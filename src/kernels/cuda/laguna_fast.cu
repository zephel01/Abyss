// src/kernels/cuda/laguna_fast.cu - see include/strata/kernels/laguna_fast.hpp.
#include "strata/kernels/laguna_fast.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace strata::kernels::lag {
namespace {

constexpr int kD = kHeadDim;
constexpr int kWarps = 4;          // attention: a warp a (row, head), 4 a block
constexpr int kSplitKeys = 256;    // ... a split's keys at the least when the rows are few

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(expf(x)); }   // (ggml's)

// block (b, t), 128 threads (a dim each): b < n_head q head b of row t, else KV head b - n_head
__global__ void __launch_bounds__(kD) prep_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                                  const float* __restrict__ v, const float* __restrict__ q_norm,
                                                  const float* __restrict__ k_norm, float eps, int n_head, int n_head_kv,
                                                  int n_rot, const double* __restrict__ freq, float mscale, int64_t p0,
                                                  int ring, uint16_t* __restrict__ kc, uint16_t* __restrict__ vc,
                                                  float* __restrict__ q_out) {
    __shared__ float sv[kD];
    __shared__ float sred[kD / 32];
    const int b = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const bool is_q = b < n_head;
    const int g = is_q ? b : b - n_head;
    const float x = is_q ? q[((size_t) t * n_head + g) * kD + d] : k[((size_t) t * n_head_kv + g) * kD + d];
    float ss = warp_sum(x * x);
    if ((d & 31) == 0) sred[d >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int i = 0; i < kD / 32; ++i) ss += sred[i];
    const float xn = x * rsqrtf(ss / (float) kD + eps) * (is_q ? q_norm[d] : k_norm[d]);
    sv[d] = xn;
    __syncthreads();
    const int64_t p = p0 + t;
    float y = xn;
    if (d < n_rot) {   // NEOX pairs (i, i + n_rot / 2)
        const int half = n_rot / 2;
        const int i = d < half ? d : d - half;
        double sn, cs;
        sincos((double) p * freq[i], &sn, &cs);
        const float c = (float) cs * mscale, s = (float) sn * mscale;
        const float x0 = sv[i], x1 = sv[i + half];
        y = d < half ? x0 * c - x1 * s : x0 * s + x1 * c;
    }
    if (is_q) {
        q_out[((size_t) t * n_head + g) * kD + d] = y;
    } else {
        const size_t row = ((size_t) (p % ring) * n_head_kv + g) * kD;
        kc[row + d] = f2h(y);
        vc[row + d] = f2h(v[((size_t) t * n_head_kv + g) * kD + d]);
    }
}

// grid (ceil(T * n_head / 4), splits), 128 threads; warp w: (row, head) = blockIdx.x * 4 + w over its split's keys.
// A lane scores one key of a 32-key step (the whole 128-dim dot against q in shared memory), the step's softmax online
// across the warp, then the V rows: a lane 4 dims.  splits 1: out directly; else the partial (max, sum, acc[128]).
__global__ void __launch_bounds__(kWarps * 32) attn_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kc,
                                                           const uint16_t* __restrict__ vc, int ring, int n_head,
                                                           int n_head_kv, int64_t p0, int T, int window, float scale,
                                                           float* __restrict__ part, float* __restrict__ out) {
    __shared__ float sq[kWarps][kD];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int idx = blockIdx.x * kWarps + w;
    if (idx >= T * n_head) return;
    const int t = idx / n_head, h = idx % n_head;
    const int grp = h / (n_head / n_head_kv);
    const int64_t p = p0 + t;
    const int64_t lo = window > 0 && p - window + 1 > 0 ? p - window + 1 : 0;
    const int64_t n = p - lo + 1;
    const int splits = gridDim.y, sp = blockIdx.y;
    const int64_t len = ((n + splits - 1) / splits + 31) / 32 * 32;
    const int64_t a = lo + (int64_t) sp * len, e = a + len < p + 1 ? a + len : p + 1;
    const float* qr = q + ((size_t) t * n_head + h) * kD;
#pragma unroll
    for (int i = 0; i < kD / 32; ++i) sq[w][lane + 32 * i] = qr[lane + 32 * i] * scale;
    __syncwarp();
    float m = -INFINITY, l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const size_t kstride = (size_t) n_head_kv * kD;
    for (int64_t base = a; base < e; base += 32) {
        const int64_t j = base + lane;
        float sc = -INFINITY;
        if (j < e) {
            const int rj = (int) (base % ring) + lane;   // (j % ring, the step's row once)
            const uint4* kp = reinterpret_cast<const uint4*>(kc + (size_t) (rj < ring ? rj : rj - ring) * kstride +
                                                             (size_t) grp * kD);
            float dot = 0.0f;
#pragma unroll
            for (int c = 0; c < kD / 8; ++c) {
                const uint4 u = kp[c];
                const __half2* hp = reinterpret_cast<const __half2*>(&u);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 f = __half22float2(hp[i]);
                    dot += f.x * sq[w][c * 8 + 2 * i] + f.y * sq[w][c * 8 + 2 * i + 1];
                }
            }
            sc = dot;
        }
        const float mn = fmaxf(m, warp_max(sc));
        const float corr = expf(m - mn);
        const float pj = j < e ? expf(sc - mn) : 0.0f;
        l = l * corr + warp_sum(pj);
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] *= corr;
        const int nk = e - base < 32 ? (int) (e - base) : 32;
        // the step's V rows: the ring row once a step, the loads of 8 keys issued before their sums (the sums in key
        // order, as one at a time: the same bits, without a dependent load a key)
        const int rb = (int) (base % ring);
        const uint16_t* vg = vc + (size_t) grp * kD + lane * 4;
        const auto vrow = [&](int jj) {
            const int r = rb + jj < ring ? rb + jj : rb + jj - ring;
            return *reinterpret_cast<const uint2*>(vg + (size_t) r * kstride);
        };
        const auto add = [&](float pp, uint2 u) {
            const __half2* hp = reinterpret_cast<const __half2*>(&u);
            const float2 f0 = __half22float2(hp[0]), f1 = __half22float2(hp[1]);
            acc[0] += pp * f0.x;
            acc[1] += pp * f0.y;
            acc[2] += pp * f1.x;
            acc[3] += pp * f1.y;
        };
        int jj = 0;
        for (; jj + 8 <= nk; jj += 8) {
            uint2 u[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) u[i] = vrow(jj + i);
#pragma unroll
            for (int i = 0; i < 8; ++i) add(__shfl_sync(0xffffffffu, pj, jj + i), u[i]);
        }
        for (; jj < nk; ++jj) add(__shfl_sync(0xffffffffu, pj, jj), vrow(jj));
        m = mn;
    }
    if (splits == 1) {
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        float* o = out + ((size_t) t * n_head + h) * kD + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) o[i] = acc[i] * inv;
    } else {
        float* pr = part + ((size_t) sp * T * n_head + idx) * (kD + 2);
        if (lane == 0) {
            pr[0] = m;
            pr[1] = l;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) pr[2 + lane * 4 + i] = acc[i];
    }
}

// The same attention as attn_kernel for few rows (a token, a verify window), a block a (row, head, split) instead of a
// warp: the split's key scores first, every warp at once (each key's dot as attn_kernel's lane computes it: the same
// bits), into shared memory; then warp 0 runs attn_kernel's online softmax and V sums over them in key order.  The
// scores of a key no longer wait for the previous step's softmax - the dot was the step's long serial part.
// grid (T * n_head, splits), 128 threads, dynamic shared memory: the split's scores (len floats).
__global__ void __launch_bounds__(kWarps * 32) attn_few_kernel(const float* __restrict__ q,
                                                               const uint16_t* __restrict__ kc,
                                                               const uint16_t* __restrict__ vc, int ring, int n_head,
                                                               int n_head_kv, int64_t p0, int T, int window, float scale,
                                                               float* __restrict__ part, float* __restrict__ out) {
    extern __shared__ float ssc[];
    __shared__ float sq[kD];
    const int idx = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
    const int t = idx / n_head, h = idx % n_head;
    const int grp = h / (n_head / n_head_kv);
    const int64_t p = p0 + t;
    const int64_t lo = window > 0 && p - window + 1 > 0 ? p - window + 1 : 0;
    const int64_t n = p - lo + 1;
    const int splits = gridDim.y, sp = blockIdx.y;
    const int64_t len = ((n + splits - 1) / splits + 31) / 32 * 32;
    const int64_t a = lo + (int64_t) sp * len, e = a + len < p + 1 ? a + len : p + 1;
    const float* qr = q + ((size_t) t * n_head + h) * kD;
    sq[tid] = qr[tid] * scale;
    __syncthreads();
    const size_t kstride = (size_t) n_head_kv * kD;
    const int ra = (int) (a % ring);
    for (int64_t j = a + tid; j < e; j += kWarps * 32) {
        const int rj = ra + (int) (j - a);   // (j % ring: a split is shorter than the ring)
        const uint4* kp = reinterpret_cast<const uint4*>(kc + (size_t) (rj < ring ? rj : rj - ring) * kstride +
                                                         (size_t) grp * kD);
        float dot = 0.0f;
#pragma unroll
        for (int c = 0; c < kD / 8; ++c) {
            const uint4 u = kp[c];
            const __half2* hp = reinterpret_cast<const __half2*>(&u);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 f = __half22float2(hp[i]);
                dot += f.x * sq[c * 8 + 2 * i] + f.y * sq[c * 8 + 2 * i + 1];
            }
        }
        ssc[j - a] = dot;
    }
    __syncthreads();
    if (tid >= 32) return;
    float m = -INFINITY, l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint16_t* vg = vc + (size_t) grp * kD + lane * 4;
    for (int64_t base = a; base < e; base += 32) {
        const int64_t j = base + lane;
        const float sc = j < e ? ssc[j - a] : -INFINITY;
        const float mn = fmaxf(m, warp_max(sc));
        const float corr = expf(m - mn);
        const float pj = j < e ? expf(sc - mn) : 0.0f;
        l = l * corr + warp_sum(pj);
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] *= corr;
        const int nk = e - base < 32 ? (int) (e - base) : 32;
        const int rb = (int) (base % ring);
        const auto vrow = [&](int jj) {
            const int r = rb + jj < ring ? rb + jj : rb + jj - ring;
            return *reinterpret_cast<const uint2*>(vg + (size_t) r * kstride);
        };
        const auto add = [&](float pp, uint2 u) {
            const __half2* hp = reinterpret_cast<const __half2*>(&u);
            const float2 f0 = __half22float2(hp[0]), f1 = __half22float2(hp[1]);
            acc[0] += pp * f0.x;
            acc[1] += pp * f0.y;
            acc[2] += pp * f1.x;
            acc[3] += pp * f1.y;
        };
        int jj = 0;
        for (; jj + 8 <= nk; jj += 8) {
            uint2 u[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) u[i] = vrow(jj + i);
#pragma unroll
            for (int i = 0; i < 8; ++i) add(__shfl_sync(0xffffffffu, pj, jj + i), u[i]);
        }
        for (; jj < nk; ++jj) add(__shfl_sync(0xffffffffu, pj, jj), vrow(jj));
        m = mn;
    }
    if (splits == 1) {
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        float* o = out + ((size_t) t * n_head + h) * kD + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) o[i] = acc[i] * inv;
    } else {
        float* pr = part + ((size_t) sp * T * n_head + idx) * (kD + 2);
        if (lane == 0) {
            pr[0] = m;
            pr[1] = l;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) pr[2 + lane * 4 + i] = acc[i];
    }
}

// block a (row, head), 128 threads: the splits' partials merged
__global__ void __launch_bounds__(kD) merge_kernel(const float* __restrict__ part, int rows, int splits,
                                                   float* __restrict__ out) {
    const int idx = blockIdx.x, d = threadIdx.x;
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) m = fmaxf(m, part[((size_t) s * rows + idx) * (kD + 2)]);
    float l = 0.0f, acc = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float* pr = part + ((size_t) s * rows + idx) * (kD + 2);
        if (pr[1] == 0.0f) continue;
        const float c = expf(pr[0] - m);
        l += pr[1] * c;
        acc += pr[2 + d] * c;
    }
    out[(size_t) idx * kD + d] = l > 0.0f ? acc / l : 0.0f;
}

__global__ void gate_kernel(float* __restrict__ o, const float* __restrict__ g, int n_head) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    o[((size_t) t * n_head + h) * kD + d] *= softplus_f(g[(size_t) t * n_head + h]);
}

}  // namespace

void rope_freqs(int n_rot, double base, double freq_scale, int orig_ctx, double beta_fast, double beta_slow,
                double* freq, float& mscale) {
    const int half = n_rot / 2;
    const bool yarn = freq_scale != 1.0;
    // ggml_rope_yarn_corr_dims, in float as there (the floor / ceil decide which pairs ramp)
    const auto corr = [&](float beta) {
        return (float) n_rot * logf((float) orig_ctx / (beta * 2.0f * 3.14159265358979323846f)) / (2.0f * logf((float) base));
    };
    const float low = std::max(0.0f, floorf(corr((float) beta_fast)));
    const float high = std::min((float) (n_rot - 1), ceilf(corr((float) beta_slow)));
    for (int i = 0; i < half; ++i) {
        const double f = std::pow(base, -2.0 * i / n_rot);
        if (!yarn) {
            freq[i] = f;
            continue;
        }
        const float y = ((float) i - low) / std::max(0.001f, high - low);
        const double mix = 1.0 - std::min(1.0f, std::max(0.0f, y));   // (ext_factor 1)
        freq[i] = f * (freq_scale * (1.0 - mix) + mix);
    }
    mscale = yarn ? 1.0f + 0.1f * logf(1.0f / (float) freq_scale) : 1.0f;
}

void attn_prep_rows(const float* q, const float* k, const float* v, const float* q_norm, const float* k_norm,
                    float eps, int n_head, int n_head_kv, int n_rot, const double* freq, float mscale, int64_t p0,
                    int T, int ring, uint16_t* kc, uint16_t* vc, float* q_out, cudaStream_t s) {
    if (T <= 0) return;
    prep_kernel<<<dim3(n_head + n_head_kv, T), kD, 0, s>>>(q, k, v, q_norm, k_norm, eps, n_head, n_head_kv, n_rot, freq,
                                                          mscale, p0, ring, kc, vc, q_out);
}

size_t attn_part_floats(int rows, int n_head, int max_splits) {
    return (size_t) rows * n_head * std::max(max_splits, 1) * (kD + 2);
}

void attn(const float* q, const uint16_t* kc, const uint16_t* vc, int ring, int n_head, int n_head_kv, int64_t p0,
          int T, int window, float scale, float* part, int max_splits, float* out, cudaStream_t s) {
    if (T <= 0) return;
    const int pairs = T * n_head;
    const int64_t keys = window > 0 ? std::min<int64_t>(window, p0 + T) : p0 + T;
    int splits = 1;
    if (part != nullptr && pairs < 1024) {   // (few rows - a token, a short window: the keys spread over blocks)
        splits = (int) std::min<int64_t>((keys + kSplitKeys - 1) / kSplitKeys, max_splits);
        splits = std::max(splits, 1);
    }
    // few rows (a token, a verify window): a block a (row, head, split), its scores at once (attn_few_kernel: the
    // same bits) while a split's scores fit in shared memory; many rows (the prompt): a warp a (row, head)
    const int64_t len = ((keys + splits - 1) / splits + 31) / 32 * 32;   // (a split's keys at the most)
    if (pairs < 1024 && len <= 11776) {   // (46 KB of scores + q: under the 48 KB a block has without opting in)
        attn_few_kernel<<<dim3((unsigned) pairs, (unsigned) splits), kWarps * 32, (size_t) len * sizeof(float), s>>>(
            q, kc, vc, ring, n_head, n_head_kv, p0, T, window, scale, part, out);
    } else {
        const dim3 grid((unsigned) ((pairs + kWarps - 1) / kWarps), (unsigned) splits);
        attn_kernel<<<grid, kWarps * 32, 0, s>>>(q, kc, vc, ring, n_head, n_head_kv, p0, T, window, scale, part, out);
    }
    if (splits > 1) merge_kernel<<<pairs, kD, 0, s>>>(part, pairs, splits, out);
}

void gate_rows(float* o, const float* g, int T, int n_head, cudaStream_t s) {
    if (T <= 0) return;
    gate_kernel<<<dim3(n_head, T), kD, 0, s>>>(o, g, n_head);
}

}  // namespace strata::kernels::lag
