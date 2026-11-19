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

// --kv int8 (laguna_fast.hpp kv 2): a cache row (a position, a KV head) is 4 groups of 32 dims, scale = max |x| / 127
// (FP16), code = round(x / scale) - the 4 scales (8 B), then the 128 codes in dim order: 136 B against FP16's 256
constexpr int kRow8 = 136;
// thread d's value x (dim d < 128, a warp a group) into the int8 row at b
__device__ __forceinline__ void q8_put(uint8_t* b, int d, float x) {
    const float amax = warp_max(fabsf(x));
    const float sc = amax / 127.0f, id = sc != 0.0f ? 1.0f / sc : 0.0f;
    reinterpret_cast<int8_t*>(b + 8)[d] = (int8_t) max(-127, min(127, __float2int_rn(x * id)));
    if ((d & 31) == 0) reinterpret_cast<uint16_t*>(b)[d >> 5] = f2h(sc);
}
// a row's address: FP16 [row][n_head_kv][128] halves, int8 [row][n_head_kv] rows of kRow8 bytes
template <int F>
__device__ __forceinline__ const uint8_t* kv_at(const uint16_t* c, size_t row, int n_head_kv, int grp) {
    return F == 2 ? reinterpret_cast<const uint8_t*>(c) + (row * n_head_kv + grp) * kRow8
                  : reinterpret_cast<const uint8_t*>(c + (row * n_head_kv + grp) * kD);
}
// q . k over a whole key row (sq: q * scale); FP16: the sums as attn_kernel always took them; int8: code * scale in F32
template <int F>
__device__ __forceinline__ float key_dot(const uint8_t* r, const float* sq) {
    float dot = 0.0f;
    if constexpr (F == 2) {
#pragma unroll
        for (int gq = 0; gq < kD / 32; ++gq) {
            const float d = __half2float(reinterpret_cast<const __half*>(r)[gq]);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint2 w = *reinterpret_cast<const uint2*>(r + 8 + gq * 32 + c * 8);
                const int8_t* q8 = reinterpret_cast<const int8_t*>(&w);
#pragma unroll
                for (int i = 0; i < 8; i += 2) {
                    const int o = gq * 32 + c * 8 + i;
                    dot += (float) q8[i] * d * sq[o] + (float) q8[i + 1] * d * sq[o + 1];
                }
            }
        }
    } else {
        const uint4* kp = reinterpret_cast<const uint4*>(r);
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
    }
    return dot;
}
// lane's 4 dims of a V row: FP16 the 4 halves, int8 the 4 codes and their group's scale - as a uint2 to keep the loads
// ahead of their sums
template <int F>
__device__ __forceinline__ uint2 v_load(const uint8_t* r, int lane) {
    if constexpr (F == 2)
        return make_uint2(*reinterpret_cast<const uint32_t*>(r + 8 + lane * 4),
                          (uint32_t) reinterpret_cast<const uint16_t*>(r)[lane >> 3]);
    else
        return *reinterpret_cast<const uint2*>(r + lane * 8);
}
template <int F>
__device__ __forceinline__ void v_add(float* acc, float pp, uint2 u) {
    if constexpr (F == 2) {
        const int8_t* q8 = reinterpret_cast<const int8_t*>(&u.x);
        const float d = __half2float(__ushort_as_half((unsigned short) u.y));
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] += pp * ((float) q8[i] * d);
    } else {
        const __half2* hp = reinterpret_cast<const __half2*>(&u);
        const float2 f0 = __half22float2(hp[0]), f1 = __half22float2(hp[1]);
        acc[0] += pp * f0.x;
        acc[1] += pp * f0.y;
        acc[2] += pp * f1.x;
        acc[3] += pp * f1.y;
    }
}
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(expf(x)); }   // (ggml's)

// block (b, t), 128 threads (a dim each): b < n_head q head b of row t, else KV head b - n_head
template <int F>
__global__ void __launch_bounds__(kD) prep_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                                  const float* __restrict__ v, const float* __restrict__ q_norm,
                                                  const float* __restrict__ k_norm, float eps, int n_head, int n_head_kv,
                                                  int n_rot, const double* __restrict__ freq, float mscale, int64_t p0,
                                                  int ring, uint16_t* __restrict__ kc, uint16_t* __restrict__ vc,
                                                  float* __restrict__ q_out, const float2* __restrict__ cs_tab) {
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
        float c, s;
        if (cs_tab != nullptr) {   // (rope_table's: the same values, computed once a row)
            const float2 v = cs_tab[(size_t) t * half + i];
            c = v.x;
            s = v.y;
        } else {
            double sn, cs;
            sincos((double) p * freq[i], &sn, &cs);
            c = (float) cs * mscale;
            s = (float) sn * mscale;
        }
        const float x0 = sv[i], x1 = sv[i + half];
        y = d < half ? x0 * c - x1 * s : x0 * s + x1 * c;
    }
    if (is_q) {
        q_out[((size_t) t * n_head + g) * kD + d] = y;
    } else if constexpr (F == 2) {
        const size_t row = ((size_t) (p % ring) * n_head_kv + g) * kRow8;
        q8_put(reinterpret_cast<uint8_t*>(kc) + row, d, y);
        q8_put(reinterpret_cast<uint8_t*>(vc) + row, d, v[((size_t) t * n_head_kv + g) * kD + d]);
    } else {
        const size_t row = ((size_t) (p % ring) * n_head_kv + g) * kD;
        kc[row + d] = f2h(y);
        vc[row + d] = f2h(v[((size_t) t * n_head_kv + g) * kD + d]);
    }
}

// grid (ceil(T * n_head / 4), splits), 128 threads; warp w: (row, head) = blockIdx.x * 4 + w over its split's keys.
// A lane scores one key of a 32-key step (the whole 128-dim dot against q in shared memory), the step's softmax online
// across the warp, then the V rows: a lane 4 dims.  splits 1: out directly; else the partial (max, sum, acc[128]).
template <int F>
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
    for (int64_t base = a; base < e; base += 32) {
        const int64_t j = base + lane;
        float sc = -INFINITY;
        if (j < e) {
            const int rj = (int) (base % ring) + lane;   // (j % ring, the step's row once)
            sc = key_dot<F>(kv_at<F>(kc, (size_t) (rj < ring ? rj : rj - ring), n_head_kv, grp), sq[w]);
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
        const auto vrow = [&](int jj) {
            const int r = rb + jj < ring ? rb + jj : rb + jj - ring;
            return v_load<F>(kv_at<F>(vc, (size_t) r, n_head_kv, grp), lane);
        };
        const auto add = [&](float pp, uint2 u) { v_add<F>(acc, pp, u); };
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
template <int F>
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
    const int ra = (int) (a % ring);
    for (int64_t j = a + tid; j < e; j += kWarps * 32) {
        const int rj = ra + (int) (j - a);   // (j % ring: a split is shorter than the ring)
        ssc[j - a] = key_dot<F>(kv_at<F>(kc, (size_t) (rj < ring ? rj : rj - ring), n_head_kv, grp), sq);
    }
    __syncthreads();
    if (tid >= 32) return;
    float m = -INFINITY, l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
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
            return v_load<F>(kv_at<F>(vc, (size_t) r, n_head_kv, grp), lane);
        };
        const auto add = [&](float pp, uint2 u) { v_add<F>(acc, pp, u); };
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

// block t, a thread a pair: prep_kernel's (cos, sin) x mscale of row t
__global__ void rope_table_kernel(const double* __restrict__ freq, int half, float mscale, int64_t p0,
                                  float2* __restrict__ cs) {
    const int t = blockIdx.x, i = threadIdx.x;
    if (i >= half) return;
    double sn, c;
    sincos((double) (p0 + t) * freq[i], &sn, &c);
    cs[(size_t) t * half + i] = make_float2((float) c * mscale, (float) sn * mscale);
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
                    int T, int ring, uint16_t* kc, uint16_t* vc, float* q_out, cudaStream_t s, const float2* cs, int kv) {
    if (T <= 0) return;
    (kv == 2 ? prep_kernel<2> : prep_kernel<0>)<<<dim3(n_head + n_head_kv, T), kD, 0, s>>>(
        q, k, v, q_norm, k_norm, eps, n_head, n_head_kv, n_rot, freq, mscale, p0, ring, kc, vc, q_out, cs);
}

void rope_table(const double* freq, int n_rot, float mscale, int64_t p0, int T, float2* cs, cudaStream_t s) {
    if (T <= 0 || n_rot <= 0) return;
    rope_table_kernel<<<T, kD, 0, s>>>(freq, n_rot / 2, mscale, p0, cs);
}

size_t attn_part_floats(int rows, int n_head, int max_splits) {
    return (size_t) rows * n_head * std::max(max_splits, 1) * (kD + 2);
}

void attn(const float* q, const uint16_t* kc, const uint16_t* vc, int ring, int n_head, int n_head_kv, int64_t p0,
          int T, int window, float scale, float* part, int max_splits, float* out, cudaStream_t s, int kv) {
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
        (kv == 2 ? attn_few_kernel<2> : attn_few_kernel<0>)<<<dim3((unsigned) pairs, (unsigned) splits), kWarps * 32, (size_t) len * sizeof(float), s>>>(
            q, kc, vc, ring, n_head, n_head_kv, p0, T, window, scale, part, out);
    } else {
        const dim3 grid((unsigned) ((pairs + kWarps - 1) / kWarps), (unsigned) splits);
        (kv == 2 ? attn_kernel<2> : attn_kernel<0>)<<<grid, kWarps * 32, 0, s>>>(q, kc, vc, ring, n_head, n_head_kv, p0, T, window, scale, part, out);
    }
    if (splits > 1) merge_kernel<<<pairs, kD, 0, s>>>(part, pairs, splits, out);
}

void gate_rows(float* o, const float* g, int T, int n_head, cudaStream_t s) {
    if (T <= 0) return;
    gate_kernel<<<dim3(n_head, T), kD, 0, s>>>(o, g, n_head);
}

}  // namespace strata::kernels::lag
