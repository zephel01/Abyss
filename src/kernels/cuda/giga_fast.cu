// src/kernels/cuda/giga_fast.cu - see include/strata/kernels/giga_fast.hpp.
#include "strata/kernels/giga_fast.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdio>

namespace strata::kernels::giga {
namespace {

std::atomic<int> g_errors{0};
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        g_errors.fetch_add(1);
        std::fprintf(stderr, "giga_fast %s: %s\n", what, cudaGetErrorString(e));
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// a block's sum (blockDim.x a multiple of 32, <= 1024); every thread gets it.  A fixed order: the same bits every run
__device__ __forceinline__ float block_sum(float v, float* sred) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    if (lane == 0) sred[warp] = v;
    __syncthreads();
    float t = 0.0f;
    if (warp == 0) {
        t = lane < nw ? sred[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) sred[0] = t;
    }
    __syncthreads();
    t = sred[0];
    __syncthreads();
    return t;
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ uint16_t f2h(float x) { return __half_as_ushort(__float2half(x)); }

constexpr int kNormThreads = 512;

// one block: the sum of squares, the rank dots of down against the gate's input (strided over the threads, then the
// block's sums), the activation, then every element's up row and the gated output
__global__ void gated_norm_kernel(float* h, const float* add, const float* __restrict__ w,
                                  const float* __restrict__ down, const float* __restrict__ up, float gate_w, float eps,
                                  int n, int rank, int form, float* y) {
    __shared__ float sred[32];
    __shared__ float t[kRankMax];
    const int tid = threadIdx.x;
    h += (size_t) blockIdx.x * n;   // (a row a block)
    y += (size_t) blockIdx.x * n;
    if (add != nullptr) add += (size_t) blockIdx.x * n;
    float ss = 0.0f;
    for (int i = tid; i < n; i += kNormThreads) {
        float x = h[i];
        if (add != nullptr) {
            x += add[i];
            h[i] = x;
        }
        ss += x * x;
    }
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) n + eps);
    const int in_form = form & kNormIn, act = (form >> kNormActShift) & kNormAct, gf = (form >> kNormGateShift) & kNormGate;
    if (!(form & kNormPlain)) {
        for (int r = 0; r < rank; ++r) {
            const float* d = down + (size_t) r * n;
            float a = 0.0f;
            for (int i = tid; i < n; i += kNormThreads) {
                float gi = h[i];
                if (in_form >= 1) gi *= inv;
                if (in_form == 2) gi *= w[i];
                a += d[i] * gi;
            }
            a = block_sum(a, sred);
            if (tid == 0) {
                float v = a;
                if (act == 1) v = a * sigm(a);
                else if (act == 2) v = fmaxf(a, 0.0f);
                else if (act == 3) v = 0.5f * a * (1.0f + erff(a * 0.70710678118654752f));
                else if (act == 4) v = tanhf(a);
                else if (act == 5) v = sigm(a);
                t[r] = v;
            }
        }
        __syncthreads();
    }
    for (int i = tid; i < n; i += kNormThreads) {
        float g = 1.0f;
        if (!(form & kNormPlain)) {
            const float* u = up + (size_t) i * rank;
            float a = 0.0f;
            for (int r = 0; r < rank; ++r) a += u[r] * t[r];
            g = gf == 0 ? gate_w * sigm(a) : gf == 1 ? sigm(gate_w * a) : gf == 2 ? 1.0f + gate_w * tanhf(a) : sigm(a);
        }
        y[i] = h[i] * inv * w[i] * g;
    }
}

// a block a value head, S threads (S 128)
__global__ void gdn_out_kernel(const float* __restrict__ o, const float* __restrict__ z, const float* __restrict__ nw,
                               float* y, int S, float eps, float scale, int silu) {
    __shared__ float sred[32];
    const int tid = threadIdx.x;
    const size_t at = (size_t) blockIdx.x * S + tid;
    const float x = tid < S ? o[at] : 0.0f;
    const float ss = block_sum(x * x, sred);
    if (tid >= S) return;
    const float zz = z[at];
    const float g = silu ? zz * sigm(zz) : scale * sigm(zz);
    y[at] = x * rsqrtf(ss / (float) S + eps) * nw[tid] * g;
}

__global__ void gdn_expand_kernel(const float* __restrict__ h, float* __restrict__ out, int h_k, int h_v, int S) {
    const int v = blockIdx.x, part = blockIdx.y, c = threadIdx.x;
    if (c >= S) return;
    const int rep = h_v / h_k;
    h += (size_t) blockIdx.z * (size_t) (2 * h_k + h_v) * S;   // (a row a z)
    out += (size_t) blockIdx.z * (size_t) 3 * h_v * S;
    const size_t dst = ((size_t) part * h_v + v) * S + c;
    if (part == 2) out[dst] = h[((size_t) 2 * h_k + v) * S + c];
    else out[dst] = h[((size_t) part * h_k + v / rep) * S + c];
}

__global__ void rope_table_kernel(int64_t pos, const double* __restrict__ freq, float mscale, float2* cs) {
    const int j = threadIdx.x;
    if (j >= kRot / 2) return;
    const double th = (double) pos * freq[j];
    cs[j] = make_float2(mscale * (float) cos(th), mscale * (float) sin(th));
}

// block 0: the cache row; block 1 + h: head h's rope part.  512 threads
__global__ void mla_prep_kernel(float* q, const float* __restrict__ lat, const float* __restrict__ kv_rot,
                                const float2* __restrict__ cs, uint16_t* __restrict__ row, int interleave) {
    const int tid = threadIdx.x;
    const int half = kRot / 2;
    if (blockIdx.x == 0) {
        row[tid] = f2h(lat[tid]);   // (blockDim 512 = kLat)
        if (tid < half) {
            const float2 c = cs[tid];
            const int i0 = interleave ? 2 * tid : tid, i1 = interleave ? 2 * tid + 1 : tid + half;
            const float x0 = kv_rot[i0], x1 = kv_rot[i1];
            row[kLat + i0] = f2h(x0 * c.x - x1 * c.y);
            row[kLat + i1] = f2h(x0 * c.y + x1 * c.x);
        }
        return;
    }
    if (tid >= half) return;
    float* qr = q + (size_t) (blockIdx.x - 1) * kQHead + kNope;
    const float2 c = cs[tid];
    const int i0 = interleave ? 2 * tid : tid, i1 = interleave ? 2 * tid + 1 : tid + half;
    const float x0 = qr[i0], x1 = qr[i1];
    qr[i0] = x0 * c.x - x1 * c.y;
    qr[i1] = x0 * c.y + x1 * c.x;
}

__global__ void scale_kernel(float* x, float a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}

// block 0: the key (kIdxKey threads); block 1 + h: iq's head h (its first kRot dims roped, half-split pairs)
__global__ void idx_prep_kernel(float* iq, const float* __restrict__ kraw, const float* __restrict__ nw, float eps,
                                const float2* __restrict__ cs, uint16_t* __restrict__ row) {
    __shared__ float sred[32];
    __shared__ float kn[kIdxKey];
    const int tid = threadIdx.x, half = kRot / 2;
    if (blockIdx.x == 0) {
        const float x = kraw[tid];
        const float ss = block_sum(x * x, sred);
        kn[tid] = x * rsqrtf(ss / (float) kIdxKey + eps) * nw[tid];
        __syncthreads();
        float v = kn[tid];
        if (tid < half) {
            const float2 c = cs[tid];
            v = kn[tid] * c.x - kn[tid + half] * c.y;
        } else if (tid < kRot) {
            const float2 c = cs[tid - half];
            v = kn[tid] * c.x + kn[tid - half] * c.y;
        }
        row[tid] = f2h(v);
        return;
    }
    if (tid >= half) return;
    float* q = iq + (size_t) (blockIdx.x - 1) * kIdxKey;
    const float2 c = cs[tid];
    const float x0 = q[tid], x1 = q[tid + half];
    q[tid] = x0 * c.x - x1 * c.y;
    q[tid + half] = x1 * c.x + x0 * c.y;
}

// 4 positions a block, 64 threads (the heads) a position: the dots, then the heads' weighted relu sum in a fixed order
__global__ void __launch_bounds__(256) idx_score_kernel(const float* __restrict__ iq, const float* __restrict__ w,
                                                        const uint16_t* __restrict__ keys, int64_t n, float scale,
                                                        float* __restrict__ score) {
    __shared__ float key[4][kIdxKey];
    __shared__ float sred[8];
    const int tid = threadIdx.x, q = tid >> 6, h = tid & 63;
    const int64_t pos = (int64_t) blockIdx.x * 4 + q;
    for (int i = tid; i < 4 * kIdxKey; i += 256) {
        const int64_t pp = (int64_t) blockIdx.x * 4 + i / kIdxKey;
        key[i / kIdxKey][i % kIdxKey] = pp < n ? __half2float(__ushort_as_half(keys[(size_t) pp * kIdxKey + i % kIdxKey])) : 0.0f;
    }
    __syncthreads();
    const float* qh = iq + (size_t) h * kIdxKey;
    float a = 0.0f;
#pragma unroll 8
    for (int d = 0; d < kIdxKey; ++d) a += qh[d] * key[q][d];
    a = warp_sum(w[h] * fmaxf(a, 0.0f));
    if ((tid & 31) == 0) sred[tid >> 5] = a;
    __syncthreads();
    if (h == 0 && pos < n) score[pos] = scale * (sred[2 * q] + sred[2 * q + 1]);
}

// a float's bits as an unsigned key in the floats' order
__device__ __forceinline__ uint32_t fkey(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// one block of 1024: a radix select (4 digits of 8 bits) for the k-th largest key, then the selected positions written
// in ascending order (thread t owns a contiguous run of positions; the runs' counts prefix-summed by thread 0)
__global__ void __launch_bounds__(1024) idx_topk_kernel(const float* __restrict__ score, int64_t n, int k, int* __restrict__ sel) {
    __shared__ int hist[256];
    __shared__ int cgt[1024], ceq[1024], off[1024];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int s_need;
    const int tid = threadIdx.x;
    if (tid == 0) { s_prefix = 0u; s_mask = 0u; s_need = k; }
    for (int pass = 3; pass >= 0; --pass) {
        const int shift = pass * 8;
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        const uint32_t prefix = s_prefix, mask = s_mask;
        for (int64_t i = tid; i < n; i += 1024) {
            const uint32_t u = fkey(score[i]);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255u], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int need = s_need, b = 255;
            for (; b > 0; --b) {
                if (hist[b] >= need) break;
                need -= hist[b];
            }
            s_need = need;
            s_prefix = prefix | ((uint32_t) b << shift);
            s_mask = mask | (255u << shift);
        }
        __syncthreads();
    }
    const uint32_t thr = s_prefix;
    const int need = s_need;   // (the keys equal to thr to take: the first ones)
    const int64_t run = (n + 1023) / 1024, i0 = (int64_t) tid * run, i1 = i0 + run < n ? i0 + run : n;
    int gt = 0, eq = 0;
    for (int64_t i = i0; i < i1; ++i) {
        const uint32_t u = fkey(score[i]);
        gt += u > thr;
        eq += u == thr;
    }
    cgt[tid] = gt;
    ceq[tid] = eq;
    __syncthreads();
    if (tid == 0) {
        int o = 0, e = 0;
        for (int t = 0; t < 1024; ++t) {
            const int take = need - e > 0 ? (ceq[t] < need - e ? ceq[t] : need - e) : 0;
            off[t] = o;
            o += cgt[t] + take;
            const int et = ceq[t];
            ceq[t] = e;   // (the equal keys before this run)
            e += et;
        }
    }
    __syncthreads();
    int o = off[tid], e = ceq[tid];
    for (int64_t i = i0; i < i1; ++i) {
        const uint32_t u = fkey(score[i]);
        if (u > thr) sel[o++] = (int) i;
        else if (u == thr) {
            if (e < need) sel[o++] = (int) i;
            ++e;
        }
    }
}

__global__ void gather_rows_kernel(const uint16_t* __restrict__ cache, const int* __restrict__ sel, uint16_t* __restrict__ out) {
    const size_t src = (size_t) sel[blockIdx.x] * kRow, dst = (size_t) blockIdx.x * kRow;
    for (int i = threadIdx.x; i < kRow; i += blockDim.x) out[dst + i] = cache[src + i];
}

}  // namespace

void idx_prep(float* iq, int n_heads, const float* kraw, const float* k_norm, float eps, const float2* cs,
              uint16_t* key_row, cudaStream_t s) {
    idx_prep_kernel<<<1 + (iq != nullptr ? n_heads : 0), kIdxKey, 0, s>>>(iq, kraw, k_norm, eps, cs, key_row);
    check("idx_prep");
}

void idx_score(const float* iq, const float* w, const uint16_t* keys, int64_t n, float scale, float* score,
               cudaStream_t s) {
    idx_score_kernel<<<(unsigned) ((n + 3) / 4), 256, 0, s>>>(iq, w, keys, n, scale, score);
    check("idx_score");
}

void idx_topk(const float* score, int64_t n, int k, int* sel, cudaStream_t s) {
    idx_topk_kernel<<<1, 1024, 0, s>>>(score, n, k, sel);
    check("idx_topk");
}

void gather_rows(const uint16_t* cache, const int* sel, int k, uint16_t* out, cudaStream_t s) {
    gather_rows_kernel<<<k, 192, 0, s>>>(cache, sel, out);
    check("gather_rows");
}

void gated_norm(float* h, const float* add, const float* w, const float* down, const float* up, float gate_w, float eps,
                int n, int rank, int form, float* y, cudaStream_t s, int rows) {
    if (rank > kRankMax || rows < 1) {
        g_errors.fetch_add(1);
        return;
    }
    gated_norm_kernel<<<rows, kNormThreads, 0, s>>>(h, add, w, down, up, gate_w, eps, n, rank, form, y);
    check("gated_norm");
}

void gdn_out(const float* o, const float* z, const float* ssm_norm, float* y, int h_v, int S, float eps, float scale,
             bool silu, cudaStream_t s, int rows) {
    gdn_out_kernel<<<h_v * rows, (S + 31) / 32 * 32, 0, s>>>(o, z, ssm_norm, y, S, eps, scale, silu ? 1 : 0);
    check("gdn_out");
}

void gdn_expand(const float* h, float* out, int h_k, int h_v, int S, cudaStream_t s, int rows) {
    gdn_expand_kernel<<<dim3(h_v, 3, rows), (S + 31) / 32 * 32, 0, s>>>(h, out, h_k, h_v, S);
    check("gdn_expand");
}

void rope_table(int64_t pos, const double* freq, float mscale, float2* cs, cudaStream_t s) {
    rope_table_kernel<<<1, 32, 0, s>>>(pos, freq, mscale, cs);
    check("rope_table");
}

void mla_prep(float* q, int n_head, const float* lat, const float* kv_rot, const float2* cs, uint16_t* row,
              bool interleave, cudaStream_t s) {
    mla_prep_kernel<<<1 + n_head, kLat, 0, s>>>(q, lat, kv_rot, cs, row, interleave ? 1 : 0);
    check("mla_prep");
}

void scale(float* x, float a, int n, cudaStream_t s) {
    scale_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, a, n);
    check("scale");
}

int launch_errors() { return g_errors.load(); }

}  // namespace strata::kernels::giga
