// src/kernels/cuda/hy4_fast.cu - see include/strata/kernels/hy4_fast.hpp.
#include "strata/kernels/hy4_fast.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "dsa_topk.cuh"

namespace strata::kernels::hy4 {
namespace {

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
// a block-wide sum (every thread gets it); sred: 32 floats of shared memory
__device__ float block_sum(float v, float* sred) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sred[warp] = v;
    __syncthreads();
    float t = lane < nw ? sred[lane] : 0.0f;
    t = warp_sum(t);
    return t;
}
__device__ float block_max(float v, float* sred) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sred[warp] = v;
    __syncthreads();
    float t = lane < nw ? sred[lane] : -INFINITY;
    t = warp_max(t);
    return t;
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }

// the interleaved-pair rotation of (x0, x1) = dims (2j, 2j+1) of a 64-dim rope part at pos: angle pos * base^(-2j/64)
// (the angle in float64: at long positions a float angle is off by more than the pair's own rounding)
__device__ __forceinline__ void rot_pair(float& x0, float& x1, int j, Pos3 p3, float base) {
    const int64_t pos = j < 30 ? (j % 3 == 0 ? p3.t : j % 3 == 1 ? p3.h : p3.w) : p3.t;   // the pair's M-RoPE plane
    const double inv = pow((double) base, -2.0 * (double) j / (double) kRot);
    double sd, cd;
    sincos((double) pos * inv, &sd, &cd);
    const float c = (float) cd, s = (float) sd;
    const float a = x0, b = x1;
    x0 = a * c - b * s;
    x1 = b * c + a * s;
}

// ---- the hyper-connections
__global__ void __launch_bounds__(512) hc_mix_kernel(const float* __restrict__ R, const float* __restrict__ fn, int D,
                                                     float eps, float* __restrict__ mixes) {
    __shared__ float sred[32];
    const int m = blockIdx.x;
    R += (size_t) blockIdx.y * D;             // (a prompt chunk's row: blockIdx.y; the token path launches one)
    mixes += (size_t) blockIdx.y * gridDim.x;
    const float* w = fn + (size_t) m * D;
    float dot = 0.0f, ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float r = R[i];
        dot += w[i] * r;
        ss += r * r;
    }
    dot = block_sum(dot, sred);
    ss = block_sum(ss, sred);
    if (threadIdx.x == 0) mixes[m] = dot * rsqrtf(ss / (float) D + eps);
}

__global__ void __launch_bounds__(1024) hc_pre_kernel(const float* __restrict__ mixes, const float* __restrict__ scale,
                                                      const float* __restrict__ base, const float* __restrict__ R, int hc,
                                                      int dim, float hc_eps, float mag, const float* __restrict__ norm_w,
                                                      float eps, float* __restrict__ x, float* __restrict__ post,
                                                      int mix_ld) {
    __shared__ float sred[32];
    __shared__ float s_pre[8];
    {   // (a prompt chunk's row: blockIdx.x; the token path launches one)
        const size_t r = blockIdx.x;
        mixes += r * mix_ld;
        R += r * hc * dim;
        x += r * dim;
        if (post != nullptr) post += r * hc;
    }
    if (threadIdx.x < hc) s_pre[threadIdx.x] = sigm(mixes[threadIdx.x] * scale[0] + base[threadIdx.x]) + hc_eps;
    if (post != nullptr && threadIdx.x < hc)
        post[threadIdx.x] = mag * sigm(mixes[hc + threadIdx.x] * scale[1] + base[hc + threadIdx.x]) + hc_eps;
    __syncthreads();
    constexpr int kPer = 4;   // dim <= 4096
    float v[kPer];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < kPer; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        float a = 0.0f;
        if (i < dim)
            for (int c = 0; c < hc; ++c) a += s_pre[c] * R[(size_t) c * dim + i];
        v[k] = a;
        ss += a * a;
    }
    if (norm_w != nullptr) {
        ss = block_sum(ss, sred);
        const float r = rsqrtf(ss / (float) dim + eps);
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int i = threadIdx.x + k * blockDim.x;
            if (i < dim) x[i] = v[k] * r * norm_w[i];
        }
    } else {
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int i = threadIdx.x + k * blockDim.x;
            if (i < dim) x[i] = v[k];
        }
    }
}

__global__ void __launch_bounds__(1024) hc_add_kernel(float* __restrict__ R, const float* __restrict__ post,
                                                      float* __restrict__ a, const float* __restrict__ a_norm, float eps,
                                                      int hc, int dim) {
    __shared__ float sred[32];
    R += (size_t) blockIdx.x * hc * dim;   // (a prompt chunk's row: blockIdx.x; the token path launches one)
    post += (size_t) blockIdx.x * hc;
    a += (size_t) blockIdx.x * dim;
    float r = 1.0f;
    if (a_norm != nullptr) {
        float ss = 0.0f;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += a[i] * a[i];
        ss = block_sum(ss, sred);
        r = rsqrtf(ss / (float) dim + eps);
    }
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = a_norm != nullptr ? a[i] * r * a_norm[i] : a[i];
        if (a_norm != nullptr) a[i] = v;
        for (int c = 0; c < hc; ++c) R[(size_t) c * dim + i] += post[c] * v;
    }
}

__global__ void __launch_bounds__(1024) rms_kernel(const float* __restrict__ x, const float* __restrict__ w, float eps,
                                                   int n, float* __restrict__ y, const float* __restrict__ add) {
    __shared__ float sred[32];
    x += (size_t) blockIdx.x * n;   // (a prompt chunk's row: blockIdx.x; the token path launches one)
    y += (size_t) blockIdx.x * n;
    if (add != nullptr) add += (size_t) blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i] + (add != nullptr ? add[i] : 0.0f);
        ss += v * v;
    }
    ss = block_sum(ss, sred);
    const float r = rsqrtf(ss / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = (x[i] + (add != nullptr ? add[i] : 0.0f)) * r;
        if (w != nullptr) v *= w[i];
        y[i] = v;
    }
}

// ---- the attention
__device__ __forceinline__ void mla_prep_body(float* __restrict__ q, int n_head, const float* __restrict__ kv,
                                              const float* __restrict__ kv_norm, float eps, Pos3 pos, float base,
                                              uint16_t* __restrict__ row) {
    __shared__ float sred[32];
    const int t = threadIdx.x;
    const float l = t < kLat ? kv[t] : 0.0f;
    const float ss = block_sum(l * l, sred);
    if (t < kLat) row[t] = f2h(l * rsqrtf(ss / (float) kLat + eps) * kv_norm[t]);
    if (t < kRot / 2) {
        float x0 = kv[kLat + 2 * t], x1 = kv[kLat + 2 * t + 1];
        rot_pair(x0, x1, t, pos, base);
        row[kLat + 2 * t] = f2h(x0);
        row[kLat + 2 * t + 1] = f2h(x1);
    }
    for (int i = t; i < n_head * (kRot / 2); i += blockDim.x) {
        const int h = i / (kRot / 2), j = i % (kRot / 2);
        float* qr = q + (size_t) h * (kNope + kRot) + kNope;
        float x0 = qr[2 * j], x1 = qr[2 * j + 1];
        rot_pair(x0, x1, j, pos, base);
        qr[2 * j] = x0;
        qr[2 * j + 1] = x1;
    }
}
__global__ void __launch_bounds__(512) mla_prep_kernel(float* __restrict__ q, int n_head, const float* __restrict__ kv,
                                                       const float* __restrict__ kv_norm, float eps, Pos3 pos,
                                                       float base, uint16_t* __restrict__ row) {
    mla_prep_body(q, n_head, kv, kv_norm, eps, pos, base, row);
}
// a prompt chunk: row t is position p0 + t (rotated at pos[t]), its cache row (p0 + t - row_base) % ring
__global__ void __launch_bounds__(512) mla_prep_rows_kernel(float* __restrict__ q, int n_head, const float* __restrict__ kv,
                                                            const float* __restrict__ kv_norm, float eps,
                                                            const Pos3* __restrict__ pos, int64_t p0, float base,
                                                            uint16_t* __restrict__ cache, int64_t row_base, int ring) {
    const int t = blockIdx.x;
    const int64_t r = (p0 + t - row_base) % ring;
    mla_prep_body(q + (size_t) t * n_head * (kNope + kRot), n_head, kv + (size_t) t * kRow, kv_norm, eps, pos[t], base,
                  cache + (size_t) r * kRow);
}

// a Q8_0 block: FP16 d, 32 int8
__device__ __forceinline__ float q8_0_val(const uint8_t* blk, int i) {
    const uint16_t d = (uint16_t) blk[0] | ((uint16_t) blk[1] << 8);
    return h2f(d) * (float) (int8_t) blk[2 + i];
}

__global__ void __launch_bounds__(256) mla_absorb_kernel(const float* __restrict__ q, const uint8_t* __restrict__ kb,
                                                         int n_head, float* __restrict__ qabs) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n_head * kLat) return;
    const int h = row / kLat;
    const uint8_t* w = kb + (size_t) row * (kNope / 32) * 34;
    const float* qn = q + (size_t) h * (kNope + kRot);
    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < kNope / 32; ++b) acc += q8_0_val(w + b * 34, lane) * qn[b * 32 + lane];
    acc = warp_sum(acc);
    if (lane == 0) qabs[row] = acc;
}

// the same over float rows (k_b decoded at load from a type the kernel above does not read)
__global__ void __launch_bounds__(256) mla_absorb_f32_kernel(const float* __restrict__ q, const float* __restrict__ kb,
                                                             int n_head, float* __restrict__ qabs) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n_head * kLat) return;
    const int h = row / kLat;
    const float* w = kb + (size_t) row * kNope;
    const float* qn = q + (size_t) h * (kNope + kRot);
    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < kNope / 32; ++b) acc += w[b * 32 + lane] * qn[b * 32 + lane];
    acc = warp_sum(acc);
    if (lane == 0) qabs[row] = acc;
}

constexpr int kAttnThreads = 256;
__global__ void __launch_bounds__(kAttnThreads) mla_attend_kernel(const float* __restrict__ qabs, const float* __restrict__ q,
                                                                  const uint16_t* __restrict__ cache, int ring,
                                                                  const int* __restrict__ cells, int n_cells, int64_t lo,
                                                                  const float* __restrict__ sinks, float scale,
                                                                  float* __restrict__ ctx) {
    __shared__ float s_q[kRow];
    __shared__ float s_p[kMaxCells];
    __shared__ int s_row[kMaxCells];
    __shared__ float sred[32];
    const int h = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5, nw = blockDim.x >> 5;
    for (int i = t; i < kLat; i += blockDim.x) s_q[i] = qabs[(size_t) h * kLat + i];
    for (int i = t; i < kRot; i += blockDim.x) s_q[kLat + i] = q[(size_t) h * (kNope + kRot) + kNope + i];
    for (int i = t; i < n_cells; i += blockDim.x) {
        const int64_t p = cells != nullptr ? (int64_t) cells[i] : lo + i;
        s_row[i] = p < 0 ? -1 : (int) (p % ring);
    }
    __syncthreads();
    for (int i = warp; i < n_cells; i += nw) {
        const int r = s_row[i];
        float dot = -INFINITY;
        if (r >= 0) {
            const __half2* kr = reinterpret_cast<const __half2*>(cache + (size_t) r * kRow);
            float acc = 0.0f;
#pragma unroll
            for (int k = 0; k < kRow / 64; ++k) {
                const int e = lane + 32 * k;
                const float2 f = __half22float2(kr[e]);
                acc += f.x * s_q[2 * e] + f.y * s_q[2 * e + 1];
            }
            dot = warp_sum(acc) * scale;
        }
        if (lane == 0) s_p[i] = dot;
    }
    __syncthreads();
    const float sink = sinks[h];
    float mx = sink;
    for (int i = t; i < n_cells; i += blockDim.x) mx = fmaxf(mx, s_p[i]);
    mx = block_max(mx, sred);
    float den = 0.0f;
    for (int i = t; i < n_cells; i += blockDim.x) {
        const float e = s_p[i] == -INFINITY ? 0.0f : expf(s_p[i] - mx);
        s_p[i] = e;
        den += e;
    }
    den = block_sum(den, sred) + expf(sink - mx);
    const float inv = 1.0f / den;
    __syncthreads();
    for (int c = t; c < kLat; c += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < n_cells; ++i) {
            const int r = s_row[i];
            if (r >= 0) acc += s_p[i] * h2f(cache[(size_t) r * kRow + c]);
        }
        ctx[(size_t) h * kLat + c] = acc * inv;
    }
}

// a Q6_K block (256 values, 210 bytes: ql[128], qh[64], int8 scales[16], FP16 d): value i
__device__ __forceinline__ float q6_k_val(const uint8_t* blk, int i) {
    const int n = i >> 7, r = i & 127, grp = r >> 5, l = r & 31;
    const uint8_t* ql = blk + n * 64;
    const uint8_t* qh = blk + 128 + n * 32;
    const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192) + n * 8;
    const uint16_t dh = (uint16_t) blk[208] | ((uint16_t) blk[209] << 8);
    const int lo4 = (grp & 1) ? ql[l + 32] : ql[l];
    const int nib = (grp >> 1) ? (lo4 >> 4) : (lo4 & 15);
    const int hi = (qh[l] >> (2 * grp)) & 3;
    const int qv = (nib | (hi << 4)) - 32;
    return h2f(dh) * (float) sc[(l >> 4) + 2 * grp] * (float) qv;
}

__global__ void __launch_bounds__(256) mla_out_kernel(const float* __restrict__ ctx, const uint8_t* __restrict__ vb,
                                                      const float* __restrict__ gate, int n_head, float* __restrict__ o) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n_head * kV) return;
    const int h = row / kV;
    const uint8_t* w = vb + (size_t) row * (kLat / 256) * 210;
    const float* c = ctx + (size_t) h * kLat;
    float acc = 0.0f;
    for (int e = lane; e < kLat; e += 32) acc += q6_k_val(w + (e >> 8) * 210, e & 255) * c[e];
    acc = warp_sum(acc);
    if (lane == 0) o[row] = acc * sigm(gate[row]);
}

__global__ void __launch_bounds__(256) mla_out_f32_kernel(const float* __restrict__ ctx, const float* __restrict__ vb,
                                                          const float* __restrict__ gate, int n_head,
                                                          float* __restrict__ o) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n_head * kV) return;
    const int h = row / kV;
    const float* w = vb + (size_t) row * kLat;
    const float* c = ctx + (size_t) h * kLat;
    float acc = 0.0f;
    for (int e = lane; e < kLat; e += 32) acc += w[e] * c[e];
    acc = warp_sum(acc);
    if (lane == 0) o[row] = acc * sigm(gate[row]);
}

// ---- the indexer
// the 4 token keys' rows: tk[m] (slot m of the group)
__device__ __forceinline__ void idx_pool_body(const float* const* tk, const float* __restrict__ wv,
                                              const float* __restrict__ wg, const float* __restrict__ ape,
                                              const float* __restrict__ norm, float eps, Pos3 pos0, float base,
                                              float* __restrict__ pooled) {
    constexpr int K = 128;
    __shared__ float s_tk[4 * K];
    __shared__ float s_out[K];
    __shared__ float sred[32];
    const int c = threadIdx.x;
    for (int i = c; i < 4 * K; i += blockDim.x) s_tk[i] = tk[i / K][i % K];
    __syncthreads();
    float val[4], gt[4];
    for (int j = 0; j < 4; ++j) {
        float a = 0.0f, b = 0.0f;
        for (int e = 0; e < K; ++e) {
            a += wv[(size_t) c * K + e] * s_tk[j * K + e];
            b += wg[(size_t) c * K + e] * s_tk[j * K + e];
        }
        val[j] = a;
        gt[j] = b + ape[(size_t) j * K + c];
    }
    float mx = fmaxf(fmaxf(gt[0], gt[1]), fmaxf(gt[2], gt[3]));
    float den = 0.0f, acc = 0.0f;
    for (int j = 0; j < 4; ++j) {
        const float e = expf(gt[j] - mx);
        den += e;
        acc += e * val[j];
    }
    acc /= den;
    const float ss = block_sum(acc * acc, sred);
    s_out[c] = acc * rsqrtf(ss / (float) K + eps) * norm[c];
    __syncthreads();
    if (c < kRot / 2) {
        float x0 = s_out[K - kRot + 2 * c], x1 = s_out[K - kRot + 2 * c + 1];
        rot_pair(x0, x1, c, pos0, base);
        s_out[K - kRot + 2 * c] = x0;
        s_out[K - kRot + 2 * c + 1] = x1;
    }
    __syncthreads();
    pooled[c] = s_out[c];
}
__global__ void __launch_bounds__(128) idx_pool_kernel(const float* __restrict__ tk4, const float* __restrict__ wv,
                                                       const float* __restrict__ wg, const float* __restrict__ ape,
                                                       const float* __restrict__ norm, float eps, Pos3 pos0, float base,
                                                       float* __restrict__ pooled) {
    const float* tk[4] = {tk4, tk4 + 128, tk4 + 256, tk4 + 384};
    idx_pool_body(tk, wv, wg, ape, norm, eps, pos0, base, pooled);
}
// a prompt sub-batch (positions pt0 .. pt0 + tn): group g0 + blockIdx.x, its keys from tkb (rows of the sub-batch) or,
// before pt0, from tk4's slots; pos[i] is position pt0 - 3 + i
__global__ void __launch_bounds__(128) idx_pool_rows_kernel(const float* __restrict__ tkb, int64_t pt0,
                                                            const float* __restrict__ tk4, const float* __restrict__ wv,
                                                            const float* __restrict__ wg, const float* __restrict__ ape,
                                                            const float* __restrict__ norm, float eps,
                                                            const Pos3* __restrict__ pos, float base, int64_t g0,
                                                            float* __restrict__ pooled) {
    const int64_t g = g0 + blockIdx.x;
    const float* tk[4];
    for (int m = 0; m < 4; ++m) {
        const int64_t j = 4 * g + m;
        tk[m] = j >= pt0 ? tkb + (size_t) (j - pt0) * 128 : tk4 + (size_t) (j % 4) * 128;
    }
    idx_pool_body(tk, wv, wg, ape, norm, eps, pos[4 * g - (pt0 - 3)], base, pooled + (size_t) g * 128);
}
// the sub-batch's last positions' token keys into tk4's slots (after the pools read the older ones)
__global__ void idx_tk4_kernel(const float* __restrict__ tkb, int64_t pt0, int tn, float* __restrict__ tk4) {
    const int64_t j0 = pt0 + (tn > 4 ? tn - 4 : 0);
    for (int i = threadIdx.x; i < 4 * 128; i += blockDim.x) {
        const int64_t j = j0 + i / 128;
        if (j < pt0 + tn) tk4[(size_t) (j % 4) * 128 + i % 128] = tkb[(size_t) (j - pt0) * 128 + i % 128];
    }
}

__device__ __forceinline__ void idx_query_body(float* __restrict__ iq, int heads, float* __restrict__ w, Pos3 pos,
                                               float base) {
    constexpr int K = 128;
    for (int i = threadIdx.x; i < heads * (kRot / 2); i += blockDim.x) {
        const int h = i / (kRot / 2), j = i % (kRot / 2);
        float* r = iq + (size_t) h * K + (K - kRot);
        float x0 = r[2 * j], x1 = r[2 * j + 1];
        rot_pair(x0, x1, j, pos, base);
        r[2 * j] = x0;
        r[2 * j + 1] = x1;
    }
    const float sc = rsqrtf((float) heads) * rsqrtf((float) K);
    for (int h = threadIdx.x; h < heads; h += blockDim.x) w[h] *= sc;
}
__global__ void idx_query_kernel(float* __restrict__ iq, int heads, float* __restrict__ w, Pos3 pos, float base) {
    idx_query_body(iq, heads, w, pos, base);
}
__global__ void idx_query_rows_kernel(float* __restrict__ iq, int heads, float* __restrict__ w,
                                      const Pos3* __restrict__ pos, float base) {
    const int t = blockIdx.x;
    idx_query_body(iq + (size_t) t * heads * 128, heads, w + (size_t) t * heads, pos[t], base);
}

__global__ void __launch_bounds__(256) idx_score_kernel(const float* __restrict__ iq, const float* __restrict__ w,
                                                        int heads, const float* __restrict__ pooled, int n_groups,
                                                        float* __restrict__ score) {
    constexpr int K = 128;
    __shared__ float s_iq[32 * K];
    __shared__ float s_w[32];
    for (int i = threadIdx.x; i < heads * K; i += blockDim.x) s_iq[i] = iq[i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) s_w[i] = w[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (g >= n_groups) return;
    const float* pk = pooled + (size_t) g * K;
    float pv[K / 32];
#pragma unroll
    for (int k = 0; k < K / 32; ++k) pv[k] = pk[lane + 32 * k];
    float acc = 0.0f;
    for (int h = 0; h < heads; ++h) {
        float d = 0.0f;
#pragma unroll
        for (int k = 0; k < K / 32; ++k) d += s_iq[h * K + lane + 32 * k] * pv[k];
        d = warp_sum(d);
        acc += s_w[h] * fmaxf(d, 0.0f);
    }
    if (lane == 0) score[g] = acc;
}

__global__ void __launch_bounds__(1024) idx_select_kernel(const float* __restrict__ score, int n_vis, int top,
                                                          int64_t pos, int* __restrict__ cells, int* __restrict__ n_out) {
    const int k = min(top, n_vis);
    if (k > 0) strata::kernels::dsa::select_top_pools(score, n_vis, k, 4, cells);
    __syncthreads();
    if (threadIdx.x == 0) {
        const int64_t own = pos / 4;
        for (int m = 0; m < 4; ++m) cells[k * 4 + m] = own * 4 + m <= pos ? (int) (own * 4 + m) : -1;
        *n_out = k * 4 + 4;
    }
}

// ---- Engram
__global__ void __launch_bounds__(128) eng_rows_kernel(const float* __restrict__ rows, const float* __restrict__ lnorm,
                                                       int d, float row_eps, float* __restrict__ e, int ln_heads) {
    __shared__ float sred[32];
    const int h = blockIdx.x, t = threadIdx.x;   // (a prompt chunk: rows of ln_heads heads one after the other)
    const float v = t < d ? rows[(size_t) h * d + t] : 0.0f;
    const float ss = block_sum(v * v, sred);
    if (t < d) e[(size_t) h * d + t] = v * rsqrtf(ss / (float) d + row_eps) * lnorm[(size_t) (h % ln_heads) * d + t];
}

__global__ void __launch_bounds__(1024) eng_add_kernel(const float* __restrict__ kv, float* __restrict__ R,
                                                       const float* __restrict__ gate, const float* __restrict__ hc_scale,
                                                       const float* __restrict__ pre_gamma, int hc, int dim, float eps,
                                                       bool vfirst, const unsigned char* __restrict__ skip) {
    __shared__ float sred[32];
    const int c = blockIdx.x;
    if (skip != nullptr && skip[blockIdx.y]) return;   // (a prompt chunk's row: blockIdx.y; an image / pad row adds nothing)
    kv += (size_t) blockIdx.y * (hc + 1) * dim;
    R += (size_t) blockIdx.y * hc * dim;
    const float* val = vfirst ? kv : kv + (size_t) hc * dim;
    const float* key = vfirst ? kv + (size_t) (1 + c) * dim : kv + (size_t) c * dim;
    float* Rc = R + (size_t) c * dim;
    const float* gw = gate + (size_t) c * dim;
    float sv = 0.0f, sk = 0.0f, sq = 0.0f, kq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = val[i], k = key[i], q = Rc[i];
        sv += v * v;
        sk += k * k;
        sq += q * q;
        kq += k * q * gw[i];
    }
    sv = block_sum(sv, sred);
    sk = block_sum(sk, sred);
    sq = block_sum(sq, sred);
    kq = block_sum(kq, sred);
    const float rv = rsqrtf(sv / (float) dim + eps);
    float gl = kq * rsqrtf(sk / (float) dim + eps) * rsqrtf(sq / (float) dim + eps) * rsqrtf((float) dim);
    gl = gl == 0.0f ? 0.0f : copysignf(sqrtf(fmaxf(fabsf(gl), 1e-6f)), gl);   // (sign(0) = 0)
    const float g = sigm(gl);
    __syncthreads();
    const float* sc = hc_scale + (size_t) c * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float m = pre_gamma[i] * val[i] * rv;
        Rc[i] += g * m + sc[i] * m;
    }
}

__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, int n, float* __restrict__ h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float a = g[i];
        h[i] = a / (1.0f + expf(-a)) * u[i];
    }
}

__global__ void e4m3_kernel(const uint8_t* __restrict__ src, float* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = src[i], e = (c >> 3) & 15, m = c & 7;
    float v = e == 0 ? (float) m * 0.001953125f : ldexpf((float) (8 + m), e - 10);
    if (e == 15 && m == 7) v = 0.0f;   // (NaN in E4M3FN: never written by a quantizer; read as 0)
    dst[i] = (c & 128) ? -v : v;
}

// ---- the prompt path's own pieces (a chunk's rows at once)
// The attention of T query rows (positions p0 + t), block = (row, 16 heads), 8 warps of 2 heads: the row's keys in
// chunks of 32 cache rows (FP16 in shared memory, as in the cache), scores over the 576 dims (scale * (q_abs . lat +
// q_rot . k_rot)) and an online softmax per head that starts from the head's sink (m = sink, l = 1: the token path's
// denominator term), thread c owning the context columns c and c + 256 of the 16 heads.  A row's keys: cells[t] when
// n_cells[t] >= 0, else positions [lo, p] (lo = p - window + 1, at least 0; window 0: from 0).  Position j lives in
// cache row (j - row_base) % ring.  F32 arithmetic on the FP16 rows - the token kernel's, in another summing order.
constexpr int AR_HG = 16, AR_CH = 32;
__global__ void __launch_bounds__(256) attend_rows_kernel(const float* __restrict__ qabs, const float* __restrict__ q,
                                                          const uint16_t* __restrict__ cache, int64_t row_base, int ring,
                                                          const int* __restrict__ cells, const int* __restrict__ n_cells,
                                                          int cells_ld, int64_t p0, int window,
                                                          const float* __restrict__ sinks, float scale, int n_head,
                                                          float* __restrict__ ctx) {
    __shared__ __align__(16) uint16_t sK[AR_CH * kRow];
    __shared__ float sP[AR_HG * AR_CH];
    __shared__ float s_m[AR_HG], s_l[AR_HG], s_sc[AR_HG];
    __shared__ int s_row[AR_CH];
    const int t = blockIdx.x, h0 = blockIdx.y * AR_HG, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int64_t p = p0 + t;
    const int nc = cells != nullptr ? n_cells[t] : -1;
    const int* cl = cells != nullptr ? cells + (size_t) t * cells_ld : nullptr;
    const int64_t lo = window > 0 ? (p - window + 1 > 0 ? p - window + 1 : 0) : 0;
    const int ns = nc >= 0 ? nc : (int) (p - lo + 1);
    // the warp's 2 heads: dims lane + 32 j - j < 16 the absorbed latent query, 16 and 17 the rope part
    constexpr int QJ = kRow / 32;
    float q0[QJ], q1[QJ];
    {
        const int ha = h0 + 2 * warp;
        const float* qa = qabs + ((size_t) t * n_head + ha) * kLat;
        const float* qr = q + ((size_t) t * n_head + ha) * (kNope + kRot) + kNope;
#pragma unroll
        for (int j = 0; j < kLat / 32; ++j) {
            q0[j] = qa[lane + 32 * j];
            q1[j] = qa[kLat + lane + 32 * j];
        }
#pragma unroll
        for (int j = kLat / 32; j < QJ; ++j) {
            q0[j] = qr[lane + 32 * j - kLat];
            q1[j] = qr[kNope + kRot + lane + 32 * j - kLat];
        }
    }
    float acc0[AR_HG], acc1[AR_HG];
#pragma unroll
    for (int hh = 0; hh < AR_HG; ++hh) acc0[hh] = acc1[hh] = 0.0f;
    if (tid < AR_HG) {
        s_m[tid] = sinks[h0 + tid];
        s_l[tid] = 1.0f;
    }
    for (int s0 = 0; s0 < ns; s0 += AR_CH) {
        __syncthreads();
        if (tid < AR_CH) {
            const int i = s0 + tid;
            int64_t j = -1;
            if (i < ns) j = nc >= 0 ? (int64_t) cl[i] : lo + i;
            s_row[tid] = j < 0 ? -1 : (int) ((j - row_base) % ring);
        }
        __syncthreads();
        for (int i = tid; i < AR_CH * (kRow / 8); i += blockDim.x) {
            const int r = i / (kRow / 8), c8 = i - r * (kRow / 8);
            const int row = s_row[r];
            ((uint4*) sK)[i] = row < 0 ? make_uint4(0u, 0u, 0u, 0u) : ((const uint4*) (cache + (size_t) row * kRow))[c8];
        }
        __syncthreads();
        for (int sl = 0; sl < AR_CH; ++sl) {
            float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
            for (int j = 0; j < QJ; ++j) {
                const float l = h2f(sK[sl * kRow + lane + 32 * j]);
                a0 += q0[j] * l;
                a1 += q1[j] * l;
            }
            a0 = warp_sum(a0);
            a1 = warp_sum(a1);
            if (lane == 0) {
                const bool ok = s_row[sl] >= 0;
                sP[(2 * warp) * AR_CH + sl] = ok ? a0 * scale : -INFINITY;
                sP[(2 * warp + 1) * AR_CH + sl] = ok ? a1 * scale : -INFINITY;
            }
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int h = 2 * warp + hh;
            const float v = sP[h * AR_CH + lane];
            const float m_old = s_m[h];
            const float m_new = fmaxf(m_old, warp_max(v));
            const float e = v == -INFINITY ? 0.0f : expf(v - m_new);
            const float l = warp_sum(e);
            sP[h * AR_CH + lane] = e;
            __syncwarp();
            if (lane == 0) {
                const float sc = expf(m_old - m_new);
                s_sc[h] = sc;
                s_l[h] = s_l[h] * sc + l;
                s_m[h] = m_new;
            }
        }
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < AR_HG; ++hh) {
            const float sc = s_sc[hh];
            acc0[hh] *= sc;
            acc1[hh] *= sc;
        }
        for (int sl = 0; sl < AR_CH; ++sl) {
            const float l0 = h2f(sK[sl * kRow + tid]), l1 = h2f(sK[sl * kRow + tid + 256]);
#pragma unroll
            for (int hh = 0; hh < AR_HG; ++hh) {
                const float pp = sP[hh * AR_CH + sl];
                acc0[hh] += pp * l0;
                acc1[hh] += pp * l1;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int hh = 0; hh < AR_HG; ++hh) {
        const float inv = 1.0f / s_l[hh];
        float* o = ctx + ((size_t) t * n_head + h0 + hh) * kLat;
        o[tid] = acc0[hh] * inv;
        o[tid + 256] = acc1[hh] * inv;
    }
}

// q's no-rope parts (each head's first 192 of 256) -> FP16 [t][h][192] (the absorb product's input)
__global__ void q_nope_f16_kernel(const float* __restrict__ q, int64_t n, uint16_t* __restrict__ q16) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t r = i / kNope, k = i - r * kNope;
        q16[i] = f2h(q[r * (kNope + kRot) + k]);
    }
}
// o *= sigmoid(gate), o16 = FP16(o)
__global__ void gate_rows_kernel(float* __restrict__ o, const float* __restrict__ gate, int64_t n,
                                 uint16_t* __restrict__ o16) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float v = o[i] * sigm(gate[i]);
        o[i] = v;
        o16[i] = f2h(v);
    }
}
// cache rows of positions [j0, j0 + n): src row (j - sb) % sr -> dst row (j - db) % dr (576 FP16 each)
__global__ void ring_copy_kernel(const uint16_t* __restrict__ src, int64_t sb, int sr, uint16_t* __restrict__ dst,
                                 int64_t db, int dr, int64_t j0) {
    const int64_t j = j0 + blockIdx.x;
    const uint4* a = (const uint4*) (src + (size_t) ((j - sb) % sr) * kRow);
    uint4* b = (uint4*) (dst + (size_t) ((j - db) % dr) * kRow);
    for (int i = threadIdx.x; i < kRow / 8; i += blockDim.x) b[i] = a[i];
}
// the indexer's selection per row (idx_select_kernel's): rows up to dense positions attend densely (n_cells -1), the
// others the top min(top, own) groups of score[t][0, own) and their own group's cells (own = p / 4)
__global__ void __launch_bounds__(1024) idx_select_rows_kernel(const float* __restrict__ score, int score_ld, int64_t p0,
                                                               int dense, int top, int* __restrict__ cells, int cells_ld,
                                                               int* __restrict__ n_cells) {
    const int t = blockIdx.x;
    const int64_t pos = p0 + t;
    if (pos + 1 <= dense) {
        if (threadIdx.x == 0) n_cells[t] = -1;
        return;
    }
    const int own = (int) (pos / 4);
    const int k = min(top, own);
    int* c = cells + (size_t) t * cells_ld;
    if (k > 0) strata::kernels::dsa::select_top_pools(score + (size_t) t * score_ld, own, k, 4, c);
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int m = 0; m < 4; ++m) c[k * 4 + m] = (int64_t) own * 4 + m <= pos ? own * 4 + m : -1;
        n_cells[t] = k * 4 + 4;
    }
}
// y[t][i] += b[i] (n a row)
__global__ void add_bias_kernel(float* __restrict__ y, const float* __restrict__ b, int n, int64_t total) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < total; i += (int64_t) gridDim.x * blockDim.x)
        y[i] += b[i % n];
}

int g_launch_errors = 0;
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        ++g_launch_errors;
        std::fprintf(stderr, "hy4 kernel %s: %s\n", what, cudaGetErrorString(e));
    }
}

}  // namespace

void hc_mix(const float* R, const float* fn, int hc, int dim, int n_mix, float eps, float* mixes, cudaStream_t s) {
    hc_mix_kernel<<<n_mix, 512, 0, s>>>(R, fn, hc * dim, eps, mixes);
    check("hc_mix");
}
void hc_pre(const float* mixes, const float* scale, const float* base, const float* R, int hc, int dim, float hc_eps,
            float mag, const float* norm_w, float eps, float* x, float* post, cudaStream_t s) {
    hc_pre_kernel<<<1, 1024, 0, s>>>(mixes, scale, base, R, hc, dim, hc_eps, mag, norm_w, eps, x, post, 0);
    check("hc_pre");
}
void hc_add(float* R, const float* post, float* a, const float* a_norm, float eps, int hc, int dim, cudaStream_t s) {
    hc_add_kernel<<<1, 1024, 0, s>>>(R, post, a, a_norm, eps, hc, dim);
    check("hc_add");
}
void rms(const float* x, const float* w, float eps, int n, float* y, cudaStream_t s, const float* add) {
    rms_kernel<<<1, 1024, 0, s>>>(x, w, eps, n, y, add);
    check("rms");
}
void mla_prep(float* q, int n_head, const float* kv, const float* kv_norm, float eps, Pos3 pos, float base,
              uint16_t* cache_row, cudaStream_t s) {
    mla_prep_kernel<<<1, 512, 0, s>>>(q, n_head, kv, kv_norm, eps, pos, base, cache_row);
    check("mla_prep");
}
void mla_absorb(const float* q, const void* k_b, int n_head, float* qabs, cudaStream_t s, bool f32) {
    const int rows = n_head * kLat;
    if (f32)
        mla_absorb_f32_kernel<<<(rows + 7) / 8, 256, 0, s>>>(q, (const float*) k_b, n_head, qabs);
    else
        mla_absorb_kernel<<<(rows + 7) / 8, 256, 0, s>>>(q, (const uint8_t*) k_b, n_head, qabs);
    check("mla_absorb");
}
void mla_attend(const float* qabs, const float* q, const uint16_t* cache, int ring, const int* cells, int n_cells,
                int64_t lo, int64_t hi, const float* sinks, float scale, int n_head, float* ctx, cudaStream_t s) {
    const int n = cells != nullptr ? n_cells : (int) (hi - lo + 1);
    if (n > kMaxCells || n < 1) {
        ++g_launch_errors;
        std::fprintf(stderr, "hy4 mla_attend: %d keys (1..%d)\n", n, kMaxCells);
        return;
    }
    mla_attend_kernel<<<n_head, kAttnThreads, 0, s>>>(qabs, q, cache, ring, cells, n, lo, sinks, scale, ctx);
    check("mla_attend");
}
void mla_out(const float* ctx, const void* v_b, const float* gate, int n_head, float* o, cudaStream_t s, bool f32) {
    const int rows = n_head * kV;
    if (f32)
        mla_out_f32_kernel<<<(rows + 7) / 8, 256, 0, s>>>(ctx, (const float*) v_b, gate, n_head, o);
    else
        mla_out_kernel<<<(rows + 7) / 8, 256, 0, s>>>(ctx, (const uint8_t*) v_b, gate, n_head, o);
    check("mla_out");
}
void idx_pool(const float* tk4, const float* wv, const float* wg, const float* ape, const float* norm, float eps,
              Pos3 pos0, float base, float* pooled, cudaStream_t s) {
    idx_pool_kernel<<<1, 128, 0, s>>>(tk4, wv, wg, ape, norm, eps, pos0, base, pooled);
    check("idx_pool");
}
void idx_query(float* iq, int heads, float* w, Pos3 pos, float base, cudaStream_t s) {
    idx_query_kernel<<<1, 512, 0, s>>>(iq, heads, w, pos, base);
    check("idx_query");
}
void idx_score(const float* iq, const float* w, int heads, const float* pooled, int n_groups, float* score,
               cudaStream_t s) {
    if (n_groups <= 0) return;
    idx_score_kernel<<<(n_groups + 7) / 8, 256, 0, s>>>(iq, w, heads, pooled, n_groups, score);
    check("idx_score");
}
void idx_select(const float* score, int n_vis, int top, int64_t pos, int* cells, int* n_out, cudaStream_t s) {
    idx_select_kernel<<<1, 1024, 0, s>>>(score, n_vis, top, pos, cells, n_out);
    check("idx_select");
}
void eng_rows(const float* rows, const float* lnorm, int n_heads, int d, float row_eps, float* e, cudaStream_t s) {
    eng_rows_kernel<<<n_heads, 128, 0, s>>>(rows, lnorm, d, row_eps, e, n_heads);
    check("eng_rows");
}
void eng_add(const float* kv, float* R, const float* gate, const float* hc_scale, const float* pre_gamma, int hc,
             int dim, float eps, bool vfirst, cudaStream_t s) {
    eng_add_kernel<<<hc, 1024, 0, s>>>(kv, R, gate, hc_scale, pre_gamma, hc, dim, eps, vfirst, nullptr);
    check("eng_add");
}
void swiglu(const float* g, const float* u, int n, float* h, cudaStream_t s) {
    swiglu_kernel<<<(n + 255) / 256, 256, 0, s>>>(g, u, n, h);
    check("swiglu");
}
void e4m3_to_f32(const uint8_t* src, float* dst, int n, cudaStream_t s) {
    e4m3_kernel<<<(n + 255) / 256, 256, 0, s>>>(src, dst, n);
    check("e4m3");
}

// ---- the prompt path
static unsigned grid_for(int64_t n) { return (unsigned) std::min<int64_t>(4096, (n + 255) / 256); }
void hc_mix_rows(const float* R, const float* fn, int hc, int dim, int n_mix, float eps, float* mixes, int T,
                 cudaStream_t s) {
    if (T <= 0) return;
    hc_mix_kernel<<<dim3((unsigned) n_mix, (unsigned) T), 512, 0, s>>>(R, fn, hc * dim, eps, mixes);
    check("hc_mix_rows");
}
void hc_pre_rows(const float* mixes, int mix_ld, const float* scale, const float* base, const float* R, int hc, int dim,
                 float hc_eps, float mag, const float* norm_w, float eps, float* x, float* post, int T, cudaStream_t s) {
    if (T <= 0) return;
    hc_pre_kernel<<<(unsigned) T, 1024, 0, s>>>(mixes, scale, base, R, hc, dim, hc_eps, mag, norm_w, eps, x, post, mix_ld);
    check("hc_pre_rows");
}
void hc_add_rows(float* R, const float* post, float* a, const float* a_norm, float eps, int hc, int dim, int T,
                 cudaStream_t s) {
    if (T <= 0) return;
    hc_add_kernel<<<(unsigned) T, 1024, 0, s>>>(R, post, a, a_norm, eps, hc, dim);
    check("hc_add_rows");
}
void rms_rows(const float* x, const float* w, float eps, int n, float* y, int T, cudaStream_t s, const float* add) {
    if (T <= 0) return;
    rms_kernel<<<(unsigned) T, 1024, 0, s>>>(x, w, eps, n, y, add);
    check("rms_rows");
}
void mla_prep_rows(float* q, int n_head, const float* kv, const float* kv_norm, float eps, const Pos3* pos, int64_t p0,
                   float base, uint16_t* cache, int64_t row_base, int ring, int T, cudaStream_t s) {
    if (T <= 0) return;
    mla_prep_rows_kernel<<<(unsigned) T, 512, 0, s>>>(q, n_head, kv, kv_norm, eps, pos, p0, base, cache, row_base, ring);
    check("mla_prep_rows");
}
void q_nope_f16(const float* q, int n_head, int T, uint16_t* q16, cudaStream_t s) {
    const int64_t n = (int64_t) T * n_head * kNope;
    if (n <= 0) return;
    q_nope_f16_kernel<<<grid_for(n), 256, 0, s>>>(q, n, q16);
    check("q_nope_f16");
}
void mla_attend_rows(const float* qabs, const float* q, const uint16_t* cache, int64_t row_base, int ring,
                     const int* cells, const int* n_cells, int cells_ld, int64_t p0, int window, const float* sinks,
                     float scale, int n_head, int T, float* ctx, cudaStream_t s) {
    if (T <= 0) return;
    if (n_head % AR_HG != 0) {
        ++g_launch_errors;
        std::fprintf(stderr, "hy4 mla_attend_rows: %d heads (a multiple of %d)\n", n_head, AR_HG);
        return;
    }
    attend_rows_kernel<<<dim3((unsigned) T, (unsigned) (n_head / AR_HG)), 256, 0, s>>>(
        qabs, q, cache, row_base, ring, cells, n_cells, cells_ld, p0, window, sinks, scale, n_head, ctx);
    check("mla_attend_rows");
}
void gate_rows(float* o, const float* gate, int64_t n, uint16_t* o16, cudaStream_t s) {
    if (n <= 0) return;
    gate_rows_kernel<<<grid_for(n), 256, 0, s>>>(o, gate, n, o16);
    check("gate_rows");
}
void ring_copy(const uint16_t* src, int64_t src_base, int src_ring, uint16_t* dst, int64_t dst_base, int dst_ring,
               int64_t j0, int n, cudaStream_t s) {
    if (n <= 0) return;
    ring_copy_kernel<<<(unsigned) n, 128, 0, s>>>(src, src_base, src_ring, dst, dst_base, dst_ring, j0);
    check("ring_copy");
}
void idx_pool_rows(const float* tkb, int64_t pt0, int tn, float* tk4, const float* wv, const float* wg,
                   const float* ape, const float* norm, float eps, const Pos3* pos, float base, float* pooled,
                   cudaStream_t s) {
    if (tn <= 0) return;
    const int64_t g0 = pt0 / 4, g1 = (pt0 + tn) / 4;
    if (g1 > g0) {
        idx_pool_rows_kernel<<<(unsigned) (g1 - g0), 128, 0, s>>>(tkb, pt0, tk4, wv, wg, ape, norm, eps, pos, base, g0,
                                                                   pooled);
        check("idx_pool_rows");
    }
    idx_tk4_kernel<<<1, 512, 0, s>>>(tkb, pt0, tn, tk4);
    check("idx_tk4");
}
void idx_query_rows(float* iq, int heads, float* w, const Pos3* pos, float base, int T, cudaStream_t s) {
    if (T <= 0) return;
    idx_query_rows_kernel<<<(unsigned) T, 512, 0, s>>>(iq, heads, w, pos, base);
    check("idx_query_rows");
}
void idx_select_rows(const float* score, int score_ld, int64_t p0, int dense, int top, int* cells, int cells_ld,
                     int* n_cells, int T, cudaStream_t s) {
    if (T <= 0) return;
    idx_select_rows_kernel<<<(unsigned) T, 1024, 0, s>>>(score, score_ld, p0, dense, top, cells, cells_ld, n_cells);
    check("idx_select_rows");
}
void eng_rows_rows(const float* rows, const float* lnorm, int n_heads, int d, float row_eps, float* e, int T,
                   cudaStream_t s) {
    if (T <= 0) return;
    eng_rows_kernel<<<(unsigned) (n_heads * T), 128, 0, s>>>(rows, lnorm, d, row_eps, e, n_heads);
    check("eng_rows_rows");
}
void eng_add_rows(const float* kv, float* R, const float* gate, const float* hc_scale, const float* pre_gamma, int hc,
                  int dim, float eps, bool vfirst, const unsigned char* skip, int T, cudaStream_t s) {
    if (T <= 0) return;
    eng_add_kernel<<<dim3((unsigned) hc, (unsigned) T), 1024, 0, s>>>(kv, R, gate, hc_scale, pre_gamma, hc, dim, eps,
                                                                       vfirst, skip);
    check("eng_add_rows");
}
void add_bias_rows(float* y, const float* b, int n, int T, cudaStream_t s) {
    const int64_t total = (int64_t) n * T;
    if (total <= 0) return;
    add_bias_kernel<<<grid_for(total), 256, 0, s>>>(y, b, n, total);
    check("add_bias_rows");
}
void swiglu_n(const float* g, const float* u, int64_t n, float* h, cudaStream_t s) {
    for (int64_t i0 = 0; i0 < n; i0 += (int64_t) 1 << 30) {
        const int m = (int) std::min<int64_t>((int64_t) 1 << 30, n - i0);
        swiglu_kernel<<<(m + 255) / 256, 256, 0, s>>>(g + i0, u + i0, m, h + i0);
    }
    check("swiglu_n");
}
int launch_errors() { return g_launch_errors; }

}  // namespace strata::kernels::hy4
