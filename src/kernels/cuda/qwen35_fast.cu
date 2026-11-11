// src/kernels/cuda/qwen35_fast.cu - see include/strata/kernels/qwen35_fast.hpp.
#if defined(STRATA_USE_HIP)
// (as glm_batch.cu: the headers rocWMMA pulls in define the __shfl_*_sync names the hip_compat macros remap)
#pragma push_macro("__shfl_xor_sync")
#undef __shfl_xor_sync
#pragma push_macro("__shfl_down_sync")
#undef __shfl_down_sync
#pragma push_macro("__shfl_up_sync")
#undef __shfl_up_sync
#pragma push_macro("__shfl_sync")
#undef __shfl_sync
#include <rocwmma/rocwmma.hpp>
#pragma pop_macro("__shfl_xor_sync")
#pragma pop_macro("__shfl_down_sync")
#pragma pop_macro("__shfl_up_sync")
#pragma pop_macro("__shfl_sync")
#endif
#include "strata/kernels/qwen35_fast.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/pdl.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if !defined(STRATA_USE_HIP)
#include <mma.h>
#include <cooperative_groups.h>
#include <cuda_pipeline.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <type_traits>

namespace strata::kernels::q35 {
namespace {

constexpr int kD = 256;       // head_dim
constexpr int kChunk = 256;   // attention positions a block
constexpr int kWarps = 4;     // ... its warps
constexpr int kMinChunks = 130;   // the chunks a row and head attn_split's scratch always holds

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(expf(x)); }   // (ggml's)
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
// ggml's block_q8_1 (36 bytes: d and the sum as half2, 32 codes) and glm_fast's q8_1_store_warp - the same arithmetic
// as gf::quantize_q8_1, so a kernel that also writes the q8_1 of its output gives the same bytes as a quantize after it
struct Q81 {
    __half2 ds;
    int8_t qs[32];
};
__device__ __forceinline__ void q81_store_warp(float v, Q81* dst, int lane) {
    const float amax = warp_max(fabsf(v));
    const float sum = warp_sum(v);
    const float d = fminf(amax / 127.0f, 65504.0f);
    const int8_t q = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(v / d)));
    dst->qs[lane] = q;
    if (lane == 0) dst->ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}
// a 256-thread block's sum (sh: 8 floats)
__device__ __forceinline__ float block_sum256(float v, float* sh) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = threadIdx.x < 8 ? sh[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    return sh[0];
}
__device__ __forceinline__ float h2f(uint16_t u) { return __half2float(__ushort_as_half(u)); }
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }

// ---- --kv q4_0 (kv_q4.hpp's idea on this file's layout): a cache row (one position, one KV head) holds K or V
// rotated by the orthonormal 256-point Walsh-Hadamard H, in 8 groups of 32 dims, each ggml q4_0's arithmetic
// (scale = the value of largest |x| / -8, code = clamp(trunc(x / scale + 8.5), 0, 15), x ~ (code - 8) * scale), laid
// out for the kernels' lanes: the 8 FP16 scales first (16 B), then 32 words of 8 codes - word c the dims 8c .. 8c + 7,
// dim 8c + i in bits 4i .. 4i + 3 - so a lane's 8 dims are one aligned 32-bit load and one scale.  144 B a row (FP16:
// 512).  Q is rotated the same way (<Hq, Hk> = <q, k>) and the output rotated back (H is its own inverse).
constexpr int kQ4Row = 144;
// lane `col`'s 8 dims of cache row ch (= position * n_head_kv + KV head) of a q4_0 cache
__device__ __forceinline__ void q4_ld8(const uint16_t* c, size_t ch, int col, float* f) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(c) + ch * kQ4Row;
    const float d = h2f(reinterpret_cast<const uint16_t*>(b)[col >> 2]);
    const uint32_t w = reinterpret_cast<const uint32_t*>(b + 16)[col];
#pragma unroll
    for (int i = 0; i < 8; ++i) f[i] = (float) ((int) ((w >> (4 * i)) & 15u) - 8) * d;
}
// ... as 8 FP16 values (the kernels that stage FP16 tiles in shared memory)
__device__ __forceinline__ uint4 q4_ld8h(const uint16_t* c, size_t ch, int col) {
    float f[8];
    q4_ld8(c, ch, col, f);
    uint4 u;
    uint32_t* p = reinterpret_cast<uint32_t*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = (uint32_t) f2h(f[2 * i]) | ((uint32_t) f2h(f[2 * i + 1]) << 16);
    return u;
}
// H x over a 256-thread block (thread d holds sh[d]; every thread calls it); the result sh[d] * (1 / 16)
__device__ __forceinline__ void fwht256_sh(float* sh, int d) {
#pragma unroll
    for (int h = 1; h < 256; h <<= 1) {
        __syncthreads();
        const float a = sh[d], b = sh[d ^ h];
        __syncthreads();
        sh[d] = (d & h) ? b - a : a + b;
    }
    __syncthreads();
}
// thread d's value x (dim d of a rotated row, d < 256) into the q4_0 row at b: a warp is one group of 32
__device__ __forceinline__ void q4_put(uint8_t* b, int d, float x) {
    float amax = fabsf(x), mval = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {   // (equal |x|: the larger value, so every lane agrees on the scale)
        const float a = __shfl_xor_sync(0xffffffffu, amax, o), v = __shfl_xor_sync(0xffffffffu, mval, o);
        if (a > amax || (a == amax && v > mval)) {
            amax = a;
            mval = v;
        }
    }
    const float sc = mval / -8.0f, id = sc != 0.0f ? 1.0f / sc : 0.0f;
    const int q = min(15, max(0, __float2int_rz(x * id + 8.5f)));
    int w = q << (4 * (d & 7));
    w |= __shfl_xor_sync(0xffffffffu, w, 1);
    w |= __shfl_xor_sync(0xffffffffu, w, 2);
    w |= __shfl_xor_sync(0xffffffffu, w, 4);
    if ((d & 7) == 0) reinterpret_cast<uint32_t*>(b + 16)[d >> 3] = (uint32_t) w;
    if ((d & 31) == 0) reinterpret_cast<uint16_t*>(b)[d >> 5] = f2h(sc);
}
// --kv int8 (and k8v4's K): unrotated, 8 groups of 32 dims, scale = max |x| / 127, code = round(x / scale) - the 8 FP16
// scales, then the 256 int8 codes in dim order (a lane's 8 dims one aligned 64-bit load).  272 B a row
constexpr int kQ8Row = 272;
__device__ __forceinline__ void q8_ld8(const uint16_t* c, size_t ch, int col, float* f) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(c) + ch * kQ8Row;
    const float d = h2f(reinterpret_cast<const uint16_t*>(b)[col >> 2]);
    const uint2 w = reinterpret_cast<const uint2*>(b + 16)[col];
    const int8_t* q = reinterpret_cast<const int8_t*>(&w);
#pragma unroll
    for (int i = 0; i < 8; ++i) f[i] = (float) q[i] * d;
}
__device__ __forceinline__ void q8_put(uint8_t* b, int d, float x) {
    const float amax = warp_max(fabsf(x));
    const float sc = amax / 127.0f, id = sc != 0.0f ? 1.0f / sc : 0.0f;
    reinterpret_cast<int8_t*>(b + 16)[d] = (int8_t) max(-127, min(127, __float2int_rn(x * id)));
    if ((d & 31) == 0) reinterpret_cast<uint16_t*>(b)[d >> 5] = f2h(sc);
}
// The cache formats (the host's `kv` argument): 0 FP16, 1 q4_0 (K and V rotated), 2 int8, 3 k8v4 (K int8, V rotated
// q4_0).  A half's format: 0 FP16, 1 q4_0, 2 int8
__host__ __device__ constexpr int kv_kf(int kv) { return kv == 1 ? 1 : kv >= 2 ? 2 : 0; }
__host__ __device__ constexpr int kv_vf(int kv) { return kv == 1 || kv == 3 ? 1 : kv == 2 ? 2 : 0; }
__host__ __device__ constexpr int kv_row(int f) { return f == 1 ? kQ4Row : f == 2 ? kQ8Row : 2 * kD; }
template <int F>
__device__ __forceinline__ void kv_ld8(const uint16_t* c, size_t ch, int col, float* f) {
    if constexpr (F == 1) q4_ld8(c, ch, col, f);
    else if constexpr (F == 2) q8_ld8(c, ch, col, f);
    else {
        const __half2* h = reinterpret_cast<const __half2*>(c + ch * kD + col * 8);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 x = __half22float2(h[i]);
            f[2 * i] = x.x;
            f[2 * i + 1] = x.y;
        }
    }
}
template <int F>
__device__ __forceinline__ uint4 kv_ld8h(const uint16_t* c, size_t ch, int col) {
    if constexpr (F == 0) return *reinterpret_cast<const uint4*>(c + ch * kD + col * 8);
    float f[8];
    kv_ld8<F>(c, ch, col, f);
    uint4 u;
    uint32_t* p = reinterpret_cast<uint32_t*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[i] = (uint32_t) f2h(f[2 * i]) | ((uint32_t) f2h(f[2 * i + 1]) << 16);
    return u;
}

__global__ void silu_kernel(float* __restrict__ x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * sigmoid_f(x[i]);
}
__global__ void scale_kernel(float* __restrict__ x, int n, float a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
}
__global__ void gdn_gates_kernel(const float* __restrict__ alpha, const float* __restrict__ dt, const float* __restrict__ A,
                                 float* __restrict__ beta, float* __restrict__ gate, int h_v) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= h_v) return;
    gate[h] = softplus_f(alpha[h] + dt[h]) * A[h];
    beta[h] = sigmoid_f(beta[h]);
}
__global__ void scale_sigmoid_kernel(float* __restrict__ x, int n, const float* __restrict__ g) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= sigmoid_f(g[0]);
}
__global__ void scale_sigmoid_rows_kernel(float* __restrict__ x, int n, const float* __restrict__ g) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i < n) x[(size_t) t * n + i] *= sigmoid_f(g[t]);
}
__global__ void attn_gate_kernel(float* __restrict__ o, const float* __restrict__ qf, int n_head) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int h = blockIdx.x, d = threadIdx.x;
    if (h < n_head) o[h * kD + d] *= sigmoid_f(qf[(size_t) h * 2 * kD + kD + d]);
}

// block b < n_head: q head b (its q half of q_full); b >= n_head: KV head b - n_head.  256 threads, one a dim
__global__ void __launch_bounds__(kD) attn_prep_kernel(const float* __restrict__ qf, const float* __restrict__ k,
                                                       const float* __restrict__ v, const float* __restrict__ q_norm,
                                                       const float* __restrict__ k_norm, float eps, int n_head,
                                                       int n_head_kv, int n_rot, float rope_base, int p,
                                                       uint16_t* __restrict__ kc, uint16_t* __restrict__ vc,
                                                       float* __restrict__ q) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    __shared__ float sv[kD];
    __shared__ float sred[8];
    const int b = blockIdx.x, d = threadIdx.x;
    const bool is_q = b < n_head;
    const int g = is_q ? 0 : b - n_head;
    const float x = is_q ? qf[(size_t) b * 2 * kD + d] : k[(size_t) g * kD + d];
    const float ss = block_sum256(x * x, sred);
    const float inv = rsqrtf(ss / (float) kD + eps);
    const float xn = x * inv * (is_q ? q_norm[d] : k_norm[d]);
    sv[d] = xn;
    __syncthreads();
    float y = xn;
    if (d < n_rot) {   // NEOX pairs (i, i + n_rot / 2), theta_i = p * base^(-2 i / n_rot)
        const int half = n_rot / 2;
        const int i = d < half ? d : d - half;
        const float theta = (float) p * powf(rope_base, -2.0f * (float) i / (float) n_rot);
        const float c = cosf(theta), s = sinf(theta);
        const float x0 = sv[i], x1 = sv[i + half];
        y = d < half ? x0 * c - x1 * s : x0 * s + x1 * c;
    }
    if (is_q) {
        q[(size_t) b * kD + d] = y;
    } else {
        const size_t row = ((size_t) p * n_head_kv + g) * kD;
        kc[row + d] = f2h(y);
        vc[row + d] = f2h(v[(size_t) g * kD + d]);
    }
}

// grid (n_head, chunks), 128 threads: warp w takes positions t0 + w, t0 + w + 4, ..; a lane holds 8 dims of q (scaled)
// and of the V accumulator; the online softmax per warp, the 4 warps merged into the chunk's partial (max, sum, acc)
template <int KV>
__global__ void __launch_bounds__(kWarps * 32) attn_part_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kc,
                                                                const uint16_t* __restrict__ vc, int n_head, int n_head_kv,
                                                                int n_pos, float scale, float* __restrict__ part,
                                                                int n_chunks) {
    __shared__ float sm[kWarps], sl[kWarps];
    __shared__ float sacc[kWarps][kD];
    const int h = blockIdx.x, c = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = h / (n_head / n_head_kv);
    const int t0 = c * kChunk, t1 = min(n_pos, t0 + kChunk);
    float qv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) qv[i] = q[(size_t) h * kD + lane * 8 + i] * scale;
    float m = -INFINITY, l = 0.0f, acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    for (int t = t0 + warp; t < t1; t += kWarps) {
        if constexpr (KV != 0) {
            float kf[8], vf[8];
            kv_ld8<kv_kf(KV)>(kc, (size_t) t * n_head_kv + g, lane, kf);
            kv_ld8<kv_vf(KV)>(vc, (size_t) t * n_head_kv + g, lane, vf);
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) dot += qv[i] * kf[i];
            const float sc = warp_sum(dot);
            const float mn = fmaxf(m, sc);
            const float a = expf(m - mn), pr = expf(sc - mn);
            l = l * a + pr;
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] = acc[i] * a + pr * vf[i];
            m = mn;
            continue;
        }
        const size_t row = ((size_t) t * n_head_kv + g) * kD + lane * 8;
        const uint4 kw = *(const uint4*) (kc + row);
        const uint4 vw = *(const uint4*) (vc + row);
        const uint16_t* kh = (const uint16_t*) &kw;
        const uint16_t* vh = (const uint16_t*) &vw;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) dot += qv[i] * h2f(kh[i]);
        const float sc = warp_sum(dot);
        const float mn = fmaxf(m, sc);
        const float a = expf(m - mn), pr = expf(sc - mn);
        l = l * a + pr;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = acc[i] * a + pr * h2f(vh[i]);
        m = mn;
    }
    if (lane == 0) {
        sm[warp] = m;
        sl[warp] = l;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) sacc[warp][lane * 8 + i] = acc[i];
    __syncthreads();
    if (warp == 0) {
        float M = sm[0];
#pragma unroll
        for (int w = 1; w < kWarps; ++w) M = fmaxf(M, sm[w]);
        float L = 0.0f, f[kWarps];
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            f[w] = sm[w] == -INFINITY ? 0.0f : expf(sm[w] - M);
            L += sl[w] * f[w];
        }
        float* out = part + ((size_t) h * n_chunks + c) * (kD + 2);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float o = 0.0f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) o += sacc[w][lane * 8 + i] * f[w];
            out[lane * 8 + i] = o;
        }
        if (lane == 0) {
            out[kD] = M;
            out[kD + 1] = L;
        }
    }
}

// attn_part_kernel for many rows at once, each over ITS OWN selected cells: row r attends to kc / vc rows
// ids[r * ld_ids + j], j < steps[r * 4 + 3] (the QSA selection, read in place: no gathered copy).  grid (n_head / HPW,
// n_chunks, rows), 128 threads: a block takes HPW heads of one KV group, so a K / V row loaded once serves them all;
// per head the arithmetic is attn_part_kernel's step for step (the same warp / position split, dot order, online
// softmax and 4-warp merge), so the partials are bit-identical to gather_rows + attn_part_kernel's.  A chunk past a
// row's width writes the empty partial (max -inf, sum 0), which attn_merge_kernel adds as exact zeros.
template <int HPW>
__global__ void __launch_bounds__(kWarps * 32) attn_part_ids_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kc,
                                                                    const uint16_t* __restrict__ vc,
                                                                    const int32_t* __restrict__ ids, int ld_ids,
                                                                    const int32_t* __restrict__ steps, int n_head,
                                                                    int n_head_kv, float scale, float* __restrict__ part,
                                                                    int n_chunks) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    __shared__ float sm[kWarps][HPW], sl[kWarps][HPW];
    __shared__ float sacc[kWarps][HPW][kD];
    const int h0 = blockIdx.x * HPW, c = blockIdx.y, r = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = h0 / (n_head / n_head_kv);
    const int n_pos = steps[(size_t) r * 4 + 3];
    const int t0 = c * kChunk, t1 = min(n_pos, t0 + kChunk);
    const int32_t* id = ids != nullptr ? ids + (size_t) r * ld_ids : nullptr;   // null: the cells 0 .. n_pos - 1
    const float* qr = q + (size_t) r * n_head * kD;
    float qv[HPW][8], m[HPW], l[HPW], acc[HPW][8];
#pragma unroll
    for (int j = 0; j < HPW; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qv[j][i] = qr[(size_t) (h0 + j) * kD + lane * 8 + i] * scale;
            acc[j][i] = 0.0f;
        }
        m[j] = -INFINITY;
        l[j] = 0.0f;
    }
    // kU of the warp's keys at a time: their ids and K / V rows loaded first, their dots next (independent), then the
    // softmax steps in key order - the loads and dots overlap, each key's arithmetic and the order stay attn_part's
    constexpr int kU = 8;
    for (int tb = t0 + warp; tb < t1; tb += kWarps * kU) {
        uint4 kw[kU], vw[kU];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const int t = tb + u * kWarps;
            if (t < t1) {
                const size_t row = ((size_t) (id != nullptr ? id[t] : t) * n_head_kv + g) * kD + lane * 8;
                kw[u] = *(const uint4*) (kc + row);
                vw[u] = *(const uint4*) (vc + row);
            }
        }
        float sc[kU][HPW];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            if (tb + u * kWarps >= t1) break;
            const uint16_t* kh = (const uint16_t*) &kw[u];
#pragma unroll
            for (int j = 0; j < HPW; ++j) {
                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) dot += qv[j][i] * h2f(kh[i]);
                sc[u][j] = warp_sum(dot);
            }
        }
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            if (tb + u * kWarps >= t1) break;
            const uint16_t* vh = (const uint16_t*) &vw[u];
#pragma unroll
            for (int j = 0; j < HPW; ++j) {
                const float mn = fmaxf(m[j], sc[u][j]);
                const float a = expf(m[j] - mn), pr = expf(sc[u][j] - mn);
                l[j] = l[j] * a + pr;
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[j][i] = acc[j][i] * a + pr * h2f(vh[i]);
                m[j] = mn;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < HPW; ++j) {
        if (lane == 0) {
            sm[warp][j] = m[j];
            sl[warp][j] = l[j];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) sacc[warp][j][lane * 8 + i] = acc[j][i];
    }
    __syncthreads();
    // the 4 warps' merge, a head per warp in turn (attn_part_kernel's warp-0 merge, the same order)
    for (int j = warp; j < HPW; j += kWarps) {
        float M = sm[0][j];
#pragma unroll
        for (int w = 1; w < kWarps; ++w) M = fmaxf(M, sm[w][j]);
        float L = 0.0f, f[kWarps];
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            f[w] = sm[w][j] == -INFINITY ? 0.0f : expf(sm[w][j] - M);
            L += sl[w][j] * f[w];
        }
        float* out = part + (((size_t) r * n_head + h0 + j) * n_chunks + c) * (kD + 2);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float o = 0.0f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) o += sacc[w][j][lane * 8 + i] * f[w];
            out[lane * 8 + i] = o;
        }
        if (lane == 0) {
            out[kD] = M;
            out[kD + 1] = L;
        }
    }
}

// grid (n_head, rows), 256 threads (a dim each): the chunks' partials merged (part and out row-major: [row][head])
__global__ void __launch_bounds__(kD) attn_merge_kernel(const float* __restrict__ part, int n_chunks, float* __restrict__ out) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int h = blockIdx.y * gridDim.x + blockIdx.x, d = threadIdx.x;
    const float* P = part + (size_t) h * n_chunks * (kD + 2);
    float M = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, P[(size_t) c * (kD + 2) + kD]);
    float L = 0.0f, o = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float* pc = P + (size_t) c * (kD + 2);
        const float f = pc[kD] == -INFINITY ? 0.0f : expf(pc[kD] - M);
        L += pc[kD + 1] * f;
        o += pc[d] * f;
    }
    out[(size_t) h * kD + d] = o / L;
}

// ---------------------------------------------------------------- the prompt path (rows = a sub-batch's tokens)
// R[t][i] (row stride ldr) += add[t][i], x[t][i] = rms(R[t]) * w[i], x16 its FP16; a block a row
__global__ void __launch_bounds__(256) add_rms_rows_kernel(float* __restrict__ R, int ldr, const float* __restrict__ add,
                                                           const float* __restrict__ w, float eps, int n,
                                                           float* __restrict__ x, uint16_t* __restrict__ x16) {
    __shared__ float sred[8];
    const int t = blockIdx.x;
    float* r = R + (size_t) t * ldr;
    const float* a = add != nullptr ? add + (size_t) t * n : nullptr;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) {
        float v = r[i];
        if (a != nullptr) {
            v += a[i];
            r[i] = v;
        }
        ss += v * v;
    }
    ss = block_sum256(ss, sred);
    const float inv = rsqrtf(ss / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += 256) {
        const float y = r[i] * inv * w[i];
        x[(size_t) t * n + i] = y;
        x16[(size_t) t * n + i] = f2h(y);
    }
}
// out[t][c] = silu(sum_i kW[c][i] * in(t - (dc - 1) + i, c)), the rows before the chunk from the state (cs[c][0] the
// oldest - gdn_conv_kernel's layout); grid (C / 256, T)
__global__ void __launch_bounds__(256) conv_rows_kernel(const float* __restrict__ x, int C, const float* __restrict__ kW,
                                                        int dc, const float* __restrict__ cs, float* __restrict__ out) {
    const int c = blockIdx.x * 256 + threadIdx.x, t = blockIdx.y;
    if (c >= C) return;
    const float* w = kW + (size_t) c * dc;
    float acc = 0.0f;
    for (int i = 0; i < dc; ++i) {
        const int tau = t - (dc - 1) + i;
        const float v = tau >= 0 ? x[(size_t) tau * C + c] : cs[(size_t) c * (dc - 1) + (tau + dc - 1)];
        acc += v * w[i];
    }
    out[(size_t) t * C + c] = acc * sigmoid_f(acc);
}
// the state after T rows: the last dc - 1 inputs (the state's own rows when T < dc - 1); dc <= 9
__global__ void __launch_bounds__(256) conv_state_rows_kernel(const float* __restrict__ x, int T, int C, int dc,
                                                              float* __restrict__ cs) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    float* st = cs + (size_t) c * (dc - 1);
    float nv[8];
    for (int i = 0; i < dc - 1; ++i) {
        const int tau = T - (dc - 1) + i;
        nv[i] = tau >= 0 ? x[(size_t) tau * C + c] : st[tau + dc - 1];
    }
    for (int i = 0; i < dc - 1; ++i) st[i] = nv[i];
}
// row t's q head hh (hh < h_k) or k head hh - h_k, in place: x / sqrt(sum x^2 + eps) (the sum in double, as
// gdn_l2_kernel's), q also scaled; grid (T, 2 h_k), S threads
__global__ void __launch_bounds__(128) gdn_prep_rows_kernel(float* __restrict__ h, int C, int h_k, int S, float eps,
                                                            float qscale) {
    __shared__ double sh[4];
    const int t = blockIdx.x, hh = blockIdx.y, i = threadIdx.x;
    float* p = h + (size_t) t * C + (size_t) hh * S;   // the q heads, then the k heads: head hh of [q | k]
    const float v = i < S ? p[i] : 0.0f;
    double acc = (double) v * (double) v;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((i & 31) == 0) sh[i >> 5] = acc;
    __syncthreads();
    const double sum = sh[0] + sh[1] + sh[2] + sh[3];
    const float inv = (float) (1.0 / sqrt(sum + (double) eps)) * (hh < h_k ? qscale : 1.0f);
    if (i < S) p[i] = v * inv;
}
__global__ void gdn_gates_rows_kernel(float* __restrict__ alpha, float* __restrict__ beta, const float* __restrict__ dt,
                                      const float* __restrict__ A, int n, int h_v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int h = i % h_v;
    alpha[i] = softplus_f(alpha[i] + dt[h]) * A[h];   // (the gate, in place)
    beta[i] = sigmoid_f(beta[i]);
}
// gdn_prep_rows_kernel's blocks (y < 2 h_k) and, past them, the gates of row t (gdn_gates_rows over its h_v heads):
// one launch for both; grid (T, 2 h_k + ceil(h_v / 128))
__global__ void __launch_bounds__(128) gdn_prep_gates_kernel(float* __restrict__ h, int C, int h_k, int S, float eps,
                                                             float qscale, float* __restrict__ alpha,
                                                             float* __restrict__ beta, const float* __restrict__ dt,
                                                             const float* __restrict__ A, int h_v) {
    __shared__ double sh[4];
    const int t = blockIdx.x, hh = blockIdx.y, i = threadIdx.x;
    if (hh >= 2 * h_k) {
        const int hv = (hh - 2 * h_k) * 128 + i;
        if (hv < h_v) {
            const size_t at = (size_t) t * h_v + hv;
            alpha[at] = softplus_f(alpha[at] + dt[hv]) * A[hv];
            beta[at] = sigmoid_f(beta[at]);
        }
        return;
    }
    float* p = h + (size_t) t * C + (size_t) hh * S;
    const float v = i < S ? p[i] : 0.0f;
    double acc = (double) v * (double) v;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if ((i & 31) == 0) sh[i >> 5] = acc;
    __syncthreads();
    const double sum = sh[0] + sh[1] + sh[2] + sh[3];
    const float inv = (float) (1.0 / sqrt(sum + (double) eps)) * (hh < h_k ? qscale : 1.0f);
    if (i < S) p[i] = v * inv;
}
// the decode conv (gdn_conv_kernel's state slide) with the SiLU in the same launch
__global__ void conv_silu_step_kernel(float* __restrict__ cs, const float* __restrict__ x, const float* __restrict__ kW,
                                      float* __restrict__ out, int C, int dc) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* st = cs + (size_t) c * (dc - 1);
    const float* w = kW + (size_t) c * dc;
    float acc = 0.0f;
    for (int i = 0; i < dc - 1; ++i) acc += st[i] * w[i];   // kernel[0] reads the OLDEST state row
    acc += x[c] * w[dc - 1];                                // the new input lands in the LAST tap
    out[c] = acc * sigmoid_f(acc);
    for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];     // slide: drop the oldest, append the newest
    st[dc - 2] = x[c];
}
// the recurrence over T rows: a block a v head, a thread a state column j held in registers, the rows in order.  The
// state (S, h_v, S) and the math are gdn_step's (gdn.cu): s *= dec, d = (v - s.k) beta, s += k d, o = s.q
template <int S>
__global__ void __launch_bounds__(S) gdn_scan_kernel(float* __restrict__ state, const float* __restrict__ h, int T, int C,
                                                     const float* __restrict__ gate, const float* __restrict__ beta,
                                                     int h_k, int h_v, float* __restrict__ o) {
    __shared__ float ks[S], qs[S];
    const int hv = blockIdx.x, j = threadIdx.x, src = hv % h_k;   // (MODULO head pairing, as gdn_step)
    const size_t stride = (size_t) h_v * S;
    float* col = state + (size_t) hv * S + j;
    float st[S];
#pragma unroll
    for (int i = 0; i < S; ++i) st[i] = col[(size_t) i * stride];
    for (int t = 0; t < T; ++t) {
        const float* row = h + (size_t) t * C;
        __syncthreads();
        qs[j] = row[src * S + j];
        ks[j] = row[(h_k + src) * S + j];
        __syncthreads();
        const float dec = expf(gate[(size_t) t * h_v + hv]), b = beta[(size_t) t * h_v + hv];
        const float v = row[(2 * h_k + hv) * S + j];
        float sk = 0.0f;
#pragma unroll
        for (int i = 0; i < S; ++i) {
            st[i] *= dec;
            sk += st[i] * ks[i];
        }
        const float d = (v - sk) * b;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < S; ++i) {
            st[i] += ks[i] * d;
            dot += st[i] * qs[i];
        }
        o[((size_t) t * h_v + hv) * S + j] = dot;
    }
#pragma unroll
    for (int i = 0; i < S; ++i) col[(size_t) i * stride] = st[i];
}
// gdn_scan_kernel with the launches around it (gdn_core): prep (raw rows: the q / k heads' L2 norm and the gates computed
// here - gdn_prep_gates_kernel's arithmetic, each block its own copy, nothing written back), the recurrence, then the
// output norm with the sigmoid gate (gdn.cu's gdn_out_norm_kernel<false>: one warp's sums in its order) and the q8_1
// of y (quantize_q8_1's) - one launch for four or five, the same bits.  A block a v head, S = 128 threads
template <int S>
__global__ void __launch_bounds__(S) gdn_core_kernel(float* __restrict__ state, const float* __restrict__ h, int T, int C,
                                                     const float* __restrict__ gate, const float* __restrict__ beta,
                                                     int h_k, int h_v, bool prep, float eps, float qscale,
                                                     const float* __restrict__ dt, const float* __restrict__ A,
                                                     const float* __restrict__ z, const float* __restrict__ ssm_norm,
                                                     float out_eps, float* __restrict__ y, Q81* __restrict__ yq) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    __shared__ float ks[S], qs[S], os[S];
    __shared__ double sh[2][4];
    __shared__ float s_inv;
    const int hv = blockIdx.x, j = threadIdx.x, src = hv % h_k;   // (MODULO head pairing, as gdn_step)
    const size_t stride = (size_t) h_v * S;
    float* col = state + (size_t) hv * S + j;
    float st[S];
#pragma unroll
    for (int i = 0; i < S; ++i) st[i] = col[(size_t) i * stride];
    for (int t = 0; t < T; ++t) {
        const float* row = h + (size_t) t * C;
        float qv = row[src * S + j], kv = row[(h_k + src) * S + j];
        float g = gate[(size_t) t * h_v + hv], b = beta[(size_t) t * h_v + hv];
        if (prep) {   // (gdn_prep_gates_kernel: a double sum a head - a warp's xor tree, then the 4 warps in order)
            double aq = (double) qv * (double) qv, ak = (double) kv * (double) kv;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                aq += __shfl_xor_sync(0xffffffffu, aq, o);
                ak += __shfl_xor_sync(0xffffffffu, ak, o);
            }
            __syncthreads();   // (the previous row's readers of sh)
            if ((j & 31) == 0) {
                sh[0][j >> 5] = aq;
                sh[1][j >> 5] = ak;
            }
            __syncthreads();
            const double sq = sh[0][0] + sh[0][1] + sh[0][2] + sh[0][3];
            const double sk2 = sh[1][0] + sh[1][1] + sh[1][2] + sh[1][3];
            qv = qv * ((float) (1.0 / sqrt(sq + (double) eps)) * qscale);
            kv = kv * ((float) (1.0 / sqrt(sk2 + (double) eps)) * 1.0f);
            g = softplus_f(g + dt[hv]) * A[hv];
            b = sigmoid_f(b);
        }
        __syncthreads();
        qs[j] = qv;
        ks[j] = kv;
        __syncthreads();
        const float dec = expf(g);
        const float v = row[(2 * h_k + hv) * S + j];
        float sk = 0.0f;
#pragma unroll
        for (int i = 0; i < S; ++i) {
            st[i] *= dec;
            sk += st[i] * ks[i];
        }
        const float d = (v - sk) * b;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < S; ++i) {
            st[i] += ks[i] * d;
            dot += st[i] * qs[i];
        }
        // the output norm of this (row, head): warp 0's lanes sum o[l], o[l + 32], .. in double, then shfl_down
        os[j] = dot;
        __syncthreads();
        if (j < 32) {
            double acc = 0.0;
            for (int i = j; i < S; i += 32) acc += (double) os[i] * (double) os[i];
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
            if (j == 0) s_inv = (float) (1.0 / sqrt(acc / (double) S + (double) out_eps));
        }
        __syncthreads();
        const size_t at = ((size_t) t * h_v + hv) * S + j;
        const float yv = os[j] * s_inv * ssm_norm[j] * sigmoid_f(z[at]);
        y[at] = yv;
        if (yq != nullptr) q81_store_warp(yv, yq + at / 32, j & 31);
    }
#pragma unroll
    for (int i = 0; i < S; ++i) col[(size_t) i * stride] = st[i];
}
#if !defined(STRATA_USE_HIP)
// The token path's delta-net middle after the qkv / z projection in ONE launch (gdn_mid): alpha and beta (F32 rows:
// glm_fast's row_dot_f32 - a lane's float4 chain, then the xor tree - and mv_kernel's 1 * s + 0), the conv + SiLU
// (conv_silu_step_kernel's), then gdn_core_kernel's prep, recurrence, output norm and y's q8_1 - the same arithmetic
// in the same order, so the same bits (gdn_core_parity).  A cluster of R = h_v / h_k blocks per k head: block (src,
// rank) takes v head src + h_k * rank (the MODULO pairing); every block of the cluster computes the k head's q / k conv
// from the old state, and after the cluster barrier rank 0 alone slides that state (each block its own v channels).
// The block also pulls its state column and its slice of the next GEMV's weights (pf, pf_bytes) toward L2 at the start.
__device__ __forceinline__ float mid_row_dot_f32(const float* row, const float* x, int n_in, int lane) {
    float s = 0.0f;
    const float4* r4 = (const float4*) row;
    const float4* x4 = (const float4*) x;
    for (int k = lane; k < n_in / 4; k += 32) {
        const float4 a = r4[k], b = x4[k];
        s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    return warp_sum(s);
}
__device__ __forceinline__ float mid_row_dot_bf16(const uint16_t* row, const float* x, int n_in, int lane) {   // (row_dot_bf16)
    float s = 0.0f;
    const uint4* r8 = (const uint4*) row;
    const float4* x4 = (const float4*) x;
    for (int k = lane; k < n_in / 8; k += 32) {
        const uint4 w = r8[k];
        const float4 a = x4[2 * k], b = x4[2 * k + 1];
        s += __uint_as_float(w.x << 16) * a.x + __uint_as_float(w.x & 0xffff0000u) * a.y +
             __uint_as_float(w.y << 16) * a.z + __uint_as_float(w.y & 0xffff0000u) * a.w +
             __uint_as_float(w.z << 16) * b.x + __uint_as_float(w.z & 0xffff0000u) * b.y +
             __uint_as_float(w.w << 16) * b.z + __uint_as_float(w.w & 0xffff0000u) * b.w;
    }
    return warp_sum(s);
}
__device__ __forceinline__ float mid_conv4(float s0, float s1, float s2, float x, float4 w) {   // (conv_silu_step's, d_conv 4)
    float acc = 0.0f;
    acc += s0 * w.x;
    acc += s1 * w.y;
    acc += s2 * w.z;
    acc += x * w.w;
    return acc * sigmoid_f(acc);
}
template <int S>
__global__ void __launch_bounds__(S) gdn_mid_kernel(float* __restrict__ state, float* cs,
                                                    const float* __restrict__ qkv, const float* __restrict__ kW, int dc,
                                                    const float* __restrict__ xf, const void* __restrict__ wa,
                                                    const void* __restrict__ wb, bool ab_bf16, int n_in, int h_k, int h_v,
                                                    float eps, float qscale, const float* __restrict__ dt,
                                                    const float* __restrict__ A, const float* __restrict__ z,
                                                    const float* __restrict__ ssm_norm, float out_eps,
                                                    float* __restrict__ h_out, float* __restrict__ ab_out,
                                                    float* __restrict__ y, Q81* __restrict__ yq,
                                                    const uint8_t* __restrict__ pf, size_t pf_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    namespace cg = cooperative_groups;
    cg::cluster_group cl = cg::this_cluster();
    __shared__ float ks[S], qs[S], os[S];
    __shared__ double sh[2][4];
    __shared__ float s_inv, s_ab[2];
    const int R = h_v / h_k, rank = (int) cl.block_rank(), src = blockIdx.x / R, hv = src + h_k * rank;
    const int j = threadIdx.x, lane = j & 31, warp = j >> 5;
    const size_t stride = (size_t) h_v * S;
    float* col = state + (size_t) hv * S + j;
    // the next GEMV's weights toward L2 (this block's slice)
    if (pf != nullptr) {
        const size_t per = (pf_bytes / gridDim.x + 127) & ~(size_t) 127, b0 = per * blockIdx.x;
        const size_t b1 = b0 + per < pf_bytes ? b0 + per : pf_bytes;
        for (size_t o = b0 + (size_t) j * 128; o < b1; o += (size_t) S * 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(pf + o));
    }
    // into shared memory with cp.async (one round trip, no registers held): this v head's state slice [S][S], x and
    // its alpha / beta rows.  Then alpha (warp 0) and beta (warp 1): mv_kernel's row (alpha 1, no bias)
    extern __shared__ __align__(16) unsigned char mid_sm[];
    float* sst = (float*) mid_sm;                                   // [i][j]: state row i of this head
    float* sx = sst + S * S;
    const int rb = ab_bf16 ? n_in * 2 : n_in * 4;   // (a row's bytes)
    unsigned char* sw = (unsigned char*) (sx + n_in);
    {
        for (int c = j; c < S * S / 4; c += S) {   // (16-byte pieces: row c / 32, piece c % 32)
            const int i = c / (S / 4), p = c % (S / 4);
            __pipeline_memcpy_async(sst + i * S + p * 4, state + (size_t) i * stride + (size_t) hv * S + p * 4, 16);
        }
        const char* gx = (const char*) xf;
        const char* ga = (const char*) wa + (size_t) hv * rb;
        const char* gb = (const char*) wb + (size_t) hv * rb;
        for (int o = j * 16; o < n_in * 4; o += S * 16) __pipeline_memcpy_async((unsigned char*) sx + o, gx + o, 16);
        for (int o = j * 16; o < rb; o += S * 16) {
            __pipeline_memcpy_async(sw + o, ga + o, 16);
            __pipeline_memcpy_async(sw + rb + o, gb + o, 16);
        }
        __pipeline_commit();
    }
    // meanwhile the small inputs into registers (one round trip): this thread's q, k (the k head's) and v (the v
    // head's) channels - the input, the old conv state, the taps (d_conv 4) - and the gates' and the norm's numbers
    const int cq = src * S + j, ck = (h_k + src) * S + j, cv = (2 * h_k + hv) * S + j;
    float* sq = cs + (size_t) cq * 3;
    float* sk = cs + (size_t) ck * 3;
    float* sv = cs + (size_t) cv * 3;
    const float xq = qkv[cq], xk = qkv[ck], xv = qkv[cv];
    // (the k head's q / k state through volatile loads: rank 0 overwrites it after the cluster barrier, so these reads
    // must not be moved past it)
    const volatile float* vq = sq;
    const volatile float* vk = sk;
    const float q0 = vq[0], q1 = vq[1], q2 = vq[2], k0 = vk[0], k1 = vk[1], k2 = vk[2], v0 = sv[0], v1 = sv[1], v2 = sv[2];
    const float4 wq = *(const float4*) (kW + (size_t) cq * 4), wk = *(const float4*) (kW + (size_t) ck * 4),
                 wv = *(const float4*) (kW + (size_t) cv * 4);
    const size_t at = (size_t) hv * S + j;
    const float zt = z[at], nrm = ssm_norm[j], dth = dt[hv], Ah = A[hv];
    // the conv + SiLU (conv_silu_step_kernel's sum: kernel[0] on the oldest state row, the input on the last tap)
    float qv = mid_conv4(q0, q1, q2, xq, wq);
    float kv = mid_conv4(k0, k1, k2, xk, wk);
    const float v = mid_conv4(v0, v1, v2, xv, wv);
    if (h_out != nullptr) {
        h_out[cv] = v;
        if (rank == 0) {
            h_out[cq] = qv;
            h_out[ck] = kv;
        }
    }
    sv[0] = v1;   // (slide: drop the oldest, append the newest - the v channels are this block's alone)
    sv[1] = v2;
    sv[2] = xv;
    __pipeline_wait_prior(0);
    __syncthreads();
    if (warp < 2) {
        const unsigned char* w = sw + (size_t) warp * rb;
        const float s = ab_bf16 ? mid_row_dot_bf16((const uint16_t*) w, sx, n_in, lane)
                                : mid_row_dot_f32((const float*) w, sx, n_in, lane);
        if (lane == 0) {
            const float yv = 1.0f * s + 0.0f;
            s_ab[warp] = yv;
            if (ab_out != nullptr) ab_out[warp * h_v + hv] = yv;
        }
    }
    cl.sync();   // (every block of the cluster has read the k head's old q / k state - and s_ab is visible)
    if (rank == 0) {
        sq[0] = q1;
        sq[1] = q2;
        sq[2] = xq;
        sk[0] = k1;
        sk[1] = k2;
        sk[2] = xk;
    }
    float g = s_ab[0], b = s_ab[1];
    {   // (gdn_core_kernel's prep)
        double aq = (double) qv * (double) qv, ak = (double) kv * (double) kv;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            aq += __shfl_xor_sync(0xffffffffu, aq, o);
            ak += __shfl_xor_sync(0xffffffffu, ak, o);
        }
        if ((j & 31) == 0) {
            sh[0][j >> 5] = aq;
            sh[1][j >> 5] = ak;
        }
        __syncthreads();
        const double sq2 = sh[0][0] + sh[0][1] + sh[0][2] + sh[0][3];
        const double sk2 = sh[1][0] + sh[1][1] + sh[1][2] + sh[1][3];
        qv = qv * ((float) (1.0 / sqrt(sq2 + (double) eps)) * qscale);
        kv = kv * ((float) (1.0 / sqrt(sk2 + (double) eps)) * 1.0f);
        g = softplus_f(g + dth) * Ah;
        b = sigmoid_f(b);
    }
    qs[j] = qv;
    ks[j] = kv;
    __syncthreads();
    // gdn_core_kernel's recurrence, the state column read from shared memory twice (s *= dec recomputed: one
    // multiply, the same float) instead of held in registers - written back as it goes
    const float dec = expf(g);
    float skd = 0.0f;
#pragma unroll 16
    for (int i = 0; i < S; ++i) {
        const float sd = __fmul_rn(sst[i * S + j], dec);
        skd = __fmaf_rn(sd, ks[i], skd);
    }
    const float d = (v - skd) * b;
    float dot = 0.0f;
#pragma unroll 16
    for (int i = 0; i < S; ++i) {
        const float sn = __fmaf_rn(ks[i], d, __fmul_rn(sst[i * S + j], dec));   // (the products the register
        dot = __fmaf_rn(sn, qs[i], dot);                                        //  version's FMAs take, spelled out)
        col[(size_t) i * stride] = sn;
    }
    os[j] = dot;
    __syncthreads();
    if (j < 32) {
        double acc = 0.0;
        for (int i = j; i < S; i += 32) acc += (double) os[i] * (double) os[i];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        if (j == 0) s_inv = (float) (1.0 / sqrt(acc / (double) S + (double) out_eps));
    }
    __syncthreads();
    const float yv = os[j] * s_inv * nrm * sigmoid_f(zt);
    y[at] = yv;
    if (yq != nullptr) q81_store_warp(yv, yq + at / 32, j & 31);
#endif
}
#endif
// attn_prep_kernel over rows: grid (n_head + n_head_kv, T), row t at position p0 + t
__global__ void __launch_bounds__(kD) attn_prep_rows_kernel(const float* __restrict__ qf, const float* __restrict__ k,
                                                            const float* __restrict__ v, const float* __restrict__ q_norm,
                                                            const float* __restrict__ k_norm, float eps, int n_head,
                                                            int n_head_kv, int n_rot, float rope_base, int64_t p0,
                                                            uint16_t* __restrict__ kc, uint16_t* __restrict__ vc,
                                                            float* __restrict__ q) {
    __shared__ float sv[kD];
    __shared__ float sred[8];
    const int b = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const bool is_q = b < n_head;
    const int g = is_q ? 0 : b - n_head;
    const int64_t p = p0 + t;
    const float* qrow = qf + (size_t) t * 2 * kD * n_head;
    const float* krow = k + (size_t) t * kD * n_head_kv;
    const float* vrow = v + (size_t) t * kD * n_head_kv;
    const float x = is_q ? qrow[(size_t) b * 2 * kD + d] : krow[(size_t) g * kD + d];
    const float ss = block_sum256(x * x, sred);
    const float inv = rsqrtf(ss / (float) kD + eps);
    const float xn = x * inv * (is_q ? q_norm[d] : k_norm[d]);
    sv[d] = xn;
    __syncthreads();
    float y = xn;
    if (d < n_rot) {
        const int half = n_rot / 2;
        const int i = d < half ? d : d - half;
        const float theta = (float) p * powf(rope_base, -2.0f * (float) i / (float) n_rot);
        const float c = cosf(theta), sn = sinf(theta);
        const float x0 = sv[i], x1 = sv[i + half];
        y = d < half ? x0 * c - x1 * sn : x0 * sn + x1 * c;
    }
    if (is_q) {
        q[((size_t) t * n_head + b) * kD + d] = y;
    } else {
        const size_t row = ((size_t) p * n_head_kv + g) * kD;
        kc[row + d] = f2h(y);
        vc[row + d] = f2h(vrow[(size_t) g * kD + d]);
    }
}
// attn_prep_rows_kernel for a q4_0 cache: the same norms and RoPE, then q, k and v rotated by H; k and v go to the
// cache rows as q4_0 (kQ4Row bytes a row).  T 1 at p0 = p is attn_prep's token.  n_head 0: the K / V blocks alone
template <int KV>
__global__ void __launch_bounds__(kD) attn_prep_q4_kernel(const float* __restrict__ qf, const float* __restrict__ k,
                                                          const float* __restrict__ v, const float* __restrict__ q_norm,
                                                          const float* __restrict__ k_norm, float eps, int n_head,
                                                          int n_head_kv, int n_rot, float rope_base, int64_t p0,
                                                          uint16_t* __restrict__ kc, uint16_t* __restrict__ vc,
                                                          float* __restrict__ q) {
    __shared__ float sv[kD], sk[kD], sw[kD];
    __shared__ float sred[8];
    const int b = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const bool is_q = b < n_head;
    const int g = is_q ? 0 : b - n_head;
    const int64_t p = p0 + t;
    const float* qrow = qf + (size_t) t * 2 * kD * n_head;
    const float* krow = k + (size_t) t * kD * n_head_kv;
    const float* vrow = v + (size_t) t * kD * n_head_kv;
    const float x = is_q ? qrow[(size_t) b * 2 * kD + d] : krow[(size_t) g * kD + d];
    const float ss = block_sum256(x * x, sred);
    const float inv = rsqrtf(ss / (float) kD + eps);
    const float xn = x * inv * (is_q ? q_norm[d] : k_norm[d]);
    sv[d] = xn;
    __syncthreads();
    float y = xn;
    if (d < n_rot) {
        const int half = n_rot / 2;
        const int i = d < half ? d : d - half;
        const float theta = (float) p * powf(rope_base, -2.0f * (float) i / (float) n_rot);
        const float c = cosf(theta), sn = sinf(theta);
        const float x0 = sv[i], x1 = sv[i + half];
        y = d < half ? x0 * c - x1 * sn : x0 * sn + x1 * c;
    }
    constexpr int KF = kv_kf(KV), VF = kv_vf(KV);   // (q4_0 halves rotated; q is rotated with K)
    sk[d] = y;
    sw[d] = is_q ? 0.0f : vrow[(size_t) g * kD + d];
    if constexpr (KF == 1) {
        fwht256_sh(sk, d);
        y = sk[d] * (1.0f / 16.0f);
    }
    float vv = sw[d];
    if constexpr (VF == 1) {
        if (!is_q) {   // (block-uniform)
            fwht256_sh(sw, d);
            vv = sw[d] * (1.0f / 16.0f);
        }
    }
    if (is_q) {
        q[((size_t) t * n_head + b) * kD + d] = y;
    } else {
        const size_t ch = (size_t) p * n_head_kv + g;
        uint8_t* kb = reinterpret_cast<uint8_t*>(kc) + ch * kv_row(KF);
        uint8_t* vb = reinterpret_cast<uint8_t*>(vc) + ch * kv_row(VF);
        if constexpr (KF == 1) q4_put(kb, d, y);
        else q8_put(kb, d, y);
        if constexpr (VF == 1) q4_put(vb, d, vv);
        else q8_put(vb, d, vv);
    }
}
// attn_gate_rows_kernel after rotating the q4_0 cache's attention output back: o[t][h] = H o[t][h], then the gate
__global__ void __launch_bounds__(kD) attn_gate_q4_kernel(float* __restrict__ o, const float* __restrict__ qf, int n_head) {
    __shared__ float sh[kD];
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    float* row = o + ((size_t) t * n_head + h) * kD;
    sh[d] = row[d];
    fwht256_sh(sh, d);
    row[d] = sh[d] * (1.0f / 16.0f) * sigmoid_f(qf[((size_t) t * n_head + h) * 2 * kD + kD + d]);
}
__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 x = __half22float2(h[i]);
        f[2 * i] = x.x;
        f[2 * i + 1] = x.y;
    }
}
// the causal attention of 4 queries a block over one KV head's group: 8 warps x HW heads, a lane 8 dims; the keys
// stream past once, each (query, head) pair an online softmax in registers.  grid (ceil(T / 4), n_head_kv)
template <int HW, int KV>
__global__ void __launch_bounds__(256) attn_prompt_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kc,
                                                          const uint16_t* __restrict__ vc, int n_head, int n_head_kv,
                                                          int64_t p0, int T, float scale, float* __restrict__ out) {
    constexpr int QB = 4;
    const int g = blockIdx.y, t0 = blockIdx.x * QB, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int G = n_head / n_head_kv, h0 = g * G + warp * HW;
    if (warp * HW >= G) return;   // (a group of fewer than 8 HW heads: qwen4exp's 12 on HW 2 - 6 warps; no block sync)
    const int nq = min(QB, T - t0);
    float qr[QB][HW][8], acc[QB][HW][8], m[QB][HW], l[QB][HW];
#pragma unroll
    for (int qi = 0; qi < QB; ++qi)
#pragma unroll
        for (int hi = 0; hi < HW; ++hi) {
            const float* qp = q + ((size_t) (t0 + (qi < nq ? qi : 0)) * n_head + h0 + hi) * kD + lane * 8;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                qr[qi][hi][e] = qp[e] * scale;
                acc[qi][hi][e] = 0.0f;
            }
            m[qi][hi] = -INFINITY;
            l[qi][hi] = 0.0f;
        }
    const int64_t pend = p0 + t0 + nq;   // the keys: positions < pend, each query's own and before
    for (int64_t kp = 0; kp < pend; ++kp) {
        float kf[8], vf[8];
        if constexpr (KV != 0) {
            kv_ld8<kv_kf(KV)>(kc, (size_t) kp * n_head_kv + g, lane, kf);
            kv_ld8<kv_vf(KV)>(vc, (size_t) kp * n_head_kv + g, lane, vf);
        } else {
            const size_t row = ((size_t) kp * n_head_kv + g) * kD + lane * 8;
            unpack8(*reinterpret_cast<const uint4*>(kc + row), kf);
            unpack8(*reinterpret_cast<const uint4*>(vc + row), vf);
        }
#pragma unroll
        for (int hi = 0; hi < HW; ++hi)
#pragma unroll
            for (int qi = 0; qi < QB; ++qi) {
                float sdot = 0.0f;
#pragma unroll
                for (int e = 0; e < 8; ++e) sdot += qr[qi][hi][e] * kf[e];
                sdot = warp_sum(sdot);
                if (qi < nq && kp <= p0 + t0 + qi) {   // (warp-uniform)
                    const float mn = fmaxf(m[qi][hi], sdot);
                    const float corr = expf(m[qi][hi] - mn), pe = expf(sdot - mn);
                    l[qi][hi] = l[qi][hi] * corr + pe;
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[qi][hi][e] = acc[qi][hi][e] * corr + pe * vf[e];
                    m[qi][hi] = mn;
                }
            }
    }
#pragma unroll
    for (int qi = 0; qi < QB; ++qi) {
        if (qi >= nq) break;
#pragma unroll
        for (int hi = 0; hi < HW; ++hi) {
            float* op = out + ((size_t) (t0 + qi) * n_head + h0 + hi) * kD + lane * 8;
            const float inv = 1.0f / l[qi][hi];
#pragma unroll
            for (int e = 0; e < 8; ++e) op[e] = acc[qi][hi][e] * inv;
        }
    }
}
// ---------------------------------------------------------------- the prompt's attention on tensor cores
// A block takes 8 tokens (a warp each) and one KV head's group of 16 heads: a warp's 16 rows are its token's heads, so
// one K / V tile in shared memory serves the 8 tokens' 128 rows.  WMMA 16x16x16 (rocWMMA on HIP, nvcuda::wmma on
// CUDA): FP16 operands (q scaled, then rounded; the caches are FP16), F32 sums.  Two passes over the keys: the first
// takes each row's max and sum, the second P = exp(s - max) / sum in FP16 against V - so the accumulators never need a
// rescale (their element-to-row layout is the library's).  grid (ceil(T / 8), n_head_kv)
#if defined(STRATA_USE_HIP)
namespace q35wm = rocwmma;
using wh16 = _Float16;
__device__ __forceinline__ wh16 to_wh16(float x) { return (wh16) x; }
#else
namespace q35wm = nvcuda::wmma;
using wh16 = __half;
__device__ __forceinline__ wh16 to_wh16(float x) { return __float2half(x); }
#endif
constexpr int kPT = 8;          // tokens a block
constexpr int kPB = 32;         // keys a tile
constexpr int kKP = kD + 16;    // sK's row pitch in halves (spreads the banks; keeps 32-byte fragment rows)
constexpr size_t kPromptTcSmem = (size_t) kPB * kKP * 2 + (size_t) kPB * kD * 2 + (size_t) kPT * 16 * kPB * 4 +
                                 (size_t) kPT * 16 * kPB * 2;

__global__ void q16_scale_kernel(const float* __restrict__ q, size_t n, float scale, wh16* __restrict__ q16) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) q16[i] = to_wh16(q[i] * scale);
}

template <int KV>
__global__ void __launch_bounds__(256) attn_prompt_tc_kernel(const wh16* __restrict__ q16, const uint16_t* __restrict__ kc,
                                                             const uint16_t* __restrict__ vc, int n_head, int n_head_kv,
                                                             int64_t p0, int T, float* __restrict__ out) {
    namespace wm = q35wm;
    extern __shared__ __align__(128) unsigned char q35_tc_sm[];
    wh16* sK = (wh16*) q35_tc_sm;                     // [kPB][kKP]
    wh16* sV = sK + kPB * kKP;                         // [kPB][kD]
    float* sS = (float*) (sV + kPB * kD);              // [warp][16][kPB]
    wh16* sP = (wh16*) (sS + kPT * 16 * kPB);          // [warp][16][kPB]
    const int g = blockIdx.y, t0 = blockIdx.x * kPT, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nt = min(kPT, T - t0), t = t0 + warp;
    const bool active = warp < nt;
    const int64_t lim = p0 + t;                        // the warp's last key
    const int64_t kend = p0 + t0 + nt;                 // the block's keys: [0, kend)
    float* wS = sS + warp * 16 * kPB;
    wh16* wP = sP + warp * 16 * kPB;
    wm::fragment<wm::matrix_a, 16, 16, 16, wh16, wm::row_major> qf[kD / 16];
    if (active) {
        const wh16* qa = q16 + ((size_t) t * n_head + (size_t) g * 16) * kD;
#pragma unroll
        for (int k = 0; k < kD / 16; ++k) wm::load_matrix_sync(qf[k], qa + k * 16, kD);
    }
    const int r = lane >> 1, hf = lane & 1;            // the softmax: a row a lane pair, 16 keys a lane
    const auto load_tile = [&](int64_t kt, bool with_v) {
#pragma unroll
        for (int it = 0; it < kPB * (kD / 8) / 256; ++it) {
            const int idx = it * 256 + tid, row = idx / (kD / 8), col = idx % (kD / 8);
            const int64_t kp = kt + row;
            const uint4 z = make_uint4(0, 0, 0, 0);
            if constexpr (KV != 0) {
                const size_t ch = (size_t) kp * n_head_kv + g;
                *reinterpret_cast<uint4*>(sK + row * kKP + col * 8) = kp < kend ? kv_ld8h<kv_kf(KV)>(kc, ch, col) : z;
                if (with_v) *reinterpret_cast<uint4*>(sV + row * kD + col * 8) = kp < kend ? kv_ld8h<kv_vf(KV)>(vc, ch, col) : z;
            } else {
            const size_t off = ((size_t) kp * n_head_kv + g) * kD + col * 8;
            *reinterpret_cast<uint4*>(sK + row * kKP + col * 8) = kp < kend ? *reinterpret_cast<const uint4*>(kc + off) : z;
            if (with_v)
                *reinterpret_cast<uint4*>(sV + row * kD + col * 8) =
                    kp < kend ? *reinterpret_cast<const uint4*>(vc + off) : z;
            }
        }
    };
    const auto scores = [&] {   // S = Q K^T over the tile into wS [16][kPB]
#pragma unroll
        for (int n = 0; n < kPB / 16; ++n) {
            wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
            wm::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int k = 0; k < kD / 16; ++k) {
                wm::fragment<wm::matrix_b, 16, 16, 16, wh16, wm::col_major> b;
                wm::load_matrix_sync(b, sK + n * 16 * kKP + k * 16, kKP);
                wm::mma_sync(acc, qf[k], b, acc);
            }
            wm::store_matrix_sync(wS + n * 16, acc, kPB, wm::mem_row_major);
        }
    };
    // pass 1: each row's max and sum
    float m = -INFINITY, l = 0.0f;
    for (int64_t kt = 0; kt < kend; kt += kPB) {
        __syncthreads();
        load_tile(kt, false);
        __syncthreads();
        const bool on = active && kt <= lim;
        if (on) scores();
        __syncthreads();
        if (on) {
            float sv[16], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int64_t kp = kt + hf * 16 + j;
                sv[j] = kp <= lim ? wS[r * kPB + hf * 16 + j] : -INFINITY;
                mx = fmaxf(mx, sv[j]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            const float mn = fmaxf(m, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j) sum += expf(sv[j] - mn);
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            l = (m == -INFINITY ? 0.0f : l * expf(m - mn)) + sum;
            m = mn;
        }
    }
    // pass 2: O = sum P V with P normalized
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    wm::fragment<wm::accumulator, 16, 16, 16, float> of[kD / 16];
#pragma unroll
    for (int n = 0; n < kD / 16; ++n) wm::fill_fragment(of[n], 0.0f);
    for (int64_t kt = 0; kt < kend; kt += kPB) {
        __syncthreads();
        load_tile(kt, true);
        __syncthreads();
        const bool on = active && kt <= lim;
        if (on) scores();
        __syncthreads();
        if (on) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int64_t kp = kt + hf * 16 + j;
                const float pv = kp <= lim ? expf(wS[r * kPB + hf * 16 + j] - m) * inv : 0.0f;
                wP[r * kPB + hf * 16 + j] = to_wh16(pv);
            }
        }
        __syncthreads();
        if (on) {
            wm::fragment<wm::matrix_a, 16, 16, 16, wh16, wm::row_major> pa[kPB / 16];
#pragma unroll
            for (int kk = 0; kk < kPB / 16; ++kk) wm::load_matrix_sync(pa[kk], wP + kk * 16, kPB);
#pragma unroll
            for (int n = 0; n < kD / 16; ++n)
#pragma unroll
                for (int kk = 0; kk < kPB / 16; ++kk) {
                    wm::fragment<wm::matrix_b, 16, 16, 16, wh16, wm::row_major> vb;
                    wm::load_matrix_sync(vb, sV + kk * 16 * kD + n * 16, kD);
                    wm::mma_sync(of[n], pa[kk], vb, of[n]);
                }
        }
    }
    if (active) {
        float* op = out + ((size_t) t * n_head + (size_t) g * 16) * kD;
#pragma unroll
        for (int n = 0; n < kD / 16; ++n) wm::store_matrix_sync(op + n * 16, of[n], kD, wm::mem_row_major);
    }
}

// a verify window's attention (T <= 8 rows): attn_prompt_kernel's block (4 queries x one KV head's group) over one
// chunk of kWinChunk key positions, its partials (max, sum, the unnormalized accumulator) into part
// [row][head][chunk] for attn_merge_kernel - the window's few rows alone gave attn_prompt 1-4 blocks, each walking
// every cached position.  grid (ceil(T / 4), n_head_kv, n_chunks)
constexpr int kWinChunk = 128;
template <int HW, int KV>
__global__ void __launch_bounds__(256) attn_window_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kc,
                                                          const uint16_t* __restrict__ vc, int n_head, int n_head_kv,
                                                          int64_t p0, int T, float scale, float* __restrict__ part,
                                                          int n_chunks) {
    constexpr int QB = 4;
    const int g = blockIdx.y, t0 = blockIdx.x * QB, c = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int G = n_head / n_head_kv, h0 = g * G + warp * HW;
    const int nq = min(QB, T - t0);
    float qr[QB][HW][8], acc[QB][HW][8], m[QB][HW], l[QB][HW];
#pragma unroll
    for (int qi = 0; qi < QB; ++qi)
#pragma unroll
        for (int hi = 0; hi < HW; ++hi) {
            const float* qp = q + ((size_t) (t0 + (qi < nq ? qi : 0)) * n_head + h0 + hi) * kD + lane * 8;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                qr[qi][hi][e] = qp[e] * scale;
                acc[qi][hi][e] = 0.0f;
            }
            m[qi][hi] = -INFINITY;
            l[qi][hi] = 0.0f;
        }
    const int64_t k0 = (int64_t) c * kWinChunk, k1 = min(k0 + kWinChunk, p0 + t0 + nq);
    for (int64_t kp = k0; kp < k1; ++kp) {
        float kf[8], vf[8];
        if constexpr (KV != 0) {
            kv_ld8<kv_kf(KV)>(kc, (size_t) kp * n_head_kv + g, lane, kf);
            kv_ld8<kv_vf(KV)>(vc, (size_t) kp * n_head_kv + g, lane, vf);
        } else {
            const size_t row = ((size_t) kp * n_head_kv + g) * kD + lane * 8;
            unpack8(*reinterpret_cast<const uint4*>(kc + row), kf);
            unpack8(*reinterpret_cast<const uint4*>(vc + row), vf);
        }
#pragma unroll
        for (int hi = 0; hi < HW; ++hi)
#pragma unroll
            for (int qi = 0; qi < QB; ++qi) {
                float sdot = 0.0f;
#pragma unroll
                for (int e = 0; e < 8; ++e) sdot += qr[qi][hi][e] * kf[e];
                sdot = warp_sum(sdot);
                if (qi < nq && kp <= p0 + t0 + qi) {   // (warp-uniform)
                    const float mn = fmaxf(m[qi][hi], sdot);
                    const float corr = expf(m[qi][hi] - mn), pe = expf(sdot - mn);
                    l[qi][hi] = l[qi][hi] * corr + pe;
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[qi][hi][e] = acc[qi][hi][e] * corr + pe * vf[e];
                    m[qi][hi] = mn;
                }
            }
    }
#pragma unroll
    for (int qi = 0; qi < QB; ++qi) {
        if (qi >= nq) break;
#pragma unroll
        for (int hi = 0; hi < HW; ++hi) {
            float* op = part + (((size_t) (t0 + qi) * n_head + h0 + hi) * n_chunks + c) * (kD + 2);
#pragma unroll
            for (int e = 0; e < 8; ++e) op[lane * 8 + e] = acc[qi][hi][e];
            if (lane == 0) {
                op[kD] = m[qi][hi];
                op[kD + 1] = l[qi][hi];
            }
        }
    }
}
// the decode / verify-window attention over one chunk of keys, its partials into part [row][head][chunk] for
// attn_merge_kernel: a block takes RB rows x one KV head's group of G heads (RB * G = 8 * PW query-head pairs), a warp
// PW pairs, a lane 8 dims of each pair's q.  The keys go 8 at a time: a lane's 8 partial dots are summed over the warp
// in one transposed butterfly (9 shuffles for 8 keys, not 5 a key), and the softmax runs per tile of 32 keys (one max
// and one rescale a tile, not one a key); the tile's probabilities pass through shared memory to the P.V sums.
// grid (ceil(T / RB), n_head_kv, n_chunks)
template <int PW, int KV>
__global__ void __launch_bounds__(256) attn_split_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kc,
                                                         const uint16_t* __restrict__ vc, int n_head, int n_head_kv,
                                                         int64_t p0, int T, float scale, float* __restrict__ part,
                                                         int n_chunks, int chunk) {
    __shared__ __align__(16) float sp[8][PW][32];
    __shared__ __align__(16) uint4 sk[32][kD / 8], sv[32][kD / 8];   // the tile's K / V rows (one KV head), FP16
    const int g = blockIdx.y, c = blockIdx.z, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int G = n_head / n_head_kv, RB = 8 * PW / G, r0 = blockIdx.x * RB, nr = min(RB, T - r0);
    float qr[PW][8], acc[PW][8], m[PW], l[PW];
    int64_t lim[PW];   // a pair's last key (its row's position); -1: no pair
#pragma unroll
    for (int j = 0; j < PW; ++j) {
        const int pi = warp * PW + j, r = pi / G, h = g * G + pi % G;
        const bool ok = r < nr;
        lim[j] = ok ? p0 + r0 + r : -1;
        const float* qp = q + ((size_t) (r0 + (ok ? r : 0)) * n_head + h) * kD + lane * 8;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            qr[j][e] = ok ? qp[e] * scale : 0.0f;
            acc[j][e] = 0.0f;
        }
        m[j] = -INFINITY;
        l[j] = 0.0f;
    }
    // the key a lane holds after the butterfly: lane bits 4, 3, 2 (bits 1, 0 hold copies)
    const int kk = ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1);
    const int64_t k0 = (int64_t) c * chunk, k1 = min(k0 + chunk, p0 + r0 + nr);
    for (int64_t kt = k0; kt < k1; kt += 32) {
        const int nk = (int) min((int64_t) 32, k1 - kt);
#pragma unroll
        for (int it = 0; it < 4; ++it) {   // 32 rows x 32 uint4 each for K and V, 256 threads
            const int idx = it * 256 + threadIdx.x, row = idx >> 5, col = idx & 31;
            const uint4 z = make_uint4(0, 0, 0, 0);
            if constexpr (KV != 0) {
                const size_t ch = (size_t) (kt + row) * n_head_kv + g;
                sk[row][col] = row < nk ? kv_ld8h<kv_kf(KV)>(kc, ch, col) : z;
                sv[row][col] = row < nk ? kv_ld8h<kv_vf(KV)>(vc, ch, col) : z;
            } else {
            const size_t off = ((size_t) (kt + row) * n_head_kv + g) * kD + col * 8;
            sk[row][col] = row < nk ? *reinterpret_cast<const uint4*>(kc + off) : z;
            sv[row][col] = row < nk ? *reinterpret_cast<const uint4*>(vc + off) : z;
            }
        }
        __syncthreads();
        float sc[PW][4];
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            float v[PW][8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float kf[8];
                unpack8(sk[sub * 8 + i][lane], kf);
#pragma unroll
                for (int j = 0; j < PW; ++j) {
                    float d = 0.0f;
#pragma unroll
                    for (int e = 0; e < 8; ++e) d += qr[j][e] * kf[e];
                    v[j][i] = d;
                }
            }
            // butterfly: 8 values over 32 lanes -> lane holds key kk's sum
            const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
#pragma unroll
            for (int j = 0; j < PW; ++j) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float send = b4 ? v[j][i] : v[j][i + 4], keep = b4 ? v[j][i + 4] : v[j][i];
                    v[j][i] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const float send = b3 ? v[j][i] : v[j][i + 2], keep = b3 ? v[j][i + 2] : v[j][i];
                    v[j][i] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
                }
                {
                    const float send = b2 ? v[j][0] : v[j][1], keep = b2 ? v[j][1] : v[j][0];
                    v[j][0] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
                }
                float t = v[j][0];
                t += __shfl_xor_sync(0xffffffffu, t, 2);
                t += __shfl_xor_sync(0xffffffffu, t, 1);
                const int64_t kp = kt + sub * 8 + kk;
                sc[j][sub] = (kp < k1 && kp <= lim[j]) ? t : -INFINITY;
            }
        }
        // the tile's softmax, a pair at a time (the 8 key-lanes of a quad-group hold distinct keys: reduce over bits 4..2)
#pragma unroll
        for (int j = 0; j < PW; ++j) {
            float mx = fmaxf(fmaxf(sc[j][0], sc[j][1]), fmaxf(sc[j][2], sc[j][3]));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
            const float mn = fmaxf(m[j], mx);
            float ps = 0.0f, pr[4];
#pragma unroll
            for (int sub = 0; sub < 4; ++sub) {
                pr[sub] = mn == -INFINITY ? 0.0f : expf(sc[j][sub] - mn);
                ps += pr[sub];
            }
            ps += __shfl_xor_sync(0xffffffffu, ps, 16);
            ps += __shfl_xor_sync(0xffffffffu, ps, 8);
            ps += __shfl_xor_sync(0xffffffffu, ps, 4);
            const float corr = m[j] == -INFINITY ? 0.0f : expf(m[j] - mn);
            l[j] = l[j] * corr + ps;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[j][e] *= corr;
            m[j] = mn;
            if ((lane & 3) == 0) {
#pragma unroll
                for (int sub = 0; sub < 4; ++sub) sp[warp][j][sub * 8 + kk] = pr[sub];
            }
        }
        __syncthreads();
        for (int i4 = 0; i4 < nk; i4 += 4) {
            float vf[4][8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                unpack8(sv[i4 + i][lane], vf[i]);
            }
#pragma unroll
            for (int j = 0; j < PW; ++j) {
                const float4 p4 = *reinterpret_cast<const float4*>(&sp[warp][j][i4]);
                const float pv[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[j][e] += pv[i] * vf[i][e];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < PW; ++j) {
        if (lim[j] < 0) continue;
        const int pi = warp * PW + j, r = pi / G, h = g * G + pi % G;
        float* op = part + (((size_t) (r0 + r) * n_head + h) * n_chunks + c) * (kD + 2);
#pragma unroll
        for (int e = 0; e < 8; ++e) op[lane * 8 + e] = acc[j][e];
        if (lane == 0) {
            op[kD] = m[j];
            op[kD + 1] = l[j];
        }
    }
}
__global__ void attn_gate_rows_kernel(float* __restrict__ o, const float* __restrict__ qf, int n_head) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    o[((size_t) t * n_head + h) * kD + d] *= sigmoid_f(qf[((size_t) t * n_head + h) * 2 * kD + kD + d]);
}
// out[0] = exp(l[tok] - max) / sum exp(l - max): one block of 256
__global__ void __launch_bounds__(256) top_prob_kernel(const float* __restrict__ l, int n, const int* __restrict__ tok,
                                                       float* __restrict__ out) {
    __shared__ float sred[8];
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += 256) m = fmaxf(m, l[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) sred[threadIdx.x >> 5] = m;
    __syncthreads();
    m = sred[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) m = fmaxf(m, sred[w]);
    __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) sum += expf(l[i] - m);
    sum = block_sum256(sum, sred);
    if (threadIdx.x == 0) out[0] = expf(l[tok[0]] - m) / sum;
}
// ffn[t] *= sigmoid(x[t] . w): a block a row
__global__ void __launch_bounds__(256) shexp_gate_rows_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                              float* __restrict__ ffn, int n) {
    __shared__ float sred[8];
    const int t = blockIdx.x;
    float dot = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) dot += x[(size_t) t * n + i] * w[i];
    dot = block_sum256(dot, sred);
    const float gs = sigmoid_f(dot);
    for (int i = threadIdx.x; i < n; i += 256) ffn[(size_t) t * n + i] *= gs;
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "qwen35 %s: %s\n", what, cudaGetErrorString(e));
}
// qwen4exp (Qwen3.8-Flash-Next) hyper-connections - llama.cpp's src/models/qwen4exp.cpp build_hc_mix /
// build_hc_combine: the residual is hc streams [hc][n_embd] a row, no norm tensors (the mixer is the norm)
// block (c, t): row t's stream c - first R[c] += out * 2 sigmoid(inject[c] / hc) (the last block's combine, comb the
// identity; out == nullptr: none), then xn[c] = rms(R[c]) * gamma[c]
__global__ void __launch_bounds__(256) hc4_combine_norm_kernel(float* __restrict__ R, const float* __restrict__ out,
                                                               const float* __restrict__ inject,
                                                               const float* __restrict__ gamma, float eps, int n_embd,
                                                               int hc, int inj_stride, float* __restrict__ xn,
                                                               Q81* __restrict__ xq) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    __shared__ float sh[8];
    const int c = blockIdx.x, t = blockIdx.y;
    float* r = R + ((size_t) t * hc + c) * n_embd;
    float ss = 0.0f;
    if (out != nullptr) {
        const float* o = out + (size_t) t * n_embd;
        const float w = 2.0f * sigmoid_f(inject[(size_t) t * inj_stride + c] / (float) hc);
        for (int e = threadIdx.x; e < n_embd; e += 256) {
            const float v = r[e] + o[e] * w;
            r[e] = v;
            ss += v * v;
        }
    } else {
        for (int e = threadIdx.x; e < n_embd; e += 256) ss += r[e] * r[e];
    }
    const float sc = rsqrtf(block_sum256(ss, sh) / (float) n_embd + eps);
    float* x = xn + ((size_t) t * hc + c) * n_embd;
    const float* gm = gamma + (size_t) c * n_embd;
    const size_t q0 = ((size_t) t * hc + c) * n_embd;
    for (int e = threadIdx.x; e < n_embd; e += 256) {
        const float v = r[e] * sc * gm[e];
        x[e] = v;
        if (xq != nullptr) q81_store_warp(v, xq + (q0 + e) / 32, threadIdx.x & 31);   // (n_embd % 256 == 0)
    }
}
// ks[j] = kc[ids[j]], vs[j] = vc[ids[j]]: a block a row, 8 bytes a thread
__global__ void gather_rows_kernel(const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
                                   const int32_t* __restrict__ ids, int row, uint16_t* __restrict__ ks,
                                   uint16_t* __restrict__ vs) {
    const size_t src = (size_t) ids[blockIdx.x] * row, dst = (size_t) blockIdx.x * row;
    for (int i = threadIdx.x * 4; i < row; i += blockDim.x * 4) {
        *(uint2*) (ks + dst + i) = *(const uint2*) (kc + src + i);
        *(uint2*) (vs + dst + i) = *(const uint2*) (vc + src + i);
    }
}
// lo[i] = silu(lo[i] * a)
__global__ void silu_scale_kernel(float* __restrict__ x, int n, float a, Q81* __restrict__ xq) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {   // (with xq: n % 32 == 0, whole warps)
        const float v = x[i] * a;
        const float y = v / (1.0f + expf(-v));
        x[i] = y;
        if (xq != nullptr) q81_store_warp(y, xq + i / 32, threadIdx.x & 31);
    }
}
// mixed[t][e] = (1 / hc) sum_c xn[t][c][e] sigmoid(gate[t][c][e])
__global__ void hc4_mix_kernel(const float* __restrict__ xn, const float* __restrict__ gate, int n_embd, int hc,
                               float* __restrict__ mixed, Q81* __restrict__ xq) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int e = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (e >= n_embd) return;   // (with xq: n_embd % 32 == 0, whole warps)
    const size_t b = (size_t) t * hc * n_embd + e;
    float m = 0.0f;
    for (int c = 0; c < hc; ++c) m += xn[b + (size_t) c * n_embd] * sigmoid_f(gate[b + (size_t) c * n_embd]);
    m = m / (float) hc;
    mixed[(size_t) t * n_embd + e] = m;
    if (xq != nullptr) q81_store_warp(m, xq + ((size_t) t * n_embd + e) / 32, threadIdx.x & 31);
}
// ---- qwen4exp's hyper-connection mix in two launches (hc4_fused): the arithmetic of hc4_combine_norm_kernel,
// glm_fast's BF16 row dot (row_dot_bf16 / mv_kernel: a warp a row, 8 weights a lane a step, then the warp's sum),
// silu_scale_kernel and hc4_mix_kernel, expression for expression - the outputs are the same bits
constexpr int kHcMaxHE = 10240;
__device__ __forceinline__ float hc_bf16_dot8(const uint4 w, const float4 a, const float4 b) {
    return __uint_as_float(w.x << 16) * a.x + __uint_as_float(w.x & 0xffff0000u) * a.y +
           __uint_as_float(w.y << 16) * a.z + __uint_as_float(w.y & 0xffff0000u) * a.w +
           __uint_as_float(w.z << 16) * b.x + __uint_as_float(w.z & 0xffff0000u) * b.y +
           __uint_as_float(w.w << 16) * b.z + __uint_as_float(w.w & 0xffff0000u) * b.w;
}
// Q8_0 (the UD 3.1 packs' hc down / up): glm_fast's dot_q8_0 against a q8_1 activation, expression for expression
// (row: the weight row, 34-byte blocks; x: the activation's q8_1 blocks; call k = block kbx's ints iqs, iqs + 1)
__device__ __forceinline__ int hc_int_b2(const uint8_t* x, int i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__device__ __forceinline__ float hc_q8_0_dot(const uint8_t* row, const Q81* x, int kbx, int iqs) {
    const uint8_t* b = row + (size_t) kbx * 34;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = hc_int_b2(b + 2, iqs + i);
        const int u = ((const int*) x[kbx].qs)[iqs + i];
        sumi = __dp4a(v, u, sumi);
    }
    return __half2float(*(const __half*) b) * __low2float(x[kbx].ds) * float(sumi);
}
// a warp's share of [w, w + bytes) into L2 (glm_fast's mv_prefetch_l2): weights only, so before STRATA_PDL_ENTRY
__device__ __forceinline__ void hc_prefetch_l2(const void* w, size_t bytes, int lane) {
#if defined(__CUDA_ARCH__) && !defined(STRATA_USE_HIP)
    const char* c = (const char*) w;
    for (size_t o = (size_t) lane * 128; o < bytes; o += (size_t) 32 * 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(c + o));
#else
    (void) w; (void) bytes; (void) lane;
#endif
}
// a: block b - the hc streams' combine (R + out * 2 sigmoid(inj_prev / hc), not written back) and grouped norm into
// shared memory (each stream as hc4_combine_norm_kernel's block: 256 threads, the same partial sums), then its 8 warps'
// rows of down (lr rows) and inject (hc rows after them): y = the row's dot + 0 (mv's alpha 1, no bias)
__global__ void __launch_bounds__(256) hc4_fused_a_kernel(const float* __restrict__ R, const float* __restrict__ out,
                                                          const float* __restrict__ inj_prev,
                                                          const float* __restrict__ gamma, float eps, int n_embd,
                                                          int hc, const uint16_t* __restrict__ down, int lr,
                                                          const void* __restrict__ inject, bool inj_f32,
                                                          float* __restrict__ lo, float* __restrict__ inj_out,
                                                          float* __restrict__ sc_out, int inj_stride) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    __shared__ __align__(16) float xs[kHcMaxHE];
    {   // column blockIdx.y of a window (hc4_fused_cols; the token path's grid has one row)
        const int t = blockIdx.y;
        R += (size_t) t * hc * n_embd;
        if (out != nullptr) out += (size_t) t * n_embd;
        inj_prev += (size_t) t * inj_stride;
        lo += (size_t) t * lr;
        if (inj_out != nullptr) inj_out += (size_t) t * hc;
        sc_out += (size_t) t * 4;
    }
    __shared__ float sh[4][8];
    __shared__ float scs[4];
    // the four streams at once (hc == 4): each thread's partial sum of a stream runs over the same elements in the same
    // order as hc4_combine_norm_kernel's thread, and the four block sums take block_sum256's tree - two barriers
    // instead of eight, and the four streams' reads in flight together
    float ss[4] = {0.0f, 0.0f, 0.0f, 0.0f}, w[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (out != nullptr)
        for (int c = 0; c < 4; ++c) w[c] = 2.0f * sigmoid_f(inj_prev[c] / (float) hc);
    for (int e = threadIdx.x; e < n_embd; e += 256) {
        const float o = out != nullptr ? out[e] : 0.0f;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float r = R[(size_t) c * n_embd + e];
            if (out != nullptr) {
                const float v = r + o * w[c];
                xs[c * n_embd + e] = v;
                ss[c] += v * v;
            } else {
                xs[c * n_embd + e] = r;
                ss[c] += r * r;
            }
        }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const float v = warp_sum(ss[c]);
        if ((threadIdx.x & 31) == 0) sh[c][threadIdx.x >> 5] = v;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            float t = threadIdx.x < 8 ? sh[c][threadIdx.x] : 0.0f;
            t = warp_sum(t);
            if (threadIdx.x == 0) {
                const float sc = rsqrtf(t / (float) n_embd + eps);
                scs[c] = sc;
                if (blockIdx.x == 0) sc_out[c] = sc;
            }
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_embd; e += 256) {
#pragma unroll
        for (int c = 0; c < 4; ++c) xs[c * n_embd + e] = xs[c * n_embd + e] * scs[c] * gamma[(size_t) c * n_embd + e];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5), n_in = hc * n_embd;
    const uint16_t* wr;
    float* y;
    if (row < lr) {
        wr = down + (size_t) row * n_in;
        y = lo + row;
    } else if (inject != nullptr && row < lr + hc) {
        y = inj_out + (row - lr);
        if (inj_f32) {   // (the injects are widened to F32 at load: row_dot_f32's sum)
            const float4* r4 = (const float4*) ((const float*) inject + (size_t) (row - lr) * n_in);
            const float4* x4 = (const float4*) xs;
            float s = 0.0f;
            for (int k = lane; k < n_in / 4; k += 32) {
                const float4 a = r4[k], b = x4[k];
                s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
            s = warp_sum(s);
            if (lane == 0) *y = s + 0.0f;
            return;
        }
        wr = (const uint16_t*) inject + (size_t) (row - lr) * n_in;
    } else {
        return;
    }
    const uint4* r8 = (const uint4*) wr;
    const float4* x4 = (const float4*) xs;
    float s = 0.0f;
#pragma unroll 8
    for (int k = lane; k < n_in / 8; k += 32) {
        const uint4 w = r8[k];
        const float4 a = x4[2 * k], b = x4[2 * k + 1];
        s += hc_bf16_dot8(w, a, b);
    }
    s = warp_sum(s);
    if (lane == 0) *y = s + 0.0f;
}
// a, Q8_0 down (hc4_fused_a_q8_kernel): mv()'s down is mv_kernel_ts<8, DS> - DS warps a row, warp p summing the
// calls of its k range, the DS sums added 0 -> DS - 1 - and with lo % 32 == 0 range p is streams 4p / DS .. 4(p+1) / DS
// - 1 of xn exactly.  So block (g, p) needs only those streams: their combine and norm (hc4_combine_norm_kernel's
// sums, as hc4_fused_a_kernel's), their q8_1 in shared memory (the same bytes), then its 8 warps' range-p sums of rows
// 8g .. 8g + 7 to part[p][row] (the warp sum, as mv_kernel_ts' s_part); b adds the parts in order.  The inject rows
// (F32, row_dot_f32's chain over all four streams: not divisible) are the last block's, with the four streams' floats -
// its weight rows pulled into L2 before the PDL wait and read 8 float4 ahead (the same adds in the same order)
template<int DS>
__global__ void __launch_bounds__(256) hc4_fused_a_q8_kernel(const float* __restrict__ R, const float* __restrict__ out,
                                                             const float* __restrict__ inj_prev,
                                                             const float* __restrict__ gamma, float eps, int n_embd,
                                                             int hc, const uint8_t* __restrict__ down, int lr,
                                                             const float* __restrict__ inject,
                                                             float* __restrict__ part, float* __restrict__ inj_out,
                                                             float* __restrict__ sc_out, int inj_stride) {
    constexpr int NS = 4 / DS;   // (streams a down block)
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, n_in = hc * n_embd;
    const int ndb = lr / 8 * DS;   // (down blocks; the inject block after them)
    const bool injb = (int) blockIdx.x >= ndb;
    const int g = blockIdx.x / DS, p = blockIdx.x % DS;
    const int c0 = injb ? 0 : p * NS, nc = injb ? 4 : NS;
    const size_t rb = (size_t) n_in / 32 * 34;
    // the inject block: warp w takes row w % 4's lane chains, steps j (k = lane + 32 j) of half w / 4 - the first half
    // summed as it goes, the second's terms kept in registers and added onto the first's chain after it (row_dot_f32's
    // adds, in order); each warp's first 8 weight float4 loaded before the PDL wait, its half pulled into L2
    constexpr int kJh = kHcMaxHE / 4 / 32 / 2;   // (a half's steps, at most)
    const int Jh = n_in / 4 / 32 / 2, ih = warp >> 2, ir = warp & 3;
    const float4* r4 = injb && inject != nullptr ? (const float4*) (inject + (size_t) ir * n_in) + lane + 32 * ih * Jh : nullptr;
    float4 pre[8];
    if (!injb) {
        hc_prefetch_l2(down + (size_t) (g * 8 + warp) * rb + (size_t) c0 * n_embd / 32 * 34, (size_t) nc * n_embd / 32 * 34,
                       lane);
    } else if (r4 != nullptr) {
        hc_prefetch_l2((const float*) (inject + (size_t) ir * n_in) + (size_t) ih * Jh * 128, (size_t) Jh * 512, lane);
#pragma unroll
        for (int u = 0; u < 8; ++u) pre[u] = r4[32 * u];
    }
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any read of what an earlier kernel writes)
    if (injb && inject == nullptr) return;
    __shared__ __align__(16) float xs[kHcMaxHE];
    {   // column blockIdx.y of a window
        const int t = blockIdx.y;
        R += (size_t) t * hc * n_embd;
        if (out != nullptr) out += (size_t) t * n_embd;
        inj_prev += (size_t) t * inj_stride;
        part += (size_t) t * DS * lr;
        if (inj_out != nullptr) inj_out += (size_t) t * hc;
        sc_out += (size_t) t * 4;
    }
    __shared__ float sh[4][8];
    __shared__ float scs[4];
    float ss[4] = {0.0f, 0.0f, 0.0f, 0.0f}, w[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (out != nullptr)
#pragma unroll
        for (int c = 0; c < 4; ++c)
            if (c < nc) w[c] = 2.0f * sigmoid_f(inj_prev[c0 + c] / (float) hc);
    constexpr int kNe = kHcMaxHE / 4 / 256;   // (a thread's elements a stream, at most: unrolled, the loads in flight)
#pragma unroll
    for (int ie = 0; ie < kNe; ++ie) {
        const int e = threadIdx.x + 256 * ie;
        if (e >= n_embd) break;
        const float o = out != nullptr ? out[e] : 0.0f;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (c >= nc) break;
            const int cs = c0 + c;
            const float r = R[(size_t) cs * n_embd + e];
            if (out != nullptr) {
                const float v = r + o * w[c];
                xs[c * n_embd + e] = v;
                ss[c] += v * v;
            } else {
                xs[c * n_embd + e] = r;
                ss[c] += r * r;
            }
        }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        if (c >= nc) break;
        const float v = warp_sum(ss[c]);
        if (lane == 0) sh[c][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (c >= nc) break;
            float t = threadIdx.x < 8 ? sh[c][threadIdx.x] : 0.0f;
            t = warp_sum(t);
            if (threadIdx.x == 0) {
                const float sc = rsqrtf(t / (float) n_embd + eps);
                scs[c] = sc;
                if (!injb && g == 0) sc_out[c0 + c] = sc;
            }
        }
    }
    __syncthreads();
    if (injb) {
#pragma unroll
        for (int ie = 0; ie < kNe; ++ie) {
            const int e = threadIdx.x + 256 * ie;
            if (e >= n_embd) break;
#pragma unroll
            for (int c = 0; c < 4; ++c) xs[c * n_embd + e] = xs[c * n_embd + e] * scs[c] * gamma[(size_t) c * n_embd + e];
        }
        __syncthreads();
        __shared__ float chain[4][32];
        const float4* x4 = (const float4*) xs + lane + 32 * ih * Jh;
        float s = 0.0f, tm[kJh];
#pragma unroll
        for (int c8 = 0; c8 < kJh / 8; ++c8)
            if (c8 * 8 < Jh) {   // (Jh % 8 == 0: n_embd % 256 == 0)
                float4 a[8];
#pragma unroll
                for (int u = 0; u < 8; ++u) a[u] = c8 == 0 ? pre[u] : r4[32 * (c8 * 8 + u)];
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const float4 b = x4[32 * (c8 * 8 + u)];
                    const float term = a[u].x * b.x + a[u].y * b.y + a[u].z * b.z + a[u].w * b.w;
                    if (ih == 0) s += term;
                    else tm[c8 * 8 + u] = term;
                }
            }
        if (ih == 0) chain[ir][lane] = s;
        __syncthreads();
        if (ih == 0) return;
        s = chain[ir][lane];
#pragma unroll
        for (int j = 0; j < kJh; ++j)
            if (j < Jh) s += tm[j];
        s = warp_sum(s);
        if (lane == 0) inj_out[ir] = s + 0.0f;
        return;
    }
    // the streams' xn, each thread's values first (n_embd <= 2560: 10 a stream), then their q8_1 over the floats
    constexpr int kNi = kHcMaxHE / 4 / 256;
    const int ni = n_embd / 256;
    float v[kNi][NS];
#pragma unroll
    for (int i = 0; i < kNi; ++i)
        if (i < ni) {
            const int e = threadIdx.x + 256 * i;
#pragma unroll
            for (int c = 0; c < NS; ++c) v[i][c] = xs[c * n_embd + e] * scs[c] * gamma[(size_t) (c0 + c) * n_embd + e];
        }
    __syncthreads();
    Q81* xq = (Q81*) xs;   // (block kb of the streams' run: kb = (c * n_embd + e) / 32, the call k's block k / 4 - k0 / 4)
#pragma unroll
    for (int i = 0; i < kNi; ++i)
        if (i < ni) {
            const int e = threadIdx.x + 256 * i;
#pragma unroll
            for (int c = 0; c < NS; ++c) q81_store_warp(v[i][c], xq + (c * n_embd + e) / 32, lane);
        }
    __syncthreads();
    const int row = g * 8 + warp, nk = n_in / 32 * 4, k0 = nk * p / DS, k1 = nk * (p + 1) / DS;
    const uint8_t* wr = down + (size_t) row * rb;
    float s = 0.0f;
    for (int k = k0 + lane; k < k1; k += 32) s += hc_q8_0_dot(wr, xq - k0 / 4, k / 4, 2 * (k % 4));
    s = warp_sum(s);
    if (lane == 0) part[(size_t) p * lr + row] = s;
}
// ---- the window's hyper-connection mix over T columns in two launches (hc4_fused_cols): q4x_mix_cols' five launches
// (hc4_combine_norm over the columns, mvn down + inject, silu_scale, mvn up, hc4_mix), each column's sums in
// mvn_kernel_f's order (the same as mv's: a warp a row, 8 weights a lane a step, then the warp's sum) - the same bits
// a: hc4_fused_a_kernel with a column a grid row (blockIdx.y; two columns a block in 80 KB of shared memory measured
// slower: 14 against 9-10 us a launch, fewer blocks in flight)
// b: block b - embedding columns e0 = 32 b .. e0 + 31 of every window column: silu(lo / hc) of the T columns into
// shared memory, each warp's 8 up rows loaded once and summed against every column (mv's order), then warp t: column
// t's R combine written, xn, the mix over the streams and its q8_1
template<int WT>
__global__ void __launch_bounds__(512) hc4_fused_cols_b_kernel(float* __restrict__ R, const float* __restrict__ out,
                                                               const float* __restrict__ inj_prev, int inj_stride,
                                                               const float* __restrict__ gamma, int n_embd, int hc,
                                                               int T, const uint16_t* __restrict__ up, int lr,
                                                               const float* __restrict__ lo, float a,
                                                               const float* __restrict__ sc, float* __restrict__ xn,
                                                               int ds,
                                                               float* __restrict__ gate, float* __restrict__ mixed,
                                                               Q81* __restrict__ xq) {
    __shared__ __align__(16) float ls[8 * 512];
    __shared__ float gs[8 * 4 * 32];
    __shared__ Q81 lq[WT == 8 ? 8 * 16 : 1];   // (Q8_0 up: lo's q8_1 a column, silu_scale_kernel's bytes)
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, e0 = blockIdx.x * 32, HE = hc * n_embd;
    const int nk = lr / 8;   // (a row's calls: BF16 8 weights, Q8_0 a quarter block - lr / 8 either way)
    uint4 w0[8], w1[8];      // (Q8_0: x, y the call's two ints, z its block's scale)
#pragma unroll
    for (int j = 0; j < 8; ++j) {   // (the weights: nothing an earlier kernel writes - loaded before the PDL wait)
        const int rl = warp * 8 + j;
        const bool ok = (rl >> 5) < hc;
        const size_t ro = (size_t) ((ok ? rl >> 5 : 0) * n_embd + e0 + (rl & 31));
        if constexpr (WT == 8) {
            const uint8_t* rb = (const uint8_t*) up + ro * (size_t) (lr / 32 * 34);
            w0[j] = make_uint4(0, 0, 0, 0);
            w1[j] = make_uint4(0, 0, 0, 0);
            if (ok && lane < nk) {
                const int k = lane, kb = k / 4, iq = 2 * (k % 4);
                w0[j] = make_uint4((unsigned) hc_int_b2(rb + kb * 34 + 2, iq), (unsigned) hc_int_b2(rb + kb * 34 + 2, iq + 1),
                                   (unsigned) *(const uint16_t*) (rb + kb * 34), 0);
            }
            if (ok && lane + 32 < nk) {
                const int k = lane + 32, kb = k / 4, iq = 2 * (k % 4);
                w1[j] = make_uint4((unsigned) hc_int_b2(rb + kb * 34 + 2, iq), (unsigned) hc_int_b2(rb + kb * 34 + 2, iq + 1),
                                   (unsigned) *(const uint16_t*) (rb + kb * 34), 0);
            }
        } else {
            const uint4* r8 = (const uint4*) (up + ro * lr);
            w0[j] = ok && lane < nk ? r8[lane] : make_uint4(0, 0, 0, 0);
            w1[j] = ok && lane + 32 < nk ? r8[lane + 32] : make_uint4(0, 0, 0, 0);
        }
    }
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any read of what an earlier kernel writes)
    for (int i = threadIdx.x; i < T * lr; i += 512) {   // (silu_scale_kernel's expression)
        float l;
        if constexpr (WT == 8) {   // (lo: a's ds parts a row, [T][ds][lr] - mv_kernel_ts' sum of them, then + 0)
            const int t = i / lr, r = i % lr;
            l = 0.0f;
            for (int q = 0; q < ds; ++q) l += lo[((size_t) t * ds + q) * lr + r];
            l = l + 0.0f;
        } else {
            l = lo[i];
        }
        const float v = l * a;
        ls[(i / lr) * 512 + i % lr] = v / (1.0f + expf(-v));
    }
    __syncthreads();
    if constexpr (WT == 8) {   // (lr % 32 == 0: a q8_1 block never spans two columns, as in silu_scale's flat run)
        const int nqb = lr / 32;
        for (int pq = warp; pq < T * nqb; pq += 16)
            q81_store_warp(ls[(pq / nqb) * 512 + (pq % nqb) * 32 + lane], lq + (pq / nqb) * 16 + pq % nqb, lane);
        __syncthreads();
    }
    // (hc == 4: the 16 warps' 8 rows are the 128 rows of the 32 columns' 4 streams)
    for (int t = 0; t < T; ++t) {
        const float4* x4 = (const float4*) (ls + t * 512);
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float s = 0.0f;
            if constexpr (WT == 8) {   // (dot_q8_0: the two dp4a, then d_w d_x sumi)
                const Q81* xq = lq + t * 16;
                if (lane < nk) {
                    const int kb = lane / 4, iq = 2 * (lane % 4);
                    int sumi = __dp4a((int) w0[j].x, ((const int*) xq[kb].qs)[iq], 0);
                    sumi = __dp4a((int) w0[j].y, ((const int*) xq[kb].qs)[iq + 1], sumi);
                    s += __half2float(__ushort_as_half((unsigned short) w0[j].z)) * __low2float(xq[kb].ds) * float(sumi);
                }
                if (lane + 32 < nk) {
                    const int k = lane + 32, kb = k / 4, iq = 2 * (k % 4);
                    int sumi = __dp4a((int) w1[j].x, ((const int*) xq[kb].qs)[iq], 0);
                    sumi = __dp4a((int) w1[j].y, ((const int*) xq[kb].qs)[iq + 1], sumi);
                    s += __half2float(__ushort_as_half((unsigned short) w1[j].z)) * __low2float(xq[kb].ds) * float(sumi);
                }
            } else {
                if (lane < nk) s += hc_bf16_dot8(w0[j], x4[2 * lane], x4[2 * lane + 1]);
                if (lane + 32 < nk) s += hc_bf16_dot8(w1[j], x4[2 * (lane + 32)], x4[2 * (lane + 32) + 1]);
            }
            v[j] = s;
        }
        // the 8 rows' warp sums as a reduce-scatter: each xor step of warp_sum's butterfly (v + the partner's v) on
        // the half of the rows a lane keeps - every row's sum the same pairs in the same order, 9 shuffles, not 40
        const bool h4 = (lane & 16) != 0, h3 = (lane & 8) != 0, h2 = (lane & 4) != 0;
        float u[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float r = __shfl_xor_sync(0xffffffffu, h4 ? v[i] : v[4 + i], 16);
            u[i] = (h4 ? v[4 + i] : v[i]) + r;
        }
        float q[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float r = __shfl_xor_sync(0xffffffffu, h3 ? u[i] : u[2 + i], 8);
            q[i] = (h3 ? u[2 + i] : u[i]) + r;
        }
        float z;
        {
            const float r = __shfl_xor_sync(0xffffffffu, h2 ? q[0] : q[1], 4);
            z = (h2 ? q[1] : q[0]) + r;
        }
        z += __shfl_xor_sync(0xffffffffu, z, 2);
        z += __shfl_xor_sync(0xffffffffu, z, 1);
        if ((lane & 3) == 0) {
            const int j = (h4 ? 4 : 0) + (h3 ? 2 : 0) + (h2 ? 1 : 0), rl = warp * 8 + j;
            const int row = (rl >> 5) * n_embd + e0 + (rl & 31);
            const float gv = z + 0.0f;
            gs[t * 128 + rl] = gv;
            gate[(size_t) t * HE + row] = gv;
        }
    }
    __syncthreads();
    if (warp >= T) return;
    const int t = warp, e = e0 + lane;
    float m = 0.0f;
    for (int c = 0; c < hc; ++c) {
        float* r = R + (size_t) t * HE + (size_t) c * n_embd;
        float v = r[e];
        if (out != nullptr) {
            const float w = 2.0f * sigmoid_f(inj_prev[(size_t) t * inj_stride + c] / (float) hc);
            v = r[e] + out[(size_t) t * n_embd + e] * w;
            r[e] = v;
        }
        const float xv = v * sc[(size_t) t * 4 + c] * gamma[(size_t) c * n_embd + e];
        xn[(size_t) t * HE + (size_t) c * n_embd + e] = xv;
        m += xv * sigmoid_f(gs[t * 128 + c * 32 + lane]);
    }
    m = m / (float) hc;
    mixed[(size_t) t * n_embd + e] = m;
    if (xq != nullptr) q81_store_warp(m, xq + ((size_t) t * n_embd + e) / 32, lane);
}
}  // namespace

bool hc4_fused_ok(int n_embd, int hc, int lr, int down_type, int up_type, int inject_type) {
    constexpr int kBF16 = 30, kQ8_0 = 8;   // (ggml's GGML_TYPE_BF16 / Q8_0, glm_fast's kTypeBF16)
    if (!(hc == 4 && n_embd % 256 == 0 && hc * n_embd <= kHcMaxHE && lr % 8 == 0 && lr > 0 && lr <= 512)) return false;
    if (down_type == kBF16 && up_type == kBF16) return inject_type < 0 || inject_type == kBF16 || inject_type == 0;
    // Q8_0 down / up (unsloth UD 3.1): F32 or no inject, and mv()'s own split for these shapes - down 1, 2 or 4 warps a
    // row, up one warp (mv_kernel_t<8>); anything else (STRATA_GLM_MV_SPLIT, the generic kernel, HIP) keeps the five
    if (down_type == kQ8_0 && up_type == kQ8_0 && (inject_type < 0 || inject_type == 0) && lr % 32 == 0 &&
        n_embd % 512 == 0) {   // (the inject block's halves: whole 8-step chunks)
        const int ds = strata::kernels::glmf::mv_split_rule(kQ8_0, lr);
        return (ds == 1 || ds == 2 || ds == 4) && strata::kernels::glmf::mv_split_rule(kQ8_0, hc * n_embd) == 1;
    }
    return false;
}
void hc4_fused(float* R, const float* out, const float* inj_prev, const float* gamma, float eps, int n_embd, int hc,
               int lr, const void* down, const void* inject, const void* up, float* lo, float* inj_out, float* sc,
               float* xn, float* gate, float* x, void* xq, cudaStream_t s, bool inject_f32, int wtype, void* scratch) {
    hc4_fused_cols(R, out, inj_prev, hc, gamma, eps, n_embd, hc, lr, 1, down, inject, up, lo, inj_out, sc, xn, gate, x,
                   xq, s, inject_f32, wtype, scratch);
}
bool hc4_fused_cols_ok(int n_embd, int hc, int lr, int T, int down_type, int up_type, int inject_type) {
    return T >= 1 && T <= 8 && hc4_fused_ok(n_embd, hc, lr, down_type, up_type, inject_type);
}
void hc4_fused_cols(float* R, const float* out, const float* inj_prev, int inj_stride, const float* gamma, float eps,
                    int n_embd, int hc, int lr, int T, const void* down, const void* inject, const void* up, float* lo,
                    float* inj_out, float* sc, float* xn, float* gate, float* x, void* xq, cudaStream_t s,
                    bool inject_f32, int wtype, void* scratch) {
    if (T <= 0) return;
    if (wtype == 8) {   // (Q8_0: a's down parts in scratch, [T][ds][lr] floats)
        const int ds = strata::kernels::glmf::mv_split_rule(8, lr);
        float* part = (float*) scratch;
        const dim3 ga((unsigned) (lr / 8 * ds + (inject != nullptr ? 1 : 0)), (unsigned) T);
#define HC4_AQ(DS)                                                                                                     \
        pdl_launch(hc4_fused_a_q8_kernel<DS>, ga, 256, 0, s, R, out, inj_prev, gamma, eps, n_embd, hc,                 \
                   (const uint8_t*) down, lr, (const float*) inject, part, inj_out, sc, inj_stride)
        if (ds == 4) HC4_AQ(4);
        else if (ds == 2) HC4_AQ(2);
        else HC4_AQ(1);
#undef HC4_AQ
        check("hc4_fused_cols_a (Q8_0)");
        pdl_launch(hc4_fused_cols_b_kernel<8>, n_embd / 32, 512, 0, s, R, out, inj_prev, inj_stride, gamma, n_embd, hc,
                   T, (const uint16_t*) up, lr, (const float*) part, 1.0f / (float) hc, sc, xn, ds, gate, x, (Q81*) xq);
        check("hc4_fused_cols_b (Q8_0)");
        return;
    }
    const int rows = lr + (inject != nullptr ? hc : 0);
    pdl_launch(hc4_fused_a_kernel, dim3((unsigned) ((rows + 7) / 8), (unsigned) T), 256, 0, s,
        R, out, inj_prev, gamma, eps, n_embd, hc, (const uint16_t*) down, lr, inject, inject_f32, lo, inj_out, sc,
        inj_stride);
    check("hc4_fused_cols_a");
    pdl_launch(hc4_fused_cols_b_kernel<30>, n_embd / 32, 512, 0, s, R, out, inj_prev, inj_stride, gamma, n_embd, hc, T,
               (const uint16_t*) up, lr, lo, 1.0f / (float) hc, sc, xn, 1, gate, x, (Q81*) xq);
    check("hc4_fused_cols_b");
}

void silu(float* x, int n, cudaStream_t s) {
    if (n <= 0) return;
    silu_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, n);
    check("silu");
}
void scale(float* x, int n, float a, cudaStream_t s) {
    if (n <= 0) return;
    scale_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, n, a);
    check("scale");
}
void gdn_gates(const float* alpha, const float* dt_bias, const float* ssm_a, float* beta, float* gate, int h_v,
               cudaStream_t s) {
    if (h_v <= 0) return;
    gdn_gates_kernel<<<(h_v + 127) / 128, 128, 0, s>>>(alpha, dt_bias, ssm_a, beta, gate, h_v);
    check("gdn_gates");
}
// fn(std::integral_constant<int, kv>) for a quantized cache format kv (1..3; the FP16 calls stay as they were)
template <class Fn>
static void kv_switch(int kv, Fn&& fn) {
    switch (kv) {
        case 1: fn(std::integral_constant<int, 1>{}); break;
        case 2: fn(std::integral_constant<int, 2>{}); break;
        default: fn(std::integral_constant<int, 3>{}); break;
    }
}
void attn_prep(const float* q_full, const float* k, const float* v, const float* q_norm, const float* k_norm, float eps,
               int n_head, int n_head_kv, int head_dim, int n_rot, float rope_base, int p, uint16_t* kc, uint16_t* vc,
               float* q, cudaStream_t s, int kv) {
    if (head_dim != kD) {
        std::fprintf(stderr, "qwen35 attn_prep: head_dim %d (the kernels take %d)\n", head_dim, kD);
        return;
    }
    if (kv != 0) {
        kv_switch(kv, [&](auto K) {
            attn_prep_q4_kernel<K()><<<dim3((unsigned) (n_head + n_head_kv), 1), kD, 0, s>>>(
                q_full, k, v, q_norm, k_norm, eps, n_head, n_head_kv, n_rot, rope_base, (int64_t) p, kc, vc, q);
        });
        check("attn_prep");
        return;
    }
    pdl_launch(attn_prep_kernel, n_head + n_head_kv, kD, 0, s, q_full, k, v, q_norm, k_norm, eps, n_head, n_head_kv, n_rot,
                                                       rope_base, p, kc, vc, q);
    check("attn_prep");
}
// attn_split_kernel over the keys < p0 + T for T rows; part holds at least kMinChunks chunks a row and head.
// false: a shape it does not take (the caller runs its old kernel)
static bool attn_split(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_head_kv, int64_t p0,
                       int T, float scale, float* part, float* out, cudaStream_t s, int kv = 0) {
    static const int mode = [] { const char* v = getenv("STRATA_Q35_ATTN"); return v ? std::atoi(v) : 0; }();
    static const int chunk_env = [] { const char* v = getenv("STRATA_Q35_ATTN_CHUNK"); return v ? std::atoi(v) : 0; }();
    const int G = n_head_kv > 0 ? n_head / n_head_kv : 0;
    if (mode == 1 || G * n_head_kv != n_head || G % 8 != 0 || G > 64) return false;
    static const int pw_max = [] { const char* v = getenv("STRATA_Q35_ATTN_PW"); return v ? std::atoi(v) : 2; }();
    const int RB = std::max(1, std::min(T, pw_max * 8 / G)), PW = RB * G / 8;
    const int64_t n_keys = p0 + T;
    static const int chunk_min = [] { const char* v = getenv("STRATA_Q35_ATTN_CHUNK_MIN"); return v ? std::max(32, std::atoi(v)) : 64; }();
    // ~128 chunks a KV group (part holds kMinChunks), at least chunk_min keys each (short contexts: more blocks), a
    // multiple of 32
    const int64_t fit = (n_keys + kMinChunks - 2) / (kMinChunks - 1);   // the smallest chunk that fits part
    int chunk = chunk_env >= 32 ? chunk_env : (int) std::max<int64_t>(chunk_min, n_keys / 128);
    chunk = (int) (std::max<int64_t>(chunk, fit) + 31) / 32 * 32;
    const int n_chunks = (int) ((n_keys + chunk - 1) / chunk);
    const dim3 grid((unsigned) ((T + RB - 1) / RB), (unsigned) n_head_kv, (unsigned) n_chunks);
    switch (PW) {
#define Q35_SPLIT(n) case n: if (kv != 0) kv_switch(kv, [&](auto K) { attn_split_kernel<n, K()><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, part, n_chunks, chunk); }); \
                              else attn_split_kernel<n, 0><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, part, n_chunks, chunk); break;
        Q35_SPLIT(1) Q35_SPLIT(2) Q35_SPLIT(3) Q35_SPLIT(4) Q35_SPLIT(5) Q35_SPLIT(6) Q35_SPLIT(7) Q35_SPLIT(8)
#undef Q35_SPLIT
        default: return false;
    }
    pdl_launch(attn_merge_kernel, dim3((unsigned) n_head, (unsigned) T), kD, 0, s, part, n_chunks, out);
    return true;
}
size_t attn_part_floats(int n_head, int head_dim, int max_ctx) {
    const size_t chunks = std::max<size_t>(((size_t) max_ctx + kChunk - 1) / kChunk + 1, kMinChunks);
    return (size_t) n_head * chunks * ((size_t) head_dim + 2);
}
void attn_decode(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_head_kv, int head_dim,
                 int n_pos, float scale, float* part, float* out, cudaStream_t s, int kv) {
    if (head_dim != kD || n_pos <= 0 || n_head % n_head_kv != 0) {
        std::fprintf(stderr, "qwen35 attn_decode: head_dim %d, n_pos %d, heads %d / %d\n", head_dim, n_pos, n_head, n_head_kv);
        return;
    }
    if (attn_split(q, kc, vc, n_head, n_head_kv, n_pos - 1, 1, scale, part, out, s, kv)) {
        check("attn_decode");
        return;
    }
    const int n_chunks = (n_pos + kChunk - 1) / kChunk;
    if (kv != 0)
        kv_switch(kv, [&](auto K) {
            attn_part_kernel<K()><<<dim3((unsigned) n_head, (unsigned) n_chunks), kWarps * 32, 0, s>>>(
                q, kc, vc, n_head, n_head_kv, n_pos, scale, part, n_chunks);
        });
    else
    attn_part_kernel<0><<<dim3((unsigned) n_head, (unsigned) n_chunks), kWarps * 32, 0, s>>>(q, kc, vc, n_head, n_head_kv,
                                                                                            n_pos, scale, part, n_chunks);
    pdl_launch(attn_merge_kernel, n_head, kD, 0, s, part, n_chunks, out);
    check("attn_decode");
}
size_t attn_rows_part_floats(int rows, int n_head, int head_dim, int max_ids) {
    return (size_t) rows * n_head * (size_t) ((max_ids + kChunk - 1) / kChunk) * ((size_t) head_dim + 2);
}
static int rows_hpw(int G, int rows, int n_chunks) {
    static const int env = [] { const char* v = getenv("STRATA_Q35_ROWS_HPW"); return v ? std::atoi(v) : 0; }();
    // RTX 5090, 24 / 2 heads, 2,051 cells (qsa_rows_parity): a row 2 (1.2x gather + attn_decode; 6: 0.8x), 2-5 rows
    // 4 (2.1-4.4x), more 6 (5.3x at 8, 6.5x at 64)
    int want = env > 0 ? env : rows <= 1 ? 2 : rows <= 5 ? 4 : 6;
    (void) n_chunks;
    for (; want > 1; --want)
        if (want != 5 && G % want == 0) break;
    return want;
}
bool attn_decode_rows(const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* ids, int ld_ids,
                      const int32_t* steps, int rows, int max_ids, int n_head, int n_head_kv, int head_dim, float scale,
                      float* part, float* out, cudaStream_t s, int hpw) {
    if (rows <= 0) return true;
    if (head_dim != kD || max_ids <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0) return false;
    // attn_decode's own kernel for this shape: attn_split takes G % 8 == 0 (G <= 64) unless STRATA_Q35_ATTN=1, and
    // the rows here must match attn_decode bit for bit, so those shapes stay on the caller's per-row path
    static const int mode = [] { const char* v = getenv("STRATA_Q35_ATTN"); return v ? std::atoi(v) : 0; }();
    const int G = n_head / n_head_kv;
    if (mode != 1 && G % 8 == 0 && G <= 64) return false;
    const int n_chunks = (max_ids + kChunk - 1) / kChunk;
    // heads a block: more share a K / V row load, fewer give more blocks (a token's single row: 1 = attn_part's grid)
    if (hpw <= 0 || G % hpw != 0 || hpw == 5 || hpw > 6) hpw = rows_hpw(G, rows, n_chunks);
    const dim3 grid((unsigned) (n_head / hpw), (unsigned) n_chunks, (unsigned) rows);
    switch (hpw) {
#define Q35_IDS(n) case n: pdl_launch(attn_part_ids_kernel<n>, grid, kWarps * 32, 0, s, q, kc, vc, ids, ld_ids, steps, n_head, n_head_kv, scale, part, n_chunks); break;
        Q35_IDS(1) Q35_IDS(2) Q35_IDS(3) Q35_IDS(4) Q35_IDS(6)
#undef Q35_IDS
        default: return false;
    }
    pdl_launch(attn_merge_kernel, dim3((unsigned) n_head, (unsigned) rows), kD, 0, s, part, n_chunks, out);
    check("attn_decode_rows");
    return true;
}
__global__ void copy_f32_kernel(float* __restrict__ dst, const float* __restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
void copy_f32(float* dst, const float* src, int n, cudaStream_t s) {
    if (n <= 0) return;
    copy_f32_kernel<<<(n + 255) / 256, 256, 0, s>>>(dst, src, n);
    check("copy_f32");
}
__global__ void q4_step_set_kernel(int32_t* step, int32_t* pos, int s0, int s1, int s2, int s3, int p) {
    const int t = threadIdx.x;
    pos[t] = p;
    if (t == 0) {
        step[0] = s0;
        step[1] = s1;
        step[2] = s2;
        step[3] = s3;
    }
}
void q4_step_set(int32_t* step, int32_t* pos, int s0, int s1, int s2, int s3, int p, cudaStream_t s) {
    q4_step_set_kernel<<<1, 64, 0, s>>>(step, pos, s0, s1, s2, s3, p);
    check("q4_step_set");
}
void attn_gate(float* o, const float* q_full, int n_head, int head_dim, cudaStream_t s, int kv) {
    if (head_dim != kD) return;
    if (kv == 1 || kv == 3) {   // (a rotated V: the output back)
        attn_gate_q4_kernel<<<dim3((unsigned) n_head, 1), kD, 0, s>>>(o, q_full, n_head);
        check("attn_gate");
        return;
    }
    pdl_launch(attn_gate_kernel, n_head, kD, 0, s, o, q_full, n_head);
    check("attn_gate");
}
void scale_sigmoid(float* x, int n, const float* g, cudaStream_t s) {
    if (n <= 0) return;
    pdl_launch(scale_sigmoid_kernel, (n + 255) / 256, 256, 0, s, x, n, g);
    check("scale_sigmoid");
}
void scale_sigmoid_rows(float* x, int T, int n, const float* g, cudaStream_t s) {
    if (n <= 0 || T <= 0) return;
    scale_sigmoid_rows_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(x, n, g);
    check("scale_sigmoid_rows");
}

// ---- the prompt path
void add_rms_rows(float* R, int ldr, const float* add, const float* w, float eps, int T, int n, float* x, void* x16,
                  cudaStream_t s) {
    if (T <= 0) return;
    add_rms_rows_kernel<<<T, 256, 0, s>>>(R, ldr, add, w, eps, n, x, (uint16_t*) x16);
    check("add_rms_rows");
}
void conv_rows(const float* x, int T, int C, const float* kW, int d_conv, float* conv_state, float* out,
               cudaStream_t s) {
    if (T <= 0 || C <= 0) return;
    if (d_conv < 2 || d_conv > 9) {
        std::fprintf(stderr, "qwen35 conv_rows: d_conv %d\n", d_conv);
        return;
    }
    conv_rows_kernel<<<dim3((unsigned) ((C + 255) / 256), (unsigned) T), 256, 0, s>>>(x, C, kW, d_conv, conv_state, out);
    conv_state_rows_kernel<<<(C + 255) / 256, 256, 0, s>>>(x, T, C, d_conv, conv_state);
    check("conv_rows");
}
void gdn_prep_rows(float* h, int T, int C, int h_k, int S, float eps, cudaStream_t s) {
    if (T <= 0 || h_k <= 0) return;
    if (S <= 0 || S > 128) {
        std::fprintf(stderr, "qwen35 gdn_prep_rows: S %d\n", S);
        return;
    }
    gdn_prep_rows_kernel<<<dim3((unsigned) T, (unsigned) (2 * h_k)), 128, 0, s>>>(h, C, h_k, S, eps,
                                                                                   1.0f / std::sqrt((float) S));
    check("gdn_prep_rows");
}
void gdn_prep_gates(float* h, int T, int C, int h_k, int S, float eps, float* alpha, float* beta, const float* dt_bias,
                    const float* ssm_a, int h_v, cudaStream_t s) {
    if (T <= 0 || h_k <= 0) return;
    if (S <= 0 || S > 128) {
        std::fprintf(stderr, "qwen35 gdn_prep_gates: S %d\n", S);
        return;
    }
    const unsigned gb = (unsigned) ((h_v + 127) / 128);
    gdn_prep_gates_kernel<<<dim3((unsigned) T, (unsigned) (2 * h_k) + gb), 128, 0, s>>>(
        h, C, h_k, S, eps, 1.0f / std::sqrt((float) S), alpha, beta, dt_bias, ssm_a, h_v);
    check("gdn_prep_gates");
}
void conv_state_rows(const float* x, int T, int C, int d_conv, float* conv_state, cudaStream_t s) {
    if (T <= 0 || C <= 0 || d_conv < 2 || d_conv > 9) return;
    conv_state_rows_kernel<<<(C + 255) / 256, 256, 0, s>>>(x, T, C, d_conv, conv_state);
    check("conv_state_rows");
}
void conv_silu_step(float* conv_state, const float* x, const float* kW, int C, int d_conv, float* out, cudaStream_t s) {
    if (C <= 0 || d_conv < 2) return;
    pdl_launch(conv_silu_step_kernel, (C + 255) / 256, 256, 0, s, conv_state, x, kW, out, C, d_conv);
    check("conv_silu_step");
}
void gdn_gates_rows(float* alpha, float* beta, const float* dt_bias, const float* ssm_a, int T, int h_v, cudaStream_t s) {
    const int n = T * h_v;
    if (n <= 0) return;
    gdn_gates_rows_kernel<<<(n + 255) / 256, 256, 0, s>>>(alpha, beta, dt_bias, ssm_a, n, h_v);
    check("gdn_gates_rows");
}
void gdn_scan(float* state, const float* h, int T, int C, const float* gate, const float* beta, int S, int h_k, int h_v,
              float* o, cudaStream_t s) {
    if (T <= 0 || h_v <= 0) return;
    if (S != 128 || h_k <= 0 || h_v % h_k != 0) {
        std::fprintf(stderr, "qwen35 gdn_scan: S %d, heads %d / %d\n", S, h_k, h_v);
        return;
    }
    gdn_scan_kernel<128><<<h_v, 128, 0, s>>>(state, h, T, C, gate, beta, h_k, h_v, o);
    check("gdn_scan");
}
void gdn_core(float* state, const float* h, int T, int C, const float* gate, const float* beta, int S, int h_k, int h_v,
              bool prep, float eps, const float* dt_bias, const float* ssm_a, const float* z, const float* ssm_norm,
              float out_eps, float* y, void* yq, cudaStream_t s) {
    if (T <= 0 || h_v <= 0) return;
    if (S != 128 || h_k <= 0 || h_v % h_k != 0) {
        std::fprintf(stderr, "qwen35 gdn_core: S %d, heads %d / %d\n", S, h_k, h_v);
        return;
    }
    pdl_launch(gdn_core_kernel<128>, h_v, 128, 0, s, state, h, T, C, gate, beta, h_k, h_v, prep, eps, 1.0f / std::sqrt((float) S),
                                             dt_bias, ssm_a, z, ssm_norm, out_eps, y, (Q81*) yq);
    check("gdn_core");
}
static size_t gdn_mid_smem(int n_in, bool bf16) { return (size_t) 128 * 128 * 4 + (size_t) n_in * 4 + (size_t) 2 * n_in * (bf16 ? 2 : 4); }
bool gdn_mid_ok(int S, int h_k, int h_v, int d_conv, int n_in) {
#if defined(STRATA_USE_HIP)
    (void) S; (void) h_k; (void) h_v; (void) d_conv; (void) n_in;
    return false;
#else
    if (S != 128 || h_k <= 0 || h_v % h_k != 0 || h_v / h_k > 8 || d_conv != 4 || n_in % 8 != 0) return false;
    static int cc = -1, smem_max = 0;
    if (cc < 0) {
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        cc = major;
        smem_max = optin;
        if (cc >= 9 && cudaFuncSetAttribute(gdn_mid_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - 4096) != cudaSuccess) {
            (void) cudaGetLastError();
            cc = 0;
        }
    }
    return cc >= 9 && gdn_mid_smem(n_in, false) + 4096 <= (size_t) smem_max;
#endif
}
bool gdn_mid(float* state, float* conv_state, const float* qkv, const float* conv_w, int d_conv, const float* x,
             const void* w_alpha, const void* w_beta, bool ab_bf16, int n_in, int S, int h_k, int h_v, float eps,
             const float* dt_bias, const float* ssm_a, const float* z, const float* ssm_norm, float out_eps, float* h_out,
             float* ab_out, float* y, void* yq, const void* pf, size_t pf_bytes, cudaStream_t s) {
#if defined(STRATA_USE_HIP)
    (void) state; (void) conv_state; (void) qkv; (void) conv_w; (void) d_conv; (void) x; (void) w_alpha; (void) w_beta; (void) ab_bf16;
    (void) n_in; (void) S; (void) h_k; (void) h_v; (void) eps; (void) dt_bias; (void) ssm_a; (void) z; (void) ssm_norm;
    (void) out_eps; (void) h_out; (void) ab_out; (void) y; (void) yq; (void) pf; (void) pf_bytes; (void) s;
    return false;
#else
    if (!gdn_mid_ok(S, h_k, h_v, d_conv, n_in)) return false;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(h_v);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = gdn_mid_smem(n_in, ab_bf16);
    cfg.stream = s;
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = h_v / h_k;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    int na = 1;
    if (pdl_attr_ok(s)) {
        at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[1].val.programmaticStreamSerializationAllowed = 1;
        na = 2;
    }
    cfg.attrs = at;
    cfg.numAttrs = na;
    const cudaError_t e = cudaLaunchKernelEx(&cfg, gdn_mid_kernel<128>, state, conv_state, qkv, conv_w, d_conv, x,
                                             w_alpha, w_beta, ab_bf16, n_in, h_k, h_v, eps, 1.0f / std::sqrt((float) S), dt_bias,
                                             ssm_a, z, ssm_norm, out_eps, h_out, ab_out, y, (Q81*) yq,
                                             (const uint8_t*) pf, pf_bytes);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qwen35 gdn_mid: %s\n", cudaGetErrorString(e));
        return false;
    }
    return true;
#endif
}
void attn_prep_rows(const float* q_full, const float* k, const float* v, const float* q_norm, const float* k_norm,
                    float eps, int n_head, int n_head_kv, int head_dim, int n_rot, float rope_base, int64_t p0, int T,
                    uint16_t* kc, uint16_t* vc, float* q, cudaStream_t s, int kv) {
    if (T <= 0) return;
    if (head_dim != kD) {
        std::fprintf(stderr, "qwen35 attn_prep_rows: head_dim %d (the kernels take %d)\n", head_dim, kD);
        return;
    }
    if (kv != 0) {
        kv_switch(kv, [&](auto K) {
            attn_prep_q4_kernel<K()><<<dim3((unsigned) (n_head + n_head_kv), (unsigned) T), kD, 0, s>>>(
                q_full, k, v, q_norm, k_norm, eps, n_head, n_head_kv, n_rot, rope_base, p0, kc, vc, q);
        });
        check("attn_prep_rows");
        return;
    }
    attn_prep_rows_kernel<<<dim3((unsigned) (n_head + n_head_kv), (unsigned) T), kD, 0, s>>>(
        q_full, k, v, q_norm, k_norm, eps, n_head, n_head_kv, n_rot, rope_base, p0, kc, vc, q);
    check("attn_prep_rows");
}
void attn_prompt(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_head_kv, int head_dim,
                 int64_t p0, int T, float scale, float* out, cudaStream_t s, int kv) {
    if (T <= 0) return;
    const int G = n_head_kv > 0 ? n_head / n_head_kv : 0;
    if (head_dim != kD || G * n_head_kv != n_head || (G != 8 && G != 12 && G != 16 && G != 32)) {
        std::fprintf(stderr, "qwen35 attn_prompt: head_dim %d, heads %d / %d (a group of 8, 12, 16 or 32)\n", head_dim,
                     n_head, n_head_kv);
        return;
    }
    static const int pmode = [] { const char* v = getenv("STRATA_Q35_PATTN"); return v ? std::atoi(v) : 0; }();
    if (pmode != 1 && G == 16) {
        // the scaled q in FP16 for the tensor-core kernel (a scratch kept between calls; qwen35moe runs on one GPU)
        static wh16* q16 = nullptr;
        static size_t cap = 0;
        static bool smem_ok = false, smem_okq[4] = {false, false, false, false};
        const size_t n = (size_t) T * n_head * kD;
        if (n > cap) {
            if (q16 != nullptr) cudaFree(q16);
            q16 = nullptr;
            cap = 0;
            if (cudaMalloc(&q16, n * sizeof(wh16)) == cudaSuccess) cap = n;
            else cudaGetLastError();
        }
        if (kv == 0 && !smem_ok)
            smem_ok = cudaFuncSetAttribute(attn_prompt_tc_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           (int) kPromptTcSmem) == cudaSuccess;
        if (kv != 0 && !smem_okq[kv & 3])
            kv_switch(kv, [&](auto K) {
                smem_okq[K()] = cudaFuncSetAttribute(attn_prompt_tc_kernel<K()>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     (int) kPromptTcSmem) == cudaSuccess;
            });
        if (q16 != nullptr && (kv != 0 ? smem_okq[kv & 3] : smem_ok)) {
            q16_scale_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(q, n, scale, q16);
            const dim3 tg((unsigned) ((T + kPT - 1) / kPT), (unsigned) n_head_kv);
            if (kv != 0)
                kv_switch(kv, [&](auto K) {
                    attn_prompt_tc_kernel<K()><<<tg, 256, kPromptTcSmem, s>>>(q16, kc, vc, n_head, n_head_kv, p0, T, out);
                });
            else
            attn_prompt_tc_kernel<0><<<tg, 256, kPromptTcSmem, s>>>(
                q16, kc, vc, n_head, n_head_kv, p0, T, out);
            check("attn_prompt");
            return;
        }
        cudaGetLastError();
    }
    const dim3 grid((unsigned) ((T + 3) / 4), (unsigned) n_head_kv);
    if (kv != 0) {
        kv_switch(kv, [&](auto K) {
            if (G == 8) attn_prompt_kernel<1, K()><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, out);
            else if (G == 16 || G == 12) attn_prompt_kernel<2, K()><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, out);
            else attn_prompt_kernel<4, K()><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, out);
        });
    } else if (G == 8) attn_prompt_kernel<1, 0><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, out);
    else if (G == 16 || G == 12) attn_prompt_kernel<2, 0><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, out);
    else attn_prompt_kernel<4, 0><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, out);
    check("attn_prompt");
}
size_t attn_window_floats(int n_head, int head_dim, int max_ctx, int max_rows) {
    const size_t chunks = std::max<size_t>(((size_t) max_ctx + kWinChunk - 1) / kWinChunk + 1, kMinChunks);
    return (size_t) max_rows * n_head * chunks * ((size_t) head_dim + 2);
}
void attn_window(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_head_kv, int head_dim,
                 int64_t p0, int T, float scale, float* part, float* out, cudaStream_t s, int kv) {
    if (T <= 0) return;
    const int G = n_head_kv > 0 ? n_head / n_head_kv : 0;
    if (head_dim != kD || G * n_head_kv != n_head || (G != 8 && G != 16 && G != 32)) {
        std::fprintf(stderr, "qwen35 attn_window: head_dim %d, heads %d / %d (a group of 8, 16 or 32)\n", head_dim,
                     n_head, n_head_kv);
        return;
    }
    if (attn_split(q, kc, vc, n_head, n_head_kv, p0, T, scale, part, out, s, kv)) {
        check("attn_window");
        return;
    }
    const int n_chunks = (int) ((p0 + T + kWinChunk - 1) / kWinChunk);
    const dim3 grid((unsigned) ((T + 3) / 4), (unsigned) n_head_kv, (unsigned) n_chunks);
    if (kv != 0) {
        kv_switch(kv, [&](auto K) {
            if (G == 8) attn_window_kernel<1, K()><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, part, n_chunks);
            else if (G == 16) attn_window_kernel<2, K()><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, part, n_chunks);
            else attn_window_kernel<4, K()><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, part, n_chunks);
        });
    } else if (G == 8) attn_window_kernel<1, 0><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, part, n_chunks);
    else if (G == 16) attn_window_kernel<2, 0><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, part, n_chunks);
    else attn_window_kernel<4, 0><<<grid, 256, 0, s>>>(q, kc, vc, n_head, n_head_kv, p0, T, scale, part, n_chunks);
    pdl_launch(attn_merge_kernel, dim3((unsigned) n_head, (unsigned) T), kD, 0, s, part, n_chunks, out);
    check("attn_window");
}
void attn_gate_rows(float* o, const float* q_full, int T, int n_head, int head_dim, cudaStream_t s, int kv) {
    if (T <= 0 || head_dim != kD) return;
    if (kv == 1 || kv == 3) {   // (a rotated V: the output back)
        attn_gate_q4_kernel<<<dim3((unsigned) n_head, (unsigned) T), kD, 0, s>>>(o, q_full, n_head);
        check("attn_gate_rows");
        return;
    }
    attn_gate_rows_kernel<<<dim3((unsigned) n_head, (unsigned) T), kD, 0, s>>>(o, q_full, n_head);
    check("attn_gate_rows");
}
void attn_kv_rows(const float* k, const float* v, const float* k_norm, float eps, int n_head_kv, int head_dim, int n_rot,
                  float rope_base, int64_t p0, int T, uint16_t* kc, uint16_t* vc, cudaStream_t s, int kv) {
    if (T <= 0 || head_dim != kD) return;
    if (kv != 0) {
        kv_switch(kv, [&](auto K) {
            attn_prep_q4_kernel<K()><<<dim3((unsigned) n_head_kv, (unsigned) T), kD, 0, s>>>(
                k, k, v, k_norm, k_norm, eps, 0, n_head_kv, n_rot, rope_base, p0, kc, vc, nullptr);
        });
        check("attn_kv_rows");
        return;
    }
    // n_head 0: every block a KV head (the q blocks come first in attn_prep_rows_kernel's numbering)
    attn_prep_rows_kernel<<<dim3((unsigned) n_head_kv, (unsigned) T), kD, 0, s>>>(k, k, v, k_norm, k_norm, eps, 0,
                                                                                n_head_kv, n_rot, rope_base, p0, kc, vc,
                                                                                nullptr);
    check("attn_kv_rows");
}
void top_prob(const float* logits, int n, const int* tok, float* out, cudaStream_t s) {
    if (n <= 0) return;
    top_prob_kernel<<<1, 256, 0, s>>>(logits, n, tok, out);
    check("top_prob");
}
void shexp_gate_rows(const float* x, const float* w, float* ffn, int T, int n, cudaStream_t s) {
    if (T <= 0) return;
    shexp_gate_rows_kernel<<<T, 256, 0, s>>>(x, w, ffn, n);
    check("shexp_gate_rows");
}

void hc4_combine_norm(float* R, const float* out, const float* inject, int inj_stride, const float* gamma, float eps,
                      int n_embd, int hc, int T, float* xn, cudaStream_t s, void* xq) {
    if (T <= 0) return;
    if (xq != nullptr && n_embd % 256 != 0) {
        std::fprintf(stderr, "q35 hc4_combine_norm: the q8_1 output needs n_embd %% 256 == 0 (%d)\n", n_embd);
        return;
    }
    pdl_launch(hc4_combine_norm_kernel, dim3((unsigned) hc, (unsigned) T), 256, 0, s, R, out, inject, gamma, eps, n_embd, hc,
                                                                            inj_stride, xn, (Q81*) xq);
    check("hc4_combine_norm");
}
void gather_rows(const uint16_t* kc, const uint16_t* vc, const int32_t* ids, int n, int row, uint16_t* ks, uint16_t* vs,
                 cudaStream_t s) {
    if (n <= 0) return;
    gather_rows_kernel<<<n, 128, 0, s>>>(kc, vc, ids, row, ks, vs);
    check("gather_rows");
}
void silu_scale(float* x, int n, float a, cudaStream_t s, void* xq) {
    if (n <= 0) return;
    if (xq != nullptr && n % 32 != 0) {
        std::fprintf(stderr, "q35 silu_scale: the q8_1 output needs n %% 32 == 0 (%d)\n", n);
        return;
    }
    pdl_launch(silu_scale_kernel, (n + 255) / 256, 256, 0, s, x, n, a, (Q81*) xq);
    check("silu_scale");
}
void hc4_mix(const float* xn, const float* gate, int n_embd, int hc, int T, float* mixed, cudaStream_t s, void* xq) {
    if (T <= 0) return;
    if (xq != nullptr && n_embd % 32 != 0) {
        std::fprintf(stderr, "q35 hc4_mix: the q8_1 output needs n_embd %% 32 == 0 (%d)\n", n_embd);
        return;
    }
    pdl_launch(hc4_mix_kernel, dim3((unsigned) ((n_embd + 255) / 256), (unsigned) T), 256, 0, s, xn, gate, n_embd, hc, mixed,
                                                                                       (Q81*) xq);
    check("hc4_mix");
}

}  // namespace strata::kernels::q35
