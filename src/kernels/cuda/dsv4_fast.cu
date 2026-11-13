// src/kernels/cuda/dsv4_fast.cu - see include/strata/kernels/dsv4_fast.hpp.
//
// Transcribed from llama.cpp's deepseek4 graph (src/models/deepseek4.cpp, MIT) and ggml's CUDA rope (rope.cu: the
// NORM-mode rope_yarn, offset by ggml_rope_set_offset); docs/deepseek4/SPEC.md has the math these kernels follow.
#include "strata/kernels/dsv4_fast.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include "dsa_topk.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::kernels::dsv4 {
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
// block-wide sum / max; every thread gets the result.  sh: >= 32 floats.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += sh[i];
    __syncthreads();
    return t;
}
__device__ __forceinline__ float block_max(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = -INFINITY;
    for (int i = 0; i < nw; ++i) t = fmaxf(t, sh[i]);
    __syncthreads();
    return t;
}
// one q8_1 block from 32 values held one per lane (the glm fast path's q8_1_store_warp)
__device__ __forceinline__ void q8_1_store_warp(float v, block_q8_1* dst, int lane) {
    const float amax = warp_max(fabsf(v));
    const float sum = warp_sum(v);
    const float d = fminf(amax / 127.0f, 65504.0f);
    const int8_t q = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(v / d)));
    dst->qs[lane] = q;
    if (lane == 0) dst->ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}
__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t) v << 16); }

// ggml's rope_yarn (rope.cu), for the pair starting at iw (even, relative to the rotated span)
__device__ __forceinline__ void rope_cs(const Rope& r, int pos, int iw, bool forward, float& c, float& s) {
    const float theta_extrap = (float) pos * powf(r.theta_scale, iw / 2.0f);
    const float theta_interp = r.freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = r.attn_factor;
    if (r.ext_factor != 0.0f) {
        const float y = (float) (iw / 2 - r.corr_lo) / fmaxf(0.001f, r.corr_hi - r.corr_lo);
        const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
        const float ramp_mix = ramp * r.ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / r.freq_scale);
    }
    c = cosf(theta) * mscale;
    s = sinf(theta) * mscale;
    if (!forward) s = -s;
}
// rotates the last n_rot values of v[0, dim) in place (NORM mode: adjacent pairs).  Called by threads
// [0, n_rot / 2) of the block with t = threadIdx.x; v in shared memory.
__device__ __forceinline__ void rope_pair(float* v, int dim, int n_rot, const Rope& r, int pos, bool forward, int t) {
    const int iw = 2 * t;
    const int i0 = dim - n_rot + iw;
    float c, s;
    rope_cs(r, pos, iw, forward, c, s);
    const float x0 = v[i0], x1 = v[i0 + 1];
    v[i0] = x0 * c - x1 * s;
    v[i0 + 1] = x0 * s + x1 * c;
}

// ---------------------------------------------------------------- prep
// block 0: qr; block 1: kv -> raw ring; blocks 2.. : the compressor rings
__global__ void __launch_bounds__(512) prep_kernel(const PrepArgs a) {
    __shared__ float sh[32];
    __shared__ float v[1024];
    const int tid = threadIdx.x;
    if (blockIdx.x == 0) {
        float ss = 0.0f;
        for (int i = tid; i < a.q_lora; i += blockDim.x) ss += a.qr_raw[i] * a.qr_raw[i];
        const float inv = rsqrtf(block_sum(ss, sh) / (float) a.q_lora + a.eps);
        for (int i = tid; i < a.q_lora; i += blockDim.x) {   // blockDim % 32 == 0: each warp a whole q8_1 block
            const float y = a.qr_raw[i] * inv * a.q_a_norm[i];
            a.qr[i] = y;
            q8_1_store_warp(y, (block_q8_1*) a.qr_q + i / 32, tid & 31);
        }
        return;
    }
    if (blockIdx.x == 1) {
        const int D = a.head_dim;
        float ss = 0.0f;
        for (int i = tid; i < D; i += blockDim.x) ss += a.kv_raw[i] * a.kv_raw[i];
        const float inv = rsqrtf(block_sum(ss, sh) / (float) D + a.eps);
        for (int i = tid; i < D; i += blockDim.x) v[i] = a.kv_raw[i] * inv * a.kv_norm[i];
        __syncthreads();
        if (tid < a.n_rot / 2) rope_pair(v, D, a.n_rot, a.rope, a.p, true, tid);
        __syncthreads();
        uint16_t* row = a.raw + (size_t) (a.p % a.ring) * D;
        for (int i = tid; i < D; i += blockDim.x) row[i] = __half_as_ushort(__float2half(v[i]));
        return;
    }
    const int c = blockIdx.x - 2;
    if (c >= a.n_comp) return;
    const Comp& C = a.comp[c];
    const int W = (C.overlap ? 2 : 1) * C.D;
    const int state = (C.overlap ? 2 : 1) * C.ratio;
    const int row = a.p % ring_rows(state);
    const float* ape = C.ape + (size_t) (a.p % C.ratio) * W;
    for (int i = tid; i < W; i += blockDim.x) {
        C.ring_kv[(size_t) row * W + i] = a.comp_kv[c][i];
        C.ring_sc[(size_t) row * W + i] = a.comp_sc[c][i] + ape[i];
    }
}

// one block per finishing compressor: the per-channel softmax-weighted sum of the block's rows, rms * norm, RoPE at
// the block's first position, FP16 into the cache row b
__global__ void __launch_bounds__(512) compress_kernel(const PrepArgs a) {
    __shared__ float sh[32];
    __shared__ float v[512];
    const Comp& C = a.comp[blockIdx.x];
    const int D = C.D, R = C.ratio, tid = threadIdx.x;
    const int W = (C.overlap ? 2 : 1) * D;
    const int state = (C.overlap ? 2 : 1) * R;
    const int s0 = a.p + 1 - R;
    const int b = a.p / R;
    float ss = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        // the rows: the overlap's previous window from the FIRST half of the projection, then this window from the
        // SECOND half; one window of the whole projection otherwise.  Positions < 0 are kv 0, score -inf.
        float m = -INFINITY;
        const int n = C.overlap ? 2 * R : R;
        for (int t = 0; t < n; ++t) {
            const int pos = C.overlap ? s0 - R + t : s0 + t;
            if (pos < 0) continue;
            const int half = C.overlap && t >= R ? D : 0;
            m = fmaxf(m, C.ring_sc[(size_t) (pos % ring_rows(state)) * W + half + d]);
        }
        float den = 0.0f, num = 0.0f;
        for (int t = 0; t < n; ++t) {
            const int pos = C.overlap ? s0 - R + t : s0 + t;
            if (pos < 0) continue;
            const int half = C.overlap && t >= R ? D : 0;
            const size_t at = (size_t) (pos % ring_rows(state)) * W + half + d;
            const float e = expf(C.ring_sc[at] - m);
            den += e;
            num += e * C.ring_kv[at];
        }
        const float y = num / den;
        v[d] = y;
        ss += y * y;
    }
    const float inv = rsqrtf(block_sum(ss, sh) / (float) D + a.eps);
    for (int d = tid; d < D; d += blockDim.x) v[d] = v[d] * inv * C.norm[d];
    __syncthreads();
    if (tid < a.n_rot / 2) rope_pair(v, D, a.n_rot, a.comp_rope, s0, true, tid);
    __syncthreads();
    uint16_t* row = C.cache + (size_t) b * D;
    for (int d = tid; d < D; d += blockDim.x) row[d] = __half_as_ushort(__float2half(v[d]));
}

// ---------------------------------------------------------------- q / iq
__global__ void __launch_bounds__(128) q_prep_kernel(float* q, int n_head, int head_dim, int n_rot, Rope rope, float* iq,
                                                     int idx_dim, Rope comp_rope, int p, float eps) {
    __shared__ float sh[32];
    __shared__ float v[512];
    const int tid = threadIdx.x;
    if ((int) blockIdx.x < n_head) {
        float* x = q + (size_t) blockIdx.x * head_dim;
        float ss = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            v[i] = x[i];
            ss += v[i] * v[i];
        }
        const float inv = rsqrtf(block_sum(ss, sh) / (float) head_dim + eps);
        for (int i = tid; i < head_dim; i += blockDim.x) v[i] *= inv;
        __syncthreads();
        if (tid < n_rot / 2) rope_pair(v, head_dim, n_rot, rope, p, true, tid);
        __syncthreads();
        for (int i = tid; i < head_dim; i += blockDim.x) x[i] = v[i];
        return;
    }
    float* x = iq + (size_t) (blockIdx.x - n_head) * idx_dim;
    for (int i = tid; i < idx_dim; i += blockDim.x) v[i] = x[i];
    __syncthreads();
    if (tid < n_rot / 2) rope_pair(v, idx_dim, n_rot, comp_rope, p, true, tid);
    __syncthreads();
    for (int i = tid; i < idx_dim; i += blockDim.x) x[i] = v[i];
}

// ---------------------------------------------------------------- the lightning indexer
constexpr int IDX_WARPS = 8;
// one warp per compressed row; iq (idx_heads x idx_dim, idx_dim == 128) and w in shared memory
__global__ void __launch_bounds__(IDX_WARPS * 32) idx_score_kernel(const float* __restrict__ iq,
                                                                   const uint16_t* __restrict__ K,
                                                                   const float* __restrict__ w, int idx_heads,
                                                                   int n_vis, float* __restrict__ score) {
    extern __shared__ float s_iq[];   // idx_heads * 128, then idx_heads weights
    float* s_w = s_iq + idx_heads * 128;
    for (int i = threadIdx.x; i < idx_heads * 128; i += blockDim.x) s_iq[i] = iq[i];
    for (int i = threadIdx.x; i < idx_heads; i += blockDim.x) s_w[i] = w[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int j = blockIdx.x * IDX_WARPS + (threadIdx.x >> 5);
    if (j >= n_vis) return;
    const uint16_t* kr = K + (size_t) j * 128 + lane * 4;
    const float k0 = __half2float(__ushort_as_half(kr[0])), k1 = __half2float(__ushort_as_half(kr[1]));
    const float k2 = __half2float(__ushort_as_half(kr[2])), k3 = __half2float(__ushort_as_half(kr[3]));
    float acc = 0.0f;
    for (int h = 0; h < idx_heads; ++h) {
        const float* qh = s_iq + h * 128 + lane * 4;
        const float d = warp_sum(qh[0] * k0 + qh[1] * k1 + qh[2] * k2 + qh[3] * k3);
        acc += fmaxf(d, 0.0f) * s_w[h];
    }
    if (lane == 0) score[j] = acc;
}
// The same scores on tensor cores (sm_80+, 64 heads): K[n x 128] . Q^T as mma.sync m16n8k16, a warp 16 rows x the 64
// heads. Q (F32) goes in as two FP16 halves, hi + lo, so the products keep ~22 bits of q (the K rows are FP16 already);
// a score differs from the warp kernel's in its summation order only (~1e-6 relative). 84 -> 12 us at 31k rows (125k
// tokens), 13 -> 7 at 2k, on an RTX 5090. STRATA_DS4_IDX=1: the warp kernel above (A/B).
#if defined(__HIPCC__) || defined(STRATA_USE_HIP)   // AMD: no mma.sync; the host keeps the warp kernel
#define STRATA_DS4_IDX_SM80 0
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define STRATA_DS4_IDX_SM80 1
#else
#define STRATA_DS4_IDX_SM80 0
#endif
constexpr int IDXM_WARPS = 8;
__device__ __forceinline__ void idx_mma16816(float* d, const unsigned* a, unsigned b0, unsigned b1) {
#if !STRATA_DS4_IDX_SM80
    __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ unsigned idx_h2(__half x, __half y) {   // x in the low half
    return (unsigned) __half_as_ushort(x) | ((unsigned) __half_as_ushort(y) << 16);
}
__global__ void __launch_bounds__(IDXM_WARPS * 32) idx_score_mma_kernel(const float* __restrict__ iq,
                                                                        const uint16_t* __restrict__ K,
                                                                        const float* __restrict__ w, int n_vis,
                                                                        float* __restrict__ score) {
    // Q's B fragments in fragment order, [n tile 8][k step 8][lane 32] x {hi k, k+1 | hi k+8, k+9 | lo .. | lo ..}
    __shared__ uint4 s_b[8 * 8 * 32];
    __shared__ float s_w[64];
    for (int i = threadIdx.x; i < 8 * 8 * 32; i += blockDim.x) {
        const int ln = i & 31, ks = (i >> 5) & 7, nt = i >> 8;
        const float* qn = iq + (nt * 8 + (ln >> 2)) * 128 + ks * 16 + (ln & 3) * 2;
        const float v[4] = {qn[0], qn[1], qn[8], qn[9]};
        __half h[4], l[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            h[t] = __float2half_rn(v[t]);
            l[t] = __float2half_rn(v[t] - __half2float(h[t]));
        }
        s_b[i] = make_uint4(idx_h2(h[0], h[1]), idx_h2(h[2], h[3]), idx_h2(l[0], l[1]), idx_h2(l[2], l[3]));
    }
    if (threadIdx.x < 64) s_w[threadIdx.x] = w[threadIdx.x];
    __syncthreads();
    const int lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int row0 = (blockIdx.x * IDXM_WARPS + (threadIdx.x >> 5)) * 16;
    if (row0 >= n_vis) return;
    const int ra = row0 + g, rb = ra + 8;   // the A fragment's rows (and the accumulator's)
    const unsigned* Ka = reinterpret_cast<const unsigned*>(K + (size_t) ra * 128);
    const unsigned* Kb = reinterpret_cast<const unsigned*>(K + (size_t) rb * 128);
    unsigned a[8][4];
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const int c = ks * 8 + t4;   // in pairs of halves
        a[ks][0] = ra < n_vis ? Ka[c] : 0u;
        a[ks][1] = rb < n_vis ? Kb[c] : 0u;
        a[ks][2] = ra < n_vis ? Ka[c + 4] : 0u;
        a[ks][3] = rb < n_vis ? Kb[c + 4] : 0u;
    }
    float sa = 0.0f, sb = 0.0f;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const uint4 b = s_b[(nt * 8 + ks) * 32 + lane];
            idx_mma16816(d, a[ks], b.x, b.y);
            idx_mma16816(d, a[ks], b.z, b.w);
        }
        const float w0 = s_w[nt * 8 + t4 * 2], w1 = s_w[nt * 8 + t4 * 2 + 1];
        sa += fmaxf(d[0], 0.0f) * w0 + fmaxf(d[1], 0.0f) * w1;
        sb += fmaxf(d[2], 0.0f) * w0 + fmaxf(d[3], 0.0f) * w1;
    }
    sa += __shfl_xor_sync(0xffffffffu, sa, 1);
    sa += __shfl_xor_sync(0xffffffffu, sa, 2);
    sb += __shfl_xor_sync(0xffffffffu, sb, 1);
    sb += __shfl_xor_sync(0xffffffffu, sb, 2);
    if (t4 == 0) {
        if (ra < n_vis) score[ra] = sa;
        if (rb < n_vis) score[rb] = sb;
    }
}
#if defined(STRATA_USE_HIP)
// The same scores on RDNA's WMMA (gfx11 / gfx12, wave32; HIP builds): a wave = 16 K rows x the 64 heads, K (FP16) x
// Q^T with Q as FP16 hi + lo (two WMMAs, F32 accumulate), as idx_score_mma_kernel. Radeon 8060S (gfx1151): 1121 ->
// 174 us at 31k rows, 95 -> 23 at 2k, the scores within 6e-6 of the warp kernel's. STRATA_DS4_IDX=1: the warp kernel.
using iw_h16 = _Float16 __attribute__((ext_vector_type(16)));
using iw_f8 = float __attribute__((ext_vector_type(8)));
__device__ __forceinline__ iw_f8 iw_wmma(iw_h16 a, iw_h16 b, iw_f8 c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#elif defined(__gfx1200__) || defined(__gfx1201__)
    using h8 = _Float16 __attribute__((ext_vector_type(8)));
    const int k0 = (threadIdx.x & 31) / 16 * 8;
    h8 aa, bb;
#pragma unroll
    for (int i = 0; i < 8; ++i) { aa[i] = a[k0 + i]; bb[i] = b[k0 + i]; }
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(aa, bb, c);
#else
    __builtin_trap();
    return c;
#endif
}
__device__ __forceinline__ int iw_row(int i, int hi) {   // the accumulator's element i of half `hi`: its row
#if defined(__gfx1200__) || defined(__gfx1201__)
    return 8 * hi + i;
#else
    return 2 * i + hi;
#endif
}
constexpr int IW_WAVES = 8;
__global__ void __launch_bounds__(IW_WAVES * 32) idx_score_wmma_kernel(const float* __restrict__ iq,
                                                                       const uint16_t* __restrict__ K,
                                                                       const float* __restrict__ w, int n_vis,
                                                                       float* __restrict__ score) {
    __shared__ __align__(32) _Float16 s_qh[64][128], s_ql[64][128];
    __shared__ float s_w[64];
    for (int i = threadIdx.x; i < 64 * 128; i += blockDim.x) {
        const float v = iq[i];
        const _Float16 h = (_Float16) v;
        s_qh[i / 128][i % 128] = h;
        s_ql[i / 128][i % 128] = (_Float16) (v - (float) h);
    }
    if (threadIdx.x < 64) s_w[threadIdx.x] = w[threadIdx.x];
    __syncthreads();
    const int lane = threadIdx.x & 31, l16 = lane & 15, hi = lane >> 4;
    const int row0 = (blockIdx.x * IW_WAVES + (threadIdx.x >> 5)) * 16;
    if (row0 >= n_vis) return;
    const int ra = min(row0 + l16, n_vis - 1);   // (rows past the end: a valid row's, never written)
    const uint4* kr = reinterpret_cast<const uint4*>(K + (size_t) ra * 128);
    float s[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = 0.0f;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        iw_f8 d = {};
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            union { uint4 u[2]; iw_h16 v; } a;
            a.u[0] = kr[ks * 2];
            a.u[1] = kr[ks * 2 + 1];
            const iw_h16 bh = *reinterpret_cast<const iw_h16*>(&s_qh[nt * 16 + l16][ks * 16]);
            const iw_h16 bl = *reinterpret_cast<const iw_h16*>(&s_ql[nt * 16 + l16][ks * 16]);
            d = iw_wmma(a.v, bh, d);
            d = iw_wmma(a.v, bl, d);
        }
        const float wh = s_w[nt * 16 + l16];
#pragma unroll
        for (int i = 0; i < 8; ++i) s[i] += fmaxf(d[i], 0.0f) * wh;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {   // the 16 heads of each half-wave
        s[i] += __shfl_xor(s[i], 1, 32);
        s[i] += __shfl_xor(s[i], 2, 32);
        s[i] += __shfl_xor(s[i], 4, 32);
        s[i] += __shfl_xor(s[i], 8, 32);
    }
    if (l16 == 0)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = row0 + iw_row(i, hi);
            if (r < n_vis) score[r] = s[i];
        }
}
#endif
__global__ void __launch_bounds__(1024) idx_select_kernel(const float* __restrict__ score, int n_vis, int top,
                                                          int* __restrict__ sel) {
    strata::kernels::dsa::select_top_pools(score, n_vis, top, 1, sel);
}

// ---------------------------------------------------------------- attention
constexpr int ATT_THREADS = 256;
// one block per head; logits of every key in shared memory (dynamic: n_raw + n_comp floats)
__global__ void __launch_bounds__(ATT_THREADS) attend_kernel(const float* __restrict__ q, const uint16_t* __restrict__ raw,
                                                             int ring, int n_raw, const uint16_t* __restrict__ comp,
                                                             const int* __restrict__ comp_sel, int n_comp,
                                                             const float* __restrict__ sinks, int n_rot, Rope rope,
                                                             int p, float scale, float* __restrict__ out) {
    constexpr int D = 512;
    extern __shared__ float s_l[];
    __shared__ float s_q[D];
    __shared__ float s_o[D];
    __shared__ float sh[32];
    const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n = n_raw + n_comp;
    for (int i = tid; i < D; i += ATT_THREADS) s_q[i] = q[(size_t) h * D + i];
    __syncthreads();
    const auto key = [&](int j) -> const uint16_t* {
        if (j < n_raw) {
            const int pos = p - n_raw + 1 + j;
            return raw + (size_t) (pos % ring) * D;
        }
        const int r = comp_sel != nullptr ? comp_sel[j - n_raw] : j - n_raw;
        return comp + (size_t) r * D;
    };
    // 1. the logits, one warp per key: lane l takes the half2 pairs l, l + 32, ... (coalesced)
    for (int j = warp; j < n; j += ATT_THREADS / 32) {
        const __half2* k2 = (const __half2*) key(j);
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < D / 64; ++i) {
            const float2 kv = __half22float2(k2[lane + 32 * i]);
            const int e = 2 * (lane + 32 * i);
            acc += s_q[e] * kv.x + s_q[e + 1] * kv.y;
        }
        acc = warp_sum(acc);
        if (lane == 0) s_l[j] = acc * scale;
    }
    __syncthreads();
    // 2. the softmax with the sink (it joins the max and the denominator, no value)
    const float sink = sinks[h];
    float m = sink;
    for (int j = tid; j < n; j += ATT_THREADS) m = fmaxf(m, s_l[j]);
    m = block_max(m, sh);
    float den = 0.0f;
    for (int j = tid; j < n; j += ATT_THREADS) {
        const float e = expf(s_l[j] - m);
        s_l[j] = e;
        den += e;
    }
    den = block_sum(den, sh) + expf(sink - m);
    const float rden = 1.0f / den;
    // 3. o = sum_j w_j k_j: thread t owns the pair (2t, 2t+1)
    float o0 = 0.0f, o1 = 0.0f;
    for (int j = 0; j < n; ++j) {
        const float2 kv = __half22float2(((const __half2*) key(j))[tid]);
        const float wj = s_l[j];
        o0 += wj * kv.x;
        o1 += wj * kv.y;
    }
    s_o[2 * tid] = o0 * rden;
    s_o[2 * tid + 1] = o1 * rden;
    __syncthreads();
    // 4. de-rope (V = K carries the rotation): the inverse rotation at the query's position
    if (tid < n_rot / 2) rope_pair(s_o, D, n_rot, rope, p, false, tid);
    __syncthreads();
    for (int i = tid; i < D; i += ATT_THREADS) out[(size_t) h * D + i] = s_o[i];
}

// the split attention (flash-decoding): every head reads the same keys (K = V, one 512-wide row a position), so a
// block takes AS_H heads and a chunk of AS_KEYS keys - each key tile is read from L2 once for 8 heads, and
// n_head / 8 x ceil(n / 64) blocks fill the GPU where one block a head left most SMs idle.  Each block leaves its
// chunk's max, denominator and unnormalized sum; attend_combine_kernel merges them with the sink.
constexpr int AS_H = 8, AS_TILE = 16;
__global__ void __launch_bounds__(256) attend_split_kernel(const float* __restrict__ q, const uint16_t* __restrict__ raw,
                                                           int ring, int n_raw, const uint16_t* __restrict__ comp,
                                                           const int* __restrict__ comp_sel, int n_comp, int p,
                                                           float scale, int ck, int S, float* __restrict__ part) {
    constexpr int D = 512;
    __shared__ float s_q[AS_H][D];
    __shared__ __align__(16) __half2 s_k[AS_TILE][D / 2];
    __shared__ float s_w[AS_H][AS_TILE];
    __shared__ float s_m[AS_H], s_a[AS_H], s_den[AS_H];
    const int h0 = blockIdx.x * AS_H, c = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n = n_raw + n_comp, j0 = c * ck, j1 = min(n, j0 + ck);
    for (int i = tid; i < AS_H * D; i += 256) s_q[i / D][i % D] = q[(size_t) h0 * D + i];
    if (tid < AS_H) { s_m[tid] = -INFINITY; s_den[tid] = 0.0f; }
    const auto key = [&](int j) -> const uint16_t* {
        if (j < n_raw) return raw + (size_t) ((p - n_raw + 1 + j) % ring) * D;
        const int r = comp_sel != nullptr ? comp_sel[j - n_raw] : j - n_raw;
        return comp + (size_t) r * D;
    };
    float o[AS_H][2];
#pragma unroll
    for (int h = 0; h < AS_H; ++h) o[h][0] = o[h][1] = 0.0f;
    for (int t0 = j0; t0 < j1; t0 += AS_TILE) {
        const int nt = min(AS_TILE, j1 - t0);
        __syncthreads();   // the previous tile's s_k / s_w are done with
        for (int i = tid; i < nt * (D / 8); i += 256) {   // 16-byte loads, 64 a key
            const int kk = i / (D / 8), cc = i % (D / 8);
            ((uint4*) s_k[kk])[cc] = ((const uint4*) key(t0 + kk))[cc];
        }
        __syncthreads();
        // logits: warp w is head w, lanes across the row
        {
            const int h = warp;
            float mx = -INFINITY;
            float l[AS_TILE];
#pragma unroll
            for (int kk = 0; kk < AS_TILE; ++kk) {
                float acc = 0.0f;
                if (kk < nt) {
#pragma unroll
                    for (int i = 0; i < D / 64; ++i) {
                        const float2 kv = __half22float2(s_k[kk][lane + 32 * i]);
                        const int e = 2 * (lane + 32 * i);
                        acc += s_q[h][e] * kv.x + s_q[h][e + 1] * kv.y;
                    }
                    acc = warp_sum(acc) * scale;
                    mx = fmaxf(mx, acc);
                }
                l[kk] = acc;
            }
            const float m_old = s_m[h], m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int kk = 0; kk < AS_TILE; ++kk) {
                const float e = kk < nt ? expf(l[kk] - m_new) : 0.0f;
                sum += e;
                if (lane == kk) s_w[h][kk] = e;
            }
            if (lane == 0) {
                const float a = expf(m_old - m_new);   // m_old = -inf on the first tile: 0
                s_a[h] = a;
                s_den[h] = s_den[h] * a + sum;
                s_m[h] = m_new;
            }
        }
        __syncthreads();
        // values: thread t owns the pair (2t, 2t+1) of every head
#pragma unroll
        for (int h = 0; h < AS_H; ++h) { o[h][0] *= s_a[h]; o[h][1] *= s_a[h]; }
        for (int kk = 0; kk < nt; ++kk) {
            const float2 kv = __half22float2(s_k[kk][tid]);
#pragma unroll
            for (int h = 0; h < AS_H; ++h) {
                const float w = s_w[h][kk];
                o[h][0] += w * kv.x;
                o[h][1] += w * kv.y;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < AS_H; ++h) {
        float* P = part + ((size_t) (h0 + h) * S + c) * (D + 2);
        if (tid == 0) { P[0] = s_m[h]; P[1] = s_den[h]; }
        P[2 + 2 * tid] = o[h][0];
        P[2 + 2 * tid + 1] = o[h][1];
    }
}

// one value a lane -> one q8_1 block a warp (glm_fast's q8_1_store_warp, the same bits)
__device__ __forceinline__ void q8_1_store_warp_ds(float v, block_q8_1* dst, int lane) {
    const float amax = warp_max(fabsf(v));
    const float sum = warp_sum(v);
    const float d = fminf(amax / 127.0f, 65504.0f);
    const int8_t q = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(v / d)));
    dst->qs[lane] = q;
    if (lane == 0) dst->ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}
__global__ void oq_kernel(const float* __restrict__ x, block_q8_1* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) q8_1_store_warp_ds(x[i], y + i / 32, threadIdx.x & 31);
}

__global__ void __launch_bounds__(256) attend_combine_kernel(const float* __restrict__ part, int S,
                                                             const float* __restrict__ sinks, int n_rot, Rope rope,
                                                             int p, float* __restrict__ out,
                                                             block_q8_1* __restrict__ oq) {
    constexpr int D = 512;
    __shared__ float s_o[D];
    const int h = blockIdx.x, tid = threadIdx.x;
    const float* P = part + (size_t) h * S * (D + 2);
    const float sink = sinks[h];
    float M = sink;
    for (int c = 0; c < S; ++c) M = fmaxf(M, P[(size_t) c * (D + 2)]);
    float den = expf(sink - M), o0 = 0.0f, o1 = 0.0f;
    for (int c = 0; c < S; ++c) {
        const float* Pc = P + (size_t) c * (D + 2);
        const float a = expf(Pc[0] - M);
        den += Pc[1] * a;
        o0 += Pc[2 + 2 * tid] * a;
        o1 += Pc[2 + 2 * tid + 1] * a;
    }
    const float rden = 1.0f / den;
    s_o[2 * tid] = o0 * rden;
    s_o[2 * tid + 1] = o1 * rden;
    __syncthreads();
    if (tid < n_rot / 2) rope_pair(s_o, D, n_rot, rope, p, false, tid);
    __syncthreads();
    for (int i = tid; i < D; i += 256) out[(size_t) h * D + i] = s_o[i];
    if (oq != nullptr) {   // the head's 16 q8_1 blocks, two a warp (the separate quantize launch's bits)
        const int w = tid >> 5, lane = tid & 31;
#pragma unroll
        for (int b = 0; b < 2; ++b)
            q8_1_store_warp_ds(s_o[(2 * w + b) * 32 + lane], oq + (size_t) h * (D / 32) + 2 * w + b, lane);
    }
}

// ---------------------------------------------------------------- the hyper-connection head
__global__ void __launch_bounds__(1024) head_prep_kernel(const float* __restrict__ R, const uint16_t* __restrict__ fn,
                                                         const float* __restrict__ scale, const float* __restrict__ base,
                                                         float hc_eps, const float* __restrict__ norm_w, float eps,
                                                         int n_embd, float* __restrict__ x, block_q8_1* __restrict__ xq) {
    __shared__ float sh[32];
    __shared__ float s_pre[4];
    const int tid = threadIdx.x, F = 4 * n_embd;
    float ss = 0.0f, m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
    for (int i = tid; i < F; i += blockDim.x) {
        const float r = R[i];
        ss += r * r;
        m0 += bf(fn[i]) * r;
        m1 += bf(fn[(size_t) F + i]) * r;
        m2 += bf(fn[(size_t) 2 * F + i]) * r;
        m3 += bf(fn[(size_t) 3 * F + i]) * r;
    }
    const float inv = rsqrtf(block_sum(ss, sh) / (float) F + eps);
    m0 = block_sum(m0, sh);
    m1 = block_sum(m1, sh);
    m2 = block_sum(m2, sh);
    m3 = block_sum(m3, sh);
    if (tid == 0) {
        const float mm[4] = {m0, m1, m2, m3};
        for (int s = 0; s < 4; ++s) s_pre[s] = 1.0f / (1.0f + expf(-(mm[s] * inv * scale[0] + base[s]))) + hc_eps;
    }
    __syncthreads();
    float hs = 0.0f;
    for (int e = tid; e < n_embd; e += blockDim.x) {
        const float h = s_pre[0] * R[e] + s_pre[1] * R[n_embd + e] + s_pre[2] * R[2 * n_embd + e] +
                        s_pre[3] * R[3 * n_embd + e];
        x[e] = h;
        hs += h * h;
    }
    const float inv2 = rsqrtf(block_sum(hs, sh) / (float) n_embd + eps);
    for (int e = tid; e < n_embd; e += blockDim.x) {
        const float y = x[e] * inv2 * norm_w[e];
        x[e] = y;
        q8_1_store_warp(y, xq + e / 32, tid & 31);
    }
}


// ================================================================ the prompt path
__global__ void win_load_kernel(const uint16_t* __restrict__ ring, int n_ring, uint16_t* __restrict__ keys, int n_hist,
                                int p0, int D) {
    const int i = blockIdx.x;
    const int pos = p0 - n_hist + i;
    const uint16_t* src = ring + (size_t) (pos % n_ring) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) keys[(size_t) i * D + d] = src[d];
}
__global__ void win_store_kernel(uint16_t* __restrict__ ring, int n_ring, const uint16_t* __restrict__ keys, int n_total,
                                 int n_keep, int pos0, int D) {
    const int r = n_total - n_keep + blockIdx.x;
    const int pos = pos0 + r;
    uint16_t* dst = ring + (size_t) (pos % n_ring) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) dst[d] = keys[(size_t) r * D + d];
}

__global__ void __launch_bounds__(512) prep_batch_kernel(const BatchPrepArgs a) {
    __shared__ float sh[32];
    __shared__ float v[512];
    const int t = blockIdx.x, tid = threadIdx.x;
    if (blockIdx.y == 0) {
        const float* in = a.qr_raw + (size_t) t * a.q_lora;
        float ss = 0.0f;
        for (int i = tid; i < a.q_lora; i += blockDim.x) ss += in[i] * in[i];
        const float inv = rsqrtf(block_sum(ss, sh) / (float) a.q_lora + a.eps);
        for (int i = tid; i < a.q_lora; i += blockDim.x) {
            const float y = in[i] * inv * a.q_a_norm[i];
            a.qr[(size_t) t * a.q_lora + i] = y;
            a.qr16[(size_t) t * a.q_lora + i] = __half_as_ushort(__float2half(y));
        }
        return;
    }
    const int D = a.head_dim;
    const float* in = a.kv_raw + (size_t) t * D;
    float ss = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) ss += in[i] * in[i];
    const float inv = rsqrtf(block_sum(ss, sh) / (float) D + a.eps);
    for (int i = tid; i < D; i += blockDim.x) v[i] = in[i] * inv * a.kv_norm[i];
    __syncthreads();
    if (tid < a.n_rot / 2) rope_pair(v, D, a.n_rot, a.rope, a.p0 + t, true, tid);
    __syncthreads();
    uint16_t* row = a.keys + (size_t) (a.n_hist + t) * D;
    for (int i = tid; i < D; i += blockDim.x) row[i] = __half_as_ushort(__float2half(v[i]));
}

struct BatchCompArgs {
    BatchComp c[2];
    int n_comp, p0, T, n_rot;
    Rope rope;
    float eps;
};
// one block per (block ending in the batch, compressor)
__global__ void __launch_bounds__(512) compress_batch_kernel(const BatchCompArgs a) {
    __shared__ float sh[32];
    __shared__ float v[512];
    const BatchComp& B = a.c[blockIdx.y];
    const Comp& C = B.c;
    const int D = C.D, R = C.ratio, tid = threadIdx.x;
    const int W = (C.overlap ? 2 : 1) * D;
    const int state = (C.overlap ? 2 : 1) * R;
    const int b_lo = (a.p0 + R) / R - 1, b_hi = (a.p0 + a.T) / R - 1;
    const int b = b_lo + (int) blockIdx.x;
    if (b > b_hi || b < 0) return;
    const int s0 = b * R;
    const int n = C.overlap ? 2 * R : R;
    // a row's kv and score for channel col: this batch's projections (with the ape) or the ring (positions < p0)
    const auto row = [&](int pos, int col, float& kv, float& sc) {
        if (pos >= a.p0) {
            const size_t at = (size_t) (pos - a.p0) * W + col;
            kv = B.kv[at];
            sc = B.sc[at] + C.ape[(size_t) (pos % R) * W + col];
        } else {
            const size_t at = (size_t) (pos % ring_rows(state)) * W + col;
            kv = C.ring_kv[at];
            sc = C.ring_sc[at];
        }
    };
    float ss = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float m = -INFINITY;
        for (int t = 0; t < n; ++t) {
            const int pos = C.overlap ? s0 - R + t : s0 + t;
            if (pos < 0) continue;
            float kv, sc;
            row(pos, (C.overlap && t >= R ? D : 0) + d, kv, sc);
            m = fmaxf(m, sc);
        }
        float den = 0.0f, num = 0.0f;
        for (int t = 0; t < n; ++t) {
            const int pos = C.overlap ? s0 - R + t : s0 + t;
            if (pos < 0) continue;
            float kv, sc;
            row(pos, (C.overlap && t >= R ? D : 0) + d, kv, sc);
            const float e = expf(sc - m);
            den += e;
            num += e * kv;
        }
        const float y = num / den;
        v[d] = y;
        ss += y * y;
    }
    const float inv = rsqrtf(block_sum(ss, sh) / (float) D + a.eps);
    for (int d = tid; d < D; d += blockDim.x) v[d] = v[d] * inv * C.norm[d];
    __syncthreads();
    if (tid < a.n_rot / 2) rope_pair(v, D, a.n_rot, a.rope, s0, true, tid);
    __syncthreads();
    uint16_t* out = C.cache + (size_t) b * D;
    for (int d = tid; d < D; d += blockDim.x) out[d] = __half_as_ushort(__float2half(v[d]));
}
// the ring takes the batch's last `state` rows (after every block above read the old ones)
__global__ void comp_ring_kernel(const BatchCompArgs a) {
    const BatchComp& B = a.c[blockIdx.y];
    const Comp& C = B.c;
    const int W = (C.overlap ? 2 : 1) * C.D, state = (C.overlap ? 2 : 1) * C.ratio;
    const int first = max(a.p0, a.p0 + a.T - state);
    const int pos = first + (int) blockIdx.x;
    if (pos >= a.p0 + a.T) return;
    const size_t src = (size_t) (pos - a.p0) * W, dst = (size_t) (pos % ring_rows(state)) * W;
    const float* ape = C.ape + (size_t) (pos % C.ratio) * W;
    for (int i = threadIdx.x; i < W; i += blockDim.x) {
        C.ring_kv[dst + i] = B.kv[src + i];
        C.ring_sc[dst + i] = B.sc[src + i] + ape[i];
    }
}

__global__ void __launch_bounds__(128) q_prep_batch_kernel(float* q, int n_head, int head_dim, int n_rot, Rope rope,
                                                           float* iq, int idx_heads, int idx_dim, Rope comp_rope,
                                                           int p0, float eps) {
    __shared__ float sh[32];
    __shared__ float v[512];
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, p = p0 + t;
    if (h < n_head) {
        float* x = q + ((size_t) t * n_head + h) * head_dim;
        float ss = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            v[i] = x[i];
            ss += v[i] * v[i];
        }
        const float inv = rsqrtf(block_sum(ss, sh) / (float) head_dim + eps);
        for (int i = tid; i < head_dim; i += blockDim.x) v[i] *= inv;
        __syncthreads();
        if (tid < n_rot / 2) rope_pair(v, head_dim, n_rot, rope, p, true, tid);
        __syncthreads();
        for (int i = tid; i < head_dim; i += blockDim.x) x[i] = v[i];
        return;
    }
    float* x = iq + ((size_t) t * idx_heads + (h - n_head)) * idx_dim;
    for (int i = tid; i < idx_dim; i += blockDim.x) v[i] = x[i];
    __syncthreads();
    if (tid < n_rot / 2) rope_pair(v, idx_dim, n_rot, comp_rope, p, true, tid);
    __syncthreads();
    for (int i = tid; i < idx_dim; i += blockDim.x) x[i] = v[i];
}

// a tile of 32 tokens x 32 compressed rows over every indexer head (idx_dim 128): K once, each head's q in turn
constexpr int IQT = 32, IRT = 32, IPAD = 129;
__global__ void __launch_bounds__(256) idx_score_batch_kernel(const float* __restrict__ iq, const uint16_t* __restrict__ K,
                                                              const float* __restrict__ w, int idx_heads, int ratio,
                                                              int p0, int T, float* __restrict__ score, int ld) {
    __shared__ float s_k[IRT * IPAD];
    __shared__ float s_q[IQT * IPAD];
    __shared__ float s_w[IQT * 64];
    const int tid = threadIdx.x;
    const int t0 = blockIdx.y * IQT, j0 = blockIdx.x * IRT;
    const int max_vis = (p0 + min(T, t0 + IQT) - 1 + 1) / ratio;   // the tile's last token sees the most rows
    if (j0 >= max_vis) {
        // rows nobody in the tile sees: -inf for the tokens in range
        for (int i = tid; i < IQT * IRT; i += blockDim.x) {
            const int tq = i / IRT, r = i % IRT;
            if (t0 + tq < T && j0 + r < ld) score[(size_t) (t0 + tq) * ld + j0 + r] = -INFINITY;
        }
        return;
    }
    for (int i = tid; i < IRT * 128; i += blockDim.x) {
        const int r = i / 128, k = i % 128;
        s_k[r * IPAD + k] = j0 + r < max_vis ? __half2float(__ushort_as_half(K[(size_t) (j0 + r) * 128 + k])) : 0.0f;
    }
    for (int i = tid; i < IQT * idx_heads; i += blockDim.x) {
        const int tq = i / idx_heads, h = i % idx_heads;
        s_w[tq * 64 + h] = t0 + tq < T ? w[(size_t) (t0 + tq) * idx_heads + h] : 0.0f;
    }
    const int tq = tid / 8, tr = tid % 8;
    constexpr int NR = IRT / 8;   // rows a thread scores: tr, tr + 8, ..
    float acc[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) acc[i] = 0.0f;
    for (int h = 0; h < idx_heads; ++h) {
        __syncthreads();
        for (int i = tid; i < IQT * 128; i += blockDim.x) {
            const int q = i / 128, k = i % 128;
            s_q[q * IPAD + k] = t0 + q < T ? iq[((size_t) (t0 + q) * idx_heads + h) * 128 + k] : 0.0f;
        }
        __syncthreads();
        float d[NR];
#pragma unroll
        for (int i = 0; i < NR; ++i) d[i] = 0.0f;
        const float* qq = s_q + tq * IPAD;
        for (int k = 0; k < 128; ++k) {
            const float x = qq[k];
#pragma unroll
            for (int i = 0; i < NR; ++i) d[i] += x * s_k[(tr + 8 * i) * IPAD + k];
        }
        const float wh = s_w[tq * 64 + h];
#pragma unroll
        for (int i = 0; i < NR; ++i) acc[i] += fmaxf(d[i], 0.0f) * wh;
    }
    const int t = t0 + tq;
    if (t >= T) return;
    const int n_vis = (p0 + t + 1) / ratio;
#pragma unroll
    for (int i = 0; i < NR; ++i) {
        const int j = j0 + tr + 8 * i;
        if (j < ld) score[(size_t) t * ld + j] = j < n_vis ? acc[i] : -INFINITY;
    }
}
// The same scores on tensor cores (sm_80+, 64 heads), as the decode kernel idx_score_mma_kernel computes them: a warp
// holds 2 x 16 K rows as A fragments for a tile of 32 tokens, each token's Q split into hi + lo FP16 B fragments in
// shared memory (double buffered: the next token converts while this one multiplies). Rows a token does not see are
// left unwritten (the select reads n_vis of them). 177 -> 21.6 ms for 4096 tokens at 120k on an RTX 5090.
constexpr int IBM_WARPS = 8, IBM_RG = 2, IBM_TT = 32, IBM_ROWS = IBM_WARPS * IBM_RG * 16;
constexpr int IBM_SMEM = 2 * 2048 * (int) sizeof(uint4);
__device__ __forceinline__ void ibm_load_q(uint4* sb, float* sw, const float* __restrict__ q, const float* __restrict__ w) {
    for (int i = threadIdx.x; i < 2048; i += IBM_WARPS * 32) {   // [n tile 8][k step 8][lane 32], as s_b above
        const int ln = i & 31, ks = (i >> 5) & 7, nt = i >> 8;
        const float* qn = q + (nt * 8 + (ln >> 2)) * 128 + ks * 16 + (ln & 3) * 2;
        const float2 v01 = *reinterpret_cast<const float2*>(qn), v89 = *reinterpret_cast<const float2*>(qn + 8);
        const float v[4] = {v01.x, v01.y, v89.x, v89.y};
        __half h[4], l[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            h[t] = __float2half_rn(v[t]);
            l[t] = __float2half_rn(v[t] - __half2float(h[t]));
        }
        sb[i] = make_uint4(idx_h2(h[0], h[1]), idx_h2(h[2], h[3]), idx_h2(l[0], l[1]), idx_h2(l[2], l[3]));
    }
    if (threadIdx.x < 64) sw[threadIdx.x] = w[threadIdx.x];
}
__global__ void __launch_bounds__(IBM_WARPS * 32) idx_score_batch_mma_kernel(const float* __restrict__ iq,
                                                                             const uint16_t* __restrict__ K,
                                                                             const float* __restrict__ w, int ratio,
                                                                             int p0, int T, float* __restrict__ score,
                                                                             int ld) {
    extern __shared__ uint4 s_bq[];   // [2][2048]
    __shared__ float s_wq[2][64];
    const int t0 = blockIdx.y * IBM_TT, j0 = blockIdx.x * IBM_ROWS;
    const int tn = min(IBM_TT, T - t0);
    int tf = 0;   // the tile's first token that sees row j0
    while (tf < tn && (p0 + t0 + tf + 1) / ratio <= j0) ++tf;
    if (tf >= tn) return;
    const int lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
    const int wrow = j0 + (threadIdx.x >> 5) * IBM_RG * 16;
    const int nrow = (p0 + t0 + tn) / ratio;   // the most rows a token of the tile sees
    unsigned a[IBM_RG][8][4];
#pragma unroll
    for (int r = 0; r < IBM_RG; ++r) {
        const int ra = wrow + r * 16 + g, rb = ra + 8;
        const unsigned* Ka = reinterpret_cast<const unsigned*>(K + (size_t) ra * 128);
        const unsigned* Kb = reinterpret_cast<const unsigned*>(K + (size_t) rb * 128);
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const int c = ks * 8 + t4;
            a[r][ks][0] = ra < nrow ? Ka[c] : 0u;
            a[r][ks][1] = rb < nrow ? Kb[c] : 0u;
            a[r][ks][2] = ra < nrow ? Ka[c + 4] : 0u;
            a[r][ks][3] = rb < nrow ? Kb[c + 4] : 0u;
        }
    }
    ibm_load_q(s_bq, s_wq[0], iq + (size_t) (t0 + tf) * 64 * 128, w + (size_t) (t0 + tf) * 64);
    for (int i = tf; i < tn; ++i) {
        const int buf = (i - tf) & 1;
        __syncthreads();
        if (i + 1 < tn)
            ibm_load_q(s_bq + (buf ^ 1) * 2048, s_wq[buf ^ 1], iq + (size_t) (t0 + i + 1) * 64 * 128,
                       w + (size_t) (t0 + i + 1) * 64);
        const int t = t0 + i, n_vis = (p0 + t + 1) / ratio;
        if (wrow >= n_vis) continue;
        const uint4* sb = s_bq + buf * 2048;
        const float* sw = s_wq[buf];
        float sa[IBM_RG], sc[IBM_RG];
#pragma unroll
        for (int r = 0; r < IBM_RG; ++r) sa[r] = sc[r] = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            float d[IBM_RG][4];
#pragma unroll
            for (int r = 0; r < IBM_RG; ++r) d[r][0] = d[r][1] = d[r][2] = d[r][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const uint4 b = sb[(nt * 8 + ks) * 32 + lane];
#pragma unroll
                for (int r = 0; r < IBM_RG; ++r) {
                    idx_mma16816(d[r], a[r][ks], b.x, b.y);
                    idx_mma16816(d[r], a[r][ks], b.z, b.w);
                }
            }
            const float w0 = sw[nt * 8 + t4 * 2], w1 = sw[nt * 8 + t4 * 2 + 1];
#pragma unroll
            for (int r = 0; r < IBM_RG; ++r) {
                sa[r] += fmaxf(d[r][0], 0.0f) * w0 + fmaxf(d[r][1], 0.0f) * w1;
                sc[r] += fmaxf(d[r][2], 0.0f) * w0 + fmaxf(d[r][3], 0.0f) * w1;
            }
        }
        float* out = score + (size_t) t * ld;
#pragma unroll
        for (int r = 0; r < IBM_RG; ++r) {
            float x = sa[r], y = sc[r];
            x += __shfl_xor_sync(0xffffffffu, x, 1);
            x += __shfl_xor_sync(0xffffffffu, x, 2);
            y += __shfl_xor_sync(0xffffffffu, y, 1);
            y += __shfl_xor_sync(0xffffffffu, y, 2);
            const int ra = wrow + r * 16 + g, rb = ra + 8;
            if (t4 == 0) {
                if (ra < n_vis) out[ra] = x;
                if (rb < n_vis) out[rb] = y;
            }
        }
    }
}
#if defined(STRATA_USE_HIP)
// The prompt path's scores on WMMA (HIP, gfx11 / gfx12): Q converted once (iwb_prep: each 128-float row -> 128 FP16 hi
// + 128 FP16 lo, in place - the same 512 bytes; the indexer is its last reader), then a wave holds 2 x 16 K rows as A
// fragments for a tile of 32 tokens and reads the B fragments straight from the converted Q (L1/L2: no LDS, no
// barrier). Rows a token does not see are left unwritten. 1,024 tokens at 120k on a Radeon 8060S: 214 -> 36.5 ms
// (prep 0.24), within 3e-6 of the SIMT kernel's scores. (Q through LDS a token at a time: 143 ms.)
constexpr int IWB2_WAVES = 8, IWB2_RG = 2, IWB2_TT = 32, IWB2_ROWS = IWB2_WAVES * IWB2_RG * 16;
__global__ void __launch_bounds__(256) iwb_prep_kernel(float* __restrict__ iq, int rows) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= rows) return;
    float* x = iq + (size_t) r * 128;
    const float4 v = reinterpret_cast<const float4*>(x)[lane];
    const float f[4] = {v.x, v.y, v.z, v.w};
    _Float16 h[4], l[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        h[i] = (_Float16) f[i];
        l[i] = (_Float16) (f[i] - (float) h[i]);
    }
    __builtin_amdgcn_wave_barrier();   // (every lane has read its floats before any writes: one wave a row)
    _Float16* o = reinterpret_cast<_Float16*>(x);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        o[lane * 4 + i] = h[i];
        o[128 + lane * 4 + i] = l[i];
    }
}
// H heads (64: deepseek4; 32: deepseek41, v1.3.63)
template <int H>
__global__ void __launch_bounds__(IWB2_WAVES * 32) idx_score_batch_wmma2_kernel(const float* __restrict__ iqp,
                                                                                const uint16_t* __restrict__ K,
                                                                                const float* __restrict__ w, int ratio,
                                                                                int p0, int T, float* __restrict__ score,
                                                                                int ld) {
    const int t0 = blockIdx.y * IWB2_TT, j0 = blockIdx.x * IWB2_ROWS;
    const int tn = min(IWB2_TT, T - t0);
    const int lane = threadIdx.x & 31, l16 = lane & 15, hi = lane >> 4;
    const int wrow = j0 + (threadIdx.x >> 5) * IWB2_RG * 16;
    const int nrow = (p0 + t0 + tn) / ratio;
    if (wrow >= nrow) return;
    iw_h16 a[IWB2_RG][8];
#pragma unroll
    for (int g = 0; g < IWB2_RG; ++g) {
        const int ra = min(wrow + g * 16 + l16, nrow - 1);
        const uint4* kr = reinterpret_cast<const uint4*>(K + (size_t) ra * 128);
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            union { uint4 u[2]; iw_h16 v; } x;
            x.u[0] = kr[ks * 2];
            x.u[1] = kr[ks * 2 + 1];
            a[g][ks] = x.v;
        }
    }
    for (int i = 0; i < tn; ++i) {
        const int t = t0 + i, n_vis = (p0 + t + 1) / ratio;
        if (wrow >= n_vis) continue;
        const _Float16* q = reinterpret_cast<const _Float16*>(iqp + (size_t) t * H * 128);   // [head][hi 128 | lo 128]
        const float* wt = w + (size_t) t * H;
        float s[IWB2_RG][8];
#pragma unroll
        for (int g = 0; g < IWB2_RG; ++g)
#pragma unroll
            for (int r = 0; r < 8; ++r) s[g][r] = 0.0f;
#pragma unroll
        for (int nt = 0; nt < H / 16; ++nt) {
            const _Float16* qh = q + (size_t) (nt * 16 + l16) * 256;
            iw_f8 d[IWB2_RG] = {};
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const iw_h16 bh = *reinterpret_cast<const iw_h16*>(qh + ks * 16);
                const iw_h16 bl = *reinterpret_cast<const iw_h16*>(qh + 128 + ks * 16);
#pragma unroll
                for (int g = 0; g < IWB2_RG; ++g) {
                    d[g] = iw_wmma(a[g][ks], bh, d[g]);
                    d[g] = iw_wmma(a[g][ks], bl, d[g]);
                }
            }
            const float wh = wt[nt * 16 + l16];
#pragma unroll
            for (int g = 0; g < IWB2_RG; ++g)
#pragma unroll
                for (int r = 0; r < 8; ++r) s[g][r] += fmaxf(d[g][r], 0.0f) * wh;
        }
        float* out = score + (size_t) t * ld;
#pragma unroll
        for (int g = 0; g < IWB2_RG; ++g)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                float x = s[g][r];
                x += __shfl_xor(x, 1, 32);
                x += __shfl_xor(x, 2, 32);
                x += __shfl_xor(x, 4, 32);
                x += __shfl_xor(x, 8, 32);
                const int j = wrow + g * 16 + iw_row(r, hi);
                if (l16 == 0 && j < n_vis) out[j] = x;
            }
    }
}
#endif
__global__ void __launch_bounds__(1024) idx_select_batch_kernel(const float* __restrict__ score, int ld, int ratio, int p0,
                                                                int top_k, int* __restrict__ sel, int sel_ld) {
    const int t = blockIdx.x;
    const int n_vis = (p0 + t + 1) / ratio;
    if (n_vis <= 0) return;
    strata::kernels::dsa::select_top_pools(score + (size_t) t * ld, n_vis, min(top_k, n_vis), 1, sel + (size_t) t * sel_ld);

}

// The compressed rows token p attends: deepseek4's ratio 128 every visible row, ratio 4 sel's min(top_k, visible);
// deepseek41 (ratio passed negated: -1 / -2) every visible row while they fit top_k, else sel's top_k - the token
// path's rule (dsv41_fast_path.cu).  use_sel: the rows come from sel
__device__ __forceinline__ int comp_rows(int ratio, int p, int top_k, bool& use_sel) {
    use_sel = false;
    if (ratio == 128) return (p + 1) / 128;
    if (ratio == 4) {
        use_sel = true;
        return min(top_k, (p + 1) / 4);
    }
    if (ratio < 0) {
        const int vis = (p + 1) / -ratio;
        use_sel = vis > top_k;
        return use_sel ? top_k : vis;
    }
    return 0;
}

// 8 heads of one token per block, online softmax over tiles of 32 keys (the sink seeds the max and the sum)
constexpr int AB_H = 8, AB_TILE = 32;
__global__ void __launch_bounds__(256) attend_batch_kernel(const float* __restrict__ q, const uint16_t* __restrict__ keys,
                                                           int n_hist, int n_swa, const uint16_t* __restrict__ comp,
                                                           const int* __restrict__ sel, int top_k, int sel_ld, int ratio,
                                                           const float* __restrict__ sinks, int n_head, int n_rot,
                                                           Rope rope, int p0, float scale, float* __restrict__ out) {
    constexpr int D = 512;
    __shared__ float s_q[AB_H][D];
    __shared__ float s_l[AB_TILE][AB_H];
    __shared__ float s_m[AB_H], s_sum[AB_H], s_scale[AB_H];
    const int t = blockIdx.x, h0 = blockIdx.y * AB_H, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int p = p0 + t;
    for (int i = tid; i < AB_H * D; i += blockDim.x)
        s_q[i / D][i % D] = q[((size_t) t * n_head + h0 + i / D) * D + i % D];
    if (tid < AB_H) {
        s_m[tid] = sinks[h0 + tid];
        s_sum[tid] = 1.0f;
    }
    const int n_raw = min(p + 1, n_swa);
    const int raw0 = p - n_raw + 1 - (p0 - n_hist);   // the window's first row in keys
    bool use_sel;
    const int n_c = comp_rows(ratio, p, top_k, use_sel);
    const int n = n_raw + n_c;
    const auto key = [&](int j) -> const __half2* {
        if (j < n_raw) return (const __half2*) (keys + (size_t) (raw0 + j) * D);
        const int r = use_sel ? sel[(size_t) t * sel_ld + (j - n_raw)] : j - n_raw;
        return (const __half2*) (comp + (size_t) r * D);
    };
    float acc[AB_H][2];
#pragma unroll
    for (int h = 0; h < AB_H; ++h) acc[h][0] = acc[h][1] = 0.0f;
    __syncthreads();
    for (int j0 = 0; j0 < n; j0 += AB_TILE) {
        const int nt = min(AB_TILE, n - j0);
        // 1. the tile's logits: warp w takes keys w, w + 8, .. ; lane l the half2 pairs l, l + 32, ..
        for (int jj = warp; jj < nt; jj += 8) {
            const __half2* k2 = key(j0 + jj);
            float part[AB_H];
#pragma unroll
            for (int h = 0; h < AB_H; ++h) part[h] = 0.0f;
#pragma unroll
            for (int i = 0; i < D / 64; ++i) {
                const float2 kv = __half22float2(k2[lane + 32 * i]);
                const int e = 2 * (lane + 32 * i);
#pragma unroll
                for (int h = 0; h < AB_H; ++h) part[h] += s_q[h][e] * kv.x + s_q[h][e + 1] * kv.y;
            }
#pragma unroll
            for (int h = 0; h < AB_H; ++h) {
                const float v = warp_sum(part[h]);
                if (lane == 0) s_l[jj][h] = v * scale;
            }
        }
        __syncthreads();
        // 2. the running max and sum per head; the tile's weights replace its logits
        if (tid < AB_H) {
            const int h = tid;
            float m = s_m[h];
            for (int jj = 0; jj < nt; ++jj) m = fmaxf(m, s_l[jj][h]);
            const float sc = expf(s_m[h] - m);
            float sum = s_sum[h] * sc;
            for (int jj = 0; jj < nt; ++jj) {
                const float e = expf(s_l[jj][h] - m);
                s_l[jj][h] = e;
                sum += e;
            }
            s_m[h] = m;
            s_sum[h] = sum;
            s_scale[h] = sc;
        }
        __syncthreads();
        // 3. the values: thread t owns the pair (2t, 2t + 1) of every head
#pragma unroll
        for (int h = 0; h < AB_H; ++h) {
            acc[h][0] *= s_scale[h];
            acc[h][1] *= s_scale[h];
        }
        for (int jj = 0; jj < nt; ++jj) {
            const float2 kv = __half22float2(key(j0 + jj)[tid]);
#pragma unroll
            for (int h = 0; h < AB_H; ++h) {
                const float wj = s_l[jj][h];
                acc[h][0] += wj * kv.x;
                acc[h][1] += wj * kv.y;
            }
        }
        __syncthreads();
    }
    // 4. normalise, de-rope (V = K carries the rotation) and write
#pragma unroll
    for (int h = 0; h < AB_H; ++h) {
        const float r = 1.0f / s_sum[h];
        s_q[h][2 * tid] = acc[h][0] * r;
        s_q[h][2 * tid + 1] = acc[h][1] * r;
    }
    __syncthreads();
    {
        const int h = tid / (n_rot / 2), pr = tid % (n_rot / 2);
        if (h < AB_H) rope_pair(s_q[h], D, n_rot, rope, p, false, pr);
    }
    __syncthreads();
    for (int i = tid; i < AB_H * D; i += blockDim.x)
        out[((size_t) t * n_head + h0 + i / D) * D + i % D] = s_q[i / D][i % D];
}

// The same attention on tensor cores (sm_80+, head_dim 512, 16 | n_head): a block = one token x 16 heads, warp w owns
// dims [128w, 128w + 128) of Q, K (= V) and O, so each key is read once for 16 heads (the kernel above: once for 8).
// Per tile of 32 keys: the warps load their slices, S partials (QK over 128 dims) meet in shared memory for the online
// softmax, and P comes back for P.V on each warp's slice. Q and P go in as FP16 hi + lo pairs, so the result keeps
// ~fp32 accuracy (rel. RMS 2e-6 against the kernel above). 4,096 tokens at 120k, RTX 5090: 26.7 -> 6.5 ms (CSA layers),
// 37.1 -> 8.9 ms (HCA). STRATA_DS4_ATTN=1: the kernel above (A/B).
__device__ __forceinline__ void abm_ldsm_t(const uint16_t* src, unsigned& b0, unsigned& b1, unsigned& b2, unsigned& b3) {
#if !STRATA_DS4_IDX_SM80
    (void) src;
    b0 = b1 = b2 = b3 = 0u;
    __trap();
#else
    const unsigned addr = (unsigned) __cvta_generic_to_shared(src);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                 : "r"(addr));
#endif
}
__device__ __forceinline__ unsigned abm_h2(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const unsigned*>(&h);
}
constexpr int ABM_H = 16, ABM_TILE = 32, ABM_KS = 136;   // heads a block, keys a tile, a key row's stride (halves)
__global__ void __launch_bounds__(128) attend_batch_mma_kernel(const float* __restrict__ q, const uint16_t* __restrict__ keys,
                                                               int n_hist, int n_swa, const uint16_t* __restrict__ comp,
                                                               const int* __restrict__ sel, int top_k, int sel_ld, int ratio,
                                                               const float* __restrict__ sinks, int n_head, int n_rot,
                                                               Rope rope, int p0, float scale, float* __restrict__ out) {
    constexpr int D = 512;
    __shared__ __align__(16) uint16_t s_k[4][ABM_TILE][ABM_KS];   // reused for O (16 x 512 floats) at the end
    __shared__ float s_s[4][ABM_H][ABM_TILE + 1];
    __shared__ __align__(16) uint16_t s_p[2][ABM_H][40];          // P hi, lo
    __shared__ float s_scale[ABM_H], s_sum[ABM_H];
    const int t = blockIdx.x, h0 = blockIdx.y * ABM_H, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    const int p = p0 + t;
    const int n_raw = min(p + 1, n_swa);
    const int raw0 = p - n_raw + 1 - (p0 - n_hist);
    bool use_sel;
    const int n_c = comp_rows(ratio, p, top_k, use_sel);
    const int n = n_raw + n_c;
    // Q's A fragments for this warp's dims: [k step 8][4], hi (and lo)
    unsigned qa[8][4], ql[8][4];
    {
        const float* qg = q + ((size_t) t * n_head + h0 + g) * D + w * 128;
        const float* qg8 = qg + 8 * D;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const int c = ks * 16 + t4 * 2;
            const float2 v[4] = {*reinterpret_cast<const float2*>(qg + c), *reinterpret_cast<const float2*>(qg8 + c),
                                 *reinterpret_cast<const float2*>(qg + c + 8), *reinterpret_cast<const float2*>(qg8 + c + 8)};
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                qa[ks][r] = abm_h2(v[r].x, v[r].y);
                {
                    const float2 hv = __half22float2(*reinterpret_cast<const __half2*>(&qa[ks][r]));
                    ql[ks][r] = abm_h2(v[r].x - hv.x, v[r].y - hv.y);
                }
            }
        }
    }
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    // the softmax's state: thread tid owns head tid / 8 (all 8 threads of the group hold it)
    const int sh = tid >> 3, sub = tid & 7;
    float m = sinks[h0 + sh], sum = 1.0f;
    for (int j0 = 0; j0 < n; j0 += ABM_TILE) {
        // 1. this warp's slice of the tile's keys (zeros past the end)
#pragma unroll 4
        for (int i = lane; i < ABM_TILE * 16; i += 32) {
            const int kk = i >> 4, ch = i & 15, j = j0 + kk;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (j < n) {
                const uint16_t* row = j < n_raw ? keys + (size_t) (raw0 + j) * D
                                                : comp + (size_t) (use_sel ? sel[(size_t) t * sel_ld + (j - n_raw)] : j - n_raw) * D;
                v = *reinterpret_cast<const uint4*>(row + w * 128 + ch * 8);
            }
            *reinterpret_cast<uint4*>(&s_k[w][kk][ch * 8]) = v;
        }
        __syncwarp();
        // 2. S partials over this warp's 128 dims: 16 heads x 32 keys
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const uint16_t* kr = &s_k[w][nt * 8 + g][t4 * 2];
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const unsigned b0 = *reinterpret_cast<const unsigned*>(kr + ks * 16);
                const unsigned b1 = *reinterpret_cast<const unsigned*>(kr + ks * 16 + 8);
                idx_mma16816(d, qa[ks], b0, b1);
                idx_mma16816(d, ql[ks], b0, b1);
            }
            s_s[w][g][nt * 8 + t4 * 2] = d[0];
            s_s[w][g][nt * 8 + t4 * 2 + 1] = d[1];
            s_s[w][g + 8][nt * 8 + t4 * 2] = d[2];
            s_s[w][g + 8][nt * 8 + t4 * 2 + 1] = d[3];
        }
        __syncthreads();
        // 3. online softmax: head sh, keys sub + 8i
        {
            float l[4], tmax = -INFINITY;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int kk = sub + 8 * i;
                l[i] = j0 + kk < n ? (s_s[0][sh][kk] + s_s[1][sh][kk] + s_s[2][sh][kk] + s_s[3][sh][kk]) * scale : -INFINITY;
                tmax = fmaxf(tmax, l[i]);
            }
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 4));
            const float mn = fmaxf(m, tmax), sc = expf(m - mn);
            float ps = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float e = expf(l[i] - mn);
                const __half eh = __float2half_rn(e);
                s_p[0][sh][sub + 8 * i] = __half_as_ushort(eh);
                s_p[1][sh][sub + 8 * i] = __half_as_ushort(__float2half_rn(e - __half2float(eh)));
                ps += e;
            }
            ps += __shfl_xor_sync(0xffffffffu, ps, 1);
            ps += __shfl_xor_sync(0xffffffffu, ps, 2);
            ps += __shfl_xor_sync(0xffffffffu, ps, 4);
            sum = sum * sc + ps;
            m = mn;
            if (sub == 0) s_scale[sh] = sc;
        }
        __syncthreads();
        // 4. O = O * scale + P . V over this warp's 128 dims
        {
            const float sa = s_scale[g], sb = s_scale[g + 8];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                o[i][0] *= sa; o[i][1] *= sa; o[i][2] *= sb; o[i][3] *= sb;
            }
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                unsigned pa[4], pl[4];
                const int c = ks * 16 + t4 * 2;
                pa[0] = *reinterpret_cast<const unsigned*>(&s_p[0][g][c]);
                pa[1] = *reinterpret_cast<const unsigned*>(&s_p[0][g + 8][c]);
                pa[2] = *reinterpret_cast<const unsigned*>(&s_p[0][g][c + 8]);
                pa[3] = *reinterpret_cast<const unsigned*>(&s_p[0][g + 8][c + 8]);
                {
                    pl[0] = *reinterpret_cast<const unsigned*>(&s_p[1][g][c]);
                    pl[1] = *reinterpret_cast<const unsigned*>(&s_p[1][g + 8][c]);
                    pl[2] = *reinterpret_cast<const unsigned*>(&s_p[1][g][c + 8]);
                    pl[3] = *reinterpret_cast<const unsigned*>(&s_p[1][g + 8][c + 8]);
                }
                const int kr = ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
                for (int np = 0; np < 8; ++np) {   // two n tiles (16 dims) per ldmatrix
                    unsigned b0, b1, b2, b3;
                    abm_ldsm_t(&s_k[w][kr][np * 16 + (lane >> 4) * 8], b0, b1, b2, b3);
                    idx_mma16816(o[2 * np], pa, b0, b1);
                    idx_mma16816(o[2 * np + 1], pa, b2, b3);
                    {
                        idx_mma16816(o[2 * np], pl, b0, b1);
                        idx_mma16816(o[2 * np + 1], pl, b2, b3);
                    }
                }
            }
        }
        __syncthreads();
    }
    // 5. normalise, de-rope (V = K carries the rotation) and write
    if (sub == 0) s_sum[sh] = sum;
    __syncthreads();
    float* s_o = reinterpret_cast<float*>(&s_k[0][0][0]);   // [16][512]
    {
        const float ra = 1.0f / s_sum[g], rb = 1.0f / s_sum[g + 8];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int dcol = w * 128 + i * 8 + t4 * 2;
            *reinterpret_cast<float2*>(&s_o[g * D + dcol]) = make_float2(o[i][0] * ra, o[i][1] * ra);
            *reinterpret_cast<float2*>(&s_o[(g + 8) * D + dcol]) = make_float2(o[i][2] * rb, o[i][3] * rb);
        }
    }
    __syncthreads();
    for (int i = tid; i < ABM_H * (n_rot / 2); i += blockDim.x) rope_pair(s_o + (i / (n_rot / 2)) * D, D, n_rot, rope, p, false, i % (n_rot / 2));
    __syncthreads();
    float4* og = reinterpret_cast<float4*>(out + ((size_t) t * n_head + h0) * D);
    for (int i = tid; i < ABM_H * D / 4; i += blockDim.x) og[i] = reinterpret_cast<const float4*>(s_o)[i];
}

#if defined(STRATA_USE_HIP)
// The prompt path's attention on RDNA's WMMA (HIP, gfx11 / gfx12; the HIP counterpart of attend_batch_mma_kernel, as
// glm_batch.cu's mla_attn_wmma2_kernel): a block = one token x 16 heads, 8 waves, wave w owns
// dims [64w, 64w + 64) of Q, K (= V) and O. Per tile of 16 keys: the keys into LDS (FP16), each wave's QK partial
// over its 64 dims (Q as FP16 hi + lo, scaled by a power of two per head and wave into FP16's range), the partials
// summed for the online softmax (the sink seeds the max and the sum), P as FP16 hi + lo for P.V on the wave's dims.
// Then normalise, de-rope (V = K carries the rotation) and write. Radeon 8060S, 1,024 tokens at 30k: 37.0 -> 15.3 ms
// (CSA layers), 18.5 -> 10.0 (HCA), rel. RMS 4e-6 against the kernel above. STRATA_DS4_ATTN=1: the kernel above.
constexpr int AW_H = 16;
__global__ void __launch_bounds__(256) attend_batch_wmma_kernel(const float* __restrict__ q, const uint16_t* __restrict__ keys,
                                                                int n_hist, int n_swa, const uint16_t* __restrict__ comp,
                                                                const int* __restrict__ sel, int top_k, int sel_ld, int ratio,
                                                                const float* __restrict__ sinks, int n_head, int n_rot,
                                                                Rope rope, int p0, float scale, float* __restrict__ out) {
    constexpr int D = 512;
    __shared__ __align__(16) _Float16 sl[16][D];
    __shared__ __align__(16) _Float16 ph[16][16], pl[16][16];
    __shared__ float scores[8][16][16], qi[8][16];
    __shared__ float sm[16], denom[16], rescale[16];
    __shared__ const uint16_t* rows[16];
    __shared__ float s_o[AW_H][D];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int t = blockIdx.x, h0 = blockIdx.y * AW_H, p = p0 + t;
    const int n_raw = min(p + 1, n_swa);
    const int raw0 = p - n_raw + 1 - (p0 - n_hist);
    bool use_sel;
    const int n_c = comp_rows(ratio, p, top_k, use_sel);
    const int n = n_raw + n_c;
    // Q: head l16, dims [64 wave, +64), scaled into FP16's range (the scale's inverse undoes it on the scores)
    const float* qr = q + ((size_t) t * n_head + h0 + l16) * D + wave * 64;
    float mx = 0.0f;
#pragma unroll
    for (int k = 0; k < 64; ++k) mx = fmaxf(mx, fabsf(qr[k]));
    int ex = 0;
    if (mx > 0.0f) { frexpf(mx, &ex); ex = max(-60, min(60, 14 - ex)); }
    const float f = ldexpf(1.0f, ex);
    if (hi == 0) qi[wave][l16] = ldexpf(1.0f, -ex);
    iw_h16 qh[4], ql[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float v = qr[16 * k + j] * f;
            qh[k][j] = (_Float16) v;
            ql[k][j] = (_Float16) (v - (float) qh[k][j]);
        }
    iw_f8 acc[4] = {};
    if (tid < 16) { sm[tid] = sinks[h0 + tid]; denom[tid] = 1.0f; }
    for (int s0 = 0; s0 < n; s0 += 16) {
        __syncthreads();
        if (tid < 16) {
            const int j = s0 + tid;
            rows[tid] = j >= n ? nullptr
                        : j < n_raw ? keys + (size_t) (raw0 + j) * D
                                    : comp + (size_t) (use_sel ? sel[(size_t) t * sel_ld + (j - n_raw)] : j - n_raw) * D;
        }
        __syncthreads();
        for (int i = tid; i < 16 * D / 8; i += 256) {
            const int r = i / (D / 8), col = (i % (D / 8)) * 8;
            *reinterpret_cast<uint4*>(&sl[r][col]) = rows[r] == nullptr ? make_uint4(0u, 0u, 0u, 0u)
                                                                         : *reinterpret_cast<const uint4*>(rows[r] + col);
        }
        __syncthreads();
        iw_f8 dot = {};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const iw_h16 b = *reinterpret_cast<const iw_h16*>(&sl[l16][wave * 64 + 16 * k]);
            dot = iw_wmma(qh[k], b, dot);
            dot = iw_wmma(ql[k], b, dot);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = iw_row(i, hi);
            scores[wave][row][l16] = dot[i] * qi[wave][row];
        }
        __syncthreads();
        {   // the online softmax: head r, key col
            const int r = tid >> 4, col = tid & 15;
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) v += scores[w][r][col];
            v = s0 + col < n ? v * scale : -INFINITY;
            float m = v;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
            const float old_m = sm[r], new_m = fmaxf(old_m, m);
            const float pv = v == -INFINITY ? 0.0f : expf(v - new_m);
            float sum = pv;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
            const _Float16 phv = (_Float16) pv;
            ph[r][col] = phv;
            pl[r][col] = (_Float16) (pv - (float) phv);
            __syncthreads();   // (every thread has read sm[r] before it changes)
            if (col == 0) {
                const float sc = expf(old_m - new_m);
                rescale[r] = sc;
                denom[r] = denom[r] * sc + sum;
                sm[r] = new_m;
            }
        }
        __syncthreads();
        const iw_h16 a_hi = *reinterpret_cast<const iw_h16*>(ph[l16]), a_lo = *reinterpret_cast<const iw_h16*>(pl[l16]);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[nt][i] *= rescale[iw_row(i, hi)];
            iw_h16 b;
#pragma unroll
            for (int k = 0; k < 16; ++k) b[k] = sl[k][wave * 64 + nt * 16 + l16];
            acc[nt] = iw_wmma(a_hi, b, acc[nt]);
            acc[nt] = iw_wmma(a_lo, b, acc[nt]);
        }
    }
    __syncthreads();
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = iw_row(i, hi);
            s_o[r][wave * 64 + nt * 16 + l16] = acc[nt][i] / denom[r];
        }
    __syncthreads();
    for (int i = tid; i < AW_H * (n_rot / 2); i += 256) rope_pair(s_o[i / (n_rot / 2)], D, n_rot, rope, p, false, i % (n_rot / 2));
    __syncthreads();
    float4* og = reinterpret_cast<float4*>(out + ((size_t) t * n_head + h0) * D);
    for (int i = tid; i < AW_H * D / 4; i += 256) og[i] = reinterpret_cast<const float4*>(&s_o[0][0])[i];
}
#endif

__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = __half_as_ushort(__float2half(x[i]));
}

// ---------------------------------------------------------------- DSpark (the block drafter, src/core/dspark.cu)
// the drafter's context rows: prep's kv half, one block a row, at positions p0 + t
__global__ void __launch_bounds__(512) kv_rows_kernel(const float* __restrict__ kv_raw, const float* __restrict__ kv_norm,
                                                      int D, int n_rot, Rope rope, int p0, uint16_t* __restrict__ ring,
                                                      int n_ring, float eps) {
    __shared__ float sh[32];
    __shared__ float v[1024];
    const int t = blockIdx.x, tid = threadIdx.x;
    const float* x = kv_raw + (size_t) t * D;
    float ss = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) ss += x[i] * x[i];
    const float inv = rsqrtf(block_sum(ss, sh) / (float) D + eps);
    for (int i = tid; i < D; i += blockDim.x) v[i] = x[i] * inv * kv_norm[i];
    __syncthreads();
    if (tid < n_rot / 2) rope_pair(v, D, n_rot, rope, p0 + t, true, tid);
    __syncthreads();
    uint16_t* row = ring + (size_t) ((p0 + t) % n_ring) * D;
    for (int i = tid; i < D; i += blockDim.x) row[i] = __half_as_ushort(__float2half(v[i]));
}

// a target layer's output -> the drafter's feature: the mean of the 4 streams, row y of n
__global__ void stream_mean_kernel(const float* __restrict__ R, size_t rs, int E, float* __restrict__ out, size_t os) {
    const float* Rr = R + (size_t) blockIdx.y * rs;
    float* o = out + (size_t) blockIdx.y * os;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < E) o[e] = 0.25f * (Rr[e] + Rr[E + e] + Rr[2 * E + e] + Rr[3 * E + e]);
}

// the drafter's router, a block a row: sc = sqrt(softplus(logit)), the top k of sc + bias (ties: the lower id), the
// weights sc / sum * w_scale over the experts with a VRAM slot (tab[e] != 0; the others are left out)
__global__ void __launch_bounds__(256) dspark_route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                           int n_expert, int k, float w_scale,
                                                           const unsigned long long* __restrict__ tab,
                                                           unsigned long long* __restrict__ plan_ptr,
                                                           float* __restrict__ plan_w, int* __restrict__ ids) {
    __shared__ float s_sc[512], s_sel[512];
    __shared__ int s_pick[8];
    const int r = blockIdx.x, tid = threadIdx.x;
    const float* lg = logits + (size_t) r * n_expert;
    for (int e = tid; e < n_expert; e += blockDim.x) {
        const float x = lg[e];
        const float sc = sqrtf(x > 20.0f ? x : log1pf(expf(x)));
        s_sc[e] = sc;
        s_sel[e] = sc + bias[e];
    }
    __syncthreads();
    if (tid < 32) {
        for (int i = 0; i < k; ++i) {
            float bv = -INFINITY;
            int bi = 0x7fffffff;
            for (int e = tid; e < n_expert; e += 32) {
                const float v = s_sel[e];
                if (v > bv || (v == bv && e < bi)) { bv = v; bi = e; }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (bi >= n_expert) bi = i;   // (NaN logits: any id)
            if (tid == 0) {
                s_pick[i] = bi;
                s_sel[bi] = -INFINITY;
            }
            __syncwarp();
        }
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i)
            if (tab[s_pick[i]] != 0ull) sum += s_sc[s_pick[i]];
        const float m = w_scale / fmaxf(sum, 6.103515625e-5f);
        int n = 0;
        for (int i = 0; i < k; ++i) {
            const int e = s_pick[i];
            ids[r * k + i] = e;
            const unsigned long long ptr = tab[e];
            if (ptr == 0ull) continue;
            plan_ptr[r * 8 + n] = ptr;
            plan_w[r * 8 + n] = s_sc[e] * m;
            ++n;
        }
        for (; n < 8; ++n) {
            plan_ptr[r * 8 + n] = 0ull;
            plan_w[r * 8 + n] = 0.0f;
        }
    }
}

// head_prep_kernel keeping h (the drafter's confidence reads it before the norm)
__global__ void __launch_bounds__(1024) dspark_head_kernel(const float* __restrict__ R, const uint16_t* __restrict__ fn,
                                                           const float* __restrict__ scale, const float* __restrict__ base,
                                                           float hc_eps, const float* __restrict__ norm_w, float eps,
                                                           int n_embd, float* __restrict__ hout, float* __restrict__ x,
                                                           block_q8_1* __restrict__ xq) {
    __shared__ float sh[32];
    __shared__ float s_pre[4];
    const int tid = threadIdx.x, F = 4 * n_embd;
    float ss = 0.0f, m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
    for (int i = tid; i < F; i += blockDim.x) {
        const float r = R[i];
        ss += r * r;
        m0 += bf(fn[i]) * r;
        m1 += bf(fn[(size_t) F + i]) * r;
        m2 += bf(fn[(size_t) 2 * F + i]) * r;
        m3 += bf(fn[(size_t) 3 * F + i]) * r;
    }
    const float inv = rsqrtf(block_sum(ss, sh) / (float) F + eps);
    m0 = block_sum(m0, sh);
    m1 = block_sum(m1, sh);
    m2 = block_sum(m2, sh);
    m3 = block_sum(m3, sh);
    if (tid == 0) {
        const float mm[4] = {m0, m1, m2, m3};
        for (int s = 0; s < 4; ++s) s_pre[s] = 1.0f / (1.0f + expf(-(mm[s] * inv * scale[0] + base[s]))) + hc_eps;
    }
    __syncthreads();
    float hs = 0.0f;
    for (int e = tid; e < n_embd; e += blockDim.x) {
        const float h = s_pre[0] * R[e] + s_pre[1] * R[n_embd + e] + s_pre[2] * R[2 * n_embd + e] +
                        s_pre[3] * R[3 * n_embd + e];
        hout[e] = h;
        x[e] = h;
        hs += h * h;
    }
    const float inv2 = rsqrtf(block_sum(hs, sh) / (float) n_embd + eps);
    for (int e = tid; e < n_embd; e += blockDim.x) {
        const float y = x[e] * inv2 * norm_w[e];
        x[e] = y;
        q8_1_store_warp_ds(y, xq + e / 32, tid & 31);
    }
}

// a row of the drafter's Markov chain: m = w1[*prev] (rank 256, F16), out[v] = logits[v] + w2[v] . m (Q8_0 rows of
// 256 - V4.1's drafter: F16 rows - a warp a row, lane l the values 8 l .. 8 l + 7); block 0 also the confidence
// sigmoid(cp . [h; m] (+ *cb, V4.1's bias))
template <bool F16W>
__global__ void __launch_bounds__(256) dspark_markov_kernel(const float* __restrict__ logits,
                                                            const void* __restrict__ w2v,
                                                            const __half* __restrict__ w1, int n_vocab,
                                                            const int* __restrict__ prev, const float* __restrict__ cp,
                                                            const float* __restrict__ cb,
                                                            const float* __restrict__ h, int n_embd,
                                                            float* __restrict__ out, float* __restrict__ conf) {
    constexpr int RANK = 256;
    __shared__ float s_m[RANK];
    __shared__ float sh[32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int pv = *prev;
    for (int i = tid; i < RANK; i += blockDim.x) s_m[i] = __half2float(w1[(size_t) pv * RANK + i]);
    __syncthreads();
    if (blockIdx.x == 0) {
        float a = 0.0f;
        for (int i = tid; i < n_embd; i += blockDim.x) a += cp[i] * h[i];
        for (int i = tid; i < RANK; i += blockDim.x) a += cp[n_embd + i] * s_m[i];
        a = block_sum(a, sh);
        if (cb != nullptr) a += *cb;
        if (tid == 0) *conf = 1.0f / (1.0f + expf(-a));
    }
    const int wpb = blockDim.x >> 5;
    if constexpr (F16W) {
        const __half* w2 = (const __half*) w2v;
        const int j0 = lane * 8;
        for (int v = blockIdx.x * wpb + warp; v < n_vocab; v += gridDim.x * wpb) {
            const __half* r = w2 + (size_t) v * RANK + j0;
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) acc += __half2float(r[j]) * s_m[j0 + j];
            acc = warp_sum(acc);
            if (lane == 0) out[v] = logits[v] + acc;
        }
    } else {
        const block_q8_0* w2 = (const block_q8_0*) w2v;
        const int b = lane >> 2, j0 = (lane & 3) * 8;
        for (int v = blockIdx.x * wpb + warp; v < n_vocab; v += gridDim.x * wpb) {
            const block_q8_0& B = w2[(size_t) v * (RANK / 32) + b];
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) acc += (float) B.qs[j0 + j] * s_m[b * 32 + j0 + j];
            acc = warp_sum(acc * __half2float(B.d));
            if (lane == 0) out[v] = logits[v] + acc;
        }
    }
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "dsv4_fast %s: %s\n", what, cudaGetErrorString(e));
}

}  // namespace

Rope make_rope(int n_rot, float base, float freq_scale, float ext_factor, int n_ctx_orig, float beta_fast,
               float beta_slow) {
    Rope r;
    r.theta_scale = powf(base, -2.0f / (float) n_rot);
    r.freq_scale = freq_scale;
    r.ext_factor = ext_factor;
    r.attn_factor = ext_factor == 0.0f ? 1.0f : 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));
    if (ext_factor != 0.0f && n_ctx_orig > 0) {
        // ggml_rope_yarn_corr_dims
        const auto corr_dim = [&](float b) {
            // (pi spelled out: MSVC's <cmath> has no M_PI without _USE_MATH_DEFINES - issue #2)
            return (float) n_rot * logf((float) n_ctx_orig / (b * 2.0f * 3.14159265358979323846f)) / (2.0f * logf(base));
        };
        r.corr_lo = std::max(0.0f, floorf(corr_dim(beta_fast)));
        r.corr_hi = std::min((float) (n_rot - 1), ceilf(corr_dim(beta_slow)));
    }
    return r;
}

void prep(const PrepArgs& a, cudaStream_t s) {
    prep_kernel<<<2 + a.n_comp, 512, 0, s>>>(a);
    check("prep");
}

void compress(const PrepArgs& a, cudaStream_t s) {
    // only the compressors whose block ends at p (the caller passes those first in a.comp)
    if (a.n_comp <= 0) return;
    compress_kernel<<<a.n_comp, 512, 0, s>>>(a);
    check("compress");
}

void q_prep(float* q, int n_head, int head_dim, int n_rot, const Rope& rope, float* iq, int idx_heads, int idx_dim,
            const Rope& comp_rope, int p, float eps, cudaStream_t s) {
    const int blocks = n_head + (iq != nullptr ? idx_heads : 0);
    q_prep_kernel<<<blocks, 128, 0, s>>>(q, n_head, head_dim, n_rot, rope, iq, idx_dim, comp_rope, p, eps);
    check("q_prep");
}

// the tensor-core indexer kernels: sm_80+ in a CUDA build, 64 heads, unless STRATA_DS4_IDX=1. The first call on a
// device also lets the batch kernel take its 64 KB of shared memory there.
static bool idx_use_mma(int idx_heads) {
    static const bool warp_only = [] { const char* v = getenv("STRATA_DS4_IDX"); return v && std::atoi(v) == 1; }();
#if !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
    if (warp_only || idx_heads != 64) return false;
    static int state[64] = {};   // per device: 0 unknown, 1 yes, -1 no
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    if (state[dev] == 0) {
        int major = 0;
        const bool ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        major >= 8 &&
                        cudaFuncSetAttribute(idx_score_batch_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             IBM_SMEM) == cudaSuccess;
        state[dev] = ok ? 1 : -1;
    }
    return state[dev] == 1;
#else
    (void) warp_only;
    (void) idx_heads;
    return false;
#endif
}
#if defined(STRATA_USE_HIP)
// HIP: the WMMA kernels on gfx11 / gfx12 (RDNA3 / 3.5 / 4; gfx1103 has no WMMA), 64 heads, unless STRATA_DS4_IDX=1
static bool hip_wmma_arch();
static bool idx_use_wmma(int idx_heads) {
    static const bool warp_only = [] { const char* v = getenv("STRATA_DS4_IDX"); return v && std::atoi(v) == 1; }();
    return !warp_only && idx_heads == 64 && hip_wmma_arch();
}
// the current device has RDNA's wave32 WMMA (gfx11 / gfx12; gfx1103 has none)
static bool hip_wmma_arch() {
    static int state[64] = {};   // per device: 0 unknown, 1 yes, -1 no
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) return false;
    if (state[dev] == 0) {
        hipDeviceProp_t p{};
        bool ok = hipGetDeviceProperties(&p, dev) == hipSuccess;
        if (ok) {
            const std::string a = p.gcnArchName;
            const std::string base = a.substr(0, a.find(':'));
            ok = base == "gfx1100" || base == "gfx1101" || base == "gfx1102" || base == "gfx1150" || base == "gfx1151" ||
                 base == "gfx1200" || base == "gfx1201";
        }
        state[dev] = ok ? 1 : -1;
    }
    return state[dev] == 1;
}
#endif
void indexer(const float* iq, const uint16_t* K, const float* w, int idx_heads, int idx_dim, int n_vis, int top_k,
             float* score, int* sel, cudaStream_t s) {
    if (n_vis <= 0) return;
    (void) idx_dim;   // 128 (the kernels' layout)
#if defined(STRATA_USE_HIP)
    if (idx_use_wmma(idx_heads)) {
        idx_score_wmma_kernel<<<(n_vis + IW_WAVES * 16 - 1) / (IW_WAVES * 16), IW_WAVES * 32, 0, s>>>(iq, K, w, n_vis,
                                                                                                     score);
        check("idx_score_wmma");
        idx_select_kernel<<<1, 1024, 0, s>>>(score, n_vis, std::min(top_k, n_vis), sel);
        check("idx_select");
        return;
    }
#endif
    if (idx_use_mma(idx_heads)) {
        idx_score_mma_kernel<<<(n_vis + IDXM_WARPS * 16 - 1) / (IDXM_WARPS * 16), IDXM_WARPS * 32, 0, s>>>(iq, K, w, n_vis,
                                                                                                         score);
        check("idx_score_mma");
        idx_select_kernel<<<1, 1024, 0, s>>>(score, n_vis, std::min(top_k, n_vis), sel);
        check("idx_select");
        return;
    }
    const size_t sm = (size_t) (idx_heads * 128 + idx_heads) * sizeof(float);
    idx_score_kernel<<<(n_vis + IDX_WARPS - 1) / IDX_WARPS, IDX_WARPS * 32, sm, s>>>(iq, K, w, idx_heads, n_vis, score);
    check("idx_score");
    idx_select_kernel<<<1, 1024, 0, s>>>(score, n_vis, std::min(top_k, n_vis), sel);
    check("idx_select");
}

void indexer_scores(const float* iq, const uint16_t* K, const float* w, int idx_heads, int n_vis, float* score,
                    cudaStream_t s) {
    if (n_vis <= 0) return;
    if (idx_use_mma(idx_heads)) {
        idx_score_mma_kernel<<<(n_vis + IDXM_WARPS * 16 - 1) / (IDXM_WARPS * 16), IDXM_WARPS * 32, 0, s>>>(iq, K, w, n_vis,
                                                                                                         score);
        check("idx_score_mma");
        return;
    }
    const size_t sm = (size_t) (idx_heads * 128 + idx_heads) * sizeof(float);
    idx_score_kernel<<<(n_vis + IDX_WARPS - 1) / IDX_WARPS, IDX_WARPS * 32, sm, s>>>(iq, K, w, idx_heads, n_vis, score);
    check("idx_score");
}

void indexer_select(const float* score, int n_vis, int top_k, int* sel, cudaStream_t s) {
    if (n_vis <= 0) return;
    idx_select_kernel<<<1, 1024, 0, s>>>(score, n_vis, std::min(top_k, n_vis), sel);
    check("idx_select");
}

void attend(const float* q, const uint16_t* raw, int ring, int n_raw, const uint16_t* comp, const int* comp_sel,
            int n_comp, const float* sinks, int n_head, int head_dim, int n_rot, const Rope& rope, int p, float* out,
            cudaStream_t s, int lane, void* oq) {
    const int n = n_raw + n_comp;
    const float scale = 1.0f / sqrtf((float) head_dim);
    // STRATA_DS4_OQ_FUSE (default 1): out's q8_1 written by the combine kernel's epilogue; 0: a separate launch
    static const bool oq_fuse = [] { const char* v = getenv("STRATA_DS4_OQ_FUSE"); return v == nullptr || std::atoi(v) != 0; }();
    const auto oq_separate = [&] {
        if (oq == nullptr) return;
        const int no = n_head * head_dim;
        oq_kernel<<<(no + 255) / 256, 256, 0, s>>>(out, (block_q8_1*) oq, no);
        check("oq");
    };
    // STRATA_DS4_ATTN=1: the one-block-a-head kernel (A/B); STRATA_DS4_ATTN_CHECK=1: both, compared every 997th call
    static const int mode = [] { const char* v = getenv("STRATA_DS4_ATTN"); return v ? std::atoi(v) : 0; }();
    static const bool chk = getenv("STRATA_DS4_ATTN_CHECK") != nullptr;
    const auto old = [&](float* o) {
        static bool attr = false;
        if (!attr) {
            cudaFuncSetAttribute(attend_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
            attr = true;
        }
        attend_kernel<<<n_head, ATT_THREADS, (size_t) n * sizeof(float), s>>>(q, raw, ring, n_raw, comp, comp_sel,
                                                                            n_comp, sinks, n_rot, rope, p, scale, o);
        check("attend");
    };
    if (mode == 1 || head_dim != 512 || n_head % AS_H != 0) { old(out); oq_separate(); return; }
    // the partials: [n_head][S][2 + 512] floats, per device, grown when a longer context needs more chunks
    // the chunk: about two blocks an SM (n_head / 8 x S blocks), in whole tiles, 16..256 keys
    static const int nsm = [] {
        int d = 0, v = 0;
        cudaGetDevice(&d);
        cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, d);
        return v > 0 ? v : 80;
    }();
    const int hb = n_head / AS_H;
    int ck = (n * hb + 2 * nsm - 1) / (2 * nsm);
    ck = std::min(256, std::max(AS_TILE, (ck + AS_TILE - 1) / AS_TILE * AS_TILE));
    const int S = (n + ck - 1) / ck;
    static float* parts[16][8] = {};   // [device][lane]: a verify window's columns (8) - a column past them took the
    static size_t parts_n[16][8] = {};  // one-block-a-head kernel, ~56 us instead of ~26
    int dev = 0;
    cudaGetDevice(&dev);
    const size_t need = (size_t) n_head * std::max(S, 64) * (head_dim + 2);
    if (dev < 0 || dev >= 16 || lane < 0 || lane >= 8) { old(out); oq_separate(); return; }
    float*& pt = parts[dev][lane];
    size_t& pn = parts_n[dev][lane];
    if (pn < need) {
        if (pt) cudaFree(pt);
        if (cudaMalloc(&pt, need * sizeof(float)) != cudaSuccess) { pt = nullptr; pn = 0; old(out); oq_separate(); return; }
        pn = need;
    }
    attend_split_kernel<<<dim3((unsigned) hb, (unsigned) S), 256, 0, s>>>(q, raw, ring, n_raw, comp,
                                                                                      comp_sel, n_comp, p, scale, ck, S,
                                                                                      pt);
    check("attend_split");
    attend_combine_kernel<<<n_head, 256, 0, s>>>(pt, S, sinks, n_rot, rope, p, out,
                                                  oq_fuse ? (block_q8_1*) oq : nullptr);
    check("attend_combine");
    if (!oq_fuse) oq_separate();
    if (chk) {
        static long calls = 0;
        if (calls++ % 997 == 0) {
            static float* ref = nullptr;
            const size_t no = (size_t) n_head * head_dim;
            if (!ref) cudaMalloc(&ref, 64 * 1024 * sizeof(float));
            if (no > 64 * 1024) return;
            old(ref);
            std::vector<float> a(no), b(no);
            cudaMemcpyAsync(a.data(), out, no * sizeof(float), cudaMemcpyDeviceToHost, s);
            cudaMemcpyAsync(b.data(), ref, no * sizeof(float), cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
            double dn = 0, dd = 0, mx = 0;
            for (size_t i = 0; i < no; ++i) {
                const double d = std::fabs((double) a[i] - b[i]);
                dn += d; dd += std::fabs((double) b[i]); mx = std::max(mx, d);
            }
            std::fprintf(stderr, "dsv4 attend check: p %d n %d (raw %d comp %d) S %d: rel %.2e, max abs %.2e\n", p, n,
                         n_raw, n_comp, S, dn / (dd + 1e-30), mx);
        }
    }
}

void head_prep(const float* R, const uint16_t* fn, const float* scale, const float* base, float hc_eps,
               const float* norm_w, float eps, int n_embd, float* x, void* xq, cudaStream_t s) {
    head_prep_kernel<<<1, 1024, 0, s>>>(R, fn, scale, base, hc_eps, norm_w, eps, n_embd, x, (block_q8_1*) xq);
    check("head_prep");
}


void win_load(const uint16_t* ring, int n_ring, uint16_t* keys, int n_hist, int p0, int D, cudaStream_t s) {
    if (n_hist <= 0) return;
    win_load_kernel<<<n_hist, 256, 0, s>>>(ring, n_ring, keys, n_hist, p0, D);
    check("win_load");
}
void win_store(uint16_t* ring, int n_ring, const uint16_t* keys, int n_hist, int p0, int T, int D, cudaStream_t s) {
    const int n_total = n_hist + T, n_keep = std::min(n_ring, n_total);
    if (n_keep <= 0) return;
    win_store_kernel<<<n_keep, 256, 0, s>>>(ring, n_ring, keys, n_total, n_keep, p0 - n_hist, D);
    check("win_store");
}
void prep_batch(const BatchPrepArgs& a, cudaStream_t s) {
    if (a.T <= 0) return;
    prep_batch_kernel<<<dim3((unsigned) a.T, 2), 512, 0, s>>>(a);
    check("prep_batch");
}
void compress_batch(const BatchComp* comps, int n_comp, int p0, int T, int n_rot, const Rope& comp_rope, float eps,
                    cudaStream_t s) {
    if (n_comp <= 0 || T <= 0) return;
    BatchCompArgs a{};
    int max_blocks = 0, max_state = 0;
    for (int i = 0; i < n_comp; ++i) {
        a.c[i] = comps[i];
        const int R = comps[i].c.ratio;
        max_blocks = std::max(max_blocks, (p0 + T) / R - ((p0 + R) / R - 1));
        max_state = std::max(max_state, (comps[i].c.overlap ? 2 : 1) * R);
    }
    a.n_comp = n_comp;
    a.p0 = p0;
    a.T = T;
    a.n_rot = n_rot;
    a.rope = comp_rope;
    a.eps = eps;
    if (max_blocks > 0) {
        compress_batch_kernel<<<dim3((unsigned) max_blocks, (unsigned) n_comp), 512, 0, s>>>(a);
        check("compress_batch");
    }
    comp_ring_kernel<<<dim3((unsigned) std::min(T, max_state), (unsigned) n_comp), 256, 0, s>>>(a);
    check("comp_ring");
}
void q_prep_batch(float* q, int n_head, int head_dim, int n_rot, const Rope& rope, float* iq, int idx_heads,
                  int idx_dim, const Rope& comp_rope, int p0, int T, float eps, cudaStream_t s) {
    if (T <= 0) return;
    const int hy = n_head + (iq != nullptr ? idx_heads : 0);
    q_prep_batch_kernel<<<dim3((unsigned) T, (unsigned) hy), 128, 0, s>>>(q, n_head, head_dim, n_rot, rope, iq, idx_heads,
                                                                        idx_dim, comp_rope, p0, eps);
    check("q_prep_batch");
}
void indexer_batch(float* iq, const uint16_t* K, const float* w, int idx_heads, int ratio, int p0, int T, int top_k,
                   float* score, int ld, int* sel, cudaStream_t s) {
    const int max_vis = (p0 + T) / ratio;
    if (T <= 0 || max_vis <= 0) return;
#if defined(STRATA_USE_HIP)
    if (idx_use_wmma(idx_heads)) {
        iwb_prep_kernel<<<(unsigned) ((T * 64 + 7) / 8), 256, 0, s>>>(iq, T * 64);
        check("iwb_prep");
        const dim3 grid((unsigned) ((max_vis + IWB2_ROWS - 1) / IWB2_ROWS), (unsigned) ((T + IWB2_TT - 1) / IWB2_TT));
        idx_score_batch_wmma2_kernel<64><<<grid, IWB2_WAVES * 32, 0, s>>>(iq, K, w, ratio, p0, T, score, ld);
        check("idx_score_batch_wmma");
        idx_select_batch_kernel<<<T, 1024, 0, s>>>(score, ld, ratio, p0, top_k, sel, top_k);
        check("idx_select_batch");
        return;
    }
#endif
    if (idx_use_mma(idx_heads)) {
        const dim3 grid((unsigned) ((max_vis + IBM_ROWS - 1) / IBM_ROWS), (unsigned) ((T + IBM_TT - 1) / IBM_TT));
        idx_score_batch_mma_kernel<<<grid, IBM_WARPS * 32, IBM_SMEM, s>>>(iq, K, w, ratio, p0, T, score, ld);
        check("idx_score_batch_mma");
    } else {
        const dim3 grid((unsigned) ((max_vis + IRT - 1) / IRT), (unsigned) ((T + IQT - 1) / IQT));
        idx_score_batch_kernel<<<grid, 256, 0, s>>>(iq, K, w, idx_heads, ratio, p0, T, score, ld);
        check("idx_score_batch");
    }
    idx_select_batch_kernel<<<T, 1024, 0, s>>>(score, ld, ratio, p0, top_k, sel, top_k);
    check("idx_select_batch");
}
void indexer_batch_scores(float* iq, const uint16_t* K, const float* w, int idx_heads, int ratio, int p0, int T,
                          float* score, int ld, cudaStream_t s) {
    const int max_vis = (p0 + T) / ratio;
    if (T <= 0 || max_vis <= 0) return;
#if defined(STRATA_USE_HIP)
    // deepseek41's 32 heads on WMMA too (v1.3.63; iq is converted in place, as indexer_batch does: these rows' last
    // reader).  (The CUDA mma kernel is 64 heads: the plain one below.)
    if (idx_heads == 32 && idx_use_wmma(64)) {
        iwb_prep_kernel<<<(unsigned) ((T * 32 + 7) / 8), 256, 0, s>>>(iq, T * 32);
        check("iwb_prep");
        const dim3 grid((unsigned) ((max_vis + IWB2_ROWS - 1) / IWB2_ROWS), (unsigned) ((T + IWB2_TT - 1) / IWB2_TT));
        idx_score_batch_wmma2_kernel<32><<<grid, IWB2_WAVES * 32, 0, s>>>(iq, K, w, ratio, p0, T, score, ld);
        check("idx_score_batch_wmma");
        return;
    }
#endif
    const dim3 grid((unsigned) ((max_vis + IRT - 1) / IRT), (unsigned) ((T + IQT - 1) / IQT));
    idx_score_batch_kernel<<<grid, 256, 0, s>>>(iq, K, w, idx_heads, ratio, p0, T, score, ld);
    check("idx_score_batch");
}
void indexer_batch_select(const float* score, int ld, int ratio, int p0, int T, int top_k, int* sel, int sel_ld,
                          cudaStream_t s) {
    if (T <= 0) return;
    idx_select_batch_kernel<<<T, 1024, 0, s>>>(score, ld, ratio, p0, top_k, sel, sel_ld);
    check("idx_select_batch");
}
// the prompt path's tensor-core attention: sm_80+ in a CUDA build, head_dim 512, 16 | n_head, unless STRATA_DS4_ATTN=1
static bool attn_use_mma(int n_head, int head_dim) {
    static const bool old = [] { const char* v = getenv("STRATA_DS4_ATTN"); return v && std::atoi(v) == 1; }();
#if !defined(__HIPCC__) && !defined(STRATA_USE_HIP)
    if (old || head_dim != 512 || n_head % ABM_H != 0) return false;
    static int state[64] = {};   // per device: 0 unknown, 1 yes, -1 no
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    if (state[dev] == 0) {
        int major = 0;
        state[dev] = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && major >= 8 ? 1 : -1;
    }
    return state[dev] == 1;
#else
    (void) old;
    (void) n_head;
    (void) head_dim;
    return false;
#endif
}
void attend_batch(const float* q, const uint16_t* keys, int n_hist, int n_swa, const uint16_t* comp, const int* sel,
                  int top_k, int ratio, const float* sinks, int n_head, int head_dim, int n_rot, const Rope& rope, int p0,
                  int T, float* out, cudaStream_t s, int sel_ld) {
    if (T <= 0) return;
    if (sel_ld <= 0) sel_ld = top_k;
#if defined(STRATA_USE_HIP)
    static const bool old_attn = [] { const char* v = getenv("STRATA_DS4_ATTN"); return v && std::atoi(v) == 1; }();
    if (!old_attn && head_dim == 512 && n_head % AW_H == 0 && n_rot <= 512 && hip_wmma_arch()) {
        attend_batch_wmma_kernel<<<dim3((unsigned) T, (unsigned) (n_head / AW_H)), 256, 0, s>>>(
            q, keys, n_hist, n_swa, comp, sel, top_k, sel_ld, ratio, sinks, n_head, n_rot, rope, p0, 1.0f / sqrtf((float) head_dim),
            out);
        check("attend_batch_wmma");
        return;
    }
#endif
    if (attn_use_mma(n_head, head_dim)) {
        attend_batch_mma_kernel<<<dim3((unsigned) T, (unsigned) (n_head / ABM_H)), 128, 0, s>>>(
            q, keys, n_hist, n_swa, comp, sel, top_k, sel_ld, ratio, sinks, n_head, n_rot, rope, p0, 1.0f / sqrtf((float) head_dim),
            out);
        check("attend_batch_mma");
        return;
    }
    attend_batch_kernel<<<dim3((unsigned) T, (unsigned) (n_head / AB_H)), 256, 0, s>>>(
        q, keys, n_hist, n_swa, comp, sel, top_k, sel_ld, ratio, sinks, n_head, n_rot, rope, p0, 1.0f / sqrtf((float) head_dim),
        out);
    check("attend_batch");
}
void to_f16(const float* x, uint16_t* y, int64_t n, cudaStream_t s) {
    if (n <= 0) return;
    to_f16_kernel<<<(unsigned) std::min<int64_t>(4096, (n + 255) / 256), 256, 0, s>>>(x, y, n);
    check("to_f16");
}


// ---------------------------------------------------------------- DSpark
void kv_rows(const float* kv_raw, int n, const float* kv_norm, int head_dim, int n_rot, const Rope& rope, int p0,
             uint16_t* ring, int n_ring, float eps, cudaStream_t s) {
    if (n <= 0) return;
    kv_rows_kernel<<<n, 512, 0, s>>>(kv_raw, kv_norm, head_dim, n_rot, rope, p0, ring, n_ring, eps);
    check("kv_rows");
}
void stream_mean(const float* R, int n, size_t rs, int n_embd, float* out, size_t os, cudaStream_t s) {
    if (n <= 0) return;
    stream_mean_kernel<<<dim3((unsigned) ((n_embd + 255) / 256), (unsigned) n), 256, 0, s>>>(R, rs, n_embd, out, os);
    check("stream_mean");
}
bool dspark_route(const float* logits, const float* bias, int n_expert, int k, float w_scale,
                  const unsigned long long* tab, int nr, unsigned long long* plan_ptr, float* plan_w, int* ids,
                  cudaStream_t s) {
    if (n_expert > 512 || k > 8 || nr <= 0) return false;
    dspark_route_kernel<<<nr, 256, 0, s>>>(logits, bias, n_expert, k, w_scale, tab, plan_ptr, plan_w, ids);
    check("dspark_route");
    return true;
}
void dspark_head(const float* R, const uint16_t* fn, const float* scale, const float* base, float hc_eps,
                 const float* norm_w, float eps, int n_embd, float* h, float* x, void* xq, cudaStream_t s) {
    dspark_head_kernel<<<1, 1024, 0, s>>>(R, fn, scale, base, hc_eps, norm_w, eps, n_embd, h, x, (block_q8_1*) xq);
    check("dspark_head");
}
void dspark_markov(const float* logits, const void* w2, const uint16_t* w1, int n_vocab, const int* prev,
                   const float* cp, const float* h, int n_embd, float* out, float* conf, cudaStream_t s) {
    dspark_markov_kernel<false><<<1024, 256, 0, s>>>(logits, w2, (const __half*) w1, n_vocab, prev, cp, nullptr, h,
                                                     n_embd, out, conf);
    check("dspark_markov");
}
void dspark_markov_f16(const float* logits, const uint16_t* w2, const uint16_t* w1, int n_vocab, const int* prev,
                       const float* cp, const float* cb, const float* h, int n_embd, float* out, float* conf,
                       cudaStream_t s) {
    dspark_markov_kernel<true><<<1024, 256, 0, s>>>(logits, w2, (const __half*) w1, n_vocab, prev, cp, cb, h, n_embd,
                                                    out, conf);
    check("dspark_markov_f16");
}

}  // namespace strata::kernels::dsv4
