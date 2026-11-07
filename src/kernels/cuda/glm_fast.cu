// src/kernels/cuda/glm_fast.cu - see include/strata/kernels/glm_fast.hpp.
//
// The dot products are llama.cpp's vecdotq (ggml/src/ggml-cuda/vecdotq.cuh, MIT, third_party/ggml/LICENSE),
// the same expressions iq_kernels.cu and native_mmvq.cu transcribe; the block structs and codebooks come
// from ggml-common.h.  Every reduction below is parallel, so the summation ORDER differs from the
// serial reference kernels - the values agree to float rounding, which the fast-vs-reference check in
// glm_pack_test pins on the real model.
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/exl3.hpp"
#if defined(STRATA_USE_HIP)
#include "strata/kernels/glm_expert_bench.hpp"
#include <cstring>
#endif

#if !defined(STRATA_USE_HIP)
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include "dsa_topk.cuh"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <map>
#include <mutex>
#include <cstdlib>

#if defined(STRATA_USE_HIP)
// The HC grid exchanges FP32 partials between workgroups. An agent-scope
// acquire load preserves visibility after their publication fences; HIP only
// supplies CUDA's __ldcg spelling for half types.
__device__ __forceinline__ float __ldcg(const float* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
#endif

namespace strata::kernels::glmf {
namespace {

// ---------------------------------------------------------------- small helpers
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
// block-wide sum; every thread gets the total.  sh: >= 32 floats.  blockDim % 32 == 0.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += sh[i];
    return t;
}
__device__ __forceinline__ float block_max(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = -INFINITY;
    for (int i = 0; i < nw; ++i) t = fmaxf(t, sh[i]);
    return t;
}
// one q8_1 block from 32 values held one per lane (quantize.cu's arithmetic)
__device__ __forceinline__ void q8_1_store_warp(float v, block_q8_1* dst, int lane) {
    const float amax = warp_max(fabsf(v));
    const float sum = warp_sum(v);
    // the scale and the sum stay finite in FP16 (Strata #1448: an activation past ~8.3M overflowed the scale to inf,
    // and the dot products to NaN); below that nothing changes
    const float d = fminf(amax / 127.0f, 65504.0f);
    const int8_t q = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(v / d)));
    dst->qs[lane] = q;
    if (lane == 0) dst->ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}
__device__ __forceinline__ float dsigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t) v << 16); }

// ---------------------------------------------------------------- llama.cpp vecdotq helpers
__device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__device__ __forceinline__ int get_int_b4(const void* x, const int& i32) { return ((const int*) x)[i32]; }
__device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}
__device__ __forceinline__ int2 get_int_from_table_16(const int& q4, const int8_t* table) {
    const uint32_t* table32 = (const uint32_t*) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// --- K quants (bq8_1 = the q8_1 blocks of this weight block's 256 values)
__device__ __forceinline__ float dot_q4_K(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q4_K* bq4 = (const block_q4_K*) vbq + kbx;
    int v[2], u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = (const int*) (bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];
    const uint16_t* scales = (const uint16_t*) bq4->scales;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm], s2 = scales[jm + 2], s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(bq4->dm);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float dot_q5_K(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q5_K* bq5 = (const block_q5_K*) vbq + kbx;
    int vl[2], vh[2], u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = (const int*) (bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = (const int*) (bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    const uint16_t* scales = (const uint16_t*) bq5->scales;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm], s2 = scales[jm + 2], s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = __dp4a(v0i, u[2 * i], __dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i], __dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(bq5->dm);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

__device__ __forceinline__ float dot_q6_K(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q6_K* w = (const block_q6_K*) vbq + kbx;
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = get_int_b2(w->ql, iqs);
    const int vh = get_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    const int8_t* scales = w->scales + scale_offset;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int u = ((const int*) bq8_1[bq8_offset + 2 * i].qs)[iqs % 8];
        const float d8 = __low2float(bq8_1[bq8_offset + 2 * i].ds);
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8 * (__dp4a(vi, u, 0) * sc);
    }
    return __half2float(w->d) * sumf;
}

__device__ __forceinline__ float dot_q8_0(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q8_0* w = (const block_q8_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = get_int_b2(w->qs, iqs + i);
        const int u = ((const int*) bq8_1->qs)[iqs + i];
        sumi = __dp4a(v, u, sumi);
    }
    return __half2float(w->d) * __low2float(bq8_1->ds) * float(sumi);
}

// --- i-quants (iq_kernels.cu's transcriptions)
// --- Q2_K / Q3_K (the NextN block's experts): llama.cpp's vec_dot_q2_K_q8_1 / vec_dot_q3_K_q8_1, VDR 1 - 16 calls per
// 256-block, call iqs reading int (iqs % 8) of the 4 q8_1 blocks 4 * (iqs / 8) .. + 3
__device__ __forceinline__ float dot_q2_K(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q2_K* b = (const block_q2_K*) vbq + kbx;
    const int bq8_offset = 4 * (iqs / 8);
    const uint8_t* scales = b->scales + (iqs - iqs % 8 + (iqs % 8) / 4);
    const int v = get_int_b4(b->qs, iqs);
    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const block_q8_1* q8 = bq8_1 + bq8_offset + i;
        const int u = get_int_b4(q8->qs, iqs % 8);
        const float d8 = __low2float(q8->ds);
        const int sc = scales[2 * i];
        const int vi = (v >> (2 * i)) & 0x03030303;
        sumf_d += d8 * (__dp4a(vi, u, 0) * (sc & 0xF));
        int m = sc >> 4;
        m |= m << 8;
        m |= m << 16;
        sumf_m += d8 * __dp4a(m, u, 0);
    }
    const float2 dm = __half22float2(b->dm);
    return dm.x * sumf_d - dm.y * sumf_m;
}
__device__ __forceinline__ float dot_q3_K(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_q3_K* b = (const block_q3_K*) vbq + kbx;
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const float d = __half2float(b->d);
    const int vl = get_int_b2(b->qs, iqs);
    const int vh = ~get_int_b2(b->hmask, iqs % 8) >> bq8_offset;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const block_q8_1* q8 = bq8_1 + bq8_offset + i;
        const int u = get_int_b4(q8->qs, iqs % 8);
        const float d8 = __low2float(q8->ds);
        const int isc = scale_offset + 2 * i;
        const int sc_low = (b->scales[isc % 8] >> (4 * (isc / 8))) & 0xF;
        const int sc_high = ((b->scales[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        sumf += d8 * (__dp4a((int) __vsubss4(vil, vih), u, 0) * sc);
    }
    return d * sumf;
}
__device__ __forceinline__ float dot_iq2_xxs(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = __dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = __dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}
__device__ __forceinline__ float dot_iq3_xxs(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = __dp4a(grid_l, u0, sumi);
        sumi = __dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}
__device__ __forceinline__ float dot_iq1_s(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_iq1_s* bq1 = (const block_iq1_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq1->qh[iqs];
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi = __dp4a(grid0, u0, sumi);
        sumi = __dp4a(grid1, u1, sumi);
    }
    const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    const float2 ds = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x * sumi + ds.y * delta);
}
__device__ __forceinline__ float dot_iq4_xs(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = __dp4a(v.x, u0, sumi);
        sumi = __dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs / 4].ds);
    return d * sumi;
}

// qk = values per block, ipb = dot calls per block, step = iqs stride, bytes = block bytes
template<int T> struct F;
template<> struct F<12> { static constexpr int qk = 256, ipb = 16, step = 2, bytes = sizeof(block_q4_K);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q4_K(v, y, kbx, iqs); } };
template<> struct F<13> { static constexpr int qk = 256, ipb = 16, step = 2, bytes = sizeof(block_q5_K);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q5_K(v, y, kbx, iqs); } };
template<> struct F<14> { static constexpr int qk = 256, ipb = 32, step = 1, bytes = sizeof(block_q6_K);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q6_K(v, y, kbx, iqs); } };
template<> struct F<8> { static constexpr int qk = 32, ipb = 4, step = 2, bytes = sizeof(block_q8_0);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q8_0(v, y, kbx, iqs); } };
template<> struct F<16> { static constexpr int qk = 256, ipb = 8, step = 2, bytes = sizeof(block_iq2_xxs);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_iq2_xxs(v, y, kbx, iqs); } };
template<> struct F<18> { static constexpr int qk = 256, ipb = 8, step = 2, bytes = sizeof(block_iq3_xxs);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_iq3_xxs(v, y, kbx, iqs); } };
template<> struct F<19> { static constexpr int qk = 256, ipb = 8, step = 1, bytes = sizeof(block_iq1_s);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_iq1_s(v, y, kbx, iqs); } };
template<> struct F<10> { static constexpr int qk = 256, ipb = 16, step = 1, bytes = sizeof(block_q2_K);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q2_K(v, y, kbx, iqs); } };
template<> struct F<11> { static constexpr int qk = 256, ipb = 16, step = 1, bytes = sizeof(block_q3_K);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_q3_K(v, y, kbx, iqs); } };
template<> struct F<23> { static constexpr int qk = 256, ipb = 8, step = 4, bytes = sizeof(block_iq4_xs);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return dot_iq4_xs(v, y, kbx, iqs); } };

// one row against one q8_1 activation, the whole warp (lane-strided calls)
template<int T>
__device__ __forceinline__ float row_dot(const uint8_t* row, const block_q8_1* x, int n_in, int lane) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
    float s = 0.0f;
    for (int k = lane; k < nb * Fm::ipb; k += 32) {
        const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
        s += Fm::dot(row, x + kbx * (Fm::qk / 32), kbx, iqs);
    }
    return warp_sum(s);
}
__device__ __forceinline__ float row_dot_f32(const float* row, const float* x, int n_in, int lane) {
    float s = 0.0f;
    const float4* r4 = (const float4*) row;
    const float4* x4 = (const float4*) x;
    for (int k = lane; k < n_in / 4; k += 32) {
        const float4 a = r4[k], b = x4[k];
        s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    return warp_sum(s);
}
__device__ __forceinline__ float row_dot_bf16(const uint16_t* row, const float* x, int n_in, int lane) {
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

// Q8_0 weights read as floats (STRATA_GLM_DENSE_Q8: the GGUF's own 34-byte blocks - an FP16 scale and 32 int8 - in
// place of the pack's BF16 expansion, half the bytes; rows of whole blocks, so value i of a run is block i / 32's)
__device__ __forceinline__ float q8_0_at(const uint8_t* W, size_t i) {
    const uint8_t* b = W + (i >> 5) * 34;
    return __half2float(*(const __half*) b) * (float) (int8_t) b[2 + (i & 31)];
}
// values 4 * i4 .. 4 * i4 + 3 (one block's quarter of a quarter: never across blocks); the codes are 2-byte aligned
__device__ __forceinline__ float4 q8_0_at4(const uint8_t* W, size_t i4) {
    const uint8_t* b = W + (i4 >> 3) * 34;
    const float d = __half2float(*(const __half*) b);
    const uint16_t* q = (const uint16_t*) (b + 2) + (i4 & 7) * 2;
    const uint32_t v = (uint32_t) q[0] | ((uint32_t) q[1] << 16);
    return make_float4(d * (float) (int8_t) (v & 0xffu), d * (float) (int8_t) ((v >> 8) & 0xffu),
                       d * (float) (int8_t) ((v >> 16) & 0xffu), d * (float) (int8_t) (v >> 24));
}
// a Q8_0 row (n_in % 32 == 0) . x (F32, 16-byte aligned)
__device__ __forceinline__ float row_dot_q8_0f(const uint8_t* row, const float* x, int n_in, int lane) {
    float s = 0.0f;
    const float4* x4 = (const float4*) x;
    for (int k = lane; k < n_in / 4; k += 32) {
        const float4 w = q8_0_at4(row, (size_t) k), a = x4[k];
        s += w.x * a.x + w.y * a.y + w.z * a.z + w.w * a.w;
    }
    return warp_sum(s);
}

template<int T>
__device__ __forceinline__ size_t rbytes(int n_in) { return (size_t) (n_in / F<T>::qk) * F<T>::bytes; }

// ---- the same i-quant dots with the codebook in SHARED memory (the expert kernels): the grid lookups are
// scattered across the 32 lanes, and from global memory every one of them replays through L1
__device__ __forceinline__ float dot_iq1_s_t(const void* vbq, const block_q8_1* bq8_1, const int& kbx, const int& iqs,
                                             const uint32_t* tab) {
    const block_iq1_s* bq1 = (const block_iq1_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq1->qh[iqs];
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = tab[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi = __dp4a(grid0, u0, sumi);
        sumi = __dp4a(grid1, u1, sumi);
    }
    const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    const float2 ds = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x * sumi + ds.y * delta);
}
__device__ __forceinline__ float dot_iq2_xxs_t(const void* vbq, const block_q8_1* bq8_1, const int& kbx,
                                               const int& iqs, const uint32_t* tab) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = __dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = __dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}
__device__ __forceinline__ float dot_iq3_xxs_t(const void* vbq, const block_q8_1* bq8_1, const int& kbx,
                                               const int& iqs, const uint32_t* tab) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(tab[q3[l0 + 0]], tab[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = __dp4a(grid_l, u0, sumi);
        sumi = __dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}
template<int T> __device__ __forceinline__ float dot_t(const void* v, const block_q8_1* y, int kbx, int iqs,
                                                       const uint32_t* tab);
template<> __device__ __forceinline__ float dot_t<19>(const void* v, const block_q8_1* y, int kbx, int iqs, const uint32_t* tab) { return dot_iq1_s_t(v, y, kbx, iqs, tab); }
template<> __device__ __forceinline__ float dot_t<16>(const void* v, const block_q8_1* y, int kbx, int iqs, const uint32_t* tab) { return dot_iq2_xxs_t(v, y, kbx, iqs, tab); }
template<> __device__ __forceinline__ float dot_t<18>(const void* v, const block_q8_1* y, int kbx, int iqs, const uint32_t* tab) { return dot_iq3_xxs_t(v, y, kbx, iqs, tab); }
template<> __device__ __forceinline__ float dot_t<23>(const void* v, const block_q8_1* y, int kbx, int iqs, const uint32_t*) { return dot_iq4_xs(v, y, kbx, iqs); }
// u32 words of the type's codebook (0 = none needed)
template<int T> struct TabWords { static constexpr int n = 0; };
template<> struct TabWords<19> { static constexpr int n = 2048; };
template<> struct TabWords<16> { static constexpr int n = 512; };
template<> struct TabWords<18> { static constexpr int n = 256; };
template<> struct TabWords<17> { static constexpr int n = 1024; };
template<> struct TabWords<22> { static constexpr int n = 2048; };
template<> struct TabWords<21> { static constexpr int n = 512; };
template<> struct TabWords<29> { static constexpr int n = 2048; };
template<int T>
__device__ __forceinline__ void load_tab(uint32_t* s_tab) {
    const uint32_t* src = T == 19 || T == 29 ? (const uint32_t*) iq1s_grid_gpu
                          : T == 16 ? (const uint32_t*) iq2xxs_grid
                          : T == 18 ? (const uint32_t*) iq3xxs_grid
                          : T == 17 ? (const uint32_t*) iq2xs_grid
                          : T == 22 ? (const uint32_t*) iq2s_grid
                          : T == 21 ? (const uint32_t*) iq3s_grid : nullptr;
    for (int i = threadIdx.x; i < TabWords<T>::n; i += blockDim.x) s_tab[i] = src[i];
}
// ---- the activation half of a dot, loaded ONCE and reused across rows: every i-quant call k reads exactly one
// q8_1 block (index kbx * qk/32 + k % ipb) and all 8 of its ints
struct XV {
    int u[8];
    float d, s;
};
__device__ __forceinline__ XV load_xv(const block_q8_1* xb) {
    XV v;
    const int* q = (const int*) xb->qs;
#pragma unroll
    for (int i = 0; i < 8; ++i) v.u[i] = q[i];
    const float2 ds = __half22float2(xb->ds);
    v.d = ds.x;
    v.s = ds.y;
    return v;
}
template<int T> __device__ __forceinline__ float dw(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab);
template<> __device__ __forceinline__ float dw<19>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq1_s* bq1 = (const block_iq1_s*) row + kbx;
    const int qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq1->qh[iqs];
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = tab[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        sumi = __dp4a((grid >> 0) & 0x0F0F0F0F, x.u[l0 + 0], sumi);
        sumi = __dp4a((grid >> 4) & 0x0F0F0F0F, x.u[l0 + 1], sumi);
    }
    const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    return d1q * (x.d * sumi + x.s * delta);
}
template<> __device__ __forceinline__ float dw<16>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) row + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        sumi = __dp4a((int) __vsub4(grid_pos.x ^ signs0, signs0), x.u[k0 + 0], sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        sumi = __dp4a((int) __vsub4(grid_pos.y ^ signs1, signs1), x.u[k0 + 1], sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    return __half2float(bq2->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw<18>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) row + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(tab[q3[l0 + 0]], tab[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        sumi = __dp4a((int) __vsub4(grid_pos.x ^ signs0, signs0), x.u[l0 + 0], sumi);
        sumi = __dp4a((int) __vsub4(grid_pos.y ^ signs1, signs1), x.u[l0 + 1], sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    return __half2float(bq3->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw<23>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t*) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) row + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = __dp4a(v.x, x.u[j + 0], sumi);
        sumi = __dp4a(v.y, x.u[j + 4], sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    return __half2float(bq4->d) * x.d * sumi;
}
// ---- the other i-quants of the Maya quants (docs/maya/QUANT-PLAN.md): vecdotq's IQ2_XS, IQ2_S, IQ3_S and IQ1_M (the
// expressions iq_kernels.cu transcribes) in the same activation-preloaded, table-in-shared-memory form
template<> __device__ __forceinline__ float dw<17>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) row + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        if (l0 < 4) {
            sumi0 = __dp4a(grid_l, x.u[l0 + 0], sumi0);
            sumi0 = __dp4a(grid_h, x.u[l0 + 1], sumi0);
        } else {
            sumi1 = __dp4a(grid_l, x.u[l0 + 0], sumi1);
            sumi1 = __dp4a(grid_h, x.u[l0 + 1], sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    return __half2float(bq2->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw<22>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq2_s* bq2 = (const block_iq2_s*) row + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)];
        const int signs0 =
            __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 =
            __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        if (l0 < 4) {
            sumi0 = __dp4a(grid_l, x.u[l0 + 0], sumi0);
            sumi0 = __dp4a(grid_h, x.u[l0 + 1], sumi0);
        } else {
            sumi1 = __dp4a(grid_l, x.u[l0 + 0], sumi1);
            sumi1 = __dp4a(grid_h, x.u[l0 + 1], sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    return __half2float(bq2->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw<21>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq3_s* bq3 = (const block_iq3_s*) row + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(tab[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)], tab[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 =
            __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 =
            __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        sumi = __dp4a(grid_l, x.u[l0 + 0], sumi);
        sumi = __dp4a(grid_h, x.u[l0 + 1], sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    return __half2float(bq3->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw<29>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t* tab) {
    const block_iq1_m* bq1 = (const block_iq1_m*) row + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = tab[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        sumi[l0 / 4] = __dp4a((grid >> 0) & 0x0F0F0F0F, x.u[l0 + 0], sumi[l0 / 4]);
        sumi[l0 / 4] = __dp4a((grid >> 4) & 0x0F0F0F0F, x.u[l0 + 1], sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = __dp4a(x.u[l0 + 0], 0x01010101, sumy);
        sumy = __dp4a(x.u[l0 + 1], 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * x.d;
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}
// HIP sign decode adapted from Strata src/prefill/moe_fused_iq.cu (MIT).
// Every IQ2/IQ3 grid byte is nonzero and <= 62, so two's-complement negation
// can use a whole-word add without carries across byte boundaries. Multiplying
// a four-bit sign nibble by 0x10204080 puts its bits at each byte's high bit;
// after masking/shifting, these are the +1 corrections and *255 gives the masks.
// The integer scale divisions and floating-point accumulation order stay intact.
#if defined(STRATA_USE_HIP)
__device__ __forceinline__ uint4 direct_signs(unsigned b, bool parity) {
    b &= parity ? 127u : 255u;
    if (parity) b |= (__popc(b) & 1) << 7;
    const unsigned lo = (((b & 15u) * 0x10204080u) & 0x80808080u) >> 7;
    const unsigned hi = ((((b >> 4) & 15u) * 0x10204080u) & 0x80808080u) >> 7;
    return make_uint4(lo * 255u, lo, hi * 255u, hi);
}
__device__ __forceinline__ uint2 quant_load8(const void* p) {
    uint2 words;
    __builtin_memcpy(&words, p, sizeof(words)); // GGUF super-blocks are only 2-byte aligned.
    return words;
}
template<int T> constexpr int sign_count = T == 22 ? 256 : 128;
template<int T>
__device__ __forceinline__ void load_signs(uint4* signs) {
    for (int i = threadIdx.x; i < sign_count<T>; i += blockDim.x) {
        unsigned b = i;
        if constexpr (T != 22) b |= (__popc(b) & 1) << 7;
        unsigned lo = 0, hi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            lo |= (0u - ((b >> j) & 1u)) & (255u << (8 * j));
            hi |= (0u - ((b >> (j + 4)) & 1u)) & (255u << (8 * j));
        }
        signs[i] = make_uint4(lo, lo & 0x01010101u, hi, hi & 0x01010101u);
    }
}
template<int T>
__device__ __forceinline__ float dw_rdna(const uint8_t* row, int kbx, int iqs, const XV& x,
                                        const uint32_t* tab, const uint4* signs, bool direct, bool wide);
template<> __device__ __forceinline__ float dw_rdna<16>(const uint8_t* row, int kbx, int iqs,
    const XV& x, const uint32_t* tab, const uint4* sign_tab, bool direct, bool wide) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) row + kbx;
    const int q2 = wide ? quant_load8(bq2->qs + 2 * iqs).x : get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = wide ? quant_load8(bq2->qs + 2 * iqs).y : get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[aux8[k0 / 2]];
        const uint4 signs = direct ? direct_signs(aux32 >> (7 * k0 / 2), true) : sign_tab[(aux32 >> (7 * k0 / 2)) & 127];
        sumi = __dp4a((int) ((grid_pos.x ^ signs.x) + signs.y), x.u[k0 + 0], sumi);
        sumi = __dp4a((int) ((grid_pos.y ^ signs.z) + signs.w), x.u[k0 + 1], sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    return __half2float(bq2->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw_rdna<17>(const uint8_t* row, int kbx, int iqs,
    const XV& x, const uint32_t* tab, const uint4* sign_tab, bool direct, bool wide) {
    (void) wide;
    const block_iq2_xs* bq2 = (const block_iq2_xs*) row + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[q2[l0 / 2] & 0x1FF];
        const uint4 signs = direct ? direct_signs(q2[l0 / 2] >> 9, true) : sign_tab[(q2[l0 / 2] >> 9) & 127];
        const int grid_l = (int) ((grid_pos.x ^ signs.x) + signs.y);
        const int grid_h = (int) ((grid_pos.y ^ signs.z) + signs.w);
        if (l0 < 4) {
            sumi0 = __dp4a(grid_l, x.u[l0 + 0], sumi0);
            sumi0 = __dp4a(grid_h, x.u[l0 + 1], sumi0);
        } else {
            sumi1 = __dp4a(grid_l, x.u[l0 + 0], sumi1);
            sumi1 = __dp4a(grid_h, x.u[l0 + 1], sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;   // (as dw<17>)
    return __half2float(bq2->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw_rdna<18>(const uint8_t* row, int kbx, int iqs,
    const XV& x, const uint32_t* tab, const uint4* sign_tab, bool direct, bool wide) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) row + kbx;
    const uint2 q3_wide = wide ? quant_load8(bq3->qs + 4 * iqs) : make_uint2(0, 0);
    const int2 q3_packed = wide ? make_int2(q3_wide.x, q3_wide.y) :
        make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(tab[q3[l0 + 0]], tab[q3[l0 + 1]]);
        const uint4 signs = direct ? direct_signs(aux32 >> (7 * l0 / 2), true) : sign_tab[(aux32 >> (7 * l0 / 2)) & 127];
        sumi = __dp4a((int) ((grid_pos.x ^ signs.x) + signs.y), x.u[l0 + 0], sumi);
        sumi = __dp4a((int) ((grid_pos.y ^ signs.z) + signs.w), x.u[l0 + 1], sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    return __half2float(bq3->d) * x.d * sumi;
}
template<> __device__ __forceinline__ float dw_rdna<22>(const uint8_t* row, int kbx, int iqs,
    const XV& x, const uint32_t* tab, const uint4* sign_tab, bool direct, bool wide) {
    (void) wide;
    const block_iq2_s* bq2 = (const block_iq2_s*) row + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) tab)[qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)];
        const uint4 signs = direct ? direct_signs(signs_packed_8[l0 / 2], false) : sign_tab[signs_packed_8[l0 / 2]];
        const int grid_l = ((grid_pos.x ^ signs.x) + signs.y);
        const int grid_h = ((grid_pos.y ^ signs.z) + signs.w);
        if (l0 < 4) {
            sumi0 = __dp4a(grid_l, x.u[l0 + 0], sumi0);
            sumi0 = __dp4a(grid_h, x.u[l0 + 1], sumi0);
        } else {
            sumi1 = __dp4a(grid_l, x.u[l0 + 0], sumi1);
            sumi1 = __dp4a(grid_h, x.u[l0 + 1], sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    return __half2float(bq2->d) * x.d * sumi;
}
template<int T, int NR>
__device__ __forceinline__ void rows_dot_rdna(const uint8_t* const* rows, const block_q8_1* x,
                                             int n_in, int lane, const uint32_t* tab,
                                             const uint4* signs, float* s, bool direct = false, bool wide = false) {
    float acc[NR] = {};
    for (int k = lane; k < n_in / 32; k += 32) {
        const XV xv = load_xv(x + k);
#pragma unroll
        for (int r = 0; r < NR; ++r)
            acc[r] += dw_rdna<T>(rows[r], k / 8, 2 * (k % 8), xv, tab, signs, direct, wide);
    }
#pragma unroll
    for (int r = 0; r < NR; ++r) s[r] = warp_sum(acc[r]);
}
#endif

// their F entries (the dense GEMV and row_dot read the codebook from global memory; the call's q8_1 block is
// iqs / step of the super-block's)
#define GLMF_IQ_F(T, BLOCK, STEP, TABLE)                                                                          \
    template<> struct F<T> { static constexpr int qk = 256, ipb = 8, step = STEP, bytes = sizeof(BLOCK);         \
        __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) {                      \
            return dw<T>((const uint8_t*) v, kbx, iqs, load_xv(y + iqs / STEP), (const uint32_t*) TABLE); } };
GLMF_IQ_F(17, block_iq2_xs, 2, iq2xs_grid)
GLMF_IQ_F(22, block_iq2_s, 2, iq2s_grid)
GLMF_IQ_F(21, block_iq3_s, 2, iq3s_grid)
GLMF_IQ_F(29, block_iq1_m, 1, iq1s_grid_gpu)
#undef GLMF_IQ_F
// MXFP4 (deepseek4's converter keeps the routed experts' native FP4; unsloth's quants leave some down rows in it):
// vecdotq's vec_dot_mxfp4_q8_1 over a whole 32-value block (its two calls of 2 ints each in one)
__device__ __forceinline__ int get_int_b1(const void* x, const int& i32) {
    const uint8_t* x8 = (const uint8_t*) x;
    return (int) ((uint32_t) x8[4 * i32] | ((uint32_t) x8[4 * i32 + 1] << 8) | ((uint32_t) x8[4 * i32 + 2] << 16) |
                  ((uint32_t) x8[4 * i32 + 3] << 24));
}
__device__ __forceinline__ float e8m0_to_fp32(uint8_t x) {
    const uint32_t bits = x == 0 ? 0x00400000u : (uint32_t) x << 23;
    return __uint_as_float(bits);
}
template<> __device__ __forceinline__ float dw<39>(const uint8_t* row, int kbx, int, const XV& x, const uint32_t*) {
    const block_mxfp4* b = (const block_mxfp4*) row + kbx;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int2 v = get_int_from_table_16(get_int_b1(b->qs, l), kvalues_mxfp4);
        sumi = __dp4a(v.x, x.u[l], sumi);
        sumi = __dp4a(v.y, x.u[l + 4], sumi);
    }
    return e8m0_to_fp32(b->e) * 0.5f * x.d * (float) sumi;
}
template<> struct F<39> { static constexpr int qk = 32, ipb = 1, step = 1, bytes = sizeof(block_mxfp4);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int) {
        return dw<39>((const uint8_t*) v, kbx, 0, load_xv(y), nullptr); } };
// NVFP4 (NVIDIA's FP4 checkpoints, e.g. magnitudedev's Qwen3.8-Flash-Next NVFP4): 64 values a block in four
// sub-blocks of 16, each with a UE4M3 scale (x0.5: the E2M1 table is doubled as MXFP4's), then the 4-bit codes - sub
// s's 16 values are the low nibbles of qs[8s .. 8s+7] (values 0-7) and their high nibbles (8-15).  A call covers half
// a block (32 values: two sub-blocks) against one q8_1 block.  The tensor's second-level scale (per expert) is not
// here: moe_gate_up / moe_down apply it (MoeDev::nv_scale)
__device__ __forceinline__ float ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (x >> 3) & 0xF, m = x & 7;
    const float v = e == 0 ? (float) m * (1.0f / 512.0f) : __uint_as_float((uint32_t) (e + 120) << 23 | (uint32_t) m << 20);
    return 0.5f * v;
}
template<> __device__ __forceinline__ float dw<40>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t*) {
    const uint8_t* b = row + (size_t) kbx * 36;   // [4 UE4M3 scales | 32 bytes of codes]
    float acc = 0.0f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int sb = 2 * iqs + h;
        const uint8_t* q = b + 4 + 8 * sb;
        const int2 v0 = get_int_from_table_16(get_int_b1(q, 0), kvalues_mxfp4);
        const int2 v1 = get_int_from_table_16(get_int_b1(q, 1), kvalues_mxfp4);
        int sumi = __dp4a(v0.x, x.u[4 * h + 0], 0);
        sumi = __dp4a(v1.x, x.u[4 * h + 1], sumi);
        sumi = __dp4a(v0.y, x.u[4 * h + 2], sumi);
        sumi = __dp4a(v1.y, x.u[4 * h + 3], sumi);
        acc += ue4m3_to_fp32(b[sb]) * (float) sumi;
    }
    return x.d * acc;
}
template<> struct F<40> { static constexpr int qk = 64, ipb = 2, step = 1, bytes = 36;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) {
        return dw<40>((const uint8_t*) v, kbx, iqs, load_xv(y + iqs), nullptr); } };
// IQ4_NL (qwen4exp's down experts, orcarouter's Qwen3.8): 32 values a block, an FP16 scale and the iq4nl table -
// vecdotq's vec_dot_iq4_nl_q8_1 over the whole block (as MXFP4 above, without its x0.5)
template<> __device__ __forceinline__ float dw<20>(const uint8_t* row, int kbx, int, const XV& x, const uint32_t*) {
    const block_iq4_nl* b = (const block_iq4_nl*) row + kbx;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int2 v = get_int_from_table_16(get_int_b1(b->qs, l), kvalues_iq4nl);
        sumi = __dp4a(v.x, x.u[l], sumi);
        sumi = __dp4a(v.y, x.u[l + 4], sumi);
    }
    return __half2float(b->d) * x.d * (float) sumi;
}
template<> struct F<20> { static constexpr int qk = 32, ipb = 1, step = 1, bytes = sizeof(block_iq4_nl);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int) {
        return dw<20>((const uint8_t*) v, kbx, 0, load_xv(y), nullptr); } };
// Q5_1 (unsloth's UD-Q4_K_XL of Qwen3.8: the 640-wide down experts): 32 values, FP16 scale and min, the 5th bits in
// qh - vecdotq's vec_dot_q5_1_q8_1 over the whole block (its min term m5 * s8 once a block)
template<> __device__ __forceinline__ float dw<7>(const uint8_t* row, int kbx, int, const XV& x, const uint32_t*) {
    const block_q5_1* b = (const block_q5_1*) row + kbx;
    const int qh = get_int_b4(b->qh, 0);
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int vl = get_int_b4(b->qs, i), vh = qh >> (4 * i);
        int vi0 = vl & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = __dp4a(vi0, x.u[i], sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = __dp4a(vi1, x.u[i + 4], sumi);
    }
    const float2 dm = __half22float2(b->dm);
    return dm.x * x.d * (float) sumi + dm.y * x.s;
}
template<> struct F<7> { static constexpr int qk = 32, ipb = 1, step = 1, bytes = sizeof(block_q5_1);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int) {
        return dw<7>((const uint8_t*) v, kbx, 0, load_xv(y), nullptr); } };
// Q5_0 (unsloth's UD 3.1 UD-IQ2_XS of Qwen3.8: one layer's 640-wide down experts): Q5_1 without the min - codes
// 0..31 meaning -16..15.  The bytes are made signed (code - 16) before the dot, so the sum is ggml-cpu's exact integer
// sum: vecdotq's form (sumi * d8 - 16 * s8) subtracts the q8_1 block's FP16 sum s8 from a sum of about the same size,
// and that cancellation nearly doubled the layer's error against the double reference (STRATA_GLM_CPU_LANE_CHECK on
// that layer: 0.024 in vecdotq's form, 0.014 in this one; its IQ4_NL neighbours 0.013-0.017).  The
// block is 22 bytes, so its ints are read two bytes at a time (get_int_b2)
template<> __device__ __forceinline__ float dw<6>(const uint8_t* row, int kbx, int, const XV& x, const uint32_t*) {
    const block_q5_0* b = (const block_q5_0*) row + kbx;
    const int qh = get_int_b2(b->qh, 0);
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int vl = get_int_b2(b->qs, i), vh = qh >> (4 * i);
        int vi0 = vl & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = __dp4a((int) __vsub4(vi0, 0x10101010), x.u[i], sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = __dp4a((int) __vsub4(vi1, 0x10101010), x.u[i + 4], sumi);
    }
    return __half2float(b->d) * x.d * (float) sumi;
}
template<> struct F<6> { static constexpr int qk = 32, ipb = 1, step = 1, bytes = sizeof(block_q5_0);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int) {
        return dw<6>((const uint8_t*) v, kbx, 0, load_xv(y), nullptr); } };
// Q2_0 (ISTA-DASLab's GSQ-RCO Qwen3.8: down experts and shared experts): 64 values, one FP16 scale, 2-bit codes
// 0..3 meaning -1..2 - vecdotq's vec_dot_q2_0_q8_1, a call per q8_1 block (iqs: the block's half).  The codes are
// spread into bytes with shifts and masks (no byte permute: the same on CUDA and HIP)
__device__ __forceinline__ int q2_0_bytes(int q) {   // 4 codes (8 bits) -> 4 signed bytes, code - 1
    const int b = (q & 0x03) | ((q & 0x0C) << 6) | ((q & 0x30) << 12) | ((q & 0xC0) << 18);
    return __vsub4(b, 0x01010101);
}
template<> __device__ __forceinline__ float dw<42>(const uint8_t* row, int kbx, int iqs, const XV& x, const uint32_t*) {
    const block_q2_0* b = (const block_q2_0*) row + kbx;
    const uint16_t* qs = (const uint16_t*) b->qs + iqs * 4;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        sumi = __dp4a(q2_0_bytes(q & 0xFF), x.u[2 * j], sumi);
        sumi = __dp4a(q2_0_bytes(q >> 8), x.u[2 * j + 1], sumi);
    }
    return __half2float(b->d) * x.d * (float) sumi;
}
template<> struct F<42> { static constexpr int qk = 64, ipb = 2, step = 1, bytes = sizeof(block_q2_0);
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) {
        return dw<42>((const uint8_t*) v, kbx, iqs, load_xv(y + iqs), nullptr); } };
// NR rows against ONE activation, the whole warp; s[r] gets each row's full sum
// the K-quant calls read 4 q8_1 blocks' int (iqs % 8) each: loaded ONCE per call position for all rows
struct XK {
    int u[4];
    float d8[4];
};
__device__ __forceinline__ XK load_xk(const block_q8_1* xb, int iqs) {
    XK v;
    const int off = 4 * (iqs / 8);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v.u[i] = get_int_b4(xb[off + i].qs, iqs % 8);
        v.d8[i] = __low2float(xb[off + i].ds);
    }
    return v;
}
template<int T> __device__ __forceinline__ float dwk(const uint8_t* row, int kbx, int iqs, const XK& x);
template<> __device__ __forceinline__ float dwk<10>(const uint8_t* row, int kbx, int iqs, const XK& x) {
    const block_q2_K* b = (const block_q2_K*) row + kbx;
    const uint8_t* scales = b->scales + (iqs - iqs % 8 + (iqs % 8) / 4);
    const int v = get_int_b4(b->qs, iqs);
    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int sc = scales[2 * i];
        sumf_d += x.d8[i] * (__dp4a((v >> (2 * i)) & 0x03030303, x.u[i], 0) * (sc & 0xF));
        int m = sc >> 4;
        m |= m << 8;
        m |= m << 16;
        sumf_m += x.d8[i] * __dp4a(m, x.u[i], 0);
    }
    const float2 dm = __half22float2(b->dm);
    return dm.x * sumf_d - dm.y * sumf_m;
}
template<> __device__ __forceinline__ float dwk<11>(const uint8_t* row, int kbx, int iqs, const XK& x) {
    const block_q3_K* b = (const block_q3_K*) row + kbx;
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const int vl = get_int_b2(b->qs, iqs);
    const int vh = ~get_int_b2(b->hmask, iqs % 8) >> bq8_offset;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int sc_low = (b->scales[isc % 8] >> (4 * (isc / 8))) & 0xF;
        const int sc_high = ((b->scales[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        sumf += x.d8[i] * (__dp4a((int) __vsubss4(vil, vih), x.u[i], 0) * ((sc_low | sc_high) - 32));
    }
    return __half2float(b->d) * sumf;
}
template<int T, int NR>
__device__ __forceinline__ void rows_dot(const uint8_t* const* rows, const block_q8_1* x, int n_in, int lane,
                                         const uint32_t* tab, float* s) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
    float acc[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) acc[r] = 0.0f;
    if constexpr (T == 10 || T == 11) {
        for (int k = lane; k < nb * Fm::ipb; k += 32) {
            const int kbx = k / Fm::ipb, iqs = k % Fm::ipb;
            const XK xk = load_xk(x + kbx * (Fm::qk / 32), iqs);
#pragma unroll
            for (int r = 0; r < NR; ++r) acc[r] += dwk<T>(rows[r], kbx, iqs, xk);
        }
    } else if constexpr (T == 12 || T == 13 || T == 14 || T == 8) {
        // Q4_K / Q5_K / Q6_K experts (K-quant GGUFs other than Maya's): no shared-activation form yet - each row
        // takes the dense GEMV's dot (row_dot), the activation re-read per row
        for (int k = lane; k < nb * Fm::ipb; k += 32) {
            const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
#pragma unroll
            for (int r = 0; r < NR; ++r) acc[r] += Fm::dot(rows[r], x + kbx * (Fm::qk / 32), kbx, iqs);
        }
    } else {
        for (int k = lane; k < nb * Fm::ipb; k += 32) {
            const int kbx = k / Fm::ipb, ki = k % Fm::ipb, iqs = Fm::step * ki;
            const XV xv = load_xv(x + kbx * (Fm::qk / 32) + ki);
#pragma unroll
            for (int r = 0; r < NR; ++r) acc[r] += dw<T>(rows[r], kbx, iqs, xv, tab);
        }
    }
#pragma unroll
    for (int r = 0; r < NR; ++r) s[r] = warp_sum(acc[r]);
}

template<int T>
__device__ __forceinline__ float row_dot_t(const uint8_t* row, const block_q8_1* x, int n_in, int lane,
                                           const uint32_t* tab) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
    float s = 0.0f;
#pragma unroll 4
    for (int k = lane; k < nb * Fm::ipb; k += 32) {
        const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
        s += dot_t<T>(row, x + kbx * (Fm::qk / 32), kbx, iqs, tab);
    }
    return warp_sum(s);
}

__device__ __forceinline__ float any_row_dot(int type, const void* w, int row, const void* xq, const float* xf,
                                             int n_in, int lane) {
    switch (type) {
        case kTypeF32: return row_dot_f32((const float*) w + (size_t) row * n_in, xf, n_in, lane);
        case kTypeBF16: return row_dot_bf16((const uint16_t*) w + (size_t) row * n_in, xf, n_in, lane);
#define GF_CASE(T) case T: return row_dot<T>((const uint8_t*) w + (size_t) row * rbytes<T>(n_in), (const block_q8_1*) xq, n_in, lane);
        GF_CASE(12) GF_CASE(13) GF_CASE(14) GF_CASE(8) GF_CASE(16) GF_CASE(18) GF_CASE(19) GF_CASE(23) GF_CASE(10) GF_CASE(11)
        GF_CASE(17) GF_CASE(22) GF_CASE(21) GF_CASE(29) GF_CASE(20) GF_CASE(7) GF_CASE(42)
#undef GF_CASE
        default: return 0.0f;
    }
}

// a row's bytes in any_row_dot's types (mv_kernel's L2 pull; 0 for a type it does not take)
__device__ __forceinline__ size_t any_row_bytes(int type, int n_in) {
    switch (type) {
        case kTypeF32: return (size_t) n_in * 4;
        case kTypeBF16: return (size_t) n_in * 2;
#define GF_CASE(T) case T: return rbytes<T>(n_in);
        GF_CASE(12) GF_CASE(13) GF_CASE(14) GF_CASE(8) GF_CASE(16) GF_CASE(18) GF_CASE(19) GF_CASE(23) GF_CASE(10) GF_CASE(11)
        GF_CASE(17) GF_CASE(22) GF_CASE(21) GF_CASE(29) GF_CASE(20) GF_CASE(7) GF_CASE(42)
#undef GF_CASE
        default: return 0;
    }
}

#if defined(STRATA_USE_HIP)
// ... over part p of split of a row's k-blocks (mv_kernel_gs: split warps a row): the warp's partial sum, unreduced
template<int T>
__device__ __forceinline__ float row_dot_part(const uint8_t* row, const block_q8_1* x, int n_in, int lane, int p,
                                              int split) {
    using Fm = F<T>;
    const int nk = n_in / Fm::qk * Fm::ipb;
    float s = 0.0f;
    for (int k = nk * p / split + lane; k < nk * (p + 1) / split; k += 32) {
        const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
        s += Fm::dot(row, x + kbx * (Fm::qk / 32), kbx, iqs);
    }
    return s;
}
__device__ __forceinline__ float any_row_dot_part(int type, const void* w, int row, const void* xq, const float* xf,
                                                  int n_in, int lane, int p, int split) {
    switch (type) {
        case kTypeF32: {
            const float4* r4 = (const float4*) ((const float*) w + (size_t) row * n_in);
            const float4* x4 = (const float4*) xf;
            const int n4 = n_in / 4;
            float s = 0.0f;
            for (int k = n4 * p / split + lane; k < n4 * (p + 1) / split; k += 32) {
                const float4 a = r4[k], b = x4[k];
                s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
            return s;
        }
        case kTypeBF16: {
            const uint4* r8 = (const uint4*) ((const uint16_t*) w + (size_t) row * n_in);
            const float4* x4 = (const float4*) xf;
            const int n8 = n_in / 8;
            float s = 0.0f;
            for (int k = n8 * p / split + lane; k < n8 * (p + 1) / split; k += 32) {
                const uint4 v = r8[k];
                const float4 a = x4[2 * k], b = x4[2 * k + 1];
                s += __uint_as_float(v.x << 16) * a.x + __uint_as_float(v.x & 0xffff0000u) * a.y +
                     __uint_as_float(v.y << 16) * a.z + __uint_as_float(v.y & 0xffff0000u) * a.w +
                     __uint_as_float(v.z << 16) * b.x + __uint_as_float(v.z & 0xffff0000u) * b.y +
                     __uint_as_float(v.w << 16) * b.z + __uint_as_float(v.w & 0xffff0000u) * b.w;
            }
            return s;
        }
#define GF_CASE(T) case T: return row_dot_part<T>((const uint8_t*) w + (size_t) row * rbytes<T>(n_in), (const block_q8_1*) xq, n_in, lane, p, split);
        GF_CASE(12) GF_CASE(13) GF_CASE(14) GF_CASE(8) GF_CASE(16) GF_CASE(18) GF_CASE(19) GF_CASE(23) GF_CASE(10) GF_CASE(11)
        GF_CASE(17) GF_CASE(22) GF_CASE(21) GF_CASE(29) GF_CASE(20) GF_CASE(7) GF_CASE(42)
#undef GF_CASE
        default: return 0.0f;
    }
}

// gfx1201, a full 64-CU die: a dispatch whose waves land at or near a multiple of the die's wave slots delays the NEXT
// dispatch's start by 8-14 us (ROCm/TheRock#9300, ROCm 10.0 / HIP 7.15; only a kernel that can fill every slot).  An AI
// PRO R9700 met it ~275 times a DeepSeek V4 token: the 4096-row GEMVs (a wave a row) and the expert down (8192 waves).
// Blocks that return at once move a launch out of the window (Strata PR #2492: a 2560 -> 512 matvec 16.5 -> 5.0 us);
// every row and sum stays the same.  The window is HIP's CU count (32 on the R9700: WGPs) x 32 = 1024: the "full" 2048
// (x 64, issue #9) measured no better there (2 of 4 runs at the no-padding level).  gfx11 does not show the stall
// (issue #9, RX 7900 XT: none 40.4, 1024-style 39.9, 2048-style 40.7 tok/s, within the runs' spread), so the padding is
// on only for gfx12 and the archs outside gfx11.  STRATA_GLM_GRID_PAD=0: none anywhere, =1: on everywhere.
static int grid_pad(int blocks, int wpb) {
    static const int env = getenv("STRATA_GLM_GRID_PAD") == nullptr ? -1 : std::atoi(getenv("STRATA_GLM_GRID_PAD")) != 0;
    static thread_local int dev = -1, slots = 0;
    int d = 0;
    if (env == 0 || wpb <= 0 || hipGetDevice(&d) != hipSuccess) return blocks;
    if (d != dev) {
        int cu = 0;
        hipDeviceProp_t prop{};
        const bool gfx11 = hipGetDeviceProperties(&prop, d) == hipSuccess && std::strncmp(prop.gcnArchName, "gfx11", 5) == 0;
        slots = (env == 1 || !gfx11) && hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, d) == hipSuccess
                    ? cu * 32 : 0;
        dev = d;
    }
    if (slots <= 0) return blocks;
    const int win = std::max(16, 4 * wpb), r = (int) (((long long) blocks * wpb) % slots);
    if (r < 16) return blocks + (16 - r + wpb - 1) / wpb;                     // at a multiple or just past it
    if (r > slots - win) return blocks + (slots - r + 16 + wpb - 1) / wpb;    // just below one
    return blocks;
}
#else
static inline int grid_pad(int blocks, int) { return blocks; }
#endif

// ---------------------------------------------------------------- per-GPU choices for the dense GEMVs
// Picked once per device from its architecture; a choice changes only where it was measured on that GPU, every other
// GPU keeps the old default.  STRATA_GLM_MV_Q6K_SPLIT (0 / 1) overrides; the line "glm fast: dense GEMV tune" says
// what a device got.
//  - q6k_split: Q6_K GEMVs of up to 4,096 blocks 4 warps a row.  RTX 5090: faster (wo_a 24.1 -> 18.1 us).  gfx12 (AI PRO
//    R9700, DeepSeek V4.1 UD-IQ1_S decode, rocprofv3): slower - 8.78 ms a token split 4 ways, 8.2 split 2, 7.45 one
//    warp a row (475 -> 562 GB/s).  (Measured and not taken: DeepSeek V4.1's F32 routers, 384 rows, 4 warps a row on
//    gfx12 - 19.3 -> 18.6 us, a small GEMV's floor there)
struct MvTune {
    bool q6k_split = true;
};
static const MvTune& mv_tune() {
    static thread_local int dev = -1;
    static thread_local MvTune t;
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d == dev) return t;
    dev = d;
    t = MvTune{};
    char arch[64] = "?";
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, d) == cudaSuccess) {
#if defined(STRATA_USE_HIP)
        std::snprintf(arch, sizeof arch, "%s", prop.gcnArchName);
        if (char* c = std::strchr(arch, ':')) *c = 0;
        const bool gfx12 = std::strncmp(arch, "gfx12", 5) == 0;   // RDNA 4 (gfx1200 / gfx1201)
        const bool gfx1100 = std::strncmp(arch, "gfx1100", 7) == 0; // RX 7900 XT / XTX: 1 warp a row measured best (abyss_gpu_tune)
        t.q6k_split = !(gfx12 || gfx1100);
#else
        std::snprintf(arch, sizeof arch, "sm_%d%d", prop.major, prop.minor);
#endif
    }
    if (const char* v = getenv("STRATA_GLM_MV_Q6K_SPLIT")) t.q6k_split = std::atoi(v) != 0;
    std::fprintf(stderr, "glm fast: dense GEMV tune for device %d (%s, %d CUs): Q6_K %s\n", d, arch,
                 prop.multiProcessorCount, t.q6k_split ? "4 warps a row (up to 4,096 blocks)" : "a warp a row");
    return t;
}

// (abyss_gpu_tune: a forced choice for the calls that follow, -1 = mv_tune's)
static std::atomic<int> g_q6k_force{-1};
std::atomic<int> g_split_force{-1};   // gpu_tune: 1 / 2 / 4 rows-split for every typed launch (-1: the rule)

// ---------------------------------------------------------------- multi-job GEMV
struct MvBatch {
    MvJob j[kMaxMvJobs];
    int blk_end[kMaxMvJobs];
    int n;
    int pf;   // (PDL launches: the typed kernels pull their weight rows into L2 before the wait - mv_prefetch_l2)
};
constexpr int MV_ROWS = 8;   // rows (warps) per block

// a warp's share of [w, w + bytes) into L2, 128 B lines (part of parts: a row split over warps).  Only for weights -
// nothing an earlier kernel writes - so it may run before STRATA_PDL_ENTRY(): with PDL the block is dispatched during
// the previous kernel's tail and its first weight lines arrive while that kernel finishes.  A hint: no value changes
__device__ __forceinline__ void mv_prefetch_l2(const void* w, size_t bytes, int lane, int part, int parts) {
#if defined(__CUDA_ARCH__) && !defined(STRATA_USE_HIP)
    const char* c = (const char*) w;
    for (size_t o = (size_t) (lane * parts + part) * 128; o < bytes; o += (size_t) 32 * parts * 128)
        asm volatile("prefetch.global.L2 [%0];" ::"l"(c + o));
#else
    (void) w; (void) bytes; (void) lane; (void) part; (void) parts;
#endif
}

__global__ void __launch_bounds__(256) mv_kernel(const __grid_constant__ MvBatch b) {
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row = (bid - blk0) * MV_ROWS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= J.n_out) return;   // (no global access before this: an exit is the trigger)
    if (b.pf) {
        const size_t rbt = any_row_bytes(J.type, J.n_in);
        mv_prefetch_l2((const uint8_t*) J.w + (size_t) row * rbt, rbt, lane, 0, 1);
    }
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any read of what an earlier kernel writes)
    const float s = any_row_dot(J.type, J.w, row, J.xq, J.xf, J.n_in, lane);
    if (lane == 0) J.y[row] = J.alpha * s + (J.bias ? J.bias[row] : 0.0f);
}

// ... the same for jobs of ONE quantized type: compiled for that type alone (the switch over every type costs the
// generic kernel registers and occupancy - a Q6_K 8192 x 4096 GEMV on a V100: 53.4 -> 47.8 us, the same arithmetic)
template<int T>
__global__ void __launch_bounds__(256) mv_kernel_t(const __grid_constant__ MvBatch b) {
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row = (bid - blk0) * MV_ROWS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= J.n_out) return;   // (no global access before this: an exit is the trigger)
    if (b.pf) mv_prefetch_l2((const uint8_t*) J.w + (size_t) row * rbytes<T>(J.n_in), rbytes<T>(J.n_in), lane, 0, 1);
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any read of what an earlier kernel writes)
    const float s = row_dot<T>((const uint8_t*) J.w + (size_t) row * rbytes<T>(J.n_in), (const block_q8_1*) J.xq,
                               J.n_in, lane);
    if (lane == 0) J.y[row] = J.alpha * s + (J.bias ? J.bias[row] : 0.0f);
}

// ... and with SPLIT warps a row, for the small jobs: a row's k-blocks are shared by SPLIT warps and their partial
// sums added through shared memory.  One warp a row left wo_b (4,096 rows of 8.7 KB: 512 blocks) at 1.5 TB/s and
// the shared expert's gate/up (512 blocks of Q5_K) at 1.4 against q_b's 1.6 (32,768 short rows, 4,096 blocks) on an
// RTX 5090: too few warps in flight for the latency.  The sums are not bit-identical to the one-warp kernel's
template<int T, int SPLIT>
__global__ void __launch_bounds__(256) mv_kernel_ts(const __grid_constant__ MvBatch b) {
    constexpr int RPB = MV_ROWS / SPLIT;
    __shared__ float s_part[MV_ROWS];
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = (bid - blk0) * RPB + warp / SPLIT, part = warp % SPLIT;
    if (b.pf && row < J.n_out)
        mv_prefetch_l2((const uint8_t*) J.w + (size_t) row * rbytes<T>(J.n_in), rbytes<T>(J.n_in), lane, part, SPLIT);
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any read of what an earlier kernel writes)
    using Fm = F<T>;
    float s = 0.0f;
    if (row < J.n_out) {
        const uint8_t* w = (const uint8_t*) J.w + (size_t) row * rbytes<T>(J.n_in);
        const block_q8_1* x = (const block_q8_1*) J.xq;
        const int nk = J.n_in / Fm::qk * Fm::ipb;
        const int k0 = nk * part / SPLIT, k1 = nk * (part + 1) / SPLIT;
        for (int k = k0 + lane; k < k1; k += 32) {
            const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
            s += Fm::dot(w, x + kbx * (Fm::qk / 32), kbx, iqs);
        }
    }
    s = warp_sum(s);
    if (lane == 0) s_part[warp] = s;
    __syncthreads();
    if (part == 0 && lane == 0 && row < J.n_out) {
        float t = 0.0f;
#pragma unroll
        for (int p = 0; p < SPLIT; ++p) t += s_part[warp + p];
        J.y[row] = J.alpha * t + (J.bias ? J.bias[row] : 0.0f);
    }
}

#if defined(STRATA_USE_HIP)
// ... the same for jobs of ANY types in one launch (HIP, mv's small calls of several types): the type is switched per
// block (one job a block), which RDNA's wave32 affords - 16 waves a SIMD up to 96 VGPRs
template<int SPLIT>
__global__ void __launch_bounds__(256) mv_kernel_gs(const __grid_constant__ MvBatch b) {
    constexpr int RPB = MV_ROWS / SPLIT;
    __shared__ float s_part[MV_ROWS];
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = (bid - blk0) * RPB + warp / SPLIT, part = warp % SPLIT;
    float s = 0.0f;
    if (row < J.n_out) s = any_row_dot_part(J.type, J.w, row, J.xq, J.xf, J.n_in, lane, part, SPLIT);
    s = warp_sum(s);
    if (lane == 0) s_part[warp] = s;
    __syncthreads();
    if (part == 0 && lane == 0 && row < J.n_out) {
        float t = 0.0f;
#pragma unroll
        for (int p = 0; p < SPLIT; ++p) t += s_part[warp + p];
        J.y[row] = J.alpha * t + (J.bias ? J.bias[row] : 0.0f);
    }
}
#endif

// ... and against NC activations at once (a verify window's tokens): the weight row is read once, every column's
// q8_1 blocks at xq + c * n_in / 32, its outputs at y + c * n_out (columns contiguous, as native_mmvq's)
template<int T, int NC>
__global__ void __launch_bounds__(256) mvn_kernel_t(const __grid_constant__ MvBatch b) {
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row = (bid - blk0) * MV_ROWS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= J.n_out) return;
    using Fm = F<T>;
    const uint8_t* w = (const uint8_t*) J.w + (size_t) row * rbytes<T>(J.n_in);
    const block_q8_1* x = (const block_q8_1*) J.xq;
    const int nb = J.n_in / Fm::qk, xs = J.xs ? J.xs : J.n_in / 32;
    const size_t ys = J.ys ? (size_t) J.ys : (size_t) J.n_out;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    for (int k = lane; k < nb * Fm::ipb; k += 32) {
        const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[c] += Fm::dot(w, x + (size_t) c * xs + kbx * (Fm::qk / 32), kbx, iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_sum(acc[c]);
        if (lane == 0) J.y[(size_t) c * ys + row] = J.alpha * v + (J.bias ? J.bias[row] : 0.0f);
    }
}

// BF16 / F32 rows against NC float activations (xf + c * n_in)
template<int NC>
__global__ void __launch_bounds__(256) mvn_kernel_f(const __grid_constant__ MvBatch b) {
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row = (bid - blk0) * MV_ROWS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= J.n_out) return;
    const size_t xs = J.xs ? (size_t) J.xs : (size_t) J.n_in, ys = J.ys ? (size_t) J.ys : (size_t) J.n_out;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    if (J.type == kTypeBF16) {
        const uint4* r8 = (const uint4*) ((const uint16_t*) J.w + (size_t) row * J.n_in);
        for (int k = lane; k < J.n_in / 8; k += 32) {
            const uint4 wv = r8[k];
            const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);
            const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);
            const float w4 = __uint_as_float(wv.z << 16), w5 = __uint_as_float(wv.z & 0xffff0000u);
            const float w6 = __uint_as_float(wv.w << 16), w7 = __uint_as_float(wv.w & 0xffff0000u);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const float4* x4 = (const float4*) (J.xf + (size_t) c * xs);
                const float4 a = x4[2 * k], bb = x4[2 * k + 1];
                acc[c] += w0 * a.x + w1 * a.y + w2 * a.z + w3 * a.w + w4 * bb.x + w5 * bb.y + w6 * bb.z + w7 * bb.w;
            }
        }
    } else {   // F32
        const float4* r4 = (const float4*) ((const float*) J.w + (size_t) row * J.n_in);
        for (int k = lane; k < J.n_in / 4; k += 32) {
            const float4 a = r4[k];
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const float4 xx = ((const float4*) (J.xf + (size_t) c * xs))[k];
                acc[c] += a.x * xx.x + a.y * xx.y + a.z * xx.z + a.w * xx.w;
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_sum(acc[c]);
        if (lane == 0) J.y[(size_t) c * ys + row] = J.alpha * v + (J.bias ? J.bias[row] : 0.0f);
    }
}

__global__ void quantize_kernel(const float* __restrict__ x, block_q8_1* __restrict__ y, int n) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;   // n % 32 == 0: whole warps only
    q8_1_store_warp(x[i], y + i / 32, threadIdx.x & 31);
}

// ... and SPLIT warps a row against NC activations (mvn_mv): each column's partial sums exactly as mv_kernel_ts makes
// them for one activation (the same k ranges a warp, the same lane-strided loop, the parts added in order)
template<int T, int SPLIT, int NC>
__global__ void __launch_bounds__(256) mvn_kernel_ts(const __grid_constant__ MvBatch b) {
    constexpr int RPB = MV_ROWS / SPLIT;
    __shared__ float s_part[NC][MV_ROWS];
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const MvJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = (bid - blk0) * RPB + warp / SPLIT, part = warp % SPLIT;
    using Fm = F<T>;
    const int xs = J.xs ? J.xs : J.n_in / 32;
    const size_t ys = J.ys ? (size_t) J.ys : (size_t) J.n_out;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    if (row < J.n_out) {
        const uint8_t* w = (const uint8_t*) J.w + (size_t) row * rbytes<T>(J.n_in);
        const block_q8_1* x = (const block_q8_1*) J.xq;
        const int nk = J.n_in / Fm::qk * Fm::ipb;
        const int k0 = nk * part / SPLIT, k1 = nk * (part + 1) / SPLIT;
        for (int k = k0 + lane; k < k1; k += 32) {
            const int kbx = k / Fm::ipb, iqs = Fm::step * (k % Fm::ipb);
#pragma unroll
            for (int c = 0; c < NC; ++c) acc[c] += Fm::dot(w, x + (size_t) c * xs + kbx * (Fm::qk / 32), kbx, iqs);
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_sum(acc[c]);
        if (lane == 0) s_part[c][warp] = v;
    }
    __syncthreads();
    if (part == 0 && lane == 0 && row < J.n_out) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float t = 0.0f;
#pragma unroll
            for (int p = 0; p < SPLIT; ++p) t += s_part[c][warp + p];
            J.y[(size_t) c * ys + row] = J.alpha * t + (J.bias ? J.bias[row] : 0.0f);
        }
    }
}

// ---------------------------------------------------------------- mHC
constexpr int HC_CHUNK = 512;
constexpr int HC_THREADS = 256;
constexpr int HC_MIX = 24;      // (2 + hc) * hc for hc = 4

__global__ void __launch_bounds__(HC_THREADS) hc_kernel(const __grid_constant__ HcArgs a) {
    const int hc = 4;
    const int n_embd = a.n_embd;
    const int hc_dim = hc * n_embd;
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ float sR[HC_CHUNK];
    __shared__ float sred[32];
    __shared__ float spart[HC_MIX];
    __shared__ bool s_last;

    // ---- phase 1: this block's chunk of R_new, its sum of squares and the 24 partial dots
    float ss = 0.0f;
    for (int i = tid; i < HC_CHUNK; i += HC_THREADS) {
        const int idx = b * HC_CHUNK + i;
        float r;
        if (a.block_out != nullptr) {
            const int d = idx / n_embd, e = idx - d * n_embd;
            r = a.block_out[e] * a.post_in[d];
#pragma unroll
            for (int s = 0; s < hc; ++s) r += a.comb_in[d * hc + s] * a.R_old[s * n_embd + e];
            a.R_new[idx] = r;
        } else {
            r = a.R_old[idx];
        }
        sR[i] = r;
        ss += r * r;
    }
    __syncthreads();
    for (int m = warp; m < HC_MIX; m += HC_THREADS / 32) {
        const size_t w0 = (size_t) m * hc_dim + (size_t) b * HC_CHUNK;
        const uint16_t* wr = a.w_fn + w0;
        float acc = 0.0f;
        if (a.w_fn_q8 != nullptr)
            for (int i = lane; i < HC_CHUNK; i += 32) acc += sR[i] * q8_0_at((const uint8_t*) a.w_fn_q8, w0 + i);
        else
            for (int i = lane; i < HC_CHUNK; i += 32) acc += sR[i] * bf(wr[i]);
        acc = warp_sum(acc);
        if (lane == 0) spart[m] = acc;
    }
    ss = block_sum(ss, sred);
    if (tid < HC_MIX) a.part[(size_t) b * 25 + tid] = spart[tid];
    if (tid == 0) a.part[(size_t) b * 25 + 24] = ss;
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(a.counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!s_last) return;

    // ---- phase 2 (the last block): the gates, the Sinkhorn, mixed, the norm, q8_1
    __shared__ float smix[HC_MIX + 1];
    __shared__ float s_pre[4];
    __shared__ float s_mixed[kMaxEmbd];
    if (tid == 0) *a.counter = 0;
    if (tid < 25) {
        float acc = 0.0f;
        for (int bb = 0; bb < (int) gridDim.x; ++bb) acc += __ldcg(&a.part[(size_t) bb * 25 + tid]);
        smix[tid] = acc;
    }
    __syncthreads();
    const float inv = rsqrtf(smix[24] / (float) hc_dim + a.norm_eps);
    if (warp == 0) {
        if (lane < hc) {
            const float pre_v = dsigmoid(smix[lane] * inv * a.w_scale[0] + a.w_base[lane]) + a.hc_eps;
            const float post_v = 2.0f * dsigmoid(smix[hc + lane] * inv * a.w_scale[1] + a.w_base[hc + lane]);
            a.pre[lane] = pre_v;
            s_pre[lane] = pre_v;
            a.post[lane] = post_v;
        }
        // comb[d][s] on lane d*4 + s (lanes 0..15); lanes 16..31 shadow harmlessly
        const int l = lane & 15, d = l >> 2, s = l & 3;
        float c = smix[2 * hc + d + hc * s] * inv * a.w_scale[2] + a.w_base[2 * hc + d + hc * s];
        // softmax over dst (d) per column s: lanes sharing s differ in bits 2..3
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 4));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
        const float ex = __expf(c - mx);
        float den = ex + __shfl_xor_sync(0xffffffffu, ex, 4);
        den += __shfl_xor_sync(0xffffffffu, den, 8);
        c = ex / den + a.hc_eps;
        // dst-normalise: sum over s (bits 0..1); src-normalise: sum over d (bits 2..3)
        {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        for (int it = 1; it < a.iters; ++it) {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 4);
            sm += __shfl_xor_sync(0xffffffffu, sm, 8);
            c = c / (a.hc_eps + sm);
            sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        if (lane < 16) a.comb[d * hc + s] = c;
    }
    __syncthreads();
    const float* Rn = a.block_out != nullptr ? a.R_new : a.R_old;
    float ss2 = 0.0f;
    for (int e = tid; e < n_embd; e += HC_THREADS) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < hc; ++s) acc += (a.pre_use != nullptr ? a.pre_use[s] : s_pre[s]) * __ldcg(&Rn[s * n_embd + e]);
        s_mixed[e] = acc;
        ss2 += acc * acc;
    }
    ss2 = block_sum(ss2, sred);
    const float inv2 = rsqrtf(ss2 / (float) n_embd + a.norm_eps);
    for (int e = tid; e < n_embd; e += HC_THREADS) {   // warp w holds 32 consecutive e at every step
        const float xv = s_mixed[e] * inv2 * a.norm_w[e];
        a.x[e] = xv;
        q8_1_store_warp(xv, (block_q8_1*) a.xq + e / 32, lane);
    }
}

// ---- hc, take two: the grid splits by EMBEDDING range (all 4 streams of 128 values per block), so after the partial
// dots every block can finish its own range - two grid barriers (the blocks are co-resident: 32 blocks, checked
// at launch) instead of one block doing the whole tail.  Same arithmetic as hc_kernel.
constexpr int HC2_EB = 128;
__device__ __forceinline__ void grid_barrier(unsigned long long* ctr, unsigned int nb) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned long long v = atomicAdd(ctr, 1ull);
        const unsigned long long target = (v / nb + 1) * nb;
        while (*(volatile unsigned long long*) ctr < target) {
        }
        __threadfence();
    }
    __syncthreads();
}

__global__ void __launch_bounds__(HC_THREADS) hc2_kernel(const __grid_constant__ HcArgs a) {
    const int hc = 4;
    const int n_embd = a.n_embd;
    const int hc_dim = hc * n_embd;
    const unsigned int nb = gridDim.x;
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int e0 = b * HC2_EB;
    unsigned long long* ctr = (unsigned long long*) a.counter;
    __shared__ float sR[4][HC2_EB];
    __shared__ float sred[32];
    __shared__ float spart[HC_MIX];
    __shared__ float smix[HC_MIX + 1];
    __shared__ float s_pre[4];
    __shared__ float s_mixed[HC2_EB];

    // ---- phase 1: this block's 4 x 128 values of R_new, their sum of squares and the 24 partial dots
    float ss = 0.0f;
    for (int i = tid; i < 4 * HC2_EB; i += HC_THREADS) {
        const int d = i / HC2_EB, j = i - d * HC2_EB, e = e0 + j;
        float r;
        if (a.block_out != nullptr) {
            r = a.block_out[e] * a.post_in[d];
#pragma unroll
            for (int s = 0; s < hc; ++s) r += a.comb_in[d * hc + s] * a.R_old[s * n_embd + e];
            a.R_new[d * n_embd + e] = r;
        } else {
            r = a.R_old[d * n_embd + e];
        }
        sR[d][j] = r;
        ss += r * r;
    }
    __syncthreads();
    for (int m = warp; m < HC_MIX; m += HC_THREADS / 32) {
        const size_t w0 = (size_t) m * hc_dim + e0;
        const uint16_t* wr = a.w_fn + w0;
        float acc = 0.0f;
        for (int i = lane; i < 4 * HC2_EB; i += 32) {
            const int d = i / HC2_EB, j = i - d * HC2_EB;
            acc += sR[d][j] * (a.w_fn_q8 != nullptr ? q8_0_at((const uint8_t*) a.w_fn_q8, w0 + (size_t) d * n_embd + j)
                                                    : bf(wr[(size_t) d * n_embd + j]));
        }
        acc = warp_sum(acc);
        if (lane == 0) spart[m] = acc;
    }
    ss = block_sum(ss, sred);
    if (tid < HC_MIX) a.part[(size_t) b * 25 + tid] = spart[tid];
    if (tid == 0) a.part[(size_t) b * 25 + 24] = ss;
    grid_barrier(ctr, nb);

    // ---- phase 2 (every block, redundantly): the gates and the Sinkhorn from the partial sums
    if (tid < 25) {
        float acc = 0.0f;
        for (int bb = 0; bb < (int) nb; ++bb) acc += __ldcg(&a.part[(size_t) bb * 25 + tid]);
        smix[tid] = acc;
    }
    __syncthreads();
    const float inv = rsqrtf(smix[24] / (float) hc_dim + a.norm_eps);
    if (warp == 0) {
        if (lane < hc) {
            const float pre_v = dsigmoid(smix[lane] * inv * a.w_scale[0] + a.w_base[lane]) + a.hc_eps;
            s_pre[lane] = pre_v;
            if (b == 0) {
                a.pre[lane] = pre_v;
                a.post[lane] = 2.0f * dsigmoid(smix[hc + lane] * inv * a.w_scale[1] + a.w_base[hc + lane]);
            }
        }
        const int l = lane & 15, d = l >> 2, s = l & 3;
        float c = smix[2 * hc + d + hc * s] * inv * a.w_scale[2] + a.w_base[2 * hc + d + hc * s];
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 4));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
        const float ex = __expf(c - mx);
        float den = ex + __shfl_xor_sync(0xffffffffu, ex, 4);
        den += __shfl_xor_sync(0xffffffffu, den, 8);
        c = ex / den + a.hc_eps;
        {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        for (int it = 1; it < a.iters; ++it) {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 4);
            sm += __shfl_xor_sync(0xffffffffu, sm, 8);
            c = c / (a.hc_eps + sm);
            sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        if (b == 0 && lane < 16) a.comb[d * hc + s] = c;
    }
    __syncthreads();
    // ---- phase 3: mixed for this block's 128 values, the norm's sum of squares across the grid, x and q8_1
    float ss2 = 0.0f;
    if (tid < HC2_EB) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < hc; ++s) acc += (a.pre_use != nullptr ? a.pre_use[s] : s_pre[s]) * sR[s][tid];
        s_mixed[tid] = acc;
        ss2 = acc * acc;
    }
    ss2 = block_sum(ss2, sred);
    float* part2 = a.part + (size_t) nb * 25;
    if (tid == 0) part2[b] = ss2;
    grid_barrier(ctr, nb);
    float tot = 0.0f;
    for (int bb = 0; bb < (int) nb; ++bb) tot += __ldcg(&part2[bb]);
    const float inv2 = rsqrtf(tot / (float) n_embd + a.norm_eps);
    if (tid < HC2_EB) {   // warps 0..3: 32 consecutive values each
        const int e = e0 + tid;
        const float xv = s_mixed[tid] * inv2 * a.norm_w[e];
        a.x[e] = xv;
        q8_1_store_warp(xv, (block_q8_1*) a.xq + e / 32, lane);
    }
}

// ---- hc, take three: hc2's arithmetic with the latency taken out - every block issues its 24 weight rows' loads
// (8 bf16 per lane per stream segment, uint4) BEFORE it computes its R values, the partial sums are reduced by whole
// warps (lane = block) instead of one thread walking 32 of them, and the second barrier's total likewise.
// 32 blocks of 128 embedding values (n_embd 4096); with HIP up to 64 (DeepSeek V4.1's 5120: 40 - a lane reduces two
// partials there) and V4.1's lagged collapse (pre_use)
__global__ void __launch_bounds__(HC_THREADS) hc3_kernel(const __grid_constant__ HcArgs a) {
    const int hc = 4;
    const int n_embd = a.n_embd;
    const unsigned int nb = gridDim.x;   // 32 (up to 64)
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int e0 = b * HC2_EB;
    unsigned long long* ctr = (unsigned long long*) a.counter;
    __shared__ float sR[4][HC2_EB];
    __shared__ float sred[32];
    __shared__ float smix[HC_MIX + 11];
    __shared__ float s_pre[4];
    __shared__ float s_mixed[HC2_EB];
    // the weights first: warp w owns mix rows w, w + 8, w + 16; per row and stream segment d a lane loads the 4 bf16
    // values j = 4 * lane .. 4 * lane + 3 of [e0, e0 + 128)
    // (Q8_0: the 4 values' codes in wv.x and their block's scale in wv.y, as float bits)
    uint2 wv[3][4];
    const bool wq8 = a.w_fn_q8 != nullptr;
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        const int m = warp + 8 * r;
        const size_t w0 = (size_t) m * (hc * n_embd) + e0 + 4 * lane;
        const uint16_t* wr = a.w_fn + w0;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            if (wq8) {
                const size_t i = w0 + (size_t) d * n_embd;
                const uint8_t* bp = (const uint8_t*) a.w_fn_q8 + (i >> 5) * 34;
                const uint16_t* q = (const uint16_t*) (bp + 2) + ((i & 31) >> 1);
                wv[r][d].x = (uint32_t) q[0] | ((uint32_t) q[1] << 16);
                wv[r][d].y = __float_as_uint(__half2float(*(const __half*) bp));
            } else {
                wv[r][d] = *(const uint2*) (wr + (size_t) d * n_embd);
            }
        }
    }
    // the CPU LANE's answer for the last route (moe_cpu_wait, fused): block 0 alone polls the host-mapped answer and
    // passes its seq on in device memory (cpu_seq[2]), where the other blocks wait - each poll is a PCIe read the
    // CPU's memory serves while the lane reads its experts.  Then every block adds its own 128 values into
    // block_out (moe_cpu_wait's float add, written back as it leaves them) and reads them from shared memory
    __shared__ float s_bo[HC2_EB];
    __shared__ unsigned int s_need;
    const bool cpu = a.cpu_seq != nullptr && a.block_out != nullptr;
    if (cpu) {
        if (tid == 0) {
            s_need = *a.cpu_seq;
            if (s_need != 0u) {
                volatile unsigned int* seen = (volatile unsigned int*) a.cpu_seq + 2;   // (as moe_cpu_wait's)
                const long long t0 = clock64();   // bounded like moe_cpu_wait (~60 s)
                if (b == 0) {
                    const CpuAnswer* r = (const CpuAnswer*) a.cpu_ans + (s_need % kCpuAnswers);
                    while (r->seq != s_need) {
                        if (clock64() - t0 > (1ll << 36)) break;
                    }
                    __threadfence_system();
                    *seen = s_need;
                } else {
                    while (*seen != s_need) {
                        if (clock64() - t0 > (1ll << 36)) break;
                    }
                    __threadfence_system();
                }
            }
        }
        __syncthreads();
        if (tid < HC2_EB) {
            float o = a.block_out[e0 + tid];
            if (s_need != 0u)
                o += ((const volatile float*) ((const CpuAnswer*) a.cpu_ans + (s_need % kCpuAnswers))->part)[e0 + tid];
            ((float*) a.block_out)[e0 + tid] = o;
            s_bo[tid] = o;
        }
        __syncthreads();
    }
    // R_new for this block's 4 x 128 values (thread t: stream t / 64... two values per thread)
    float ss = 0.0f;
    for (int i = tid; i < 4 * HC2_EB; i += HC_THREADS) {
        const int d = i / HC2_EB, j = i - d * HC2_EB, e = e0 + j;
        float r;
        if (a.block_out != nullptr) {
            r = (cpu ? s_bo[j] : a.block_out[e]) * a.post_in[d];
#pragma unroll
            for (int s = 0; s < hc; ++s) r += a.comb_in[d * hc + s] * a.R_old[s * n_embd + e];
            a.R_new[d * n_embd + e] = r;
        } else {
            r = a.R_old[d * n_embd + e];
        }
        sR[d][j] = r;
        ss += r * r;
    }
    __syncthreads();
    float* part = a.part;   // [m][block]: 25 rows of nb partials (the 25th: the sum of squares)
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        const int m = warp + 8 * r;
        float acc = 0.0f;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const uint2 w = wv[r][d];
            const float* x = &sR[d][4 * lane];
            if (wq8) {
                const float sc = __uint_as_float(w.y);
                acc += sc * ((float) (int8_t) (w.x & 0xffu) * x[0] + (float) (int8_t) ((w.x >> 8) & 0xffu) * x[1] +
                             (float) (int8_t) ((w.x >> 16) & 0xffu) * x[2] + (float) (int8_t) (w.x >> 24) * x[3]);
            } else {
                acc += __uint_as_float(w.x << 16) * x[0] + __uint_as_float(w.x & 0xffff0000u) * x[1] +
                       __uint_as_float(w.y << 16) * x[2] + __uint_as_float(w.y & 0xffff0000u) * x[3];
            }
        }
        acc = warp_sum(acc);
        if (lane == 0) part[(size_t) m * nb + b] = acc;
    }
    ss = block_sum(ss, sred);
    if (tid == 0) part[(size_t) HC_MIX * nb + b] = ss;
    if (a.gram) {
        // the streams' Gram partials <R_s, R_t> (s <= t: 10) over this block's 128 values, one pair a warp - the
        // mixed vector's norm is sum_s sum_t pre_s pre_t <R_s, R_t>, so phase 3 needs no barrier of its own
        const int gs[10] = {0, 0, 0, 0, 1, 1, 1, 2, 2, 3}, gt[10] = {0, 1, 2, 3, 1, 2, 3, 2, 3, 3};
        for (int pr = warp; pr < 10; pr += 8) {
            float g = 0.0f;
            for (int j = lane; j < HC2_EB; j += 32) g += sR[gs[pr]][j] * sR[gt[pr]][j];
            g = warp_sum(g);
            if (lane == 0) part[(size_t) (HC_MIX + 1 + pr) * nb + b] = g;
        }
    }
    grid_barrier(ctr, nb);
    // the 25 (+ 10 Gram) totals: warp w reduces rows w, w + 8, ..., one partial per lane
    const int n_rows = a.gram ? HC_MIX + 11 : HC_MIX + 1;
    for (int m = warp; m < n_rows; m += 8) {
        float v = lane < (int) nb ? __ldcg(&part[(size_t) m * nb + lane]) : 0.0f;
        if (nb > 32u && lane + 32 < (int) nb) v += __ldcg(&part[(size_t) m * nb + lane + 32]);
        const float t = warp_sum(v);
        if (lane == 0) smix[m] = t;
    }
    __syncthreads();
    const float inv = rsqrtf(smix[24] / (float) (hc * n_embd) + a.norm_eps);
    if (warp == 0) {
        if (lane < hc) {
            const float pre_v = dsigmoid(smix[lane] * inv * a.w_scale[0] + a.w_base[lane]) + a.hc_eps;
            s_pre[lane] = pre_v;
            if (b == 0) {
                a.pre[lane] = pre_v;
                a.post[lane] = 2.0f * dsigmoid(smix[hc + lane] * inv * a.w_scale[1] + a.w_base[hc + lane]);
            }
        }
        const int l = lane & 15, d = l >> 2, s = l & 3;
        float c = smix[2 * hc + d + hc * s] * inv * a.w_scale[2] + a.w_base[2 * hc + d + hc * s];
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 4));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
        const float ex = __expf(c - mx);
        float den = ex + __shfl_xor_sync(0xffffffffu, ex, 4);
        den += __shfl_xor_sync(0xffffffffu, den, 8);
        c = ex / den + a.hc_eps;
        {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        for (int it = 1; it < a.iters; ++it) {
            float sm = c + __shfl_xor_sync(0xffffffffu, c, 4);
            sm += __shfl_xor_sync(0xffffffffu, sm, 8);
            c = c / (a.hc_eps + sm);
            sm = c + __shfl_xor_sync(0xffffffffu, c, 1);
            sm += __shfl_xor_sync(0xffffffffu, sm, 2);
            c = c / (a.hc_eps + sm);
        }
        if (b == 0 && lane < 16) a.comb[d * hc + s] = c;
    }
    __syncthreads();
    // the collapse weights: this sublayer's pre, or deepseek41's lagged one (the previous sublayer's)
    float pw[4];
#pragma unroll
    for (int s = 0; s < hc; ++s) pw[s] = a.pre_use != nullptr ? a.pre_use[s] : s_pre[s];
    float ss2 = 0.0f;
    if (tid < HC2_EB) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < hc; ++s) acc += pw[s] * sR[s][tid];
        s_mixed[tid] = acc;
        ss2 = acc * acc;
    }
    float tot;
    if (a.gram) {
        const int gs[10] = {0, 0, 0, 0, 1, 1, 1, 2, 2, 3}, gt[10] = {0, 1, 2, 3, 1, 2, 3, 2, 3, 3};
        tot = 0.0f;
#pragma unroll
        for (int pr = 0; pr < 10; ++pr)
            tot += (gs[pr] == gt[pr] ? 1.0f : 2.0f) * pw[gs[pr]] * pw[gt[pr]] * smix[HC_MIX + 1 + pr];
    } else {
        ss2 = block_sum(ss2, sred);
        float* part2 = a.part + (size_t) (HC_MIX + 1) * nb;
        if (tid == 0) part2[b] = ss2;
        grid_barrier(ctr, nb);
        float v = lane < (int) nb ? __ldcg(&part2[lane]) : 0.0f;
        if (nb > 32u && lane + 32 < (int) nb) v += __ldcg(&part2[lane + 32]);
        tot = warp_sum(v);
    }
    const float inv2 = rsqrtf(tot / (float) n_embd + a.norm_eps);
    if (tid < HC2_EB) {
        const int e = e0 + tid;
        const float xv = s_mixed[tid] * inv2 * a.norm_w[e];
        a.x[e] = xv;
        q8_1_store_warp(xv, (block_q8_1*) a.xq + e / 32, lane);
    }
}

__global__ void hc_post_kernel(const float* __restrict__ block_out, const float* __restrict__ R_old,
                               const float* __restrict__ post, const float* __restrict__ comb, int n_embd,
                               float* __restrict__ R_new) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 4 * n_embd) return;
    const int d = idx / n_embd, e = idx - d * n_embd;
    float acc = block_out[e] * post[d];
#pragma unroll
    for (int s = 0; s < 4; ++s) acc += comb[d * 4 + s] * R_old[s * n_embd + e];
    R_new[idx] = acc;
}

__global__ void head_prep_kernel(const float* __restrict__ R, const float* __restrict__ norm_w, float eps,
                                 int n_embd, float* __restrict__ x, block_q8_1* __restrict__ xq) {
    __shared__ float sred[32];
    __shared__ float sm[kMaxEmbd];
    const int tid = threadIdx.x, lane = tid & 31;
    float ss = 0.0f;
    for (int e = tid; e < n_embd; e += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < 4; ++s) acc += R[e + (size_t) n_embd * s];
        acc = acc / 4.0f;
        sm[e] = acc;
        ss += acc * acc;
    }
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) n_embd + eps);
    for (int e = tid; e < n_embd; e += blockDim.x) {
        const float v = sm[e] * inv * norm_w[e];
        x[e] = v;
        q8_1_store_warp(v, xq + e / 32, lane);
    }
}

__global__ void embed_streams_kernel(const float* __restrict__ emb, float* __restrict__ R, int n_embd) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 4 * n_embd) return;
    R[idx] = emb[idx % n_embd];
}

// token_embd rows for T tokens whose ids are read on the device (a host-mapped int or the argmax's output); the table
// may be host-mapped (a few KB a row).  Q8_0: a thread a block of 32.  Q5_K: a thread a column l of a super-block
// (ggml's dequantize_row_q5_K: 4 pairs of 32 per super-block).
__global__ void embed_q8_rows_kernel(const block_q8_0* __restrict__ table, const int* __restrict__ toks, int n_embd,
                                     float* __restrict__ out) {
    const int t = blockIdx.y;
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int nb = n_embd / 32;
    if (b >= nb) return;
    const block_q8_0* q = table + (size_t) toks[t] * nb + b;
    const float d = __half2float(q->d);
    float* y = out + (size_t) t * n_embd + (size_t) b * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) y[i] = d * (float) q->qs[i];
}

__device__ __forceinline__ void emb_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

__global__ void embed_q5k_rows_kernel(const block_q5_K* __restrict__ table, const int* __restrict__ toks, int n_embd,
                                      float* __restrict__ out) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // super-block * 32 + l
    const int nsb = n_embd / 256;
    const int sb = i >> 5, l = i & 31;
    if (sb >= nsb) return;
    const block_q5_K* q = table + (size_t) toks[t] * nsb + sb;
    const float2 dm = __half22float2(q->dm);
    const float d = dm.x, dmin = dm.y;
    float* y = out + (size_t) t * n_embd + (size_t) sb * 256;
    uint8_t u1 = 1, u2 = 2;
    const uint8_t qh = q->qh[l];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint8_t sc, m;
        emb_scale_min_k4(2 * j, q->scales, sc, m);
        const float d1 = d * sc, m1 = dmin * m;
        emb_scale_min_k4(2 * j + 1, q->scales, sc, m);
        const float d2 = d * sc, m2 = dmin * m;
        const uint8_t ql = q->qs[32 * j + l];
        y[64 * j + l] = d1 * (float) ((ql & 0xF) + ((qh & u1) ? 16 : 0)) - m1;
        y[64 * j + 32 + l] = d2 * (float) ((ql >> 4) + ((qh & u2) ? 16 : 0)) - m2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

// Q4_K rows (GLM-5.3-Flash UD-IQ2_XXS's token_embd): a thread a (super-block, l) - its 8 values, ggml's
// dequantize_row_q4_K order (64-value groups: low nibbles with scale 2j, high nibbles with 2j+1)
__global__ void embed_q4k_rows_kernel(const block_q4_K* __restrict__ table, const int* __restrict__ toks, int n_embd,
                                      float* __restrict__ out) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // super-block * 32 + l
    const int nsb = n_embd / 256;
    const int sb = i >> 5, l = i & 31;
    if (sb >= nsb) return;
    const block_q4_K* q = table + (size_t) toks[t] * nsb + sb;
    const float2 dm = __half22float2(q->dm);
    const float d = dm.x, dmin = dm.y;
    float* y = out + (size_t) t * n_embd + (size_t) sb * 256;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint8_t sc, m;
        emb_scale_min_k4(2 * j, q->scales, sc, m);
        const float d1 = __fmul_rn(d, sc), m1 = __fmul_rn(dmin, m);
        emb_scale_min_k4(2 * j + 1, q->scales, sc, m);
        const float d2 = __fmul_rn(d, sc), m2 = __fmul_rn(dmin, m);
        const uint8_t ql = q->qs[32 * j + l];
        // (two roundings, as ggml's dequantize_row_q4_K on the host under -ffp-contract=off: nvcc would fuse a * b - c)
        y[64 * j + l] = __fsub_rn(__fmul_rn(d1, (float) (ql & 0xF)), m1);
        y[64 * j + 32 + l] = __fsub_rn(__fmul_rn(d2, (float) (ql >> 4)), m2);
    }
}

// Q6_K rows (DeepSeek V4.1 UD-IQ1_S's token_embd): a thread a (super-block, l) - its 8 values, ggml's order
__global__ void embed_q6k_rows_kernel(const block_q6_K* __restrict__ table, const int* __restrict__ toks, int n_embd,
                                      float* __restrict__ out) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // super-block * 32 + l
    const int nsb = n_embd / 256;
    const int sb = i >> 5, l = i & 31;
    if (sb >= nsb) return;
    const block_q6_K* q = table + (size_t) toks[t] * nsb + sb;
    const float d = __half2float(*(const __half*) &q->d);
    float* y = out + (size_t) t * n_embd + (size_t) sb * 256;
    const int is = l / 16;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint8_t* ql = q->ql + 64 * h;
        const uint8_t qh = q->qh[32 * h + l];
        const int8_t* sc = q->scales + 8 * h;
        y[128 * h + l] = d * sc[is] * (float) ((int) ((ql[l] & 0xF) | ((qh & 3) << 4)) - 32);
        y[128 * h + l + 32] = d * sc[is + 2] * (float) ((int) ((ql[l + 32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
        y[128 * h + l + 64] = d * sc[is + 4] * (float) ((int) ((ql[l] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
        y[128 * h + l + 96] = d * sc[is + 6] * (float) ((int) ((ql[l + 32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
    }
}

// x_t -= s Q (Q^T x_t) for each row t (a refusal-direction edit applied to a block's output at run time - the same
// as W' = W - s Q Q^T W on the weight that wrote x, without re-quantizing it): Q [n][r] row-major, r <= 4
__global__ void abl_project_kernel(float* __restrict__ x, int ld, int n, const float* __restrict__ q, int r, float s,
                                   const int* __restrict__ on) {
    if (on != nullptr && *on == 0) return;
    float* xr = x + (size_t) blockIdx.x * ld;
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        for (int j = 0; j < r; ++j) c[j] += q[(size_t) i * r + j] * v;
    }
    __shared__ float sh[4][256];
    for (int j = 0; j < 4; ++j) sh[j][threadIdx.x] = c[j];
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        if (threadIdx.x < w)
            for (int j = 0; j < 4; ++j) sh[j][threadIdx.x] += sh[j][threadIdx.x + w];
        __syncthreads();
    }
    float cs[4];
    for (int j = 0; j < 4; ++j) cs[j] = s * sh[j][0];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float d = 0.0f;
        for (int j = 0; j < r; ++j) d += q[(size_t) i * r + j] * cs[j];
        xr[i] -= d;
    }
}

// ---------------------------------------------------------------- KDA
__global__ void __launch_bounds__(128) kda_prep_kernel(const __grid_constant__ KdaPrepArgs a) {
    const int h = blockIdx.x, role = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int hd = a.head_dim;
    const int d_inner = a.n_head * hd;
    __shared__ float sred[32];
    __shared__ float sv[128];
    if (role < 3) {
        const int c = h * hd + tid;
        const int hist = a.d_conv - 1;
        float* st = a.conv_state + (size_t) role * d_inner * hist + (size_t) hist * c;
        const float* w = a.conv_w[role] + (size_t) a.d_conv * c;
        const float xp = a.proj[role][c];
        float acc = 0.0f;
        for (int tap = 0; tap < hist; ++tap) acc += w[tap] * st[tap];
        acc += w[hist] * xp;
        float y = acc / (1.0f + __expf(-acc));
        for (int j = 0; j + 1 < hist; ++j) st[j] = st[j + 1];
        st[hist - 1] = xp;
        if (role < 2) {
            const float ss = block_sum(y * y, sred);
            y = y * rsqrtf(ss + 1e-6f);
        }
        a.out[role][c] = y;
        return;
    }
    // roles 3..6: g1 = lb * sigmoid(-(f_b . fa + dt_bias) * ssm_a[h]) for a quarter of the head's rows each;
    // roles 7..10: g2 = g_b . ga, likewise (a quarter = 32 rows: 8 per warp)
    const bool is_g1 = role < 7;
    const int quarter = (role - 3) & 3;
    const float* in = is_g1 ? a.fa : a.ga;
    const uint16_t* W = is_g1 ? a.f_b : a.g_b;
    const uint8_t* W8 = (const uint8_t*) (is_g1 ? a.f_b_q8 : a.g_b_q8);
    sv[tid] = in[tid];
    __syncthreads();
    const int rbase = quarter * (hd / 4) + warp * (hd / 16);
    for (int r = rbase; r < rbase + hd / 16; ++r) {
        const int c = h * hd + r;
        const uint16_t* wr = W + (size_t) c * hd;
        float acc = 0.0f;
        if (W8 != nullptr)
            for (int j = lane; j < hd; j += 32) acc += q8_0_at(W8, (size_t) c * hd + j) * sv[j];
        else
            for (int j = lane; j < hd; j += 32) acc += bf(wr[j]) * sv[j];
        acc = warp_sum(acc);
        if (lane == 0) {
            if (is_g1) {
                const float v = acc + a.dt_bias[c];
                a.g1[c] = a.lower_bound * dsigmoid(-v * a.ssm_a[h]);
            } else {
                a.g2[c] = acc;
            }
        }
    }
}

__global__ void __launch_bounds__(256) kda_rec_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                                      const float* __restrict__ v, const float* __restrict__ g1,
                                                      const float* __restrict__ beta_raw, float* __restrict__ state,
                                                      const float* __restrict__ g2, const float* __restrict__ norm_w,
                                                      float eps, int n_head, block_q8_1* __restrict__ out) {
    constexpr int HD = 128;
    const int h = blockIdx.x, t = threadIdx.x, j = t & (HD - 1), half = t >> 7, lane = t & 31;
    __shared__ float sq[HD], sk[HD], sdec[HD], sdel[HD];
    __shared__ float sacc[2][HD];
    __shared__ float sred[32];
    if (t < HD) {
        sq[t] = q[h * HD + t];
        sk[t] = k[h * HD + t];
        sdec[t] = __expf(g1[h * HD + t]);
    }
    const float b = 1.0f / (1.0f + expf(-beta_raw[h]));
    __syncthreads();
    float* Sh = state + (size_t) h * HD * HD;
    float reg[64];
    float A = 0.0f;
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) {
        const int i = half * 64 + ii;
        const float s = Sh[i * HD + j] * sdec[i];
        reg[ii] = s;
        A += s * sk[i];
    }
    sacc[half][j] = A;
    __syncthreads();
    if (half == 0) sdel[j] = b * (v[h * HD + j] - (sacc[0][j] + sacc[1][j]));
    __syncthreads();
    const float dl = sdel[j];
    float O = 0.0f;
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) {
        const int i = half * 64 + ii;
        const float s2 = reg[ii] + sk[i] * dl;
        Sh[i * HD + j] = s2;
        O += s2 * sq[i];
    }
    __syncthreads();
    sacc[half][j] = O;
    __syncthreads();
    const float o = (sacc[0][j] + sacc[1][j]) * rsqrtf((float) HD);
    // out gate over the head (threads 0..127 own j; the upper half mirrors and does not write)
    float ss = half == 0 ? o * o : 0.0f;
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) HD + eps);
    if (half == 0) {
        const float gv = o * inv * norm_w[j] * dsigmoid(g2[h * HD + j]);
        q8_1_store_warp(gv, out + h * (HD / 32) + (j >> 5), lane);
    }
}

// kda_rec split by columns (GB10 has 48 SMs: one block per head is 64 blocks = two waves on Spark and 64 of the 5090's
// 170 SMs). A column j of the state only needs its own column: blocks (head, quarter) take 32 columns each, a thread
// one (column, half) and the same 64-row loop as kda_rec_kernel, so A, the delta and O are the same sums in the same
// order. The head's RMS needs all 128 columns: kda_out_kernel does it after, with kda_rec_kernel's block_sum layout
// (256 threads, the upper half zero), so the result is bit-identical.
__global__ void __launch_bounds__(64) kda_rec_cols_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                                          const float* __restrict__ v, const float* __restrict__ g1,
                                                          const float* __restrict__ beta_raw, float* __restrict__ state,
                                                          float* __restrict__ o_out) {
    constexpr int HD = 128;
    const int h = blockIdx.x, t = threadIdx.x, jl = t & 31, half = t >> 5, j = blockIdx.y * 32 + jl;
    __shared__ float sq[HD], sk[HD], sdec[HD];
    __shared__ float sacc[2][32];
    for (int i = t; i < HD; i += 64) {
        sq[i] = q[h * HD + i];
        sk[i] = k[h * HD + i];
        sdec[i] = __expf(g1[h * HD + i]);
    }
    const float b = 1.0f / (1.0f + expf(-beta_raw[h]));
    __syncthreads();
    float* Sh = state + (size_t) h * HD * HD;
    float reg[64];
    float A = 0.0f;
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) {
        const int i = half * 64 + ii;
        const float s = Sh[i * HD + j] * sdec[i];
        reg[ii] = s;
        A += s * sk[i];
    }
    sacc[half][jl] = A;
    __syncthreads();
    const float dl = b * (v[h * HD + j] - (sacc[0][jl] + sacc[1][jl]));
    float O = 0.0f;
#pragma unroll
    for (int ii = 0; ii < 64; ++ii) {
        const int i = half * 64 + ii;
        const float s2 = reg[ii] + sk[i] * dl;
        Sh[i * HD + j] = s2;
        O += s2 * sq[i];
    }
    __syncthreads();
    sacc[half][jl] = O;
    __syncthreads();
    if (half == 0) o_out[h * HD + j] = (sacc[0][jl] + sacc[1][jl]) * rsqrtf((float) HD);
}

__global__ void __launch_bounds__(256) kda_out_kernel(const float* __restrict__ o_in, const float* __restrict__ g2,
                                                      const float* __restrict__ norm_w, float eps,
                                                      block_q8_1* __restrict__ out) {
    constexpr int HD = 128;
    const int h = blockIdx.x, t = threadIdx.x, j = t & (HD - 1), half = t >> 7, lane = t & 31;
    __shared__ float sred[32];
    const float o = o_in[h * HD + j];
    float ss = half == 0 ? o * o : 0.0f;
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) HD + eps);
    if (half == 0) {
        const float gv = o * inv * norm_w[j] * dsigmoid(g2[h * HD + j]);
        q8_1_store_warp(gv, out + h * (HD / 32) + (j >> 5), lane);
    }
}

// the DSA latent cache is FP16 (uint16_t bits): read / written in F32
__device__ __forceinline__ float lat_f(uint16_t v) { return __half2float(__ushort_as_half(v)); }
__device__ __forceinline__ uint16_t lat_h(float v) { return __half_as_ushort(__float2half(v)); }
// the INT8 latent records (lat8_rec_bytes): value c of the record at rec
__device__ __forceinline__ float lat8_f(const uint8_t* rec, int kv_lora, int c) {
    return (float) ((const int8_t*) rec)[c] * __half2float(((const __half*) (rec + kv_lora))[c >> 5]);
}
// latent value c of cell: FP16 rows or INT8 records
__device__ __forceinline__ float lat_any(const uint16_t* lat, bool q8, int kv_lora, int cell, int c) {
    return q8 ? lat8_f((const uint8_t*) lat + (size_t) lat8_rec_bytes(kv_lora) * cell, kv_lora, c)
              : lat_f(lat[(size_t) kv_lora * cell + c]);
}
// one latent row into its INT8 record: lane l of the warp holding values [32 w, 32 w + 32) stores value c
__device__ __forceinline__ void lat8_store_warp(float y, uint8_t* rec, int kv_lora, int c, int lane) {
    float amax = fabsf(y);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const __half d16 = __float2half(amax / 127.0f);
    const float d = __half2float(d16);
    const int q = d > 0.0f ? max(-127, min(127, __float2int_rn(y / d))) : 0;
    ((int8_t*) rec)[c] = (int8_t) q;
    if (lane == 0) ((__half*) (rec + kv_lora))[c >> 5] = d16;
}
// ---------------------------------------------------------------- DSA
__global__ void __launch_bounds__(512) dsa_prep_kernel(const __grid_constant__ DsaPrepArgs a) {
    __shared__ float sred[32];
    const int tid = threadIdx.x, lane = tid & 31;
    if (blockIdx.x == 0) {
        // q_a norm -> qr (f32) and qr_q (q8_1); q_lora <= 2048 (stride 512 keeps warps on 32-runs)
        float vals[4];
        float ss = 0.0f;
        int n = 0;
        for (int e = tid; e < a.q_lora; e += 512, ++n) {
            vals[n] = a.qr_raw[e];
            ss += vals[n] * vals[n];
        }
        ss = block_sum(ss, sred);
        const float inv = rsqrtf(ss / (float) a.q_lora + a.eps);
        n = 0;
        for (int e = tid; e < a.q_lora; e += 512, ++n) {
            const float y = vals[n] * inv * a.q_a_norm[e];
            a.qr[e] = y;
            q8_1_store_warp(y, (block_q8_1*) a.qr_q + e / 32, lane);
        }
        return;
    }
    if (blockIdx.x == 1) {
        float v = tid < a.kv_lora ? a.kv_raw[tid] : 0.0f;
        const float ss = block_sum(v * v, sred);
        const float inv = rsqrtf(ss / (float) a.kv_lora + a.eps);
        if (tid < a.kv_lora) {   // (kv_lora % 32 == 0: whole warps)
            const float y = v * inv * a.kv_norm[tid];
            if (a.lat_q8)
                lat8_store_warp(y, (uint8_t*) a.lat + (size_t) lat8_rec_bytes(a.kv_lora) * a.p, a.kv_lora, tid, tid & 31);
            else
                a.lat[(size_t) a.kv_lora * a.p + tid] = lat_h(y);
        }
        return;
    }
    // block 2: the indexer key (layer norm) and the compressor gate into their caches; the pool
    const int K = a.idx_key;
    const float v = tid < K ? a.ik_raw[tid] : 0.0f;
    const float s1 = block_sum(v, sred);
    const float s2 = block_sum(v * v, sred);
    const float mu = s1 / (float) K;
    const float inv = rsqrtf(s2 / (float) K - mu * mu + a.eps);
    if (tid < K) {
        a.ik_cache[(size_t) K * (a.p % a.ring) + tid] = (v - mu) * inv * a.k_norm_w[tid] + a.k_norm_b[tid];
        a.ig_cache[(size_t) K * (a.p % a.ring) + tid] = a.ig_raw[tid];
    }
    if ((a.p + 1) % a.kpool != 0) return;
    __syncthreads();
    __threadfence_block();
    const int pi = (a.p + 1) / a.kpool - 1;
    if (tid < K) {
        float lg[16];
        float mx = -INFINITY;
        for (int m = 0; m < a.kpool; ++m) {
            lg[m] = a.ig_cache[(size_t) tid + (size_t) K * ((pi * a.kpool + m) % a.ring)] + a.ape[tid + K * m];
            mx = fmaxf(mx, lg[m]);
        }
        float den = 0.0f;
        for (int m = 0; m < a.kpool; ++m) {
            lg[m] = expf(lg[m] - mx);
            den += lg[m];
        }
        float acc = 0.0f;
        for (int m = 0; m < a.kpool; ++m)
            acc += (lg[m] / den) * a.ik_cache[(size_t) tid + (size_t) K * ((pi * a.kpool + m) % a.ring)];
        a.pooled[(size_t) tid + (size_t) K * pi] = acc;
    }
}

__global__ void __launch_bounds__(256) dsa_score_kernel(const float* __restrict__ iq, const float* __restrict__ pooled,
                                                        const float* __restrict__ iw, int key_dim, int idx_heads,
                                                        int n_pools, float* __restrict__ score) {
    extern __shared__ float s_iq[];   // idx_heads * key_dim, then idx_heads weights
    float* s_iw = s_iq + idx_heads * key_dim;
    for (int i = threadIdx.x; i < idx_heads * key_dim; i += blockDim.x) s_iq[i] = iq[i];
    for (int i = threadIdx.x; i < idx_heads; i += blockDim.x) s_iw[i] = iw[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int p = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (p >= n_pools) return;
    const float* pk = pooled + (size_t) key_dim * p;
    float acc = 0.0f;
    for (int h = 0; h < idx_heads; ++h) {
        float dot = 0.0f;
        for (int e = lane; e < key_dim; e += 32) dot += s_iq[h * key_dim + e] * pk[e];
        dot = warp_sum(dot);
        acc += fmaxf(dot, 0.0f) * s_iw[h];
    }
    if (lane == 0) score[p] = acc;
}

__global__ void __launch_bounds__(1024) dsa_select_kernel(const float* __restrict__ score, int n_vis, int kpool,
                                                          int top_pools, int n_sel, int pos, int* __restrict__ cells) {
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) cells[i] = -1;
    __syncthreads();
    // the top pools in rank order (score descending, ties by the lower index), O(n_vis) - src/kernels/cuda/dsa_topk.cuh
    if (n_vis > 0) strata::kernels::dsa::select_top_pools(score, n_vis, min(top_pools, n_vis), kpool, cells);
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int m = 0; m < kpool - 1; ++m) {
            const int cell = n_vis * kpool + m;
            if (cell <= pos) cells[top_pools * kpool + m] = cell;
        }
    }
}

// One block per head: q_abs = wk_b_h . q_h (kv_lora rows of qk_nope), scores over the selected latents,
// softmax, ctx, out = wv_b_h . ctx (v_head rows of kv_lora), q8_1 of out.
__global__ void __launch_bounds__(256) mla_kernel(const float* __restrict__ q, const uint16_t* __restrict__ wk_b,
                                                  const uint16_t* __restrict__ wv_b, const uint16_t* __restrict__ lat,
                                                  const int* __restrict__ cells, int n_sel, int qk_nope, int kv_lora,
                                                  int v_head, block_q8_1* __restrict__ out, bool q8,
                                                  const uint8_t* __restrict__ wk8, const uint8_t* __restrict__ wv8) {
    extern __shared__ float smem[];
    float* s_q = smem;                    // qk_nope
    float* s_qa = s_q + qk_nope;          // kv_lora
    float* s_ctx = s_qa + kv_lora;        // kv_lora
    float* s_out = s_ctx + kv_lora;       // v_head
    float* s_p = s_out + v_head;          // n_sel
    __shared__ float sred[32];
    const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    for (int i = tid; i < qk_nope; i += blockDim.x) s_q[i] = q[(size_t) qk_nope * h + i];
    __syncthreads();
    const uint16_t* wk = wk_b + (size_t) qk_nope * kv_lora * h;
    for (int c = warp; c < kv_lora; c += nw) {
        const uint16_t* wr = wk + (size_t) qk_nope * c;
        const size_t w0 = (size_t) qk_nope * kv_lora * h + (size_t) qk_nope * c;
        float acc = 0.0f;
        for (int e = lane; e < qk_nope; e += 32) acc += (wk8 != nullptr ? q8_0_at(wk8, w0 + e) : bf(wr[e])) * s_q[e];
        acc = warp_sum(acc);
        if (lane == 0) s_qa[c] = acc;
    }
    __syncthreads();
    const float scale = rsqrtf((float) qk_nope);
    for (int s = warp; s < n_sel; s += nw) {
        const int cell = cells[s];
        float dot = -INFINITY;
        if (cell >= 0) {
            float acc = 0.0f;
            for (int e = lane; e < kv_lora; e += 32) acc += s_qa[e] * lat_any(lat, q8, kv_lora, cell, e);
            dot = warp_sum(acc) * scale;
        }
        if (lane == 0) s_p[s] = dot;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int s = tid; s < n_sel; s += blockDim.x) mx = fmaxf(mx, s_p[s]);
    mx = block_max(mx, sred);
    float den = 0.0f;
    for (int s = tid; s < n_sel; s += blockDim.x) {
        const float e = s_p[s] == -INFINITY ? 0.0f : expf(s_p[s] - mx);
        s_p[s] = e;
        den += e;
    }
    den = block_sum(den, sred);
    const float invd = 1.0f / den;
    for (int c = tid; c < kv_lora; c += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_sel; ++s) {
            const int cell = cells[s];
            if (cell >= 0) acc += (s_p[s] * invd) * lat_any(lat, q8, kv_lora, cell, c);
        }
        s_ctx[c] = acc;
    }
    __syncthreads();
    const uint16_t* wv = wv_b + (size_t) kv_lora * v_head * h;
    for (int vv = warp; vv < v_head; vv += nw) {
        const uint16_t* wr = wv + (size_t) kv_lora * vv;
        const size_t w0 = (size_t) kv_lora * v_head * h + (size_t) kv_lora * vv;
        float acc = 0.0f;
        for (int c = lane; c < kv_lora; c += 32) acc += (wv8 != nullptr ? q8_0_at(wv8, w0 + c) : bf(wr[c])) * s_ctx[c];
        acc = warp_sum(acc);
        if (lane == 0) s_out[vv] = acc;
    }
    __syncthreads();
    for (int vv = tid; vv < v_head; vv += blockDim.x)   // warp-aligned runs of 32
        q8_1_store_warp(s_out[vv], out + ((size_t) v_head * h + vv) / 32, lane);
}

// ---- MLA, take two: the absorbed projections as head-wise GEMVs over the whole GPU (warp per row, 32 rows per
// block - 4 per warp) and the attention proper in between
// out[h * R + r] = W[h][r][:] . in[h * K : h * K + K] (W BF16, rows of K); q8 != nullptr also writes the q8_1
template<bool Q8>
__global__ void __launch_bounds__(256) headwise_gemv_kernel(const void* __restrict__ W, const float* __restrict__ in,
                                                           int R, int K, float* __restrict__ out,
                                                           block_q8_1* __restrict__ q8) {
    __shared__ float s_o[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row0 = blockIdx.x * 32;   // R % 32 == 0: a block never straddles heads
    const int h = row0 / R;
    const float* x = in + (size_t) h * K;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int row = row0 + warp * 4 + j;
        const float s = Q8 ? row_dot_q8_0f((const uint8_t*) W + (size_t) row * (K / 32) * 34, x, K, lane)
                           : row_dot_bf16((const uint16_t*) W + (size_t) row * K, x, K, lane);
        if (lane == 0) s_o[warp * 4 + j] = s;
    }
    __syncthreads();
    if (warp == 0) {
        const float v = s_o[lane];
        if (out) out[row0 + lane] = v;
        if (q8) q8_1_store_warp(v, q8 + row0 / 32, lane);
    }
}

__global__ void __launch_bounds__(256) mla_attn_kernel(const float* __restrict__ q_abs, const uint16_t* __restrict__ lat,
                                                       const int* __restrict__ cells, int n_sel, int qk_nope,
                                                       int kv_lora, float* __restrict__ ctx, bool q8) {
    extern __shared__ float smem[];
    float* s_qa = smem;               // kv_lora
    float* s_p = s_qa + kv_lora;      // n_sel
    __shared__ float sred[32];
    const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    for (int i = tid; i < kv_lora; i += blockDim.x) s_qa[i] = q_abs[(size_t) kv_lora * h + i];
    __syncthreads();
    const float scale = rsqrtf((float) qk_nope);
    for (int s = warp; s < n_sel; s += nw) {
        const int cell = cells[s];
        float dot = -INFINITY;
        if (cell >= 0) {
            float acc = 0.0f;
            for (int e = lane; e < kv_lora; e += 32) acc += s_qa[e] * lat_any(lat, q8, kv_lora, cell, e);
            dot = warp_sum(acc) * scale;
        }
        if (lane == 0) s_p[s] = dot;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int s = tid; s < n_sel; s += blockDim.x) mx = fmaxf(mx, s_p[s]);
    mx = block_max(mx, sred);
    float den = 0.0f;
    for (int s = tid; s < n_sel; s += blockDim.x) {
        const float e = s_p[s] == -INFINITY ? 0.0f : expf(s_p[s] - mx);
        s_p[s] = e;
        den += e;
    }
    den = block_sum(den, sred);
    const float invd = 1.0f / den;
    for (int c = tid; c < kv_lora; c += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_sel; ++s) {
            const int cell = cells[s];
            if (cell >= 0) acc += (s_p[s] * invd) * lat_any(lat, q8, kv_lora, cell, c);
        }
        ctx[(size_t) kv_lora * h + c] = acc;
    }
}

// ---- MLA attention, split over the selected cells (flash-decoding): every head reads the SAME latents, so a block
// loads a chunk of MLA_CHUNK latent rows into shared memory once and scores it against ALL heads; per head it keeps
// its chunk's max, sum of exponentials and exp-weighted context, and the combine kernel merges the chunks.  At 2k
// selected cells this reads the latents once per layer instead of once per head (64x less).
// The grid is (chunks, head groups of MLA_HPB): ~20 chunks at 600 cells would leave most SMs idle with every head in
// one block (208 us a call on a V100); a group re-reads its chunk's latents, which is cheap next to the arithmetic.
// The latents stay FP16 in shared memory, as in the cache (34 KB at kv_lora 512; F32 rows were over Turing's 64 KB).
constexpr int MLA_CHUNK = 32;
constexpr int MLA_HPB = 16;
__global__ void __launch_bounds__(256) mla_split_kernel(const float* __restrict__ q_abs, const uint16_t* __restrict__ lat,
                                                        const int* __restrict__ cells, int n_sel, int n_head, int qk_nope,
                                                        int kv_lora, float* __restrict__ part, bool q8) {
    extern __shared__ float smem[];
    uint16_t* sL = (uint16_t*) smem;                   // MLA_CHUNK x kv_lora, FP16
    float* sS = smem + MLA_CHUNK * kv_lora / 2;        // MLA_HPB x MLA_CHUNK scores -> weights
    __shared__ int s_cell[MLA_CHUNK];
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    const int h0 = blockIdx.y * MLA_HPB, nh = min(MLA_HPB, n_head - h0);
    const int s0 = b * MLA_CHUNK;
    const int ns = min(MLA_CHUNK, n_sel - s0);
    if (tid < MLA_CHUNK) s_cell[tid] = tid < ns ? cells[s0 + tid] : -1;
    __syncthreads();
    for (int i = tid; i < MLA_CHUNK * kv_lora; i += blockDim.x) {
        const int s = i / kv_lora, c = i - s * kv_lora;
        const int cell = s_cell[s];
        sL[i] = cell < 0 ? (uint16_t) 0 : q8 ? lat_h(lat_any(lat, true, kv_lora, cell, c)) : lat[(size_t) kv_lora * cell + c];
    }
    __syncthreads();
    const float scale = rsqrtf((float) qk_nope);
    // scores: a warp per head, the head's q in registers (kv_lora / 32 per lane), one warp reduction per cell
    for (int hh = warp; hh < nh; hh += nw) {
        float q[16];
        const float* qh = q_abs + (size_t) kv_lora * (h0 + hh);
#pragma unroll
        for (int j = 0; j < 16; ++j) q[j] = (lane + 32 * j) < kv_lora ? qh[lane + 32 * j] : 0.0f;
        for (int s = 0; s < MLA_CHUNK; ++s) {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j)
                if (lane + 32 * j < kv_lora) acc += q[j] * lat_f(sL[s * kv_lora + lane + 32 * j]);
            acc = warp_sum(acc);
            if (lane == 0) sS[hh * MLA_CHUNK + s] = s_cell[s] >= 0 ? acc * scale : -INFINITY;
        }
    }
    __syncthreads();
    // per head: the chunk's max and sum; the scores become exp-weights
    float* pm = part + ((size_t) b * n_head + h0) * (kv_lora + 2);
    for (int hh = warp; hh < nh; hh += nw) {
        const float v = lane < MLA_CHUNK ? sS[hh * MLA_CHUNK + lane] : -INFINITY;
        const float m = warp_max(v);
        const float e = (v == -INFINITY) ? 0.0f : expf(v - m);
        const float l = warp_sum(e);
        if (lane < MLA_CHUNK) sS[hh * MLA_CHUNK + lane] = e;
        if (lane == 0) {
            pm[(size_t) hh * (kv_lora + 2) + kv_lora] = m;
            pm[(size_t) hh * (kv_lora + 2) + kv_lora + 1] = l;
        }
    }
    __syncthreads();
    // exp-weighted context per head over this chunk
    for (int i = tid; i < nh * kv_lora; i += blockDim.x) {
        const int hh = i / kv_lora, c = i - hh * kv_lora;
        float acc = 0.0f;
#pragma unroll 8
        for (int s = 0; s < MLA_CHUNK; ++s) acc += sS[hh * MLA_CHUNK + s] * lat_f(sL[s * kv_lora + c]);
        pm[(size_t) hh * (kv_lora + 2) + c] = acc;
    }
}

__global__ void mla_combine_kernel(const float* __restrict__ part, int n_chunks, int n_head, int kv_lora,
                                   float* __restrict__ ctx) {
    const int h = blockIdx.x;
    float M = -INFINITY;
    for (int b = 0; b < n_chunks; ++b) M = fmaxf(M, part[((size_t) b * n_head + h) * (kv_lora + 2) + kv_lora]);
    float L = 0.0f;
    for (int b = 0; b < n_chunks; ++b) {
        const float* pb = part + ((size_t) b * n_head + h) * (kv_lora + 2);
        if (pb[kv_lora] != -INFINITY) L += expf(pb[kv_lora] - M) * pb[kv_lora + 1];
    }
    const float invL = L > 0.0f ? 1.0f / L : 0.0f;
    for (int c = threadIdx.x; c < kv_lora; c += blockDim.x) {
        float acc = 0.0f;
        for (int b = 0; b < n_chunks; ++b) {
            const float* pb = part + ((size_t) b * n_head + h) * (kv_lora + 2);
            if (pb[kv_lora] != -INFINITY) acc += expf(pb[kv_lora] - M) * pb[c];
        }
        ctx[(size_t) kv_lora * h + c] = acc * invL;
    }
}

__global__ void swiglu_q8_kernel(const float* __restrict__ gate, const float* __restrict__ up, float limit, int n,
                                 block_q8_1* __restrict__ hq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;   // n % 32 == 0
    const float g = fminf(gate[i], limit);
    const float u = fminf(fmaxf(up[i], -limit), limit);
    q8_1_store_warp((g / (1.0f + expf(-g))) * u, hq + i / 32, threadIdx.x & 31);
}

// ---------------------------------------------------------------- routed experts
constexpr int ROUTE_THREADS = 320;

// the k largest of sel[0..E) in descending order, ties by the lower index - exactly the stable-descending rank rule
// (rank(e) = #{f: sel f > sel e} + #{f < e: sel f == sel e}) - as k rounds of a block argmax; sel is consumed
// (the picked entries become -inf).  All threads of the block call it.
__device__ void topk_argmax(float* sel, int E, int k, int* out, float* s_bv, int* s_bi) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int e = tid; e < E; e += blockDim.x) {
            const float v = sel[e];
            if (v > bv || (v == bv && e < bi)) {
                bv = v;
                bi = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            s_bv[warp] = bv;
            s_bi[warp] = bi;
        }
        __syncthreads();
        if (tid == 0) {
            float v = s_bv[0];
            int i = s_bi[0];
            for (int w = 1; w < nw; ++w)
                if (s_bv[w] > v || (s_bv[w] == v && s_bi[w] < i)) {
                    v = s_bv[w];
                    i = s_bi[w];
                }
            out[r] = i >= 0 && i < E && isfinite(v) ? i : -1;
            if (out[r] >= 0) sel[i] = -INFINITY;
        }
        __syncthreads();
    }
}

struct RouteArgs {
    const float* logits;
    const float* bias;
    int n_expert, k;
    float w_scale;
    int norm_w;
    int layer;
    const float* x;
    int n_embd;
    MoeDev d;
    const float* sh_gate;
    const float* sh_up;
    float sh_limit;
    int n_ff_sh;
    block_q8_1* sh_hq;
    const float* pred_logits;   // the next layer's router on this layer's input (nullptr: no prediction)
    const float* pred_bias;
    int max_pf;                 // prefetch at most this many predicted non-resident experts of the next layer
    const float* ahead_logits;  // n_ahead x n_expert: the next layers' routers on this layer's input (LOOKAHEAD)
    const float* ahead_bias[kAhead];
    int n_ahead;
    int skip_from;              // experts not in VRAM at route rank >= skip_from are left out (no fetch, no wait):
                                // the draft block's, and the trunk's ROUTE TAIL SKIP (0: all of them; >= k: none)
    unsigned long long cpu_plan;   // the CPU LANE: of f RAM-tier experts, (plan >> 4 f) & 15 go to the host (0: off)
    int promote_min;            // STRATA_GLM_PROMOTE_MIN: keep a fetched expert only when its aged route count
                                // clears this; 0 keeps the old rule (a spare, if one is free)
    int pf_rank;                // STRATA_GLM_PREFETCH_RANK: prefetch only the prediction's first pf_rank ranks
    int gating;                 // 0: sigmoid (glm5-next), 1: sqrt(softplus) (deepseek4), 2: softmax top-k (qwen35moe)
    const int* hash_ids;        // deepseek4's hash-routed layers: the token's k expert ids (no selection)
    float skip_w;               // ... and those not in VRAM whose share of the route's probability is below skip_w
    int early_x;                // 1: the request's x (16 KB, host-mapped) is written at the kernel's start, beside
                                // the scores and the plan, instead of after them (STRATA_GLM_ROUTE_EARLY)
    int skip_disk;              // 1: the tail skip leaves out only the experts in no tier (disk-only), not RAM ones
    const block_q8_1* xq;       // x's q8_1 (nullptr: none): copied into the request beside x, for the iGPU lane
    float affinity;             // STRATA_GLM_ROUTE_AFFINITY (Project Maya #252): added to a VRAM-resident expert's
                                // selection score - its weight stays its own probability.  LOSSY (0: off)
    float affinity_ram;         // STRATA_GLM_ROUTE_AFFINITY_RAM: the same for an expert in the RAM tier (not on disk
                                // only); a VRAM one gets the larger of the two.  LOSSY (0: off)
    int prof;                   // STRATA_GLM_ROUTE_PROF: per-phase cycle sums (diagnostics)
    int win_more;               // MoeRequest::win_more (a batched verify window's column: more routes of the layer follow)
};
// the request's xq (x's q8_1 as the device computed it: 36-byte blocks, copied as words) and its flag; the system
// fence before the request's seq covers it as it does x
__device__ __forceinline__ void route_put_xq(const RouteArgs& a, MoeRequest* rq, int tid) {
    if (tid == 0) rq->xq_ok = a.xq != nullptr && a.n_embd % 32 == 0 && a.n_embd <= kMaxEmbd ? 1 : 0;
    if (a.xq == nullptr || a.n_embd % 32 != 0 || a.n_embd > kMaxEmbd) return;
    const uint32_t* s = (const uint32_t*) a.xq;
    uint32_t* d = (uint32_t*) rq->xq;
    for (int e = tid; e < a.n_embd / 32 * 9; e += ROUTE_THREADS) d[e] = s[e];
}

// the router's per-expert probability (gating 2, softmax: the logit itself - the top-k by logit is the top-k by
// softmax, and the plan takes the softmax over the k chosen, which is the renormalised top-k of the full softmax)
__device__ __forceinline__ float route_score(float x, int gating) {
    if (gating == 1) return sqrtf(x > 20.0f ? x : logf(1.0f + expf(x)));   // ggml's softplus, then sqrt
    if (gating == 2) return x;
    return 1.0f / (1.0f + expf(-x));
}

// the top-k of sel[0, E) (E <= 512) by one warp: 16 scores a lane in registers, k rounds of shuffles; the same
// selection rule and ties as topk_argmax (the larger score, then the lower index); invalid ranks -> -1
__device__ void warp_topk_scan(const float* sel, int E, int k, int* out) {
    const int lane = threadIdx.x & 31;
    float v[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) v[j] = lane + 32 * j < E ? sel[lane + 32 * j] : -INFINITY;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            if (lane + 32 * j < E && (v[j] > bv || (v[j] == bv && lane + 32 * j < bi))) {
                bv = v[j];
                bi = lane + 32 * j;
            }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
#pragma unroll
        for (int j = 0; j < 16; ++j)
            if (lane + 32 * j == bi) v[j] = -INFINITY;
        if (lane == 0) out[r] = bi >= 0 && bi < E && isfinite(bv) ? bi : -1;
    }
}

// CUDA sm_80+: the same selection with the warp's REDUX instructions - a round is each lane's best of its 16 (a
// tree, the lower entry on a tie), then the warp's max key and the lowest index among the lanes holding it: two
// reductions instead of five shuffle pairs.  The scores as order-preserving keys (-0 as +0: equal scores, equal
// keys).  The same ids in the same order as the scan (both are the k first of one total order; checked against it
// over every E / k / tie / -inf / -0 pattern).  RTX 5090, Qwen3.8's k = 10 of 512: 8.8k -> 5.0k cycles
#if !defined(STRATA_USE_HIP)
__device__ __forceinline__ unsigned topk_key(float v) {
    const unsigned u = __float_as_uint(v == 0.0f ? 0.0f : v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
#endif
__device__ void warp_topk(const float* sel, int E, int k, int* out) {
#if !defined(STRATA_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int lane = threadIdx.x & 31;
    unsigned key[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int e = lane + 32 * j;
        key[j] = e < E ? topk_key(sel[e]) : 0u;   // (past E or taken: 0 - below every score's key, -inf's is 0x007fffff)
    }
    for (int r = 0; r < k; ++r) {
        unsigned bk[8], bj[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const bool t = key[2 * j + 1] > key[2 * j];
            bk[j] = t ? key[2 * j + 1] : key[2 * j];
            bj[j] = t ? 2 * j + 1 : 2 * j;
        }
#pragma unroll
        for (int w = 4; w >= 1; w >>= 1)
#pragma unroll
            for (int j = 0; j < w; ++j) {
                const bool t = bk[2 * j + 1] > bk[2 * j];
                bk[j] = t ? bk[2 * j + 1] : bk[2 * j];
                bj[j] = t ? bj[2 * j + 1] : bj[2 * j];
            }
        const unsigned m = __reduce_max_sync(0xffffffffu, bk[0]);
        const unsigned mi = (unsigned) lane + 32u * bj[0];
        const unsigned bi = __reduce_min_sync(0xffffffffu, bk[0] == m ? mi : 0xffffffffu);
        if (mi == bi) {
#pragma unroll
            for (int j = 0; j < 16; ++j)
                if (j == (int) bj[0]) key[j] = 0u;
        }
        // (-inf, or nothing left: no expert, as the scan's non-finite best)
        if (lane == 0) out[r] = bi < (unsigned) E && m > 0x007fffffu ? (int) bi : -1;
    }
#else
    warp_topk_scan(sel, E, k, out);
#endif
}

// STRATA_GLM_ROUTE_PROF=1 (diagnostics, CUDA): thread 0 of the route block sums SM cycles per phase (0 start, 1 x / prev
// error, 2 scores, 3 top-k, 4 next-layer prediction, 5 lookahead + table loads, 6 plan, 7 request + fence, 8 the seq
// publish) and prints the means every 43,000 routes
#if !defined(STRATA_USE_HIP)
__device__ unsigned long long g_route_prof[10];
#define ROUTE_PROF(i)                                                                                         \
    if (a.prof && tid == 0) {                                                                                 \
        const unsigned long long t_ = clock64();                                                             \
        if ((i) > 0) atomicAdd(&g_route_prof[(i)], t_ - prof_t);                                              \
        prof_t = t_;                                                                                          \
        if ((i) == 8 && (atomicAdd(&g_route_prof[0], 1ull) + 1) % 43000ull == 0ull) {                         \
            const double n = (double) g_route_prof[0];                                                        \
            printf("route prof (cycles a route, %.0f routes): x %.0f | scores %.0f | topk %.0f | pred %.0f | ahead+tab %.0f | plan %.0f | rq+fence %.0f | publish %.0f\n", n, \
                   g_route_prof[1] / n, g_route_prof[2] / n, g_route_prof[3] / n, g_route_prof[4] / n,         \
                   g_route_prof[5] / n, g_route_prof[6] / n, g_route_prof[7] / n, g_route_prof[8] / n);       \
        }                                                                                                     \
    }
#else
#define ROUTE_PROF(i)
#endif
__global__ void __launch_bounds__(ROUTE_THREADS) moe_route_kernel(const __grid_constant__ RouteArgs a) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int tid = threadIdx.x, lane = tid & 31;
    if (blockIdx.x > 0) {
        // the shared expert's swiglu + q8_1 (warp-aligned 32-runs)
        const int i = (blockIdx.x - 1) * ROUTE_THREADS + tid;
        if (i >= a.n_ff_sh) return;
        const float g = fminf(a.sh_gate[i], a.sh_limit);
        const float u = fminf(fmaxf(a.sh_up[i], -a.sh_limit), a.sh_limit);
        q8_1_store_warp((g / (1.0f + expf(-g))) * u, a.sh_hq + i / 32, lane);
        return;
    }
    const int E = a.n_expert;
    unsigned long long prof_t = 0;
    ROUTE_PROF(0)
    __shared__ float s_p[512], s_sel[512];
    __shared__ int s_ids[kMaxK];
    __shared__ float s_w[kMaxK];
    __shared__ unsigned int s_mask, s_seq;
    __shared__ int bad;
    // an earlier route's error: its device copy (the host-mapped flag is a PCIe round trip, which waited ~100 us
    // behind the tier moves' copies) - read now, used at the bad check below
    const int prev_err = tid != 0                       ? 0
                         : a.d.route_error_d != nullptr ? *a.d.route_error_d
                         : a.d.route_error != nullptr   ? *a.d.route_error
                                                        : 0;
    if (tid == 0) bad = 0;
    // EARLY X: this route's request slot is known now ((*seq + 1) % kRingSize: only this kernel moves seq, and routes
    // are stream-ordered), so its x goes out over PCIe while the scores, the top-k and the plan run - it was ~4.4 us
    // of scalar stores at the end, on the lane's critical path.  The system fence before seq below covers it
    if (a.early_x) {
        MoeRequest* rqe = (MoeRequest*) a.d.ring + ((*a.d.seq + 1) % kRingSize);
        const float4* xs = (const float4*) a.x;
        float4* xd = (float4*) rqe->x;
        for (int e = tid; e < a.n_embd / 4; e += ROUTE_THREADS) xd[e] = xs[e];
        route_put_xq(a, rqe, tid);
    }
    __syncthreads();
    ROUTE_PROF(1)
    // the CPU lane's frequency counts age: every 4096 routes all of them halve (the whole block, before any use)
    if (a.cpu_plan != 0ull && a.d.dcnt != nullptr && ((*a.d.seq + 1) & 4095u) == 0u)
        for (int i = tid; i < a.d.n_keys; i += ROUTE_THREADS) a.d.dcnt[i] >>= 1;
    for (int e = tid; e < E; e += ROUTE_THREADS) {
        const float p = route_score(a.logits[e], a.gating);
        s_p[e] = p;
        s_sel[e] = a.bias ? p + a.bias[e] : p;
        if ((a.affinity > 0.0f || a.affinity_ram > 0.0f) && a.hash_ids == nullptr) {
            const size_t key = (size_t) a.layer * E + e;
            if (a.d.tab[key] != 0ull) s_sel[e] += fmaxf(a.affinity, a.affinity_ram);
            else if (a.affinity_ram > 0.0f && a.d.tab[a.d.n_keys + key] != 0ull) s_sel[e] += a.affinity_ram;
        }
        if (!isfinite(a.logits[e]) || !isfinite(s_sel[e])) atomicMax(&bad, 1);
    }
    __shared__ float s_bv[32];
    __shared__ int s_bi[32];
    __syncthreads();
    ROUTE_PROF(2)
    if (a.hash_ids != nullptr) {
        if (tid < a.k) s_ids[tid] = a.hash_ids[tid];
        __syncthreads();
    } else if (E <= 512) {
        if (tid < 32) warp_topk(s_sel, E, a.k, s_ids);
        __syncthreads();
    } else {
        topk_argmax(s_sel, E, a.k, s_ids, s_bv, s_bi);
    }
    if (tid == 0)
        for (int i = 0; i < a.k; ++i)
            if (s_ids[i] < 0 || s_ids[i] >= E) atomicMax(&bad, 1);
    ROUTE_PROF(3)
    // the next layer's predicted top-k (same selection rule, its own bias)
    __shared__ int s_pred[kMaxK];
    if (tid < kMaxK) s_pred[tid] = -1;
    if (a.pred_logits != nullptr) {
        __syncthreads();
        for (int e = tid; e < E; e += ROUTE_THREADS) {
            const float p = route_score(a.pred_logits[e], a.gating);
            s_sel[e] = a.pred_bias ? p + a.pred_bias[e] : p;   // s_sel is free again: the selection is done
            if (!isfinite(a.pred_logits[e]) || !isfinite(s_sel[e])) atomicMax(&bad, 2);
        }
        __syncthreads();
        topk_argmax(s_sel, E, a.k, s_pred, s_bv, s_bi);
        if (tid == 0)
            for (int i = 0; i < a.k; ++i)
                if (s_pred[i] < 0 || s_pred[i] >= E) atomicMax(&bad, 2);
    }
    ROUTE_PROF(4)
    // LOOKAHEAD: one warp per later layer, its top-k in registers (E <= 512; the same selection rule and ties)
    __shared__ short s_ah[kAhead][kMaxK];
    if ((tid >> 5) < a.n_ahead) {
        const int w = tid >> 5;
        const float* lg = a.ahead_logits + (size_t) w * E;
        const float* bs = a.ahead_bias[w];
        float v[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int e = lane + 32 * j;
            float x = -INFINITY;
            if (e < E) {
                const float p = route_score(lg[e], a.gating);
                x = bs ? p + bs[e] : p;
                if (!isfinite(lg[e]) || !isfinite(x)) atomicMax(&bad, 3);
            }
            v[j] = x;
        }
        for (int r = 0; r < a.k; ++r) {
            float bv = -INFINITY;
            int bi = 0x7fffffff;
#pragma unroll
            for (int j = 0; j < 16; ++j)
                if (lane + 32 * j < E && (v[j] > bv || (v[j] == bv && lane + 32 * j < bi))) {
                    bv = v[j];
                    bi = lane + 32 * j;
                }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) {
                    bv = ov;
                    bi = oi;
                }
            }
#pragma unroll
            for (int j = 0; j < 16; ++j)
                if (lane + 32 * j == bi) v[j] = -INFINITY;
            if (lane == 0) {
                const bool valid = bi >= 0 && bi < E && isfinite(bv);
                s_ah[w][r] = (short) (valid ? bi : -1);
                if (!valid) atomicMax(&bad, 3);
            }
        }
    }
    // (4) the route's table entries, one thread an expert: tid 0's plan below then reads shared memory, not k
    // dependent global loads (the RAM-tier table, the VRAM table, the route counts)
    __shared__ unsigned long long s_tab[kMaxK], s_rtab[kMaxK];
    __shared__ unsigned int s_cnt[kMaxK], s_seq0;
    if (tid < a.k && s_ids[tid] >= 0 && s_ids[tid] < E) {
        const size_t key = (size_t) a.layer * E + s_ids[tid];
        s_tab[tid] = a.d.tab[key];
        s_rtab[tid] = a.d.tab[a.d.n_keys + key];
        s_cnt[tid] = a.cpu_plan != 0ull && a.skip_from > 0 && a.d.dcnt != nullptr ? ++a.d.dcnt[key] : 0u;
    }
    if (tid == 32) s_seq0 = *a.d.seq;
    if (tid == 0 && prev_err != 0) atomicMax(&bad, 1);
    __syncthreads();
    ROUTE_PROF(5)
    if (bad != 0) {
        if (tid == 0) {
            // No probability/table lookup or DMA for an invalid route. Still publish and retire its sequence.
            if (a.d.route_error_d != nullptr && *a.d.route_error_d == 0) *a.d.route_error_d = 4 * a.layer + bad;
            if (a.d.route_error != nullptr && *a.d.route_error == 0) *a.d.route_error = 4 * a.layer + bad;
            *a.d.wait_seq = 0u;
            if (a.d.cpu_seq != nullptr) {
                a.d.cpu_seq[1] = a.d.cpu_seq[0];
                a.d.cpu_seq[0] = 0u;
            }
            *a.d.cpu_flag = 0;
            *a.d.pf_n = 0;
            for (int i = 0; i < a.k; ++i) {
                a.d.plan_ptr[i] = a.d.fetch_src[i] = 0ull;
                a.d.plan_w[i] = 0.0f;
                a.d.plan_id[i] = -1;
            }
            const unsigned int seq = *a.d.seq + 1;
            *a.d.seq = seq;
            MoeRequest* rq = (MoeRequest*) a.d.ring + (seq % kRingSize);
            rq->layer = a.layer;
            rq->error = bad;
            __threadfence_system();
            rq->seq = seq;
            __threadfence_system();
        }
        return;
    }
    __shared__ unsigned int s_fetch, s_promo, s_cpu, s_skip;
    __shared__ unsigned long long s_pp[kMaxK], s_cs[kMaxK];
    __shared__ int s_npf, s_pfid[kMaxK];
    __shared__ unsigned long long s_pfptr[kMaxK];
    // The PLAN, one lane an entry (warp 0): it was one thread walking the k entries, ~6 us of dependent shared and
    // global loads and FP64 a route on the GPU's critical path (route -> lane request, route -> the expert kernels).
    // The same decisions, made in parallel: the sum in route order, the host set as the to_cpu smallest by (route
    // count, later entry first) - what picking the coldest to_cpu times over gives - and the spares by prefix count.
    if (tid < 32) {
        const int i = lane;
        const bool on = i < a.k;
        float p = on ? s_p[s_ids[i]] : 0.0f;
        if (a.gating == 2) {   // softmax over the k chosen logits
            float mx = on ? p : -INFINITY;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            p = on ? expf(p - mx) : 0.0f;
        }
        double sum = 0.0;
#pragma unroll
        for (int j = 0; j < kMaxK; ++j) {
            const float pj = __shfl_sync(0xffffffffu, p, j);
            if (j < a.k) sum += (double) pj;
        }
        const float inv = (float) (a.norm_w ? 1.0 / fmax(sum, 6.103515625e-5) : 1.0);
        // the entries a miss leaves out: rank >= skip_from, or a share of the route's probability below skip_w
        const bool direct_i = on && a.d.ram_direct != 0 && s_tab[i] == 0ull && s_rtab[i] != 0ull;   // (read in place)
        const bool cut_i = on && !direct_i && (!a.skip_disk || s_rtab[i] == 0ull) &&
                           (i >= a.skip_from || (a.skip_w > 0.0f && (double) p < (double) a.skip_w * sum));
        unsigned long long* rtab = a.d.tab + a.d.n_keys;
        unsigned long long* spares = a.d.tab + 2 * (size_t) a.d.n_keys + (size_t) a.layer * kSpares;
        const unsigned long long tab_i = on ? s_tab[i] : 0ull, rtab_i = on ? s_rtab[i] : 0ull;
        // the CPU LANE: of this route's f RAM-tier experts, cpu_take(f) (the plan's split table) are computed by the
        // host from their RAM-tier blobs while the device pulls the others over PCIe - the COLDEST by the route counts
        // (ties: the later in route order), so the hot ones are still promoted into VRAM
        bool host_i = false;
        if (a.cpu_plan != 0ull && a.skip_from > 0) {
            const bool cand = on && !cut_i && !direct_i && tab_i == 0ull && rtab_i != 0ull;
            const unsigned int cm = (unsigned int) __ballot_sync(0xffffffffu, cand);
            const int f_ram = __popc(cm);
            const int f_tab = f_ram < 15 ? f_ram : 15;   // (cpu_take's: the split table holds f = 0..15)
            const int to_cpu = min((int) ((a.cpu_plan >> (4 * f_tab)) & 15ull), f_ram);
            const unsigned int c_i = on ? s_cnt[i] : 0u;
            int rank = 0;
#pragma unroll
            for (int j = 0; j < kMaxK; ++j) {
                const unsigned int cj = __shfl_sync(0xffffffffu, c_i, j);
                if (((cm >> j) & 1u) && (cj < c_i || (cj == c_i && j > i))) ++rank;
            }
            const bool helper_i = cand && rank >= to_cpu && a.d.helper_owned != nullptr &&
                                  a.d.helper_owned[(size_t) a.layer * E + s_ids[i]] != 0;
            // a hot helper-owned expert may still be promoted (#24): without it the VRAM tier stops adapting during
            // decode (the helper serves exactly the hot RAM-tier experts) and the next prompt is slower. One a route,
            // only into a spare no other miss of this route will take
            bool hp_i = false;
            if (a.d.helper_owned != nullptr && a.d.helper_promote != 0u && a.promote_min >= 0) {
                const bool hot = helper_i && s_cnt[i] >= a.d.helper_promote;
                const unsigned int hm = (unsigned int) __ballot_sync(0xffffffffu, hot);
                const bool other = on && !cut_i && !direct_i && tab_i == 0ull && !(cand && (rank < to_cpu || helper_i));
                const int n_other = __popc((unsigned int) __ballot_sync(0xffffffffu, other));
                if (hm != 0u) {
                    const unsigned long long* spl = a.d.tab + 2 * (size_t) a.d.n_keys + (size_t) a.layer * kSpares;
                    int navail = 0;
#pragma unroll
                    for (int j = 0; j < kSpares; ++j) navail += spl[j] != 0ull ? 1 : 0;
                    hp_i = hot && navail - n_other >= 1 && (hm & ((1u << i) - 1u)) == 0u;
                }
            }
            host_i = cand && (rank < to_cpu || (helper_i && !hp_i));
        }
        const float w = (float) ((double) p * inv * a.w_scale);
        const size_t key = on ? (size_t) a.layer * E + s_ids[i] : 0;
        unsigned long long ptr = tab_i, src = 0ull, pp = 0ull, cs = 0ull;
        bool b_skip = false, b_cpu = false, needs = false;
        if (on && ptr == 0ull) {
            if (cut_i) {
                // left out: the expert kernels skip a null plan entry (a draft only has to be a good guess; a trunk
                // route's low-ranked expert adds little to the token)
                b_skip = true;
            } else if (direct_i) {
                // UMA DIRECT: the expert kernels read the RAM-tier blob where it is (host-mapped, the same DRAM)
                ptr = rtab_i;
            } else {
                src = rtab_i;
                if (src != 0ull && host_i) {
                    // to the host: a null plan entry (the expert kernels skip it), no spare, no fetch
                    b_cpu = true;
                    cs = src;
                    src = 0ull;
                } else {
                    needs = true;
                }
            }
        }
        // not in VRAM: land it in a spare slot of this layer (it becomes resident: the table says so from here on)
        // or, with none left, in this entry's scratch slot.  promote_min > 1: keep it only when its aged route count
        // clears the bar - one-offs take scratch and do not churn the tier.  The r-th entry that wants one (route
        // order) takes the r-th free spare.
        const unsigned int rc = needs && a.d.dcnt != nullptr ? a.d.dcnt[key] : 0u;
        // promote_min < 0 (STRATA_GLM_DISK_LANE): a miss never claims a spare - the host computes it from its RAM landing
        const bool want = needs && a.promote_min >= 0 && (a.promote_min <= 1 || rc >= (unsigned int) a.promote_min);
        unsigned long long sp[kSpares];
#pragma unroll
        for (int j = 0; j < kSpares; ++j) sp[j] = spares[j];
        const unsigned int wm = (unsigned int) __ballot_sync(0xffffffffu, want);   // (all spare reads before the claims)
        bool b_promo = false;
        if (want) {
            const int r = __popc(wm & ((1u << i) - 1u));
            int c = 0;
#pragma unroll
            for (int j = 0; j < kSpares; ++j)
                if (sp[j] != 0ull) {
                    if (c == r) {
                        ptr = sp[j];
                        spares[j] = 0ull;
                        a.d.tab[key] = ptr;
                        b_promo = true;
                        pp = ptr;
                    }
                    ++c;
                }
        }
        if (needs && !b_promo) ptr = a.d.scratch[i];
        if (on) {
            s_w[i] = w;
            s_pp[i] = pp;
            s_cs[i] = cs;
            a.d.plan_ptr[i] = ptr;
            a.d.fetch_src[i] = src;
            a.d.plan_w[i] = w;
            a.d.plan_id[i] = s_ids[i];
        } else if (i < 8) {
            // (the unused slots up to 8: fetch_src holds max(8, k) entries - k_slots - and a window column's plan_w
            // follows it directly in verify_alloc's block, so a store past it zeroed that column's weights: 2026-10-31,
            // qwen4exp's k = 10 wrote [10, 16) over the window's plan_w and every verify window lost its experts)
            a.d.fetch_src[i] = 0ull;
        }
        const unsigned int skip = (unsigned int) __ballot_sync(0xffffffffu, b_skip);
        const unsigned int cpu = (unsigned int) __ballot_sync(0xffffffffu, b_cpu);
        const unsigned int promo = (unsigned int) __ballot_sync(0xffffffffu, b_promo);
        const unsigned int fetch = (unsigned int) __ballot_sync(0xffffffffu, needs && src != 0ull);
        const unsigned int mask = (unsigned int) __ballot_sync(0xffffffffu, needs && src == 0ull);   // only on disk: the host provides the source
        __syncwarp();
        if (lane == 0) {
            s_mask = mask;
            s_fetch = fetch;
            s_promo = promo;
            s_cpu = cpu;
            s_skip = skip;
            // PREFETCH: the predicted experts of the next layer that are not in VRAM but are in the RAM tier claim
            // that layer's spares now (one spare stays for its real misses); the side stream copies them while this
            // layer and the next one's attention compute, and the next route sees them as resident.  Only the
            // prediction's first pf_rank ranks (STRATA_GLM_PREFETCH_RANK): its top guesses are right almost always (Maya-M
            // on a 3090 + 3060, the next layer used rank 0 / 1 / 2 / 3 96 / 88 / 75 / 62 % of the time, ranks 4..7 52 .. 24 %)
            // - the first NON-resident guess is often a low rank, and a wrong copy costs the link what a right one saves
            int npf = 0;
            if (a.pred_logits != nullptr && a.max_pf > 0 && a.d.ram_direct == 0) {   // (direct: nothing to pull)
                const int L1 = a.layer + 1;
                unsigned long long* sp1 = a.d.tab + 2 * (size_t) a.d.n_keys + (size_t) L1 * kSpares;
                for (int i = 0; i < a.k && i < a.pf_rank && npf < a.max_pf; ++i) {
                    const int e = s_pred[i];
                    if (e < 0) continue;
                    const size_t key = (size_t) L1 * E + e;
                    if (a.d.tab[key] != 0ull) continue;
                    const unsigned long long src = rtab[key];
                    if (src == 0ull) continue;
                    int navail = 0, first = -1;
                    for (int j = 0; j < kSpares; ++j)
                        if (sp1[j] != 0ull) {
                            ++navail;
                            if (first < 0) first = j;
                        }
                    if (navail < 2) break;
                    const unsigned long long dst = sp1[first];
                    sp1[first] = 0ull;
                    a.d.tab[key] = dst;
                    a.d.pf_src[npf] = src;
                    a.d.pf_dst[npf] = dst;
                    s_pfid[npf] = e;
                    s_pfptr[npf] = dst;
                    ++npf;
                }
            }
            *a.d.pf_n = npf;
            s_npf = npf;
            const unsigned int seq = s_seq0 + 1;
            *a.d.seq = seq;
            s_seq = seq;
            *a.d.wait_seq = mask ? seq : 0u;
            if (a.d.cpu_seq != nullptr) {   // [1]: the route before this one's (expert deferral waits for that one)
                a.d.cpu_seq[1] = a.d.cpu_seq[0];
                a.d.cpu_seq[0] = cpu ? seq : 0u;
            }
            *a.d.cpu_flag = 0;
        }
    }
    __syncthreads();
    ROUTE_PROF(6)
    MoeRequest* rq = (MoeRequest*) a.d.ring + (s_seq % kRingSize);
    if ((s_cpu != 0u || (a.promote_min < 0 && s_mask != 0u)) && !a.early_x) {   // (the disk lane reads x too)
        // 16-byte stores, as EARLY X's: the 16 KB over PCIe in 1.6 us instead of 3.8 (AI PRO R9700, Gen4 x4)
        if ((a.n_embd & 3) == 0) {
            const float4* xs = (const float4*) a.x;
            float4* xd = (float4*) rq->x;
            for (int e = tid; e < a.n_embd / 4; e += ROUTE_THREADS) xd[e] = xs[e];
        } else {
            for (int e = tid; e < a.n_embd; e += ROUTE_THREADS) rq->x[e] = a.x[e];
        }
        route_put_xq(a, rq, tid);
    }
    // the request's fields (host-mapped: each store crosses PCIe) a thread an entry - thread 0 storing them one after
    // another was ~13k cycles of the route with the fence (RTX 5090); the same bytes, every thread's fence below
    if (tid == 0) {
        rq->layer = a.layer;
        rq->error = 0;
        rq->miss_mask = s_mask;
        rq->fetch_mask = s_fetch;
        rq->promo_mask = s_promo;
        rq->cpu_mask = s_cpu;
        rq->skip_mask = s_skip;
        rq->pf_n = s_npf;
        rq->win_more = a.win_more;
    }
    if (tid >= 32 && tid < 32 + a.k) {
        const int i = tid - 32;
        rq->ids[i] = s_ids[i];
        rq->w[i] = s_w[i];
        rq->promo_ptr[i] = s_pp[i];
        rq->pred[i] = s_pred[i];
        rq->cpu_src[i] = s_cs[i];
    }
    if (tid >= 64 && tid < 64 + s_npf) {
        const int i = tid - 64;
        rq->pf_ids[i] = s_pfid[i];
        rq->pf_ptr[i] = s_pfptr[i];
    }
    if (tid >= 96 && tid < 96 + kAhead * kMaxK) {
        const int d2 = (tid - 96) / kMaxK, i = (tid - 96) % kMaxK;
        rq->ahead[d2][i] = d2 < a.n_ahead && i < a.k ? s_ah[d2][i] : (short) -1;
    }
    __threadfence_system();
    __syncthreads();
    ROUTE_PROF(7)
    if (tid == 0) {
        __threadfence_system();
        rq->seq = s_seq;
#if defined(__HIPCC__)
        // The CPU polls this signal without a stream query. As in Strata #697,
        // publish the signal itself after publishing the request payload.
        __threadfence_system();
#endif
    }
    ROUTE_PROF(8)
}

// moe_wait's work for a route with misses (`s_need`, the whole block): the host's response, applied
__device__ void moe_wait_apply(const MoeDev& d, int n_embd, unsigned int s_need) {
    const int tid = threadIdx.x;
    const MoeResponse* r = (const MoeResponse*) d.resp + (s_need % kCpuAnswers);   // (the host answers seq there)
    if (tid == 0) {
        // spin on the host's answer (bounded: a dead host must not hang the GPU forever - ~60 s)
        const long long t0 = clock64();
        while (r->seq != s_need) {
            if (clock64() - t0 > (1ll << 36)) break;
        }
        __threadfence_system();
    }
    __syncthreads();
    // the disk-only experts: the host read each into host-mapped memory and answers with that SOURCE; the
    // fetch kernel copies it into the slot the route already planned
    const unsigned int pm = ((volatile const MoeResponse*) r)->ptr_mask;
    if (tid < kMaxK && (pm >> tid) & 1u) d.fetch_src[tid] = ((volatile const MoeResponse*) r)->ptr[tid];
    // the disk lane: the host computed these from their RAM landing - no fetch, no expert kernels (null plan entries)
    const unsigned int hm = ((volatile const MoeResponse*) r)->host_mask;
    if (tid < kMaxK && (hm >> tid) & 1u) {
        d.fetch_src[tid] = 0ull;
        d.plan_ptr[tid] = 0ull;
    }
    const int nu = ((volatile const MoeResponse*) r)->n_upd;
    for (int u = tid; u < nu && u < 64; u += blockDim.x)
        d.tab[((volatile const MoeResponse*) r)->upd_key[u]] = ((volatile const MoeResponse*) r)->upd_val[u];
    if (((volatile const MoeResponse*) r)->cpu) {
        for (int e = tid; e < n_embd; e += blockDim.x) d.cpu_part[e] = ((volatile const MoeResponse*) r)->cpu_part[e];
        if (tid == 0) *d.cpu_flag = 1;
    }
}

__global__ void moe_wait_kernel(MoeDev d, int n_embd) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    __shared__ unsigned int s_need;
    if (threadIdx.x == 0) s_need = *d.wait_seq;
    __syncthreads();
    if (s_need == 0) return;
    moe_wait_apply(d, n_embd, s_need);
}

// the CPU LANE's answer for the last route (when it sent experts to the host): spin, then out += its weighted sum.
// which: 0 the last route's (cpu_seq[0]), 1 the route before it (cpu_seq[1]: EXPERT DEFERRAL - the host's part of
// layer l is added one layer late, into layer l + 1's FFN output, so the lane is never waited for)
__device__ __forceinline__ void cpu_wait_add(const unsigned int* cpu_seq, const void* cpu_ans, int n_embd, float* out,
                                             int which, int blk, int nblk) {
    __shared__ unsigned int s_need;
    const int tid = threadIdx.x;
    if (tid == 0) s_need = cpu_seq[which];
    __syncthreads();
    if (s_need == 0) return;
    const CpuAnswer* r = (const CpuAnswer*) cpu_ans + (s_need % kCpuAnswers);
    if (tid == 0) {
#if defined(STRATA_USE_HIP)
        // block 0 alone polls the host-mapped answer and passes its seq on in device memory (cpu_seq[2]), where the
        // other blocks wait: every poll is a PCIe read the CPU's memory serves while the lane reads its experts
        // (hc3's 32 pollers made an R9700 host's CPU lane ~12% slower; not measured on NVIDIA, which keeps 8)
        volatile unsigned int* seen = (volatile unsigned int*) cpu_seq + 2;
        const long long t0 = clock64();   // bounded like moe_wait (~60 s)
        if (blk == 0) {
            while (r->seq != s_need) {
                if (clock64() - t0 > (1ll << 36)) break;
            }
            __threadfence_system();
            *seen = s_need;
        } else {
            while (*seen != s_need) {
                if (clock64() - t0 > (1ll << 36)) break;
            }
            __threadfence_system();
        }
#else
        const long long t0 = clock64();   // bounded like moe_wait (~60 s)
        while (r->seq != s_need) {
            if (clock64() - t0 > (1ll << 36)) break;
        }
        __threadfence_system();
#endif
    }
    __syncthreads();
    // the host's part: 16-byte uncached loads spread over the grid's blocks (each block waits for the answer
    // itself) - one block reading 4096 floats a value at a time took ~30 us a layer over PCIe Gen4 x4, 8 blocks ~9
    if ((n_embd & 3) == 0) {
        const float4* src = (const float4*) r->part;
        float4* dst = (float4*) out;
        const int n4 = n_embd / 4;
        for (int i = blk * blockDim.x + tid; i < n4; i += nblk * blockDim.x) {
#if defined(__HIPCC__)
            const volatile float* vp = (const volatile float*) (src + i);
            const float4 v = make_float4(vp[0], vp[1], vp[2], vp[3]);
#else
            const float4 v = __ldcv(src + i);
#endif
            float4 o = dst[i];
            o.x += v.x; o.y += v.y; o.z += v.z; o.w += v.w;
            dst[i] = o;
        }
    } else {
        for (int e = blk * blockDim.x + tid; e < n_embd; e += nblk * blockDim.x)
            out[e] += ((volatile const CpuAnswer*) r)->part[e];
    }
}

__global__ void moe_cpu_wait_kernel(MoeDev d, int n_embd, float* out, int which) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    cpu_wait_add(d.cpu_seq, d.cpu_ans, n_embd, out, which, blockIdx.x, gridDim.x);
}

// a verify window's columns in one launch (blockIdx.y the column): each column's answer is waited for and added as
// moe_cpu_wait does it (the same adds), a launch a layer instead of one a column
__global__ void moe_cpu_wait_cols_kernel(CpuWaitCols c, const void* cpu_ans, int n_embd) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int t = blockIdx.y;
    if (c.seq[t] == nullptr) return;
    cpu_wait_add(c.seq[t], cpu_ans, n_embd, c.out[t], 0, blockIdx.x, gridDim.x);
}

// EXPERT DEFERRAL, the half-layer form: the CPU lane's part of layer l's MoE, waited for after layer l + 1's
// attention and written straight into the hyper-connection streams with layer l's FFN write weights (post, saved
// before layer l + 1's attention-side read overwrote them): R[d][e] += post[d] * part[e].  Layer l + 1's attention
// misses it, its router and experts see it.  (Adding it into layer l + 1's FFN output instead - a whole layer late
// and with l + 1's weights - cost KL 0.21 against 0.012.)
__global__ void hc_defer_add_kernel(MoeDev d, const float* __restrict__ post, int n_embd, float* __restrict__ R) {
    __shared__ unsigned int s_need;
    const int tid = threadIdx.x;
    if (tid == 0) s_need = d.cpu_seq[0];   // (the last route's: the next layer's route has not run yet)
    __syncthreads();
    if (s_need == 0) return;
    const CpuAnswer* r = (const CpuAnswer*) d.cpu_ans + (s_need % kCpuAnswers);
    if (tid == 0) {
        const long long t0 = clock64();
        while (r->seq != s_need) {
            if (clock64() - t0 > (1ll << 36)) break;
        }
        __threadfence_system();
    }
    __syncthreads();
    const float p0 = post[0], p1 = post[1], p2 = post[2], p3 = post[3];
    for (int e = blockIdx.x * blockDim.x + tid; e < n_embd; e += gridDim.x * blockDim.x) {
#if defined(__HIPCC__)
        const float v = ((volatile const float*) r->part)[e];
#else
        const float v = __ldcv(r->part + e);
#endif
        R[e] += p0 * v;
        R[n_embd + e] += p1 * v;
        R[2 * n_embd + e] += p2 * v;
        R[3 * n_embd + e] += p3 * v;
    }
}
__global__ void copy_floats_kernel(float* __restrict__ dst, const float* __restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void __launch_bounds__(256) moe_prefetch_kernel(MoeDev d, size_t n16) {
    const int n = *d.pf_n;
    for (int i = 0; i < n; ++i) {
        const uint4* s = (const uint4*) d.pf_src[i];
        uint4* t = (uint4*) d.pf_dst[i];
        for (size_t j = blockIdx.x * (size_t) blockDim.x + threadIdx.x; j < n16; j += (size_t) gridDim.x * blockDim.x)
            t[j] = s[j];
    }
}

// one block per SM, 16-byte loads: the PCIe pull measured 11.5 GB/s (a little over cudaMemcpy's 10.7)
__device__ __forceinline__ void moe_fetch_body(const MoeDev& d, int k, size_t n16) {
    for (int i = 0; i < k; ++i) {
        const unsigned long long src = d.fetch_src[i];
        if (src == 0ull) continue;
        const uint4* s = (const uint4*) src;
        uint4* t = (uint4*) d.plan_ptr[i];
        for (size_t j = blockIdx.x * (size_t) blockDim.x + threadIdx.x; j < n16; j += (size_t) gridDim.x * blockDim.x)
            t[j] = s[j];
    }
}

__global__ void __launch_bounds__(256) moe_fetch_kernel(MoeDev d, int k, size_t n16) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    moe_fetch_body(d, k, n16);
}

// STREAMED: the host posted the response before its disk reads finished, and each read job sets a bit of
// ready[i] (part * n_chunks + chunk, fast_read_part's layout) once its chunk is in the landing slot.  Block b copies
// the chunks whose bit is b modulo the grid, each as soon as it is set - the PCIe pull (~6 GB/s) runs behind the
// SSD (~3.4) instead of after it, so only the last chunk's copy stays exposed.  Entries outside stream_mask are
// whole blobs, as above.  A bit that never comes (a dead host) gives up after ~60 s like moe_wait.
// gu / uu / dn: the gate, up and down parts' bytes (uu == gu but in a mixed gate/up layer)
__device__ void moe_fetch_stream_body(const MoeDev& d, int k, size_t n16, unsigned long long gu, unsigned long long dn,
                                      int rch, unsigned int s_need, unsigned long long uu) {
    const volatile MoeResponse* r = s_need != 0u ? (const volatile MoeResponse*) d.resp + (s_need % kCpuAnswers) : nullptr;
    const unsigned int sm = r != nullptr ? r->stream_mask : 0u;
    for (int i = 0; i < k; ++i) {
        const unsigned long long src = d.fetch_src[i];
        if (src == 0ull) continue;
        const uint8_t* s = (const uint8_t*) src;
        uint8_t* t = (uint8_t*) d.plan_ptr[i];
        if (((sm >> i) & 1u) == 0u) {
            const uint4* s4 = (const uint4*) s;
            uint4* t4 = (uint4*) t;
            for (size_t j = blockIdx.x * (size_t) blockDim.x + threadIdx.x; j < n16; j += (size_t) gridDim.x * blockDim.x)
                t4[j] = s4[j];
            continue;
        }
        for (int bit = blockIdx.x; bit < 3 * rch; bit += gridDim.x) {
            const int part = bit / rch, c = bit % rch;
            const unsigned long long len = part == 2 ? dn : part == 1 ? uu : gu;
            const unsigned long long piece = ((len + (unsigned long long) rch - 1) / rch + 4095ull) & ~4095ull;
            const unsigned long long c0 = min(len, (unsigned long long) c * piece), c1 = min(len, c0 + piece);
            if (c1 <= c0) continue;
            const unsigned long long base = (part == 0 ? 0ull : part == 1 ? gu : gu + uu) + c0;
            if (threadIdx.x == 0) {
                const long long t0 = clock64();
                while (((r->ready[i] >> bit) & 1u) == 0u) {
                    if (clock64() - t0 > (1ll << 36)) break;
                }
                __threadfence_system();
            }
            __syncthreads();
            const size_t nb = (size_t) (c1 - c0), m16 = nb / 16;
            const uint4* s4 = (const uint4*) (s + base);
            uint4* t4 = (uint4*) (t + base);
            for (size_t j = threadIdx.x; j < m16; j += blockDim.x) t4[j] = s4[j];
            for (size_t j = m16 * 16 + threadIdx.x; j < nb; j += blockDim.x) t[base + j] = s[base + j];
            __syncthreads();
        }
    }
}

__global__ void __launch_bounds__(256) moe_fetch_stream_kernel(MoeDev d, int k, size_t n16, unsigned long long gu,
                                                               unsigned long long dn, int rch, unsigned long long uu) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    __shared__ unsigned int s_need;
    if (threadIdx.x == 0) s_need = *d.wait_seq;
    __syncthreads();
    moe_fetch_stream_body(d, k, n16, gu, dn, rch, s_need, uu);
}

// moe_wait and moe_fetch in one launch: a route without disk misses - the rule once warm - goes straight to the
// copies; with misses, block 0 waits for the host's response and applies it as moe_wait does, then passes the seq on
// in device memory (wait_seq[1]), where the other blocks wait before they read the plan
template <bool STREAM>
__global__ void __launch_bounds__(256) moe_wait_fetch_kernel(MoeDev d, int n_embd, int k, size_t n16,
                                                             unsigned long long gu, unsigned long long dn, int rch,
                                                             unsigned long long uu) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    __shared__ unsigned int s_need;
    if (threadIdx.x == 0) s_need = *d.wait_seq;
    __syncthreads();
    const unsigned int need = s_need;
    if (need != 0u) {
        volatile unsigned int* done = (volatile unsigned int*) d.wait_seq + 1;
        if (blockIdx.x == 0) {
            moe_wait_apply(d, n_embd, need);
            __syncthreads();
            if (threadIdx.x == 0) {
                __threadfence();
                *done = need;
            }
        } else {
            if (threadIdx.x == 0) {
                const long long t0 = clock64();   // bounded like moe_wait (~60 s)
                while (*done != need) {
                    if (clock64() - t0 > (1ll << 36)) break;
                }
                __threadfence();
            }
            __syncthreads();
        }
    }
    if (STREAM) moe_fetch_stream_body(d, k, n16, gu, dn, rch, need, uu);
    else moe_fetch_body(d, k, n16);
}

// moe_wait_fetch over a verify window's columns in one launch (blockIdx.y the column, its plan in mds[y], device):
// each column's blocks do what moe_wait_fetch_kernel's do for it - the same copies, a launch a layer instead of a
// moe_wait and a moe_fetch a column
template <bool STREAM>
__global__ void __launch_bounds__(256) moe_wait_fetch_cols_kernel(const MoeDev* __restrict__ mds, int n_embd, int k,
                                                                  size_t n16, unsigned long long gu,
                                                                  unsigned long long dn, int rch,
                                                                  unsigned long long uu) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const MoeDev& d = mds[blockIdx.y];
    __shared__ unsigned int s_need;
    if (threadIdx.x == 0) s_need = *d.wait_seq;
    __syncthreads();
    const unsigned int need = s_need;
    if (need != 0u) {
        volatile unsigned int* done = (volatile unsigned int*) d.wait_seq + 1;
        if (blockIdx.x == 0) {
            moe_wait_apply(d, n_embd, need);
            __syncthreads();
            if (threadIdx.x == 0) {
                __threadfence();
                *done = need;
            }
        } else {
            if (threadIdx.x == 0) {
                const long long t0 = clock64();   // bounded like moe_wait (~60 s)
                while (*done != need) {
                    if (clock64() - t0 > (1ll << 36)) break;
                }
                __threadfence();
            }
            __syncthreads();
        }
    }
    if (STREAM) moe_fetch_stream_body(d, k, n16, gu, dn, rch, need, uu);
    else moe_fetch_body(d, k, n16);
}

// a verify window's columns in one expert launch (moe_gate_up_cols / moe_down_cols): the per-column table (device)
// and the byte / float strides of the columns' buffers; nc 1 and ca null is the one-column launch
struct MoeCols {
    const MoeColArgs* ca = nullptr;
    int nc = 1;
    size_t xq_cs = 0, hq_cs = 0, shq_cs = 0, out_cs = 0;
};
constexpr int GU_ROWS = 32;   // h values (gate rows = up rows) per block: one q8_1 block of h
constexpr int GU_WARPS = 16;  // ... 2 gate + 2 up rows per warp (8 per warp measured 90 vs 65 us a layer: occupancy)

// grid (n_ff / GU_ROWS, k [+ 1]): the extra row of blocks (blockIdx.y == k) computes the SHARED expert's down rows
// into sh_out while the routed experts' blocks run - in moe_down it was a serial tail (~25 us of 72)
// TU: up's type when it differs from gate's (a mixed layer: unsloth's UD-IQ1_S Qwen3.8 keeps IQ2_S gate + IQ2_XXS up in
// some layers) - each role's rows at its own size and codebook, one dot pass a role; TU == TG is the one-pass kernel
#if defined(STRATA_USE_HIP)
template<int TG, int NW = GU_WARPS, bool LUT = false, bool GLOBAL = false, bool SPLIT = false,
         bool DIRECT = false, bool WIDE = false, int MINWAVES = 0, int TU = TG>
__attribute__((amdgpu_waves_per_eu(MINWAVES > 0 ? MINWAVES : 1)))
__global__ void __launch_bounds__(NW * 32)
#else
template<int TG, int TU = TG>
__global__ void __launch_bounds__(GU_WARPS * 32)
#endif
moe_gate_up_kernel(MoeDev d, int k, int n_embd, int n_ff, float limit,
                                                                    const block_q8_1* __restrict__ xq,
                                                                    block_q8_1* __restrict__ hq,
                                                                    const uint8_t* __restrict__ sh_down, int sh_type,
                                                                    const block_q8_1* __restrict__ sh_hq, int n_ff_sh,
                                                                    float* __restrict__ sh_out,
                                                                    const MoeColArgs* __restrict__ ca = nullptr,
                                                                    size_t xq_cs = 0, size_t hq_cs = 0,
                                                                    size_t shq_cs = 0, size_t out_cs = 0) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
#if !defined(STRATA_USE_HIP)
    constexpr int NW = GU_WARPS;
#endif
    const int ei = blockIdx.y, chunk = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    // a window's columns in one grid (moe_gate_up_cols): column blockIdx.z's plan and buffers
    const unsigned long long* plan_ptr = d.plan_ptr;
    const int* plan_id = d.plan_id;
    if (ca != nullptr) {
        const int col = blockIdx.z;
        plan_ptr = ca[col].plan_ptr;
        plan_id = ca[col].plan_id;
        xq = (const block_q8_1*) ((const uint8_t*) xq + (size_t) col * xq_cs);
        hq = (block_q8_1*) ((uint8_t*) hq + (size_t) col * hq_cs);
        if (sh_hq != nullptr) sh_hq = (const block_q8_1*) ((const uint8_t*) sh_hq + (size_t) col * shq_cs);
        if (sh_out != nullptr) sh_out += (size_t) col * out_cs;
    }
#if defined(STRATA_USE_HIP)
    if constexpr (!SPLIT)
#endif
    if (ei == k) {
        const int per = (n_embd + (int) gridDim.x - 1) / (int) gridDim.x;
        const int r1 = min(n_embd, (chunk + 1) * per);
        for (int r = chunk * per + warp; r < r1; r += NW) {
            const float v = any_row_dot(sh_type, sh_down, r, sh_hq, nullptr, n_ff_sh, lane);
            if (lane == 0) sh_out[r] = v;
        }
        return;
    }
    const unsigned long long ptr = plan_ptr[ei];
    if (ptr == 0ull) return;
    __shared__ float sg[GU_ROWS], su[GU_ROWS];
    __shared__ uint32_t s_tab[TabWords<TG>::n > 0 ? TabWords<TG>::n : 1];
    __shared__ uint32_t s_tab_u[TU != TG && TabWords<TU>::n > 0 ? TabWords<TU>::n : 1];   // (a mixed layer's up)
#if defined(STRATA_USE_HIP)
    __shared__ uint4 s_sign[LUT && !DIRECT ? sign_count<TG> : 1];
    if constexpr (LUT && !DIRECT) load_signs<TG>(s_sign);
    if constexpr (!GLOBAL)
#endif
    load_tab<TG>(s_tab);
    if constexpr (TU != TG) load_tab<TU>(s_tab_u);
    __syncthreads();
    const size_t rb = rbytes<TG>(n_embd);
    const size_t rbu = rbytes<TU>(n_embd);
    const uint8_t* blob = (const uint8_t*) ptr;
    // each warp: 2 gate + 2 up rows in ONE pass (one activation load per call for all four)
    constexpr int RPW = GU_ROWS / NW;
    const uint8_t* rows[2 * RPW];
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
        const int r = chunk * GU_ROWS + warp * RPW + j;
        rows[j] = blob + (size_t) r * rb;
        rows[RPW + j] = blob + (size_t) n_ff * rb + (size_t) r * rbu;
    }
    float s[2 * RPW];
    const uint32_t* tab = s_tab;
    if constexpr (TU != TG) {
        // a mixed layer: the gate rows, then the up rows, each at its own type (each row's sum is the same dot)
        rows_dot<TG, RPW>(rows, xq, n_embd, lane, tab, s);
        rows_dot<TU, RPW>(rows + RPW, xq, n_embd, lane, s_tab_u, s + RPW);
    } else {
#if defined(STRATA_USE_HIP)
    if constexpr (GLOBAL) tab = TG == 16 ? (const uint32_t*) iq2xxs_grid :
                                TG == 18 ? (const uint32_t*) iq3xxs_grid : (const uint32_t*) iq2s_grid;
    if constexpr (LUT) {
        rows_dot_rdna<TG, 2 * RPW>(rows, xq, n_embd, lane, tab, s_sign, s, DIRECT, WIDE);
    } else
#endif
    rows_dot<TG, 2 * RPW>(rows, xq, n_embd, lane, tab, s);
    }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < RPW; ++j) {
            sg[warp * RPW + j] = s[j];
            su[warp * RPW + j] = s[RPW + j];
        }
    }
    __syncthreads();
    if (warp == 0) {
        // NVFP4: the expert's second-level scale (gate and up share it in NVIDIA's checkpoints; [2 * id] = theirs)
        float sc = 1.0f;
        if constexpr (TG == 40)
            if (d.nv_scale != nullptr && plan_id != nullptr && plan_id[ei] >= 0) sc = d.nv_scale[2 * plan_id[ei]];
        const float g = fminf(sc * sg[lane], limit);
        const float u = fminf(fmaxf(sc * su[lane], -limit), limit);
        q8_1_store_warp((g / (1.0f + expf(-g))) * u, hq + (size_t) ei * (n_ff / 32) + chunk, lane);
    }
}

constexpr int DOWN_ROWS = 4;

#if defined(STRATA_USE_HIP)
template<int TD, int NR = DOWN_ROWS, bool LUT = false, bool GLOBAL = false, bool DIRECT = false, bool WIDE = false>
#else
template<int TD>
#endif
__global__ void __launch_bounds__(256) moe_down_kernel(MoeDev d, int k, int n_embd, int n_ff, size_t down_off,
                                                       const block_q8_1* __restrict__ hq,
                                                       const float* __restrict__ sh_out, float* __restrict__ out,
                                                       const MoeColArgs* __restrict__ ca = nullptr,
                                                       size_t hq_cs = 0, size_t out_cs = 0) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
#if !defined(STRATA_USE_HIP)
    constexpr int NR = DOWN_ROWS;
#else
    if ((int) blockIdx.x * NR >= n_embd) return;   // (grid_pad's blocks)
#endif
    // NR output rows per block; warp w computes those rows for expert w (one activation load per call
    // for all of them), the combine runs in plan order like the reference's axpy chain
    __shared__ unsigned long long s_ptr[kMaxK];
    __shared__ float s_w[kMaxK];
    __shared__ float s_part[kMaxK][NR];
    __shared__ uint32_t s_tab[TabWords<TD>::n > 0 ? TabWords<TD>::n : 1];
#if defined(STRATA_USE_HIP)
    __shared__ uint4 s_sign[LUT && !DIRECT ? sign_count<TD> : 1];
    if constexpr (LUT && !DIRECT) load_signs<TD>(s_sign);
    if constexpr (!GLOBAL)
#endif
    load_tab<TD>(s_tab);
    // a window's columns in one grid (moe_down_cols): column blockIdx.z's plan, CPU-lane part and buffers
    const unsigned long long* plan_ptr = d.plan_ptr;
    const float* plan_w = d.plan_w;
    const int* plan_id = d.plan_id;
    const int* cpu_flag = d.cpu_flag;
    const float* cpu_part = d.cpu_part;
    if (ca != nullptr) {
        const MoeColArgs& c = ca[blockIdx.z];
        plan_ptr = c.plan_ptr;
        plan_w = c.plan_w;
        plan_id = c.plan_id;
        cpu_flag = c.cpu_flag;
        cpu_part = c.cpu_part;
        hq = (const block_q8_1*) ((const uint8_t*) hq + (size_t) blockIdx.z * hq_cs);
        out += (size_t) blockIdx.z * out_cs;
        if (sh_out != nullptr) sh_out += (size_t) blockIdx.z * out_cs;
    }
    if (threadIdx.x < kMaxK) {
        s_ptr[threadIdx.x] = (int) threadIdx.x < k ? plan_ptr[threadIdx.x] : 0ull;
        float w = (int) threadIdx.x < k ? plan_w[threadIdx.x] : 0.0f;
        // NVFP4: the expert's down scale_2 ([2 * id + 1]) folds into its weight (the down sum is linear)
        if constexpr (TD == 40)
            if ((int) threadIdx.x < k && d.nv_scale != nullptr && plan_id != nullptr && plan_id[threadIdx.x] >= 0)
                w *= d.nv_scale[2 * plan_id[threadIdx.x] + 1];
        s_w[threadIdx.x] = w;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int r0 = blockIdx.x * NR;
    const size_t drb = rbytes<TD>(n_ff);
    // warp w computes expert w, then w + 8 (top-k above 8: qwen4exp's 10)
    for (int warp = threadIdx.x >> 5; warp < 8 || warp < k; warp += 8) {
        float s[NR];
        if (warp < k && s_ptr[warp] != 0ull) {
            const uint8_t* rows[NR];
#pragma unroll
            for (int j = 0; j < NR; ++j) rows[j] = (const uint8_t*) s_ptr[warp] + down_off + (size_t) (r0 + j) * drb;
            const uint32_t* tab = s_tab;
#if defined(STRATA_USE_HIP)
            if constexpr (GLOBAL) tab = TD == 16 ? (const uint32_t*) iq2xxs_grid :
                                        TD == 18 ? (const uint32_t*) iq3xxs_grid : (const uint32_t*) iq2s_grid;
            if constexpr (LUT) {
                rows_dot_rdna<TD, NR>(rows, hq + (size_t) warp * (n_ff / 32), n_ff, lane, tab, s_sign, s, DIRECT, WIDE);
            } else
#endif
            rows_dot<TD, NR>(rows, hq + (size_t) warp * (n_ff / 32), n_ff, lane, tab, s);
        } else {
#pragma unroll
            for (int j = 0; j < NR; ++j) s[j] = 0.0f;
        }
        if (lane == 0)
#pragma unroll
            for (int j = 0; j < NR; ++j) s_part[warp][j] = s[j];
    }
    __syncthreads();
    if (threadIdx.x < NR) {
        const int j = threadIdx.x, r = r0 + j;
        float m = 0.0f;
        for (int i = 0; i < k; ++i) m += s_w[i] * s_part[i][j];
        if (*cpu_flag) m += cpu_part[r];
        out[r] = m + (sh_out != nullptr ? sh_out[r] : 0.0f);
    }
}

// ---------------------------------------------------------------- the prompt path's lightly routed experts
// An expert with a few rows (tokens) of a prompt chunk: its weights read ONCE, each 32-value sub-block decoded once
// and dotted with up to LR_MT tokens at a time (MMQ computes tiles of 64-128 tokens - mostly empty at ~10 rows an
// expert, as at a 300-token prompt).  The sub-block decode is the decode path's dw<T>, split from its dot.
constexpr int LR_MT = 4;
#ifndef GLMF_LR_MT2
#define GLMF_LR_MT2 8
#endif
constexpr int LR_MT2 = GLMF_LR_MT2;   // tokens a pass of the light kernels (rows_multi2)
template<int T>
__device__ __forceinline__ void dwm(const uint8_t* row, int kbx, int iqs, const XV* x, int nt, const uint32_t* tab,
                                    float* acc) {
    int g[8];
    if constexpr (T == 19) {
        const block_iq1_s* bq1 = (const block_iq1_s*) row + kbx;
        const int qs_packed = get_int_b2(bq1->qs, iqs);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq1->qh[iqs];
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int grid = tab[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
            g[l0] = (grid >> 0) & 0x0F0F0F0F;
            g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
        }
        const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
        const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
#pragma unroll
        for (int t = 0; t < LR_MT; ++t) {
            if (t >= nt) break;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) sumi = __dp4a(g[i], x[t].u[i], sumi);
            acc[t] += d1q * (x[t].d * sumi + x[t].s * delta);
        }
    } else if constexpr (T == 18) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) row + kbx;
        const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(tab[q3[l0 + 0]], tab[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            g[l0] = (int) __vsub4(grid_pos.x ^ signs0, signs0);
            g[l0 + 1] = (int) __vsub4(grid_pos.y ^ signs1, signs1);
        }
        const int ls = aux32 >> 28;
        const float d = __half2float(bq3->d);
#pragma unroll
        for (int t = 0; t < LR_MT; ++t) {
            if (t >= nt) break;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) sumi = __dp4a(g[i], x[t].u[i], sumi);
            sumi = (ls * sumi + sumi / 2) / 2;
            acc[t] += d * x[t].d * sumi;
        }
    } else {   // 16: IQ2_XXS
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) row + kbx;
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t* aux8 = (const uint8_t*) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2*) tab)[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            g[k0] = (int) __vsub4(grid_pos.x ^ signs0, signs0);
            g[k0 + 1] = (int) __vsub4(grid_pos.y ^ signs1, signs1);
        }
        const int ls = aux32 >> 27 | 1;
        const float d = __half2float(bq2->d);
#pragma unroll
        for (int t = 0; t < LR_MT; ++t) {
            if (t >= nt) break;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) sumi = __dp4a(g[i], x[t].u[i], sumi);
            sumi = sumi * ls / 8;
            acc[t] += d * x[t].d * sumi;
        }
    }
}

// one 32-value sub-block decoded (dwm's first half), then dotted with any number of activations
struct Dec {
    int g[8];
    float a, b;   // IQ1_S: d1q, delta; IQ3_XXS / IQ2_XXS / IQ2_XS: d, -
    int ls, ls1;  // IQ2_XS: the two halves' scales
};
template<int T>
__device__ __forceinline__ void dec_sub(const uint8_t* row, int kbx, int iqs, const uint32_t* tab, Dec& o) {
    if constexpr (T == 19) {
        const block_iq1_s* bq1 = (const block_iq1_s*) row + kbx;
        const int qs_packed = get_int_b2(bq1->qs, iqs);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq1->qh[iqs];
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int grid = tab[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
            o.g[l0] = (grid >> 0) & 0x0F0F0F0F;
            o.g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
        }
        o.a = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
        o.b = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    } else if constexpr (T == 18) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) row + kbx;
        const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(tab[q3[l0 + 0]], tab[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            o.g[l0] = (int) __vsub4(grid_pos.x ^ signs0, signs0);
            o.g[l0 + 1] = (int) __vsub4(grid_pos.y ^ signs1, signs1);
        }
        o.ls = aux32 >> 28;
        o.a = __half2float(bq3->d);
    } else if constexpr (T == 17) {   // IQ2_XS (dw<17>'s decode)
        const block_iq2_xs* bq2 = (const block_iq2_xs*) row + kbx;
        const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
        const uint16_t* q2 = (const uint16_t*) &q2_packed;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2 grid_pos = ((const uint2*) tab)[q2[l0 / 2] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            o.g[l0] = (int) __vsub4(grid_pos.x ^ signs0, signs0);
            o.g[l0 + 1] = (int) __vsub4(grid_pos.y ^ signs1, signs1);
        }
        o.ls = bq2->scales[iqs / 2] & 0x0F;
        o.ls1 = bq2->scales[iqs / 2] >> 4;
        o.a = __half2float(bq2->d);
    } else {
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) row + kbx;
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t* aux8 = (const uint8_t*) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2*) tab)[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            o.g[k0] = (int) __vsub4(grid_pos.x ^ signs0, signs0);
            o.g[k0 + 1] = (int) __vsub4(grid_pos.y ^ signs1, signs1);
        }
        o.ls = aux32 >> 27 | 1;
        o.a = __half2float(bq2->d);
    }
}
template<int T>
__device__ __forceinline__ float dec_dot(const Dec& o, const XV& x) {
    if constexpr (T == 17) {
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) s0 = __dp4a(o.g[i], x.u[i], s0);
#pragma unroll
        for (int i = 4; i < 8; ++i) s1 = __dp4a(o.g[i], x.u[i], s1);
        return o.a * x.d * ((s0 * o.ls + s1 * o.ls1 + (s0 + s1) / 2) / 4);
    }
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) sumi = __dp4a(o.g[i], x.u[i], sumi);
    if constexpr (T == 19) return o.a * (x.d * sumi + x.s * o.b);
    else if constexpr (T == 18) return o.a * x.d * ((o.ls * sumi + sumi / 2) / 2);
    else return o.a * x.d * (sumi * o.ls / 8);
}

// NR weight rows x MT tokens a pass: each k step decodes the NR rows' sub-blocks once, then every token's activation
// is loaded once and dotted with all NR (dwm decoded per row and per token group)
template<int T, int NR, int MT, typename XF>
__device__ __forceinline__ void rows_multi2(const uint8_t* const* rows, int n_in, int ntok, int lane, const uint32_t* tab,
                                            XF x_of, float (&out)[NR][MT], int t0) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
    const int nt = min(MT, ntok - t0);
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int t = 0; t < MT; ++t) out[r][t] = 0.0f;
    for (int k = lane; k < nb * Fm::ipb; k += 32) {
        const int kbx = k / Fm::ipb, ki = k % Fm::ipb, iqs = Fm::step * ki;
        Dec dr[NR];
#pragma unroll
        for (int r = 0; r < NR; ++r) dec_sub<T>(rows[r], kbx, iqs, tab, dr[r]);
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (t >= nt) break;
            const XV xv = load_xv(x_of(t0 + t) + kbx * (Fm::qk / 32) + ki);
#pragma unroll
            for (int r = 0; r < NR; ++r) out[r][t] += dec_dot<T>(dr[r], xv);
        }
    }
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int t = 0; t < MT; ++t) out[r][t] = warp_sum(out[r][t]);
}

// NR weight rows (pointers) x the rows' tokens, LR_MT tokens a pass; x(t) = the q8_1 activation of token t
template<int T, int NR, typename XF>
__device__ __forceinline__ void rows_multi(const uint8_t* const* rows, int n_in, int ntok, int lane, const uint32_t* tab,
                                           XF x_of, float (&out)[NR][LR_MT], int t0) {
    using Fm = F<T>;
    const int nb = n_in / Fm::qk;
    const int nt = min(LR_MT, ntok - t0);
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int t = 0; t < LR_MT; ++t) out[r][t] = 0.0f;
    for (int k = lane; k < nb * Fm::ipb; k += 32) {
        const int kbx = k / Fm::ipb, ki = k % Fm::ipb, iqs = Fm::step * ki;
        XV xv[LR_MT];
#pragma unroll
        for (int t = 0; t < LR_MT; ++t)
            if (t < nt) xv[t] = load_xv(x_of(t0 + t) + kbx * (Fm::qk / 32) + ki);
#pragma unroll
        for (int r = 0; r < NR; ++r) dwm<T>(rows[r], kbx, iqs, xv, nt, tab, out[r]);
    }
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int t = 0; t < LR_MT; ++t) out[r][t] = warp_sum(out[r][t]);
}

// light[i] = {slot, r0, nr}: grid (n_ff / 32, n); 16 warps x (1 gate + 1 up row)... as the decode kernel: 2 + 2
template<int TG>
__global__ void __launch_bounds__(GU_WARPS * 32) rows_gate_up_kernel(const uint8_t* __restrict__ base, size_t stride, const unsigned long long* __restrict__ ptrs,
                                                                     const int* __restrict__ light,
                                                                     const int* __restrict__ row_tok,
                                                                     const block_q8_1* __restrict__ xq, int n_embd,
                                                                     int n_ff, float* __restrict__ out, int ld) {
    const int i = blockIdx.y, chunk = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ uint32_t s_tab[TabWords<TG>::n > 0 ? TabWords<TG>::n : 1];
    load_tab<TG>(s_tab);
    __syncthreads();
    const int slot = light[3 * i], R0 = light[3 * i + 1], NR = light[3 * i + 2];
    const uint8_t* blob = ptrs != nullptr ? (const uint8_t*) ptrs[i] : base + (size_t) slot * stride;
    const size_t rb = rbytes<TG>(n_embd);
    constexpr int RPW = GU_ROWS / GU_WARPS;
    const uint8_t* rows[2 * RPW];
    int col[2 * RPW];
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
        const int r = chunk * GU_ROWS + warp * RPW + j;
        rows[j] = blob + (size_t) r * rb;
        rows[RPW + j] = blob + (size_t) (n_ff + r) * rb;
        col[j] = r;
        col[RPW + j] = n_ff + r;
    }
    const int nblk = n_embd / 32;
    const auto x_of = [&](int t) { return xq + (size_t) row_tok[R0 + t] * nblk; };
    for (int t0 = 0; t0 < NR; t0 += LR_MT2) {
        float s[2 * RPW][LR_MT2];
        rows_multi2<TG, 2 * RPW, LR_MT2>(rows, n_embd, NR, lane, s_tab, x_of, s, t0);
        if (lane == 0)
#pragma unroll
            for (int j = 0; j < 2 * RPW; ++j)
#pragma unroll
                for (int t = 0; t < LR_MT2; ++t)
                    if (t0 + t < NR) out[(size_t) (R0 + t0 + t) * ld + col[j]] = s[j][t];
    }
}

// h = swiglu(gate, up) of the light rows [rlo, rhi) (GU at out rows, ld floats apart) -> q8_1 rows (n_ff / 32 each)
__global__ void rows_swiglu_q8_kernel(const float* __restrict__ gu, int ld, int rlo, int rhi, int n_ff, float limit,
                                      block_q8_1* __restrict__ hq) {
    const int64_t w = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31, per = n_ff / 32;
    const int64_t r = rlo + w / per, b = w % per;
    if (r >= rhi) return;
    const float* row = gu + (size_t) r * ld;
    const float gv = fminf(row[b * 32 + lane], limit);
    const float uv = fminf(fmaxf(row[n_ff + b * 32 + lane], -limit), limit);
    q8_1_store_warp((gv / (1.0f + expf(-gv))) * uv, hq + (size_t) (r - rlo) * per + b, lane);
}

// the down rows: grid (n_embd / 32, n); 16 warps x 2 rows; out row r (ld floats) cols [0, n_embd)
template<int TD>
__global__ void __launch_bounds__(GU_WARPS * 32) rows_down_kernel(const uint8_t* __restrict__ base, size_t stride, const unsigned long long* __restrict__ ptrs,
                                                                  size_t down_off, const int* __restrict__ light,
                                                                  const block_q8_1* __restrict__ hq, int rlo, int n_ff,
                                                                  int n_embd, float* __restrict__ out, int ld) {
    const int i = blockIdx.y, chunk = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ uint32_t s_tab[TabWords<TD>::n > 0 ? TabWords<TD>::n : 1];
    load_tab<TD>(s_tab);
    __syncthreads();
    const int slot = light[3 * i], R0 = light[3 * i + 1], NR = light[3 * i + 2];
    const uint8_t* blob = (ptrs != nullptr ? (const uint8_t*) ptrs[i] : base + (size_t) slot * stride) + down_off;
    const size_t rb = rbytes<TD>(n_ff);
    constexpr int RPW = 2;
    const uint8_t* rows[RPW];
    int col[RPW];
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
        col[j] = chunk * (GU_WARPS * RPW) + warp * RPW + j;
        rows[j] = blob + (size_t) col[j] * rb;
    }
    const int per = n_ff / 32;
    const auto x_of = [&](int t) { return hq + (size_t) (R0 + t - rlo) * per; };
    for (int t0 = 0; t0 < NR; t0 += LR_MT2) {
        float s[RPW][LR_MT2];
        rows_multi2<TD, RPW, LR_MT2>(rows, n_ff, NR, lane, s_tab, x_of, s, t0);
        if (lane == 0)
#pragma unroll
            for (int j = 0; j < RPW; ++j)
#pragma unroll
                for (int t = 0; t < LR_MT2; ++t)
                    if (t0 + t < NR) out[(size_t) (R0 + t0 + t) * ld + col[j]] = s[j][t];
    }
}

// x rows (ld floats apart) -> q8_1 rows (n / 32 blocks each): a warp a block
__global__ void rows_q8_kernel(const float* __restrict__ x, int ld, int rows, int n, block_q8_1* __restrict__ q) {
    const int64_t w = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31, per = n / 32;
    if (w >= (int64_t) rows * per) return;
    const int64_t r = w / per, b = w % per;
    q8_1_store_warp(x[(size_t) r * ld + b * 32 + lane], q + (size_t) r * per + b, lane);
}

// ---------------------------------------------------------------- the NextN block's glue
// cat = [rms(emb) * enorm, rms(h) * hnorm] as q8_1 (eh_proj's input); n_embd <= 4096, one block of 1024
__global__ void __launch_bounds__(1024) mtp_in_kernel(const float* __restrict__ emb, const float* __restrict__ h,
                                                      const float* __restrict__ enorm, const float* __restrict__ hnorm,
                                                      float eps, int n, block_q8_1* __restrict__ catq) {
    __shared__ float sred[32];
    const int tid = threadIdx.x, lane = tid & 31;
    float a[4], b[4];
    float sa = 0.0f, sb = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = tid + 1024 * i;
        a[i] = e < n ? emb[e] : 0.0f;
        b[i] = e < n ? h[e] : 0.0f;
        sa += a[i] * a[i];
        sb += b[i] * b[i];
    }
    sa = block_sum(sa, sred);
    sb = block_sum(sb, sred);
    const float ia = rsqrtf(sa / (float) n + eps), ib = rsqrtf(sb / (float) n + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = tid + 1024 * i;
        if (e >= n) break;
        q8_1_store_warp(a[i] * ia * enorm[e], catq + e / 32, lane);
        q8_1_store_warp(b[i] * ib * hnorm[e], catq + (n + e) / 32, lane);
    }
}
// h += add (when given), then x = rms(h) * w and its q8_1; n_embd <= 4096, one block of 1024
__global__ void __launch_bounds__(1024) rms_q8_kernel(float* __restrict__ h, const float* __restrict__ add,
                                                      const float* __restrict__ w, float eps, int n,
                                                      float* __restrict__ x, block_q8_1* __restrict__ xq) {
    __shared__ float sred[32];
    const int tid = threadIdx.x, lane = tid & 31;
    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = tid + 1024 * i;
        float t = e < n ? h[e] : 0.0f;
        if (add != nullptr && e < n) {
            t += add[e];
            h[e] = t;
        }
        v[i] = t;
        ss += t * t;
    }
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) n + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = tid + 1024 * i;
        if (e >= n) break;
        const float xv = v[i] * inv * w[e];
        x[e] = xv;
        q8_1_store_warp(xv, xq + e / 32, lane);
    }
}

// one block: float4 loads, four in flight per thread (the vocabulary is ~0.6 MB; scalar loads ran at ~10 GB/s)
__global__ void __launch_bounds__(1024) argmax_kernel(const float* __restrict__ x, int n, int* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    float best = -INFINITY;
    int bi = 0x7fffffff;
    const auto take = [&](float v, int i) {
        if (v > best || (v == best && i < bi)) {
            best = v;
            bi = i;
        }
    };
    const int B = (int) blockDim.x;
    int tail = 0;
    if ((((uintptr_t) x) & 15u) == 0) {
        const float4* x4 = (const float4*) x;
        const int n4 = n >> 2;
        int i = threadIdx.x;
        for (; i + 3 * B < n4; i += 4 * B) {
            float4 a[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) a[u] = x4[i + u * B];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int b = 4 * (i + u * B);
                take(a[u].x, b);
                take(a[u].y, b + 1);
                take(a[u].z, b + 2);
                take(a[u].w, b + 3);
            }
        }
        for (; i < n4; i += B) {
            const float4 a = x4[i];
            take(a.x, 4 * i);
            take(a.y, 4 * i + 1);
            take(a.z, 4 * i + 2);
            take(a.w, 4 * i + 3);
        }
        tail = 4 * n4;
    }
    for (int i = tail + threadIdx.x; i < n; i += B) take(x[i], i);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { sv[w] = best; si[w] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < (int) (blockDim.x >> 5); ++i)
            if (sv[i] > best || (sv[i] == best && si[i] < bi)) { best = sv[i]; bi = si[i]; }
        *out = bi;
    }
}

int mv_type_ok(int t) {
    switch (t) {
        case kTypeF32: case kTypeBF16: case 12: case 13: case 14: case 8: case 16: case 18: case 19: case 23: case 10: case 11:
        case 17: case 22: case 21: case 29: case 20: case 7: case 42: return 1;
        case exl3::kTypeEXL3: return 1;   // forwarded to the EXL3 kernels (mv below)
        default: return 0;
    }
}

// a failed launch is STICKY: the forward that issued it reports an error (launch_errors) instead of computing on
std::atomic<int> g_launch_errors{0};
thread_local bool g_helper_launch = false;
thread_local cudaError_t g_helper_error = cudaSuccess;
void launch_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm_fast %s: %s\n", what, cudaGetErrorString(e));
        if (g_helper_launch) {
            if (g_helper_error == cudaSuccess) g_helper_error = e;
        } else {
            g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

}  // namespace

bool mv_supported(int type) { return mv_type_ok(type) != 0; }

bool moe_supported(int type) {   // moe_gate_up's and moe_down's instantiations
    return type == 10 || type == 11 || type == 12 || type == 13 || type == 14 || type == 16 || type == 17 ||
           type == 18 || type == 19 || type == 20 || type == 21 || type == 22 || type == 23 || type == 29 || type == 39 ||
           type == 7 || type == 6 || type == 8 || type == 42 || type == 40;   // (40: NVFP4, its scale_2 in MoeDev::nv_scale)   // (Qwen3.8 GGUFs: unsloth's Q5_1 / Q5_0 / Q8_0, ISTA's Q2_0 down experts)
}

int launch_errors() { return g_launch_errors.load(std::memory_order_relaxed) + exl3::launch_errors(); }
void helper_launch_begin() {
    g_helper_error = cudaSuccess;
    g_helper_launch = true;
}
cudaError_t helper_launch_end() {
    g_helper_launch = false;
    return g_helper_error;
}

size_t row_bytes(int type, int64_t n_in) {
    switch (type) {
        case kTypeF32: return (size_t) n_in * 4;
        case kTypeBF16: return (size_t) n_in * 2;
        case 12: return (size_t) (n_in / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n_in / 256) * sizeof(block_q5_K);
        case 14: return (size_t) (n_in / 256) * sizeof(block_q6_K);
        case 8: return (size_t) (n_in / 32) * sizeof(block_q8_0);
        case 16: return (size_t) (n_in / 256) * sizeof(block_iq2_xxs);
        case 18: return (size_t) (n_in / 256) * sizeof(block_iq3_xxs);
        case 19: return (size_t) (n_in / 256) * sizeof(block_iq1_s);
        case 23: return (size_t) (n_in / 256) * sizeof(block_iq4_xs);
        case 10: return (size_t) (n_in / 256) * sizeof(block_q2_K);
        case 11: return (size_t) (n_in / 256) * sizeof(block_q3_K);
        case 17: return (size_t) (n_in / 256) * sizeof(block_iq2_xs);
        case 22: return (size_t) (n_in / 256) * sizeof(block_iq2_s);
        case 21: return (size_t) (n_in / 256) * sizeof(block_iq3_s);
        case 29: return (size_t) (n_in / 256) * sizeof(block_iq1_m);
        case 39: return (size_t) (n_in / 32) * sizeof(block_mxfp4);
        case 40: return (size_t) (n_in / 64) * 36;
        case 20: return (size_t) (n_in / 32) * sizeof(block_iq4_nl);
        case 7: return (size_t) (n_in / 32) * sizeof(block_q5_1);
        case 6: return (size_t) (n_in / 32) * sizeof(block_q5_0);
        case 42: return (size_t) (n_in / 64) * sizeof(block_q2_0);
        default: return 0;
    }
}

bool mvn_supported(int type) {   // (qwen4exp's i-quant projections too: IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S)
    return type == 8 || type == 12 || type == 13 || type == 14 || type == 18 || type == 20 || type == 21 || type == 22 ||
           type == 23 || type == 7 || type == 42 ||
           type == kTypeBF16 || type == kTypeF32;
}

bool mvn(const MvJob* jobs, int n, int nc, cudaStream_t s) {
    if (nc == 1) return mv(jobs, n, s);
    if (n <= 0 || n > kMaxMvJobs || nc < 1) return false;
    if (nc > 8) {
        // a wider call: runs of 2-8 columns (never 1: the one-column kernels sum in another order), each column's
        // sum the same as in any run
        for (int c0 = 0; c0 < nc;) {
            const int rem = nc - c0, take = rem > 8 ? (rem == 9 ? 5 : 8) : rem;
            MvJob sub[kMaxMvJobs];
            for (int i = 0; i < n; ++i) {
                const MvJob& J = jobs[i];
                sub[i] = J;
                const bool f = J.type == kTypeBF16 || J.type == kTypeF32;
                const size_t xs = J.xs ? (size_t) J.xs : (size_t) (f ? J.n_in : J.n_in / 32);
                if (f) {
                    if (J.xf != nullptr) sub[i].xf = J.xf + (size_t) c0 * xs;
                } else {
                    if (J.xq != nullptr) sub[i].xq = (const uint8_t*) J.xq + (size_t) c0 * xs * sizeof(block_q8_1);
                    if (J.xf != nullptr) sub[i].xf = J.xf + (size_t) c0 * J.n_in;
                }
                sub[i].y = J.y + (size_t) c0 * (J.ys ? (size_t) J.ys : (size_t) J.n_out);
            }
            if (!mvn(sub, n, take, s)) return false;
            c0 += take;
        }
        return true;
    }
    for (int i = 0; i < n; ++i)
        if (!mvn_supported(jobs[i].type)) {
            std::fprintf(stderr, "glm_fast mvn: type %d unsupported\n", jobs[i].type);
            return false;
        }
    // one launch per type present (as mv's typed kernels), each compiled for the column count
    constexpr int kTypes[] = {14, 8, 12, 13, 18, 20, 21, 22, 23, 7, 42, kTypeBF16};
    bool done[kMaxMvJobs] = {};
    for (const int T : kTypes) {
        MvBatch b{};
        int acc = 0, m = 0;
        for (int i = 0; i < n; ++i)
            if (!done[i] && (jobs[i].type == T || (T == kTypeBF16 && jobs[i].type == kTypeF32))) {
                b.j[m] = jobs[i];
                acc += (jobs[i].n_out + MV_ROWS - 1) / MV_ROWS;
                b.blk_end[m++] = acc;
                done[i] = true;
            }
        if (m == 0) continue;
        b.n = m;
#define GF_MVN(TT)                                                                      \
        switch (nc) {                                                                   \
            case 2: mvn_kernel_t<TT, 2><<<acc, 256, 0, s>>>(b); break;                  \
            case 3: mvn_kernel_t<TT, 3><<<acc, 256, 0, s>>>(b); break;                  \
            case 4: mvn_kernel_t<TT, 4><<<acc, 256, 0, s>>>(b); break;                  \
            case 5: mvn_kernel_t<TT, 5><<<acc, 256, 0, s>>>(b); break;                  \
            case 6: mvn_kernel_t<TT, 6><<<acc, 256, 0, s>>>(b); break;                  \
            case 7: mvn_kernel_t<TT, 7><<<acc, 256, 0, s>>>(b); break;                  \
            default: mvn_kernel_t<TT, 8><<<acc, 256, 0, s>>>(b); break;                 \
        }
        switch (T) {
            case 14: GF_MVN(14) break;
            case 8: GF_MVN(8) break;
            case 12: GF_MVN(12) break;
            case 13: GF_MVN(13) break;
            case 18: GF_MVN(18) break;
            case 20: GF_MVN(20) break;
            case 21: GF_MVN(21) break;
            case 22: GF_MVN(22) break;
            case 23: GF_MVN(23) break;
            case 7: GF_MVN(7) break;
            case 42: GF_MVN(42) break;
            default:
                switch (nc) {
                    case 2: mvn_kernel_f<2><<<acc, 256, 0, s>>>(b); break;
                    case 3: mvn_kernel_f<3><<<acc, 256, 0, s>>>(b); break;
                    case 4: mvn_kernel_f<4><<<acc, 256, 0, s>>>(b); break;
                    case 5: mvn_kernel_f<5><<<acc, 256, 0, s>>>(b); break;
                    case 6: mvn_kernel_f<6><<<acc, 256, 0, s>>>(b); break;
                    case 7: mvn_kernel_f<7><<<acc, 256, 0, s>>>(b); break;
                    default: mvn_kernel_f<8><<<acc, 256, 0, s>>>(b); break;
                }
        }
#undef GF_MVN
    }
    launch_check("mvn");
    return true;
}

void mv_force_q6k_split(int split4) { g_q6k_force.store(split4); }
void mv_force_split(int split) { g_split_force.store(split == 2 || split == 4 || split == 1 ? split : -1); }
bool mv_q6k_split_choice() { return mv_tune().q6k_split; }

bool mvn_mv(const MvJob* jobs, int n, int nc, cudaStream_t s) {
    if (nc == 1) return mv(jobs, n, s);
    if (n <= 0 || n > kMaxMvJobs || nc < 1) return false;
    // a mv() a column: its pointers at column c (the strides as mvn's)
    const auto per_column = [&](const MvJob* js, int m) {
        for (int c = 0; c < nc; ++c) {
            MvJob sub[kMaxMvJobs];
            for (int i = 0; i < m; ++i) {
                const MvJob& J = js[i];
                sub[i] = J;
                const bool f = J.type == kTypeBF16 || J.type == kTypeF32;
                const size_t xs = J.xs ? (size_t) J.xs : (size_t) (f ? J.n_in : J.n_in / 32);
                if (f) {
                    if (J.xf != nullptr) sub[i].xf = J.xf + (size_t) c * xs;
                } else {
                    if (J.xq != nullptr) sub[i].xq = (const uint8_t*) J.xq + (size_t) c * xs * sizeof(block_q8_1);
                    if (J.xf != nullptr) sub[i].xf = J.xf + (size_t) c * J.n_in;
                }
                sub[i].y = J.y + (size_t) c * (J.ys ? (size_t) J.ys : (size_t) J.n_out);
                sub[i].xs = sub[i].ys = 0;
            }
            if (!mv(sub, m, s)) return false;
        }
        return true;
    };
#if defined(STRATA_USE_HIP)
    return per_column(jobs, n);   // (HIP's mv fuses small calls of mixed types: not reproduced here)
#else
    if (nc > 8) {   // (runs of up to 8 columns: a column's sum does not depend on the run)
        for (int c0 = 0; c0 < nc; c0 += 8) {
            const int take = std::min(8, nc - c0);
            MvJob sub[kMaxMvJobs];
            for (int i = 0; i < n; ++i) {
                const MvJob& J = jobs[i];
                sub[i] = J;
                const bool f = J.type == kTypeBF16 || J.type == kTypeF32;
                const size_t xs = J.xs ? (size_t) J.xs : (size_t) (f ? J.n_in : J.n_in / 32);
                if (f) {
                    if (J.xf != nullptr) sub[i].xf = J.xf + (size_t) c0 * xs;
                } else {
                    if (J.xq != nullptr) sub[i].xq = (const uint8_t*) J.xq + (size_t) c0 * xs * sizeof(block_q8_1);
                    if (J.xf != nullptr) sub[i].xf = J.xf + (size_t) c0 * J.n_in;
                }
                sub[i].y = J.y + (size_t) c0 * (J.ys ? (size_t) J.ys : (size_t) J.n_out);
                if (!sub[i].xs) sub[i].xs = (int) xs;
                if (!sub[i].ys) sub[i].ys = J.n_out;
            }
            if (!mvn_mv(sub, n, take, s)) return false;
        }
        return true;
    }
    for (int i = 0; i < n; ++i)
        if (!mv_type_ok(jobs[i].type) || jobs[i].type == exl3::kTypeEXL3) {
            std::fprintf(stderr, "glm_fast mvn_mv: type %d unsupported\n", jobs[i].type);
            return false;
        }
    // mv()'s rule, for the same call: the typed kernels (Q6_K, Q8_0, Q4_K, Q5_K) with the split of a row its per-type
    // block count picks, the rest one warp a row (row_dot<T> / the F32 and BF16 dots - mvn's kernels sum the same way)
    static const bool typed = getenv("STRATA_GLM_MV_GENERIC") == nullptr;
    static const int split_env = [] { const char* v = getenv("STRATA_GLM_MV_SPLIT"); return v ? std::atoi(v) : -1; }();
    const int q6k_force = g_q6k_force.load(std::memory_order_relaxed);
    const bool q6k_split = q6k_force >= 0 ? q6k_force != 0 : mv_tune().q6k_split;
    bool done[kMaxMvJobs] = {};
    constexpr int kTyped[] = {14, 8, 12, 13};
    if (typed) {
        for (const int T : kTyped) {
            MvBatch b{};
            int acc = 0, m = 0;
            for (int i = 0; i < n; ++i)
                if (jobs[i].type == T) {
                    b.j[m] = jobs[i];
                    acc += (jobs[i].n_out + MV_ROWS - 1) / MV_ROWS;
                    b.blk_end[m++] = acc;
                    done[i] = true;
                }
            if (m == 0) continue;
            b.n = m;
            const int sforce = g_split_force.load(std::memory_order_relaxed);
            const int split = sforce > 0 ? sforce
                            : split_env >= 0 ? (split_env == 2 || split_env == 4 ? split_env : 1)
                            : (acc < 256 || (q6k_split && T == 14 && acc <= 4096)) ? 4 : 1;
            if (split > 1) {
                const int rpb = MV_ROWS / split;
                acc = 0;
                for (int q = 0; q < m; ++q) {
                    acc += (b.j[q].n_out + rpb - 1) / rpb;
                    b.blk_end[q] = acc;
                }
            }
#define GF_MVNS(TT, SP)                                                                  \
            switch (nc) {                                                                \
                case 2: mvn_kernel_ts<TT, SP, 2><<<acc, 256, 0, s>>>(b); break;          \
                case 3: mvn_kernel_ts<TT, SP, 3><<<acc, 256, 0, s>>>(b); break;          \
                case 4: mvn_kernel_ts<TT, SP, 4><<<acc, 256, 0, s>>>(b); break;          \
                case 5: mvn_kernel_ts<TT, SP, 5><<<acc, 256, 0, s>>>(b); break;          \
                case 6: mvn_kernel_ts<TT, SP, 6><<<acc, 256, 0, s>>>(b); break;          \
                case 7: mvn_kernel_ts<TT, SP, 7><<<acc, 256, 0, s>>>(b); break;          \
                default: mvn_kernel_ts<TT, SP, 8><<<acc, 256, 0, s>>>(b); break;         \
            }
#define GF_MVN1(TT)                                                                      \
            switch (nc) {                                                                \
                case 2: mvn_kernel_t<TT, 2><<<acc, 256, 0, s>>>(b); break;               \
                case 3: mvn_kernel_t<TT, 3><<<acc, 256, 0, s>>>(b); break;               \
                case 4: mvn_kernel_t<TT, 4><<<acc, 256, 0, s>>>(b); break;               \
                case 5: mvn_kernel_t<TT, 5><<<acc, 256, 0, s>>>(b); break;               \
                case 6: mvn_kernel_t<TT, 6><<<acc, 256, 0, s>>>(b); break;               \
                case 7: mvn_kernel_t<TT, 7><<<acc, 256, 0, s>>>(b); break;               \
                default: mvn_kernel_t<TT, 8><<<acc, 256, 0, s>>>(b); break;              \
            }
#define GF_MVNT(TT)                                                                      \
            if (split == 4) { GF_MVNS(TT, 4) } else if (split == 2) { GF_MVNS(TT, 2) } else { GF_MVN1(TT) }
            switch (T) {
                case 14: GF_MVNT(14) break;
                case 8: GF_MVNT(8) break;
                case 12: GF_MVNT(12) break;
                case 13: GF_MVNT(13) break;
            }
#undef GF_MVNT
#undef GF_MVN1
#undef GF_MVNS
        }
    }
    // the rest: mvn's one-warp-a-row kernels where it has them (the same loop as row_dot<T> / row_dot_bf16 / _f32),
    // else a mv() a column
    MvJob rest[kMaxMvJobs], other[kMaxMvJobs];
    int nr = 0, no = 0;
    for (int i = 0; i < n; ++i) {
        if (done[i]) continue;
        if (mvn_supported(jobs[i].type)) rest[nr++] = jobs[i];
        else other[no++] = jobs[i];
    }
    if (nr > 0 && !mvn(rest, nr, nc, s)) return false;
    if (no > 0 && !per_column(other, no)) return false;
    launch_check("mvn_mv");
    return true;
#endif
}

bool mv(const MvJob* jobs, int n, cudaStream_t s) {
    if (n <= 0 || n > kMaxMvJobs) return false;
    for (int i = 0; i < n; ++i)
        if (!mv_type_ok(jobs[i].type)) {
            std::fprintf(stderr, "glm_fast mv: type %d unsupported\n", jobs[i].type);
            return false;
        }
#if defined(STRATA_USE_HIP)
    // HIP: a small call of several types in ONE launch, 4 warps a row (mv_kernel_gs) - an AI PRO R9700 spends 15-29 us
    // on a small GEMV whatever its size (an RTX 5090 3-6), and DeepSeek V4's projections of x went out as two or three
    // launches a layer (Q5_K q_a, the Q8_0 compressors, the F32 indexer weights).  STRATA_GLM_MV_FUSE=0: one per type
    static const bool fuse = getenv("STRATA_GLM_MV_FUSE") == nullptr || std::atoi(getenv("STRATA_GLM_MV_FUSE")) != 0;
    if (fuse && n > 1) {
        int rows = 0;
        bool mixed = false, exl = false;
        for (int i = 0; i < n; ++i) {
            rows += jobs[i].n_out;
            mixed = mixed || jobs[i].type != jobs[0].type;
            exl = exl || jobs[i].type == exl3::kTypeEXL3;
        }
        if (mixed && !exl && rows <= 8192) {
            MvBatch b{};
            int acc = 0;
            for (int i = 0; i < n; ++i) {
                b.j[i] = jobs[i];
                acc += (jobs[i].n_out + MV_ROWS / 4 - 1) / (MV_ROWS / 4);
                b.blk_end[i] = acc;
            }
            b.n = n;
            mv_kernel_gs<4><<<grid_pad(acc, 8), 256, 0, s>>>(b);
            launch_check("mv (one launch)");
            return true;
        }
    }
#endif
    // the jobs of the common dense types go to a kernel compiled for their type (one launch per type present), the
    // rest to the generic one; the jobs are independent, so their order does not matter
    static const bool typed = getenv("STRATA_GLM_MV_GENERIC") == nullptr;
    // STRATA_GLM_MV_SPLIT: 0 one warp a row always; 2 / 4 that split for every typed launch; unset: 4 warps a row
    // under 256 blocks (mv_kernel_ts).  nsys on an RTX 5090: the 128-block Q5_K q_a 4.3 -> 3.0 us and the 192-block
    // HCA projections 6.3 -> 5.3, but the 512-block wo_b stays ~23.5 (1.5 TB/s is its long rows' ceiling) and the
    // 512-block Q5_K shared gate/up got slower (8.4 -> 9.3: the dot's arithmetic, not latency) - so only the small ones.
    // Q6_K is the exception: one warp a row leaves its 6-bit dot latency-bound at every size measured (the attn-q6k
    // pack's wo_a 1,024 blocks 24.1 -> 18.1 us, wo_b 512 blocks 23.0 -> 19.8, q_b 4,096 blocks 19.2 -> 18.5 with 4
    // warps a row; 1.14 -> 1.52 TB/s on wo_a), so Q6_K splits up to 4,096 blocks (STRATA_GLM_MV_SPLIT=0: never) - where
    // mv_tune says so (not gfx12)
    const int q6k_force = g_q6k_force.load(std::memory_order_relaxed);
    const bool q6k_split = q6k_force >= 0 ? q6k_force != 0 : mv_tune().q6k_split;
    static const int split_env = [] { const char* v = getenv("STRATA_GLM_MV_SPLIT"); return v ? std::atoi(v) : -1; }();
    constexpr int kTyped[] = {14, 8, 12, 13};
    bool done[kMaxMvJobs] = {};
    // EXL3 weights (w: an exl3::Mat): their own kernels, F32 input when the job has it, else the q8_1
    {
        exl3::MvJob xj[kMaxMvJobs];
        int nx = 0;
        for (int i = 0; i < n; ++i)
            if (jobs[i].type == exl3::kTypeEXL3) {
                xj[nx].m = (const exl3::Mat*) jobs[i].w;
                xj[nx].xf = jobs[i].xf;
                xj[nx].xq = jobs[i].xq;
                xj[nx].y = jobs[i].y;
                xj[nx].bias = jobs[i].bias;
                xj[nx].alpha = jobs[i].alpha;
                ++nx;
                done[i] = true;
            }
        if (nx > 0 && !exl3::mv(xj, nx, s)) return false;
        if (nx == n) return true;
    }
    // the typed kernels' early L2 pull of their weight rows (mv_prefetch_l2): with PDL by default
    // (STRATA_GLM_PDL_PREFETCH=0: PDL without it), STRATA_GLM_PDL_PREFETCH=2: also without PDL (A/B)
    static const int pf_env = [] { const char* v = getenv("STRATA_GLM_PDL_PREFETCH"); return v == nullptr ? 1 : std::atoi(v); }();
    const int pf = pf_env >= 2 || (pf_env == 1 && pdl_enabled()) ? 1 : 0;
    if (typed) {
        for (const int T : kTyped) {
            MvBatch b{};
            int acc = 0, m = 0;
            for (int i = 0; i < n; ++i)
                if (jobs[i].type == T) {
                    b.j[m] = jobs[i];
                    acc += (jobs[i].n_out + MV_ROWS - 1) / MV_ROWS;
                    b.blk_end[m++] = acc;
                    done[i] = true;
                }
            if (m == 0) continue;
            b.n = m;
            b.pf = pf;
            const int sforce = g_split_force.load(std::memory_order_relaxed);
            const int split = sforce > 0 ? sforce
                            : split_env >= 0 ? (split_env == 2 || split_env == 4 ? split_env : 1)
                            : (acc < 256 || (q6k_split && T == 14 && acc <= 4096)) ? 4 : 1;
            if (split > 1) {   // the block ranges at MV_ROWS / split rows a block
                const int rpb = MV_ROWS / split;
                acc = 0;
                for (int q = 0; q < m; ++q) {
                    acc += (b.j[q].n_out + rpb - 1) / rpb;
                    b.blk_end[q] = acc;
                }
            }
#define GF_MV_T(TT) \
            switch (split) { \
                case 4: pdl_launch(mv_kernel_ts<TT, 4>, grid_pad(acc, 8), 256, 0, s, b); break; \
                case 2: pdl_launch(mv_kernel_ts<TT, 2>, grid_pad(acc, 8), 256, 0, s, b); break; \
                default: pdl_launch(mv_kernel_t<TT>, grid_pad(acc, 8), 256, 0, s, b); break; \
            }
            switch (T) {
                case 14: GF_MV_T(14) break;
                case 8: GF_MV_T(8) break;
                case 12: GF_MV_T(12) break;
                case 13: GF_MV_T(13) break;
            }
#undef GF_MV_T
        }
    }
    MvBatch b{};
    int acc = 0, m = 0;
    for (int i = 0; i < n; ++i) {
        if (done[i]) continue;
        b.j[m] = jobs[i];
        acc += (jobs[i].n_out + MV_ROWS - 1) / MV_ROWS;
        b.blk_end[m++] = acc;
    }
    if (m > 0) {
        b.n = m;
        b.pf = pf;
        pdl_launch(mv_kernel, grid_pad(acc, 8), 256, 0, s, b);
    }
    launch_check("mv");
    return true;
}

void quantize_q8_1(const float* x, void* xq, int n, cudaStream_t s) {
    pdl_launch(quantize_kernel, (n + 255) / 256, 256, 0, s, x, (block_q8_1*) xq, n);
    launch_check("quantize");
}

// the grid-barrier kernel needs every block resident at once; checked once per process (and per device count
// of SMs), the single-tail kernel otherwise.  STRATA_GLM_HC1=1 forces the old one (A/B).
// hc3's widths: 4096 (32 blocks); with HIP DeepSeek V4.1's 5120 too (40 blocks, its lagged collapse; HC3V41-NUMBERS,
// STRATA_GLM_HC3_V41=0: hc2 as before).  NVIDIA keeps hc2 for V4.1 (not measured there)
static bool hc3_width(int n_embd) {
#if defined(STRATA_USE_HIP)
    static const bool v41 = [] { const char* v = getenv("STRATA_GLM_HC3_V41"); return v == nullptr || std::atoi(v) != 0; }();
    if (v41 && n_embd == 5120) return true;
#endif
    return n_embd == 4096;
}

static int hc_mode(int n_embd) {
    static int mode = -1;
    if (mode < 0) {
        int dev = 0, sms = 0, per = 0, per3 = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, hc2_kernel, HC_THREADS, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per3, hc3_kernel, HC_THREADS, 0);
        mode = (getenv("STRATA_GLM_HC1") == nullptr && per * sms >= n_embd / HC2_EB) ? 2 : 1;
        if (mode == 2 && getenv("STRATA_GLM_HC2") == nullptr && hc3_width(n_embd) && per3 * sms >= n_embd / HC2_EB)
            mode = 3;
    }
    return mode;
}

bool hc_fuses_cpu(int n_embd) { return hc_mode(n_embd) == 3 && hc3_width(n_embd); }

void hc(const HcArgs& a, cudaStream_t s) {
    const int mode = hc_mode(a.n_embd);
    if (a.cpu_seq != nullptr && a.block_out != nullptr && mode != 3) {
        // (a caller that asked hc_fuses_cpu() never lands here: the CPU lane's answer in its own launch first)
        MoeDev d;
        d.cpu_seq = (unsigned int*) a.cpu_seq;
        d.cpu_ans = a.cpu_ans;
        moe_cpu_wait(d, a.n_embd, (float*) a.block_out, s);
        HcArgs b = a;
        b.cpu_seq = nullptr;
        hc(b, s);
        return;
    }
    if (mode == 3) {   // (deepseek41's lagged collapse: HIP only, hc3_width)
        HcArgs g3 = a;
        // (nsys on an RTX 5090: 8.5 -> 8.1 us a launch; the sums round differently, so off unless asked)
        static const bool gram = [] { const char* v = getenv("STRATA_GLM_HC_GRAM"); return v != nullptr && std::atoi(v) != 0; }();
        g3.gram = gram ? 1 : 0;
        hc3_kernel<<<a.n_embd / HC2_EB, HC_THREADS, 0, s>>>(g3);
        launch_check("hc3");
        return;
    }
    if (mode == 2 && a.n_embd % HC2_EB == 0) {
        hc2_kernel<<<a.n_embd / HC2_EB, HC_THREADS, 0, s>>>(a);
        launch_check("hc2");
        return;
    }
    const int blocks = 4 * a.n_embd / HC_CHUNK;
    hc_kernel<<<blocks, HC_THREADS, 0, s>>>(a);
    launch_check("hc");
}

void hc_post(const float* block_out, const float* R_old, const float* post, const float* comb, int n_embd,
             float* R_new, cudaStream_t s) {
    hc_post_kernel<<<(4 * n_embd + 255) / 256, 256, 0, s>>>(block_out, R_old, post, comb, n_embd, R_new);
    launch_check("hc_post");
}

void head_prep(const float* R, const float* norm_w, float eps, int n_embd, float* x, void* xq, cudaStream_t s) {
    head_prep_kernel<<<1, 1024, 0, s>>>(R, norm_w, eps, n_embd, x, (block_q8_1*) xq);
    launch_check("head_prep");
}

void abl_project(float* x, int rows, int ld, int n, const float* q, int r, float s, cudaStream_t st, const int* on) {
    if (rows <= 0 || r < 1 || r > 4) return;
    abl_project_kernel<<<rows, 256, 0, st>>>(x, ld, n, q, r, s, on);
    launch_check("abl_project");
}

__global__ void abl_capture_kernel(const float* __restrict__ x, int ld, int n, int streams, float* __restrict__ cap) {
    const float* xr = x + (size_t) blockIdx.y * ld;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.0f;
    for (int k = 0; k < streams; ++k) a += xr[(size_t) k * n + i];
    cap[(size_t) blockIdx.y * n + i] = a / (float) streams;
}

void abl_capture(const float* x, int rows, int ld, int n, int streams, float* cap, cudaStream_t st) {
    if (rows <= 0) return;
    abl_capture_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) rows), 256, 0, st>>>(x, ld, n, streams, cap);
    launch_check("abl_capture");
}

void embed_streams(const float* emb, float* R, int n_embd, cudaStream_t s) {
    embed_streams_kernel<<<(4 * n_embd + 255) / 256, 256, 0, s>>>(emb, R, n_embd);
    launch_check("embed_streams");
}

bool embed_rows_q(int type, const void* table, const int* toks, int T, int n_embd, float* out, cudaStream_t s) {
    if (type == 8 && n_embd % 32 == 0) {
        const int nb = n_embd / 32;
        embed_q8_rows_kernel<<<dim3((unsigned) ((nb + 127) / 128), (unsigned) T), 128, 0, s>>>(
            (const block_q8_0*) table, toks, n_embd, out);
        launch_check("embed_q8_rows");
        return true;
    }
    if (type == 12 && n_embd % 256 == 0) {
        const int n = n_embd / 256 * 32;
        embed_q4k_rows_kernel<<<dim3((unsigned) ((n + 127) / 128), (unsigned) T), 128, 0, s>>>(
            (const block_q4_K*) table, toks, n_embd, out);
        launch_check("embed_q4k_rows");
        return true;
    }
    if (type == 13 && n_embd % 256 == 0) {
        const int n = n_embd / 256 * 32;
        embed_q5k_rows_kernel<<<dim3((unsigned) ((n + 127) / 128), (unsigned) T), 128, 0, s>>>(
            (const block_q5_K*) table, toks, n_embd, out);
        launch_check("embed_q5k_rows");
        return true;
    }
    if (type == 14 && n_embd % 256 == 0) {
        const int n = n_embd / 256 * 32;
        embed_q6k_rows_kernel<<<dim3((unsigned) ((n + 127) / 128), (unsigned) T), 128, 0, s>>>(
            (const block_q6_K*) table, toks, n_embd, out);
        launch_check("embed_q6k_rows");
        return true;
    }
    return false;
}

bool embed_rows_q_supported(int type, int n_embd) {
    return (type == 8 && n_embd % 32 == 0) || ((type == 12 || type == 13 || type == 14) && n_embd % 256 == 0);
}

void kda_prep(const KdaPrepArgs& a, cudaStream_t s) {
    kda_prep_kernel<<<dim3(a.n_head, 11), 128, 0, s>>>(a);   // 3 conv roles + 2 x 4 gate quarters
    launch_check("kda_prep");
}

void kda_rec(const float* q, const float* k, const float* v, const float* g1, const float* beta_raw, float* state,
             const float* g2, const float* norm_w, float eps, int n_head, int head_dim, void* out_q8_1,
             cudaStream_t s) {
    if (head_dim != 128) {
        std::fprintf(stderr, "glm_fast kda_rec: head_dim %d (the kernel is built for 128)\n", head_dim);
        return;
    }
    // the column split (STRATA_GLM_KDA_SPLIT=1, default off - no gain measured on GB10 or the 5090; off: one block a head); its o buffer is made outside a graph capture
    static const bool split_on = [] { const char* e = std::getenv("STRATA_GLM_KDA_SPLIT"); return e != nullptr && std::atoi(e) != 0; }();
    static std::mutex mu;
    static std::map<int, std::pair<float*, int>> obuf;   // device -> (buffer, floats)
    float* ob = nullptr;
    if (split_on) {
        int dev = 0;
        cudaGetDevice(&dev);
        std::lock_guard<std::mutex> lk(mu);
        auto it = obuf.find(dev);
        if (it != obuf.end() && it->second.second >= n_head * 128) {
            ob = it->second.first;
        } else {
            cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(s, &cs);
            if (cs == cudaStreamCaptureStatusNone) {
                float* p = nullptr;
                if (cudaMalloc(&p, (size_t) n_head * 128 * sizeof(float)) == cudaSuccess) {
                    if (it != obuf.end()) cudaFree(it->second.first);
                    obuf[dev] = {p, n_head * 128};
                    ob = p;
                } else {
                    cudaGetLastError();
                }
            }
        }
    }
    if (ob) {
        kda_rec_cols_kernel<<<dim3(n_head, 4), 64, 0, s>>>(q, k, v, g1, beta_raw, state, ob);
        kda_out_kernel<<<n_head, 256, 0, s>>>(ob, g2, norm_w, eps, (block_q8_1*) out_q8_1);
    } else {
        kda_rec_kernel<<<n_head, 256, 0, s>>>(q, k, v, g1, beta_raw, state, g2, norm_w, eps, n_head,
                                              (block_q8_1*) out_q8_1);
    }
    launch_check("kda_rec");
}

void dsa_prep(const DsaPrepArgs& a, cudaStream_t s) {
    dsa_prep_kernel<<<3, 512, 0, s>>>(a);
    launch_check("dsa_prep");
}

void dsa_score(const float* iq, const float* pooled, const float* iw, int key_dim, int idx_heads, int n_pools,
               float* score, cudaStream_t s) {
    if (n_pools <= 0) return;
    const size_t smem = ((size_t) idx_heads * key_dim + idx_heads) * sizeof(float);
    dsa_score_kernel<<<(n_pools + 7) / 8, 256, smem, s>>>(iq, pooled, iw, key_dim, idx_heads, n_pools, score);
    launch_check("dsa_score");
}

void dsa_select(const float* score, int n_vis, int kpool, int top_pools, int n_sel, int pos, int* cells,
                cudaStream_t s) {
    dsa_select_kernel<<<1, 1024, 0, s>>>(score, n_vis, kpool, top_pools, n_sel, pos, cells);
    launch_check("dsa_select");
}

void mla(const float* q, const uint16_t* wk_b, const uint16_t* wv_b, const uint16_t* lat, const int* cells, int n_sel,
         int n_head, int qk_nope, int kv_lora, int v_head, void* out_q8_1, cudaStream_t s, bool lat_q8,
         const void* wk_b_q8, const void* wv_b_q8) {
    const uint8_t* wk8 = (const uint8_t*) wk_b_q8;
    const uint8_t* wv8 = (const uint8_t*) wv_b_q8;
    // scratch for q_abs and ctx (n_head * kv_lora each), per device
    static float* scratch[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (getenv("STRATA_GLM_MLA1") != nullptr || dev >= 16 || kv_lora % 32 || v_head % 32) {
        const size_t smem = ((size_t) qk_nope + 2 * kv_lora + v_head + n_sel) * sizeof(float);
        mla_kernel<<<n_head, 256, smem, s>>>(q, wk_b, wv_b, lat, cells, n_sel, qk_nope, kv_lora, v_head,
                                             (block_q8_1*) out_q8_1, lat_q8, wk8, wv8);
        launch_check("mla");
        return;
    }
    if (scratch[dev] == nullptr && cudaMalloc(&scratch[dev], (size_t) 2 * n_head * kv_lora * sizeof(float)) != cudaSuccess) {
        scratch[dev] = nullptr;
        cudaGetLastError();
        const size_t smem = ((size_t) qk_nope + 2 * kv_lora + v_head + n_sel) * sizeof(float);
        mla_kernel<<<n_head, 256, smem, s>>>(q, wk_b, wv_b, lat, cells, n_sel, qk_nope, kv_lora, v_head,
                                             (block_q8_1*) out_q8_1, lat_q8, wk8, wv8);
        launch_check("mla");
        return;
    }
    float* q_abs = scratch[dev];
    float* ctx = q_abs + (size_t) n_head * kv_lora;
    if (wk8 != nullptr)
        headwise_gemv_kernel<true><<<n_head * kv_lora / 32, 256, 0, s>>>(wk8, q, kv_lora, qk_nope, q_abs, nullptr);
    else
        headwise_gemv_kernel<false><<<n_head * kv_lora / 32, 256, 0, s>>>(wk_b, q, kv_lora, qk_nope, q_abs, nullptr);
    // the split attention: chunks of MLA_CHUNK cells, partials per (chunk, head), then the merge
    static float* part[16] = {};
    static int part_chunks[16] = {};
    const int n_chunks = (n_sel + MLA_CHUNK - 1) / MLA_CHUNK;
    const size_t smem = (size_t) MLA_CHUNK * kv_lora * sizeof(uint16_t) + (size_t) MLA_HPB * MLA_CHUNK * sizeof(float);
    // the opt-in must be EXACTLY what the launch needs (or at most the device's opt-in minus the kernel's static
    // shared memory): asking for the whole 96 KB fails on a V100 (s_cell is static), and every launch then failed
    // with "invalid argument" - the attention silently contributed nothing (fixed 2026-10-06)
    static int attr_smem[16] = {};
    static bool split_ok[16] = {true, true, true, true, true, true, true, true,
                                true, true, true, true, true, true, true, true};
    if (split_ok[dev] && attr_smem[dev] < (int) smem) {
        if (cudaFuncSetAttribute(mla_split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) == cudaSuccess) {
            attr_smem[dev] = (int) smem;
        } else {
            cudaGetLastError();
            split_ok[dev] = false;
            std::fprintf(stderr, "glm_fast mla: the split kernel cannot get %zu bytes of shared memory - the one-pass "
                                 "attention instead\n", smem);
        }
    }
    if (split_ok[dev] && kv_lora <= 512 && n_head <= 128 && getenv("STRATA_GLM_MLA_ATTN1") == nullptr) {
        if (part_chunks[dev] < n_chunks) {
            if (part[dev]) cudaFree(part[dev]);
            const int want = std::max(n_chunks, 80);
            if (cudaMalloc(&part[dev], (size_t) want * n_head * (kv_lora + 2) * sizeof(float)) != cudaSuccess) {
                cudaGetLastError();
                part[dev] = nullptr;
                part_chunks[dev] = 0;
            } else {
                part_chunks[dev] = want;
            }
        }
    }
    if (split_ok[dev] && part[dev] != nullptr && part_chunks[dev] >= n_chunks && kv_lora <= 512 &&
        getenv("STRATA_GLM_MLA_ATTN1") == nullptr) {
        mla_split_kernel<<<dim3(n_chunks, (n_head + MLA_HPB - 1) / MLA_HPB), 256, smem, s>>>(
            q_abs, lat, cells, n_sel, n_head, qk_nope, kv_lora, part[dev], lat_q8);
        mla_combine_kernel<<<n_head, 256, 0, s>>>(part[dev], n_chunks, n_head, kv_lora, ctx);
    } else {
        mla_attn_kernel<<<n_head, 256, ((size_t) kv_lora + n_sel) * sizeof(float), s>>>(q_abs, lat, cells, n_sel,
                                                                                        qk_nope, kv_lora, ctx, lat_q8);
    }
    if (wv8 != nullptr)
        headwise_gemv_kernel<true><<<n_head * v_head / 32, 256, 0, s>>>(wv8, ctx, v_head, kv_lora, nullptr,
                                                                        (block_q8_1*) out_q8_1);
    else
        headwise_gemv_kernel<false><<<n_head * v_head / 32, 256, 0, s>>>(wv_b, ctx, v_head, kv_lora, nullptr,
                                                                         (block_q8_1*) out_q8_1);
    launch_check("mla2");
}

template<typename T>
__global__ void q8_0_widen_kernel(const uint8_t* __restrict__ src, T* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float v = q8_0_at(src, (size_t) i);
        if constexpr (sizeof(T) == 4) dst[i] = v;
        else dst[i] = __half_as_ushort(__float2half_rn(v));
    }
}
void q8_0_to_f32(const void* src, float* dst, int64_t n, cudaStream_t s) {
    q8_0_widen_kernel<float><<<1024, 256, 0, s>>>((const uint8_t*) src, dst, n);
    launch_check("q8_0_to_f32");
}
void q8_0_to_f16(const void* src, uint16_t* dst, int64_t n, cudaStream_t s) {
    q8_0_widen_kernel<uint16_t><<<1024, 256, 0, s>>>((const uint8_t*) src, dst, n);
    launch_check("q8_0_to_f16");
}

void swiglu_q8(const float* gate, const float* up, float limit, int n, void* hq, cudaStream_t s) {
    swiglu_q8_kernel<<<(n + 255) / 256, 256, 0, s>>>(gate, up, limit, n, (block_q8_1*) hq);
    launch_check("swiglu_q8");
}

__global__ void sh_swiglu_kernel(const float* __restrict__ sh_gate, const float* __restrict__ sh_up, float sh_limit,
                                 int n_ff_sh, block_q8_1* __restrict__ sh_hq) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    // moe_route_kernel's blocks > 0, unchanged: the shared expert's swiglu + q8_1 (warp-aligned 32-runs)
    const int i = blockIdx.x * ROUTE_THREADS + threadIdx.x;
    if (i >= n_ff_sh) return;
    const float g = fminf(sh_gate[i], sh_limit);
    const float u = fminf(fmaxf(sh_up[i], -sh_limit), sh_limit);
    q8_1_store_warp((g / (1.0f + expf(-g))) * u, sh_hq + i / 32, threadIdx.x & 31);
}

void sh_swiglu(const float* sh_gate, const float* sh_up, float sh_limit, int n_ff_sh, void* sh_hq, cudaStream_t s) {
    pdl_launch(sh_swiglu_kernel, (n_ff_sh + ROUTE_THREADS - 1) / ROUTE_THREADS, ROUTE_THREADS, 0, s, sh_gate, sh_up, sh_limit,
                                                                                            n_ff_sh, (block_q8_1*) sh_hq);
    launch_check("sh_swiglu");
}

void moe_route(const float* logits, const float* bias, int n_expert, int k, float w_scale, bool norm_w, int layer,
               const float* x, int n_embd, const MoeDev& d, const float* sh_gate, const float* sh_up, float sh_limit,
               int n_ff_sh, void* sh_hq, cudaStream_t s, const float* pred_logits, const float* pred_bias,
               int max_prefetch, const float* ahead_logits, const float* const* ahead_bias, int n_ahead,
               int skip_from, unsigned long long cpu_plan, int promote_min, int pf_rank, int gating,
               const int* hash_ids, float skip_w, int skip_disk, const void* xq, int win_more) {
    RouteArgs a{logits, bias, n_expert, k, w_scale, norm_w ? 1 : 0, layer, x, n_embd, d,
                sh_gate, sh_up, sh_limit, n_ff_sh, (block_q8_1*) sh_hq, pred_logits, pred_bias, max_prefetch,
                ahead_logits, {}, 0, skip_from, d.cpu_seq != nullptr ? cpu_plan : 0ull, promote_min,
                pf_rank > 0 ? pf_rank : k, gating, hash_ids, skip_w, 0, skip_disk, (const block_q8_1*) xq, 0.0f, 0.0f};
    static const float affinity = [] {
        const char* v = getenv("STRATA_GLM_ROUTE_AFFINITY");
        return v ? std::max(0.0f, (float) std::atof(v)) : 0.0f;
    }();
    static const float affinity_ram = [] {
        const char* v = getenv("STRATA_GLM_ROUTE_AFFINITY_RAM");
        return v ? std::max(0.0f, (float) std::atof(v)) : 0.0f;
    }();
    a.affinity = affinity;
    a.affinity_ram = affinity_ram;
    static const int prof = [] { const char* v = getenv("STRATA_GLM_ROUTE_PROF"); return v != nullptr && std::atoi(v) != 0; }();
    a.prof = prof;
    a.win_more = win_more;
    // STRATA_GLM_ROUTE_EARLY=1: x written at the route kernel's start (float4, beside the scores and the plan);
    // default 0: after the plan, as before - measured the same on an RTX 5090 + iGPU lane (the system fence, not the
    // store, is what the request waits for): 3 x 2 answers, 83.4 / 85.6 against 83.6 / 91.2 tok/s
    static const int early = [] {
        const char* v = getenv("STRATA_GLM_ROUTE_EARLY");
        return v != nullptr && std::atoi(v) != 0;
    }();
    a.early_x = early && (n_embd & 3) == 0 ? 1 : 0;
    if (ahead_logits != nullptr && ahead_bias != nullptr && n_expert <= 512) {
        a.n_ahead = std::min(n_ahead, kAhead);
        for (int i = 0; i < a.n_ahead; ++i) a.ahead_bias[i] = ahead_bias[i];
    }
    const int blocks = 1 + (n_ff_sh + ROUTE_THREADS - 1) / ROUTE_THREADS;
    pdl_launch(moe_route_kernel, blocks, ROUTE_THREADS, 0, s, a);
    launch_check("moe_route");
}

void moe_wait(const MoeDev& d, int n_embd, cudaStream_t s) {
    pdl_launch(moe_wait_kernel, 1, 256, 0, s, d, n_embd);
    launch_check("moe_wait");
}

void hc_defer_add(const MoeDev& d, const float* post, int n_embd, float* R, cudaStream_t s) {
    if (d.cpu_seq == nullptr) return;
    static const int cwb = [] { const char* v = getenv("STRATA_GLM_CPU_WAIT_BLOCKS"); return v ? std::max(1, std::atoi(v)) : 8; }();
    hc_defer_add_kernel<<<cwb, 128, 0, s>>>(d, post, n_embd, R);
    launch_check("hc_defer_add");
}
void copy_floats(float* dst, const float* src, int n, cudaStream_t s) {
    copy_floats_kernel<<<(n + 127) / 128, 128, 0, s>>>(dst, src, n);
    launch_check("copy_floats");
}

void moe_cpu_wait(const MoeDev& d, int n_embd, float* out, cudaStream_t s, int mode) {
    if (d.cpu_seq == nullptr) return;
    static const int cwb = [] { const char* v = getenv("STRATA_GLM_CPU_WAIT_BLOCKS"); return v ? std::max(1, std::atoi(v)) : 8; }();
    // mode bit 2: the route before the last one's answer (deferred from the layer before), bit 1: the last route's
    if (mode & 2) pdl_launch(moe_cpu_wait_kernel, cwb, 128, 0, s, d, n_embd, out, 1);
    if (mode & 1) pdl_launch(moe_cpu_wait_kernel, cwb, 128, 0, s, d, n_embd, out, 0);   // STRATA_GLM_CPU_WAIT_BLOCKS: the blocks (A/B)
    launch_check("moe_cpu_wait");
}

void moe_cpu_wait_cols(const CpuWaitCols& c, const void* cpu_ans, int n_embd, cudaStream_t s) {
    if (c.n <= 0) return;
    static const int cwb = [] { const char* v = getenv("STRATA_GLM_CPU_WAIT_BLOCKS"); return v ? std::max(1, std::atoi(v)) : 8; }();
    pdl_launch(moe_cpu_wait_cols_kernel, dim3((unsigned) cwb, (unsigned) c.n), 128, 0, s, c, cpu_ans, n_embd);
    launch_check("moe_cpu_wait_cols");
}

void moe_prefetch(const MoeDev& d, size_t blob_bytes, cudaStream_t s) {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) sms = 80;
    }
    // half the SMs: it runs beside the main stream's kernels.  STRATA_GLM_PREFETCH_BLOCKS=<n>: n blocks instead - a
    // PCIe pull needs few SMs to keep the link busy, and every SM it holds is one the main stream's kernels wait for
    static const int blocks_env = [] {
        const char* v = getenv("STRATA_GLM_PREFETCH_BLOCKS");
        return v != nullptr ? std::max(1, std::atoi(v)) : 0;
    }();
    moe_prefetch_kernel<<<blocks_env > 0 ? blocks_env : std::max(1, sms / 2), 256, 0, s>>>(d, (blob_bytes + 15) / 16);
    launch_check("moe_prefetch");
}

__global__ void __launch_bounds__(256) zero_copy_kernel(const uint4* __restrict__ s, uint4* __restrict__ t, size_t n16) {
    for (size_t j = blockIdx.x * (size_t) blockDim.x + threadIdx.x; j < n16; j += (size_t) gridDim.x * blockDim.x)
        t[j] = s[j];
}

void zero_copy(const void* src, void* dst, size_t bytes, cudaStream_t s) {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) sms = 80;
    }
    zero_copy_kernel<<<sms, 256, 0, s>>>((const uint4*) src, (uint4*) dst, (bytes + 15) / 16);   // (moe_fetch's grid)
    launch_check("zero_copy");
}

void moe_fetch(const MoeDev& d, int k, size_t blob_bytes, cudaStream_t s, size_t gu_bytes, size_t dn_bytes, int n_chunks,
               size_t up_bytes) {
    if (up_bytes == 0) up_bytes = gu_bytes;
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) sms = 80;
    }
    if (n_chunks > 0 && gu_bytes > 0 && dn_bytes > 0)
        pdl_launch(moe_fetch_stream_kernel, sms, 256, 0, s, d, k, (blob_bytes + 15) / 16, gu_bytes, dn_bytes, n_chunks,
                   up_bytes);
    else
        pdl_launch(moe_fetch_kernel, sms, 256, 0, s, d, k, (blob_bytes + 15) / 16);
    launch_check("moe_fetch");
}

void moe_wait_fetch(const MoeDev& d, int n_embd, int k, size_t blob_bytes, cudaStream_t s, size_t gu_bytes,
                    size_t dn_bytes, int n_chunks, size_t up_bytes) {
    if (up_bytes == 0) up_bytes = gu_bytes;
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) sms = 80;
    }
    if (n_chunks > 0 && gu_bytes > 0 && dn_bytes > 0)
        pdl_launch(moe_wait_fetch_kernel<true>, sms, 256, 0, s, d, n_embd, k, (blob_bytes + 15) / 16, gu_bytes, dn_bytes,
                   n_chunks, up_bytes);
    else
        pdl_launch(moe_wait_fetch_kernel<false>, sms, 256, 0, s, d, n_embd, k, (blob_bytes + 15) / 16,
                   (unsigned long long) 0, (unsigned long long) 0, 0, (unsigned long long) 0);
    launch_check("moe_wait_fetch");
}

void moe_wait_fetch_cols(const MoeDev* mds_dev, int n_cols, int n_embd, int k, size_t blob_bytes, cudaStream_t s,
                         size_t gu_bytes, size_t dn_bytes, int n_chunks, size_t up_bytes) {
    if (up_bytes == 0) up_bytes = gu_bytes;
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) sms = 80;
    }
    if (n_cols <= 0) return;
    // (the SMs shared among the columns: a column's copies are PCIe-bound, ~a dozen blocks keep the link busy)
    const dim3 grid((unsigned) std::max(16, sms / n_cols), (unsigned) n_cols);
    if (n_chunks > 0 && gu_bytes > 0 && dn_bytes > 0)
        pdl_launch(moe_wait_fetch_cols_kernel<true>, grid, 256, 0, s, mds_dev, n_embd, k, (blob_bytes + 15) / 16,
                   gu_bytes, dn_bytes, n_chunks, up_bytes);
    else
        pdl_launch(moe_wait_fetch_cols_kernel<false>, grid, 256, 0, s, mds_dev, n_embd, k, (blob_bytes + 15) / 16,
                   (unsigned long long) 0, (unsigned long long) 0, 0, (unsigned long long) 0);
    launch_check("moe_wait_fetch_cols");
}

#if defined(STRATA_USE_HIP)
namespace {
// The RDNA expert kernels of the measured decode geometry, by the current device's architecture (hip_expert_kernel_bench
// --sweep; every sum bit-identical to the original kernels'): gfx1100 and gfx1201 all of them (an AI PRO R9700, 8
// experts: IQ2_XS / IQ2_XXS gate/up 90.6 / 85.3 -> 75.6 / 70.2 us, IQ3_XXS down 59.3 -> 48.8, IQ2_S down 56.7 ->
// 45.4); gfx1151 all but the IQ3_XXS down, which regresses there.  Other architectures and shapes keep the original
// kernels; STRATA_HIP_EXPERTS_LEGACY=1 the original ones everywhere.
enum class RdnaArch { none, gfx1100, gfx1151, gfx1201 };
RdnaArch rdna_expert_arch() {
    static const bool legacy = [] {
        const char* v = std::getenv("STRATA_HIP_EXPERTS_LEGACY");
        return v && v[0] == '1';
    }();
    if (legacy) return RdnaArch::none;
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) return RdnaArch::none;
    static thread_local int cached_device = -1;
    static thread_local RdnaArch arch = RdnaArch::none;
    if (cached_device != device) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, device) != hipSuccess) return RdnaArch::none;
        // the kernels are RDNA3 (gfx11, wave32 + sudot4), RDNA3.5 (gfx115x) and RDNA4 (gfx120x) ones: the sibling
        // chips of each family run the same code objects (the device guards in intrinsics.hpp / dsv4_fast.cu name
        // them all), so an RX 7800 XT / 7700 (gfx1101), 7600 (gfx1102), Radeon 890M (gfx1150) and RX 9060 (gfx1200)
        // take their family's kernels instead of the legacy ones (20-30% slower a call, bit-identical sums).
        // gfx1103 (the 780M) stays on the legacy kernels: its build is the portable one (cmake/hip_backend.cmake)
        const char* a = prop.gcnArchName;
        const auto is = [a](const char* name) { return std::strncmp(a, name, 7) == 0; };
        arch = is("gfx1100") || is("gfx1101") || is("gfx1102") ? RdnaArch::gfx1100
             : is("gfx1151") || is("gfx1150") ? RdnaArch::gfx1151
             : is("gfx1201") || is("gfx1200") ? RdnaArch::gfx1201 : RdnaArch::none;
        cached_device = device;
    }
    return arch;
}
bool rdna3_expert_device(bool allow_gfx1151 = true) {
    const RdnaArch a = rdna_expert_arch();
    return a == RdnaArch::gfx1100 || a == RdnaArch::gfx1201 || (allow_gfx1151 && a == RdnaArch::gfx1151);
}
// Splitting shared down removes its dynamic type switch from routed gate/up's
// register budget. The routed grid keeps exactly the original dot/reduction order.
__global__ void __launch_bounds__(256) rdna_shared_down_kernel(const uint8_t* weights, int type,
                                                         const block_q8_1* h, int n_in,
                                                         int n_out, float* out) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + wave;
    if (row < n_out) {
        const float v = any_row_dot(type, weights, row, h, nullptr, n_in, lane);
        if (lane == 0) out[row] = v;
    }
}
template<int T, int NW, bool GLOBAL = false, bool LUT = true, bool SPLIT = false,
         bool DIRECT = false, bool WIDE = false, int MINWAVES = 0>
void launch_expert_gu(const MoeDev& d, int k, int n_embd, int n_ff, float limit, const void* xq, void* hq,
               const void* sh_down, int sh_type, const void* sh_hq, int n_ff_sh, float* sh_out, cudaStream_t s,
               const MoeCols& mc = MoeCols{}) {
    const bool shared = sh_down && sh_out;
    const dim3 grid(n_ff / GU_ROWS, k + (shared && !SPLIT ? 1 : 0), (unsigned) mc.nc);
    pdl_launch(moe_gate_up_kernel<T, NW, LUT, GLOBAL, SPLIT, DIRECT, WIDE, MINWAVES>, grid, NW * 32, 0, s, d, shared ? k : k + 1, n_embd, n_ff,
        limit, (const block_q8_1*) xq, (block_q8_1*) hq, (const uint8_t*) sh_down, sh_type,
        (const block_q8_1*) sh_hq, n_ff_sh, sh_out, mc.ca, mc.xq_cs, mc.hq_cs, mc.shq_cs, mc.out_cs);
    if constexpr (SPLIT) {
        for (int c = 0; shared && c < mc.nc; ++c)
            rdna_shared_down_kernel<<<(n_embd + 7) / 8, 256, 0, s>>>(
                (const uint8_t*) sh_down, sh_type, (const block_q8_1*) ((const uint8_t*) sh_hq + (size_t) c * mc.shq_cs),
                n_ff_sh, n_embd, sh_out + (size_t) c * mc.out_cs);
    }
}
template<int T, int NR, bool GLOBAL = false, bool LUT = true, bool DIRECT = false, bool WIDE = false>
void launch_expert_down(const MoeDev& d, int k, int n_embd, int n_ff, size_t off, const void* hq,
               const float* sh, float* out, cudaStream_t s, const MoeCols& mc = MoeCols{}) {
    pdl_launch(moe_down_kernel<T, NR, LUT, GLOBAL, DIRECT, WIDE>, dim3((unsigned) grid_pad(n_embd / NR, 8), 1u, (unsigned) mc.nc),
                                                       256, 0, s, d, k, n_embd, n_ff, off, (const block_q8_1*) hq, sh,
                                                                    out, mc.ca, mc.hq_cs, mc.out_cs);
}
} // namespace
#endif

namespace {
// the gate/up type pairs of a mixed layer that have a kernel (moe_gate_up_kernel<G, U>): the 2-bit i-quants among
// themselves (unsloth's UD-IQ1_S Qwen3.8: IQ2_S / IQ2_XXS both ways), the 3-bit and the 1-bit pairs
#define GLMF_GU_PAIRS(X) X(16, 17) X(17, 16) X(16, 22) X(22, 16) X(17, 22) X(22, 17) X(18, 21) X(21, 18) X(19, 29) X(29, 19)

bool gate_up_mixed(int gu_type, int up_type, const MoeDev& d, int k, int n_embd, int n_ff, float limit, const void* xq,
                   void* hq, const void* sh_down, int sh_type, const void* sh_hq, int n_ff_sh, float* sh_out,
                   cudaStream_t s, const MoeCols& mc) {
    const bool sh = sh_down != nullptr && sh_out != nullptr;
    const dim3 grid((unsigned) (n_ff / GU_ROWS), (unsigned) (k + (sh ? 1 : 0)), (unsigned) mc.nc);
    const auto* X = (const block_q8_1*) xq;
    auto* H = (block_q8_1*) hq;
    const auto* SD = (const uint8_t*) sh_down;
    const auto* SH = (const block_q8_1*) sh_hq;
    const int kk = sh ? k : k + 1;
#if defined(STRATA_USE_HIP)
#define GLMF_GU2_K(G, U) moe_gate_up_kernel<G, GU_WARPS, false, false, false, false, false, 0, U>
#else
#define GLMF_GU2_K(G, U) moe_gate_up_kernel<G, U>
#endif
#define GLMF_GU2(G, U)                                                                                                  \
    if (gu_type == G && up_type == U) {                                                                                 \
        GLMF_GU2_K(G, U)<<<grid, GU_WARPS * 32, 0, s>>>(d, kk, n_embd, n_ff, limit, X, H, SD, sh_type, SH, n_ff_sh,    \
                                                        sh_out, mc.ca, mc.xq_cs, mc.hq_cs, mc.shq_cs, mc.out_cs);       \
        launch_check("moe_gate_up mixed");                                                                              \
        return true;                                                                                                    \
    }
    GLMF_GU_PAIRS(GLMF_GU2)
#undef GLMF_GU2
#undef GLMF_GU2_K
    std::fprintf(stderr, "glm_fast moe_gate_up: gate type %d with up type %d unsupported\n", gu_type, up_type);
    return false;
}

void gate_up_impl(int gu_type, const MoeDev& d, int k, int n_embd, int n_ff, float limit, const void* xq, void* hq,
                  const void* sh_down, int sh_type, const void* sh_hq, int n_ff_sh, float* sh_out, cudaStream_t s,
                  const MoeCols& mc, int up_type = -1) {
    if (up_type >= 0 && up_type != gu_type) {
        gate_up_mixed(gu_type, up_type, d, k, n_embd, n_ff, limit, xq, hq, sh_down, sh_type, sh_hq, n_ff_sh, sh_out, s,
                      mc);
        return;
    }
#if defined(STRATA_USE_HIP)
    // IQ2_XS (DeepSeek V4's gate/up) the same way, measured on gfx1151 (the Radeon 8060S lane) and gfx1201 (an AI PRO
    // R9700: 90.6 -> 75.6 us for 8 experts), the RX 7900 XT (gfx1100: 78.4 -> 60.1 us), bit-identical sums
    static const bool gu17_legacy = std::getenv("ABYSS_IGPU_GU17_LEGACY") != nullptr;
    const RdnaArch ra = rdna_expert_arch();
    if (gu_type == 17 && n_embd == 4096 && n_ff == 2048 && (ra == RdnaArch::gfx1100 || ra == RdnaArch::gfx1151 || ra == RdnaArch::gfx1201) &&
        !gu17_legacy) {
        launch_expert_gu<17, 16, false, true, false, true>(d, k, n_embd, n_ff, limit,
            xq, hq, sh_down, sh_type, sh_hq, n_ff_sh, sh_out, s, mc);
        launch_check("moe_gate_up RDNA3.5");
        return;
    }
    // DeepSeek V4.1's 5120 / 2304 (unsloth's UD quants): IQ2_XS / IQ2_S gate/up the same LDS-grid kernel - 8 experts
    // (hip_expert_kernel_bench --sweep --shape 5120x2304), an AI PRO R9700 (gfx1201) 125 / 131 -> 101 / 106 us, the
    // Radeon 8060S (gfx1151, the iGPU lane) 261 / 301 -> 249 / 273, the RX 7900 XT (gfx1100) 103 / 116 -> 80 / 87 us; bit-identical.
    if ((gu_type == 17 || gu_type == 22) && n_embd == 5120 && n_ff == 2304 &&
        (ra == RdnaArch::gfx1100 || ra == RdnaArch::gfx1201 || ra == RdnaArch::gfx1151)) {
        if (gu_type == 17)
            launch_expert_gu<17, 16, false, true, false, true>(d, k, n_embd, n_ff, limit,
                xq, hq, sh_down, sh_type, sh_hq, n_ff_sh, sh_out, s, mc);
        else
            launch_expert_gu<22, 16, false, true, false, true>(d, k, n_embd, n_ff, limit,
                xq, hq, sh_down, sh_type, sh_hq, n_ff_sh, sh_out, s, mc);
        launch_check("moe_gate_up RDNA V4.1");
        return;
    }
    // (any shape of whole super-blocks: DeepSeek V4.1's 5120 / 2304 too - the kernel takes the shape at run time;
    // ABYSS_IGPU_GU16_4096=1: GLM-5.3-Flash's 4096 / 2048 only, as before)
    static const bool gu16_any = std::getenv("ABYSS_IGPU_GU16_4096") == nullptr;
    if (gu_type == 16 && ((n_embd == 4096 && n_ff == 2048) || (gu16_any && n_embd % 256 == 0 && n_ff % GU_ROWS == 0)) &&
        rdna3_expert_device()) {
        // Packed sign expansion, original 16-wave/four-row geometry, grid in LDS.
        launch_expert_gu<16, 16, false, true, false, true>(d, k, n_embd, n_ff, limit,
            xq, hq, sh_down, sh_type, sh_hq, n_ff_sh, sh_out, s, mc);
        launch_check("moe_gate_up RDNA3");
        return;
    }
#endif
    const bool sh = sh_down != nullptr && sh_out != nullptr;
    const dim3 grid((unsigned) (n_ff / GU_ROWS), (unsigned) (k + (sh ? 1 : 0)), (unsigned) mc.nc);
    const auto* X = (const block_q8_1*) xq;
    auto* H = (block_q8_1*) hq;
    const auto* SD = (const uint8_t*) sh_down;
    const auto* SH = (const block_q8_1*) sh_hq;
    const int kk = sh ? k : k + 1;   // no shared expert: no block has blockIdx.y == kk
    switch (gu_type) {
#define GLMF_GU(T) \
        case T: pdl_launch(moe_gate_up_kernel<T>, grid, GU_WARPS * 32, 0, s, d, kk, n_embd, n_ff, limit, X, H, SD, sh_type, SH, \
                                                                     n_ff_sh, sh_out, mc.ca, mc.xq_cs, mc.hq_cs,      \
                                                                     mc.shq_cs, mc.out_cs); break;
        GLMF_GU(16) GLMF_GU(18) GLMF_GU(19) GLMF_GU(23) GLMF_GU(10) GLMF_GU(11) GLMF_GU(17) GLMF_GU(22) GLMF_GU(21) GLMF_GU(29)
        GLMF_GU(12) GLMF_GU(13) GLMF_GU(14) GLMF_GU(39) GLMF_GU(20) GLMF_GU(7) GLMF_GU(6) GLMF_GU(8) GLMF_GU(42) GLMF_GU(40)
#undef GLMF_GU
        default: std::fprintf(stderr, "glm_fast moe_gate_up: type %d unsupported\n", gu_type); return;
    }
    launch_check("moe_gate_up");
}
}  // namespace

void moe_gate_up(int gu_type, const MoeDev& d, int k, int n_embd, int n_ff, float limit, const void* xq, void* hq,
                 const void* sh_down, int sh_type, const void* sh_hq, int n_ff_sh, float* sh_out, cudaStream_t s,
                 int up_type) {
    gate_up_impl(gu_type, d, k, n_embd, n_ff, limit, xq, hq, sh_down, sh_type, sh_hq, n_ff_sh, sh_out, s, MoeCols{},
                 up_type);
}

bool moe_gate_up_pair_supported(int gu_type, int up_type) {
    if (up_type < 0 || up_type == gu_type) return moe_supported(gu_type);
#define GLMF_GU2_OK(G, U) if (gu_type == G && up_type == U) return true;
    GLMF_GU_PAIRS(GLMF_GU2_OK)
#undef GLMF_GU2_OK
    return false;
}

void moe_gate_up_cols(int gu_type, const MoeColArgs* cols, int nc, int k, int n_embd, int n_ff, float limit,
                      const void* xq, size_t xq_cs, void* hq, size_t hq_cs, const void* sh_down, int sh_type,
                      const void* sh_hq, size_t sh_hq_cs, int n_ff_sh, float* sh_out, size_t out_cs, cudaStream_t s,
                      const float* nv_scale, int up_type) {
    MoeCols mc;
    mc.ca = cols;
    mc.nc = nc;
    mc.xq_cs = xq_cs;
    mc.hq_cs = hq_cs;
    mc.shq_cs = sh_hq_cs;
    mc.out_cs = out_cs;
    MoeDev d{};
    d.nv_scale = nv_scale;   // (NVFP4: the layer's scales, read through each column's plan_id)
    gate_up_impl(gu_type, d, k, n_embd, n_ff, limit, xq, hq, sh_down, sh_type, sh_hq, n_ff_sh, sh_out, s, mc, up_type);
}

#if defined(STRATA_USE_HIP)
namespace hip_expert_bench {

void gate_up(Variant v, int type, const MoeDev& d, int k, int n_embd, int n_ff, float limit,
             const void* xq, void* hq, const void* sh_down, int sh_type, const void* sh_hq,
             int n_ff_sh, float* sh_out, cudaStream_t s) {
    if (v == Variant::production) return moe_gate_up(type, d, k, n_embd, n_ff, limit, xq, hq,
                                                   sh_down, sh_type, sh_hq, n_ff_sh, sh_out, s);
#define BENCH_GU(T) case T: \
    if (v == Variant::baseline) launch_expert_gu<T,16,false,false>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::direct_lds) launch_expert_gu<T,16,false,true,false,true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::direct_signs) launch_expert_gu<T,16,true,true,false,true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::vector_load) launch_expert_gu<T,16,true,true,false,false,true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::occupancy8) launch_expert_gu<T,16,true,true,true,false,false,8>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::split_shared) launch_expert_gu<T, 16, true, true, true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::global_plain) launch_expert_gu<T, 16, true, false>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::global_waves8) launch_expert_gu<T, 8, true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::waves8) launch_expert_gu<T, 8>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else if (v == Variant::global_grid) launch_expert_gu<T, 16, true>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); \
    else launch_expert_gu<T, 16>(d,k,n_embd,n_ff,limit,xq,hq,sh_down,sh_type,sh_hq,n_ff_sh,sh_out,s); break;
    switch (type) { BENCH_GU(16) BENCH_GU(17) BENCH_GU(18) BENCH_GU(22)
        default: std::fprintf(stderr, "expert bench: unsupported type %d\n", type); std::abort(); }
#undef BENCH_GU
    launch_check("expert bench gate/up");
}

void down(Variant v, int type, const MoeDev& d, int k, int n_embd, int n_ff, size_t off,
          const void* hq, const float* sh, float* out, cudaStream_t s) {
    if (v == Variant::production) return moe_down(type,d,k,n_embd,n_ff,off,hq,sh,out,s);
#define BENCH_DN(T) case T: \
    if (v == Variant::baseline) launch_expert_down<T,4,false,false>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::direct_lds) launch_expert_down<T,2,false,true,true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::direct_rows4) launch_expert_down<T,4,true,true,true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::direct_lds_rows4) launch_expert_down<T,4,false,true,true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::direct_signs) launch_expert_down<T,2,true,true,true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::vector_load) launch_expert_down<T,2,true,true,false,true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::global_plain) launch_expert_down<T, 4, true, false>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::global_rows2) launch_expert_down<T, 2, true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::global_rows8) launch_expert_down<T, 8, true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::rows2) launch_expert_down<T, 2>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::rows8) launch_expert_down<T, 8>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else if (v == Variant::global_grid) launch_expert_down<T, 4, true>(d,k,n_embd,n_ff,off,hq,sh,out,s); \
    else launch_expert_down<T, 4>(d,k,n_embd,n_ff,off,hq,sh,out,s); break;
    switch (type) { BENCH_DN(16) BENCH_DN(17) BENCH_DN(18) BENCH_DN(22)
        default: std::fprintf(stderr, "expert bench: unsupported type %d\n", type); std::abort(); }
#undef BENCH_DN
    launch_check("expert bench down");
}
} // namespace hip_expert_bench
#endif

bool rows_experts(int gu_type, int d_type, const uint8_t* base, size_t stride, size_t down_off, const int* light, int n,
                  const int* row_tok, const float* x, int n_tok, int n_embd, int n_ff, float limit, int rlo, int rhi,
                  void* xq_scratch, void* hq_scratch, float* out, int ld, cudaStream_t s,
                  const unsigned long long* ptrs) {
    const auto ok = [](int t) { return t == 16 || t == 17 || t == 18 || t == 19; };
    if (!ok(gu_type) || !ok(d_type) || n_embd % 256 || n_ff % (GU_ROWS) || n_embd % (GU_WARPS * 2)) return false;
    if (n <= 0 || rhi <= rlo) return true;
    auto* xq = (block_q8_1*) xq_scratch;
    auto* hq = (block_q8_1*) hq_scratch;
    rows_q8_kernel<<<(unsigned) (((int64_t) n_tok * (n_embd / 32) * 32 + 255) / 256), 256, 0, s>>>(x, n_embd, n_tok,
                                                                                                 n_embd, xq);
    const dim3 ggu((unsigned) (n_ff / GU_ROWS), (unsigned) n), gdn((unsigned) (n_embd / (GU_WARPS * 2)), (unsigned) n);
    switch (gu_type) {
        case 16: rows_gate_up_kernel<16><<<ggu, GU_WARPS * 32, 0, s>>>(base, stride, ptrs, light, row_tok, xq, n_embd, n_ff, out, ld); break;
        case 17: rows_gate_up_kernel<17><<<ggu, GU_WARPS * 32, 0, s>>>(base, stride, ptrs, light, row_tok, xq, n_embd, n_ff, out, ld); break;
        case 18: rows_gate_up_kernel<18><<<ggu, GU_WARPS * 32, 0, s>>>(base, stride, ptrs, light, row_tok, xq, n_embd, n_ff, out, ld); break;
        default: rows_gate_up_kernel<19><<<ggu, GU_WARPS * 32, 0, s>>>(base, stride, ptrs, light, row_tok, xq, n_embd, n_ff, out, ld); break;
    }
    rows_swiglu_q8_kernel<<<(unsigned) (((int64_t) (rhi - rlo) * (n_ff / 32) * 32 + 255) / 256), 256, 0, s>>>(
        out, ld, rlo, rhi, n_ff, limit, hq);
    switch (d_type) {
        case 16: rows_down_kernel<16><<<gdn, GU_WARPS * 32, 0, s>>>(base, stride, ptrs, down_off, light, hq, rlo, n_ff, n_embd, out, ld); break;
        case 17: rows_down_kernel<17><<<gdn, GU_WARPS * 32, 0, s>>>(base, stride, ptrs, down_off, light, hq, rlo, n_ff, n_embd, out, ld); break;
        case 19: rows_down_kernel<19><<<gdn, GU_WARPS * 32, 0, s>>>(base, stride, ptrs, down_off, light, hq, rlo, n_ff, n_embd, out, ld); break;
        default: rows_down_kernel<18><<<gdn, GU_WARPS * 32, 0, s>>>(base, stride, ptrs, down_off, light, hq, rlo, n_ff, n_embd, out, ld); break;
    }
    launch_check("rows_experts");
    return true;
}

namespace {
void down_impl(int d_type, const MoeDev& d, int k, int n_embd, int n_ff, size_t down_off, const void* hq,
               const float* sh_out, float* out, cudaStream_t s, const MoeCols& mc) {
#if defined(STRATA_USE_HIP)
    // DeepSeek V4.1's 5120 / 2304: four rows a warp with the grid in LDS - 8 experts (hip_expert_kernel_bench --sweep
    // --shape 5120x2304), IQ2_XXS / IQ2_S / IQ3_XXS / IQ2_XS down: an AI PRO R9700 (gfx1201) 66.6 / 81.0 / 76.7 / 71.1
    // -> 45.9 / 52.1 / 55.9 / 50.3 us, the Radeon 8060S (gfx1151, the iGPU lane) 151.7 / 189.8 / 183.9 / 164.8 ->
    // 117.7 / 155.6 / 177.3 / 137.1, the RX 7900 XT (gfx1100) 64.4 / 77.2 / 72.0 / 70.0 -> 46.6 / 54.6 / 58.6 / 51.8 us; bit-identical.
    const RdnaArch ra = rdna_expert_arch();
    if ((d_type == 16 || d_type == 22 || d_type == 18 || d_type == 17) && n_embd == 5120 && n_ff == 2304 &&
        (ra == RdnaArch::gfx1100 || ra == RdnaArch::gfx1201 || ra == RdnaArch::gfx1151)) {
        switch (d_type) {
            case 16: launch_expert_down<16, 4, false, true, true>(d, k, n_embd, n_ff, down_off, hq, sh_out, out, s, mc); break;
            case 17: launch_expert_down<17, 4, false, true, true>(d, k, n_embd, n_ff, down_off, hq, sh_out, out, s, mc); break;
            case 18: launch_expert_down<18, 4, false, true, true>(d, k, n_embd, n_ff, down_off, hq, sh_out, out, s, mc); break;
            default: launch_expert_down<22, 4, false, true, true>(d, k, n_embd, n_ff, down_off, hq, sh_out, out, s, mc); break;
        }
        launch_check("moe_down RDNA V4.1");
        return;
    }
    // IQ3_XXS down regresses on gfx1151; let it use the original dispatch below.
    if ((d_type == 22 || d_type == 18) && n_embd == 4096 && n_ff == 2048 && rdna3_expert_device(d_type == 22)) {
        // Two rows lower the register budget; IQ2_S prefers global grid reads,
        // while the smaller IQ3_XXS grid is faster in LDS on gfx1100.
        if (d_type == 22) launch_expert_down<22, 2, true, true, true>(d, k, n_embd, n_ff, down_off, hq, sh_out, out, s, mc);
        else launch_expert_down<18, 2, false, true, true>(d, k, n_embd, n_ff, down_off, hq, sh_out, out, s, mc);
        launch_check("moe_down RDNA3");
        return;
    }
#endif
    const int blocks = n_embd / DOWN_ROWS;   // n_embd % DOWN_ROWS == 0 (4096)
    const auto* H = (const block_q8_1*) hq;
    const dim3 grid((unsigned) grid_pad(blocks, 8), 1u, (unsigned) mc.nc);
    switch (d_type) {
#define GLMF_DN(T) \
        case T: pdl_launch(moe_down_kernel<T>, grid, 256, 0, s, d, k, n_embd, n_ff, down_off, H, sh_out, out, mc.ca, mc.hq_cs, \
                                                         mc.out_cs); break;
        GLMF_DN(16) GLMF_DN(18) GLMF_DN(19) GLMF_DN(23) GLMF_DN(10) GLMF_DN(11) GLMF_DN(17) GLMF_DN(22) GLMF_DN(21) GLMF_DN(29)
        GLMF_DN(12) GLMF_DN(13) GLMF_DN(14) GLMF_DN(39) GLMF_DN(20) GLMF_DN(7) GLMF_DN(6) GLMF_DN(8) GLMF_DN(42) GLMF_DN(40)
#undef GLMF_DN
        default: std::fprintf(stderr, "glm_fast moe_down: type %d unsupported\n", d_type); return;
    }
    launch_check("moe_down");
}
}  // namespace

void moe_down(int d_type, const MoeDev& d, int k, int n_embd, int n_ff, size_t down_off, const void* hq,
              const float* sh_out, float* out, cudaStream_t s) {
    down_impl(d_type, d, k, n_embd, n_ff, down_off, hq, sh_out, out, s, MoeCols{});
}

void moe_down_cols(int d_type, const MoeColArgs* cols, int nc, int k, int n_embd, int n_ff, size_t down_off,
                   const void* hq, size_t hq_cs, const float* sh_out, float* out, size_t out_cs, cudaStream_t s,
                   const float* nv_scale) {
    MoeCols mc;
    mc.ca = cols;
    mc.nc = nc;
    mc.hq_cs = hq_cs;
    mc.out_cs = out_cs;
    MoeDev d{};
    d.nv_scale = nv_scale;
    down_impl(d_type, d, k, n_embd, n_ff, down_off, hq, sh_out, out, s, mc);
}

namespace {
__global__ void tab_update_kernel(unsigned long long* tab, const int* keys, const unsigned long long* vals, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) tab[keys[i]] = vals[i];
}
}  // namespace

void tab_update(unsigned long long* tab, const int* keys, const unsigned long long* vals, int n, cudaStream_t s) {
    if (n <= 0) return;
    tab_update_kernel<<<(n + 255) / 256, 256, 0, s>>>(tab, keys, vals, n);
    launch_check("tab_update");
}

void mtp_in(const float* emb, const float* h, const float* enorm, const float* hnorm, float eps, int n, void* cat_q8_1,
            cudaStream_t s) {
    mtp_in_kernel<<<1, 1024, 0, s>>>(emb, h, enorm, hnorm, eps, n, (block_q8_1*) cat_q8_1);
    launch_check("mtp_in");
}

void rms_q8(float* h, const float* add, const float* w, float eps, int n, float* x, void* xq, cudaStream_t s) {
    rms_q8_kernel<<<1, 1024, 0, s>>>(h, add, w, eps, n, x, (block_q8_1*) xq);
    launch_check("rms_q8");
}

void argmax(const float* x, int n, int* out, cudaStream_t s) {
    argmax_kernel<<<1, 1024, 0, s>>>(x, n, out);
    launch_check("argmax");
}

}  // namespace strata::kernels::glmf
