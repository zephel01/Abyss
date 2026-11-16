// src/prefill/moe_own.cu - see include/strata/prefill/moe_own.hpp.
//
// The grouped GEMM.  A CTA is NW warps and takes one tile: one expert, TC of its activation rows (tokens) and 32 * NW
// rows of its weights - two 16-row fragments a warp: gate and up rows r..r+15 (the SwiGLU needs both in one thread) or
// down rows r..r+31.  It walks K in steps of 32:
//   - the weights' A fragments are decoded in registers from the raw quantized bytes (moe_own_decode.cuh), once a step
//     for all of the tile's tokens; the raw bytes of the next weight chunk come to shared memory by cp.async while the
//     current one is multiplied (whole 16-byte words around each row's chunk: the rows are not 16-byte aligned);
//   - the activations (int8, TC x 128 values a half chunk) and their block scales come by cp.async too, two buffers;
//   - mma.sync m16n8k32 s8 (two m16n8k16 where a scale covers 16 values) a group of 8 tokens, the int32 sums scaled
//     and added in FP32 in registers.
// One barrier a 128 values.  The tiles of a set are listed exactly (a prefix table of the experts' token tiles - in
// the launch's arguments for a small set, made by a small kernel for a large one): no empty CTAs, no fix-up pass.
// (CTAs that take tiles from a counter until none is left measured the same as one CTA a tile.)
//
// The tensor-core part (mma.sync, cp.async) is NVIDIA's; the decoders and the tile plan are not - a WMMA backend
// (gfx12: 16x16x16 int8 tiles) replaces `tile_body` only.
#include "strata/prefill/moe_own.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#if !defined(__HIPCC__)
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include "moe_own_decode.cuh"
#endif

namespace strata::prefill::own {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill own: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// STRATA_PROMPT_OWN_STATS=1: the launches' shapes, printed when the context ends
struct Stats { long launches = 0, coarse = 0, inline_ = 0, experts = 0, rows = 0, tiles = 0; };
Stats& stats() { static Stats t; return t; }
bool stats_on() { static const bool on = getenv("STRATA_PROMPT_OWN_STATS") != nullptr; return on; }

// ---------------------------------------------------------------- the activations
// A warp takes 128 values of a row: a thread 4 of them, 8 threads a 32-value block.  d = max|x| / 127 over the block,
// q = round(x * (127 / max|x|)).  The int8 plane holds the 128 values' 4-value words in the GEMM's order (below).  
__global__ void quant_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids, uint32_t* __restrict__ q,
                             float* __restrict__ d, int cols, int64_t ld, int64_t rows, bool sums) {
    const int64_t wi = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // the warp: (row, 128 values)
    const int lane = threadIdx.x & 31, per_row = cols / 128;
    const int64_t r = wi / per_row;
    if (r >= rows) return;
    const int c = (int) (wi - r * per_row);
    const float* src = x + (ids != nullptr ? (int64_t) ids[r] : r) * ld + (int64_t) c * 128 + lane * 4;
    float v0, v1, v2, v3;
    if (((uintptr_t) src & 15) == 0) {   // (the engine's rows: one 16-byte load)
        const float4 v = *(const float4*) src;
        v0 = v.x; v1 = v.y; v2 = v.z; v3 = v.w;
    } else {
        v0 = src[0]; v1 = src[1]; v2 = src[2]; v3 = src[3];
    }
    float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 4));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const uint32_t w = (uint32_t) (__float2int_rn(v0 * inv) & 255) | (uint32_t) (__float2int_rn(v1 * inv) & 255) << 8 |
                       (uint32_t) (__float2int_rn(v2 * inv) & 255) << 16 | (uint32_t) (__float2int_rn(v3 * inv) & 255) << 24;
    // (the GEMM's order: block i's word w - values 4 w .. 4 w + 3 - at t * 8 + i * 2 + half, w = t + 4 half: a thread
    // t of the GEMM reads its two words of all four blocks in two 16-byte loads)
    const int wd = lane & 7, bi = lane >> 3;
    q[(r * cols + (int64_t) c * 128) / 4 + (wd & 3) * 8 + bi * 2 + (wd >> 2)] = w;
    // the block's scale, and - a second plane - its sum of activations as rounded (the K-quants' offsets)
    int si = (int) (int8_t) (w & 255) + (int) (int8_t) ((w >> 8) & 255) + (int) (int8_t) ((w >> 16) & 255) + (int) (int8_t) (w >> 24);
    si += __shfl_xor_sync(0xFFFFFFFFu, si, 1);
    si += __shfl_xor_sync(0xFFFFFFFFu, si, 2);
    si += __shfl_xor_sync(0xFFFFFFFFu, si, 4);
    if ((lane & 7) == 0) {
        const int64_t nb = cols / 32;
        d[r * nb + c * 4 + (lane >> 3)] = amax / 127.0f;
        if (sums) d[rows * nb + r * nb + c * 4 + (lane >> 3)] = (amax / 127.0f) * (float) si;
    }
}

#if !defined(__HIPCC__)
// ---------------------------------------------------------------- the tile plan
// Tile shapes.  Fine (0): 4 warps (64 gate/up or 128 down rows), 64 tokens - two CTAs an SM, more and smaller tiles:
// a small set still fills the GPU.  Coarse (1): 8 warps, 128 tokens - the weights are decoded once for twice the
// tokens (a set of many rows an expert: ~10% less time).  Wide (2): 8 warps, 64 tokens - the types whose raw chunk is
// large (Q6_K, Q8_0: ~10 16-byte words a row and chunk): two fine CTAs or a coarse one do not fit an SM's shared
// memory, a wide one does - their only shape.
template<int CF> struct Cfg {
    static constexpr int NW = CF ? 8 : 4;               // warps a CTA
    static constexpr int TC = CF == 1 ? 128 : 64;       // tokens a tile
    static constexpr int NT = NW * 32;
};

// tiles[e] = token tiles (of tc tokens) of experts < e (n + 1 entries); n <= 1024.  (A set of up to kInline experts
// carries the table in the launch's arguments instead: no extra launch for the many small sets of a short prompt.)
constexpr int kInline = 64;
__global__ void tiles_kernel(const int32_t* __restrict__ bounds, int32_t* __restrict__ tiles, int n, int tc) {
    __shared__ int a[1025], b[1025];
    for (int i = threadIdx.x; i <= n; i += blockDim.x) a[i] = i == 0 ? 0 : (bounds[i] - bounds[i - 1] + tc - 1) / tc;
    __syncthreads();
    int* src = a;
    int* dst = b;
    for (int step = 1; step <= n; step *= 2) {
        for (int i = threadIdx.x; i <= n; i += blockDim.x) dst[i] = src[i] + (i >= step ? src[i - step] : 0);
        __syncthreads();
        int* t = src;
        src = dst;
        dst = t;
    }
    for (int i = threadIdx.x; i <= n; i += blockDim.x) tiles[i] = src[i];
}

struct KArgs {
    const uint8_t* w;       // expert 0's blob + the first matrix's offset
    size_t stride;          // bytes between experts
    size_t up_off;          // gate/up: the up matrix after the gate matrix
    const uint8_t* xq;      // activations: int8 [rows][K]
    const float* xd;        // their scales [rows][K / 32]
    const float* xsum;      // their block sums (scale x the sum of the int8), the same shape
    const int32_t* bounds;
    const int32_t* tiles;   // the tile table on the device, or null: itiles
    int32_t itiles[kInline + 1];
    float* dst;
    int64_t ld;
    int n, row0, w_rows, K;
    int tc;                 // tokens a tile in this launch (a multiple of 8, at most the shape's TC)
    float limit;
};

// (mma.sync s8 and cp.async need sm_80: device code for an older target traps - built() is false on such a device)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define STRATA_OWN_SM80 1
#else
#define STRATA_OWN_SM80 0
#endif
__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
#if STRATA_OWN_SM80
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
#else
    __trap();
#endif
}
__device__ __forceinline__ void cp_commit() {
#if STRATA_OWN_SM80
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void cp_wait_all() {
#if STRATA_OWN_SM80
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

// C[16 x 8] = A[16 x 32] B[32 x 8], int8 -> int32
__device__ __forceinline__ void mma32(int c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if STRATA_OWN_SM80
    const int z = 0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(z));
#else
    c[0] = c[1] = c[2] = c[3] = (int) (a0 + a1 + a2 + a3 + b0 + b1);
    __trap();
#endif
}
// C[16 x 8] = A[16 x 16] B[16 x 8]
__device__ __forceinline__ void mma16(int c[4], uint32_t a0, uint32_t a1, uint32_t b0) {
#if STRATA_OWN_SM80
    const int z = 0;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(z));
#else
    c[0] = c[1] = c[2] = c[3] = (int) (a0 + a1 + b0);
    __trap();
#endif
}

constexpr size_t kSmemSM = 102400, kSmemCTA = 101376;   // (an SM's shared memory and what one CTA may take: 100 KB cards)
// 16-byte words a slot holds (a chunk from any offset its alignment allows), and a CTA's shared memory
template<int TY, int CF> struct Lay {
    static constexpr int WH = Rng<TY>::HDR > 0 ? (16 - Dec<TY>::AL + Rng<TY>::HDR + 15) / 16 : 0;   // the header's words
    static constexpr int WD = (16 - Dec<TY>::AL + Dec<TY>::CB + 15) / 16;                              // the data's
    static constexpr int W2 = Rng2<TY>::CB2 > 0 ? (16 - Dec<TY>::AL + Rng2<TY>::CB2 + 15) / 16 : 0;      // the second data range's
    static constexpr int WS = WH + WD + W2;
    static constexpr int BUF = 2 * Cfg<CF>::NT * WS + 2 * Cfg<CF>::TC * 8 + (Sums<TY>::on ? 4 : 2) * Cfg<CF>::TC;   // 16-byte words
    static constexpr size_t smem = (size_t) BUF * 16 + (size_t) Dec<TY>::GW * 4;   // (+ the codebook)
};

// One tile of ng groups of 8 tokens (ng <= TC / 8: the loops over the groups are unrolled to that and each group is
// skipped past ng - the sums stay in registers, a tile costs its own groups).  e: the expert; rt: the row tile; b0, b1:
// the tile's first activation row and the expert's end.
template<int TY, bool GLU, int CF>
__device__ __forceinline__ void tile_body(const KArgs& a, uint4* smem, int e, int rt, int b0, int b1, int ng) {
    typedef Dec<TY> D;
    constexpr int NW = Cfg<CF>::NW, TC = Cfg<CF>::TC, NT = Cfg<CF>::NT, NG = TC / 8;
    constexpr int WS = Lay<TY, CF>::WS, WH = Lay<TY, CF>::WH, WD = Lay<TY, CF>::WD, W2 = Lay<TY, CF>::W2, HPW = D::WCH / 128, SPH = 4;   // (half chunks a weight chunk; steps a half chunk)
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    uint4* wbuf = smem;                       // [2][NT slots][WS]
    uint4* xbuf = wbuf + 2 * NT * WS;         // [2][TC][8]
    uint4* sbuf = xbuf + 2 * TC * 8;          // [2][TC]
    uint4* ubuf = sbuf + 2 * TC;              // [2][TC]: the block sums (the types that need them)
#if defined(OWN_GRID_GLOBAL)
    const uint32_t* grid = D::table();
#else
    const uint32_t* grid = (const uint32_t*) (smem + Lay<TY, CF>::BUF);   // (own_kernel copied it)
#endif
    const uint8_t* we = a.w + (size_t) e * a.stride;
    const size_t row_bytes = (size_t) (a.K / D::WCH) * Rng<TY>::RB;
    const int n_h = a.K / 128, n_wc = a.K / D::WCH;

    // slot s (0 .. NT-1) = fragment f (s / (NT / 2)), warp, row in the fragment -> that row's bytes
    const auto slot_row = [&](int s) -> const uint8_t* {
        const int f = s / (NT / 2), ww = (s / 16) % NW, rr = s % 16;
        const int row = GLU ? rt * (NW * 16) + ww * 16 + rr : rt * (NW * 32) + (2 * ww + f) * 16 + rr;
        return we + (GLU && f ? a.up_off : (size_t) 0) + (size_t) min(row, a.w_rows - 1) * row_bytes;
    };
    const uint8_t* my_slot = slot_row(threadIdx.x);
    const auto issue_w = [&](int c) {   // this thread's slot: the 16-byte words around its row's chunk c
        uint4* dst = wbuf + ((c & 1) * NT + threadIdx.x) * WS;
        if constexpr (WH > 0) {   // (the chunk's header range first)
            const uintptr_t src = (uintptr_t) (my_slot + Rng<TY>::hdr(c));
            const uint8_t* p = (const uint8_t*) (src & ~(uintptr_t) 15);
            const int last = (int) (((src & 15) + Rng<TY>::HDR - 1) / 16);
#pragma unroll
            for (int k = 0; k < WH; ++k) cp16(dst + k, p + 16 * min(k, last));
        }
        const uintptr_t src = (uintptr_t) (my_slot + Rng<TY>::dat(c));
        const uint8_t* p = (const uint8_t*) (src & ~(uintptr_t) 15);
        const int last = (int) (((src & 15) + D::CB - 1) / 16);   // (no word past the chunk: it may be past the pool)
#pragma unroll
        for (int k = 0; k < WD; ++k) cp16(dst + WH + k, p + 16 * min(k, last));
        if constexpr (W2 > 0) {
            const uintptr_t src2 = (uintptr_t) (my_slot + Rng2<TY>::dat2(c));
            const uint8_t* p2 = (const uint8_t*) (src2 & ~(uintptr_t) 15);
            const int last2 = (int) (((src2 & 15) + Rng2<TY>::CB2 - 1) / 16);
#pragma unroll
            for (int k = 0; k < W2; ++k) cp16(dst + WH + WD + k, p2 + 16 * min(k, last2));
        }
    };
    const auto issue_x = [&](int h) {
        uint4* xd = xbuf + (h & 1) * TC * 8;
#pragma unroll
        for (int k = 0; k < TC * 8 / NT; ++k) {
            const int i = threadIdx.x + k * NT, tok = i >> 3, word = i & 7;
            if (tok < ng * 8) {
                const int64_t r = min(b0 + tok, b1 - 1) - a.row0;
                cp16(xd + i, a.xq + r * a.K + (int64_t) h * 128 + word * 16);
            }
        }
        if (threadIdx.x < ng * 8) {
            const int64_t r = min(b0 + (int) threadIdx.x, b1 - 1) - a.row0;
            cp16(sbuf + (h & 1) * TC + threadIdx.x, a.xd + r * (a.K / 32) + (int64_t) h * 4);
            if constexpr (Sums<TY>::on) cp16(ubuf + (h & 1) * TC + threadIdx.x, a.xsum + r * (a.K / 32) + (int64_t) h * 4);
        }
    };

    // the four rows this thread decodes: fragment f, rows g and g + 8 of its warp's 16
    const uint8_t* rowp[2][2];
    int soff[2][2];
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
        for (int h8 = 0; h8 < 2; ++h8) {
            soff[f][h8] = f * (NT / 2) + wp * 16 + g + 8 * h8;
            rowp[f][h8] = slot_row(soff[f][h8]);
        }

    float acc[2][NG][4];
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
        for (int j = 0; j < NG; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[f][j][k] = 0.0f;

    issue_x(0);
    issue_w(0);
    cp_commit();
    const uint8_t* sp[2][2];
    const uint8_t* hp[2][2] = {};   // (the chunk's header, the types that have one)
    const uint8_t* sp2[2][2] = {};  // (its second data range)
    float hd[2][2];
    for (int h = 0; h < n_h; ++h) {
        cp_wait_all();
        __syncthreads();   // half chunk h (and its weights) landed for every thread; h - 1 is done by all
        const int c = h / HPW;
        if (h + 1 < n_h) issue_x(h + 1);
        if (h % HPW == 0 && c + 1 < n_wc) issue_w(c + 1);
        cp_commit();
        if (h % HPW == 0) {
#pragma unroll
            for (int f = 0; f < 2; ++f)
#pragma unroll
                for (int h8 = 0; h8 < 2; ++h8) {
                    const uint8_t* slot = (const uint8_t*) (wbuf + ((c & 1) * NT + soff[f][h8]) * WS);
                    sp[f][h8] = slot + WH * 16 + ((uintptr_t) (rowp[f][h8] + Rng<TY>::dat(c)) & 15);
                    if constexpr (W2 > 0) sp2[f][h8] = slot + (WH + WD) * 16 + ((uintptr_t) (rowp[f][h8] + Rng2<TY>::dat2(c)) & 15);
                    if constexpr (WH > 0) {
                        hp[f][h8] = slot + ((uintptr_t) (rowp[f][h8] + Rng<TY>::hdr(c)) & 15);
                        hd[f][h8] = D::head(hp[f][h8]);
                    } else {
                        hd[f][h8] = D::head(sp[f][h8]);
                    }
                }
        }
        const uint4* xw = xbuf + (h & 1) * TC * 8;
        const uint4* xs = sbuf + (h & 1) * TC;
        // the half chunk's four steps decoded, then a token group at a time: its activation words and scales for the
        // four steps come in four 16-byte loads (the kernel's time goes with its loads)
        Step st[SPH][2][2];
#pragma unroll
        for (int i = 0; i < SPH; ++i)
#pragma unroll
            for (int f = 0; f < 2; ++f)
#pragma unroll
                for (int h8 = 0; h8 < 2; ++h8) {
                    if constexpr (W2 > 0) st[i][f][h8] = D::step_h(sp[f][h8], hp[f][h8], h / HPW, (h % HPW) * SPH + i, t, hd[f][h8], sp2[f][h8]);
                    else if constexpr (WH > 0) st[i][f][h8] = D::step_h(sp[f][h8], hp[f][h8], h / HPW, (h % HPW) * SPH + i, t, hd[f][h8]);
                    else st[i][f][h8] = D::step(sp[f][h8], (h % HPW) * SPH + i, t, hd[f][h8], grid);
                }
#pragma unroll
        for (int j = 0; j < NG; ++j) {
            if (j >= ng) break;
            const uint4 b01 = xw[(j * 8 + g) * 8 + t * 2], b23 = xw[(j * 8 + g) * 8 + t * 2 + 1];
            const uint4 s0 = xs[j * 8 + 2 * t], s1 = xs[j * 8 + 2 * t + 1];
            uint4 u0 = {}, u1 = {};
            if constexpr (Sums<TY>::on) {
                u0 = ubuf[(h & 1) * TC + j * 8 + 2 * t];
                u1 = ubuf[(h & 1) * TC + j * 8 + 2 * t + 1];
            }
            const float ua[4] = {__uint_as_float(u0.x), __uint_as_float(u0.y), __uint_as_float(u0.z), __uint_as_float(u0.w)};
            const float ub[4] = {__uint_as_float(u1.x), __uint_as_float(u1.y), __uint_as_float(u1.z), __uint_as_float(u1.w)};
            const uint32_t bw[8] = {b01.x, b01.y, b01.z, b01.w, b23.x, b23.y, b23.z, b23.w};
            const float xa[4] = {__uint_as_float(s0.x), __uint_as_float(s0.y), __uint_as_float(s0.z), __uint_as_float(s0.w)};
            const float xb[4] = {__uint_as_float(s1.x), __uint_as_float(s1.y), __uint_as_float(s1.z), __uint_as_float(s1.w)};
#pragma unroll
            for (int i = 0; i < SPH; ++i) {
                const uint32_t bl = bw[2 * i], bh = bw[2 * i + 1];
                const float x0 = xa[i], x1 = xb[i];
#pragma unroll
                for (int f = 0; f < 2; ++f) {
                    float v[4];
                    if constexpr (D::SUB == 32) {
                        int cc[4];
                        mma32(cc, st[i][f][0].lo, st[i][f][1].lo, st[i][f][0].hi, st[i][f][1].hi, bl, bh);
                        v[0] = (float) cc[0] * st[i][f][0].s;
                        v[1] = (float) cc[1] * st[i][f][0].s;
                        v[2] = (float) cc[2] * st[i][f][1].s;
                        v[3] = (float) cc[3] * st[i][f][1].s;
                    } else {
                        int cl[4], ch[4];
                        mma16(cl, st[i][f][0].lo, st[i][f][1].lo, bl);
                        mma16(ch, st[i][f][0].hi, st[i][f][1].hi, bh);
                        v[0] = (float) (cl[0] * st[i][f][0].m0 + ch[0] * st[i][f][0].m1) * hd[f][0];
                        v[1] = (float) (cl[1] * st[i][f][0].m0 + ch[1] * st[i][f][0].m1) * hd[f][0];
                        v[2] = (float) (cl[2] * st[i][f][1].m0 + ch[2] * st[i][f][1].m1) * hd[f][1];
                        v[3] = (float) (cl[3] * st[i][f][1].m0 + ch[3] * st[i][f][1].m1) * hd[f][1];
                    }
                    acc[f][j][0] = fmaf(v[0], x0, acc[f][j][0]);
                    acc[f][j][1] = fmaf(v[1], x1, acc[f][j][1]);
                    acc[f][j][2] = fmaf(v[2], x0, acc[f][j][2]);
                    acc[f][j][3] = fmaf(v[3], x1, acc[f][j][3]);
                    if constexpr (Sums<TY>::on) {   // (the rows' offsets x the tokens' block sums)
                        acc[f][j][0] = fmaf(-st[i][f][0].o, ua[i], acc[f][j][0]);
                        acc[f][j][1] = fmaf(-st[i][f][0].o, ub[i], acc[f][j][1]);
                        acc[f][j][2] = fmaf(-st[i][f][1].o, ua[i], acc[f][j][2]);
                        acc[f][j][3] = fmaf(-st[i][f][1].o, ub[i], acc[f][j][3]);
                    }
                }
            }
        }
    }

    // the write-back: acc[f][j][k] is row (k < 2 ? g : g + 8) of fragment f, token j * 8 + 2 t + (k & 1)
    const int ntok = min(b1 - b0, a.tc);   // (the rest of the expert is other tiles)
#pragma unroll
    for (int j = 0; j < NG; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int tok = j * 8 + 2 * t + (k & 1), rr = g + 8 * (k >> 1);
            if (tok >= ntok) continue;
            float* out = a.dst + (int64_t) (b0 + tok - a.row0) * a.ld;
            if constexpr (GLU) {
                const int row = rt * (NW * 16) + wp * 16 + rr;
                if (row < a.w_rows) {
                    const float gt = fminf(acc[0][j][k], a.limit);
                    const float up = fminf(fmaxf(acc[1][j][k], -a.limit), a.limit);
                    out[row] = (gt / (1.0f + expf(-gt))) * up;
                }
            } else {
#pragma unroll
                for (int f = 0; f < 2; ++f) {
                    const int row = rt * (NW * 32) + (2 * wp + f) * 16 + rr;
                    if (row < a.w_rows) out[row] = acc[f][j][k];
                }
            }
        }
}

template<int TY, bool GLU, int CF>
__global__ void __launch_bounds__(Cfg<CF>::NT) own_kernel(const KArgs a) {
    constexpr int NW = Cfg<CF>::NW;
    extern __shared__ uint4 smem[];
    {   // the codebook into shared memory: the decode's lookups stay off the global-memory path
        uint32_t* g = (uint32_t*) (smem + Lay<TY, CF>::BUF);
        const uint32_t* src = Dec<TY>::table();
        for (int i = threadIdx.x; i < Dec<TY>::GW; i += Cfg<CF>::NT) g[i] = src[i];
        __syncthreads();
    }
    const int n_rt = (a.w_rows + (GLU ? NW * 16 : NW * 32) - 1) / (GLU ? NW * 16 : NW * 32);
    const int tt = blockIdx.x / n_rt, rt = blockIdx.x % n_rt;
    const auto tile_at = [&](int i) { return a.tiles != nullptr ? a.tiles[i] : a.itiles[i]; };
    int lo = 0, hi = a.n;   // the expert whose token tiles hold tt
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (tile_at(mid) <= tt) lo = mid;
        else hi = mid;
    }
    const int e = lo, b1 = a.bounds[e + 1], b0 = a.bounds[e] + (tt - tile_at(e)) * a.tc;
    tile_body<TY, GLU, CF>(a, smem, e, rt, b0, b1, (min(b1 - b0, a.tc) + 7) / 8);
}

// the type takes the wide shape only: two fine CTAs do not fit an SM
template<int TY> struct Wide { static constexpr bool on = 2 * Lay<TY, 0>::smem > kSmemSM; };

template<int TY, bool GLU, int CF>
void launch_c(const KArgs& a, unsigned grid, cudaStream_t s) {
    static_assert(Lay<TY, CF>::smem <= kSmemCTA, "the shape does not fit a CTA's shared memory");
    static bool once = false;
    if (!once) {
        ck(cudaFuncSetAttribute(own_kernel<TY, GLU, CF>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) Lay<TY, CF>::smem),
           "shared memory size");
        once = true;
    }
    own_kernel<TY, GLU, CF><<<grid, Cfg<CF>::NT, Lay<TY, CF>::smem, s>>>(a);
    ck(cudaGetLastError(), GLU ? "gate_up" : "down");
}
template<int TY, bool GLU>
void launch_t(const KArgs& a, unsigned grid, int shape, cudaStream_t s) {
    if constexpr (Wide<TY>::on) {   // (a large raw chunk: the wide shape only)
        launch_c<TY, GLU, 2>(a, grid, s);
    } else {
        if (shape == 1) launch_c<TY, GLU, 1>(a, grid, s);
        else launch_c<TY, GLU, 0>(a, grid, s);
    }
}

template<bool GLU>
void launch(int type, const KArgs& a, unsigned grid, int shape, cudaStream_t s) {
    switch (type) {
        case 16: launch_t<16, GLU>(a, grid, shape, s); break;
        case 17: launch_t<17, GLU>(a, grid, shape, s); break;
        case 18: launch_t<18, GLU>(a, grid, shape, s); break;
        case 22: launch_t<22, GLU>(a, grid, shape, s); break;
        case 19: launch_t<19, GLU>(a, grid, shape, s); break;
        case 29: launch_t<29, GLU>(a, grid, shape, s); break;
        case 20: launch_t<20, GLU>(a, grid, shape, s); break;
        case 42: launch_t<42, GLU>(a, grid, shape, s); break;
        case 23: launch_t<23, GLU>(a, grid, shape, s); break;
        case 12: launch_t<12, GLU>(a, grid, shape, s); break;
        case 14: launch_t<14, GLU>(a, grid, shape, s); break;
        case 8: launch_t<8, GLU>(a, grid, shape, s); break;
        case 39: launch_t<39, GLU>(a, grid, shape, s); break;
        default:
            std::fprintf(stderr, "prefill own: type %d is not covered\n", type);
            std::exit(1);
    }
}

int align_of(int type) { return type == 39 ? 1 : 2; }
bool wide_type(int type) {
    switch (type) {
        case 16: return Wide<16>::on;
        case 17: return Wide<17>::on;
        case 18: return Wide<18>::on;
        case 22: return Wide<22>::on;
        case 19: return Wide<19>::on;
        case 29: return Wide<29>::on;
        case 20: return Wide<20>::on;
        case 42: return Wide<42>::on;
        case 23: return Wide<23>::on;
        case 12: return Wide<12>::on;
        case 14: return Wide<14>::on;
        case 8: return Wide<8>::on;
        case 39: return Wide<39>::on;
        default: return false;
    }
}
#endif   // !__HIPCC__

}  // namespace

#if defined(__HIPCC__)
bool built() { return false; }
bool tuned_device() { return false; }
bool supported(int) { return false; }
#else
namespace {
// the current device's compute capability major * 10 + minor (cached a device: a box may hold different cards)
int device_cc() {
    static int cc[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return 0;
    if (cc[dev] == 0) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return 0;
        cc[dev] = p.major * 10 + p.minor;
    }
    return cc[dev];
}
}  // namespace
bool tuned_device() { return device_cc() == 120; }
bool built() { return device_cc() >= 80; }   // (int8 mma.sync and cp.async)
bool supported(int t) {
    return built() && (t == 16 || t == 17 || t == 18 || t == 22 || t == 39 || t == 19 || t == 29 || t == 20 || t == 42 || t == 23 || t == 12 || t == 14 || t == 8);
}
#endif

bool usable(int t, int64_t rows, int64_t cols) {
    // (a chunk of the type's blocks: 256 values, 128 for the 32- and 64-value block types)
    const int64_t chunk = (t == 39 || t == 20 || t == 42 || t == 8) ? 128 : 256;   // (IQ4_XS, Q4_K, Q6_K: 256-value blocks)
    return supported(t) && rows > 0 && cols > 0 && cols % chunk == 0 && rows % 16 == 0;
}

size_t act_bytes(int64_t rows, int64_t cols, bool sums) {
    return (size_t) rows * (size_t) cols + (size_t) rows * (size_t) (cols / 32) * (sums ? 8 : 4);
}
bool needs_sums(int t) { return t == 12; }

void quantize(const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld, int64_t rows, void* stream, bool sums) {
    if (rows <= 0) return;
    const int64_t n = rows * (cols / 128) * 32;   // threads
    quant_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(
        x, ids, (uint32_t*) xq, (float*) ((uint8_t*) xq + rows * cols), (int) cols, ld, rows, sums);
    ck(cudaGetLastError(), "quantize");
}

Context::Context() {}
Context::~Context() {
    if (tiles_ != nullptr) cudaFree(tiles_);
#if !defined(__HIPCC__)
    if (stats_on() && stats().launches > 0) {
        const Stats& t = stats();
        std::fprintf(stderr, "prefill own: %ld launches (%ld coarse, %ld with the tile table inline), %.1f experts and %.0f "
                             "rows a launch, %.0f tiles a launch\n", t.launches, t.coarse, t.inline_,
                     (double) t.experts / t.launches, (double) t.rows / t.launches, (double) t.tiles / t.launches);
    }
#endif
}

#if defined(__HIPCC__)
void Context::gate_up(const Set&, int, int64_t, int64_t, size_t, float, void*) {
    std::fprintf(stderr, "prefill own: no kernels in this build\n");
    std::exit(1);
}
void Context::down(const Set&, int, int64_t, int64_t, size_t, void*) {
    std::fprintf(stderr, "prefill own: no kernels in this build\n");
    std::exit(1);
}
#else
namespace {
void check_ptr(const Set& s, int type, size_t off) {
    if ((((uintptr_t) s.w + off) | s.stride) % align_of(type) != 0 || (uintptr_t) s.xq % 16 != 0) {
        std::fprintf(stderr, "prefill own: a matrix of type %d or its activations are not aligned\n", type);
        std::exit(1);
    }
}
int sm_count() {
    static const int n = [] {
        int dev = 0;
        cudaDeviceProp p{};
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&p, dev);
        return std::max(1, p.multiProcessorCount);
    }();
    return n;
}
// The tile shape for the set, its tile table on the device, the launch.  rows_w: the matrix's rows a tile of 16-row
// fragments covers twice over (gate/up: 16 a warp; down: 32 a warp).
template<bool GLU>
void run(const Set& s, int type, KArgs& a, int32_t*& tiles, cudaStream_t st) {
    if (s.n > 1024) {
        std::fprintf(stderr, "prefill own: a set of %d experts (at most 1024)\n", s.n);
        std::exit(1);
    }
    if (tiles == nullptr) ck(cudaMalloc(&tiles, (size_t) 1025 * sizeof(int32_t)), "tile table");
    // The tile shape and the tokens a tile, from a small cost model fitted to measurements (tests/moe_own_test
    // --bench, RTX 5090): a tile's time is its decode (about 6 token groups' worth; 3 to 10 choose alike) plus its token groups; the GPU
    // runs `slots` tiles at once; up to two rounds are whole rounds (a launch of 176 coarse tiles on 170 SMs took
    // two), more are a stream.  Large tiles share the decode (many rows: coarse, 128 tokens); a small set wants enough
    // tiles to fill the GPU once (2 experts of 384 rows in 64-token tiles: a third of the SMs idle).
    // STRATA_PROMPT_OWN_TILE=<tokens>: fixed (8-64 fine, 72-128 coarse).  The sums do not depend on the choice.
    static const int tile_env = [] {
        const char* v = getenv("STRATA_PROMPT_OWN_TILE");
        return v != nullptr ? std::clamp(std::atoi(v) / 8 * 8, 8, 128) : 0;
    }();
    long nonempty = 0, rows_all = 0;
    for (int e = 0; e < s.n; ++e) {
        const int r = s.host_bounds[e + 1] - s.host_bounds[e];
        nonempty += r > 0;
        rows_all += r;
    }
    const int per_warp = GLU ? 16 : 32;
    // token tiles of tc tokens, and their token groups
    const auto token_tiles = [&](int tc, long& groups) {
        long t = 0;
        groups = 0;
        for (int e = 0; e < s.n; ++e) {
            const int r = s.host_bounds[e + 1] - s.host_bounds[e];
            t += (r + tc - 1) / tc;
            groups += (long) (r / tc) * (tc / 8) + (r % tc + 7) / 8;
        }
        return t;
    };
    const bool wide = wide_type(type);   // (launch_t: the wide shape - 8 warps, 64 tokens, one CTA an SM)
    const auto count = [&](int tc, bool crs) {
        const int rows_tile = (crs || wide ? Cfg<1>::NW : Cfg<0>::NW) * per_warp;
        long g = 0;
        return token_tiles(tc, g) * ((a.w_rows + rows_tile - 1) / rows_tile);
    };
    int tc = tile_env;
    if (tc == 0) {
        const long sms = sm_count();
        double best = 1e300;
        for (int c = 8; c <= (wide ? Cfg<2>::TC : Cfg<1>::TC); c += 8) {
            const bool crs = c > Cfg<0>::TC;
            long groups = 0;
            const long tt = token_tiles(c, groups);
            if (tt <= 0) break;
            const long tiles = count(c, crs), slots = crs || wide ? sms : 2 * sms;
            const double rounds = tiles <= slots ? 1.0 : tiles <= 2 * slots ? 2.0 : (double) tiles / (double) slots;
            // (a coarse tile: twice the warps of a fine one on the SM, the same time)
            const double est = rounds * (6.0 + (double) groups / (double) tt);
            if (est <= best) {   // (ties: the larger tile, less decode)
                best = est;
                tc = c;
            }
        }
        if (tc == 0) return;
    }
    if (wide) tc = std::min(tc, Cfg<2>::TC);
    const bool coarse = tc > Cfg<0>::TC;
    const long total = count(tc, coarse);
    if (total <= 0) return;
    a.tc = tc;
    if (s.n <= kInline) {
        a.tiles = nullptr;
        a.itiles[0] = 0;
        for (int e = 0; e < s.n; ++e) a.itiles[e + 1] = a.itiles[e] + (s.host_bounds[e + 1] - s.host_bounds[e] + tc - 1) / tc;
    } else {
        tiles_kernel<<<1, 256, 0, st>>>(s.bounds, tiles, s.n, tc);
        ck(cudaGetLastError(), "tiles");
        a.tiles = tiles;
    }
    launch<GLU>(type, a, (unsigned) total, coarse ? 1 : 0, st);
    if (stats_on()) {
        Stats& t = stats();
        ++t.launches;
        t.coarse += coarse;
        t.inline_ += s.n <= kInline;
        t.experts += nonempty;
        t.rows += rows_all;
        t.tiles += total;
    }
}
}  // namespace

void Context::gate_up(const Set& s, int type, int64_t n_ff, int64_t n_embd, size_t up_off, float limit, void* stream) {
    if (s.n <= 0 || s.rows <= 0) return;
    check_ptr(s, type, 0);
    check_ptr(s, type, up_off);
    KArgs a{};
    a.w = (const uint8_t*) s.w;
    a.stride = s.stride;
    a.up_off = up_off;
    a.xq = (const uint8_t*) s.xq;
    a.xd = (const float*) ((const uint8_t*) s.xq + s.rows * n_embd);
    a.xsum = a.xd + s.rows * (n_embd / 32);
    a.bounds = s.bounds;
    a.dst = s.dst;
    a.ld = s.ld;
    a.n = s.n;
    a.row0 = s.row0;
    a.w_rows = (int) n_ff;
    a.K = (int) n_embd;
    a.limit = limit;
    run<true>(s, type, a, tiles_, (cudaStream_t) stream);
}

void Context::down(const Set& s, int type, int64_t n_embd, int64_t n_ff, size_t down_off, void* stream) {
    if (s.n <= 0 || s.rows <= 0) return;
    check_ptr(s, type, down_off);
    KArgs a{};
    a.w = (const uint8_t*) s.w + down_off;
    a.stride = s.stride;
    a.xq = (const uint8_t*) s.xq;
    a.xd = (const float*) ((const uint8_t*) s.xq + s.rows * n_ff);
    a.xsum = a.xd + s.rows * (n_ff / 32);
    a.bounds = s.bounds;
    a.dst = s.dst;
    a.ld = s.ld;
    a.n = s.n;
    a.row0 = s.row0;
    a.w_rows = (int) n_embd;
    a.K = (int) n_ff;
    a.limit = 0.0f;
    run<false>(s, type, a, tiles_, (cudaStream_t) stream);
}
#endif

}  // namespace strata::prefill::own
