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
#include <string>
#include <vector>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include "moe_own_decode.cuh"

#if defined(__HIPCC__)
#include <cstring>
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
#if defined(__HIPCC__)
    // (the WMMA backend: the values in their own order - a lane reads 16 consecutive ones)
    q[(r * cols + (int64_t) c * 128) / 4 + bi * 8 + wd] = w;
#else
    q[(r * cols + (int64_t) c * 128) / 4 + (wd & 3) * 8 + bi * 2 + (wd >> 2)] = w;
#endif
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
// The token tiles of r tokens: tc a tile, and the last one takes what is left up to the shape's tmax - 2,048 tokens are
// 17 tiles of 120 (the last 128), not 16 of 128: with 20 row tiles that is 340 tiles, two full rounds of 170 SMs.
__host__ __device__ inline int tok_tiles(int r, int tc, int tmax) { return r <= 0 ? 0 : max(1, (r - (tmax - tc) + tc - 1) / tc); }
__global__ void tiles_kernel(const int32_t* __restrict__ bounds, int32_t* __restrict__ tiles, int n, int tc, int tmax) {
    __shared__ int a[1025], b[1025];
    for (int i = threadIdx.x; i <= n; i += blockDim.x) a[i] = i == 0 ? 0 : tok_tiles(bounds[i] - bounds[i - 1], tc, tmax);
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
    ptrdiff_t up_off;       // gate/up: the up matrix from the gate matrix (two tensors: any distance)
    const uint8_t* xq;      // activations: int8 [rows][K]
    const float* xd;        // their scales [rows][K / 32]
    const float* xsum;      // their block sums (scale x the sum of the int8), the same shape
    const int32_t* bounds;
    const int32_t* tiles;   // the tile table on the device, or null: itiles
    int32_t itiles[kInline + 1];
    float* dst;
    int64_t ld;
    int n, row0, w_rows, K;
    int order;              // 1: an expert's tiles by row tile, then token tile (0: the other way, an A/B)
    int tc;                 // tokens a tile in this launch (a multiple of 8, at most the shape's TC)
    int al4;                // the matrix (its experts, both of gate / up) starts on a multiple of 4
    int tmax;               // the shape's TC: an expert's last tile holds up to this many (tok_tiles)
    float limit;
};

#if !defined(__HIPCC__)
// ================================================================ the NVIDIA backend (mma.sync, cp.async)
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
    // One weight buffer instead of two (W1): the types whose raw chunk is large - two buffers of the coarse shape's
    // 256 rows do not fit a CTA's shared memory (Q5_K, Q6_K, Q8_0, IQ3_S, Q3_K).  The half chunk's steps are decoded
    // into registers before its token groups run, so the next chunk's bytes come into the same buffer under the groups
    // (one more barrier a chunk) - instead of a narrower tile.
    // A slot's stride is an odd number of 16-byte words (WSS the data's, WHS the header's): shared memory has 32
    // banks of 4 bytes, and a warp's eight rows read their slots at once - with a stride of 8 words (IQ3_S, Q6_K)
    // the eight land in one bank and the load takes eight turns; odd strides spread them over eight banks.
    // IQ3_S on a matrix whose rows are 16-byte multiples: gate/up 0.758 -> 0.55 ms (tests/moe_own_test --bench).
    // The odd stride is not taken where its extra word costs the type a shape: two fine CTAs an SM (IQ2_S, whose
    // codebook is 8 KB: 102,400 bytes for two, exactly the SM's), or the second weight buffer.
    static constexpr size_t bytes(int nt, int tc, int slot, bool w1) {   // a CTA's shared memory with that slot stride
        return (size_t) ((w1 ? 1 : 2) * nt * slot + 2 * tc * 8 + (Sums<TY>::on ? 4 : 2) * tc) * 16 + (size_t) Dec<TY>::GW * 4;
    }
    static constexpr int kEven = WS, kOdd = ((WS - WH) | 1) + (WH > 0 ? (WH | 1) : 0);
    static constexpr bool w1_even = bytes(Cfg<1>::NT, Cfg<1>::TC, kEven, false) > kSmemCTA,
                          w1_odd = bytes(Cfg<1>::NT, Cfg<1>::TC, kOdd, false) > kSmemCTA;
    static constexpr bool odd = !(w1_odd && !w1_even) &&
                                !(2 * bytes(Cfg<0>::NT, Cfg<0>::TC, kOdd, w1_odd) > kSmemSM && 2 * bytes(Cfg<0>::NT, Cfg<0>::TC, kEven, w1_even) <= kSmemSM);
    static constexpr int WSS = odd ? ((WS - WH) | 1) : WS - WH, WHS = WH > 0 ? (odd ? (WH | 1) : WH) : 0;
    static constexpr bool W1 = odd ? w1_odd : w1_even;
    static constexpr int BUF = (W1 ? 1 : 2) * Cfg<CF>::NT * (WSS + WHS) + 2 * Cfg<CF>::TC * 8 + (Sums<TY>::on ? 4 : 2) * Cfg<CF>::TC;   // 16-byte words
    static constexpr size_t smem = (size_t) BUF * 16 + (size_t) Dec<TY>::GW * 4;   // (+ the codebook)
};

// One tile of ng groups of 8 tokens (ng <= TC / 8: the loops over the groups are unrolled to that and each group is
// skipped past ng - the sums stay in registers, a tile costs its own groups).  e: the expert; rt: the row tile; b0, b1:
// the tile's first activation row and the expert's end.
template<int TY, bool GLU, int CF>
__device__ __forceinline__ void tile_body(const KArgs& a, uint4* smem, int e, int rt, int b0, int b1, int ng) {
    typedef Dec<TY> D;
    constexpr int NW = Cfg<CF>::NW, TC = Cfg<CF>::TC, NT = Cfg<CF>::NT, NG = TC / 8;
    constexpr bool W1 = Lay<TY, CF>::W1;
    constexpr int WB = W1 ? 1 : 2;   // weight buffers
    constexpr int WS = Lay<TY, CF>::WS, WH = Lay<TY, CF>::WH, WD = Lay<TY, CF>::WD, W2 = Lay<TY, CF>::W2, HPW = D::WCH / 128, SPH = 4;   // (half chunks a weight chunk; steps a half chunk)
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    constexpr int WSD = Lay<TY, CF>::WSS, WHS = Lay<TY, CF>::WHS;   // a slot's data words and its header's (in hbuf), as strides
    uint4* wbuf = smem;                       // [WB][NT slots][WSD]
    uint4* hbuf = wbuf + WB * NT * WSD;       // [WB][NT slots][WH]: the 256-value blocks' headers, one a block (both its half chunks)
    uint4* xbuf = hbuf + WB * NT * WHS;       // [2][TC][8]
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
        return we + (GLU && f ? a.up_off : (ptrdiff_t) 0) + (size_t) min(row, a.w_rows - 1) * row_bytes;
    };
    const uint8_t* my_slot = slot_row(threadIdx.x);
    const auto issue_w = [&](int c) {   // this thread's slot: the 16-byte words around its row's chunk c
        uint4* dst = wbuf + ((W1 ? 0 : (c & 1)) * NT + threadIdx.x) * WSD;
        if constexpr (WH > 0) {   // (the block's header with its first half chunk)
            if ((c & 1) == 0) {
                uint4* hdst = hbuf + ((W1 ? 0 : ((c >> 1) & 1)) * NT + threadIdx.x) * WHS;
                const uintptr_t src = (uintptr_t) (my_slot + Rng<TY>::hdr(c));
                const uint8_t* p = (const uint8_t*) (src & ~(uintptr_t) 15);
                const int last = (int) (((src & 15) + Rng<TY>::HDR - 1) / 16);
#pragma unroll
                for (int k = 0; k < WH; ++k)
                    if (k <= last) cp16(hdst + k, p + 16 * k);   // (only the words the range touches: none past it, no copy twice)
            }
        }
        const uintptr_t src = (uintptr_t) (my_slot + Rng<TY>::dat(c));
        const uint8_t* p = (const uint8_t*) (src & ~(uintptr_t) 15);
        const int last = (int) (((src & 15) + D::CB - 1) / 16);   // (no word past the chunk: it may be past the pool)
#pragma unroll
        for (int k = 0; k < WD; ++k)
            if (k <= last) cp16(dst + k, p + 16 * k);
        if constexpr (W2 > 0) {
            const uintptr_t src2 = (uintptr_t) (my_slot + Rng2<TY>::dat2(c));
            const uint8_t* p2 = (const uint8_t*) (src2 & ~(uintptr_t) 15);
            const int last2 = (int) (((src2 & 15) + Rng2<TY>::CB2 - 1) / 16);
#pragma unroll
            for (int k = 0; k < W2; ++k)
                if (k <= last2) cp16(dst + WD + k, p2 + 16 * k);
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
        if (!W1 && h % HPW == 0 && c + 1 < n_wc) issue_w(c + 1);
        cp_commit();
        if (h % HPW == 0) {
#pragma unroll
            for (int f = 0; f < 2; ++f)
#pragma unroll
                for (int h8 = 0; h8 < 2; ++h8) {
                    const uint8_t* slot = (const uint8_t*) (wbuf + ((W1 ? 0 : (c & 1)) * NT + soff[f][h8]) * WSD);
                    sp[f][h8] = slot + ((uintptr_t) (rowp[f][h8] + Rng<TY>::dat(c)) & 15);
                    if constexpr (W2 > 0) sp2[f][h8] = slot + WD * 16 + ((uintptr_t) (rowp[f][h8] + Rng2<TY>::dat2(c)) & 15);
                    if constexpr (WH > 0) {
                        const uint8_t* hslot = (const uint8_t*) (hbuf + ((W1 ? 0 : ((c >> 1) & 1)) * NT + soff[f][h8]) * WHS);
                        hp[f][h8] = hslot + ((uintptr_t) (rowp[f][h8] + Rng<TY>::hdr(c)) & 15);
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
        if constexpr (Four<TY>::on) {   // (the half's four blocks at once, their shared bytes read once)
#pragma unroll
            for (int f = 0; f < 2; ++f)
#pragma unroll
                for (int h8 = 0; h8 < 2; ++h8) {
                    Step four[SPH];
                    if constexpr (W2 > 0) D::steps4(sp[f][h8], hp[f][h8], c, t, hd[f][h8], sp2[f][h8], four);
                    else D::steps4(sp[f][h8], hp[f][h8], c, t, hd[f][h8], four);
#pragma unroll
                    for (int i = 0; i < SPH; ++i) st[i][f][h8] = four[i];
                }
        } else {
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
        }
        if (W1 && h % HPW == HPW - 1 && c + 1 < n_wc) {
            __syncthreads();   // every thread has decoded this chunk: its buffer takes the next one, under the groups
            issue_w(c + 1);
            cp_commit();
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
                int ol[4] = {0, 0, 0, 0}, oh[4] = {0, 0, 0, 0};   // (OFS16: the tokens' sums of each 16 activations)
                if constexpr (Ofs16<TY>::on) {
                    mma16(ol, 0x01010101u, 0x01010101u, bl);
                    mma16(oh, 0x01010101u, 0x01010101u, bh);
                }
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
                    } else if constexpr (Fs16<TY>::on) {   // (a float scale a 16 values)
                        int cl[4], ch[4];
                        mma16(cl, st[i][f][0].lo, st[i][f][1].lo, bl);
                        mma16(ch, st[i][f][0].hi, st[i][f][1].hi, bh);
                        v[0] = fmaf((float) cl[0], st[i][f][0].s, (float) ch[0] * st[i][f][0].s2);
                        v[1] = fmaf((float) cl[1], st[i][f][0].s, (float) ch[1] * st[i][f][0].s2);
                        v[2] = fmaf((float) cl[2], st[i][f][1].s, (float) ch[2] * st[i][f][1].s2);
                        v[3] = fmaf((float) cl[3], st[i][f][1].s, (float) ch[3] * st[i][f][1].s2);
                    } else {
                        int cl[4], ch[4];
                        mma16(cl, st[i][f][0].lo, st[i][f][1].lo, bl);
                        mma16(ch, st[i][f][0].hi, st[i][f][1].hi, bh);
                        v[0] = (float) (cl[0] * st[i][f][0].m0 + ch[0] * st[i][f][0].m1) * hd[f][0];
                        v[1] = (float) (cl[1] * st[i][f][0].m0 + ch[1] * st[i][f][0].m1) * hd[f][0];
                        v[2] = (float) (cl[2] * st[i][f][1].m0 + ch[2] * st[i][f][1].m1) * hd[f][1];
                        v[3] = (float) (cl[3] * st[i][f][1].m0 + ch[3] * st[i][f][1].m1) * hd[f][1];
                    }
                    if constexpr (Ofs16<TY>::on) {   // (the offsets x each 16 values' sum of activations)
                        v[0] -= fmaf(st[i][f][0].o, (float) ol[0], st[i][f][0].o2 * (float) oh[0]);
                        v[1] -= fmaf(st[i][f][0].o, (float) ol[1], st[i][f][0].o2 * (float) oh[1]);
                        v[2] -= fmaf(st[i][f][1].o, (float) ol[2], st[i][f][1].o2 * (float) oh[2]);
                        v[3] -= fmaf(st[i][f][1].o, (float) ol[3], st[i][f][1].o2 * (float) oh[3]);
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
    const int ntok = b1 - b0;   // (b1: the tile's end)
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
    const auto tile_at = [&](int i) { return a.tiles != nullptr ? a.tiles[i] : a.itiles[i]; };
    int lo = 0, hi = a.n;   // the expert whose tiles hold this one
    {
        const int tt0 = blockIdx.x / n_rt;
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (tile_at(mid) <= tt0) lo = mid;
            else hi = mid;
        }
    }
    // An expert's tiles in the order row tile, then token tile: the CTAs that run together share weight rows (a dense
    // layer: 170 CTAs on 170 different row tiles pulled 125 MB of 4-bit weights through the 100 MB L2 for every
    // token tile - the weight copies ran at the memory's bandwidth).  The tokens' activations are the smaller side.
    const int e = lo, b1 = a.bounds[e + 1];
    const int local = (int) blockIdx.x - tile_at(e) * n_rt, ntt = tile_at(e + 1) - tile_at(e);
    const int rt = a.order ? local / ntt : local % n_rt, ttl = a.order ? local % ntt : local / n_rt;
    const int b0 = a.bounds[e] + ttl * a.tc;
    const int ntok = ttl == ntt - 1 ? b1 - b0 : a.tc;   // (the expert's last tile: the rest, at most the shape's TC)
    tile_body<TY, GLU, CF>(a, smem, e, rt, b0, b0 + ntok, (ntok + 7) / 8);
}

// the type takes the wide shape only: two fine CTAs do not fit an SM
template<int TY> struct Wide {
    static constexpr bool on = 2 * Lay<TY, 0>::smem > kSmemSM;
    // (two fine CTAs fit, a coarse one does not: Q3_K - the fine shape only)
    static constexpr bool fine_only = !on && Lay<TY, 1>::smem > kSmemCTA;
};
// (the shapes the measurements were made with: two fine CTAs an SM and the coarse shape for the types of the default
// list - a layout change that drops one shows here, not as a slower prompt)
#define STRATA_OWN_BOTH(TY) static_assert(!Wide<TY>::on && !Wide<TY>::fine_only, "own GEMM: type " #TY " lost a tile shape")
STRATA_OWN_BOTH(16); STRATA_OWN_BOTH(17); STRATA_OWN_BOTH(18); STRATA_OWN_BOTH(22); STRATA_OWN_BOTH(19); STRATA_OWN_BOTH(20);
STRATA_OWN_BOTH(42); STRATA_OWN_BOTH(23); STRATA_OWN_BOTH(12); STRATA_OWN_BOTH(13); STRATA_OWN_BOTH(14);
#undef STRATA_OWN_BOTH

template<int TY, bool GLU, int CF>
void launch_c(const KArgs& a, unsigned grid, cudaStream_t s) {
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
    } else if constexpr (Wide<TY>::fine_only) {
        launch_c<TY, GLU, 0>(a, grid, s);
    } else {
        if (shape == 1) launch_c<TY, GLU, 1>(a, grid, s);
        else launch_c<TY, GLU, 0>(a, grid, s);
    }
}


#else   // __HIPCC__
// ================================================================ the AMD backend: WMMA int8 (wave32)
// v_wmma_i32_16x16x16_iu8: D[16 x 16] = A[16 x 16] B[16 x 16] + C, int8 -> int32, one wave.
//   gfx11 (RDNA3 / 3.5): A lane l holds the 16 k of row l % 16 (both halves of the wave the same), B lane l the 16 k
//     of column l % 16, D lane l holds D[2 i + l / 16][l % 16] (i < 8).
//   gfx12 (RDNA4): A lane l holds A[l % 16][8 (l / 16) + j], B lane l B[8 (l / 16) + j][l % 16] (j < 8), D lane l
//     holds D[8 (l / 16) + i][l % 16].
// The tile: one shape - 4 waves, 64 tokens (LDS is 64 KiB a work group), single buffers filled with plain loads (no
// cp.async here): a wave takes 16 rows of two fragments as on NVIDIA, a lane decodes all 32 values of its row
// (l % 16) a step, the rows' scales go through a small LDS table of the wave (a lane's D holds 8 rows, not its own),
// a token group is 16 tokens (lane l: token l % 16).  One k16 WMMA for the first 16 values of a step, one for the
// second.  The per-arch choices are the two macros below and the fragment helpers; the tile's size is one constant
// set (gfx12 on an R9700 will want its own - it is not measured yet).
#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_OWN_W12 1
#else
#define STRATA_OWN_W12 0
#endif
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
#define STRATA_OWN_W11 1
#else
#define STRATA_OWN_W11 0
#endif
typedef int ow_i2 __attribute__((ext_vector_type(2)));
typedef int ow_i4 __attribute__((ext_vector_type(4)));
typedef int ow_i8 __attribute__((ext_vector_type(8)));
// D = A B + C; a16 / b16: the lane's 16 int8 of its row / its token column (gfx12 takes the half its lane group holds)
__device__ __forceinline__ ow_i8 wmma(uint4 a16, uint4 b16, ow_i8 c, int hi) {
#if STRATA_OWN_W11
    (void) hi;
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ow_i4{(int) a16.x, (int) a16.y, (int) a16.z, (int) a16.w}, true,
                                                      ow_i4{(int) b16.x, (int) b16.y, (int) b16.z, (int) b16.w}, c, false);
#elif STRATA_OWN_W12
    const ow_i2 a = hi ? ow_i2{(int) a16.z, (int) a16.w} : ow_i2{(int) a16.x, (int) a16.y};
    const ow_i2 b = hi ? ow_i2{(int) b16.z, (int) b16.w} : ow_i2{(int) b16.x, (int) b16.y};
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
#else
    (void) a16; (void) b16; (void) hi;
    __builtin_trap();   // (an architecture without WMMA: built() is false there)
    return c;
#endif
}
// the D row (0-15) of a lane's element i, and where a row's scale sits so that a lane's eight are consecutive
#if STRATA_OWN_W12
#define OW_DROW(i, hi) (8 * (hi) + (i))
#define OW_POS(r) (r)
#else
#define OW_DROW(i, hi) (2 * (i) + (hi))
#define OW_POS(r) ((((r) & 1) << 3) + ((r) >> 1))
#endif

constexpr int HNW = 4, HTC = 64, HNT = HNW * 32, HNG = HTC / 16;
template<int TY> struct HLay {
    static constexpr int WH = Rng<TY>::HDR > 0 ? (16 - Dec<TY>::AL + Rng<TY>::HDR + 15) / 16 : 0;
    static constexpr int WD = (16 - Dec<TY>::AL + Dec<TY>::CB + 15) / 16;
    static constexpr int W2 = Rng2<TY>::CB2 > 0 ? (16 - Dec<TY>::AL + Rng2<TY>::CB2 + 15) / 16 : 0;
    static constexpr int WS = WH + WD + W2;
    static constexpr int GW = Dec<TY>::GW > 0 ? Dec<TY>::GW : 4;
};

template<int TY, bool GLU>
__global__ void __launch_bounds__(HNT) own_kernel(const KArgs a) {
    typedef Dec<TY> D;
    constexpr int WS = HLay<TY>::WS, WH = HLay<TY>::WH, WD = HLay<TY>::WD, W2 = HLay<TY>::W2, HPW = D::WCH / 128, SPH = 4;
    __shared__ __align__(16) uint4 wbuf[HNT * WS];          // the rows' raw chunks: a slot a thread
    __shared__ __align__(16) uint4 xbuf[HTC * 8];           // the tokens' 128 int8 of the half chunk
    __shared__ __align__(16) uint4 sbuf[HTC], ubuf[HTC];    // their four scales, their four block sums
    __shared__ __align__(16) float scl[HNW][2][SPH][2][16]; // a wave's rows' scales: [fragment][step][first / second 16][OW_POS(row)]
    __shared__ __align__(16) float ofs[HNW][2][SPH][2][16]; // their offsets (SUMS, OFS16 types)
    __shared__ __align__(16) uint32_t gridb[HLay<TY>::GW];
    {
        const uint32_t* src = D::table();
        for (int i = threadIdx.x; i < D::GW; i += HNT) gridb[i] = src[i];
    }
    const uint32_t* grid = gridb;
    const int n_rt = (a.w_rows + (GLU ? HNW * 16 : HNW * 32) - 1) / (GLU ? HNW * 16 : HNW * 32);
    const auto tile_at = [&](int i) { return a.tiles != nullptr ? a.tiles[i] : a.itiles[i]; };
    int lo = 0, hi_e = a.n;
    {
        const int tt0 = blockIdx.x / n_rt;
        while (hi_e - lo > 1) {
            const int mid = (lo + hi_e) / 2;
            if (tile_at(mid) <= tt0) lo = mid;
            else hi_e = mid;
        }
    }
    // (an expert's tiles in the order row tile, then token tile - see the NVIDIA kernel)
    const int e = lo;
    const int local = (int) blockIdx.x - tile_at(e) * n_rt, ntt = tile_at(e + 1) - tile_at(e);
    const int rt = a.order ? local / ntt : local % n_rt, ttl = a.order ? local % ntt : local / n_rt;
    const int b0 = a.bounds[e] + ttl * a.tc;
    const int ntok = ttl == ntt - 1 ? a.bounds[e + 1] - b0 : a.tc, ng = (ntok + 15) / 16, b1 = b0 + ntok;   // (the last tile: the rest)
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, r = lane & 15, hi = lane >> 4;
    const uint8_t* we = a.w + (size_t) e * a.stride;
    const size_t row_bytes = (size_t) (a.K / D::WCH) * Rng<TY>::RB;
    const int n_h = a.K / 128;
    const auto slot_row = [&](int s) -> const uint8_t* {
        const int f = s / (HNT / 2), ww = (s / 16) % HNW, rr = s % 16;
        const int row = GLU ? rt * (HNW * 16) + ww * 16 + rr : rt * (HNW * 32) + (2 * ww + f) * 16 + rr;
        return we + (GLU && f ? a.up_off : (ptrdiff_t) 0) + (size_t) min(row, a.w_rows - 1) * row_bytes;
    };
    const uint8_t* my_slot = slot_row(threadIdx.x);
    // a range of the row into the slot: the 16-byte words around it (none past it)
    const auto copy_range = [&](uint4* dst, const uint8_t* srcp, int bytes, int words) {
        const uintptr_t src = (uintptr_t) srcp;
        const uint4* pw = (const uint4*) (src & ~(uintptr_t) 15);
        const int last = (int) (((src & 15) + bytes - 1) / 16);
#pragma unroll
        for (int k = 0; k < words; ++k) dst[k] = pw[min(k, last)];
    };
    const uint8_t* rowp[2];
    int soff[2];
#pragma unroll
    for (int f = 0; f < 2; ++f) {
        soff[f] = f * (HNT / 2) + wp * 16 + r;
        rowp[f] = slot_row(soff[f]);
    }
    float acc[2][HNG][8];
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
        for (int j = 0; j < HNG; ++j)
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[f][j][k] = 0.0f;
    const uint8_t* sp[2];
    const uint8_t* hp[2] = {};
    const uint8_t* sp2[2] = {};
    float hd[2];
    for (int h = 0; h < n_h; ++h) {
        __syncthreads();   // everyone is done with the buffers of half chunk h - 1 (and the codebook is in place)
        const int c = h / HPW;
        if (h % HPW == 0) {
            uint4* dst = wbuf + threadIdx.x * WS;
            if constexpr (WH > 0) {
                if ((c & 1) == 0) copy_range(dst, my_slot + Rng<TY>::hdr(c), Rng<TY>::HDR, WH);   // (the block's header once)
            }
            copy_range(dst + WH, my_slot + Rng<TY>::dat(c), D::CB, WD);
            if constexpr (W2 > 0) copy_range(dst + WH + WD, my_slot + Rng2<TY>::dat2(c), Rng2<TY>::CB2, W2);
        }
#pragma unroll
        for (int k = 0; k < HTC * 8 / HNT; ++k) {
            const int i = threadIdx.x + k * HNT, tok = i >> 3, word = i & 7;
            if (tok < ng * 16) {
                const int64_t rr = min(b0 + tok, b1 - 1) - a.row0;
                xbuf[i] = *(const uint4*) (a.xq + rr * a.K + (int64_t) h * 128 + word * 16);
            }
        }
        if ((int) threadIdx.x < ng * 16) {
            const int64_t rr = min(b0 + (int) threadIdx.x, b1 - 1) - a.row0;
            sbuf[threadIdx.x] = *(const uint4*) (a.xd + rr * (a.K / 32) + (int64_t) h * 4);
            if constexpr (Sums<TY>::on) ubuf[threadIdx.x] = *(const uint4*) (a.xsum + rr * (a.K / 32) + (int64_t) h * 4);
        }
        __syncthreads();
        if (h % HPW == 0) {
#pragma unroll
            for (int f = 0; f < 2; ++f) {
                const uint8_t* slot = (const uint8_t*) (wbuf + soff[f] * WS);
                sp[f] = slot + WH * 16 + ((uintptr_t) (rowp[f] + Rng<TY>::dat(c)) & 15);
                if constexpr (W2 > 0) sp2[f] = slot + (WH + WD) * 16 + ((uintptr_t) (rowp[f] + Rng2<TY>::dat2(c)) & 15);
                if constexpr (WH > 0) {
                    hp[f] = slot + ((uintptr_t) (rowp[f] + Rng<TY>::hdr(c)) & 15);
                    hd[f] = D::head(hp[f]);
                } else {
                    hd[f] = D::head(sp[f]);
                }
            }
        }
        // the lane's row of both fragments, the half chunk's four steps: its 32 int8 a step, its scales into the wave's
        // table (the lanes of one wave write and then read it: no barrier)
        uint4 a_lo[SPH][2], a_hi[SPH][2];
#pragma unroll
        for (int i = 0; i < SPH; ++i)
#pragma unroll
            for (int f = 0; f < 2; ++f) {
                Step st[4];
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    if constexpr (W2 > 0) st[t] = D::step_h(sp[f], hp[f], h / HPW, (h % HPW) * SPH + i, t, hd[f], sp2[f]);
                    else if constexpr (WH > 0) st[t] = D::step_h(sp[f], hp[f], h / HPW, (h % HPW) * SPH + i, t, hd[f]);
                    else st[t] = D::step(sp[f], (h % HPW) * SPH + i, t, hd[f], grid);
                }
                a_lo[i][f] = make_uint4(st[0].lo, st[1].lo, st[2].lo, st[3].lo);
                a_hi[i][f] = make_uint4(st[0].hi, st[1].hi, st[2].hi, st[3].hi);
                if (hi == 0) {
                    float s0, s1;   // the first and the second 16 values' scales
                    if constexpr (Fs16<TY>::on) {
                        s0 = st[0].s;
                        s1 = st[0].s2;
                    } else if constexpr (D::SUB == 16) {
                        s0 = (float) st[0].m0 * hd[f];
                        s1 = (float) st[0].m1 * hd[f];
                    } else {
                        s0 = s1 = st[0].s;
                    }
                    scl[wp][f][i][0][OW_POS(r)] = s0;
                    scl[wp][f][i][1][OW_POS(r)] = s1;
                    if constexpr (Sums<TY>::on || Ofs16<TY>::on) ofs[wp][f][i][0][OW_POS(r)] = st[0].o;
                    if constexpr (Ofs16<TY>::on) ofs[wp][f][i][1][OW_POS(r)] = st[0].o2;
                }
            }
        __syncwarp();
#pragma unroll
        for (int j = 0; j < HNG; ++j) {
            if (j >= ng) break;
            const int tok = j * 16 + r;
            const uint4 xs4 = sbuf[tok];
            const float xs[4] = {__uint_as_float(xs4.x), __uint_as_float(xs4.y), __uint_as_float(xs4.z), __uint_as_float(xs4.w)};
            uint4 us4 = {};
            if constexpr (Sums<TY>::on) us4 = ubuf[tok];
            const float us[4] = {__uint_as_float(us4.x), __uint_as_float(us4.y), __uint_as_float(us4.z), __uint_as_float(us4.w)};
#pragma unroll
            for (int i = 0; i < SPH; ++i) {
                const uint4 b_lo = xbuf[tok * 8 + i * 2], b_hi = xbuf[tok * 8 + i * 2 + 1];
#pragma unroll
                for (int f = 0; f < 2; ++f) {
                    const ow_i8 z = {0, 0, 0, 0, 0, 0, 0, 0};
                    float v[8];
                    if constexpr (D::SUB == 32 && !Fs16<TY>::on) {   // one scale the 32 values: both k16 into one sum
                        ow_i8 d = wmma(a_lo[i][f], b_lo, z, hi);
                        d = wmma(a_hi[i][f], b_hi, d, hi);
#pragma unroll
                        for (int k = 0; k < 8; ++k) v[k] = (float) d[k] * scl[wp][f][i][0][8 * hi + k];
                    } else {
                        const ow_i8 d0 = wmma(a_lo[i][f], b_lo, z, hi), d1 = wmma(a_hi[i][f], b_hi, z, hi);
#pragma unroll
                        for (int k = 0; k < 8; ++k)
                            v[k] = fmaf((float) d0[k], scl[wp][f][i][0][8 * hi + k], (float) d1[k] * scl[wp][f][i][1][8 * hi + k]);
                    }
                    if constexpr (Ofs16<TY>::on) {   // (the offsets x each 16 values' sum of activations: a product with ones)
                        const uint4 ones = make_uint4(0x01010101u, 0x01010101u, 0x01010101u, 0x01010101u);
                        const ow_i8 o0 = wmma(ones, b_lo, z, hi), o1 = wmma(ones, b_hi, z, hi);
#pragma unroll
                        for (int k = 0; k < 8; ++k)
                            v[k] -= fmaf(ofs[wp][f][i][0][8 * hi + k], (float) o0[k], ofs[wp][f][i][1][8 * hi + k] * (float) o1[k]);
                    }
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        acc[f][j][k] = fmaf(v[k], xs[i], acc[f][j][k]);
                        if constexpr (Sums<TY>::on) acc[f][j][k] = fmaf(-ofs[wp][f][i][0][8 * hi + k], us[i], acc[f][j][k]);
                    }
                }
            }
        }
    }
    // the write-back: acc[f][j][k] is row OW_DROW(k, hi) of fragment f, token j * 16 + r
#pragma unroll
    for (int j = 0; j < HNG; ++j)
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int tok = j * 16 + r, rr = OW_DROW(k, hi);
            if (tok >= ntok) continue;
            float* out = a.dst + (int64_t) (b0 + tok - a.row0) * a.ld;
            if constexpr (GLU) {
                const int row = rt * (HNW * 16) + wp * 16 + rr;
                if (row < a.w_rows) {
                    const float gt = fminf(acc[0][j][k], a.limit);
                    const float up = fminf(fmaxf(acc[1][j][k], -a.limit), a.limit);
                    out[row] = (gt / (1.0f + expf(-gt))) * up;
                }
            } else {
#pragma unroll
                for (int f = 0; f < 2; ++f) {
                    const int row = rt * (HNW * 32) + (2 * wp + f) * 16 + rr;
                    if (row < a.w_rows) out[row] = acc[f][j][k];
                }
            }
        }
}

template<int TY> struct Wide { static constexpr bool on = false, fine_only = true; };   // (one shape here)
template<int TY, bool GLU>
void launch_t(const KArgs& a, unsigned grid, int, cudaStream_t s) {
    own_kernel<TY, GLU><<<grid, HNT, 0, s>>>(a);
    ck(cudaGetLastError(), GLU ? "gate_up" : "down");
}
#endif   // backends

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
#if defined(__HIPCC__)
        case 12: launch_t<12, GLU>(a, grid, shape, s); break;
        case 13: launch_t<13, GLU>(a, grid, shape, s); break;
#else   // (the matrix on a multiple of 4: the decode's 4-byte loads are one load each, Q5_K -3%)
        case 12: a.al4 ? launch_t<112, GLU>(a, grid, shape, s) : launch_t<12, GLU>(a, grid, shape, s); break;
        case 13: a.al4 ? launch_t<113, GLU>(a, grid, shape, s) : launch_t<13, GLU>(a, grid, shape, s); break;
#endif
        case 14: launch_t<14, GLU>(a, grid, shape, s); break;
        case 8: launch_t<8, GLU>(a, grid, shape, s); break;
        case 40: launch_t<40, GLU>(a, grid, shape, s); break;
        case 2: launch_t<2, GLU>(a, grid, shape, s); break;
        case 3: launch_t<3, GLU>(a, grid, shape, s); break;
        case 6: launch_t<6, GLU>(a, grid, shape, s); break;
        case 11: launch_t<11, GLU>(a, grid, shape, s); break;
        case 10: launch_t<10, GLU>(a, grid, shape, s); break;
        case 21: launch_t<21, GLU>(a, grid, shape, s); break;
        case 39: launch_t<39, GLU>(a, grid, shape, s); break;
        default:
            std::fprintf(stderr, "prefill own: type %d is not covered\n", type);
            std::exit(1);
    }
}

int align_of(int type) { return type == 39 || type == 40 ? 1 : 2; }
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
        case 40: return Wide<40>::on;
        case 2: return Wide<2>::on;
        case 3: return Wide<3>::on;
        case 6: return Wide<6>::on;
        case 13: return Wide<13>::on;
        case 11: return Wide<11>::on;
        case 10: return Wide<10>::on;
        case 21: return Wide<21>::on;
        case 39: return Wide<39>::on;
        default: return false;
    }
}
bool fine_only_type(int type) {
    switch (type) {
        case 16: return Wide<16>::fine_only;
        case 17: return Wide<17>::fine_only;
        case 18: return Wide<18>::fine_only;
        case 22: return Wide<22>::fine_only;
        case 19: return Wide<19>::fine_only;
        case 29: return Wide<29>::fine_only;
        case 20: return Wide<20>::fine_only;
        case 42: return Wide<42>::fine_only;
        case 23: return Wide<23>::fine_only;
        case 12: return Wide<12>::fine_only;
        case 14: return Wide<14>::fine_only;
        case 8: return Wide<8>::fine_only;
        case 40: return Wide<40>::fine_only;
        case 2: return Wide<2>::fine_only;
        case 3: return Wide<3>::fine_only;
        case 6: return Wide<6>::fine_only;
        case 13: return Wide<13>::fine_only;
        case 11: return Wide<11>::fine_only;
        case 10: return Wide<10>::fine_only;
        case 21: return Wide<21>::fine_only;
        case 39: return Wide<39>::fine_only;
        default: return false;
    }
}

}  // namespace

#if defined(__HIPCC__)
// the current device has WMMA int8 and this binary carries its code: gfx11 (RDNA3 / 3.5), gfx12 (RDNA4)
bool built() {
    int dev = 0;
    hipDeviceProp_t p{};
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) return false;
    static const char* const ok[] = {"gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201"};
    for (const char* a : ok)
        if (std::strncmp(p.gcnArchName, a, 7) == 0) return true;
    return false;
}
bool tuned_device() { return false; }   // (no AMD card is measured against MMQ yet: MMQ stays the default)
bool supported(int t) {
    return built() && (t == 16 || t == 17 || t == 18 || t == 22 || t == 39 || t == 19 || t == 29 || t == 20 || t == 42 || t == 23 || t == 12 || t == 14 || t == 8 || t == 40 || t == 2 || t == 3 || t == 6 || t == 13 || t == 11 || t == 10 || t == 21);
}
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
    return built() && (t == 16 || t == 17 || t == 18 || t == 22 || t == 39 || t == 19 || t == 29 || t == 20 || t == 42 || t == 23 || t == 12 || t == 14 || t == 8 || t == 40 || t == 2 || t == 3 || t == 6 || t == 13 || t == 11 || t == 10 || t == 21);
}
#endif

// The default lists: the types measured not slower than MMQ on every model and prompt length, on the GPUs they were
// measured on (tuned_device(): sm_120, the RTX 5090; any other GPU keeps MMQ until it is measured).
// RTX 5090 (worklog/2026-11-16_no-llamacpp.md sections 8 and 11):
//   2026-11-16: IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18, IQ2_S 22, IQ1_S 19, IQ4_NL 20, Q2_0 42 - experts and dense layers.
//   2026-11-17: IQ4_XS 23, Q4_K 12, Q5_K 13 - experts and dense layers (Qwen3.8-27B UD-Q4_K_XL, Ornith-35B,
//   Ornith-122B at 4K-128K); Q6_K 14 - experts only (Ornith-35B's down matrices; a dense layer's Q6_K is 3% of the
//   prompt slower on Qwen3.8-27B UD-Q6_K: its two multiplies a 32 values); Q4_0 2 - dense layers only (Qwen3.8-27B
//   L-Q4_0 +9-11%; no model with Q4_0 experts here).
// In neither: MXFP4 and NVFP4 (MMQ multiplies in FP4 there), IQ1_M (no model here to measure with; MMQ has no IQ1_M:
// the layer takes the FP16 path), Q4_1 / Q5_0, Q3_K, Q2_K, IQ3_S (faster in the kernel bench, no model run yet),
// Q8_0 (Qwen3.8-27B Q8_0 -5%).
static const char* const kDefaultTypes = "16,17,18,22,19,20,42,23,12,13,14";
static const char* const kDefaultDense = "16,17,18,22,19,20,42,23,12,13,2";

bool type_on(int type, bool mmq_built, bool dense) {
    // (the lists once; which one applies follows the current device)
    static const auto parse = [](const std::string& list) {
        std::vector<int> v;
        if (list == "all") {
            for (int t = 0; t < 64; ++t) v.push_back(t);
            return v;
        }
        for (size_t i = 0; i < list.size();) {
            size_t j = list.find(',', i);
            if (j == std::string::npos) j = list.size();
            v.push_back(std::atoi(list.substr(i, j - i).c_str()));
            i = j + 1;
        }
        return v;
    };
    static const bool mmq_only = [] { const char* m = getenv("STRATA_PROMPT_MMQ"); return m != nullptr && std::atoi(m) != 0; }();
    static const char* const env = getenv("STRATA_PROMPT_OWN_TYPES");
    static const std::vector<int> from_env = parse(env != nullptr ? env : ""), all = parse("all"), tuned = parse(kDefaultTypes),
                                  tuned_dense = parse(kDefaultDense);
    if (mmq_only || !supported(type)) return false;
    // (a build without MMQ: every type the own GEMM covers - the other choice there is the FP16 path)
    // (a dense layer's products - one matrix, the tokens as one "expert" - have their own list)
    const std::vector<int>& on = env != nullptr ? from_env : !mmq_built ? all : tuned_device() ? (dense ? tuned_dense : tuned) : from_env;
    return std::find(on.begin(), on.end(), type) != on.end();
}

bool usable(int t, int64_t rows, int64_t cols) {
    // (a chunk of the type's blocks: 256 values, 128 for the 32- and 64-value block types)
    const int64_t chunk = (t == 39 || t == 20 || t == 42 || t == 8 || t == 40 || t == 2 || t == 3 || t == 6) ? 128 : 256;   // (IQ4_XS, Q4_K, Q6_K: 256-value blocks)
    return supported(t) && rows > 0 && cols > 0 && cols % chunk == 0 && rows % 16 == 0;
}

size_t act_bytes(int64_t rows, int64_t cols, bool sums) {
    return (size_t) rows * (size_t) cols + (size_t) rows * (size_t) (cols / 32) * (sums ? 8 : 4);
}
bool needs_sums(int t) { return t == 12 || t == 13 || t == 3; }
bool act_natural_order() {
#if defined(__HIPCC__)
    return true;
#else
    return false;
#endif
}

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
    if (stats_on() && stats().launches > 0) {
        const Stats& t = stats();
        std::fprintf(stderr, "prefill own: %ld launches (%ld coarse, %ld with the tile table inline), %.1f experts and %.0f "
                             "rows a launch, %.0f tiles a launch\n", t.launches, t.coarse, t.inline_,
                     (double) t.experts / t.launches, (double) t.rows / t.launches, (double) t.tiles / t.launches);
    }
}

namespace {
bool& align4() { static bool on = true; return on; }
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
    const bool wide = wide_type(type);   // (launch_t: the wide shape - 8 warps, 64 tokens, one CTA an SM)
#if defined(__HIPCC__)
    const auto tmax_of = [&](int) { return HTC; };
#else
    const auto tmax_of = [&](int tc) { return tc > Cfg<0>::TC ? Cfg<1>::TC : Cfg<0>::TC; };   // (the tile's shape's)
#endif
    // token tiles of tc tokens, and their token groups
    const auto token_tiles = [&](int tc, long& groups) {
        long t = 0;
        groups = 0;
        for (int e = 0; e < s.n; ++e) {
            const int r = s.host_bounds[e + 1] - s.host_bounds[e];
            const int n = tok_tiles(r, tc, tmax_of(tc));
            t += n;
            if (n > 0) groups += (long) (n - 1) * (tc / 8) + (r - (n - 1) * tc + 7) / 8;
        }
        return t;
    };
    const auto count = [&](int tc, bool crs) {
        const int rows_tile = (crs || wide ? Cfg<1>::NW : Cfg<0>::NW) * per_warp;
        long g = 0;
        return token_tiles(tc, g) * ((a.w_rows + rows_tile - 1) / rows_tile);
    };
    int tc = tile_env;
    if (tc == 0) {
        const long sms = sm_count();
        double best = 1e300;
#if defined(__HIPCC__)
        const int c_step = 16, c_max = HTC;   // (one shape; a token group is 16)
#else
        const int c_step = 8, c_max = wide || fine_only_type(type) ? Cfg<2>::TC : Cfg<1>::TC;
#endif
        for (int c = c_step; c <= c_max; c += c_step) {
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
#if defined(__HIPCC__)
    tc = std::clamp((tc + 15) / 16 * 16, 16, HTC);
#endif
    if (wide) tc = std::min(tc, Cfg<2>::TC);
    const bool coarse = tc > Cfg<0>::TC;
    const long total = count(tc, coarse);
    if (total <= 0) return;
    a.tc = tc;
    a.tmax = tmax_of(tc);
    // the tiles' order (own_kernel): row tiles outside when an expert's weights are more than the L2 holds well
    // (64 MB) and well over its activations (4 x) - a dense gate/up of 2 x 17,408 rows: 100 MB, +9%.  Everything else ran
    // the same or faster with the token tiles outside (a dense down, the projections, the experts' sets - measured,
    // tests/moe_own_test --bench with 1 expert of exactly 512 to 4,096 rows).
    {
        int max_rows = 0;
        for (int e = 0; e < s.n; ++e) max_rows = std::max(max_rows, s.host_bounds[e + 1] - s.host_bounds[e]);
        // (bytes a 256 values, about: the choice's threshold only)
        const int b256 = type == 16 ? 66 : type == 17 ? 74 : type == 22 ? 82 : type == 18 ? 98 : type == 19 ? 50 : type == 29 ? 56 :
                         type == 14 ? 210 : type == 8 ? 272 : type == 13 || type == 6 ? 176 : type == 3 ? 160 : type == 11 || type == 21 ? 110 :
                         type == 10 ? 84 : 140;
        const int64_t w_bytes = (int64_t) a.w_rows * (GLU ? 2 : 1) * (a.K / 256) * b256, x_bytes = (int64_t) max_rows * a.K;
        a.order = w_bytes > ((int64_t) 64 << 20) && w_bytes > 4 * x_bytes ? 1 : 0;
    }
    if (s.n <= kInline) {
        a.tiles = nullptr;
        a.itiles[0] = 0;
        for (int e = 0; e < s.n; ++e) a.itiles[e + 1] = a.itiles[e] + tok_tiles(s.host_bounds[e + 1] - s.host_bounds[e], tc, a.tmax);
    } else {
        tiles_kernel<<<1, 256, 0, st>>>(s.bounds, tiles, s.n, tc, a.tmax);
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

void test_align4(bool on) { align4() = on; }

void Context::gate_up(const Set& s, int type, int64_t n_ff, int64_t n_embd, size_t up_off, float limit, void* stream) {
    if (s.n <= 0 || s.rows <= 0) return;
    check_ptr(s, type, 0);
    check_ptr(s, type, up_off);
    KArgs a{};
    a.w = (const uint8_t*) s.w;
    a.stride = s.stride;
    a.up_off = (ptrdiff_t) up_off;
    a.al4 = align4() && (((uintptr_t) s.w | s.stride | up_off) & 3) == 0;
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
    a.al4 = align4() && (((uintptr_t) a.w | s.stride) & 3) == 0;
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

}  // namespace strata::prefill::own
