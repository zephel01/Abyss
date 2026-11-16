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
// One barrier a 128 values.  The tiles of a set are listed exactly (a prefix table of the experts' token tiles, made
// by a small kernel from the device bounds): no empty CTAs, no fix-up pass.
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

// ---------------------------------------------------------------- the activations
// one thread a 32-value block: d = max|x| / 127, q = round(x / d)
__global__ void quant_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids, uint32_t* __restrict__ q,
                             float* __restrict__ d, int nb, int64_t ld, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / nb;
    const int kb = (int) (i - r * nb);
    const float* src = x + (ids != nullptr ? (int64_t) ids[r] : r) * ld + (int64_t) kb * 32;
    float v[32];
    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        v[k] = src[k];
        amax = fmaxf(amax, fabsf(v[k]));
    }
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    uint32_t* dst = q + i * 8;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        uint32_t w = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) w |= (uint32_t) (__float2int_rn(v[4 * k + b] * inv) & 255) << (8 * b);
        dst[k] = w;
    }
    d[i] = amax / 127.0f;
}

#if !defined(__HIPCC__)
// ---------------------------------------------------------------- the tile plan
constexpr int NW = 4;     // warps a CTA
constexpr int TC = 64;    // tokens a tile
constexpr int NT = NW * 32;

// tiles[e] = token tiles of experts < e (n + 1 entries); n <= 1024
__global__ void tiles_kernel(const int32_t* __restrict__ bounds, int32_t* __restrict__ tiles, int n) {
    __shared__ int a[1025], b[1025];
    for (int i = threadIdx.x; i <= n; i += blockDim.x) a[i] = i == 0 ? 0 : (bounds[i] - bounds[i - 1] + TC - 1) / TC;
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
    const int32_t* bounds;
    const int32_t* tiles;
    float* dst;
    int64_t ld;
    int n, row0, w_rows, K;
    float limit;
};

__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

// C[16 x 8] = A[16 x 32] B[32 x 8], int8 -> int32
__device__ __forceinline__ void mma32(int c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    const int z = 0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(z));
}
// C[16 x 8] = A[16 x 16] B[16 x 8]
__device__ __forceinline__ void mma16(int c[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    const int z = 0;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(z));
}

// 16-byte words a slot holds (a chunk from any offset its alignment allows), and a CTA's shared memory
template<int TY> struct Lay {
    static constexpr int WS = (16 - Dec<TY>::AL + Dec<TY>::CB + 15) / 16;
    static constexpr size_t smem = (size_t) (2 * NT * WS + 2 * TC * 8 + 2 * TC) * 16;
};

// One tile with NG groups of 8 tokens.  e: the expert; rt: the row tile; b0, b1: the tile's first activation row
// and the expert's end.
template<int TY, bool GLU, int NG>
__device__ __forceinline__ void tile_body(const KArgs& a, uint4* smem, int e, int rt, int b0, int b1) {
    typedef Dec<TY> D;
    constexpr int WS = Lay<TY>::WS, HPW = D::WCH / 128, SPH = 4;   // (half chunks a weight chunk; steps a half chunk)
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    uint4* wbuf = smem;                       // [2][NT slots][WS]
    uint4* xbuf = wbuf + 2 * NT * WS;         // [2][TC][8]
    uint4* sbuf = xbuf + 2 * TC * 8;          // [2][TC]
    const uint8_t* we = a.w + (size_t) e * a.stride;
    const size_t row_bytes = (size_t) (a.K / D::WCH) * D::CB;
    const int n_h = a.K / 128, n_wc = a.K / D::WCH;

    // slot s (0 .. NT-1) = fragment f (s / (NT / 2)), warp, row in the fragment -> that row's bytes
    const auto slot_row = [&](int s) -> const uint8_t* {
        const int f = s / (NT / 2), ww = (s / 16) % NW, rr = s % 16;
        const int row = GLU ? rt * (NW * 16) + ww * 16 + rr : rt * (NW * 32) + (2 * ww + f) * 16 + rr;
        return we + (GLU && f ? a.up_off : (size_t) 0) + (size_t) min(row, a.w_rows - 1) * row_bytes;
    };
    const uint8_t* my_slot = slot_row(threadIdx.x);
    const auto issue_w = [&](int c) {   // this thread's slot: the 16-byte words around its row's chunk c
        const uintptr_t src = (uintptr_t) (my_slot + (size_t) c * D::CB);
        const uint8_t* p = (const uint8_t*) (src & ~(uintptr_t) 15);
        const int last = (int) (((src & 15) + D::CB - 1) / 16);   // (no word past the chunk: it may be past the pool)
        uint4* dst = wbuf + ((c & 1) * NT + threadIdx.x) * WS;
#pragma unroll
        for (int k = 0; k < WS; ++k) cp16(dst + k, p + 16 * min(k, last));
    };
    const auto issue_x = [&](int h) {
        uint4* xd = xbuf + (h & 1) * TC * 8;
#pragma unroll
        for (int k = 0; k < TC * 8 / NT; ++k) {
            const int i = threadIdx.x + k * NT, tok = i >> 3, word = i & 7;
            if (tok < NG * 8) {
                const int64_t r = min(b0 + tok, b1 - 1) - a.row0;
                cp16(xd + i, a.xq + r * a.K + (int64_t) h * 128 + word * 16);
            }
        }
        if (threadIdx.x < NG * 8) {
            const int64_t r = min(b0 + (int) threadIdx.x, b1 - 1) - a.row0;
            cp16(sbuf + (h & 1) * TC + threadIdx.x, a.xd + r * (a.K / 32) + (int64_t) h * 4);
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
                    const uintptr_t src = (uintptr_t) (rowp[f][h8] + (size_t) c * D::CB);
                    sp[f][h8] = (const uint8_t*) (wbuf + ((c & 1) * NT + soff[f][h8]) * WS) + (src & 15);
                    hd[f][h8] = D::head(sp[f][h8]);
                }
        }
        const uint32_t* xw = (const uint32_t*) (xbuf + (h & 1) * TC * 8);
        const float* xs = (const float*) (sbuf + (h & 1) * TC);
#pragma unroll
        for (int i = 0; i < SPH; ++i) {
            const int ib = (h % HPW) * SPH + i;
            Step st[2][2];
#pragma unroll
            for (int f = 0; f < 2; ++f)
#pragma unroll
                for (int h8 = 0; h8 < 2; ++h8) st[f][h8] = D::step(sp[f][h8], ib, t, hd[f][h8]);
#pragma unroll
            for (int j = 0; j < NG; ++j) {
                const uint32_t bl = xw[(j * 8 + g) * 32 + i * 8 + t], bh = xw[(j * 8 + g) * 32 + i * 8 + 4 + t];
                const float x0 = xs[(j * 8 + 2 * t) * 4 + i], x1 = xs[(j * 8 + 2 * t + 1) * 4 + i];
#pragma unroll
                for (int f = 0; f < 2; ++f) {
                    float v[4];
                    if constexpr (D::SUB == 32) {
                        int cc[4];
                        mma32(cc, st[f][0].lo, st[f][1].lo, st[f][0].hi, st[f][1].hi, bl, bh);
                        v[0] = (float) cc[0] * st[f][0].s;
                        v[1] = (float) cc[1] * st[f][0].s;
                        v[2] = (float) cc[2] * st[f][1].s;
                        v[3] = (float) cc[3] * st[f][1].s;
                    } else {
                        int cl[4], ch[4];
                        mma16(cl, st[f][0].lo, st[f][1].lo, bl);
                        mma16(ch, st[f][0].hi, st[f][1].hi, bh);
                        v[0] = (float) (cl[0] * st[f][0].m0 + ch[0] * st[f][0].m1) * hd[f][0];
                        v[1] = (float) (cl[1] * st[f][0].m0 + ch[1] * st[f][0].m1) * hd[f][0];
                        v[2] = (float) (cl[2] * st[f][1].m0 + ch[2] * st[f][1].m1) * hd[f][1];
                        v[3] = (float) (cl[3] * st[f][1].m0 + ch[3] * st[f][1].m1) * hd[f][1];
                    }
                    acc[f][j][0] = fmaf(v[0], x0, acc[f][j][0]);
                    acc[f][j][1] = fmaf(v[1], x1, acc[f][j][1]);
                    acc[f][j][2] = fmaf(v[2], x0, acc[f][j][2]);
                    acc[f][j][3] = fmaf(v[3], x1, acc[f][j][3]);
                }
            }
        }
    }

    // the write-back: acc[f][j][k] is row (k < 2 ? g : g + 8) of fragment f, token j * 8 + 2 t + (k & 1)
    const int ntok = b1 - b0;
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

template<int TY, bool GLU>
__global__ void __launch_bounds__(NT) own_kernel(const KArgs a) {
    extern __shared__ uint4 smem[];
    const int n_rt = (a.w_rows + (GLU ? NW * 16 : NW * 32) - 1) / (GLU ? NW * 16 : NW * 32);
    const int tt = blockIdx.x / n_rt, rt = blockIdx.x % n_rt;
    int lo = 0, hi = a.n;   // the expert whose token tiles hold tt
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (a.tiles[mid] <= tt) lo = mid;
        else hi = mid;
    }
    const int e = lo, b1 = a.bounds[e + 1], b0 = a.bounds[e] + (tt - a.tiles[e]) * TC;
    const int ntok = b1 - b0;
    if (ntok <= 8) tile_body<TY, GLU, 1>(a, smem, e, rt, b0, b1);
    else if (ntok <= 16) tile_body<TY, GLU, 2>(a, smem, e, rt, b0, b1);
    else if (ntok <= 32) tile_body<TY, GLU, 4>(a, smem, e, rt, b0, b1);
    else tile_body<TY, GLU, 8>(a, smem, e, rt, b0, b1);
}

template<int TY, bool GLU>
void launch_t(const KArgs& a, unsigned grid, cudaStream_t s) {
    static bool once = false;
    if (!once) {
        ck(cudaFuncSetAttribute(own_kernel<TY, GLU>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) Lay<TY>::smem),
           "shared memory size");
        once = true;
    }
    own_kernel<TY, GLU><<<grid, NT, Lay<TY>::smem, s>>>(a);
    ck(cudaGetLastError(), GLU ? "gate_up" : "down");
}

template<bool GLU>
void launch(int type, const KArgs& a, unsigned grid, cudaStream_t s) {
    switch (type) {
        case 16: launch_t<16, GLU>(a, grid, s); break;
        case 17: launch_t<17, GLU>(a, grid, s); break;
        case 18: launch_t<18, GLU>(a, grid, s); break;
        case 39: launch_t<39, GLU>(a, grid, s); break;
        default:
            std::fprintf(stderr, "prefill own: type %d is not covered\n", type);
            std::exit(1);
    }
}

int align_of(int type) { return type == 39 ? 1 : 2; }
#endif   // !__HIPCC__

}  // namespace

#if defined(__HIPCC__)
bool built() { return false; }
bool supported(int) { return false; }
#else
bool built() {
    static const bool ok = [] {
        int dev = 0;
        cudaDeviceProp p{};
        if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
        return p.major >= 8;   // (int8 mma.sync and cp.async)
    }();
    return ok;
}
bool supported(int t) { return built() && (t == 16 || t == 17 || t == 18 || t == 39); }
#endif

bool usable(int t, int64_t rows, int64_t cols) {
    return supported(t) && rows > 0 && cols > 0 && cols % 256 == 0 && rows % 16 == 0;
}

size_t act_bytes(int64_t rows, int64_t cols) { return (size_t) rows * (size_t) cols + (size_t) rows * (size_t) (cols / 32) * 4; }

void quantize(const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld, int64_t rows, void* stream) {
    if (rows <= 0) return;
    const int64_t n = rows * (cols / 32);
    quant_kernel<<<(unsigned) ((n + 127) / 128), 128, 0, (cudaStream_t) stream>>>(
        x, ids, (uint32_t*) xq, (float*) ((uint8_t*) xq + rows * cols), (int) (cols / 32), ld, n);
    ck(cudaGetLastError(), "quantize");
}

Context::Context() {}
Context::~Context() {
    if (tiles_ != nullptr) cudaFree(tiles_);
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
// the set's tile table on the device; -> the number of token tiles
int plan(const Set& s, int32_t*& tiles, int& cap, cudaStream_t st) {
    if (s.n > 1024) {
        std::fprintf(stderr, "prefill own: a set of %d experts (at most 1024)\n", s.n);
        std::exit(1);
    }
    if (tiles == nullptr) {
        cap = 1025;
        ck(cudaMalloc(&tiles, (size_t) cap * sizeof(int32_t)), "tile table");
    }
    int total = 0;
    for (int e = 0; e < s.n; ++e) total += (s.host_bounds[e + 1] - s.host_bounds[e] + TC - 1) / TC;
    if (total > 0) {
        tiles_kernel<<<1, 256, 0, st>>>(s.bounds, tiles, s.n);
        ck(cudaGetLastError(), "tiles");
    }
    return total;
}
void check_ptr(const Set& s, int type, size_t off) {
    if ((((uintptr_t) s.w + off) | s.stride) % align_of(type) != 0 || (uintptr_t) s.xq % 16 != 0) {
        std::fprintf(stderr, "prefill own: a matrix of type %d or its activations are not aligned\n", type);
        std::exit(1);
    }
}
}  // namespace

void Context::gate_up(const Set& s, int type, int64_t n_ff, int64_t n_embd, size_t up_off, float limit, void* stream) {
    if (s.n <= 0 || s.rows <= 0) return;
    const cudaStream_t st = (cudaStream_t) stream;
    check_ptr(s, type, 0);
    check_ptr(s, type, up_off);
    const int total = plan(s, tiles_, cap_, st);
    if (total <= 0) return;
    KArgs a{};
    a.w = (const uint8_t*) s.w;
    a.stride = s.stride;
    a.up_off = up_off;
    a.xq = (const uint8_t*) s.xq;
    a.xd = (const float*) ((const uint8_t*) s.xq + s.rows * n_embd);
    a.bounds = s.bounds;
    a.tiles = tiles_;
    a.dst = s.dst;
    a.ld = s.ld;
    a.n = s.n;
    a.row0 = s.row0;
    a.w_rows = (int) n_ff;
    a.K = (int) n_embd;
    a.limit = limit;
    const int n_rt = (int) ((n_ff + NW * 16 - 1) / (NW * 16));
    launch<true>(type, a, (unsigned) total * (unsigned) n_rt, st);
}

void Context::down(const Set& s, int type, int64_t n_embd, int64_t n_ff, size_t down_off, void* stream) {
    if (s.n <= 0 || s.rows <= 0) return;
    const cudaStream_t st = (cudaStream_t) stream;
    check_ptr(s, type, down_off);
    const int total = plan(s, tiles_, cap_, st);
    if (total <= 0) return;
    KArgs a{};
    a.w = (const uint8_t*) s.w + down_off;
    a.stride = s.stride;
    a.xq = (const uint8_t*) s.xq;
    a.xd = (const float*) ((const uint8_t*) s.xq + s.rows * n_ff);
    a.bounds = s.bounds;
    a.tiles = tiles_;
    a.dst = s.dst;
    a.ld = s.ld;
    a.n = s.n;
    a.row0 = s.row0;
    a.w_rows = (int) n_embd;
    a.K = (int) n_ff;
    a.limit = 0.0f;
    const int n_rt = (int) ((n_embd + NW * 32 - 1) / (NW * 32));
    launch<false>(type, a, (unsigned) total * (unsigned) n_rt, st);
}
#endif

}  // namespace strata::prefill::own
