// src/prefill/moe_fp4.cuh - the opt-in FP4-activation products of NVFP4 expert weights (STRATA_INK_FP4ACT), CUDA
// sm_120a only: the block-scaled FP4 tensor-core MMA (mma.sync m16n8k64 kind::mxf4nvf4, a UE4M3 scale a 16 values on
// both sides).  Included by moe_own.cu (the NVIDIA backend, inside strata::prefill::own); a HIP build never sees it.
//
// The activations: a row's values over one float scale (amax / (6 x 448)), then each 16-value block as E2M1 codes
// under its own UE4M3 scale (round to nearest, ties to the even code) - one term, about 4 bits.  Two terms: the
// first term's residual quantized the same way (same row scale, its own block scales): x ~ q1 + q2, about 8 bits; both
// terms go through the tensor cores into the same FP32 accumulator.  Lossy (not the exact path's sums): opt-in only.
//
// The weights are read as the GGUF holds them (block_nvfp4: 4 UE4M3 scales, then 32 bytes of codes, a 64-value block):
// inside a 16-value block the codes are in the file's order (byte j: values j and j + 8), and the activations are
// written in that same order, so the MMA's k positions pair the same two values on both sides (the activations keep a
// 64-value block's eight 4-byte words as 0, 4, 1, 5, 2, 6, 3, 7 - storage only: a thread's two B registers, words c and
// c + 4, are one 8-byte load).  Each gate/up/down product: one launch a set of experts, a CTA 128 output rows x 128
// tokens of one expert, cp.async stages of 128 values, F32 sums.

namespace fp4 {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200) && defined(__CUDA_ARCH_FEAT_SM120_ALL)
#define STRATA_FP4_MMA 1
#else
#define STRATA_FP4_MMA 0
#endif

constexpr float kRowMax = 6.0f * 448.0f;   // (the largest E2M1 code times the largest finite UE4M3 scale)

__device__ __forceinline__ uint32_t e2m1_code(float v) {   // round to nearest, ties to the even code; saturates at 6
    const float a = fabsf(v);
    const uint32_t c = (uint32_t) (a > 0.25f) + (uint32_t) (a >= 0.75f) + (uint32_t) (a > 1.25f) + (uint32_t) (a >= 1.75f) +
                       (uint32_t) (a > 2.5f) + (uint32_t) (a >= 3.5f) + (uint32_t) (a > 5.0f);
    return c | (v < 0.0f && c != 0 ? 8u : 0u);
}
__device__ __forceinline__ float e2m1_value(uint32_t c) {
    const uint32_t m = c & 7u;
    const float a = m < 4u ? 0.5f * (float) m : m == 7u ? 6.0f : (float) (m - 2u);
    return (c & 8u) ? -a : a;
}
__device__ __forceinline__ uint32_t ue4m3_code(float v) {   // v >= 0: round to nearest, saturating at 448
    return (uint32_t) __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ float ue4m3_value(uint32_t c) {
    __nv_fp8_e4m3 f;
    f.__x = (__nv_fp8_storage_t) c;
    return (float) f;
}

// two values -> one byte of E2M1 codes (lo in bits 0-3), round to nearest even, saturating; and back
__device__ __forceinline__ uint32_t e2m1x2(float lo, float hi) {
#if STRATA_FP4_MMA
    uint16_t r;
    asm("{\n.reg .b8 t;\ncvt.rn.satfinite.e2m1x2.f32 t, %2, %1;\nmov.b16 %0, {t, 0};\n}" : "=h"(r) : "f"(lo), "f"(hi));
    return r;
#else
    return e2m1_code(lo) | (e2m1_code(hi) << 4);
#endif
}
__device__ __forceinline__ float2 e2m1x2_value(uint32_t b) {
#if STRATA_FP4_MMA
    uint32_t h;
    const uint16_t in = (uint16_t) b;
    asm("{\n.reg .b8 t, u;\nmov.b16 {t, u}, %1;\ncvt.rn.f16x2.e2m1x2 %0, t;\n}" : "=r"(h) : "h"(in));
    return __half22float2(*(const __half2*) &h);
#else
    return make_float2(e2m1_value(b & 15u), e2m1_value(b >> 4));
#endif
}

__device__ __forceinline__ float block_err(const float v[16], float s, float is) {
    float e = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float2 d = e2m1x2_value(e2m1x2(v[j] * is, v[j + 8] * is));
        const float a = fmaf(-d.x, s, v[j]), b = fmaf(-d.y, s, v[j + 8]);
        e = fmaf(a, a, fmaf(b, b, e));
    }
    return e;
}
// One 16-value block v[16] (already over the row scale) -> 8 bytes of codes (byte j: values j, j + 8) and its scale's
// code (amax / 6, round to nearest; `search`: the code of the least squared error among it and its two neighbours);
// v becomes the residual (v minus what the codes hold).
__device__ __forceinline__ void quant_block(float v[16], uint2& q, uint32_t& sc, bool search) {
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) m = fmaxf(m, fabsf(v[i]));
    sc = ue4m3_code(m / 6.0f);
    if (search && sc > 1u && sc < 0x7Eu) {
        uint32_t best = sc;
        float s = ue4m3_value(sc), be = block_err(v, s, 1.0f / s);
#pragma unroll
        for (int d = -1; d <= 1; d += 2) {
            s = ue4m3_value(sc + d);
            const float e = block_err(v, s, 1.0f / s);
            if (e < be) {
                be = e;
                best = sc + d;
            }
        }
        sc = best;
    }
    const float s = ue4m3_value(sc), is = s > 0.0f ? 1.0f / s : 0.0f;
    uint32_t w[2] = {0u, 0u};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t b = e2m1x2(v[j] * is, v[j + 8] * is);
        w[j >> 2] |= b << (8 * (j & 3));
        const float2 d = e2m1x2_value(b);
        v[j] = fmaf(-d.x, s, v[j]);
        v[j + 8] = fmaf(-d.y, s, v[j + 8]);
    }
    q = make_uint2(w[0], w[1]);
}

// one CTA a row (128 threads), cols / 16 <= 2 x 128 blocks: row i of the output is row ids[i] of x (i when ids is null)
__global__ void __launch_bounds__(128) quant_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids,
                                                     uint8_t* __restrict__ xq, float* __restrict__ rs, int cols, int64_t ld,
                                                     int64_t rows, int terms, bool search, const int32_t* __restrict__ r_ids,
                                                     const int32_t* __restrict__ r_pos, int k_routes, const float* __restrict__ nv,
                                                     int row_base) {
    const int64_t r = blockIdx.x;
    const int nb = cols / 16, t = threadIdx.x;
    const float* xr = x + (ids != nullptr ? (int64_t) ids[r] : r) * ld;
    float v[2][16];
    float m = 0.0f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int b = t + 128 * h;
        if (b < nb) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 f = ((const float4*) (xr + 16 * b))[i];
                v[h][4 * i] = f.x;
                v[h][4 * i + 1] = f.y;
                v[h][4 * i + 2] = f.z;
                v[h][4 * i + 3] = f.w;
            }
#pragma unroll
            for (int i = 0; i < 16; ++i) m = fmaxf(m, fabsf(v[h][i]));
        }
    }
    __shared__ float red[4];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((t & 31) == 0) red[t >> 5] = m;
    __syncthreads();
    m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    const float scale = m > 0.0f ? m / kRowMax : 1.0f, inv = 1.0f / scale;
    if (t == 0) {
        // (nv: the row's expert's NVFP4 gate/up scale into its row scale - the route of token ids[r] whose sorted row is
        // row_base + r)
        float x = scale;
        if (nv != nullptr) {
            const int64_t tok = ids[r];
            for (int k = 0; k < k_routes; ++k)
                if (r_pos[tok * k_routes + k] == row_base + (int) r) x *= nv[2 * r_ids[tok * k_routes + k]];
        }
        rs[r] = x;
    }
    const int64_t qplane = rows * (int64_t) (cols / 2), splane = rows * (int64_t) nb;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int b = t + 128 * h;
        if (b >= nb) continue;
#pragma unroll
        for (int i = 0; i < 16; ++i) v[h][i] *= inv;
        for (int k = 0; k < terms; ++k) {
            uint2 q;
            uint32_t sc;
            quant_block(v[h], q, sc, search && k == terms - 1);
            // (a 64-value block's 8 words as (0, 4, 1, 5, 2, 6, 3, 7): a thread's two B registers are one 8-byte load)
            uint32_t* qb = (uint32_t*) (xq + k * qplane + r * (cols / 2) + 32 * (b >> 2));
            const int w0 = 2 * (b & 3);
            qb[2 * (w0 & 3) + (w0 >> 2)] = q.x;
            qb[2 * ((w0 + 1) & 3) + ((w0 + 1) >> 2)] = q.y;
            xq[terms * qplane + k * splane + r * nb + b] = (uint8_t) sc;
        }
    }
}

constexpr int kTM = 128;      // a CTA's output rows (4 warps x 32)
// A stage's rows in shared memory:
//   weights:     the file's 72 bytes (two blocks: scale word, 8 code words), in 80 (the 4-byte fragment loads hit 32 banks)
//   activations: the two blocks' code words (in the order above), then their scale words, in 96 (24 words: the 8-byte
//                fragment loads of a half warp hit 32 banks)
constexpr int kWRow = 80;
constexpr int kXRow = 96;
constexpr int kMaxSet = 256;  // experts a launch (the tile prefix in the arguments)

struct Args {
    const uint8_t* w;   // expert 0's matrix
    size_t stride;      // bytes between two experts' matrices
    const int32_t* bounds;
    int n, row0;
    const uint8_t* xq;  // the activations' planes (quant_kernel's), rows rows
    const float* rs;
    int64_t rows;
    int K, N;
    float* dst;
    int64_t ld;
    size_t up_off;      // GLU: the up matrix's bytes past the gate's
    float limit;        // GLU: the SwiGLU's clamp (negative: inkling's SITU)
    int32_t tiles[kMaxSet + 1];   // token tiles before expert e
};

template <int TERMS, int TN> struct Smem {
    static constexpr int W = kTM * kWRow, X = TERMS * TN * kXRow;
    static constexpr int stage = W + X;
};

__device__ __forceinline__ void cp8(void* smem, const void* gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp16g(void* smem, const void* gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void mma_fp4(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
#if STRATA_FP4_MMA
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
#else
    (void) c; (void) a; (void) b0; (void) b1; (void) sa; (void) sb;
    __trap();
#endif
}

__global__ void has_kernel(int* o) { *o = STRATA_FP4_MMA; }

// dst[row][n] = sum_k W_e[n][k] x[row][k] over the set's experts: grid (N / 128, the set's token tiles), 256 threads
// GLU: gate rows [64 x, 64 x + 64) in the tile's first 64 rows and the same up rows in its last 64, h = act(gate, up)
// into dst row r's columns [64 x, 64 x + 64) (glm_batch.cu swiglu_rows' expression); grid (N / 64, tiles) then
template <int TERMS, int TN, int NST, bool GLU> __global__ void __launch_bounds__(256) gemm_kernel(const __grid_constant__ Args a) {
#if STRATA_FP4_MMA
    extern __shared__ __align__(16) uint8_t sm[];
    constexpr int NT = TN / 16;   // n tiles of 8 tokens a warp (2 warps across the tokens)
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp & 3, wn = warp >> 2;
    const int g = blockIdx.y, n0 = blockIdx.x * (GLU ? kTM / 2 : kTM);
    // the tile's expert: the last e with tiles[e] <= g
    int lo = 0, hi = a.n;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (a.tiles[mid] <= g) lo = mid; else hi = mid;
    }
    const int e = lo;
    const int r_lo = a.bounds[e] + (g - a.tiles[e]) * TN, r_end = a.bounds[e + 1];
    const int r_hi = min(r_lo + TN, r_end);
    const int K = a.K, nk = K / 128;
    const int64_t wrow = (int64_t) (K / 64) * 36;
    const uint8_t* W = a.w + (size_t) e * a.stride + (int64_t) n0 * wrow;
    const int64_t qplane = a.rows * (int64_t) (K / 2), splane = a.rows * (int64_t) (K / 16);
    const uint8_t* XQ = a.xq;
    const uint8_t* XS = a.xq + TERMS * qplane;
    using L = Smem<TERMS, TN>;
    const auto load = [&](int kb, int slot) {
        uint8_t* st = sm + slot * L::stage;
        // weights: 128 rows x 9 pairs of words
        for (int q = tid; q < kTM * 9; q += 256) {
            const int row = q / 9, wd = q - row * 9;
            const uint8_t* src = GLU && row >= kTM / 2 ? W + a.up_off + (row - kTM / 2) * wrow : W + row * wrow;
            cp8(st + row * kWRow + 8 * wd, src + kb * 72 + 8 * wd);
        }
        // the activations' codes: TN rows x 4 chunks of 16 bytes a term; their scales: 8 bytes a row a term
#pragma unroll
        for (int k = 0; k < TERMS; ++k) {
#pragma unroll
            for (int i = 0; i < (TN * 4) / 256; ++i) {
                const int q = tid + 256 * i, row = q >> 2, c = q & 3;
                const int r = min(r_lo + row, r_hi - 1) - a.row0;
                cp16g(st + L::W + (k * TN + row) * kXRow + 16 * c, XQ + k * qplane + (int64_t) r * (K / 2) + kb * 64 + 16 * c);
            }
            if (tid < TN) {
                const int r = min(r_lo + tid, r_hi - 1) - a.row0;
                cp8(st + L::W + (k * TN + tid) * kXRow + 64, XS + k * splane + (int64_t) r * (K / 16) + kb * 8);
            }
        }
    };
    float acc[2][NT][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.0f;
#pragma unroll
    for (int s = 0; s < NST - 1; ++s) {
        if (s < nk) load(s, s);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int kb = 0; kb < nk; ++kb) {
        cp_wait<NST - 2>();
        __syncthreads();
        if (kb + NST - 1 < nk) load(kb + NST - 1, (kb + NST - 1) % NST);
        asm volatile("cp.async.commit_group;\n" ::);
        const uint8_t* st = sm + (kb % NST) * L::stage;
        const uint32_t* Ws = (const uint32_t*) st;
        const uint32_t* Xs = (const uint32_t*) (st + L::W);
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            uint32_t af[2][4], sa[2];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int R = 32 * wm + 16 * mt;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    af[mt][i] = Ws[(R + (lane >> 2) + 8 * (i & 1)) * (kWRow / 4) + 9 * b + 1 + (lane & 3) + 4 * (i >> 1)];
                sa[mt] = Ws[(R + (lane >> 2) + 8 * (lane & 1)) * (kWRow / 4) + 9 * b];
            }
#pragma unroll
            for (int k = 0; k < TERMS; ++k)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const int tok = k * TN + wn * (TN / 2) + 8 * nt + (lane >> 2);
                    const uint2 bv = *(const uint2*) (Xs + tok * (kXRow / 4) + 8 * b + 2 * (lane & 3));
                    const uint32_t sb = Xs[tok * (kXRow / 4) + 16 + b];
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) mma_fp4(acc[mt][nt], af[mt], bv.x, bv.y, sa[mt], sb);
                }
        }
    }
    cp_wait<0>();
    if constexpr (GLU) {
        // the up warps (wm 2, 3) hand their sums to the gate warps of the same rows (wm 0, 1) through shared memory,
        // in their fragments' order
        __syncthreads();
        float* xch = (float*) sm;
        const int slot = (wm & 1) * 2 + wn;
        if (wm >= 2) {
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                    for (int c = 0; c < 4; ++c) xch[((slot * 2 + mt) * NT * 4 + nt * 4 + c) * 32 + lane] = acc[mt][nt][c];
        }
        __syncthreads();
        if (wm >= 2) return;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int row = r_lo + wn * (TN / 2) + 8 * nt + 2 * (lane & 3) + j;
                if (row >= r_hi) continue;
                const float s = a.rs[row - a.row0];
                float* d = a.dst + (int64_t) (row - a.row0) * a.ld + n0 + 32 * wm + (lane >> 2);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        const int c = 2 * hh + j;
                        const float gs = acc[mt][nt][c] * s, us = xch[((slot * 2 + mt) * NT * 4 + nt * 4 + c) * 32 + lane] * s;
                        float h;
                        if (a.limit < 0.0f) {
                            h = (4.0f * tanhf(gs * 0.25f)) * (1.0f / (1.0f + expf(-gs))) * (25.0f * tanhf(us * 0.04f));
                        } else {
                            const float gg = fminf(gs, a.limit), uu = fminf(fmaxf(us, -a.limit), a.limit);
                            h = (gg / (1.0f + expf(-gg))) * uu;
                        }
                        d[16 * mt + 8 * hh] = h;
                    }
            }
        return;
    }
    // write-back: d0, d1 = out row lane/4, tokens 2 (lane % 4) + {0, 1}; d2, d3 = out row + 8
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = r_lo + wn * (TN / 2) + 8 * nt + 2 * (lane & 3) + j;
            if (row >= r_hi) continue;
            const float s = a.rs[row - a.row0];
            float* d = a.dst + (int64_t) (row - a.row0) * a.ld + n0 + 32 * wm + (lane >> 2);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                d[16 * mt] = acc[mt][nt][j] * s;
                d[16 * mt + 8] = acc[mt][nt][2 + j] * s;
            }
        }
#else
    (void) a;
    __trap();
#endif
}

}  // namespace fp4
