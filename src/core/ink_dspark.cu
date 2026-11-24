// src/core/ink_dspark.cu - Inkling 2 Mini's DSpark drafter (unsloth/Inkling-2-Mini-GGUF dspark/, arch dflash, 3 dense
// layers, block 8, an RNN Markov head of rank 256), loaded with STRATA_INK_DSPARK=<the GGUF> on an inkling target.  It
// proposes up to 7 tokens of a verify window (inkling_fast_path.cu's ink_window).  The arithmetic
// (Abyss-tools worklog/2026-11-22_inkling-dspark-spec.md, written on Abyss's own structure):
//   * the taps: the residual entering layer applications 5, 23 and 41 (before their depth mix and controllers) and
//     the head's row (the output depth mix after output_norm, visit 48) of every position;
//   * mix(r, norm) = rms(sum_j softmax(dspark_fuse[r])_j tap_j) * norm: rows 0..2 give drafter layer i's context
//     feature c_i (blk.i.dflare_norm), row 3 the Markov head's context (dspark_norm);
//   * a layer's context rows: K = rope_p(rms_head(attn_k_ctx c_i) * attn_k_ctx_norm), V = attn_v_ctx c_i (FP16 rings:
//     layers 0 / 2 attend the last 1024 positions, layer 1 all of them - its ring holds STRATA_INK_DSPARK_CTX
//     positions, default 32768; past that it sees the last ones: a draft is a proposal, the window decides);
//   * the block: row 0 the anchor (the pending token's target embedding, rms * token_embd_norm), rows 1..7 mask_embd,
//     at positions pos .. pos + 7, through 3 pre-norm dense layers (per-head q / k norms, NEOX RoPE over 128 dims base
//     1e6, 24 q heads over 8 kv heads, no mask over the context and the whole block, SwiGLU 9216), output_norm -> y;
//   * the head: row s's logits = W_lm (y_s + hidden_out.b) + F latent_s, F = W_lm hidden_out folded once at load
//     (FP16 [vocab][256]); latent_s from the GRU-like step z = joint_proj [state | markov_w1[prev] | y_s | f_rnn] + b,
//     state = sigmoid(z0) state + (1 - sigmoid(z0)) tanh(z1), latent = tanh(z2), state 0 and prev the anchor at the
//     start, f_rnn = mix(3) at pos - 1.  The pick is the argmax (ties to the lower id, NaN as -inf) over each row's
//     candidates: the top 16 of each of 64 vocabulary chunks of W_lm (y_s + b) (1024 a row), scored exactly;
//     its confidence the softmax over the candidates' softcapped scores.
// The context follows the sequence as DFlash 2's does (dflash2.cu): the token path, the verify window (the kept
// columns, at spec_rewind) and the prompt path leave their taps (ink_tap accumulates the four mixes), inkd_commit
// turns them into ring rows.
#include "glm_fast_state.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/pdl.hpp"

#include "strata/qtypes.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;

namespace {
constexpr size_t kQ8 = 36;   // bytes of a q8_1 block (32 values)
constexpr int kL = 3, kNB = 8, kND = kNB - 1;   // drafter layers; block rows; drafts at most
constexpr int kHD = 128, kNH = 24, kNKV = 8, kGq = kNH / kNKV, kKVD = kNKV * kHD, kQD = kNH * kHD, kRank = 256;
constexpr int kZ = 3 * kRank;                    // joint_proj's rows: gate, candidate, latent
constexpr int kLocalWin = 1024, kLocalRing = 2048;
constexpr int kTail = 1024;                      // a prompt chunk's last rows kept for the local layers
constexpr int kCh = 256;                         // a commit's rows a pass
constexpr int kTkCh = 64, kTopK = 16, kNC = kTkCh * kTopK;   // the candidates a row
constexpr int kSub = 32;                         // the attention's keys a sub-chunk
constexpr int kMaxSplit = 192;                   // the attention's key ranges a kv head at most
constexpr int kQ8_0 = 8, kF32 = 0, kBF16 = 30;
static_assert(kGq == 3, "the attention's layout: 3 q heads a kv head");

struct Alpha {   // softmax(dspark_fuse[r]) - rows 0..2 the layers' context, 3 the Markov head's
    float a[4][4];
};

struct IkLayer {
    WSlot q, k, v, o, gate, up, down, kc, vc;
    float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *kc_norm = nullptr,
          *dflare = nullptr;
    __half *rk = nullptr, *rv = nullptr;   // [ring][kKVD]
    int ring = 0, win = 0;                 // rows; positions a draft attends (local 1024, global the ring)
    int64_t lo = 0;                        // the first context position these rings hold contiguously
};

struct IkState {
    std::unique_ptr<strata::GgufFile> gguf;
    IkLayer L[kL];
    Alpha al{};
    float *out_norm = nullptr, *mk_norm = nullptr, *mask = nullptr, *ho_b = nullptr, *jb = nullptr;
    uint16_t *w1 = nullptr, *wsta = nullptr, *wdt = nullptr;   // BF16 bits: markov_w1 [V][256]; joint_proj's halves
    __half* F = nullptr;                                             // the fold [V][256]
    double* inv = nullptr;                                           // the RoPE frequencies [64]
    int ff = 9216, n_max = kND, V = 0, ctx_g = 32768;
    float eps = 1e-6f, softcap = 30.0f;
    // the taps' accumulators: acc_g [rows_g][E] (layer 1's), acc_l [kTail][3][E] (layers 0, 2, the Markov head's)
    float *acc_g = nullptr, *acc_l = nullptr;
    int rows_g = 0;                  // acc_g's rows (grown at the first long prompt chunk)
    uint64_t acc_gen = 0;            // (acc_g reallocated: the token graphs that write it are captured again)
    int tap_n = 0, tap_ng = 0, tap_nl = 0;   // the rows being accumulated: all, layer 1's (last), the others' (last)
    bool tap_prompt = false;
    // the deferred single tokens' rows (STRATA_INK_DSPARK_DEFER): positions pend_p0 .. pend_p0 + pend_n - 1 in the
    // accumulators' rows 0 .. pend_n - 1; roff_d the next token's row (its taps read it)
    bool defer = false;
    int64_t pend_p0 = 0;
    int pend_n = 0;
    int64_t* roff_d = nullptr;       // (roff_d[0]: the row, launches one by one; roff_d[1]: pend_p0, the graphs' base)
    const int64_t *tap_pa = nullptr, *tap_pb = nullptr;   // the token's taps' row: *tap_pa - *tap_pb (tap_pb null: 0)
    uint64_t flushes = 0;
    // a commit's pass
    float *cf = nullptr, *kk = nullptr, *vv = nullptr;
    void* cq = nullptr;
    float* frnn = nullptr;           // f_rnn of the context's last position
    // the draft
    float *x = nullptr, *xn = nullptr, *q = nullptr, *kb = nullptr, *vb = nullptr, *o = nullptr, *a = nullptr,
          *gt = nullptr, *up = nullptr, *hn = nullptr, *xz = nullptr, *zs = nullptr, *logits = nullptr, *tk_v = nullptr,
          *pm = nullptr, *pl = nullptr, *pacc = nullptr, *res_conf_d = nullptr;
    void *xq = nullptr, *oq = nullptr, *gq = nullptr, *hq = nullptr;
    int *tk_i = nullptr, *res_d = nullptr;
    int *y_h = nullptr, *y_d = nullptr, *res_h = nullptr;
    float *res_conf_h = nullptr, *emb_h = nullptr;
    std::vector<void*> allocs;
    size_t bytes = 0;
    FILE* dump = nullptr;
    int64_t ctx_end = 0, win_p0 = -1, frnn_pos = -1;
    int win_T = 0;
    uint64_t drafts = 0, commits = 0, prompt_rows = 0, prompt_cut = 0;
    double ms_draft = 0.0;
    // STRATA_INK_DSPARK_PROF=1: the draft's phases on the device (events)
    static constexpr int kPh = 6;
    bool prof = false;
    cudaEvent_t ev[kPh + 1] = {};
    double ms_ph[kPh] = {};
};

// ------------------------------------------------------------------------------------------------ kernels
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// a block's sum (blockDim a multiple of 32); every thread gets it
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float r = lane < nw ? sh[lane] : 0.0f;
    return warp_sum(r);
}

__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

// a q8_1 block (glm_fast's layout: the scale and the sum as FP16, 32 int8) from 32 values held one per lane
struct Q81 {
    __half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81) == kQ8, "q8_1 block");
__device__ __forceinline__ void q81_store_warp(float v, Q81* dst, int lane) {
    const float amax = warp_max(fabsf(v));
    const float sum = warp_sum(v);
    const float d = fminf(amax / 127.0f, 65504.0f);
    const int8_t q = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(v / d)));
    dst->qs[lane] = q;
    if (lane == 0) dst->ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}

// y[row] = rms(x[row]) * w (rows ldx / ldy floats apart; 256 threads, E a multiple of 256 up to 256 kRmsPer), the
// row as q8_1 blocks into yq (null: none)
constexpr int kRmsPer = 16;
__global__ void __launch_bounds__(256) k_rms(const float* __restrict__ x, int64_t ldx, const float* __restrict__ w,
                                             float eps, int E, float* __restrict__ y, int64_t ldy, Q81* __restrict__ yq) {
    __shared__ float sh[32];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float xv[kRmsPer];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < kRmsPer; ++k) {
        const int i = threadIdx.x + k * 256;
        xv[k] = i < E ? xr[i] : 0.0f;
        ss += xv[k] * xv[k];
    }
    ss = block_sum(ss, sh);
    const float r = rsqrtf(ss / (float) E + eps);
    float* yr = y + (size_t) blockIdx.x * ldy;
    Q81* qr = yq != nullptr ? yq + (size_t) blockIdx.x * (E / 32) : nullptr;
#pragma unroll
    for (int k = 0; k < kRmsPer; ++k) {
        const int i = threadIdx.x + k * 256;
        if (i < E) {
            const float v = xv[k] * r * w[i];
            yr[i] = v;
            if (qr != nullptr) q81_store_warp(v, qr + i / 32, threadIdx.x & 31);
        }
    }
}

// a tap's rows into the mixes' accumulators: row r of n (x rows ld apart; normed with nw first when not null) adds
// alpha[i][j] x to acc_g's row r - (n - ng) (when r >= n - ng) and to acc_l's slots 0 / 1 / 2 (rows 0, 2, 3 of the
// mix) of row r - (n - nl); j == 0 sets them
__global__ void __launch_bounds__(256) k_tap(const float* __restrict__ x, int64_t ld, int n, const float* __restrict__ nw,
                                             float eps, int E, int j, int ng, int nl, float* __restrict__ acc_g,
                                             float* __restrict__ acc_l, Alpha al, const int64_t* __restrict__ pa,
                                             const int64_t* __restrict__ pb) {
    STRATA_PDL_ENTRY();   // (the token / window graphs' launches: PDL, as the kernels around it)
    __shared__ float sh[32];
    const int r = blockIdx.x;
    const int ro = pa != nullptr ? (int) (*pa - (pb != nullptr ? *pb : 0)) : 0;   // (a deferred token: its row past the others)
    const float* xr = x + (size_t) r * ld;
    float sc = 1.0f;
    if (nw != nullptr) {
        float ss = 0.0f;
        for (int i = threadIdx.x; i < E; i += 256) ss += xr[i] * xr[i];
        ss = block_sum(ss, sh);
        sc = rsqrtf(ss / (float) E + eps);
    }
    const int rg = r - (n - ng) + ro, rl = r - (n - nl) + ro;
    for (int i = threadIdx.x; i < E; i += 256) {
        const float v = nw != nullptr ? xr[i] * sc * nw[i] : xr[i];
        if (rg >= 0) {
            float* d = acc_g + (size_t) rg * E + i;
            *d = (j == 0 ? 0.0f : *d) + al.a[1][j] * v;
        }
        if (rl >= 0) {
            float* d = acc_l + (size_t) rl * 3 * E + i;
            d[0] = (j == 0 ? 0.0f : d[0]) + al.a[0][j] * v;
            d[E] = (j == 0 ? 0.0f : d[E]) + al.a[2][j] * v;
            d[2 * (size_t) E] = (j == 0 ? 0.0f : d[2 * (size_t) E]) + al.a[3][j] * v;
        }
    }
}

__global__ void k_set_i64(int64_t* d, int64_t v) {
    STRATA_PDL_ENTRY();
    *d = v;
}

// per head (blockIdx.x) of row blockIdx.y, 128 threads: rms * w (w null: none), NEOX RoPE at position pos0 + row
// (inv null: none); in place (ring null) or as FP16 into ring row (pos0 + row) % ring_n
__global__ void __launch_bounds__(kHD) k_norm_rope(float* __restrict__ x, int ld, const float* __restrict__ w, float eps,
                                                   long long pos0, const double* __restrict__ inv,
                                                   __half* __restrict__ ring, int ring_n, int ring_ld) {
    __shared__ float sh[32];
    __shared__ float buf[kHD];
    const int d = threadIdx.x, h = blockIdx.x, row = blockIdx.y;
    float* xr = x + (size_t) row * ld + (size_t) h * kHD;
    float v = xr[d];
    if (w != nullptr) {
        const float ss = block_sum(v * v, sh);
        v = v * rsqrtf(ss / (float) kHD + eps) * w[d];
    }
    const long long pos = pos0 + row;
    if (inv != nullptr) {
        buf[d] = v;
        __syncthreads();
        const int i = d & (kHD / 2 - 1);
        double ang = (double) pos * inv[i];
        ang -= 6.283185307179586 * floor(ang / 6.283185307179586);
        float sn, cs;
        sincosf((float) ang, &sn, &cs);
        const float a = buf[i], b = buf[i + kHD / 2];
        v = d < kHD / 2 ? a * cs - b * sn : a * sn + b * cs;
    }
    if (ring != nullptr) ring[(size_t) (pos % ring_n) * ring_ld + (size_t) h * kHD + d] = __float2half(v);
    else xr[d] = v;
}

// the attention's partial sums: block (split, kv head) over keys [split * per, ..) of the n_ctx context rows (ring
// rows (lo + j) % ring_n) then the block's kNB rows (kb / vb), for the block's 8 rows x 3 heads of that kv head;
// online softmax over sub-chunks of kSub keys -> pm / pl / pacc[split][kvh][24][(128)]
constexpr int kQn = kNB * kGq;   // the queries a kv head
__global__ void __launch_bounds__(256) k_attn_part(const float* __restrict__ q, const __half* __restrict__ rk,
                                                   const __half* __restrict__ rv, int ring_n, long long lo, int n_ctx,
                                                   const float* __restrict__ kb, const float* __restrict__ vb, int per,
                                                   float scale, float* __restrict__ pm, float* __restrict__ pl,
                                                   float* __restrict__ pacc) {
    __shared__ float qs[kQn][kHD];
    __shared__ float ks[kSub][kHD + 1];
    __shared__ float vs[kSub][kHD];
    __shared__ float sc[kQn][kSub];
    __shared__ float mrow[kQn], lrow[kQn], arow[kQn];
    const int sp = blockIdx.x, kvh = blockIdx.y, t = threadIdx.x;
    const int S = n_ctx + kNB, j0 = sp * per, j1 = min(S, j0 + per);
    for (int i = t; i < kQn * kHD; i += 256) {
        const int qi = i / kHD, d = i % kHD, r = qi / kGq, g = qi % kGq;
        qs[qi][d] = q[(size_t) r * kQD + (size_t) (kvh * kGq + g) * kHD + d];
    }
    if (t < kQn) {
        mrow[t] = -INFINITY;
        lrow[t] = 0.0f;
    }
    const int dd = t % kHD, h2 = t / kHD;   // (the values' thread: dim dd of queries h2, h2 + 2, ..)
    float acc[kQn / 2];
#pragma unroll
    for (int k = 0; k < kQn / 2; ++k) acc[k] = 0.0f;
    for (int c0 = j0; c0 < j1; c0 += kSub) {
        const int n = min(kSub, j1 - c0);
        __syncthreads();
        for (int i = t; i < kSub * kHD; i += 256) {
            const int jj = i / kHD, d = i % kHD, j = c0 + jj;
            float kv = 0.0f, vv = 0.0f;
            if (jj < n) {
                if (j < n_ctx) {
                    const size_t row = (size_t) ((lo + j) % ring_n) * kKVD + (size_t) kvh * kHD + d;
                    kv = __half2float(rk[row]);
                    vv = __half2float(rv[row]);
                } else {
                    const size_t row = (size_t) (j - n_ctx) * kKVD + (size_t) kvh * kHD + d;
                    kv = kb[row];
                    vv = vb[row];
                }
            }
            ks[jj][d] = kv;
            vs[jj][d] = vv;
        }
        __syncthreads();
        {   // the scores: thread t's key t % 32 against queries 3 (t / 32) .. + 2
            const int jj = t % kSub, q0 = (t / kSub) * 3;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
#pragma unroll 8
            for (int d = 0; d < kHD; ++d) {
                const float kv = ks[jj][d];
                s0 += qs[q0][d] * kv;
                s1 += qs[q0 + 1][d] * kv;
                s2 += qs[q0 + 2][d] * kv;
            }
            const bool ok = jj < n;
            sc[q0][jj] = ok ? s0 * scale : -INFINITY;
            sc[q0 + 1][jj] = ok ? s1 * scale : -INFINITY;
            sc[q0 + 2][jj] = ok ? s2 * scale : -INFINITY;
        }
        __syncthreads();
        if (t < kQn) {
            float mx = -INFINITY;
            for (int jj = 0; jj < n; ++jj) mx = fmaxf(mx, sc[t][jj]);
            const float mn = fmaxf(mrow[t], mx);
            const float a = mrow[t] == -INFINITY ? 0.0f : __expf(mrow[t] - mn);
            float l = 0.0f;
            for (int jj = 0; jj < kSub; ++jj) {
                const float e = jj < n ? __expf(sc[t][jj] - mn) : 0.0f;
                sc[t][jj] = e;
                l += e;
            }
            lrow[t] = lrow[t] * a + l;
            mrow[t] = mn;
            arow[t] = a;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kQn / 2; ++k) {
            const int qi = h2 + 2 * k;
            float v = acc[k] * arow[qi];
            for (int jj = 0; jj < n; ++jj) v += sc[qi][jj] * vs[jj][dd];
            acc[k] = v;
        }
    }
    __syncthreads();
    const size_t base = ((size_t) sp * kNKV + kvh) * kQn;
    if (t < kQn) {
        pm[base + t] = mrow[t];
        pl[base + t] = lrow[t];
    }
#pragma unroll
    for (int k = 0; k < kQn / 2; ++k) pacc[(base + h2 + 2 * k) * kHD + dd] = acc[k];
}

// the splits' partial sums -> out[row][head * 128 + d] (and its q8_1 blocks); block (head, row), 128 threads
__global__ void __launch_bounds__(kHD) k_attn_comb(const float* __restrict__ pm, const float* __restrict__ pl,
                                                   const float* __restrict__ pacc, int nsplit, float* __restrict__ out,
                                                   Q81* __restrict__ outq) {
    const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    const int kvh = h / kGq, qi = r * kGq + h % kGq;
    float M = -INFINITY;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, pm[((size_t) s * kNKV + kvh) * kQn + qi]);
    float L = 0.0f, acc = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const size_t b = ((size_t) s * kNKV + kvh) * kQn + qi;
        if (pm[b] == -INFINITY) continue;
        const float e = __expf(pm[b] - M);
        L += pl[b] * e;
        acc += pacc[b * kHD + d] * e;
    }
    const size_t o = (size_t) r * kQD + (size_t) h * kHD + d;
    const float v = L > 0.0f ? acc / L : 0.0f;
    out[o] = v;
    q81_store_warp(v, outq + o / 32, d & 31);
}

__global__ void k_add(float* __restrict__ x, const float* __restrict__ a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += a[i];
}

// g = silu(g) * u, and its q8_1 blocks (n a multiple of 32)
__global__ void k_swiglu(float* __restrict__ g, const float* __restrict__ u, int n, Q81* __restrict__ gq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float a = g[i];
        const float v = a / (1.0f + __expf(-a)) * u[i];
        g[i] = v;
        q81_store_warp(v, gq + i / 32, threadIdx.x & 31);
    }
}

// y = x + b and its q8_1 blocks (rows of E; b one row)
__global__ void k_bias_q8(const float* __restrict__ x, const float* __restrict__ b, int E, int n, float* __restrict__ y,
                          Q81* __restrict__ yq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i] + b[i % E];
        y[i] = v;
        q81_store_warp(v, yq + i / 32, threadIdx.x & 31);
    }
}

// rows [r][V] -> each chunk's top kTopK (value, index): the chunk's floats in shared memory, kTopK block argmaxes
// (ties to the lower index, NaN as -inf)
__global__ void __launch_bounds__(256) k_topk_part(const float* __restrict__ logits, int V, int chunk,
                                                   float* __restrict__ tv, int* __restrict__ ti) {
    extern __shared__ float vals[];
    __shared__ float bv[8];
    __shared__ int bi[8];
    const int r = blockIdx.y, ch = blockIdx.x, lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int i0 = ch * chunk, n = max(0, min(chunk, V - i0));
    const float* lr = logits + (size_t) r * V + i0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = lr[i];
        vals[i] = v == v ? v : -INFINITY;
    }
    __syncthreads();
    for (int k = 0; k < kTopK; ++k) {
        float v = -INFINITY;
        int idx = 0x7fffffff;
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (vals[i] > v || (vals[i] == v && i < idx)) { v = vals[i]; idx = i; }
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffffu, v, o);
            const int i2 = __shfl_xor_sync(0xffffffffu, idx, o);
            if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
        }
        if (lane == 0) { bv[wp] = v; bi[wp] = idx; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float bvv = bv[0];
            int bii = bi[0];
            for (int w2 = 1; w2 < 8; ++w2)
                if (bv[w2] > bvv || (bv[w2] == bvv && bi[w2] < bii)) { bvv = bv[w2]; bii = bi[w2]; }
            const size_t o = ((size_t) r * gridDim.x + ch) * kTopK + k;
            tv[o] = bvv;
            ti[o] = bii < n ? i0 + bii : -1;
            if (bii < n) vals[bii] = -INFINITY;
        }
        __syncthreads();
    }
}

// the Markov chain (one block of 1024): for s = 0 .. nd - 1 (block row s + 1), z = zs[s] + jb + wdt^T [state | w1[prev]],
// the state's update, latent, each candidate's score cv + F[id] . latent, the argmax (ties to the lower id, NaN -inf)
// and its confidence (the softmax over the candidates' softcapped scores); prev = the pick.  out: the drafts, conf
constexpr int kMkT = 1024;
__global__ void __launch_bounds__(kMkT) k_markov(const float* __restrict__ zs, const float* __restrict__ jb,
                                                 const uint16_t* __restrict__ wdt, const uint16_t* __restrict__ w1,
                                                 const __half* __restrict__ F, const float* __restrict__ cv,
                                                 const int* __restrict__ ci, int nc, int nd, const int* __restrict__ y,
                                                 float softcap, int V, int* __restrict__ out, float* __restrict__ conf) {
    __shared__ float st[kRank], xin[2 * kRank], z[kZ], lat[kRank], scs[kNC];
    __shared__ float wv[32], wsum[32];
    __shared__ int wi[32];
    __shared__ int prev_s;
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5;
    if (t < kRank) st[t] = 0.0f;
    if (t == 0) prev_s = *y;
    __syncthreads();
    for (int s = 0; s < nd; ++s) {
        const int prev = prev_s;
        if (t < kRank) {
            xin[t] = st[t];
            xin[kRank + t] = prev >= 0 && prev < V ? bf16f(w1[(size_t) prev * kRank + t]) : 0.0f;
        }
        __syncthreads();
        if (t < kZ) {   // (4 sums in flight: the rows' loads overlap)
            float a4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
            for (int k = 0; k < 2 * kRank; k += 4) {
#pragma unroll
                for (int u = 0; u < 4; ++u) a4[u] += bf16f(wdt[(size_t) (k + u) * kZ + t]) * xin[k + u];
            }
            z[t] = zs[(size_t) s * kZ + t] + jb[t] + ((a4[0] + a4[1]) + (a4[2] + a4[3]));
        }
        __syncthreads();
        if (t < kRank) {
            const float g = 1.0f / (1.0f + __expf(-z[t]));
            st[t] = g * st[t] + (1.0f - g) * tanhf(z[kRank + t]);
            lat[t] = tanhf(z[2 * kRank + t]);
        }
        __syncthreads();
        // the candidates: a thread each (its F row's 32 16-byte pieces in flight at once)
        float bvv = -INFINITY;
        int bii = 0x7fffffff;
        for (int c = t; c < nc; c += kMkT) {
            const int id = ci[(size_t) s * nc + c];
            float sc = -INFINITY;
            if (id >= 0 && id < V) {
                const uint4* row = reinterpret_cast<const uint4*>(F + (size_t) id * kRank);
                float dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 8
                for (int p = 0; p < kRank / 8; ++p) {
                    const uint4 raw = row[p];
                    const __half2* h = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float2 f = __half22float2(h[e]);
                        dot[e] += f.x * lat[p * 8 + 2 * e] + f.y * lat[p * 8 + 2 * e + 1];
                    }
                }
                sc = cv[(size_t) s * nc + c] + ((dot[0] + dot[1]) + (dot[2] + dot[3]));
                if (!(sc == sc)) sc = -INFINITY;
            }
            scs[c] = sc;
            if (sc > bvv || (sc == bvv && id < bii)) { bvv = sc; bii = id; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_xor_sync(0xffffffffu, bvv, o);
            const int i2 = __shfl_xor_sync(0xffffffffu, bii, o);
            if (v2 > bvv || (v2 == bvv && i2 < bii)) { bvv = v2; bii = i2; }
        }
        if (lane == 0) { wv[wp] = bvv; wi[wp] = bii; }
        __syncthreads();
        if (t == 0) {
            float b = wv[0];
            int bi = wi[0];
            for (int w2 = 1; w2 < 32; ++w2)
                if (wv[w2] > b || (wv[w2] == b && wi[w2] < bi)) { b = wv[w2]; bi = wi[w2]; }
            wv[0] = b;
            wi[0] = bi;
        }
        __syncthreads();
        const float best = wv[0];
        const int bid = wi[0];
        // the confidence: 1 / sum_c exp(cap(s_c) - cap(best))
        const float mb = softcap > 0.0f ? softcap * tanhf(best / softcap) : best;
        float e = 0.0f;
        for (int c = t; c < nc; c += kMkT) {
            const float v = scs[c];
            if (v != -INFINITY) e += __expf((softcap > 0.0f ? softcap * tanhf(v / softcap) : v) - mb);
        }
        e = warp_sum(e);
        if (lane == 0) wsum[wp] = e;
        __syncthreads();
        if (t == 0) {
            float tot = 0.0f;
            for (int w2 = 0; w2 < 32; ++w2) tot += wsum[w2];
            const bool ok = best != -INFINITY && bid >= 0 && bid < V;
            out[s] = ok ? bid : -1;
            conf[s] = ok && tot > 0.0f ? 1.0f / tot : 0.0f;
            prev_s = ok ? bid : -1;
        }
        __syncthreads();
    }
}

bool mv_rows(const WSlot& w, int n_in, int n_out, const float* x, const void* xq, float* y, int rows, cudaStream_t s) {
    gf::MvJob j = {w.q, xq, x, y, nullptr, 1.0f, w.type, n_in, n_out};
    if (w.type == kF32) j.w = w.f32;
    return gf::mvn(&j, 1, rows, s);
}

int64_t env_i(const char* n, int64_t def) {
    const char* v = getenv(n);
    return v != nullptr && v[0] ? std::atoll(v) : def;
}

}  // namespace

// ------------------------------------------------------------------------------------------------ load
// the drafter where setup and the Hugging Face layout keep it, from the pack's folder up: <d>/<name>, <d>/dspark/<name>
// (unsloth's repo folder), and <d>/[models/]unsloth/Inkling-2-Mini-GGUF/[dspark/]<name> (the abliterated model's pack
// finds unsloth's drafter there too)
std::string Glm5Model::inkd_find() const {
    namespace fs = std::filesystem;
    static const char* kName = "dspark-Inkling-2-Mini-Q8_0.gguf";
    if (pack_dir_.empty()) return "";
    std::error_code ec;
    fs::path d = fs::absolute(fs::path(pack_dir_), ec);
    if (ec) return "";
    const fs::path ud = fs::path("unsloth") / "Inkling-2-Mini-GGUF";
    for (int up = 0; up < 6 && !d.empty(); ++up) {
        for (const fs::path& c : {d / kName, d / "dspark" / kName, d / ud / "dspark" / kName, d / ud / kName,
                                  d / "models" / ud / "dspark" / kName, d / "models" / ud / kName})
            if (fs::is_regular_file(c, ec) && fs::file_size(c, ec) > ((uintmax_t) 100 << 20)) return c.string();
        if (d == d.parent_path()) break;
        d = d.parent_path();
    }
    return "";
}

size_t Glm5Model::inkd_bytes() const {
    const auto* D = (const IkState*) inkd_;
    return D != nullptr ? D->bytes : 0;
}

bool Glm5Model::inkd_load(std::string& err) {
    (void) err;
    const Glm5Geometry& g = g_;
    // STRATA_INK_DSPARK=<the GGUF> loads that drafter; =0 (or off / none) keeps it off.  Unset (2026-11-24): the file
    // beside the pack (inkd_find) loads on one CUDA GPU with one lane, a discrete card, not under STRATA_DETERMINISTIC
    // - there the expert tiers stay where they start and the CPU lane is off, so a window's extra columns fetch the
    // RAM-tier experts they route over PCIe (a 165K prompt on an RTX 5090, 1,024 tokens: 3.1 -> 9.2 RAM fetches a
    // token, 85.0 -> 62.0 tok/s); the tests that compare drafts name the file.  fast_setup then keeps it only when the
    // expert pool comes out the same as without it (inkd_auto_)
    const char* path = getenv("STRATA_INK_DSPARK");
    inkd_auto_ = false;
    if (!g.ink) return true;
    if (path != nullptr && (std::strcmp(path, "0") == 0 || std::strcmp(path, "off") == 0 || std::strcmp(path, "none") == 0))
        return true;
    std::string found;
    if (path == nullptr || !path[0]) {
#if defined(STRATA_USE_HIP)
        return true;   // (not measured on HIP; STRATA_INK_DSPARK=<file> loads it)
#else
        const FastState* F = fast_;
        if (n_parts_ != 1 || l0_ != 0 || l1_ != g.n_layers || F == nullptr || F->unified_memory ||
            ink_parallel_env() != 1)
            return true;
        found = inkd_find();
        if (found.empty()) return true;
        if (const char* dv = getenv("STRATA_DETERMINISTIC"); dv != nullptr && std::atoi(dv) != 0) {
            std::fprintf(stderr, "ink dspark: %s not loaded - STRATA_DETERMINISTIC (STRATA_INK_DSPARK=<file> loads it)\n",
                         found.c_str());
            return true;
        }
        path = found.c_str();
        inkd_auto_ = true;
#endif
    }
    const auto refuse = [&](const std::string& why) {
        std::fprintf(stderr, "ink dspark: not loaded - %s\n", why.c_str());
        inkd_destroy();
        return true;   // the target runs on without the drafter
    };
    if (n_parts_ != 1 || l0_ != 0 || l1_ != g.n_layers) return refuse("one device only (no split)");
    if (g.ink_slots() != 48) return refuse("the target's 48 layer applications (Inkling 2 Mini)");
    auto* D = new IkState();
    inkd_ = D;
    try {
        D->gguf = std::make_unique<strata::GgufFile>(path);
    } catch (const std::exception& e) {
        return refuse(std::string("cannot open ") + path + ": " + e.what());
    }
    const strata::GgufFile& G = *D->gguf;
    const auto num = [&](const std::string& key, double def) {
        const strata::MetaValue* v = G.get(key);
        return v != nullptr && v->is_num() ? v->num() : def;
    };
    const strata::MetaValue* arch = G.get("general.architecture");
    if (arch == nullptr || arch->s != "dflash") return refuse("not a dflash GGUF");
    const strata::MetaValue* tl = G.get("dflash.target_layers");
    static const int kTaps[4] = {5, 23, 41, 48};
    bool taps_ok = tl != nullptr && tl->items.size() == 4;
    for (int i = 0; taps_ok && i < 4; ++i) taps_ok = (int) tl->items[(size_t) i].num() == kTaps[i];
    const int E = g.n_embd, V = g.n_vocab;
    D->V = V;
    D->ff = (int) num("dflash.feed_forward_length", 0);
    D->eps = (float) num("dflash.attention.layer_norm_rms_epsilon", 1e-6);
    D->softcap = (float) num("dflash.final_logit_softcapping", 30.0);
    const double rope_base = num("dflash.rope.freq_base", 1e6);
    if (!taps_ok || (int) num("dflash.block_count", 0) != kL || (int) num("dflash.embedding_length", 0) != E ||
        D->ff <= 0 || D->ff % 256 != 0 || (int) num("dflash.attention.head_count", 0) != kNH ||
        (int) num("dflash.attention.head_count_kv", 0) != kNKV || (int) num("dflash.attention.key_length", 0) != kHD ||
        (int) num("dflash.attention.value_length", 0) != kHD || (int) num("dflash.block_size", 0) != kNB ||
        (int) num("dflash.dspark.markov_rank", 0) != kRank || (int) num("dflash.attention.sliding_window", 0) != kLocalWin ||
        (int) num("dflash.rope.dimension_count", kHD) != kHD || E % 256 != 0 || E > 256 * kRmsPer || V <= 0)
        return refuse("the drafter's layout (3 layers over taps 5/23/41/48, the target's width, 24 / 8 heads of 128, "
                      "block 8, Markov rank 256, a 1024-row window)");
    D->n_max = (int) std::max<int64_t>(1, std::min<int64_t>(kND, env_i("STRATA_INK_DSPARK_N", kND)));
    D->ctx_g = (int) std::max<int64_t>(2048, std::min<int64_t>(env_i("STRATA_INK_DSPARK_CTX", 32768),
                                                               std::max<int64_t>(max_ctx_, 2048)));
    std::string why;
    bool ok = true;
    const auto dev = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (!ok || cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; cudaGetLastError(); return nullptr; }
        D->allocs.push_back(p);
        D->bytes += bytes;
        return p;
    };
    const auto find = [&](const std::string& n, int type, std::initializer_list<uint64_t> shape) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = G.find(n);
        if (t == nullptr) { if (why.empty()) why = n + " missing"; return nullptr; }
        bool good = (int) t->type == type && t->shape.size() == shape.size();
        size_t i = 0;
        for (uint64_t d : shape) good = good && t->shape[i++] == d;
        if (!good && why.empty()) why = n + " has another type or shape";
        return good ? t : nullptr;
    };
    const auto raw = [&](const std::string& n, int64_t n_in, int64_t n_out, WSlot& w) {
        const strata::TensorInfo* t0 = G.find(n);
        const int ty = t0 != nullptr ? (int) t0->type : kQ8_0;
        if (t0 != nullptr && (!gf::mvn_supported(ty) || ty == kF32)) {
            if (why.empty()) why = n + ": a type the drafter's kernels do not serve";
            ok = false;
            return;
        }
        const strata::TensorInfo* t = find(n, ty, {(uint64_t) n_in, (uint64_t) n_out});
        if (t == nullptr) { ok = false; return; }
        const size_t bytes = gf::row_bytes(ty, n_in) * (size_t) n_out;
        void* p = dev(bytes);
        if (p == nullptr) return;
        ok = cudaMemcpy(p, G.tensor_data(*t), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        w = WSlot{nullptr, ty, p, n_in, n_out};
    };
    const auto f32 = [&](const std::string& n, std::initializer_list<uint64_t> shape) -> float* {
        const strata::TensorInfo* t = find(n, kF32, shape);
        if (t == nullptr) { ok = false; return nullptr; }
        float* p = (float*) dev(t->elements() * sizeof(float));
        if (p != nullptr)
            ok = cudaMemcpy(p, G.tensor_data(*t), t->elements() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
        return p;
    };
    const uint64_t uE = (uint64_t) E;
    const int FF = D->ff;
    for (int l = 0; l < kL && ok; ++l) {
        IkLayer& Ly = D->L[l];
        const std::string P = "blk." + std::to_string(l) + ".";
        raw(P + "attn_q.weight", E, kQD, Ly.q);
        raw(P + "attn_k.weight", E, kKVD, Ly.k);
        raw(P + "attn_v.weight", E, kKVD, Ly.v);
        raw(P + "attn_output.weight", kQD, E, Ly.o);
        raw(P + "ffn_gate.weight", E, FF, Ly.gate);
        raw(P + "ffn_up.weight", E, FF, Ly.up);
        raw(P + "ffn_down.weight", FF, E, Ly.down);
        raw(P + "attn_k_ctx.weight", E, kKVD, Ly.kc);
        raw(P + "attn_v_ctx.weight", E, kKVD, Ly.vc);
        Ly.attn_norm = f32(P + "attn_norm.weight", {uE});
        Ly.ffn_norm = f32(P + "ffn_norm.weight", {uE});
        Ly.q_norm = f32(P + "attn_q_norm.weight", {(uint64_t) kHD});
        Ly.k_norm = f32(P + "attn_k_norm.weight", {(uint64_t) kHD});
        Ly.kc_norm = f32(P + "attn_k_ctx_norm.weight", {(uint64_t) kHD});
        Ly.dflare = f32(P + "dflare_norm.weight", {uE});
        // (sliding_window_pattern [T, F, T]: layers 0 and 2 the window, layer 1 every position)
        const bool local = l != 1;
        Ly.ring = local ? kLocalRing : D->ctx_g;
        Ly.win = local ? kLocalWin : D->ctx_g;
        Ly.rk = (__half*) dev((size_t) Ly.ring * kKVD * sizeof(__half));
        Ly.rv = (__half*) dev((size_t) Ly.ring * kKVD * sizeof(__half));
    }
    D->out_norm = f32("output_norm.weight", {uE});
    D->mk_norm = f32("dspark_norm.weight", {uE});
    D->mask = f32("mask_embd.weight", {uE});
    D->ho_b = f32("dspark_hidden_out.bias", {uE});
    D->jb = f32("joint_proj.bias", {(uint64_t) kZ});
    if (!ok) return refuse(why.empty() ? "the weights did not upload (VRAM?)" : why);
    {   // the fuse's rows (softmax over the 4 taps)
        const strata::TensorInfo* t = find("dspark_fuse.weight", kF32, {4, 4});
        if (t == nullptr) return refuse(why);
        const float* fw = (const float*) G.tensor_data(*t);
        for (int r = 0; r < 4; ++r) {
            float m = -INFINITY, s = 0.0f;
            for (int j = 0; j < 4; ++j) m = std::max(m, fw[r * 4 + j]);
            for (int j = 0; j < 4; ++j) s += std::exp(fw[r * 4 + j] - m);
            for (int j = 0; j < 4; ++j) D->al.a[r][j] = std::exp(fw[r * 4 + j] - m) / s;
        }
    }
    // markov_w1 [V(pad)][256] BF16 as it is (the first V rows); joint_proj [768][5120] BF16 split: the state / prev
    // columns transposed ([512][768], the chain's reads coalesced), the hidden / f_rnn columns ([768][4608], mvn)
    {
        const strata::TensorInfo* tw = G.find("markov_w1.weight");
        if (tw == nullptr || (int) tw->type != kBF16 || tw->shape.size() != 2 || tw->shape[0] != (uint64_t) kRank ||
            tw->shape[1] < (uint64_t) V)
            return refuse("markov_w1.weight (BF16 256 x vocab)");
        D->w1 = (uint16_t*) dev((size_t) V * kRank * 2);
        if (ok) ok = cudaMemcpy(D->w1, G.tensor_data(*tw), (size_t) V * kRank * 2, cudaMemcpyHostToDevice) == cudaSuccess;
        const strata::TensorInfo* tj = find("joint_proj.weight", kBF16, {(uint64_t) 2 * kRank + 2 * uE, (uint64_t) kZ});
        if (tj == nullptr) return refuse(why);
        const uint16_t* jw = (const uint16_t*) G.tensor_data(*tj);
        const size_t jin = (size_t) 2 * kRank + 2 * E, jsi = (size_t) 2 * E;
        std::vector<uint16_t> dt((size_t) 2 * kRank * kZ), sta((size_t) kZ * jsi);
        for (size_t o = 0; o < (size_t) kZ; ++o) {
            for (size_t k = 0; k < (size_t) 2 * kRank; ++k) dt[k * kZ + o] = jw[o * jin + k];
            std::memcpy(&sta[o * jsi], &jw[o * jin + 2 * kRank], jsi * 2);
        }
        D->wdt = (uint16_t*) dev(dt.size() * 2);
        D->wsta = (uint16_t*) dev(sta.size() * 2);
        if (ok)
            ok = cudaMemcpy(D->wdt, dt.data(), dt.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemcpy(D->wsta, sta.data(), sta.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess;
        if (!gf::mvn_supported(kBF16)) return refuse("mvn has no BF16 rows");
    }
    if (!ok) return refuse("the Markov head's weights did not upload (VRAM?)");
    // the fold F = W_lm . hidden_out ([V][256] FP16): W_lm's rows dequantized to FP16 a slice at a time, cuBLAS
    const auto ow_it = ws_map_.find("output.weight");
    if (ow_it == ws_map_.end()) return refuse("the target's output.weight");
    const WSlot ow = ow_it->second;
    if (ow.type != kF32 && !strata::kernels::dequant_bf16_supported(ow.type))
        return refuse("the target's output.weight type " + std::to_string(ow.type) + " (no FP16 dequant)");
    {
        const strata::TensorInfo* th = find("dspark_hidden_out.weight", kBF16, {(uint64_t) kRank, uE});
        if (th == nullptr) return refuse(why);
        const uint16_t* hw = (const uint16_t*) G.tensor_data(*th);
        std::vector<__half> h16((size_t) E * kRank);   // [E][256] row-major = column-major [256][E]
        for (size_t i = 0; i < h16.size(); ++i) {
            uint32_t b = (uint32_t) hw[i] << 16;
            float f;
            std::memcpy(&f, &b, 4);
            h16[i] = __float2half(f);
        }
        D->F = (__half*) dev((size_t) V * kRank * sizeof(__half));
        constexpr int kSl = 8192;
        void* hdev = nullptr;
        void* wsl = nullptr;
        cublasHandle_t bh = nullptr;
        bool fok = ok && cudaMalloc(&hdev, h16.size() * 2) == cudaSuccess &&
                   cudaMalloc(&wsl, (size_t) kSl * E * 2) == cudaSuccess && cublasCreate(&bh) == CUBLAS_STATUS_SUCCESS;
        cudaStream_t s = fast_->cs;
        if (fok) fok = cudaMemcpy(hdev, h16.data(), h16.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess &&
                       cublasSetStream(bh, s) == CUBLAS_STATUS_SUCCESS;
        std::vector<float> f32tmp;
        for (int r0 = 0; fok && r0 < V; r0 += kSl) {
            const int n = std::min(kSl, V - r0);
            if (ow.type == kF32) {
                f32tmp.resize((size_t) n * E);
                std::vector<__half> hh((size_t) n * E);
                fok = cudaMemcpy(f32tmp.data(), ow.f32 + (size_t) r0 * E, hh.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
                for (size_t i = 0; i < hh.size(); ++i) hh[i] = __float2half(f32tmp[i]);
                fok = fok && cudaMemcpy(wsl, hh.data(), hh.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess;
            } else {
                strata::kernels::dequant_f16(ow.type, ow.q, r0, n, E, (uint16_t*) wsl, s);
            }
            const float one = 1.0f, zero = 0.0f;
            // column-major: F^T[256][n] = H^T[256][E] . W^T[E][n]
            fok = fok && cublasGemmEx(bh, CUBLAS_OP_N, CUBLAS_OP_N, kRank, n, E, &one, hdev, CUDA_R_16F, kRank, wsl,
                                      CUDA_R_16F, E, &zero, D->F + (size_t) r0 * kRank, CUDA_R_16F, kRank,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
        }
        if (fok) fok = cudaStreamSynchronize(s) == cudaSuccess;
        if (bh) cublasDestroy(bh);
        if (hdev) cudaFree(hdev);
        if (wsl) cudaFree(wsl);
        if (!fok) {
            cudaGetLastError();
            return refuse("the fold W_lm . hidden_out");
        }
    }
    {   // the RoPE frequencies
        double inv[kHD / 2];
        for (int i = 0; i < kHD / 2; ++i) inv[i] = std::pow(rope_base, -2.0 * (double) i / (double) kHD);
        D->inv = (double*) dev(sizeof inv);
        if (ok) ok = cudaMemcpy(D->inv, inv, sizeof inv, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    // the buffers
    const auto fb = [&](float*& p, size_t n) { p = (float*) dev(n * sizeof(float)); };
    const auto bb = [&](void*& p, size_t n) { p = dev(n); };
    D->rows_g = std::max(kVerifyMax, 64);
    fb(D->acc_g, (size_t) D->rows_g * E);
    fb(D->acc_l, (size_t) kTail * 3 * E);
    fb(D->cf, (size_t) 2 * kCh * E);
    bb(D->cq, (size_t) 2 * kCh * E / 32 * kQ8);
    fb(D->kk, (size_t) 2 * kCh * kKVD);
    fb(D->vv, (size_t) 2 * kCh * kKVD);
    fb(D->frnn, E);
    D->roff_d = (int64_t*) dev(64);
    fb(D->x, (size_t) kNB * E);
    fb(D->xn, (size_t) kNB * E);
    bb(D->xq, (size_t) kNB * E / 32 * kQ8);
    fb(D->q, (size_t) kNB * kQD);
    fb(D->kb, (size_t) kNB * kKVD);
    fb(D->vb, (size_t) kNB * kKVD);
    fb(D->o, (size_t) kNB * kQD);
    bb(D->oq, (size_t) kNB * kQD / 32 * kQ8);
    fb(D->a, (size_t) kNB * E);
    fb(D->gt, (size_t) kNB * FF);
    fb(D->up, (size_t) kNB * FF);
    bb(D->gq, (size_t) kNB * FF / 32 * kQ8);
    fb(D->hn, (size_t) kNB * E);
    bb(D->hq, (size_t) kNB * E / 32 * kQ8);
    fb(D->xz, (size_t) kND * 2 * E);
    fb(D->zs, (size_t) kND * kZ);
    fb(D->logits, (size_t) kND * V);
    fb(D->tk_v, (size_t) kND * kNC);
    D->tk_i = (int*) dev((size_t) kND * kNC * sizeof(int));
    fb(D->pm, (size_t) kMaxSplit * kNKV * kQn);
    fb(D->pl, (size_t) kMaxSplit * kNKV * kQn);
    fb(D->pacc, (size_t) kMaxSplit * kNKV * kQn * kHD);
    D->res_d = (int*) dev(16 * sizeof(int));
    fb(D->res_conf_d, 16);
    if (!ok) return refuse("the buffers did not allocate (VRAM?)");
    ok = cudaHostAlloc((void**) &D->y_h, 64, cudaHostAllocMapped) == cudaSuccess &&
         cudaHostGetDevicePointer((void**) &D->y_d, D->y_h, 0) == cudaSuccess &&
         cudaHostAlloc((void**) &D->res_h, 64, cudaHostAllocDefault) == cudaSuccess &&
         cudaHostAlloc((void**) &D->res_conf_h, 64, cudaHostAllocDefault) == cudaSuccess &&
         cudaHostAlloc((void**) &D->emb_h, (size_t) E * sizeof(float), cudaHostAllocDefault) == cudaSuccess;
    if (!ok) return refuse("the pinned buffers");
    if (const char* v = getenv("STRATA_INK_DSPARK_DUMP"); v != nullptr && v[0]) D->dump = std::fopen(v, "w");
    // STRATA_INK_DSPARK_DEFER=0: every single token's context rows committed right after it, as before.  Default:
    // they wait in the accumulators (up to 64 rows) and go in one pass before the next draft / window / prompt chunk -
    // the per-token commit (~12 small launches behind the token, ~1% of a token's time) mostly gone where the gate
    // keeps the drafter off; the drafter's context the same rows
    if (const char* v = getenv("STRATA_INK_DSPARK_DEFER"); v == nullptr || std::atoi(v) != 0) D->defer = true;
    if (const char* v = getenv("STRATA_INK_DSPARK_PROF"); v != nullptr && std::atoi(v) != 0) {
        D->prof = true;
        for (cudaEvent_t& e : D->ev) D->prof = D->prof && cudaEventCreate(&e) == cudaSuccess;
    }
    std::fprintf(stderr, "ink dspark: %s - %d layers over taps 5/23/41/48, up to %d drafts a step; the global layer's "
                         "context %d positions; %.2f GB on CUDA%d\n",
                 path, kL, D->n_max, D->ctx_g, (double) D->bytes / 1073741824.0, dev_);
    return true;
}

void Glm5Model::inkd_destroy() {
    auto* D = (IkState*) inkd_;
    inkd_auto_ = false;
    if (D == nullptr) return;
    inkd_ = nullptr;
    for (void* p : D->allocs) cudaFree(p);
    if (D->dump) std::fclose(D->dump);
    for (cudaEvent_t e : D->ev)
        if (e) cudaEventDestroy(e);
    for (void* p : {(void*) D->y_h, (void*) D->res_h, (void*) D->res_conf_h, (void*) D->emb_h})
        if (p) cudaFreeHost(p);
    delete D;
}

int Glm5Model::inkd_max() const {
    const auto* D = (const IkState*) inkd_;
    return D != nullptr ? D->n_max : 0;
}

void Glm5Model::inkd_report() const {
    const auto* D = (const IkState*) inkd_;
    if (D == nullptr || D->drafts == 0) return;
    std::fprintf(stderr, "ink dspark: %llu drafts, %.2f ms each (host to results); %llu context commits (%llu of deferred rows), "
                         "%llu prompt rows (%llu past the global layer's buffer)\n",
                 (unsigned long long) D->drafts, D->ms_draft / (double) D->drafts, (unsigned long long) D->commits,
                 (unsigned long long) D->flushes, (unsigned long long) D->prompt_rows, (unsigned long long) D->prompt_cut);
    if (D->prof) {
        static const char* names[IkState::kPh] = {"block", "layer 0", "layer 1", "layer 2", "head", "markov"};
        std::fprintf(stderr, "ink dspark: device ms a draft -");
        double t = 0.0;
        for (int i = 0; i < IkState::kPh; ++i) {
            std::fprintf(stderr, " %s %.3f", names[i], D->ms_ph[i] / (double) D->drafts);
            t += D->ms_ph[i];
        }
        std::fprintf(stderr, " | total %.3f\n", t / (double) D->drafts);
    }
}

void Glm5Model::inkd_reset() {
    auto* D = (IkState*) inkd_;
    if (D == nullptr) return;
    inkd_flush();
    D->win_p0 = -1;
    // a restored snapshot (pos_ inside the context): the rows before it are still this sequence's
    if (pos_ > 0 && pos_ <= D->ctx_end) {
        D->ctx_end = pos_;
        for (auto& Ly : D->L) Ly.lo = std::min<int64_t>(std::max<int64_t>(Ly.lo, pos_ - Ly.ring), pos_);
        if (D->frnn_pos != pos_ - 1) D->frnn_pos = -1;
    } else {
        D->ctx_end = pos_;
        for (auto& Ly : D->L) Ly.lo = pos_;
        D->frnn_pos = -1;
    }
}

// ------------------------------------------------------------------------------------------------ the context
void Glm5Model::ink_tap(int visit, const float* x, int64_t ld, int n, const float* norm_w, bool prompt, cudaStream_t s,
                        bool pend) {
    auto* D = (IkState*) inkd_;
    if (D == nullptr || n <= 0) return;
    const int j = visit == 5 ? 0 : visit == 23 ? 1 : visit == 41 ? 2 : visit == 48 ? 3 : -1;
    if (j < 0) return;
    const int E = g_.n_embd;
    if (j == 0 && !pend && D->pend_n > 0) {   // (other rows over the deferred ones: those committed first)
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(s, &cs);
        if (cs == cudaStreamCaptureStatusNone) inkd_flush();
    }
    if (j == 0) {   // (the rows' first tap: how many of them the accumulators hold)
        int ng = n;
        if (prompt && n > D->rows_g) {   // (a long chunk: acc_g grown - halving while the VRAM refuses)
            int want = std::min(n, D->ctx_g);
            float* p = nullptr;
            while (want > D->rows_g && cudaMalloc(&p, (size_t) want * E * sizeof(float)) != cudaSuccess) {
                cudaGetLastError();
                p = nullptr;
                want /= 2;
            }
            if (p != nullptr) {
                cudaStreamSynchronize(s);
                auto it = std::find(D->allocs.begin(), D->allocs.end(), (void*) D->acc_g);
                if (it != D->allocs.end()) *it = p;
                cudaFree(D->acc_g);
                D->bytes += (size_t) (want - D->rows_g) * E * sizeof(float);
                D->acc_g = p;
                D->rows_g = want;
                ++D->acc_gen;
            }
            ng = std::min(n, D->rows_g);
        }
        D->tap_n = n;
        D->tap_ng = std::min(ng, D->rows_g);
        D->tap_nl = std::min(n, kTail);
        D->tap_prompt = prompt;
    }
    if (n != D->tap_n) return;   // (a tap of other rows than the first one's: a mismatch - not taken)
    const bool pd = pend && D->defer && n == 1;
    strata::kernels::pdl_launch(k_tap, dim3((unsigned) n), dim3(256), 0, s, x, ld, n, norm_w, g_.norm_eps, E, j, D->tap_ng,
                                D->tap_nl, D->acc_g, D->acc_l, D->al, pd ? D->tap_pa : nullptr, pd ? D->tap_pb : nullptr);
}

uint64_t Glm5Model::inkd_acc_gen() const {
    const auto* D = (const IkState*) inkd_;
    return D == nullptr ? 0 : D->acc_gen;
}

void Glm5Model::inkd_tok_taps(int n) {   // (a replayed token / window graph: its taps' bookkeeping, as ink_tap's first sets it)
    auto* D = (IkState*) inkd_;
    if (D == nullptr) return;
    D->tap_n = n;
    D->tap_ng = std::min(n, D->rows_g);
    D->tap_nl = std::min(n, kTail);
    D->tap_prompt = false;
}

bool Glm5Model::inkd_commit(int64_t p0, int n, bool prompt, cudaStream_t s) {
    auto* D = (IkState*) inkd_;
    if (D == nullptr || n <= 0) return true;
    if (n != D->tap_n || prompt != D->tap_prompt) {   // (no taps for these rows: the context starts over after them)
        D->ctx_end = p0 + n;
        for (auto& Ly : D->L) Ly.lo = p0 + n;
        D->frnn_pos = -1;
        return true;
    }
    const int E = g_.n_embd;
    const int ng = D->tap_ng, nl = D->tap_nl;
    if (prompt) {
        D->prompt_rows += (uint64_t) n;
        D->prompt_cut += (uint64_t) (n - ng);
    }
    const bool gap = p0 != D->ctx_end;
    // layer 1 (every position): acc_g's rows = positions p0 + n - ng ..
    for (int c0 = 0; c0 < ng; c0 += kCh) {
        const int m = std::min(kCh, ng - c0);
        const IkLayer& Ly = D->L[1];
        k_rms<<<m, 256, 0, s>>>(D->acc_g + (size_t) c0 * E, E, Ly.dflare, D->eps, E, D->cf, E, (Q81*) D->cq);
        gf::MvJob j[2] = {{Ly.kc.q, D->cq, D->cf, D->kk, nullptr, 1.0f, Ly.kc.type, E, kKVD},
                          {Ly.vc.q, D->cq, D->cf, D->vv, nullptr, 1.0f, Ly.vc.type, E, kKVD}};
        if (!gf::mvn(j, 2, m, s)) return false;
        const long long pp = (long long) (p0 + n - ng + c0);
        k_norm_rope<<<dim3(kNKV, m), kHD, 0, s>>>(D->kk, kKVD, Ly.kc_norm, D->eps, pp, D->inv, Ly.rk, Ly.ring, kKVD);
        k_norm_rope<<<dim3(kNKV, m), kHD, 0, s>>>(D->vv, kKVD, nullptr, D->eps, pp, nullptr, Ly.rv, Ly.ring, kKVD);
    }
    // layers 0 and 2 (the window): acc_l's rows = positions p0 + n - nl ..
    for (int c0 = 0; c0 < nl; c0 += kCh) {
        const int m = std::min(kCh, nl - c0);
        const float* src = D->acc_l + (size_t) c0 * 3 * E;
        for (int w = 0; w < 2; ++w)
            k_rms<<<m, 256, 0, s>>>(src + (size_t) w * E, (int64_t) 3 * E, D->L[2 * w].dflare, D->eps, E,
                                    D->cf + (size_t) w * kCh * E, E, (Q81*) D->cq + (size_t) w * kCh * (E / 32));
        gf::MvJob j[4];
        for (int w = 0; w < 2; ++w) {
            const IkLayer& Ly = D->L[2 * w];
            const void* xq = (const Q81*) D->cq + (size_t) w * kCh * (E / 32);
            const float* xf = D->cf + (size_t) w * kCh * E;
            j[2 * w] = {Ly.kc.q, xq, xf, D->kk + (size_t) w * kCh * kKVD, nullptr, 1.0f, Ly.kc.type, E, kKVD};
            j[2 * w + 1] = {Ly.vc.q, xq, xf, D->vv + (size_t) w * kCh * kKVD, nullptr, 1.0f, Ly.vc.type, E, kKVD};
        }
        if (!gf::mvn(j, 4, m, s)) return false;
        const long long pp = (long long) (p0 + n - nl + c0);
        for (int w = 0; w < 2; ++w) {
            const IkLayer& Ly = D->L[2 * w];
            k_norm_rope<<<dim3(kNKV, m), kHD, 0, s>>>(D->kk + (size_t) w * kCh * kKVD, kKVD, Ly.kc_norm, D->eps, pp,
                                                       D->inv, Ly.rk, Ly.ring, kKVD);
            k_norm_rope<<<dim3(kNKV, m), kHD, 0, s>>>(D->vv + (size_t) w * kCh * kKVD, kKVD, nullptr, D->eps, pp, nullptr,
                                                       Ly.rv, Ly.ring, kKVD);
        }
    }
    // f_rnn: the last row's mix 3
    k_rms<<<1, 256, 0, s>>>(D->acc_l + ((size_t) (nl - 1) * 3 + 2) * E, E, D->mk_norm, D->eps, E, D->frnn, E, nullptr);
    D->frnn_pos = p0 + n - 1;
    const int64_t end = p0 + n;
    for (int l = 0; l < kL; ++l) {
        IkLayer& Ly = D->L[l];
        const int64_t first = p0 + n - (l == 1 ? ng : nl);
        if (gap || first != p0 || Ly.lo > p0) Ly.lo = first;   // (rows before first not written: it starts there)
        Ly.lo = std::max<int64_t>(Ly.lo, end - Ly.ring);
    }
    D->ctx_end = end;
    D->tap_n = 0;
    ++D->commits;
    return cudaGetLastError() == cudaSuccess;
}

void Glm5Model::inkd_tok_begin(int64_t p, const int64_t* pos_d) {
    auto* D = (IkState*) inkd_;
    if (D == nullptr || !D->defer) return;
    if (D->pend_n > 0 && p < D->pend_p0 + D->pend_n) D->pend_n = (int) std::max<int64_t>(0, p - D->pend_p0);   // (a rewind)
    if (D->pend_n > 0 && (p != D->pend_p0 + D->pend_n || D->pend_n >= std::min(64, D->rows_g))) inkd_flush();
    const bool fresh = D->pend_n == 0;
    if (fresh) D->pend_p0 = p;
    if (pos_d != nullptr) {   // (the graph: the row from its position word less the first deferred one's - set once)
        if (fresh || D->tap_pb != D->roff_d + 1)
            strata::kernels::pdl_launch(k_set_i64, dim3(1), dim3(1), 0, fast_->cs, D->roff_d + 1, D->pend_p0);
        D->tap_pa = pos_d;
        D->tap_pb = D->roff_d + 1;
    } else {
        strata::kernels::pdl_launch(k_set_i64, dim3(1), dim3(1), 0, fast_->cs, D->roff_d, (int64_t) D->pend_n);
        D->tap_pa = D->roff_d;
        D->tap_pb = nullptr;
    }
}

bool Glm5Model::inkd_tok_end(int64_t p, cudaStream_t s) {
    auto* D = (IkState*) inkd_;
    if (D == nullptr) return true;
    if (!D->defer) return inkd_commit(p, 1, false, s);
    if (D->tap_n != 1 || D->tap_prompt || p != D->pend_p0 + D->pend_n) {   // (no taps for it: as the commit's mismatch)
        if (!inkd_flush()) return false;
        return inkd_commit(p, 1, false, s);
    }
    ++D->pend_n;
    D->tap_n = 0;
    return true;
}

bool Glm5Model::inkd_flush() {
    auto* D = (IkState*) inkd_;
    if (D == nullptr || D->pend_n <= 0) return true;
    const int n = D->pend_n;
    D->pend_n = 0;
    D->tap_n = n;
    D->tap_ng = std::min(n, D->rows_g);
    D->tap_nl = std::min(n, kTail);
    D->tap_prompt = false;
    ++D->flushes;
    return inkd_commit(D->pend_p0, n, false, fast_->cs);
}

void Glm5Model::inkd_settle(int64_t pos) {
    auto* D = (IkState*) inkd_;
    if (D == nullptr || D->win_p0 < 0) return;
    const int64_t p0 = D->win_p0;
    const int n = (int) std::max<int64_t>(0, std::min<int64_t>(D->win_T, (pos >= 0 ? pos : pos_) - p0));
    D->win_p0 = -1;
    if (n > 0) {
        D->tap_n = n;   // (the window's first n columns: the accumulators' rows 0..n-1)
        D->tap_ng = std::min(D->tap_ng, n);
        D->tap_nl = std::min(D->tap_nl, n);
        inkd_commit(p0, n, false, fast_->cs);
    } else {
        D->tap_n = 0;
    }
}

void Glm5Model::inkd_window(int64_t p0, int T) {
    auto* D = (IkState*) inkd_;
    if (D == nullptr) return;
    D->win_p0 = p0;
    D->win_T = T;
}

// ------------------------------------------------------------------------------------------------ the draft
bool Glm5Model::inkd_draft(int32_t y, int32_t* drafts, float* conf, std::string& err) {
    auto* D = (IkState*) inkd_;
    for (int k = 0; k < kND; ++k) {   // (the callers' arrays hold kVerifyMax - 1 = 7)
        drafts[k] = -1;
        conf[k] = 0.0f;
    }
    if (D == nullptr) { err = "ink dspark: not loaded"; return false; }
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto t0 = std::chrono::steady_clock::now();
    inkd_settle();
    if (!inkd_flush()) { err = "ink dspark: the deferred rows' commit"; return false; }
    const int64_t pos = pos_;
    if (D->ctx_end != pos || D->frnn_pos != pos - 1) return true;   // (no context for this position: no drafts)
    const int E = g.n_embd, V = D->V, FF = D->ff, ND = D->n_max;
    int ph = 0;
    const auto mark = [&] { if (D->prof) cudaEventRecord(D->ev[ph++], s); };
    mark();
    // the block's rows: the anchor's embedding (rms * token_embd_norm), then mask_embd
    D->y_h[0] = y;
    if (F->emb_tab_d != nullptr) {
        gf::embed_rows_q(pack_emb_type_, F->emb_tab_d, D->y_d, 1, E, D->a, s);
    } else if (F->emb_q8_d != nullptr) {
        gf::embed_rows_q(kQ8_0, F->emb_q8_d, D->y_d, 1, E, D->a, s);
    } else {
        const strata::qt::Traits* tt = strata::qt::traits((strata::qt::Type) pack_emb_type_);
        if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) { err = "ink dspark: the embedding"; return false; }
        tt->to_float(pack_emb_src_ + (size_t) y * strata::qt::row_size((strata::qt::Type) pack_emb_type_, E), D->emb_h, E);
        cudaMemcpyAsync(D->a, D->emb_h, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s);
    }
    k_rms<<<1, 256, 0, s>>>(D->a, E, w_.at("token_embd_norm.weight"), g.norm_eps, E, D->x, E, nullptr);
    for (int i = 1; i < kNB; ++i)
        cudaMemcpyAsync(D->x + (size_t) i * E, D->mask, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, s);
    const float scale = 1.0f / std::sqrt((float) kHD);
    const int nE = kNB * E;
    for (int l = 0; l < kL; ++l) {
        const IkLayer& Ly = D->L[l];
        mark();
        // ---- attention
        k_rms<<<kNB, 256, 0, s>>>(D->x, E, Ly.attn_norm, D->eps, E, D->xn, E, (Q81*) D->xq);
        gf::MvJob j[3] = {{Ly.q.q, D->xq, D->xn, D->q, nullptr, 1.0f, Ly.q.type, E, kQD},
                          {Ly.k.q, D->xq, D->xn, D->kb, nullptr, 1.0f, Ly.k.type, E, kKVD},
                          {Ly.v.q, D->xq, D->xn, D->vb, nullptr, 1.0f, Ly.v.type, E, kKVD}};
        if (!gf::mvn(j, 3, kNB, s)) { err = "ink dspark: q / k / v"; return false; }
        k_norm_rope<<<dim3(kNH, kNB), kHD, 0, s>>>(D->q, kQD, Ly.q_norm, D->eps, (long long) pos, D->inv, nullptr, 0, 0);
        k_norm_rope<<<dim3(kNKV, kNB), kHD, 0, s>>>(D->kb, kKVD, Ly.k_norm, D->eps, (long long) pos, D->inv, nullptr, 0, 0);
        const int64_t lo = std::max<int64_t>(Ly.lo, pos - Ly.win);
        const int n_ctx = (int) std::max<int64_t>(0, pos - lo);
        const int S = n_ctx + kNB;
        int per = std::max(64, ((S + 159) / 160 + kSub - 1) / kSub * kSub);
        int nsplit = (S + per - 1) / per;
        while (nsplit > kMaxSplit) {
            per *= 2;
            nsplit = (S + per - 1) / per;
        }
        k_attn_part<<<dim3(nsplit, kNKV), 256, 0, s>>>(D->q, Ly.rk, Ly.rv, Ly.ring, (long long) lo, n_ctx, D->kb, D->vb,
                                                       per, scale, D->pm, D->pl, D->pacc);
        k_attn_comb<<<dim3(kNH, kNB), kHD, 0, s>>>(D->pm, D->pl, D->pacc, nsplit, D->o, (Q81*) D->oq);
        if (!mv_rows(Ly.o, kQD, E, D->o, D->oq, D->a, kNB, s)) { err = "ink dspark: attn_output"; return false; }
        k_add<<<(nE + 255) / 256, 256, 0, s>>>(D->x, D->a, nE);
        // ---- the FFN
        k_rms<<<kNB, 256, 0, s>>>(D->x, E, Ly.ffn_norm, D->eps, E, D->xn, E, (Q81*) D->xq);
        gf::MvJob ju[2] = {{Ly.gate.q, D->xq, D->xn, D->gt, nullptr, 1.0f, Ly.gate.type, E, FF},
                           {Ly.up.q, D->xq, D->xn, D->up, nullptr, 1.0f, Ly.up.type, E, FF}};
        if (!gf::mvn(ju, 2, kNB, s)) { err = "ink dspark: ffn gate / up"; return false; }
        k_swiglu<<<(kNB * FF + 255) / 256, 256, 0, s>>>(D->gt, D->up, kNB * FF, (Q81*) D->gq);
        if (!mv_rows(Ly.down, FF, E, D->gt, D->gq, D->a, kNB, s)) { err = "ink dspark: ffn_down"; return false; }
        k_add<<<(nE + 255) / 256, 256, 0, s>>>(D->x, D->a, nE);
    }
    // ---- the head: rows 1 .. ND (row 0 is the anchor's)
    mark();
    k_rms<<<ND, 256, 0, s>>>(D->x + E, E, D->out_norm, D->eps, E, D->hn, E, nullptr);
    k_bias_q8<<<(ND * E + 255) / 256, 256, 0, s>>>(D->hn, D->ho_b, E, ND * E, D->xn, (Q81*) D->hq);
    // the static part of joint_proj: [y_s | f_rnn]
    cudaMemcpy2DAsync(D->xz, (size_t) 2 * E * sizeof(float), D->hn, (size_t) E * sizeof(float), (size_t) E * sizeof(float),
                      (size_t) ND, cudaMemcpyDeviceToDevice, s);
    for (int i = 0; i < ND; ++i)
        cudaMemcpyAsync(D->xz + (size_t) i * 2 * E + E, D->frnn, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, s);
    gf::MvJob jz = {D->wsta, nullptr, D->xz, D->zs, nullptr, 1.0f, kBF16, 2 * E, kZ};
    if (!gf::mvn(&jz, 1, ND, s)) { err = "ink dspark: joint_proj"; return false; }
    const WSlot& ow = ws_map_.at("output.weight");
    if (!mv_rows(ow, E, V, D->xn, D->hq, D->logits, ND, s)) { err = "ink dspark: output head"; return false; }
    const int chunk = (V + kTkCh - 1) / kTkCh;
    k_topk_part<<<dim3(kTkCh, ND), 256, (size_t) chunk * sizeof(float), s>>>(D->logits, V, chunk, D->tk_v, D->tk_i);
    mark();
    k_markov<<<1, kMkT, 0, s>>>(D->zs, D->jb, D->wdt, D->w1, D->F, D->tk_v, D->tk_i, kNC, ND, D->y_d, D->softcap, V,
                                D->res_d, D->res_conf_d);
    mark();
    cudaMemcpyAsync(D->res_h, D->res_d, (size_t) ND * sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(D->res_conf_h, D->res_conf_d, (size_t) ND * sizeof(float), cudaMemcpyDeviceToHost, s);
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        err = std::string("ink dspark: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        err = std::string("ink dspark: ") + cudaGetErrorString(e);
        return false;
    }
    for (int i = 0; i < ND; ++i) {
        if (D->res_h[i] < 0) break;
        drafts[i] = D->res_h[i];
        conf[i] = D->res_conf_h[i];
    }
    // STRATA_INK_DSPARK_DUMP=<file> (debug): a line a draft - the position (y's), y, the context rows | the drafts | conf
    if (D->dump != nullptr) {
        std::fprintf(D->dump, "%lld %d %lld |", (long long) pos, y, (long long) (pos - D->L[1].lo));
        for (int i = 0; i < ND; ++i) std::fprintf(D->dump, " %d", drafts[i]);
        std::fprintf(D->dump, " |");
        for (int i = 0; i < ND; ++i) std::fprintf(D->dump, " %.4f", conf[i]);
        std::fprintf(D->dump, "\n");
        std::fflush(D->dump);
    }
    for (int i = 0; D->prof && i < IkState::kPh; ++i) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, D->ev[i], D->ev[i + 1]);
        D->ms_ph[i] += ms;
    }
    ++D->drafts;
    D->ms_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace strata::core
