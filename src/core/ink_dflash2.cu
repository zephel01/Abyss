// src/core/ink_dflash2.cu - Inkling 2 Mini's DFlash 2 drafter (incoai/Inkling-2-Mini-DFlash2, Apache-2.0, converted by
// tools/ink_dflash2_convert.py: arch dflash with the selector's tensors), loaded through STRATA_INK_DSPARK=<the GGUF>
// (inkd_load reads the layout from the file).  It proposes up to 7 tokens of Inkling's verify window as the DSpark
// drafter does, and it reads the same context: ink_dspark.cu's taps (the residual entering layer applications 5 / 23
// / 41 and the head's normed row, visit 48), here the four rows side by side ([f5 | f23 | f41 | f48], 4 E a position)
// instead of DSpark's mixes; inkd_commit hands them over (ikf_rows), inkd_draft asks for the drafts (ikf_draft).
// The arithmetic is Qwen3.8's DFlash 2's (dflash2.cu, written from the same block drafter):
//   * the context: g = rms(fc feat) * enc.output_norm; each layer's K = rope_p(rms_head(wk g) * k_norm), V = wv g, kept
//     in rings of the last 2048 positions (FP16, positions mod 2048 - the drafter's sliding window);
//   * the block: row 0 the pending token's target embedding, rows 1..7 the mask token's (200056), each rms * the
//     target's token_embd_norm (the target's own input rows), at positions pos .. pos + 7; five layers of attn_norm
//     -> the 2-tap conv (w[side][tap][c] = base + attn_conv_proj xn, a weight a group of 16 channels) -> q / k / v,
//     per-head q / k norms, NEOX RoPE base 1e4 over 128 dims -> attention over the context and the whole block (no
//     mask, 32 q heads over 8 kv heads) -> wo -> the conv's side 1 -> the residual; the FFN the same way (SwiGLU
//     6912).  output_norm, the TARGET's output head over the vocabulary's rows: row i proposes position pos + i;
//   * the selector: each row's top 16 logits, score = cap tanh((logit + sum_r succ[c][r] pred[prev][r] gate_i[r]) / cap)
//     (cap 30, the checkpoint's "after_correction"; gate = selector_hidden hn, prev the row before's pick, row 1's the
//     pending token), the greedy walk; a pick's confidence exp(s_best) / (sum of the 16's exp(s) + the vocabulary's
//     other softcapped mass) - with the tail the decode's cost table tells the drafts apart (RTX 5090, a 4K prompt +
//     512: code 220.7 -> 234.5 tok/s, prose 197.5 -> 207.8, Japanese 202.8 -> 220.0; the 16 alone were overconfident:
//     wide windows that kept little);
//   * the draft attention split over the keys (64 ranges a kv head, all 8 rows x 4 heads a block): 1.64 -> 1.53 ms a
//     draft at a full 2048-row window.
// The drafts are proposals: the verify window keeps only the target's own tokens (greedy the same tokens, sampled
// the same distribution - the window's rule, nothing here).
#include "glm_fast_state.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"

#include "strata/qtypes.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;

namespace {
constexpr size_t kQ8 = 36;     // bytes of a q8_1 block (32 values)
constexpr int kL = 5;          // drafter layers
constexpr int kNF = 4;         // taps a position
constexpr int kRing = 2048;    // the context window, positions mod kRing
constexpr int kCh = 256;       // a commit's rows a pass
constexpr int kHD = 128, kNH = 32, kNKV = 8, kKVD = kNKV * kHD, kQD = kNH * kHD, kRank = 256, kTopK = 16, kGrp = 16;
constexpr int kNB = 8;         // y + 7 masks (block_size)
constexpr int kTkCh = 64;      // the top-k's chunks a row
constexpr int kQ8_0 = 8, kF32 = 0;

struct IkfPar {   // a draft's step values the kernels read on the device (one captured graph serves every draft)
    long long pos, lo;
    int n_ctx, pad;
};

struct IkfLayer {
    WSlot q, k, v, o, gate, up, down, acp, fcp;
    float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *acb = nullptr,
          *fcb = nullptr;
    __half *rk = nullptr, *rv = nullptr;   // [kRing][kKVD]
};

struct IkfState {
    IkfLayer L[kL];
    WSlot fc, sel_h;
    const uint8_t *sel_pred = nullptr, *sel_succ = nullptr;   // Q8_0 rows [Vs][256]
    int Vs = 0, V = 0, ff = 6912, nb = kNB, mask = 200056, cap_mode = 1;   // cap_mode: 1 after the correction, 0 before
    float *enc_norm = nullptr, *out_norm = nullptr, *mask_row = nullptr;
    float eps = 1e-6f, rope_base = 10000.0f, softcap = 30.0f;
    int win = kRing;
    bool tail = true;
    bool split = true;   // (the attention split over the keys: STRATA_INK_DFLASH_SPLIT=0 k_attn, a block a row)
    float *pm = nullptr, *pl = nullptr, *pacc = nullptr;    // (the confidence with the vocabulary's tail: STRATA_INK_DFLASH_TAIL=0 the 16 alone)
    float2* lse = nullptr;   // (the context positions a draft reads: STRATA_INK_DFLASH_CTX, at most kRing)
    // the commit's pass
    void* featq = nullptr;
    float *gpre = nullptr, *gv = nullptr, *kk = nullptr, *vv = nullptr;
    void* gq = nullptr;
    float2* rope_c = nullptr;
    // the draft
    float *x = nullptr, *xn = nullptr, *dyn = nullptr, *xc = nullptr, *q = nullptr, *kb = nullptr, *vb = nullptr,
          *o = nullptr, *a = nullptr, *gt = nullptr, *up = nullptr, *f = nullptr, *hn = nullptr, *logits = nullptr,
          *gate = nullptr, *tk_v = nullptr, *res_conf_d = nullptr;
    void *xnq = nullptr, *xcq = nullptr, *oq = nullptr, *gq2 = nullptr, *hnq = nullptr;
    int *tk_i = nullptr, *res_d = nullptr;
    int *y_h = nullptr, *y_d = nullptr, *res_h = nullptr;
    float *res_conf_h = nullptr, *emb_h = nullptr;
    float2* rope_d = nullptr;
    IkfPar *par_h = nullptr, *par_d = nullptr;
    bool graph_on = true;   // STRATA_INK_DFLASH_GRAPH=0: the draft's kernels launched one by one
    cudaStream_t gs = nullptr;
    cudaGraphExec_t gexec = nullptr;
    size_t att_smem = 0;
    int launch_err0 = 0;
    std::vector<void*> allocs;
    size_t bytes = 0;
    FILE* dump = nullptr;
    uint64_t drafts = 0, drafted = 0, commits = 0, rows = 0;
    double ms_draft = 0.0;
    uint64_t hist[kNB] = {};
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
__device__ float block_sum(float v, float* sh) {   // (blockDim a multiple of 32; every thread gets it)
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float r = lane < nw ? sh[lane] : 0.0f;
    return warp_sum(r);
}

struct Q81 {   // glm_fast's q8_1 block: the scale and the sum as FP16, 32 int8
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

// y[row] = rms(x[row]) * w (256 threads, E a multiple of 256 up to 256 kRmsPer), and the row's q8_1 blocks (null: none)
constexpr int kRmsPer = 16;
__global__ void __launch_bounds__(256) k_rms(const float* __restrict__ x, int ldx, const float* __restrict__ w,
                                             float eps, int E, float* __restrict__ y, int ldy, Q81* __restrict__ yq) {
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

// the block's first row: rms(emb) * w (the anchor), the other rows the mask row; one block of 256 a row
__global__ void __launch_bounds__(256) k_block_rows(const float* __restrict__ emb, const float* __restrict__ w,
                                                    float eps, int E, const float* __restrict__ mask_row, int nb,
                                                    float* __restrict__ x) {
    __shared__ float sh[32];
    const int r = blockIdx.x;
    float* xr = x + (size_t) r * E;
    if (r > 0) {
        for (int i = threadIdx.x; i < E; i += 256) xr[i] = mask_row[i];
        return;
    }
    float ss = 0.0f;
    for (int i = threadIdx.x; i < E; i += 256) ss += emb[i] * emb[i];
    ss = block_sum(ss, sh);
    const float sc = rsqrtf(ss / (float) E + eps);
    for (int i = threadIdx.x; i < E; i += 256) xr[i] = emb[i] * sc * w[i];
    (void) nb;
}

// the two-tap conv over the block's rows: w_tap[c] = base[tap][c] + dyn[t][off + tap * G + c / grp], out[t] = w0 x[t] +
// w1 x[t - 1] (x[-1] = 0); res != null: res[t] += out[t] instead of out (and out's q8_1 blocks when outq is not null)
__global__ void k_conv(const float* __restrict__ x, const float* __restrict__ dyn, int dyn_ld, int off, int G, int grp,
                       const float* __restrict__ base, int E, float* __restrict__ out, float* __restrict__ res,
                       Q81* __restrict__ outq) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= E) return;
    const float* d = dyn + (size_t) t * dyn_ld + off;
    const float w0 = base[c] + d[c / grp], w1 = base[E + c] + d[G + c / grp];
    const float prev = t > 0 ? x[(size_t) (t - 1) * E + c] : 0.0f;
    const float v = w0 * x[(size_t) t * E + c] + w1 * prev;
    if (res != nullptr) {
        res[(size_t) t * E + c] += v;
    } else {
        out[(size_t) t * E + c] = v;
        if (outq != nullptr) q81_store_warp(v, outq + ((size_t) t * E + c) / 32, threadIdx.x & 31);
    }
}

// the RoPE angles of rows pos0 .. pos0 + rows - 1 (par: pos0 from the device): tab[row][i] = (cos, sin) in double
__global__ void k_rope_tab(long long pos0, const IkfPar* __restrict__ par, int rows, float base, float2* __restrict__ tab) {
    if (par != nullptr) pos0 = par->pos;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= rows * (kHD / 2)) return;
    const int row = k / (kHD / 2), i = k % (kHD / 2);
    const double inv = pow((double) base, -2.0 * (double) i / (double) kHD);
    double sn, cs;
    sincos((double) (pos0 + row) * inv, &sn, &cs);
    tab[k] = make_float2((float) cs, (float) sn);
}

// per head (blockIdx.x) of row blockIdx.y: rms * w (w null: none), NEOX RoPE from tab (null: none); in place (ring
// null) or as FP16 into ring row (pos0 + row) % ring_n.  Heads nh0 and up are a second tensor's (x2 / w2 / ld2)
__global__ void k_norm_rope(float* __restrict__ x, int ld, const float* __restrict__ w, float eps, long long pos0,
                            const float2* __restrict__ tab, __half* __restrict__ ring, int ring_n, int ring_ld,
                            int nh0, float* __restrict__ x2, int ld2, const float* __restrict__ w2) {
    __shared__ float sh[32];
    __shared__ float buf[kHD];
    const int d = threadIdx.x, row = blockIdx.y;
    int h = blockIdx.x;
    if (h >= nh0) {
        h -= nh0;
        x = x2;
        ld = ld2;
        w = w2;
    }
    float* xr = x + (size_t) row * ld + (size_t) h * kHD;
    float v = xr[d];
    if (w != nullptr) {
        const float ss = block_sum(v * v, sh);
        v = v * rsqrtf(ss / (float) kHD + eps) * w[d];
    }
    buf[d] = v;
    __syncthreads();
    if (tab != nullptr) {
        const int i = d & (kHD / 2 - 1);
        const float2 cs = tab[(size_t) row * (kHD / 2) + i];
        const float a = buf[i], b = buf[i + kHD / 2];
        v = d < kHD / 2 ? a * cs.x - b * cs.y : a * cs.y + b * cs.x;
    }
    if (ring != nullptr) ring[(size_t) ((pos0 + row) % ring_n) * ring_ld + (size_t) h * kHD + d] = __float2half(v);
    else xr[d] = v;
}

// attention of block row blockIdx.y for the 4 q heads of kv head blockIdx.x over the context's n_ctx ring rows
// (positions par->lo ..) and the block's nb rows (dflash2.cu's kernel: K / V chunks through shared memory, read once
// for the 4 heads; the scores in dynamic shared memory [4][n_ctx + nb])
constexpr int kAttCh = 128, kAttT = 512, kAttPer = kAttCh * kHD / 8 / kAttT;
constexpr int kHpk = kNH / kNKV;
static_assert(kHpk == 4 && kAttT == kHpk * kHD, "the attention's layout");
__device__ __forceinline__ void attn_chunk_load(const __half* __restrict__ ring, int64_t ring0, int ring_n, int kvh,
                                                int c0, int n_ctx, uint4 (&r)[kAttPer]) {
#pragma unroll
    for (int u = 0; u < kAttPer; ++u) {
        const int piece = u * kAttT + threadIdx.x, row = piece / (kHD / 8), col = piece % (kHD / 8);
        const int j = c0 + row;
        if (j < n_ctx) {
            int64_t rr = ring0 + j;
            if (rr >= ring_n) rr -= ring_n;
            r[u] = *reinterpret_cast<const uint4*>(ring + (size_t) rr * kKVD + (size_t) kvh * kHD + col * 8);
        } else {
            r[u] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}
__device__ __forceinline__ void attn_chunk_store(__half* __restrict__ buf, const uint4 (&r)[kAttPer]) {
#pragma unroll
    for (int u = 0; u < kAttPer; ++u) reinterpret_cast<uint4*>(buf)[u * kAttT + threadIdx.x] = r[u];
}
__device__ __forceinline__ float attn_dot4(const float4 (&qv)[kHpk], const float4 kv, int lane) {
    float p[kHpk];
#pragma unroll
    for (int g = 0; g < kHpk; ++g) p[g] = qv[g].x * kv.x + qv[g].y * kv.y + qv[g].z * kv.z + qv[g].w * kv.w;
    const bool hi = (lane & 16) != 0, m8 = (lane & 8) != 0;
    const float r1 = __shfl_xor_sync(0xffffffffu, hi ? p[0] : p[1], 16);
    const float r2 = __shfl_xor_sync(0xffffffffu, hi ? p[2] : p[3], 16);
    const float x = (hi ? p[1] : p[0]) + r1, y = (hi ? p[3] : p[2]) + r2;
    const float r3 = __shfl_xor_sync(0xffffffffu, m8 ? x : y, 8);
    float z = (m8 ? y : x) + r3;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    return z;
}
__global__ void __launch_bounds__(kAttT) k_attn(const float* __restrict__ q, const __half* __restrict__ rk,
                                                const __half* __restrict__ rv, const IkfPar* __restrict__ par, int ring_n,
                                                const float* __restrict__ kb, const float* __restrict__ vb, int nb,
                                                float scale, float* __restrict__ out, Q81* __restrict__ outq) {
    extern __shared__ float sc[];
    __shared__ float sh[32], sh2[32];
    __shared__ float lsum[kHpk], mx[kHpk];
    __shared__ __align__(16) __half cb[kAttCh * kHD];
    constexpr int kW = kAttT / 32;
    const int kvh = blockIdx.x, t = blockIdx.y;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const long long lo = par->lo;
    const int n_ctx = par->n_ctx;
    const int S = n_ctx + nb;
    const int my_g = ((lane >> 4) & 1) | (((lane >> 3) & 1) << 1);
    const bool writer = (lane & 7) == 0;
    float4 qv[kHpk];
#pragma unroll
    for (int g = 0; g < kHpk; ++g)
        qv[g] = *reinterpret_cast<const float4*>(q + (size_t) t * kQD + (size_t) (kvh * kHpk + g) * kHD + lane * 4);
    const int64_t ring0 = lo % ring_n;
    uint4 pf[kAttPer];
    if (n_ctx > 0) attn_chunk_load(rk, ring0, ring_n, kvh, 0, n_ctx, pf);
    for (int c0 = 0; c0 < n_ctx; c0 += kAttCh) {
        __syncthreads();
        attn_chunk_store(cb, pf);
        __syncthreads();
        if (c0 + kAttCh < n_ctx) attn_chunk_load(rk, ring0, ring_n, kvh, c0 + kAttCh, n_ctx, pf);
        const int n = min(kAttCh, n_ctx - c0);
        for (int i = wp; i < n; i += kW) {
            const __half* kp = cb + (size_t) i * kHD + lane * 4;
            const __half2 k01 = *reinterpret_cast<const __half2*>(kp), k23 = *reinterpret_cast<const __half2*>(kp + 2);
            const float2 f01 = __half22float2(k01), f23 = __half22float2(k23);
            const float dot = attn_dot4(qv, make_float4(f01.x, f01.y, f23.x, f23.y), lane);
            if (writer) sc[(size_t) my_g * S + c0 + i] = dot * scale;
        }
    }
    for (int j = n_ctx + wp; j < S; j += kW) {
        const float4 kv = *reinterpret_cast<const float4*>(kb + (size_t) (j - n_ctx) * kKVD + (size_t) kvh * kHD + lane * 4);
        const float dot = attn_dot4(qv, kv, lane);
        if (writer) sc[(size_t) my_g * S + j] = dot * scale;
    }
    if (n_ctx > 0) attn_chunk_load(rv, ring0, ring_n, kvh, 0, n_ctx, pf);
    __syncthreads();
    const int g = threadIdx.x >> 7, tg = threadIdx.x & 127;
    float* sr = sc + (size_t) g * S;
    float m = -INFINITY;
    for (int j = tg; j < S; j += 128) m = fmaxf(m, sr[j]);
    m = warp_max(m);
    if (lane == 0) sh[wp] = m;
    __syncthreads();
    if (tg == 0) mx[g] = fmaxf(fmaxf(sh[4 * g], sh[4 * g + 1]), fmaxf(sh[4 * g + 2], sh[4 * g + 3]));
    __syncthreads();
    m = mx[g];
    float l = 0.0f;
    for (int j = tg; j < S; j += 128) {
        const float e = __expf(sr[j] - m);
        sr[j] = e;
        l += e;
    }
    l = warp_sum(l);
    if (lane == 0) sh2[wp] = l;
    __syncthreads();
    if ((wp & 3) == 0) {
        float r = lane < 4 ? sh2[wp + lane] : 0.0f;
        r = warp_sum(r);
        if (lane == 0) lsum[g] = r;
    }
    const int d = tg;
    float acc = 0.0f;
    for (int c0 = 0; c0 < n_ctx; c0 += kAttCh) {
        __syncthreads();
        attn_chunk_store(cb, pf);
        __syncthreads();
        if (c0 + kAttCh < n_ctx) attn_chunk_load(rv, ring0, ring_n, kvh, c0 + kAttCh, n_ctx, pf);
        const int n = min(kAttCh, n_ctx - c0);
        const float* sp = sr + c0;
        if (n == kAttCh) {
#pragma unroll 16
            for (int i = 0; i < kAttCh; ++i) acc += sp[i] * __half2float(cb[(size_t) i * kHD + d]);
        } else {
            for (int i = 0; i < n; ++i) acc += sp[i] * __half2float(cb[(size_t) i * kHD + d]);
        }
    }
    __syncthreads();
    for (int jj = 0; jj < nb; ++jj) acc += sr[n_ctx + jj] * vb[(size_t) jj * kKVD + (size_t) kvh * kHD + d];
    const size_t o = (size_t) t * kQD + (size_t) (kvh * kHpk + g) * kHD + d;
    const float v = acc / lsum[g];
    out[o] = v;
    q81_store_warp(v, outq + o / 32, lane);
}

// the attention split over the keys (STRATA_INK_DFLASH_SPLIT, default on): block (split, kv head) takes keys
// [split * per, ..) of the context's n_ctx ring rows then the block's nb rows, for all nb rows x 4 q heads of that kv
// head at once (each K / V row read once a block, not once a row); online softmax over sub-chunks of kSub keys ->
// pm / pl / pacc[split][kvh][32][(128)], k_attn_comb sums the splits.  Another sum order than k_attn: the drafts are
// proposals, the window keeps the target's tokens
constexpr int kSub = 16, kSplit = 64, kQn = kNB * kHpk;   // (32 queries a kv head)
__global__ void __launch_bounds__(256) k_attn_part(const float* __restrict__ q, const __half* __restrict__ rk,
                                                   const __half* __restrict__ rv, const IkfPar* __restrict__ par,
                                                   int ring_n, const float* __restrict__ kb, const float* __restrict__ vb,
                                                   float scale, float* __restrict__ pm, float* __restrict__ pl,
                                                   float* __restrict__ pacc) {
    __shared__ float qs[kQn][kHD];
    __shared__ float ks[kSub][kHD + 1];
    __shared__ float vs[kSub][kHD];
    __shared__ float sc[kQn][kSub];
    __shared__ float mrow[kQn], lrow[kQn], arow[kQn];
    const int sp = blockIdx.x, kvh = blockIdx.y, t = threadIdx.x;
    const long long lo = par->lo;
    const int n_ctx = par->n_ctx, S = n_ctx + kNB;
    const int per = ((S + kSplit - 1) / kSplit + kSub - 1) / kSub * kSub;
    const int j0 = sp * per, j1 = min(S, j0 + per);
    for (int i = t; i < kQn * kHD; i += 256) {
        const int qi = i / kHD, d = i % kHD, r = qi / kHpk, g = qi % kHpk;
        qs[qi][d] = q[(size_t) r * kQD + (size_t) (kvh * kHpk + g) * kHD + d];
    }
    if (t < kQn) {
        mrow[t] = -INFINITY;
        lrow[t] = 0.0f;
    }
    const int dd = t % kHD, h2 = t / kHD;   // (the values' thread: dim dd of queries h2, h2 + 2, ..)
    float acc[kQn / 2];
#pragma unroll
    for (int k = 0; k < kQn / 2; ++k) acc[k] = 0.0f;
    const int64_t ring0 = lo % ring_n;
    for (int c0 = j0; c0 < j1; c0 += kSub) {
        const int n = min(kSub, j1 - c0);
        __syncthreads();
        for (int i = t; i < kSub * kHD; i += 256) {
            const int jj = i / kHD, d = i % kHD, j = c0 + jj;
            float kv = 0.0f, vv = 0.0f;
            if (jj < n) {
                if (j < n_ctx) {
                    int64_t rr = ring0 + j;
                    if (rr >= ring_n) rr -= ring_n;
                    const size_t row = (size_t) rr * kKVD + (size_t) kvh * kHD + d;
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
        {   // the scores: thread t's key t % 16 against queries 2 (t / 16), + 1
            const int jj = t % kSub, q0 = (t / kSub) * 2;
            float s0 = 0.0f, s1 = 0.0f;
#pragma unroll 8
            for (int d = 0; d < kHD; ++d) {
                const float kv = ks[jj][d];
                s0 += qs[q0][d] * kv;
                s1 += qs[q0 + 1][d] * kv;
            }
            const bool ok = jj < n;
            sc[q0][jj] = ok ? s0 * scale : -INFINITY;
            sc[q0 + 1][jj] = ok ? s1 * scale : -INFINITY;
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

// the splits' partial sums -> out[row][head * 128 + d] and its q8_1 blocks; block (head, row), 128 threads
__global__ void __launch_bounds__(kHD) k_attn_comb(const float* __restrict__ pm, const float* __restrict__ pl,
                                                   const float* __restrict__ pacc, float* __restrict__ out,
                                                   Q81* __restrict__ outq) {
    const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    const int kvh = h / kHpk, qi = r * kHpk + h % kHpk;
    float M = -INFINITY;
    for (int s = 0; s < kSplit; ++s) M = fmaxf(M, pm[((size_t) s * kNKV + kvh) * kQn + qi]);
    float L = 0.0f, acc = 0.0f;
    for (int s = 0; s < kSplit; ++s) {
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

__device__ __forceinline__ float capf(float v, float cap) { return cap > 0.0f ? cap * tanhf(v / cap) : v; }

__global__ void k_swiglu(float* __restrict__ g, const float* __restrict__ u, int n, Q81* __restrict__ gq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float a = g[i];
        const float v = a / (1.0f + __expf(-a)) * u[i];
        g[i] = v;
        q81_store_warp(v, gq + i / 32, threadIdx.x & 31);
    }
}

// rows [r][V] -> each chunk's top kTopK (value, index): the chunk's floats in shared memory, kTopK block argmaxes
// (ties to the lower index, NaN as -inf); and the chunk's softcapped mass (max, sum exp(cap(v) - max)) into lse
__global__ void __launch_bounds__(256) k_topk_part(const float* __restrict__ logits, int V, int chunk, float cap,
                                                   float* __restrict__ tv, int* __restrict__ ti, float2* __restrict__ lse) {
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
    {   // the chunk's softcapped mass (before the top-k takes its values out)
        float m = -INFINITY;
        for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, vals[i]);
        m = warp_max(m);
        if (lane == 0) bv[wp] = m;
        __syncthreads();
        m = bv[0];
        for (int w2 = 1; w2 < 8; ++w2) m = fmaxf(m, bv[w2]);
        const float mc = m == -INFINITY ? -INFINITY : capf(m, cap);
        float e = 0.0f;
        if (mc != -INFINITY)
            for (int i = threadIdx.x; i < n; i += blockDim.x)
                if (vals[i] != -INFINITY) e += __expf(capf(vals[i], cap) - mc);
        e = warp_sum(e);
        __syncthreads();
        if (lane == 0) bv[wp] = e;
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = 0.0f;
            for (int w2 = 0; w2 < 8; ++w2) t += bv[w2];
            lse[(size_t) r * gridDim.x + ch] = make_float2(mc, t);
        }
        __syncthreads();
    }
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

__device__ __forceinline__ float q8_0_at(const uint8_t* row, int r) {
    const uint8_t* b = row + (size_t) (r >> 5) * 34;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    return d * (float) (int8_t) b[2 + (r & 31)];
}

// the greedy walk (one block of 256): rows 0 .. rows - 1 (block rows 1 ..), each row's top kTopK out of its chunks'
// candidates, the selector's scores (softcapped after the correction, cap_mode 1, or the logits before it, 0), the
// pick and its confidence - its share of the 16 scores plus the vocabulary's other softcapped mass (lse: the chunks'
// (max, sum), the tail = the row's mass less its top 16's); stops below p_min.  out: n, then the tokens; conf: the
// picks' confidences
__global__ void __launch_bounds__(256) k_select(const float* __restrict__ tv, const int* __restrict__ ti,
                                                const float2* __restrict__ lse, int n_ch, int n_cand,
                                                const float* __restrict__ gate, const uint8_t* __restrict__ pred,
                                                const uint8_t* __restrict__ succ, int Vs, const int* __restrict__ y,
                                                int rows, float cap, int cap_mode, float p_min, int* __restrict__ out,
                                                float* __restrict__ conf) {
    __shared__ float cv[kTkCh * kTopK];
    __shared__ int ci[kTkCh * kTopK];
    __shared__ float pg[kRank];
    __shared__ float top_v[kTopK], score[kTopK];
    __shared__ int top_i[kTopK];
    __shared__ float bv[8];
    __shared__ int bi[8];
    __shared__ int prev, stop;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const size_t rb = (size_t) (kRank / 32) * 34;
    if (threadIdx.x == 0) {
        prev = y[0];
        stop = 0;
        out[0] = 0;
    }
    __syncthreads();
    for (int r = 0; r < rows; ++r) {
        for (int i = threadIdx.x; i < n_cand; i += blockDim.x) {
            cv[i] = tv[(size_t) r * n_cand + i];
            ci[i] = ti[(size_t) r * n_cand + i];
            if (ci[i] < 0 || ci[i] >= Vs) cv[i] = -INFINITY;
        }
        const bool pv = prev >= 0 && prev < Vs;
        for (int i = threadIdx.x; i < kRank; i += blockDim.x)
            pg[i] = pv ? q8_0_at(pred + (size_t) prev * rb, i) * gate[(size_t) r * kRank + i] : 0.0f;
        __syncthreads();
        for (int k = 0; k < kTopK; ++k) {
            float v = -INFINITY;
            int idx = 0x7fffffff;
            for (int i = threadIdx.x; i < n_cand; i += blockDim.x)
                if (cv[i] > v || (cv[i] == v && i < idx)) { v = cv[i]; idx = i; }
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
                top_v[k] = bvv;
                top_i[k] = bii < n_cand ? ci[bii] : 0;
                if (bii < n_cand) cv[bii] = -INFINITY;
            }
            __syncthreads();
        }
        for (int c = 2 * wp; c < 2 * wp + 2; ++c) {   // (warp wp: candidates 2 wp, 2 wp + 1)
            const uint8_t* srow = succ + (size_t) top_i[c] * rb;
            float s = 0.0f;
            for (int i = lane; i < kRank; i += 32) s += q8_0_at(srow, i) * pg[i];
            s = warp_sum(s);
            if (lane == 0) {
                const float u = top_v[c];
                score[c] = u == -INFINITY ? -INFINITY : cap_mode == 1 ? capf(u + s, cap) : capf(u, cap) + s;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int b = 0;
            for (int c = 1; c < kTopK; ++c)
                if (score[c] > score[b]) b = c;
            const float ref = score[b];
            float z = 0.0f, m16 = 0.0f, all = 0.0f;
            for (int c = 0; c < kTopK; ++c)
                if (score[c] != -INFINITY) {
                    z += expf(score[c] - ref);
                    m16 += expf(capf(top_v[c], cap) - ref);
                }
            if (lse != nullptr) {
                for (int c = 0; c < n_ch; ++c) {
                    const float2 l = lse[(size_t) r * n_ch + c];
                    if (l.x != -INFINITY) all += l.y * expf(l.x - ref);
                }
                z += fmaxf(0.0f, all - m16);
            }
            const float p = ref == -INFINITY ? 0.0f : 1.0f / z;
            if (p <= 0.0f || p < p_min) {
                stop = 1;
            } else {
                out[1 + r] = top_i[b];
                conf[r] = p;
                out[0] = r + 1;
                prev = top_i[b];
            }
        }
        __syncthreads();
        if (stop) break;
    }
}

}  // namespace

// ------------------------------------------------------------------------------------------------ load
bool Glm5Model::ikf_load(const strata::GgufFile& G, const char* path, std::string& why) {
    const Glm5Geometry& g = g_;
    auto* D = new IkfState();
    ikf_ = D;
    const auto num = [&](const std::string& key, double def) {
        const strata::MetaValue* v = G.get(key);
        return v != nullptr && v->is_num() ? v->num() : def;
    };
    const int E = g.n_embd, V = g.n_vocab;
    D->V = V;
    D->ff = (int) num("dflash.feed_forward_length", 0);
    D->eps = (float) num("dflash.attention.layer_norm_rms_epsilon", 1e-6);
    D->rope_base = (float) num("dflash.rope.freq_base", 10000.0);
    D->softcap = (float) num("dflash.final_logit_softcapping", 30.0);
    if (const strata::MetaValue* m = G.get("dflash.selector_softcap_mode"); m != nullptr && m->s == "before_correction")
        D->cap_mode = 0;
    D->mask = (int) num("tokenizer.ggml.mask_token_id", 200056.0);
    const int FF = D->ff;
    if ((int) num("dflash.block_count", 0) != kL || (int) num("dflash.embedding_length", 0) != E || FF <= 0 ||
        FF % 256 != 0 || (int) num("dflash.attention.head_count", 0) != kNH ||
        (int) num("dflash.attention.head_count_kv", 0) != kNKV || (int) num("dflash.attention.key_length", 0) != kHD ||
        (int) num("dflash.attention.value_length", 0) != kHD || (int) num("dflash.conv_kernel_size", 0) != 2 ||
        (int) num("dflash.conv_group_size", 0) != kGrp || (int) num("dflash.selector_rank", 0) != kRank ||
        (int) num("dflash.selector_top_k", 0) != kTopK || (int) num("dflash.attention.sliding_window", 0) != kRing ||
        (int) num("dflash.block_size", 0) != kNB || E % kGrp != 0 || E % 256 != 0 || E > 256 * kRmsPer ||
        D->mask < 0 || D->mask >= V) {
        why = "the DFlash 2 layout (5 layers over the 4 taps, the target's width, 32 / 8 heads of 128, an ff of 256s, a "
              "2-tap conv in groups of 16, selector rank 256 top-16, a 2048-row window, block 8)";
        return false;
    }
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
    const int G2 = 4 * (E / kGrp);   // conv_proj's rows: [side][tap][group]
    for (int l = 0; l < kL && ok; ++l) {
        IkfLayer& Ly = D->L[l];
        const std::string P = "blk." + std::to_string(l) + ".";
        raw(P + "attn_q.weight", E, kQD, Ly.q);
        raw(P + "attn_k.weight", E, kKVD, Ly.k);
        raw(P + "attn_v.weight", E, kKVD, Ly.v);
        raw(P + "attn_output.weight", kQD, E, Ly.o);
        raw(P + "ffn_gate.weight", E, FF, Ly.gate);
        raw(P + "ffn_up.weight", E, FF, Ly.up);
        raw(P + "ffn_down.weight", FF, E, Ly.down);
        raw(P + "attn_conv_proj.weight", E, G2, Ly.acp);
        raw(P + "ffn_conv_proj.weight", E, G2, Ly.fcp);
        Ly.attn_norm = f32(P + "attn_norm.weight", {uE});
        Ly.ffn_norm = f32(P + "ffn_norm.weight", {uE});
        Ly.q_norm = f32(P + "attn_q_norm.weight", {(uint64_t) kHD});
        Ly.k_norm = f32(P + "attn_k_norm.weight", {(uint64_t) kHD});
        Ly.acb = f32(P + "attn_conv_base", {uE, 2, 2});
        Ly.fcb = f32(P + "ffn_conv_base", {uE, 2, 2});
        Ly.rk = (__half*) dev((size_t) kRing * kKVD * sizeof(__half));
        Ly.rv = (__half*) dev((size_t) kRing * kKVD * sizeof(__half));
    }
    raw("fc.weight", (int64_t) kNF * E, E, D->fc);
    raw("selector_hidden.weight", E, kRank, D->sel_h);
    D->enc_norm = f32("enc.output_norm.weight", {uE});
    D->out_norm = f32("output_norm.weight", {uE});
    {   // the selector's codebooks: Q8_0 rows of 256, at least the target's vocabulary
        const strata::TensorInfo* tp = G.find("selector_predecessor.weight");
        D->Vs = tp != nullptr && tp->shape.size() == 2 ? (int) tp->shape[1] : 0;
        if (D->Vs < V) {
            if (why.empty()) why = "selector_predecessor.weight (Q8_0 256 x at least the vocabulary)";
            ok = false;
        }
        for (int which = 0; which < 2 && ok; ++which) {
            const char* n = which == 0 ? "selector_predecessor.weight" : "selector_successor.weight";
            const strata::TensorInfo* t = find(n, kQ8_0, {(uint64_t) kRank, (uint64_t) D->Vs});
            if (t == nullptr) { ok = false; break; }
            const size_t bytes = gf::row_bytes(kQ8_0, kRank) * (size_t) D->Vs;
            void* p = dev(bytes);
            if (p != nullptr) ok = cudaMemcpy(p, G.tensor_data(*t), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
            (which == 0 ? D->sel_pred : D->sel_succ) = (const uint8_t*) p;
        }
    }
    if (!ok) {
        if (why.empty()) why = "the weights did not upload (VRAM?)";
        return false;
    }
    const auto ow_it = ws_map_.find("output.weight");
    const auto tn_it = w_.find("token_embd_norm.weight");
    const strata::qt::Traits* tt = strata::qt::traits((strata::qt::Type) pack_emb_type_);
    if (ow_it == ws_map_.end() || tn_it == w_.end() || tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        why = "the target's output head, token_embd_norm or embedding";
        return false;
    }
    if (!gf::mvn_supported(ow_it->second.type) && ow_it->second.type != kF32) {
        why = "the target's output.weight type " + std::to_string(ow_it->second.type);
        return false;
    }
    // the buffers
    const auto fb = [&](float*& p, size_t n) { p = (float*) dev(n * sizeof(float)); };
    const auto bb = [&](void*& p, size_t n) { p = dev(n); };
    const size_t FE = (size_t) kNF * E, NB = (size_t) kNB;
    bb(D->featq, (size_t) kCh * FE / 32 * kQ8);
    fb(D->gpre, (size_t) kCh * E);
    fb(D->gv, (size_t) kCh * E);
    bb(D->gq, (size_t) kCh * E / 32 * kQ8);
    fb(D->kk, (size_t) kL * kCh * kKVD);
    fb(D->vv, (size_t) kL * kCh * kKVD);
    D->rope_c = (float2*) dev((size_t) kCh * (kHD / 2) * sizeof(float2));
    fb(D->mask_row, E);
    fb(D->x, NB * E);
    fb(D->xn, NB * E);
    bb(D->xnq, NB * E / 32 * kQ8);
    fb(D->dyn, NB * G2);
    fb(D->xc, NB * E);
    bb(D->xcq, NB * E / 32 * kQ8);
    fb(D->q, NB * kQD);
    fb(D->kb, NB * kKVD);
    fb(D->vb, NB * kKVD);
    fb(D->o, NB * kQD);
    bb(D->oq, NB * kQD / 32 * kQ8);
    fb(D->a, NB * E);
    fb(D->gt, NB * FF);
    fb(D->up, NB * FF);
    bb(D->gq2, NB * FF / 32 * kQ8);
    fb(D->f, NB * E);
    fb(D->hn, NB * E);
    bb(D->hnq, NB * E / 32 * kQ8);
    fb(D->logits, NB * (size_t) V);
    fb(D->gate, NB * kRank);
    fb(D->tk_v, NB * kTkCh * kTopK);
    D->tk_i = (int*) dev(NB * kTkCh * kTopK * sizeof(int));
    D->lse = (float2*) dev(NB * kTkCh * sizeof(float2));
    fb(D->pm, (size_t) kSplit * kNKV * kQn);
    fb(D->pl, (size_t) kSplit * kNKV * kQn);
    fb(D->pacc, (size_t) kSplit * kNKV * kQn * kHD);
    D->rope_d = (float2*) dev(NB * (kHD / 2) * sizeof(float2));
    D->par_d = (IkfPar*) dev(sizeof(IkfPar));
    D->res_d = (int*) dev(16 * sizeof(int));
    fb(D->res_conf_d, 16);
    if (!ok) {
        why = "the buffers did not allocate (VRAM?)";
        return false;
    }
    ok = cudaHostAlloc((void**) &D->y_h, 64, cudaHostAllocMapped) == cudaSuccess &&
         cudaHostGetDevicePointer((void**) &D->y_d, D->y_h, 0) == cudaSuccess &&
         cudaHostAlloc((void**) &D->res_h, 64, cudaHostAllocDefault) == cudaSuccess &&
         cudaHostAlloc((void**) &D->res_conf_h, 64, cudaHostAllocDefault) == cudaSuccess &&
         cudaHostAlloc((void**) &D->emb_h, (size_t) E * sizeof(float), cudaHostAllocDefault) == cudaSuccess &&
         cudaHostAlloc((void**) &D->par_h, sizeof(IkfPar), cudaHostAllocDefault) == cudaSuccess;
    if (!ok) {
        why = "the pinned buffers";
        return false;
    }
    {   // the mask row once: rms(emb(mask)) * token_embd_norm (the target's eps), as the anchor's row each draft
        tt->to_float(pack_emb_src_ + (size_t) D->mask * strata::qt::row_size((strata::qt::Type) pack_emb_type_, E),
                     D->emb_h, E);
        ok = cudaMemcpy(D->a, D->emb_h, (size_t) E * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
        if (ok) {
            k_rms<<<1, 256>>>(D->a, E, tn_it->second, g.norm_eps, E, D->mask_row, E, nullptr);
            ok = cudaDeviceSynchronize() == cudaSuccess;
        }
    }
    D->att_smem = (size_t) kHpk * (kRing + kNB) * sizeof(float);
    if (ok && D->att_smem > 16 * 1024)
        ok = cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) D->att_smem) == cudaSuccess;
    if (const char* v = getenv("STRATA_INK_DFLASH_GRAPH")) D->graph_on = std::atoi(v) != 0;
    if (const char* v = getenv("STRATA_INK_DFLASH_SPLIT")) D->split = std::atoi(v) != 0;
    if (const char* v = getenv("STRATA_INK_DFLASH_TAIL")) D->tail = std::atoi(v) != 0;
    if (const char* v = getenv("STRATA_INK_DFLASH_CTX")) D->win = std::max(0, std::min(kRing, std::atoi(v)));
    if (ok && D->graph_on) ok = cudaStreamCreateWithFlags(&D->gs, cudaStreamNonBlocking) == cudaSuccess;
    if (!ok) {
        cudaGetLastError();
        why = "the mask row / the attention's shared memory / the graph's stream";
        return false;
    }
    D->launch_err0 = gf::launch_errors();
    if (const char* v = getenv("STRATA_INK_DSPARK_DUMP"); v != nullptr && v[0]) D->dump = std::fopen(v, "w");
    std::fprintf(stderr, "ink dflash2: %s - %d layers over taps 5/23/41/48, up to %d drafts a step, a %d-position "
                         "window, selector softcap %.0f %s the correction; %.2f GB on CUDA%d\n",
                 path, kL, kNB - 1, kRing, (double) D->softcap, D->cap_mode == 1 ? "after" : "before",
                 (double) D->bytes / 1073741824.0, dev_);
    return true;
}

void Glm5Model::ikf_destroy() {
    auto* D = (IkfState*) ikf_;
    if (D == nullptr) return;
    ikf_ = nullptr;
    for (void* p : D->allocs) cudaFree(p);
    if (D->dump) std::fclose(D->dump);
    for (void* p : {(void*) D->y_h, (void*) D->res_h, (void*) D->res_conf_h, (void*) D->emb_h, (void*) D->par_h})
        if (p) cudaFreeHost(p);
    if (D->gexec) cudaGraphExecDestroy(D->gexec);
    if (D->gs) cudaStreamDestroy(D->gs);
    delete D;
}

size_t Glm5Model::ikf_bytes() const {
    const auto* D = (const IkfState*) ikf_;
    return D != nullptr ? D->bytes : 0;
}

void Glm5Model::ikf_report() const {
    const auto* D = (const IkfState*) ikf_;
    if (D == nullptr || D->drafts == 0) return;
    std::fprintf(stderr, "ink dflash2: %llu drafts, %.2f ms each (host to results), %.2f drafted a step (lengths",
                 (unsigned long long) D->drafts, D->ms_draft / (double) D->drafts,
                 (double) D->drafted / (double) D->drafts);
    for (int i = 0; i < kNB; ++i) std::fprintf(stderr, " %llu", (unsigned long long) D->hist[i]);
    std::fprintf(stderr, "); %llu context commits, %llu rows\n", (unsigned long long) D->commits,
                 (unsigned long long) D->rows);
}

// ------------------------------------------------------------------------------------------------ the context
// positions p_first .. p_first + n - 1 (n <= kRing) from their four taps ([n][4 E]) into the rings
bool Glm5Model::ikf_rows(const float* feat, int64_t p_first, int n, cudaStream_t s) {
    auto* D = (IkfState*) ikf_;
    if (D == nullptr || n <= 0) return true;
    const int E = g_.n_embd;
    const size_t FE = (size_t) kNF * E;
    if (n > kRing) {
        feat += (size_t) (n - kRing) * FE;
        p_first += n - kRing;
        n = kRing;
    }
    for (int c0 = 0; c0 < n; c0 += kCh) {
        const int m = std::min(kCh, n - c0);
        const float* fr = feat + (size_t) c0 * FE;
        gf::quantize_q8_1(fr, D->featq, (int) (m * FE), s);
        gf::MvJob jf = {D->fc.q, D->featq, fr, D->gpre, nullptr, 1.0f, D->fc.type, (int) FE, E};
        if (!gf::mvn(&jf, 1, m, s)) return false;
        k_rms<<<m, 256, 0, s>>>(D->gpre, E, D->enc_norm, D->eps, E, D->gv, E, (Q81*) D->gq);
        gf::MvJob jk[kL], jv[kL];
        for (int l = 0; l < kL; ++l) {
            jk[l] = {D->L[l].k.q, D->gq, D->gv, D->kk + (size_t) l * kCh * kKVD, nullptr, 1.0f, D->L[l].k.type, E, kKVD};
            jv[l] = {D->L[l].v.q, D->gq, D->gv, D->vv + (size_t) l * kCh * kKVD, nullptr, 1.0f, D->L[l].v.type, E, kKVD};
        }
        if (!gf::mvn(jk, kL, m, s) || !gf::mvn(jv, kL, m, s)) return false;
        const long long pp = (long long) (p_first + c0);
        k_rope_tab<<<(m * (kHD / 2) + 127) / 128, 128, 0, s>>>(pp, nullptr, m, D->rope_base, D->rope_c);
        for (int l = 0; l < kL; ++l) {
            k_norm_rope<<<dim3(kNKV, m), kHD, 0, s>>>(D->kk + (size_t) l * kCh * kKVD, kKVD, D->L[l].k_norm, D->eps, pp,
                                                       D->rope_c, D->L[l].rk, kRing, kKVD, kNKV, nullptr, 0, nullptr);
            k_norm_rope<<<dim3(kNKV, m), kHD, 0, s>>>(D->vv + (size_t) l * kCh * kKVD, kKVD, nullptr, D->eps, pp, nullptr,
                                                       D->L[l].rv, kRing, kKVD, kNKV, nullptr, 0, nullptr);
        }
    }
    ++D->commits;
    D->rows += (uint64_t) n;
    return cudaGetLastError() == cudaSuccess;
}

// ------------------------------------------------------------------------------------------------ the draft
// the drafts for the pending token y at position pos over the context lo .. pos - 1 (the rings hold it)
bool Glm5Model::ikf_draft(int32_t y, int64_t pos, int64_t lo, int32_t* drafts, float* conf, std::string& err) {
    auto* D = (IkfState*) ikf_;
    if (D == nullptr) { err = "ink dflash2: not loaded"; return false; }
    FastState* F = fast_;
    cudaStream_t s = F->cs;
    const auto t0 = std::chrono::steady_clock::now();
    const int E = g_.n_embd, V = D->V, NB = D->nb, G2 = 4 * (E / kGrp), FF = D->ff, R = NB - 1;
    lo = std::max<int64_t>(lo, pos - D->win);
    const int n_ctx = (int) std::max<int64_t>(0, std::min<int64_t>(pos - lo, D->win));
    D->y_h[0] = y;
    *D->par_h = IkfPar{(long long) pos, (long long) (pos - n_ctx), n_ctx, 0};
    const bool host_emb = F->emb_tab_d == nullptr && F->emb_q8_d == nullptr;
    if (host_emb) {
        const strata::qt::Traits* tt = strata::qt::traits((strata::qt::Type) pack_emb_type_);
        tt->to_float(pack_emb_src_ + (size_t) y * strata::qt::row_size((strata::qt::Type) pack_emb_type_, E), D->emb_h, E);
    }
    const float* tok_norm = w_.at("token_embd_norm.weight");
    const float eps_t = g_.norm_eps;
    const WSlot& ow = ws_map_.at("output.weight");
    const auto enqueue = [&](cudaStream_t s) -> bool {
        cudaMemcpyAsync(D->par_d, D->par_h, sizeof(IkfPar), cudaMemcpyHostToDevice, s);
        if (!host_emb)
            gf::embed_rows_q(F->emb_tab_d ? pack_emb_type_ : kQ8_0, F->emb_tab_d ? F->emb_tab_d : F->emb_q8_d, D->y_d, 1,
                             E, D->a, s);
        else
            cudaMemcpyAsync(D->a, D->emb_h, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s);
        k_block_rows<<<NB, 256, 0, s>>>(D->a, tok_norm, eps_t, E, D->mask_row, NB, D->x);
        const dim3 cg((unsigned) ((E + 255) / 256), (unsigned) NB);
        const float scale = 1.0f / std::sqrt((float) kHD);
        k_rope_tab<<<(NB * (kHD / 2) + 127) / 128, 128, 0, s>>>(0, D->par_d, NB, D->rope_base, D->rope_d);
        for (int l = 0; l < kL; ++l) {
            const IkfLayer& Ly = D->L[l];
            // ---- attention
            k_rms<<<NB, 256, 0, s>>>(D->x, E, Ly.attn_norm, D->eps, E, D->xn, E, (Q81*) D->xnq);
            gf::MvJob jc = {Ly.acp.q, D->xnq, D->xn, D->dyn, nullptr, 1.0f, Ly.acp.type, E, G2};
            if (!gf::mvn(&jc, 1, NB, s)) { err = "ink dflash2: attn_conv_proj"; return false; }
            k_conv<<<cg, 256, 0, s>>>(D->xn, D->dyn, G2, 0, E / kGrp, kGrp, Ly.acb, E, D->xc, nullptr, (Q81*) D->xcq);
            gf::MvJob j[3] = {{Ly.q.q, D->xcq, D->xc, D->q, nullptr, 1.0f, Ly.q.type, E, kQD},
                              {Ly.k.q, D->xcq, D->xc, D->kb, nullptr, 1.0f, Ly.k.type, E, kKVD},
                              {Ly.v.q, D->xcq, D->xc, D->vb, nullptr, 1.0f, Ly.v.type, E, kKVD}};
            if (!gf::mvn(j, 3, NB, s)) { err = "ink dflash2: q / k / v"; return false; }
            k_norm_rope<<<dim3(kNH + kNKV, NB), kHD, 0, s>>>(D->q, kQD, Ly.q_norm, D->eps, (long long) pos, D->rope_d,
                                                             nullptr, 0, 0, kNH, D->kb, kKVD, Ly.k_norm);
            if (D->split && NB == kNB) {
                k_attn_part<<<dim3(kSplit, kNKV), 256, 0, s>>>(D->q, Ly.rk, Ly.rv, D->par_d, kRing, D->kb, D->vb, scale,
                                                               D->pm, D->pl, D->pacc);
                k_attn_comb<<<dim3(kNH, NB), kHD, 0, s>>>(D->pm, D->pl, D->pacc, D->o, (Q81*) D->oq);
            } else {
                k_attn<<<dim3(kNKV, NB), kAttT, D->att_smem, s>>>(D->q, Ly.rk, Ly.rv, D->par_d, kRing, D->kb, D->vb, NB,
                                                                  scale, D->o, (Q81*) D->oq);
            }
            gf::MvJob jo = {Ly.o.q, D->oq, D->o, D->a, nullptr, 1.0f, Ly.o.type, kQD, E};
            if (!gf::mvn(&jo, 1, NB, s)) { err = "ink dflash2: attn_output"; return false; }
            k_conv<<<cg, 256, 0, s>>>(D->a, D->dyn, G2, 2 * (E / kGrp), E / kGrp, kGrp, Ly.acb + 2 * (size_t) E, E,
                                      nullptr, D->x, nullptr);
            // ---- the FFN
            k_rms<<<NB, 256, 0, s>>>(D->x, E, Ly.ffn_norm, D->eps, E, D->xn, E, (Q81*) D->xnq);
            gf::MvJob jf = {Ly.fcp.q, D->xnq, D->xn, D->dyn, nullptr, 1.0f, Ly.fcp.type, E, G2};
            if (!gf::mvn(&jf, 1, NB, s)) { err = "ink dflash2: ffn_conv_proj"; return false; }
            k_conv<<<cg, 256, 0, s>>>(D->xn, D->dyn, G2, 0, E / kGrp, kGrp, Ly.fcb, E, D->xc, nullptr, (Q81*) D->xcq);
            gf::MvJob ju[2] = {{Ly.gate.q, D->xcq, D->xc, D->gt, nullptr, 1.0f, Ly.gate.type, E, FF},
                               {Ly.up.q, D->xcq, D->xc, D->up, nullptr, 1.0f, Ly.up.type, E, FF}};
            if (!gf::mvn(ju, 2, NB, s)) { err = "ink dflash2: ffn gate / up"; return false; }
            k_swiglu<<<(NB * FF + 255) / 256, 256, 0, s>>>(D->gt, D->up, NB * FF, (Q81*) D->gq2);
            gf::MvJob jd = {Ly.down.q, D->gq2, D->gt, D->f, nullptr, 1.0f, Ly.down.type, FF, E};
            if (!gf::mvn(&jd, 1, NB, s)) { err = "ink dflash2: ffn_down"; return false; }
            k_conv<<<cg, 256, 0, s>>>(D->f, D->dyn, G2, 2 * (E / kGrp), E / kGrp, kGrp, Ly.fcb + 2 * (size_t) E, E,
                                      nullptr, D->x, nullptr);
        }
        // ---- the head over rows 1 .. NB - 1 (row 0 is y's): the target's head over the vocabulary, the gate, the walk
        k_rms<<<R, 256, 0, s>>>(D->x + E, E, D->out_norm, D->eps, E, D->hn, E, (Q81*) D->hnq);
        gf::MvJob jh = {ow.q, D->hnq, D->hn, D->logits, nullptr, 1.0f, ow.type, E, V};
        if (ow.type == kF32) jh.w = ow.f32;
        if (!gf::mvn(&jh, 1, R, s)) { err = "ink dflash2: output head"; return false; }
        gf::MvJob jg = {D->sel_h.q, D->hnq, D->hn, D->gate, nullptr, 1.0f, D->sel_h.type, E, kRank};
        if (!gf::mvn(&jg, 1, R, s)) { err = "ink dflash2: selector_hidden"; return false; }
        const int chunk = (V + kTkCh - 1) / kTkCh;
        k_topk_part<<<dim3(kTkCh, R), 256, (size_t) chunk * sizeof(float), s>>>(D->logits, V, chunk, D->softcap, D->tk_v,
                                                                                D->tk_i, D->lse);
        k_select<<<1, 256, 0, s>>>(D->tk_v, D->tk_i, D->tail ? D->lse : nullptr, kTkCh, kTkCh * kTopK, D->gate, D->sel_pred, D->sel_succ, D->Vs, D->y_d, R,
                                   D->softcap, D->cap_mode, 0.0f, D->res_d, D->res_conf_d);
        cudaMemcpyAsync(D->res_h, D->res_d, (size_t) (1 + R) * sizeof(int), cudaMemcpyDeviceToHost, s);
        cudaMemcpyAsync(D->res_conf_h, D->res_conf_d, (size_t) R * sizeof(float), cudaMemcpyDeviceToHost, s);
        return true;
    };
    bool launched = false;
    if (D->graph_on && D->gexec == nullptr) {   // (the first draft: its kernels captured once)
        cudaGraph_t gr = nullptr;
        bool okc = cudaStreamBeginCapture(D->gs, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
        const bool enq = okc && enqueue(D->gs);
        okc = cudaStreamEndCapture(D->gs, &gr) == cudaSuccess && okc && enq &&
              cudaGraphInstantiate(&D->gexec, gr, 0) == cudaSuccess;
        if (gr) cudaGraphDestroy(gr);
        if (!okc) {
            cudaGetLastError();
            if (D->gexec) cudaGraphExecDestroy(D->gexec);
            D->gexec = nullptr;
            D->graph_on = false;
            err.clear();
            std::fprintf(stderr, "ink dflash2: the draft's graph did not capture - kernel by kernel\n");
        }
    }
    if (D->gexec != nullptr) launched = cudaGraphLaunch(D->gexec, s) == cudaSuccess;
    if (!launched && !enqueue(s)) return false;
    if (cudaStreamSynchronize(s) != cudaSuccess || gf::launch_errors() != D->launch_err0) {
        err = std::string("ink dflash2: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        err = std::string("ink dflash2: ") + cudaGetErrorString(e);
        return false;
    }
    const int n = std::max(0, std::min(D->res_h[0], R));
    for (int i = 0; i < n; ++i) {
        drafts[i] = D->res_h[1 + i];
        conf[i] = D->res_conf_h[i];
    }
    // STRATA_INK_DSPARK_DUMP=<file> (debug): a line a draft - the position (y's), y, the context rows | drafts | conf
    if (D->dump != nullptr) {
        std::fprintf(D->dump, "%lld %d %d |", (long long) pos, y, n_ctx);
        for (int i = 0; i < n; ++i) std::fprintf(D->dump, " %d", drafts[i]);
        std::fprintf(D->dump, " |");
        for (int i = 0; i < n; ++i) std::fprintf(D->dump, " %.4f", conf[i]);
        std::fprintf(D->dump, "\n");
        std::fflush(D->dump);
    }
    ++D->drafts;
    D->drafted += (uint64_t) n;
    ++D->hist[n];
    D->ms_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace strata::core
