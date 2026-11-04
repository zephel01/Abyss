// src/core/dflash2.cu - DFlash 2, Qwen3.8-Flash-Next's block drafter (nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF,
// arch dflash), loaded with STRATA_Q4X_DFLASH=<the GGUF> on a qwen4exp target.  Five small dense layers read the
// target's residual entering layers 6/16/26/36/46 and propose up to 5 tokens of a verify window at once.  The arithmetic
// (worklog/2026-11-04_dflash2-survey.md; the numpy reference is Abyss-tools bench/dflash2/dfl2_probe.py, taken from
// llama.cpp's src/models/dflash.cpp and common/speculative.cpp):
//   * the context: feat[p] = [h6; h16; h26; h36; h46], h_L the mean of the 4 streams entering layer L at position p;
//     g = rms(fc feat) * enc.output_norm; each drafter layer's K = rope_p(rms_head(wk g) * k_norm), V = wv g, kept in a
//     ring of kRing positions (FP16, positions mod kRing); a draft reads the last kRing of them;
//   * the block: [y, mask x n] (y the pending token, the mask token's embedding after it, both the TARGET's
//     token_embd) at positions pos .. pos + n.  A layer: attn_norm -> the conv's side 0 (dyn = attn_conv_proj xn,
//     w[side][tap][c] = base[side][tap][c] + dyn[side][tap][c / 16], out[t] = w0 x[t] + w1 x[t - 1] inside the block)
//     -> q / k / v, the per-head q / k norms, NEOX RoPE over 128 dims (base 1e4) -> attention over the context and the
//     whole block (no mask; 32 q heads over 8 kv heads) -> wo -> the conv's side 1 (the same dyn) -> the residual; the
//     FFN the same shape (ffn_norm, ffn_conv_proj, SwiGLU 7680).  output_norm, the TARGET's output head: row i
//     proposes the token at position pos + i (DSpark's row i proposes pos + i + 1);
//   * the selector: each row's top-16 logits are its candidates, score = logit + sum_r succ[cand][r] pred[prev][r]
//     gate_i[r] (gate = selector_hidden hn, prev the row before's pick, row 1's y), a greedy walk; a row whose pick
//     has confidence 1 / sum exp(s - s_max) below STRATA_Q4X_DFLASH_PMIN (default 0.9) ends the drafts.
// The context follows the sequence as DSpark's does: the token path, the verify window (the kept columns, at
// spec_rewind) and the prompt path (a chunk's last kRing rows) leave their features, dflash_commit turns them into
// ring rows.
#include "glm_fast_state.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/dsv4_fast.hpp"
#include "strata/kernels/glm_fast.hpp"

#include "ggml.h"

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
namespace ds = strata::kernels::dsv4;

namespace {
constexpr size_t kQ8 = 36;     // bytes of a q8_1 block (32 values)
constexpr int kL = 5;          // drafter layers = feature layers
constexpr int kRing = 2048;    // the context window (the GGUF's sliding_window), positions mod kRing
constexpr int kCh = 256;       // the commit's rows a pass
constexpr int kHD = 128, kNH = 32, kNKV = 8, kFF = 7680, kRank = 256, kTopK = 16, kGrp = 16;
constexpr int kNbMax = 6;      // y + 5 masks
constexpr int kTkCh = 64;      // the top-k's chunks a row
constexpr int kQ8_0 = 8, kF32 = 0;

struct DflLayer {
    WSlot q, k, v, o, gate, up, down, acp, fcp;
    float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *acb = nullptr,
          *fcb = nullptr;
    __half *rk = nullptr, *rv = nullptr;   // [kRing][kNKV * kHD]
};

struct DflState {
    std::unique_ptr<strata::GgufFile> gguf;
    DflLayer L[kL];
    WSlot fc, sel_h;
    const uint8_t *sel_pred = nullptr, *sel_succ = nullptr;   // Q8_0 rows [V][256]
    float *enc_norm = nullptr, *out_norm = nullptr, *emb_mask = nullptr;
    int cap[kL] = {6, 16, 26, 36, 46};
    int n_max = 5, nb = 6, mask = 248077;
    float p_min = 0.9f, eps = 1e-6f, rope_base = 10000.0f;   // (0.9: list +4%, code even against 0.75 on an RTX 5090)
    // features: the token's / the window's columns ([kVerifyMax][5 E]) and a prompt chunk's last rows ([kRing][5 E])
    float *feat = nullptr, *feat_p = nullptr;
    void* featq = nullptr;
    float *gpre = nullptr, *gv = nullptr, *kk = nullptr, *vv = nullptr;   // the commit's pass
    void* gq = nullptr;
    // the draft
    float *x = nullptr, *xn = nullptr, *dyn = nullptr, *xc = nullptr, *q = nullptr, *kb = nullptr, *vb = nullptr,
          *o = nullptr, *a = nullptr, *gt = nullptr, *up = nullptr, *f = nullptr, *hn = nullptr, *logits = nullptr,
          *gate = nullptr, *tk_v = nullptr, *res_conf_d = nullptr;
    void *xnq = nullptr, *xcq = nullptr, *oq = nullptr, *gq2 = nullptr, *hnq = nullptr;
    int *tk_i = nullptr, *res_d = nullptr;
    int *y_h = nullptr, *y_d = nullptr, *res_h = nullptr;
    float *res_conf_h = nullptr, *emb_h = nullptr;
    std::vector<void*> allocs;
    size_t bytes = 0;
    FILE* dump = nullptr;
    int launch_err0 = 0;
    int64_t ctx_lo = 0, ctx_end = 0, win_p0 = -1;
    int win_T = 0;
    uint64_t drafts = 0, drafted = 0, commits = 0;
    double ms_draft = 0.0;
    uint64_t hist[kNbMax + 1] = {};
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
// a block's sum (blockDim a multiple of 32, <= 1024); every thread gets it
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float r = lane < nw ? sh[lane] : 0.0f;
    r = warp_sum(r);
    return r;
}
__device__ float block_max(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float r = lane < nw ? sh[lane] : -INFINITY;
    r = warp_max(r);
    return r;
}

// y[row] = rms(x[row]) * w (rows ldx / ldy floats apart)
__global__ void k_rms(const float* __restrict__ x, int ldx, const float* __restrict__ w, float eps, int E,
                      float* __restrict__ y, int ldy) {
    __shared__ float sh[32];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, sh);
    const float r = rsqrtf(ss / (float) E + eps);
    float* yr = y + (size_t) blockIdx.x * ldy;
    for (int i = threadIdx.x; i < E; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}

// the two-tap conv over the block's rows: w_tap[c] = base[tap][c] + dyn[t][off + tap * G + c / grp], out[t] = w0 x[t] +
// w1 x[t - 1] (x[-1] = 0); res != null: res[t] += out[t] instead of out
__global__ void k_conv(const float* __restrict__ x, const float* __restrict__ dyn, int dyn_ld, int off, int G, int grp,
                       const float* __restrict__ base, int E, float* __restrict__ out, float* __restrict__ res) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= E) return;
    const float* d = dyn + (size_t) t * dyn_ld + off;
    const float w0 = base[c] + d[c / grp], w1 = base[E + c] + d[G + c / grp];
    const float prev = t > 0 ? x[(size_t) (t - 1) * E + c] : 0.0f;
    const float v = w0 * x[(size_t) t * E + c] + w1 * prev;
    if (res != nullptr) res[(size_t) t * E + c] += v;
    else out[(size_t) t * E + c] = v;
}

// per head (blockIdx.x) of row blockIdx.y: rms * w (w null: none), NEOX RoPE at pos0 + row over 128 dims (rope false:
// none); the result in place (ring null) or as FP16 into ring row (pos0 + row) % ring_n
__global__ void k_norm_rope(float* __restrict__ x, int ld, const float* __restrict__ w, float eps, long long pos0,
                            int rope, float base, __half* __restrict__ ring, int ring_n, int ring_ld) {
    __shared__ float sh[32];
    __shared__ float buf[kHD];
    const int d = threadIdx.x, h = blockIdx.x, row = blockIdx.y;
    float* xr = x + (size_t) row * ld + (size_t) h * kHD;
    float v = xr[d];
    if (w != nullptr) {
        const float ss = block_sum(v * v, sh);
        v = v * rsqrtf(ss / (float) kHD + eps) * w[d];
    }
    buf[d] = v;
    __syncthreads();
    const long long pos = pos0 + row;
    if (rope) {
        const int i = d & (kHD / 2 - 1);
        const double inv = pow((double) base, -2.0 * (double) i / (double) kHD);
        double sn, cs;
        sincos((double) pos * inv, &sn, &cs);
        const float a = buf[i], b = buf[i + kHD / 2];
        v = d < kHD / 2 ? a * (float) cs - b * (float) sn : a * (float) sn + b * (float) cs;
    }
    if (ring != nullptr) ring[(size_t) (pos % ring_n) * ring_ld + (size_t) h * kHD + d] = __float2half(v);
    else xr[d] = v;
}

// attention of block row blockIdx.y, q head blockIdx.x over the context's n_ctx ring rows (positions lo ..) and the
// block's nb rows (kb / vb, [nb][kNKV * kHD] floats); 128 threads, scores in dynamic shared memory
__global__ void __launch_bounds__(128) k_attn(const float* __restrict__ q, const __half* __restrict__ rk,
                                              const __half* __restrict__ rv, long long lo, int n_ctx, int ring_n,
                                              const float* __restrict__ kb, const float* __restrict__ vb, int nb,
                                              float scale, float* __restrict__ out) {
    extern __shared__ float sc[];
    __shared__ float sh[32];
    const int h = blockIdx.x, t = blockIdx.y, kvh = h / (kNH / kNKV);
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int S = n_ctx + nb, KVD = kNKV * kHD;
    const float* qr = q + (size_t) t * kNH * kHD + (size_t) h * kHD;
    const float4 qv = *reinterpret_cast<const float4*>(qr + lane * 4);
    for (int j = wp; j < S; j += 4) {
        float4 kv;
        if (j < n_ctx) {
            const __half* kp = rk + (size_t) ((lo + j) % ring_n) * KVD + (size_t) kvh * kHD + lane * 4;
            const __half2 k01 = *reinterpret_cast<const __half2*>(kp), k23 = *reinterpret_cast<const __half2*>(kp + 2);
            const float2 f01 = __half22float2(k01), f23 = __half22float2(k23);
            kv = make_float4(f01.x, f01.y, f23.x, f23.y);
        } else {
            kv = *reinterpret_cast<const float4*>(kb + (size_t) (j - n_ctx) * KVD + (size_t) kvh * kHD + lane * 4);
        }
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        dot = warp_sum(dot);
        if (lane == 0) sc[j] = dot * scale;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int j = threadIdx.x; j < S; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = block_max(m, sh);
    float l = 0.0f;
    for (int j = threadIdx.x; j < S; j += blockDim.x) {
        const float e = __expf(sc[j] - m);
        sc[j] = e;
        l += e;
    }
    l = block_sum(l, sh);   // (its barriers order the writes above before the reads below)
    const int d = threadIdx.x;
    float acc = 0.0f;
    for (int j = 0; j < n_ctx; ++j)
        acc += sc[j] * __half2float(rv[(size_t) ((lo + j) % ring_n) * KVD + (size_t) kvh * kHD + d]);
    for (int j = 0; j < nb; ++j) acc += sc[n_ctx + j] * vb[(size_t) j * KVD + (size_t) kvh * kHD + d];
    out[(size_t) t * kNH * kHD + (size_t) h * kHD + d] = acc / l;
}

__global__ void k_swiglu(float* __restrict__ g, const float* __restrict__ u, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float a = g[i];
        g[i] = a / (1.0f + __expf(-a)) * u[i];
    }
}

// rows [r][V] -> each chunk's top kTopK (value, index) - the chunk's floats in shared memory, kTopK block argmaxes
__global__ void __launch_bounds__(256) k_topk_part(const float* __restrict__ logits, int V, int chunk,
                                                   float* __restrict__ tv, int* __restrict__ ti) {
    extern __shared__ float vals[];
    __shared__ float bv[8];
    __shared__ int bi[8];
    const int r = blockIdx.y, ch = blockIdx.x, lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int i0 = ch * chunk, n = max(0, min(chunk, V - i0));
    const float* lr = logits + (size_t) r * V + i0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) vals[i] = lr[i];
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

__device__ __forceinline__ float q8_0_at(const uint8_t* row, int r) {
    const uint8_t* b = row + (size_t) (r >> 5) * 34;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    return d * (float) (int8_t) b[2 + (r & 31)];
}

// the greedy walk (one block of 256): rows 1 .. nb - 1 (tv / ti / gate index r = row - 1), each row's top kTopK out
// of its chunks' candidates, the selector's scores, the pick and its confidence; stops below p_min.  out: n, then the
// tokens; conf: the picks' confidences
__global__ void __launch_bounds__(256) k_select(const float* __restrict__ tv, const int* __restrict__ ti, int n_cand,
                                                const float* __restrict__ gate, const uint8_t* __restrict__ pred,
                                                const uint8_t* __restrict__ succ, const int* __restrict__ y, int rows,
                                                float p_min, int* __restrict__ out, float* __restrict__ conf) {
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
            if (ci[i] < 0) cv[i] = -INFINITY;
        }
        // pred[prev] . gate_r
        for (int i = threadIdx.x; i < kRank; i += blockDim.x)
            pg[i] = q8_0_at(pred + (size_t) prev * rb, i) * gate[(size_t) r * kRank + i];
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
        // the scores: warp wp takes candidates 2 wp, 2 wp + 1
        for (int c = 2 * wp; c < 2 * wp + 2; ++c) {
            const uint8_t* srow = succ + (size_t) top_i[c] * rb;
            float s = 0.0f;
            for (int i = lane; i < kRank; i += 32) s += q8_0_at(srow, i) * pg[i];
            s = warp_sum(s);
            if (lane == 0) score[c] = top_v[c] + s;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int b = 0;
            for (int c = 1; c < kTopK; ++c)
                if (score[c] > score[b]) b = c;
            float z = 0.0f;
            for (int c = 0; c < kTopK; ++c) z += expf(score[c] - score[b]);
            const float p = 1.0f / z;
            if (p < p_min) {
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

int cap_slot(const DflState* D, int il) {
    for (int k = 0; k < kL; ++k)
        if (D->cap[k] == il) return k;
    return -1;
}

void rms_rows(const float* x, int ldx, const float* w, float eps, int E, int rows, float* y, int ldy, cudaStream_t s) {
    k_rms<<<rows, 256, 0, s>>>(x, ldx, w, eps, E, y, ldy);
}
}  // namespace

bool Glm5Model::dflash_load(std::string& err) {
    (void) err;
    const char* path = getenv("STRATA_Q4X_DFLASH");
    if (path == nullptr || !path[0]) return true;
    const Glm5Geometry& g = g_;
    const auto refuse = [&](const std::string& why) {
        std::fprintf(stderr, "dflash: not loaded - %s\n", why.c_str());
        dflash_destroy();
        return true;   // the target runs on without the drafter
    };
    if (!g.q4x || n_parts_ != 1 || l0_ != 0 || l1_ != g.n_layers || g.hc != 4)
        return refuse("Qwen3.8-Flash-Next (qwen4exp) on one device only (no split)");
    auto* D = new DflState();
    dfl_ = D;
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
    bool layers_ok = tl != nullptr && tl->items.size() == (size_t) kL;
    for (int i = 0; layers_ok && i < kL; ++i) {
        D->cap[i] = (int) tl->items[(size_t) i].num();
        layers_ok = D->cap[i] > 0 && D->cap[i] < g.n_layers;
    }
    D->eps = (float) num("dflash.attention.layer_norm_rms_epsilon", 1e-6);
    D->rope_base = (float) num("dflash.rope.freq_base", 10000.0);
    const int E = g.n_embd, V = g.n_vocab;
    if (!layers_ok || (int) num("dflash.block_count", 0) != kL || (int) num("dflash.embedding_length", 0) != E ||
        (int) num("dflash.feed_forward_length", 0) != kFF || (int) num("dflash.attention.head_count", 0) != kNH ||
        (int) num("dflash.attention.head_count_kv", 0) != kNKV || (int) num("dflash.attention.key_length", 0) != kHD ||
        (int) num("dflash.attention.value_length", 0) != kHD || (int) num("dflash.conv_kernel_size", 0) != 2 ||
        (int) num("dflash.conv_group_size", 0) != kGrp || (int) num("dflash.selector_rank", 0) != kRank ||
        (int) num("dflash.selector_top_k", 0) != kTopK || (int) num("dflash.attention.sliding_window", 0) != kRing ||
        E % (kGrp * 32) != 0)
        return refuse("the drafter's layout (5 layers over 5 target layers, the target's width, 32 / 8 heads of 128, "
                      "ff 7680, a 2-tap conv in groups of 16, selector rank 256 top-16, a 2048-row window)");
    if (const char* v = getenv("STRATA_Q4X_DFLASH_MASK")) D->mask = std::atoi(v);
    if (D->mask < 0 || D->mask >= V) return refuse("the mask token id");
    if (const char* v = getenv("STRATA_Q4X_DFLASH_N")) D->n_max = std::max(1, std::min(kNbMax - 1, std::atoi(v)));
    if (const char* v = getenv("STRATA_Q4X_DFLASH_PMIN")) D->p_min = (float) std::atof(v);
    D->nb = D->n_max + 1;
    std::string why;
    const auto find = [&](const std::string& n, int type, std::initializer_list<uint64_t> shape) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = G.find(n);
        if (t == nullptr) { if (why.empty()) why = n + " missing"; return nullptr; }
        bool ok = (int) t->type == type && t->shape.size() == shape.size();
        size_t i = 0;
        for (uint64_t d : shape) ok = ok && t->shape[i++] == d;
        if (!ok && why.empty()) why = n + " has another type or shape";
        return ok ? t : nullptr;
    };
    bool ok = true;
    const auto dev = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (!ok || cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; cudaGetLastError(); return nullptr; }
        D->allocs.push_back(p);
        D->bytes += bytes;
        return p;
    };
    const auto raw = [&](const std::string& n, int64_t n_in, int64_t n_out, WSlot& w) {
        const strata::TensorInfo* t = find(n, kQ8_0, {(uint64_t) n_in, (uint64_t) n_out});
        if (t == nullptr) { ok = false; return; }
        const size_t bytes = gf::row_bytes(kQ8_0, n_in) * (size_t) n_out;
        void* p = dev(bytes);
        if (p == nullptr) return;
        ok = cudaMemcpy(p, G.tensor_data(*t), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        w = WSlot{nullptr, kQ8_0, p, n_in, n_out};
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
        DflLayer& Ly = D->L[l];
        const std::string P = "blk." + std::to_string(l) + ".";
        raw(P + "attn_q.weight", E, kNH * kHD, Ly.q);
        raw(P + "attn_k.weight", E, kNKV * kHD, Ly.k);
        raw(P + "attn_v.weight", E, kNKV * kHD, Ly.v);
        raw(P + "attn_output.weight", kNH * kHD, E, Ly.o);
        raw(P + "ffn_gate.weight", E, kFF, Ly.gate);
        raw(P + "ffn_up.weight", E, kFF, Ly.up);
        raw(P + "ffn_down.weight", kFF, E, Ly.down);
        raw(P + "attn_conv_proj.weight", E, G2, Ly.acp);
        raw(P + "ffn_conv_proj.weight", E, G2, Ly.fcp);
        Ly.attn_norm = f32(P + "attn_norm.weight", {uE});
        Ly.ffn_norm = f32(P + "ffn_norm.weight", {uE});
        Ly.q_norm = f32(P + "attn_q_norm.weight", {(uint64_t) kHD});
        Ly.k_norm = f32(P + "attn_k_norm.weight", {(uint64_t) kHD});
        Ly.acb = f32(P + "attn_conv_base", {uE, 2, 2});
        Ly.fcb = f32(P + "ffn_conv_base", {uE, 2, 2});
        Ly.rk = (__half*) dev((size_t) kRing * kNKV * kHD * sizeof(__half));
        Ly.rv = (__half*) dev((size_t) kRing * kNKV * kHD * sizeof(__half));
    }
    raw("fc.weight", (int64_t) kL * E, E, D->fc);
    raw("selector_hidden.weight", E, kRank, D->sel_h);
    D->enc_norm = f32("enc.output_norm.weight", {uE});
    D->out_norm = f32("output_norm.weight", {uE});
    for (int which = 0; which < 2 && ok; ++which) {
        const char* n = which == 0 ? "selector_predecessor.weight" : "selector_successor.weight";
        const strata::TensorInfo* t = find(n, kQ8_0, {(uint64_t) kRank, (uint64_t) V});
        if (t == nullptr) { ok = false; break; }
        const size_t bytes = gf::row_bytes(kQ8_0, kRank) * (size_t) V;
        void* p = dev(bytes);
        if (p != nullptr) ok = cudaMemcpy(p, G.tensor_data(*t), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        (which == 0 ? D->sel_pred : D->sel_succ) = (const uint8_t*) p;
    }
    if (!ok) return refuse(why.empty() ? "the weights did not upload (VRAM?)" : why);
    if (!gf::mvn_supported(kQ8_0)) return refuse("the Q8_0 kernels");
    if (ws_map_.find("output.weight") == ws_map_.end()) return refuse("the target's output.weight");
    // the mask token's embedding, once
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) return refuse("the target's embedding");
    {
        std::vector<float> e((size_t) E);
        tt->to_float(pack_emb_src_ + (size_t) D->mask * ggml_row_size((ggml_type) pack_emb_type_, E), e.data(), E);
        D->emb_mask = (float*) dev((size_t) E * sizeof(float));
        if (ok) ok = cudaMemcpy(D->emb_mask, e.data(), (size_t) E * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    // the buffers
    const auto fb = [&](float*& p, size_t n) { p = (float*) dev(n * sizeof(float)); };
    const auto bb = [&](void*& p, size_t n) { p = dev(n); };
    const size_t FE = (size_t) kL * E, NB = (size_t) kNbMax;
    fb(D->feat, (size_t) kVerifyMax * FE);
    fb(D->feat_p, (size_t) kRing * FE);
    bb(D->featq, (size_t) kCh * FE / 32 * kQ8);
    fb(D->gpre, (size_t) kCh * E);
    fb(D->gv, (size_t) kCh * E);
    bb(D->gq, (size_t) kCh * E / 32 * kQ8);
    fb(D->kk, (size_t) kL * kCh * kNKV * kHD);
    fb(D->vv, (size_t) kL * kCh * kNKV * kHD);
    fb(D->x, NB * E);
    fb(D->xn, NB * E);
    bb(D->xnq, NB * E / 32 * kQ8);
    fb(D->dyn, NB * G2);
    fb(D->xc, NB * E);
    bb(D->xcq, NB * E / 32 * kQ8);
    fb(D->q, NB * kNH * kHD);
    fb(D->kb, NB * kNKV * kHD);
    fb(D->vb, NB * kNKV * kHD);
    fb(D->o, NB * kNH * kHD);
    bb(D->oq, NB * kNH * kHD / 32 * kQ8);
    fb(D->a, NB * E);
    fb(D->gt, NB * kFF);
    fb(D->up, NB * kFF);
    bb(D->gq2, NB * kFF / 32 * kQ8);
    fb(D->f, NB * E);
    fb(D->hn, NB * E);
    bb(D->hnq, NB * E / 32 * kQ8);
    fb(D->logits, NB * (size_t) V);
    fb(D->gate, NB * kRank);
    fb(D->tk_v, NB * kTkCh * kTopK);
    D->tk_i = (int*) dev(NB * kTkCh * kTopK * sizeof(int));
    D->res_d = (int*) dev(16 * sizeof(int));
    fb(D->res_conf_d, 16);
    if (!ok) return refuse("the buffers did not allocate (VRAM?)");
    ok = cudaHostAlloc((void**) &D->y_h, 64, cudaHostAllocMapped) == cudaSuccess &&
         cudaHostGetDevicePointer((void**) &D->y_d, D->y_h, 0) == cudaSuccess &&
         cudaHostAlloc((void**) &D->res_h, 64, cudaHostAllocDefault) == cudaSuccess &&
         cudaHostAlloc((void**) &D->res_conf_h, 64, cudaHostAllocDefault) == cudaSuccess &&
         cudaHostAlloc((void**) &D->emb_h, (size_t) E * sizeof(float), cudaHostAllocDefault) == cudaSuccess;
    if (!ok) return refuse("the pinned buffers");
    D->launch_err0 = gf::launch_errors();
    if (const char* v = getenv("STRATA_Q4X_DFLASH_DUMP"); v != nullptr && v[0]) D->dump = std::fopen(v, "w");
    std::fprintf(stderr, "dflash: %s - %d layers over target layers %d/%d/%d/%d/%d, up to %d drafts a step (p_min %.2f), "
                         "%.2f GB on CUDA%d\n",
                 path, kL, D->cap[0], D->cap[1], D->cap[2], D->cap[3], D->cap[4], D->n_max, D->p_min,
                 (double) D->bytes / 1073741824.0, dev_);
    return true;
}

void Glm5Model::dflash_destroy() {
    auto* D = (DflState*) dfl_;
    if (D == nullptr) return;
    dfl_ = nullptr;
    for (void* p : D->allocs) cudaFree(p);
    if (D->dump) std::fclose(D->dump);
    for (void* p : {(void*) D->y_h, (void*) D->res_h, (void*) D->res_conf_h, (void*) D->emb_h})
        if (p) cudaFreeHost(p);
    delete D;
}

void Glm5Model::dflash_report() const {
    const auto* D = (const DflState*) dfl_;
    if (D == nullptr || D->drafts == 0) return;
    std::fprintf(stderr, "dflash: %llu drafts, %.2f ms each (host to results), %.2f drafted a step (lengths",
                 (unsigned long long) D->drafts, D->ms_draft / (double) D->drafts,
                 (double) D->drafted / (double) D->drafts);
    for (int i = 0; i <= D->n_max; ++i) std::fprintf(stderr, " %llu", (unsigned long long) D->hist[i]);
    std::fprintf(stderr, "); %llu context commits\n", (unsigned long long) D->commits);
}

void Glm5Model::dflash_reset() {
    auto* D = (DflState*) dfl_;
    if (D == nullptr) return;
    D->win_p0 = -1;
    // a restored snapshot (pos_ inside the context): the ring rows of the positions before it that no later position
    // wrote over are still this sequence's
    if (pos_ > D->ctx_lo && pos_ <= D->ctx_end) {
        D->ctx_lo = std::max<int64_t>(D->ctx_lo, D->ctx_end - kRing);
        if (D->ctx_lo >= pos_) D->ctx_lo = pos_;
        D->ctx_end = pos_;
    } else {
        D->ctx_lo = D->ctx_end = pos_;
    }
}

void Glm5Model::dflash_capture(int il, const float* R, int n, size_t rs, bool prompt, cudaStream_t s) {
    auto* D = (DflState*) dfl_;
    if (D == nullptr || n <= 0) return;
    const int k = cap_slot(D, il);
    if (k < 0) return;
    const int E = g_.n_embd;
    const size_t FE = (size_t) kL * E;
    if (prompt) {
        const int m = std::min(n, kRing);
        ds::stream_mean(R + (size_t) (n - m) * rs, m, rs, E, D->feat_p + (size_t) k * E, FE, s);
    } else {
        ds::stream_mean(R, std::min(n, (int) kVerifyMax), rs, E, D->feat + (size_t) k * E, FE, s);
    }
}

bool Glm5Model::dflash_commit(int64_t p0, int n, bool prompt, cudaStream_t s) {
    auto* D = (DflState*) dfl_;
    if (D == nullptr || n <= 0) return true;
    const int E = g_.n_embd, KVD = kNKV * kHD;
    const size_t FE = (size_t) kL * E;
    if (n > kRing) {
        p0 += n - kRing;
        n = kRing;
    }
    const float* feat = prompt ? D->feat_p : D->feat;
    for (int c0 = 0; c0 < n; c0 += kCh) {
        const int m = std::min(kCh, n - c0);
        const float* fr = feat + (size_t) c0 * FE;
        gf::quantize_q8_1(fr, D->featq, (int) (m * FE), s);
        gf::MvJob j = {D->fc.q, D->featq, fr, D->gpre, nullptr, 1.0f, D->fc.type, (int) FE, E};
        if (!gf::mvn(&j, 1, m, s)) return false;
        rms_rows(D->gpre, E, D->enc_norm, D->eps, E, m, D->gv, E, s);
        gf::quantize_q8_1(D->gv, D->gq, m * E, s);
        gf::MvJob jk[kL], jv[kL];
        for (int l = 0; l < kL; ++l) {
            jk[l] = {D->L[l].k.q, D->gq, D->gv, D->kk + (size_t) l * kCh * KVD, nullptr, 1.0f, kQ8_0, E, KVD};
            jv[l] = {D->L[l].v.q, D->gq, D->gv, D->vv + (size_t) l * kCh * KVD, nullptr, 1.0f, kQ8_0, E, KVD};
        }
        if (!gf::mvn(jk, kL, m, s) || !gf::mvn(jv, kL, m, s)) return false;
        for (int l = 0; l < kL; ++l) {
            k_norm_rope<<<dim3(kNKV, m), kHD, 0, s>>>(D->kk + (size_t) l * kCh * KVD, KVD, D->L[l].k_norm, D->eps,
                                                       (long long) (p0 + c0), 1, D->rope_base, D->L[l].rk, kRing, KVD);
            k_norm_rope<<<dim3(kNKV, m), kHD, 0, s>>>(D->vv + (size_t) l * kCh * KVD, KVD, nullptr, D->eps,
                                                       (long long) (p0 + c0), 0, D->rope_base, D->L[l].rv, kRing, KVD);
        }
    }
    if (p0 > D->ctx_end || p0 < D->ctx_lo) D->ctx_lo = p0;   // (a gap: the context starts over)
    D->ctx_end = p0 + n;
    ++D->commits;
    return true;
}

void Glm5Model::dflash_settle() {
    auto* D = (DflState*) dfl_;
    if (D == nullptr || D->win_p0 < 0) return;
    const int64_t p0 = D->win_p0;
    const int n = (int) std::max<int64_t>(0, std::min<int64_t>(D->win_T, pos_ - p0));
    D->win_p0 = -1;
    dflash_commit(p0, n, false, fast_->cs);
}

void Glm5Model::dflash_window(int64_t p0, int T) {
    auto* D = (DflState*) dfl_;
    if (D == nullptr) return;
    D->win_p0 = p0;
    D->win_T = T;
}

bool Glm5Model::dflash_draft(int32_t y, int32_t* drafts, float* conf, std::string& err) {
    auto* D = (DflState*) dfl_;
    for (int k = 0; k < 5; ++k) {
        drafts[k] = -1;
        conf[k] = -1.0f;   // (the decode's threshold for this drafter is 0: a slot without a draft must stay below)
    }
    if (D == nullptr) { err = "dflash: not loaded"; return false; }
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto t0 = std::chrono::steady_clock::now();
    dflash_settle();
    const int64_t pos = pos_;
    const int E = g.n_embd, V = g.n_vocab, NB = D->nb, KVD = kNKV * kHD, G2 = 4 * (E / kGrp);
    const size_t q8e = (size_t) (E / 32) * kQ8;
    int64_t lo = std::max<int64_t>(D->ctx_lo, pos - kRing);
    int n_ctx = D->ctx_end == pos ? (int) (pos - lo) : 0;
    if (n_ctx <= 0) {
        n_ctx = 0;
        lo = pos;
    }
    // the rows: y's embedding, then the mask token's
    D->y_h[0] = y;
    if (F->emb_tab_d != nullptr || F->emb_q8_d != nullptr) {
        gf::embed_rows_q(F->emb_tab_d ? pack_emb_type_ : kQ8_0, F->emb_tab_d ? F->emb_tab_d : F->emb_q8_d, D->y_d, 1, E,
                         D->x, s);
    } else {
        const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
        tt->to_float(pack_emb_src_ + (size_t) y * ggml_row_size((ggml_type) pack_emb_type_, E), D->emb_h, E);
        cudaMemcpyAsync(D->x, D->emb_h, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s);
    }
    for (int i = 1; i < NB; ++i)
        cudaMemcpyAsync(D->x + (size_t) i * E, D->emb_mask, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, s);
    const dim3 cg((unsigned) ((E + 255) / 256), (unsigned) NB);
    const float scale = 1.0f / std::sqrt((float) kHD);
    const size_t att_smem = (size_t) (n_ctx + NB) * sizeof(float);
    for (int l = 0; l < kL; ++l) {
        const DflLayer& Ly = D->L[l];
        // ---- attention
        rms_rows(D->x, E, Ly.attn_norm, D->eps, E, NB, D->xn, E, s);
        gf::quantize_q8_1(D->xn, D->xnq, NB * E, s);
        gf::MvJob jd = {Ly.acp.q, D->xnq, D->xn, D->dyn, nullptr, 1.0f, kQ8_0, E, G2};
        if (!gf::mvn(&jd, 1, NB, s)) { err = "dflash: attn_conv_proj"; return false; }
        k_conv<<<cg, 256, 0, s>>>(D->xn, D->dyn, G2, 0, E / kGrp, kGrp, Ly.acb, E, D->xc, nullptr);
        gf::quantize_q8_1(D->xc, D->xcq, NB * E, s);
        gf::MvJob j[3] = {{Ly.q.q, D->xcq, D->xc, D->q, nullptr, 1.0f, kQ8_0, E, kNH * kHD},
                          {Ly.k.q, D->xcq, D->xc, D->kb, nullptr, 1.0f, kQ8_0, E, KVD},
                          {Ly.v.q, D->xcq, D->xc, D->vb, nullptr, 1.0f, kQ8_0, E, KVD}};
        if (!gf::mvn(j, 3, NB, s)) { err = "dflash: q / k / v"; return false; }
        k_norm_rope<<<dim3(kNH, NB), kHD, 0, s>>>(D->q, kNH * kHD, Ly.q_norm, D->eps, (long long) pos, 1, D->rope_base,
                                                  nullptr, 0, 0);
        k_norm_rope<<<dim3(kNKV, NB), kHD, 0, s>>>(D->kb, KVD, Ly.k_norm, D->eps, (long long) pos, 1, D->rope_base,
                                                   nullptr, 0, 0);
        k_attn<<<dim3(kNH, NB), 128, att_smem, s>>>(D->q, Ly.rk, Ly.rv, (long long) lo, n_ctx, kRing, D->kb, D->vb, NB,
                                                    scale, D->o);
        gf::quantize_q8_1(D->o, D->oq, NB * kNH * kHD, s);
        gf::MvJob jo = {Ly.o.q, D->oq, D->o, D->a, nullptr, 1.0f, kQ8_0, kNH * kHD, E};
        if (!gf::mvn(&jo, 1, NB, s)) { err = "dflash: attn_output"; return false; }
        k_conv<<<cg, 256, 0, s>>>(D->a, D->dyn, G2, 2 * (E / kGrp), E / kGrp, kGrp, Ly.acb + 2 * (size_t) E, E, nullptr,
                                  D->x);
        // ---- the FFN
        rms_rows(D->x, E, Ly.ffn_norm, D->eps, E, NB, D->xn, E, s);
        gf::quantize_q8_1(D->xn, D->xnq, NB * E, s);
        gf::MvJob jf = {Ly.fcp.q, D->xnq, D->xn, D->dyn, nullptr, 1.0f, kQ8_0, E, G2};
        if (!gf::mvn(&jf, 1, NB, s)) { err = "dflash: ffn_conv_proj"; return false; }
        k_conv<<<cg, 256, 0, s>>>(D->xn, D->dyn, G2, 0, E / kGrp, kGrp, Ly.fcb, E, D->xc, nullptr);
        gf::quantize_q8_1(D->xc, D->xcq, NB * E, s);
        gf::MvJob ju[2] = {{Ly.gate.q, D->xcq, D->xc, D->gt, nullptr, 1.0f, kQ8_0, E, kFF},
                           {Ly.up.q, D->xcq, D->xc, D->up, nullptr, 1.0f, kQ8_0, E, kFF}};
        if (!gf::mvn(ju, 2, NB, s)) { err = "dflash: ffn gate / up"; return false; }
        k_swiglu<<<(NB * kFF + 255) / 256, 256, 0, s>>>(D->gt, D->up, NB * kFF);
        gf::quantize_q8_1(D->gt, D->gq2, NB * kFF, s);
        gf::MvJob jn = {Ly.down.q, D->gq2, D->gt, D->f, nullptr, 1.0f, kQ8_0, kFF, E};
        if (!gf::mvn(&jn, 1, NB, s)) { err = "dflash: ffn_down"; return false; }
        k_conv<<<cg, 256, 0, s>>>(D->f, D->dyn, G2, 2 * (E / kGrp), E / kGrp, kGrp, Ly.fcb + 2 * (size_t) E, E, nullptr,
                                  D->x);
    }
    // ---- the head over rows 1 .. NB - 1 (row 0 is y's), the selector's gate, the walk
    const int R = NB - 1;
    rms_rows(D->x + E, E, D->out_norm, D->eps, E, R, D->hn, E, s);
    gf::quantize_q8_1(D->hn, D->hnq, R * E, s);
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, D->hnq, D->hn, D->logits, nullptr, 1.0f, ow.type, E, V};
    if (ow.type == 0) o.w = ow.f32;
    if (gf::mvn_supported(ow.type)) {
        if (!gf::mvn(&o, 1, R, s)) { err = "dflash: output head"; return false; }
    } else {
        for (int i = 0; i < R; ++i) {
            gf::MvJob ot = o;
            ot.xq = (const uint8_t*) D->hnq + (size_t) i * q8e;
            ot.xf = D->hn + (size_t) i * E;
            ot.y = D->logits + (size_t) i * V;
            if (!gf::mv(&ot, 1, s)) { err = "dflash: output head (type " + std::to_string(ow.type) + ")"; return false; }
        }
    }
    gf::MvJob js = {D->sel_h.q, D->hnq, D->hn, D->gate, nullptr, 1.0f, kQ8_0, E, kRank};
    if (!gf::mvn(&js, 1, R, s)) { err = "dflash: selector_hidden"; return false; }
    const int chunk = (V + kTkCh - 1) / kTkCh;
    k_topk_part<<<dim3(kTkCh, R), 256, (size_t) chunk * sizeof(float), s>>>(D->logits, V, chunk, D->tk_v, D->tk_i);
    k_select<<<1, 256, 0, s>>>(D->tk_v, D->tk_i, kTkCh * kTopK, D->gate, D->sel_pred, D->sel_succ, D->y_d, R, D->p_min,
                               D->res_d, D->res_conf_d);
    cudaMemcpyAsync(D->res_h, D->res_d, (size_t) (1 + R) * sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(D->res_conf_h, D->res_conf_d, (size_t) R * sizeof(float), cudaMemcpyDeviceToHost, s);
    if (cudaStreamSynchronize(s) != cudaSuccess || gf::launch_errors() != D->launch_err0) {
        err = std::string("dflash: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { err = std::string("dflash: ") + cudaGetErrorString(e); return false; } }
    const int n = std::max(0, std::min(D->res_h[0], std::min(R, 5)));
    for (int i = 0; i < n; ++i) {
        drafts[i] = D->res_h[1 + i];
        conf[i] = D->res_conf_h[i];
    }
    // STRATA_Q4X_DFLASH_DUMP=<file> (debug): a line a draft - the position (y's), y, the context rows | the drafts
    if (D->dump != nullptr) {
        std::fprintf(D->dump, "%lld %d %d |", (long long) pos, y, n_ctx);
        for (int i = 0; i < n; ++i) std::fprintf(D->dump, " %d", drafts[i]);
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
