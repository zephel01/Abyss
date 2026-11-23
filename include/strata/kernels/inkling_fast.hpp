// include/strata/kernels/inkling_fast.hpp - the Inkling 2 (Thinking Machines Inkling 2 Mini, GGUF arch inkling2_loop)
// kernels: the depth-attention mixes of the attention-residual blocks, the loop controllers' elementwise step, the short
// causal convolutions (static taps, plus per-head input-dependent taps on the full layers' K / V), the L2 clips of q /
// k / r with the full layers' log-scaled positions, the attention with a relative-position bias and a per-query sink
// logit over an FP16 KV cache (a ring for a window layer), the sigmoid output gate and the final logit softcap.  The
// math follows the model's definition (llama.cpp's src/models/inkling2.cpp reads the same GGUF); the kernels are the
// engine's own.  Every kernel takes T rows (the token path T = 1, a prompt sub-batch more).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::ink {

constexpr int kHeadDim = 128;
constexpr int kRel = 16;          // the relative-position bias's dims a head
constexpr int kConv = 4;          // the short convolutions' taps
constexpr int kMaxStates = 8;     // a mix's states at most (the earlier blocks + the current residual)
constexpr int kMaxSplits = 64;    // a row's attention keys over up to this many blocks

/// The states of a mix: n pointers, each T rows of n_embd floats (row t at p[i] + t * stride)
struct States {
    const float* p[kMaxStates] = {};
    int n = 0;
    int64_t stride = 0;
};

/// y[t] = rms(x[t]) (* gamma when non-null), rows of n floats: x row t at x + t xs, y at y + t ys
void rms_rows(const float* x, int64_t xs, const float* gamma, float eps, int n, int T, float* y, int64_t ys,
              cudaStream_t s);

/// The depth attention: the states normalised (rms, eps 1e-6, no gamma), each query's softmax over them of
/// q . state / sqrt(n_embd), the weighted sum of the NORMALISED states.  query [nq][n_embd]; out [T][nq][n_embd]
void mix(const States& st, const float* query, int nq, int n_embd, int T, float* out, cudaStream_t s,
         float* part = nullptr);   // part: mix_part_floats(T, n_embd) of the caller's scratch (null: a shared one, grown
                                   // on demand - not for a captured graph, whose address must stay)
size_t mix_part_floats(int T, int n_embd);

/// The loop controller's input: cat[t] = [rms(cur[t]) | rms(msg[t])] (2 n_embd, eps 1e-6, no gamma).  cur row t at
/// cur + t cs, msg row t at msg + t ms; cat [T][2 n_embd]
void ctrl_cat(const float* cur, int64_t cs, const float* msg, int64_t ms, int n_embd, int T, float* cat, cudaStream_t s);

/// out = cur - tanh(f) cur + tanh(g) msg (the controller's forget / gain on the message), n floats a row, T rows
void ctrl_apply(const float* cur, int64_t cs, const float* msg, int64_t ms, const float* g, const float* f, int n, int T,
                float* out, int64_t os, cudaStream_t s);

/// The short causal convolution of T rows in order, channel c: x_t = in[t][c], with the state's last kConv - 1 inputs
/// of the channel before row 0 (state [C][kConv - 1], oldest first; updated to the last ones):
///   out[t][c] = (x_t + sum_k w[c][k] x_{t-3+k} + dyn) * scale,  dyn = sum_lag x_{t-lag} dw[t][c / head_ch][lag]
/// (dw [T][C / head_ch][kConv], null: none), scale = 1 / sqrt((dw ? 2 : 1) kConv + 1).  out may be in.
void conv_rows(const float* in, int64_t is, const float* w, const float* dw, int64_t dws, int head_ch, float* state,
               int C, int T, float* out, int64_t os, cudaStream_t s);

/// The verify window's conv_rows: the same arithmetic a row, the state read but left as it is (in and out must not
/// overlap - the window's inputs stay for conv_commit)
void conv_win(const float* in, int64_t is, const float* w, const float* dw, int64_t dws, int head_ch, float* state,
              int C, int T, float* out, int64_t os, cudaStream_t s);

/// After a verify window: every slot's conv state ([K | V | attn post | ffn post] channels x 3, K / V KV channels,
/// the posts E) moved past the window's first n rows - slot b's rows at save + b slot_floats, [part][tm][part's
/// channels].  states_d: n_slots device pointers
void conv_commit(float* const* states_d, int n_slots, const float* save, size_t slot_floats, int KV, int E, int tm, int n,
                 cudaStream_t s);

/// conv_rows over the T rows in parallel (the same arithmetic a row): out must not overlap in
void conv_rows_par(const float* in, int64_t is, const float* w, const float* dw, int64_t dws, int head_ch, float* state,
                   int C, int T, float* out, int64_t os, cudaStream_t s);

/// The L2 clip of T rows of H heads of `dim` (x[t][h][:], rows at x + t xs): x *= scale[h][:] (null: none), then
/// x *= min(1, thr / (|x| + 1e-6)), then (log_floor > 0) x *= 1 + alpha log(max(1, (p0 + t + 1) / log_floor))
void clip_heads(float* x, int64_t xs, const float* scale, int dim, int H, float thr, int64_t p0, float log_floor,
                float alpha, int T, cudaStream_t s);

/// The KV rows of T positions p0 + t into the FP16 caches' rows (p0 + t) % ring ([ring][n_head_kv][128]): k as it is,
/// v normalised per head (rms, eps, no gamma)
void kv_store(const float* k, int64_t ks, const float* v, int64_t vs, float eps, int n_head_kv, int64_t p0, int T,
              int ring, uint16_t* kc, uint16_t* vc, cudaStream_t s);

/// A decode token's (T = 1) conv_rows of k and v (the head's channels: head_ch = 128), clip_heads of q, k and rel and
/// kv_store in one launch - the same arithmetic as those calls one after another (STRATA_INK_FUSE=0: those calls)
struct PostProj {
    float *qf, *kf, *vf, *rf, *conv;            // (conv: the layer's state; v's at conv + KV * 3)
    const float *kd, *vd, *k_conv, *v_conv;     // (kd / vd: the dynamic taps, null on a window layer)
    const float *q_clip, *rel_clip;
    int64_t dws, KV, HD;
    int H, Hkv, ring;
    int64_t p;
    float qk_clip, rel_clip_thr, lf, alpha, eps;
    uint16_t *kc, *vc;
    const int64_t* pd = nullptr;                // (non-null: the position read from this device word instead of p)
};
void post_proj(const PostProj& a, cudaStream_t s);

/// A decode token's (T = 1) fused steps, one launch each and the same arithmetic as the calls they stand for
/// (STRATA_INK_FUSE=0: those calls).  n_embd <= 4096, a multiple of 32.
/// mix_tok: [pre] conv_rows of pre_in (E channels, taps pre_w, state pre_state) into pre_tmp and add_rows of it into
/// pre_out (pre_add), or conv_rows into pre_out; then the mix (out [nq][E]); then [norm] rms_q8 of n_norm rows - row j from the
/// mix's row (norm_rs ? j : 0), gamma[j] -> x[j] / xq[j] - or [cat] ctrl_cat of n_cat controllers: cur rows out + c
/// cat_cs, msg rows msg + c E -> cat
struct MixTok {
    const float* pre_in = nullptr;
    const float* pre_w = nullptr;
    float* pre_state = nullptr;
    float* pre_tmp = nullptr;
    float* pre_out = nullptr;
    bool pre_add = false;
    States st;
    const float* query = nullptr;
    int nq = 0;
    float* out = nullptr;
    int n_norm = 0;
    int norm_rs = 0;
    const float* gamma[3] = {};
    float* x[3] = {};
    void* xq[3] = {};
    float eps = 0.0f;
    int n_cat = 0;
    int64_t cat_cs = 0;
    const float* msg = nullptr;
    float* cat = nullptr;
    float* part = nullptr;   // mix_tok_floats() of scratch (the chunks' partials; rows > 1: mix_part_floats(rows, ..))
    // the verify window's (rows > 1, up to 8, no controllers): row t of the states at st.p[s] + t st.stride, of out at
    // out + t out_rs, of x[j] / xq[j] n_embd apart; the pre step conv_win's (the state read, left: row t from pre_in's
    // rows 0 .. t), then add_rows' add (pre_add) or into pre_out; pre_rows: that kind with one row
    int rows = 1;
    bool pre_rows = false;
    int64_t out_rs = 0;
    int pre_s = -1;   // (set by mix_tok: the state that is pre_out, -1 none)
};
size_t mix_tok_floats();
void mix_tok(const MixTok& a, int n_embd, cudaStream_t s);

/// ctrl_apply of n rows (cur rows cs apart, msg / g / f / out rows n_embd apart) and each row's rms_q8 with gamma[j]
/// into x[j] / xq[j]
struct CtrlTok {
    const float* cur = nullptr;
    int64_t cs = 0;
    const float *msg = nullptr, *g = nullptr, *f = nullptr;
    float* out = nullptr;
    int n = 0;
    const float* gamma[3] = {};
    float* x[3] = {};
    void* xq[3] = {};
    float eps = 0.0f;
};
void ctrl_tok(const CtrlTok& a, int n_embd, cudaStream_t s);

/// attn of one row (T = 1) followed by gate_sigmoid with gate and quantize_q8_1 into outq (out kept too); pd non-null:
/// the position read from that device word (p0 still gives the split count)
void attn_tok(const float* q, const float* r, const float* sink, const float* rel_bias, int extent, const uint16_t* kc,
              const uint16_t* vc, int ring, int n_head, int n_head_kv, int64_t p0, const int64_t* pd, int window,
              float scale, float* part, int max_splits, const float* gate, float* out, void* outq, cudaStream_t s);
/// The split count attn gives a row at position p (a graph's key)
int attn_splits(int64_t p, int window, int max_splits);
/// The split grid attn(rows) launches for a window whose last row is at p_last: the last row's count rounded up to a
/// power of two, 8 at the least (each row still its own count - a captured window's graph key)
int attn_rows_grid(int64_t p_last, int window, int max_splits);
/// *d = p (a captured graph's position word, set before its launch)
void set_pos(int64_t* d, int64_t p, cudaStream_t s);

/// The verify window's post_proj over T rows at positions a.p .. (a.pd non-null: from that device word): k / v convolved from kraw / vraw (rows KV apart)
/// into a.kf / a.vf with the state read but left (conv_win's), the clips and the cache rows of every row - each row
/// post_proj's arithmetic
void post_proj_rows(const PostProj& a, const float* kraw, const float* vraw, int T, cudaStream_t s);

/// The verify window's loop controllers, n of them over T rows in one launch each (ctrl_cat's / ctrl_apply's rows):
/// controller c's current rows cur[c] + t cs, its message rows msg + c E + t ms; cat rows at cat + c cat_c + t 2E;
/// g / f rows at g / f + c gf_c + t E; out rows out[c] + t E
struct CtrlRows {
    const float* cur[4] = {};
    int64_t cs = 0;
    const float* msg = nullptr;
    int64_t ms = 0;
    float* cat = nullptr;
    int64_t cat_c = 0;
    const float *g = nullptr, *f = nullptr;
    int64_t gf_c = 0;
    float* out[4] = {};
};
void ctrl_cat_n(const CtrlRows& a, int n, int E, int T, cudaStream_t s);
void ctrl_apply_n(const CtrlRows& a, int n, int E, int T, cudaStream_t s);

/// The scratch (floats) attn needs for T rows of n_head heads in up to max_splits key ranges.
size_t attn_part_floats(int T, int n_head, int max_splits);

/// out[t][h] = sum_j p_j V[j], p over the visible keys j of row t (p = p0 + t; j in lo..p, lo = max(0, p - window + 1),
/// window 0: lo = 0) and one sink: logit_j = scale q[t][h] . K[j] + b(p - j), the sink's logit sink[t][h] (no value);
/// b(d) = r[t][h] . rel_bias[d] for d < extent, else 0 (rel_bias [extent][kRel]).  q [T][H][128] (rows qs apart),
/// r [T][H][kRel] (rs), sink [T][H] (ss); K / V of KV head h / (H / n_head_kv) from cache row j % ring.
void attn(const float* q, int64_t qs, const float* r, int64_t rs, const float* sink, int64_t ss, const float* rel_bias,
          int extent, const uint16_t* kc, const uint16_t* vc, int ring, int n_head, int n_head_kv, int64_t p0, int T,
          int window, float scale, float* part, int max_splits, float* out, int64_t os, cudaStream_t s,
          bool rows = false,    // rows: each row the split count the token path gives its position (the verify window)
          const int64_t* p0d = nullptr);   // (rows: non-null, p0 read from this device word - p0 still gives the grid)

/// attn for a prompt's rows (T of them at once) on tensor cores: the same math, S = q K^T and P V in FP16 with F32
/// accumulation (P rounded to FP16), the bias as FP16 pairs (about F32).  False (nothing launched): the caller runs attn - AMD, a GPU
/// before sm_80, an unsupported head grouping, or STRATA_INK_PREFILL_ATTN=0.  max_splits: part's capacity.
bool attn_prompt(const float* q, int64_t qs, const float* r, int64_t rs, const float* sink, int64_t ss,
                 const float* rel_bias, int extent, const uint16_t* kc, const uint16_t* vc, int ring, int n_head,
                 int n_head_kv, int64_t p0, int T, int window, float scale, float* part, int max_splits, float* out,
                 int64_t os, cudaStream_t s);
/// Whether attn_prompt runs at all on this build and setting (CUDA, STRATA_INK_PREFILL_ATTN not 0).
bool attn_prompt_on();

/// o[t][i] *= sigmoid(g[t][i]), n floats a row (rows os / gs apart), T rows
void gate_sigmoid(float* o, int64_t os, const float* g, int64_t gs, int n, int T, cudaStream_t s);

/// h[t] += a[t], n floats a row
void add_rows(float* h, int64_t hs, const float* a, int64_t as, int n, int T, cudaStream_t s);

/// x = cap tanh(x / cap), n floats (cap <= 0: nothing)
void softcap(float* x, int64_t n, float cap, cudaStream_t s);

}  // namespace strata::kernels::ink
