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
void mix(const States& st, const float* query, int nq, int n_embd, int T, float* out, cudaStream_t s);

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

/// The L2 clip of T rows of H heads of `dim` (x[t][h][:], rows at x + t xs): x *= scale[h][:] (null: none), then
/// x *= min(1, thr / (|x| + 1e-6)), then (log_floor > 0) x *= 1 + alpha log(max(1, (p0 + t + 1) / log_floor))
void clip_heads(float* x, int64_t xs, const float* scale, int dim, int H, float thr, int64_t p0, float log_floor,
                float alpha, int T, cudaStream_t s);

/// The KV rows of T positions p0 + t into the FP16 caches' rows (p0 + t) % ring ([ring][n_head_kv][128]): k as it is,
/// v normalised per head (rms, eps, no gamma)
void kv_store(const float* k, int64_t ks, const float* v, int64_t vs, float eps, int n_head_kv, int64_t p0, int T,
              int ring, uint16_t* kc, uint16_t* vc, cudaStream_t s);

/// The scratch (floats) attn needs for T rows of n_head heads in up to max_splits key ranges.
size_t attn_part_floats(int T, int n_head, int max_splits);

/// out[t][h] = sum_j p_j V[j], p over the visible keys j of row t (p = p0 + t; j in lo..p, lo = max(0, p - window + 1),
/// window 0: lo = 0) and one sink: logit_j = scale q[t][h] . K[j] + b(p - j), the sink's logit sink[t][h] (no value);
/// b(d) = r[t][h] . rel_bias[d] for d < extent, else 0 (rel_bias [extent][kRel]).  q [T][H][128] (rows qs apart),
/// r [T][H][kRel] (rs), sink [T][H] (ss); K / V of KV head h / (H / n_head_kv) from cache row j % ring.
void attn(const float* q, int64_t qs, const float* r, int64_t rs, const float* sink, int64_t ss, const float* rel_bias,
          int extent, const uint16_t* kc, const uint16_t* vc, int ring, int n_head, int n_head_kv, int64_t p0, int T,
          int window, float scale, float* part, int max_splits, float* out, int64_t os, cudaStream_t s);

/// o[t][i] *= sigmoid(g[t][i]), n floats a row (rows os / gs apart), T rows
void gate_sigmoid(float* o, int64_t os, const float* g, int64_t gs, int n, int T, cudaStream_t s);

/// h[t] += a[t], n floats a row
void add_rows(float* h, int64_t hs, const float* a, int64_t as, int n, int T, cudaStream_t s);

/// x = cap tanh(x / cap), n floats (cap <= 0: nothing)
void softcap(float* x, int64_t n, float cap, cudaStream_t s);

}  // namespace strata::kernels::ink
