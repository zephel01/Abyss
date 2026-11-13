// include/strata/kernels/laguna_fast.hpp - the Laguna (poolside Laguna-S-2.1, gguf arch `laguna`) attention kernels:
// head_dim 128, GQA groups of any size (6 and 9 in Laguna-S), q / k RMS norms, NEOX RoPE over the first n_rot dims
// with a per-pair frequency table (YaRN on the full-attention layers, plain on the sliding-window ones) and a
// magnitude factor, an FP16 KV cache that may be a ring (a sliding-window layer keeps window + slack rows: position j
// lives in row j % ring), causal attention with an optional window, and the per-head softplus output gate.  The math
// follows llama.cpp's src/models/laguna.cpp; the kernels are the engine's own.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::lag {

constexpr int kHeadDim = 128;
constexpr int kMaxSplits = 64;   // a token's attention: its keys over up to this many blocks (256 keys at the least)

/// The RoPE pair frequencies (n_rot / 2 of them, float64) for theta_i(p) = p * freq[i]: base^(-2i / n_rot), with
/// YaRN (ggml's rope_yarn: ramp between corr dims from beta_fast / beta_slow over orig_ctx, ext_factor 1) when
/// freq_scale != 1.  mscale (out): the cos / sin factor (1 + 0.1 ln(1 / freq_scale) under YaRN, else 1).
void rope_freqs(int n_rot, double base, double freq_scale, int orig_ctx, double beta_fast, double beta_slow,
                double* freq, float& mscale);

/// T rows at positions p0 + t: q [T][n_head][128] -> rms(q) * q_norm, RoPE -> q_out; k [T][n_head_kv][128] ->
/// rms(k) * k_norm, RoPE, and v as they are, into the FP16 caches' rows (p0 + t) % ring ([ring][n_head_kv][128]).
/// freq: device, n_rot / 2 doubles (rope_freqs).  cs: rope_table's rows for these positions and this freq / mscale
/// (null: computed here - the same values).
void attn_prep_rows(const float* q, const float* k, const float* v, const float* q_norm, const float* k_norm,
                    float eps, int n_head, int n_head_kv, int n_rot, const double* freq, float mscale, int64_t p0,
                    int T, int ring, uint16_t* kc, uint16_t* vc, float* q_out, cudaStream_t s,
                    const float2* cs = nullptr);

/// The (cos, sin) x mscale attn_prep_rows uses for rows p0 .. p0 + T - 1: cs [T][n_rot / 2] (a token's layers of one
/// kind share them: one table a token instead of a sincos a layer)
void rope_table(const double* freq, int n_rot, float mscale, int64_t p0, int T, float2* cs, cudaStream_t s);

/// The scratch (floats) attn needs for `rows` query rows of n_head heads split into up to max_splits key ranges.
size_t attn_part_floats(int rows, int n_head, int max_splits);

/// out[t][h] = softmax(scale * q[t][h] . K[j]) V[j] over the positions j of row t (p = p0 + t): lo..p with lo =
/// max(0, p - window + 1) (window 0: lo = 0), K / V of KV head h / (n_head / n_head_kv) read from cache row j % ring.
/// The caller keeps every needed position in the ring (ring >= window + T - 1 for a window layer, ring > p0 + T - 1
/// otherwise).  part: attn_part_floats(T, n_head, max_splits) floats; with few rows the keys are split across blocks.
void attn(const float* q, const uint16_t* kc, const uint16_t* vc, int ring, int n_head, int n_head_kv, int64_t p0,
          int T, int window, float scale, float* part, int max_splits, float* out, cudaStream_t s);

/// o[t][h][d] *= softplus(g[t][h]) (the per-head output gate), T rows of n_head heads of 128.
void gate_rows(float* o, const float* g, int T, int n_head, cudaStream_t s);

}  // namespace strata::kernels::lag
