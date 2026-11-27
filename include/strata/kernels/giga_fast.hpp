// include/strata/kernels/giga_fast.hpp - ai-sage GigaChat4's (GGUF arch gigachat4) one-token decode kernels.
//
// The pieces the shared kernels do not cover (src/core/giga_fast_path.cu runs them in order; the delta-net's conv /
// gates / recurrence are qwen35_fast's, the MLA's absorb / attention ling_fast's, its gated output raon_fast's):
//   * GatedNorm: an RMSNorm whose output is scaled by a rank-r gate,
//       n = x * rsqrt(mean(x^2) + eps), t = P_in . g_in (r values), u = P_out . act(t), y = n * w * G(u)
//     the model's (form 6): g_in = n * w, act SiLU, G = gate_w * sigmoid; the other forms are checks.  P_in is the
//     model's gate_up_projection (n -> r: the GGUF's *_gate_up), P_out its gate_down_projection (*_gate_down)
//   * the delta-net's closing norm with the scaled sigmoid gate: y = rms(o_head) * ssm_norm * scale * sigmoid(z)
//   * the value heads' key pairing as repeat_interleave (q / k head v / (h_v / h_k) for value head v) - the recurrence
//     kernel pairs v % h_k: gdn_expand writes q / k once a value head
//   * the MLA's rotary table with YaRN (the pair frequencies from the host, cos / sin x mscale) and the cache row
//     from an already normed latent: FP16 [512 latent | 64 roped key], q's rope parts roped in place
//   * the DSA indexer: its keys, the scores of every position, the top_k positions and their cache rows gathered
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::giga {

constexpr int kRankMax = 16;   // the gate's rank
constexpr int kLat = 512, kRot = 64, kRow = kLat + kRot, kNope = 128, kQHead = kNope + kRot;

/// GatedNorm's form: an int of fields (STRATA_GIGA_NORM in giga_fast_path.cu)
///   bits 0-1  the gate's input: 0 x as it is, 1 the rms-normed x, 2 the rms-normed x * w
///   bits 2-4  the activation between down and up: 0 none, 1 SiLU, 2 ReLU, 3 GELU (erf), 4 tanh, 5 sigmoid
///   bits 5-6  the gate: 0 gate_w * sigmoid(u), 1 sigmoid(gate_w * u), 2 1 + gate_w * tanh(u), 3 sigmoid(u) (no weight)
///   bit 7     y without the gate (the plain RMSNorm: a check)
constexpr int kNormIn = 3, kNormActShift = 2, kNormAct = 7, kNormGateShift = 5, kNormGate = 3, kNormPlain = 128;

/// h += add (add null: none; in place), then y = GatedNorm(h).  y may be h (add null).  rank <= 16; p_in [rank][n],
/// p_out [n][rank] (F32 rows).  rows > 1 (a prompt path): rows of n floats in h / add / y, each the token path's
/// arithmetic
void gated_norm(float* h, const float* add, const float* w, const float* p_in, const float* p_out, float gate_w, float eps,
                int n, int rank, int form, float* y, cudaStream_t s, int rows = 1);
/// y[h][c] = o[h][c] * rsqrt(mean_c(o[h]^2) + eps) * ssm_norm[c] * G(z[h][c]); G = scale * sigmoid (silu false) or SiLU
void gdn_out(const float* o, const float* z, const float* ssm_norm, float* y, int h_v, int S, float eps, float scale,
             bool silu, cudaStream_t s, int rows = 1);
/// h [q: h_k S | k: h_k S | v: h_v S] -> out [q: h_v S | k: h_v S | v: h_v S], q / k head v / (h_v / h_k) a value head
/// (rows > 1: rows of (2 h_k + h_v) S in, 3 h_v S out)
void gdn_expand(const float* h, float* out, int h_k, int h_v, int S, cudaStream_t s, int rows = 1);
/// cs[j] = mscale * (cos, sin)(pos * freq[j]), j < kRot / 2 (freq: the pairs' frequencies as doubles on the device)
void rope_table(int64_t pos, const double* freq, float mscale, float2* cs, cudaStream_t s);
/// q [n_head][kQHead]: each head's rope part roped in place (interleaved pairs: dims 2j, 2j + 1); row (FP16 [kRow]) =
/// lat [kLat] as it is | kv_rot [kRot] roped.  interleave false: the half-split pairs (j, j + kRot / 2)
void mla_prep(float* q, int n_head, const float* lat, const float* kv_rot, const float2* cs, uint16_t* row,
              bool interleave, cudaStream_t s);
/// x[i] *= a
void scale(float* x, float a, int n, cudaStream_t s);

// ---- the DSA indexer (the model's GigaChat4DSAIndexer): a key of kIdxKey dims a position, a query of n_heads x
// kIdxKey, score(s) = scale * sum_h w[h] * relu(iq[h] . key[s]); past top_k positions the attention reads the top_k
// best-scoring positions alone.  The rotary here is the half-split form (pairs j, j + kRot / 2) on a row's first kRot
// dims, the attention's table
constexpr int kIdxKey = 128;
/// key_row (FP16 [kIdxKey]) = rope(rms(kraw) * k_norm); iq (null: none) [n_heads][kIdxKey]: each head roped in place
void idx_prep(float* iq, int n_heads, const float* kraw, const float* k_norm, float eps, const float2* cs,
              uint16_t* key_row, cudaStream_t s);
/// score[s] = scale * sum_h w[h] * relu(iq[h] . keys[s]), s < n (keys FP16 rows of kIdxKey; n_heads 64)
void idx_score(const float* iq, const float* w, const uint16_t* keys, int64_t n, float scale, float* score,
               cudaStream_t s);
/// sel[0, k): the positions of the k largest of score[0, n) (n > k), ascending; equal scores: the lower positions
void idx_topk(const float* score, int64_t n, int k, int* sel, cudaStream_t s);
/// out[i] = cache[sel[i]], rows of kRow FP16, i < k
void gather_rows(const uint16_t* cache, const int* sel, int k, uint16_t* out, cudaStream_t s);

// ---- the prompt path (src/core/giga_prefill.cu): the same arithmetic over rows (row t = position p0 + t)
/// gated_norm over rows with strides: h rows ldh apart, add rows lda apart, y rows ldy apart (n floats of each)
void gated_norm_ld(float* h, int ldh, const float* add, int lda, const float* w, const float* p_in, const float* p_out,
                   float gate_w, float eps, int n, int rank, int form, float* y, int ldy, cudaStream_t s, int rows);
/// rope_table + mla_prep a row (interleaved pairs): q rows of ldq floats ([n_head][kQHead] in each) roped in place; the
/// cache rows (cache: the first row's place) = lat rows [kLat] | the roped kv rows' [kLat, kRow) (kv rows of kRow)
void mla_prep_rows(float* q, int ldq, int n_head, const float* lat, const float* kv, const double* freq, int64_t p0,
                   int rows, uint16_t* cache, cudaStream_t s);
/// idx_prep's key a row: keys (the first row's place) from kraw rows of kIdxKey
void idx_keys_rows(const float* kraw, const float* k_norm, float eps, const double* freq, int64_t p0, int rows,
                   uint16_t* keys, cudaStream_t s);
/// idx_prep's query rope a row: iq rows of n_heads x kIdxKey, row t at position p0 + t
void idx_q_rope_rows(float* iq, int n_heads, const double* freq, int64_t p0, int rows, cudaStream_t s);
/// idx_score a row on tensor cores (the query heads as FP16, F32 sums): score[t * ld + s] over s <= p0 + t; 64 heads
void idx_score_rows(const float* iq, const float* w, const uint16_t* keys, int64_t p0, int rows, float* score, int64_t ld,
                    cudaStream_t s);
/// idx_topk a row: row t picks k of its p0 + t + 1 scores (> k) into sel + t * k
void idx_topk_rows(const float* score, int64_t ld, int64_t p0, int rows, int k, int* sel, cudaStream_t s);
/// The prompt's MLA attention on tensor cores (raon_fast's mla_prompt_attn; scale on q): row t at position p0 + t reads
/// cache rows [0, p0 + t], or - past k positions, sel given - the k rows of sel + (t - sel_row0) * k.  qabs
/// [T][n_head][kLat] in, ctx the same shape out (apart), q rows of ldq floats.  prompt_attn_ok false: not here
bool prompt_attn_ok();
void mla_prompt_attn(const float* qabs, float* ctx, const float* q, int ldq, const uint16_t* cache, int64_t p0, int T,
                     int n_head, float scale, const int* sel, int k, int sel_row0, cudaStream_t s);
/// launches that failed (all giga kernels)
int launch_errors();

}  // namespace strata::kernels::giga
