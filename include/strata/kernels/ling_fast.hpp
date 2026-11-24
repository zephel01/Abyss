// include/strata/kernels/ling_fast.hpp - inclusionAI Ling-3.x's (GGUF arch bailingmoe3) one-token decode kernels.
//
// The pieces the shared GLM / qwen35 kernels do not cover (src/core/ling_fast_path.cu runs them in order):
//   * KDA's prep with full-width gates: the three short convs (+ SiLU, the history slide), the q / k L2 norms, and
//     g1 = lb * sigmoid((f + dt_bias) * a) per channel, a = exp(A_log) per head (the recurrence is glm_fast's kda_rec)
//   * the MLA attention over every position: the latent rms(kv[0, 512)) * kv_norm and the roped key kv[512, 576) into
//     the cache row (FP16 [576]), q's rope dims roped in place; q_abs = k_b . q_nope per head; the scores
//     scale * (q_abs . lat + q_rot . k_rot), a softmax over the positions, ctx = sum p lat (in fixed chunks, summed in
//     chunk order), o = (v_b . ctx) * sigmoid(gate) per head
//   * the routed experts' group mask: the router's scores sigmoid(logit) + bias in n_group groups, the best topk_group
//     groups by the sum of their best two kept - the others' bias -1e30, so the route never picks them
// RoPE is the interleaved-pair form (pair j = dims 2j, 2j + 1, frequency base^(-2j / 64)) - the model's
// apply_rotary_pos_emb_interleave (its config's rope_interleave).
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::ling {

constexpr int kLat = 512;           // the latent
constexpr int kRot = 64;            // the rope key / a q head's rope part
constexpr int kRow = kLat + kRot;   // a cache row (FP16)
constexpr int kNope = 128;
constexpr int kQHead = kNope + kRot;
constexpr int kV = 128;
constexpr int kChunk = 256;         // the context sums' positions a block

/// KDA prep (one token).  proj: q | k | v (3 x d_inner, as the projections wrote them); conv_w[r]: [d_inner][d_conv];
/// conv_state: 3 x d_inner x (d_conv - 1) (glm_fast's layout); out: q | k | v after the convs (q, k L2-normed);
/// f: f_a's output (d_inner); g1: the log decay (d_inner)
void kda_prep(const float* proj, const float* const conv_w[3], float* conv_state, float* out, const float* f,
              const float* dt_bias, const float* a, float lb, float* g1, int n_head, int head_dim, int d_conv,
              cudaStream_t s);
/// The rope pair frequencies of a token (cos, sin for kRot / 2 pairs at position pos) into cs (device, kRot / 2)
void rope_table(int64_t pos, double base, float2* cs, cudaStream_t s);
/// MLA prep for one token: q [n_head][kQHead] (its rope parts roped in place), kv [kRow] (the kv_a projection) ->
/// cache_row (FP16 [kRow]: the latent rms-normed by kv_norm, then the roped key)
void mla_prep(float* q, int n_head, const float* kv, const float* kv_norm, float eps, const float2* cs,
              uint16_t* cache_row, cudaStream_t s);
/// q_abs[h][c] = k_b[h][c] . q_nope[h] (k_b F32 [n_head][kLat][kNope])
void mla_absorb(const float* q, const float* k_b, int n_head, float* qabs, cudaStream_t s);
/// The attention over cache rows [0, n): ctx[h][c] = softmax_t(scale * (q_abs_h . lat_t + q_rot_h . k_rot_t)) . lat_t[c].
/// sc: [n_head][n] scratch; part: [ceil(n / kChunk)][n_head][kLat] scratch
void mla_attend(const float* qabs, const float* q, const uint16_t* cache, int64_t n, float scale, int n_head,
                float* sc, float* part, float* ctx, cudaStream_t s);
/// the floats mla_attend's sc / part take for n positions
inline size_t mla_sc_floats(int64_t n, int n_head) { return (size_t) n * (size_t) n_head; }
inline size_t mla_part_floats(int64_t n, int n_head) {
    return (size_t) ((n + kChunk - 1) / kChunk) * (size_t) n_head * kLat;
}
/// o[h][v] = (v_b[h][v] . ctx[h]) * sigmoid(gate[h]) (v_b F32 [n_head][kV][kLat])
void mla_out(const float* ctx, const float* v_b, const float* gate, int n_head, float* o, cudaStream_t s);
/// The group mask: out[e] = bias[e] (0 without one) where e's group is kept, else -1e30
void group_bias(const float* logits, const float* bias, int n_expert, int n_group, int topk_group, float* out,
                cudaStream_t s);
/// launches that failed (all ling kernels)
int launch_errors();

}  // namespace strata::kernels::ling
