// include/strata/kernels/ernie_fast.hpp - baidu ERNIE-5.5-Flash-Preview's decode kernels beyond DeepSeek V4's: one token.
//
// The math is the survey's (worklog 2026-11-25_ernie55-port-survey.md section 1), checked against the float64
// reference recipes/ernie55/ref.py:
//   * the MoE's two routers: s = sigmoid(l0) + sigmoid(l1);
//   * VHA, the value heads' mix after the attention: o_h <- sum_g M[h][g] o_g (M = I + V U^T, folded at load);
//   * the output gate: z *= sigmoid(g), the sigmoid rounded to BF16 first (both reference stacks);
//   * SiTU: 4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25) (the expert kernels' situ_act, for the dense FFN and the shared
//     expert, whose down projections are BF16 GEMVs on the float activation);
//   * MultiMax on the logits: x + t0 relu(r0 - x) + t1 relu(x - r1) + t2 relu(r2 - x)^2 + t3 relu(x - r3)^2;
//   * the MLA layers' attention: q = [nope | rope] a head, one 512 latent + a 64-dim rope key a position (F32 cache),
//     kv_b absorbed into the query (q_lat = W_k^T q_nope) and the output (v = W_v o_lat), plain RoPE on adjacent pairs,
//     a sink a head in the softmax's denominator.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::ernie {

void router2(const float* l0, const float* l1, float* s, int n, cudaStream_t st);
/// o [H][D] in place (H <= 64): o[h][d] = sum_g M[h][g] o[g][d]
void vha(float* o, const float* M, int H, int D, cudaStream_t st);
/// z[i] *= sigmoid(gl[i]) (rounded to BF16 when bf16)
void gate_mul(float* z, const float* gl, int n, int bf16, cudaStream_t st);
/// h = SiTU(g, u)
void situ(const float* g, const float* u, float* h, int n, cudaStream_t st);
/// y = x / sqrt(mean(x^2) + eps) * w (one row of n <= 16384)
void rms(const float* x, const float* w, float* y, int n, float eps, cudaStream_t st);
/// MultiMax in place over n logits (r, t: 4 values each, device)
void multimax(float* x, int n, const float* r, const float* t, cudaStream_t st);

/// the MLA layer's cache row of position p: [rms(kva[0, L)) * norm | rope(kva[L, L + R))] (L 512, R 64)
void mla_kv(const float* kva, const float* norm, float* row, int L, int R, int p, float base, float eps, cudaStream_t st);
/// RoPE in place on q's dims [nope, nope + R) of every head (H heads of nope + R)
void mla_q_rope(float* q, int H, int nope, int R, int p, float base, cudaStream_t st);
/// q_lat[h][c] = sum_{i < nope} q[h][i] kvb[(h (nope + V) + i) L + c]   (kvb BF16 [H (nope + V)][L])
void mla_absorb(const float* q, const uint16_t* kvb, float* q_lat, int H, int nope, int R, int V, int L, cudaStream_t st);
/// over the n cache rows: scores, the softmax with the head's sink, o_lat[h] = sum_j p_j row_j[0, L)
void mla_attend(const float* q_lat, const float* q, const float* cache, int n, const float* sinks, float scale,
                float* score, float* o_lat, int H, int nope, int R, int L, cudaStream_t st);
/// o[h][d] = sum_c kvb[(h (nope + V) + nope + d) L + c] o_lat[h][c]
void mla_v(const float* o_lat, const uint16_t* kvb, float* o, int H, int nope, int V, int L, cudaStream_t st);

int launch_errors();

}  // namespace strata::kernels::ernie
