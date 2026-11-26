// include/strata/kernels/raon_fast.hpp - KRAFTON Raon-1's (GGUF arch raon1) one-token decode kernels.
//
// The pieces the shared kernels do not cover (src/core/raon_fast_path.cu runs them in order; the RDA's convs / gates
// and the MLA's attention are ling_fast's kda_prep / mla_prep / mla_absorb / mla_attend, the recurrence glm_fast's
// kda_rec):
//   * Raon-mHC, a site (attention or FFN) of a layer on the 4 streams R [4][n_embd]:
//       z = (W . flat(R)) * rsqrt(mean(flat(R)^2) + hc_eps), W the selector [17][4 n_embd] (BF16)
//       h_pre = sigmoid(s0 z[0:4] + b[0:4]), h_post = post * sigmoid(s1 z[4:8] + b[4:8]),
//       t = s2 z[8:17] + b[8:17] as 3 x 3, E = Q t Q^T (Q: the 4 x 3 Helmert basis), B = (I + 11^T / 4) / 2,
//       rho = max(0, max_ij(-E_ij / B_ij)), Phi = B + E / (1 + rho), H = (1 - g) I + g Phi, g = sigmoid(s3)
//     mhc_pre: the coefficients (coef: h_pre 4 | h_post 4 | H 16) and the branch input rms(sum_n h_pre[n] R[n]) * norm
//     (x, its q8_1 by the caller); mhc_post: R[i] = sum_j H[i][j] R[j] + h_post[i] * out
//   * the RDA's write gate folded into the values: v[h] *= sigmoid(b_h + w_h) / sigmoid(b_h) (b: the erase logits the
//     recurrence takes as its beta, w: the write delta), so the KDA update stays S += sigmoid(b) (v' - S k) k^T
//   * the gated MLA's output with a gate per channel: o[h][v] = (v_b[h][v] . ctx[h]) * sigmoid(gate[h * 128 + v])
//   * y += alpha * c (the reasoning-effort steering vector on a layer's MoE output)
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::raon {

constexpr int kStreams = 4;
constexpr int kSel = 17;     // the selector's rows: pre 4, post 4, the residual chart's 9
constexpr int kCoef = 24;    // h_pre 4 | h_post 4 | H 16 (row-major)
constexpr int kZRow = 32;    // a row's z scratch

/// One mHC site's coefficients and branch input.  R: [4][n_embd]; fn: BF16 [17][4 n_embd]; base [17]; scale [4]
/// (s0..s3 = alpha_pre, alpha_post, alpha_z, eta); norm_w [n_embd]; z: scratch (kZRow floats); coef: out (kCoef floats);
/// x: out (n_embd floats, the normed input).  rows > 1 (the prompt path): rows of R (4 n_embd), z (kZRow), coef
/// (kCoef) and x (n_embd) - each row the token path's arithmetic
void mhc_pre(const float* R, const uint16_t* fn, const float* base, const float* scale, const float* norm_w,
             float hc_eps, float norm_eps, float post_factor, int n_embd, float* z, float* coef, float* x,
             cudaStream_t s, int rows = 1);
/// R[i][e] = sum_j H[i][j] R[j][e] + h_post[i] * out[e] (coef from mhc_pre)
void mhc_post(float* R, const float* out, const float* coef, int n_embd, cudaStream_t s, int rows = 1);
/// v[h * hd + c] *= exp(logsigmoid(b[h] + w[h]) - logsigmoid(b[h]))
void rda_ratio(float* v, const float* b, const float* w, int n_head, int head_dim, cudaStream_t s, int rows = 1);
/// o[h][v] = (v_b[h][v] . ctx[h]) * sigmoid(gate[h * kV + v]); v_b F32 [n_head][128][512], ctx [n_head][512]
void mla_out_gated(const float* ctx, const float* v_b, const float* gate, int n_head, float* o, cudaStream_t s);
/// y[i] += alpha * c[i]
void axpy(float* y, const float* c, float alpha, int n, cudaStream_t s, int rows = 1);
/// the prompt path: x[t] = rms(in[t]) * w over rows of n floats (any n), x16 its FP16
void rms_rows_any(const float* in, const float* w, float eps, int rows, int n, float* x, void* x16, cudaStream_t s);
/// the prompt path: the MLA cache rows (512 latent rms-normed | 64 pe as they are, FP16) of rows positions from kv rows
/// of 576 floats, into cache (the first row's place)
void mla_cache_rows(const float* kv, const float* kv_norm, float eps, int rows, uint16_t* cache, cudaStream_t s);
/// o[i] *= sigmoid(gate[i])
void gate_rows(float* o, const float* gate, int64_t n, cudaStream_t s);
/// The prompt's causal MLA attention on tensor cores (ling_fast's mla_prompt_kernel generalised: n_head heads in
/// passes of 32, no gate): ctx[t][h] = softmax_j(scale (q_abs[t][h] . lat_j + q_pe[t][h] . pe_j)) . lat_j over
/// j <= p0 + t.  qabs [T][n_head][512] in, ctx [T][n_head][512] out (separate), q rows of ldq floats with each
/// head's pe at h * 192 + 128.  False where it does not run (not sm_80+, the WMMA layout off)
bool prompt_attn_ok();
void mla_prompt_attn(const float* qabs, float* ctx, const float* q, int ldq, const uint16_t* cache, int64_t p0, int T,
                     int n_head, float scale, cudaStream_t s);
/// launches that failed (all raon kernels)
int launch_errors();

}  // namespace strata::kernels::raon
