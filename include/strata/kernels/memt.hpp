// include/strata/kernels/memt.hpp - MemT (MemTensor's native memory: MemT-1.0-Nano, MemT-1.0-Flash) kernels: the
// memory read beside an attention layer and the turn-end write (worklogs 2026-11-25_memt-nano.md / _memt-flash.md).
//
// A memory layer keeps a state S [H][D][D] (FP32; key rows k, value columns v).  For each row t and head h:
//   read:  qn = q / sqrt(sum q^2 + 1e-6);  r[v] = scale * sum_k qn[k] S'[h][k][v]  (S' = S, or S rounded to BF16:
//          Flash's reference reads a BF16 copy of the state);
//          out[v] = r[v] / sqrt(mean(r^2) + eps) * onorm[v] * act(g[v] + gbias[v]), act(x) = c tanh(silu(x) / c)
//          (c > 0) or silu, added to the attention output ao.
//   write: kn = k / sqrt(sum k^2 + 1e-6);  u = kn * sqrt(sigmoid(b));  wv = sigmoid(w) * v;
//          lam = exp(-exp(dsl[h]) * softplus(f + bias));  then over the rows in order:
//          S <- diag(lam) S;  e = u^T S;  S <- S + u (wv - e)^T
// D is 256 (Nano) or 512 (Flash), H any.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::memt {

constexpr int kD = 256;   // (Nano's head dim)

/// ao[t][h*D + v] += the read of q (rows [T][H*D], raw projections) against S, gated by g (rows [T][H*D], g_down's
/// output; gbias [H*D] or nullptr); onorm [D]
void read_rows(const float* q, const float* g, const float* gbias, const float* S, const float* onorm, float* ao,
               int T, int H, int D, bool bf16_state, float scale, float eps, float softcap, cudaStream_t s);
// one token's read (the decode): the state's rows split over kSplit blocks a head (part: H * kSplit * D floats of
// scratch), then summed in a fixed order - the same math as read_rows with T = 1, spread over more of the GPU
constexpr int kSplit = 16;
void read_tok(const float* q, const float* g, const float* gbias, const float* S, const float* onorm, float* ao,
              float* part, int H, int D, bool bf16_state, float scale, float eps, float softcap, cudaStream_t s);
/// ... the same with the state stored as BF16 (S [H][D][D] BF16 bits: the values the rounded FP32 read sees)
void read_rows16(const float* q, const float* g, const float* gbias, const uint16_t* S, const float* onorm, float* ao,
                 int T, int H, int D, float scale, float eps, float softcap, cudaStream_t s);
void read_tok16(const float* q, const float* g, const float* gbias, const uint16_t* S, const float* onorm, float* ao,
                float* part, int H, int D, float scale, float eps, float softcap, cudaStream_t s);
/// y[i] = x[i] rounded to BF16 (nearest even), n values
void to_bf16(const float* x, uint16_t* y, size_t n, cudaStream_t s);
/// The write's inputs in place: k -> u, w -> wv (reads v), f -> lam (f: f_down's output, [T][H*D]); b [T][H*D]
void write_prep(float* k, const float* v, const float* b, float* w, float* f, const float* bias, const float* dsl,
                int T, int H, int D, cudaStream_t s);
/// S (in / out) over the T rows of u, wv, lam ([T][H*D] each), in order
void write_scan(float* S, const float* u, const float* wv, const float* lam, int T, int H, int D, cudaStream_t s);
/// In place, rows x [T][ld]: each of the G groups of `width` floats RMS-normed (eps) and scaled by w [G*width]
void group_rms(float* x, const float* w, int T, int G, int width, int ld, float eps, cudaStream_t s);

}  // namespace strata::kernels::memt
