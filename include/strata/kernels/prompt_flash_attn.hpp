// include/strata/kernels/prompt_flash_attn.hpp - the prompt's causal attention over an FP16 KV cache on tensor cores
// (CUDA sm_80+: mma.sync m16n8k16, F32 sums), one pass with an online softmax in registers.  For the GQA models whose
// prompt attention ran on CUDA cores or on a two-pass WMMA kernel: Laguna (head_dim 128, groups 6 / 9, window layers
// on a ring), the qwen35 family (head_dim 256: Ornith, Qwen3.8-27B).
//
// The math is the attention's: softmax(scale q . k) v over each row's keys, with q * scale (* log2 e) rounded to FP16,
// P (unnormalized, in [0, 1]) rounded to FP16 for the P V product, the row sums of P and the outputs in F32.  It is a
// sum-order (and operand-rounding) change against a CUDA-core kernel, not the same bits.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::fpa {

/// out[t][h] = softmax(scale * q[t][h] . K[j]) V[j] over the positions j of row t (p = p0 + t): lo..p with lo =
/// max(0, p - window + 1) (window 0: lo = 0), K / V of KV head h / (n_head / n_head_kv) read from cache row j % ring
/// (ring 0: row j) of kc / vc ([rows][n_head_kv][head_dim] FP16).  q, out: [T][n_head][head_dim] F32.
/// head_dim 128 or 256.  Returns false (nothing launched) where this kernel does not run: a HIP build, a device under
/// sm_80, another head_dim, STRATA_PROMPT_FA_OLD=1 - the caller then runs its own (the old) kernel.
bool attn(const float* q, const uint16_t* kc, const uint16_t* vc, int64_t ring, int n_head, int n_head_kv,
          int head_dim, int64_t p0, int T, int window, float scale, float* out, cudaStream_t s);

}  // namespace strata::kernels::fpa
