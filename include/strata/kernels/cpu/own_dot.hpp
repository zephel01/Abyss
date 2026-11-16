// include/strata/kernels/cpu/own_dot.hpp - Strata's own CPU dot products of GGUF weight rows with quantized
// activations, and the activation quantizers they take (src/kernels/cpu/own_dot_body.hpp explains the arithmetic).
// They replace ggml-cpu's `vec_dot` / `from_float` for the native (GGUF-form) experts: the same activation bytes and
// the same result bits as ggml-cpu's x86 AVX2 dot products, for every expert type the engine runs.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// The activation format a weight type's dot takes (8 = Q8_0, 9 = Q8_1, 15 = Q8_K), -1 when there is no own dot.
int own_vec_dot_type(int type) noexcept;
/// x (n floats) -> the activation blocks of act_type (Q8_0 / Q8_1 / Q8_K); false for any other type.
bool own_quant_act(int act_type, const float* x, void* dst, int64_t n) noexcept;
/// One row's dot product (the portable reference, every type own_vec_dot_type knows).
float own_dot(int type, int n, const void* w, const void* act) noexcept;
/// The same, by the AVX2 + FMA build where the CPU has it (same bits).
float own_dot_fast(int type, int n, const void* w, const void* act) noexcept;
/// out[t][r] = w_r . act[t] for rows [r0, r1) (AVX2 + FMA build of the same arithmetic where the CPU has it).
void own_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
              int r0, int r1);

}  // namespace strata::kernels::cpu
