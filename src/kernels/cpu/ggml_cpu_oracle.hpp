// src/kernels/cpu/ggml_cpu_oracle.hpp - ggml-cpu's dot products and activation quantizers, when the build has ggml
// (STRATA_GGML=ON): the old CPU expert path (STRATA_CPU_GGML=1) and the oracle own_dot_test compares with.
// Without ggml every call answers false / NaN.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

bool ggml_cpu_have() noexcept;
/// ggml-cpu's vec_dot_type for a type, -1 when it has no dot
int ggml_cpu_vec_dot_type(int type) noexcept;
bool ggml_cpu_quant(int act_type, const float* x, void* dst, int64_t n) noexcept;
float ggml_cpu_dot(int type, int n, const void* w, const void* act) noexcept;
/// ggml-base's row dequantizer and fp16 conversion (for qtypes_test)
bool ggml_base_to_float(int type, const void* src, float* dst, int64_t n) noexcept;
uint16_t ggml_base_fp32_to_fp16(float f) noexcept;
bool ggml_base_q8_0_ref(const float* x, void* dst, int64_t n) noexcept;
/// ggml-base's block geometry and name of a type id (false: no ggml, or the id is past ggml's GGML_TYPE_COUNT)
bool ggml_base_type_info(int type, int64_t* blck, size_t* size, const char** name) noexcept;

}  // namespace strata::kernels::cpu
