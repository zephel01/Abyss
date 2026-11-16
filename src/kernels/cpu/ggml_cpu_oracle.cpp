// src/kernels/cpu/ggml_cpu_oracle.cpp - see ggml_cpu_oracle.hpp.
#include "ggml_cpu_oracle.hpp"

#include <cmath>
#include <mutex>

#if STRATA_HAVE_GGML_CPU
#include "ggml.h"
#include "ggml-cpu.h"

namespace strata::kernels::cpu {
namespace {
const ggml_type_traits_cpu* tr(int t) {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
    return t >= 0 && t < GGML_TYPE_COUNT ? ggml_get_type_traits_cpu((ggml_type) t) : nullptr;
}
}  // namespace
bool ggml_cpu_have() noexcept { return true; }
int ggml_cpu_vec_dot_type(int t) noexcept {
    const ggml_type_traits_cpu* c = tr(t);
    return c != nullptr && c->vec_dot != nullptr ? (int) c->vec_dot_type : -1;
}
bool ggml_cpu_quant(int a, const float* x, void* dst, int64_t n) noexcept {
    const ggml_type_traits_cpu* c = tr(a);
    if (c == nullptr || c->from_float == nullptr) return false;
    c->from_float(x, dst, n);
    return true;
}
float ggml_cpu_dot(int t, int n, const void* w, const void* act) noexcept {
    const ggml_type_traits_cpu* c = tr(t);
    if (c == nullptr || c->vec_dot == nullptr) return NAN;
    float s = 0.0f;
    c->vec_dot(n, &s, 0, w, 0, act, 0, 1);
    return s;
}
bool ggml_base_to_float(int t, const void* src, float* dst, int64_t n) noexcept {
    if (t < 0 || t >= GGML_TYPE_COUNT) return false;
    const ggml_type_traits* b = ggml_get_type_traits((ggml_type) t);
    if (b == nullptr || b->to_float == nullptr) return false;
    b->to_float(src, dst, n);
    return true;
}
uint16_t ggml_base_fp32_to_fp16(float f) noexcept { return ggml_fp32_to_fp16(f); }
bool ggml_base_q8_0_ref(const float* x, void* dst, int64_t n) noexcept {
    ggml_get_type_traits(GGML_TYPE_Q8_0)->from_float_ref(x, dst, n);
    return true;
}
bool ggml_base_type_info(int t, int64_t* blck, size_t* size, const char** name) noexcept {
    if (t < 0 || t >= GGML_TYPE_COUNT) return false;
    *blck = ggml_blck_size((ggml_type) t);
    *size = ggml_type_size((ggml_type) t);
    *name = ggml_type_name((ggml_type) t);
    return true;
}
}  // namespace strata::kernels::cpu
#else
namespace strata::kernels::cpu {
bool ggml_cpu_have() noexcept { return false; }
int ggml_cpu_vec_dot_type(int) noexcept { return -1; }
bool ggml_cpu_quant(int, const float*, void*, int64_t) noexcept { return false; }
float ggml_cpu_dot(int, int, const void*, const void*) noexcept { return NAN; }
bool ggml_base_to_float(int, const void*, float*, int64_t) noexcept { return false; }
uint16_t ggml_base_fp32_to_fp16(float) noexcept { return 0; }
bool ggml_base_q8_0_ref(const float*, void*, int64_t) noexcept { return false; }
bool ggml_base_type_info(int, int64_t*, size_t*, const char**) noexcept { return false; }
}  // namespace strata::kernels::cpu
#endif
