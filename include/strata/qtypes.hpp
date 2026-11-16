// include/strata/qtypes.hpp - the GGUF tensor types without ggml: ids, block geometry, names, fp16/bf16 conversion,
// row dequantizers and the few row quantizers the engine calls on the host.
//
// The ids are the GGUF file format's tensor type ids (the same numbers ggml's enum carries; own_dot_test checks ids, block sizes and names against ggml), so a
// type read from a GGUF header or a pack's native_experts.txt is used as is.  The block layouts are the published
// ones in third_party/ggml/ggml-common.h (MIT, kept with its license as the format's description); everything
// here is Strata's own code written from those layouts.
//
// to_float gives the values a GGUF block means, in single-precision steps a plain C compiler takes without FMA
// contraction (each product and sum rounded on its own): the same bits ggml-base's dequantizers give, which
// own_dot_test checks against a ggml build when one is configured (STRATA_GGML=ON).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::qt {

/// GGUF tensor type ids.
enum Type : int {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23,
    I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, IQ1_M = 29, BF16 = 30,
    TQ1_0 = 34, TQ2_0 = 35, MXFP4 = 39, NVFP4 = 40, Q1_0 = 41, Q2_0 = 42,
    COUNT = 43
};

using ToFloat = void (*)(const void* src, float* dst, int64_t n);
using FromFloat = void (*)(const float* src, void* dst, int64_t n);

/// One type's facts.  name: ggml's spelling ("q4_K", "iq2_xxs", "bf16") so log lines read as before.
struct Traits {
    const char* name = nullptr;
    int64_t blck_size = 0;          ///< values per block (0: no such type)
    size_t type_size = 0;           ///< bytes per block
    bool is_quantized = false;
    ToFloat to_float = nullptr;     ///< row dequantizer (n a multiple of blck_size)
    FromFloat from_float_ref = nullptr;   ///< reference row quantizer (Q8_0 only; nullptr otherwise)
};

/// nullptr for an id outside [0, COUNT); a Traits with blck_size 0 for an unused id.
const Traits* traits(int type) noexcept;
const char* type_name(int type) noexcept;   ///< "?" for an unknown id
int64_t blck_size(int type) noexcept;       ///< 0 for an unknown id
size_t type_size(int type) noexcept;        ///< 0 for an unknown id
size_t row_size(int type, int64_t n) noexcept;   ///< type_size * n / blck_size

/// IEEE binary16 <-> binary32 (round to nearest even; NaN stays NaN, overflow gives infinity)
float fp16_to_fp32(uint16_t h) noexcept;
uint16_t fp32_to_fp16(float f) noexcept;
void fp32_to_fp16_row(const float* src, uint16_t* dst, int64_t n) noexcept;
/// bfloat16: the upper 16 bits; fp32_to_bf16 rounds to nearest even (a NaN keeps its sign and gets a quiet bit)
float bf16_to_fp32(uint16_t h) noexcept;
uint16_t fp32_to_bf16(float f) noexcept;

/// Quantizes rows of n_per_row floats (src rows from `start`, a multiple of n_per_row) into dst; returns the bytes
/// written.  Q8_0 is the reference quantizer (from_float_ref).  Q4_K and IQ4_NL are Strata's own simple quantizers
/// (per-block scale fits, no importance matrix): only the opt-in 4-bit MTP draft copies use them (a draft decides
/// the acceptance, never the output).  0 for any other type.
size_t quantize_chunk(int type, const float* src, void* dst, int64_t start, int64_t nrows, int64_t n_per_row) noexcept;

}  // namespace strata::qt
