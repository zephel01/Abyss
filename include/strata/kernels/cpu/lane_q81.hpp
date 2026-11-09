// include/strata/kernels/cpu/lane_q81.hpp - the CPU lane's expert rows in the DEVICE's arithmetic.
//
// The GPU's expert kernels (glm_fast.cu: moe_gate_up_kernel / moe_down_kernel - rows_dot, dw<T>, dwk<T>, the vecdotq
// expressions) quantize an activation to q8_1 (an FP16 scale and an FP16 sum a 32 values, glm_fast's
// q8_1_store_warp) and combine each 32 (or 16) values' integer sum with the weight's scales in their own integer steps
// (IQ2_XXS's sumi * ls / 8 truncated, ...).  ggml-cpu's dots take Q8_K (one scale a 256 values) instead, so an expert
// the CPU lane computed differed from the same expert on the GPU by ~2% - and which experts run on the CPU changes
// every token.  These rows take the device's q8_1 activation and its integer and float steps, so a lane expert gives
// the device's numbers to float rounding (the order of the float additions is the only difference).
//
// The activation ("Q81 act", lane_q81_act_bytes(n)): the q8_1 codes as n int8, then the n / 32 FP16-rounded scales
// and sums as floats, then the sums of every 8 codes as int16 (the K-quants' minimum terms, IQ1_M's delta).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// The ggml types the device's expert kernels take (glm_fast moe_supported), all covered here.
bool lane_q81_type(int ggml_type) noexcept;
/// Bytes of one Q81 activation of n values (n % 32 == 0).
size_t lane_q81_act_bytes(int64_t n) noexcept;
/// x (n floats, n % 32 == 0) -> a Q81 activation: glm_fast's q8_1_store_warp bit for bit (amax / 127 capped at 65504,
/// roundf(x / d), the codes clamped to +-127, the sum in the warp's butterfly order, both rounded to FP16).
void lane_q81_quant(const float* x, int64_t n, void* dst);
/// The device's own q8_1 blocks (36 bytes a 32 values) -> a Q81 activation (the same values; checks, tests).
void lane_q81_from_blocks(const void* blocks, int64_t n, void* dst);

/// out[t][r] = row r . act[t] for rows [r0, r1), `nt` tokens: the fastest form this CPU has (AVX-512 VBMI, AVX2,
/// else the portable one), every form the device's arithmetic.
void lane_q81_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                   float* const* out, int r0, int r1);
/// Whether lane_q81_rows has a vector form for this type on this CPU (false: the portable scalar one).
bool lane_q81_fast(int ggml_type) noexcept;
/// An expert's gate/up rows [r0, r1) for nt tokens and the device's epilogue (moe_gate_up_kernel): g = min(scale * gate,
/// limit), u = clamp(scale * up, +-limit), ff[t][r] = g / (1 + exp(-g)) * u (limit infinite: no clamp; scale: NVFP4's
/// second-level scale, else 1).  The rows of gate at gate_base + r * gate_row, of up likewise (a mixed layer: up's own
/// type).
void lane_q81_gu_rows(int gate_type, int up_type, const uint8_t* gate_base, size_t gate_row, const uint8_t* up_base,
                      size_t up_row, int n, const void* const* act, int nt, float* const* ff, int r0, int r1,
                      float limit, float scale);

/// The reference: the device's dot functions transcribed (dp4a and friends emulated), one call at a time, the calls'
/// floats summed in double.  For the tests.
double lane_q81_ref_dot(int ggml_type, const uint8_t* row, int n, const void* act);
/// The portable rows (the reference's calls, summed in float); what lane_q81_rows uses where no vector form exists.
void lane_q81_rows_scalar(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                          float* const* out, int r0, int r1);

// (the vector forms; lane_q81_rows picks them)
bool lane_q81_avx2_type(int ggml_type) noexcept;
void lane_q81_avx2_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                        float* const* out, int r0, int r1);
/// (x86 GCC / Clang builds: lane_q81_avx2_rows' types with AVX-VNNI - only on a CPU that has it)
void lane_q81_vnni_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                        float* const* out, int r0, int r1);
bool lane_q81_vbmi_type(int ggml_type) noexcept;
void lane_q81_vbmi_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                        float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
