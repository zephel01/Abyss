// include/strata/kernels/cpu/iq_avx512.hpp - plan v0.3 P6: AVX-512 multi-token dot products for the i-quant
// expert formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS) and the K-quants Q4_K / Q5_K against Q8_K
// activations (ggml's block_q8_K).  Each token's result is bit-identical to ggml-cpu's AVX2 dot for that pair.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

bool iq512_supported(int ggml_type) noexcept;
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
void iq512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void iq512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);
/// Q5_K's per-block min term as one fmadd (GCC / Clang's contraction of ggml's line, the default) or a multiply
/// then an add (MSVC's) - set from native_expert.cpp's check against this build's ggml-cpu.
void iq512_set_q5k_fused(bool fused) noexcept;

/// iq_avx512_vbmi.cpp: IQ2_XXS (16) / IQ2_XS (17) / IQ3_XXS (18) / IQ2_S (22) rows for 1+ tokens with the grid looked
/// up in registers, IQ1_S (19) with its grid gathered, each token bit-identical to ggml-cpu's dot; needs
/// cpu_avx512_ok() (VNNI, VBMI).  iq512v_ready() builds its tables on the first call.
bool iq512v_supported(int ggml_type) noexcept;
bool iq512v_ready() noexcept;
/// IQ1_S's per-block delta term as one fmadd (GCC / Clang's contraction of ggml's line, the default) or a multiply then
/// an add (MSVC's) - set from native_expert.cpp's check against this build's ggml-cpu.
void iq512v_set_iq1s_fused(bool fused) noexcept;
/// The CPU lane's newer AVX-512 decodes, the same bits as the older ones (lane_q81_test): IQ2_S by gathering
/// iq2s_grid's qwords (~1.5x a block on Zen 5), IQ3_XXS by the grid's byte planes (~1.2x), IQ4_NL in 512-bit lanes
/// (~1.9x over AVX2).  Off unless set: Qwen3.8's CPU lane turns them on (glm_fast_path.cu); STRATA_IQ2S_GATHER,
/// STRATA_IQ3XXS_PLANES, STRATA_IQ4NL_512 = 0 / 1 override each.
void iq512v_set_lane_kernels(bool on) noexcept;
void iq512v_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                 float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu
