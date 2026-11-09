// src/kernels/cpu/lane_q81_gpu_tables.cpp - the device's own copy of IQ1_S's grid (ggml-common.h's iq1s_grid_gpu: the
// values + 1 as nibbles, only compiled for the GPU backends) for lane_q81's reference: the table the GPU reads, not
// one rebuilt from the CPU's grid.
#include <cstdint>

#define __device__
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

namespace strata::kernels::cpu {
const uint32_t* lane_q81_iq1s_grid_gpu() { return iq1s_grid_gpu; }
}  // namespace strata::kernels::cpu
