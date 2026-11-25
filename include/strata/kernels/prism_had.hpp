// include/strata/kernels/prism_had.hpp - PrismML's blockwise Hadamard transform (Ternary Bonsai GGUFs, prism.hadamard.*).
//
// The rotated weights W' were stored as W (S P)^T H (H the normalized Sylvester-Walsh-Hadamard matrix on each block of B
// input values, H[r][c] = (-1)^popcount(r & c) / sqrt(B), symmetric and its own inverse; S a +-1 sign per input value;
// P an optional reorder of the input), so W x = W' (H S P x): the input goes through rotate() before the product.  The
// embedding table stores rotated rows z = H S h: the lookup's row goes through unrotate() (h = S H z).
// The butterflies run in FP32 in shared memory (log2 B stages); 1 / sqrt(B) is exact for even log2 B (B = 1024: 1/32).
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::prism {

struct Rot {
    const float* signs = nullptr;   // n values on the device, or nullptr (identity signs)
    int n = 0;                      // the row's width (a multiple of B)
    int B = 1024;                   // the block (a power of two, 32..1024)
    int hd = 0, nk = 0, rep = 0;    // rep > 1: P, the GDN value heads tiled [hd, nk, rep] -> grouped [hd, rep, nk]
};

/// rows of y = H_B (S P x); y / y16 (FP16 bits) may each be nullptr; x == y is allowed when there is no P.
/// x, y, y16: row r at r * ldx / ldy / ld16 (elements).  False for a shape the kernel does not take.
/// yq (optional): each row's values also as q8_1 blocks (36 bytes per 32, rows n / 32 blocks apart) - the bits of
/// glmf::quantize_q8_1 over y, in the same launch.
bool rotate(const Rot& r, const float* x, int ldx, float* y, int ldy, uint16_t* y16, int ld16, int rows, cudaStream_t s,
            void* yq = nullptr);
/// rows of y = S H_B z (the embedding row restored; no P); in place allowed.
bool unrotate(const Rot& r, const float* z, int ldz, float* y, int ldy, uint16_t* y16, int ld16, int rows, cudaStream_t s);

}  // namespace strata::kernels::prism
