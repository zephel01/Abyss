// src/kernels/cuda/rope.cu - P2.S2: NEOX partial RoPE.
//
// Semantics from `ref/qsa.py` (`rope_freqs` + `rope_neox`), which `bench/micro/rope_xcheck` validates against
// the oracle:
//
//     inv[i] = theta ** (-2i / n_rot)          i in [0, n_rot/2)
//     ang    = pos * inv
//     cos, sin = cos(ang), sin(ang)
//     out[:half]     = a*cos - b*sin            a = x[:half], b = x[half:n_rot]
//     out[half:n_rot] = a*sin + b*cos
//
// THE PAIRING IS NEOX - (i, i + n_rot/2), NOT the adjacent pair (2i, 2i+1).  `ggml.h` writes NEOX
// n_dims=8 as [ccccssss]; the adjacent-pair ("NORMAL") ordering is the other common convention and choosing
// wrongly rotates the wrong dimensions together, which produces correctly-shaped output with scrambled
// content.  It is also PARTIAL: only the first n_rot of head_dim are touched, here 64 of 256.
//
// WHY THE TABLE IS BUILT ON THE HOST.  The reference computes the frequencies in FLOAT64 (`theta ** (...)`
// on a float64 arange, cos/sin in float64).  Reproducing that on device means double-precision pow and cos,
// which are slow on a consumer GPU and - more to the point - not guaranteed to agree with the host's libm.
// Building the table once on the host and passing it in makes the TABLE exactly the reference's values and
// leaves the kernel a pure rotation, so each half can be checked tightly instead of both halves sharing one
// loose tolerance.  The engine wants a cached table anyway: it is per (n_rot, theta, position), not per token.
#include "strata/kernels/rope.hpp"
#include "strata/kernels/mrope.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    const int half = n_rot / 2;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            // float64 throughout, in the reference's order: inv, then ang, then cos/sin
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            cos_tab[(size_t) p * half + i] = (float) std::cos(ang);
            sin_tab[(size_t) p * half + i] = (float) std::sin(ang);
        }
    }
}

namespace {

// ONE THREAD PER ROW, not per (row, pair).  The first version had one thread per pair and did the tail copy
// with 	hreadIdx.x strided loops - so up to 32 threads wrote into the SAME row, with no barrier between the
// copy and the rotation, and a thread's copy of orow[i] could land AFTER another thread had rotated it.  A
// race that silently writes the unrotated value is worse than a wrong answer: it is intermittent.
//
// A row is head_dim floats (256 here) and there are batch x heads of them, so one thread per row is a
// handful of threads for a token and removes the ordering question entirely rather than synchronising it.
__global__ void rope_neox_kernel(const float* __restrict__ x, float* __restrict__ out, long long rows,
                                 int head_dim, int n_rot, const float* __restrict__ cos_tab,
                                 const float* __restrict__ sin_tab, const int* __restrict__ pos,
                                 const int32_t* __restrict__ mtab) {
    const long long r = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int half = n_rot / 2;
    const float* xr = x + r * head_dim;
    float* orow = out + r * head_dim;

    for (int d = n_rot; d < head_dim; ++d) orow[d] = xr[d];      // the PARTIAL rotation's untouched tail
    for (int i = 0; i < half; ++i) {
        const size_t toff = (size_t) mrope_pos(mtab, pos[r], i) * half;   // the image path's per-pair position
        // THE PAIRING AND THE SIGNS COME FROM `rope_neox_pair`, shared with the indexer's pooling kernel.
        rope_neox_pair(xr[i], xr[half + i], cos_tab[toff + i], sin_tab[toff + i], orow[i], orow[half + i]);
    }
}

// rope_neox_kernel's arithmetic with a thread a PAIR (and a thread a tail element) instead of a thread a row: a
// token's few rows (the QSA indexer's query heads) as one thread each walked 64+ pairs serially (~4.5 us).  Each
// output element is written by exactly one thread, which read its own inputs only: safe in place, the same bits
__global__ void rope_neox_pairs_kernel(const float* __restrict__ x, float* __restrict__ out, long long rows,
                                       int head_dim, int n_rot, const float* __restrict__ cos_tab,
                                       const float* __restrict__ sin_tab, const int* __restrict__ pos,
                                       const int32_t* __restrict__ mtab) {
    const int half = n_rot / 2, per = half + (head_dim - n_rot);
    const long long g = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const long long r = g / per;
    if (r >= rows) return;
    const int j = (int) (g - r * per);
    const float* xr = x + r * head_dim;
    float* orow = out + r * head_dim;
    if (j >= half) {
        const int d = n_rot + (j - half);
        orow[d] = xr[d];
        return;
    }
    const int i = j;
    const size_t toff = (size_t) mrope_pos(mtab, pos[r], i) * half;
    rope_neox_pair(xr[i], xr[half + i], cos_tab[toff + i], sin_tab[toff + i], orow[i], orow[half + i]);
}

}  // namespace

void rope_neox_apply_pairs(const float* x, float* out, int64_t rows, int head_dim, int n_rot, const float* cos_tab,
                           const float* sin_tab, const int* pos, void* stream) {
    if (rows <= 0 || n_rot <= 0) return;
    if (n_rot % 2 != 0 || n_rot > head_dim) {
        std::fprintf(stderr, "rope_neox_apply_pairs: n_rot %d must be even and <= head_dim %d\n", n_rot, head_dim);
        std::exit(1);
    }
    const long long total = rows * (long long) (n_rot / 2 + (head_dim - n_rot));
    const int threads = 256;
    const unsigned grid = (unsigned) ((total + threads - 1) / threads);
    rope_neox_pairs_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(x, out, rows, head_dim, n_rot, cos_tab,
                                                                       sin_tab, pos, mrope_table());
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "rope_neox_apply_pairs launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

void rope_neox_apply(const float* x, float* out, int64_t rows, int head_dim, int n_rot, const float* cos_tab,
                     const float* sin_tab, const int* pos, void* stream) {
    if (rows <= 0 || n_rot <= 0) return;
    if (n_rot % 2 != 0 || n_rot > head_dim) {
        std::fprintf(stderr, "rope_neox_apply: n_rot %d must be even and <= head_dim %d\n", n_rot, head_dim);
        std::exit(1);
    }
    const int threads = 128;
    const unsigned grid = (unsigned) ((rows + threads - 1) / threads);
    rope_neox_kernel<<<grid, threads, 0, (cudaStream_t) stream>>>(x, out, rows, head_dim, n_rot, cos_tab,
                                                                 sin_tab, pos, mrope_table());
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "rope_neox_apply launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) cudaDeviceSynchronize();
}

}  // namespace strata::kernels
