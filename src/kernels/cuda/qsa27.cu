// src/kernels/cuda/qsa27.cu - see include/strata/kernels/qsa27.hpp.
#include "strata/kernels/qsa27.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels::qsa27 {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa27 %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// the sum of v over the block's 128 threads (4 warps), every thread gets it
__device__ __forceinline__ float sum128(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5;
    __syncthreads();   // (red may still be read from the previous call)
    if ((threadIdx.x & 31) == 0) red[w] = v;
    __syncthreads();
    return (red[0] + red[1]) + (red[2] + red[3]);
}

// one block (128 threads, a dim each) per block of cells this call completes: b = b0 + blockIdx.x
__global__ void __launch_bounds__(kDim) pool_kernel(const float* __restrict__ raw, int64_t ld, int64_t pos0, int64_t b0,
                                                    const float* __restrict__ w_pre, const float* __restrict__ w_blk,
                                                    float eps, const float* __restrict__ tail, float* __restrict__ pooled,
                                                    const float* __restrict__ cos_tab, const float* __restrict__ sin_tab) {
    __shared__ float red[4];
    __shared__ float row[kDim];
    const int d = threadIdx.x;
    const int64_t b = b0 + blockIdx.x;
    float m = 0.0f;
    for (int i = 0; i < kBlock; ++i) {
        const int64_t c = b * kBlock + i;
        const float v = c >= pos0 ? raw[(c - pos0) * ld + d] : tail[(size_t) i * kDim + d];
        const float ss = sum128(v * v, red);
        m += v * rsqrtf(ss / (float) kDim + eps) * w_pre[d];
    }
    m *= 1.0f / (float) kBlock;
    const float ss = sum128(m * m, red);
    row[d] = m * rsqrtf(ss / (float) kDim + eps) * w_blk[d];
    __syncthreads();
    float o = row[d];
    if (d < kRot) {   // NEOX: (i, i + 32) at the block's first cell
        const int i = d < kRot / 2 ? d : d - kRot / 2;
        const size_t t = (size_t) (b * kBlock) * (kRot / 2) + i;
        const float c = cos_tab[t], s = sin_tab[t];
        o = d < kRot / 2 ? row[d] * c - row[d + kRot / 2] * s : row[d - kRot / 2] * s + row[d] * c;
    }
    pooled[b * kDim + d] = o;
}

// the cells [c0, c1) of the block the call leaves incomplete into their tail slots
__global__ void __launch_bounds__(kDim) tail_kernel(const float* __restrict__ raw, int64_t ld, int64_t pos0, int64_t c0,
                                                    int64_t c1, float* __restrict__ tail) {
    const int d = threadIdx.x;
    for (int64_t c = c0; c < c1; ++c) tail[(size_t) (c % kBlock) * kDim + d] = raw[(c - pos0) * ld + d];
}

__global__ void own_block_kernel(int32_t* __restrict__ ids, int64_t ld, const int32_t* __restrict__ steps_in,
                                 int32_t* __restrict__ steps_out, int nq) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    const int32_t* si = steps_in + (size_t) q * kStepCount;
    int32_t* so = steps_out + (size_t) q * kStepCount;
    const int32_t n_kv = si[kStepNKv], w = si[kStepWidth];
    int32_t wo = w;
    if (n_kv % kBlock == 0 && n_kv > w && w > 0) {
        int32_t* r = ids + (size_t) q * ld;
        if (r[w - 1] != n_kv - 1) {   // (ascending: the own block is the last one when it was selected)
            for (int i = 0; i < kBlock; ++i) r[w + i] = n_kv - kBlock + i;
            wo = w + kBlock;
        }
    }
    so[kStepPos] = si[kStepPos];
    so[kStepNKv] = n_kv;
    so[kStepNBid] = si[kStepNBid];
    so[kStepWidth] = wo;
}

__global__ void pick_kernel(float* __restrict__ scores, int64_t max_blocks, const int32_t* __restrict__ steps, int mode) {
    const int q = blockIdx.y;
    const int64_t b = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t n_bid = steps[(size_t) q * kStepCount + kStepNBid];
    if (b >= n_bid || b >= max_blocks) return;
    scores[(size_t) q * max_blocks + b] = mode == 1 ? (float) b : -(float) b;
}

}  // namespace

void pick_scores(float* scores, int64_t max_blocks, const int32_t* steps, int nq, int mode, cudaStream_t s) {
    if (nq <= 0 || mode == 0) return;
    pick_kernel<<<dim3((unsigned) ((max_blocks + 255) / 256), (unsigned) nq), 256, 0, s>>>(scores, max_blocks, steps, mode);
    check("pick");
}

void step_fill(int32_t* h, int64_t pos, int top_k) {
    const int64_t n_kv = pos + 1;
    h[kStepPos] = (int32_t) pos;
    h[kStepNKv] = (int32_t) n_kv;
    h[kStepNBid] = (int32_t) (n_kv / kBlock);
    h[kStepWidth] = (int32_t) width(n_kv, top_k);
}

void key_append_rows(const float* raw, int64_t ld, int64_t pos0, int n, const float* w_pre, const float* w_blk,
                     float eps, float* tail, float* pooled, const float* cos_tab, const float* sin_tab,
                     cudaStream_t s) {
    if (n <= 0) return;
    const int64_t end = pos0 + n;
    const int64_t b0 = pos0 / kBlock, nb = end / kBlock - b0;   // the blocks whose last cell is in [pos0, end)
    if (nb > 0) {
        pool_kernel<<<(unsigned) nb, kDim, 0, s>>>(raw, ld, pos0, b0, w_pre, w_blk, eps, tail, pooled, cos_tab,
                                                   sin_tab);
        check("pool");
    }
    if (end % kBlock != 0) {
        const int64_t c0 = end - end % kBlock;
        tail_kernel<<<1, kDim, 0, s>>>(raw, ld, pos0, c0 > pos0 ? c0 : pos0, end, tail);
        check("tail");
    }
}

void own_block(int32_t* ids, int64_t ld, const int32_t* steps_in, int32_t* steps_out, int nq, cudaStream_t s) {
    if (nq <= 0) return;
    own_block_kernel<<<(unsigned) ((nq + 63) / 64), 64, 0, s>>>(ids, ld, steps_in, steps_out, nq);
    check("own_block");
}

}  // namespace strata::kernels::qsa27
