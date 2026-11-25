// src/kernels/cuda/prism_had.cu - see include/strata/kernels/prism_had.hpp.
#include "strata/kernels/prism_had.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels::prism {
namespace {

constexpr int kThreads = 256;

struct Q81 {   // glm_fast.cu's block_q8_1
    __half2 ds;
    int8_t qs[32];
};
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// glm_fast.cu's q8_1_store_warp, the same arithmetic (so the products read the bits quantize_q8_1 would write)
__device__ __forceinline__ void q81_store(float v, Q81* dst, int lane) {
    const float amax = wmax(fabsf(v));
    const float sum = wsum(v);
    const float d = fminf(amax / 127.0f, 65504.0f);
    const int8_t q = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(v / d)));
    dst->qs[lane] = q;
    if (lane == 0) dst->ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}

// one block of B values of one row: gather (P, S or none), the butterflies, the scale (and S after them for the inverse)
template <bool INV>
__global__ void __launch_bounds__(kThreads) had_kernel(const float* __restrict__ x, int ldx, float* y, int ldy,
                                                       uint16_t* y16, int ld16, const float* __restrict__ signs, int B,
                                                       int hd, int nk, int rep, float scale, Q81* yq, int n) {
    __shared__ float sh[1024];
    const int blk = blockIdx.x, row = blockIdx.y, t = threadIdx.x;
    const float* xr = x + (size_t) row * ldx;
    const int base = blk * B;
    for (int i = t; i < B; i += kThreads) {
        const int gi = base + i;
        int src = gi;
        if (!INV && rep > 1) {   // grouped position gi = h + hd (r + rep k) reads the tiled one h + hd (k + nk r)
            const int h = gi % hd, q = gi / hd, rr = q % rep, k = q / rep;
            src = h + hd * (k + nk * rr);
        }
        float v = xr[src];
        if (!INV && signs != nullptr) v *= signs[gi];
        sh[i] = v;
    }
    __syncthreads();
    for (int h = 1; h < B; h <<= 1) {
        for (int p = t; p < B / 2; p += kThreads) {
            const int i = (p / h) * 2 * h + (p % h), j = i + h;
            const float a = sh[i], b = sh[j];
            sh[i] = a + b;
            sh[j] = a - b;
        }
        __syncthreads();
    }
    float* yr = y != nullptr ? y + (size_t) row * ldy : nullptr;
    uint16_t* hr = y16 != nullptr ? y16 + (size_t) row * ld16 : nullptr;
    for (int i = t; i < B; i += kThreads) {
        const int gi = base + i;
        float v = sh[i] * scale;
        if (INV && signs != nullptr) v *= signs[gi];
        if (yr != nullptr) yr[gi] = v;
        if (hr != nullptr) hr[gi] = __half_as_ushort(__float2half_rn(v));
        if (yq != nullptr) q81_store(v, yq + (size_t) row * (n / 32) + gi / 32, t & 31);   // (a warp: 32 in a row)
    }
}

// B = 1024 in registers (the Ternary Bonsai files' block): 128 threads, thread t holding values 8t .. 8t + 7 for the
// stages h = 1, 2, 4 (its own values), h = 8 .. 128 (a warp's lanes, shuffles), then - through shared memory - the
// columns t and t + 128 of the 4 x 256 view for h = 256, 512.  The same pairs, the same order of stages and the same
// FP32 adds as had_kernel: the same bits, ~1/3 of its time (no barrier a stage)
template <bool INV>
__global__ void __launch_bounds__(128) had1024_kernel(const float* __restrict__ x, int ldx, float* y, int ldy,
                                                      uint16_t* y16, int ld16, const float* __restrict__ signs, int hd,
                                                      int nk, int rep, float scale, Q81* yq, int n) {
    __shared__ float sh[1024];
    const int blk = blockIdx.x, row = blockIdx.y, t = threadIdx.x, lane = t & 31;
    const float* xr = x + (size_t) row * ldx;
    const int base = blk * 1024;
    float v[8];
    if (!INV && rep > 1) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int gi = base + 8 * t + i;
            const int h = gi % hd, q = gi / hd, rr = q % rep, k = q / rep;
            v[i] = xr[h + hd * (k + nk * rr)];
        }
    } else {
        const float4* x4 = (const float4*) (xr + base + 8 * t);
        const float4 a = x4[0], b = x4[1];
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    }
    if (!INV && signs != nullptr) {
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] *= signs[base + 8 * t + i];
    }
#pragma unroll
    for (int h = 1; h < 8; h <<= 1)
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if ((i & h) == 0) {
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {   // h = 8 m: the partner is lane ^ m; the lower one keeps a + b
        const bool lower = (lane & m) == 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float o = __shfl_xor_sync(0xffffffffu, v[i], m);
            v[i] = lower ? v[i] + o : o - v[i];
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) sh[8 * t + i] = v[i];
    __syncthreads();
#pragma unroll
    for (int c2 = 0; c2 < 2; ++c2) {
        const int c = t + 128 * c2;   // column c: values c, c + 256, c + 512, c + 768
        float a0 = sh[c], a1 = sh[c + 256], a2 = sh[c + 512], a3 = sh[c + 768];
        float b0 = a0 + a1, b1 = a0 - a1, b2 = a2 + a3, b3 = a2 - a3;   // h = 256
        float r[4] = {b0 + b2, b1 + b3, b0 - b2, b1 - b3};               // h = 512
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int gi = base + c + 256 * k;
            float o = r[k] * scale;
            if (INV && signs != nullptr) o *= signs[gi];
            if (y != nullptr) y[(size_t) row * ldy + gi] = o;
            if (y16 != nullptr) y16[(size_t) row * ld16 + gi] = __half_as_ushort(__float2half_rn(o));
            if (yq != nullptr) q81_store(o, yq + (size_t) row * (n / 32) + gi / 32, lane);
        }
    }
}

bool shape_ok(const Rot& r, int rows) {
    if (r.B < 32 || r.B > 1024 || (r.B & (r.B - 1)) != 0 || r.n <= 0 || r.n % r.B != 0 || rows <= 0 || rows > 65535)
        return false;
    if (r.rep > 1 && (r.hd <= 0 || r.nk <= 0 || r.hd * r.nk * r.rep != r.n)) return false;
    return true;
}

// STRATA_PRISM_HAD=0: the shared-memory kernel for every block (A/B; the same bits)
bool fast_on() {
    static const bool on = [] { const char* v = getenv("STRATA_PRISM_HAD"); return v == nullptr || std::atoi(v) != 0; }();
    return on;
}

}  // namespace

bool rotate(const Rot& r, const float* x, int ldx, float* y, int ldy, uint16_t* y16, int ld16, int rows, cudaStream_t s,
            void* yq) {
    if (!shape_ok(r, rows) || (r.rep > 1 && x == y) || (yq != nullptr && r.B < kThreads)) return false;
    const dim3 grid((unsigned) (r.n / r.B), (unsigned) rows);
    if (r.B == 1024 && fast_on() && (r.rep > 1 || (((uintptr_t) x | (size_t) ldx * 4) & 15) == 0))
        had1024_kernel<false><<<grid, 128, 0, s>>>(x, ldx, y, ldy, y16, ld16, r.signs, r.hd, r.nk, r.rep, 1.0f / 32.0f,
                                                   (Q81*) yq, r.n);
    else
    had_kernel<false><<<grid, kThreads, 0, s>>>(x, ldx, y, ldy, y16, ld16, r.signs, r.B, r.hd, r.nk, r.rep,
                                                (float) (1.0 / std::sqrt((double) r.B)), (Q81*) yq, r.n);
    return cudaGetLastError() == cudaSuccess;
}

bool unrotate(const Rot& r, const float* z, int ldz, float* y, int ldy, uint16_t* y16, int ld16, int rows, cudaStream_t s) {
    if (!shape_ok(r, rows)) return false;
    const dim3 grid((unsigned) (r.n / r.B), (unsigned) rows);
    if (r.B == 1024 && fast_on() && (((uintptr_t) z | (size_t) ldz * 4) & 15) == 0)
        had1024_kernel<true><<<grid, 128, 0, s>>>(z, ldz, y, ldy, y16, ld16, r.signs, 0, 0, 0, 1.0f / 32.0f, nullptr, r.n);
    else
    had_kernel<true><<<grid, kThreads, 0, s>>>(z, ldz, y, ldy, y16, ld16, r.signs, r.B, 0, 0, 0,
                                               (float) (1.0 / std::sqrt((double) r.B)), nullptr, r.n);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace strata::kernels::prism
