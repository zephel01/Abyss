// src/kernels/cuda/raon_fast.cu - see include/strata/kernels/raon_fast.hpp.
#include "strata/kernels/raon_fast.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdio>

namespace strata::kernels::raon {
namespace {

std::atomic<int> g_errors{0};
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        g_errors.fetch_add(1);
        std::fprintf(stderr, "raon_fast %s: %s\n", what, cudaGetErrorString(e));
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// a block's sum (blockDim.x a multiple of 32, <= 1024); every thread gets it
__device__ __forceinline__ float block_sum(float v, float* sred) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sred[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += sred[i];
    return t;
}
__device__ __forceinline__ float bf(uint16_t u) { return __uint_as_float((uint32_t) u << 16); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---------------------------------------------------------------- Raon-mHC
// block r < 17: z[r] = W[r] . flat(R); block 17: sum flat(R)^2
__global__ void __launch_bounds__(256) mhc_z_kernel(const float* __restrict__ R, const uint16_t* __restrict__ fn, int n4,
                                                    float* __restrict__ z) {
    __shared__ float sred[32];
    const int r = blockIdx.x;
    float acc = 0.0f;
    if (r < kSel) {
        const uint16_t* w = fn + (size_t) r * n4;
        for (int i = threadIdx.x; i < n4; i += blockDim.x) acc += bf(w[i]) * R[i];
    } else {
        for (int i = threadIdx.x; i < n4; i += blockDim.x) acc += R[i] * R[i];
    }
    acc = block_sum(acc, sred);
    if (threadIdx.x == 0) z[r] = acc;
}

// one block of 1024: the chart (thread 0, in double), then the branch input's mix and rms norm
__global__ void __launch_bounds__(1024) mhc_coef_kernel(const float* __restrict__ R, const float* __restrict__ z,
                                                        const float* __restrict__ base, const float* __restrict__ scale,
                                                        const float* __restrict__ norm_w, float hc_eps, float norm_eps,
                                                        float post_factor, int n, float* __restrict__ coef,
                                                        float* __restrict__ x) {
    __shared__ float sc[kCoef];
    __shared__ float sred[32];
    const int tid = threadIdx.x;
    if (tid == 0) {
        const double inv = 1.0 / sqrt((double) z[kSel] / (double) (kStreams * n) + (double) hc_eps);
        double zz[kSel];
        for (int r = 0; r < kSel; ++r) zz[r] = (double) z[r] * inv;
        const double s0 = scale[0], s1 = scale[1], s2 = scale[2], s3 = scale[3];
        for (int i = 0; i < 4; ++i) {
            sc[i] = (float) (1.0 / (1.0 + exp(-(s0 * zz[i] + (double) base[i]))));
            sc[4 + i] = (float) ((double) post_factor / (1.0 + exp(-(s1 * zz[4 + i] + (double) base[4 + i]))));
        }
        double t[3][3], Q[4][3];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) t[a][b] = s2 * zz[8 + 3 * a + b] + (double) base[8 + 3 * a + b];
        for (int i = 0; i < 4; ++i)
            for (int k = 1; k < 4; ++k) {
                const double nk = 1.0 / sqrt((double) k * (k + 1.0));
                Q[i][k - 1] = i < k ? nk : i == k ? -(double) k * nk : 0.0;
            }
        double Em[4][4], rho = 0.0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                double acc = 0.0;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) acc += Q[i][a] * t[a][b] * Q[j][b];
                Em[i][j] = acc;
                const double B = i == j ? 0.625 : 0.125;
                rho = fmax(rho, -acc / B);
            }
        const double gm = 1.0 / (1.0 + exp(-s3));
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                const double B = i == j ? 0.625 : 0.125;
                const double phi = B + Em[i][j] / (1.0 + rho);
                sc[8 + 4 * i + j] = (float) ((i == j ? 1.0 - gm : 0.0) + gm * phi);
            }
    }
    __syncthreads();
    if (tid < kCoef) coef[tid] = sc[tid];
    float ss = 0.0f;
    for (int e = tid; e < n; e += blockDim.x) {
        const float v = sc[0] * R[e] + sc[1] * R[n + e] + sc[2] * R[2 * n + e] + sc[3] * R[3 * n + e];
        x[e] = v;
        ss += v * v;
    }
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) n + norm_eps);
    for (int e = tid; e < n; e += blockDim.x) x[e] = x[e] * inv * norm_w[e];
}

__global__ void mhc_post_kernel(float* __restrict__ R, const float* __restrict__ out, const float* __restrict__ coef,
                                int n) {
    __shared__ float sc[kCoef];
    if (threadIdx.x < kCoef) sc[threadIdx.x] = coef[threadIdx.x];
    __syncthreads();
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    const float s0 = R[e], s1 = R[n + e], s2 = R[2 * n + e], s3 = R[3 * n + e], o = out[e];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float* h = sc + 8 + 4 * i;
        R[(size_t) i * n + e] = h[0] * s0 + h[1] * s1 + h[2] * s2 + h[3] * s3 + sc[4 + i] * o;
    }
}

// ---------------------------------------------------------------- RDA's write gate
__device__ __forceinline__ float logsig(float x) {   // log sigmoid(x), stable both ways
    return x >= 0.0f ? -log1pf(expf(-x)) : x - log1pf(expf(x));
}
__global__ void rda_ratio_kernel(float* __restrict__ v, const float* __restrict__ b, const float* __restrict__ w, int hd) {
    const int h = blockIdx.x;
    __shared__ float r;
    if (threadIdx.x == 0) r = expf(logsig(b[h] + w[h]) - logsig(b[h]));
    __syncthreads();
    for (int c = threadIdx.x; c < hd; c += blockDim.x) v[(size_t) h * hd + c] *= r;
}

// ---------------------------------------------------------------- the gated MLA's output
constexpr int kLat = 512, kV = 128;
// grid (n_head, kV / 8), 8 warps: a warp a row v of v_b
__global__ void mla_out_gated_kernel(const float* __restrict__ ctx, const float* __restrict__ v_b,
                                     const float* __restrict__ gate, float* __restrict__ o) {
    const int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int v = blockIdx.y * 8 + warp;
    const float4* w = reinterpret_cast<const float4*>(v_b + ((size_t) h * kV + v) * kLat);
    const float4* x = reinterpret_cast<const float4*>(ctx + (size_t) h * kLat);
    float d = 0.0f;
#pragma unroll
    for (int i = 0; i < kLat / 128; ++i) {
        const float4 a = w[lane + 32 * i], b = x[lane + 32 * i];
        d += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    d = warp_sum(d);
    if (lane == 0) o[(size_t) h * kV + v] = d * sigm(gate[(size_t) h * kV + v]);
}

__global__ void axpy_kernel(float* __restrict__ y, const float* __restrict__ c, float alpha, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * c[i];
}

}  // namespace

void mhc_pre(const float* R, const uint16_t* fn, const float* base, const float* scale, const float* norm_w,
             float hc_eps, float norm_eps, float post_factor, int n_embd, float* z, float* coef, float* x,
             cudaStream_t s) {
    mhc_z_kernel<<<kSel + 1, 256, 0, s>>>(R, fn, kStreams * n_embd, z);
    mhc_coef_kernel<<<1, 1024, 0, s>>>(R, z, base, scale, norm_w, hc_eps, norm_eps, post_factor, n_embd, coef, x);
    check("mhc_pre");
}

void mhc_post(float* R, const float* out, const float* coef, int n_embd, cudaStream_t s) {
    mhc_post_kernel<<<(n_embd + 255) / 256, 256, 0, s>>>(R, out, coef, n_embd);
    check("mhc_post");
}

void rda_ratio(float* v, const float* b, const float* w, int n_head, int head_dim, cudaStream_t s) {
    rda_ratio_kernel<<<n_head, 128, 0, s>>>(v, b, w, head_dim);
    check("rda_ratio");
}

void mla_out_gated(const float* ctx, const float* v_b, const float* gate, int n_head, float* o, cudaStream_t s) {
    mla_out_gated_kernel<<<dim3(n_head, kV / 8), 256, 0, s>>>(ctx, v_b, gate, o);
    check("mla_out_gated");
}

void axpy(float* y, const float* c, float alpha, int n, cudaStream_t s) {
    axpy_kernel<<<(n + 255) / 256, 256, 0, s>>>(y, c, alpha, n);
    check("axpy");
}

int launch_errors() { return g_errors.load(); }

}  // namespace strata::kernels::raon
