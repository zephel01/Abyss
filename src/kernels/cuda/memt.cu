// src/kernels/cuda/memt.cu - MemT's memory read and turn-end write (include/strata/kernels/memt.hpp has the math).
// FP32 throughout; the sums run in a fixed order (the same bits run to run).
#include "strata/kernels/memt.hpp"

#include <cuda_runtime.h>

namespace strata::kernels::memt {

namespace {

constexpr int kRows = 16;   // the read's rows a block (each state row read once for them)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// grid (ceil(T / kRows), H), block kD
__global__ void k_read_rows(const float* __restrict__ q, const float* __restrict__ g, const float* __restrict__ S,
                            const float* __restrict__ onorm, float* __restrict__ ao, int T, int H, float scale,
                            float eps, float softcap) {
    __shared__ float buf[kRows][kD];
    __shared__ float inv[kRows];
    const int h = blockIdx.y, t0 = blockIdx.x * kRows, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n = min(kRows, T - t0);
    const size_t W = (size_t) H * kD;
    for (int j = 0; j < kRows; ++j) buf[j][tid] = j < n ? q[(size_t) (t0 + j) * W + (size_t) h * kD + tid] : 0.0f;
    __syncthreads();
    // the L2 norms: 2 rows a warp
    for (int j = warp; j < kRows; j += kD / 32) {
        float ss = 0.0f;
        for (int i = lane; i < kD; i += 32) ss += buf[j][i] * buf[j][i];
        ss = warp_sum(ss);
        if (lane == 0) inv[j] = rsqrtf(ss + 1e-6f);
    }
    __syncthreads();
    float acc[kRows];
#pragma unroll
    for (int j = 0; j < kRows; ++j) acc[j] = 0.0f;
    const float* Sh = S + (size_t) h * kD * kD + tid;
    for (int k = 0; k < kD; ++k) {
        const float sv = Sh[(size_t) k * kD];
#pragma unroll
        for (int j = 0; j < kRows; ++j) acc[j] = fmaf(buf[j][k] * inv[j], sv, acc[j]);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kRows; ++j) buf[j][tid] = scale * acc[j];
    __syncthreads();
    for (int j = warp; j < kRows; j += kD / 32) {
        float ss = 0.0f;
        for (int i = lane; i < kD; i += 32) ss += buf[j][i] * buf[j][i];
        ss = warp_sum(ss);
        if (lane == 0) inv[j] = rsqrtf(ss / (float) kD + eps);
    }
    __syncthreads();
    const float w = onorm[tid];
    for (int j = 0; j < n; ++j) {
        const size_t at = (size_t) (t0 + j) * W + (size_t) h * kD + tid;
        float a = silu(g[at]);
        if (softcap > 0.0f) a = softcap * tanhf(a / softcap);
        ao[at] += buf[j][tid] * inv[j] * w * a;
    }
}

// grid (H, kSplit), block kD: rows k of the slice, every column v (a thread) - part[h][slice][v]
__global__ void k_read_part(const float* __restrict__ q, const float* __restrict__ S, float* __restrict__ part, int H) {
    __shared__ float qs[kD];
    __shared__ float red[kD / 32];
    const int h = blockIdx.x, sl = blockIdx.y, tid = threadIdx.x;
    const float qv = q[(size_t) h * kD + tid];
    qs[tid] = qv;
    float ss = warp_sum(qv * qv);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int i = 0; i < kD / 32; ++i) ss += red[i];
    const float inv = rsqrtf(ss + 1e-6f);
    constexpr int R = kD / kSplit;
    const float* Sh = S + (size_t) h * kD * kD + (size_t) sl * R * kD + tid;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < R; ++k) acc = fmaf(qs[sl * R + k] * inv, Sh[(size_t) k * kD], acc);
    part[((size_t) h * kSplit + sl) * kD + tid] = acc;
}

// grid H, block kD: the slices summed in order, RMS-normed, o_norm, the gate, += ao
__global__ void k_read_fin(const float* __restrict__ part, const float* __restrict__ g, const float* __restrict__ onorm,
                           float* __restrict__ ao, float scale, float eps, float softcap) {
    __shared__ float red[kD / 32];
    const int h = blockIdx.x, tid = threadIdx.x;
    float r = 0.0f;
#pragma unroll
    for (int sl = 0; sl < kSplit; ++sl) r += part[((size_t) h * kSplit + sl) * kD + tid];
    r *= scale;
    float ss = warp_sum(r * r);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int i = 0; i < kD / 32; ++i) ss += red[i];
    const float inv = rsqrtf(ss / (float) kD + eps);
    const size_t at = (size_t) h * kD + tid;
    float a = silu(g[at]);
    if (softcap > 0.0f) a = softcap * tanhf(a / softcap);
    ao[at] += r * inv * onorm[tid] * a;
}

// grid (T, H), block kD
__global__ void k_write_prep(float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ b,
                             float* __restrict__ w, float* __restrict__ f, const float* __restrict__ bias,
                             const float* __restrict__ dsl, int H) {
    __shared__ float part[kD / 32];
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const size_t at = (size_t) t * H * kD + (size_t) h * kD + tid;
    const float kv = k[at];
    float ss = warp_sum(kv * kv);
    if ((tid & 31) == 0) part[tid >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int i = 0; i < kD / 32; ++i) ss += part[i];
    k[at] = kv * rsqrtf(ss + 1e-6f) * sqrtf(sigm(b[at]));
    w[at] = sigm(w[at]) * v[at];
    f[at] = expf(-expf(dsl[h]) * softplus(f[at] + bias[(size_t) h * kD + tid]));
}

// grid (H, kD / 8), block 256: a warp a value column, a lane 8 key rows (k = lane + 32 i)
__global__ void k_write_scan(float* __restrict__ S, const float* __restrict__ u, const float* __restrict__ wv,
                             const float* __restrict__ lam, int T, int H) {
    const int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int v = blockIdx.y * 8 + warp;
    float* Sh = S + (size_t) h * kD * kD + v;
    float s[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = Sh[(size_t) (lane + 32 * i) * kD];
    const size_t W = (size_t) H * kD;
    for (int t = 0; t < T; ++t) {
        const float* ut = u + (size_t) t * W + (size_t) h * kD;
        const float* lt = lam + (size_t) t * W + (size_t) h * kD;
        float uk[8], e = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uk[i] = ut[lane + 32 * i];
            s[i] *= lt[lane + 32 * i];
            e = fmaf(uk[i], s[i], e);
        }
        e = warp_sum(e);
        const float d = wv[(size_t) t * W + (size_t) h * kD + v] - e;
#pragma unroll
        for (int i = 0; i < 8; ++i) s[i] = fmaf(uk[i], d, s[i]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) Sh[(size_t) (lane + 32 * i) * kD] = s[i];
}

}  // namespace

void read_rows(const float* q, const float* g, const float* S, const float* onorm, float* ao, int T, int H,
               float scale, float eps, float softcap, cudaStream_t s) {
    if (T <= 0) return;
    k_read_rows<<<dim3((unsigned) ((T + kRows - 1) / kRows), (unsigned) H), kD, 0, s>>>(q, g, S, onorm, ao, T, H, scale,
                                                                                       eps, softcap);
}

void read_tok(const float* q, const float* g, const float* S, const float* onorm, float* ao, float* part, int H,
              float scale, float eps, float softcap, cudaStream_t s) {
    k_read_part<<<dim3((unsigned) H, (unsigned) kSplit), kD, 0, s>>>(q, S, part, H);
    k_read_fin<<<(unsigned) H, kD, 0, s>>>(part, g, onorm, ao, scale, eps, softcap);
}

void write_prep(float* k, const float* v, const float* b, float* w, float* f, const float* bias, const float* dsl,
                int T, int H, cudaStream_t s) {
    if (T <= 0) return;
    k_write_prep<<<dim3((unsigned) T, (unsigned) H), kD, 0, s>>>(k, v, b, w, f, bias, dsl, H);
}

void write_scan(float* S, const float* u, const float* wv, const float* lam, int T, int H, cudaStream_t s) {
    if (T <= 0) return;
    k_write_scan<<<dim3((unsigned) H, (unsigned) (kD / 8)), 256, 0, s>>>(S, u, wv, lam, T, H);
}

}  // namespace strata::kernels::memt
