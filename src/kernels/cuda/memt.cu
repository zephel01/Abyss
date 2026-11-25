// src/kernels/cuda/memt.cu - MemT's memory read and turn-end write (include/strata/kernels/memt.hpp has the math).
// FP32 throughout (the BF16 state read rounds each state value to BF16 first); the sums run in a fixed order (the same
// bits run to run).  Every kernel is a template on the head dim D (256: Nano, 512: Flash), a block D threads.
#include "strata/kernels/memt.hpp"

#include <cuda_bf16.h>
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
// the state's value: M 0 FP32 as it is, 1 FP32 rounded to BF16, 2 stored as BF16 (the same values as 1)
template <int M>
__device__ __forceinline__ float st_ld(const void* p, size_t i) {
    if constexpr (M == 2) return __uint_as_float((unsigned) ((const uint16_t*) p)[i] << 16);
    else if constexpr (M == 1) return __bfloat162float(__float2bfloat16_rn(((const float*) p)[i]));
    else return ((const float*) p)[i];
}

// grid (ceil(T / kRows), H), block D
template <int D, int M>
__global__ void k_read_rows(const float* __restrict__ q, const float* __restrict__ g, const float* __restrict__ gbias,
                            const void* __restrict__ S, const float* __restrict__ onorm, float* __restrict__ ao,
                            int T, int H, float scale, float eps, float softcap) {
    __shared__ float buf[kRows][D];
    __shared__ float inv[kRows];
    const int h = blockIdx.y, t0 = blockIdx.x * kRows, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n = min(kRows, T - t0);
    const size_t W = (size_t) H * D;
    for (int j = 0; j < kRows; ++j) buf[j][tid] = j < n ? q[(size_t) (t0 + j) * W + (size_t) h * D + tid] : 0.0f;
    __syncthreads();
    // the L2 norms: kRows / (D / 32) rows a warp
    for (int j = warp; j < kRows; j += D / 32) {
        float ss = 0.0f;
        for (int i = lane; i < D; i += 32) ss += buf[j][i] * buf[j][i];
        ss = warp_sum(ss);
        if (lane == 0) inv[j] = rsqrtf(ss + 1e-6f);
    }
    __syncthreads();
    float acc[kRows];
#pragma unroll
    for (int j = 0; j < kRows; ++j) acc[j] = 0.0f;
    const size_t Sh = (size_t) h * D * D + tid;
    for (int k = 0; k < D; ++k) {
        const float sv = st_ld<M>(S, Sh + (size_t) k * D);
#pragma unroll
        for (int j = 0; j < kRows; ++j) acc[j] = fmaf(buf[j][k] * inv[j], sv, acc[j]);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kRows; ++j) buf[j][tid] = scale * acc[j];
    __syncthreads();
    for (int j = warp; j < kRows; j += D / 32) {
        float ss = 0.0f;
        for (int i = lane; i < D; i += 32) ss += buf[j][i] * buf[j][i];
        ss = warp_sum(ss);
        if (lane == 0) inv[j] = rsqrtf(ss / (float) D + eps);
    }
    __syncthreads();
    const float w = onorm[tid];
    const float gb = gbias != nullptr ? gbias[(size_t) h * D + tid] : 0.0f;
    for (int j = 0; j < n; ++j) {
        const size_t at = (size_t) (t0 + j) * W + (size_t) h * D + tid;
        float a = silu(gbias != nullptr ? g[at] + gb : g[at]);
        if (softcap > 0.0f) a = softcap * tanhf(a / softcap);
        ao[at] += buf[j][tid] * inv[j] * w * a;
    }
}

// grid (H, kSplit), block D: rows k of the slice, every column v (a thread) - part[h][slice][v]
template <int D, int M>
__global__ void k_read_part(const float* __restrict__ q, const void* __restrict__ S, float* __restrict__ part, int H) {
    __shared__ float qs[D];
    __shared__ float red[D / 32];
    const int h = blockIdx.x, sl = blockIdx.y, tid = threadIdx.x;
    const float qv = q[(size_t) h * D + tid];
    qs[tid] = qv;
    float ss = warp_sum(qv * qv);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int i = 0; i < D / 32; ++i) ss += red[i];
    const float inv = rsqrtf(ss + 1e-6f);
    constexpr int R = D / kSplit;
    const size_t Sh = (size_t) h * D * D + (size_t) sl * R * D + tid;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < R; ++k) acc = fmaf(qs[sl * R + k] * inv, st_ld<M>(S, Sh + (size_t) k * D), acc);
    part[((size_t) h * kSplit + sl) * D + tid] = acc;
}

// grid H, block D: the slices summed in order, RMS-normed, o_norm, the gate, += ao
template <int D>
__global__ void k_read_fin(const float* __restrict__ part, const float* __restrict__ g, const float* __restrict__ gbias,
                           const float* __restrict__ onorm, float* __restrict__ ao, float scale, float eps,
                           float softcap) {
    __shared__ float red[D / 32];
    const int h = blockIdx.x, tid = threadIdx.x;
    float r = 0.0f;
#pragma unroll
    for (int sl = 0; sl < kSplit; ++sl) r += part[((size_t) h * kSplit + sl) * D + tid];
    r *= scale;
    float ss = warp_sum(r * r);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int i = 0; i < D / 32; ++i) ss += red[i];
    const float inv = rsqrtf(ss / (float) D + eps);
    const size_t at = (size_t) h * D + tid;
    float a = silu(gbias != nullptr ? g[at] + gbias[at] : g[at]);
    if (softcap > 0.0f) a = softcap * tanhf(a / softcap);
    ao[at] += r * inv * onorm[tid] * a;
}

// grid (T, H), block D
template <int D>
__global__ void k_write_prep(float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ b,
                             float* __restrict__ w, float* __restrict__ f, const float* __restrict__ bias,
                             const float* __restrict__ dsl, int H) {
    __shared__ float part[D / 32];
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const size_t at = (size_t) t * H * D + (size_t) h * D + tid;
    const float kv = k[at];
    float ss = warp_sum(kv * kv);
    if ((tid & 31) == 0) part[tid >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int i = 0; i < D / 32; ++i) ss += part[i];
    k[at] = kv * rsqrtf(ss + 1e-6f) * sqrtf(sigm(b[at]));
    w[at] = sigm(w[at]) * v[at];
    f[at] = expf(-expf(dsl[h]) * softplus(f[at] + bias[(size_t) h * D + tid]));
}

// grid (H, D / 8), block 256: a warp a value column, a lane D / 32 key rows (k = lane + 32 i)
template <int D>
__global__ void k_write_scan(float* __restrict__ S, const float* __restrict__ u, const float* __restrict__ wv,
                             const float* __restrict__ lam, int T, int H) {
    constexpr int R = D / 32;
    const int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int v = blockIdx.y * 8 + warp;
    float* Sh = S + (size_t) h * D * D + v;
    float s[R];
#pragma unroll
    for (int i = 0; i < R; ++i) s[i] = Sh[(size_t) (lane + 32 * i) * D];
    const size_t W = (size_t) H * D;
    for (int t = 0; t < T; ++t) {
        const float* ut = u + (size_t) t * W + (size_t) h * D;
        const float* lt = lam + (size_t) t * W + (size_t) h * D;
        float uk[R], e = 0.0f;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            uk[i] = ut[lane + 32 * i];
            s[i] *= lt[lane + 32 * i];
            e = fmaf(uk[i], s[i], e);
        }
        e = warp_sum(e);
        const float d = wv[(size_t) t * W + (size_t) h * D + v] - e;
#pragma unroll
        for (int i = 0; i < R; ++i) s[i] = fmaf(uk[i], d, s[i]);
    }
#pragma unroll
    for (int i = 0; i < R; ++i) Sh[(size_t) (lane + 32 * i) * D] = s[i];
}

// grid (T, G), block 256
__global__ void k_group_rms(float* __restrict__ x, const float* __restrict__ w, int width, int ld, float eps) {
    __shared__ float red[8];
    const int t = blockIdx.x, gi = blockIdx.y, tid = threadIdx.x;
    float* r = x + (size_t) t * ld + (size_t) gi * width;
    float ss = 0.0f;
    for (int i = tid; i < width; i += 256) ss += r[i] * r[i];
    ss = warp_sum(ss);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) ss += red[i];
    const float inv = rsqrtf(ss / (float) width + eps);
    const float* wg = w + (size_t) gi * width;
    for (int i = tid; i < width; i += 256) r[i] = r[i] * inv * wg[i];
}

template <int D>
void read_rows_t(const float* q, const float* g, const float* gbias, const void* S, const float* onorm, float* ao,
                 int T, int H, int m, float scale, float eps, float softcap, cudaStream_t s) {
    const dim3 grid((unsigned) ((T + kRows - 1) / kRows), (unsigned) H);
    if (m == 2) k_read_rows<D, 2><<<grid, D, 0, s>>>(q, g, gbias, S, onorm, ao, T, H, scale, eps, softcap);
    else if (m == 1) k_read_rows<D, 1><<<grid, D, 0, s>>>(q, g, gbias, S, onorm, ao, T, H, scale, eps, softcap);
    else k_read_rows<D, 0><<<grid, D, 0, s>>>(q, g, gbias, S, onorm, ao, T, H, scale, eps, softcap);
}

template <int D>
void read_tok_t(const float* q, const float* g, const float* gbias, const void* S, const float* onorm, float* ao,
                float* part, int H, int m, float scale, float eps, float softcap, cudaStream_t s) {
    const dim3 grid((unsigned) H, (unsigned) kSplit);
    if (m == 2) k_read_part<D, 2><<<grid, D, 0, s>>>(q, S, part, H);
    else if (m == 1) k_read_part<D, 1><<<grid, D, 0, s>>>(q, S, part, H);
    else k_read_part<D, 0><<<grid, D, 0, s>>>(q, S, part, H);
    k_read_fin<D><<<(unsigned) H, D, 0, s>>>(part, g, gbias, onorm, ao, scale, eps, softcap);
}

// round to nearest even (what __float2bfloat16_rn gives a finite value; NaN -> 0x7fff) - memt_path.cu's host copy
// rounds the same way
__global__ void k_to_bf16(const float* __restrict__ x, uint16_t* __restrict__ y, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const unsigned u = __float_as_uint(x[i]);
        y[i] = (u & 0x7fffffffu) > 0x7f800000u ? (uint16_t) 0x7fff : (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
}

}  // namespace

void read_rows(const float* q, const float* g, const float* gbias, const float* S, const float* onorm, float* ao,
               int T, int H, int D, bool bf16_state, float scale, float eps, float softcap, cudaStream_t s) {
    if (T <= 0) return;
    if (D == 512) read_rows_t<512>(q, g, gbias, S, onorm, ao, T, H, bf16_state ? 1 : 0, scale, eps, softcap, s);
    else read_rows_t<256>(q, g, gbias, S, onorm, ao, T, H, bf16_state ? 1 : 0, scale, eps, softcap, s);
}

void read_rows16(const float* q, const float* g, const float* gbias, const uint16_t* S, const float* onorm, float* ao,
                 int T, int H, int D, float scale, float eps, float softcap, cudaStream_t s) {
    if (T <= 0) return;
    if (D == 512) read_rows_t<512>(q, g, gbias, S, onorm, ao, T, H, 2, scale, eps, softcap, s);
    else read_rows_t<256>(q, g, gbias, S, onorm, ao, T, H, 2, scale, eps, softcap, s);
}

void read_tok(const float* q, const float* g, const float* gbias, const float* S, const float* onorm, float* ao,
              float* part, int H, int D, bool bf16_state, float scale, float eps, float softcap, cudaStream_t s) {
    if (D == 512) read_tok_t<512>(q, g, gbias, S, onorm, ao, part, H, bf16_state ? 1 : 0, scale, eps, softcap, s);
    else read_tok_t<256>(q, g, gbias, S, onorm, ao, part, H, bf16_state ? 1 : 0, scale, eps, softcap, s);
}

void read_tok16(const float* q, const float* g, const float* gbias, const uint16_t* S, const float* onorm, float* ao,
                float* part, int H, int D, float scale, float eps, float softcap, cudaStream_t s) {
    if (D == 512) read_tok_t<512>(q, g, gbias, S, onorm, ao, part, H, 2, scale, eps, softcap, s);
    else read_tok_t<256>(q, g, gbias, S, onorm, ao, part, H, 2, scale, eps, softcap, s);
}

void to_bf16(const float* x, uint16_t* y, size_t n, cudaStream_t s) {
    if (n == 0) return;
    k_to_bf16<<<1024, 256, 0, s>>>(x, y, n);
}

void write_prep(float* k, const float* v, const float* b, float* w, float* f, const float* bias, const float* dsl,
                int T, int H, int D, cudaStream_t s) {
    if (T <= 0) return;
    if (D == 512) k_write_prep<512><<<dim3((unsigned) T, (unsigned) H), 512, 0, s>>>(k, v, b, w, f, bias, dsl, H);
    else k_write_prep<256><<<dim3((unsigned) T, (unsigned) H), 256, 0, s>>>(k, v, b, w, f, bias, dsl, H);
}

void write_scan(float* S, const float* u, const float* wv, const float* lam, int T, int H, int D, cudaStream_t s) {
    if (T <= 0) return;
    if (D == 512) k_write_scan<512><<<dim3((unsigned) H, 512u / 8), 256, 0, s>>>(S, u, wv, lam, T, H);
    else k_write_scan<256><<<dim3((unsigned) H, 256u / 8), 256, 0, s>>>(S, u, wv, lam, T, H);
}

void group_rms(float* x, const float* w, int T, int G, int width, int ld, float eps, cudaStream_t s) {
    if (T <= 0) return;
    k_group_rms<<<dim3((unsigned) T, (unsigned) G), 256, 0, s>>>(x, w, width, ld, eps);
}

}  // namespace strata::kernels::memt
