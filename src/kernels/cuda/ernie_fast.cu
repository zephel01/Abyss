// src/kernels/cuda/ernie_fast.cu - ERNIE-5.5-Flash-Preview's decode kernels (include/strata/kernels/ernie_fast.hpp).
// Correctness first: plain loops, F32 sums, the RoPE angles in double.
#include "strata/kernels/ernie_fast.hpp"

#include <cuda_fp16.h>

#include <atomic>
#include <cmath>
#include <cstdio>

namespace strata::kernels::ernie {

namespace {
std::atomic<int> g_errors{0};
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        g_errors.fetch_add(1);
        std::fprintf(stderr, "ernie kernels: %s: %s\n", what, cudaGetErrorString(e));
    }
}

__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float bf16_round(float x) {   // round to nearest even, as torch's .to(bfloat16)
    unsigned int u = __float_as_uint(x);
    if ((u & 0x7f800000u) == 0x7f800000u) return x;
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    return __uint_as_float(u);
}

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((unsigned int) v << 16); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// a block's sum / max (blockDim a multiple of 32, <= 1024); every thread gets the result
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = lane < nw ? sh[lane] : 0.0f;
    return warp_sum(t);
}
__device__ float block_max(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = lane < nw ? sh[lane] : -INFINITY;
    return warp_max(t);
}

// the plain RoPE angle of pair i (of R / 2) at position p: p * base^(-2 i / R), in double
__device__ __forceinline__ void rope_cs(int p, int i, int R, float base, float& c, float& s) {
    const double th = (double) p * pow((double) base, -2.0 * (double) i / (double) R);
    double sd, cd;
    sincos(th, &sd, &cd);
    c = (float) cd;
    s = (float) sd;
}

__global__ void router2_kernel(const float* __restrict__ l0, const float* __restrict__ l1, float* __restrict__ s, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) s[i] = sigm(l0[i]) + sigm(l1[i]);
}

__global__ void vha_kernel(float* __restrict__ o, const float* __restrict__ M, int H, int D) {
    __shared__ float sM[64 * 64];
    for (int i = threadIdx.x; i < H * H; i += blockDim.x) sM[i] = M[i];
    __syncthreads();
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float col[64];
#pragma unroll
    for (int g = 0; g < 64; ++g) col[g] = g < H ? o[(size_t) g * D + d] : 0.0f;
    for (int h = 0; h < H; ++h) {
        float a = 0.0f;
#pragma unroll
        for (int g = 0; g < 64; ++g)
            if (g < H) a += sM[h * H + g] * col[g];
        o[(size_t) h * D + d] = a;
    }
}

__global__ void gate_mul_kernel(float* __restrict__ z, const float* __restrict__ gl, int n, int b16) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = sigm(gl[i]);
    z[i] *= b16 ? bf16_round(g) : g;
}

__global__ void situ_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = g[i], b = u[i];
    h[i] = (4.0f * tanhf(a * 0.25f)) * (1.0f / (1.0f + expf(-a))) * (25.0f * tanhf(b * 0.04f));
}

__global__ void rms_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int n,
                           float eps) {
    __shared__ float sh[32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    const float inv = rsqrtf(block_sum(ss, sh) / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

__global__ void multimax_kernel(float* __restrict__ x, int n, const float* __restrict__ r, const float* __restrict__ t) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    const float a = fmaxf(r[0] - v, 0.0f), b = fmaxf(v - r[1], 0.0f), c = fmaxf(r[2] - v, 0.0f), d = fmaxf(v - r[3], 0.0f);
    x[i] = v + t[0] * a + t[1] * b + t[2] * c * c + t[3] * d * d;
}

__global__ void mla_kv_kernel(const float* __restrict__ kva, const float* __restrict__ norm, float* __restrict__ row,
                              int L, int R, int p, float base, float eps) {
    __shared__ float sh[32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < L; i += blockDim.x) ss += kva[i] * kva[i];
    const float inv = rsqrtf(block_sum(ss, sh) / (float) L + eps);
    for (int i = threadIdx.x; i < L; i += blockDim.x) row[i] = kva[i] * inv * norm[i];
    for (int i = threadIdx.x; i < R / 2; i += blockDim.x) {
        float c, s;
        rope_cs(p, i, R, base, c, s);
        const float x0 = kva[L + 2 * i], x1 = kva[L + 2 * i + 1];
        row[L + 2 * i] = x0 * c - x1 * s;
        row[L + 2 * i + 1] = x0 * s + x1 * c;
    }
}

__global__ void mla_q_rope_kernel(float* __restrict__ q, int nope, int R, int p, float base) {
    const int h = blockIdx.x;
    float* qh = q + (size_t) h * (nope + R) + nope;
    for (int i = threadIdx.x; i < R / 2; i += blockDim.x) {
        float c, s;
        rope_cs(p, i, R, base, c, s);
        const float x0 = qh[2 * i], x1 = qh[2 * i + 1];
        qh[2 * i] = x0 * c - x1 * s;
        qh[2 * i + 1] = x0 * s + x1 * c;
    }
}

// block h, a thread a latent column c (blockDim = L)
__global__ void mla_absorb_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kvb, float* __restrict__ q_lat,
                                  int nope, int R, int V, int L) {
    extern __shared__ float sq[];
    const int h = blockIdx.x;
    for (int i = threadIdx.x; i < nope; i += blockDim.x) sq[i] = q[(size_t) h * (nope + R) + i];
    __syncthreads();
    const uint16_t* W = kvb + (size_t) h * (nope + V) * L;
    for (int c = threadIdx.x; c < L; c += blockDim.x) {
        float a = 0.0f;
        for (int i = 0; i < nope; ++i) a += sq[i] * bf(W[(size_t) i * L + c]);
        q_lat[(size_t) h * L + c] = a;
    }
}

// block h (256 threads): a warp a cache row for the scores; then the softmax; then o_lat
__global__ void mla_attend_kernel(const float* __restrict__ q_lat, const float* __restrict__ q,
                                  const float* __restrict__ cache, const int* __restrict__ rows, int n,
                                  const float* __restrict__ sinks, float scale, float* __restrict__ score,
                                  float* __restrict__ o_lat, int nope, int R, int L) {
    extern __shared__ float sm[];   // [L + R] the query (latent | rope)
    __shared__ float sh[32];
    const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, nw = blockDim.x >> 5;
    const int W = L + R;
    for (int i = tid; i < L; i += blockDim.x) sm[i] = q_lat[(size_t) h * L + i];
    for (int i = tid; i < R; i += blockDim.x) sm[L + i] = q[(size_t) h * (nope + R) + nope + i];
    __syncthreads();
    float* sc = score + (size_t) h * n;
    for (int j = w; j < n; j += nw) {
        const float* row = cache + (size_t) (rows ? rows[j] : j) * W;
        float a = 0.0f;
        for (int i = lane; i < W; i += 32) a += sm[i] * row[i];
        a = warp_sum(a);
        if (lane == 0) sc[j] = a * scale;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int j = tid; j < n; j += blockDim.x) m = fmaxf(m, sc[j]);
    const float sink = sinks[h];
    m = fmaxf(block_max(m, sh), sink);
    float den = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        const float e = expf(sc[j] - m);
        sc[j] = e;
        den += e;
    }
    den = block_sum(den, sh) + expf(sink - m);
    const float inv = 1.0f / den;
    __syncthreads();
    for (int c = tid; c < L; c += blockDim.x) {
        float a = 0.0f;
        for (int j = 0; j < n; ++j) a += sc[j] * cache[(size_t) (rows ? rows[j] : j) * W + c];
        o_lat[(size_t) h * L + c] = a * inv;
    }
}

// block h (256 threads): a warp an output dim d
__global__ void mla_v_kernel(const float* __restrict__ o_lat, const uint16_t* __restrict__ kvb, float* __restrict__ o,
                             int nope, int V, int L) {
    extern __shared__ float so[];
    const int h = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int i = threadIdx.x; i < L; i += blockDim.x) so[i] = o_lat[(size_t) h * L + i];
    __syncthreads();
    const uint16_t* Wv = kvb + ((size_t) h * (nope + V) + nope) * L;
    for (int d = w; d < V; d += nw) {
        const uint16_t* row = Wv + (size_t) d * L;
        float a = 0.0f;
        for (int c = lane; c < L; c += 32) a += bf(row[c]) * so[c];
        a = warp_sum(a);
        if (lane == 0) o[(size_t) h * V + d] = a;
    }
}

// the DSA indexer's key of position p: LayerNorm (weight + bias) of k_raw [D], RoPE rotate-half (NEOX) on its first R
// dims (pairs i, i + R/2), FP16 -> row.  One block of D (128) threads.
__global__ void idx_key_kernel(const float* __restrict__ kraw, const float* __restrict__ w, const float* __restrict__ b,
                               uint16_t* __restrict__ row, int D, int R, int p, float base, float eps) {
    __shared__ float sh[32];
    __shared__ float v[512];
    const int i = threadIdx.x;
    const float x = i < D ? kraw[i] : 0.0f;
    const float mu = block_sum(x, sh) / (float) D;
    const float d = i < D ? x - mu : 0.0f;
    const float var = block_sum(d * d, sh) / (float) D;
    const float inv = rsqrtf(var + eps);
    if (i < D) v[i] = d * inv * w[i] + b[i];
    __syncthreads();
    float y = i < D ? v[i] : 0.0f;
    if (i < R) {
        const int h = R / 2, k = i < h ? i : i - h;
        float c, s;
        rope_cs(p, k, R, base, c, s);
        y = i < h ? v[i] * c - v[i + h] * s : v[i] * c + v[i - h] * s;
    }
    if (i < D) row[i] = __half_as_ushort(__float2half_rn(y));
}

// the indexer's queries: RoPE rotate-half on the first R dims of each of the H heads of D (block h)
__global__ void idx_q_rope_kernel(float* __restrict__ iq, int D, int R, int p, float base) {
    float* qh = iq + (size_t) blockIdx.x * D;
    const int h = R / 2;
    for (int k = threadIdx.x; k < h; k += blockDim.x) {
        float c, s;
        rope_cs(p, k, R, base, c, s);
        const float x0 = qh[k], x1 = qh[k + h];
        qh[k] = x0 * c - x1 * s;
        qh[k + h] = x1 * c + x0 * s;
    }
}

__global__ void iota_kernel(int* __restrict__ out, int start, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = start + i;
}
}  // namespace

int launch_errors() { return g_errors.load(); }

void router2(const float* l0, const float* l1, float* s, int n, cudaStream_t st) {
    router2_kernel<<<(n + 255) / 256, 256, 0, st>>>(l0, l1, s, n);
    check("router2");
}
void vha(float* o, const float* M, int H, int D, cudaStream_t st) {
    if (H > 64) {
        g_errors.fetch_add(1);
        std::fprintf(stderr, "ernie kernels: vha takes <= 64 heads (%d)\n", H);
        return;
    }
    vha_kernel<<<(D + 127) / 128, 128, 0, st>>>(o, M, H, D);
    check("vha");
}
void gate_mul(float* z, const float* gl, int n, int bf16, cudaStream_t st) {
    gate_mul_kernel<<<(n + 255) / 256, 256, 0, st>>>(z, gl, n, bf16);
    check("gate_mul");
}
void situ(const float* g, const float* u, float* h, int n, cudaStream_t st) {
    situ_kernel<<<(n + 255) / 256, 256, 0, st>>>(g, u, h, n);
    check("situ");
}
void rms(const float* x, const float* w, float* y, int n, float eps, cudaStream_t st) {
    rms_kernel<<<1, 1024, 0, st>>>(x, w, y, n, eps);
    check("rms");
}
void multimax(float* x, int n, const float* r, const float* t, cudaStream_t st) {
    multimax_kernel<<<(n + 255) / 256, 256, 0, st>>>(x, n, r, t);
    check("multimax");
}
void mla_kv(const float* kva, const float* norm, float* row, int L, int R, int p, float base, float eps, cudaStream_t st) {
    mla_kv_kernel<<<1, 512, 0, st>>>(kva, norm, row, L, R, p, base, eps);
    check("mla_kv");
}
void mla_q_rope(float* q, int H, int nope, int R, int p, float base, cudaStream_t st) {
    mla_q_rope_kernel<<<H, 32, 0, st>>>(q, nope, R, p, base);
    check("mla_q_rope");
}
void mla_absorb(const float* q, const uint16_t* kvb, float* q_lat, int H, int nope, int R, int V, int L, cudaStream_t st) {
    mla_absorb_kernel<<<H, 512, (size_t) nope * sizeof(float), st>>>(q, kvb, q_lat, nope, R, V, L);
    check("mla_absorb");
}
void mla_attend(const float* q_lat, const float* q, const float* cache, const int* rows, int n, const float* sinks,
                float scale, float* score, float* o_lat, int H, int nope, int R, int L, cudaStream_t st) {
    mla_attend_kernel<<<H, 256, (size_t) (L + R) * sizeof(float), st>>>(q_lat, q, cache, rows, n, sinks, scale, score,
                                                                         o_lat, nope, R, L);
    check("mla_attend");
}
void mla_v(const float* o_lat, const uint16_t* kvb, float* o, int H, int nope, int V, int L, cudaStream_t st) {
    mla_v_kernel<<<H, 256, (size_t) L * sizeof(float), st>>>(o_lat, kvb, o, nope, V, L);
    check("mla_v");
}

void idx_key(const float* kraw, const float* w, const float* b, uint16_t* row, int D, int R, int p, float base, float eps,
             cudaStream_t st) {
    if (D > 512 || D % 32 != 0 || R > D || R % 2 != 0) {
        g_errors.fetch_add(1);
        std::fprintf(stderr, "ernie kernels: idx_key takes D <= 512 (a multiple of 32), R <= D (%d, %d)\n", D, R);
        return;
    }
    idx_key_kernel<<<1, D, 0, st>>>(kraw, w, b, row, D, R, p, base, eps);
    check("idx_key");
}
void idx_q_rope(float* iq, int H, int D, int R, int p, float base, cudaStream_t st) {
    idx_q_rope_kernel<<<H, 32, 0, st>>>(iq, D, R, p, base);
    check("idx_q_rope");
}
void iota(int* out, int start, int n, cudaStream_t st) {
    if (n <= 0) return;
    iota_kernel<<<(n + 255) / 256, 256, 0, st>>>(out, start, n);
    check("iota");
}

}  // namespace strata::kernels::ernie
