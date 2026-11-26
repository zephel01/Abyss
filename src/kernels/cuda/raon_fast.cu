// src/kernels/cuda/raon_fast.cu - see include/strata/kernels/raon_fast.hpp.
#include "strata/kernels/raon_fast.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdio>

// the prompt attention's WMMA (as ling_fast's: sm_80+; HIP: none - prompt_attn_ok false, the rows a token at a time)
#if defined(__HIPCC__) || defined(STRATA_USE_HIP)
#define STRATA_RAON_PA 0
#else
#define STRATA_RAON_PA 1
#include <mma.h>
#endif

namespace strata::kernels::raon {
namespace {
constexpr int kLat = 512, kPe = 64, kZ = 32;

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
    R += (size_t) blockIdx.y * n4;   // (a row of the prompt path: its 4 streams, its z)
    z += (size_t) blockIdx.y * kZ;
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
    R += (size_t) blockIdx.x * kStreams * n;
    z += (size_t) blockIdx.x * kZ;
    coef += (size_t) blockIdx.x * kCoef;
    x += (size_t) blockIdx.x * n;
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
    R += (size_t) blockIdx.y * kStreams * n;
    out += (size_t) blockIdx.y * n;
    coef += (size_t) blockIdx.y * kCoef;
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
    v += (size_t) blockIdx.y * gridDim.x * hd;
    b += (size_t) blockIdx.y * gridDim.x;
    w += (size_t) blockIdx.y * gridDim.x;
    __shared__ float r;
    if (threadIdx.x == 0) r = expf(logsig(b[h] + w[h]) - logsig(b[h]));
    __syncthreads();
    for (int c = threadIdx.x; c < hd; c += blockDim.x) v[(size_t) h * hd + c] *= r;
}

// ---------------------------------------------------------------- the gated MLA's output
constexpr int kV = 128;
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
    if (i < n) y[(size_t) blockIdx.y * n + i] += alpha * c[i];
}

// ---------------------------------------------------------------- the prompt path's rows
// x[t] = rms(in[t]) * w (any n), and its FP16
__global__ void __launch_bounds__(256) rms_any_kernel(const float* __restrict__ in, const float* __restrict__ w, float eps,
                                                      int n, float* __restrict__ x, __half* __restrict__ x16) {
    __shared__ float sred[32];
    const size_t o = (size_t) blockIdx.x * n;
    float ss = 0.0f;
    for (int e = threadIdx.x; e < n; e += blockDim.x) ss += in[o + e] * in[o + e];
    ss = block_sum(ss, sred);
    const float inv = rsqrtf(ss / (float) n + eps);
    for (int e = threadIdx.x; e < n; e += blockDim.x) {
        const float v = in[o + e] * inv * w[e];
        x[o + e] = v;
        x16[o + e] = __float2half(v);
    }
}

// the MLA cache rows of T positions: the latent rms-normed (mla_prep's block 0, the same sums), the pe dims as they are
__global__ void mla_rows_kernel(const float* __restrict__ kv, const float* __restrict__ kv_norm, float eps,
                                uint16_t* __restrict__ cache) {
    __shared__ float sred[32];
    const int t = blockIdx.x, tid = threadIdx.x;   // (blockDim 512 = kLat)
    const float* r = kv + (size_t) t * (kLat + kPe);
    uint16_t* row = cache + (size_t) t * (kLat + kPe);
    const float x = r[tid];
    const float ss = block_sum(x * x, sred);
    const float inv = rsqrtf(ss / (float) kLat + eps);
    row[tid] = __half_as_ushort(__float2half(x * inv * kv_norm[tid]));
    if (tid < kPe) row[kLat + tid] = __half_as_ushort(__float2half(r[kLat + tid]));
}

__global__ void gate_rows_kernel(float* __restrict__ o, const float* __restrict__ gate, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] *= sigm(gate[i]);
}

#if STRATA_RAON_PA
// ---------------------------------------------------------------- the prompt's MLA attention (tensor cores)
// ling_fast's mla_prompt_kernel with the heads in passes of 32 (blockIdx.y: heads h0 .. h0 + 31, the last pass ending
// at n_head - a pass overlapping the one before writes the same values), q_abs in and ctx out apart, no gate
namespace wm = nvcuda::wmma;
constexpr int kRow = kLat + kPe, kNope = 128, kQHead = kNope + kPe;
constexpr int kPaH = 32, kPaK = 32;
constexpr int kPaLd = kRow + 8, kPaSld = kPaK + 4, kPaPld = kPaK + 8;
constexpr size_t kPaSmem = (size_t) 2 * kPaH * kPaLd * 2 + (size_t) 2 * kPaH * kPaSld * 4 + (size_t) kPaH * kPaPld * 2 +
                           (size_t) 3 * kPaH * 4;
__device__ __forceinline__ int acc_row(int lane, int e) { return (lane >> 2) + 8 * ((e >> 1) & 1); }
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__global__ void layout_probe_kernel(int* bad) {
    __shared__ float m[16 * 16];
    for (int i = threadIdx.x; i < 256; i += 32) m[i] = (float) (i / 16);
    __syncwarp();
    wm::fragment<wm::accumulator, 16, 16, 16, float> f;
    wm::load_matrix_sync(f, m, 16, wm::mem_row_major);
    for (int e = 0; e < f.num_elements; ++e)
        if (f.x[e] != (float) acc_row(threadIdx.x, e)) atomicExch(bad, 1);
    if (f.num_elements != 8) atomicExch(bad, 1);
}

__global__ void __launch_bounds__(256) mla_prompt_kernel(const float* __restrict__ qabs, float* __restrict__ ctx,
                                                         const float* __restrict__ q, int ldq,
                                                         const uint16_t* __restrict__ cache, int64_t p0, int n_head,
                                                         float scale) {
    extern __shared__ __align__(16) unsigned char smem[];
    __half* sQ = (__half*) smem;
    __half* sK = sQ + kPaH * kPaLd;
    float* sS = (float*) (sK + kPaH * kPaLd);
    __half* sP = (__half*) (sS + 2 * kPaH * kPaSld);
    float* sm = (float*) (sP + kPaH * kPaPld);
    float* sl = sm + kPaH;
    float* sa = sl + kPaH;
    const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int h0 = min((int) blockIdx.y * kPaH, n_head - kPaH);
    const int64_t p = p0 + t;
    const float* qin = qabs + ((size_t) t * n_head + h0) * kLat;
    float* qout = ctx + ((size_t) t * n_head + h0) * kLat;
    for (int i = tid; i < kPaH * kRow; i += 256) {
        const int h = i / kRow, d = i - h * kRow;
        const float v = d < kLat ? qin[(size_t) h * kLat + d]
                                 : q[(size_t) t * ldq + (size_t) (h0 + h) * kQHead + kNope + (d - kLat)];
        sQ[h * kPaLd + d] = __float2half(v * scale);
    }
    if (tid < kPaH) { sm[tid] = -INFINITY; sl[tid] = 0.0f; }
    wm::fragment<wm::accumulator, 16, 16, 16, float> o[2][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) wm::fill_fragment(o[r][c], 0.0f);
    for (int64_t j0 = 0; j0 <= p; j0 += kPaK) {
        __syncthreads();
        for (int i = tid; i < kPaK * (kRow / 8); i += 256) {
            const int r = i / (kRow / 8), c8 = i - r * (kRow / 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (j0 + r <= p) v = *(const uint4*) (cache + (size_t) (j0 + r) * kRow + c8 * 8);
            *(uint4*) (sK + r * kPaLd + c8 * 8) = v;
        }
        __syncthreads();
        {
            const int f = warp & 3, kh = warp >> 2, rt = f >> 1, ct = f & 1;
            wm::fragment<wm::accumulator, 16, 16, 16, float> acc;
            wm::fill_fragment(acc, 0.0f);
            for (int ks = kh * (kRow / 32); ks < (kh + 1) * (kRow / 32); ++ks) {
                wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> a;
                wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> b;
                wm::load_matrix_sync(a, sQ + rt * 16 * kPaLd + ks * 16, kPaLd);
                wm::load_matrix_sync(b, sK + ct * 16 * kPaLd + ks * 16, kPaLd);
                wm::mma_sync(acc, a, b, acc);
            }
            wm::store_matrix_sync(sS + kh * kPaH * kPaSld + rt * 16 * kPaSld + ct * 16, acc, kPaSld, wm::mem_row_major);
        }
        __syncthreads();
        for (int rr = 0; rr < 4; ++rr) {
            const int r = warp * 4 + rr;
            float sv = sS[r * kPaSld + lane] + sS[kPaH * kPaSld + r * kPaSld + lane];
            if (j0 + lane > p) sv = -INFINITY;
            const float mx = warp_max(sv);
            const float m_old = sm[r], m_new = fmaxf(m_old, mx);
            const float e = __expf(sv - m_new);
            const float sum = warp_sum(e);
            sP[r * kPaPld + lane] = __float2half(e);
            __syncwarp();
            if (lane == 0) {
                const float al = __expf(m_old - m_new);
                sl[r] = sl[r] * al + sum;
                sm[r] = m_new;
                sa[r] = al;
            }
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int c = 0; c < 4; ++c)
#pragma unroll
                for (int e = 0; e < 8; ++e) o[r][c].x[e] *= sa[r * 16 + acc_row(lane, e)];
#pragma unroll
        for (int ks = 0; ks < kPaK / 16; ++ks) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> a;
                wm::load_matrix_sync(a, sP + r * 16 * kPaPld + ks * 16, kPaPld);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::row_major> b;
                    wm::load_matrix_sync(b, sK + ks * 16 * kPaLd + (warp * 4 + c) * 16, kPaLd);
                    wm::mma_sync(o[r][c], a, b, o[r][c]);
                }
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
#pragma unroll
            for (int e = 0; e < 8; ++e) o[r][c].x[e] *= 1.0f / sl[r * 16 + acc_row(lane, e)];
            wm::store_matrix_sync(qout + (size_t) r * 16 * kLat + (warp * 4 + c) * 16, o[r][c], kLat, wm::mem_row_major);
        }
}
#endif

}  // namespace

bool prompt_attn_ok() {
#if STRATA_RAON_PA
    static const int ok = [] {
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess || major < 8)
            return 0;
        if (cudaFuncSetAttribute(mla_prompt_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kPaSmem) !=
            cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        int* d = nullptr;
        int bad = 1;
        if (cudaMalloc(&d, 4) == cudaSuccess && cudaMemset(d, 0, 4) == cudaSuccess) {
            layout_probe_kernel<<<1, 32>>>(d);
            if (cudaMemcpy(&bad, d, 4, cudaMemcpyDeviceToHost) != cudaSuccess) bad = 1;
        }
        if (d != nullptr) cudaFree(d);
        cudaGetLastError();
        if (bad) std::fprintf(stderr, "raon: the WMMA accumulator layout is not the expected one - the prompt's MLA a row at a time\n");
        return bad ? 0 : 1;
    }();
    return ok != 0;
#else
    return false;
#endif
}

void mla_prompt_attn(const float* qabs, float* ctx, const float* q, int ldq, const uint16_t* cache, int64_t p0, int T,
                     int n_head, float scale, cudaStream_t s) {
#if STRATA_RAON_PA
    if (T <= 0 || n_head < kPaH) { g_errors.fetch_add(T > 0 ? 1 : 0); return; }
    mla_prompt_kernel<<<dim3(T, (n_head + kPaH - 1) / kPaH), 256, kPaSmem, s>>>(qabs, ctx, q, ldq, cache, p0, n_head, scale);
    check("mla_prompt_attn");
#else
    (void) qabs; (void) ctx; (void) q; (void) ldq; (void) cache; (void) p0; (void) T; (void) n_head; (void) scale; (void) s;
    g_errors.fetch_add(1);
#endif
}

void mhc_pre(const float* R, const uint16_t* fn, const float* base, const float* scale, const float* norm_w,
             float hc_eps, float norm_eps, float post_factor, int n_embd, float* z, float* coef, float* x,
             cudaStream_t s, int rows) {
    if (rows < 1) return;
    mhc_z_kernel<<<dim3(kSel + 1, rows), 256, 0, s>>>(R, fn, kStreams * n_embd, z);
    mhc_coef_kernel<<<rows, 1024, 0, s>>>(R, z, base, scale, norm_w, hc_eps, norm_eps, post_factor, n_embd, coef, x);
    check("mhc_pre");
}

void mhc_post(float* R, const float* out, const float* coef, int n_embd, cudaStream_t s, int rows) {
    if (rows < 1) return;
    mhc_post_kernel<<<dim3((n_embd + 255) / 256, rows), 256, 0, s>>>(R, out, coef, n_embd);
    check("mhc_post");
}

void rda_ratio(float* v, const float* b, const float* w, int n_head, int head_dim, cudaStream_t s, int rows) {
    if (rows < 1) return;
    rda_ratio_kernel<<<dim3(n_head, rows), 128, 0, s>>>(v, b, w, head_dim);
    check("rda_ratio");
}

void rms_rows_any(const float* in, const float* w, float eps, int rows, int n, float* x, void* x16, cudaStream_t s) {
    if (rows < 1) return;
    rms_any_kernel<<<rows, 256, 0, s>>>(in, w, eps, n, x, (__half*) x16);
    check("rms_rows_any");
}

void mla_cache_rows(const float* kv, const float* kv_norm, float eps, int rows, uint16_t* cache, cudaStream_t s) {
    if (rows < 1) return;
    mla_rows_kernel<<<rows, kLat, 0, s>>>(kv, kv_norm, eps, cache);
    check("mla_cache_rows");
}

void gate_rows(float* o, const float* gate, int64_t n, cudaStream_t s) {
    if (n < 1) return;
    gate_rows_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(o, gate, n);
    check("gate_rows");
}

void mla_out_gated(const float* ctx, const float* v_b, const float* gate, int n_head, float* o, cudaStream_t s) {
    mla_out_gated_kernel<<<dim3(n_head, kV / 8), 256, 0, s>>>(ctx, v_b, gate, o);
    check("mla_out_gated");
}

void axpy(float* y, const float* c, float alpha, int n, cudaStream_t s, int rows) {
    if (rows < 1) return;
    axpy_kernel<<<dim3((n + 255) / 256, rows), 256, 0, s>>>(y, c, alpha, n);
    check("axpy");
}

int launch_errors() { return g_errors.load(); }

}  // namespace strata::kernels::raon
