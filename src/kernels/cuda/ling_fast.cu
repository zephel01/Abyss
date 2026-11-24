// src/kernels/cuda/ling_fast.cu - see include/strata/kernels/ling_fast.hpp.
#include "strata/kernels/ling_fast.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdio>

namespace strata::kernels::ling {
namespace {

std::atomic<int> g_errors{0};
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        g_errors.fetch_add(1);
        std::fprintf(stderr, "ling_fast %s: %s\n", what, cudaGetErrorString(e));
    }
}

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
__device__ __forceinline__ float block_max(float v, float* sred) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sred[warp] = v;
    __syncthreads();
    float t = -INFINITY;
    for (int i = 0; i < nw; ++i) t = fmaxf(t, sred[i]);
    return t;
}
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }

// ---------------------------------------------------------------- KDA
struct KdaArgs {
    const float* proj;
    const float* conv_w[3];
    float* conv_state;
    float* out;
    const float* f;
    const float* dt_bias;
    const float* a;
    float lb;
    float* g1;
    int n_head, head_dim, d_conv;
};

// grid (n_head, 4): roles 0..2 the q / k / v convs of the head's channels, role 3 its gates; a thread a channel
__global__ void kda_prep_kernel(const __grid_constant__ KdaArgs a) {
    __shared__ float sred[32];
    const int h = blockIdx.x, role = blockIdx.y, tid = threadIdx.x;
    const int d_inner = a.n_head * a.head_dim;
    const int c = h * a.head_dim + tid;
    if (role < 3) {
        const int hist = a.d_conv - 1;
        float* st = a.conv_state + (size_t) role * d_inner * hist + (size_t) hist * c;
        const float* w = a.conv_w[role] + (size_t) a.d_conv * c;
        const float xp = a.proj[(size_t) role * d_inner + c];
        float acc = 0.0f;
        for (int tap = 0; tap < hist; ++tap) acc += w[tap] * st[tap];
        acc += w[hist] * xp;
        float y = acc / (1.0f + expf(-acc));
        for (int j = 0; j + 1 < hist; ++j) st[j] = st[j + 1];
        st[hist - 1] = xp;
        if (role < 2) {   // (ggml_l2_norm: x / max(|x|, eps))
            const float ss = block_sum(y * y, sred);
            y = y / fmaxf(sqrtf(ss), 1e-6f);
        }
        a.out[(size_t) role * d_inner + c] = y;
        return;
    }
    const float v = (a.f[c] + a.dt_bias[c]) * a.a[h];
    a.g1[c] = a.lb * (1.0f / (1.0f + expf(-v)));
}

// ---------------------------------------------------------------- MLA
__global__ void rope_table_kernel(int64_t pos, double base, float2* cs) {
    const int j = threadIdx.x;
    if (j >= kRot / 2) return;
    const double th = (double) pos * pow(base, -2.0 * (double) j / (double) kRot);
    cs[j] = make_float2((float) cos(th), (float) sin(th));
}

// block 0: the cache row (the latent's rms norm, the key's rope); block 1 + h: q head h's rope part
__global__ void mla_prep_kernel(float* q, const float* __restrict__ kv, const float* __restrict__ kv_norm, float eps,
                                const float2* __restrict__ cs, uint16_t* __restrict__ row) {
    __shared__ float sred[32];
    const int tid = threadIdx.x;
    if (blockIdx.x == 0) {
        const float x = kv[tid];   // (blockDim 512 = kLat)
        const float ss = block_sum(x * x, sred);
        const float inv = rsqrtf(ss / (float) kLat + eps);
        row[tid] = f2h(x * inv * kv_norm[tid]);
        if (tid < kRot / 2) {
            const float2 c = cs[tid];
            const float x0 = kv[kLat + 2 * tid], x1 = kv[kLat + 2 * tid + 1];
            row[kLat + 2 * tid] = f2h(x0 * c.x - x1 * c.y);
            row[kLat + 2 * tid + 1] = f2h(x0 * c.y + x1 * c.x);
        }
        return;
    }
    if (tid >= kRot / 2) return;
    float* qr = q + (size_t) (blockIdx.x - 1) * kQHead + kNope;
    const float2 c = cs[tid];
    const float x0 = qr[2 * tid], x1 = qr[2 * tid + 1];
    qr[2 * tid] = x0 * c.x - x1 * c.y;
    qr[2 * tid + 1] = x0 * c.y + x1 * c.x;
}

// grid (n_head, kLat / 8), 8 warps: a warp a row c of k_b
__global__ void mla_absorb_kernel(const float* __restrict__ q, const float* __restrict__ k_b, float* __restrict__ qabs) {
    const int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c = blockIdx.y * 8 + warp;
    const float4 w = reinterpret_cast<const float4*>(k_b + ((size_t) h * kLat + c) * kNope)[lane];
    const float4 x = reinterpret_cast<const float4*>(q + (size_t) h * kQHead)[lane];
    const float d = warp_sum(w.x * x.x + w.y * x.y + w.z * x.z + w.w * x.w);
    if (lane == 0) qabs[(size_t) h * kLat + c] = d;
}

// the scores: grid (ceil(n / 32), n_head / 8), 8 warps of 4 positions; the 8 heads' query rows (q_abs, then q_rot) in
// shared memory; a lane takes 9 of a row's 288 half pairs
constexpr int kScHeads = 8;
__global__ void mla_scores_kernel(const float* __restrict__ qabs, const float* __restrict__ q,
                                  const uint16_t* __restrict__ cache, int64_t n, float scale, float* __restrict__ sc) {
    __shared__ float sq[kScHeads][kRow];
    const int h0 = blockIdx.y * kScHeads, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int i = threadIdx.x; i < kScHeads * kRow; i += blockDim.x) {
        const int hh = i / kRow, d = i % kRow;
        sq[hh][d] = d < kLat ? qabs[(size_t) (h0 + hh) * kLat + d] : q[(size_t) (h0 + hh) * kQHead + kNope + (d - kLat)];
    }
    __syncthreads();
    for (int i = 0; i < 4; ++i) {
        const int64_t t = (int64_t) blockIdx.x * 32 + warp * 4 + i;
        if (t >= n) break;
        const __half2* r = reinterpret_cast<const __half2*>(cache + (size_t) t * kRow);
        float acc[kScHeads];
#pragma unroll
        for (int hh = 0; hh < kScHeads; ++hh) acc[hh] = 0.0f;
#pragma unroll
        for (int m = 0; m < kRow / 64; ++m) {
            const int k = lane + 32 * m;
            const float2 v = __half22float2(r[k]);
#pragma unroll
            for (int hh = 0; hh < kScHeads; ++hh) acc[hh] += sq[hh][2 * k] * v.x + sq[hh][2 * k + 1] * v.y;
        }
#pragma unroll
        for (int hh = 0; hh < kScHeads; ++hh) {
            const float s = warp_sum(acc[hh]);
            if (lane == 0) sc[(size_t) (h0 + hh) * n + t] = s * scale;
        }
    }
}

// the softmax of a head's row in place: grid n_head, 256 threads
__global__ void mla_softmax_kernel(float* sc, int64_t n) {
    __shared__ float sred[32];
    float* r = sc + (size_t) blockIdx.x * n;
    float mx = -INFINITY;
    for (int64_t t = threadIdx.x; t < n; t += blockDim.x) mx = fmaxf(mx, r[t]);
    mx = block_max(mx, sred);
    float sum = 0.0f;
    for (int64_t t = threadIdx.x; t < n; t += blockDim.x) {
        const float e = expf(r[t] - mx);
        r[t] = e;
        sum += e;
    }
    sum = block_sum(sum, sred);
    const float inv = 1.0f / sum;
    for (int64_t t = threadIdx.x; t < n; t += blockDim.x) r[t] *= inv;
}

// a chunk's context partials: grid (chunks, ceil(n_head / 32)), 256 threads a pair of latent dims each; the chunk's
// probabilities of 32 heads in shared memory
constexpr int kCtxHeads = 32;
__global__ void __launch_bounds__(256) mla_ctx_kernel(const float* __restrict__ p, const uint16_t* __restrict__ cache,
                                                      int64_t n, int n_head, float* __restrict__ part) {
    __shared__ float sp[kCtxHeads][kChunk];
    const int64_t t0 = (int64_t) blockIdx.x * kChunk;
    const int h0 = blockIdx.y * kCtxHeads, nh = min(kCtxHeads, n_head - h0);
    const int nt = (int) (n - t0 < (int64_t) kChunk ? n - t0 : (int64_t) kChunk);
    for (int i = threadIdx.x; i < kCtxHeads * kChunk; i += blockDim.x) {
        const int hh = i / kChunk, tt = i % kChunk;
        sp[hh][tt] = hh < nh && tt < nt ? p[(size_t) (h0 + hh) * n + t0 + tt] : 0.0f;
    }
    __syncthreads();
    float ax[kCtxHeads], ay[kCtxHeads];
#pragma unroll
    for (int hh = 0; hh < kCtxHeads; ++hh) ax[hh] = ay[hh] = 0.0f;
    const int k = threadIdx.x;   // half pair k: dims 2k, 2k + 1
    for (int tt = 0; tt < nt; ++tt) {
        const float2 v = __half22float2(reinterpret_cast<const __half2*>(cache + (size_t) (t0 + tt) * kRow)[k]);
#pragma unroll
        for (int hh = 0; hh < kCtxHeads; ++hh) {
            ax[hh] += sp[hh][tt] * v.x;
            ay[hh] += sp[hh][tt] * v.y;
        }
    }
    for (int hh = 0; hh < nh; ++hh) {
        float* o = part + ((size_t) blockIdx.x * n_head + h0 + hh) * kLat;
        o[2 * k] = ax[hh];
        o[2 * k + 1] = ay[hh];
    }
}

// the chunks' partials summed in chunk order: grid n_head, kLat threads
__global__ void mla_ctx_sum_kernel(const float* __restrict__ part, int chunks, int n_head, float* __restrict__ ctx) {
    const int h = blockIdx.x, c = threadIdx.x;
    float s = 0.0f;
    for (int i = 0; i < chunks; ++i) s += part[((size_t) i * n_head + h) * kLat + c];
    ctx[(size_t) h * kLat + c] = s;
}

// grid (n_head, kV / 8), 8 warps: a warp a row v of v_b
__global__ void mla_out_kernel(const float* __restrict__ ctx, const float* __restrict__ v_b,
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
    if (lane == 0) o[(size_t) h * kV + v] = d * (1.0f / (1.0f + expf(-gate[h])));
}

// ---------------------------------------------------------------- the routed experts' groups
__global__ void group_bias_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int E, int G, int TG,
                                  float* __restrict__ out) {
    __shared__ float s[512];
    __shared__ float gs[32];
    __shared__ int keep[32];
    for (int e = threadIdx.x; e < E; e += blockDim.x)
        s[e] = 1.0f / (1.0f + expf(-logits[e])) + (bias != nullptr ? bias[e] : 0.0f);
    __syncthreads();
    const int per = E / G;
    if (threadIdx.x < G) {   // a group's score: its best two
        float b1 = -INFINITY, b2 = -INFINITY;
        for (int i = 0; i < per; ++i) {
            const float v = s[threadIdx.x * per + i];
            if (v > b1) {
                b2 = b1;
                b1 = v;
            } else if (v > b2) {
                b2 = v;
            }
        }
        gs[threadIdx.x] = b1 + b2;
        keep[threadIdx.x] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0)
        for (int r = 0; r < TG; ++r) {   // the best TG groups (ties: the lower index)
            int best = -1;
            for (int gi = 0; gi < G; ++gi)
                if (!keep[gi] && (best < 0 || gs[gi] > gs[best])) best = gi;
            keep[best] = 1;
        }
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x)
        out[e] = keep[e / per] ? (bias != nullptr ? bias[e] : 0.0f) : -1e30f;
}

}  // namespace

void kda_prep(const float* proj, const float* const conv_w[3], float* conv_state, float* out, const float* f,
              const float* dt_bias, const float* a, float lb, float* g1, int n_head, int head_dim, int d_conv,
              cudaStream_t s) {
    KdaArgs k{proj, {conv_w[0], conv_w[1], conv_w[2]}, conv_state, out, f, dt_bias, a, lb, g1, n_head, head_dim, d_conv};
    kda_prep_kernel<<<dim3(n_head, 4), head_dim, 0, s>>>(k);
    check("kda_prep");
}

void rope_table(int64_t pos, double base, float2* cs, cudaStream_t s) {
    rope_table_kernel<<<1, 32, 0, s>>>(pos, base, cs);
    check("rope_table");
}

void mla_prep(float* q, int n_head, const float* kv, const float* kv_norm, float eps, const float2* cs,
              uint16_t* cache_row, cudaStream_t s) {
    mla_prep_kernel<<<1 + n_head, kLat, 0, s>>>(q, kv, kv_norm, eps, cs, cache_row);
    check("mla_prep");
}

void mla_absorb(const float* q, const float* k_b, int n_head, float* qabs, cudaStream_t s) {
    mla_absorb_kernel<<<dim3(n_head, kLat / 8), 256, 0, s>>>(q, k_b, qabs);
    check("mla_absorb");
}

void mla_attend(const float* qabs, const float* q, const uint16_t* cache, int64_t n, float scale, int n_head,
                float* sc, float* part, float* ctx, cudaStream_t s) {
    mla_scores_kernel<<<dim3((unsigned) ((n + 31) / 32), n_head / kScHeads), 256, 0, s>>>(qabs, q, cache, n, scale, sc);
    mla_softmax_kernel<<<n_head, 256, 0, s>>>(sc, n);
    const int chunks = (int) ((n + kChunk - 1) / kChunk);
    mla_ctx_kernel<<<dim3(chunks, (n_head + kCtxHeads - 1) / kCtxHeads), 256, 0, s>>>(sc, cache, n, n_head, part);
    mla_ctx_sum_kernel<<<n_head, kLat, 0, s>>>(part, chunks, n_head, ctx);
    check("mla_attend");
}

void mla_out(const float* ctx, const float* v_b, const float* gate, int n_head, float* o, cudaStream_t s) {
    mla_out_kernel<<<dim3(n_head, kV / 8), 256, 0, s>>>(ctx, v_b, gate, o);
    check("mla_out");
}

void group_bias(const float* logits, const float* bias, int n_expert, int n_group, int topk_group, float* out,
                cudaStream_t s) {
    group_bias_kernel<<<1, 512, 0, s>>>(logits, bias, n_expert, n_group, topk_group, out);
    check("group_bias");
}

int launch_errors() { return g_errors.load(); }

}  // namespace strata::kernels::ling
