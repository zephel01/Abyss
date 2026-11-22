// src/kernels/cuda/inkling_fast.cu - see include/strata/kernels/inkling_fast.hpp.
#include "strata/kernels/inkling_fast.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels::ink {
namespace {

constexpr int kD = kHeadDim;

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

// a block's sum of one value a thread (blockDim.x a multiple of 32, <= 1024); every thread gets it
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (int) blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    float t = lane < nw ? sh[lane] : 0.0f;
    t = warp_sum(t);
    return t;
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "inkling kernels: %s: %s\n", what, cudaGetErrorString(e));
}

// ---------------------------------------------------------------- norms
__global__ void rms_rows_kernel(const float* __restrict__ x, int64_t xs, const float* __restrict__ gamma, float eps,
                                int n, float* __restrict__ y, int64_t ys) {
    __shared__ float sh[32];
    const float* xr = x + (int64_t) blockIdx.x * xs;
    float* yr = y + (int64_t) blockIdx.x * ys;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = gamma != nullptr ? xr[i] * inv * gamma[i] : xr[i] * inv;
}

// ---------------------------------------------------------------- the depth attention
// two kernels over chunks of kMixChunk elements (grid (chunks, T)): the chunk's sums of squares of the states and the
// queries' dots with them (part[t][chunk][kMixSums]), then every chunk block adds the chunks' partials, takes the
// softmax and writes its chunk of the weighted sum
constexpr int kMixChunk = 256;
constexpr int kMixSums = kMaxStates * 5;

__global__ void __launch_bounds__(kMixChunk) mix_part_kernel(States st, const float* __restrict__ query, int nq, int E,
                                                             float* __restrict__ part) {
    __shared__ float wp[kMixChunk / 32][kMixSums];
    const int t = blockIdx.y, c = blockIdx.x, S = st.n;
    const int i = c * kMixChunk + threadIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float acc[kMixSums];
#pragma unroll
    for (int k = 0; k < kMixSums; ++k) acc[k] = 0.0f;
    if (i < E) {
        float qv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) qv[q] = q < nq ? query[(size_t) q * E + i] : 0.0f;
#pragma unroll
        for (int s = 0; s < kMaxStates; ++s) {
            if (s >= S) break;
            const float v = st.p[s][(int64_t) t * st.stride + i];
            acc[s] = v * v;
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[kMaxStates + q * kMaxStates + s] = qv[q] * v;
        }
    }
#pragma unroll
    for (int k = 0; k < kMixSums; ++k) {
        const float v = warp_sum(acc[k]);
        if (lane == 0) wp[warp][k] = v;
    }
    __syncthreads();
    if (threadIdx.x < kMixSums) {
        float a = 0.0f;
#pragma unroll
        for (int w = 0; w < kMixChunk / 32; ++w) a += wp[w][threadIdx.x];
        part[((size_t) t * gridDim.x + c) * kMixSums + threadIdx.x] = a;
    }
}

__global__ void __launch_bounds__(kMixChunk) mix_out_kernel(States st, const float* __restrict__ part, int nq, int E,
                                                            float* __restrict__ out) {
    __shared__ float red[kMixSums];
    __shared__ float w[4][kMaxStates];
    __shared__ float inv[kMaxStates];
    const int t = blockIdx.y, c = blockIdx.x, S = st.n, nc = gridDim.x;
    if (threadIdx.x < kMixSums) {
        float a = 0.0f;
        for (int k = 0; k < nc; ++k) a += part[((size_t) t * nc + k) * kMixSums + threadIdx.x];
        red[threadIdx.x] = a;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float sc = 1.0f / sqrtf((float) E);
        for (int s = 0; s < S; ++s) inv[s] = rsqrtf(red[s] / (float) E + 1e-6f);
        for (int q = 0; q < nq; ++q) {
            float m = -INFINITY;
            float l[kMaxStates];
            for (int s = 0; s < S; ++s) {
                l[s] = red[kMaxStates + q * kMaxStates + s] * inv[s] * sc;
                m = fmaxf(m, l[s]);
            }
            float z = 0.0f;
            for (int s = 0; s < S; ++s) {
                l[s] = expf(l[s] - m);
                z += l[s];
            }
            for (int s = 0; s < S; ++s) w[q][s] = l[s] / z;
        }
    }
    __syncthreads();
    const int i = c * kMixChunk + threadIdx.x;
    if (i >= E) return;
    float v[kMaxStates];
#pragma unroll
    for (int s = 0; s < kMaxStates; ++s) v[s] = s < S ? st.p[s][(int64_t) t * st.stride + i] * inv[s] : 0.0f;
    for (int q = 0; q < nq; ++q) {
        float o = 0.0f;
#pragma unroll
        for (int s = 0; s < kMaxStates; ++s)
            if (s < S) o += w[q][s] * v[s];
        out[((int64_t) t * nq + q) * E + i] = o;
    }
}

// the partials' scratch (grown on demand; the engine runs one stream a device here)
float* mix_scratch(size_t floats) {
    static float* p = nullptr;
    static size_t have = 0;
    if (floats > have) {
        if (p != nullptr) cudaFree(p);
        p = nullptr;
        have = 0;
        if (cudaMalloc(&p, floats * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        have = floats;
    }
    return p;
}

// ---------------------------------------------------------------- the loop controller
__global__ void ctrl_apply_kernel(const float* __restrict__ cur, int64_t cs, const float* __restrict__ msg, int64_t ms,
                                  const float* __restrict__ g, const float* __restrict__ f, int n, float* __restrict__ out,
                                  int64_t os) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float c = cur[(int64_t) t * cs + i], m = msg[(int64_t) t * ms + i];
    const float fv = tanhf(f[(int64_t) t * n + i]), gv = tanhf(g[(int64_t) t * n + i]);
    out[(int64_t) t * os + i] = (c - fv * c) + gv * m;
}

// ---------------------------------------------------------------- the short convolutions
// grid (C / 128, T): row t from the inputs t - 3 .. t (the state's for t < 3), as conv_kernel's step
__global__ void conv_par_kernel(const float* __restrict__ in, int64_t is, const float* __restrict__ w,
                                const float* __restrict__ dw, int64_t dws, int head_ch, const float* __restrict__ state,
                                int C, float* __restrict__ out, int64_t os) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= C) return;
    float xv[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int tt = t - 3 + k;
        xv[k] = tt >= 0 ? in[(int64_t) tt * is + c] : state[(size_t) c * 3 + (3 + tt)];
    }
    const float x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
    const float w0 = w[(size_t) c * 4 + 0], w1 = w[(size_t) c * 4 + 1], w2 = w[(size_t) c * 4 + 2], w3 = w[(size_t) c * 4 + 3];
    const float scale = 1.0f / sqrtf(dw != nullptr ? 9.0f : 5.0f);
    float o = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
    if (dw != nullptr) {
        const float* d = dw + (int64_t) t * dws + (size_t) (c / head_ch) * 4;
        o = o + x3 * d[0];
        o = o + x2 * d[1];
        o = o + x1 * d[2];
        o = o + x0 * d[3];
    }
    out[(int64_t) t * os + c] = (x3 + o) * scale;
}
// the state after T rows: the last 3 inputs (the old state's where T < 3)
__global__ void conv_state_kernel(const float* __restrict__ in, int64_t is, float* __restrict__ state, int C, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int tt = T - 3 + k;
        v[k] = tt >= 0 ? in[(int64_t) tt * is + c] : state[(size_t) c * 3 + (3 + tt)];
    }
#pragma unroll
    for (int k = 0; k < 3; ++k) state[(size_t) c * 3 + k] = v[k];
}

// conv_kernel's channel c over T rows in order (the decode's fused post_proj_kernel runs the same function)
__device__ __forceinline__ void conv_channel(const float* __restrict__ in, int64_t is, const float* __restrict__ w,
                                             const float* __restrict__ dw, int64_t dws, int head_ch,
                                             float* __restrict__ state, int c, int T, float* __restrict__ out,
                                             int64_t os) {
    float x0 = state[(size_t) c * 3 + 0], x1 = state[(size_t) c * 3 + 1], x2 = state[(size_t) c * 3 + 2];
    const float w0 = w[(size_t) c * 4 + 0], w1 = w[(size_t) c * 4 + 1], w2 = w[(size_t) c * 4 + 2], w3 = w[(size_t) c * 4 + 3];
    const float scale = 1.0f / sqrtf(dw != nullptr ? 9.0f : 5.0f);
    const int h = c / head_ch;
    for (int t = 0; t < T; ++t) {
        const float x3 = in[(int64_t) t * is + c];
        float o = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
        if (dw != nullptr) {
            const float* d = dw + (int64_t) t * dws + (size_t) h * 4;
            o = o + x3 * d[0];
            o = o + x2 * d[1];
            o = o + x1 * d[2];
            o = o + x0 * d[3];
        }
        out[(int64_t) t * os + c] = (x3 + o) * scale;
        x0 = x1;
        x1 = x2;
        x2 = x3;
    }
    state[(size_t) c * 3 + 0] = x0;
    state[(size_t) c * 3 + 1] = x1;
    state[(size_t) c * 3 + 2] = x2;
}
__global__ void conv_kernel(const float* __restrict__ in, int64_t is, const float* __restrict__ w,
                            const float* __restrict__ dw, int64_t dws, int head_ch, float* __restrict__ state, int C, int T,
                            float* __restrict__ out, int64_t os) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    conv_channel(in, is, w, dw, dws, head_ch, state, c, T, out, os);
}

// ---------------------------------------------------------------- clips
// a warp a (row, head)
__device__ __forceinline__ void clip_head(float* __restrict__ x, int64_t xs, const float* __restrict__ scale, int dim,
                                          float thr, int64_t p0, float log_floor, float alpha, int t, int h, int lane) {
    float* r = x + (int64_t) t * xs + (size_t) h * dim;
    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int d = lane + 32 * k;
        v[k] = 0.0f;
        if (d < dim) {
            v[k] = r[d];
            if (scale != nullptr) v[k] = v[k] * scale[(size_t) h * dim + d];
            ss += v[k] * v[k];
        }
    }
    ss = warp_sum(ss);
    const float nrm = sqrtf(ss) + 1e-6f;
    const float fac = fminf(fmaxf(thr / nrm, 0.0f), 1.0f);
    float tau = 1.0f;
    if (log_floor > 0.0f) {
        const float pf = (float) (p0 + t) + 1.0f;
        tau = alpha * logf(fmaxf(pf * (1.0f / log_floor), 1.0f)) + 1.0f;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int d = lane + 32 * k;
        if (d < dim) {
            float o = v[k] * fac;
            if (log_floor > 0.0f) o = o * tau;
            r[d] = o;
        }
    }
}
__global__ void clip_kernel(float* __restrict__ x, int64_t xs, const float* __restrict__ scale, int dim, int H, float thr,
                            int64_t p0, float log_floor, float alpha, int T) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (wid >= T * H) return;
    clip_head(x, xs, scale, dim, thr, p0, log_floor, alpha, wid / H, wid % H, lane);
}

// ---------------------------------------------------------------- the KV rows
// a warp a (row, KV head): k as it is, v normalised, both FP16
__device__ __forceinline__ void kv_store_head(const float* __restrict__ k, int64_t ks, const float* __restrict__ v,
                                              int64_t vs, float eps, int n_head_kv, int64_t p0, int ring,
                                              __half* __restrict__ kc, __half* __restrict__ vc, int t, int h, int lane) {
    const float* kr = k + (int64_t) t * ks + (size_t) h * kD;
    const float* vr = v + (int64_t) t * vs + (size_t) h * kD;
    float vv[4];
    float ss = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        vv[q] = vr[lane * 4 + q];
        ss += vv[q] * vv[q];
    }
    ss = warp_sum(ss);
    const float inv = rsqrtf(ss / (float) kD + eps);
    const size_t row = ((size_t) ((p0 + t) % ring) * n_head_kv + h) * kD;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        kc[row + lane * 4 + q] = __float2half(kr[lane * 4 + q]);
        vc[row + lane * 4 + q] = __float2half(vv[q] * inv);
    }
}
__global__ void kv_store_kernel(const float* __restrict__ k, int64_t ks, const float* __restrict__ v, int64_t vs, float eps,
                                int n_head_kv, int64_t p0, int T, int ring, __half* __restrict__ kc,
                                __half* __restrict__ vc) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (wid >= T * n_head_kv) return;
    kv_store_head(k, ks, v, vs, eps, n_head_kv, p0, ring, kc, vc, wid / n_head_kv, wid % n_head_kv, lane);
}

// a decode token's conv_rows (k, v) + clip_heads (q, k, rel) + kv_store in one launch, the same functions as the
// separate kernels (the same bits): a warp a KV head (its k / v channels' convs, k's clip, the cache rows), a q head's
// clip or a rel head's clip
__global__ void post_proj_kernel(PostProj a) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (wid < a.Hkv) {
        const int h = wid;
        for (int d = lane; d < kD; d += 32) {
            const int c = h * kD + d;
            conv_channel(a.kf, a.KV, a.k_conv, a.kd, a.dws, kD, a.conv, c, 1, a.kf, a.KV);
            conv_channel(a.vf, a.KV, a.v_conv, a.vd, a.dws, kD, a.conv + (size_t) a.KV * 3, c, 1, a.vf, a.KV);
        }
        __syncwarp();
        clip_head(a.kf, a.KV, nullptr, kD, a.qk_clip, a.p, 0.0f, 0.0f, 0, h, lane);
        __syncwarp();
        kv_store_head(a.kf, a.KV, a.vf, a.KV, a.eps, a.Hkv, a.p, a.ring, (__half*) a.kc, (__half*) a.vc, 0, h, lane);
    } else if (wid < a.Hkv + a.H) {
        clip_head(a.qf, a.HD, a.q_clip, kD, a.qk_clip, a.p, a.lf, a.alpha, 0, wid - a.Hkv, lane);
    } else if (wid < a.Hkv + 2 * a.H) {
        clip_head(a.rf, (int64_t) a.H * kRel, a.rel_clip, kRel, a.rel_clip_thr, a.p, a.lf, a.alpha, 0, wid - a.Hkv - a.H,
                  lane);
    }
}

// ---------------------------------------------------------------- attention
constexpr int kAttnWarps = 4;
constexpr int kTile = 32 * kAttnWarps;   // keys a tile: a warp's 32
constexpr int kMaxG = 8;                 // q heads a KV head at most

// the part rows: [T][H][max_splits] of (m, l, acc[128])
constexpr int kPart = kD + 2;

// grid (splits, n_head_kv, T): block (split, kv head, row) runs the G q heads of the KV head over its key range
template <int kG>
__global__ void __launch_bounds__(kAttnWarps * 32) attn_part_kernel(
    const float* __restrict__ q, int64_t qs, const float* __restrict__ r, int64_t rs, const float* __restrict__ relb,
    int extent, const __half* __restrict__ kc, const __half* __restrict__ vc, int ring, int H, int Hkv, int64_t p0,
    int window, float scale, float* __restrict__ part, int max_splits, int f16) {
    const int split = blockIdx.x, kvh = blockIdx.y, t = blockIdx.z;
    const int nsplit = gridDim.x;
    constexpr int G = kG;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t p = p0 + t;
    const int64_t lo = window > 0 ? (p - window + 1 > 0 ? p - window + 1 : (int64_t) 0) : 0;
    const int64_t nk = p - lo + 1;
    const int64_t per = (nk + nsplit - 1) / nsplit;
    const int64_t k0 = lo + (int64_t) split * per, k1 = (p + 1 < k0 + per ? p + 1 : k0 + per);
    __shared__ float s_r[kG][kRel];
    __shared__ float s_b[kG][kTile];
    __shared__ float s_m[kAttnWarps][kG], s_l[kAttnWarps][kG];
    __shared__ float s_acc[kAttnWarps][kG][kD];
    __shared__ float s_q[kG][kD];
    for (int i = threadIdx.x; i < G * kRel; i += blockDim.x)
        s_r[i / kRel][i % kRel] = r[(int64_t) t * rs + (size_t) (kvh * G + i / kRel) * kRel + i % kRel];
    float qv[kG][4];
#pragma unroll
    for (int g = 0; g < kG; ++g)
#pragma unroll
        for (int c = 0; c < 4; ++c)
            qv[g][c] = g < G ? q[(int64_t) t * qs + (size_t) (kvh * G + g) * kD + lane * 4 + c] * scale : 0.0f;
    if (f16 & 2)
#pragma unroll
        for (int g = 0; g < kG; ++g)
#pragma unroll
            for (int c = 0; c < 4; ++c) qv[g][c] = __half2float(__float2half(qv[g][c]));
    if (warp == 0)
#pragma unroll
        for (int g = 0; g < kG; ++g)
#pragma unroll
            for (int c = 0; c < 4; ++c) s_q[g][lane * 4 + c] = qv[g][c];
    float m[kG], l[kG], acc[kG][4];
#pragma unroll
    for (int g = 0; g < kG; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[g][c] = 0.0f;
    }
    __syncthreads();
    for (int64_t ks = k0; ks < k1; ks += kTile) {
        {   // the tile's biases: thread i the key ks + i
            const int64_t j = ks + threadIdx.x;
            const int64_t d = p - j;
            for (int g = 0; g < G; ++g) {
                float b = 0.0f;
                if (j < k1 && d < extent) {
                    const float* rb = relb + (size_t) d * kRel;
#pragma unroll
                    for (int e = 0; e < kRel; ++e) b += s_r[g][e] * rb[e];
                }
                s_b[g][threadIdx.x] = (f16 & 1) ? __half2float(__float2half(b)) : b;
            }
        }
        __syncthreads();
        // a lane a key: its logits for the G heads (q from shared memory), the warp's online-softmax step, then
        // the 32 keys' values (a lane 4 dims) weighted by their probabilities
        const int64_t w0 = ks + (int64_t) warp * 32;
        if (w0 < k1) {
            const int64_t j = w0 + lane;
            const bool ok = j < k1;
            float lg[kG];
#pragma unroll
            for (int g = 0; g < kG; ++g) lg[g] = 0.0f;
            if (ok) {
                const uint4* kr = reinterpret_cast<const uint4*>(kc + ((size_t) (j % ring) * Hkv + kvh) * kD);
                uint4 kw[kD / 8];
#pragma unroll
                for (int d8 = 0; d8 < kD / 8; ++d8) kw[d8] = kr[d8];   // (the row's 256 bytes, all loads in flight)
#pragma unroll
                for (int d8 = 0; d8 < kD / 8; ++d8) {
                    const __half2* h2 = reinterpret_cast<const __half2*>(&kw[d8]);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float2 kf = __half22float2(h2[e]);
                        const int d = d8 * 8 + 2 * e;
#pragma unroll
                        for (int g = 0; g < kG; ++g)
                            if (g < G) lg[g] += s_q[g][d] * kf.x + s_q[g][d + 1] * kf.y;
                    }
                }
            }
            float pj[kG];
#pragma unroll
            for (int g = 0; g < kG; ++g) {
                pj[g] = 0.0f;
                if (g >= G) continue;
                const float x = ok ? lg[g] + s_b[g][warp * 32 + lane] : -INFINITY;
                const float mt = warp_max(x);
                const float mn = fmaxf(m[g], mt);
                const float cf = expf(m[g] - mn);
                pj[g] = ok ? expf(x - mn) : 0.0f;
                l[g] = l[g] * cf + warp_sum(pj[g]);
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[g][c] *= cf;
                m[g] = mn;
            }
            const int nkk = (int) (k1 - w0 < 32 ? k1 - w0 : 32);
            for (int kk = 0; kk < nkk; ++kk) {
                const size_t row = ((size_t) ((w0 + kk) % ring) * Hkv + kvh) * kD + lane * 4;
                const float2 va = __half22float2(*reinterpret_cast<const __half2*>(vc + row));
                const float2 vb = __half22float2(*reinterpret_cast<const __half2*>(vc + row + 2));
#pragma unroll
                for (int g = 0; g < kG; ++g) {
                    if (g >= G) break;
                    const float pk = __shfl_sync(0xffffffffu, pj[g], kk);
                    acc[g][0] += pk * va.x;
                    acc[g][1] += pk * va.y;
                    acc[g][2] += pk * vb.x;
                    acc[g][3] += pk * vb.y;
                }
            }
        }
        __syncthreads();
    }
    // the warps' runs into one
    for (int g = 0; g < G; ++g) {
        if (lane == 0) {
            s_m[warp][g] = m[g];
            s_l[warp][g] = l[g];
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) s_acc[warp][g][lane * 4 + c] = acc[g][c];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < G * kD; i += blockDim.x) {
        const int g = i / kD, d = i % kD;
        float M = -INFINITY;
        for (int w = 0; w < kAttnWarps; ++w) M = fmaxf(M, s_m[w][g]);
        float L = 0.0f, A = 0.0f;
        if (M > -INFINITY)
            for (int w = 0; w < kAttnWarps; ++w) {
                const float f = expf(s_m[w][g] - M);
                L += s_l[w][g] * f;
                A += s_acc[w][g][d] * f;
            }
        float* pr = part + (((size_t) t * H + (size_t) (kvh * G + g)) * max_splits + split) * kPart;
        pr[2 + d] = A;
        if (d == 0) {
            pr[0] = M;
            pr[1] = L;
        }
    }
}

// grid (H, T): the splits and the sink into out
__global__ void attn_combine_kernel(const float* __restrict__ part, int nsplit, int max_splits, const float* __restrict__ sink,
                                    int64_t ss, int H, float* __restrict__ out, int64_t os) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const float* pr = part + ((size_t) t * H + h) * max_splits * kPart;
    const float sk = sink[(int64_t) t * ss + h];
    float M = sk;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, pr[(size_t) s * kPart]);
    float L = expf(sk - M), A = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const float ms = pr[(size_t) s * kPart];
        if (ms == -INFINITY) continue;
        const float f = expf(ms - M);
        L += pr[(size_t) s * kPart + 1] * f;
        A += pr[(size_t) s * kPart + 2 + d] * f;
    }
    out[(int64_t) t * os + (size_t) h * kD + d] = A / L;
}

// ---------------------------------------------------------------- elementwise
__global__ void gate_kernel(float* __restrict__ o, int64_t os, const float* __restrict__ g, int64_t gs, int n) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = g[(int64_t) t * gs + i];
    o[(int64_t) t * os + i] *= 1.0f / (1.0f + expf(-gv));
}
__global__ void add_kernel(float* __restrict__ h, int64_t hs, const float* __restrict__ a, int64_t as, int n) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    h[(int64_t) t * hs + i] += a[(int64_t) t * as + i];
}
__global__ void softcap_kernel(float* __restrict__ x, int64_t n, float cap) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = tanhf(x[i] * (1.0f / cap)) * cap;
}

}  // namespace

void rms_rows(const float* x, int64_t xs, const float* gamma, float eps, int n, int T, float* y, int64_t ys,
              cudaStream_t s) {
    if (T <= 0) return;
    rms_rows_kernel<<<T, 256, 0, s>>>(x, xs, gamma, eps, n, y, ys);
    check("rms_rows");
}

void mix(const States& st, const float* query, int nq, int n_embd, int T, float* out, cudaStream_t s) {
    if (T <= 0 || st.n <= 0 || st.n > kMaxStates || nq < 1 || nq > 4) {
        std::fprintf(stderr, "inkling kernels: mix: %d states, %d queries (1..%d, 1..4)\n", st.n, nq, kMaxStates);
        return;
    }
    const int nc = (n_embd + kMixChunk - 1) / kMixChunk;
    float* part = mix_scratch((size_t) T * nc * kMixSums);
    if (part == nullptr) {
        std::fprintf(stderr, "inkling kernels: mix: no scratch\n");
        return;
    }
    mix_part_kernel<<<dim3((unsigned) nc, (unsigned) T), kMixChunk, 0, s>>>(st, query, nq, n_embd, part);
    mix_out_kernel<<<dim3((unsigned) nc, (unsigned) T), kMixChunk, 0, s>>>(st, part, nq, n_embd, out);
    check("mix");
}

void ctrl_cat(const float* cur, int64_t cs, const float* msg, int64_t ms, int n_embd, int T, float* cat, cudaStream_t s) {
    rms_rows(cur, cs, nullptr, 1e-6f, n_embd, T, cat, 2 * (int64_t) n_embd, s);
    rms_rows(msg, ms, nullptr, 1e-6f, n_embd, T, cat + n_embd, 2 * (int64_t) n_embd, s);
}

void ctrl_apply(const float* cur, int64_t cs, const float* msg, int64_t ms, const float* g, const float* f, int n, int T,
                float* out, int64_t os, cudaStream_t s) {
    if (T <= 0) return;
    ctrl_apply_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(cur, cs, msg, ms, g, f, n, out, os);
    check("ctrl_apply");
}

void conv_rows(const float* in, int64_t is, const float* w, const float* dw, int64_t dws, int head_ch, float* state,
               int C, int T, float* out, int64_t os, cudaStream_t s) {
    if (T <= 0) return;
    conv_kernel<<<(C + 127) / 128, 128, 0, s>>>(in, is, w, dw, dws, head_ch, state, C, T, out, os);
    check("conv_rows");
}

void conv_rows_par(const float* in, int64_t is, const float* w, const float* dw, int64_t dws, int head_ch, float* state,
                   int C, int T, float* out, int64_t os, cudaStream_t s) {
    if (T <= 0) return;
    conv_par_kernel<<<dim3((unsigned) ((C + 127) / 128), (unsigned) T), 128, 0, s>>>(in, is, w, dw, dws, head_ch, state, C,
                                                                                    out, os);
    conv_state_kernel<<<(C + 127) / 128, 128, 0, s>>>(in, is, state, C, T);
    check("conv_rows_par");
}

void post_proj(const PostProj& a, cudaStream_t s) {
    const int warps = a.Hkv + 2 * a.H;
    post_proj_kernel<<<(warps + 7) / 8, 256, 0, s>>>(a);
    check("post_proj");
}

void clip_heads(float* x, int64_t xs, const float* scale, int dim, int H, float thr, int64_t p0, float log_floor,
                float alpha, int T, cudaStream_t s) {
    if (T <= 0) return;
    if (dim > 128) {
        std::fprintf(stderr, "inkling kernels: clip_heads: dim %d > 128\n", dim);
        return;
    }
    const int warps = T * H;
    clip_kernel<<<(warps + 7) / 8, 256, 0, s>>>(x, xs, scale, dim, H, thr, p0, log_floor, alpha, T);
    check("clip_heads");
}

void kv_store(const float* k, int64_t ks, const float* v, int64_t vs, float eps, int n_head_kv, int64_t p0, int T,
              int ring, uint16_t* kc, uint16_t* vc, cudaStream_t s) {
    if (T <= 0) return;
    const int warps = T * n_head_kv;
    kv_store_kernel<<<(warps + 7) / 8, 256, 0, s>>>(k, ks, v, vs, eps, n_head_kv, p0, T, ring, (__half*) kc, (__half*) vc);
    check("kv_store");
}

// STRATA_INK_F16=<bits> (a debug A/B against llama.cpp's flash attention): 1 the bias rounded to FP16, 2 the scaled q
static int f16_env() {
    static const int v = getenv("STRATA_INK_F16") != nullptr ? std::atoi(getenv("STRATA_INK_F16")) : 0;
    return v;
}

size_t attn_part_floats(int T, int n_head, int max_splits) {
    return (size_t) T * n_head * max_splits * kPart;
}

void attn(const float* q, int64_t qs, const float* r, int64_t rs, const float* sink, int64_t ss, const float* rel_bias,
          int extent, const uint16_t* kc, const uint16_t* vc, int ring, int n_head, int n_head_kv, int64_t p0, int T,
          int window, float scale, float* part, int max_splits, float* out, int64_t os, cudaStream_t s) {
    if (T <= 0) return;
    if (n_head % n_head_kv != 0 || n_head / n_head_kv > kMaxG) {
        std::fprintf(stderr, "inkling kernels: attn: %d heads over %d KV heads\n", n_head, n_head_kv);
        return;
    }
    // the splits: the longest row's keys, a tile (128) at the least a split - a decode token fills the GPU
    // (r12 B1, 32-key splits at a short context: +0.5% at a 31-token prompt, within noise - not taken)
    const int64_t p1 = p0 + T - 1;
    const int64_t nk = window > 0 ? std::min<int64_t>(p1 + 1, window) : p1 + 1;
    int ns = (int) std::min<int64_t>(max_splits, std::max<int64_t>(1, (nk + kTile - 1) / kTile));
    ns = std::max(1, ns);
    const dim3 grid((unsigned) ns, (unsigned) n_head_kv, (unsigned) T);
#define INK_ATTN(GG) attn_part_kernel<GG><<<grid, kAttnWarps * 32, 0, s>>>(q, qs, r, rs, rel_bias, extent, \
        (const __half*) kc, (const __half*) vc, ring, n_head, n_head_kv, p0, window, scale, part, max_splits, f16_env())
    switch (n_head / n_head_kv) {
        case 1: INK_ATTN(1); break;
        case 2: INK_ATTN(2); break;
        case 3: INK_ATTN(3); break;
        case 4: INK_ATTN(4); break;
        case 6: INK_ATTN(6); break;
        case 8: INK_ATTN(8); break;
        default: std::fprintf(stderr, "inkling kernels: attn: %d q heads a KV head\n", n_head / n_head_kv); return;
    }
#undef INK_ATTN
    check("attn_part");
    attn_combine_kernel<<<dim3((unsigned) n_head, (unsigned) T), kD, 0, s>>>(part, ns, max_splits, sink, ss, n_head, out, os);
    check("attn_combine");
}

void gate_sigmoid(float* o, int64_t os, const float* g, int64_t gs, int n, int T, cudaStream_t s) {
    if (T <= 0) return;
    gate_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(o, os, g, gs, n);
    check("gate_sigmoid");
}

void add_rows(float* h, int64_t hs, const float* a, int64_t as, int n, int T, cudaStream_t s) {
    if (T <= 0) return;
    add_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(h, hs, a, as, n);
    check("add_rows");
}

void softcap(float* x, int64_t n, float cap, cudaStream_t s) {
    if (cap <= 0.0f || n <= 0) return;
    softcap_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(x, n, cap);
    check("softcap");
}

}  // namespace strata::kernels::ink
