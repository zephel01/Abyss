// src/kernels/cuda/inkling_fast.cu - see include/strata/kernels/inkling_fast.hpp.
#include "strata/kernels/inkling_fast.hpp"
#include "strata/kernels/pdl.hpp"

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
// a row's rms (the block's 256 threads): rms_rows_kernel's, and the verify window's controller inputs (ctrl_cat_n)
__device__ __forceinline__ void rms_row(const float* __restrict__ xr, const float* __restrict__ gamma, float eps, int n,
                                        float* __restrict__ yr, float* sh) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = gamma != nullptr ? xr[i] * inv * gamma[i] : xr[i] * inv;
}
__global__ void rms_rows_kernel(const float* __restrict__ x, int64_t xs, const float* __restrict__ gamma, float eps,
                                int n, float* __restrict__ y, int64_t ys) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    __shared__ float sh[32];
    const float* xr = x + (int64_t) blockIdx.x * xs;
    float* yr = y + (int64_t) blockIdx.x * ys;
    rms_row(xr, gamma, eps, n, yr, sh);
}
// grid (T, n, 2): controller c's row t - z 0 rms(cur), z 1 rms(msg) - into cat + c cat_c + t 2E (+ E)
__global__ void ctrl_cat_n_kernel(CtrlRows a, int E) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    __shared__ float sh[32];
    const int t = blockIdx.x, c = blockIdx.y, z = blockIdx.z;
    const float* xr = z == 0 ? a.cur[c] + (int64_t) t * a.cs : a.msg + (int64_t) c * E + (int64_t) t * a.ms;
    rms_row(xr, nullptr, 1e-6f, E, a.cat + (int64_t) c * a.cat_c + (int64_t) t * 2 * E + (int64_t) z * E, sh);
}
// grid (E / 256, T, n): ctrl_apply_kernel's step for controller c
__global__ void ctrl_apply_n_kernel(CtrlRows a, int E) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int t = blockIdx.y, c = blockIdx.z;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    const float cv = a.cur[c][(int64_t) t * a.cs + i], m = a.msg[(int64_t) c * E + (int64_t) t * a.ms + i];
    const float* g = a.g + (int64_t) c * a.gf_c;
    const float* f = a.f + (int64_t) c * a.gf_c;
    const float fv = tanhf(f[(int64_t) t * E + i]), gv = tanhf(g[(int64_t) t * E + i]);
    a.out[c][(int64_t) t * E + i] = (cv - fv * cv) + gv * m;
}

// ---------------------------------------------------------------- the depth attention
// two kernels over chunks of kMixChunk elements (grid (chunks, T)): the chunk's sums of squares of the states and the
// queries' dots with them (part[t][chunk][kMixSums]), then every chunk block adds the chunks' partials, takes the
// softmax and writes its chunk of the weighted sum
constexpr int kMixChunk = 256;
constexpr int kMixSums = kMaxStates * 5;

__global__ void __launch_bounds__(kMixChunk) mix_part_kernel(States st, const float* __restrict__ query, int nq, int E,
                                                             float* __restrict__ part) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
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
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
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

// a row's output from its 3 inputs before (x0 oldest) and its own (x3), the taps w0..w3 and the dynamic taps d (null:
// none)
__device__ __forceinline__ float conv_step(float x0, float x1, float x2, float x3, float w0, float w1, float w2, float w3,
                                           const float* __restrict__ d, float scale) {
    float o = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
    if (d != nullptr) {
        o = o + x3 * d[0];
        o = o + x2 * d[1];
        o = o + x1 * d[2];
        o = o + x0 * d[3];
    }
    return (x3 + o) * scale;
}
// conv_channel's row t alone (no dynamic taps), the state read: rows 0 .. t - 1 of in give row t's inputs before it
__device__ __forceinline__ float conv_row_keep(const float* __restrict__ in, int64_t is, const float* __restrict__ w,
                                               const float* __restrict__ state, int c, int t) {
    float x0 = state[(size_t) c * 3 + 0], x1 = state[(size_t) c * 3 + 1], x2 = state[(size_t) c * 3 + 2];
    const float w0 = w[(size_t) c * 4 + 0], w1 = w[(size_t) c * 4 + 1], w2 = w[(size_t) c * 4 + 2], w3 = w[(size_t) c * 4 + 3];
    for (int r = 0; r < t; ++r) {
        x0 = x1;
        x1 = x2;
        x2 = in[(int64_t) r * is + c];
    }
    return conv_step(x0, x1, x2, in[(int64_t) t * is + c], w0, w1, w2, w3, nullptr, 1.0f / sqrtf(5.0f));
}
// conv_kernel's channel c over T rows in order (the decode's fused post_proj_kernel runs the same function; the
// verify window's conv_win_kernel too, with kKeep: the state read, not written - conv_commit moves it later)
template <bool kKeep = false>
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
        out[(int64_t) t * os + c] = conv_step(x0, x1, x2, x3, w0, w1, w2, w3, dw != nullptr ? dw + (int64_t) t * dws + (size_t) h * 4 : nullptr, scale);
        x0 = x1;
        x1 = x2;
        x2 = x3;
    }
    if (kKeep) return;
    state[(size_t) c * 3 + 0] = x0;
    state[(size_t) c * 3 + 1] = x1;
    state[(size_t) c * 3 + 2] = x2;
}
// the verify window's: conv_kernel's rows over a state left as it is (in and out apart: the inputs stay for
// conv_commit)
__global__ void conv_win_kernel(const float* __restrict__ in, int64_t is, const float* __restrict__ w,
                                const float* __restrict__ dw, int64_t dws, int head_ch, float* __restrict__ state, int C,
                                int T, float* __restrict__ out, int64_t os) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    conv_channel<true>(in, is, w, dw, dws, head_ch, state, c, T, out, os);
}
// the states after a window's first n rows: slot b's channel c (of C) from its last 3 inputs - the window's rows
// (save + b * slot_floats, [part][tm][part channels], parts KV, KV, E, E) or the state's
__global__ void conv_commit_kernel(float* const* __restrict__ states, const float* __restrict__ save, size_t slot_floats,
                                   int KV, int E, int tm, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y;
    const int C = 2 * KV + 2 * E;
    if (c >= C) return;
    float* st = states[b] + (size_t) c * 3;
    int off, pc;
    if (c < KV) { off = 0; pc = KV; }
    else if (c < 2 * KV) { off = KV; pc = KV; }
    else if (c < 2 * KV + E) { off = 2 * KV; pc = E; }
    else { off = 2 * KV + E; pc = E; }
    const float* sv = save + (size_t) b * slot_floats + (size_t) tm * off + (size_t) (c - off);
    float v[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int i = n + k;   // (the sequence: the state's 3, then the window's rows)
        v[k] = i < 3 ? st[i] : sv[(size_t) (i - 3) * pc];
    }
#pragma unroll
    for (int k = 0; k < 3; ++k) st[k] = v[k];
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
// grid Hkv + ceil(2 H / 4) blocks of 128: block h < Hkv a KV head - thread d its k and v channel's conv, then warp 0
// the k clip and the cache rows; the other blocks a q head's or a rel head's clip a warp
__global__ void __launch_bounds__(kD) post_proj_kernel(PostProj a) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int64_t p = a.pd != nullptr ? *a.pd : a.p;   // (a captured graph's token: the position from the device word)
    const int lane = threadIdx.x & 31;
    if ((int) blockIdx.x < a.Hkv) {
        const int h = blockIdx.x;
        const int c = h * kD + (int) threadIdx.x;
        conv_channel(a.kf, a.KV, a.k_conv, a.kd, a.dws, kD, a.conv, c, 1, a.kf, a.KV);
        conv_channel(a.vf, a.KV, a.v_conv, a.vd, a.dws, kD, a.conv + (size_t) a.KV * 3, c, 1, a.vf, a.KV);
        __syncthreads();
        if (threadIdx.x >= 32) return;
        clip_head(a.kf, a.KV, nullptr, kD, a.qk_clip, p, 0.0f, 0.0f, 0, h, lane);
        __syncwarp();
        kv_store_head(a.kf, a.KV, a.vf, a.KV, a.eps, a.Hkv, p, a.ring, (__half*) a.kc, (__half*) a.vc, 0, h, lane);
        return;
    }
    const int wid = ((int) blockIdx.x - a.Hkv) * (kD / 32) + (int) (threadIdx.x >> 5);
    if (wid < a.H) {
        clip_head(a.qf, a.HD, a.q_clip, kD, a.qk_clip, p, a.lf, a.alpha, 0, wid, lane);
    } else if (wid < 2 * a.H) {
        clip_head(a.rf, (int64_t) a.H * kRel, a.rel_clip, kRel, a.rel_clip_thr, p, a.lf, a.alpha, 0, wid - a.H, lane);
    }
}

// the verify window's post_proj over T rows (the state read, left; the raw k / v rows kept apart): a block of kD
// threads a KV head (a thread a channel's k and v convs over the rows, then warp 0 each row's k clip and cache rows),
// then blocks of 4 warps, a warp a (row, q head)'s clip or a (row, rel head)'s - post_proj_kernel's functions
__global__ void __launch_bounds__(kD) post_proj_rows_kernel(PostProj a, const float* __restrict__ kraw,
                                                              const float* __restrict__ vraw, int T) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    if (a.pd != nullptr) a.p = *a.pd;   // (a captured window: the first position from the device word)
    const int lane = threadIdx.x & 31;
    // (as post_proj_kernel: a block a KV head - thread d its k / v channel's conv over the rows, then warp 0 each
    // row's k clip and cache rows; the other blocks a (row, q head)'s or a (row, rel head)'s clip a warp)
    if ((int) blockIdx.x < a.Hkv) {
        const int h = blockIdx.x;
        const int c = h * kD + (int) threadIdx.x;
        conv_channel<true>(kraw, a.KV, a.k_conv, a.kd, a.dws, kD, a.conv, c, T, a.kf, a.KV);
        conv_channel<true>(vraw, a.KV, a.v_conv, a.vd, a.dws, kD, a.conv + (size_t) a.KV * 3, c, T, a.vf, a.KV);
        __syncthreads();
        if (threadIdx.x >= 32) return;
        for (int t = 0; t < T; ++t) {
            clip_head(a.kf, a.KV, nullptr, kD, a.qk_clip, a.p, 0.0f, 0.0f, t, h, lane);
            __syncwarp();
            kv_store_head(a.kf, a.KV, a.vf, a.KV, a.eps, a.Hkv, a.p, a.ring, (__half*) a.kc, (__half*) a.vc, t, h, lane);
        }
        return;
    }
    const int wid = ((int) blockIdx.x - a.Hkv) * (kD / 32) + (int) (threadIdx.x >> 5);
    if (wid < T * a.H) {
        clip_head(a.qf, a.HD, a.q_clip, kD, a.qk_clip, a.p, a.lf, a.alpha, wid / a.H, wid % a.H, lane);
    } else if (wid < 2 * T * a.H) {
        const int w = wid - T * a.H;
        clip_head(a.rf, (int64_t) a.H * kRel, a.rel_clip, kRel, a.rel_clip_thr, a.p, a.lf, a.alpha, w / a.H, w % a.H, lane);
    }
}

// the batched decode's conv_rows: grid (C / 128, T), row t one step over its own state (conv_kernel's T = 1 step)
__global__ void conv_batch_kernel(const float* __restrict__ in, int64_t is, const float* __restrict__ w,
                                  float* const* __restrict__ st, int64_t soff, int C, float* __restrict__ out, int64_t os) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= C) return;
    conv_channel(in + (int64_t) t * is, is, w, nullptr, 0, C, st[t] + soff, c, 1, out + (int64_t) t * os, os);
}

// the batched decode's post_proj: grid (post_proj_kernel's blocks, T) - block (b, t) post_proj_kernel's block b on row
// t, which is its own conversation (its position, its conv state, its caches)
__global__ void __launch_bounds__(kD) post_proj_batch_kernel(PostProj a, BatchRows br, const float* __restrict__ kraw,
                                                               const float* __restrict__ vraw) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int t = blockIdx.y;
    const int64_t p = br.pos[t];
    const int lane = threadIdx.x & 31;
    float* kf = a.kf + (int64_t) t * a.KV;
    float* vf = a.vf + (int64_t) t * a.KV;
    if ((int) blockIdx.x < a.Hkv) {
        const int h = blockIdx.x;
        const int c = h * kD + (int) threadIdx.x;
        float* conv = br.conv[t];
        conv_channel(kraw + (int64_t) t * a.KV, a.KV, a.k_conv, a.kd != nullptr ? a.kd + (int64_t) t * a.dws : nullptr,
                     a.dws, kD, conv, c, 1, kf, a.KV);
        conv_channel(vraw + (int64_t) t * a.KV, a.KV, a.v_conv, a.vd != nullptr ? a.vd + (int64_t) t * a.dws : nullptr,
                     a.dws, kD, conv + (size_t) a.KV * 3, c, 1, vf, a.KV);
        __syncthreads();
        if (threadIdx.x >= 32) return;
        clip_head(kf, a.KV, nullptr, kD, a.qk_clip, p, 0.0f, 0.0f, 0, h, lane);
        __syncwarp();
        kv_store_head(kf, a.KV, vf, a.KV, a.eps, a.Hkv, p, a.ring, (__half*) br.kc[t], (__half*) br.vc[t], 0, h, lane);
        return;
    }
    const int wid = ((int) blockIdx.x - a.Hkv) * (kD / 32) + (int) (threadIdx.x >> 5);
    if (wid < a.H) {
        clip_head(a.qf + (int64_t) t * a.HD, a.HD, a.q_clip, kD, a.qk_clip, p, a.lf, a.alpha, 0, wid, lane);
    } else if (wid < 2 * a.H) {
        clip_head(a.rf + (int64_t) t * a.H * kRel, (int64_t) a.H * kRel, a.rel_clip, kRel, a.rel_clip_thr, p, a.lf,
                  a.alpha, 0, wid - a.H, lane);
    }
}

// ---------------------------------------------------------------- attention
constexpr int kAttnWarps = 4;
constexpr int kTile = 32 * kAttnWarps;   // keys a tile: a warp's 32
constexpr int kMaxG = 8;                 // q heads a KV head at most

// the part rows: [T][H][max_splits] of (m, l, acc[128])
constexpr int kPart = kD + 2;

// grid (splits, n_head_kv, T): block (split, kv head, row) runs the G q heads of the KV head over its key range
// a row's split count: its keys over tiles, at most max_splits (the token path's choice for that row alone)
__host__ __device__ __forceinline__ int attn_row_splits(int64_t nk, int max_splits) {
    const int64_t n = (nk + kTile - 1) / kTile;
    return (int) (n < 1 ? 1 : n > max_splits ? max_splits : n);
}

// kRows (the verify window's): each row its own split count, the one a token at that position alone gets - the same
// key ranges, so the same sums (blocks past a row's count leave at once)
template <int kG, bool kRows = false>
__global__ void __launch_bounds__(kAttnWarps * 32) attn_part_kernel(
    const float* __restrict__ q, int64_t qs, const float* __restrict__ r, int64_t rs, const float* __restrict__ relb,
    int extent, const __half* __restrict__ kc, const __half* __restrict__ vc, int ring, int H, int Hkv, int64_t p0,
    int window, float scale, float* __restrict__ part, int max_splits, int f16, const int64_t* __restrict__ p0d = nullptr,
    BatchRows br = BatchRows(), int64_t tc_from = -1) {
    STRATA_PDL_ENTRY();
    // (kRows: grid (T, splits, n_head_kv) - a split's rows next to each other, so the window's columns read the same
    // keys at about the same time and the L2 serves the later ones; the blocks compute what they did before)
    const int split = kRows ? (int) blockIdx.y : (int) blockIdx.x, kvh = kRows ? (int) blockIdx.z : (int) blockIdx.y;
    const int t = kRows ? (int) blockIdx.x : (int) blockIdx.z;
    constexpr int G = kG;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    // (p0d: a captured graph's token; br: the batched decode - row t its own conversation's position and caches)
    const int64_t p = kRows && br.pos != nullptr ? br.pos[t] : (p0d != nullptr ? *p0d : p0) + t;
    if (kRows && br.pos != nullptr) {
        kc = (const __half*) br.kc[t];
        vc = (const __half*) br.vc[t];
    }
    const int64_t lo = window > 0 ? (p - window + 1 > 0 ? p - window + 1 : (int64_t) 0) : 0;
    const int64_t nk = p - lo + 1;
    const int nsplit = kRows ? attn_row_splits(nk, max_splits) : (int) gridDim.x;   // (kRows: the row's own count)
    if (kRows && split >= nsplit) return;
    if (kRows && tc_from >= 0 && window == 0 && nk > tc_from) return;   // (attn_dec_tc_kernel's row)
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
            // (the 32 keys' value rows loaded at once, then added in key order: the same sums as a row at a time;
            // a row at a time waited on each load - the decode's attention was latency-bound)
            const int64_t r0 = w0 % ring;
            const size_t hoff = (size_t) kvh * kD + (size_t) lane * 4, rstride = (size_t) Hkv * kD;
            if (nkk == 32) {
                uint2 vr[32];
#pragma unroll
                for (int kk = 0; kk < 32; ++kk) {
                    int64_t rr = r0 + kk;
                    if (rr >= ring) rr -= ring;
                    vr[kk] = *reinterpret_cast<const uint2*>(vc + (size_t) rr * rstride + hoff);
                }
#pragma unroll
                for (int kk = 0; kk < 32; ++kk) {
                    const float2 va = __half22float2(*reinterpret_cast<const __half2*>(&vr[kk].x));
                    const float2 vb = __half22float2(*reinterpret_cast<const __half2*>(&vr[kk].y));
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
            } else {
                for (int kk = 0; kk < nkk; ++kk) {
                    int64_t rr = r0 + kk;
                    if (rr >= ring) rr -= ring;
                    const size_t row = (size_t) rr * rstride + hoff;
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

// grid (H, T): the splits and the sink into out (rows_p0 >= 0: the verify window's, each row its own split count)
__global__ void attn_combine_kernel(const float* __restrict__ part, int nsplit, int max_splits, const float* __restrict__ sink,
                                    int64_t ss, int H, float* __restrict__ out, int64_t os, int64_t rows_p0 = -1,
                                    int window = 0, const int64_t* __restrict__ rows_p0d = nullptr,
                                    const int64_t* __restrict__ rows_pos = nullptr) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    if (rows_p0 >= 0) {   // (rows_p0d: a captured window's first position, from the device word; rows_pos: a row's own)
        const int64_t p = rows_pos != nullptr ? rows_pos[t] : (rows_p0d != nullptr ? *rows_p0d : rows_p0) + t;
        nsplit = attn_row_splits(window > 0 ? (p + 1 < window ? p + 1 : (int64_t) window) : p + 1, max_splits);
    }
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

// ---------------------------------------------------------------- the prompt path's tensor-core attention
// attn's math over a block of tokens at once: S = (scale q) K^T and O = P V on f16 tensor cores (F32 accumulation),
// the relative bias added to S from small MMAs on FP16 pairs (about F32), an online softmax in registers.
// mma.sync / ldmatrix / cp.async need sm_80; AMD compiles none of it (attn_prompt returns false: attn runs).
#if defined(__HIPCC__) || defined(STRATA_USE_HIP)
#define STRATA_INK_TC 0
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define STRATA_INK_TC 1
#else
#define STRATA_INK_TC 2   // (device code for an older sm: traps; the host refuses such a device)
#endif

#if STRATA_INK_TC != 0
constexpr int kTcBK = 64;   // keys a tile
constexpr int kTcTW = 40;   // a bias table row (floats): 32 keys + 8 (conflict-free reads)

__device__ __forceinline__ void tc_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if STRATA_INK_TC != 1
    __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ void tc_ldsm(uint32_t* r, const void* p) {
#if STRATA_INK_TC != 1
    __trap();
#else
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
#endif
}
__device__ __forceinline__ void tc_ldsm_t(uint32_t* r, const void* p) {
#if STRATA_INK_TC != 1
    __trap();
#else
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
#endif
}
__device__ __forceinline__ void tc_cp16(void* dst, const void* src, bool on) {
#if STRATA_INK_TC != 1
    __trap();
#else
    const unsigned d = (unsigned) __cvta_generic_to_shared(dst);
    const int n = on ? 16 : 0;   // (0: zero-filled)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
#endif
}
__device__ __forceinline__ void tc_commit() {
#if STRATA_INK_TC == 1
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void tc_wait1() {
#if STRATA_INK_TC == 1
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}
__device__ __forceinline__ uint32_t tc_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}
// a K / V tile row's 16-byte chunk c in shared memory (XOR-swizzled: ldmatrix's 8 rows hit 8 bank groups)
__device__ __forceinline__ int tc_at(int key, int c) {
    return key * kD + ((c ^ (key & 7)) << 3);
}

// grid (ceil(T / (16 MT)), n_head_kv, splits), G MT warps: block (b, kv head, split) runs tokens 16 MT b .. + 16 MT - 1
// of the KV head's G q heads over the split's share of the block's key tiles (keys lo .. the last token's position);
// warp w: q head w % G, tokens 16 (w / G) .. + 15 (an m16 tile).  K and V tiles (64 keys) stream through one shared
// buffer each (cp.async: the next K loads during P V, the next V during the next S).  splits 1: the sink and the
// division here, out written; else part rows (m, l, acc) for attn_combine_kernel.
template <int G, int MT>
__global__ void __launch_bounds__(G * MT * 32, 2) attn_tc_kernel(
    const float* __restrict__ q, int64_t qs, const float* __restrict__ r, int64_t rs, const float* __restrict__ sink,
    int64_t ss, const float* __restrict__ relb, int extent, const __half* __restrict__ kc, const __half* __restrict__ vc,
    int ring, int H, int Hkv, int64_t p0, int T, int window, float scale, float* __restrict__ part, int max_splits,
    float* __restrict__ out, int64_t os) {
    constexpr int NW = G * MT, BQ = 16 * MT;
    __shared__ __align__(128) __half sK[kTcBK * kD];
    __shared__ __align__(128) __half sV[kTcBK * kD];
    __shared__ __align__(16) float sB[NW][16 * kTcTW];
    const int kvh = blockIdx.y, split = blockIdx.z, nsplit = gridDim.z;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = warp % G, mt = warp / G, h = kvh * G + g;
    const int tf = blockIdx.x * BQ, tl = min(T, tf + BQ) - 1;
    // (positions as int: a context of 2^31 tokens is far off)
    const int pf = (int) p0 + tf, pl = (int) p0 + tl;
    const int lo = window > 0 ? max(pf - window + 1, 0) : 0;
    const int ntile = (int) ((pl - lo + kTcBK) / kTcBK);
    const int per = (ntile + nsplit - 1) / nsplit;
    const int i0 = split * per, i1 = min(ntile, i0 + per);
    // the warp's tokens and this lane's two accumulator rows (tokens ra, ra + 8)
    const int wt = tf + mt * 16;
    const bool wact = wt <= tl;
    const int wp0 = (int) p0 + wt, wp1 = (int) p0 + min(wt + 15, tl);
    const int ra = wt + (lane >> 2), rb = ra + 8;
    const int pa = (int) p0 + ra, pb = (int) p0 + rb;
    const int qd = 2 * (lane & 3);
    uint32_t qf[8][4];
    {
        const float* qa = q + (int64_t) min(ra, tl) * qs + (size_t) h * kD;
        const float* qb = q + (int64_t) min(rb, tl) * qs + (size_t) h * kD;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const int c = 16 * ks + qd;
            qf[ks][0] = tc_h2(qa[c] * scale, qa[c + 1] * scale);
            qf[ks][1] = tc_h2(qb[c] * scale, qb[c + 1] * scale);
            qf[ks][2] = tc_h2(qa[c + 8] * scale, qa[c + 9] * scale);
            qf[ks][3] = tc_h2(qb[c + 8] * scale, qb[c + 9] * scale);
        }
    }
    float o[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[n][e] = 0.0f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;
    uint32_t rf[4], rl[4];   // r's A fragment (the warp's 16 tokens x kRel) in FP16, and what FP16 left out
    {
        const float* r0 = r + (int64_t) min(ra, tl) * rs + (size_t) h * kRel + qd;
        const float* r1 = r + (int64_t) min(rb, tl) * rs + (size_t) h * kRel + qd;
        const float v[4][2] = {{r0[0], r0[1]}, {r1[0], r1[1]}, {r0[8], r0[9]}, {r1[8], r1[9]}};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            rf[i] = tc_h2(v[i][0], v[i][1]);
            const float2 hv = __half22float2(*reinterpret_cast<const __half2*>(&rf[i]));
            rl[i] = tc_h2(v[i][0] - hv.x, v[i][1] - hv.y);
        }
    }
    const auto load = [&](__half* dst, const __half* src, int j0) {
        for (int i = threadIdx.x; i < kTcBK * 16; i += NW * 32) {
            const int key = i >> 4, c = i & 15;
            const int j = j0 + key;
            const bool on = j <= pl;
            tc_cp16(dst + tc_at(key, c), src + ((size_t) ((on ? j : lo) % ring) * Hkv + kvh) * kD + c * 8, on);
        }
    };
    if (i0 < i1) load(sK, kc, lo + i0 * kTcBK);
    tc_commit();
    if (i0 < i1) load(sV, vc, lo + i0 * kTcBK);
    tc_commit();
    float* tb = sB[warp];
    for (int it = i0; it < i1; ++it) {
        const int j0 = lo + it * kTcBK, j1 = j0 + kTcBK - 1;
        tc_wait1();
        __syncthreads();   // K(it) in
        // the warp's keys: wp1 at the most, wp0 - window + 1 at the least
        const bool run = wact && j0 <= wp1 && (window <= 0 || j1 >= wp0 - window + 1);
        uint32_t pf16[4][4];
        if (run) {
            // S = q K^T a half (32 keys) at a time, then the half's bias b(d) = r . rel_bias[d] (d < extent): for a
            // quarter (16 keys) the 31 distances it spans, U[t][u] = r_t . rel_bias[D0 - 15 + u], 8 of them a
            // m16n8k16 x 3 (r and rel_bias each as FP16 + its FP16 remainder, F32 sums) into shared memory, read back
            // along the diagonals
            float sc[8][4];
            const bool bias = wp0 - j1 < extent;
#pragma unroll
            for (int hb = 0; hb < 2; ++hb) {
#pragma unroll
                for (int n = 4 * hb; n < 4 * hb + 4; ++n) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) sc[n][e] = 0.0f;
#pragma unroll
                    for (int kp = 0; kp < 4; ++kp) {
                        uint32_t b[4];
                        tc_ldsm(b, sK + tc_at(8 * n + (lane & 7), 4 * kp + (lane >> 3)));
                        tc_mma(sc[n], qf[2 * kp], b[0], b[1]);
                        tc_mma(sc[n], qf[2 * kp + 1], b[2], b[3]);
                    }
                }
                if (bias) {
#pragma unroll
                    for (int qb = 2 * hb; qb < 2 * hb + 2; ++qb) {
                        const int base = wp0 - j0 - 16 * qb - 15;   // U's column 0: d of token 0 x key 16 qb + 15
                        float u[4][4];
#pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            const int d = base + 8 * nt + (lane >> 2);
                            float2 x0 = make_float2(0.0f, 0.0f), x1 = x0;
                            if (d >= 0 && d < extent) {
                                const float* rb = relb + (size_t) d * kRel + qd;
                                x0 = __ldg(reinterpret_cast<const float2*>(rb));
                                x1 = __ldg(reinterpret_cast<const float2*>(rb + 8));
                            }
#pragma unroll
                            for (int e = 0; e < 4; ++e) u[nt][e] = 0.0f;
                            const uint32_t b0 = tc_h2(x0.x, x0.y), b1 = tc_h2(x1.x, x1.y);
                            tc_mma(u[nt], rf, b0, b1);
                            // + r's and rel_bias's FP16 remainders: about F32's products (FP16 alone: the 9K text's
                            // KL 0.0046-0.0048 against 0.0039)
                            const float2 h0 = __half22float2(*reinterpret_cast<const __half2*>(&b0));
                            const float2 h1 = __half22float2(*reinterpret_cast<const __half2*>(&b1));
                            tc_mma(u[nt], rl, b0, b1);
                            tc_mma(u[nt], rf, tc_h2(x0.x - h0.x, x0.y - h0.y), tc_h2(x1.x - h1.x, x1.y - h1.y));
                        }
#pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            *reinterpret_cast<float2*>(tb + (lane >> 2) * kTcTW + 8 * nt + qd) =
                                make_float2(u[nt][0], u[nt][1]);
                            *reinterpret_cast<float2*>(tb + ((lane >> 2) + 8) * kTcTW + 8 * nt + qd) =
                                make_float2(u[nt][2], u[nt][3]);
                        }
                        __syncwarp();
                        // S[t][c] (c the key in the quarter) += U[t][t - c + 15]
#pragma unroll
                        for (int m = 0; m < 2; ++m)
#pragma unroll
                            for (int e = 0; e < 2; ++e) {
                                const int c = 8 * m + qd + e, ta = lane >> 2;
                                sc[2 * qb + m][e] += tb[ta * kTcTW + ta - c + 15];
                                sc[2 * qb + m][2 + e] += tb[(ta + 8) * kTcTW + ta + 8 - c + 15];
                            }
                        __syncwarp();
                    }
                }
            }
            // the mask (causal, the window) unless every key is visible to every row of the warp
            const bool all = j1 <= wp0 && (window <= 0 || j0 >= wp1 - window + 1);
            if (!all) {
#pragma unroll
                for (int n = 0; n < 8; ++n)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int j = j0 + 8 * n + qd + e;
                        if (j > pa || (window > 0 && j <= pa - window)) sc[n][e] = -INFINITY;
                        if (j > pb || (window > 0 && j <= pb - window)) sc[n][2 + e] = -INFINITY;
                    }
            }
            // the online softmax: a row's max over its quad
            float xa = -INFINITY, xb = -INFINITY;
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                xa = fmaxf(xa, fmaxf(sc[n][0], sc[n][1]));
                xb = fmaxf(xb, fmaxf(sc[n][2], sc[n][3]));
            }
#pragma unroll
            for (int o2 = 1; o2 <= 2; o2 <<= 1) {
                xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, o2));
                xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, o2));
            }
            const float na = fmaxf(ma, xa), nb = fmaxf(mb, xb);
            const float ua = na == -INFINITY ? 0.0f : na, ub = nb == -INFINITY ? 0.0f : nb;
            const float fa = __expf(ma - ua), fb = __expf(mb - ub);
            ma = na;
            mb = nb;
            float sa = 0.0f, sb = 0.0f;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const __half2 a0 = __floats2half2_rn(__expf(sc[2 * kk][0] - ua), __expf(sc[2 * kk][1] - ua));
                const __half2 a1 = __floats2half2_rn(__expf(sc[2 * kk][2] - ub), __expf(sc[2 * kk][3] - ub));
                const __half2 a2 = __floats2half2_rn(__expf(sc[2 * kk + 1][0] - ua), __expf(sc[2 * kk + 1][1] - ua));
                const __half2 a3 = __floats2half2_rn(__expf(sc[2 * kk + 1][2] - ub), __expf(sc[2 * kk + 1][3] - ub));
                const float2 f0 = __half22float2(a0), f1 = __half22float2(a1), f2 = __half22float2(a2),
                             f3 = __half22float2(a3);
                sa += f0.x + f0.y + f2.x + f2.y;   // (the sums of the FP16 weights P V multiplies)
                sb += f1.x + f1.y + f3.x + f3.y;
                pf16[kk][0] = *reinterpret_cast<const uint32_t*>(&a0);
                pf16[kk][1] = *reinterpret_cast<const uint32_t*>(&a1);
                pf16[kk][2] = *reinterpret_cast<const uint32_t*>(&a2);
                pf16[kk][3] = *reinterpret_cast<const uint32_t*>(&a3);
            }
            la = la * fa + sa;
            lb = lb * fb + sb;
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                o[n][0] *= fa;
                o[n][1] *= fa;
                o[n][2] *= fb;
                o[n][3] *= fb;
            }
        }
        __syncthreads();   // sK free
        if (it + 1 < i1) load(sK, kc, j0 + kTcBK);
        tc_commit();
        tc_wait1();
        __syncthreads();   // V(it) in
        if (run) {
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                for (int np = 0; np < 8; ++np) {
                    uint32_t b[4];
                    tc_ldsm_t(b, sV + tc_at(16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1), 2 * np + (lane >> 4)));
                    tc_mma(o[2 * np], pf16[kk], b[0], b[1]);
                    tc_mma(o[2 * np + 1], pf16[kk], b[2], b[3]);
                }
        }
        __syncthreads();   // sV free
        if (it + 1 < i1) load(sV, vc, j0 + kTcBK);
        tc_commit();
    }
    // the rows' sums over the quad
#pragma unroll
    for (int o2 = 1; o2 <= 2; o2 <<= 1) {
        la += __shfl_xor_sync(0xffffffffu, la, o2);
        lb += __shfl_xor_sync(0xffffffffu, lb, o2);
    }
    if (!wact) return;
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
        const int t = hr == 0 ? ra : rb;
        if (t > tl) continue;
        const float m = hr == 0 ? ma : mb, l = hr == 0 ? la : lb;
        if (nsplit == 1) {
            const float sk = sink[(int64_t) t * ss + h];
            const float M = fmaxf(m, sk);
            const float f = m == -INFINITY ? 0.0f : expf(m - M);
            const float inv = 1.0f / (l * f + expf(sk - M));
            float* orow = out + (int64_t) t * os + (size_t) h * kD;
#pragma unroll
            for (int n = 0; n < 16; ++n)
                *reinterpret_cast<float2*>(orow + 8 * n + qd) =
                    make_float2(o[n][2 * hr] * f * inv, o[n][2 * hr + 1] * f * inv);
        } else {
            float* pr = part + (((size_t) t * H + h) * max_splits + split) * kPart;
            if ((lane & 3) == 0) {
                pr[0] = m;
                pr[1] = l;
            }
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                pr[2 + 8 * n + qd] = o[n][2 * hr];
                pr[3 + 8 * n + qd] = o[n][2 * hr + 1];
            }
        }
    }
}

// ---------------------------------------------------------------- the token path's tensor-core attention (global layers)
// attn_part_kernel's part rows (m, l, acc) for a decode token, a verify window's rows or a batch's rows on f16 tensor
// cores, at F32's precision: q and the probabilities each go in as an FP16 pair (the value and what FP16 left out, two
// MMAs; K / V are the FP16 cache, exact), F32 sums, the bias in F32 as attn_part_kernel adds it - the sums' order is
// what changes.  A split is a fixed range of keys, C = 128 2^k a split (the least with ceil(keys / C) <= max_splits),
// the same on every GPU, so a window's rows share their splits' tiles: the block reads each 64-key tile once for all
// its rows (T G <= 16 MT rows: token i, q head g at row i G + g).  Warp w: m-tile w / 4, keys
// 16 (w % 4) .. + 15 of every tile; the 4 key warps' runs merged in warp order.  A row's sums do not depend on the
// other rows in the block (masked keys add exact zeros), so a window row is the decode token's bits at its position.
// Rows with keys <= from are not this kernel's (attn_part_kernel's, the old bits); a window across a C boundary takes
// its two groups one after the other.
__host__ __device__ __forceinline__ int64_t dec_chunk(int64_t nk, int max_splits) {
    int64_t c = 128;
    while ((nk + c - 1) / c > max_splits) c *= 2;
    return c;
}

template <int G, int MT>
__global__ void __launch_bounds__(MT * 4 * 32) attn_dec_tc_kernel(
    const float* __restrict__ q, int64_t qs, const float* __restrict__ r, int64_t rs, const float* __restrict__ relb,
    int extent, const __half* __restrict__ kc, const __half* __restrict__ vc, int ring, int H, int Hkv, int64_t p0,
    int T, float scale, float* __restrict__ part, int max_splits, int64_t from, const int64_t* __restrict__ p0d,
    BatchRows br) {
    STRATA_PDL_ENTRY();
    constexpr int NW = MT * 4;
    __shared__ __align__(128) __half sKV[2 * kTcBK * kD];   // (the K tile, the V tile; after the keys the merge's)
    __half* const sK = sKV;
    __half* const sV = sKV + kTcBK * kD;
    // (32 KB of shared memory in all: r is read from global memory where the bias needs it and
    // the merge's m / l go after its partial sums in the same buffer)
    const int split = blockIdx.x, kvh = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int mt = warp >> 2, kw = warp & 3;
    const bool bt = br.pos != nullptr;
    const int rt0 = bt ? (int) blockIdx.z : 0, nr = bt ? 1 : T;   // (the block's first row and its row count)
    int64_t pf;
    if (bt) {
        pf = br.pos[rt0];
        kc = (const __half*) br.kc[rt0];
        vc = (const __half*) br.vc[rt0];
    } else {
        pf = p0d != nullptr ? *p0d : p0;
    }
    const int qd = 2 * (lane & 3);
    const int ra = mt * 16 + (lane >> 2), rb = ra + 8;   // this lane's two rows (block rows)
    const int ta = ra / G, tb = rb / G;                   // their tokens
    const int64_t pa = pf + ta, pb = pf + tb;
    const int ha = kvh * G + ra % G, hb = kvh * G + rb % G;
    const float* const ra_r = r + (int64_t) (rt0 + min(ta, nr - 1)) * rs + (size_t) ha * kRel;
    const float* const rb_r = r + (int64_t) (rt0 + min(tb, nr - 1)) * rs + (size_t) hb * kRel;
    // q (scaled in F32, as attn_part_kernel) as FP16 + its FP16 remainder: the A fragments of the warp's m-tile
    uint32_t qh[8][4], ql[8][4];
    {
        const float* qa = q + (int64_t) (rt0 + min(ta, nr - 1)) * qs + (size_t) ha * kD;
        const float* qb = q + (int64_t) (rt0 + min(tb, nr - 1)) * qs + (size_t) hb * kD;
        const bool va = ta < nr, vb = tb < nr;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const int c = 16 * ks + qd;
            const float x[4][2] = {{va ? qa[c] * scale : 0.0f, va ? qa[c + 1] * scale : 0.0f},
                                   {vb ? qb[c] * scale : 0.0f, vb ? qb[c + 1] * scale : 0.0f},
                                   {va ? qa[c + 8] * scale : 0.0f, va ? qa[c + 9] * scale : 0.0f},
                                   {vb ? qb[c + 8] * scale : 0.0f, vb ? qb[c + 9] * scale : 0.0f}};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                qh[ks][i] = tc_h2(x[i][0], x[i][1]);
                const float2 hv = __half22float2(*reinterpret_cast<const __half2*>(&qh[ks][i]));
                ql[ks][i] = tc_h2(x[i][0] - hv.x, x[i][1] - hv.y);
            }
        }
    }
    // the passes: the rows of the last row's C, then (a window across a C boundary) the first row's
    const int64_t c_last = pf + nr - 1 + 1 > from ? dec_chunk(pf + nr, max_splits) : 0;
    const int64_t c_first = pf + 1 > from ? dec_chunk(pf + 1, max_splits) : 0;
    __syncthreads();
    for (int pass = 0; pass < 2; ++pass) {
        const int64_t C = pass == 0 ? c_last : c_first;
        if (C == 0 || (pass == 1 && c_first == c_last)) continue;
        // a row's in this pass: a real row, past from, its C this one
        const auto in_pass = [&](int t) {
            if (t >= nr) return false;
            const int64_t nk = pf + t + 1;
            return nk > from && dec_chunk(nk, max_splits) == C;
        };
        const bool ina = in_pass(ta), inb = in_pass(tb);
        int tlo = nr, thi = -1;   // (the pass's rows: a contiguous run)
        for (int t = 0; t < nr; ++t)
            if (in_pass(t)) {
                tlo = min(tlo, t);
                thi = max(thi, t);
            }
        const int64_t pmin = pf + tlo, pmax = pf + thi;
        const int64_t k0 = (int64_t) split * C, k1 = k0 + C < pmax + 1 ? k0 + C : pmax + 1;
        const int ntile = k0 < k1 ? (int) ((k1 - k0 + kTcBK - 1) / kTcBK) : 0;
        float o[16][4];
#pragma unroll
        for (int n = 0; n < 16; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) o[n][e] = 0.0f;
        float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;
        const auto load = [&](__half* dst, const __half* src, int64_t j0) {
            for (int i = threadIdx.x; i < kTcBK * 16; i += NW * 32) {
                const int key = i >> 4, c = i & 15;
                const int64_t j = j0 + key;
                const bool on = j <= pmax;
                tc_cp16(dst + tc_at(key, c), src + ((size_t) ((on ? j : k0) % ring) * Hkv + kvh) * kD + c * 8, on);
            }
        };
        if (ntile > 0) load(sK, kc, k0);
        tc_commit();
        if (ntile > 0) load(sV, vc, k0);
        tc_commit();
        for (int it = 0; it < ntile; ++it) {
            const int64_t j0 = k0 + (int64_t) it * kTcBK, jw = j0 + 16 * kw;   // (the warp's 16 keys from jw)
            tc_wait1();
            __syncthreads();   // K(it) in
            float sc[2][4];
#pragma unroll
            for (int n = 0; n < 2; ++n) {
#pragma unroll
                for (int e = 0; e < 4; ++e) sc[n][e] = 0.0f;
#pragma unroll
                for (int kp = 0; kp < 4; ++kp) {
                    uint32_t b[4];
                    tc_ldsm(b, sK + tc_at(16 * kw + 8 * n + (lane & 7), 4 * kp + (lane >> 3)));
                    tc_mma(sc[n], qh[2 * kp], b[0], b[1]);
                    tc_mma(sc[n], ql[2 * kp], b[0], b[1]);
                    tc_mma(sc[n], qh[2 * kp + 1], b[2], b[3]);
                    tc_mma(sc[n], ql[2 * kp + 1], b[2], b[3]);
                }
            }
            // the bias (attn_part_kernel's F32 dot) where a row's distance is under extent, the mask past the row
            const bool bias = jw + 15 > pmin - extent;
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const bool hi = e >= 2;
                    const int64_t j = jw + 8 * n + qd + (e & 1);
                    const int64_t p = hi ? pb : pa;
                    if (!(hi ? inb : ina) || j > p) {
                        sc[n][e] = -INFINITY;
                        continue;
                    }
                    const int64_t d = p - j;
                    if (bias && d < extent) {
                        const float* rr = hi ? rb_r : ra_r;
                        const float* rv = relb + (size_t) d * kRel;
                        float bsum = 0.0f;
#pragma unroll
                        for (int e2 = 0; e2 < kRel; ++e2) bsum += __ldg(rr + e2) * __ldg(rv + e2);
                        sc[n][e] += bsum;
                    }
                }
            // the online softmax: a row's max over its quad (16 keys)
            float xa = fmaxf(fmaxf(sc[0][0], sc[0][1]), fmaxf(sc[1][0], sc[1][1]));
            float xb = fmaxf(fmaxf(sc[0][2], sc[0][3]), fmaxf(sc[1][2], sc[1][3]));
#pragma unroll
            for (int o2 = 1; o2 <= 2; o2 <<= 1) {
                xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, o2));
                xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, o2));
            }
            const float na = fmaxf(ma, xa), nb = fmaxf(mb, xb);
            const float ua = na == -INFINITY ? 0.0f : na, ub = nb == -INFINITY ? 0.0f : nb;
            const float fa = expf(ma - ua), fb = expf(mb - ub);
            ma = na;
            mb = nb;
            // the probabilities: F32 into the sums l, x 2^15 (no FP16 subnormals) as an FP16 pair into P V
            uint32_t ph[4], pl[4];
            float sa = 0.0f, sb = 0.0f;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const float e0 = expf(sc[n][0] - ua), e1 = expf(sc[n][1] - ua);
                const float e2 = expf(sc[n][2] - ub), e3 = expf(sc[n][3] - ub);
                sa += e0 + e1;
                sb += e2 + e3;
                const float s0 = e0 * 32768.0f, s1 = e1 * 32768.0f, s2 = e2 * 32768.0f, s3 = e3 * 32768.0f;
                ph[2 * n] = tc_h2(s0, s1);
                ph[2 * n + 1] = tc_h2(s2, s3);
                const float2 h01 = __half22float2(*reinterpret_cast<const __half2*>(&ph[2 * n]));
                const float2 h23 = __half22float2(*reinterpret_cast<const __half2*>(&ph[2 * n + 1]));
                pl[2 * n] = tc_h2(s0 - h01.x, s1 - h01.y);
                pl[2 * n + 1] = tc_h2(s2 - h23.x, s3 - h23.y);
            }
            la = la * fa + sa;
            lb = lb * fb + sb;
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                o[n][0] *= fa;
                o[n][1] *= fa;
                o[n][2] *= fb;
                o[n][3] *= fb;
            }
            __syncthreads();   // sK free
            if (it + 1 < ntile) load(sK, kc, j0 + kTcBK);
            tc_commit();
            tc_wait1();
            __syncthreads();   // V(it) in
            {
                // A: P (rows x the warp's 16 keys) - n-tile 0 the keys 0..7, 1 the keys 8..15
                const uint32_t ah[4] = {ph[0], ph[1], ph[2], ph[3]}, al[4] = {pl[0], pl[1], pl[2], pl[3]};
#pragma unroll
                for (int np = 0; np < 8; ++np) {
                    uint32_t b[4];
                    tc_ldsm_t(b, sV + tc_at(16 * kw + (lane & 7) + 8 * ((lane >> 3) & 1), 2 * np + (lane >> 4)));
                    tc_mma(o[2 * np], ah, b[0], b[1]);
                    tc_mma(o[2 * np], al, b[0], b[1]);
                    tc_mma(o[2 * np + 1], ah, b[2], b[3]);
                    tc_mma(o[2 * np + 1], al, b[2], b[3]);
                }
            }
            __syncthreads();   // sV free
            if (it + 1 < ntile) load(sV, vc, j0 + kTcBK);
            tc_commit();
        }
        // the rows' sums over the quad
#pragma unroll
        for (int o2 = 1; o2 <= 2; o2 <<= 1) {
            la += __shfl_xor_sync(0xffffffffu, la, o2);
            lb += __shfl_xor_sync(0xffffffffu, lb, o2);
        }
        // the 4 key warps' runs into one, an m-tile at a time through the (now free) K / V tiles; 2^-15 back out
        float* sx = reinterpret_cast<float*>(sKV);   // [3][16][128] floats = 24 KB of the 32, then m / l [2][3][16]
        float (*s_ml)[3][16] = reinterpret_cast<float (*)[3][16]>(sx + 3 * 16 * kD);
        static_assert(sizeof(sKV) >= (3 * 16 * kD + 2 * 3 * 16) * sizeof(float), "the merge's scratch");
        for (int m = 0; m < MT; ++m) {
            __syncthreads();
            if (mt == m && kw > 0) {
                float* dst = sx + (size_t) (kw - 1) * 16 * kD;
                const int r0 = lane >> 2;
#pragma unroll
                for (int n = 0; n < 16; ++n) {
                    *reinterpret_cast<float2*>(dst + r0 * kD + 8 * n + qd) = make_float2(o[n][0], o[n][1]);
                    *reinterpret_cast<float2*>(dst + (r0 + 8) * kD + 8 * n + qd) = make_float2(o[n][2], o[n][3]);
                }
                if ((lane & 3) == 0) {
                    s_ml[0][kw - 1][r0] = ma;
                    s_ml[0][kw - 1][r0 + 8] = mb;
                    s_ml[1][kw - 1][r0] = la;
                    s_ml[1][kw - 1][r0 + 8] = lb;
                }
            }
            __syncthreads();
            if (mt == m && kw == 0) {
#pragma unroll
                for (int hr = 0; hr < 2; ++hr) {
                    if (!(hr == 0 ? ina : inb)) continue;
                    const int r0 = (lane >> 2) + 8 * hr;
                    const float m0 = hr == 0 ? ma : mb, l0 = hr == 0 ? la : lb;
                    float M = m0;
                    for (int w = 0; w < 3; ++w) M = fmaxf(M, s_ml[0][w][r0]);
                    float f[4] = {0.0f, 0.0f, 0.0f, 0.0f}, L = 0.0f;
                    if (M > -INFINITY) {
                        f[0] = expf(m0 - M);
                        L = l0 * f[0];
                        for (int w = 0; w < 3; ++w) {
                            f[w + 1] = expf(s_ml[0][w][r0] - M);
                            L += s_ml[1][w][r0] * f[w + 1];
                        }
                    }
                    const int t = (hr == 0 ? ta : tb), h = hr == 0 ? ha : hb;
                    float* pr = part + (((size_t) (rt0 + t) * H + (size_t) h) * max_splits + split) * kPart;
#pragma unroll
                    for (int n = 0; n < 16; ++n)
#pragma unroll
                        for (int e = 0; e < 2; ++e) {
                            const int d = 8 * n + qd + e;
                            float A = o[n][2 * hr + e] * f[0];
                            for (int w = 0; w < 3; ++w) A += sx[(size_t) w * 16 * kD + r0 * kD + d] * f[w + 1];
                            pr[2 + d] = A * (1.0f / 32768.0f);
                        }
                    if ((lane & 3) == 0) {
                        pr[0] = M;
                        pr[1] = L;
                    }
                }
            }
        }
        __syncthreads();   // (the scratch free before a second pass's loads)
    }
}
#endif  // STRATA_INK_TC != 0

// ---------------------------------------------------------------- elementwise
__global__ void gate_kernel(float* __restrict__ o, int64_t os, const float* __restrict__ g, int64_t gs, int n) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = g[(int64_t) t * gs + i];
    o[(int64_t) t * os + i] *= 1.0f / (1.0f + expf(-gv));
}
__global__ void add_kernel(float* __restrict__ h, int64_t hs, const float* __restrict__ a, int64_t as, int n) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    h[(int64_t) t * hs + i] += a[(int64_t) t * as + i];
}
__global__ void softcap_kernel(float* __restrict__ x, int64_t n, float cap) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = tanhf(x[i] * (1.0f / cap)) * cap;
}

// ---------------------------------------------------------------- the decode token's fused steps (T = 1)
// Each is the same arithmetic as the kernels it stands for, element for element and sum for sum: glm_fast.cu's
// rms_q8_kernel<4> (one block of 1024, thread tid the elements tid + 1024 i, the warps' sums added in order) and
// q8_1_store_warp, and this file's mix_part / mix_out (a chunk's 256 threads), rms_row (256 threads), conv_channel,
// add_kernel, ctrl_apply_kernel, attn_combine_kernel, gate_kernel.
constexpr int kTokK = 4;                        // rms_q8_kernel<4>: n_embd <= 4096
constexpr int kTokChunks = 4096 / kMixChunk;    // the mix's chunks at most

struct Q81 {   // ggml's block_q8_1
    __half2 ds;
    int8_t qs[32];
};
// glm_fast.cu's block_sum (rms_q8_kernel's)
__device__ __forceinline__ float gblock_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += sh[i];
    return t;
}
// glm_fast.cu's q8_1_store_warp
__device__ __forceinline__ void q81_store_warp(float v, Q81* dst, int lane) {
    const float amax = warp_max(fabsf(v));
    const float sum = warp_sum(v);
    const float d = fminf(amax / 127.0f, 65504.0f);
    const int8_t q = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(v / d)));
    dst->qs[lane] = q;
    if (lane == 0) dst->ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
}
// rms_q8_kernel<4>'s row from v[i] = the element tid + 1024 i (0 past n): x = rms * w, its q8_1
__device__ __forceinline__ void rms_q8_tok(const float (&v)[kTokK], const float* __restrict__ w, float eps, int n,
                                           float* __restrict__ x, Q81* __restrict__ xq, float* sh) {
    const int tid = threadIdx.x, lane = tid & 31;
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < kTokK; ++i) ss += v[i] * v[i];
    ss = gblock_sum(ss, sh);
    const float inv = rsqrtf(ss / (float) n + eps);
#pragma unroll
    for (int i = 0; i < kTokK; ++i) {
        const int e = tid + 1024 * i;
        if (e >= n) break;
        const float xv = v[i] * inv * w[e];
        x[e] = xv;
        q81_store_warp(xv, xq + e / 32, lane);
    }
}
// rms_row's row (no gamma) on the 256 threads of group g (lt their index): this file's block_sum over the group
__device__ __forceinline__ void rms_row_group(const float* __restrict__ xr, float eps, int n, float* __restrict__ yr,
                                              bool on, int g, int lt, float (*shg)[32]) {
    const int lane = lt & 31, warp = lt >> 5;
    float ss = 0.0f;
    if (on)
        for (int i = lt; i < n; i += 256) ss += xr[i] * xr[i];
    ss = warp_sum(ss);
    __syncthreads();
    if (lane == 0) shg[g][warp] = ss;
    __syncthreads();
    float t = lane < 8 ? shg[g][lane] : 0.0f;
    t = warp_sum(t);
    const float inv = rsqrtf(t / (float) n + eps);
    if (on)
        for (int i = lt; i < n; i += 256) yr[i] = xr[i] * inv;
}

// two launches (PDL lets the second start as the first ends):
// mix_tok_part_kernel, grid chunks x 256: mix_part_kernel's chunk (the pre step first on the chunk's channels; the sums
// past the states / queries are zeros nobody reads: skipped)
__global__ void __launch_bounds__(kMixChunk) mix_tok_part_kernel(MixTok a, int E) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    __shared__ float wp[kMixChunk / 32][kMixSums];
    const int c = blockIdx.x, S = a.st.n, nq = a.nq;
    const int i = c * kMixChunk + threadIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t = blockIdx.y;   // (the verify window's row, a.rows of them; a token: 0)
    if (a.pre_in != nullptr && i < E && a.pre_bs != nullptr) {   // (the batched decode: mix_pre's step, row t)
        float* st = a.pre_bs[t] + a.pre_bso + (size_t) i * 3;
        const float x0 = st[0], x1 = st[1], x2 = st[2], x3 = a.pre_in[(int64_t) t * E + i];
        const float* w = a.pre_w + (size_t) i * 4;
        const float v = conv_step(x0, x1, x2, x3, w[0], w[1], w[2], w[3], nullptr, 1.0f / sqrtf(5.0f));
        st[0] = x1;
        st[1] = x2;
        st[2] = x3;
        float* o = a.pre_out + (int64_t) t * E + i;
        *o = a.pre_add ? *o + v : v;
    } else if (a.pre_in != nullptr && i < E) {
        if (a.rows > 1 || a.pre_rows) {   // (the window's conv_win: rows 0 .. t over the state left as it is)
            const float v = conv_row_keep(a.pre_in, E, a.pre_w, a.pre_state, i, t);
            if (a.pre_add) a.pre_out[(int64_t) t * E + i] += v;   // (add_rows' add of conv_win's row)
            else a.pre_out[(int64_t) t * E + i] = v;
        } else {
            conv_channel(a.pre_in, E, a.pre_w, nullptr, 0, E, a.pre_state, i, 1, a.pre_add ? a.pre_tmp : a.pre_out, E);
            if (a.pre_add) a.pre_out[i] += a.pre_tmp[i];
        }
    }
    float acc[kMixSums];
#pragma unroll
    for (int k = 0; k < kMixSums; ++k) acc[k] = 0.0f;
    if (i < E) {
        float qv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) qv[q] = q < nq ? a.query[(size_t) q * E + i] : 0.0f;
#pragma unroll
        for (int s = 0; s < kMaxStates; ++s) {
            if (s >= S) break;
            const float v = a.st.p[s][(int64_t) t * a.st.stride + i];
            acc[s] = v * v;
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[kMaxStates + q * kMaxStates + s] = qv[q] * v;
        }
    }
#pragma unroll
    for (int k = 0; k < kMixSums; ++k) {
        const int s = k % kMaxStates, q = k / kMaxStates - 1;
        if (s >= S || q >= nq) continue;   // (uniform)
        const float v = warp_sum(acc[k]);
        if (lane == 0) wp[warp][k] = v;
    }
    __syncthreads();
    if (threadIdx.x < kMixSums) {
        const int s = threadIdx.x % kMaxStates, q = threadIdx.x / kMaxStates - 1;
        if (s < S && q < nq) {
            float sm = 0.0f;
#pragma unroll
            for (int w = 0; w < kMixChunk / 32; ++w) sm += wp[w][threadIdx.x];
            a.part[((size_t) t * gridDim.x + c) * kMixSums + threadIdx.x] = sm;
        }
    }
}
// mix_tok_rows_kernel, grid jobs x 1024: mix_out_kernel's sums and softmax, then job j's row - norm mode: the mix's
// row q (norm_rs ? j : 0) and its rms_q8 with gamma[j]; cat mode: job 2 c (+ 1) the rms of controller c's cur row
// (the mix's row cat_cs ? c : 0) or msg row; neither: the mix's row j.  The mix's rows are written once each.
__global__ void __launch_bounds__(1024) mix_tok_rows_kernel(MixTok a, int E) {
    STRATA_PDL_ENTRY();
    __shared__ float red[kMixSums];
    __shared__ float w[4][kMaxStates];
    __shared__ float inv[kMaxStates];
    __shared__ float sh[32];
    __shared__ float shg[4][32];
    const int tid = threadIdx.x, j = blockIdx.x, t = blockIdx.y;   // (t: the window's row)
    const int S = a.st.n, nq = a.nq;
    const int nc = (E + kMixChunk - 1) / kMixChunk;
    int q = j, z = 0;   // (the mix's row this job needs; z 1: a msg row, no mix)
    bool write = true;
    if (a.n_norm > 0) {
        q = a.norm_rs ? j : 0;
        write = a.norm_rs || j == 0;
    } else if (a.n_cat > 0) {
        z = j & 1;
        q = a.cat_cs ? (j >> 1) : 0;
        write = z == 0 && (a.cat_cs || j == 0);
    }
    float o[kTokK];
#pragma unroll
    for (int k = 0; k < kTokK; ++k) o[k] = 0.0f;
    if (z == 0) {
        if (tid < kMaxStates + kMaxStates) {   // (the squares, and query q's dots)
            const int k = tid < kMaxStates ? tid : kMaxStates + q * kMaxStates + (tid - kMaxStates);
            const int s = k % kMaxStates;
            float sm = 0.0f;
            if (s < S)
                for (int c = 0; c < nc; ++c) sm += a.part[((size_t) t * nc + c) * kMixSums + k];
            red[k] = sm;
        }
        __syncthreads();
        if (tid < S) inv[tid] = rsqrtf(red[tid] / (float) E + 1e-6f);
        __syncthreads();
        if (tid == 0) {   // (mix_out_kernel's thread 0, query q's row)
            const float sc = 1.0f / sqrtf((float) E);
            float m = -INFINITY;
            float l[kMaxStates];
            for (int s = 0; s < S; ++s) {
                l[s] = red[kMaxStates + q * kMaxStates + s] * inv[s] * sc;
                m = fmaxf(m, l[s]);
            }
            float zz = 0.0f;
            for (int s = 0; s < S; ++s) {
                l[s] = expf(l[s] - m);
                zz += l[s];
            }
            for (int s = 0; s < S; ++s) w[0][s] = l[s] / zz;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kTokK; ++k) {
            const int e = tid + 1024 * k;
            if (e >= E) continue;
            float v[kMaxStates];
#pragma unroll
            for (int s = 0; s < kMaxStates; ++s) v[s] = s < S ? a.st.p[s][(int64_t) t * a.st.stride + e] * inv[s] : 0.0f;
            float oo = 0.0f;
#pragma unroll
            for (int s = 0; s < kMaxStates; ++s)
                if (s < S) oo += w[0][s] * v[s];
            o[k] = oo;
            if (write) a.out[(int64_t) t * a.out_rs + (size_t) q * E + e] = oo;
        }
    }
    if (a.n_norm > 0) {
        rms_q8_tok(o, a.gamma[j], a.eps, E, a.x[j] + (int64_t) t * E, (Q81*) a.xq[j] + (int64_t) t * (E / 32), sh);
    } else if (a.n_cat > 0) {
        // rms_row's 256-thread layout on group 0, the row from shared memory... no: from the values just computed,
        // through the block's scratch (the mix's row is out's row q only when written by this block)
        __shared__ float row[1024 * kTokK];
        const float* xr;
        if (z == 0) {
#pragma unroll
            for (int k = 0; k < kTokK; ++k) {
                const int e = tid + 1024 * k;
                if (e < E) row[e] = o[k];
            }
            xr = row;
        } else {
            xr = a.msg + (size_t) (j >> 1) * E;
        }
        __syncthreads();
        const int c = j >> 1;
        rms_row_group(xr, 1e-6f, E, a.cat + (size_t) c * 2 * E + (size_t) z * E, tid < 256, tid >> 8, tid & 255, shg);
    }
}

// a value the grid's other blocks may have written in this launch: through L2
__device__ __forceinline__ float ld_cg(const float* p) {
#if defined(__HIP_PLATFORM_AMD__)
    return *(const volatile float*) p;
#else
    return __ldcg(p);
#endif
}
// mix_tok_fused_kernel's pre step on element i of row t: mix_tok_part_kernel's (conv_channel's T = 1 step, or
// conv_row_keep's row; then the add), the same arithmetic; returns the value written to pre_out
__device__ __forceinline__ float mix_pre(const MixTok& a, int E, int i, int t) {
    if (a.pre_bs != nullptr) {   // (the batched decode: row t its own conversation's step - the token path's, below)
        float* st = a.pre_bs[t] + a.pre_bso + (size_t) i * 3;
        const float x0 = st[0], x1 = st[1], x2 = st[2], x3 = a.pre_in[(int64_t) t * E + i];
        const float* w = a.pre_w + (size_t) i * 4;
        const float v = conv_step(x0, x1, x2, x3, w[0], w[1], w[2], w[3], nullptr, 1.0f / sqrtf(5.0f));
        st[0] = x1;
        st[1] = x2;
        st[2] = x3;
        float* o = a.pre_out + (int64_t) t * E + i;
        const float h = a.pre_add ? *o + v : v;
        *o = h;
        return h;
    }
    if (a.rows > 1 || a.pre_rows) {
        const float v = conv_row_keep(a.pre_in, E, a.pre_w, a.pre_state, i, t);
        float* o = a.pre_out + (int64_t) t * E + i;
        const float h = a.pre_add ? *o + v : v;
        *o = h;
        return h;
    }
    float* st = a.pre_state + (size_t) i * 3;
    const float x0 = st[0], x1 = st[1], x2 = st[2], x3 = a.pre_in[i];
    const float* w = a.pre_w + (size_t) i * 4;
    const float v = conv_step(x0, x1, x2, x3, w[0], w[1], w[2], w[3], nullptr, 1.0f / sqrtf(5.0f));
    st[0] = x1;
    st[1] = x2;
    st[2] = x3;
    if (!a.pre_add) {
        a.pre_out[i] = v;
        return v;
    }
    a.pre_tmp[i] = v;
    const float h = a.pre_out[i] + v;
    a.pre_out[i] = h;
    return h;
}

// warp_sum's result (lane 0's) from the 32 lanes' values in one thread: after each xor step o = 16 .. 1 lane i < o
// holds its value plus lane i + o's - the same adds (the products before them rounded: no contraction)
__device__ __forceinline__ float lane0_tree(float (&p)[32]) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
        for (int i = 0; i < o; ++i) p[i] = __fadd_rn(p[i], p[i + o]);
    return p[0];
}

// gblock_sum for a block of 1,024 (the 32 warps' sums added in order)
__device__ __forceinline__ float gblock_sum32(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float r[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) r[i] = sh[i];
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) t += r[i];
    return t;
}

// the cross-block counters (thread 0 of a block, after a __syncthreads over the block's writes): add with release
// (returns the old value), spin with acquire until *c reaches n - the grid's few blocks are all resident at once
__device__ __forceinline__ unsigned add_release(unsigned* c, unsigned v) {
#if defined(__HIP_PLATFORM_AMD__)
    __threadfence();
    return atomicAdd(c, v);
#else
    unsigned old;
    asm volatile("atom.add.release.gpu.global.u32 %0, [%1], %2;" : "=r"(old) : "l"(c), "r"(v) : "memory");
    return old;
#endif
}
__device__ __forceinline__ void spin_until(const unsigned* c, unsigned n) {
#if defined(__HIP_PLATFORM_AMD__)
    while (*(const volatile unsigned*) c < n) __builtin_amdgcn_s_sleep(1);
    __threadfence();
#else
    for (;;) {
        unsigned v;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(c) : "memory");
        if (v >= n) break;
        __nanosleep(20);
    }
#endif
}

// grid (chunks, rows) x 1024, one launch for mix_tok_part_kernel + mix_tok_rows_kernel, every sum in their order.
// A: chunk block c, element c 256 + tid on thread tid < 256: the pre step, its states and queries into xs; then a
//    thread a (warp of the chunk, sum) adds that warp's 32 products as warp_sum would (lane0_tree), the 8 warps' sums
//    to part.  B (every block, once all chunks are in - cnt[t]): the chunks' sums added in order, the softmax of each
//    row's query (mix_out_kernel's thread 0 code, a state a thread), the weighted sums of the chunk's elements (from
//    xs) to out.  C (the row's last `jobs` blocks past B - cnt[8 + t] - once every row element is in): job j of
//    mix_tok_rows_kernel - the norm (rms_q8_tok; the q8_1 blocks' amax and sum a block a thread, lane0_tree) or a
//    controller input's rms (rms_row_group).  The counters go back to 0 at the end (cnt[16 + t]: the jobs past C's
//    wait).  nr: the mix's rows the jobs need (0 .. nr - 1); jobs 0: none (the rows are the output).
__global__ void __launch_bounds__(1024) mix_tok_fused_kernel(MixTok a, int E, int nr, int jobs, unsigned* __restrict__ cnt) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    __shared__ float xs[kMaxStates + 4][kMixChunk + 1];
    __shared__ float wp[kMixChunk / 32][kMixSums];
    __shared__ float pr[kTokChunks][kMixSums];
    __shared__ float red[kMixSums];
    __shared__ float lx[4][kMaxStates];
    __shared__ float w[4][kMaxStates];
    __shared__ float inv[kMaxStates];
    __shared__ float sh[32];
    __shared__ float shg[4][32];
    __shared__ float row[1024 * kTokK + 1024 * kTokK / 32];   // (norm: element e at e + e / 32 - no bank conflicts)
    __shared__ float qd[1024 * kTokK / 32], qa[1024 * kTokK / 32];
    __shared__ int job;
    const int tid = threadIdx.x, c = blockIdx.x, t = blockIdx.y, nc = gridDim.x;
    const int S = a.st.n, nq = a.nq, hs = a.pre_s;
    // ---- A
    if (tid < kMixChunk) {
        const int i = c * kMixChunk + tid;
        const bool in = i < E;
        float qv[4], sv[kMaxStates];
#pragma unroll
        for (int q = 0; q < 4; ++q) qv[q] = in && q < nq ? a.query[(size_t) q * E + i] : 0.0f;
#pragma unroll
        for (int s = 0; s < kMaxStates; ++s) sv[s] = in && s < S && s != hs ? a.st.p[s][(int64_t) t * a.st.stride + i] : 0.0f;
        if (a.pre_in != nullptr && in) {
            const float h = mix_pre(a, E, i, t);
#pragma unroll
            for (int s = 0; s < kMaxStates; ++s)
                if (s == hs) sv[s] = h;
        }
#pragma unroll
        for (int s = 0; s < kMaxStates; ++s) xs[s][tid] = sv[s];
#pragma unroll
        for (int q = 0; q < 4; ++q) xs[kMaxStates + q][tid] = qv[q];
    }
    __syncthreads();
    const int K = S * (1 + nq);   // (the sums: S squares, then S dots a query)
    if (tid < (kMixChunk / 32) * K) {
        const int ww = tid / K, kk = tid % K;
        const int s = kk % S, q = kk / S - 1;   // (q -1: the square)
        const float* xr = &xs[s][ww * 32];
        const float* yr = q < 0 ? xr : &xs[kMaxStates + q][ww * 32];
        float p[32];
#pragma unroll
        for (int l = 0; l < 32; ++l) p[l] = __fmul_rn(yr[l], xr[l]);
        wp[ww][q < 0 ? s : kMaxStates + q * kMaxStates + s] = lane0_tree(p);
    }
    __syncthreads();
    if (tid < kMixSums) {
        const int s = tid % kMaxStates, q = tid / kMaxStates - 1;
        if (s < S && q < nq) {
            float sm = 0.0f;
#pragma unroll
            for (int k = 0; k < kMixChunk / 32; ++k) sm += wp[k][tid];
            a.part[((size_t) t * nc + c) * kMixSums + tid] = sm;
        }
    }
    __syncthreads();
    // ---- B
    if (tid == 0) {
        add_release(&cnt[t], 1u);
        spin_until(&cnt[t], (unsigned) nc);
    }
    __syncthreads();
    if (tid < nc * kMixSums) {   // (every chunk's sums at once)
        const int cc = tid / kMixSums, k = tid % kMixSums;
        const int s = k % kMaxStates, q = k / kMaxStates - 1;
        if (s < S && q < nr) pr[cc][k] = ld_cg(a.part + ((size_t) t * nc + cc) * kMixSums + k);
    }
    __syncthreads();
    if (tid < kMixSums) {   // (mix_tok_rows_kernel's sums over the chunks)
        const int s = tid % kMaxStates, q = tid / kMaxStates - 1;
        if (s < S && q < nr) {
            float sm = 0.0f;
            for (int k = 0; k < nc; ++k) sm += pr[k][tid];
            red[tid] = sm;
            if (q < 0) inv[s] = rsqrtf(sm / (float) E + 1e-6f);
        }
    }
    __syncthreads();
    const int qs_ = tid / kMaxStates, ss_ = tid % kMaxStates;   // (the softmax: query qs_'s state ss_)
    const bool sm_on = qs_ < nr && ss_ < S;
    if (sm_on) lx[qs_][ss_] = red[kMaxStates + qs_ * kMaxStates + ss_] * inv[ss_] * (1.0f / sqrtf((float) E));
    __syncthreads();
    float ex = 0.0f;
    if (sm_on) {
        float m = -INFINITY;
        for (int s = 0; s < S; ++s) m = fmaxf(m, lx[qs_][s]);
        ex = expf(lx[qs_][ss_] - m);
    }
    __syncthreads();
    if (sm_on) lx[qs_][ss_] = ex;
    __syncthreads();
    if (sm_on) {
        float zz = 0.0f;
        for (int s = 0; s < S; ++s) zz += lx[qs_][s];
        w[qs_][ss_] = ex / zz;
    }
    __syncthreads();
    if (tid < kMixChunk) {   // (mix_tok_rows_kernel's weighted sums, the chunk's elements)
        const int i = c * kMixChunk + tid;
        if (i < E) {
            float v[kMaxStates];
#pragma unroll
            for (int s = 0; s < kMaxStates; ++s) v[s] = s < S ? xs[s][tid] * inv[s] : 0.0f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                if (q >= nr) break;
                float oo = 0.0f;
#pragma unroll
                for (int s = 0; s < kMaxStates; ++s)
                    if (s < S) oo += w[q][s] * v[s];
                a.out[(int64_t) t * a.out_rs + (size_t) q * E + i] = oo;
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        const unsigned tk = add_release(&cnt[8 + t], 1u);
        if (tk == (unsigned) (nc - 1)) {   // (every block is past B's wait)
            cnt[t] = 0u;
            if (jobs == 0) cnt[8 + t] = 0u;
        }
        job = (int) tk - (nc - jobs);
    }
    __syncthreads();
    const int j = job;
    if (j < 0) return;
    // ---- C: mix_tok_rows_kernel's job j
    int q = j, z = 0;
    if (a.n_norm > 0) {
        q = a.norm_rs ? j : 0;
    } else {
        z = j & 1;
        q = a.cat_cs ? (j >> 1) : 0;
    }
    float gv[kTokK];
#pragma unroll
    for (int k = 0; k < kTokK; ++k) {
        const int e = tid + 1024 * k;
        gv[k] = a.n_norm > 0 && e < E ? a.gamma[j][e] : 0.0f;
    }
    if (tid == 0) {
        spin_until(&cnt[8 + t], (unsigned) nc);
        if (atomicAdd(&cnt[16 + t], 1u) == (unsigned) (jobs - 1)) {   // (every job is past the wait)
            cnt[8 + t] = 0u;
            cnt[16 + t] = 0u;
        }
    }
    __syncthreads();
    float o[kTokK];
#pragma unroll
    for (int k = 0; k < kTokK; ++k) {
        const int e = tid + 1024 * k;
        o[k] = z == 0 && e < E ? ld_cg(a.out + (int64_t) t * a.out_rs + (size_t) q * E + e) : 0.0f;
    }
    if (a.n_norm > 0) {   // rms_q8_tok
        float ss = 0.0f;
#pragma unroll
        for (int k = 0; k < kTokK; ++k) ss += o[k] * o[k];
        ss = gblock_sum32(ss, sh);
        const float iv = rsqrtf(ss / (float) E + a.eps);
        float* x = a.x[j] + (int64_t) t * E;
        Q81* xq = (Q81*) a.xq[j] + (int64_t) t * (E / 32);
        float xv[kTokK];
#pragma unroll
        for (int k = 0; k < kTokK; ++k) {
            const int e = tid + 1024 * k;
            xv[k] = o[k] * iv * gv[k];
            if (e < E) {
                x[e] = xv[k];
                row[e + e / 32] = xv[k];
            }
        }
        __syncthreads();
        for (int g = tid; g < E / 32; g += 1024) {   // (q8_1_store_warp's amax and sum of block g)
            float p[32], am = 0.0f;
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                p[l] = row[g * 33 + l];
                am = fmaxf(am, fabsf(p[l]));
            }
            const float sum = lane0_tree(p);
            const float d = fminf(am / 127.0f, 65504.0f);
            qd[g] = d;
            qa[g] = am;
            xq[g].ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kTokK; ++k) {
            const int e = tid + 1024 * k;
            if (e >= E) break;
            const float d = qd[e / 32];
            xq[e / 32].qs[e & 31] = qa[e / 32] == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(xv[k] / d)));
        }
    } else {   // (mix_tok_rows_kernel's controller input rms)
        const float* xr;
        if (z == 0) {
#pragma unroll
            for (int k = 0; k < kTokK; ++k) {
                const int e = tid + 1024 * k;
                if (e < E) row[e] = o[k];
            }
            xr = row;
        } else {
            xr = a.msg + (size_t) (j >> 1) * E;
        }
        __syncthreads();
        const int cc = j >> 1;
        rms_row_group(xr, 1e-6f, E, a.cat + (size_t) cc * 2 * E + (size_t) z * E, tid < 256, tid >> 8, tid & 255, shg);
    }
}

// conv_par_kernel's value of channel c, row t (no dynamic taps): the same arithmetic
__device__ __forceinline__ float conv_par_one(const float* __restrict__ in, int64_t is, const float* __restrict__ w,
                                              const float* __restrict__ state, int c, int t) {
    float xv[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int tt = t - 3 + k;
        xv[k] = tt >= 0 ? in[(int64_t) tt * is + c] : state[(size_t) c * 3 + (3 + tt)];
    }
    const float x0 = xv[0], x1 = xv[1], x2 = xv[2], x3 = xv[3];
    const float w0 = w[(size_t) c * 4 + 0], w1 = w[(size_t) c * 4 + 1], w2 = w[(size_t) c * 4 + 2], w3 = w[(size_t) c * 4 + 3];
    const float scale = 1.0f / sqrtf(5.0f);
    float o = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
    return (x3 + o) * scale;
}

// grid rows x 256 (prompts): mix_part_kernel + mix_out_kernel of row t in one block (mix_norm: + rms_rows with gamma +
// f32_to_f16; a PromptPre: conv_par_kernel's rows of pre_in into pre_out, + add_kernel's add, first), every sum in
// their order.  Chunk after chunk, element c 256 + tid on thread tid: the
// chunk's states and queries into xs, the chunk's 8 warps' sums as warp_sum would (lane0_tree, a thread a (warp,
// sum)) added warp after warp, and those added chunk after chunk (mix_out_kernel's part loop) - the states read
// once for the sums; the softmax of query q a state a thread (mix_out_kernel's thread 0 code); the weighted sums,
// thread tid's elements tid + 256 c (the states read again) - rms_rows' layout, so the norms' sums of squares are
// added on the way in its order, then its block_sum and the scaled rows.  nr: the rows (the queries).
__global__ void __launch_bounds__(kMixChunk) mix_rows_kernel(MixTok a, int E, int nr) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    __shared__ float xs[kMaxStates + 4][kMixChunk + 1];
    __shared__ float wp[kMixChunk / 32][kMixSums];
    __shared__ float red[kMixSums];
    __shared__ float lx[4][kMaxStates];
    __shared__ float w[4][kMaxStates];
    __shared__ float inv[kMaxStates];
    __shared__ float sh[3][32];
    const int tid = threadIdx.x, t = blockIdx.x, lane = tid & 31, warp = tid >> 5;
    const int S = a.st.n, nq = a.nq;
    const int nc = (E + kMixChunk - 1) / kMixChunk;
    const int K = S * (1 + nq);   // (the sums: S squares, then S dots a query)
    const int ks = tid % kMaxStates, kq = tid / kMaxStates - 1;
    const bool k_on = tid < kMixSums && ks < S && kq < nq;
    float acc = 0.0f;   // (thread k: sum k, the chunks' added in order)
    for (int c = 0; c < nc; ++c) {
        const int i = c * kMixChunk + tid;
        const bool in = i < E;
        float h = 0.0f;
        if (a.pre_in != nullptr && in) {   // (the pre step: row t of pre_out, the state pre_s)
            const float v = conv_par_one(a.pre_in, E, a.pre_w, a.pre_state, i, t);
            float* o = a.pre_out + (int64_t) t * E + i;
            h = a.pre_add ? *o + v : v;
            *o = h;
        }
#pragma unroll
        for (int s = 0; s < kMaxStates; ++s)
            xs[s][tid] = in && s < S ? (s == a.pre_s ? h : a.st.p[s][(int64_t) t * a.st.stride + i]) : 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) xs[kMaxStates + q][tid] = in && q < nq ? a.query[(size_t) q * E + i] : 0.0f;
        __syncthreads();
        for (int g = tid; g < (kMixChunk / 32) * K; g += kMixChunk) {
            const int ww = g / K, kk = g % K;
            const int s = kk % S, q = kk / S - 1;   // (q -1: the square)
            const float* xr = &xs[s][ww * 32];
            const float* yr = q < 0 ? xr : &xs[kMaxStates + q][ww * 32];
            float p[32];
#pragma unroll
            for (int l = 0; l < 32; ++l) p[l] = __fmul_rn(yr[l], xr[l]);
            wp[ww][q < 0 ? s : kMaxStates + q * kMaxStates + s] = lane0_tree(p);
        }
        __syncthreads();
        if (k_on) {
            float pc = 0.0f;
#pragma unroll
            for (int k = 0; k < kMixChunk / 32; ++k) pc += wp[k][tid];
            acc += pc;
        }
    }
    if (k_on) red[tid] = acc;
    __syncthreads();
    if (tid < S) inv[tid] = rsqrtf(red[tid] / (float) E + 1e-6f);
    __syncthreads();
    const int qs_ = tid / kMaxStates, ss_ = tid % kMaxStates;   // (the softmax: query qs_'s state ss_)
    const bool sm_on = qs_ < nr && ss_ < S;
    if (sm_on) lx[qs_][ss_] = red[kMaxStates + qs_ * kMaxStates + ss_] * inv[ss_] * (1.0f / sqrtf((float) E));
    __syncthreads();
    float ex = 0.0f;
    if (sm_on) {
        float m = -INFINITY;
        for (int s = 0; s < S; ++s) m = fmaxf(m, lx[qs_][s]);
        ex = expf(lx[qs_][ss_] - m);
    }
    __syncthreads();
    if (sm_on) lx[qs_][ss_] = ex;
    __syncthreads();
    if (sm_on) {
        float zz = 0.0f;
        for (int s = 0; s < S; ++s) zz += lx[qs_][s];
        w[qs_][ss_] = ex / zz;
    }
    __syncthreads();
    const int nn = a.n_rn;
    float ss[3] = {0.0f, 0.0f, 0.0f};   // (rms_row's sums of squares, job j's row)
    float* orow = a.out + (int64_t) t * a.out_rs;
    for (int c = 0; c < nc; ++c) {
        const int i = c * kMixChunk + tid;
        if (i >= E) break;
        float v[kMaxStates];
#pragma unroll
        for (int s = 0; s < kMaxStates; ++s) v[s] = s < S ? a.st.p[s][(int64_t) t * a.st.stride + i] * inv[s] : 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (q >= nr) break;
            float oo = 0.0f;
#pragma unroll
            for (int s = 0; s < kMaxStates; ++s)
                if (s < S) oo += w[q][s] * v[s];
            orow[(size_t) q * E + i] = oo;
#pragma unroll
            for (int j = 0; j < 3; ++j)
                if (j < nn && q == (a.rn_rs ? j : 0)) ss[j] = __fmaf_rn(oo, oo, ss[j]);   // (rms_row's FFMA: a shared
                                                                                        // oo * oo would be FMUL + FADD)
        }
    }
    if (nn == 0) return;
    // rms_row's block_sum of each job's row (the block's 256 threads)
#pragma unroll
    for (int j = 0; j < 3; ++j) ss[j] = warp_sum(ss[j]);
    if (lane == 0)
#pragma unroll
        for (int j = 0; j < 3; ++j) sh[j][warp] = ss[j];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        if (j >= nn) break;
        float tt = lane < kMixChunk / 32 ? sh[j][lane] : 0.0f;
        tt = warp_sum(tt);
        const float iv = rsqrtf(tt / (float) E + a.eps);
        const float* xr = orow + (size_t) (a.rn_rs ? j : 0) * E;
        const float* gm = a.rn_g[j];
        float* yr = a.rn_y[j] + (int64_t) t * E;
        __half* yh = a.rn_h[j] != nullptr ? (__half*) a.rn_h[j] + (int64_t) t * E : nullptr;
        for (int i = tid; i < E; i += kMixChunk) {
            const float y = xr[i] * iv * gm[i];
            yr[i] = y;
            if (yh != nullptr) yh[i] = __float2half(y);
        }
    }
}

// grid n x 1024: row j's ctrl_apply_kernel step and its rms_q8 with gamma[j]
__global__ void __launch_bounds__(1024) ctrl_tok_kernel(CtrlTok a, int E) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    __shared__ float sh[32];
    const int tid = threadIdx.x, j = blockIdx.x;
    float v[kTokK];
#pragma unroll
    for (int k = 0; k < kTokK; ++k) {
        const int e = tid + 1024 * k;
        v[k] = 0.0f;
        if (e >= E) continue;
        const float c = a.cur[(int64_t) j * a.cs + e], m = a.msg[(int64_t) j * E + e];
        const float fv = tanhf(a.f[(int64_t) j * E + e]), gv = tanhf(a.g[(int64_t) j * E + e]);
        const float y = (c - fv * c) + gv * m;
        a.out[(int64_t) j * E + e] = y;
        v[k] = y;
    }
    rms_q8_tok(v, a.gamma[j], a.eps, E, a.x[j], (Q81*) a.xq[j], sh);
}

// attn_combine_kernel's row (T 1) + gate_kernel + quantize_kernel: grid H, kD threads
__global__ void attn_combine_gq_kernel(const float* __restrict__ part, int nsplit, int max_splits,
                                       const float* __restrict__ sink, const float* __restrict__ gate,
                                       float* __restrict__ out, Q81* __restrict__ outq) {
    STRATA_PDL_ENTRY();
    const int h = blockIdx.x, d = threadIdx.x;
    const float* pr = part + (size_t) h * max_splits * kPart;
    const float sk = sink[h];
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
    const size_t i = (size_t) h * kD + d;
    float o = A / L;
    const float gv = gate[i];
    o *= 1.0f / (1.0f + expf(-gv));
    out[i] = o;
    q81_store_warp(o, outq + i / 32, d & 31);
}

// ---------------------------------------------------------------- the loop controllers' BF16 GEMVs (ctl_mv)
struct CtlBatch {
    CtlJob j[kMaxCtlJobs];
    int blk_end[kMaxCtlJobs];
    int n;
    int pf;
};
// a warp's share of a weight row into L2 (glm_fast's mv_prefetch_l2): weights only - nothing an earlier kernel
// writes - so before the PDL wait, the row's first lines arriving while the kernel before finishes.  A hint
__device__ __forceinline__ void ctl_prefetch(const void* w, size_t bytes, int lane) {
#if defined(__CUDA_ARCH__) && !defined(STRATA_USE_HIP)
    const char* c = (const char*) w;
    for (size_t o = (size_t) lane * 128; o < bytes; o += (size_t) 32 * 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(c + o));
#else
    (void) w;
    (void) bytes;
    (void) lane;
#endif
}
// a warp a row, rows (blockDim / 32) a block: glm_fast's F32 sums (row_dot_f32 / mvn_kernel_f's F32 branch) on the
// rows' BF16 halves - the four weights a step widened to the F32 values they were, the same products and adds in the
// same order, then warp_sum and mv's alpha * s + 0 (alpha 1).  IT > 0: a lane's IT steps (float4 index lane + 32 i,
// i < IT, n_in <= 128 IT) unrolled whole, every load issued before the sums (the down's 36 steps a lane: 9 trips to
// memory in a 4-step unroll).  Loading the weights into registers before the PDL wait was slower (RTX 5090: the
// controllers' GEMVs 276 -> 344 us a token).  IT 0: the plain loop
template <int NC, int IT>
__global__ void __launch_bounds__(256) ctl_mv_kernel(const __grid_constant__ CtlBatch b) {
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const CtlJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row = (bid - blk0) * (int) (blockDim.x >> 5) + (int) (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= J.n_out) return;   // (no global access before this: an exit is the trigger)
    const uint2* r4 = (const uint2*) (J.w + (size_t) row * J.n_in);
    const int n4 = J.n_in / 4;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    if constexpr (IT > 0) {
        if (b.pf) ctl_prefetch(r4, (size_t) J.n_in * 2, lane);
        STRATA_PDL_ENTRY();   // (pdl.hpp: before any read of what an earlier kernel writes)
        uint2 wr[IT];
#pragma unroll
        for (int i = 0; i < IT; ++i) {
            const int k = lane + 32 * i;
            wr[i] = k < n4 ? r4[k] : make_uint2(0u, 0u);
        }
        // (the activation loads unconditional - a step past the row reads its last float4 and adds nothing - so they
        // issue together rather than one branch at a time)
#pragma unroll
        for (int i = 0; i < IT; ++i) {
            const int k = lane + 32 * i;
            const bool in = k < n4;
            const int kk = in ? k : n4 - 1;
            const uint2 v = wr[i];
            const float w0 = __uint_as_float(v.x << 16), w1 = __uint_as_float(v.x & 0xffff0000u);
            const float w2 = __uint_as_float(v.y << 16), w3 = __uint_as_float(v.y & 0xffff0000u);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const float4 xx = ((const float4*) (J.x + (size_t) c * J.n_in))[kk];
                const float t = w0 * xx.x + w1 * xx.y + w2 * xx.z + w3 * xx.w;
                acc[c] = in ? acc[c] + t : acc[c];
            }
        }
    } else {
        if (b.pf) ctl_prefetch(r4, (size_t) J.n_in * 2, lane);
        STRATA_PDL_ENTRY();   // (pdl.hpp: before any read of what an earlier kernel writes)
#pragma unroll 4
        for (int k = lane; k < n4; k += 32) {
            const uint2 v = r4[k];
            const float w0 = __uint_as_float(v.x << 16), w1 = __uint_as_float(v.x & 0xffff0000u);
            const float w2 = __uint_as_float(v.y << 16), w3 = __uint_as_float(v.y & 0xffff0000u);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const float4 xx = ((const float4*) (J.x + (size_t) c * J.n_in))[k];
                acc[c] += w0 * xx.x + w1 * xx.y + w2 * xx.z + w3 * xx.w;
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_sum(acc[c]);
        if (lane == 0) J.y[(size_t) c * J.n_out + row] = 1.0f * v + 0.0f;
    }
}
// ... SPLIT warps a row (STRATA_INK_CTL_SPLIT, opt-in): warp p sums float4 steps [p n4 / SPLIT, (p + 1) n4 / SPLIT)
// lane-strided, then warp 0 adds the parts in order - another sum order than the F32 GEMV's (not the same bits)
template <int NC, int SPLIT>
__global__ void __launch_bounds__(256) ctl_mv_split_kernel(const __grid_constant__ CtlBatch b) {
    __shared__ float part[SPLIT][NC];
    const int bid = blockIdx.x;
    int ji = 0;
    while (ji < b.n - 1 && bid >= b.blk_end[ji]) ++ji;
    const CtlJob& J = b.j[ji];
    const int blk0 = ji ? b.blk_end[ji - 1] : 0;
    const int row = bid - blk0;   // (a block a row)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint2* r4 = (const uint2*) (J.w + (size_t) row * J.n_in);
    const int n4 = J.n_in / 4, k0 = n4 * warp / SPLIT, k1 = n4 * (warp + 1) / SPLIT;
    if (b.pf) ctl_prefetch(r4 + k0, (size_t) (k1 - k0) * 8, lane);
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any read of what an earlier kernel writes)
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
#pragma unroll 4
    for (int k = k0 + lane; k < k1; k += 32) {
        const uint2 v = r4[k];
        const float w0 = __uint_as_float(v.x << 16), w1 = __uint_as_float(v.x & 0xffff0000u);
        const float w2 = __uint_as_float(v.y << 16), w3 = __uint_as_float(v.y & 0xffff0000u);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float4 xx = ((const float4*) (J.x + (size_t) c * J.n_in))[k];
            acc[c] += w0 * xx.x + w1 * xx.y + w2 * xx.z + w3 * xx.w;
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_sum(acc[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (warp == 0 && lane < NC) {
        float t = 0.0f;
#pragma unroll
        for (int p = 0; p < SPLIT; ++p) t += part[p][lane];
        J.y[(size_t) lane * J.n_out + row] = t;
    }
}
// four F32 values -> their BF16 high halves (element 2i the low half of word i); bad: a low half was not zero
__global__ void ctl_pack_kernel(const uint4* __restrict__ x, uint2* __restrict__ y, int64_t n4, int* bad) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    const uint4 u = x[i];
    if (((u.x | u.y | u.z | u.w) & 0xffffu) != 0u) *bad = 1;
    y[i] = make_uint2((u.x >> 16) | (u.y & 0xffff0000u), (u.z >> 16) | (u.w & 0xffff0000u));
}
__global__ void ctl_unpack_kernel(const uint2* __restrict__ x, uint4* __restrict__ y, int64_t n4) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    const uint2 v = x[i];
    y[i] = make_uint4(v.x << 16, v.x & 0xffff0000u, v.y << 16, v.y & 0xffff0000u);
}

}  // namespace

bool ctl_mv(const CtlJob* jobs, int n, int nc, int rpb, cudaStream_t s, int split) {
    if (n <= 0 || n > kMaxCtlJobs || nc < 1 || (rpb != 1 && rpb != 2 && rpb != 4 && rpb != 8)) return false;
    if (split == 2 || split == 4 || split == 8) {   // (opt-in: another sum order)
        const int pf = strata::kernels::pdl_enabled() ? 1 : 0;
        for (int c0 = 0; c0 < nc; c0 += 8) {
            const int m = std::min(8, nc - c0);
            CtlBatch b{};
            int acc = 0;
            for (int i = 0; i < n; ++i) {
                b.j[i] = jobs[i];
                b.j[i].x = jobs[i].x + (size_t) c0 * jobs[i].n_in;
                b.j[i].y = jobs[i].y + (size_t) c0 * jobs[i].n_out;
                acc += jobs[i].n_out;
                b.blk_end[i] = acc;
            }
            b.n = n;
            b.pf = pf;
            const dim3 grid((unsigned) acc), block((unsigned) (32 * split));
#define INK_CTLS(SP)                                                                                        \
            switch (m) {                                                                                    \
                case 1: strata::kernels::pdl_launch(ctl_mv_split_kernel<1, SP>, grid, block, 0, s, b); break; \
                case 2: strata::kernels::pdl_launch(ctl_mv_split_kernel<2, SP>, grid, block, 0, s, b); break; \
                case 3: strata::kernels::pdl_launch(ctl_mv_split_kernel<3, SP>, grid, block, 0, s, b); break; \
                case 4: strata::kernels::pdl_launch(ctl_mv_split_kernel<4, SP>, grid, block, 0, s, b); break; \
                case 5: strata::kernels::pdl_launch(ctl_mv_split_kernel<5, SP>, grid, block, 0, s, b); break; \
                case 6: strata::kernels::pdl_launch(ctl_mv_split_kernel<6, SP>, grid, block, 0, s, b); break; \
                case 7: strata::kernels::pdl_launch(ctl_mv_split_kernel<7, SP>, grid, block, 0, s, b); break; \
                default: strata::kernels::pdl_launch(ctl_mv_split_kernel<8, SP>, grid, block, 0, s, b); break; \
            }
            if (split == 2) { INK_CTLS(2) } else if (split == 4) { INK_CTLS(4) } else { INK_CTLS(8) }
#undef INK_CTLS
        }
        check("ctl_mv (split)");
        return true;
    }
    for (int i = 0; i < n; ++i)
        if (jobs[i].n_in % 4 != 0 || jobs[i].n_out <= 0) return false;
    const int pf = strata::kernels::pdl_enabled() ? 1 : 0;
    int it = 0;
    for (int i = 0; i < n; ++i) it = std::max(it, (jobs[i].n_in / 4 + 31) / 32);
    for (int c0 = 0; c0 < nc; c0 += 8) {   // (runs of up to 8 columns: a column's sum does not depend on the run)
        const int m = std::min(8, nc - c0);
        CtlBatch b{};
        int acc = 0;
        for (int i = 0; i < n; ++i) {
            b.j[i] = jobs[i];
            b.j[i].x = jobs[i].x + (size_t) c0 * jobs[i].n_in;
            b.j[i].y = jobs[i].y + (size_t) c0 * jobs[i].n_out;
            acc += (jobs[i].n_out + rpb - 1) / rpb;
            b.blk_end[i] = acc;
        }
        b.n = n;
        b.pf = pf;
        const dim3 grid((unsigned) acc), block((unsigned) (32 * rpb));
#define INK_CTL(IT)                                                                                   \
        switch (m) {                                                                                  \
            case 1: strata::kernels::pdl_launch(ctl_mv_kernel<1, IT>, grid, block, 0, s, b); break;   \
            case 2: strata::kernels::pdl_launch(ctl_mv_kernel<2, IT>, grid, block, 0, s, b); break;   \
            case 3: strata::kernels::pdl_launch(ctl_mv_kernel<3, IT>, grid, block, 0, s, b); break;   \
            case 4: strata::kernels::pdl_launch(ctl_mv_kernel<4, IT>, grid, block, 0, s, b); break;   \
            case 5: strata::kernels::pdl_launch(ctl_mv_kernel<5, IT>, grid, block, 0, s, b); break;   \
            case 6: strata::kernels::pdl_launch(ctl_mv_kernel<6, IT>, grid, block, 0, s, b); break;   \
            case 7: strata::kernels::pdl_launch(ctl_mv_kernel<7, IT>, grid, block, 0, s, b); break;   \
            default: strata::kernels::pdl_launch(ctl_mv_kernel<8, IT>, grid, block, 0, s, b); break;  \
        }
        // (the steps a lane of the call's longest row, rounded up to a compiled count; longer: the plain loop)
        switch (it <= 1 ? 1 : it <= 2 ? 2 : it <= 4 ? 4 : it <= 8 ? 8 : it <= 16 ? 16 : it <= 36 ? 36 : 0) {
            case 1: INK_CTL(1) break;
            case 2: INK_CTL(2) break;
            case 4: INK_CTL(4) break;
            case 8: INK_CTL(8) break;
            case 16: INK_CTL(16) break;
            case 36: INK_CTL(36) break;
            default: INK_CTL(0) break;
        }
#undef INK_CTL
    }
    check("ctl_mv");
    return true;
}

void ctl_pack(const float* x, uint16_t* y, int64_t n, int* bad, cudaStream_t s) {
    const int64_t n4 = n / 4;
    if (n4 <= 0) return;
    ctl_pack_kernel<<<(unsigned) ((n4 + 255) / 256), 256, 0, s>>>((const uint4*) x, (uint2*) y, n4, bad);
    check("ctl_pack");
}

void ctl_unpack(const uint16_t* x, float* y, int64_t n, cudaStream_t s) {
    const int64_t n4 = n / 4;
    if (n4 <= 0) return;
    ctl_unpack_kernel<<<(unsigned) ((n4 + 255) / 256), 256, 0, s>>>((const uint2*) x, (uint4*) y, n4);
    check("ctl_unpack");
}

void rms_rows(const float* x, int64_t xs, const float* gamma, float eps, int n, int T, float* y, int64_t ys,
              cudaStream_t s) {
    if (T <= 0) return;
    strata::kernels::pdl_launch(rms_rows_kernel, dim3((unsigned) T), dim3(256), 0, s, x, xs, gamma, eps, n, y, ys);
    check("rms_rows");
}

// STRATA_INK_MIX1=0: the mix's two launches (part + out / rows) instead of one (HIP: always the two)
bool mix1_on() {
#if defined(__HIP_PLATFORM_AMD__)
    return false;
#else
    static const bool on = [] { const char* v = std::getenv("STRATA_INK_MIX1"); return v == nullptr || std::atoi(v) != 0; }();
    return on;
#endif
}

// mix_tok_fused_kernel's row counters (rows <= 8: A's arrivals, B's, C's jobs past the wait; back to 0 by the end of a
// launch; one stream a device)
__device__ unsigned g_mix_cnt[24];
unsigned* mix_counters() {
#if defined(__HIP_PLATFORM_AMD__)
    return nullptr;   // (mix1_on() is false there)
#else
    static unsigned* p[64] = {};
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= 64) {
        cudaGetLastError();
        return nullptr;
    }
    if (p[d] == nullptr && cudaGetSymbolAddress((void**) &p[d], g_mix_cnt) != cudaSuccess) {
        cudaGetLastError();
        p[d] = nullptr;
    }
    return p[d];
#endif
}

size_t mix_part_floats(int T, int n_embd) { return (size_t) T * ((n_embd + kMixChunk - 1) / kMixChunk) * kMixSums; }

// mix_rows_kernel's rows at least: fewer go the two launches, whose chunks a row fill the GPU better (RTX 5090: the
// same at 128 rows, the one launch +1% at 256, +3% at 512, +6% at 2,048)
constexpr int kMixRowsMin = 128;
int mix_rows_min() { return kMixRowsMin; }

// a PromptPre into the MixTok (pre_s: the state that is pre_out's rows); false: not this shape
bool prompt_pre(MixTok& a, const PromptPre* pre, int n_embd) {
    if (pre == nullptr) return true;
    a.pre_in = pre->in;
    a.pre_w = pre->w;
    a.pre_state = const_cast<float*>(pre->state);
    a.pre_out = pre->out;
    a.pre_add = pre->add;
    a.pre_s = -1;
    for (int i = 0; i < a.st.n; ++i)
        if (a.st.p[i] == pre->out && a.st.stride == n_embd) a.pre_s = i;
    return a.pre_s >= 0;
}

void mix(const States& st, const float* query, int nq, int n_embd, int T, float* out, cudaStream_t s, float* part,
         const PromptPre* pre) {
    if (T <= 0 || st.n <= 0 || st.n > kMaxStates || nq < 1 || nq > 4) {
        std::fprintf(stderr, "inkling kernels: mix: %d states, %d queries (1..%d, 1..4)\n", st.n, nq, kMaxStates);
        return;
    }
    if (mix1_on() && n_embd <= 1024 * kTokK && T >= mix_rows_min()) {   // (one launch, a block a row: mix_rows_kernel)
        MixTok a;
        a.st = st;
        a.query = query;
        a.nq = nq;
        a.out = out;
        a.rows = T;
        a.out_rs = (int64_t) nq * n_embd;
        if (prompt_pre(a, pre, n_embd)) {
            strata::kernels::pdl_launch(mix_rows_kernel, dim3((unsigned) T), dim3(kMixChunk), 0, s, a, n_embd, nq);
            if (pre != nullptr) conv_state_kernel<<<(n_embd + 127) / 128, 128, 0, s>>>(pre->in, n_embd, const_cast<float*>(pre->state), n_embd, T);
            check("mix");
            return;
        }
    }
    if (pre != nullptr) {   // (the pre step's kernels as before)
        conv_rows_par(pre->in, n_embd, pre->w, nullptr, 0, n_embd, const_cast<float*>(pre->state), n_embd, T,
                      pre->add ? pre->tmp : pre->out, n_embd, s);
        if (pre->add) add_rows(pre->out, n_embd, pre->tmp, n_embd, n_embd, T, s);
    }
    const int nc = (n_embd + kMixChunk - 1) / kMixChunk;
    if (part == nullptr) part = mix_scratch((size_t) T * nc * kMixSums);
    if (part == nullptr) {
        std::fprintf(stderr, "inkling kernels: mix: no scratch\n");
        return;
    }
    strata::kernels::pdl_launch(mix_part_kernel, dim3((unsigned) nc, (unsigned) T), dim3(kMixChunk), 0, s, st, query, nq, n_embd, part);
    strata::kernels::pdl_launch(mix_out_kernel, dim3((unsigned) nc, (unsigned) T), dim3(kMixChunk), 0, s, st, (const float*) part, nq, n_embd, out);
    check("mix");
}

bool mix_norm(const States& st, const float* query, int nq, int n_embd, int T, float* out, int n, bool rs,
              const float* const* gamma, float eps, float* const* y, void* const* y16, cudaStream_t s,
              const PromptPre* pre) {
    if (!mix1_on() || n_embd > 1024 * kTokK || n_embd % 32 != 0 || T < mix_rows_min() || st.n <= 0 || st.n > kMaxStates || nq < 1 ||
        nq > 4 || n < 1 || n > 3 || (rs && n > nq))
        return false;
    MixTok a;
    a.st = st;
    a.query = query;
    a.nq = nq;
    a.out = out;
    a.rows = T;
    a.out_rs = (int64_t) nq * n_embd;
    a.eps = eps;
    a.n_rn = n;
    a.rn_rs = rs ? 1 : 0;
    for (int j = 0; j < n; ++j) {
        a.rn_g[j] = gamma[j];
        a.rn_y[j] = y[j];
        a.rn_h[j] = y16 != nullptr ? y16[j] : nullptr;
    }
    if (!prompt_pre(a, pre, n_embd)) return false;
    strata::kernels::pdl_launch(mix_rows_kernel, dim3((unsigned) T), dim3(kMixChunk), 0, s, a, n_embd, rs ? n : 1);
    if (pre != nullptr) conv_state_kernel<<<(n_embd + 127) / 128, 128, 0, s>>>(pre->in, n_embd, const_cast<float*>(pre->state), n_embd, T);
    check("mix_norm");
    return true;
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

void conv_win(const float* in, int64_t is, const float* w, const float* dw, int64_t dws, int head_ch, float* state,
              int C, int T, float* out, int64_t os, cudaStream_t s) {
    if (T <= 0) return;
    strata::kernels::pdl_launch(conv_win_kernel, dim3((unsigned) ((C + 127) / 128)), dim3(128), 0, s, in, is, w, dw, dws, head_ch, state, C, T, out, os);
    check("conv_win");
}

void post_proj_rows(const PostProj& a, const float* kraw, const float* vraw, int T, cudaStream_t s) {
    if (T <= 0) return;
    const int blocks = a.Hkv + (2 * T * a.H + kD / 32 - 1) / (kD / 32);
    strata::kernels::pdl_launch(post_proj_rows_kernel, dim3((unsigned) blocks), dim3(kD), 0, s, a, kraw, vraw, T);
    check("post_proj_rows");
}

void post_proj_batch(const PostProj& a, const BatchRows& b, const float* kraw, const float* vraw, int T, cudaStream_t s) {
    if (T <= 0) return;
    const int blocks = a.Hkv + (2 * a.H + kD / 32 - 1) / (kD / 32);
    strata::kernels::pdl_launch(post_proj_batch_kernel, dim3((unsigned) blocks, (unsigned) T), dim3(kD), 0, s, a, b, kraw,
                                vraw);
    check("post_proj_batch");
}

void conv_batch(const float* in, int64_t is, const float* w, float* const* st, int64_t soff, int C, int T, float* out,
                int64_t os, cudaStream_t s) {
    if (T <= 0) return;
    strata::kernels::pdl_launch(conv_batch_kernel, dim3((unsigned) ((C + 127) / 128), (unsigned) T), dim3(128), 0, s, in,
                                is, w, st, soff, C, out, os);
    check("conv_batch");
}

void ctrl_cat_n(const CtrlRows& a, int n, int E, int T, cudaStream_t s) {
    if (T <= 0 || n <= 0) return;
    strata::kernels::pdl_launch(ctrl_cat_n_kernel, dim3((unsigned) T, (unsigned) n, 2), dim3(256), 0, s, a, E);
    check("ctrl_cat_n");
}

void ctrl_apply_n(const CtrlRows& a, int n, int E, int T, cudaStream_t s) {
    if (T <= 0 || n <= 0) return;
    strata::kernels::pdl_launch(ctrl_apply_n_kernel, dim3((unsigned) ((E + 255) / 256), (unsigned) T, (unsigned) n), dim3(256), 0, s, a, E);
    check("ctrl_apply_n");
}

void conv_commit(float* const* states_d, int n_slots, const float* save, size_t slot_floats, int KV, int E, int tm, int n,
                 cudaStream_t s) {
    if (n <= 0 || n_slots <= 0) return;
    const int C = 2 * KV + 2 * E;
    conv_commit_kernel<<<dim3((unsigned) ((C + 127) / 128), (unsigned) n_slots), 128, 0, s>>>(states_d, save, slot_floats,
                                                                                              KV, E, tm, n);
    check("conv_commit");
}

void post_proj(const PostProj& a, cudaStream_t s) {
    const int blocks = a.Hkv + (2 * a.H + kD / 32 - 1) / (kD / 32);
    strata::kernels::pdl_launch(post_proj_kernel, dim3(blocks), dim3(kD), 0, s, a);
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

// STRATA_INK_DECODE_ATTN=0: the token path's global layers on attn_part_kernel again (the old bits);
// STRATA_INK_DECODE_ATTN_FROM=<keys> (default 65,536): a row takes attn_dec_tc_kernel past that many keys (RTX 5090,
// decode without a drafter: 32K -2%, 64K +3.5%, 165K +10%; below it the old bits).  -1: off (AMD, a GPU before sm_80)
static int64_t dec_from() {
#if STRATA_INK_TC == 0
    return -1;
#else
    static const int64_t v = [] {
        const char* e = getenv("STRATA_INK_DECODE_ATTN");
        if (e != nullptr && std::atoi(e) == 0) return (int64_t) -1;
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        if (major < 8) return (int64_t) -1;
        const char* f = getenv("STRATA_INK_DECODE_ATTN_FROM");
        return f != nullptr ? std::max<int64_t>(0, std::atoll(f)) : (int64_t) 65536;
    }();
    return v;
#endif
}

int attn_dec_mode(int64_t p_first, int64_t p_last, int window, int G, int T) {
    const int64_t from = dec_from();
    if (from < 0 || window > 0 || G < 1 || G > 3 || T < 1 || T * G > 32) return 0;
    if (p_last + 1 <= from) return 0;
    return p_first + 1 > from ? 1 : 2;
}

// attn_dec_tc_kernel over T rows (br.pos: each row its own block, MT 1), grid x ns (attn's split grid)
static void dec_launch(const float* q, int64_t qs, const float* r, int64_t rs, const float* rel_bias, int extent,
                       const uint16_t* kc, const uint16_t* vc, int ring, int n_head, int n_head_kv, int64_t p0, int T,
                       float scale, float* part, int max_splits, const int64_t* p0d, const BatchRows& bb, int ns,
                       cudaStream_t s) {
#if STRATA_INK_TC == 0
    (void) q, (void) qs, (void) r, (void) rs, (void) rel_bias, (void) extent, (void) kc, (void) vc, (void) ring;
    (void) n_head, (void) n_head_kv, (void) p0, (void) T, (void) scale, (void) part, (void) max_splits, (void) p0d;
    (void) bb, (void) ns, (void) s;
#else
    const bool bt = bb.pos != nullptr;
    const int G = n_head / n_head_kv, MT = bt ? 1 : (T * G + 15) / 16;
    const dim3 grid((unsigned) ns, (unsigned) n_head_kv, bt ? (unsigned) T : 1u);
    const int64_t from = dec_from();
#define INK_DEC(GG, MM) strata::kernels::pdl_launch(attn_dec_tc_kernel<GG, MM>, grid, dim3(MM * 128), 0, s, q, qs, r, rs, \
        rel_bias, extent, (const __half*) kc, (const __half*) vc, ring, n_head, n_head_kv, p0, T, scale, part, \
        max_splits, from, p0d, bb)
    switch (G * 4 + MT) {
        case 1 * 4 + 1: INK_DEC(1, 1); break;
        case 1 * 4 + 2: INK_DEC(1, 2); break;
        case 2 * 4 + 1: INK_DEC(2, 1); break;
        case 2 * 4 + 2: INK_DEC(2, 2); break;
        case 3 * 4 + 1: INK_DEC(3, 1); break;
        case 3 * 4 + 2: INK_DEC(3, 2); break;
        default: std::fprintf(stderr, "inkling kernels: attn_dec: G %d MT %d\n", G, MT); return;
    }
#undef INK_DEC
#endif
}

void attn(const float* q, int64_t qs, const float* r, int64_t rs, const float* sink, int64_t ss, const float* rel_bias,
          int extent, const uint16_t* kc, const uint16_t* vc, int ring, int n_head, int n_head_kv, int64_t p0, int T,
          int window, float scale, float* part, int max_splits, float* out, int64_t os, cudaStream_t s, bool rows,
          const int64_t* p0d, const BatchRows* br) {
    if (T <= 0) return;
    const BatchRows bb = br != nullptr ? *br : BatchRows();
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
    if (rows) ns = attn_rows_grid(p1, window, max_splits);   // (the rows past a row's count leave: a few grids)
    const dim3 grid((unsigned) ns, (unsigned) n_head_kv, (unsigned) T);
    const dim3 grid_rows((unsigned) T, (unsigned) ns, (unsigned) n_head_kv);   // (kRows: a split's rows adjacent)
    // the global layers' decode rows past STRATA_INK_DECODE_ATTN_FROM keys: attn_dec_tc_kernel (mode 2: both kernels,
    // each its own rows)
    const bool bt = bb.pos != nullptr;
    const int mode = (rows || T == 1) ? attn_dec_mode(bt ? 0 : p0, p1, window, n_head / n_head_kv, T) : 0;
    const int64_t tc_from = mode == 2 ? dec_from() : -1;
    if (mode != 1) {
#define INK_ATTN(GG) do { if (rows) strata::kernels::pdl_launch(attn_part_kernel<GG, true>, grid_rows, dim3(kAttnWarps * 32), \
        0, s, q, qs, r, rs, rel_bias, extent, (const __half*) kc, (const __half*) vc, ring, n_head, n_head_kv, p0, \
        window, scale, part, max_splits, f16_env(), p0d, bb, tc_from); \
        else attn_part_kernel<GG><<<grid, kAttnWarps * 32, 0, s>>>(q, qs, r, rs, rel_bias, extent, \
        (const __half*) kc, (const __half*) vc, ring, n_head, n_head_kv, p0, window, scale, part, max_splits, f16_env()); \
        } while (0)
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
    }
    if (mode != 0) {
        dec_launch(q, qs, r, rs, rel_bias, extent, kc, vc, ring, n_head, n_head_kv, p0, T, scale, part, max_splits, p0d,
                   bb, ns, s);
        check("attn_dec_tc");
    }
    if (rows)
        strata::kernels::pdl_launch(attn_combine_kernel, dim3((unsigned) n_head, (unsigned) T), dim3(kD), 0, s,
                                    (const float*) part, ns, max_splits, sink, ss, n_head, out, os,
                                    bb.pos != nullptr ? std::max<int64_t>(p0, 0) : p0, window, p0d, bb.pos);
    else
        attn_combine_kernel<<<dim3((unsigned) n_head, (unsigned) T), kD, 0, s>>>(part, ns, max_splits, sink, ss, n_head,
                                                                                out, os);
    check("attn_combine");
}

// STRATA_INK_PREFILL_ATTN=0: the prompt path's attention on attn (the token path's kernel) again
static bool tc_env() {
    static const bool v = [] {
        const char* e = getenv("STRATA_INK_PREFILL_ATTN");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return v;
}

bool attn_prompt_on() {
#if STRATA_INK_TC == 0
    return false;
#else
    return tc_env();
#endif
}

bool attn_prompt(const float* q, int64_t qs, const float* r, int64_t rs, const float* sink, int64_t ss,
                 const float* rel_bias, int extent, const uint16_t* kc, const uint16_t* vc, int ring, int n_head,
                 int n_head_kv, int64_t p0, int T, int window, float scale, float* part, int max_splits, float* out,
                 int64_t os, cudaStream_t s) {
#if STRATA_INK_TC == 0
    (void) q, (void) qs, (void) r, (void) rs, (void) sink, (void) ss, (void) rel_bias, (void) extent, (void) kc;
    (void) vc, (void) ring, (void) n_head, (void) n_head_kv, (void) p0, (void) T, (void) window, (void) scale;
    (void) part, (void) max_splits, (void) out, (void) os, (void) s;
    return false;
#else
    if (T <= 0) return true;
    if (!tc_env() || n_head_kv <= 0 || n_head % n_head_kv != 0) return false;
    static const int nsm = [] {
        int dev = 0, major = 0, n = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return major >= 8 ? std::max(1, n) : 0;
    }();
    if (nsm == 0) return false;
    const int G = n_head / n_head_kv;
    const int MT = G <= 3 ? 2 : 1;
    const int BQ = 16 * MT;
    const int nb = (T + BQ - 1) / BQ;
    // the key splits: the blocks fill the GPU twice over, a split 4 tiles at the least
    const int64_t span = window > 0 ? std::min<int64_t>(p0 + T, (int64_t) window + BQ) : p0 + T;
    const int64_t tiles = (span + kTcBK - 1) / kTcBK;
    const int64_t want = ((int64_t) 2 * nsm + (int64_t) nb * n_head_kv - 1) / ((int64_t) nb * n_head_kv);
    const int ns = (int) std::max<int64_t>(1, std::min<int64_t>({(int64_t) max_splits, want, tiles / 4}));
    const dim3 grid((unsigned) nb, (unsigned) n_head_kv, (unsigned) ns);
#define INK_TC(GG, MM) attn_tc_kernel<GG, MM><<<grid, GG * MM * 32, 0, s>>>(q, qs, r, rs, sink, ss, rel_bias, extent, \
        (const __half*) kc, (const __half*) vc, ring, n_head, n_head_kv, p0, T, window, scale, part, max_splits, out, os)
    switch (G) {
        case 1: INK_TC(1, 2); break;
        case 2: INK_TC(2, 2); break;
        case 3: INK_TC(3, 2); break;
        case 4: INK_TC(4, 1); break;
        case 6: INK_TC(6, 1); break;
        default: return false;
    }
#undef INK_TC
    check("attn_tc");
    if (ns > 1) {
        attn_combine_kernel<<<dim3((unsigned) n_head, (unsigned) T), kD, 0, s>>>(part, ns, max_splits, sink, ss, n_head,
                                                                               out, os);
        check("attn_combine");
    }
    return true;
#endif
}

void gate_sigmoid(float* o, int64_t os, const float* g, int64_t gs, int n, int T, cudaStream_t s) {
    if (T <= 0) return;
    strata::kernels::pdl_launch(gate_kernel, dim3((unsigned) ((n + 255) / 256), (unsigned) T), dim3(256), 0, s, o, os, g, gs, n);
    check("gate_sigmoid");
}

void add_rows(float* h, int64_t hs, const float* a, int64_t as, int n, int T, cudaStream_t s) {
    if (T <= 0) return;
    strata::kernels::pdl_launch(add_kernel, dim3((unsigned) ((n + 255) / 256), (unsigned) T), dim3(256), 0, s, h, hs, a, as, n);
    check("add_rows");
}

void softcap(float* x, int64_t n, float cap, cudaStream_t s) {
    if (cap <= 0.0f || n <= 0) return;
    strata::kernels::pdl_launch(softcap_kernel, dim3((unsigned) ((n + 255) / 256)), dim3(256), 0, s, x, n, cap);
    check("softcap");
}

int attn_rows_grid(int64_t p_last, int window, int max_splits) {
    const int ns = attn_splits(p_last, window, max_splits);
    int b = 8;   // (8 at the least: a window's keys up to 1,024 one grid - a new grid is a new graph to capture)
    while (b < ns) b *= 2;
    return std::min(b, std::max(1, max_splits));
}

int attn_splits(int64_t p, int window, int max_splits) {
    const int64_t nk = window > 0 ? std::min<int64_t>(p + 1, window) : p + 1;
    return std::max(1, (int) std::min<int64_t>(max_splits, std::max<int64_t>(1, (nk + kTile - 1) / kTile)));
}

void attn_tok(const float* q, const float* r, const float* sink, const float* rel_bias, int extent, const uint16_t* kc,
              const uint16_t* vc, int ring, int n_head, int n_head_kv, int64_t p0, const int64_t* pd, int window,
              float scale, float* part, int max_splits, const float* gate, float* out, void* outq, cudaStream_t s) {
    if (n_head % n_head_kv != 0 || n_head / n_head_kv > kMaxG) {
        std::fprintf(stderr, "inkling kernels: attn_tok: %d heads over %d KV heads\n", n_head, n_head_kv);
        return;
    }
    const int ns = attn_splits(p0, window, max_splits);   // (attn's for T = 1)
    const dim3 grid((unsigned) ns, (unsigned) n_head_kv, 1u);
    const int64_t qs = (int64_t) n_head * kD, rs = (int64_t) n_head * kRel;
    if (attn_dec_mode(p0, p0, window, n_head / n_head_kv, 1) == 1) {
        dec_launch(q, qs, r, rs, rel_bias, extent, kc, vc, ring, n_head, n_head_kv, p0, 1, scale, part, max_splits, pd,
                   BatchRows(), ns, s);
        check("attn_tok dec_tc");
    } else {
#define INK_ATTN(GG) strata::kernels::pdl_launch(attn_part_kernel<GG>, grid, dim3(kAttnWarps * 32), 0, s, q, qs, r, rs, \
        rel_bias, extent, (const __half*) kc, (const __half*) vc, ring, n_head, n_head_kv, p0, window, scale, part, \
        max_splits, f16_env(), pd, BatchRows(), (int64_t) -1)
    switch (n_head / n_head_kv) {
        case 1: INK_ATTN(1); break;
        case 2: INK_ATTN(2); break;
        case 3: INK_ATTN(3); break;
        case 4: INK_ATTN(4); break;
        case 6: INK_ATTN(6); break;
        case 8: INK_ATTN(8); break;
        default: std::fprintf(stderr, "inkling kernels: attn_tok: %d q heads a KV head\n", n_head / n_head_kv); return;
    }
#undef INK_ATTN
    check("attn_tok part");
    }
    strata::kernels::pdl_launch(attn_combine_gq_kernel, dim3((unsigned) n_head), dim3(kD), 0, s, (const float*) part, ns,
                                max_splits, sink, gate, out, (Q81*) outq);
    check("attn_tok combine");
}

namespace {
__global__ void set_pos_kernel(int64_t* d, int64_t p) {
    STRATA_PDL_ENTRY();
    *d = p;
}
}  // namespace
void set_pos(int64_t* d, int64_t p, cudaStream_t s) {
    strata::kernels::pdl_launch(set_pos_kernel, dim3(1), dim3(1), 0, s, d, p);
    check("set_pos");
}

size_t mix_tok_floats() { return (size_t) kTokChunks * kMixSums; }

void mix_tok(const MixTok& a, int n_embd, cudaStream_t s) {
    if (n_embd > 1024 * kTokK || n_embd % 32 != 0 || a.st.n <= 0 || a.st.n > kMaxStates || a.nq < 1 || a.nq > 4 ||
        a.n_norm > 3 || a.n_cat > 4 || (a.n_norm > 0 && a.norm_rs && a.n_norm > a.nq) || a.part == nullptr) {
        std::fprintf(stderr, "inkling kernels: mix_tok: n_embd %d, %d states, %d queries, %d norms, %d controllers\n",
                     n_embd, a.st.n, a.nq, a.n_norm, a.n_cat);
        return;
    }
    if (a.rows < 1 || a.rows > 8 || (a.rows > 1 && (a.n_cat > 0 || a.out_rs < (int64_t) a.nq * n_embd))) {
        std::fprintf(stderr, "inkling kernels: mix_tok: %d rows (1..8, no controllers, rows nq n_embd apart)\n", a.rows);
        return;
    }
    // one launch (STRATA_INK_MIX1=0: the two launches)
    const bool one = mix1_on();
    const int jobs = a.n_norm > 0 ? a.n_norm : a.n_cat > 0 ? 2 * a.n_cat : a.nq;
    const int nc = (n_embd + kMixChunk - 1) / kMixChunk;
    // (the fused kernel's: the mix's rows the jobs read, 0 .. nr - 1, and its jobs - none when the rows are the output)
    const int nr = a.n_norm > 0 ? (a.norm_rs ? a.n_norm : 1) : a.n_cat > 0 ? (a.cat_cs ? a.n_cat : 1) : a.nq;
    const int fjobs = a.n_norm > 0 || a.n_cat > 0 ? jobs : 0;
    MixTok b = a;   // (pre_s: the state that is pre_out - row t at the same place - or -1)
    b.pre_s = -1;
    bool fits = fjobs <= nc && nr <= a.nq && nc <= kTokChunks && nc * kMixSums <= 1024;
    if (a.pre_in != nullptr)
        for (int i = 0; i < a.st.n; ++i)
            if (a.st.p[i] == a.pre_out) {
                if (a.rows > 1 && a.st.stride != n_embd) fits = false;
                else b.pre_s = i;
            }
    unsigned* cnt = one && fits ? mix_counters() : nullptr;
    if (cnt != nullptr) {
        strata::kernels::pdl_launch(mix_tok_fused_kernel, dim3((unsigned) nc, (unsigned) a.rows), dim3(1024), 0, s, b, n_embd,
                                    nr, fjobs, cnt);
        check("mix_tok");
        return;
    }
    strata::kernels::pdl_launch(mix_tok_part_kernel, dim3(nc, (unsigned) a.rows), dim3(kMixChunk), 0, s, a, n_embd);
    strata::kernels::pdl_launch(mix_tok_rows_kernel, dim3(jobs, (unsigned) a.rows), dim3(1024), 0, s, a, n_embd);
    check("mix_tok");
}

void ctrl_tok(const CtrlTok& a, int n_embd, cudaStream_t s) {
    if (n_embd > 1024 * kTokK || n_embd % 32 != 0 || a.n < 1 || a.n > 3) {
        std::fprintf(stderr, "inkling kernels: ctrl_tok: n_embd %d, %d rows\n", n_embd, a.n);
        return;
    }
    strata::kernels::pdl_launch(ctrl_tok_kernel, dim3((unsigned) a.n), dim3(1024), 0, s, a, n_embd);
    check("ctrl_tok");
}

}  // namespace strata::kernels::ink
