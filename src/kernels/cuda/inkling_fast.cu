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
    __shared__ float sh[32];
    const float* xr = x + (int64_t) blockIdx.x * xs;
    float* yr = y + (int64_t) blockIdx.x * ys;
    rms_row(xr, gamma, eps, n, yr, sh);
}
// grid (T, n, 2): controller c's row t - z 0 rms(cur), z 1 rms(msg) - into cat + c cat_c + t 2E (+ E)
__global__ void ctrl_cat_n_kernel(CtrlRows a, int E) {
    __shared__ float sh[32];
    const int t = blockIdx.x, c = blockIdx.y, z = blockIdx.z;
    const float* xr = z == 0 ? a.cur[c] + (int64_t) t * a.cs : a.msg + (int64_t) c * E + (int64_t) t * a.ms;
    rms_row(xr, nullptr, 1e-6f, E, a.cat + (int64_t) c * a.cat_c + (int64_t) t * 2 * E + (int64_t) z * E, sh);
}
// grid (E / 256, T, n): ctrl_apply_kernel's step for controller c
__global__ void ctrl_apply_n_kernel(CtrlRows a, int E) {
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
__global__ void post_proj_kernel(PostProj a) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (a.pd != nullptr) a.p = *a.pd;   // (a captured graph's token: the position from the device word)
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

// the verify window's post_proj over T rows (the state read, left; the raw k / v rows kept apart): a warp a KV head
// (its channels' convs over the rows, then each row's k clip and cache rows), a (row, q head)'s clip or a (row, rel
// head)'s - post_proj_kernel's functions
__global__ void post_proj_rows_kernel(PostProj a, const float* __restrict__ kraw, const float* __restrict__ vraw, int T) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (wid < a.Hkv) {
        const int h = wid;
        for (int d = lane; d < kD; d += 32) {
            const int c = h * kD + d;
            conv_channel<true>(kraw, a.KV, a.k_conv, a.kd, a.dws, kD, a.conv, c, T, a.kf, a.KV);
            conv_channel<true>(vraw, a.KV, a.v_conv, a.vd, a.dws, kD, a.conv + (size_t) a.KV * 3, c, T, a.vf, a.KV);
        }
        __syncwarp();
        for (int t = 0; t < T; ++t) {
            clip_head(a.kf, a.KV, nullptr, kD, a.qk_clip, a.p, 0.0f, 0.0f, t, h, lane);
            __syncwarp();
            kv_store_head(a.kf, a.KV, a.vf, a.KV, a.eps, a.Hkv, a.p, a.ring, (__half*) a.kc, (__half*) a.vc, t, h, lane);
        }
    } else if (wid < a.Hkv + T * a.H) {
        const int w = wid - a.Hkv;
        clip_head(a.qf, a.HD, a.q_clip, kD, a.qk_clip, a.p, a.lf, a.alpha, w / a.H, w % a.H, lane);
    } else if (wid < a.Hkv + 2 * T * a.H) {
        const int w = wid - a.Hkv - T * a.H;
        clip_head(a.rf, (int64_t) a.H * kRel, a.rel_clip, kRel, a.rel_clip_thr, a.p, a.lf, a.alpha, w / a.H, w % a.H, lane);
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
    int window, float scale, float* __restrict__ part, int max_splits, int f16, const int64_t* __restrict__ p0d = nullptr) {
    STRATA_PDL_ENTRY();
    const int split = blockIdx.x, kvh = blockIdx.y, t = blockIdx.z;
    constexpr int G = kG;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t p = (p0d != nullptr ? *p0d : p0) + t;   // (p0d: a captured graph's token)
    const int64_t lo = window > 0 ? (p - window + 1 > 0 ? p - window + 1 : (int64_t) 0) : 0;
    const int64_t nk = p - lo + 1;
    const int nsplit = kRows ? attn_row_splits(nk, max_splits) : (int) gridDim.x;
    if (kRows && split >= nsplit) return;
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
                                    int window = 0) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    if (rows_p0 >= 0) {
        const int64_t p = rows_p0 + t;
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
#endif  // STRATA_INK_TC != 0

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
// rms_q8_tok for nr <= 3 rows at once (each row's sums as rms_q8_tok's; their reductions share the barriers)
__device__ __forceinline__ void rms_q8_tok3(const float (&v)[3][kTokK], int nr, const float* const* w, float eps, int n,
                                            float* const* x, void* const* xq, float (*sh3)[32]) {
    const int tid = threadIdx.x, lane = tid & 31, wi = tid >> 5, nw = blockDim.x >> 5;
    float ss[3] = {0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        if (j >= nr) break;
#pragma unroll
        for (int i = 0; i < kTokK; ++i) ss[j] += v[j][i] * v[j][i];
        ss[j] = warp_sum(ss[j]);
    }
    __syncthreads();
    if (lane == 0)
        for (int j = 0; j < nr; ++j) sh3[j][wi] = ss[j];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        if (j >= nr) break;
        float t = 0.0f;
        for (int i = 0; i < nw; ++i) t += sh3[j][i];
        const float inv = rsqrtf(t / (float) n + eps);
        Q81* q = (Q81*) xq[j];
#pragma unroll
        for (int i = 0; i < kTokK; ++i) {
            const int e = tid + 1024 * i;
            if (e >= n) break;
            const float xv = v[j][i] * inv * w[j][e];
            x[j][e] = xv;
            q81_store_warp(xv, q + e / 32, lane);
        }
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

// grid ceil(chunks / 4) blocks of 1024: block b's groups g (256 threads) run mix_part_kernel's chunks 4 b + g - the
// pre step first on their own channels, the chunk's partials (the sums the mix reads: the states' squares and the nq
// queries' dots) to part; the last block to arrive adds the chunks' partials in order and runs mix_out_kernel's
// softmax and rows and the post step over the whole row
__device__ unsigned int g_mix_tok_ticket = 0;
__global__ void __launch_bounds__(1024) mix_tok_kernel(MixTok a, int E, float* __restrict__ part) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    __shared__ float wp[4][kMixChunk / 32][kMixSums];
    __shared__ float red[kMixSums];
    __shared__ float w[4][kMaxStates];
    __shared__ float inv[kMaxStates];
    __shared__ float sh3[3][32];
    __shared__ float shg[4][32];
    __shared__ bool last;
    const int tid = threadIdx.x, lane = tid & 31;
    const int S = a.st.n, nq = a.nq;
    const int nc = (E + kMixChunk - 1) / kMixChunk;
    const int g = tid >> 8, lt = tid & 255, lw = lt >> 5;
    const int c = (int) blockIdx.x * 4 + g;
    const int i = c * kMixChunk + lt;
    // ---- pre: conv_rows (T 1) into pre_tmp + add_rows, or conv_rows into pre_out - this chunk's channels
    if (a.pre_in != nullptr && c < nc && i < E) {
        conv_channel(a.pre_in, E, a.pre_w, nullptr, 0, E, a.pre_state, i, 1, a.pre_add ? a.pre_tmp : a.pre_out, E);
        if (a.pre_add) a.pre_out[i] += a.pre_tmp[i];
    }
    // ---- mix_part_kernel's chunk c (the sums past the states / queries are zeros nobody reads: skipped)
    {
        float acc[kMixSums];
#pragma unroll
        for (int k = 0; k < kMixSums; ++k) acc[k] = 0.0f;
        if (c < nc && i < E) {
            float qv[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) qv[q] = q < nq ? a.query[(size_t) q * E + i] : 0.0f;
#pragma unroll
            for (int s = 0; s < kMaxStates; ++s) {
                if (s >= S) break;
                const float v = a.st.p[s][i];
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
            if (lane == 0) wp[g][lw][k] = v;
        }
    }
    __syncthreads();
    if (c < nc && lt < kMixSums) {
        const int s = lt % kMaxStates, q = lt / kMaxStates - 1;
        if (s < S && q < nq) {
            float sm = 0.0f;
#pragma unroll
            for (int ww = 0; ww < kMixChunk / 32; ++ww) sm += wp[g][ww][lt];
            part[(size_t) c * kMixSums + lt] = sm;
        }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) last = atomicAdd(&g_mix_tok_ticket, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    if (tid == 0) g_mix_tok_ticket = 0;   // (every block has arrived: the next launch starts from 0)
    __threadfence();
    // ---- mix_out_kernel: the chunks' sums in order, the softmax, the rows
    if (tid < kMixSums) {
        const int s = tid % kMaxStates, q = tid / kMaxStates - 1;
        float sm = 0.0f;
        if (s < S && q < nq)
            for (int k = 0; k < nc; ++k) sm += __ldcg(part + (size_t) k * kMixSums + tid);
        red[tid] = sm;
    }
    __syncthreads();
    if (tid < S) inv[tid] = rsqrtf(red[tid] / (float) E + 1e-6f);
    __syncthreads();
    if (tid < nq) {   // (mix_out_kernel's thread 0, a query a thread)
        const float sc = 1.0f / sqrtf((float) E);
        {
            const int q = tid;
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
    float o[4][kTokK];
#pragma unroll
    for (int k = 0; k < kTokK; ++k) {
        const int e = tid + 1024 * k;
#pragma unroll
        for (int q = 0; q < 4; ++q) o[q][k] = 0.0f;
        if (e >= E) continue;
        float v[kMaxStates];
#pragma unroll
        for (int s = 0; s < kMaxStates; ++s) v[s] = s < S ? __ldcg(a.st.p[s] + e) * inv[s] : 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (q >= nq) break;
            float oo = 0.0f;
#pragma unroll
            for (int s = 0; s < kMaxStates; ++s)
                if (s < S) oo += w[q][s] * v[s];
            o[q][k] = oo;
            a.out[(size_t) q * E + e] = oo;
        }
    }
    // ---- post: rms_q8 rows, or ctrl_cat
    if (a.n_norm > 0) {
        float v[3][kTokK];
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int k = 0; k < kTokK; ++k) v[j][k] = a.norm_rs ? o[j][k] : o[0][k];
        rms_q8_tok3(v, a.n_norm, a.gamma, a.eps, E, a.x, a.xq, sh3);
    }
    if (a.n_cat > 0) {
        __syncthreads();   // (the mix's rows, written above by every thread)
        const int rows = 2 * a.n_cat;
        for (int r0 = 0; r0 < rows; r0 += 4) {
            const int r = r0 + g, cc = r >> 1, z = r & 1;
            const bool on = r < rows;
            const float* xr = on ? (z == 0 ? a.out + (size_t) cc * a.cat_cs : a.msg + (size_t) cc * E) : nullptr;
            rms_row_group(xr, 1e-6f, E, a.cat + (size_t) cc * 2 * E + (size_t) z * E, on, g, lt, shg);
        }
    }
}

__global__ void __launch_bounds__(1024) ctrl_tok_kernel(CtrlTok a, int E) {
    STRATA_PDL_ENTRY();   // (pdl.hpp: before any global access)
    __shared__ float sh3[3][32];
    const int tid = threadIdx.x;
    float v[3][kTokK];
#pragma unroll
    for (int j = 0; j < 3; ++j)
#pragma unroll
        for (int k = 0; k < kTokK; ++k) {
            const int e = tid + 1024 * k;
            v[j][k] = 0.0f;
            if (j >= a.n || e >= E) continue;
            const float c = a.cur[(int64_t) j * a.cs + e], m = a.msg[(int64_t) j * E + e];
            const float fv = tanhf(a.f[(int64_t) j * E + e]), gv = tanhf(a.g[(int64_t) j * E + e]);
            const float y = (c - fv * c) + gv * m;
            a.out[(int64_t) j * E + e] = y;
            v[j][k] = y;
        }
    rms_q8_tok3(v, a.n, a.gamma, a.eps, E, a.x, a.xq, sh3);
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

void conv_win(const float* in, int64_t is, const float* w, const float* dw, int64_t dws, int head_ch, float* state,
              int C, int T, float* out, int64_t os, cudaStream_t s) {
    if (T <= 0) return;
    conv_win_kernel<<<(C + 127) / 128, 128, 0, s>>>(in, is, w, dw, dws, head_ch, state, C, T, out, os);
    check("conv_win");
}

void post_proj_rows(const PostProj& a, const float* kraw, const float* vraw, int T, cudaStream_t s) {
    if (T <= 0) return;
    const int warps = a.Hkv + 2 * T * a.H;
    post_proj_rows_kernel<<<(warps + 7) / 8, 256, 0, s>>>(a, kraw, vraw, T);
    check("post_proj_rows");
}

void ctrl_cat_n(const CtrlRows& a, int n, int E, int T, cudaStream_t s) {
    if (T <= 0 || n <= 0) return;
    ctrl_cat_n_kernel<<<dim3((unsigned) T, (unsigned) n, 2), 256, 0, s>>>(a, E);
    check("ctrl_cat_n");
}

void ctrl_apply_n(const CtrlRows& a, int n, int E, int T, cudaStream_t s) {
    if (T <= 0 || n <= 0) return;
    ctrl_apply_n_kernel<<<dim3((unsigned) ((E + 255) / 256), (unsigned) T, (unsigned) n), 256, 0, s>>>(a, E);
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
    const int warps = a.Hkv + 2 * a.H;
    strata::kernels::pdl_launch(post_proj_kernel, dim3((warps + 7) / 8), dim3(256), 0, s, a);
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
          int window, float scale, float* part, int max_splits, float* out, int64_t os, cudaStream_t s, bool rows) {
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
#define INK_ATTN(GG) do { if (rows) attn_part_kernel<GG, true><<<grid, kAttnWarps * 32, 0, s>>>(q, qs, r, rs, rel_bias, \
        extent, (const __half*) kc, (const __half*) vc, ring, n_head, n_head_kv, p0, window, scale, part, max_splits, \
        f16_env()); \
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
    attn_combine_kernel<<<dim3((unsigned) n_head, (unsigned) T), kD, 0, s>>>(part, ns, max_splits, sink, ss, n_head, out, os,
                                                                            rows ? p0 : -1, window);
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
#define INK_ATTN(GG) strata::kernels::pdl_launch(attn_part_kernel<GG>, grid, dim3(kAttnWarps * 32), 0, s, q, qs, r, rs, \
        rel_bias, extent, (const __half*) kc, (const __half*) vc, ring, n_head, n_head_kv, p0, window, scale, part, \
        max_splits, f16_env(), pd)
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
        a.n_norm > 3 || a.n_cat > 4 || (a.n_norm > 0 && a.norm_rs && a.n_norm > a.nq)) {
        std::fprintf(stderr, "inkling kernels: mix_tok: n_embd %d, %d states, %d queries, %d norms, %d controllers\n",
                     n_embd, a.st.n, a.nq, a.n_norm, a.n_cat);
        return;
    }
    if (a.part == nullptr) {
        std::fprintf(stderr, "inkling kernels: mix_tok: no scratch\n");
        return;
    }
    const int nc = (n_embd + kMixChunk - 1) / kMixChunk;
    strata::kernels::pdl_launch(mix_tok_kernel, dim3((nc + 3) / 4), dim3(1024), 0, s, a, n_embd, a.part);
    check("mix_tok");
}

void ctrl_tok(const CtrlTok& a, int n_embd, cudaStream_t s) {
    if (n_embd > 1024 * kTokK || n_embd % 32 != 0 || a.n < 1 || a.n > 3) {
        std::fprintf(stderr, "inkling kernels: ctrl_tok: n_embd %d, %d rows\n", n_embd, a.n);
        return;
    }
    strata::kernels::pdl_launch(ctrl_tok_kernel, dim3(1), dim3(1024), 0, s, a, n_embd);
    check("ctrl_tok");
}

}  // namespace strata::kernels::ink
