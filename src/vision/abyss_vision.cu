// src/vision/abyss_vision.cu - abyss-vision: Abyss's own image encoder for Ling-3.2-air (inclusionAI's BailingMoeV3VL,
// the mmproj's clip.projector_type "ling3vl"), on the GPU with cuBLAS - no llama.cpp, no ggml.
//
// It speaks strata-vision's protocol (serve/server.py's Vision class starts either), so the server, the engine's
// GENI requests and the embeddings file stay the same:
//
//   abyss-vision --mmproj <mmproj.gguf> [--model <ignored>] [--gpu] [--max-tokens N] [--no-warmup] [--measure]
//   -> "READY <n_embd>", then per stdin line
//      ENC <picture> <output>  ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>"
//      QUIT
//
// <picture> is the server's prepared picture (Vision.prepare, serve/server.py): int32 {'LVI1', width, height} then
// width x height x 3 bytes RGB, already resized the HF processor's way (smart_resize to multiples of 32 within the
// token cap, PIL bicubic) - the resize stays in Python with Pillow, where the reference does it too.  The output is
// strata-vision's record: int32 {'SVE1', n_tokens, nx, ny, n_embd} then float32 [n_tokens][n_embd], row i at the
// merged grid's (x = i % nx, y = i / nx).
//
// The math (tools/vision/ling3vl_ref.py is its float64 numpy reference, written from the HF code):
//   (x / 255 - mean) / std -> 16x16 patches; the 3-D patch conv on one frame repeated twice = the two time slices'
//   kernels both applied (two GEMMs into one sum) + bias + the 48 x 48 learned position grid resampled bilinearly,
//   corners aligned -> 27 blocks: LayerNorm (eps from the file), qkv + bias, 2-D rotary (head_dim 72, NEOX halves of
//   36: frequencies 0..17 by the patch row, 18..35 by its column, theta 10000^(-2j/36)), full attention (scale
//   72^-0.5), out proj + bias; LayerNorm, fc1 + bias, GELU (tanh), fc2 + bias -> the merger's LayerNorm per patch ->
//   each 2x2 block's four patches side by side (row-major) -> Linear 4608 -> 2560 + bias, GELU (erf), Linear 2560 ->
//   2560 + bias.
// The projections run as FP16 x FP16 GEMMs with F32 accumulation (the weights are the file's FP16; the activations
// are rounded to FP16 for the GEMM only); the residual stream, LayerNorms, rotary and attention stay in F32.
//
// --measure: allocate everything for the largest picture (the token cap), encode one, print
// "MEM <bytes> <base> <total>" (this process's GPU memory, what the device had in use before, the device total), end.
// All buffers are sized for the cap at start, so what --measure reports is what an encode uses.
#include "strata/artifact/gguf_reader.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kPatch = 16, kMerge = 2, kSide = 48;   // patch, spatial merge, the learned position grid's side
constexpr int kHeadDim = 72;
__constant__ float kMean[3] = {0.48145466f, 0.4578275f, 0.40821073f};   // the processor's CLIP mean / std
__constant__ float kStd[3] = {0.26862954f, 0.26130258f, 0.27577711f};

struct Fail {
    std::string msg;
};
[[noreturn]] void fail(const std::string& m) { throw Fail{m}; }
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(e));
}
void ckb(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) fail(std::string(what) + ": cuBLAS status " + std::to_string((int) s));
}
void launched(const char* what) { ck(cudaGetLastError(), what); }

// ------------------------------------------------------------------------------------------------ kernels
__device__ __forceinline__ float block_sum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += sh[i];
    return t;
}

// patches: rgb [h][w][3] -> P [gy * gw + gx][c * 256 + ky * 16 + kx] FP16 (normalized)
__global__ void patchify_kernel(const uint8_t* __restrict__ rgb, int w, int gw, __half* __restrict__ P) {
    const int p = blockIdx.x, gy = p / gw, gx = p % gw;
    for (int i = threadIdx.x; i < 3 * kPatch * kPatch; i += blockDim.x) {
        const int c = i / (kPatch * kPatch), ky = (i / kPatch) % kPatch, kx = i % kPatch;
        const int y = gy * kPatch + ky, x = gx * kPatch + kx;
        const float v = (float) rgb[((size_t) y * w + x) * 3 + c] / 255.0f;
        P[(size_t) p * 3 * kPatch * kPatch + i] = __float2half((v - kMean[c]) / kStd[c]);
    }
}

// h [n][C] += bias + the position grid resampled (bilinear, corners aligned: src = i * (side - 1) / (n - 1))
__global__ void embed_add_kernel(float* __restrict__ h, const float* __restrict__ bias, const float* __restrict__ pe,
                                 int gh, int gw, int C) {
    const int p = blockIdx.x, gy = p / gw, gx = p % gw;
    const float sy = gh > 1 ? (float) gy * (float) (kSide - 1) / (float) (gh - 1) : 0.0f;
    const float sx = gw > 1 ? (float) gx * (float) (kSide - 1) / (float) (gw - 1) : 0.0f;
    const int y0 = (int) sy, x0 = (int) sx;
    const int y1 = min(y0 + 1, kSide - 1), x1 = min(x0 + 1, kSide - 1);
    const float dy = sy - (float) y0, dx = sx - (float) x0;
    const float w00 = (1.0f - dy) * (1.0f - dx), w01 = (1.0f - dy) * dx, w10 = dy * (1.0f - dx), w11 = dy * dx;
    const float* r00 = pe + (size_t) (y0 * kSide + x0) * C;
    const float* r01 = pe + (size_t) (y0 * kSide + x1) * C;
    const float* r10 = pe + (size_t) (y1 * kSide + x0) * C;
    const float* r11 = pe + (size_t) (y1 * kSide + x1) * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        h[(size_t) p * C + c] += bias[c] + (r00[c] * w00 + r01[c] * w01 + r10[c] * w10 + r11[c] * w11);
}

// one row a block: LayerNorm of x [C] -> FP16 y (and optionally F32 yf)
__global__ void layer_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                  float eps, int C, __half* __restrict__ y, float* __restrict__ yf) {
    __shared__ float sh[32];
    const float* r = x + (size_t) blockIdx.x * C;
    float s = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) s += r[c];
    const float mean = block_sum(s, sh) / (float) C;
    float v = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float d = r[c] - mean;
        v += d * d;
    }
    const float inv = rsqrtf(block_sum(v, sh) / (float) C + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float o = (r[c] - mean) * inv * w[c] + b[c];
        if (y) y[(size_t) blockIdx.x * C + c] = __float2half(o);
        if (yf) yf[(size_t) blockIdx.x * C + c] = o;
    }
}

// qkv [n][3C] += bias, then q and k rotated: pair (i, i + 36) of each head, frequency j = i % 18, by the row (i < 18)
// or the column
__global__ void qkv_rope_kernel(float* __restrict__ qkv, const float* __restrict__ bias, int gw, int C, int n_head) {
    const int p = blockIdx.x, gy = p / gw, gx = p % gw;
    float* r = qkv + (size_t) p * 3 * C;
    for (int c = threadIdx.x; c < 3 * C; c += blockDim.x) r[c] += bias[c];
    __syncthreads();
    constexpr int half = kHeadDim / 2, nf = half / 2;
    for (int t = threadIdx.x; t < 2 * n_head * half; t += blockDim.x) {
        const int which = t / (n_head * half), hh = (t / half) % n_head, i = t % half;
        const int j = i % nf;
        const float pos = (float) (i < nf ? gy : gx);
        const double th = (double) pos * pow(10000.0, -2.0 * (double) j / (double) half);
        const float cs = (float) cos(th), sn = (float) sin(th);
        float* v = r + (size_t) which * C + (size_t) hh * kHeadDim;
        const float a = v[i], bb = v[i + half];
        v[i] = a * cs - bb * sn;
        v[i + half] = bb * cs + a * sn;
    }
}

// a column of scores (one query against all n keys) a block: softmax(scale * s)
__global__ void softmax_kernel(float* __restrict__ s, int n, float scale) {
    __shared__ float sh[32];
    float* r = s + (size_t) blockIdx.x * n;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, r[i]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    __shared__ float mw[32];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) mw[w] = m;
    __syncthreads();
    m = -INFINITY;
    for (int i = 0; i < (int) ((blockDim.x + 31) >> 5); ++i) m = fmaxf(m, mw[i]);
    float t = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float e = expf((r[i] - m) * scale);
        r[i] = e;
        t += e;
    }
    const float inv = 1.0f / block_sum(t, sh);
    for (int i = threadIdx.x; i < n; i += blockDim.x) r[i] *= inv;
}

__global__ void to_half_kernel(const float* __restrict__ x, __half* __restrict__ y, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

__global__ void add_bias_kernel(float* __restrict__ x, const float* __restrict__ b, int C, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += b[i % C];
}

// x [n][C] + bias -> GELU -> FP16 (tanh form for the ViT's MLP, erf form for the projector)
__global__ void bias_gelu_kernel(const float* __restrict__ x, const float* __restrict__ b, int C, size_t n, int erf_form,
                                 __half* __restrict__ y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i] + b[i % C];
    float g;
    if (erf_form) g = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
    else g = 0.5f * v * (1.0f + tanhf(0.79788456080286536f * (v + 0.044715f * v * v * v)));
    y[i] = __float2half(g);
}

// the merger's input: block (by, bx) of the merged grid -> its four patches (dy, dx row-major) side by side, FP16
__global__ void merge_kernel(const float* __restrict__ x, int gw, int C, __half* __restrict__ y) {
    const int r = blockIdx.x, mw = gw / kMerge, by = r / mw, bx = r % mw;
    for (int i = threadIdx.x; i < 4 * C; i += blockDim.x) {
        const int s = i / C, c = i % C, dy = s / 2, dx = s % 2;
        const int p = (by * kMerge + dy) * gw + bx * kMerge + dx;
        y[(size_t) r * 4 * C + i] = __float2half(x[(size_t) p * C + c]);
    }
}

unsigned blocks(size_t n, unsigned t = 256) { return (unsigned) ((n + t - 1) / t); }

// ------------------------------------------------------------------------------------------------ the model
struct Dev {
    void* p = nullptr;
    size_t bytes = 0;
    Dev() = default;
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    ~Dev() { if (p) cudaFree(p); }
    void alloc(size_t n) {
        if (p) cudaFree(p);
        p = nullptr;
        bytes = n;
        ck(cudaMalloc(&p, n ? n : 16), "cudaMalloc");
    }
    template <class T> T* as() const { return static_cast<T*>(p); }
};

struct Layer {
    Dev ln1_w, ln1_b, ln2_w, ln2_b, qkv_w, qkv_b, out_w, out_b, up_w, up_b, down_w, down_b;
};

class Encoder {
public:
    Encoder(const std::string& mmproj, int cap) : f_(mmproj), cap_(cap) {
        const strata::MetaValue* pt = f_.get("clip.projector_type");
        if (!pt || pt->s != "ling3vl")
            fail("this encoder reads Ling-3.x's mmproj (clip.projector_type ling3vl), not '" + (pt ? pt->s : "") + "'");
        C_ = (int) num("clip.vision.embedding_length");
        F_ = (int) num("clip.vision.feed_forward_length");
        L_ = (int) num("clip.vision.block_count");
        H_ = (int) num("clip.vision.attention.head_count");
        E_ = (int) num("clip.vision.projection_dim");
        eps_ = (float) num("clip.vision.attention.layer_norm_epsilon");
        if ((int) num("clip.vision.patch_size") != kPatch || C_ / H_ != kHeadDim || C_ % H_ != 0)
            fail("unexpected vision geometry (patch 16, head_dim 72 are what this encoder implements)");
        if (const strata::MetaValue* m = f_.get("clip.vision.spatial_merge_size"); m && (int) m->num() != kMerge)
            fail("unexpected spatial merge size");
        ckb(cublasCreate(&blas_), "cublasCreate");
        up(patch0_, "v.patch_embd.weight", 1, (size_t) C_ * 3 * kPatch * kPatch);
        up(patch1_, "v.patch_embd.weight.1", 1, (size_t) C_ * 3 * kPatch * kPatch);
        up(patch_b_, "v.patch_embd.bias", 0, C_);
        up(pos_, "v.position_embd.weight", 0, (size_t) kSide * kSide * C_);
        layers_.reset(new Layer[(size_t) L_]);
        for (int l = 0; l < L_; ++l) {
            Layer& y = layers_[(size_t) l];
            const std::string p = "v.blk." + std::to_string(l) + ".";
            up(y.ln1_w, p + "ln1.weight", 0, C_);
            up(y.ln1_b, p + "ln1.bias", 0, C_);
            up(y.ln2_w, p + "ln2.weight", 0, C_);
            up(y.ln2_b, p + "ln2.bias", 0, C_);
            up(y.qkv_w, p + "attn_qkv.weight", 1, (size_t) 3 * C_ * C_);
            up(y.qkv_b, p + "attn_qkv.bias", 0, (size_t) 3 * C_);
            up(y.out_w, p + "attn_out.weight", 1, (size_t) C_ * C_);
            up(y.out_b, p + "attn_out.bias", 0, C_);
            up(y.up_w, p + "ffn_up.weight", 1, (size_t) F_ * C_);
            up(y.up_b, p + "ffn_up.bias", 0, F_);
            up(y.down_w, p + "ffn_down.weight", 1, (size_t) C_ * F_);
            up(y.down_b, p + "ffn_down.bias", 0, C_);
        }
        up(mnorm_w_, "mm.input_norm.weight", 0, C_);
        up(mnorm_b_, "mm.input_norm.bias", 0, C_);
        M_ = (int) (f_.find("mm.0.weight")->shape.size() > 1 ? f_.find("mm.0.weight")->shape[1] : 0);
        if (M_ <= 0 || (int) f_.find("mm.0.weight")->shape[0] != 4 * C_) fail("unexpected mm.0.weight");
        up(mm0_w_, "mm.0.weight", 1, (size_t) M_ * 4 * C_);
        up(mm0_b_, "mm.0.bias", 0, M_);
        up(mm2_w_, "mm.2.weight", 1, (size_t) E_ * M_);
        up(mm2_b_, "mm.2.bias", 0, E_);
        // the work buffers, for the largest picture (cap tokens = 4 cap patches)
        const size_t n = (size_t) cap_ * 4, Cs = (size_t) C_;
        rgb_.alloc(n * kPatch * kPatch * 3);
        h_.alloc(n * Cs * 4);
        half_.alloc(n * std::max<size_t>({(size_t) 3 * kPatch * kPatch, (size_t) F_, Cs, (size_t) M_}) * 2);   // per patch
        qkv_.alloc(n * 3 * Cs * 4);
        o_.alloc(n * Cs * 4);
        mlp_.alloc(n * (size_t) F_ * 4);
        // scores: all heads at once for a chunk of queries, within ~256 MB (and at least 64 queries)
        qc_ = (int) std::max<size_t>(64, std::min<size_t>(n, ((size_t) 256 << 20) / ((size_t) H_ * n * 4)));
        sc_.alloc((size_t) H_ * n * (size_t) qc_ * 4);
        mid_.alloc((size_t) cap_ * (size_t) M_ * 4);
        out_.alloc((size_t) cap_ * (size_t) E_ * 4);
        host_out_.resize((size_t) cap_ * (size_t) E_);
    }
    ~Encoder() { if (blas_) cublasDestroy(blas_); }

    int n_embd() const { return E_; }

    // rgb [h][w][3] (multiples of 32) -> the rows (host), the merged grid nx x ny
    const float* encode(const uint8_t* rgb, int w, int h, int& nx, int& ny) {
        if (w % (kPatch * kMerge) || h % (kPatch * kMerge) || w <= 0 || h <= 0)
            fail("the picture's sides must be multiples of 32 (" + std::to_string(w) + "x" + std::to_string(h) + ")");
        const int gw = w / kPatch, gh = h / kPatch, n = gw * gh;
        if (n / 4 > cap_)
            fail("the picture has " + std::to_string(n / 4) + " tokens, more than the cap " + std::to_string(cap_));
        nx = gw / kMerge;
        ny = gh / kMerge;
        const int C = C_, K = 3 * kPatch * kPatch;
        ck(cudaMemcpy(rgb_.p, rgb, (size_t) w * h * 3, cudaMemcpyHostToDevice), "the picture to the GPU");
        __half* hb = half_.as<__half>();
        float* hs = h_.as<float>();
        patchify_kernel<<<n, 256>>>(rgb_.as<uint8_t>(), w, gw, hb);
        launched("patchify");
        gemm(patch0_, hb, hs, C, n, K, 0.0f);
        gemm(patch1_, hb, hs, C, n, K, 1.0f);
        embed_add_kernel<<<n, 256>>>(hs, patch_b_.as<float>(), pos_.as<float>(), gh, gw, C);
        launched("embed");
        for (int l = 0; l < L_; ++l) {
            const Layer& y = layers_[(size_t) l];
            layer_norm_kernel<<<n, 256>>>(hs, y.ln1_w.as<float>(), y.ln1_b.as<float>(), eps_, C, hb, nullptr);
            launched("ln1");
            gemm(y.qkv_w, hb, qkv_.as<float>(), 3 * C, n, C, 0.0f);
            qkv_rope_kernel<<<n, 256>>>(qkv_.as<float>(), y.qkv_b.as<float>(), gw, C, H_);
            launched("qkv");
            attention(n);
            to_half_kernel<<<blocks((size_t) n * C), 256>>>(o_.as<float>(), hb, (size_t) n * C);
            launched("attn to half");
            gemm(y.out_w, hb, hs, C, n, C, 1.0f);
            add_bias_kernel<<<blocks((size_t) n * C), 256>>>(hs, y.out_b.as<float>(), C, (size_t) n * C);
            launched("out bias");
            layer_norm_kernel<<<n, 256>>>(hs, y.ln2_w.as<float>(), y.ln2_b.as<float>(), eps_, C, hb, nullptr);
            launched("ln2");
            gemm(y.up_w, hb, mlp_.as<float>(), F_, n, C, 0.0f);
            bias_gelu_kernel<<<blocks((size_t) n * F_), 256>>>(mlp_.as<float>(), y.up_b.as<float>(), F_,
                                                                (size_t) n * F_, 0, hb);
            launched("fc1");
            gemm(y.down_w, hb, hs, C, n, F_, 1.0f);
            add_bias_kernel<<<blocks((size_t) n * C), 256>>>(hs, y.down_b.as<float>(), C, (size_t) n * C);
            launched("fc2 bias");
        }
        // the merger: LayerNorm per patch (F32 into o_), 2x2 blocks side by side, the projector
        layer_norm_kernel<<<n, 256>>>(hs, mnorm_w_.as<float>(), mnorm_b_.as<float>(), eps_, C, nullptr, o_.as<float>());
        launched("merger norm");
        const int T = n / 4;
        merge_kernel<<<T, 256>>>(o_.as<float>(), gw, C, hb);
        launched("merge");
        gemm(mm0_w_, hb, mid_.as<float>(), M_, T, 4 * C, 0.0f);
        bias_gelu_kernel<<<blocks((size_t) T * M_), 256>>>(mid_.as<float>(), mm0_b_.as<float>(), M_, (size_t) T * M_, 1,
                                                           hb);
        launched("mm0");
        gemm(mm2_w_, hb, out_.as<float>(), E_, T, M_, 0.0f);
        add_bias_kernel<<<blocks((size_t) T * E_), 256>>>(out_.as<float>(), mm2_b_.as<float>(), E_, (size_t) T * E_);
        launched("mm2");
        ck(cudaMemcpy(host_out_.data(), out_.p, (size_t) T * E_ * 4, cudaMemcpyDeviceToHost), "the rows back");
        return host_out_.data();
    }

private:
    double num(const char* key) const {
        const strata::MetaValue* v = f_.get(key);
        if (!v || !v->is_num()) fail(std::string("the mmproj has no ") + key);
        return v->num();
    }

    // a tensor to the GPU: as FP16 (as_half, the file's F16 weights) or F32 (F32 vectors / tables)
    void up(Dev& d, const std::string& name, int as_half, size_t want) {
        const strata::TensorInfo* t = f_.find(name);
        if (!t) fail("the mmproj has no " + name);
        if (t->elements() != want) fail(name + ": unexpected size");
        const uint8_t* src = f_.tensor_data(*t);
        if (as_half) {
            if (t->type != 1) fail(name + ": expected F16 (an F16 mmproj file)");
            d.alloc(want * 2);
            ck(cudaMemcpy(d.p, src, want * 2, cudaMemcpyHostToDevice), name.c_str());
            return;
        }
        std::vector<float> v(want);
        if (t->type == 0) {
            std::memcpy(v.data(), src, want * 4);
        } else if (t->type == 1) {
            const uint16_t* s = reinterpret_cast<const uint16_t*>(src);
            for (size_t i = 0; i < want; ++i) {
                __half hv;
                std::memcpy(&hv, &s[i], 2);
                v[i] = __half2float(hv);
            }
        } else {
            fail(name + ": expected F32 or F16");
        }
        d.alloc(want * 4);
        ck(cudaMemcpy(d.p, v.data(), want * 4, cudaMemcpyHostToDevice), name.c_str());
    }

    // y [n][m] (F32, row-major) = x [n][k] (FP16) . W^T, W [m][k] (FP16, row-major); beta 1 adds to y
    void gemm(const Dev& W, const __half* x, float* y, int m, int n, int k, float beta) {
        const float alpha = 1.0f;
        ckb(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, W.p, CUDA_R_16F, k, x, CUDA_R_16F, k, &beta, y,
                         CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
            "a projection GEMM");
    }

    // o_ [n][C] = softmax(q k^T / sqrt(72)) v per head, F32, the queries in chunks of qc_
    void attention(int n) {
        const int C = C_, ld = 3 * C;
        const float* q = qkv_.as<float>();
        const float* k = q + C;
        const float* v = q + 2 * C;
        float* s = sc_.as<float>();
        const float one = 1.0f, zero = 0.0f;
        for (int q0 = 0; q0 < n; q0 += qc_) {
            const int qn = std::min(qc_, n - q0);
            // S_h (n keys x qn queries, column-major) = K_h^T Q_h
            ckb(cublasSgemmStridedBatched(blas_, CUBLAS_OP_T, CUBLAS_OP_N, n, qn, kHeadDim, &one, k, ld, kHeadDim,
                                          q + (size_t) q0 * ld, ld, kHeadDim, &zero, s, n, (long long) n * qn, H_),
                "attention scores");
            softmax_kernel<<<H_ * qn, 256>>>(s, n, 1.0f / std::sqrt((float) kHeadDim));
            launched("softmax");
            // O_h (72 x qn) = V_h P_h
            ckb(cublasSgemmStridedBatched(blas_, CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, qn, n, &one, v, ld, kHeadDim, s, n,
                                          (long long) n * qn, &zero, o_.as<float>() + (size_t) q0 * C, C, kHeadDim, H_),
                "attention values");
        }
    }

    strata::GgufFile f_;
    int cap_;
    int C_ = 0, F_ = 0, L_ = 0, H_ = 0, E_ = 0, M_ = 0, qc_ = 0;
    float eps_ = 1e-6f;
    cublasHandle_t blas_ = nullptr;
    Dev patch0_, patch1_, patch_b_, pos_, mnorm_w_, mnorm_b_, mm0_w_, mm0_b_, mm2_w_, mm2_b_;
    std::unique_ptr<Layer[]> layers_;
    Dev rgb_, h_, half_, qkv_, o_, mlp_, sc_, mid_, out_;
    std::vector<float> host_out_;
};

bool parse_enc(const std::string& line, std::string& img, std::string& out) {
    if (line.rfind("ENC ", 0) != 0) return false;
    const std::string rest = line.substr(4);
    const size_t sp = rest.rfind(' ');
    if (sp == std::string::npos || sp == 0) return false;
    img = rest.substr(0, sp);
    out = rest.substr(sp + 1);
    return !img.empty() && !out.empty();
}

// the server's prepared picture: int32 {'LVI1', w, h} + RGB bytes
bool read_picture(const std::string& path, std::vector<uint8_t>& rgb, int& w, int& h, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot read the picture " + path; return false; }
    int32_t hdr[3] = {0, 0, 0};
    const bool ok = std::fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == 0x3149564C && hdr[1] > 0 && hdr[2] > 0 &&
                    hdr[1] <= 16384 && hdr[2] <= 16384;
    if (!ok) {
        std::fclose(f);
        err = "not a prepared picture (abyss-vision reads the server's LVI1 files; the server resizes the image)";
        return false;
    }
    w = hdr[1];
    h = hdr[2];
    rgb.resize((size_t) w * h * 3);
    const bool full = std::fread(rgb.data(), 1, rgb.size(), f) == rgb.size();
    std::fclose(f);
    if (!full) { err = "short picture file " + path; return false; }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string mmproj;
    int cap = 4096;
    bool warm = true, measure = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--mmproj") mmproj = next();
        else if (a == "--model" || a == "--threads") (void) next();   // (strata-vision's; not needed here)
        else if (a == "--gpu" || a == "--no-flash-attn") {}           // always on the GPU, full-precision attention
        else if (a == "--max-tokens") cap = std::atoi(next().c_str());
        else if (a == "--no-warmup") warm = false;
        else if (a == "--measure") measure = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (mmproj.empty() || cap < 4) {
        std::fprintf(stderr, "usage: abyss-vision --mmproj <mmproj.gguf> [--max-tokens N] [--measure] [--no-warmup]\n");
        return 2;
    }
    size_t free0 = 0, total0 = 0;
    std::unique_ptr<Encoder> enc;
    try {
        ck(cudaSetDevice(0), "the GPU");
        ck(cudaFree(nullptr), "the GPU context");
        ck(cudaMemGetInfo(&free0, &total0), "the GPU's memory");
        enc = std::make_unique<Encoder>(mmproj, cap);
        if (warm || measure) {   // the largest picture once: the work buffers and cuBLAS's workspace in use now
            const int tw = std::max(1, (int) std::sqrt((double) cap)), th = std::max(1, cap / tw);
            const int side_w = tw * 32, side_h = th * 32;
            std::vector<uint8_t> grey((size_t) side_w * side_h * 3, 128);
            int nx = 0, ny = 0;
            enc->encode(grey.data(), side_w, side_h, nx, ny);
            std::fprintf(stderr, "abyss-vision: warmed up at %d image tokens\n", nx * ny);
        }
    } catch (const Fail& e) {
        std::printf("ERR %s\n", e.msg.c_str());
        std::fflush(stdout);
        return 1;
    } catch (const std::exception& e) {
        std::printf("ERR %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
    if (measure) {
        size_t free1 = 0, total1 = 0;
        cudaMemGetInfo(&free1, &total1);
        // (this process's context is in free0 already: base is what the device had in use, this context included)
        std::printf("MEM %zu %zu %zu\n", free0 > free1 ? free0 - free1 : (size_t) 0, total0 - free0, total0);
        std::fflush(stdout);
        return 0;
    }
    std::printf("READY %d\n", enc->n_embd());
    std::fflush(stdout);
    std::string line;
    std::vector<uint8_t> rgb;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "QUIT") break;
        std::string img, out, err;
        if (!parse_enc(line, img, out)) { std::printf("ERR expected: ENC <picture> <output>\n"); std::fflush(stdout); continue; }
        const auto t0 = std::chrono::steady_clock::now();
        int w = 0, h = 0, nx = 0, ny = 0;
        const float* rows = nullptr;
        if (read_picture(img, rgb, w, h, err)) {
            try {
                rows = enc->encode(rgb.data(), w, h, nx, ny);
            } catch (const Fail& e) {
                err = e.msg;
            }
        }
        if (err.empty()) {
            const int n = nx * ny, E = enc->n_embd();
            std::FILE* f = std::fopen(out.c_str(), "wb");
            if (!f) err = "cannot write " + out;
            if (f) {
                const int32_t hdr[5] = {0x31455653, n, nx, ny, E};
                const bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
                                std::fwrite(rows, sizeof(float) * (size_t) E, (size_t) n, f) == (size_t) n;
                std::fclose(f);
                if (!ok) err = "short write to " + out;
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (ok) std::printf("OK %d %d %d %.0f\n", n, nx, ny, ms);
            }
        }
        if (!err.empty()) std::printf("ERR %s\n", err.c_str());
        std::fflush(stdout);
    }
    return 0;
}
