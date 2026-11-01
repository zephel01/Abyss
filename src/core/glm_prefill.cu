// src/core/glm_prefill.cu - the glm5-next batched PROMPT path (Glm5Model members).
//
// The token path (glm_fast_path.cu) reads a prompt one token at a time: every weight of the model crosses the
// memory bus once per token, ~31-55 ms each.  Here a chunk of T tokens goes through one layer at a time:
//
//   * projections are GEMMs - quantized weights dequantized to FP16 into a scratch and multiplied on the tensor
//     cores (cuBLAS), BF16 weights widened to F32 (SGEMM, the router and the hyper-connections stay F32-exact);
//   * the recurrences (KDA's delta rule, the causal convs) walk the chunk inside one kernel, the DSA caches are
//     written for every position before any token of the chunk attends;
//   * the routed experts are grouped by expert: the ones RESIDENT in the VRAM tier are multiplied where they lie
//     (llama.cpp's MMQ over the layer's whole pool partition in one launch - the slot stride is MMQ-addressable,
//     glmfast::expert_stride), the others are copied from the pinned RAM tier (or read from disk) into a small ring
//     of staging groups on the copy stream while the resident ones compute.
//
// The kernels are the token path's arithmetic (strata/kernels/glm_batch.hpp); what differs is the projections'
// rounding (FP16 activations instead of q8_1), as between llama.cpp's batched and one-token paths.  The prompt's
// routing also feeds the expert tiers' LFU counts, so the decode that follows starts from this conversation's
// experts.  STRATA_GLM_NO_PREFILL=1 keeps the token-at-a-time prompt (A/B).
#include "glm_fast_state.hpp"
#include "strata/kernels/dsv4_fast.hpp"
#include "strata/kernels/dsv41_fast.hpp"

#include "strata/core/progress.hpp"
#include "strata/kernels/cpu/kq_avx512.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/glm_batch.hpp"
#include "strata/kernels/qwen35_fast.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/iq_kernels.hpp"
#if defined(STRATA_USE_HIP)
#include "strata/prefill/gemm.hpp"
#endif
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::core {

namespace gb = strata::kernels::glmb;
namespace mmq = strata::prefill::mmq;
namespace q35 = strata::kernels::q35;

namespace {

// the RAM free now (MemAvailable; Windows: the smaller of free RAM and free commit), 0 when unknown
int64_t avail_ram_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (int64_t) std::min(ms.ullAvailPhys, ms.ullAvailPageFile) : 0;
#else
    long long kb = 0;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f))
            if (std::sscanf(line, "MemAvailable: %lld kB", &kb) == 1) break;
        std::fclose(f);
    }
    return (int64_t) kb * 1024;
#endif
}

// carves 256-byte-aligned buffers off a base; with base == nullptr it only measures
struct Carve {
    uint8_t* base = nullptr;
    size_t off = 0;
    template <class Tp>
    Tp* take(size_t n) {
        Tp* p = base ? (Tp*) (base + off) : nullptr;
        off += (n * sizeof(Tp) + 255u) & ~(size_t) 255u;
        return p;
    }
};

constexpr int kSub = 256;   // the sub-batch the mixers, the dense FFN and the shared expert start from
// prestaging (PrefillState::pbuf): the buffer's slots on one GPU, the chunk from which it is carved, and the fewest
// predicted rows an expert is prestaged for (below that it may not be routed at all)
constexpr int kPreSlots = 160, kPreMinT = 1024, kPreMinRows = 4;
// the MMQ row-id table (iota) carries this many entries past T * n_exp_used: mul_mat_q loads a whole mmq_y tile of
// ids before it checks the row bounds, so a table cut exactly at the last row is read past its end by the last tile
// (a long prompt's / a split's "illegal memory access", cuda-memcheck's out-of-range read; Maya #645's IDS_PAD)
constexpr int kIdsPad = 512;
// resident experts routed by at most kLightRows rows of a chunk take the light kernels (gf::rows_experts) instead of
// MMQ (STRATA_GLM_PREFILL_LIGHT overrides, 0 = MMQ only); their {slot, r0, nr} ride the bounds array from kLightOff
constexpr int kLightRows = 32, kLightOff = 6144, kLightMax = (8192 - kLightOff) / 3;

struct KdaBufs {
    float* proj[3];
    float* conv[3];
    float *fa, *ga, *beta, *g1, *g2;
    uint16_t* out16;
};
KdaBufs carve_kda(Carve& c, size_t T, const Glm5Geometry& g) {
    KdaBufs b{};
    const size_t DI = (size_t) g.d_inner();
    for (int i = 0; i < 3; ++i) b.proj[i] = c.take<float>(T * DI);
    for (int i = 0; i < 3; ++i) b.conv[i] = c.take<float>(T * DI);
    b.fa = c.take<float>(T * (size_t) g.kda_head_dim);
    b.ga = c.take<float>(T * (size_t) g.kda_head_dim);
    b.beta = c.take<float>(T * (size_t) g.n_head);
    b.g1 = c.take<float>(T * DI);
    b.g2 = c.take<float>(T * DI);
    b.out16 = c.take<uint16_t>(T * DI);
    return b;
}

// qwen35moe's mixers over a sub-batch of T rows: the delta-net layers' projections, conv rows, gates, recurrence
// output and gated norm (then the ssm_out product's FP16 input); the attention layers' projections, q, output
struct Q35Bufs {
    float *qkv, *conv, *z, *alpha, *beta, *o, *y;
    uint16_t* y16;
    float *qf, *k, *v, *q, *ao;
    uint16_t* ao16;
};
Q35Bufs carve_q35(Carve& c, size_t T, const Glm5Geometry& g) {
    Q35Bufs b{};
    const size_t C = (size_t) g.ssm_conv_dim(), V = (size_t) g.ssm_inner, HV = (size_t) g.ssm_v_heads;
    const size_t HD = (size_t) g.n_head * g.head_dim, KV = (size_t) g.attn_head_kv * g.head_dim;
    b.qkv = c.take<float>(T * C);
    b.conv = c.take<float>(T * C);
    b.z = c.take<float>(T * V);
    b.alpha = c.take<float>(T * HV);
    b.beta = c.take<float>(T * HV);
    b.o = c.take<float>(T * V);
    b.y = c.take<float>(T * V);
    b.y16 = c.take<uint16_t>(T * V);
    b.qf = c.take<float>(T * 2 * HD);
    b.k = c.take<float>(T * KV);
    b.v = c.take<float>(T * KV);
    b.q = c.take<float>(T * HD);
    b.ao = c.take<float>(T * HD);
    b.ao16 = c.take<uint16_t>(T * HD);
    return b;
}
// qwen4exp over a sub-batch of T rows: the hyper-connection's normed streams (and FP16), its low-rank and gates; the
// PLE's embedding rows (and FP16), keys, values and the postops' scratch; the QSA indexer's raw keys and queries, the
// rows' step records and their queries' positions
struct Q4xBufs {
    float *xn, *gate, *lo, *emb, *key, *qn, *gated, *gate4, *val, *kraw, *iq;
    uint16_t *xn16, *lo16, *emb16;
    int32_t *steps, *ipos;
};
Q4xBufs carve_q4x(Carve& c, size_t T, const Glm5Geometry& g) {
    Q4xBufs b{};
    if (!g.q4x) return b;
    const size_t E = (size_t) g.n_embd, HE = (size_t) g.hc * E, LR = (size_t) g.hc_lr;
    b.xn = c.take<float>(T * HE);
    b.gate = c.take<float>(T * HE);
    b.lo = c.take<float>(T * LR);
    b.emb = c.take<float>(T * E);
    b.key = b.gate;   // (the PLE runs between mixes)
    b.qn = c.take<float>(T * HE);
    b.gated = b.xn;
    b.gate4 = c.take<float>(T * 4);
    b.val = c.take<float>(T * E);
    b.kraw = c.take<float>(T * (size_t) g.idx_key);
    b.iq = c.take<float>(T * (size_t) g.idx_heads * g.idx_key);
    b.xn16 = c.take<uint16_t>(T * HE);
    b.lo16 = c.take<uint16_t>(T * LR);
    b.emb16 = c.take<uint16_t>(T * E);
    b.steps = c.take<int32_t>(T * 4);
    b.ipos = c.take<int32_t>(T * (size_t) g.idx_heads);
    return b;
}
// MMQ reads K in whole tiles of 256 (FP4: 512) values, so with n_ff not a multiple of that (qwen4exp's 640) a down
// matrix's last row reads past the matrix: llama.cpp zeroes that padding at allocation (a garbage FP16 scale there
// times the zero activations is NaN - the prompt's FFN rows had NaN in column n_embd - 1).  The slots' tail bytes
// [off, off + n) of n_exp slots `stride` apart, zeroed before the product (the slot padding is never read otherwise)
__global__ void zero_tails(uint8_t* base, size_t stride, size_t off, int n) {
    uint8_t* p = base + (size_t) blockIdx.x * stride + off;
    for (int i = threadIdx.x; i < n; i += blockDim.x) p[i] = 0;
}
// dst row r = src row tok[r] (n floats each): the host experts' x rows, gathered before they go down
__global__ void gather_rows(const float* __restrict__ src, const int* __restrict__ tok, int rows, int n,
                            float* __restrict__ dst) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const float4* a = reinterpret_cast<const float4*>(src + (size_t) tok[r] * n);
    float4* b = reinterpret_cast<float4*>(dst + (size_t) r * n);
    for (int i = threadIdx.x; i < n / 4; i += blockDim.x) b[i] = a[i];
}
#if defined(STRATA_USE_HIP)
// The MLA batched products in FP16 with every head's tokens contiguous (hipBLAS picks far faster kernels than for the
// F32 ones).  The prompt attention has three paths: the rocWMMA kernel on FP16 q_abs (f16q), the wave32 WMMA kernel on
// F32 q_abs (wmma2), and the F32 kernel.  Both FP16 products and the WMMA attention are on by default;
// STRATA_GLM_MLA_F16=0 and STRATA_GLM_MLA_WMMA=0 turn them off (WMMA off keeps the F32 attention, F16 off also the F32
// products).  STRATA_GLM_PREFILL_ATTN=wmma2|f16q|f32 forces an attention path; unset, gfx12 (RDNA4) defaults to wmma2
// and gfx11 to f16q.
__global__ void pack_heads_f16(const float* __restrict__ src, uint16_t* __restrict__ dst, int T, int H, int K) {
    const int64_t n = (int64_t) T * H * K;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int k = (int) (i % K);
        const int64_t r = i / K;
        const int t = (int) (r % T), h = (int) (r / T);
        dst[i] = __half_as_ushort(__float2half(src[((int64_t) t * H + h) * K + k]));
    }
}
__global__ void bf16_to_f16(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = __half_as_ushort(__float2half(__uint_as_float((uint32_t) src[i] << 16)));
}
bool env_on(const char* name) {
    const char* v = getenv(name);
    return !(v && v[0] == '0');
}
bool mla_wmma_on() {
    static const bool on = env_on("STRATA_GLM_MLA_WMMA");
    return on;
}
bool mla_f16_on() {
    static const bool on = env_on("STRATA_GLM_MLA_F16") || mla_wmma_on();
    return on;
}
enum class MlaAttnPath { F32, F16Q, WMMA2 };
MlaAttnPath mla_attn_path() {
    if (!mla_wmma_on()) return MlaAttnPath::F32;   // STRATA_GLM_MLA_WMMA=0 keeps the F32 kernel as before
    static const char* mode = getenv("STRATA_GLM_PREFILL_ATTN");
    static const bool want_wmma2 = mode != nullptr && std::strcmp(mode, "wmma2") == 0;
    static const bool want_f16q = mode != nullptr && std::strcmp(mode, "f16q") == 0;
    static const bool want_f32 = mode != nullptr && std::strcmp(mode, "f32") == 0;
    if (want_f16q) return MlaAttnPath::F16Q;
    if (want_f32) return MlaAttnPath::F32;
    if (want_wmma2) return gb::mla_wmma2_supported() ? MlaAttnPath::WMMA2 : MlaAttnPath::F16Q;
    return gb::mla_wmma2_default() ? MlaAttnPath::WMMA2 : MlaAttnPath::F16Q;
}
#endif
struct DsaBufs {
    float *qr_raw, *qr, *kv_raw, *ik_raw, *ig_raw, *iw, *q, *iq, *score, *q_abs, *ctx, *attn;
    uint16_t *qr16, *attn16;
    int *cells, *n_sel;
    int score_ld;
};
DsaBufs carve_dsa(Carve& c, size_t T, const Glm5Geometry& g, int max_pools) {
    DsaBufs b{};
    b.qr_raw = c.take<float>(T * g.q_lora);
    b.qr = c.take<float>(T * g.q_lora);
    b.qr16 = c.take<uint16_t>(T * g.q_lora);
    b.kv_raw = c.take<float>(T * g.kv_lora);
    b.ik_raw = c.take<float>(T * g.idx_key);
    b.ig_raw = c.take<float>(T * g.idx_key);
    b.iw = c.take<float>(T * g.idx_heads);
    b.q = c.take<float>(T * (size_t) g.n_head * g.qk_nope);
    b.iq = c.take<float>(T * (size_t) g.idx_heads * g.idx_key);
    b.cells = c.take<int>(T * (size_t) g.n_sel_max());
    b.n_sel = c.take<int>(T);
    b.score_ld = std::max(1, max_pools);
    b.score = c.take<float>(T * (size_t) b.score_ld);
    b.q_abs = c.take<float>(T * (size_t) g.n_head * g.kv_lora);
    b.ctx = c.take<float>(T * (size_t) g.n_head * g.kv_lora);
    b.attn = c.take<float>(T * (size_t) g.n_head * g.v_head);
    b.attn16 = c.take<uint16_t>(T * (size_t) g.n_head * g.v_head);
    return b;
}

// deepseek4's attention mixer for a sub-batch of T tokens (src/kernels/cuda/dsv4_fast.cu's batched kernels)
struct Ds4Bufs {
    float *qr_raw, *qr, *kv_raw, *ckv, *csc, *ikv, *isc, *iw, *q, *iq, *o, *oa, *score;
    uint16_t *qr16, *o16, *oa16, *keys;
    int* sel;
    int score_ld, score_rows;
};
Ds4Bufs carve_ds4(Carve& c, size_t T, const Glm5Geometry& g, int max_rows) {
    Ds4Bufs b{};
    const size_t D = (size_t) g.head_dim, HD = (size_t) g.n_head * D, OL = (size_t) g.o_groups * g.o_lora;
    b.qr_raw = c.take<float>(T * g.q_lora);
    b.qr = c.take<float>(T * g.q_lora);
    b.qr16 = c.take<uint16_t>(T * g.q_lora);
    b.kv_raw = c.take<float>(T * D);
    b.ckv = c.take<float>(T * 2 * D);
    b.csc = c.take<float>(T * 2 * D);
    b.ikv = c.take<float>(T * 2 * g.idx_key);
    b.isc = c.take<float>(T * 2 * g.idx_key);
    b.iw = c.take<float>(T * g.idx_heads);
    b.q = c.take<float>(T * HD);
    b.iq = c.take<float>(T * (size_t) g.idx_heads * g.idx_key);
    b.o = c.take<float>(T * HD);
    b.o16 = c.take<uint16_t>(T * HD);
    b.oa = c.take<float>(T * OL);
    b.oa16 = c.take<uint16_t>(T * OL);
    b.keys = c.take<uint16_t>((T + (size_t) g.n_swa) * D);
    b.score_ld = std::max(1, max_rows);
    // at least 6 rows: deepseek41's ratio-1 rows see up to the whole context (4 rows of max_rows), and the candidate
    // blocks' scratch takes ~1/8 of a row and 2048 ints more - a sub-batch's last few tokens wrote past 4 x their rows
    b.score_rows = (int) std::max<size_t>(T, 6);
    b.score = c.take<float>((size_t) b.score_rows * b.score_ld);
    b.sel = c.take<int>(T * (size_t) g.idx_top_k);
    return b;
}

struct DenseBufs {
    float *dg, *du;
    uint16_t* dh16;
};
DenseBufs carve_dense(Carve& c, size_t T, const Glm5Geometry& g) {
    DenseBufs b{};
    b.dg = c.take<float>(T * g.n_ff_dense);
    b.du = c.take<float>(T * g.n_ff_dense);
    b.dh16 = c.take<uint16_t>(T * g.n_ff_dense);
    return b;
}

// the MoE output rows a layer keeps on the device at once: a WINDOW (each expert set's rows are added into ffn as the
// window fills - moe_combine_add - instead of every routed row living until one combine at the end: 128 KB a token),
// and as many again for the CPU experts' uploaded rows.  At least T rows (one expert never has more than the chunk's
// tokens) and the light kernels' most; never more than the chunk routes.
// (STRATA_GLM_PREFILL_WINDOW=<rows> sets it, at least the floor; 0 = every routed row, the layout before the window)
int moe_window_rows(size_t T, const Glm5Geometry& g) {
    const size_t all = T * (size_t) g.n_exp_used, floor = std::max(T, (size_t) kLightMax * kLightRows);
    static const long long env = [] {
        const char* v = getenv("STRATA_GLM_PREFILL_WINDOW");
        return v ? std::atoll(v) : -1LL;
    }();
    if (env == 0) return (int) all;
    return (int) std::min(all, env > 0 ? std::max(floor, (size_t) env) : floor);
}

// the MoE runs on the WHOLE chunk (every expert's weights are read once per chunk); the shared expert in sub-batches
struct MoeBufs {
    float *logits, *rw, *OUTP, *OUTPc;
    float *sh_g, *sh_u;
    uint16_t* sh16;
    int *ids, *rank, *counts, *base, *row_tok, *pos, *bounds;
    uint8_t *Xq, *Hq;
};
MoeBufs carve_moe(Carve& c, size_t T, size_t sub, const Glm5Geometry& g) {
    MoeBufs b{};
    const size_t rows = T * (size_t) g.n_exp_used;
    const size_t FF = (size_t) g.n_ff_exp * g.n_shared;
    b.logits = c.take<float>(T * g.n_expert);
    b.ids = c.take<int>(rows);
    b.rw = c.take<float>(rows);
    b.rank = c.take<int>(rows);
    b.pos = c.take<int>(rows);
    b.row_tok = c.take<int>(rows);
    b.counts = c.take<int>((size_t) g.n_expert);
    b.base = c.take<int>((size_t) g.n_expert);
    b.bounds = c.take<int>(8192);
    const size_t W = (size_t) moe_window_rows(T, g);
    b.Xq = c.take<uint8_t>(mmq::q8_bytes((int64_t) W, g.n_embd));
    b.Hq = c.take<uint8_t>(mmq::q8_bytes((int64_t) W, g.n_ff_exp));
    // each set's gate/up rows live in its own OUTP rows until the down product, the swiglu over their gate halves
    b.OUTP = c.take<float>(W * g.n_embd);
    b.OUTPc = c.take<float>(W * g.n_embd);
    const size_t ts = std::min(T, sub);   // the shared expert's sub-batch
    b.sh_g = c.take<float>(ts * FF);
    b.sh_u = c.take<float>(ts * FF);
    b.sh16 = c.take<uint16_t>(ts * FF);
    return b;
}

// the mixers' and the dense FFN's sub-batch for chunks of T tokens: kSub, doubled (up to 4096) while their buffers
// stay within what the chunk's MoE buffers take of the union anyway - no extra memory, fewer and larger GEMMs, and
// each projection dequantized to FP16 once a sub-batch instead of once every 256 tokens (a 14.8K chunk: 58 -> 4).
// sub_env (STRATA_GLM_PREFILL_SUB, 16-8192) sets every sub-batch instead; HIP sets it by default (prefill_sub_env).
int mixer_sub(size_t T, const Glm5Geometry& g, bool kda, bool dsa, bool dense, bool moe, int max_pools, int sub_env) {
    if (sub_env > 0) return sub_env;
#if defined(STRATA_USE_HIP)
    (void) T; (void) g; (void) kda; (void) dsa; (void) dense; (void) moe; (void) max_pools;
    return kSub;
#else
    size_t room = 0;
    if (moe) { Carve m; carve_moe(m, T, kSub, g); room = m.off; }
    int ts = kSub;
    for (int t2 = 2 * kSub; t2 <= 4096 && (size_t) t2 <= T; t2 *= 2) {
        size_t need = 0;
        if (kda) { Carve k; carve_kda(k, (size_t) t2, g); need = std::max(need, k.off); }
        if (g.qwen35()) { Carve q; carve_q35(q, (size_t) t2, g); carve_q4x(q, (size_t) t2, g); need = std::max(need, q.off); }
        if (dsa) { Carve d; g.dsv4() ? (void) carve_ds4(d, (size_t) t2, g, max_pools) : (void) carve_dsa(d, (size_t) t2, g, max_pools); need = std::max(need, d.off); }
        if (dense) { Carve d; carve_dense(d, (size_t) t2, g); need = std::max(need, d.off); }
        if (need > room) break;
        ts = t2;
    }
    return ts;
#endif
}

// STRATA_GLM_PREFILL_SUB (16-8192): every sub-batch.  HIP without it: 1024 - mixer_sub keeps 256 there, which left an
// AI PRO R9700's 64 CUs short on the dense prompt projections (DeepSeek V4 8K / 16K prompts 454 / 599 -> 491 / 659
// tok/s, GLM-5.3-Flash Maya-S-v2 448 / 569 -> 505 / 655; 2048 and 4096 no faster); setup wrote 1024 into the configs of
// 20 GB+ Radeons and APUs already
int prefill_sub_env() {
    if (const char* v = getenv("STRATA_GLM_PREFILL_SUB")) return std::clamp(std::atoi(v), 16, 8192);
#if defined(STRATA_USE_HIP)
    return 1024;
#else
    return 0;
#endif
}

void blas_ck(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) std::fprintf(stderr, "glm prefill: %s: cuBLAS status %d\n", what, (int) st);
}

}  // namespace

struct Glm5Model::PrefillState {
    int T = 0;                       // tokens per chunk
    // --prefill: the chunk and its prestage slots for a lend cap (% of the pool's slots), the cap it was sized with,
    // whether the measured pinned share may still lower it (prefill_settle), and how it reads in the log
    std::function<std::pair<int64_t, int>(int64_t)> choose;
    int64_t lend_pct = 90;
    bool lend_pct_fixed = false;
    // the slots a layer the pool keeps while a prompt runs: a route, the spares and 2 - or, on a pool that could lend
    // no chunk at that (an 8 GB card: 9 slots a layer, issue #23), the spares and 1: the prompt stages nearly every
    // expert anyway, and the decode's misses refill the tail after it (keep_small; STRATA_GLM_PREFILL_KEEP sets it)
    int64_t keep = 0, keep_small = 0;
    bool small = false;              // ... the small keep is in use (said once)
    size_t need_min = 0, lend_max = 0;   // a 256-token chunk's bytes, and what the pool can lend at most (the refusal)
    std::string mode;
    int sub_env = 0;                 // STRATA_GLM_PREFILL_SUB: every sub-batch (0: mixer_sub, the shared expert kSub)
    int sh_sub() const { return sub_env > 0 ? sub_env : kSub; }   // the shared expert's sub-batch
    int max_pools = 0;
    cublasHandle_t blas = nullptr;
    void* ws = nullptr;
#if defined(STRATA_USE_HIP)
    std::unique_ptr<strata::prefill::Gemm> lt;   // hipBLASLt with a tuning table (STRATA_HIPBLASLT_TUNING)
    bool lt_tried = false;
#endif
    std::unique_ptr<mmq::Context> mq;
    uint8_t* arena = nullptr;
    size_t arena_bytes = 0;
    // every device buffer lives in the pool's lendable tail (prefill_bind): valid only while lent
    size_t borrow_bytes = 0, ws_bytes = 0, gbuf_bytes = 0;
    bool lent = false;
    bool has_kda = false, has_dsa = false, has_dense = false, has_moe = false;
    bool v41_eng_gemm = false;       // deepseek41: Engram's engram_kv as a GEMM over a block (prefill_v41_engram_block)
    uint8_t* region = nullptr;       // the pool's lendable tail
    size_t region_bytes = 0;
    int T_bound = 0;                 // the chunk the pointers are carved for (<= T)
    int sub = kSub;                  // ... and its mixer sub-batch (mixer_sub)
    size_t need = 0;                 // ... and the bytes of the region that layout uses
    uint64_t dropped = 0;            // experts the lending evicted from VRAM (cumulative)
    uint64_t moved = 0;              // ... of which a lent slot's expert took a colder one's slot instead
    // per chunk, across layers
    float *R = nullptr, *x = nullptr, *mixer = nullptr, *ffn = nullptr;
    float *pre = nullptr, *post = nullptr, *comb = nullptr, *ss = nullptr, *mix = nullptr;
    float *pre2 = nullptr, *onehot = nullptr;   // deepseek41: the FFN reads' pre (the lag's other half), layer 0's
    std::vector<int32_t> toks_h;                // ... and the chunk's token ids on the host (the Engram / PLE hashes)
    std::vector<int32_t> q4_steps_h;            // qwen4exp: a sub-batch's QSA step records and query positions
    uint16_t* x16 = nullptr;
    int* iota = nullptr;
    uint8_t* v41_rows = nullptr;     // deepseek41: each row's top-k selection and candidate mask (prefill_v41_row)
    uint8_t* uni = nullptr;          // the per-layer region (KDA | DSA | dense | MoE)
    // weight scratch
    uint16_t* w16 = nullptr;
    int64_t w16_elems = 0;
    float* w32 = nullptr;
    int64_t w32_elems = 0;
    // non-resident experts: a ring of NG staging groups of GE slots each (the layer's slot stride apart)
    static constexpr int NG = 3, GE = 4;
    uint8_t* gbuf = nullptr;
    size_t gstride = 0;
    // pinned: the disk reads' landing ring, nland slots of gstride - deep enough that the reader runs a layer's disk
    // experts ahead of the groups that use them (12 slots kept the reader and the GPU waiting on each other: a 16k
    // prompt on Mercury read the NVMe at ~0.7 of its ~2.9 GB/s); STRATA_GLM_PREFILL_LAND=<slots>, at least NG * GE
    uint8_t* gpin = nullptr;
    int nland = NG * GE;
    cudaEvent_t ev_ready[NG] = {}, ev_free[NG] = {};
    std::vector<cudaEvent_t> ev_land;    // a landing slot's copy to the device ran: the reader may refill it
    // pinned host staging
    float* emb_h = nullptr;          // T x n_embd
    unsigned char* img_d = nullptr;  // deepseek41: the chunk's image-row flags on the device (route's per-row bias)
    bool img_on = false;             // ... and this chunk has an image row (else route takes the plain bias)
    int* toks_d = nullptr;           // deepseek4: the chunk's token ids on the device (hash routing)
    float* hop_h = nullptr;          // 2 x T x hc x n_embd: the split's hand-over (first half), double-buffered
    int* h_counts = nullptr;
    int* h_base = nullptr;
    int* h_bounds = nullptr;
    cudaEvent_t ev_hop = nullptr;
    // stats
    double ms = 0, ms_plan = 0, ms_disk = 0;
    int64_t tokens = 0, chunks = 0;
    // STRATA_GLM_PREFILL_PROF=1: events between the phases of every layer, summed per phase (debug)
    bool prof = false;
    std::vector<cudaEvent_t> pev;
    std::vector<const char*> pname;
    size_t pn = 0;
    std::map<std::string, double> pacc;
    void mark(const char* nm, cudaStream_t st) {
        if (!prof) return;
        if (pn >= pev.size()) {
            cudaEvent_t e = nullptr;
            cudaEventCreate(&e);
            pev.push_back(e);
            pname.push_back(nm);
        }
        cudaEventRecord(pev[pn], st);
        pname[pn] = nm;
        ++pn;
    }
    void collect() {
        if (!prof || pn == 0) return;
        cudaEventSynchronize(pev[pn - 1]);
        for (size_t i = 1; i < pn; ++i) {
            float t = 0.0f;
            if (cudaEventElapsedTime(&t, pev[i - 1], pev[i]) == cudaSuccess) pacc[pname[i]] += t;
        }
        pn = 0;
    }
    uint64_t staged_ram = 0, staged_vram = 0, staged_disk = 0, rows_resident = 0, rows_staged = 0, disk_issued = 0;
    // the prompt's CPU experts: x and the rows' tokens down, the outputs up, on their own stream.  ev_xd: xs's copies
    // done - the host waits for it, not for the stream (HIP on an AI PRO R9700 runs every stream in one hardware queue,
    // and a stream's synchronize there waited out the layer's whole PCIe staging queued after: ~120-150 ms a layer)
    cudaStream_t xs = nullptr;
    cudaEvent_t ev_x = nullptr, ev_out = nullptr, ev_xd = nullptr;
    float* x_h = nullptr;
    size_t x_h_n = 0;
    int* rt_h = nullptr;
    int* id_h = nullptr;      // 0, 1, 2, ..: the iGPU's row -> token map when x went down gathered (into out_h)
    bool x_gathered = false;  // this layer's host rows' x is in out_h (row r = the host's row r), not x_h
    float* out_h = nullptr;
    size_t rows_h_n = 0;
    std::vector<uint8_t> c_act, c_hq, c_pack;
    std::vector<float> c_ff;
    uint64_t cpu_experts = 0, cpu_rows = 0;
    double ms_cpu = 0, row_ms = 0.05;   // row_ms: the host's cost of a row, learned (the split's estimate)
    // the host experts on the iGPU (abyss_igpu_rows): its own cost of a row (learned) and of an expert; the host
    // buffers it reads and writes (registered with it)
    double row_ms_ig = 0.009, fix_ms_ig = 0.045;
    uint64_t ig_experts = 0, ig_rows = 0;
    std::vector<const void*> ig_reg;
    // the split's balance, learned: the host's estimates times kappa, nudged each layer by the host's time over the
    // copy stream's (ev_c0 .. ev_c1: the staged copies) - the two lanes should end together
    double kappa = 1.0;
    cudaEvent_t ev_c0 = nullptr, ev_c1 = nullptr;
    // PRESTAGING (one GPU): the copy stream idles while a layer's mixer runs (~30% of a long prompt on a 3090), so
    // the next MoE layer's likely experts - RAM-tier ones, by predicted rows - are copied then, into a buffer of NP
    // slots that every layer reuses (one slot serves each layer: worth ~a slot per layer of the pool it is borrowed
    // from, for the cost of one).  The plan computes the ones its routing hit from there and splits the rest between
    // the CPU and PCIe as before; copies still in flight at the plan count against the PCIe side.
    int NP = 0;                           // slots (of the largest MoE layer stride)
    size_t pbuf_bytes = 0;
    size_t pbuf_carved = 0;               // ... of which this chunk's carve took (prefill_carve; the prestage's cap)
    uint8_t* pbuf = nullptr;              // carved for chunks of kPreMinT tokens and more
    int pre_layer = -1;                   // the layer whose experts pbuf holds (or is receiving)
    std::vector<int> pre_e;               // ... in slot order
    static constexpr int kPreEv = 8;      // an event every kPreEv copies: how far they are at the plan
    std::vector<cudaEvent_t> ev_pdone;
    cudaEvent_t ev_pready = nullptr, ev_pfree = nullptr, ev_pstart = nullptr;
    // STRATA_GLM_PRESTAGE_ADAPT=1 (experimental): how many a layer prestages, in experts per chunk token by mixer
    // (KDA, DSA), learned from its lanes - the copy stream idle before the plan: more; the CPU done before copies the
    // plan could not move to it: fewer (0: unset).  Off: every layer fills the buffer
    double pre_rate[2] = {0.0, 0.0};
    double copy_ms = 0.0;                 // a prestage copy's time, measured (the split's PCIe estimate)
    std::vector<std::vector<int>> last_cnt;   // per layer: the routing of its last chunk (half the prediction)
    uint64_t pre_issued = 0, pre_hit = 0, pre_rows = 0;
    // STRATA_GLM_PREFILL_TRACE=1 (debug): per MoE layer of a chunk, when its lanes ended - printed after the chunk
    struct Tr {
        cudaEvent_t start = nullptr, plan = nullptr, pre = nullptr, c1 = nullptr, end = nullptr;
        int pre_n = 0, pre_left = 0, grp = 0, ncpu = 0, crows = 0;
        double dl = 0, dcpu = 0;
        bool on = false;
    };
    bool trace = false;
    std::vector<Tr> tr;
};

int Glm5Model::prefill_chunk() const { return pf_ ? pf_->T : 0; }

// --prefill as the engine reads it: STRATA_GLM_PREFILL (generate sets it from the config's args), unset = auto;
// STRATA_GLM_PREFILL_CHUNK=N (the older knob) is --prefill N
static std::string prefill_mode() {
    if (const char* c = getenv("STRATA_GLM_PREFILL_CHUNK"); c != nullptr && c[0]) return c;
    if (const char* v = getenv("STRATA_GLM_PREFILL"); v != nullptr && v[0]) return v;
    return "auto";
}

// a split's chunks are its smallest part's (prefill()): a part set up for longer ones keeps its pinned staging to that
// (a 24 GB card sized for 32K tokens beside a 16 GB card's 22K would pin ~1.5 GB of hop and embedding rows no chunk
// uses)
void Glm5Model::prefill_cap(int T, const char* why) {
    PrefillState* S = pf_;
    if (S == nullptr || S->T <= T) return;
    const size_t E = (size_t) g_.n_embd;
    const bool hop = S->hop_h != nullptr;
    cudaSetDevice(dev_);
    cudaFreeHost(S->emb_h);
    S->emb_h = nullptr;
    if (hop) cudaFreeHost(S->hop_h);
    S->hop_h = nullptr;
    if (cudaHostAlloc((void**) &S->emb_h, (size_t) T * E * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
        (hop && cudaHostAlloc((void**) &S->hop_h, (size_t) 2 * T * 4 * E * sizeof(float), cudaHostAllocDefault) !=
                    cudaSuccess)) {
        cudaGetLastError();
        std::fprintf(stderr, "glm prefill: CUDA%d pinned staging did not allocate - token by token\n", dev_);
        prefill_destroy();
        return;
    }
    std::fprintf(stderr, "glm prefill: CUDA%d chunks of %d tokens (%s)\n", dev_, T, why);
    S->T = T;
}

// Strata's lend cap, settled once the RAM tier exists (prefill_setup sized the chunk with its 90): 90% of the pool's
// slots when at least 90% of this card's expert bytes are held pinned - in the VRAM pool or the RAM tier, whose copies
// are DMA - else 85% (the rest comes from the disk through host copies, the limit there: lending more only streams
// more through them)
void Glm5Model::prefill_settle(double pinned_share) {
    PrefillState* S = pf_;
    if (S == nullptr || S->lend_pct_fixed || pinned_share >= 0.9) return;
    S->lend_pct = 85;
    const auto pick = S->choose(S->lend_pct);
    std::fprintf(stderr, "glm prefill: CUDA%d %.0f%% of the experts' bytes are held pinned: the pool lends at most 85%% "
                         "of its slots\n", dev_, 100.0 * pinned_share);
    if (pick.first == 0) {
        std::fprintf(stderr, "glm prefill: CUDA%d the expert pool can lend no prompt chunk at 85%% (a 256-token chunk needs "
                             "%zu MB, it can lend %zu MB) - token by token\n", dev_, S->need_min >> 20, S->lend_max >> 20);
        prefill_destroy();
        return;
    }
    if (pick.first >= S->T) return;
    S->NP = std::min(S->NP, pick.second);
    S->pbuf_bytes = (size_t) S->NP * S->gstride;
    prefill_cap((int) pick.first, "--prefill auto at 85%");
}

// routed experts MMQ does not read (IQ1_M: ISTA-DASLab's GSQ-RCO V4.1 quants) whose rows dequant_f16 decodes: the
// prompt path decodes each such expert to FP16 once and multiplies it by cuBLAS (run_blas_dq) - the idea of Strata
// 0.2.4's gfx12 i-quant tiles and Maya's DS4_FAT_ROWS (a prompt expert's weights decoded once, not once per row tile)
static bool mmq_layer(int gu, int d) { return mmq::supported(gu) && mmq::supported(d); }
static bool deq_expert_type(int t) {
    return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29 || strata::kernels::dequant_bf16_supported(t);
}
// rows a decoded expert's FP16 activations take beside its 2 n_ff x n_embd gate/up (run_blas_dq's X16)
static constexpr int64_t kDqRows = 512;

// the weight scratch the prompt path keeps beside a chunk's rows: {BF16 elements, F32 elements}.  deq: a layer's
// experts go through run_blas_dq (their decoded gate/up and kDqRows activation rows)
static std::pair<int64_t, int64_t> weight_scratch(const Glm5Geometry& g, bool has_dsa, bool deq = false) {
    const int64_t E = g.n_embd;
    int64_t w32 = 2 << 20;
    if (has_dsa)
        w32 = std::max<int64_t>({w32, (int64_t) g.n_head * g.kv_lora * g.qk_nope, (int64_t) g.n_head * g.v_head * g.kv_lora});
    int64_t w16 = std::max<int64_t>((int64_t) g.d_inner() * E, (int64_t) 2048 * E);
    if (deq) w16 = std::max<int64_t>(w16, ((int64_t) 2 * g.n_ff_exp + kDqRows) * E);
#if defined(STRATA_USE_HIP)
    if (has_dsa)   // the MLA products' FP16 copy of wk_b / wv_b
        w16 = std::max<int64_t>({w16, (int64_t) g.n_head * g.kv_lora * g.qk_nope, (int64_t) g.n_head * g.v_head * g.kv_lora});
#endif
    return {w16, w32};
}
static constexpr size_t kPfWorkspace = (size_t) 16 << 20;   // cuBLAS

// ---------------------------------------------------------------- setup
bool Glm5Model::prefill_setup(std::string& err) {
    (void) err;
    if (getenv("STRATA_GLM_NO_PREFILL") != nullptr) return true;
    if (prefill_mode() != "auto" && std::atoll(prefill_mode().c_str()) <= 0) {
        std::fprintf(stderr, "glm prefill: --prefill %s: the prompt runs token by token\n", prefill_mode().c_str());
        return true;
    }
    if (!mmq::built()) {
        std::fprintf(stderr, "glm prefill: this build has no MMQ kernels - the prompt runs token by token\n");
        return true;
    }
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    bool has_kda = false, has_dsa = false, has_moe = false, has_dense = false, has_deq = false, v41_tok = false;
    bool v41_eng_tok = false;
    size_t gstride = 0;
    std::string why;
    const auto deq_ok = [&](const WSlot& w, const char* nm, int il) {
        // (qwen4exp: the i-quant projections too - hgemm_q's dequant_f16 takes them)
        const bool iq = g.q4x && (w.type == 16 || w.type == 17 || w.type == 18 || w.type == 21 || w.type == 22 ||
                                  w.type == 29);
        // (qwen35 / qwen4exp: F32 rows too - unsloth's Qwen3.8 UD files keep ssm_alpha / ssm_beta F32 where ISTA's are
        // BF16, and hgemm_q widens F32 to FP16 as it does the injects; issue #25: they dropped the whole prompt path)
        const bool f32 = g.qwen35() && w.type == gf::kTypeF32;
        if (w.q == nullptr || (w.type != gf::kTypeBF16 && w.type != strata::kernels::exl3::kTypeEXL3 && !iq && !f32 &&
                               !strata::kernels::dequant_bf16_supported(w.type)))
            why += "blk." + std::to_string(il) + "." + nm + " (type " + std::to_string(w.type) + ") ";
    };
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (g.qwen35()) {
            if (Ly.recr) {
                deq_ok(Ly.qkv, "attn_qkv", il);
                deq_ok(Ly.zg, "attn_gate", il);
                deq_ok(Ly.ssm_o, "ssm_out", il);
                deq_ok(Ly.alpha_w, "ssm_alpha", il);
                deq_ok(Ly.beta_w, "ssm_beta", il);
            } else {
                deq_ok(Ly.aq, "attn_q", il);
                deq_ok(Ly.ak, "attn_k", il);
                deq_ok(Ly.av, "attn_v", il);
            }
        } else if (g.v41) {
            // deepseek41: the attention's projections batched (prefill_v41_mixer) when every one dequantizes, else
            // the token path's attention a token at a time (fast_dsv41_attn); STRATA_GLM_V41_BATCH=0: always the latter
            const size_t before = why.size();
            deq_ok(Ly.q_a, "attn_q_a", il);
            deq_ok(Ly.q_b, "attn_q_b", il);
            deq_ok(Ly.kv, "attn_kv", il);
            deq_ok(Ly.out_a, "attn_output_a", il);
            deq_ok(Ly.out_b, "attn_output_b", il);
            if (Ly.kv_src) {
                deq_ok(Ly.comp_kv, "attn_compressor_kv", il);
                if (Ly.ratio == 2) deq_ok(Ly.comp_gate, "attn_compressor_gate", il);
            }
            if (Ly.idx_src) deq_ok(Ly.idx_q_b_q, "indexer.attn_q_b", il);
            if (why.size() != before) {
                v41_tok = true;
                why.resize(before);
            }
            if (Ly.eng >= 0) {
                deq_ok(Ly.eng_kv, "engram_kv", il);
                if (why.size() != before) {
                    v41_eng_tok = true;
                    why.resize(before);
                }
            }
        } else if (g.dsv4()) {
            has_dsa = true;
            deq_ok(Ly.q_a, "attn_q_a", il);
            deq_ok(Ly.q_b, "attn_q_b", il);
            deq_ok(Ly.kv, "attn_kv", il);
            deq_ok(Ly.out_a, "attn_output_a", il);
            deq_ok(Ly.out_b, "attn_output_b", il);
            if (Ly.ratio != 0) {
                deq_ok(Ly.comp_kv, "attn_compressor_kv", il);
                deq_ok(Ly.comp_gate, "attn_compressor_gate", il);
            }
            if (Ly.ratio == 4) {
                deq_ok(Ly.idx_q_b_q, "indexer.attn_q_b", il);
                deq_ok(Ly.idx_comp_kv, "indexer_compressor_kv", il);
                deq_ok(Ly.idx_comp_gate, "indexer_compressor_gate", il);
            }
        } else if (Ly.recr) {
            has_kda = true;
            deq_ok(Ly.q, "attn_q", il);
            deq_ok(Ly.k, "attn_k", il);
            deq_ok(Ly.v, "attn_v", il);
        } else {
            has_dsa = true;
            deq_ok(Ly.q_a, "attn_q_a", il);
            if (Ly.idx_q_b == nullptr) deq_ok(Ly.idx_q_b_x, "indexer.attn_q_b", il);
            deq_ok(Ly.q_b, "attn_q_b", il);
            deq_ok(Ly.kv_a, "attn_kv_a", il);
        }
        if (!g.dsv4() && !(g.qwen35() && Ly.recr)) deq_ok(Ly.out, "attn_output", il);
        if (Ly.moe) {
            has_moe = true;
            deq_ok(Ly.sh_gate, "ffn_gate_shexp", il);
            deq_ok(Ly.sh_up, "ffn_up_shexp", il);
            deq_ok(Ly.sh_down, "ffn_down_shexp", il);
            if (!Ly.exl3 && !mmq_layer(Ly.gu_type, Ly.d_type)) {   // (EXL3: run_set_x3)
                if (deq_expert_type(Ly.gu_type) && deq_expert_type(Ly.d_type))
                    has_deq = true;   // (run_blas_dq)
                else
                    why += "blk." + std::to_string(il) + " experts (types " + std::to_string(Ly.gu_type) + "/" +
                           std::to_string(Ly.d_type) + ") ";
            }
            gstride = std::max(gstride, glmfast::expert_stride(Ly.blob, Ly.gu_type, Ly.d_type));
        } else {
            has_dense = true;
            deq_ok(Ly.ffn_gate, "ffn_gate", il);
            deq_ok(Ly.ffn_up, "ffn_up", il);
            deq_ok(Ly.ffn_down, "ffn_down", il);
        }
    }
    if (!why.empty()) {
        std::fprintf(stderr, "glm prefill: CUDA%d has no prompt path (not covered: %s) - token by token\n", dev_,
                     why.substr(0, 400).c_str());
        return true;
    }
    if (has_dsa && !g.dsv4() && (g.kv_lora != 512 || g.n_head % 16 != 0 || g.kda_head_dim != 128)) {
        std::fprintf(stderr, "glm prefill: geometry not covered by the batched kernels - token by token\n");
        return true;
    }
    if (g.v41) {
        static const bool batch_env = [] { const char* v = getenv("STRATA_GLM_V41_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
        has_dsa = batch_env && !v41_tok;   // (the mixer's buffers: carve_ds4's)
        std::fprintf(stderr, "glm prefill: CUDA%d deepseek41 attention %s\n", dev_,
                     has_dsa ? "projections batched" : v41_tok ? "a token at a time (a weight type without an FP16 decode)"
                                                               : "a token at a time (STRATA_GLM_V41_BATCH=0)");
    }
    auto* S = new PrefillState();
    pf_ = S;
    S->has_kda = has_kda;
    S->has_dsa = has_dsa;
    S->has_dense = has_dense;
    S->has_moe = has_moe;
    if (g.v41) {   // STRATA_GLM_V41_ENG_GEMM=0: Engram's adds a token at a time (the token path's GEMV)
        static const bool eng_env = [] { const char* v = getenv("STRATA_GLM_V41_ENG_GEMM"); return v == nullptr || std::atoi(v) != 0; }();
        S->v41_eng_gemm = has_dsa && eng_env && !v41_eng_tok;
    }
    S->max_pools = (int) std::max<int64_t>(1, max_ctx_ / g.idx_kpool);
    S->sub_env = prefill_sub_env();
    S->prof = getenv("STRATA_GLM_PREFILL_PROF") != nullptr;
    S->gstride = gstride;

    // ---- sizes: the chunk T from the budget (per device)
    const int64_t E = g.n_embd;
    const auto wsc = weight_scratch(g, has_dsa, has_deq);
    if (has_deq)
        std::fprintf(stderr, "glm prefill: CUDA%d experts without MMQ (IQ1_M ...) decoded to FP16 once each, then cuBLAS\n", dev_);
    const int64_t w16_elems = wsc.first, w32_elems = wsc.second;
    const size_t ws_bytes = kPfWorkspace;
    // ---- the chunk T, as Strata's --prefill (the config's args; STRATA_GLM_PREFILL, unset = auto).  `auto`: the
    // largest chunk on the 256-token grid, up to 32768 on one GPU and 8192 on a split (each the faster there: one V100,
    // 26K-token prompts 559 tok/s at 32768 against 367 at 8192; two V100s 709 at 8192 against 551 at 32768), whose
    // buffers the expert pool fast_setup carves next can lend.
    // The pool lends at most STRATA_PREFILL_LEND_PCT of its slots - Strata's 90 when at least 90% of the expert bytes
    // are held pinned (the copies are DMA), 85 when host copies are the limit; that share is known once the RAM tier
    // exists, so the chunk is sized with 90 here and prefill_settle() takes it down to 85's when the share falls short
    // - and keeps every layer a route's experts and the spares (fast_setup's floor).  The prestage buffer is Strata's
    // ring and comes out of the same loan: the chunk is the largest that leaves it whole, else the largest that leaves
    // it 16 slots, else none.  A number N: chunks of N tokens, halved until the pool can lend them (only the floor
    // applies - the operator's number).  A split reads in its smallest part's chunk (load_pack_split).
    // STRATA_GLM_PREFILL_MB=<MB> fixes what may be lent.
    const std::string mode = prefill_mode();
    const bool is_auto = mode == "auto";
    const int64_t auto_max = n_parts_ == 1 ? 32768 : 8192;
    const int64_t ceiling = is_auto ? auto_max : (int64_t) std::atoll(mode.c_str());
    // the pool to come: `per` slots a layer over the tier layers, as fast_setup sizes it from the same free VRAM
    size_t dev_free = 0, dev_total = 0;
    cudaMemGetInfo(&dev_free, &dev_total);
    const size_t avail = pool_avail(dev_free, dev_total);
    size_t blob_sum = 0, stride_sum = 0;
    for (int il = l0_; il < lt_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (!Ly.moe) continue;
        blob_sum += Ly.blob;
        stride_sum += glmfast::expert_stride(Ly.blob, Ly.gu_type, Ly.d_type);
    }
    int64_t per = std::min<int64_t>(g.n_expert, (int64_t) (avail / std::max<size_t>(1, blob_sum)));
    while (per > 0 && (size_t) per * stride_sum > avail) --per;
    // what every layer keeps from the prompt: a route's experts, the spares and 2 (fast_setup's floor).  A pool that
    // can lend no chunk at that keeps only the spares and 1 instead (issue #23: an 8 GB RTX A2000 ran Ornith-1.5-122B
    // IQ3_XXS with 9 slots a layer - no chunk against a keep of 13, 16 tok/s token by token.  A prompt runs alone and
    // stages nearly every expert of a layer whatever the pool holds, so its residents are worth little to it, and the
    // decode's misses refill the tail after it - 9 slots x 48 layers x 3.4 MB = 1.5 GB, a fraction of a second).
    // STRATA_GLM_PREFILL_KEEP=<slots> sets that small keep (1..); the full keep stays first whenever it lends a chunk
    const int64_t keep_full = g.n_exp_used + gf::kSpares + 2;
    S->keep = keep_full;
    S->keep_small = std::min<int64_t>(keep_full, gf::kSpares + 1);
    if (const char* v = getenv("STRATA_GLM_PREFILL_KEEP")) S->keep_small = std::max<int64_t>(1, std::atoll(v));
    const char* pct_env = getenv("STRATA_PREFILL_LEND_PCT");
    const char* mb_env = getenv("STRATA_GLM_PREFILL_MB");
    const size_t gbuf = has_moe ? (size_t) PrefillState::NG * PrefillState::GE * gstride : 0;
    // the prestage buffer (the ring), on a split's parts too (v1.3.27: RTX 5090 + RTX 3090 split 33 / 10, 32K-token
    // prompts 2,177-2,186 -> 2,305-2,309 tok/s, 8K about even); STRATA_GLM_PRESTAGE=<slots>, 0 none
    int ring_max = has_moe ? kPreSlots : 0;
    if (const char* v = getenv("STRATA_GLM_PRESTAGE")) ring_max = has_moe ? std::max(0, std::atoi(v)) : 0;
    // the chunk and its prestage slots for a lend cap of `pct`% of the pool's slots ({0, 0}: no chunk fits)
    S->choose = [=, this](int64_t pct) -> std::pair<int64_t, int> {
        PrefillState* S = pf_;
        // the bytes the pool can lend the prompt path when every layer keeps `keep` slots
        const auto lendable_at = [&](int64_t keep) -> size_t {
            if (mb_env != nullptr) return (size_t) (std::atof(mb_env) * 1048576.0);
            if (stride_sum == 0) return avail;
            int64_t k = std::max<int64_t>(0, per - keep);
            if (is_auto) k = std::min<int64_t>(k, pct * per / 100);
            return (size_t) k * stride_sum;
        };
        size_t lendable = lendable_at(keep_full);
        const auto nonring = [&](int64_t T) -> size_t {
            const auto pu = prefill_bytes_for((size_t) T);
            Carve c;
            c.take<uint8_t>(pu.first + pu.second);
            c.take<uint16_t>((size_t) w16_elems);
            c.take<float>((size_t) w32_elems);
            c.take<uint8_t>(ws_bytes);
            c.take<uint8_t>(gbuf);
            return c.off;
        };
        constexpr int kRingMin = 16;   // a prestage buffer below this is none (prefill_trim_prestage)
        // the prestage slots chunk T leaves beside its buffers (at most ring_max), -1 when T does not fit at all
        const auto room = [&](int64_t T) -> int64_t {
            const size_t nr = nonring(T);
            if (nr + 256 > lendable) return -1;
            if (T < kPreMinT || gstride == 0) return ring_max;   // (a chunk this short carves no prestage buffer)
            return std::min<int64_t>(ring_max, (int64_t) ((lendable - nr - 256) / gstride));
        };
        // Strata's biggest_chunk: every test rises with T, so the largest chunk that passes is a bisection on the grid
        const auto biggest = [&](int64_t floor) -> int64_t {
            int64_t lo = 1, hi = ceiling / 256, best = 0;
            while (lo <= hi) {
                const int64_t mid = lo + (hi - lo) / 2;
                if (room(mid * 256) >= floor) {
                    best = mid * 256;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            return best;
        };
        const auto pick_T = [&]() -> int64_t {
            int64_t T = 0;
            if (is_auto && n_parts_ > 1) {
                // a split: the chunk first (8192 at most - the parts' pipeline wants it whole), the prestage buffer
                // from what it leaves (none below 16 slots) - a small card keeps its chunk
                T = biggest(0);
            } else if (is_auto) {
                T = biggest(ring_max);
                if (T == 0 && ring_max > kRingMin) T = biggest(kRingMin);
                if (T == 0) T = biggest(0);
            } else {
                for (int64_t c = ceiling; c >= 256; c /= 2)
                    if (room(c) >= 0) {
                        T = c;
                        break;
                    }
                if (T == 0 && ceiling > 0 && ceiling < 256 && room(ceiling) >= 0) T = ceiling;
            }
            return T;
        };
        int64_t T = pick_T();
        int64_t keep = keep_full;
        if (T == 0 && S->keep_small < keep_full && mb_env == nullptr && stride_sum > 0) {   // the small keep (above)
            lendable = lendable_at(S->keep_small);
            T = pick_T();
            if (T > 0) {
                keep = S->keep_small;
                if (!S->small)
                    std::fprintf(stderr, "glm prefill: CUDA%d the expert pool (%lld slots a layer) can lend no prompt "
                                         "chunk while every layer keeps %lld slots - it keeps %lld while a prompt runs "
                                         "(small VRAM; STRATA_GLM_PREFILL_KEEP)\n", dev_, (long long) per,
                                 (long long) keep_full, (long long) keep);
                S->small = true;
            }
        }
        S->keep = keep;
        // (what a 256-token chunk would take against what the pool can lend at most: the refusal's "how much short")
        S->need_min = nonring(256) + 256;
        S->lend_max = lendable;
        const int64_t r = T > 0 ? room(T) : 0;
        return {T, T >= kPreMinT && r >= kRingMin ? (int) r : 0};
    };
    S->lend_pct = pct_env != nullptr ? (int64_t) std::atoi(pct_env) : 90;
    S->lend_pct_fixed = pct_env != nullptr || mb_env != nullptr || !is_auto;
    S->mode = is_auto ? "auto, up to " + std::to_string(auto_max) : mode;
    const auto pick = S->choose(S->lend_pct);
    const int64_t T = pick.first;
    if (T == 0) {
        // (how much VRAM is short for the smallest chunk, and what frees it: Maya's refusal lines say as much)
        const size_t short_b = S->need_min > S->lend_max ? S->need_min - S->lend_max : 0;
        std::fprintf(stderr, "glm prefill: CUDA%d the expert pool (%lld slots a layer, %lld kept) can lend no prompt "
                             "chunk (--prefill %s): a 256-token chunk needs %zu MB, the pool can lend %zu MB - %zu MB of "
                             "VRAM short; a smaller --max-context, --kv q4_0 or a lighter dense pack (docs/SMALL-VRAM.md) "
                             "frees it - token by token\n", dev_, (long long) per, (long long) S->keep, mode.c_str(),
                     S->need_min >> 20, S->lend_max >> 20, (short_b + 1048575) >> 20);
        prefill_destroy();
        return true;
    }
    if (!is_auto && T != ceiling)
        std::fprintf(stderr, "glm prefill: CUDA%d prompt chunk %lld -> %lld tokens so its buffers fit in the expert "
                             "pool\n", dev_, (long long) ceiling, (long long) T);
    S->T = (int) T;
    const auto pu = prefill_bytes_for((size_t) T);
    S->arena_bytes = pu.first + pu.second;
    S->w16_elems = w16_elems;
    S->w32_elems = w32_elems;
    S->ws_bytes = ws_bytes;
    S->gbuf_bytes = gbuf;
    S->NP = pick.second;
    S->pbuf_bytes = (size_t) pick.second * gstride;
    {
        Carve c;
        c.take<uint8_t>(S->arena_bytes);
        c.take<uint16_t>((size_t) w16_elems);
        c.take<float>((size_t) w32_elems);
        c.take<uint8_t>(ws_bytes);
        c.take<uint8_t>(S->gbuf_bytes);
        c.take<uint8_t>(S->pbuf_bytes);
        S->borrow_bytes = c.off;
    }
    if (cudaHostAlloc((void**) &S->emb_h, (size_t) T * E * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &S->h_counts, 4096 * sizeof(int), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostAlloc((void**) &S->h_base, 4096 * sizeof(int), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &S->h_bounds, 8192 * sizeof(int), cudaHostAllocDefault) != cudaSuccess ||
        (l1_ < g.n_layers && cudaHostAlloc((void**) &S->hop_h, (size_t) 2 * T * 4 * E * sizeof(float),
                                           cudaHostAllocDefault) != cudaSuccess)) {
        cudaGetLastError();
        std::fprintf(stderr, "glm prefill: CUDA%d pinned staging did not allocate - token by token\n", dev_);
        prefill_destroy();
        return true;
    }
    if (has_moe) {
        // the landing ring: ~3% of the free RAM per device (it is pinned before the RAM tier measures what is left,
        // so it comes out of that tier), 12 to 96 slots; halved while the pinning fails.  96 slots let the second
        // GPU of a split read ahead through a whole layer (2x V100, 30 GB RAM: its disk waits 31 s -> 3-6 s, +1%)
        constexpr int kMinLand = PrefillState::NG * PrefillState::GE;
        int want = (int) std::min<int64_t>(96, std::max<int64_t>(kMinLand, (int64_t) (0.03 * (double) avail_ram_bytes() /
                                                                                      (double) gstride)));
        // On an APU these pinned pages compete with the expert pool. A small
        // landing ring suffices for disk reads while the prompt borrows its tail.
        if (F->unified_memory) want = kMinLand;
        if (const char* v = getenv("STRATA_GLM_PREFILL_LAND")) want = std::max(kMinLand, std::atoi(v));
        int n = want;
        while (cudaHostAlloc((void**) &S->gpin, (size_t) n * gstride, cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            S->gpin = nullptr;
            if (n == kMinLand) break;
            n = std::max(kMinLand, n / 2);
        }
        if (S->gpin == nullptr) {
            std::fprintf(stderr, "glm prefill: CUDA%d the disk landing ring did not allocate - token by token\n", dev_);
            prefill_destroy();
            return true;
        }
        S->nland = n;
        S->ev_land.assign((size_t) n, nullptr);
    }
    if (g.v41 && cudaMalloc(&S->img_d, (size_t) T) != cudaSuccess) {   // the image rows' flags (exp_probs_b_vl)
        cudaGetLastError();
        S->img_d = nullptr;
        std::fprintf(stderr, "glm prefill: CUDA%d the image flag buffer did not allocate - token by token\n", dev_);
        prefill_destroy();
        return true;
    }
    if (g.dsv4() && cudaMalloc(&S->toks_d, (size_t) T * sizeof(int)) != cudaSuccess) {   // the hash routing's tokens
        cudaGetLastError();
        S->toks_d = nullptr;
        std::fprintf(stderr, "glm prefill: CUDA%d the token buffer did not allocate - token by token\n", dev_);
        prefill_destroy();
        return true;
    }
    if (cublasCreate(&S->blas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "glm prefill: CUDA%d cublasCreate failed - token by token\n", dev_);
        prefill_destroy();
        return true;
    }
    cublasSetStream(S->blas, F->cs);
    cublasSetMathMode(S->blas, CUBLAS_DEFAULT_MATH);
    S->mq = std::make_unique<mmq::Context>();
    // an event that did not create (a GPU the expert pool filled) stays null and the prompt runs token by token:
    // the staging below records and waits on every one of them, and a null event there is no event at all
    bool events_ok = true;
    const auto create_event = [&](cudaEvent_t& e, unsigned flags) {
        if (events_ok && cudaEventCreateWithFlags(&e, flags) != cudaSuccess) {
            e = nullptr;
            events_ok = false;
        }
    };
    for (int b = 0; b < PrefillState::NG; ++b) {
        create_event(S->ev_ready[b], cudaEventDisableTiming);
        create_event(S->ev_free[b], cudaEventDisableTiming);
    }
    for (auto& e : S->ev_land) create_event(e, cudaEventDisableTiming);
    create_event(S->ev_hop, cudaEventDisableTiming);
    if (S->NP > 0) {
        S->ev_pdone.assign((size_t) (S->NP + PrefillState::kPreEv - 1) / PrefillState::kPreEv, nullptr);
        for (auto& e : S->ev_pdone) create_event(e, cudaEventDisableTiming);
        create_event(S->ev_pready, 0);   // (timed, flags 0: the controller reads when the copies landed)
        create_event(S->ev_pstart, 0);
        create_event(S->ev_pfree, cudaEventDisableTiming);
    }
    if (!events_ok) {
        cudaGetLastError();
        std::fprintf(stderr, "glm prefill: CUDA%d staging events did not create - token by token\n", dev_);
        prefill_destroy();
        return true;
    }
    S->last_cnt.assign((size_t) g.n_layers, {});
    S->trace = getenv("STRATA_GLM_PREFILL_TRACE") != nullptr;
    if (S->trace) {
        S->tr.resize((size_t) g.n_layers);
        for (auto& t : S->tr)
            for (cudaEvent_t* e : {&t.start, &t.plan, &t.pre, &t.c1, &t.end}) cudaEventCreate(e);
    }
    std::fprintf(stderr, "glm prefill: CUDA%d chunks of %d tokens (--prefill %s; mixer sub-batches of %d), borrowing "
                         "%.0f MB of the expert pool while a prompt runs (activations %.0f, weight scratch %.0f, expert "
                         "staging %.0f, prestage %.0f: %d experts); disk landing ring %d experts (%.0f MB pinned)\n",
                 dev_, S->T, S->mode.c_str(),
                 mixer_sub((size_t) S->T, g, has_kda, has_dsa, has_dense, has_moe, S->max_pools, S->sub_env),
                 (double) S->borrow_bytes / 1048576.0, (double) S->arena_bytes / 1048576.0,
                 (double) (w16_elems * 2 + w32_elems * 4) / 1048576.0, (double) S->gbuf_bytes / 1048576.0,
                 (double) S->pbuf_bytes / 1048576.0, S->NP, S->nland, (double) S->nland * (double) gstride / 1048576.0);
    return true;
}

size_t Glm5Model::prefill_borrow_bytes() const { return pf_ ? pf_->borrow_bytes : 0; }

// the slots a layer the pool keeps while a prompt runs (prefill_setup: the full keep, or the small one on a pool that
// could lend no chunk at that); without a prompt path, the full keep
int Glm5Model::prefill_keep() const {
    return pf_ != nullptr && pf_->keep > 0 ? (int) pf_->keep : g_.n_exp_used + gf::kSpares + 2;
}

// a pool that cannot lend the whole borrow: the prestage buffer shrinks to what it can (none below 16 slots)
size_t Glm5Model::prefill_trim_prestage(size_t max_borrow) {
    PrefillState* S = pf_;
    if (S == nullptr) return 0;
    if (S->borrow_bytes <= max_borrow || S->pbuf_bytes == 0) return S->borrow_bytes;
    const size_t base = S->borrow_bytes - ((S->pbuf_bytes + 255u) & ~(size_t) 255u);
    int np = base < max_borrow ? (int) ((max_borrow - base) / std::max<size_t>(1, S->gstride)) : 0;
    if (np < 16) np = 0;
    std::fprintf(stderr, "glm prefill: CUDA%d prestage buffer %d -> %d experts (what the pool can lend)\n", dev_, S->NP, np);
    S->NP = np;
    S->pbuf_bytes = (size_t) np * S->gstride;
    S->borrow_bytes = base + ((S->pbuf_bytes + 255u) & ~(size_t) 255u);
    return S->borrow_bytes;
}

bool Glm5Model::prefill_bind(uint8_t* region, size_t bytes, std::string& err) {
    PrefillState* S = pf_;
    if (S == nullptr) return true;
    if (bytes < S->borrow_bytes) {
        err = "glm prefill: the lendable tail (" + std::to_string(bytes >> 20) + " MB) is smaller than the prompt path (" +
              std::to_string(S->borrow_bytes >> 20) + " MB)";
        return false;
    }
    S->region = region;
    S->region_bytes = bytes;
    prefill_carve(S->T);
    return true;
}

// a chunk of T tokens' device buffers: {the rows kept across a layer, the largest of the mixers' / FFN's / NextN cache
// fill's scratch} - the layout prefill_carve lays out
static std::pair<size_t, size_t> chunk_bytes(const Glm5Geometry& g, size_t T, bool has_kda, bool has_dsa,
                                             bool has_dense, bool has_moe, int max_pools, int sub_env, bool mtp,
                                             size_t v41_row = 0) {
    const size_t E = (size_t) g.n_embd;
    Carve a;
    a.take<float>(T * 4 * E);                    // R
    a.take<float>(T * E);                        // x
    a.take<uint16_t>(T * E);                     // x16
    a.take<float>(T * E);                        // mixer
    a.take<float>(T * E);                        // ffn
    a.take<float>(T * 4);
    a.take<float>(T * 4);
    a.take<float>(T * 4);                        // pre2
    a.take<float>(T * 4);                        // onehot
    a.take<float>(T * 16);
    a.take<float>(T);
    a.take<float>(T * 24);
    a.take<int>(T * (size_t) g.n_exp_used + kIdsPad);   // iota (+ the tile over-read's pad)
    if (v41_row) a.take<uint8_t>(T * v41_row);   // v41_rows
    size_t uni = 0;
    const size_t ts = std::min<size_t>(T, mixer_sub(T, g, has_kda, has_dsa, has_dense, has_moe, max_pools, sub_env));
    if (has_kda) { Carve k; carve_kda(k, ts, g); uni = std::max(uni, k.off); }
    if (g.qwen35()) { Carve q; carve_q35(q, ts, g); carve_q4x(q, ts, g); uni = std::max(uni, q.off); }
    if (has_dsa) { Carve d; g.dsv4() ? (void) carve_ds4(d, ts, g, max_pools) : (void) carve_dsa(d, ts, g, max_pools); uni = std::max(uni, d.off); }
    if (has_dense) { Carve d; carve_dense(d, ts, g); uni = std::max(uni, d.off); }
    if (has_moe) { Carve m; carve_moe(m, T, (size_t) (sub_env > 0 ? sub_env : kSub), g); uni = std::max(uni, m.off); }
    // the NextN block's cache fill (the last half) carves 4 n_embd + 1.5 n_embd rows of its own
    if (mtp) uni = std::max(uni, T * (size_t) (6 * g.n_embd + g.kv_lora + 2 * g.idx_key) * 4 + 8 * 256);
    return {a.off, uni};
}

Glm5Model::PinnedPrompt Glm5Model::prefill_pinned() const {
    const PrefillState* S = pf_;
    if (S == nullptr) return {};
    const size_t T = (size_t) S->T, E = (size_t) g_.n_embd;
    PinnedPrompt p;
    p.staging = T * E * sizeof(float) + (S->gpin != nullptr ? (size_t) S->nland * S->gstride : 0);
    p.hop = (size_t) 2 * T * 4 * E * sizeof(float);
    return p;
}

std::pair<size_t, size_t> Glm5Model::prefill_bytes_for(size_t T) const {
    const PrefillState* S = pf_;
    return chunk_bytes(g_, T, S->has_kda, S->has_dsa, S->has_dense, S->has_moe, S->max_pools, S->sub_env, mtp_il_ >= 0,
                       prefill_v41_row());
}

size_t Glm5Model::prefill_v41_row() const {
    if (!g_.v41) return 0;
    const size_t keep = ((size_t) (max_ctx_ + g_.cand_bs) / (size_t) g_.cand_bs + 15) / 16 * 16;
    return (size_t) g_.idx_top_k * sizeof(int) + keep;
}


// what a part with these layers lends the prompt path for a chunk of T tokens, its prestage buffer aside (the split
// search's startability gate: prefill_setup's layout without a PrefillState)
size_t glm_prefill_lend_bytes(const Glm5Geometry& g, size_t T, bool has_kda, bool has_dsa, bool has_dense,
                              bool has_moe, bool mtp, int64_t max_ctx, size_t gstride) {
    const int max_pools = (int) std::max<int64_t>(1, max_ctx / g.idx_kpool);
    const int sub_env = prefill_sub_env();
    const auto pu = chunk_bytes(g, T, has_kda, has_dsa, has_dense, has_moe, max_pools, sub_env, mtp);
    const auto ws = weight_scratch(g, has_dsa, g.v41);   // (v41: an IQ1_M part's run_blas_dq scratch, at most)
    Carve c;
    c.take<uint8_t>(pu.first + pu.second);
    c.take<uint16_t>((size_t) ws.first);
    c.take<float>((size_t) ws.second);
    c.take<uint8_t>(kPfWorkspace);
    c.take<uint8_t>(has_moe ? (size_t) Glm5Model::PrefillState::NG * Glm5Model::PrefillState::GE * gstride : 0);
    return c.off;
}

// the prompt path's device buffers for chunks of T tokens, carved from the start of the lendable region; returns the
// bytes that layout uses (a short prompt borrows - and evicts - only that much of the pool's tail)
size_t Glm5Model::prefill_carve(int Tn) {
    PrefillState* S = pf_;
    const Glm5Geometry& g = g_;
    const size_t T = (size_t) Tn, E = (size_t) g.n_embd;
    S->sub = mixer_sub(T, g, S->has_kda, S->has_dsa, S->has_dense, S->has_moe, S->max_pools, S->sub_env);
    const auto [pers, uni] = prefill_bytes_for(T);
    Carve c{S->region};
    S->arena = c.take<uint8_t>(pers + uni);
    S->w16 = c.take<uint16_t>((size_t) S->w16_elems);
    S->w32 = c.take<float>((size_t) S->w32_elems);
    S->ws = c.take<uint8_t>(S->ws_bytes);
    S->gbuf = S->gbuf_bytes ? c.take<uint8_t>(S->gbuf_bytes) : nullptr;
    // the prestage slots this chunk gets: T / 128 (16 .. the buffer's), unless STRATA_GLM_PRESTAGE_CARVE says
    // otherwise (0: the whole buffer, as before) - the rest of the buffer stays with the pool's experts
    static const long long carve_env = [] { const char* v = getenv("STRATA_GLM_PRESTAGE_CARVE"); return v ? std::atoll(v) : -1LL; }();
    S->pbuf_carved = 0;
    if (S->pbuf_bytes && Tn >= kPreMinT && S->gstride > 0) {
        const size_t all = S->pbuf_bytes / S->gstride;
        size_t slots = carve_env == 0 ? all : carve_env > 0 ? (size_t) carve_env : std::max<size_t>(16, T / 128);
        slots = std::min(slots, all);
        S->pbuf_carved = slots * S->gstride;
    }
    S->pbuf = S->pbuf_carved ? c.take<uint8_t>(S->pbuf_carved) : nullptr;   // (a short prompt: none)
    Carve a{S->arena};
    S->R = a.take<float>(T * 4 * E);
    S->x = a.take<float>(T * E);
    S->x16 = a.take<uint16_t>(T * E);
    S->mixer = a.take<float>(T * E);
    S->ffn = a.take<float>(T * E);
    S->pre = a.take<float>(T * 4);
    S->post = a.take<float>(T * 4);
    S->pre2 = a.take<float>(T * 4);
    S->onehot = a.take<float>(T * 4);
    S->comb = a.take<float>(T * 16);
    S->ss = a.take<float>(T);
    S->mix = a.take<float>(T * 24);
    S->iota = a.take<int>(T * (size_t) g.n_exp_used + kIdsPad);
    S->v41_rows = prefill_v41_row() ? a.take<uint8_t>(T * prefill_v41_row()) : nullptr;
    S->uni = S->arena + a.off;
    cublasSetWorkspace(S->blas, S->ws, S->ws_bytes);
    S->T_bound = Tn;
    S->need = c.off;
    return c.off;
}

// The pool's tail -> the prompt path: every lendable slot's expert leaves VRAM (it is only on disk until the decode
// fetches it again), a spare there leaves the spare table, and the device tables learn it before any prompt work.
bool Glm5Model::prefill_lend(std::string& err) {
    FastState* F = fast_;
    PrefillState* S = pf_;
    if (F == nullptr || S == nullptr || S->lent) return true;
    cudaSetDevice(dev_);
    const uint64_t moved0 = S->moved, dropped0 = S->dropped;
    if (!lend_tail(S->need, S->moved, S->dropped, err)) return false;
    static const bool verbose = getenv("STRATA_GLM_PREFILL_VERBOSE") != nullptr;
    if (verbose)
        std::fprintf(stderr, "glm prefill: CUDA%d lends %.0f MB of the expert pool for %d tokens: %llu experts leave "
                             "VRAM (%llu of them moved over colder ones)\n", dev_, S->need / 1048576.0, S->T_bound,
                     (unsigned long long) (S->dropped - dropped0), (unsigned long long) (S->moved - moved0));
    // the borrowed memory holds expert bytes: what the prompt path reads before writing is set up again
    mmq::iota(S->iota, (int64_t) S->T_bound * g_.n_exp_used + kIdsPad, F->cs);
    for (int b = 0; b < PrefillState::NG; ++b) cudaEventRecord(S->ev_free[b], F->cs);
    if (S->ev_pfree) cudaEventRecord(S->ev_pfree, F->cs);
    S->pre_layer = -1;
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm prefill lend", err)) return false;
    S->lent = true;
    return true;
}

// The on-demand vision encoder (serve VLEND): the whole tail is emptied the same way and its memory freed, so the
// encoder's process can allocate it; the prompt path waits (token by token) until vision_reclaim gives it back.
bool Glm5Model::vision_lend(size_t& bytes, std::string& err) {
    FastState* F = fast_;
    bytes = 0;
    if (F == nullptr || !vis_lend_ok_) {
        err = "this engine has no pool tail to lend to the vision encoder";
        return false;
    }
    if (vis_lent_) {
        bytes = F->xpool_bytes;
        return true;
    }
    cudaSetDevice(dev_);
    uint64_t moved = 0, dropped = 0;
    if (!lend_tail(F->xpool_bytes, moved, dropped, err)) return false;
    cudaFree(F->xpool);
    F->xpool = nullptr;
    for (auto& P : F->lp) P.xbase = nullptr;
    vis_lent_ = true;
    bytes = F->xpool_bytes;
    std::fprintf(stderr, "glm fast: CUDA%d %.2f GB lent to the vision encoder (%llu experts moved to colder slots, "
                         "%llu dropped)\n", dev_, (double) bytes / 1073741824.0, (unsigned long long) moved,
                 (unsigned long long) (dropped - moved));
    return true;
}

size_t Glm5Model::vision_lend_bytes() const { return fast_ != nullptr && vis_lend_ok_ ? fast_->xpool_bytes : 0; }

// ... and back once the encoder has ended: the tail is allocated again (false while the memory is still taken) and
// its slots are free - the decode's misses refill them
bool Glm5Model::vision_reclaim(std::string& err) {
    FastState* F = fast_;
    if (F == nullptr || !vis_lent_) return true;
    cudaSetDevice(dev_);
    if (cudaMalloc(&F->xpool, F->xpool_bytes) != cudaSuccess) {
        cudaGetLastError();
        F->xpool = nullptr;
        err = "the pool's tail is not free yet";
        return false;
    }
    uint8_t* b = F->xpool;
    for (int il = l0_; il < lt_; ++il) {
        if (!F->L[(size_t) il].moe) continue;
        auto& P = F->lp[(size_t) il];
        P.xbase = b;
        b += (size_t) (P.n - P.n_main) * P.stride;
    }
    if (pf_ != nullptr && !prefill_bind(F->xpool, F->xpool_bytes, err)) return false;
    {
        std::lock_guard<std::mutex> lk(F->mu);
        for (int il = l0_; il < lt_; ++il) {
            auto& P = F->lp[(size_t) il];
            for (int s = P.n_main; s < P.n; ++s)
                if (P.st[(size_t) s] == FastState::kLent) P.st[(size_t) s] = FastState::kFree;
        }
    }
    vis_lent_ = false;
    std::fprintf(stderr, "glm fast: CUDA%d the vision encoder's %.2f GB are back in the expert pool\n", dev_,
                 (double) F->xpool_bytes / 1073741824.0);
    return true;
}

// The tail's slots below xpool + limit leave the decode: a resident expert takes the slot of a colder main-slot
// resident (a device copy) or goes; a spare there leaves the spare table; the slots are kLent after this.
bool Glm5Model::lend_tail(size_t limit, uint64_t& moved, uint64_t& dropped, std::string& err) {
    FastState* F = fast_;
    // background moves in flight land first and no new ones start (fast_boundary step 5): a slot with a copy on its
    // way would otherwise be lent under it
    F->bg_hold = true;
    F->bg_drain(nullptr, SIZE_MAX);   // the paced promotions' remaining pieces go now
    // Retire metadata as well as DMA. With the hold set, the boundary cannot start replacement moves.
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->copy), "glm lending copies", err) || !fast_boundary(err))
        return false;
    if (!F->bg.empty() || !F->draining.empty()) {
        err = "glm lending: tier moves were not retired";
        return false;
    }
    const int NE = g_.n_expert;
    {
        std::lock_guard<std::mutex> lk(F->mu);
        int nu = 0;
        bool ok = true;
        // the boundary's current pinned set (its last flush's DMA may still be in flight: wait for its event)
        const int ub = F->upd_buf;
        int* kh = F->upd_key_h[ub];
        unsigned long long* vh = F->upd_val_h[ub];
        ok = glmfast::cuda_ok(cudaEventSynchronize(F->upd_ev[ub]), "glm lending set reuse", err);
        const auto flush = [&]() {
            if (nu == 0 || !ok) return;
            ok = glmfast::cuda_ok(cudaMemcpyAsync(F->upd_key_d[ub], kh, (size_t) nu * sizeof(int),
                                                  cudaMemcpyHostToDevice, F->cs), "glm lending keys", err) &&
                 glmfast::cuda_ok(cudaMemcpyAsync(F->upd_val_d[ub], vh,
                                                  (size_t) nu * sizeof(unsigned long long), cudaMemcpyHostToDevice,
                                                  F->cs), "glm lending values", err);
            if (ok) {
                gf::tab_update(F->tab, F->upd_key_d[ub], F->upd_val_d[ub], nu, F->cs);
                ok = glmfast::cuda_ok(cudaGetLastError(), "glm lending table update", err);
            }
            const cudaError_t e = cudaStreamSynchronize(F->cs);   // pinned sources may now be reused
            if (ok) ok = glmfast::cuda_ok(e, "glm lending table sync", err);
            nu = 0;
        };
        // (every key is edited at most once here, so a batch never carries two values for one key)
        const auto upd = [&](size_t k, unsigned long long v) {
            if (!ok) return;
            if (nu == FastState::kMaxUpd) flush();
            if (!ok) return;
            kh[nu] = (int) k;
            vh[nu] = v;
            ++nu;
        };
        // STRATA_GLM_LEND_DROP=1: drop the lent slots' experts as before (instead of moving them, below)
        static const bool lend_drop = getenv("STRATA_GLM_LEND_DROP") != nullptr;
        // an expert that leaves VRAM here goes to the RAM tier when it has room (a free slot, else a colder entry):
        // the decode after the prompt then finds it in RAM, not on disk (with the experts all in VRAM - a split
        // across many GPUs - the RAM tier holds little else, and every lent expert was a ~20 ms disk read back)
        std::vector<char> parked;   // (keys parked by this lending: never its victims - one table edit per key)
        const auto park = [&](int il, int s, int key) -> bool {
            if (lend_drop || F->layer_rc[(size_t) il] < 0) return false;
            if (parked.empty()) parked.assign(F->cnt.size(), 0);
            auto& R = F->rc[(size_t) F->layer_rc[(size_t) il]];
            int rs = -1;
            for (int s2 = 0; s2 < R.n && rs < 0; ++s2)
                if (R.st[(size_t) s2] == FastState::kRFree) rs = s2;
            if (rs < 0) {
                uint32_t bc = F->cnt[(size_t) key];
                for (int s2 = 0; s2 < R.n; ++s2)
                    if (R.st[(size_t) s2] == FastState::kRHold && !parked[(size_t) R.key[(size_t) s2]] &&
                        F->cnt[(size_t) R.key[(size_t) s2]] < bc) {
                        bc = F->cnt[(size_t) R.key[(size_t) s2]];
                        rs = s2;
                    }
                if (rs < 0) return false;
                const int old = R.key[(size_t) rs];
                upd(F->rtab_key((size_t) old), 0ull);
                F->ram_of[(size_t) old] = -1;
                F->left[(size_t) old] = 2;
            }
            // on the compute stream: the prompt's work behind it cannot overwrite the slot before the copy read it
            if (!glmfast::cuda_ok(cudaMemcpyAsync(R.base + (size_t) rs * R.stride,
                                                   F->lp[(size_t) il].slot_ptr(s), F->L[(size_t) il].blob,
                                                   cudaMemcpyDeviceToHost, F->cs), "glm lending park", err)) {
                ok = false;
                return false;
            }
            R.key[(size_t) rs] = key;
            R.tick[(size_t) rs] = F->clock;
            R.st[(size_t) rs] = FastState::kRHold;
            F->ram_of[(size_t) key] = rs;
            parked[(size_t) key] = 1;
            upd(F->rtab_key((size_t) key), (unsigned long long) (R.base + (size_t) rs * R.stride));
            ++F->diag_lend_park;
            return true;
        };
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe) continue;
            auto& P = F->lp[(size_t) il];
            // the layer's main-slot residents, coldest first: a lent slot's expert moves over the coldest one that is
            // colder than it (a device copy) - the prompt costs the decode its coldest experts, not whichever sat in
            // the borrowed tail (a chat turn used to drop up to ~145 warm experts to disk)
            std::vector<std::pair<uint64_t, int>> cold;   // (count << 32 | recency rank, slot)
            if (!lend_drop)
                for (int s = 0; s < P.n_main; ++s)
                    if (P.st[(size_t) s] == FastState::kResident && P.key[(size_t) s] >= 0)
                        cold.push_back({(uint64_t) F->cnt[(size_t) il * NE + P.key[(size_t) s]], s});
            std::sort(cold.begin(), cold.end(), [&](const auto& a, const auto& b) {
                return a.first != b.first ? a.first < b.first : P.tick[(size_t) a.second] < P.tick[(size_t) b.second];
            });
            size_t ci = 0;
            for (int s = P.n_main; s < P.n; ++s) {
                // a slot past the bytes this lending uses (a prompt's layout) stays with the decode
                if (P.slot_ptr(s) >= F->xpool + limit) continue;
                char& st = P.st[(size_t) s];
                if (st == FastState::kResident && P.key[(size_t) s] >= 0) {
                    const int key = il * NE + P.key[(size_t) s];
                    if (ci < cold.size() && cold[ci].first < (uint64_t) F->cnt[(size_t) key]) {
                        const int v = cold[ci++].second;   // the coldest: it goes, this expert takes its slot
                        const int vkey = il * NE + P.key[(size_t) v];
                        const bool parked = park(il, v, vkey);   // (read before the copy below overwrites the slot)
                        if (!ok || !glmfast::cuda_ok(cudaMemcpyAsync(P.slot_ptr(v), P.slot_ptr(s),
                                                                     F->L[(size_t) il].blob, cudaMemcpyDeviceToDevice,
                                                                     F->cs), "glm lending relocation", err))
                            return false;
                        upd(F->tab_key((size_t) vkey), 0ull);
                        F->slot_of[(size_t) vkey] = -1;
                        if (!parked) F->left[(size_t) vkey] = 3;
                        upd(F->tab_key((size_t) key), (unsigned long long) P.slot_ptr(v));
                        F->slot_of[(size_t) key] = v;
                        P.key[(size_t) v] = P.key[(size_t) s];
                        P.tick[(size_t) v] = P.tick[(size_t) s];
                        ++moved;
                    } else {
                        upd(F->tab_key((size_t) key), 0ull);
                        F->slot_of[(size_t) key] = -1;
                        if (!park(il, s, key)) F->left[(size_t) key] = 3;
                        if (!ok) return false;
                    }
                    ++dropped;
                    ++F->diag_lend;
                } else if (st == FastState::kSpare) {
                    for (int j = 0; j < gf::kSpares; ++j)
                        if (P.spare[j] == s) {
                            P.spare[j] = -1;
                            upd(F->spare_key(il, j), 0ull);
                        }
                }
                P.key[(size_t) s] = -1;
                st = FastState::kLent;
            }
        }
        flush();
        if (!ok) return false;
    }
    F->bg_hold = false;
    return true;
}

// ... and back: the slots are free again; the decode's boundaries hand them out as spares (misses refill them)
bool Glm5Model::prefill_return(std::string& err) {
    FastState* F = fast_;
    PrefillState* S = pf_;
    if (F == nullptr || S == nullptr || !S->lent) return true;
    cudaSetDevice(dev_);
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm return compute", err) ||
        !glmfast::cuda_ok(cudaStreamSynchronize(F->copy), "glm return copies", err)) return false;
    std::lock_guard<std::mutex> lk(F->mu);
    for (int il = l0_; il < lt_; ++il) {
        auto& P = F->lp[(size_t) il];
        for (int s = P.n_main; s < P.n; ++s)
            if (P.st[(size_t) s] == FastState::kLent) P.st[(size_t) s] = FastState::kFree;
    }
    S->lent = false;
    return true;
}

void Glm5Model::prefill_destroy() {
    PrefillState* S = pf_;
    if (S == nullptr) return;
    cudaSetDevice(dev_);
    if (fast_ && fast_->cs) cudaStreamSynchronize(fast_->cs);
    if (fast_ && fast_->copy) cudaStreamSynchronize(fast_->copy);
    if (S->prof && !S->pacc.empty()) {
        double tot = 0;
        for (auto& kv : S->pacc) tot += kv.second;
        std::fprintf(stderr, "glm prefill prof CUDA%d (%lld tokens, %.0f ms of stream time; waits on disk %.0f ms, plan %.0f "
                             "ms host; disk experts used %llu, read %llu):\n", dev_, (long long) S->tokens, tot, S->ms_disk,
                     S->ms_plan, (unsigned long long) S->staged_disk, (unsigned long long) S->disk_issued);
        std::vector<std::pair<double, std::string>> v;
        for (auto& kv : S->pacc) v.push_back({kv.second, kv.first});
        std::sort(v.rbegin(), v.rend());
        for (auto& e : v)
            std::fprintf(stderr, "  %-14s %9.1f ms  %5.1f%%  (%.3f ms/token)\n", e.second.c_str(), e.first,
                         100.0 * e.first / std::max(1e-9, tot), e.first / (double) std::max<int64_t>(1, S->tokens));
    }
    for (auto e : S->pev) cudaEventDestroy(e);
    for (auto& t : S->tr)
        for (cudaEvent_t e : {t.start, t.plan, t.pre, t.c1, t.end})
            if (e) cudaEventDestroy(e);
    if (S->xs) cudaStreamSynchronize(S->xs);
    if (fast_ && fast_->igpu_unreg)
        for (const void* p : S->ig_reg) fast_->igpu_unreg(p);
    S->ig_reg.clear();
    if (S->x_h) cudaFreeHost(S->x_h);
    if (S->rt_h) cudaFreeHost(S->rt_h);
    if (S->id_h) cudaFreeHost(S->id_h);
    if (S->out_h) cudaFreeHost(S->out_h);
    if (S->ev_x) cudaEventDestroy(S->ev_x);
    if (S->ev_out) cudaEventDestroy(S->ev_out);
    if (S->ev_xd) cudaEventDestroy(S->ev_xd);
    if (S->ev_c0) cudaEventDestroy(S->ev_c0);
    if (S->ev_c1) cudaEventDestroy(S->ev_c1);
    for (auto e : S->ev_pdone)
        if (e) cudaEventDestroy(e);
    if (S->ev_pready) cudaEventDestroy(S->ev_pready);
    if (S->ev_pstart) cudaEventDestroy(S->ev_pstart);
    if (S->ev_pfree) cudaEventDestroy(S->ev_pfree);
    if (S->xs) cudaStreamDestroy(S->xs);
    S->mq.reset();
    if (S->blas) cublasDestroy(S->blas);
    for (int b = 0; b < PrefillState::NG; ++b) {
        if (S->ev_ready[b]) cudaEventDestroy(S->ev_ready[b]);
        if (S->ev_free[b]) cudaEventDestroy(S->ev_free[b]);
    }
    for (auto e : S->ev_land)
        if (e) cudaEventDestroy(e);
    if (S->ev_hop) cudaEventDestroy(S->ev_hop);
    if (S->gpin) cudaFreeHost(S->gpin);
    if (S->emb_h) cudaFreeHost(S->emb_h);
    if (S->toks_d) cudaFree(S->toks_d);
    if (S->img_d) cudaFree(S->img_d);
    if (S->hop_h) cudaFreeHost(S->hop_h);
    if (S->h_counts) cudaFreeHost(S->h_counts);
    if (S->h_base) cudaFreeHost(S->h_base);
    if (S->h_bounds) cudaFreeHost(S->h_bounds);
    delete S;
    pf_ = nullptr;
}


// ---------------------------------------------------------------- deepseek4: the attention mixer of a sub-batch
// The token path's fast_dsv4_attn for tn tokens at once (docs/deepseek4/SPEC.md): the projections as FP16 GEMMs on
// dequantized weights, then the batched kernels of dsv4_fast.cu - the window keys in a linear buffer (the ring's
// last positions first), every block that ends in the sub-batch compressed before any query reads it, the indexer
// per token, attention with sinks over each token's window and compressed rows, the grouped output LoRA.
bool Glm5Model::prefill_ds4_mixer(int il, int64_t p0_64, int tn, const uint16_t* x16, const float* xs, float* mixer,
                                  bool skip_output, void* uni, std::string& err) {
    namespace ds = strata::kernels::dsv4;
    FastState* F = fast_;
    PrefillState* S = pf_;
    const Glm5Geometry& g = g_;
    const cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, D = g.head_dim, r = Ly.ratio, p0 = (int) p0_64;
    const int HD = g.n_head * D, OL = g.o_groups * g.o_lora, IQ = g.idx_heads * g.idx_key;
    const float one = 1.0f, zero = 0.0f;
    const auto hgemm = [&](const WSlot& w, int N, int Kd, const uint16_t* X16, int ldx, float* Y, int ldy) {
        const int rows = (int) std::max<int64_t>(1, std::min<int64_t>(N, S->w16_elems / Kd));
        for (int r0 = 0; r0 < N; r0 += rows) {
            const int n = std::min(rows, N - r0);
            if (w.type == gf::kTypeBF16)
                gb::bf16_to_f16((const uint16_t*) w.q + (size_t) r0 * Kd, S->w16, (int64_t) n * Kd, s);
            else
                strata::kernels::dequant_f16(w.type, w.q, r0, n, Kd, S->w16, s);
            blas_ck(cublasGemmEx(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, tn, Kd, &one, S->w16, CUDA_R_16F, Kd, X16,
                                 CUDA_R_16F, ldx, &zero, Y + r0, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "ds4 gemm f16");
        }
    };
    const auto make_rope = [&](bool comp) {
        return comp ? ds::make_rope(g.n_rot, g.comp_rope_base, 1.0f / g.yarn_factor, 1.0f, g.yarn_orig_ctx,
                                    g.yarn_beta_fast, g.yarn_beta_slow)
                    : ds::make_rope(g.n_rot, g.rope_base, 1.0f, 0.0f, 0, 0.0f, 0.0f);
    };
    const ds::Rope rope = make_rope(r != 0), crope = make_rope(true);
    Carve c{(uint8_t*) uni};
    Ds4Bufs B = carve_ds4(c, (size_t) tn, g, S->max_pools);
    // 1. the projections
    hgemm(Ly.q_a, g.q_lora, E, x16, E, B.qr_raw, g.q_lora);
    hgemm(Ly.kv, D, E, x16, E, B.kv_raw, D);
    const int cw = r == 4 ? 2 * D : D;
    if (r != 0) {
        hgemm(Ly.comp_kv, cw, E, x16, E, B.ckv, cw);
        hgemm(Ly.comp_gate, cw, E, x16, E, B.csc, cw);
    }
    if (r == 4) {
        hgemm(Ly.idx_comp_kv, 2 * g.idx_key, E, x16, E, B.ikv, 2 * g.idx_key);
        hgemm(Ly.idx_comp_gate, 2 * g.idx_key, E, x16, E, B.isc, 2 * g.idx_key);
        const float alpha = 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads));
        blas_ck(cublasSgemm(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, g.idx_heads, tn, E, &alpha, Ly.idx_proj_f, E, xs, E,
                            &zero, B.iw, g.idx_heads),
                "ds4 indexer weights");
    }
    S->mark("ds4_proj", s);
    // 2. the window keys (the ring's last positions, then this sub-batch's), q_a's norm, the compressors
    const int n_hist = std::min(p0, g.n_swa - 1);
    ds::win_load(Ly.raw, ds_ring_, B.keys, n_hist, p0, D, s);
    ds::BatchPrepArgs a;
    a.qr_raw = B.qr_raw;
    a.q_a_norm = Ly.q_a_norm;
    a.qr = B.qr;
    a.qr16 = B.qr16;
    a.q_lora = g.q_lora;
    a.kv_raw = B.kv_raw;
    a.kv_norm = Ly.kv_a_norm;
    a.keys = B.keys;
    a.n_hist = n_hist;
    a.head_dim = D;
    a.n_rot = g.n_rot;
    a.rope = rope;
    a.p0 = p0;
    a.T = tn;
    a.eps = g.norm_eps;
    ds::prep_batch(a, s);
    ds::win_store(Ly.raw, ds_ring_, B.keys, n_hist, p0, tn, D, s);
    ds::BatchComp comps[2];
    int nc = 0;
    if (r != 0) {
        ds::BatchComp& b = comps[nc++];
        b.c.ring_kv = Ly.ring_kv;
        b.c.ring_sc = Ly.ring_sc;
        b.c.cache = Ly.comp_cache;
        b.c.ape = Ly.comp_ape;
        b.c.norm = Ly.comp_norm;
        b.c.D = D;
        b.c.ratio = r;
        b.c.overlap = r == 4;
        b.kv = B.ckv;
        b.sc = B.csc;
    }
    if (r == 4) {
        ds::BatchComp& b = comps[nc++];
        b.c.ring_kv = Ly.iring_kv;
        b.c.ring_sc = Ly.iring_sc;
        b.c.cache = Ly.idx_cache;
        b.c.ape = Ly.idx_comp_ape;
        b.c.norm = Ly.idx_comp_norm;
        b.c.D = g.idx_key;
        b.c.ratio = 4;
        b.c.overlap = 1;
        b.kv = B.ikv;
        b.sc = B.isc;
    }
    ds::compress_batch(comps, nc, p0, tn, g.n_rot, crope, g.norm_eps, s);
    S->mark("ds4_prep", s);
    // 3. q (and the indexer's q), per-head norm and RoPE
    hgemm(Ly.q_b, HD, g.q_lora, B.qr16, g.q_lora, B.q, HD);
    if (r == 4) hgemm(Ly.idx_q_b_q, IQ, g.q_lora, B.qr16, g.q_lora, B.iq, IQ);
    ds::q_prep_batch(B.q, g.n_head, D, g.n_rot, rope, r == 4 ? B.iq : nullptr, g.idx_heads, g.idx_key, crope, p0, tn,
                     g.norm_eps, s);
    S->mark("ds4_q", s);
    // 4. the indexer, the attention
    if (r == 4) ds::indexer_batch(B.iq, Ly.idx_cache, B.iw, g.idx_heads, 4, p0, tn, g.idx_top_k, B.score, B.score_ld,
                                  B.sel, s);
    S->mark("ds4_index", s);
    ds::attend_batch(B.q, B.keys, n_hist, g.n_swa, Ly.comp_cache, B.sel, g.idx_top_k, r, Ly.sinks, g.n_head, D, g.n_rot,
                     rope, p0, tn, B.o, s);
    S->mark("ds4_attn", s);
    if (skip_output) return true;
    // 5. the grouped output LoRA, then wo_b
    ds::to_f16(B.o, B.o16, (int64_t) tn * HD, s);
    const int gin = HD / g.o_groups;
    const size_t wrow = gf::row_bytes(Ly.out_a.type, gin);
    if (wrow == 0) {
        err = "deepseek4 prefill: wo_a type " + std::to_string(Ly.out_a.type);
        return false;
    }
    for (int gi = 0; gi < g.o_groups; ++gi) {
        WSlot sl = Ly.out_a;
        sl.q = (const uint8_t*) Ly.out_a.q + (size_t) gi * g.o_lora * wrow;
        hgemm(sl, g.o_lora, gin, B.o16 + (size_t) gi * gin, HD, B.oa + (size_t) gi * g.o_lora, OL);
    }
    ds::to_f16(B.oa, B.oa16, (int64_t) tn * OL, s);
    hgemm(Ly.out_b, E, OL, B.oa16, OL, mixer, E);
    abl_apply(il, 0, mixer, tn, E, s);
    S->mark("ds4_out", s);
    return true;
}

// ---------------------------------------------------------------- deepseek41: the attention mixer of a sub-batch
// fast_dsv41_attn for tn tokens: its projections as FP16 GEMMs on dequantized weights (a token at a time each read
// the layer's ~136 MB of attention weights again: ~0.24 of its 0.31 ms on an AI PRO R9700), q_a's norm and q_b over
// the sub-batch; what has an order stays a token at a time with the token path's kernels - the window ring, the
// KV source's compressed row, q's RoPE, the indexer (the row's own selection and candidate mask: rows row0..), the
// attention - then the grouped output LoRA and wo_b batched again.  skip_output (the layer a prompt skip cuts at, the
// tail): its ring and compressed rows only, which is all a later row reads of it
bool Glm5Model::prefill_v41_mixer(int il, int64_t p0_64, int tn, int row0, const uint16_t* x16, const float* xs,
                                  float* mixer, bool skip_output, void* uni, std::string& err) {
    namespace ds = strata::kernels::dsv4;
    namespace d41 = strata::kernels::dsv41;
    FastState* F = fast_;
    PrefillState* S = pf_;
    const Glm5Geometry& g = g_;
    const cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, D = g.head_dim, r = Ly.ratio, p0 = (int) p0_64;
    const int HD = g.n_head * D, OL = g.o_groups * g.o_lora, IQ = g.idx_heads * g.idx_key;
    const float one = 1.0f, zero = 0.0f;
    const auto hgemm = [&](const WSlot& w, int N, int Kd, const uint16_t* X16, int ldx, float* Y, int ldy) {
        const int rows = (int) std::max<int64_t>(1, std::min<int64_t>(N, S->w16_elems / Kd));
        for (int r0 = 0; r0 < N; r0 += rows) {
            const int n = std::min(rows, N - r0);
            if (w.type == gf::kTypeBF16)
                gb::bf16_to_f16((const uint16_t*) w.q + (size_t) r0 * Kd, S->w16, (int64_t) n * Kd, s);
            else
                strata::kernels::dequant_f16(w.type, w.q, r0, n, Kd, S->w16, s);
            blas_ck(cublasGemmEx(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, tn, Kd, &one, S->w16, CUDA_R_16F, Kd, X16,
                                 CUDA_R_16F, ldx, &zero, Y + r0, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "v41 gemm f16");
        }
    };
    const ds::Rope rope = r != 0 ? ds::make_rope(g.n_rot, g.comp_rope_base, 1.0f / g.yarn_factor, 1.0f, g.yarn_orig_ctx,
                                                 g.yarn_beta_fast, g.yarn_beta_slow)
                                 : ds::make_rope(g.n_rot, g.rope_base, 1.0f, 0.0f, 0, 0.0f, 0.0f);
    const ds::Rope crope = ds::make_rope(g.n_rot, g.comp_rope_base, 1.0f / g.yarn_factor, 1.0f, g.yarn_orig_ctx,
                                         g.yarn_beta_fast, g.yarn_beta_slow);
    Carve c{(uint8_t*) uni};
    Ds4Bufs B = carve_ds4(c, (size_t) tn, g, S->max_pools);
    const bool attn = !skip_output;   // (a skipped output: no reader of this layer's queries either)
    // 1. the projections of x
    hgemm(Ly.q_a, g.q_lora, E, x16, E, B.qr_raw, g.q_lora);   // (prep reads it whether or not a query follows)
    hgemm(Ly.kv, D, E, x16, E, B.kv_raw, D);
    if (Ly.kv_src) {
        hgemm(Ly.comp_kv, D, E, x16, E, B.ckv, D);
        if (r == 2) hgemm(Ly.comp_gate, D, E, x16, E, B.csc, D);
    }
    if (attn && Ly.idx_src) {
        const float alpha = 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads));
        blas_ck(cublasSgemm(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, g.idx_heads, tn, E, &alpha, Ly.idx_proj_f, E, xs, E,
                            &zero, B.iw, g.idx_heads),
                "v41 indexer weights");
    }
    // STRATA_GLM_V41_ATTN_TOK=1: steps 2-3 a token at a time on the token path's kernels (A/B, v1.3.59's path)
    static const bool attn_tok = [] { const char* v = getenv("STRATA_GLM_V41_ATTN_TOK"); return v && std::atoi(v) != 0; }();
    const size_t vr = prefill_v41_row();
    if (!attn_tok) {
        S->mark("v41_proj", s);
        // 2. the window keys (the ring's last positions, then this sub-batch's: prep over the batch), q_a's norm, the
        // KV source's compressed rows
        const int n_hist = std::min(p0, g.n_swa - 1);
        ds::win_load(Ly.raw, ds_ring_, B.keys, n_hist, p0, D, s);
        ds::BatchPrepArgs a;
        a.qr_raw = B.qr_raw;
        a.q_a_norm = Ly.q_a_norm;
        a.qr = B.qr;
        a.qr16 = B.qr16;
        a.q_lora = g.q_lora;
        a.kv_raw = B.kv_raw;
        a.kv_norm = Ly.kv_a_norm;
        a.keys = B.keys;
        a.n_hist = n_hist;
        a.head_dim = D;
        a.n_rot = g.n_rot;
        a.rope = rope;
        a.p0 = p0;
        a.T = tn;
        a.eps = g.norm_eps;
        ds::prep_batch(a, s);
        ds::win_store(Ly.raw, ds_ring_, B.keys, n_hist, p0, tn, D, s);
        if (Ly.kv_src) {
            d41::PublishArgs pa;
            pa.v = B.ckv;
            pa.g = r == 2 ? B.csc : nullptr;
            pa.pend = F->v41_pend[Ly.slot];
            pa.norm = Ly.comp_norm;
            pa.idx_k = (const uint16_t*) Ly.idx_k_w.q;
            pa.idx_k_norm = Ly.idx_k_norm;
            pa.comp_cache = Ly.comp_cache;
            pa.idx_cache = Ly.idx_cache;
            pa.ratio = r;
            pa.D = D;
            pa.idx_dim = g.idx_key;
            pa.n_rot = g.n_rot;
            pa.rope = crope;
            pa.eps = g.norm_eps;
            d41::publish_batch(pa, p0, tn, FastState::kV41PendRing, s);
        }
        if (!attn) {
            S->mark("v41_attn", s);
            return true;
        }
        // 3. q, the indexer's q, their RoPE
        hgemm(Ly.q_b, HD, g.q_lora, B.qr16, g.q_lora, B.q, HD);
        if (Ly.idx_src) hgemm(Ly.idx_q_b_q, IQ, g.q_lora, B.qr16, g.q_lora, B.iq, IQ);
        d41::q_rope_batch(B.q, g.n_head, D, g.n_rot, rope, Ly.idx_src ? B.iq : nullptr, g.idx_heads, g.idx_key, crope,
                          p0, tn, s);
        S->mark("v41_q", s);
        // 4. the indexer (rows whose visible compressed rows outgrow top_k), in row groups the score buffer holds;
        // the candidate blocks / mask a row at a time on the rows past them; the selection into each row's own
        // slot, which the layers reading this source's selection take
        int* const sel0 = (int*) (S->v41_rows + (size_t) row0 * vr);
        const int sel_ld = (int) (vr / sizeof(int));
        if (r != 0 && Ly.idx_src && (p0 + tn) / r > g.idx_top_k) {
            const size_t cap = (size_t) B.score_rows * (size_t) B.score_ld;
            const int cand_min = g.cand_blocks * g.cand_bs;
            // the candidate blocks over a group (v1.3.63: a launch a group, not three or four a row - a 60K prompt
            // took ~44K rows a layer past cand_min) take their scratch from the score buffer's end: a row costs its
            // scores, its block maxima and its block selection.  STRATA_GLM_V41_CAND_ROW=1: a row at a time
            static const bool cand_row = [] { const char* v = getenv("STRATA_GLM_V41_CAND_ROW"); return v && std::atoi(v) == 1; }();
            const int maxv = std::max(1, (p0 + tn) / r);
            const bool cand = il >= g.cand_layer && maxv > cand_min, cand_grp = cand && !cand_row;
            const size_t per = (size_t) maxv + (cand_grp ? (size_t) (maxv + g.cand_bs - 1) / g.cand_bs + g.cand_blocks : 0);
            for (int g0 = 0; g0 < tn;) {
                int gn = (int) std::max<size_t>(1, std::min<size_t>((size_t) (tn - g0), cap / per));
                const int ld = std::max(1, (p0 + g0 + gn) / r);
                ds::indexer_batch_scores(B.iq + (size_t) g0 * IQ, Ly.idx_cache, B.iw + (size_t) g0 * g.idx_heads,
                                         g.idx_heads, r, p0 + g0, gn, B.score, ld, s);
                if (cand_grp) {
                    const int bld = (ld + g.cand_bs - 1) / g.cand_bs;
                    float* const bscore = B.score + (size_t) gn * ld;
                    int* const bsel = (int*) (bscore + (size_t) gn * bld);
                    int* const sel_g = sel0 + (size_t) g0 * sel_ld;
                    if (il == g.cand_layer)
                        d41::candidates_batch(B.score, ld, r, p0 + g0, gn, cand_min, g.cand_bs, g.cand_blocks, bscore,
                                              bld, bsel, sel_g, sel_ld, g.idx_top_k, s);
                    else
                        d41::candidate_mask_batch(B.score, ld, r, p0 + g0, gn, cand_min, g.cand_bs, sel_g, sel_ld,
                                                  g.idx_top_k, s);
                } else if (cand)
                    for (int t = g0; t < g0 + gn; ++t) {
                        const int n_vis = (p0 + t + 1) / r;
                        if (n_vis <= cand_min) continue;
                        float* const sc = B.score + (size_t) (t - g0) * ld;
                        unsigned char* const keep = (unsigned char*) (sel0 + (size_t) t * sel_ld + g.idx_top_k);
                        if (il == g.cand_layer)
                            d41::candidates(sc, n_vis, g.cand_bs, g.cand_blocks, F->v41_bscore, F->v41_bsel, keep, s);
                        else
                            d41::candidate_mask(sc, n_vis, g.cand_bs, keep, s);
                    }
                ds::indexer_batch_select(B.score, ld, r, p0 + g0, gn, g.idx_top_k, sel0 + (size_t) g0 * sel_ld, sel_ld, s);
                g0 += gn;
            }
        }
        S->mark("v41_index", s);
        // 5. attention over each row's window and compressed rows (all visible ones while they fit top_k)
        ds::attend_batch(B.q, B.keys, n_hist, g.n_swa, Ly.comp_cache, sel0, g.idx_top_k, r == 0 ? 0 : -r, Ly.sinks,
                         g.n_head, D, g.n_rot, rope, p0, tn, B.o, s, sel_ld);
    } else {
        // 2. q_a's norm (its q8_1 rows into o16, free until step 5), q_b and the indexer's q
        if (attn) {
            d41::rms_rows(B.qr_raw, tn, Ly.q_a_norm, g.norm_eps, g.q_lora, B.qr, (void*) B.o16, s);
            ds::to_f16(B.qr, B.qr16, (int64_t) tn * g.q_lora, s);
            hgemm(Ly.q_b, HD, g.q_lora, B.qr16, g.q_lora, B.q, HD);
            if (Ly.idx_src) hgemm(Ly.idx_q_b_q, IQ, g.q_lora, B.qr16, g.q_lora, B.iq, IQ);
        }
        S->mark("v41_proj", s);
        // 3. a token at a time: the window ring, the KV source's row, q's RoPE, the indexer, the attention
        for (int t = 0; t < tn; ++t) {
            if ((t & 255) == 255) progress_beat();   // (the watchdog: a slow layer is not a stuck one)
            const int p = p0 + t;
            ds::PrepArgs a;   // (its qr half on row t too: F->qr / qr_q, which nothing reads after)
            a.qr_raw = B.qr_raw + (size_t) t * g.q_lora;
            a.q_a_norm = Ly.q_a_norm;
            a.qr = F->qr;
            a.qr_q = F->qr_q;
            a.q_lora = g.q_lora;
            a.kv_raw = B.kv_raw + (size_t) t * D;
            a.kv_norm = Ly.kv_a_norm;
            a.raw = Ly.raw;
            a.ring = ds_ring_;
            a.head_dim = D;
            a.n_rot = g.n_rot;
            a.rope = rope;
            a.comp_rope = crope;
            a.p = p;
            a.eps = g.norm_eps;
            ds::prep(a, s);
            if (Ly.kv_src) {
                d41::PublishArgs pa;
                pa.v = B.ckv + (size_t) t * D;
                pa.g = r == 2 ? B.csc + (size_t) t * D : nullptr;
                pa.pend = F->v41_pend[Ly.slot] + (size_t) ((p / 2) % FastState::kV41PendRing) * 2 * D;
                pa.norm = Ly.comp_norm;
                pa.idx_k = (const uint16_t*) Ly.idx_k_w.q;
                pa.idx_k_norm = Ly.idx_k_norm;
                pa.comp_cache = Ly.comp_cache;
                pa.idx_cache = Ly.idx_cache;
                pa.ratio = r;
                pa.p = p;
                pa.D = D;
                pa.idx_dim = g.idx_key;
                pa.n_rot = g.n_rot;
                pa.rope = crope;
                pa.eps = g.norm_eps;
                d41::publish(pa, s);
            }
            if (!attn) continue;
            float* const q = B.q + (size_t) t * HD;
            float* const iq = Ly.idx_src ? B.iq + (size_t) t * IQ : nullptr;
            d41::q_rope(q, g.n_head, D, g.n_rot, rope, iq, g.idx_heads, g.idx_key, crope, p, s);
            int* const sel_buf = (int*) (S->v41_rows + (size_t) (row0 + t) * vr);
            unsigned char* const keep_buf = (unsigned char*) (sel_buf + g.idx_top_k);
            int n_comp = 0;
            const int* sel = nullptr;
            if (r != 0) {
                const int n_vis = (p + 1) / r;
                if (n_vis > g.idx_top_k) {
                    if (Ly.idx_src) {
                        ds::indexer_scores(iq, Ly.idx_cache, B.iw + (size_t) t * g.idx_heads, g.idx_heads, n_vis,
                                           F->v41_score, s);
                        if (il == g.cand_layer && n_vis > g.cand_blocks * g.cand_bs)
                            d41::candidates(F->v41_score, n_vis, g.cand_bs, g.cand_blocks, F->v41_bscore, F->v41_bsel,
                                            keep_buf, s);
                        if (il > g.cand_layer && n_vis > g.cand_blocks * g.cand_bs)
                            d41::candidate_mask(F->v41_score, n_vis, g.cand_bs, keep_buf, s);
                        ds::indexer_select(F->v41_score, n_vis, g.idx_top_k, sel_buf, s);
                    }
                    n_comp = g.idx_top_k;
                    sel = sel_buf;
                } else {
                    n_comp = n_vis;
                }
            }
            ds::attend(q, Ly.raw, ds_ring_, std::min(p + 1, g.n_swa), Ly.comp_cache, sel, n_comp, Ly.sinks, g.n_head, D,
                       g.n_rot, rope, p, B.o + (size_t) t * HD, s);
        }
    }
    S->mark("v41_attn", s);
    if (skip_output) return true;
    // 4. the grouped output LoRA, then wo_b
    ds::to_f16(B.o, B.o16, (int64_t) tn * HD, s);
    const int gin = HD / g.o_groups;
    const size_t wrow = gf::row_bytes(Ly.out_a.type, gin);
    if (wrow == 0) {
        err = "deepseek41 prefill: wo_a type " + std::to_string(Ly.out_a.type);
        return false;
    }
    for (int gi = 0; gi < g.o_groups; ++gi) {
        WSlot sl = Ly.out_a;
        sl.q = (const uint8_t*) Ly.out_a.q + (size_t) gi * g.o_lora * wrow;
        hgemm(sl, g.o_lora, gin, B.o16 + (size_t) gi * gin, HD, B.oa + (size_t) gi * g.o_lora, OL);
    }
    ds::to_f16(B.oa, B.oa16, (int64_t) tn * OL, s);
    hgemm(Ly.out_b, E, OL, B.oa16, OL, mixer, E);
    abl_apply(il, 0, mixer, tn, E, s);
    S->mark("v41_out", s);
    return true;
}

// deepseek41: one Engram block's adds (bn tokens, their rows kd floats apart, R's rows 4 x n_embd apart): engram_kv
// as an FP16 GEMM on its dequantized weight - the token path's GEMV read its ~160 MB again for every token, ~0.3 ms a
// token and layer on an AI PRO R9700 - then the gate and add a token a block.  The mixer's sub-batch buffers are free
// here (the Engram add comes before the layer's attention)
bool Glm5Model::prefill_v41_engram_gemm() const { return pf_ != nullptr && pf_->v41_eng_gemm; }

bool Glm5Model::prefill_v41_engram_block(int il, int bn, int kd, const float* rows, float* R, std::string& err) {
    FastState* F = fast_;
    PrefillState* S = pf_;
    const Glm5Geometry& g = g_;
    const cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int N = 5 * g.n_embd;
    const float one = 1.0f, zero = 0.0f;
    Carve c{(uint8_t*) S->uni};
    uint16_t* const x16 = c.take<uint16_t>((size_t) bn * kd);
    float* const kv = c.take<float>((size_t) bn * N);
    strata::kernels::dsv4::to_f16(rows, x16, (int64_t) bn * kd, s);
    const WSlot& w = Ly.eng_kv;
    const int step = (int) std::max<int64_t>(1, std::min<int64_t>(N, S->w16_elems / kd));
    for (int r0 = 0; r0 < N; r0 += step) {
        const int n = std::min(step, N - r0);
        if (w.type == gf::kTypeBF16)
            gb::bf16_to_f16((const uint16_t*) w.q + (size_t) r0 * kd, S->w16, (int64_t) n * kd, s);
        else
            strata::kernels::dequant_f16(w.type, w.q, r0, n, kd, S->w16, s);
        blas_ck(cublasGemmEx(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, bn, kd, &one, S->w16, CUDA_R_16F, kd, x16,
                             CUDA_R_16F, kd, &zero, kv + r0, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "v41 engram gemm");
    }
    (void) err;
    strata::kernels::dsv41::engram_add_rows(R, kv, Ly.eng_qn, Ly.eng_kn, g.n_embd, g.norm_eps, bn, s);
    return true;
}

// ---------------------------------------------------------------- one half's layers over a chunk
bool Glm5Model::prefill_half(int64_t p0, int T, std::string& err, const int32_t* next_ids) {
    FastState* F = fast_;
    PrefillState* S = pf_;
    const Glm5Geometry& g = g_;
    const cudaStream_t s = F->cs;
    const int E = g.n_embd, DI = g.d_inner(), HD = g.kda_head_dim, K = g.n_exp_used;
    const float one = 1.0f, zero = 0.0f;

    // Y[t][N] (row stride ldy) = alpha * X[t][K] (F32, row stride ldx) . W[N][K]^T, W BF16 (widened in slices)
    const auto sgemm_bf16 = [&](const uint16_t* W, int N, int Kd, const float* X, int ldx, float* Y, int ldy, int Tn,
                                float alpha) {
        const int rows = (int) std::max<int64_t>(1, std::min<int64_t>(N, S->w32_elems / Kd));
        for (int r0 = 0; r0 < N; r0 += rows) {
            const int n = std::min(rows, N - r0);
            gb::bf16_to_f32(W + (size_t) r0 * Kd, S->w32, (int64_t) n * Kd, s);
            blas_ck(cublasSgemm(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, Tn, Kd, &alpha, S->w32, Kd, X, ldx, &zero, Y + r0,
                                ldy),
                    "sgemm");
        }
    };
    // Y[t][N] = X16[t][K] . deq(W)[N][K]^T (tensor cores, F32 accumulate); beta = 1 adds to Y.  A BF16 weight (a
    // projection the GGUF keeps unquantized) is narrowed to FP16 the same way
    const auto hgemm_q = [&](const WSlot& w, int N, int Kd, const uint16_t* X16, int ldx, float* Y, int ldy, int Tn,
                             float beta) {
#if defined(STRATA_USE_HIP)
        if (!S->lt_tried) {
            S->lt_tried = true;
            if (getenv("STRATA_HIPBLASLT_TUNING")) {
                auto lt = std::make_unique<strata::prefill::Gemm>();
                std::string lt_err;
                if (lt->init_external(F->cs, S->w16, S->w16_elems, S->ws, S->ws_bytes, lt_err))
                    S->lt = std::move(lt);
                else
                    std::fprintf(stderr, "glm prefill: hipBLASLt GEMM unavailable (%s)\n", lt_err.c_str());
            }
        }
        if (S->lt) S->lt->rebind(S->w16, S->w16_elems, S->ws, S->ws_bytes);
#endif
        // an EXL3 weight: reconstructed in slices of whole 128-row output rotation blocks (at a sub-batch's 1-2K rows
        // the decode once + cuBLAS beat exl3::gemm_rows, which decodes again for every 32 rows: KDA projections 2.3 vs
        // 3.3 s over 9 GPUs for a 6.4K-token prompt)
        const bool x3w = w.type == strata::kernels::exl3::kTypeEXL3;
        int rows = (int) std::max<int64_t>(1, std::min<int64_t>(N, S->w16_elems / Kd));
        if (x3w) rows = std::max(128, rows / 128 * 128);
        for (int r0 = 0; r0 < N; r0 += rows) {
            const int n = std::min(rows, N - r0);
            if (w.type == gf::kTypeBF16)
                gb::bf16_to_f16((const uint16_t*) w.q + (size_t) r0 * Kd, S->w16, (int64_t) n * Kd, s);
            else if (w.type == gf::kTypeF32)   // (ISTA's Qwen3.8: the injects, BF16 in the GGUF, widened at load)
                gb::f32_to_f16((const float*) w.q + (size_t) r0 * Kd, S->w16, (int64_t) n * Kd, s);
            else if (x3w)
                strata::kernels::exl3::reconstruct_rows(
                    strata::kernels::exl3::view(*(const strata::kernels::exl3::Mat*) w.q, r0, n), S->w16, false, s);
            else
                strata::kernels::dequant_f16(w.type, w.q, r0, n, Kd, S->w16, s);
#if defined(STRATA_USE_HIP)
            if (S->lt && ldx == Kd) {
                S->lt->f16(X16, S->w16, Y + r0, Tn, n, Kd, ldy, beta);
                continue;
            }
#endif
            blas_ck(cublasGemmEx(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, Tn, Kd, &one, S->w16, CUDA_R_16F, Kd, X16,
                                 CUDA_R_16F, ldx, &beta, Y + r0, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "gemm f16");
        }
    };
    // qwen4exp's hyper-connection over the chunk, in sub-batches (fast_q4x_layers' q4x_mix): R += out . 2 sigmoid(the
    // writer's inject / hc) (out: null = none), the streams normed, inject = inject_w . xn (inj_out: null = none), the
    // low rank silu(down . xn / hc), gate = up . lo and x = mean_c xn[c] sigmoid(gate[c]) (and x16); w: 0 attention,
    // 1 FFN; combine_only: the combine alone (layer 1's PLE comes between it and the mix)
    const auto q4_mix = [&](int il, int w, const float* out, const float* inj_prev, float* inj_out, bool combine_only) {
        const auto& H = F->L[(size_t) il].hc4[w];
        const int HE = g.hc * E, LR = g.hc_lr;
        for (int t0 = 0; t0 < T; t0 += S->sub) {
            const int tn = std::min(S->sub, T - t0);
            Carve c{S->uni};
            Q4xBufs Q = carve_q4x(c, (size_t) tn, g);
            q35::hc4_combine_norm(S->R + (size_t) t0 * HE, out != nullptr ? out + (size_t) t0 * E : nullptr,
                                  inj_prev != nullptr ? inj_prev + (size_t) t0 * g.hc : nullptr, g.hc, H.norm,
                                  g.norm_eps, E, g.hc, tn, Q.xn, s);
            if (combine_only) continue;
            gb::f32_to_f16(Q.xn, Q.xn16, (int64_t) tn * HE, s);
            hgemm_q(H.down, LR, HE, Q.xn16, HE, Q.lo, LR, tn, 0.0f);
            if (inj_out != nullptr) hgemm_q(H.inject, g.hc, HE, Q.xn16, HE, inj_out + (size_t) t0 * g.hc, g.hc, tn, 0.0f);
            q35::silu_scale(Q.lo, tn * LR, 1.0f / (float) g.hc, s);
            gb::f32_to_f16(Q.lo, Q.lo16, (int64_t) tn * LR, s);
            hgemm_q(H.up, HE, LR, Q.lo16, LR, Q.gate, HE, tn, 0.0f);
            q35::hc4_mix(Q.xn, Q.gate, E, g.hc, tn, S->x + (size_t) t0 * E, s);
        }
        if (!combine_only) gb::f32_to_f16(S->x, S->x16, (int64_t) T * E, s);
        S->mark("q4_hc", s);
    };
    // qwen4exp's PLE (layer 1) over the chunk: the rows' n-gram hashes on the host (the token path's ple_toks carry
    // the history), the table rows in one batched read, then key / value as GEMMs and the postops in sub-batches (the
    // conv history advances across them) - R += gated + silu(conv)
    const auto q4_ple = [&](int64_t p0, std::string& e) -> bool {
        namespace k = strata::kernels;
        const int HE = g.hc * E;
        if (S->toks_h.size() != (size_t) T) { e = "glm prefill: qwen4exp has no host token ids"; return false; }
        if (F->ple_issued) {   // (a token whose layers never ran)
            if (!F->ple->collect(F->ple_emb_h, e)) return false;
            F->ple_issued = false;
        }
        const size_t nt = F->ple_toks.size();
        if (p0 < 0 || (size_t) (p0 + T) > nt) { e = "glm prefill: qwen4exp's PLE history is shorter than the context"; return false; }
        for (int t = 0; t < T; ++t) F->ple_toks[(size_t) (p0 + t)] = S->toks_h[(size_t) t];
        std::vector<int32_t> prev((size_t) T * 2);
        for (int t = 0; t < T; ++t) {
            const int64_t p = p0 + t;
            prev[(size_t) t * 2] = p >= 2 ? F->ple_toks[(size_t) (p - 2)] : -1;
            prev[(size_t) t * 2 + 1] = p >= 1 ? F->ple_toks[(size_t) (p - 1)] : -1;
        }
        static const k::PleConsts consts = k::ple_artifact_consts();
        std::vector<uint32_t> rows((size_t) T * k::PLE_N_HEADS);
        k::ngram_rows(S->toks_h.data(), prev.data(), T, consts, rows.data());
        if (F->ple_pf_n < (size_t) T * E) {
            if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "glm prompt PLE buffer", e)) return false;
            if (F->ple_pf_h != nullptr) cudaFreeHost(F->ple_pf_h);
            F->ple_pf_n = 0;
            if (cudaHostAlloc((void**) &F->ple_pf_h, (size_t) T * E * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
                F->ple_pf_h = nullptr;
                e = "glm prefill: qwen4exp's PLE rows did not allocate";
                return false;
            }
            F->ple_pf_n = (size_t) T * E;
        }
        // (the previous chunk's copies out of it are long done: every MoE layer syncs the stream)
        if (!F->ple->gather_batch(rows.data(), (size_t) T, F->ple_pf_h, e)) { e = "qwen4exp: PLE: " + e; return false; }
        for (int t0 = 0; t0 < T; t0 += S->sub) {
            const int tn = std::min(S->sub, T - t0);
            Carve c{S->uni};
            Q4xBufs Q = carve_q4x(c, (size_t) tn, g);
            cudaMemcpyAsync(Q.emb, F->ple_pf_h + (size_t) t0 * E, (size_t) tn * E * sizeof(float), cudaMemcpyHostToDevice, s);
            gb::f32_to_f16(Q.emb, Q.emb16, (int64_t) tn * E, s);
            hgemm_q(F->ple_key_w, HE, E, Q.emb16, E, Q.key, HE, tn, 0.0f);
            hgemm_q(F->ple_val_w, E, E, Q.emb16, E, Q.val, E, tn, 0.0f);
            k::native_ple_postops_batch(Q.key, S->R + (size_t) t0 * HE, Q.val, F->ple_hist, F->ple_w, Q.qn, Q.gated,
                                        Q.gate4, tn, (void*) s);
        }
        abl_res(S->R, T * g.hc, E, s);   // (the PLE's rows are a residual write too, as V4.1's Engram)
        S->mark("q4_ple", s);
        return true;
    };
    // qwen4exp's attention over a sub-batch (after attn_prep_rows wrote its K / V rows and normed, rotated q): every
    // row's indexer key into the block keys; the rows up to 2,051 cells attend densely (exact: the selection is every
    // cell), the later ones to their own selection (fast_qwen35_attn's), kQ4Grp rows' scores and selections at a time
    const auto q4_attn = [&](int il, int64_t pt0, int tn, const float* xs, const float* q, float* ao, const Q4xBufs& Q,
                             std::string& e) -> bool {
        namespace k = strata::kernels;
        const auto& Ly = F->L[(size_t) il];
        const k::QsaShapes sh = k::qsa_real_shapes();
        const int D = g.head_dim, HDq = g.n_head * D, KV = g.attn_head_kv * D, ID = g.idx_key, IH = g.idx_heads;
        const float scale = 1.0f / std::sqrt((float) D);
        if (Ly.q4_pooled == nullptr) { e = "glm prefill: qwen4exp's attention layer has no block keys"; return false; }
        auto& hs = S->q4_steps_h;
        hs.assign((size_t) tn * (4 + IH), 0);
        for (int t = 0; t < tn; ++t) {
            k::qsa_step_fill(hs.data() + (size_t) t * 4, pt0 + t, sh);
            for (int h = 0; h < IH; ++h) hs[(size_t) tn * 4 + (size_t) t * IH + h] = (int32_t) (pt0 + t);
        }
        cudaMemcpyAsync(Q.steps, hs.data(), (size_t) tn * 4 * sizeof(int32_t), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(Q.ipos, hs.data() + (size_t) tn * 4, (size_t) tn * IH * sizeof(int32_t), cudaMemcpyHostToDevice, s);
        sgemm_bf16((const uint16_t*) Ly.q4_ik.q, ID, E, xs, E, Q.kraw, ID, tn, 1.0f);
        const k::QsaIndexerBuffers ib{Ly.q4_tail, Ly.q4_dead, Ly.q4_pooled, Ly.q4_bpos};
        // STRATA_Q4X_QSA_BATCH (default on for CUDA, off on HIP): the sub-batch's indexer keys in two launches, not
        // one a row, and below a group's attention in one launch, not two a row; the same bytes either way
#if defined(STRATA_USE_HIP)
        static const bool qsa_batch = [] { const char* v = getenv("STRATA_Q4X_QSA_BATCH"); return v != nullptr && std::atoi(v) != 0; }();
#else
        static const bool qsa_batch = [] { const char* v = getenv("STRATA_Q4X_QSA_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
#endif
        if (qsa_batch)
            k::indexer_key_append_rows(Q.kraw, ID, pt0, tn, 0, Ly.q4_ikn, g.norm_eps, ib, sh, F->q4_cos, F->q4_sin,
                                       (void*) s);
        else
            for (int t = 0; t < tn; ++t)
                k::indexer_key_append(Q.kraw + (size_t) t * ID, Q.steps + (size_t) t * 4 + k::kStepPos, 0, Ly.q4_ikn,
                                      g.norm_eps, ib, sh, F->q4_cos, F->q4_sin, (void*) s);
        const int64_t wmax = sh.idx_top_k + sh.idx_block - 1;   // the rows below see every cell
        const int nd = (int) std::max<int64_t>(0, std::min<int64_t>(tn, wmax - pt0));
        if (nd > 0) q35::attn_prompt(q, Ly.kc, Ly.vc, g.n_head, g.attn_head_kv, D, pt0, nd, scale, ao, s);
        const int ns = tn - nd;
        if (ns <= 0) return true;
        float* iq = Q.iq;
        sgemm_bf16((const uint16_t*) Ly.q4_iq.q, IH * ID, E, xs + (size_t) nd * E, E, iq, IH * ID, ns, 1.0f);
        k::rms_norm_weighted(iq, Ly.q4_iqn, (int64_t) ns * IH, ID, g.norm_eps, (void*) s);
        k::rope_neox_apply(iq, iq, (int64_t) ns * IH, ID, g.n_rot, F->q4_cos, F->q4_sin, Q.ipos + (size_t) nd * IH,
                           (void*) s);
        const int G = FastState::kQ4Grp;
        for (int r0 = 0; r0 < ns; r0 += G) {
            const int ng = std::min(G, ns - r0);
            const int32_t* st = Q.steps + (size_t) (nd + r0) * 4;
            k::qsa_block_scores(Ly.q4_pooled, Ly.q4_dead, iq + (size_t) r0 * IH * ID, st, ng, F->q4_max_blocks, sh,
                                F->q4_pf_scores, (void*) s);
            k::qsa_block_topk(F->q4_pf_scores, st, ng, F->q4_max_blocks, F->q4_cap, sh, F->q4_pf_ids, (void*) s);
            // the group's rows at once, each over its own selection read in place (else a row at a time through a
            // gathered copy)
            if (qsa_batch && F->q4_pf_part != nullptr &&
                q35::attn_decode_rows(q + (size_t) (nd + r0) * HDq, Ly.kc, Ly.vc, F->q4_pf_ids, F->q4_cap, st, ng,
                                      F->q4_cap, g.n_head, g.attn_head_kv, D, scale, F->q4_pf_part,
                                      ao + (size_t) (nd + r0) * HDq, s))
                continue;
            for (int j = 0; j < ng; ++j) {
                const int t = nd + r0 + j;
                const int w = (int) k::qsa_selection_width(pt0 + t + 1, sh);
                q35::gather_rows(Ly.kc, Ly.vc, F->q4_pf_ids + (size_t) j * F->q4_cap, w, KV, F->q4_ks, F->q4_vs, s);
                q35::attn_decode(q + (size_t) t * HDq, F->q4_ks, F->q4_vs, g.n_head, g.attn_head_kv, D, w, scale,
                                 F->q35_part, ao + (size_t) t * HDq, s);
            }
        }
        S->mark("q4_qsa", s);
        return true;
    };
    // the mHC read half for the whole chunk: (write half of the previous block,) mix dots, gates, x / x16
    const auto hc_read = [&](const float* block_out, const uint16_t* fn, const float* scale, const float* base,
                             const float* norm, const float* pre_use = nullptr, float* pre_out = nullptr) {
        gb::hc_update(block_out, S->R, S->post, S->comb, S->ss, T, E, s);
        sgemm_bf16(fn, 24, 4 * E, S->R, 4 * E, S->mix, 24, T, 1.0f);
        gb::HcFinishArgs h;
        h.mix = S->mix;
        h.ss = S->ss;
        h.R = S->R;
        h.w_scale = scale;
        h.w_base = base;
        h.norm_w = norm;
        h.norm_eps = g.norm_eps;
        h.hc_eps = g.hc_eps;
        h.iters = g.sinkhorn_iters;
        h.n_embd = E;
        h.T = T;
        h.pre = pre_out != nullptr ? pre_out : S->pre;
        h.pre_use = pre_use;
        h.post = S->post;
        h.comb = S->comb;
        h.x = S->x;
        h.x16 = S->x16;
        gb::hc_finish(h, s);
    };

    // GLM_CB_DIR seam dumps of the chunk's LAST row, under the token path's names (debug only, synchronous)
    static const char* cb_dir = getenv("GLM_PF_CB_DIR") != nullptr ? getenv("GLM_PF_CB_DIR") : getenv("GLM_CB_DIR");   // (GLM_PF_CB_DIR: apart from the token path's)
    // A single-device prompt chunk only conditions later tokens; decode computes the last prompt token and its
    // logits separately. The terminal layer's outputs have no reader unless NextN is loaded. Keep its mixer/cache
    // updates, but omit the output projection and FFN (including both mHC write halves). Split boundaries and seam
    // dumps consume these rows, so keep their full path. STRATA_GLM_PREFILL_TAIL_SKIP=0 is the A/B baseline.
    const char* tail_env = getenv("STRATA_GLM_PREFILL_TAIL_SKIP");
    const bool tail_skip = !prefill_logits && l0_ == 0 && l1_ == g.n_layers && split_next_ == nullptr && mtp_il_ < 0 && dsp_ == nullptr && q4d_ == nullptr &&
                           (cb_dir == nullptr || !cb_dir[0]) && (tail_env == nullptr || std::atoi(tail_env) != 0);
    const auto dump_row = [&](const std::string& name, const float* rows, int width, int row = -1) {
        if (cb_dir == nullptr || !cb_dir[0]) return;
        cudaStreamSynchronize(s);
        std::vector<float> buf((size_t) width);
        cudaMemcpy(buf.data(), rows + (size_t) (row < 0 ? T - 1 : row) * width, (size_t) width * sizeof(float),
                   cudaMemcpyDeviceToHost);
        if (FILE* f = std::fopen((std::string(cb_dir) + "/" + name + ".f32").c_str(), "wb")) {
            std::fwrite(buf.data(), 4, buf.size(), f);
            std::fclose(f);
        }
    };

    // ---- the disk reader: one thread for the chunk, reading the experts the MoE layers append - (layer, expert) in
    //      order - into the pinned landing ring, while the resident product and the RAM-tier groups run (Mercury,
    //      2.6k-token prompt: 5.1 -> 4.5 ms/token).  A layer appends its disk-only experts once its plan is known.
    //      From kPredT tokens on (STRATA_GLM_PREFILL_PRED_T=<tokens>, 0 = never), a layer also appends the NEXT MoE
    //      layer's whole disk-only set (a superset of what it will route), so those reads overlap its attention too.
    //      With the 12-slot ring it was slower (the reader could not get ahead); with the deep ring it pays from 2k
    //      chunks (Mercury 16k prompt at 2048-token chunks: 394 -> 417 tok/s).  A slot is refilled once the copy that
    //      read it ran (ev_land) and the layers moved past it.
    const int KL = S->nland;
    constexpr int kPredT = 1024;
    struct DiskQ {
        std::mutex mu;
        std::vector<std::pair<int, int>> q;
        std::atomic<int> n{0}, landed{0}, copied{0};
        std::atomic<bool> stop{false};
        int push(int il, int e) {
            std::lock_guard<std::mutex> lk(mu);
            q.emplace_back(il, e);
            n.store((int) q.size(), std::memory_order_release);
            return (int) q.size() - 1;
        }
    } dq;
    // STRATA_GLM_DISK_QD=<n>: experts read at once (one batch across the workers) when they are queued and their
    // slots free - an NVMe gives more with a few in flight (Uranus 1.87 -> 2.2 GB/s at 2, Mercury 2.65 -> 2.96).
    // Up to 32: on Windows each read has a fixed cost, so more in flight matter more (RX 7900 XTX, PCIe 4 NVMe,
    // with STRATA_GLM_READ_CHUNKS=1: 1.66 GB/s at 2 -> 4.27 GB/s at 16)
    static constexpr int kMaxDiskQd = 32;
    static const int disk_qd = [] {
        const char* v = getenv("STRATA_GLM_DISK_QD");
        return std::max(1, std::min(kMaxDiskQd, v ? std::atoi(v) : 2));
    }();
    // STRATA_GLM_DISK_READERS=<n>: n reader threads, each with one expert in flight (its three parts read whole), every
    // expert handed on the moment it and those before it landed - instead of one reader reading disk_qd experts at a
    // time across the workers and handing the batch on when all of it landed.  A DRAM-less NVMe (Kingston NV3) reads
    // 1.6 GB/s in the 256 KB pieces the workers read and 3.0 GB/s in 2 MB ones; a deeper batch bought short prompts the
    // throughput (514 tokens: 6.8 -> 5.9 s) but cost long ones the first expert's wait (8k tokens: 9.7 -> 10.7 s)
    static const int n_readers = [] {
        const char* v = getenv("STRATA_GLM_DISK_READERS");
        return v ? std::max(0, std::min(16, std::atoi(v))) : 0;
    }();
    std::atomic<int> rnext{0};
    std::mutex land_mu;
    std::vector<int> land_seq((size_t) std::max(1, KL), -1);
    const auto reader_one = [&] {
        cudaSetDevice(dev_);
        for (;;) {
            const int k = rnext.fetch_add(1);
            for (int spin = 0; dq.n.load(std::memory_order_acquire) <= k ||
                               (k >= KL && dq.copied.load(std::memory_order_acquire) < k - KL + 1);
                 ++spin) {
                if (dq.stop.load()) return;
                if (spin < 1024) std::this_thread::yield();
                else std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
            cudaEventSynchronize(S->ev_land[(size_t) (k % KL)]);   // the slot's last copy (this chunk's or earlier)
            std::pair<int, int> it;
            {
                std::lock_guard<std::mutex> lk(dq.mu);
                it = dq.q[(size_t) k];
            }
            uint8_t* land = S->gpin + (size_t) (k % KL) * S->gstride;
            for (int part = 0; part < 3; ++part) fast_read_part(it.first, it.second, part, land, 0, 1);
            std::lock_guard<std::mutex> lk(land_mu);
            land_seq[(size_t) (k % KL)] = k;
            int l = dq.landed.load(std::memory_order_relaxed);
            while (land_seq[(size_t) (l % KL)] == l) ++l;
            dq.landed.store(l, std::memory_order_release);
        }
    };
    std::thread reader([&] {
        cudaSetDevice(dev_);
        if (n_readers > 0) {
            std::vector<std::thread> more;
            for (int i = 1; i < n_readers; ++i) more.emplace_back(reader_one);
            reader_one();
            for (auto& t : more) t.join();
            return;
        }
        // each part in pieces across the workers, as the decode's disk path (STRATA_GLM_READ_CHUNKS)
        static const int kChunks = [] {
            const char* v = getenv("STRATA_GLM_READ_CHUNKS");
            return std::max(1, std::min(16, v ? std::atoi(v) : 8));
        }();
        for (int k = 0;;) {
            while (dq.n.load(std::memory_order_acquire) <= k ||
                   (k >= KL && dq.copied.load(std::memory_order_acquire) < k - KL + 1)) {
                if (dq.stop.load()) return;
                std::this_thread::yield();
            }
            // this one, and the ones after it that are queued and whose slots are free already
            int nb = 1;
            while (nb < disk_qd && dq.n.load(std::memory_order_acquire) > k + nb &&
                   (k + nb < KL || dq.copied.load(std::memory_order_acquire) >= k + nb - KL + 1))
                ++nb;
            std::pair<int, int> it[kMaxDiskQd];
            uint8_t* land[kMaxDiskQd];
            for (int b = 0; b < nb; ++b) {
                cudaEventSynchronize(S->ev_land[(k + b) % KL]);   // the slot's last copy (this chunk's or earlier)
                std::lock_guard<std::mutex> lk(dq.mu);
                it[b] = dq.q[(size_t) (k + b)];
                land[b] = S->gpin + (size_t) ((k + b) % KL) * S->gstride;
            }
            F->workers->run(nb * 3 * kChunks, [&](int job) {
                const int b = job / (3 * kChunks), part = job % (3 * kChunks);
                fast_read_part(it[b].first, it[b].second, part / kChunks, land[b], part % kChunks, kChunks);
            });
            k += nb;
            dq.landed.store(k, std::memory_order_release);
        }
    });
    struct Joiner {
        std::thread& t;
        DiskQ& q;
        uint64_t& issued;
        ~Joiner() {
            q.stop.store(true);
            if (t.joinable()) t.join();
            issued += (uint64_t) q.n.load();
        }
    } joiner{reader, dq, S->disk_issued};
    // STRATA_GLM_PREFILL_PRED_T=<tokens>: the chunk size from which the next layer's disk set is read ahead (0 = never)
    static const int pred_t = [] {
        const char* v = getenv("STRATA_GLM_PREFILL_PRED_T");
        return v ? std::atoi(v) : kPredT;
    }();
    int pred_layer = -1, pred0 = 0;   // the predicted segment: its layer, first index and experts (ascending)
    std::vector<int> pred_e;
    // the experts of layer il that are neither resident in its main slots nor held by the RAM tier (ascending)
    const auto disk_only = [&](int il) {
        std::vector<int> v;
        const auto& P = F->lp[(size_t) il];
        const auto& RC = F->rc[(size_t) F->layer_rc[(size_t) il]];
        std::vector<char> res((size_t) g.n_expert, 0);
        std::lock_guard<std::mutex> lk(F->mu);
        for (int sl = 0; sl < P.n_main; ++sl)
            if (P.st[(size_t) sl] == FastState::kResident && P.key[(size_t) sl] >= 0) res[(size_t) P.key[(size_t) sl]] = 1;
        for (int e = 0; e < g.n_expert; ++e) {
            if (res[(size_t) e]) continue;
            const int rs = F->ram_of[(size_t) il * g.n_expert + e];
            if (rs >= 0 && RC.st[(size_t) rs] == FastState::kRHold) continue;
            const int vs = F->slot_of[(size_t) il * g.n_expert + e];   // a tail slot this prompt did not borrow
            if (vs >= 0 && P.st[(size_t) vs] == FastState::kResident) continue;
            v.push_back(e);
        }
        return v;
    };

    // ---- PRESTAGING: layer il's predicted experts -> pbuf on the copy stream, behind what it carries (the layer
    //      before's staged groups) and once the layer before has read pbuf (ev_pfree)
    // a long deepseek41 prompt's chunk before its cut (prefill_run): layers up to kv_cut only, kv_cut's FFN skipped
    const int kv_cut = pf_kv_cut_;
    const int moe_end = kv_cut >= 0 ? kv_cut : l1_;   // (MoE layers that read experts: before it)
    const auto next_moe = [&](int il) {
        while (il < moe_end && !F->L[(size_t) il].moe) ++il;
        return il < moe_end ? il : l1_;
    };
    // whether layer il's host experts go to the iGPU (abyss_igpu_rows: glm_fast's light row kernels on the APU's
    // integrated GPU, reading the RAM tier in place - an expert of ~20 rows 0.22 ms against ~1.2 on the 16 CPU
    // threads, Radeon 8060S).  STRATA_GLM_PREFILL_IGPU=0: the CPU threads.  A call it cannot take: the CPU threads.
    static const bool pf_igpu = [] {
        const char* v = getenv("STRATA_GLM_PREFILL_IGPU");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const auto igpu_lane = [&](int il) {
        const auto& Ly = F->L[(size_t) il];
        const auto ok = [](int t) { return t == 16 || t == 17 || t == 18 || t == 19; };
        return pf_igpu && F->igpu_rows != nullptr && ok(Ly.gu_type) && ok(Ly.d_type) && 2 * g.n_ff_exp <= E &&
               E % 256 == 0;
    };
    const auto prestage = [&](int il) {
        S->pre_layer = -1;
        S->pre_e.clear();
        if (S->pbuf == nullptr || T < kPreMinT || il >= l1_ || !F->L[(size_t) il].moe || F->layer_rc[(size_t) il] < 0 ||
            (tail_skip && il == g.n_layers - 1))   // the skipped terminal FFN reads no experts
            return;
        const auto& Ly = F->L[(size_t) il];
        const auto& P = F->lp[(size_t) il];
        const auto& RC = F->rc[(size_t) F->layer_rc[(size_t) il]];
        const int NE = g.n_expert;
        // an expert's predicted rows: its share of this machine's routing (the usage profile, which the prompts
        // feed) and of the layer's last chunk, half each
        double us = 0.0, ls = 0.0;
        for (int e = 0; e < NE; ++e) us += F->usage[(size_t) il * NE + e];
        const auto& lc = S->last_cnt[(size_t) il];
        for (int c : lc) ls += c;
        struct Cand {
            double pred;
            int e, rs;
        };
        std::vector<Cand> cand;
        {
            std::lock_guard<std::mutex> lk(F->mu);
            for (int e = 0; e < NE; ++e) {
                const size_t key = (size_t) il * NE + e;
                const int vs = F->slot_of[key];
                if (vs >= 0 && vs < P.n && P.st[(size_t) vs] == FastState::kResident && P.key[(size_t) vs] == e)
                    continue;   // in VRAM (a main slot, or a tail slot this prompt did not borrow)
                const int rs = F->ram_of[key];
                if (rs < 0 || RC.st[(size_t) rs] != FastState::kRHold) continue;   // (disk: the reader's)
                double sh = 0.0, w = 0.0;
                if (us > 0.0) sh += F->usage[key] / us, w += 1.0;
                if (ls > 0.0 && !lc.empty()) sh += lc[(size_t) e] / ls, w += 1.0;
                const double pred = (w > 0.0 ? sh / w : 1.0 / NE) * (double) T * K;
                if (pred >= kPreMinRows) cand.push_back(Cand{pred, e, rs});
            }
        }
        // the layer's count: the buffer's, or (STRATA_GLM_PRESTAGE_ADAPT=1) what its mixer's window took before.
        // deepseek4 adapts by default (=0: the whole buffer): with the CPU lane's AVX-512 i-quant rows the host takes
        // the lightly routed experts the full buffer would have copied - Ryzen AI Max+ 395 + RTX 5090 on PCIe Gen4 x4,
        // UD-Q2_K_XL: a 5.6K prompt 693 -> 1065 tok/s, an 18K one 1752 -> 1720
        static const bool adapt = [&] {
            const char* v = getenv("STRATA_GLM_PRESTAGE_ADAPT");
            return v != nullptr && v[0] != '\0' ? std::atoi(v) != 0 : g.arch == 1;
        }();
        const int cap = std::min(S->NP, (int) std::min<size_t>(S->pbuf_carved / P.stride,
                                                                S->ev_pdone.size() * PrefillState::kPreEv));
        // (deepseek4 before its first lesson: 64 experts for a 5.6K-token chunk, the cap from ~14K tokens on - the best
        // fixed counts measured there, so a server's first prompt does not pay the whole buffer)
        // (with the iGPU taking the host's experts, ~5x the CPU threads' rate: 32 for 8K tokens - 24 to 40 measured best)
        const double first = igpu_lane(il) ? 32.0 / 8000.0 : 64.0 / 5600.0;
        const double rate = S->pre_rate[Ly.recr ? 0 : 1] > 0.0 ? S->pre_rate[Ly.recr ? 0 : 1]
                                                                 : (adapt && g.arch == 1 ? first : 0.0);
        const int want = !adapt || rate <= 0.0 ? cap : std::max(std::min(cap, 16), std::min(cap, (int) std::lround(rate * T)));
        const int n = (int) std::min<size_t>((size_t) want, cand.size());
        if (n == 0) return;
        std::partial_sort(cand.begin(), cand.begin() + n, cand.end(),
                          [](const Cand& a, const Cand& b) { return a.pred > b.pred; });
        cudaStreamWaitEvent(F->copy, S->ev_pfree, 0);
        cudaEventRecord(S->ev_pstart, F->copy);
        for (int j = 0; j < n; ++j) {
            cudaMemcpyAsync(S->pbuf + (size_t) j * P.stride, RC.base + (size_t) cand[(size_t) j].rs * RC.stride, Ly.blob,
                            cudaMemcpyHostToDevice, F->copy);
            if ((j + 1) % PrefillState::kPreEv == 0 || j + 1 == n)
                cudaEventRecord(S->ev_pdone[(size_t) j / PrefillState::kPreEv], F->copy);
            S->pre_e.push_back(cand[(size_t) j].e);
        }
        cudaEventRecord(S->ev_pready, F->copy);
        if (S->trace) {
            cudaEventRecord(S->tr[(size_t) il].pre, F->copy);
            S->tr[(size_t) il].pre_n = n;
        }
        S->pre_layer = il;
        S->pre_issued += (uint64_t) n;
    };

    // whether layer il's plan may give experts to the CPU (STRATA_GLM_PREFILL_CPU=0: never)
    static const bool pf_cpu = [] {
        const char* v = getenv("STRATA_GLM_PREFILL_CPU");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const auto cpu_lane = [&](int il) {
        return pf_cpu && F->cpu_pool && F->cpu_c_ms > 0.0 && F->cpu_p_ms > 2.0 * F->cpu_c_ms &&
               (size_t) il < F->cpu_fmt.size() && F->cpu_fmt[(size_t) il].n_ff > 0 &&
               !F->L[(size_t) il].exl3;   // (EXL3: no multi-row CPU kernel yet - its staged experts all go over PCIe)
    };
    // the CPU lane's stream (x down, the rows' tokens down) and its pinned buffers
    const auto cpu_bufs = [&](int cpu_nrows) {
        bool ok = true;
        if (S->xs == nullptr)
            ok = cudaStreamCreateWithFlags(&S->xs, cudaStreamNonBlocking) == cudaSuccess &&
                 cudaEventCreateWithFlags(&S->ev_x, cudaEventDisableTiming) == cudaSuccess &&
                 cudaEventCreateWithFlags(&S->ev_out, cudaEventDisableTiming) == cudaSuccess &&
                 cudaEventCreateWithFlags(&S->ev_xd, cudaEventDisableTiming) == cudaSuccess;
        // (the iGPU reads x and the rows' tokens and writes the outputs in place: registered with it)
        const auto unreg = [&](const void* p) {
            if (p == nullptr) return;
            const auto it = std::find(S->ig_reg.begin(), S->ig_reg.end(), p);
            if (it == S->ig_reg.end()) return;
            if (F->igpu_unreg) F->igpu_unreg(p);
            S->ig_reg.erase(it);
        };
        const auto reg = [&](const void* p, size_t bytes) {
            if (F->igpu_rows == nullptr || F->igpu_reg == nullptr || p == nullptr) return;
            char e[256] = {0};
            if (F->igpu_reg(p, bytes, e, (int) sizeof e) == 0) S->ig_reg.push_back(p);
            else std::fprintf(stderr, "glm prefill: the iGPU cannot read a host buffer (%s) - the CPU threads compute\n", e);
        };
        if (ok && S->x_h_n < (size_t) T * E) {
            unreg(S->x_h);
            if (S->x_h) cudaFreeHost(S->x_h);
            S->x_h_n = (size_t) T * E;
            ok = cudaHostAlloc((void**) &S->x_h, S->x_h_n * sizeof(float), cudaHostAllocDefault) == cudaSuccess;
            if (ok) reg(S->x_h, S->x_h_n * sizeof(float));
        }
        if (ok && S->rows_h_n < (size_t) cpu_nrows) {
            unreg(S->rt_h);
            unreg(S->id_h);
            unreg(S->out_h);
            if (S->rt_h) cudaFreeHost(S->rt_h);
            if (S->id_h) cudaFreeHost(S->id_h);
            if (S->out_h) cudaFreeHost(S->out_h);
            S->id_h = nullptr;
            // (what this layer needs and a quarter more, at most a chunk's routes: T x K rows of n_embd floats
            // would be 2 GB of pinned memory at 16K tokens)
            S->rows_h_n = std::min((size_t) T * K, (size_t) cpu_nrows + (size_t) cpu_nrows / 4 + 64);
            ok = cudaHostAlloc((void**) &S->rt_h, S->rows_h_n * sizeof(int), cudaHostAllocDefault) == cudaSuccess &&
                 cudaHostAlloc((void**) &S->out_h, S->rows_h_n * E * sizeof(float), cudaHostAllocDefault) == cudaSuccess;
            if (ok && F->igpu_rows != nullptr &&
                cudaHostAlloc((void**) &S->id_h, S->rows_h_n * sizeof(int), cudaHostAllocDefault) == cudaSuccess) {
                for (size_t r = 0; r < S->rows_h_n; ++r) S->id_h[r] = (int) r;
                reg(S->id_h, S->rows_h_n * sizeof(int));
            }
            if (ok) {
                reg(S->rt_h, S->rows_h_n * sizeof(int));
                reg(S->out_h, S->rows_h_n * E * sizeof(float));
            }
        }
        return ok;
    };

    S->mark("start", s);
    if (g.v41) {   // the chunk's Engram row numbers (the n-gram history advances) and layer 0's one-hot pre
        if (S->toks_h.size() != (size_t) T) { err = "glm prefill: deepseek41 has no host token ids"; return false; }
        if (!dsv41_engram_hash(S->toks_h.data(), p0, T, err)) return false;
        std::vector<float> oh((size_t) T * 4, 0.0f);
        for (int t = 0; t < T; ++t) oh[(size_t) t * 4] = 1.0f;
        cudaMemcpyAsync(S->onehot, oh.data(), oh.size() * sizeof(float), cudaMemcpyHostToDevice, s);
    }
    prestage(next_moe(l0_));   // (the dense layers before the first MoE one run meanwhile)
    for (auto& t : S->tr) t.on = false;
    for (int il = l0_; il < l1_; ++il) {
        progress_at("the prompt: layer", il);   // a layer is the serve watchdog's heartbeat (issue #40)
        progress_beat();
        if (kv_cut >= 0 && il > kv_cut) break;
        const auto& Ly = F->L[(size_t) il];
        const bool skip_output = (tail_skip && il == g.n_layers - 1) || il == kv_cut;
        if (S->trace) cudaEventRecord(S->tr[(size_t) il].start, s);
        if (g.v41) {
            // deepseek41: an Engram layer writes the previous FFN alone, adds its rows, then reads without a write;
            // the read collapses with the previous sublayer's pre (layer 0: the one-hot)
            const float* blk = il == l0_ ? nullptr : S->ffn;
            if (Ly.eng >= 0) {
                if (blk != nullptr) gb::hc_update(blk, S->R, S->post, S->comb, nullptr, T, E, s);
                if (!dsv41_engram_prompt(il, T, S->R, err)) return false;
                abl_res(S->R, T * 4, E, s);   // (Engram's rows are a residual write too)
                blk = nullptr;
                S->mark("engram", s);
            }
            hc_read(blk, Ly.hc_attn_fn, Ly.hc_attn_scale, Ly.hc_attn_base, Ly.attn_norm,
                    il == l0_ ? S->onehot : S->pre2, S->pre);
            if (dsp_ != nullptr && il > l0_) dspark_capture_rows(il - 1, S->R, T, s);   // DSpark's features
        } else if (g.q4x) {
            // qwen4exp: the last FFN combined (with its inject) and the attention's mix; layer 1 first adds the PLE
            if (il == F->ple_il && F->ple != nullptr) {
                if (il > l0_) q4_mix(il, 0, S->ffn, S->post, nullptr, true);
                if (!q4_ple(p0, err)) return false;
                q4_mix(il, 0, nullptr, nullptr, S->pre, false);
            } else {
                q4_mix(il, 0, il == l0_ ? nullptr : S->ffn, S->post, S->pre, false);
            }
        } else if (g.qwen35()) {
            // a plain residual (R's first stream): it takes the last FFN, x = rms(R) * attn_norm and its FP16
            q35::add_rms_rows(S->R, 4 * E, il == l0_ ? nullptr : S->ffn, Ly.attn_norm, g.norm_eps, T, E, S->x, S->x16, s);
        } else {
            hc_read(il == l0_ ? nullptr : S->ffn, Ly.hc_attn_fn, Ly.hc_attn_scale, Ly.hc_attn_base, Ly.attn_norm);
            if (dsp_ != nullptr && il > l0_) dspark_capture_rows(il - 1, S->R, T, s);   // DSpark's features
        }
        S->mark("hc", s);
        dump_row("attn_norm-" + std::to_string(il), S->x, E);

        // ---- the mixer, in sub-batches of S->sub tokens (the recurrences carry their state across)
        for (int t0 = 0; t0 < T; t0 += S->sub) {
            const int tn = std::min(S->sub, T - t0);
            const float* xs = S->x + (size_t) t0 * E;
            const uint16_t* x16 = S->x16 + (size_t) t0 * E;
            float* mixer = S->mixer + (size_t) t0 * E;
            if (g.v41 && S->has_dsa) {
                if (!prefill_v41_mixer(il, p0 + t0, tn, t0, x16, xs, mixer, skip_output, (void*) S->uni, err))
                    return false;
            } else if (g.v41) {
                // a token at a time through the token path's attention (its caches and rings advance in order); the
                // row's own top-k selection and candidate mask, kept across the layers (a reader layer reads what the
                // last index source chose for that row, not for the sub-batch's last row)
                const size_t vr = prefill_v41_row();
                for (int t = 0; t < tn; ++t) {
                    // the watchdog's heartbeat every 256 rows too: under memory pressure (a big copy to a network
                    // share) a layer of this loop took 210 s against 5, its launches held up, and the per-layer beat
                    // alone read that as stuck
                    if ((t & 255) == 255) progress_beat();
                    cudaMemcpyAsync(F->x, xs + (size_t) t * E, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, s);
                    gf::quantize_q8_1(F->x, F->xq, E, s);
                    pf_sel_ = (int*) (S->v41_rows + (size_t) (t0 + t) * vr);
                    pf_keep_ = (unsigned char*) (pf_sel_ + g.idx_top_k);
                    const bool ok = fast_dsv41_attn(il, p0 + t0 + t, err);
                    pf_sel_ = nullptr;
                    pf_keep_ = nullptr;
                    if (!ok) return false;
                    cudaMemcpyAsync(mixer + (size_t) t * E, F->mixer, (size_t) E * sizeof(float),
                                    cudaMemcpyDeviceToDevice, s);
                }
            } else if (g.dsv4()) {
                if (!prefill_ds4_mixer(il, p0 + t0, tn, x16, xs, mixer, skip_output, (void*) S->uni, err)) return false;
            } else if (g.qwen35()) {
                // qwen35moe: the token path's layers over the sub-batch's rows (qwen35_fast_path.cu's order), the
                // recurrence and the conv carrying their state across sub-batches, the caches' rows written in order
                Carve c{S->uni};
                Q35Bufs B = carve_q35(c, (size_t) tn, g);
                const int64_t pt0 = p0 + t0;
                const std::string Ls = std::to_string(il);
                if (Ly.recr) {
                    const int C = g.ssm_conv_dim(), V = g.ssm_inner, Sd = g.ssm_state, HK = g.ssm_k_heads, HV = g.ssm_v_heads;
                    hgemm_q(Ly.qkv, C, E, x16, E, B.qkv, C, tn, 0.0f);
                    hgemm_q(Ly.zg, V, E, x16, E, B.z, V, tn, 0.0f);
                    hgemm_q(Ly.alpha_w, HV, E, x16, E, B.alpha, HV, tn, 0.0f);
                    hgemm_q(Ly.beta_w, HV, E, x16, E, B.beta, HV, tn, 0.0f);
                    S->mark("q35_proj", s);
                    q35::conv_rows(B.qkv, tn, C, Ly.conv1d, g.d_conv, Ly.gdn_conv, B.conv, s);
                    q35::gdn_prep_gates(B.conv, tn, C, HK, Sd, g.norm_eps, B.alpha, B.beta, Ly.dt_bias, Ly.ssm_a, HV, s);
                    q35::gdn_scan(Ly.gdn_state, B.conv, tn, C, B.alpha, B.beta, Sd, HK, HV, B.o, s);
                    if (g.q4x)   // (qwen4exp: a sigmoid output gate)
                        strata::kernels::gdn_out_norm(B.o, B.z, Ly.ssm_norm, B.y, (int64_t) tn * HV, Sd, g.norm_eps,
                                                      (void*) s);
                    else
                        strata::kernels::gdn_out_norm_silu(B.o, B.z, Ly.ssm_norm, B.y, (int64_t) tn * HV, Sd, g.norm_eps,
                                                           (void*) s);
                    S->mark("q35_gdn", s);
                    if (t0 + tn == T) dump_row("final_output-" + Ls, B.y, V, tn - 1);
                    if (!skip_output) {
                        gb::f32_to_f16(B.y, B.y16, (int64_t) tn * V, s);
                        hgemm_q(Ly.ssm_o, E, V, B.y16, V, mixer, E, tn, 0.0f);
                    }
                } else {
                    const int D = g.head_dim, HDq = g.n_head * D, KV = g.attn_head_kv * D;
                    hgemm_q(Ly.aq, 2 * HDq, E, x16, E, B.qf, 2 * HDq, tn, 0.0f);
                    hgemm_q(Ly.ak, KV, E, x16, E, B.k, KV, tn, 0.0f);
                    hgemm_q(Ly.av, KV, E, x16, E, B.v, KV, tn, 0.0f);
                    S->mark("q35_proj", s);
                    q35::attn_prep_rows(B.qf, B.k, B.v, Ly.q_norm, Ly.k_norm, g.norm_eps, g.n_head, g.attn_head_kv, D,
                                        g.n_rot, g.rope_base, pt0, tn, Ly.kc, Ly.vc, B.q, s);
                    if (g.q4x) {
                        Q4xBufs Q = carve_q4x(c, (size_t) tn, g);
                        if (!q4_attn(il, pt0, tn, xs, B.q, B.ao, Q, err)) return false;
                    } else {
                        q35::attn_prompt(B.q, Ly.kc, Ly.vc, g.n_head, g.attn_head_kv, D, pt0, tn,
                                         1.0f / std::sqrt((float) D), B.ao, s);
                    }
                    q35::attn_gate_rows(B.ao, B.qf, tn, g.n_head, D, s);
                    S->mark("q35_attn", s);
                    if (t0 + tn == T) dump_row("attn_gated-" + Ls, B.ao, HDq, tn - 1);
                    if (!skip_output) {
                        gb::f32_to_f16(B.ao, B.ao16, (int64_t) tn * HDq, s);
                        hgemm_q(Ly.out, E, HDq, B.ao16, HDq, mixer, E, tn, 0.0f);
                    }
                }
            } else if (Ly.recr) {
                Carve c{S->uni};
                KdaBufs B = carve_kda(c, (size_t) tn, g);
                hgemm_q(Ly.q, DI, E, x16, E, B.proj[0], DI, tn, 0.0f);
                hgemm_q(Ly.k, DI, E, x16, E, B.proj[1], DI, tn, 0.0f);
                hgemm_q(Ly.v, DI, E, x16, E, B.proj[2], DI, tn, 0.0f);
                sgemm_bf16(Ly.f_a, HD, E, xs, E, B.fa, HD, tn, 1.0f);
                sgemm_bf16(Ly.g_a, HD, E, xs, E, B.ga, HD, tn, 1.0f);
                sgemm_bf16(Ly.beta, g.n_head, E, xs, E, B.beta, g.n_head, tn, 1.0f);
                const float* pr[3] = {B.proj[0], B.proj[1], B.proj[2]};
                float* cv[3] = {B.conv[0], B.conv[1], B.conv[2]};
                float* cst = state_ + kda_conv_[(size_t) il];
                gb::kda_conv(pr, Ly.conv, cst, cv, tn, DI, g.d_conv, s);
                gb::kda_conv_state(pr, cst, tn, DI, g.d_conv, s);
                sgemm_bf16(Ly.f_b, DI, HD, B.fa, HD, B.g1, DI, tn, 1.0f);
                sgemm_bf16(Ly.g_b, DI, HD, B.ga, HD, B.g2, DI, tn, 1.0f);
                S->mark("kda_proj", s);
                gb::kda_rec(B.conv[0], B.conv[1], B.conv[2], B.g1, Ly.dt_bias, Ly.ssm_a, g.kda_lb, B.beta,
                            state_ + kda_S_[(size_t) il], B.g2, Ly.ssm_norm, g.norm_eps, g.n_head, tn, B.out16, s);
                S->mark("kda_rec", s);
                if (!skip_output) hgemm_q(Ly.out, E, DI, B.out16, DI, mixer, E, tn, 0.0f);
            } else {
                Carve c{S->uni};
                DsaBufs B = carve_dsa(c, (size_t) tn, g, S->max_pools);
                const int pt = (int) (p0 + t0);
                const float prescale = 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads));
                hgemm_q(Ly.q_a, g.q_lora, E, x16, E, B.qr_raw, g.q_lora, tn, 0.0f);
                hgemm_q(Ly.kv_a, g.kv_lora, E, x16, E, B.kv_raw, g.kv_lora, tn, 0.0f);
                sgemm_bf16(Ly.idx_k, g.idx_key, E, xs, E, B.ik_raw, g.idx_key, tn, 1.0f);
                sgemm_bf16(Ly.idx_gate, g.idx_key, E, xs, E, B.ig_raw, g.idx_key, tn, 1.0f);
                sgemm_bf16(Ly.idx_proj, g.idx_heads, E, xs, E, B.iw, g.idx_heads, tn, prescale);
                gb::DsaPrepArgs d;
                d.qr_raw = B.qr_raw;
                d.q_a_norm = Ly.q_a_norm;
                d.qr = B.qr;
                d.qr16 = B.qr16;
                d.q_lora = g.q_lora;
                d.kv_raw = B.kv_raw;
                d.kv_norm = Ly.kv_a_norm;
                d.lat = (uint16_t*) (state_ + dsa_lat_[(size_t) il]);
                d.lat_q8 = lat_q8_;
                d.kv_lora = g.kv_lora;
                d.ik_raw = B.ik_raw;
                d.k_norm_w = Ly.k_norm_w;
                d.k_norm_b = Ly.k_norm_b;
                d.ik_cache = state_ + dsa_ik_[(size_t) il];
                d.ig_raw = B.ig_raw;
                d.ig_cache = state_ + dsa_ig_[(size_t) il];
                d.idx_key = g.idx_key;
                d.ring = ik_ring_;
                d.p0 = pt;
                d.T = tn;
                d.eps = g.norm_eps;
                gb::dsa_prep(d, s);
                // the pools completed inside this sub-batch: pool pi ends at position (pi + 1) * kpool - 1
                const int kp = g.idx_kpool;
                const int pool_lo = (pt + kp) / kp - 1, pool_hi = (pt + tn) / kp - 1;
                if (pool_hi >= pool_lo)
                    gb::dsa_pool(state_ + dsa_ik_[(size_t) il], state_ + dsa_ig_[(size_t) il], Ly.ape,
                                 state_ + dsa_pool_[(size_t) il], g.idx_key, kp, pool_lo, pool_hi - pool_lo + 1, s,
                                 ik_ring_);
                hgemm_q(Ly.q_b, g.n_head * g.qk_nope, g.q_lora, B.qr16, g.q_lora, B.q, g.n_head * g.qk_nope, tn, 0.0f);
                if (Ly.idx_q_b_x.q != nullptr)   // an EXL3 pack quantizes it
                    hgemm_q(Ly.idx_q_b_x, g.idx_heads * g.idx_key, g.q_lora, B.qr16, g.q_lora, B.iq,
                            g.idx_heads * g.idx_key, tn, 0.0f);
                else
                    sgemm_bf16(Ly.idx_q_b, g.idx_heads * g.idx_key, g.q_lora, B.qr, g.q_lora, B.iq,
                               g.idx_heads * g.idx_key, tn, 1.0f);
                S->mark("dsa_proj", s);
                const int max_vis = (pt + tn) / kp;
                if (max_vis > g.top_pools_max())
                    gb::dsa_score(B.iq, state_ + dsa_pool_[(size_t) il], B.iw, g.idx_key, g.idx_heads, pt, kp, tn,
                                  max_vis, B.score, B.score_ld, s);
                gb::dsa_select(B.score, B.score_ld, pt, kp, g.top_pools_max(), g.idx_select_tail, tn, g.n_sel_max(),
                               B.cells, B.n_sel, s);
                S->mark("dsa_index", s);
                // q_abs[t][h] = wk_b[h] (kv_lora x qk_nope) . q[t][h]
#if defined(STRATA_USE_HIP)
                // FP16 products; q is packed per head into B.attn's space and the context into B.q_abs's (both idle there)
                const bool f16 = mla_f16_on() && (int64_t) g.qk_nope <= 2 * (int64_t) g.v_head;
                const MlaAttnPath attn_path = mla_attn_path();
                const bool use_f16q = f16 && attn_path == MlaAttnPath::F16Q;
                const bool use_wmma2 = attn_path == MlaAttnPath::WMMA2;
                uint16_t* const hp_q = (uint16_t*) B.attn;
                uint16_t* const hp_c = (uint16_t*) B.q_abs;
                if (f16) {
                    bf16_to_f16<<<1024, 256, 0, s>>>(Ly.k_b, S->w16, (int64_t) g.n_head * g.kv_lora * g.qk_nope);
                    pack_heads_f16<<<2048, 256, 0, s>>>(B.q, hp_q, tn, g.n_head, g.qk_nope);
                    blas_ck(hipblasGemmStridedBatchedEx(S->blas, HIPBLAS_OP_T, HIPBLAS_OP_N, g.kv_lora, tn, g.qk_nope, &one,
                                                        S->w16, HIP_R_16F, g.qk_nope, (long long) g.kv_lora * g.qk_nope,
                                                        hp_q, HIP_R_16F, g.qk_nope, (long long) tn * g.qk_nope, &zero,
                                                        B.q_abs, use_f16q ? HIP_R_16F : HIP_R_32F, g.n_head * g.kv_lora,
                                                        g.kv_lora, g.n_head, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
                            "q_abs f16");
                } else
#endif
                {
                gb::bf16_to_f32(Ly.k_b, S->w32, (int64_t) g.n_head * g.kv_lora * g.qk_nope, s);
                blas_ck(cublasSgemmStridedBatched(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, g.kv_lora, tn, g.qk_nope, &one,
                                                  S->w32, g.qk_nope, (long long) g.kv_lora * g.qk_nope, B.q,
                                                  g.n_head * g.qk_nope, g.qk_nope, &zero, B.q_abs, g.n_head * g.kv_lora,
                                                  g.kv_lora, g.n_head),
                        "q_abs");
                }
                S->mark("dsa_qabs", s);
#if defined(STRATA_USE_HIP)
                if (use_f16q)
                    gb::mla_attn_f16q((const uint16_t*) B.q_abs, (const uint16_t*) (state_ + dsa_lat_[(size_t) il]), B.cells,
                                      B.n_sel, g.n_sel_max(), g.n_head, g.kv_lora, 1.0f / std::sqrt((float) g.qk_nope), tn,
                                      B.ctx, s, lat_q8_);
                else if (use_wmma2)
                    gb::mla_attn_wmma2(B.q_abs, (const uint16_t*) (state_ + dsa_lat_[(size_t) il]), B.cells,
                                       B.n_sel, g.n_sel_max(), g.n_head, g.kv_lora, 1.0f / std::sqrt((float) g.qk_nope), tn,
                                       B.ctx, s, lat_q8_);
                else
#endif
                gb::mla_attn(B.q_abs, (const uint16_t*) (state_ + dsa_lat_[(size_t) il]), B.cells, B.n_sel, g.n_sel_max(), g.n_head,
                             g.kv_lora, 1.0f / std::sqrt((float) g.qk_nope), tn, B.ctx, s, lat_q8_);
                S->mark("dsa_attn", s);
                // out[t][h] = wv_b[h] (v_head x kv_lora) . ctx[t][h]
#if defined(STRATA_USE_HIP)
                if (f16) {
                    bf16_to_f16<<<1024, 256, 0, s>>>(Ly.v_b, S->w16, (int64_t) g.n_head * g.v_head * g.kv_lora);
                    pack_heads_f16<<<2048, 256, 0, s>>>(B.ctx, hp_c, tn, g.n_head, g.kv_lora);
                    blas_ck(hipblasGemmStridedBatchedEx(S->blas, HIPBLAS_OP_T, HIPBLAS_OP_N, g.v_head, tn, g.kv_lora, &one,
                                                        S->w16, HIP_R_16F, g.kv_lora, (long long) g.v_head * g.kv_lora,
                                                        hp_c, HIP_R_16F, g.kv_lora, (long long) tn * g.kv_lora, &zero,
                                                        B.attn, HIP_R_32F, g.n_head * g.v_head, g.v_head, g.n_head,
                                                        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
                            "mla out f16");
                } else
#endif
                {
                gb::bf16_to_f32(Ly.v_b, S->w32, (int64_t) g.n_head * g.v_head * g.kv_lora, s);
                blas_ck(cublasSgemmStridedBatched(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, g.v_head, tn, g.kv_lora, &one,
                                                  S->w32, g.kv_lora, (long long) g.v_head * g.kv_lora, B.ctx,
                                                  g.n_head * g.kv_lora, g.kv_lora, &zero, B.attn, g.n_head * g.v_head,
                                                  g.v_head, g.n_head),
                        "mla out");
                }
                gb::f32_to_f16(B.attn, B.attn16, (int64_t) tn * g.n_head * g.v_head, s);
                if (!skip_output)
                    hgemm_q(Ly.out, E, g.n_head * g.v_head, B.attn16, g.n_head * g.v_head, mixer, E, tn, 0.0f);
                if (t0 + tn == T) {
                    const std::string Ls = std::to_string(il);
                    dump_row("dsa_qr-" + Ls, B.qr, g.q_lora, tn - 1);
                    dump_row("dsa_q-" + Ls, B.q, g.n_head * g.qk_nope, tn - 1);
                    dump_row("dsa_iq-" + Ls, B.iq, g.idx_heads * g.idx_key, tn - 1);
#if defined(STRATA_USE_HIP)
                    if (!f16)   // with the FP16 products q_abs holds FP16 / the packed context
#endif
                    dump_row("pf_qabs-" + Ls, B.q_abs, g.n_head * g.kv_lora, tn - 1);
                    dump_row("pf_ctx-" + Ls, B.ctx, g.n_head * g.kv_lora, tn - 1);
                    dump_row("pf_attn-" + Ls, B.attn, g.n_head * g.v_head, tn - 1);
                }
            }
        }

        S->mark(Ly.recr ? "kda" : "dsa", s);
        if (skip_output) continue;   // every KDA/DSA cache update above still ran; these output rows feed nothing
        abl_out(il, 0, S->mixer, T, E, s);
        dump_row("mixer-" + std::to_string(il), S->mixer, E);
        if (g.v41)
            hc_read(S->mixer, Ly.hc_ffn_fn, Ly.hc_ffn_scale, Ly.hc_ffn_base, Ly.ffn_norm, S->pre, S->pre2);
        else if (g.q4x)
            q4_mix(il, 1, S->mixer, S->pre, S->post, false);
        else if (g.qwen35())
            q35::add_rms_rows(S->R, 4 * E, S->mixer, Ly.post_norm, g.norm_eps, T, E, S->x, S->x16, s);
        else
            hc_read(S->mixer, Ly.hc_ffn_fn, Ly.hc_ffn_scale, Ly.hc_ffn_base, Ly.ffn_norm);
        S->mark("hc", s);
        dump_row("ffn_norm-" + std::to_string(il), S->x, E);
        // the CPU lane's x (this layer's FFN input) goes down now, under the shared expert and the routing - not
        // after the plan, where the host's experts waited for it (~40 ms of a 16K chunk's 260 MB)
        // (not when the iGPU takes them: after the plan only their rows go down, gathered - a 32K chunk's whole x is
        // 537 MB, ~80 ms over PCIe Gen4 x4, against ~70 MB for a layer's ~4K host rows)
        bool x_down = false;
        if (Ly.moe && cpu_lane(il) && !(igpu_lane(il) && S->id_h != nullptr)) {
            if (!cpu_bufs(0)) {
                err = "glm prefill: the CPU experts' host buffers did not allocate";
                return false;
            }
            cudaEventRecord(S->ev_x, s);
            cudaStreamWaitEvent(S->xs, S->ev_x, 0);
            cudaMemcpyAsync(S->x_h, S->x, (size_t) T * E * sizeof(float), cudaMemcpyDeviceToHost, S->xs);
            x_down = true;
        }

        if (!Ly.moe) {
            for (int t0 = 0; t0 < T; t0 += S->sub) {
                const int tn = std::min(S->sub, T - t0);
                Carve c{S->uni};
                DenseBufs B = carve_dense(c, (size_t) tn, g);
                const uint16_t* x16 = S->x16 + (size_t) t0 * E;
                hgemm_q(Ly.ffn_gate, g.n_ff_dense, E, x16, E, B.dg, g.n_ff_dense, tn, 0.0f);
                hgemm_q(Ly.ffn_up, g.n_ff_dense, E, x16, E, B.du, g.n_ff_dense, tn, 0.0f);
                gb::swiglu_f16(B.dg, B.du, B.dh16, (int64_t) tn * g.n_ff_dense, g.swiglu_shexp, s);
                hgemm_q(Ly.ffn_down, E, g.n_ff_dense, B.dh16, g.n_ff_dense, S->ffn + (size_t) t0 * E, E, tn, 0.0f);
            }
            S->mark("dense_ffn", s);
            abl_out(il, 1, S->ffn, T, E, s);
            dump_row("ffn_out-" + std::to_string(il), S->ffn, E);
            continue;
        }

        // ---- MoE: the shared expert (sub-batches, straight into ffn), the routes, the expert groups
        Carve c{S->uni};
        MoeBufs M = carve_moe(c, (size_t) T, (size_t) S->sh_sub(), g);
        const int FF = g.n_ff_exp * g.n_shared, NE = g.n_expert, nff = g.n_ff_exp;
        // the shared expert: before the route (as before), or after the route's sync so its GEMMs run while the
        // host plans the expert sets (the stream was idle through the plan).  STRATA_GLM_PREFILL_SH_LATE=0: before
        static const bool sh_late = [] { const char* v = getenv("STRATA_GLM_PREFILL_SH_LATE"); return v == nullptr || std::atoi(v) != 0; }();
        const auto shared_expert = [&] {
            for (int t0 = 0, shs = S->sh_sub(); t0 < T; t0 += shs) {
                const int tn = std::min(shs, T - t0);
                const uint16_t* x16 = S->x16 + (size_t) t0 * E;
                hgemm_q(Ly.sh_gate, FF, E, x16, E, M.sh_g, FF, tn, 0.0f);
                hgemm_q(Ly.sh_up, FF, E, x16, E, M.sh_u, FF, tn, 0.0f);
                gb::swiglu_f16(M.sh_g, M.sh_u, M.sh16, (int64_t) tn * FF, g.swiglu_shexp, s);
                hgemm_q(Ly.sh_down, E, FF, M.sh16, FF, S->ffn + (size_t) t0 * E, E, tn, 0.0f);
            }
            abl_apply(il, 1, S->ffn, T, E, s);   // (S->ffn holds the shared expert alone here: the routed add later)
            if (g.qwen35() && Ly.sh_gate_w != nullptr)   // qwen35moe: the shared expert behind its sigmoid gate
                q35::shexp_gate_rows(S->x, Ly.sh_gate_w, S->ffn, T, E, s);
            S->mark("shexp", s);
        };
        if (!sh_late) shared_expert();
        if (Ly.router_type == gf::kTypeF32) {   // (deepseek41, qwen35moe: an F32 router)
#if defined(STRATA_USE_HIP)
            // the engine's own tile kernel: rocBLAS's F32 GEMM took 15.7 ms a layer here (575 rows, 256 experts, 3072),
            // 751 of a 575-token prompt's 2255 ms (an AI PRO R9700), against 0.2 ms for the same shape alone
            gb::router_f32((const float*) Ly.router, NE, S->x, T, E, M.logits, s);
#else
            blas_ck(cublasSgemm(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, NE, T, E, &one, (const float*) Ly.router, E, S->x, E,
                                &zero, M.logits, NE),
                    "router sgemm");
#endif
        } else {
            sgemm_bf16(Ly.router, NE, E, S->x, E, M.logits, NE, T, 1.0f);
        }
        S->mark("router", s);
        gb::route(M.logits, Ly.router_bias, NE, K, g.w_scale, g.norm_w != 0, T, M.ids, M.rw, s, g.gating,
                  Ly.tid2eid, S->toks_d, Ly.router_bias_vl, S->img_on ? S->img_d : nullptr);
        gb::expert_count(M.ids, T * K, NE, M.counts, M.rank, s);
        gb::copy_i32(M.counts, S->h_counts, NE, s);   // (not a copy: the engine may be busy with x, x_down)
        S->mark("route", s);
        if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "glm prompt route/count sync", err)) return false;
        if (sh_late) shared_expert();   // (into ffn, ahead of the expert sets' combines on the same stream)
        int counted = 0;
        for (int e = 0; e < NE; ++e) counted += S->h_counts[e];
        if (counted != T * K) {
            err = "glm router: prompt layer " + std::to_string(il) + " invalid expert ID or non-finite score (" +
                  std::to_string(counted) + " of " + std::to_string(T * K) + " routes)";
            return false;
        }
        const auto tp = std::chrono::steady_clock::now();
        // (the split's balance measures both lanes from here: the compute stream is idle, the event is the plan's start)
        if (S->ev_c0 == nullptr) {
            cudaEventCreate(&S->ev_c0);
            cudaEventCreate(&S->ev_c1);
        }
        cudaEventRecord(S->ev_c0, s);
        if (S->trace) {
            auto& t = S->tr[(size_t) il];
            cudaEventRecord(t.plan, s);
            t.on = true;
            t.pre_left = 0;
            if (S->pre_layer == il) {
                const int np = (int) S->pre_e.size();
                t.pre_left = np;
                for (size_t k = 0; k * PrefillState::kPreEv < (size_t) np; ++k) {
                    if (cudaEventQuery(S->ev_pdone[k]) != cudaSuccess) break;
                    t.pre_left = std::max(0, np - (int) (k + 1) * PrefillState::kPreEv);
                }
            } else {
                t.pre_n = 0;
            }
        }

        // ---- the plan (host): resident experts in slot order, then the staged ones in groups
        auto& P = F->lp[(size_t) il];
        auto& RC = F->rc[(size_t) F->layer_rc[(size_t) il]];
        struct Staged {
            int e;
            const uint8_t* ram;   // the RAM tier slot (or, dev, its VRAM slot), or null: disk
            bool dev;
        };
        std::vector<Staged> staged;
        int nb = 0;   // h_bounds fill
        int rows_res = 0, max_res = 0;
        const int b_res = nb;
        // resident experts with at most light_rows rows go to the light kernels (gf::rows_experts): their rows come
        // after the MMQ product's, which sees them as empty
        static const int light_env = [] {
            const char* v = getenv("STRATA_GLM_PREFILL_LIGHT");
            return v ? std::atoi(v) : kLightRows;
        }();
        const auto lt_ok = [](int t) { return t == 16 || t == 18 || t == 19; };
        static const int light_layer = getenv("STRATA_GLM_PREFILL_LIGHT_LAYER") ? std::atoi(getenv("STRATA_GLM_PREFILL_LIGHT_LAYER")) : -1;
        const int light_rows = lt_ok(Ly.gu_type) && lt_ok(Ly.d_type) && (light_layer < 0 || il == light_layer) ? light_env : 0;
        int n_light = 0, rows_mmq = 0;
        int b_pre = -1, rows_pre = 0, max_pre = 0;   // the prestaged set: its bounds, rows and largest expert
        const int pre_n = S->pre_layer == il ? (int) S->pre_e.size() : 0;
        int* h_light = S->h_bounds + kLightOff;
        {
            std::lock_guard<std::mutex> lk(F->mu);
            S->h_bounds[nb++] = 0;
            for (int sl = 0; sl < P.n_main; ++sl) {
                int cnt = 0;
                if (P.st[(size_t) sl] == FastState::kResident && P.key[(size_t) sl] >= 0) {
                    const int e = P.key[(size_t) sl];
                    cnt = S->h_counts[e];
                    if (cnt > 0 && cnt <= light_rows && n_light < kLightMax) {
                        S->h_counts[e] = -cnt;   // claimed (rows assigned below)
                        h_light[3 * n_light] = sl;
                        h_light[3 * n_light + 1] = e;   // (the expert until its rows are known)
                        h_light[3 * n_light + 2] = cnt;
                        ++n_light;
                        cnt = 0;
                    } else if (cnt > 0) {
                        S->h_base[e] = rows_res;
                        S->h_counts[e] = -cnt;   // claimed
                    }
                }
                rows_res += std::max(0, cnt);
                max_res = std::max(max_res, cnt);
                S->h_bounds[nb++] = rows_res;
            }
            rows_mmq = rows_res;
            for (int i = 0; i < n_light; ++i) {
                const int e = h_light[3 * i + 1];
                S->h_base[e] = rows_res;
                h_light[3 * i + 1] = rows_res;
                rows_res += h_light[3 * i + 2];
            }
            // the prestaged experts this routing hit: their rows after the resident ones, computed from pbuf
            if (S->pre_layer == il) {
                b_pre = nb;
                S->h_bounds[nb++] = 0;
                for (int e : S->pre_e) {
                    const int cnt = std::max(0, S->h_counts[e]);
                    if (cnt > 0) {
                        S->h_base[e] = rows_res + rows_pre;
                        S->h_counts[e] = -cnt;   // claimed
                        rows_pre += cnt;
                        max_pre = std::max(max_pre, cnt);
                        ++S->pre_hit;
                    }
                    S->h_bounds[nb++] = rows_pre;
                }
            }
            for (int e = 0; e < NE; ++e) {
                if (S->h_counts[e] <= 0) continue;
                const int key = il * NE + e;
                const int rs = F->ram_of[(size_t) key];
                const int vs = F->slot_of[(size_t) key];
                const uint8_t* src = nullptr;
                // still in VRAM: a tail slot past what this prompt borrows (a short prompt borrows a little of
                // the tail) - a device copy, not the disk read it used to be
                const bool dev = vs >= P.n_main && vs < P.n && P.st[(size_t) vs] == FastState::kResident &&
                                 P.key[(size_t) vs] == e;
                if (dev) src = P.slot_ptr(vs);
                else if (rs >= 0 && RC.st[(size_t) rs] == FastState::kRHold) src = RC.base + (size_t) rs * RC.stride;
                staged.push_back(Staged{e, src, dev});
            }
            // the prompt's routing feeds the LFU counts the decode's tiers evict by
            uint64_t ev = 0;
            for (int e = 0; e < NE; ++e) {
                const int cnt = std::abs(S->h_counts[e]);
                if (cnt == 0) continue;
                uint32_t& cc = F->cnt[(size_t) il * NE + e];
                cc = (uint32_t) std::min<uint64_t>((uint64_t) cc + (uint64_t) cnt, 1u << 24);
                ev += (uint64_t) cnt;
                uint32_t& uu = F->usage[(size_t) il * NE + e];
                uu = (uint32_t) std::min<uint64_t>((uint64_t) uu + (uint64_t) cnt, 1u << 30);
                if (uu >= (1u << 30))
                    for (auto& u : F->usage) u >>= 1;
            }
            F->cnt_events += ev;
            while (F->cnt_events >= 32768) {
                F->cnt_events -= 32768;
                for (auto& cc : F->cnt) cc >>= 1;
            }
            auto& lc = S->last_cnt[(size_t) il];
            lc.resize((size_t) NE);
            for (int e = 0; e < NE; ++e) lc[(size_t) e] = std::abs(S->h_counts[e]);
        }
        // RAM-tier experts first (DMA), disk ones last (their reads overlap the earlier groups)
        std::stable_partition(staged.begin(), staged.end(), [](const Staged& x) { return x.ram != nullptr; });
        // THE CPU'S SHARE (a host whose CPU lane computes an expert faster than PCIe moves it - one GPU beside two
        // sockets: 0.15 vs 2.1 ms): the least routed RAM-tier experts are computed on the host, from the RAM tier, while
        // the copy stream stages the rest - taken in ascending row count while the host's estimated time (an expert's
        // weights + STRATA_GLM_PREFILL_CPU_ROW_MS a row, times the learned balance kappa) stays under the PCIe time of
        // what is left.  Their rows go last in the sorted order; their outputs are uploaded into OUTP before the
        // combine.  STRATA_GLM_PREFILL_CPU=0: off.
        std::vector<Staged> cpu_set;
        {
            static const char* row_env = getenv("STRATA_GLM_PREFILL_CPU_ROW_MS");
            // the planning cost of a row: the env's, else learned from the layers before (their host time less an
            // expert's weights each, over their rows; one 3090 beside 2x Xeon 6152: ~0.056 ms)
            const bool ig = igpu_lane(il);
            const double row_ms = row_env != nullptr ? std::atof(row_env) : ig ? S->row_ms_ig : S->row_ms;
            const double fix_ms = ig ? S->fix_ms_ig : 2.0 * F->cpu_c_ms;
            if (cpu_lane(il)) {
                std::vector<size_t> ram_i;
                int n_pcie = 0;
                for (size_t i = 0; i < staged.size(); ++i) {
                    if (staged[i].dev) continue;
                    ++n_pcie;
                    if (staged[i].ram != nullptr) ram_i.push_back(i);
                }
                std::sort(ram_i.begin(), ram_i.end(), [&](size_t a, size_t b) {
                    return S->h_counts[staged[a].e] < S->h_counts[staged[b].e];
                });
                // (the streaming rate: the prestage copies', else the calibration's)
                const double p_ms = S->copy_ms > 0.0 ? S->copy_ms : F->cpu_ps_ms > 0.0 ? F->cpu_ps_ms : F->cpu_p_ms;
                // the prestage copies the copy stream still owes (a mixer shorter than the buffer's copies): the
                // staged groups queue behind them
                int pre_left = 0;
                if (S->pre_layer == il) {
                    const int np = (int) S->pre_e.size();
                    pre_left = np;
                    for (size_t k = 0; k * PrefillState::kPreEv < (size_t) np; ++k) {
                        if (cudaEventQuery(S->ev_pdone[k]) != cudaSuccess) break;
                        pre_left = std::max(0, np - (int) (k + 1) * PrefillState::kPreEv);
                    }
                }
                double t_cpu = 0.0, t_pcie = (double) (n_pcie + pre_left) * p_ms;
                std::vector<char> take(staged.size(), 0);
                const int cpu_cap = moe_window_rows((size_t) T, g);   // (the device rows their outputs go up into)
                int cpu_rows_taken = 0;
                for (size_t i : ram_i) {
                    // (an expert's fixed cost is about twice the lane's one-token expert: its weights, and the
                    // multi-row kernel's unpacking of every row)
                    const double c = S->kappa * (fix_ms + row_ms * S->h_counts[staged[i].e]);
                    if (t_cpu + c > t_pcie - p_ms) break;
                    if (cpu_rows_taken + S->h_counts[staged[i].e] > cpu_cap) break;
                    cpu_rows_taken += S->h_counts[staged[i].e];
                    take[i] = 1;
                    t_cpu += c;
                    t_pcie -= p_ms;
                }
                std::vector<Staged> rest;
                for (size_t i = 0; i < staged.size(); ++i) (take[i] ? cpu_set : rest).push_back(staged[i]);
                staged.swap(rest);
            }
        }
        int rows = rows_res + rows_pre;
        struct Group {
            int first, n, r0, nrows, max_rows, b_off;
        };
        std::vector<Group> groups;
        for (size_t i = 0; i < staged.size(); i += PrefillState::GE) {
            Group gr{(int) i, (int) std::min<size_t>(PrefillState::GE, staged.size() - i), rows, 0, 0, nb};
            S->h_bounds[nb++] = 0;
            for (int j = 0; j < gr.n; ++j) {
                const int e = staged[(size_t) gr.first + j].e;
                const int cnt = S->h_counts[e];
                S->h_base[e] = rows;
                rows += cnt;
                gr.nrows += cnt;
                gr.max_rows = std::max(gr.max_rows, cnt);
                S->h_bounds[nb++] = gr.nrows;
            }
            groups.push_back(gr);
        }
        const int cpu_r0 = rows;
        for (const auto& cx : cpu_set) {
            S->h_base[cx.e] = rows;
            rows += S->h_counts[cx.e];
        }
        const int cpu_nrows = rows - cpu_r0;
        if (rows != T * K) {
            err = "glm prefill: layer " + std::to_string(il) + " planned " + std::to_string(rows) + " of " +
                  std::to_string(T * K) + " routed rows";
            return false;
        }
        if (nb > kLightOff) {
            err = "glm prefill: the expert bounds overflowed";
            return false;
        }
        cudaMemcpyAsync(M.base, S->h_base, (size_t) NE * sizeof(int), cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(M.bounds, S->h_bounds, (size_t) nb * sizeof(int), cudaMemcpyHostToDevice, s);
        if (n_light > 0)
            cudaMemcpyAsync(M.bounds + kLightOff, h_light, (size_t) 3 * n_light * sizeof(int), cudaMemcpyHostToDevice, s);
        gb::expert_scatter(M.ids, M.rank, M.base, T * K, K, M.row_tok, M.pos, s);
        if (cpu_nrows > 0) {
            if (!cpu_bufs(cpu_nrows)) {
                err = "glm prefill: the CPU experts' host buffers did not allocate";
                return false;
            }
            cudaEventRecord(S->ev_x, s);   // the scatter (and x) are done up to here
            cudaStreamWaitEvent(S->xs, S->ev_x, 0);
            S->x_gathered = !x_down && igpu_lane(il) && S->id_h != nullptr;
            if (S->x_gathered) {   // (OUTPc: the host rows' outputs come back into it later, after the combine read it)
                gather_rows<<<(unsigned) cpu_nrows, 256, 0, s>>>(S->x, M.row_tok + cpu_r0, cpu_nrows, E, M.OUTPc);
                cudaEventRecord(S->ev_x, s);
                cudaStreamWaitEvent(S->xs, S->ev_x, 0);
                cudaMemcpyAsync(S->out_h, M.OUTPc, (size_t) cpu_nrows * E * sizeof(float), cudaMemcpyDeviceToHost, S->xs);
            } else if (!x_down) {
                cudaMemcpyAsync(S->x_h, S->x, (size_t) T * E * sizeof(float), cudaMemcpyDeviceToHost, S->xs);
            }
            cudaMemcpyAsync(S->rt_h, M.row_tok + cpu_r0, (size_t) cpu_nrows * sizeof(int), cudaMemcpyDeviceToHost, S->xs);
            cudaEventRecord(S->ev_xd, S->xs);   // (what the host lane waits for: xs's copies, ev_xd)
        }
        S->mark("plan", s);
        S->ms_plan += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp).count();

        // ---- the OUTPUT WINDOW: OUTP row 0 is sorted row wb; rows [w_lo, w_hi) are computed and not yet added into ffn.
        //      A set's rows go in contiguous with what it holds, else the window is added into ffn first (moe_combine_add)
        //      and starts at the set; a set larger than the window runs in parts at expert boundaries
        const int W = moe_window_rows((size_t) T, g);
        int wb = 0, w_lo = 0, w_hi = 0;
        const auto flush = [&]() {
            if (w_hi > w_lo) gb::moe_combine_add(M.OUTP, wb, w_lo, w_hi, M.pos, M.rw, T, K, E, S->ffn, s);
            w_lo = w_hi = 0;
        };
        const auto place = [&](int a, int n) {   // -> the OUTP row of sorted row a
            if (!(w_hi > w_lo && a == w_hi && a + n <= wb + W)) {
                flush();
                wb = a;
                w_lo = a;
            }
            w_hi = a + n;
            return a - wb;
        };
        const size_t qblk = mmq::q8_block_bytes();
        // the SwiGLU in the gate/up product's write-back where MMQ has it: the product
        // writes T x n_ff values instead of T x 2 n_ff that a separate kernel reads back, bit-identical to it.
        // STRATA_GLM_MMQ_GLU=0: the separate kernel.
        static const bool glu_on = [] {
            const char* v = getenv("STRATA_GLM_MMQ_GLU");
            return v == nullptr || std::atoi(v) != 0;
        }();
        const bool glu = glu_on && mmq::glu_supported(Ly.gu_type, nff);
        // one expert set through gate/up, swiglu and down: its rows [r0 + hb[0], r0 + hb[n_exp]) of the sorted order
        // (hb: the set's bounds on the host, d_bounds the same on the device), in parts of at most W rows
        // (the down matrices' over-read past their last row, see zero_tails; 0: n_ff fills MMQ's tiles)
        const int64_t dn_qk = ggml_blck_size((ggml_type) Ly.d_type);
        const size_t dn_end = Ly.down_off + (size_t) E * (size_t) (nff / dn_qk) * ggml_type_size((ggml_type) Ly.d_type);
        const int dn_over = nff % 512 == 0 ? 0
                            : (int) ((512 - nff % 512) / dn_qk * (int64_t) ggml_type_size((ggml_type) Ly.d_type));
        const auto run_set = [&](const uint8_t* wbase, int n_exp, size_t stride, const int* d_bounds, const int* hb,
                                 int r0) {
            for (int i0 = 0; i0 < n_exp;) {
                int i1 = i0 + 1;   // (one expert never has more rows than T <= W)
                while (i1 < n_exp && hb[i1 + 1] - hb[i0] <= W) ++i1;
                const int a = hb[i0], nrows = hb[i1] - a;
                int max_rows = 0;
                for (int i = i0; i < i1; ++i) max_rows = std::max(max_rows, hb[i + 1] - hb[i]);
                if (nrows > 0) {
                    const int o = place(r0 + a, nrows);
                    mmq::quantize(S->x, M.row_tok + r0 + a, M.Xq, Ly.gu_type, E, E, nrows, s);
                    // the part's bounds are the set's (from a): the activations and the destination are addressed a
                    // rows back so that bound a is the part's first row
                    float* GU = M.OUTP + (size_t) o * E;   // 2 * n_ff <= n_embd: the set's own OUTP rows hold its gate/up
                    mmq::Product gu;
                    gu.w = wbase + (size_t) i0 * stride;
                    gu.type = Ly.gu_type;
                    gu.w_rows = 2 * nff;
                    gu.w_cols = E;
                    gu.expert_bytes = stride;
                    gu.n = i1 - i0;
                    gu.xq = (const uint8_t*) M.Xq - (size_t) a * qblk;
                    gu.bounds = d_bounds + i0;
                    gu.ids = S->iota;
                    gu.total_rows = nrows;
                    gu.max_rows = max_rows;
                    gu.dst = GU - (size_t) a * (2 * nff);
                    gu.ld_dst = 2 * nff;
                    if (glu) {
                        S->mq->run_glu(gu, g.swiglu_exp, s);   // (h in each row's first n_ff floats, as below)
                    } else {
                        S->mq->run(gu, s);
                        gb::swiglu_rows(GU, GU, nrows, nff, g.swiglu_exp, s, 2 * nff);   // (in place: no T x K x n_ff buffer)
                    }
                    mmq::quantize(GU, nullptr, M.Hq, Ly.d_type, nff, 2 * nff, nrows, s);
                    if (dn_over > 0 && dn_end + (size_t) dn_over > stride) {
                        static bool said = false;
                        if (!said) std::fprintf(stderr, "glm prefill: a down matrix's MMQ over-read passes its slot (stride %zu)\n", stride);
                        said = true;
                    } else if (dn_over > 0)
                        zero_tails<<<i1 - i0, 128, 0, s>>>(const_cast<uint8_t*>(wbase) + (size_t) i0 * stride, stride,
                                                           dn_end, dn_over);
                    mmq::Product dn;
                    dn.w = wbase + (size_t) i0 * stride + Ly.down_off;
                    dn.type = Ly.d_type;
                    dn.w_rows = E;
                    dn.w_cols = nff;
                    dn.expert_bytes = stride;
                    dn.n = i1 - i0;
                    dn.xq = (const uint8_t*) M.Hq - (size_t) a * qblk;
                    dn.bounds = d_bounds + i0;
                    dn.ids = S->iota;
                    dn.total_rows = nrows;
                    dn.max_rows = max_rows;
                    dn.dst = M.OUTP + (size_t) (o - a) * E;
                    dn.ld_dst = E;
                    S->mq->run(dn, s);
                }
                i0 = i1;
            }
        };
        // EXL3 experts (MMQ does not read their trellis), each by its row count: a lightly routed one through the
        // tensor-core kernels (sm_80+; they decode the trellis once a 32-row block - cheaper than reconstructing 50 MB
        // of FP16 for a few rows), the rest reconstructed to FP16 and multiplied by cuBLAS (the decode once, the GEMM
        // at the tensor cores' FP16 rate).  Both write the window's OUTP rows as run_set's; w16 is their scratch.
        // STRATA_GLM_EXL3_MMA_ROWS=<rows>: the largest expert the tensor-core kernels take (0: none)
        static const int mma_rows = [] {
            const char* v = getenv("STRATA_GLM_EXL3_MMA_ROWS");
            return v ? std::max(0, std::atoi(v)) : 64;
        }();
        const bool use_mma = mma_rows > 0 && strata::kernels::exl3::mma_supported();
        // experts [i0, i1) of the set through the tensor-core kernels, in parts whose rows fit the window and w16's h
        const auto run_mma = [&](const uint8_t* wbase, size_t stride, const int* d_bounds, const int* hb, int r0, int i0,
                                 int i1) {
            namespace x3 = strata::kernels::exl3;
            const int cap = std::min(W, (int) std::max<int64_t>(1, S->w16_elems / nff));
            while (i0 < i1) {
                int j = i0 + 1;
                while (j < i1 && hb[j + 1] - hb[i0] <= cap) ++j;
                const int a = hb[i0], nrows = hb[j] - a;
                int max_rows = 0;
                for (int i = i0; i < j; ++i) max_rows = std::max(max_rows, hb[i + 1] - hb[i]);
                if (nrows > 0) {
                    x3::SetArgs sa;
                    sa.wbase = wbase + (size_t) i0 * stride;
                    sa.stride = stride;
                    sa.lay = Ly.xl;
                    sa.n_exp = j - i0;
                    sa.bounds = d_bounds + i0;
                    sa.row0 = a;
                    sa.max_rows = max_rows;
                    sa.n_embd = E;
                    sa.n_ff = nff;
                    sa.limit = g.swiglu_exp;
                    const int o = place(r0 + a, nrows);
                    x3::set_gate_up(sa, S->x, E, M.row_tok + r0 + a, S->w16, nff, s);
                    x3::set_down(sa, S->w16, nff, M.OUTP + (size_t) o * E, E, s);
                }
                i0 = j;
            }
        };
        // one expert through reconstruct + cuBLAS.  By default the expert's rows are rotated instead of its weights
        // (x W = H(H(x * suh) W') * svh): the decoded W' alone goes to FP16 - a plain decode, not two 128-point
        // transforms of every weight - and the inputs and outputs of its few hundred rows take the rotations, as in
        // decode.  STRATA_GLM_EXL3_ROT=0: the weights rotated (reconstruct_rows), as before
        static const bool rot_rows = [] {
            const char* v = getenv("STRATA_GLM_EXL3_ROT");
            return v == nullptr || std::atoi(v) != 0;
        }();
        const auto run_blas = [&](const uint8_t* blob, int a, int nrows, int r0) {
            namespace x3 = strata::kernels::exl3;
            const int64_t wgu = (int64_t) 2 * nff * E, wdn = (int64_t) E * nff;
            const int gu_rows = (int) std::max<int64_t>(1, (S->w16_elems - wgu) / (rot_rows ? 2 * E : E));
            const int dn_rows = (int) std::max<int64_t>(1, (S->w16_elems - wdn) / nff);
            const auto mat = [&](int r) {
                x3::Mat m;
                m.trellis = (const uint32_t*) (blob + Ly.xl.trellis[r]);
                m.suh = (const uint16_t*) (blob + Ly.xl.suh[r]);
                m.svh = (const uint16_t*) (blob + Ly.xl.svh[r]);
                m.k = r < 2 ? E : nff;
                m.n = r < 2 ? nff : E;
                m.K = Ly.xl.K[r];
                m.cb = Ly.xl.cb;
                return m;
            };
            const int o = place(r0 + a, nrows);
            float* GU = M.OUTP + (size_t) o * E;   // 2 * n_ff == n_embd: the expert's own OUTP rows hold its gate/up
            if (rot_rows) {
                // gate and up: W' of each, the rows rotated by each one's suh, the outputs by its svh
                for (int r = 0; r < 2; ++r) x3::reconstruct_inner_rows(mat(r), S->w16 + (size_t) r * nff * E, s);
                uint16_t* X16 = S->w16 + wgu;
                for (int c0 = 0; c0 < nrows; c0 += gu_rows) {
                    const int n = std::min(gu_rows, nrows - c0);
                    for (int r = 0; r < 2; ++r) {
                        uint16_t* Xr = X16 + (size_t) r * gu_rows * E;
                        x3::rows_rot_f16(S->x, E, M.row_tok + r0 + a + c0, n, E, mat(r).suh, Xr, s);
                        blas_ck(cublasGemmEx(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, nff, n, E, &one,
                                             S->w16 + (size_t) r * nff * E, CUDA_R_16F, E, Xr, CUDA_R_16F, E, &zero,
                                             GU + (size_t) c0 * E + (size_t) r * nff, CUDA_R_32F, E,
                                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                                "gemm exl3 gate/up");
                    }
                }
                for (int r = 0; r < 2; ++r) x3::rows_rot_out(GU + (size_t) r * nff, E, nrows, nff, mat(r).svh, s);
                gb::swiglu_rows(GU, GU, nrows, nff, g.swiglu_exp, s, 2 * nff);
                x3::reconstruct_inner_rows(mat(2), S->w16, s);
                uint16_t* H16 = S->w16 + wdn;
                for (int c0 = 0; c0 < nrows; c0 += dn_rows) {
                    const int n = std::min(dn_rows, nrows - c0);
                    x3::rows_rot_f16(GU + (size_t) c0 * E, E, nullptr, n, nff, mat(2).suh, H16, s);
                    blas_ck(cublasGemmEx(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, E, n, nff, &one, S->w16, CUDA_R_16F, nff,
                                         H16, CUDA_R_16F, nff, &zero, GU + (size_t) c0 * E, CUDA_R_32F, E,
                                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                            "gemm exl3 down");
                }
                x3::rows_rot_out(GU, E, nrows, E, mat(2).svh, s);
                return;
            }
            x3::reconstruct_rows(mat(0), S->w16, false, s);
            x3::reconstruct_rows(mat(1), S->w16 + (size_t) nff * E, false, s);
            uint16_t* X16 = S->w16 + wgu;
            for (int c0 = 0; c0 < nrows; c0 += gu_rows) {
                const int n = std::min(gu_rows, nrows - c0);
                x3::rows_f16(S->x, E, M.row_tok + r0 + a + c0, n, E, X16, s);
                blas_ck(cublasGemmEx(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * nff, n, E, &one, S->w16, CUDA_R_16F, E,
                                     X16, CUDA_R_16F, E, &zero, GU + (size_t) c0 * E, CUDA_R_32F, E, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                        "gemm exl3 gate/up");
            }
            gb::swiglu_rows(GU, GU, nrows, nff, g.swiglu_exp, s, 2 * nff);
            x3::reconstruct_rows(mat(2), S->w16, false, s);
            uint16_t* H16 = S->w16 + wdn;
            for (int c0 = 0; c0 < nrows; c0 += dn_rows) {
                const int n = std::min(dn_rows, nrows - c0);
                x3::rows_f16(GU + (size_t) c0 * E, E, nullptr, n, nff, H16, s);
                blas_ck(cublasGemmEx(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, E, n, nff, &one, S->w16, CUDA_R_16F, nff, H16,
                                     CUDA_R_16F, nff, &zero, GU + (size_t) c0 * E, CUDA_R_32F, E, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                        "gemm exl3 down");
            }
        };
        // in set order (the window takes the rows in order): runs of light experts to the tensor cores, the rest alone
        const auto run_set_x3 = [&](const uint8_t* wbase, int n_exp, size_t stride, const int* d_bounds, const int* hb,
                                    int r0) {
            for (int i = 0; i < n_exp;) {
                const int rows = hb[i + 1] - hb[i];
                if (use_mma && rows <= mma_rows) {
                    int j = i + 1;
                    while (j < n_exp && hb[j + 1] - hb[j] <= mma_rows) ++j;
                    run_mma(wbase, stride, d_bounds, hb, r0, i, j);
                    i = j;
                } else {
                    if (rows > 0) run_blas(wbase + (size_t) i * stride, hb[i], rows, r0);
                    ++i;
                }
            }
        };
        // experts MMQ does not read (deq_expert_type): one at a time, gate and up decoded to FP16 into w16 [2 n_ff][E]
        // (the blob's gate rows, then its up rows), the expert's rows gathered to FP16 and multiplied by cuBLAS, the
        // SwiGLU, then the down [E][n_ff] the same way - the outputs land in the expert's own OUTP rows as run_blas's
        const bool deq = !Ly.exl3 && !mmq_layer(Ly.gu_type, Ly.d_type);
        const auto run_blas_dq = [&](const uint8_t* blob, int a, int nrows, int r0) {
            const int64_t wgu = (int64_t) 2 * nff * E, wdn = (int64_t) E * nff;
            const int gu_rows = (int) std::min<int64_t>(kDqRows, std::max<int64_t>(1, (S->w16_elems - wgu) / E));
            const int dn_rows = (int) std::max<int64_t>(1, (S->w16_elems - wdn) / nff);
            const size_t up_off = mmq::matrix_bytes(Ly.gu_type, nff, E);
            const int o = place(r0 + a, nrows);
            // GU rows of 2 n_ff (gate | up) as MMQ's (2 * n_ff <= n_embd: inside the expert's own OUTP rows); the
            // down's E-wide outputs then overwrite them from the expert's first row on, so its parts run last first
            // (part c0's outputs reach only the inputs of rows past it, already multiplied)
            float* GU = M.OUTP + (size_t) o * E;
            strata::kernels::dequant_f16(Ly.gu_type, blob, 0, nff, E, S->w16, s);
            strata::kernels::dequant_f16(Ly.gu_type, blob + up_off, 0, nff, E, S->w16 + (size_t) nff * E, s);
            uint16_t* X16 = S->w16 + wgu;
            for (int c0 = 0; c0 < nrows; c0 += gu_rows) {
                const int n = std::min(gu_rows, nrows - c0);
                strata::kernels::exl3::rows_f16(S->x, E, M.row_tok + r0 + a + c0, n, E, X16, s);
                blas_ck(cublasGemmEx(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * nff, n, E, &one, S->w16, CUDA_R_16F, E,
                                     X16, CUDA_R_16F, E, &zero, GU + (size_t) c0 * 2 * nff, CUDA_R_32F, 2 * nff,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                        "gemm dq gate/up");
            }
            gb::swiglu_rows(GU, GU, nrows, nff, g.swiglu_exp, s, 2 * nff);
            strata::kernels::dequant_f16(Ly.d_type, blob + Ly.down_off, 0, E, nff, S->w16, s);
            uint16_t* H16 = S->w16 + wdn;
            for (int c0 = (nrows - 1) / dn_rows * dn_rows; c0 >= 0; c0 -= dn_rows) {
                const int n = std::min(dn_rows, nrows - c0);
                strata::kernels::exl3::rows_f16(GU + (size_t) c0 * 2 * nff, 2 * nff, nullptr, n, nff, H16, s);
                blas_ck(cublasGemmEx(S->blas, CUBLAS_OP_T, CUBLAS_OP_N, E, n, nff, &one, S->w16, CUDA_R_16F, nff, H16,
                                     CUDA_R_16F, nff, &zero, GU + (size_t) c0 * E, CUDA_R_32F, E, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                        "gemm dq down");
            }
        };
        const auto run_any = [&](const uint8_t* wbase, int n_exp, size_t stride, const int* d_bounds, const int* hb,
                                 int r0) {
            if (Ly.exl3) {
                run_set_x3(wbase, n_exp, stride, d_bounds, hb, r0);
            } else if (deq) {
                for (int i = 0; i < n_exp; ++i)
                    if (hb[i + 1] > hb[i]) run_blas_dq(wbase + (size_t) i * stride, hb[i], hb[i + 1] - hb[i], r0);
            } else {
                run_set(wbase, n_exp, stride, d_bounds, hb, r0);
            }
        };
        if (2 * nff > E) {   // (deepseek41: 2 x 2304 < 5120 - the packed rows still fit)
            err = "glm prefill: 2 * n_ff_exp > n_embd (the gate/up rows would not fit the set's output rows)";
            return false;
        }
        // the disk-only experts (staged last, ascending): their indices in the reader's order - inside the segment
        // predicted one layer ago, or appended now
        std::vector<int> dlist, gidx;   // staged indices of the disk experts; their reader indices
        for (size_t i = 0; i < staged.size(); ++i)
            if (staged[i].ram == nullptr) dlist.push_back((int) i);
        int seg_end = dq.n.load();
        if (pred_layer == il) {
            size_t p = 0;
            for (int k : dlist) {
                const int e = staged[(size_t) k].e;
                while (p < pred_e.size() && pred_e[p] != e) ++p;
                if (p == pred_e.size()) break;
                gidx.push_back(pred0 + (int) p);
            }
            seg_end = pred0 + (int) pred_e.size();
            if (gidx.size() != dlist.size()) {   // (cannot happen: the plan's disk set is within the prediction)
                dq.copied.store(seg_end, std::memory_order_release);   // the predicted slots are free again
                gidx.clear();
            }
        }
        if (gidx.size() != dlist.size()) {
            for (int k : dlist) gidx.push_back(dq.push(il, staged[(size_t) k].e));
            seg_end = dq.n.load();
        }
        pred_layer = -1;
        if (pred_t > 0 && T >= pred_t && il + 1 < l1_ && F->L[(size_t) il + 1].moe) {
            pred_e = disk_only(il + 1);
            pred0 = dq.n.load();
            for (int e : pred_e) dq.push(il + 1, e);
            pred_layer = il + 1;
        }
        // the resident experts: the lightly routed ones through the light kernels, the rest in one MMQ product over
        // the layer's whole pool partition (the light ones empty there), while the copy stream stages
        if (n_light > 0) {
            const int o = place(rows_mmq, rows_res - rows_mmq);   // (light rows: at most kLightMax * kLightRows <= W)
            if (!gf::rows_experts(Ly.gu_type, Ly.d_type, P.base, P.stride, Ly.down_off, M.bounds + kLightOff, n_light,
                                  M.row_tok, S->x, T, E, nff, g.swiglu_exp, rows_mmq, rows_res, M.Xq, M.Hq,
                                  M.OUTP + (size_t) (o - rows_mmq) * E, E, s)) {
                err = "glm prefill: the light expert kernels refused the layer's types";
                return false;
            }
        }
        S->mark("moe_light", s);
        // STRATA_GLM_PREFILL_LIGHT_CHECK=1 (debug): the light rows again through MMQ, compared row by row
        // (it reads the light rows where they were before the output window: only before any other set ran)
        static const bool light_check = getenv("STRATA_GLM_PREFILL_LIGHT_CHECK") != nullptr;
        if (light_check && n_light > 0 && wb == rows_mmq) {
            const int nl_rows = rows_res - rows_mmq;
            std::vector<float> A((size_t) nl_rows * E), B((size_t) nl_rows * E);
            cudaStreamSynchronize(s);
            cudaMemcpy(A.data(), M.OUTP, A.size() * sizeof(float), cudaMemcpyDeviceToHost);
            int* hb = S->h_bounds + 4096;
            int nbc = 0, max_l = 0, li = 0;
            hb[nbc++] = 0;
            for (int sl = 0; sl < P.n_main; ++sl) {
                int add = 0;
                if (li < n_light && h_light[3 * li] == sl) {
                    add = h_light[3 * li + 2];
                    max_l = std::max(max_l, add);
                    ++li;
                }
                hb[nbc] = hb[nbc - 1] + add;
                ++nbc;
            }
            cudaMemcpy(M.bounds + 4096, hb, (size_t) nbc * sizeof(int), cudaMemcpyHostToDevice);
            w_lo = w_hi = 0;   // (the light rows are re-run over the same window rows: not added twice)
            run_set(P.base, P.n_main, P.stride, M.bounds + 4096, hb, rows_mmq);
            cudaStreamSynchronize(s);
            cudaMemcpy(B.data(), M.OUTP, B.size() * sizeof(float), cudaMemcpyDeviceToHost);
            double num = 0, den = 0, worst = 0;
            for (int r = 0; r < nl_rows; ++r) {
                double rn = 0, rd = 0;
                for (int j = 0; j < E; ++j) {
                    const double dlt = (double) A[(size_t) r * E + j] - B[(size_t) r * E + j];
                    rn += dlt * dlt;
                    rd += (double) B[(size_t) r * E + j] * B[(size_t) r * E + j];
                }
                num += rn;
                den += rd;
                worst = std::max(worst, std::sqrt(rn / std::max(1e-30, rd)));
            }
            std::fprintf(stderr, "glm prefill light check layer %d: %d experts, %d rows, rel L2 %.3e, worst row %.3e\n", il,
                         n_light, nl_rows, std::sqrt(num / std::max(1e-30, den)), worst);
        }
        run_any(P.base, P.n_main, P.stride, M.bounds + b_res, S->h_bounds + b_res, 0);
        (void) max_res;
        S->rows_resident += (uint64_t) rows_res;
        S->mark("moe_resident", s);
        if (S->pre_layer == il) {
            if (rows_pre > 0) {
                // in pieces of kPreEv experts as their copies land (STRATA_GLM_PRESTAGE_PIECES=0: after the last)
                static const bool pieces = [] { const char* v = getenv("STRATA_GLM_PRESTAGE_PIECES"); return v == nullptr || std::atoi(v) != 0; }();
                const int npre = (int) S->pre_e.size();
                if (pieces) {
                    for (int k0 = 0, k = 0; k0 < npre; k0 += PrefillState::kPreEv, ++k) {
                        const int cnt = std::min(PrefillState::kPreEv, npre - k0);
                        cudaStreamWaitEvent(s, S->ev_pdone[(size_t) k], 0);
                        run_any(S->pbuf + (size_t) k0 * P.stride, cnt, P.stride, M.bounds + b_pre + k0,
                                S->h_bounds + b_pre + k0, rows_res);
                    }
                } else {
                    cudaStreamWaitEvent(s, S->ev_pready, 0);
                    run_any(S->pbuf, npre, P.stride, M.bounds + b_pre, S->h_bounds + b_pre, rows_res);
                }
                S->pre_rows += (uint64_t) rows_pre;
            }
            cudaEventRecord(S->ev_pfree, s);   // (the next layer's prestage overwrites pbuf after this)
            S->pre_layer = -1;
            S->mark("moe_prestaged", s);
        }
        int dk = 0;   // the next disk expert (index into dlist / gidx)
        int n_staged_pcie = 0;   // (the split's balance: the copy stream's time when the host computes experts too)
        for (const auto& st : staged) n_staged_pcie += st.dev ? 0 : 1;
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            const Group& gr = groups[gi];
            const int b = (int) (gi % PrefillState::NG);
            uint8_t* gdst = S->gbuf + (size_t) b * PrefillState::GE * P.stride;
            cudaStreamWaitEvent(F->copy, S->ev_free[b], 0);
            const int dk0 = dk;
            for (int j = 0; j < gr.n; ++j) {
                const Staged& st = staged[(size_t) gr.first + j];
                const uint8_t* src = st.ram;
                if (src == nullptr) {
                    const int gi2 = gidx[(size_t) dk];
                    // every index below this one was copied (its event recorded) or predicted and never routed:
                    // released, so the reader can get to this one however far the prediction's gaps put it
                    if (dq.copied.load() < gi2) dq.copied.store(gi2, std::memory_order_release);
                    if (dq.landed.load(std::memory_order_acquire) <= gi2) {
                        const auto td = std::chrono::steady_clock::now();
                        while (dq.landed.load(std::memory_order_acquire) <= gi2) glmfast::cpu_relax();
                        S->ms_disk += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
                    }
                    src = S->gpin + (size_t) (gi2 % KL) * S->gstride;
                    ++dk;
                }
                cudaMemcpyAsync(gdst + (size_t) j * P.stride, src, Ly.blob,
                                st.dev ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, F->copy);
                if (st.ram == nullptr) {   // its landing slot is free once this copy ran
                    cudaEventRecord(S->ev_land[gidx[(size_t) dk - 1] % KL], F->copy);
                    dq.copied.store(gidx[(size_t) dk - 1] + 1, std::memory_order_release);
                }
                S->staged_ram += st.ram != nullptr && !st.dev;
                S->staged_vram += st.dev;
            }
            (void) dk0;
            cudaEventRecord(S->ev_ready[b], F->copy);
            cudaStreamWaitEvent(s, S->ev_ready[b], 0);
            run_any(gdst, gr.n, P.stride, M.bounds + gr.b_off, S->h_bounds + gr.b_off, gr.r0);
            cudaEventRecord(S->ev_free[b], s);
            S->rows_staged += (uint64_t) gr.nrows;
        }
        dq.copied.store(std::max(dq.copied.load(), seg_end), std::memory_order_release);   // predicted, never routed
        S->staged_disk += dlist.size();
        if (cpu_nrows > 0) cudaEventRecord(S->ev_c1, F->copy);
        if (S->trace) {
            auto& t = S->tr[(size_t) il];
            cudaEventRecord(t.c1, F->copy);
            t.grp = n_staged_pcie;
            t.ncpu = (int) cpu_set.size();
            t.crows = cpu_nrows;
            t.dl = t.dcpu = 0.0;
        }
        // the next MoE layer's prestage: its copies run once this layer's are done - during the next mixer.  With
        // CPU experts it is queued after their outputs' upload, on the same stream: an upload on a stream of its own
        // waited out the whole prestage (one copy engine a direction), and the combine with it
        if (cpu_nrows == 0) prestage(next_moe(il + 1));
        if (cpu_nrows > 0) {
            namespace kc = strata::kernels::cpu;
            const auto tc = std::chrono::steady_clock::now();
            const auto& nf = F->cpu_fmt[(size_t) il];
            constexpr size_t kA = kc::kNativeActBytes, kH = kc::kNativeHBytes;
            constexpr int kGu = 32, kDn = 64;
            if (S->c_act.size() < (size_t) cpu_nrows * kA) S->c_act.resize((size_t) cpu_nrows * kA);
            if (S->c_hq.size() < (size_t) cpu_nrows * kH) S->c_hq.resize((size_t) cpu_nrows * kH);
            if (S->c_ff.size() < (size_t) cpu_nrows * nff) S->c_ff.resize((size_t) cpu_nrows * nff);
            if (!glmfast::cuda_ok(cudaEventSynchronize(S->ev_xd), "glm prompt CPU rows", err)) return false;
            const int ncx = (int) cpu_set.size();
            std::vector<int> r_of((size_t) ncx);   // each expert's first row in the CPU segment
            for (int j = 0; j < ncx; ++j) r_of[(size_t) j] = S->h_base[cpu_set[(size_t) j].e] - cpu_r0;
            const auto up = [&](int ra, int rb) {
                if (rb > ra)
                    cudaMemcpyAsync(M.OUTPc + (size_t) ra * E, S->out_h + (size_t) ra * E,
                                    (size_t) (rb - ra) * E * sizeof(float), cudaMemcpyHostToDevice, F->copy);
            };
            // the iGPU first: the whole set in one call, the outputs straight into out_h
            bool on_igpu = false;
            if (igpu_lane(il)) {
                std::vector<const void*> bl((size_t) ncx);
                std::vector<int> nr((size_t) ncx);
                for (int j = 0; j < ncx; ++j) {
                    bl[(size_t) j] = cpu_set[(size_t) j].ram;
                    nr[(size_t) j] = S->h_counts[cpu_set[(size_t) j].e];
                }
                // (gathered: x row r is out_h row r, read before the outputs overwrite it)
                const int rc = S->x_gathered
                    ? F->igpu_rows(Ly.gu_type, Ly.d_type, E, nff, Ly.down_off, g.swiglu_exp, ncx, bl.data(), r_of.data(),
                                   nr.data(), S->id_h, S->out_h, cpu_nrows, 0, cpu_nrows, S->out_h, E)
                    : F->igpu_rows(Ly.gu_type, Ly.d_type, E, nff, Ly.down_off, g.swiglu_exp, ncx, bl.data(), r_of.data(),
                                   nr.data(), S->rt_h, S->x_h, T, 0, cpu_nrows, S->out_h, E);
                if (rc == 0) {
                    on_igpu = true;
                    S->ig_experts += (uint64_t) ncx;
                    S->ig_rows += (uint64_t) cpu_nrows;
                    up(0, cpu_nrows);
                } else if (rc < 0) {
                    std::fprintf(stderr, "glm prefill: the iGPU failed on the prompt's experts - the CPU threads compute\n");
                    F->igpu_rows = nullptr;
                }
            }
            if (!on_igpu) {
                F->cpu_pool->run(cpu_nrows, [&](int r) {
                    const float* xr = S->x_gathered ? S->out_h + (size_t) r * E : S->x_h + (size_t) S->rt_h[r] * E;
                    kc::native_quant_act(nf, xr, S->c_act.data() + (size_t) r * kA);
                });
                // experts with enough rows take the multi-row AVX-512 kernel (a weight row unpacked once for all of them,
                // ~2x ggml's per-pair dots from 8 rows on - kq_avx512.hpp); their activations packed once each
                static const bool kq_on = [] {
                    const char* v = getenv("STRATA_GLM_KQ");
                    return kc::kq_avx512_ok() && (v == nullptr || std::atoi(v) != 0);
                }();
                const auto kq_use = [&](int type, int act_type, int cnt) {
                    return kq_on && kc::kq_type_ok(type) && act_type == 15 /* Q8_K */ && cnt >= (type == 11 ? 8 : 12);
                };
                // the i-quant rows (IQ2_XXS/XS/S, IQ3_XXS/S) of an expert with 2+ rows: the AVX-512 multi-token kernel
                // (iq_avx512.cpp, a weight block decoded once for 4 tokens) - DeepSeek V4's IQ2_XS gate/up and IQ3_XXS
                // down on a Ryzen AI Max+ 395, one thread: 1.2 ms a token through an expert by ggml's dots, 0.52 from 4
                // tokens on.  STRATA_GLM_PREFILL_IQ512=0: ggml's (A/B)
                static const bool iq_on = [] {
                    const char* v = getenv("STRATA_GLM_PREFILL_IQ512");
                    return kc::cpu_avx512_ok() && (v == nullptr || std::atoi(v) != 0);
                }();
                const auto iq_use = [&](int type, int act_type, int cnt) {
                    return iq_on && kc::iq512_supported(type) && act_type == 15 /* Q8_K */ && cnt >= 2;
                };
                std::vector<size_t> off_a((size_t) ncx, SIZE_MAX), off_h((size_t) ncx, SIZE_MAX);
                size_t pack_a = 0, pack_h = 0;
                for (int j = 0; j < ncx; ++j) {
                    const int cnt = S->h_counts[cpu_set[(size_t) j].e];
                    if (kq_use(nf.gu_type, nf.gu_act, cnt)) {
                        off_a[(size_t) j] = pack_a;
                        pack_a += (kc::kq_act_bytes(cnt, E) + 63) & ~(size_t) 63;
                    }
                    if (kq_use(nf.d_type, nf.d_act, cnt)) {
                        off_h[(size_t) j] = pack_h;
                        pack_h += (kc::kq_act_bytes(cnt, nff) + 63) & ~(size_t) 63;
                    }
                }
                if (S->c_pack.size() < pack_a + pack_h + 64) S->c_pack.resize(pack_a + pack_h + 64);
                uint8_t* pk = (uint8_t*) (((uintptr_t) S->c_pack.data() + 63) & ~(uintptr_t) 63);
                uint8_t* pk_h = pk + pack_a;
                const auto pack = [&](bool gu) {
                    F->cpu_pool->run(ncx, [&](int j) {
                        const size_t off = gu ? off_a[(size_t) j] : off_h[(size_t) j];
                        if (off == SIZE_MAX) return;
                        const int cnt = S->h_counts[cpu_set[(size_t) j].e], b = r_of[(size_t) j];
                        thread_local std::vector<const void*> a;
                        a.resize((size_t) cnt);
                        for (int t = 0; t < cnt; ++t)
                            a[(size_t) t] = gu ? (const void*) (S->c_act.data() + (size_t) (b + t) * kA)
                                               : (const void*) (S->c_hq.data() + (size_t) (b + t) * kH);
                        kc::kq_pack_act(a.data(), cnt, gu ? E : nff, (gu ? pk : pk_h) + off);
                    });
                };
                if (pack_a > 0) pack(true);
                F->cpu_pool->run(ncx * (nff / kGu), [&](int job) {
                    const int j = job % ncx, r0 = (job / ncx) * kGu, b = r_of[(size_t) j];
                    const int cnt = S->h_counts[cpu_set[(size_t) j].e];
                    const uint8_t* blob = cpu_set[(size_t) j].ram;
                    if (off_a[(size_t) j] != SIZE_MAX) {
                        // gate and up rows [r0, r0 + kGu) for every row of the expert, then the clamped swiglu
                        thread_local std::vector<float> gu_t;
                        thread_local std::vector<float*> go, uo;
                        gu_t.resize((size_t) 2 * cnt * kGu);
                        go.resize((size_t) cnt);
                        uo.resize((size_t) cnt);
                        for (int t = 0; t < cnt; ++t) {
                            go[(size_t) t] = gu_t.data() + (size_t) t * kGu;
                            uo[(size_t) t] = gu_t.data() + (size_t) (cnt + t) * kGu;
                        }
                        const void* pa = pk + off_a[(size_t) j];
                        kc::kq_rows_packed(nf.gu_type, blob + (size_t) r0 * nf.gu_row, nf.gu_row, E, pa, cnt, go.data(), 0,
                                           kGu);
                        kc::kq_rows_packed(nf.gu_type, blob + Ly.gu_bytes + (size_t) r0 * nf.gu_row, nf.gu_row, E, pa,
                                           cnt, uo.data(), 0, kGu);
                        const float lim = g.swiglu_exp;
                        for (int t = 0; t < cnt; ++t) {
                            float* ff = S->c_ff.data() + (size_t) (b + t) * nff + r0;
                            for (int r = 0; r < kGu; ++r) {
                                const float gg = std::fmin(go[(size_t) t][r], lim);
                                const float uu = std::fmin(std::fmax(uo[(size_t) t][r], -lim), lim);
                                ff[r] = (gg / (1.f + std::exp(-gg))) * uu;
                            }
                        }
                        return;
                    }
                    thread_local std::vector<const void*> a;
                    thread_local std::vector<float*> o;
                    a.resize((size_t) cnt);
                    o.resize((size_t) cnt);
                    for (int t = 0; t < cnt; ++t) {
                        a[(size_t) t] = S->c_act.data() + (size_t) (b + t) * kA;
                        o[(size_t) t] = S->c_ff.data() + (size_t) (b + t) * nff;
                    }
                    if (iq_use(nf.gu_type, nf.gu_act, cnt)) {
                        // gate and up rows [r0, r0 + kGu) for every token of the expert, then the clamped swiglu
                        thread_local std::vector<float> gu_t;
                        thread_local std::vector<float*> go, uo;
                        gu_t.resize((size_t) 2 * cnt * kGu);
                        go.resize((size_t) cnt);
                        uo.resize((size_t) cnt);
                        for (int t = 0; t < cnt; ++t) {
                            go[(size_t) t] = gu_t.data() + (size_t) t * kGu;
                            uo[(size_t) t] = gu_t.data() + (size_t) (cnt + t) * kGu;
                        }
                        kc::iq512_rows(nf.gu_type, blob + (size_t) r0 * nf.gu_row, nf.gu_row, E, a.data(), cnt, go.data(),
                                       0, kGu);
                        kc::iq512_rows(nf.gu_type, blob + Ly.gu_bytes + (size_t) r0 * nf.gu_row, nf.gu_row, E, a.data(),
                                       cnt, uo.data(), 0, kGu);
                        const float lim = g.swiglu_exp;
                        const bool clamp = lim != std::numeric_limits<float>::infinity();
                        for (int t = 0; t < cnt; ++t)
                            for (int r = 0; r < kGu; ++r) {
                                float gg = go[(size_t) t][r], uu = uo[(size_t) t][r];
                                if (clamp) {
                                    gg = std::fmin(gg, lim);
                                    uu = std::fmin(std::fmax(uu, -lim), lim);
                                }
                                o[(size_t) t][r0 + r] = (gg / (1.f + std::exp(-gg))) * uu;
                            }
                        return;
                    }
                    kc::native_gu_rows_split(nf, blob, blob + Ly.gu_bytes, a.data(), cnt, o.data(), r0, r0 + kGu,
                                             g.swiglu_exp);
                });
                F->cpu_pool->run(cpu_nrows, [&](int r) {
                    kc::native_quant_h(nf, S->c_ff.data() + (size_t) r * nff, S->c_hq.data() + (size_t) r * kH);
                });
                if (pack_h > 0) pack(false);
                // the down rows in two parts: the first part's outputs (about two thirds of the rows) go up on the copy
                // stream while the second part computes
                int j_split = ncx;
                for (int j = 0; j < ncx; ++j)
                    if (r_of[(size_t) j] >= cpu_nrows * 2 / 3) {
                        j_split = j;
                        break;
                    }
                const auto down = [&](int j0, int j1) {
                  const int nj = j1 - j0;
                  if (nj <= 0) return;
                  F->cpu_pool->run(nj * (E / kDn), [&](int job) {
                    const int j = j0 + job % nj, r0 = (job / nj) * kDn, b = r_of[(size_t) j];
                    const int cnt = S->h_counts[cpu_set[(size_t) j].e];
                    thread_local std::vector<const void*> h;
                    thread_local std::vector<float*> o;
                    h.resize((size_t) cnt);
                    o.resize((size_t) cnt);
                    const uint8_t* down = cpu_set[(size_t) j].ram + Ly.down_off;
                    if (off_h[(size_t) j] != SIZE_MAX) {
                        for (int t = 0; t < cnt; ++t) o[(size_t) t] = S->out_h + (size_t) (b + t) * E + r0;
                        kc::kq_rows_packed(nf.d_type, down + (size_t) r0 * nf.d_row, nf.d_row, nff, pk_h + off_h[(size_t) j],
                                           cnt, o.data(), 0, kDn);
                        return;
                    }
                    for (int t = 0; t < cnt; ++t) {
                        h[(size_t) t] = S->c_hq.data() + (size_t) (b + t) * kH;
                        o[(size_t) t] = S->out_h + (size_t) (b + t) * E;
                    }
                    if (iq_use(nf.d_type, nf.d_act, cnt)) {   // (out[t][r] for r in [r0, r0 + kDn), as the split's)
                        kc::iq512_rows(nf.d_type, down, nf.d_row, nff, h.data(), cnt, o.data(), r0, r0 + kDn);
                        return;
                    }
                    kc::native_down_rows_split(nf, down, h.data(), cnt, o.data(), r0, r0 + kDn);
                  });
                };
                const int r_split = j_split < ncx ? r_of[(size_t) j_split] : cpu_nrows;
                down(0, j_split);
                up(0, r_split);
                down(j_split, ncx);
                up(r_split, cpu_nrows);
            }
            cudaEventRecord(S->ev_out, F->copy);
            cudaStreamWaitEvent(s, S->ev_out, 0);   // (the combine reads them; the next layer's x overwrites x after)
            // (the host's time, from the plan and from its experts' start, before any wait on the copies below)
            const auto th = std::chrono::steady_clock::now();
            const double dt = std::chrono::duration<double, std::milli>(th - tc).count();
            const double dl_host = std::chrono::duration<double, std::milli>(th - tp).count();
            // the prestage count: the copy stream idle before the plan (the copies landed early) -> as many more as
            // that time moves; every other expert on the CPU and it still done before the copies -> fewer by the
            // difference; else (the plan balanced the two) the same
            // (not the chunk's first MoE layer: its copies had the dense layers before it too).  Measured before the next
            // layer's prestage is queued: it records the same events (ev_pstart, ev_pready) again, and read after it
            // they were not ready - the count stayed at the buffer's cap whatever the lanes did
            if (pre_n > 0 && il != next_moe(l0_)) {
                const double dl = dl_host;
                cudaEventSynchronize(S->ev_c1);
                float landed = 0.0f, pc = 0.0f, pt = 0.0f;
                if (cudaEventElapsedTime(&landed, S->ev_c0, S->ev_pready) == cudaSuccess &&
                    cudaEventElapsedTime(&pc, S->ev_c0, S->ev_c1) == cudaSuccess &&
                    cudaEventElapsedTime(&pt, S->ev_pstart, S->ev_pready) == cudaSuccess) {
                    const double cm = pt / pre_n;   // (the copies ran back to back from ev_pstart)
                    if (cm > 0.2 && cm < 20.0) S->copy_ms = S->copy_ms > 0.0 ? 0.8 * S->copy_ms + 0.2 * cm : cm;
                    const double pm = S->copy_ms > 0.0 ? S->copy_ms : 1.6;
                    double nn = pre_n;
                    if (landed < -3.0f) nn = pre_n + 0.8 * (-landed) / pm;
                    else if (dl + 5.0 < pc) nn = pre_n - 0.8 * (pc - dl) / pm;
                    double& r = S->pre_rate[Ly.recr ? 0 : 1];
                    r = r > 0.0 ? 0.5 * r + 0.5 * nn / T : nn / T;
                }
            }
            prestage(next_moe(il + 1));
            S->cpu_experts += (uint64_t) ncx;
            S->cpu_rows += (uint64_t) cpu_nrows;
            S->ms_cpu += dt;
            if (S->trace) {
                S->tr[(size_t) il].dcpu = dt;
                S->tr[(size_t) il].dl = dl_host;
            }
            if (cpu_nrows >= 64 && on_igpu)   // (a layer with a few rows says little about the per-row cost)
                S->row_ms_ig = 0.5 * S->row_ms_ig +
                               0.5 * std::max(0.002, (dt - (double) ncx * S->fix_ms_ig) / (double) cpu_nrows);
            else if (cpu_nrows >= 64)
                S->row_ms = 0.5 * S->row_ms +
                            0.5 * std::max(0.005, (dt - 2.0 * (double) ncx * F->cpu_c_ms) / (double) cpu_nrows);
            // the balance: when the copy stream's staged copies ended against when the host's experts did, both from
            // the plan's start (prestage copies it still owed and x's way down included)
            if (n_staged_pcie > 0) {
                const double dl = dl_host;
                cudaEventSynchronize(S->ev_c1);
                float pc = 0.0f;
                if (cudaEventElapsedTime(&pc, S->ev_c0, S->ev_c1) == cudaSuccess && pc > 1.0f && dl > 1.0)
                    S->kappa = std::min(4.0, std::max(0.25, S->kappa * std::min(1.25, std::max(0.8, std::sqrt(dl / pc)))));
            }
        }
        S->mark("moe_staged", s);
        dump_row("pf_shexp-" + std::to_string(il), S->ffn, E);
        flush();
        if (cpu_nrows > 0) gb::moe_combine_add(M.OUTPc, cpu_r0, cpu_r0, rows, M.pos, M.rw, T, K, E, S->ffn, s);
        abl_out(il, 1, S->ffn, T, E, s);   // (the whole FFN: shared + routed + the CPU lane's rows)
        S->mark("combine", s);
        if (S->trace) cudaEventRecord(S->tr[(size_t) il].end, s);
        dump_row("ffn_out-" + std::to_string(il), S->ffn, E);
    }
    // the last layer's write half: R = post x ffn + comb . R (qwen35moe's plain residual has none: the NextN rows below
    // add the last FFN themselves - this update, run with the hyper-connections' unset post / comb, broke R and with it
    // every prompt position's draft cache, so the drafter's confidences stayed under its threshold after a prompt)
    if (!tail_skip && !g.qwen35() && kv_cut < 0) {
        gb::hc_update(S->ffn, S->R, S->post, S->comb, nullptr, T, E, s);
        S->mark("hc", s);
        if (dsp_ != nullptr) {   // DSpark: the chunk's last rows into its context
            dspark_capture_rows(l1_ - 1, S->R, T, s);
            const int n = std::min(T, 128);
            if (!dspark_commit(p0 + T - n, n, true, s)) {
                err = "dspark: the prompt's context commit";
                return false;
            }
        }
    }
    // ---- the NextN block's caches for these positions (the half that carries it): position p reads h_p and the
    //      embedding of the token at p + 1, so only the cache-writing half of the block runs - eh_proj, the attention
    //      norm, kv_a and the indexer's key/gate - no queries, no attention, no FFN
    if (q4d_ != nullptr && next_ids != nullptr && kv_cut < 0) {
        // qwen4exp's draft layer: its K / V rows from the rows' final residuals (the last FFN combined into R here: the
        // chunk's R has no other reader - the prompt's last token runs on the token path) and their next tokens
        int Tv = 0;
        while (Tv < T && next_ids[Tv] >= 0) ++Tv;
        if (Tv > 0) {
            Carve c{S->uni};
            float* xn = c.take<float>((size_t) Tv * g.hc * E);
            q35::hc4_combine_norm(S->R, S->ffn, S->post, g.hc, F->q4_head[0], g.norm_eps, E, g.hc, Tv, xn, s);
            std::vector<int32_t> nx(next_ids, next_ids + Tv);
            if (!q4x_mtp_rows(p0, Tv, S->R, (size_t) g.hc * E, nx.data(), err)) return false;
            S->mark("mtp_rows", s);
        }
    }
    if (mtp_il_ >= 0 && next_ids != nullptr && kv_cut < 0) {
        int Tv = 0;   // rows whose next token is known (all but possibly the prompt's last)
        while (Tv < T && next_ids[Tv] >= 0) ++Tv;
        if (g.qwen35() && Tv > 0) {
            // qwen35moe: the block's K / V rows alone - eh_proj of [enorm(the next token's embedding) | hnorm(the trunk's
            // normed hidden)], the attention norm, k / v (k_norm, RoPE) into its caches at the rows' positions
            const auto& Ly = F->L[(size_t) mtp_il_];
            const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
            if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
                err = "glm prefill: no embedding dequantizer for the draft block";
                return false;
            }
            const size_t row_b = ggml_row_size((ggml_type) pack_emb_type_, E);
            if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "glm prompt mtp embedding", err)) return false;
            for (int t = 0; t < Tv; ++t)
                tt->to_float(pack_emb_src_ + (size_t) next_ids[t] * row_b, S->emb_h + (size_t) t * E, E);
            const int KVq = g.attn_head_kv * g.head_dim;
            Carve c{S->uni};
            float* h = c.take<float>((size_t) Tv * E);
            uint16_t* h16 = c.take<uint16_t>((size_t) Tv * E);
            float* emb = c.take<float>((size_t) Tv * E);
            uint16_t* cat16 = c.take<uint16_t>((size_t) Tv * 2 * E);
            float* hid = c.take<float>((size_t) Tv * E);
            float* kq = c.take<float>((size_t) Tv * KVq);
            float* vq = c.take<float>((size_t) Tv * KVq);
            cudaMemcpyAsync(emb, S->emb_h, (size_t) Tv * E * sizeof(float), cudaMemcpyHostToDevice, s);
            // h = output_norm(R + the last layer's FFN) a row (the tail skip is off with a draft block)
            q35::add_rms_rows(S->R, 4 * E, S->ffn, w_.at("output_norm.weight"), g.norm_eps, Tv, E, h, h16, s);
            gb::mtp_in_rows(emb, h, Ly.enorm, Ly.hnorm, g.norm_eps, Tv, E, cat16, s);
            hgemm_q(Ly.eh, E, 2 * E, cat16, 2 * E, hid, E, Tv, 0.0f);
            gb::rms_rows(hid, Ly.attn_norm, g.norm_eps, Tv, E, S->x, S->x16, s);
            hgemm_q(Ly.ak, KVq, E, S->x16, E, kq, KVq, Tv, 0.0f);
            hgemm_q(Ly.av, KVq, E, S->x16, E, vq, KVq, Tv, 0.0f);
            q35::attn_kv_rows(kq, vq, Ly.k_norm, g.norm_eps, g.attn_head_kv, g.head_dim, g.n_rot, g.rope_base, p0, Tv,
                              Ly.kc, Ly.vc, s);
            q35_row_done_ = std::max(q35_row_done_, p0 + Tv - 1);
            S->mark("mtp_rows", s);
        } else if (Tv > 0) {
            const auto& Ly = F->L[(size_t) mtp_il_];
            const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
            if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
                err = "glm prefill: no embedding dequantizer for the draft block";
                return false;
            }
            const size_t row_b = ggml_row_size((ggml_type) pack_emb_type_, E);
            if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "glm prompt mtp embedding", err)) return false;
            for (int t = 0; t < Tv; ++t)
                tt->to_float(pack_emb_src_ + (size_t) next_ids[t] * row_b, S->emb_h + (size_t) t * E, E);
            Carve c{S->uni};
            float* h = c.take<float>((size_t) Tv * E);
            float* emb = c.take<float>((size_t) Tv * E);
            uint16_t* cat16 = c.take<uint16_t>((size_t) Tv * 2 * E);
            float* hid = c.take<float>((size_t) Tv * E);
            float* kv = c.take<float>((size_t) Tv * g.kv_lora);
            float* ik = c.take<float>((size_t) Tv * g.idx_key);
            float* ig = c.take<float>((size_t) Tv * g.idx_key);
            cudaMemcpyAsync(emb, S->emb_h, (size_t) Tv * E * sizeof(float), cudaMemcpyHostToDevice, s);
            gb::head_rows(S->R, w_.at("output_norm.weight"), g.norm_eps, Tv, E, h, s);
            gb::mtp_in_rows(emb, h, Ly.enorm, Ly.hnorm, g.norm_eps, Tv, E, cat16, s);
            hgemm_q(Ly.eh, E, 2 * E, cat16, 2 * E, hid, E, Tv, 0.0f);
            gb::rms_rows(hid, Ly.attn_norm, g.norm_eps, Tv, E, S->x, S->x16, s);
            hgemm_q(Ly.kv_a, g.kv_lora, E, S->x16, E, kv, g.kv_lora, Tv, 0.0f);
            sgemm_bf16(Ly.idx_k, g.idx_key, E, S->x, E, ik, g.idx_key, Tv, 1.0f);
            sgemm_bf16(Ly.idx_gate, g.idx_key, E, S->x, E, ig, g.idx_key, Tv, 1.0f);
            gb::DsaPrepArgs d;
            d.kv_raw = kv;
            d.kv_norm = Ly.kv_a_norm;
            d.lat = (uint16_t*) (state_ + dsa_lat_[(size_t) mtp_il_]);
            d.lat_q8 = lat_q8_;
            d.kv_lora = g.kv_lora;
            d.ik_raw = ik;
            d.k_norm_w = Ly.k_norm_w;
            d.k_norm_b = Ly.k_norm_b;
            d.ik_cache = state_ + dsa_ik_[(size_t) mtp_il_];
            d.ig_raw = ig;
            d.ig_cache = state_ + dsa_ig_[(size_t) mtp_il_];
            d.idx_key = g.idx_key;
            d.ring = ik_ring_;
            d.eps = g.norm_eps;
            // in pieces the indexer's key / gate ring holds with the open pool before them (a one-GPU chunk is up to
            // 32K positions, the ring ~8K): each piece's rows are pooled before the next one overwrites the ring
            const int kp = g.idx_kpool;
            const int piece = std::max(kp, ik_ring_ - 64);
            for (int t0 = 0; t0 < Tv; t0 += piece) {
                const int tn = std::min(piece, Tv - t0);
                d.kv_raw = kv + (size_t) t0 * g.kv_lora;
                d.ik_raw = ik + (size_t) t0 * g.idx_key;
                d.ig_raw = ig + (size_t) t0 * g.idx_key;
                d.p0 = (int) p0 + t0;
                d.T = tn;
                gb::dsa_prep(d, s);
                const int pool_lo = (int) ((p0 + t0 + kp) / kp - 1), pool_hi = (int) ((p0 + t0 + tn) / kp - 1);
                if (pool_hi >= pool_lo)
                    gb::dsa_pool(state_ + dsa_ik_[(size_t) mtp_il_], state_ + dsa_ig_[(size_t) mtp_il_], Ly.ape,
                                 state_ + dsa_pool_[(size_t) mtp_il_], g.idx_key, kp, pool_lo, pool_hi - pool_lo + 1, s,
                                 ik_ring_);
            }
            S->mark("mtp_cache", s);
        }
    }
    S->collect();
    if (S->trace) {
        cudaStreamSynchronize(s);
        cudaStreamSynchronize(F->copy);
        const auto el = [](cudaEvent_t a, cudaEvent_t b) {
            float v = 0.0f;
            return cudaEventElapsedTime(&v, a, b) == cudaSuccess ? (double) v : -1.0;
        };
        std::fprintf(stderr, "glm prefill trace CUDA%d, a chunk of %d tokens (ms; plan = the routing known, the lanes "
                             "from there):\n", dev_, T);
        for (int il = l0_; il < l1_; ++il) {
            const auto& t = S->tr[(size_t) il];
            if (!t.on) continue;
            std::fprintf(stderr, "  layer %2d: start->plan %6.1f | prestage %3d, %3d left at plan, landed plan%+7.1f | copies "
                                 "%3d end plan+%6.1f | cpu %3d experts %5d rows end plan+%6.1f (compute %6.1f) | layer "
                                 "end plan+%6.1f\n",
                         il, el(t.start, t.plan), t.pre_n, t.pre_left, t.pre_n > 0 ? el(t.plan, t.pre) : 0.0, t.grp,
                         el(t.plan, t.c1), t.ncpu, t.crows, t.dl, t.dcpu, el(t.plan, t.end));
        }
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm prefill: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool Glm5Model::dump_state(const std::string& path) {
    int half = 0;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get(), ++half) {
        cudaSetDevice(m->dev_);
        cudaDeviceSynchronize();
        std::vector<float> buf(m->state_bytes_ / sizeof(float));
        if (cudaMemcpy(buf.data(), m->state_, buf.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
        const std::string base = path + "." + std::to_string(half);
        FILE* f = std::fopen((base + ".bin").c_str(), "wb");
        if (f == nullptr) return false;
        std::fwrite(buf.data(), sizeof(float), buf.size(), f);
        std::fclose(f);
        FILE* t = std::fopen((base + ".txt").c_str(), "w");
        if (t == nullptr) return false;
        std::fprintf(t, "max_ctx %lld pos %lld\n", (long long) m->max_ctx_, (long long) m->pos_);
        for (int il = m->l0_; il < m->l1_; ++il)
            std::fprintf(t, "%d %lld %lld %lld %lld %lld %lld\n", il, (long long) m->kda_S_[(size_t) il],
                         (long long) m->kda_conv_[(size_t) il], (long long) m->dsa_lat_[(size_t) il],
                         (long long) m->dsa_ik_[(size_t) il], (long long) m->dsa_ig_[(size_t) il],
                         (long long) m->dsa_pool_[(size_t) il]);
        std::fclose(t);
    }
    cudaSetDevice(dev_);
    return true;
}

// ---------------------------------------------------------------- the prompt
bool Glm5Model::prefill(const std::vector<int32_t>& tokens, std::string& err, int32_t next_token) {
    err.clear();
    prefill_next_ = next_token;
    if (fast_ == nullptr || tokens.empty()) return false;
    dspark_settle();
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())   // (a tail lent to the vision encoder: the
        if (m->pf_ == nullptr || m->fast_ == nullptr || m->vis_lent_) return false;   // token path, until it is back)
    // a handful of tokens costs less through the token path (the batched path streams every expert they touch)
    int64_t min_n = 32;
    if (const char* mn = getenv("STRATA_GLM_PREFILL_MIN")) min_n = std::max<int64_t>(1, std::atoll(mn));
    if (!prefill_logits && (int64_t) tokens.size() < min_n) return false;
    if (prefill_logits && (!g_.v41 || split_next_ || l0_ != 0 || l1_ != g_.n_layers)) {
        err = "prefill logits requires single-device DeepSeek V4.1"; return false;
    }
    // the chunk for this prompt: no bigger than it needs (rounded up to 64), so it borrows only that much; a prompt
    // longer than the largest chunk is cut into EQUAL chunks (the parts of a split pipeline chunks)
    int Tmax = pf_->T;
    for (Glm5Model* m = split_next_.get(); m != nullptr; m = m->split_next_.get()) Tmax = std::min(Tmax, m->pf_->T);
    const int64_t nn = (int64_t) tokens.size(), nchunks = (nn + Tmax - 1) / Tmax;
    // one GPU has no pipeline to balance: full chunks and the remainder last - every chunk stages nearly every expert,
    // so 8449 tokens as 8192 + 257 cost about one chunk's experts, as two equal 4225s about two (240 -> ~300 tok/s)
    const int Trun = split_next_ == nullptr ? (int) std::min<int64_t>(Tmax, (nn + 63) / 64 * 64)
                                            : (int) std::min<int64_t>(Tmax, ((nn + nchunks - 1) / nchunks + 63) / 64 * 64);
    bool ok = true;
    progress_at("the prompt: lending the expert pool's tail");   // (the serve watchdog's stages, issue #40)
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        m->prefill_carve(Trun);
        if (!m->prefill_lend(err)) { ok = false; break; }
    }
    // STRATA_GLM_STALL_TEST=<s> (a test of the serve watchdog): every batched prompt stops here for s seconds
    static const int stall_test = [] {
        const char* v = getenv("STRATA_GLM_STALL_TEST");
        return v ? std::max(0, std::atoi(v)) : 0;
    }();
    if (ok && stall_test > 0) {
        progress_at("the prompt: STRATA_GLM_STALL_TEST's pause");
        std::this_thread::sleep_for(std::chrono::seconds(stall_test));
    }
    if (ok) ok = prefill_run(tokens, err);
    progress_at("the prompt: returning the expert pool's tail");
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        std::string return_err;
        if (!m->prefill_return(return_err)) {
            if (ok) err = return_err;
            ok = false;
        }
    }
    cudaSetDevice(dev_);
    return ok;
}

bool Glm5Model::prefill_run(const std::vector<int32_t>& tokens, std::string& err) {
    const Glm5Geometry& g = g_;
    const int64_t n = (int64_t) tokens.size();
    if (pos_ + n > max_ctx_) {
        err = "glm prefill: the prompt (" + std::to_string(pos_ + n) + " positions) exceeds the context (" +
              std::to_string(max_ctx_) + ")";
        return false;
    }
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm prefill: no embedding dequantizer";
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int E = g.n_embd;
    const size_t row_b = ggml_row_size((ggml_type) pack_emb_type_, E);
    int Tc = pf_->T_bound;
    for (Glm5Model* m = split_next_.get(); m != nullptr; m = m->split_next_.get()) Tc = std::min(Tc, m->pf_->T_bound);
    const int64_t pos_start = pos_;
    // a long deepseek41 prompt's skip: the layers from the first compress-ratio-1 layer on (V4.1: 20-39) read the
    // earlier positions only through that layer's compressed KV and index keys (they own none), and each looks back
    // one window (n_swa - 1 positions) more than the layer before - so only the last keep = (n_layers - yl) (n_swa - 1)
    // + n_swa rows need them for the prompt's last row and the window rings the decode reads (exact up to the
    // experts' summing order).  The rows before the cut stop at layer yl's attention (pf_kv_cut_).  One device, the
    // whole trunk; STRATA_GLM_PROMPT_SKIP=0: off
    int yl = -1;
    int64_t cut = 0;
    static const bool skip_on = [] { const char* v = getenv("STRATA_GLM_PROMPT_SKIP"); return v == nullptr || std::atoi(v) != 0; }();
    if (skip_on && !prefill_logits && g.v41 && split_next_ == nullptr && l0_ == 0 && l1_ == g.n_layers && mtp_il_ < 0) {
        for (int il = 0; il < g.n_layers && yl < 0; ++il)
            if (g.ratio(il) == 1) yl = il;
        bool ok = yl > 0 && g.kv_src[(size_t) yl];
        for (int il = yl + 1; ok && il < g.n_layers; ++il) ok = g.ratio(il) == 1 && !g.kv_src[(size_t) il];
        const int64_t keep = (int64_t) (g.n_layers - yl) * (g.n_swa - 1) + g.n_swa;
        if (ok && n - keep >= 1024) cut = n - keep;
    }
    // the chunks' starts (every Tc rows; with a cut, the chunks before it end at it)
    std::vector<int64_t> cs;
    for (int64_t x = 0; x < cut; x += Tc) cs.push_back(x);
    for (int64_t x = cut; x < n; x += Tc) cs.push_back(x);
    cs.push_back(n);
    const int nc = (int) cs.size() - 1;
    const auto chunk_len = [&](int c) { return (int) (cs[(size_t) c + 1] - cs[(size_t) c]); };
    if (cut > 0 && getenv("STRATA_GLM_PREFILL_VERBOSE") != nullptr)
        std::fprintf(stderr, "glm prefill: prompt skip - %lld of %lld rows stop at layer %d's attention\n",
                     (long long) cut, (long long) n, yl);
    // the token after every position of chunk c (the draft block's caches read it): the prompt's own next token,
    // the caller's for the last position (-1: unknown - that position is filled by the first draft of the decode)
    const auto next_of = [&](int c, std::vector<int32_t>& out) {
        const int T = chunk_len(c);
        out.resize((size_t) T);
        for (int t = 0; t < T; ++t) {
            const int64_t j = cs[(size_t) c] + t + 1;
            out[(size_t) t] = j < n ? tokens[(size_t) j] : prefill_next_;
        }
    };
    // chunk c's embedding rows into this half's R (host-dequantized from the shard mapping)
    const auto embed = [&](int c) -> bool {
        PrefillState* S = pf_;
        const int T = chunk_len(c);
        const int64_t i = cs[(size_t) c];
        if (!glmfast::cuda_ok(cudaStreamSynchronize(fast_->cs), "glm prompt embedding", err)) return false;
        // the rows by the disk workers' threads (a 32K chunk's rows took ~100 ms on one thread, the GPU idle);
        // STRATA_GLM_PREFILL_EMB_PAR=0: one thread
        static const bool emb_par = [] { const char* v = getenv("STRATA_GLM_PREFILL_EMB_PAR"); return v == nullptr || std::atoi(v) != 0; }();
        const auto row = [&](int t) {
            tt->to_float(pack_emb_src_ + (size_t) tokens[(size_t) (i + t)] * row_b, S->emb_h + (size_t) t * E, E);
            if (const float* img = image_row(pos_start + i + t))   // an image's row in place of its <|image|> token
                std::memcpy(S->emb_h + (size_t) t * E, img, (size_t) E * sizeof(float));
        };
        if (emb_par && fast_->workers != nullptr && T >= 256) {
            const int nj = 8;
            fast_->workers->run(nj, [&](int job) { for (int t = job; t < T; t += nj) row(t); });
        } else {
            for (int t = 0; t < T; ++t) row(t);
        }
        // the embedding lands in x (free until the first layer's gates write it) and fans out to the 4 streams
        cudaMemcpyAsync(S->x, S->emb_h, (size_t) T * E * sizeof(float), cudaMemcpyHostToDevice, fast_->cs);
        if (S->toks_d != nullptr)   // (pageable source: the copy completes before the call returns)
            cudaMemcpyAsync(S->toks_d, tokens.data() + i, (size_t) T * sizeof(int), cudaMemcpyHostToDevice, fast_->cs);
        // deepseek41: which rows of the chunk are images (the router's exp_probs_b_vl); none: route takes the plain bias
        S->img_on = false;
        if (g.v41 && S->img_d != nullptr && has_image_rows()) {
            std::vector<unsigned char> fl((size_t) T, 0);
            for (int t = 0; t < T; ++t)
                if (is_image_pos(pos_start + i + t)) { fl[(size_t) t] = 1; S->img_on = true; }
            if (S->img_on)   // (pageable source: the copy completes before the call returns)
                cudaMemcpyAsync(S->img_d, fl.data(), (size_t) T, cudaMemcpyHostToDevice, fast_->cs);
        }
        abl_res(S->x, T, E, fast_->cs);
        gb::embed_rows(S->x, S->R, T, E, fast_->cs);
        if (g.v41 || g.q4x) S->toks_h.assign(tokens.begin() + i, tokens.begin() + i + T);
        return true;
    };
    // a later part's copy of the chunk's token ids (deepseek4's hash-routed layers read them; embed() fills only the
    // first part's): on its own stream, before its layers.  (A split whose later part starts below hash_layers routed
    // those layers by whatever its buffer held.)
    const auto toks_to = [&](Glm5Model* m, int c) {
        if (m->pf_ == nullptr || m->pf_->toks_d == nullptr) return;
        cudaMemcpyAsync(m->pf_->toks_d, tokens.data() + cs[(size_t) c], (size_t) chunk_len(c) * sizeof(int),
                        cudaMemcpyHostToDevice, m->fast_->cs);
    };
    std::vector<Glm5Model*> parts;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) parts.push_back(m);
    const int np = (int) parts.size();
    const bool pipelined = np > 1 && nc > 1 && getenv("STRATA_GLM_PREFILL_SERIAL") == nullptr;
    bool cancelled = false;
    if (!pipelined) {
        // one half after the other, chunk by chunk
        for (int c = 0; c < nc; ++c) {
            const int T = chunk_len(c);
            const int64_t p0 = pos_start + cs[(size_t) c];
            if (prefill_logits)
                std::fprintf(stderr, "LOGPOS prefill chunk: position %lld length %d\n", (long long) p0, T);
            cudaSetDevice(dev_);
            if (!embed(c)) return false;
            std::vector<int32_t> nx0;
            if (mtp_il_ >= 0) next_of(c, nx0);   // a single device carries the draft block itself
            pf_kv_cut_ = cs[(size_t) c + 1] <= cut ? yl : -1;
            const bool half_ok = prefill_half(p0, T, err, nx0.empty() ? nullptr : nx0.data());
            pf_kv_cut_ = -1;
            if (!half_ok) return false;
            Glm5Model* prev = this;
            for (Glm5Model* m = split_next_.get(); m != nullptr; m = m->split_next_.get()) {
                PrefillState* SP = prev->pf_;
                const size_t hop = (size_t) T * 4 * E * sizeof(float);
                cudaSetDevice(prev->dev_);
                cudaMemcpyAsync(SP->hop_h, SP->R, hop, cudaMemcpyDeviceToHost, prev->fast_->cs);
                cudaEventRecord(SP->ev_hop, prev->fast_->cs);
                cudaSetDevice(m->dev_);
                cudaStreamWaitEvent(m->fast_->cs, SP->ev_hop, 0);
                cudaMemcpyAsync(m->pf_->R, SP->hop_h, hop, cudaMemcpyHostToDevice, m->fast_->cs);
                m->pos_ = p0;
                toks_to(m, c);
                std::vector<int32_t> nx;
                if (m->mtp_il_ >= 0) next_of(c, nx);
                if (!m->prefill_half(p0, T, err, nx.empty() ? nullptr : nx.data())) return false;
                prev = m;
            }
            if (prefill_logits) {
                namespace d41 = strata::kernels::dsv41;
                namespace gf = strata::kernels::glmf;
                const WSlot& ow = ws_map_.at("output.weight");
                std::vector<float> logits((size_t) g.n_vocab);
                for (int t = 0; t < T; ++t) {
                    d41::head_prep(pf_->R + (size_t) t * 4 * E, pf_->pre2 + (size_t) t * 4,
                                   w_.at("output_norm.weight"), g.norm_eps, E,
                                   fast_->head_x, fast_->head_xq, fast_->cs);
                    gf::MvJob job = {ow.q, fast_->head_xq, fast_->head_x, sc_ + sc_logits, nullptr,
                                     1.0f, ow.type, E, g.n_vocab};
                    if (ow.type == 0) job.w = ow.f32;
                    if (!gf::mv(&job, 1, fast_->cs) ||
                        !glmfast::cuda_ok(cudaMemcpyAsync(logits.data(), sc_ + sc_logits,
                            logits.size() * sizeof(float), cudaMemcpyDeviceToHost, fast_->cs),
                            "prefill logits copy", err) ||
                        !glmfast::cuda_ok(cudaStreamSynchronize(fast_->cs), "prefill logits", err)) {
                        if (err.empty()) err = "prefill output head failed";
                        return false;
                    }
                    if (!prefill_logits(p0 + t, logits, err)) return false;
                }
            }
            pos_ = p0 + T;
            for (Glm5Model* m = split_next_.get(); m != nullptr; m = m->split_next_.get()) m->pos_ = pos_;
            ++pf_->chunks;
            if (prefill_progress && c + 1 < nc && !prefill_progress(pos_ - pos_start, n)) {
                cancelled = true;
                break;
            }
        }
    } else {
        // THE PIPELINE: part k reads chunk c while part k - 1 reads chunk c + 1 - the residual rows cross each
        // boundary through the earlier part's two pinned host slots (slot c & 1 is free again once the next part has
        // copied chunk c - 2 in).  This thread runs the first part, one thread each the later ones.
        std::mutex mu;
        std::condition_variable cv;
        // produced[k]: chunks part k has put in its slots; consumed[k]: chunks part k + 1 has copied out of them
        std::vector<int> produced((size_t) np, 0), consumed((size_t) np, 0);
        bool stop = false;
        std::string stage_err;
        const size_t slot = (size_t) Tc * 4 * E;
        const auto fail = [&](std::string e) {
            std::lock_guard<std::mutex> lk(mu);
            if (stage_err.empty()) stage_err = std::move(e);
            stop = true;
            cv.notify_all();
        };
        // part k has read chunk c: its rows into its slot for part k + 1
        const auto hand_on = [&](int k, int c, int T) -> bool {
            Glm5Model* m = parts[(size_t) k];
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return consumed[(size_t) k] >= c - 1 || stop; });
                if (stop) return false;
            }
            cudaMemcpyAsync(m->pf_->hop_h + (size_t) (c & 1) * slot, m->pf_->R, (size_t) T * 4 * E * sizeof(float),
                            cudaMemcpyDeviceToHost, m->fast_->cs);
            std::string e;
            if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm prompt hop out", e)) {
                fail(e);
                return false;
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                produced[(size_t) k] = c + 1;
            }
            cv.notify_all();
            return true;
        };
        std::vector<std::thread> later;
        for (int k = 1; k < np; ++k)
            later.emplace_back([&, k] {
                Glm5Model* m = parts[(size_t) k];
                const Glm5Model* up = parts[(size_t) k - 1];
                cudaSetDevice(m->dev_);
                for (int c = 0; c < nc; ++c) {
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&] { return produced[(size_t) k - 1] > c || stop; });
                        if (stop) return;
                    }
                    const int T = chunk_len(c);
                    const int64_t p0 = pos_start + cs[(size_t) c];
                    cudaMemcpyAsync(m->pf_->R, up->pf_->hop_h + (size_t) (c & 1) * slot,
                                    (size_t) T * 4 * E * sizeof(float), cudaMemcpyHostToDevice, m->fast_->cs);
                    std::string e2;
                    if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm prompt hop in", e2)) {
                        fail(e2);
                        return;
                    }
                    {
                        std::lock_guard<std::mutex> lk(mu);
                        consumed[(size_t) k - 1] = c + 1;
                    }
                    cv.notify_all();
                    m->pos_ = p0;
                    toks_to(m, c);
                    std::vector<int32_t> nx;
                    if (m->mtp_il_ >= 0) next_of(c, nx);
                    if (!m->prefill_half(p0, T, e2, nx.empty() ? nullptr : nx.data())) {
                        fail(e2.empty() ? "glm prefill: part " + std::to_string(k) + " failed" : e2);
                        return;
                    }
                    if (k + 1 < np) {
                        if (!hand_on(k, c, T)) return;
                        continue;
                    }
                    if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm prompt pipeline", e2)) {
                        fail(e2);
                        return;
                    }
                    if (prefill_progress && c + 1 < nc &&
                        !prefill_progress(cs[(size_t) c + 1], n)) {
                        fail("cancelled");
                        return;
                    }
                }
            });
        cudaSetDevice(dev_);
        for (int c = 0; c < nc; ++c) {
            {
                std::lock_guard<std::mutex> lk(mu);
                if (stop) break;
            }
            const int T = chunk_len(c);
            const int64_t p0 = pos_start + cs[(size_t) c];
            if (!embed(c)) { fail(err); break; }
            std::string e1;
            if (!prefill_half(p0, T, e1)) {
                fail(e1.empty() ? "glm prefill: the head half failed" : e1);
                break;
            }
            if (!hand_on(0, c, T)) break;
            ++pf_->chunks;
        }
        for (auto& t : later) t.join();
        if (stop) {
            if (stage_err == "cancelled") {
                cancelled = true;
            } else {
                err = stage_err;
                return false;
            }
        }
        pos_ = pos_start + n;
        for (Glm5Model* m : parts) m->pos_ = pos_;
    }
    if (cancelled) {
        err = "cancelled";
        return false;
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm prefill compute", err) ||
            !glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->copy), "glm prefill copies", err)) {
            cudaSetDevice(dev_);
            return false;
        }
    }
    if (gb::launch_errors() > 0 || strata::kernels::glmf::launch_errors() > 0) {
        err = "glm prefill: " + std::to_string(gb::launch_errors() + strata::kernels::glmf::launch_errors()) +
              " kernel launch(es) failed (see stderr)";
        return false;
    }
    cudaSetDevice(dev_);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    pf_->ms += ms;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->pf_->tokens += n;
    static const bool verbose = getenv("STRATA_GLM_PREFILL_VERBOSE") != nullptr;
    if (verbose) {
        uint64_t sr = 0, sv = 0, sd = 0, rr = 0, rs = 0, ce = 0, cr = 0, pi = 0, ph = 0, pr = 0, ige = 0, igr = 0;
        double plan = 0, mc = 0;
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
            sr += m->pf_->staged_ram;
            sv += m->pf_->staged_vram;
            ce += m->pf_->cpu_experts;
            cr += m->pf_->cpu_rows;
            mc += m->pf_->ms_cpu;
            sd += m->pf_->staged_disk;
            rr += m->pf_->rows_resident;
            rs += m->pf_->rows_staged;
            plan += m->pf_->ms_plan;
            pi += m->pf_->pre_issued;
            ph += m->pf_->pre_hit;
            pr += m->pf_->pre_rows;
            ige += m->pf_->ig_experts;
            igr += m->pf_->ig_rows;
        }
        std::fprintf(stderr, "glm prefill: %lld tokens in %.1f ms (%.2f ms/token, %.0f tok/s) | staged experts: %llu vram, %llu "
                             "ram, %llu disk, %llu on the host (%llu rows, %.0f ms; %llu experts / %llu rows of them on the iGPU, %.4f ms a "
                             "row; split kappa %.2f, %.3f ms a CPU row), %llu prestaged "
                             "(%llu routed, %llu rows) | rows resident %.1f%% | plan %.1f ms (cumulative)\n",
                     (long long) n, ms, ms / (double) n, 1000.0 * (double) n / ms, (unsigned long long) sv,
                     (unsigned long long) sr, (unsigned long long) sd, (unsigned long long) ce, (unsigned long long) cr, mc,
                     (unsigned long long) ige, (unsigned long long) igr, pf_->row_ms_ig, pf_->kappa,
                     pf_->row_ms, (unsigned long long) pi, (unsigned long long) ph, (unsigned long long) pr,
                     100.0 * (double) rr / (double) std::max<uint64_t>(1, rr + rs), plan);
    }
    return true;
}

}  // namespace strata::core
