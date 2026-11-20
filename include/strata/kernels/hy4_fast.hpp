// include/strata/kernels/hy4_fast.hpp - Tencent Hy4-flash's (GGUF arch hy_v4_flash) one-token decode kernels.
//
// The pieces the shared GLM / DeepSeek kernels do not cover (src/core/hy4_fast_path.cu runs them in order):
//   * the "independent" hyper-connections: 2 x hc mixes of the flat streams (no stream-mixing matrix), sigmoid gates
//   * the gated MLA attention: a shared 512 latent + a 64-dim rope key per position (FP16 cache rows of 576), per-head
//     absorbed k_b (Q8_0 rows of 192) / v_b (Q6_K rows of 512) read as the GGUF stores them, a learned sink logit per
//     head in the softmax, the output times sigmoid(attn_gate x) per head
//   * the compressed indexer's group pooling (fp32 value / gate projections over 4 token keys, a per-slot bias, a
//     softmax over the 4 slots) and its selection (the best complete groups before the query's own, then its own)
//   * the Engram add (the hashed n-gram rows already decoded to floats by the host)
// RoPE is the interleaved-pair form (pair j = dims 2j, 2j+1, frequency base^(-2j/64)) on the trailing 64 dims.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::hy4 {

constexpr int kLat = 512;           // the latent
constexpr int kRot = 64;            // the rope key / the rope part of a q head
constexpr int kRow = kLat + kRot;   // a cache row (FP16)
constexpr int kNope = 192;
constexpr int kV = 256;
constexpr int kMaxCells = 2048 + 4; // the most keys a query attends (the window 512, dense <= 2048, indexed 2044 + 4)

/// A rotary position: M-RoPE's three planes (time, height, width).  Text has t = h = w (an int64_t converts); an image
/// cell takes its own h and w.  Pair j < 30 rotates by the plane j % 3 (0 t, 1 h, 2 w), pairs 30 and 31 by t - the
/// config's mrope sections 12 / 10 / 10 interleaved.
struct Pos3 {
    int64_t t, h, w;
    Pos3(int64_t p) : t(p), h(p), w(p) {}   // NOLINT: implicit on purpose (text positions)
    Pos3(int64_t t_, int64_t h_, int64_t w_) : t(t_), h(h_), w(w_) {}
};

/// mixes[m] = (fn_m . flat(R)) * rsqrt(mean(flat(R)^2) + eps), m < n_mix; fn is F32 [n_mix][hc * dim]
void hc_mix(const float* R, const float* fn, int hc, int dim, int n_mix, float eps, float* mixes, cudaStream_t s);
/// pre_c = sigmoid(mixes[c] * scale[0] + base[c]) + hc_eps; x = sum_c pre_c R_c; x = rms(x) * norm_w (norm_w null: no
/// norm); post (non-null): post_c = mag * sigmoid(mixes[hc + c] * scale[1] + base[hc + c]) + hc_eps
void hc_pre(const float* mixes, const float* scale, const float* base, const float* R, int hc, int dim, float hc_eps,
            float mag, const float* norm_w, float eps, float* x, float* post, cudaStream_t s);
/// R_c += post_c * a (a: dim floats; a_norm non-null: a is rms-normed by it first, in place)
void hc_add(float* R, const float* post, float* a, const float* a_norm, float eps, int hc, int dim, cudaStream_t s);
/// y = rms(x + add) * w (add, w may be null; y may be x)
void rms(const float* x, const float* w, float eps, int n, float* y, cudaStream_t s, const float* add = nullptr);
/// The attention's prep for position pos: the latent rms(kv[0..512)) * kv_norm and the roped k_rot kv[512..576) into
/// the cache row (FP16), q's rope dims (each head's last 64 of 256) roped in place
void mla_prep(float* q, int n_head, const float* kv, const float* kv_norm, float eps, Pos3 pos, float base,
              uint16_t* cache_row, cudaStream_t s);
/// q_abs[h][r] = k_b_h[r] . q_nope_h (k_b: Q8_0 rows of 192, row h * 512 + r; f32: float rows - a GGUF that keeps
/// k_b in another type, decoded at load), qabs [n_head][512]
void mla_absorb(const float* q, const void* k_b, int n_head, float* qabs, cudaStream_t s, bool f32 = false);
/// Per head: scores over the keys (scale * (q_abs . lat + q_rot . k_rot)), the softmax with the head's sink in the
/// denominator, ctx[h] = sum p lat.  The keys: cells (n_cells positions) when non-null, else positions [lo, hi];
/// position j lives in cache row j % ring.
void mla_attend(const float* qabs, const float* q, const uint16_t* cache, int ring, const int* cells, int n_cells,
                int64_t lo, int64_t hi, const float* sinks, float scale, int n_head, float* ctx, cudaStream_t s);
/// o[h][v] = (v_b_h[v] . ctx_h) * sigmoid(gate[h][v]) (v_b: Q6_K rows of 512, row h * 256 + v; f32: float rows)
void mla_out(const float* ctx, const void* v_b, const float* gate, int n_head, float* o, cudaStream_t s,
             bool f32 = false);
/// The indexer's token key into its slot: tk = rms(raw) * norm (128)
/// The group pool once 4 token keys are in (tk4: 4 x 128 rows): vals = Wv tk, gates = Wg tk + ape[slot], a softmax
/// over the slots per channel, pooled = rms(sum) * norm, its last 64 dims roped at pos0 (the group's first token)
void idx_pool(const float* tk4, const float* wv, const float* wg, const float* ape, const float* norm, float eps,
              Pos3 pos0, float base, float* pooled, cudaStream_t s);
/// The query side: iq [heads][128], last 64 dims roped at pos; w *= heads^-0.5 * 128^-0.5
void idx_query(float* iq, int heads, float* w, Pos3 pos, float base, cudaStream_t s);
/// score[g] = sum_h w_h relu(iq_h . pooled_g), g < n_groups
void idx_score(const float* iq, const float* w, int heads, const float* pooled, int n_groups, float* score,
               cudaStream_t s);
/// cells: the top min(top, n_vis) groups of score[0, n_vis) (rank order, ties by the lower index) as 4 cells each,
/// then the own group's cells own * 4 .. pos; *n_out (device) the count
void idx_select(const float* score, int n_vis, int top, int64_t pos, int* cells, int* n_out, cudaStream_t s);
/// Engram: e[h][d] = rms(rows[h]) * lnorm[h] (n_heads rows of 96, eps row_eps) -> e (n_heads * 96)
void eng_rows(const float* rows, const float* lnorm, int n_heads, int d, float row_eps, float* e, cudaStream_t s);
/// Engram add: kv = [key_0 .. key_{hc-1}, value] (vfirst: [value, keys]) of dim each, bias already added;
/// R_c += sigmoid(ssqrt(<rms(key_c), rms(R_c)> . gate_c / sqrt(dim))) * main + scale_c * main,
/// main = pre_gamma * rms(value)
void eng_add(const float* kv, float* R, const float* gate, const float* hc_scale, const float* pre_gamma, int hc,
             int dim, float eps, bool vfirst, cudaStream_t s);
/// SwiGLU without a clamp into floats: h = silu(g) * u
void swiglu(const float* g, const float* u, int n, float* h, cudaStream_t s);
/// FP8 E4M3 codes -> floats (n values)
void e4m3_to_f32(const uint8_t* src, float* dst, int n, cudaStream_t s);

}  // namespace strata::kernels::hy4
