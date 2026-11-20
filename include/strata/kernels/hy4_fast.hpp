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

// ---- the batched prompt path (src/core/hy4_fast_path.cu's hy4_prompt_*): the kernels above a chunk's rows at once
// (row-major, row t = position p0 + t; the same arithmetic as their token versions), and the chunk's own pieces
void hc_mix_rows(const float* R, const float* fn, int hc, int dim, int n_mix, float eps, float* mixes, int T,
                 cudaStream_t s);
/// mixes: rows of mix_ld; R rows of hc * dim, x of dim, post of hc
void hc_pre_rows(const float* mixes, int mix_ld, const float* scale, const float* base, const float* R, int hc, int dim,
                 float hc_eps, float mag, const float* norm_w, float eps, float* x, float* post, int T, cudaStream_t s);
void hc_add_rows(float* R, const float* post, float* a, const float* a_norm, float eps, int hc, int dim, int T,
                 cudaStream_t s);
void rms_rows(const float* x, const float* w, float eps, int n, float* y, int T, cudaStream_t s,
              const float* add = nullptr);
/// pos: a device row of Pos3 per row; row t's cache row is (p0 + t - row_base) % ring
void mla_prep_rows(float* q, int n_head, const float* kv, const float* kv_norm, float eps, const Pos3* pos, int64_t p0,
                   float base, uint16_t* cache, int64_t row_base, int ring, int T, cudaStream_t s);
/// q's no-rope parts as FP16 [T][n_head][192]
void q_nope_f16(const float* q, int n_head, int T, uint16_t* q16, cudaStream_t s);
/// mla_attend for T rows (q_abs [T][n_head][512], q [T][n_head][256]): row t's keys cells[t][0, n_cells[t]) when
/// cells is non-null and n_cells[t] >= 0, else positions [max(0, p - window + 1), p] (window 0: [0, p]); position j
/// in cache row (j - row_base) % ring.  n_head a multiple of 16
void mla_attend_rows(const float* qabs, const float* q, const uint16_t* cache, int64_t row_base, int ring,
                     const int* cells, const int* n_cells, int cells_ld, int64_t p0, int window, const float* sinks,
                     float scale, int n_head, int T, float* ctx, cudaStream_t s);
/// o *= sigmoid(gate) (n values), o16 = FP16(o)
void gate_rows(float* o, const float* gate, int64_t n, uint16_t* o16, cudaStream_t s);
/// cache rows of positions [j0, j0 + n) from src (row (j - src_base) % src_ring) to dst (the same for dst)
void ring_copy(const uint16_t* src, int64_t src_base, int src_ring, uint16_t* dst, int64_t dst_base, int dst_ring,
               int64_t j0, int n, cudaStream_t s);
/// The groups that close in positions [pt0, pt0 + tn) pooled (their token keys from tkb's rows, the earlier ones from
/// tk4's slots; pos[i] is position pt0 - 3 + i), then the sub-batch's last 4 keys into tk4
void idx_pool_rows(const float* tkb, int64_t pt0, int tn, float* tk4, const float* wv, const float* wg,
                   const float* ape, const float* norm, float eps, const Pos3* pos, float base, float* pooled,
                   cudaStream_t s);
void idx_query_rows(float* iq, int heads, float* w, const Pos3* pos, float base, int T, cudaStream_t s);
/// idx_select per row: n_cells[t] = -1 (dense) while p + 1 <= dense, else the top groups of score[t] and the own group
void idx_select_rows(const float* score, int score_ld, int64_t p0, int dense, int top, int* cells, int cells_ld,
                     int* n_cells, int T, cudaStream_t s);
/// eng_rows over T rows of n_heads heads (lnorm per head)
void eng_rows_rows(const float* rows, const float* lnorm, int n_heads, int d, float row_eps, float* e, int T,
                   cudaStream_t s);
/// eng_add per row (kv rows of (hc + 1) dim, R rows of hc dim); skip[t] != 0: row t adds nothing (skip may be null)
void eng_add_rows(const float* kv, float* R, const float* gate, const float* hc_scale, const float* pre_gamma, int hc,
                  int dim, float eps, bool vfirst, const unsigned char* skip, int T, cudaStream_t s);
/// y[t][i] += b[i]
void add_bias_rows(float* y, const float* b, int n, int T, cudaStream_t s);
/// swiglu over n values (64-bit count)
void swiglu_n(const float* g, const float* u, int64_t n, float* h, cudaStream_t s);
/// launches that failed (all hy4 kernels)
int launch_errors();

}  // namespace strata::kernels::hy4
