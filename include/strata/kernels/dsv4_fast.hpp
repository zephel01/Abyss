// include/strata/kernels/dsv4_fast.hpp - the deepseek4 (DeepSeek V4 Flash) decode kernels: one token, the attention
// mixer and the hyper-connection head.  The rest of a deepseek4 layer (the mHC read/write halves, the GEMVs, the MoE
// tiers) rides the glm5-next fast path's kernels (glm_fast.hpp) - the two architectures share those pieces exactly.
//
// The arithmetic follows llama.cpp's src/models/deepseek4.cpp (docs/deepseek4/SPEC.md):
//   * attention: one shared 512-wide K=V row per token (RoPE on its last 64 dims, NORM/interleaved pairs), queried by
//     64 heads; keys = the raw sliding window (128 tokens) plus, per layer type, the compressed rows (HCA ratio 128:
//     every visible row; CSA ratio 4: the lightning indexer's top-k rows); per-head sinks; the output de-roped;
//   * compressors: a per-channel softmax over the block's tokens of (gate + ape) weights on the kv projection, then
//     an RMS norm and RoPE at the block's first position; CSA and the indexer (LID) overlap two windows of 4;
//   * the Hadamard rotations llama.cpp applies to quantized caches and to the indexer keys are orthonormal and applied
//     to both sides of every dot product, so they are omitted here (the caches are FP16, like llama.cpp's default).
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::dsv4 {

/// The RoPE parameters of one layer kind (ggml's rope_yarn, mode NORM): theta_scale = base^(-2/n_rot), the YaRN
/// correction dims and factors.  ext_factor 0 is plain RoPE.
struct Rope {
    float theta_scale = 1.0f;
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    float attn_factor = 1.0f;
    float corr_lo = 0.0f, corr_hi = 0.0f;
};
/// Host: the parameters llama.cpp's deepseek4 graph uses (attn_factor cancels YaRN's magnitude scale).
Rope make_rope(int n_rot, float base, float freq_scale, float ext_factor, int n_ctx_orig, float beta_fast,
               float beta_slow);

/// A compressor ring's spare rows: a verify window writes its columns' rows before it knows which are kept, and a
/// rejected column's row must not land on a row a later block end still reads (with exactly `state` rows, a 3-column
/// window's rejected third column overwrote the oldest row of the next block's overlap - V4 0731's answers with
/// drafts parted from the token-by-token ones).  Rows are indexed pos % (state + kCompSlack); kVerifyMax - 2 is
/// enough (the first column is always kept), 8 covers it.
constexpr int kCompSlack = 8;
__host__ __device__ inline int ring_rows(int state) { return state + kCompSlack; }

/// The compressor ring and its cache for one layer and one compressor.
struct Comp {
    float* ring_kv = nullptr;        // [ring_rows(state)][2*D or D] f32 (state = 2*ratio for overlap, ratio otherwise)
    float* ring_sc = nullptr;        // same shape: gate + ape
    uint16_t* cache = nullptr;       // [rows][D] FP16 compressed rows
    const float* ape = nullptr;      // [ratio][coff*D]
    const float* norm = nullptr;     // [D]
    int D = 512;                     // compressed row width
    int ratio = 4;
    int overlap = 0;                 // 1: two windows of `ratio` (CSA, LID), 0: one (HCA)
};

/// After the first projections of one token at position p:
///   qr = rms(qr_raw) * q_a_norm  -> qr (f32) and qr_q (q8_1);
///   kv = rope(rms(kv_raw) * kv_norm)  -> the raw ring row p % ring (FP16);
///   each compressor: its ring row p % ring_rows(state) <- (kv proj, gate proj + ape[p % ratio]).
struct PrepArgs {
    const float* qr_raw = nullptr; const float* q_a_norm = nullptr; float* qr = nullptr; void* qr_q = nullptr;
    int q_lora = 1024;
    const float* kv_raw = nullptr; const float* kv_norm = nullptr; uint16_t* raw = nullptr; int ring = 128;
    int head_dim = 512, n_rot = 64;
    Rope rope;
    int n_comp = 0;                  // 0, 1 (HCA) or 2 (CSA: attention + indexer)
    Comp comp[2];
    const float* comp_kv[2] = {nullptr, nullptr};   // this token's compressor projections
    const float* comp_sc[2] = {nullptr, nullptr};
    Rope comp_rope;                  // the compressed rows' RoPE (base 160000, YaRN)
    int p = 0;
    float eps = 1e-6f;
};
void prep(const PrepArgs& a, cudaStream_t s);
/// When (p + 1) % ratio == 0: the block b = p / ratio of each compressor -> its cache row b.
void compress(const PrepArgs& a, cudaStream_t s);

/// q (n_head x head_dim): per-head rms (no weight), RoPE on the last n_rot dims at p.  iq (idx_heads x idx_dim, may be
/// null): RoPE on its last n_rot dims at p with comp_rope.
void q_prep(float* q, int n_head, int head_dim, int n_rot, const Rope& rope, float* iq, int idx_heads, int idx_dim,
            const Rope& comp_rope, int p, float eps, cudaStream_t s);

/// The lightning indexer: score[j] = sum_h relu(iq_h . K[j]) * w_h over the n_vis compressed rows, then the top
/// min(top_k, n_vis) of them (score descending, ties by the lower row) -> sel.
void indexer(const float* iq, const uint16_t* K, const float* w, int idx_heads, int idx_dim, int n_vis, int top_k,
             float* score, int* sel, cudaStream_t s);
/// indexer's two halves (deepseek41 masks the scores in between): the scores alone, then the top-k of a score row.
void indexer_scores(const float* iq, const uint16_t* K, const float* w, int idx_heads, int n_vis, float* score,
                    cudaStream_t s);
void indexer_select(const float* score, int n_vis, int top_k, int* sel, cudaStream_t s);

/// Attention of one token at position p over the raw window (rows p-n_raw+1 .. p of the ring) and n_comp compressed
/// rows (comp_sel[i] when given, else rows 0..n_comp-1), V = K, scale 1/sqrt(head_dim), the per-head sink in the
/// softmax; the output de-roped on its last n_rot dims at p -> out (n_head x head_dim f32).
void attend(const float* q, const uint16_t* raw, int ring, int n_raw, const uint16_t* comp, const int* comp_sel,
            int n_comp, const float* sinks, int n_head, int head_dim, int n_rot, const Rope& rope, int p, float* out,
            cudaStream_t s, int lane = 0, void* oq = nullptr);   // lane (0..3): its own partials (concurrent
                                                                  // streams); oq: out's q8_1 blocks too (wo_a's input)

/// The head's prologue: m = fn . rms(flat R) (4 mixes), pre = sigmoid(m * scale + base) + hc_eps,
/// h = sum_s pre_s R_s, x = rms(h) * norm_w -> x (f32) and xq (q8_1).  fn: BF16 [4][4 * n_embd].
void head_prep(const float* R, const uint16_t* fn, const float* scale, const float* base, float hc_eps,
               const float* norm_w, float eps, int n_embd, float* x, void* xq, cudaStream_t s);

// ---------------------------------------------------------------- the prompt path: T tokens at positions p0 ..
//
// The window keys of a sub-batch live in a linear buffer `keys` of (n_hist + T) FP16 rows: rows [0, n_hist) are the
// positions p0 - n_hist .. p0 - 1 (copied from the ring first: win_load), rows n_hist + t this batch's; win_store
// writes the last window back into the ring afterwards, so the decode continues from the same state.

/// keys[0, n_hist) <- the ring rows of positions p0 - n_hist .. p0 - 1.
void win_load(const uint16_t* ring, int n_ring, uint16_t* keys, int n_hist, int p0, int D, cudaStream_t s);
/// The ring rows of the last min(n_ring, n_hist + T) positions <- keys.
void win_store(uint16_t* ring, int n_ring, const uint16_t* keys, int n_hist, int p0, int T, int D, cudaStream_t s);

/// Per token t: qr = rms(qr_raw) * q_a_norm (f32 and f16), keys[n_hist + t] = rope(rms(kv_raw) * kv_norm) at p0 + t.
struct BatchPrepArgs {
    const float* qr_raw = nullptr; const float* q_a_norm = nullptr; float* qr = nullptr; uint16_t* qr16 = nullptr;
    int q_lora = 1024;
    const float* kv_raw = nullptr; const float* kv_norm = nullptr; uint16_t* keys = nullptr; int n_hist = 0;
    int head_dim = 512, n_rot = 64;
    Rope rope;
    int p0 = 0, T = 0;
    float eps = 1e-6f;
};
void prep_batch(const BatchPrepArgs& a, cudaStream_t s);

/// One compressor over a batch: kv / sc are this batch's projections ([T][W], sc WITHOUT the ape).  Every block that
/// ends inside [p0, p0 + T) is compressed into the cache (its earlier rows come from the ring, which holds the
/// positions before p0), then the ring takes the batch's last rows.
struct BatchComp {
    Comp c;
    const float* kv = nullptr;
    const float* sc = nullptr;
};
void compress_batch(const BatchComp* comps, int n_comp, int p0, int T, int n_rot, const Rope& comp_rope, float eps,
                    cudaStream_t s);

/// q (T x n_head x head_dim): per-head rms + RoPE at p0 + t; iq (T x idx_heads x idx_dim, may be null): RoPE.
void q_prep_batch(float* q, int n_head, int head_dim, int n_rot, const Rope& rope, float* iq, int idx_heads,
                  int idx_dim, const Rope& comp_rope, int p0, int T, float eps, cudaStream_t s);

/// The lightning indexer per token t (its visible rows: (p0 + t + 1) / ratio): scores into score[t][ld] (the
/// visible rows; the tensor-core kernels leave the rest unwritten), its top min(top_k, visible) rows into
/// sel[t][top_k].  iq is scratch afterwards: the HIP WMMA path converts it to FP16 hi / lo in place.
void indexer_batch(float* iq, const uint16_t* K, const float* w, int idx_heads, int ratio, int p0, int T, int top_k,
                   float* score, int ld, int* sel, cudaStream_t s);

/// indexer_batch's two halves (deepseek41 masks the scores in between): the scores of every token's visible rows,
/// then each token's top min(top_k, visible) into sel[t * sel_ld ..].  idx_heads <= 64; on HIP gfx11 / gfx12 the
/// WMMA kernel for 32 heads (iq converted in place: the caller's last read of these rows), else the plain kernel.
void indexer_batch_scores(float* iq, const uint16_t* K, const float* w, int idx_heads, int ratio, int p0, int T,
                          float* score, int ld, cudaStream_t s);
void indexer_batch_select(const float* score, int ld, int ratio, int p0, int T, int top_k, int* sel, int sel_ld,
                          cudaStream_t s);

/// Attention per token t at p = p0 + t over its window (keys rows of positions max(0, p - n_swa + 1) .. p) and its
/// compressed rows (ratio 128: every visible one; ratio 4: sel[t][0 .. min(top_k, visible)); deepseek41, ratio
/// passed as -1 / -2: every visible row while they fit top_k, else sel[t][0 .. top_k)), sinks, de-roped.  sel_ld:
/// sel's row stride in ints (0: top_k).
void attend_batch(const float* q, const uint16_t* keys, int n_hist, int n_swa, const uint16_t* comp, const int* sel,
                  int top_k, int ratio, const float* sinks, int n_head, int head_dim, int n_rot, const Rope& rope, int p0,
                  int T, float* out, cudaStream_t s, int sel_ld = 0);

/// x (n floats) -> FP16.
void to_f16(const float* x, uint16_t* y, int64_t n, cudaStream_t s);

// ---------------------------------------------------------------- DSpark, the block drafter (src/core/dspark.cu)

/// Rows t < n: rope(rms(kv_raw[t]) * kv_norm) at position p0 + t -> ring row (p0 + t) % n_ring (FP16) - prep's kv.
void kv_rows(const float* kv_raw, int n, const float* kv_norm, int head_dim, int n_rot, const Rope& rope, int p0,
             uint16_t* ring, int n_ring, float eps, cudaStream_t s);
/// out row r (os floats apart) = the mean of the 4 streams of R row r (rs floats apart), n rows.
void stream_mean(const float* R, int n, size_t rs, int n_embd, float* out, size_t os, cudaStream_t s);
/// The drafter's router over nr rows of logits: sc = sqrt(softplus), the top k of sc + bias (ties: the lower id) ->
/// ids[r][k]; the plan (plan_ptr / plan_w, [nr][8]) keeps the experts with a VRAM slot (tab[e] != 0), weights
/// sc / sum(kept sc) * w_scale.  False: n_expert > 512 or k > 8.
bool dspark_route(const float* logits, const float* bias, int n_expert, int k, float w_scale,
                  const unsigned long long* tab, int nr, unsigned long long* plan_ptr, float* plan_w, int* ids,
                  cudaStream_t s);
/// head_prep that also keeps h = sum_s pre_s R_s (before the norm) -> h.
void dspark_head(const float* R, const uint16_t* fn, const float* scale, const float* base, float hc_eps,
                 const float* norm_w, float eps, int n_embd, float* h, float* x, void* xq, cudaStream_t s);
/// A Markov row (rank 256): m = w1[*prev] (F16), out[v] = logits[v] + w2[v] . m (w2: Q8_0 rows), and
/// *conf = sigmoid(cp . [h; m]) (cp: n_embd + 256 floats).
void dspark_markov(const float* logits, const void* w2, const uint16_t* w1, int n_vocab, const int* prev,
                   const float* cp, const float* h, int n_embd, float* out, float* conf, cudaStream_t s);
/// dspark_markov for DeepSeek V4.1's drafter: w2 F16 rows (V x 256), *conf = sigmoid(cp . [h; m] + *cb).
void dspark_markov_f16(const float* logits, const uint16_t* w2, const uint16_t* w1, int n_vocab, const int* prev,
                       const float* cp, const float* cb, const float* h, int n_embd, float* out, float* conf,
                       cudaStream_t s);

}  // namespace strata::kernels::dsv4
