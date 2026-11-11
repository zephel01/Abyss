// include/strata/kernels/glm_fast.hpp - the glm5-next FAST decode kernels (one token, fused).
//
// The correctness-first kernels (glm_hc / glm_kda / glm_dsa / router_sigmoid / native_expert_grouped)
// stay the parity reference; this family is what the decode path runs.  Same arithmetic, fused so
// a layer is ~11 launches instead of ~60, every reduction parallel, and every per-token plan read
// from DEVICE memory (the grouped expert path used to read its plan over PCIe from host-mapped
// memory - every block, every layer - which was most of its 0.4 ms per call).
//
// Weights: ggml quant types are dotted against a q8_1 activation (llama.cpp's vecdotq, the same
// expressions as iq_kernels.cu / native_mmvq.cu); the pack's big BF16 rows stay BF16 on the device
// (dense.bin stores them as BF16, so keeping them is lossless and halves their bytes); small
// parameters are F32.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::glmf {

constexpr int kTypeF32 = 0;
constexpr int kTypeBF16 = 30;

/// y[r] = alpha * (W_r . x) + (bias ? bias[r] : 0) for r in [0, n_out).  Quantized types read
/// `xq` (q8_1, n_in / 32 blocks); F32 / BF16 weights read `xf`.
struct MvJob {
    const void* w = nullptr;
    const void* xq = nullptr;
    const float* xf = nullptr;
    float* y = nullptr;
    const float* bias = nullptr;
    float alpha = 1.0f;
    int type = 0;
    int n_in = 0, n_out = 0;
    int xs = 0, ys = 0;   // mvn: column strides of the activations (q8_1 blocks, or floats for BF16 / F32) and the
                          // outputs (floats); 0 = contiguous columns (n_in / 32 blocks or n_in floats, n_out floats)
};
constexpr int kMaxMvJobs = 8;
/// One launch for up to kMaxMvJobs independent GEMVs (they share nothing but the launch).
bool mv(const MvJob* jobs, int n, cudaStream_t s);
/// The per-GPU choices mv() makes (glm_fast.cu's mv_tune) - for abyss_gpu_tune: the current device's Q6_K split (the
/// table's or STRATA_GLM_MV_Q6K_SPLIT's), and a forced one for the calls that follow (1 / 0; -1 = back to the choice)
bool mv_q6k_split_choice();
/// the warps a row mv() (and mvn_mv, a column) gives a call whose jobs of `type` add up to n_out rows: 1, 2 or 4 for
/// the typed kernels (Q6_K, Q8_0, Q4_K, Q5_K), 0 when that type goes to the generic one-warp kernel (other types, or
/// STRATA_GLM_MV_GENERIC) - for fused kernels that must sum a row as mv() does
int mv_split_rule(int type, int n_out);
/// (test) the route's 8-warp top-k against warp_topk: n blocks of 512 scores (sel_d), E_d[b] of them live, k = 1..16;
/// *bad_d counts the differing ids
int route_topk_check(const float* sel_d, const int* E_d, int n, int* bad_d, cudaStream_t s);
void mv_force_q6k_split(int split4);
void mv_force_split(int split);   // gpu_tune: every typed GEMV launch at 1 / 2 / 4 warps a row (other: the engine's rule)
/// mv() for nc activation columns at once (1..8 a launch, wider calls in runs) - a verify window's tokens, the DSpark
/// drafter's block: the weight rows are read once. Column c
/// of a job reads xq + c * n_in / 32 (q8_1 blocks) or xf + c * n_in (BF16 / F32) and writes y + c * n_out.
bool mvn(const MvJob* jobs, int n, int nc, cudaStream_t s);
bool mvn_supported(int type);
/// mvn with mv()'s sums: every column gets the same bits a mv() call of the same jobs gives it (the same split of a
/// row over warps that mv() picks for the call's per-type block count; one warp a row otherwise) - GLM-5.3's verify
/// window (STRATA_GLM_WINDOW), where the token path's bits are the reference.  Types mvn has no kernel for, and HIP:
/// a mv() a column.
bool mvn_mv(const MvJob* jobs, int n, int nc, cudaStream_t s);
/// Whether mv() serves this weight type.
bool mv_supported(int type);
/// Whether moe_gate_up() / moe_down() serve routed experts of this type.
bool moe_supported(int type);
/// Kernel launches of this family that failed since the process started (each is also printed); the forward
/// that sees the count grow reports an error instead of computing on.
int launch_errors();
// Optional helper launches report errors to their own service thread, without poisoning the primary.
void helper_launch_begin();
cudaError_t helper_launch_end();
/// Bytes of one row of `type` with n_in values (0 when unsupported).
size_t row_bytes(int type, int64_t n_in);

/// q8_1 of n floats (n % 32 == 0).
void quantize_q8_1(const float* x, void* xq, int n, cudaStream_t s);

/// The mHC read half, fused with the previous block's write half:
///   block_out != nullptr: R_new = post_in (x) block_out + comb_in . R_old  (glm_hc_post), then
///   always:               pre/post/comb from R_new (glm_hc_pre incl. the Sinkhorn), mixed =
///                         sum_s pre_s R_new_s, x = rms_norm(mixed) * norm_w, xq = q8_1(x).
/// block_out == nullptr reads R_old as the residual (R_new is ignored).  post/comb may alias
/// post_in/comb_in (they are overwritten only after every block has read them).
struct HcArgs {
    const float* block_out = nullptr;
    const float* R_old = nullptr;
    float* R_new = nullptr;
    const float* post_in = nullptr;
    const float* comb_in = nullptr;
    float* pre = nullptr;
    float* post = nullptr;
    float* comb = nullptr;
    const uint16_t* w_fn = nullptr;   // BF16 [hc_mix_dim][hc * n_embd]
    const void* w_fn_q8 = nullptr;    // or Q8_0 (the GGUF's own blocks; STRATA_GLM_DENSE_Q8) - set: w_fn is not read
    const float* w_scale = nullptr;   // 3
    const float* w_base = nullptr;    // hc_mix_dim
    const float* norm_w = nullptr;    // n_embd
    float norm_eps = 1e-5f, hc_eps = 1e-6f;
    int iters = 20, n_embd = 4096;
    float* x = nullptr;               // n_embd f32
    void* xq = nullptr;               // n_embd / 32 q8_1 blocks
    float* part = nullptr;            // scratch: (hc*n_embd / 512) * 36 floats (25 partials + 10 Gram + a spare row)
    unsigned int* counter = nullptr;  // zero-initialised; the kernel leaves it zero
    int gram = 0;                     // hc3: 1 = the mixed vector's norm from the streams' Gram partials (no second
                                      // grid barrier; set by hc() from STRATA_GLM_HC_GRAM, default on)
    const float* pre_use = nullptr;   // deepseek41: the collapse weights (the PREVIOUS sublayer's pre, 4 floats); null =
                                      // this sublayer's own pre.  pre still receives the new one (another buffer)
    const unsigned int* cpu_seq = nullptr;   // the CPU LANE's answer for the last route (MoeDev::cpu_seq[0], 0 = none):
    const void* cpu_ans = nullptr;           // added into block_out first (MoeDev::cpu_ans) - moe_cpu_wait, fused
};
void hc(const HcArgs& a, cudaStream_t s);
/// hc() runs hc3 for this width (the kernel that can take HcArgs::cpu_seq itself; the others wait in moe_cpu_wait's
/// kernel first)
bool hc_fuses_cpu(int n_embd);
/// The write half alone (the split boundary / the head): R_new = post (x) block_out + comb . R_old.
void hc_post(const float* block_out, const float* R_old, const float* post, const float* comb, int n_embd,
             float* R_new, cudaStream_t s);
/// The head's prologue: mean over the 4 streams of R, rms_norm * norm_w -> x (f32) and xq (q8_1).
void head_prep(const float* R, const float* norm_w, float eps, int n_embd, float* x, void* xq, cudaStream_t s);
/// R[s][e] = emb[e] for the 4 streams.
void embed_streams(const float* emb, float* R, int n_embd, cudaStream_t s);
// rows of x (n floats each, ld apart): x -= s Q (Q^T x), Q [n][r] row-major on the device, r <= 4 (STRATA_GLM_ABL /
// --uncensor); a no-op while the device flag `on` (if given) is 0 - the per-request switch, read on the device so a
// captured graph follows it too
void abl_project(float* x, int rows, int ld, int n, const float* q, int r, float s, cudaStream_t st,
                 const int* on = nullptr);
// rows x ([rows][n], ld apart) into cap [rows][n] (the residual's capture for --abl-build): the mean of `streams`
// consecutive n-float streams of each row
void abl_capture(const float* x, int rows, int ld, int n, int streams, float* cap, cudaStream_t st);
// out[t] = the token_embd row of toks[t] (ids read on the device; the table may be host-mapped), t < T; Q8_0 and
// Q5_K rows (false: another type)
bool embed_rows_q(int ggml_type, const void* table, const int* toks, int T, int n_embd, float* out, cudaStream_t s);
bool embed_rows_q_supported(int ggml_type, int n_embd);

/// KDA, after the projections: the three causal convs (+SiLU, + the history slide), the q/k l2
/// norms, and the two gate GEMVs (f_b with dt bias and the lower-bound squash -> g1; g_b -> g2).
struct KdaPrepArgs {
    const float* proj[3] = {nullptr, nullptr, nullptr};     // q, k, v projections (d_inner)
    const float* conv_w[3] = {nullptr, nullptr, nullptr};   // [d_inner][d_conv] f32
    float* conv_state = nullptr;                            // 3 x d_inner x (d_conv-1), this layer
    float* out[3] = {nullptr, nullptr, nullptr};
    const float* fa = nullptr;                              // head_dim (f_a output)
    const float* ga = nullptr;                              // head_dim (g_a output)
    const uint16_t* f_b = nullptr;                          // BF16 [d_inner][head_dim]
    const uint16_t* g_b = nullptr;
    const void* f_b_q8 = nullptr;                           // or Q8_0 (STRATA_GLM_DENSE_Q8): set, f_b / g_b are not
    const void* g_b_q8 = nullptr;                           // read
    const float* dt_bias = nullptr;                         // d_inner
    const float* ssm_a = nullptr;                           // n_head
    float lower_bound = -5.0f;
    float* g1 = nullptr;                                    // d_inner
    float* g2 = nullptr;                                    // d_inner
    int n_head = 64, head_dim = 128, d_conv = 4;
};
void kda_prep(const KdaPrepArgs& a, cudaStream_t s);
/// The gated delta recurrence (one token) + the output gate (rms over head_dim * norm_w *
/// sigmoid(g2)) + q8_1 of the gated output (the attn_output GEMV's input).
void kda_rec(const float* q, const float* k, const float* v, const float* g1, const float* beta_raw,
             float* state, const float* g2, const float* norm_w, float eps, int n_head, int head_dim,
             void* out_q8_1, cudaStream_t s, const float* state_in = nullptr);
/// (state_in != nullptr and != state: the recurrence reads the state from state_in and writes the new one to state -
/// the same sums, out of place; the verify window keeps the state after its column 0 that way without a copy)

/// DSA, after the first projections: q_a norm (-> qr f32 + q8_1), kv_a norm into the latent cache
/// at position p, the indexer key's layer norm into its cache, the compressor gate into its cache,
/// and the completed pool's pooled key when (p + 1) % kpool == 0.
struct DsaPrepArgs {
    const float* qr_raw = nullptr; const float* q_a_norm = nullptr; float* qr = nullptr; void* qr_q = nullptr;
    int q_lora = 1536;
    const float* kv_raw = nullptr; const float* kv_norm = nullptr; uint16_t* lat = nullptr; int kv_lora = 512;
    int lat_q8 = 0;   // the latent cache holds INT8 records (lat8_rec_bytes each) instead of FP16 rows
    const float* ik_raw = nullptr; const float* k_norm_w = nullptr; const float* k_norm_b = nullptr;
    float* ik_cache = nullptr;
    const float* ig_raw = nullptr; float* ig_cache = nullptr;
    const float* ape = nullptr; float* pooled = nullptr;
    int idx_key = 128, kpool = 4;
    int ring = 1 << 30;   // the ik / ig caches hold positions modulo ring (a multiple of kpool)
    int p = 0;
    float eps = 1e-5f;
};
void dsa_prep(const DsaPrepArgs& a, cudaStream_t s);
/// score[p] = sum_h relu(iq_h . pooled_p) * iw_h for the n_pools completed pools.
void dsa_score(const float* iq, const float* pooled, const float* iw, int key_dim, int idx_heads, int n_pools,
               float* score, cudaStream_t s);
/// The stable top-k of the pools (ties by lower index) expanded to cells, then the tail cells.
void dsa_select(const float* score, int n_vis, int kpool, int top_pools, int n_sel, int pos, int* cells,
                cudaStream_t s);
/// Absorbed MLA for one token, per head: q_abs = wk_b_h . q_h, scores over the selected latents,
/// softmax, ctx, out_h = wv_b_h . ctx; writes the q8_1 of the n_head * v_head output.
/// lat_q8: the cache holds INT8 latent records (see lat8_rec_bytes), else FP16 rows of kv_lora.
/// wk_b_q8 / wv_b_q8 (both or neither): the absorbed projections as Q8_0 blocks (the GGUF's own, STRATA_GLM_DENSE_Q8)
/// in place of the BF16 wk_b / wv_b - dotted against the F32 activations.
void mla(const float* q, const uint16_t* wk_b, const uint16_t* wv_b, const uint16_t* lat, const int* cells, int n_sel,
         int n_head, int qk_nope, int kv_lora, int v_head, void* out_q8_1, cudaStream_t s, bool lat_q8 = false,
         const void* wk_b_q8 = nullptr, const void* wv_b_q8 = nullptr);
/// mla() for nc (2..8) verify-window columns: the absorbed k_b / v_b GEMVs over all columns in one launch each (each
/// weight read once; every column's sums those of mla()'s), the attention a column at a time.  Column c: q at
/// q + c * q_cs, its cells at cells + c * cells_cs, n_sel[c] (host), its output at out + c * out_cs q8_1 blocks.
/// false: not run (the caller runs mla() a column)
bool mla_cols(int nc, const float* q, size_t q_cs, const uint16_t* wk_b, const uint16_t* wv_b, const uint16_t* lat,
              const int* cells, size_t cells_cs, const int* n_sel, int n_head, int qk_nope, int kv_lora, int v_head,
              void* out_q8_1, size_t out_cs, cudaStream_t s, bool lat_q8, const void* wk_b_q8, const void* wv_b_q8);
/// Q8_0 blocks (n values, n % 32 == 0) -> F32 / FP16 (the prompt path's weight slices)
void q8_0_to_f32(const void* src, float* dst, int64_t n, cudaStream_t s);
void q8_0_to_f16(const void* src, uint16_t* dst, int64_t n, cudaStream_t s);
/// The INT8 latent cache (STRATA_GLM_KV_INT8): per position kv_lora int8 codes, then one FP16 scale per 32 values
/// (symmetric, absmax / 127) - 544 bytes at kv_lora 512 against the FP16 row's 1024.
__host__ __device__ constexpr inline int lat8_rec_bytes(int kv_lora) { return kv_lora + kv_lora / 16; }

/// SwiGLU with GLM's clamp (gate above, up both sides) and the q8_1 of h, n values.
void swiglu_q8(const float* gate, const float* up, float limit, int n, void* hq, cudaStream_t s);

// ---------------------------------------------------------------- the routed experts
//
// Three tiers, decided ON THE DEVICE per routed expert: tab[key] != 0 -> resident in VRAM (a hit);
// else rtab[key] != 0 -> resident in the pinned RAM tier: the fetch kernel pulls it over PCIe (11.5 GB/s,
// measured) into one of the layer's SPARE slots (it is then resident: the device writes tab itself) or,
// with no spare left, into a scratch slot; else it is only on disk: the host reads it (the one case the
// device waits for).  key = layer * n_expert + expert.  tab, rtab and the spare table are ONE device
// array (tab at [0, N), rtab at [N, 2N), spares at [2N, 2N + n_layers * kSpares)) so a single update
// kernel applies the host's between-token edits.
constexpr int kSpares = 3;
/// The most routed experts a token uses (n_exp_used): the request / plan / answer records hold this many slots.
/// 16 since qwen4exp (Qwen3.8-Flash-Next: top-10); the masks below are 32-bit, one bit a slot.
constexpr int kMaxK = 16;
/// The per-device routing state (device memory unless noted).
struct MoeDev {
    const uint8_t* helper_owned = nullptr; // Decode only; null on the prompt and default paths
    unsigned long long* tab = nullptr;         // [3 tables, see above]
    int n_keys = 0;                            // N = n_layers * n_expert
    unsigned long long* scratch = nullptr;     // [kMaxK] VRAM slots for fetched experts that could not be promoted
    unsigned long long* plan_ptr = nullptr;    // [kMaxK]
    float* plan_w = nullptr;                   // [kMaxK]
    int* plan_id = nullptr;                    // [kMaxK]
    unsigned long long* fetch_src = nullptr;   // [kMaxK] host-mapped source (0 = nothing to fetch for entry i)
    unsigned long long* pf_src = nullptr;      // [kMaxK] the NEXT layer's predicted experts being prefetched: source,
    unsigned long long* pf_dst = nullptr;      // [kMaxK] ... their (claimed spare) slot,
    int* pf_n = nullptr;                       // ... and how many (the side-stream fetch kernel reads these)
    unsigned int* seq = nullptr;               // the device's request counter
    unsigned int* wait_seq = nullptr;          // the seq the wait kernel must see answered (0 = none); [1] the last
                                               // one moe_wait_fetch's block 0 applied (the other blocks wait on it)
    void* ring = nullptr;                      // host-mapped request ring (device pointer)
    const void* resp = nullptr;                // host-mapped response (device pointer)
    float* cpu_part = nullptr;                 // n_embd: the host's contribution (misses computed off-GPU)
    int* cpu_flag = nullptr;                   // 1 when cpu_part holds this layer's contribution
    unsigned int* cpu_seq = nullptr;           // the CPU LANE: [0] the seq whose answer moe_cpu_wait must see (0 = none),
                                               // [1] the route before's (expert deferral), [2] the last answer
                                               // the waits' polling block saw (the other blocks wait on it)
    const void* cpu_ans = nullptr;             // ... host-mapped CpuAnswer (device pointer)
    unsigned int* dcnt = nullptr;              // ... [n_keys] routes per key (halved every 4096 routes): the coldest
                                               //     RAM-tier experts go to the host, the hot ones are promoted
    volatile int* route_error = nullptr;       // host-mapped, first invalid route: 4 * layer + kind (1..3)
    int* route_error_d = nullptr;              // ... its device copy, which the route reads (a host-mapped read
                                               //     waits behind the copy engine's PCIe traffic: 16 -> 133 us)
    // STRATA_GLM_EXPERT_HELPER_PROMOTE (#24): a helper-owned expert whose route count reached this is fetched into a
    // free spare (promoted into VRAM) instead of going to the helper, at most one a route; 0 = never (helper only)
    unsigned int helper_promote = 0;
    // NVFP4 experts: this layer's second-level scales, [n_expert][2] = (gate = up, down), read by moe_gate_up /
    // moe_down through plan_id (MoeColArgs::plan_id for a window's columns); null for every other type
    const float* nv_scale = nullptr;
    // STRATA_GLM_UMA_DIRECT (an integrated GPU: the RAM tier is the same memory): 1 - a planned expert that is not in
    // the VRAM pool but in the RAM tier is read IN PLACE, its RAM-tier address (host-mapped) in the plan: no fetch
    // into a spare or scratch slot, no promotion or prefetch by the route, no CPU lane share, never tail-skipped
    int ram_direct = 0;
    // The CPU lane's EXACT COMBINE (STRATA_GLM_LANE_XC; null = off): xpart_floats(n_embd) floats of device memory.
    // The layer's output is summed in FIXED POINT (fx_term: each weighted expert row, the disk lane's part and the
    // shared expert's rows to an int64 of 2^-kFxShift, the sum rounded to a float once) - an integer sum, the same
    // bits in any order and whoever adds what, so it does not depend on which experts the CPU took.  A route with
    // CPU-lane experts: moe_down leaves the device's int64 sum here, moe_cpu_wait adds the CPU's (CpuAnswer::fx).
    // moe_down also posts it to the answer (CpuAnswer::gfx, gseq: the last of its blocks raises gseq - a counter after
    // the rows): a CPU that finishes after it adds the two itself and sends the float (xmask kXmFinal), so the wait
    // reads 4 bytes a row across PCIe as without the exact combine, not 8 (16 GB of VRAM, the CPU's larger share: the
    // 8-byte read cost ~2% of decode on a 5090 over Gen4 x4)
    float* xpart = nullptr;
};
/// xpart's floats: one int64 a row, then moe_down's block counter (and pad)
constexpr size_t xpart_floats(int n_embd) { return (size_t) 2 * (size_t) n_embd + 4; }
/// The exact combine's fixed point: 2^-34 (5.8e-11) a unit; a term of |v| >= 2^24 (or not finite) marks the row
/// kFxBad, and a row any of whose terms is bad comes out NaN.  Up to 16 terms of < 2^58 units: no int64 overflow
constexpr int kFxShift = 34;
constexpr long long kFxBad = -0x7fffffffffffffffLL - 1;   // INT64_MIN
/// The host's answers to routes - the CPU LANE's (CpuAnswer) and the disk misses' (MoeResponse, with its pinned
/// staging): kCpuAnswers of each, the route `seq`'s at index seq % kCpuAnswers - a verify window has several routes
/// of a layer in flight, and the host answers them in order before the device reads them.  16: a window of 8
/// columns' answers of two layers at once (expert deferral reads layer l's answers after layer l + 1's routes).
constexpr int kCpuAnswers = 16;
/// The widest n_embd the host-mapped route / answer records hold (DeepSeek V4.1 Flash: 5120).
constexpr int kMaxEmbd = 5120;
/// The CPU LANE's answer: the weighted sum of the RAM-tier experts the host computed for route `seq`.
struct CpuAnswer {
    volatile unsigned int seq;
    unsigned int xmask;      // the exact combine (MoeDev::xpart): != 0 when fx holds the CPU's sum (0: part only)
    unsigned int pad[2];
    float part[kMaxEmbd];    // the CPU's experts' weighted sum (xmask kXmFinal: the layer's whole output)
    long long fx[kMaxEmbd];  // xmask kXmFx: the CPU's experts' weighted rows summed in fixed point (kFxShift)
    // the exact combine's device half, posted by moe_down (MoeDev::xpart): its fixed-point sum, gseq = this answer's
    // seq once all of it is written
    volatile unsigned int gseq;
    unsigned int gpad[3];
    long long gfx[kMaxEmbd];
};
/// CpuAnswer::xmask: 0 part to add, kXmFx fx to add to the device's sum, kXmFinal part is the output (fx + gfx)
constexpr unsigned int kXmFx = 1u, kXmFinal = 2u;
constexpr int kRingSize = 64;
/// LOOKAHEAD: a route may also carry the top-k of the next (up to) kAhead layers' routers applied to its FFN input.
constexpr int kAhead = 4;
/// One routing request, written by the device into host-mapped memory.
struct MoeRequest {
    volatile unsigned int seq;   // written LAST (after a system fence)
    int layer;
    int error;                   // invalid route: no experts or table edits; the host still retires its sequence
    unsigned int miss_mask;      // experts the host must provide (only on disk) - the device waits for these
    unsigned int fetch_mask;     // experts fetched from the RAM tier
    unsigned int promo_mask;     // ... of which promoted into a spare slot: promo_ptr[i] is the slot
    int ids[kMaxK];
    float w[kMaxK];
    unsigned long long promo_ptr[kMaxK];
    int pred[kMaxK];                 // the NEXT layer's router applied to this layer's FFN input (-1: none)
    int pf_n;                    // of which prefetched into the next layer's spares: pf_ids / pf_ptr (slot)
    int pf_ids[kMaxK];
    unsigned long long pf_ptr[kMaxK];
    short ahead[kAhead][kMaxK];      // layer + 1 + d's router on this layer's FFN input: its top-k (-1: none)
    unsigned int cpu_mask;       // the CPU LANE: RAM-tier experts the host computes (the device skips them) ...
    unsigned long long cpu_src[kMaxK];   // ... their RAM-tier blobs
    alignas(16) float x[kMaxEmbd];   // the FFN input (16-byte stores; written at the kernel's start, or only when
                                 // cpu_mask != 0 with STRATA_GLM_ROUTE_EARLY=0)
    unsigned int skip_mask;      // experts left out (not in VRAM at rank >= skip_from): neither computed nor waited for
    int xq_ok;                   // xq holds x's q8_1 (the device's own, written with x): the iGPU lane skips its quantize
    alignas(16) unsigned char xq[kMaxEmbd / 32 * 36];
    int win_more;                // a batched verify window's column: this many more routes of the same layer follow
                                 // at once (the CPU lane answers the columns together); 0 otherwise
};
/// The host's answer to a request with misses.
struct MoeResponse {
    volatile unsigned int seq;
    unsigned int ptr_mask;            // plan entries the host made resident: ptr[i] valid
    unsigned int host_mask;           // plan entries the host COMPUTED (their sum is in cpu_part): skipped on the device
    unsigned int stream_mask;         // STREAMED entries: posted before their disk reads finished; ready[i] collects the
    unsigned int ready[kMaxK];            // chunks that landed (bit part * n_chunks + chunk), the fetch copies them as they do
    unsigned long long ptr[kMaxK];
    int n_upd;                        // table updates: tab[key] = val
    int upd_key[64];
    unsigned long long upd_val[64];
    int cpu;                          // 1: cpu_part (host-mapped, below) holds the missed experts' weighted sum
    float cpu_part[kMaxEmbd];
};
/// Router (sigmoid + bias, stable top-k, normalised weights - router_sigmoid's arithmetic; gating 1: deepseek4's
/// sqrt(softplus) probabilities, hash_ids: the k ids of a hash-routed layer instead of a selection), the plan
/// (skip_from / skip_w: experts not in VRAM at rank >= skip_from or with a weight share below skip_w are left out)
/// lookup in `tab`, the request publish, and the shared expert's swiglu + q8_1.
void moe_route(const float* logits, const float* bias, int n_expert, int k, float w_scale, bool norm_w,
               int layer, const float* x, int n_embd, const MoeDev& d, const float* sh_gate, const float* sh_up,
               float sh_limit, int n_ff_sh, void* sh_hq, cudaStream_t s, const float* pred_logits = nullptr,
               const float* pred_bias = nullptr, int max_prefetch = 0, const float* ahead_logits = nullptr,
               const float* const* ahead_bias = nullptr, int n_ahead = 0, int skip_from = 8,
               unsigned long long cpu_plan = 0, int promote_min = 0, int pf_rank = 0, int gating = 0,
               const int* hash_ids = nullptr, float skip_w = 0.0f, int skip_disk = 0, const void* xq = nullptr,
               int win_more = 0);
/// moe_route for a verify window's nc (2..8) columns of one layer in one launch, the same plans and requests as nc
/// launches in column order (each column's turn at the shared tables comes after the one before it): column t routes
/// logits + t * logit_stride over x + t * x_stride (x_stride = n_embd) into mds[t]'s plan, hash_ids[t] its hash ids
/// (null: none).  No shared expert, prediction or lookahead.  sync: 2 device words, zero (they are again after it).
/// false (nothing launched): a setting the batched launch does not take - route the columns one by one.
bool moe_route_cols(const float* logits, size_t logit_stride, const float* bias, int n_expert, int k, float w_scale,
                    bool norm_w, int layer, const float* x, size_t x_stride, const MoeDev* mds, int nc,
                    unsigned int* sync, cudaStream_t s, int skip_from = 8, unsigned long long cpu_plan = 0,
                    int promote_min = 0, int pf_rank = 0, int gating = 0, const int* const* hash_ids = nullptr,
                    float skip_w = 0.0f, int skip_disk = 0);
/// The shared expert's swiglu + q8_1 alone (what moe_route's extra blocks do when it is given sh_gate / sh_up): for
/// the token path's late shared expert, whose gate/up GEMV runs after the route (STRATA_GLM_SH_LATE).
void sh_swiglu(const float* sh_gate, const float* sh_up, float sh_limit, int n_ff_sh, void* sh_hq, cudaStream_t s);
/// The CPU LANE's split: of f RAM-tier experts in a route, cpu_take(plan, f) go to the host (4 bits per f, f = 0..15:
/// a top-k above 15 uses f = 15's entry for more).
inline int cpu_take(unsigned long long plan, int f) { return (int) ((plan >> (4 * (f < 15 ? f : 15))) & 15ull); }
/// Waits (on the device) for the host's CPU-lane answer of the last route when it had CPU experts, then adds it to
/// `out` (the layer's FFN output, which moe_down wrote first - its work runs while the CPU computes).
/// k > 0 with d.xpart (the exact combine): moe_down's fixed-point sum plus the CPU's (CpuAnswer::fx), rounded once.
void moe_cpu_wait(const MoeDev& d, int n_embd, float* out, cudaStream_t s, int mode = 1, int k = 0);
/// moe_cpu_wait (mode 1) for up to 8 columns of a verify window in one launch: column t waits for the answer its
/// seq[t] names and adds it into out[t] (seq[t] null: none)
struct CpuWaitCols {
    const unsigned int* seq[8] = {};
    float* out[8] = {};
    float* xpart[8] = {};   // (the columns' exact combine, MoeDev::xpart; k > 0: on)
    int n = 0;
    int k = 0;
};
void moe_cpu_wait_cols(const CpuWaitCols& c, const void* cpu_ans, int n_embd, cudaStream_t s);
/// Expert deferral: the last route's CPU-lane part into the 4 hyper-connection streams R with the write weights post
/// (4 floats, device): R[d][e] += post[d] * part[e].  Spins for the answer like moe_cpu_wait.
void hc_defer_add(const MoeDev& d, const float* post, int n_embd, float* R, cudaStream_t s);
/// dst[0..n) = src[0..n) by a kernel (no copy engine: those queue behind the expert moves' DMAs)
void copy_floats(float* dst, const float* src, int n, cudaStream_t s);
/// The side stream's copy of the prefetch list (pf_src -> pf_dst, pf_n entries of blob_bytes each).
void moe_prefetch(const MoeDev& d, size_t blob_bytes, cudaStream_t s);
/// Waits (on the device) for the host's response when the last route had misses, then applies it.
void moe_wait(const MoeDev& d, int n_embd, cudaStream_t s);
/// Copies every planned expert with a fetch source from the RAM tier (host-mapped) into its VRAM slot.
/// The disk-landed experts into their planned slots: whole blobs, or - for the response's streamed entries - chunk by
/// chunk as the host's reads land (gu_bytes / dn_bytes / n_chunks: the layout of fast_read_part's chunks)
/// (up_bytes: the up part's bytes when it differs from gate's - a mixed gate/up layer; 0: gu_bytes)
void moe_fetch(const MoeDev& d, int k, size_t blob_bytes, cudaStream_t s, size_t gu_bytes = 0, size_t dn_bytes = 0,
               int n_chunks = 0, size_t up_bytes = 0);
/// moe_wait then moe_fetch in one launch (the same work; a route without misses skips the wait)
void moe_wait_fetch(const MoeDev& d, int n_embd, int k, size_t blob_bytes, cudaStream_t s, size_t gu_bytes = 0,
                    size_t dn_bytes = 0, int n_chunks = 0, size_t up_bytes = 0);
/// moe_wait_fetch for a verify window's n_cols columns in one launch; mds_dev: the columns' plans (device copies)
void moe_wait_fetch_cols(const MoeDev* mds_dev, int n_cols, int n_embd, int k, size_t blob_bytes, cudaStream_t s,
                         size_t gu_bytes = 0, size_t dn_bytes = 0, int n_chunks = 0, size_t up_bytes = 0);
/// One host-mapped blob copied into VRAM the way moe_fetch copies a RAM-tier expert (SMs reading over PCIe, not the
/// copy engine): the CPU lane's calibration times the decode's own PCIe path with it.
void zero_copy(const void* src, void* dst, size_t bytes, cudaStream_t s);
/// The routed gate/up rows of the planned (resident) experts, the clamped swiglu, the q8_1 of h; and (sh_down and
/// sh_out given) the shared expert's down rows over its h in sh_hq into sh_out, by extra blocks of the same launch.
/// up_type: up's type when it differs from gate's (a mixed layer; -1 or gu_type: the same).
void moe_gate_up(int gu_type, const MoeDev& d, int k, int n_embd, int n_ff, float limit, const void* xq, void* hq,
                 const void* sh_down, int sh_type, const void* sh_hq, int n_ff_sh, float* sh_out, cudaStream_t s,
                 int up_type = -1);
/// Whether moe_gate_up / moe_gate_up_cols take this gate/up type pair (up_type -1 or gu_type: moe_supported(gu_type)).
bool moe_gate_up_pair_supported(int gu_type, int up_type);
/// out[r] = (sum_i w_i * down_i(r) . h_i over the resident experts) [+ cpu_part[r]] + sh_out[r] (when given).
/// down_off = the byte offset of the down rows inside an expert blob ([gate | up | down]).
void moe_down(int d_type, const MoeDev& d, int k, int n_embd, int n_ff, size_t down_off, const void* hq,
              const float* sh_out, float* out, cudaStream_t s);
/// What differs between a verify window's columns in one expert launch: each column's own route plan and CPU-lane
/// part (MoeDev's plan_ptr / plan_w / cpu_flag / cpu_part).
struct MoeColArgs {
    const unsigned long long* plan_ptr = nullptr;
    const float* plan_w = nullptr;
    const int* cpu_flag = nullptr;
    const float* cpu_part = nullptr;
    const int* plan_id = nullptr;   // (NVFP4: the experts' ids, for MoeDev::nv_scale)
    float* xpart = nullptr;                 // (MoeDev::xpart, MoeDev::cpu_seq: the column's exact combine)
    const unsigned int* cpu_seq = nullptr;
    const void* cpu_ans = nullptr;          // (MoeDev::cpu_ans: where moe_down posts its sum, CpuAnswer::gfx)
};
/// moe_gate_up / moe_down for nc columns in one grid (cols: nc entries in device memory): column c's xq at
/// xq + c * xq_cs bytes, hq + c * hq_cs bytes, sh_hq + c * sh_hq_cs bytes, sh_out / out + c * out_cs floats - the
/// blocks nc launches would run (the same sums), at once: a window column's ~4 resident experts a layer left the GPU
/// half idle (83 us a launch against 76 for 8 experts).
void moe_gate_up_cols(int gu_type, const MoeColArgs* cols, int nc, int k, int n_embd, int n_ff, float limit,
                      const void* xq, size_t xq_cs, void* hq, size_t hq_cs, const void* sh_down, int sh_type,
                      const void* sh_hq, size_t sh_hq_cs, int n_ff_sh, float* sh_out, size_t out_cs, cudaStream_t s, const float* nv_scale = nullptr,
                      int up_type = -1);
void moe_down_cols(int d_type, const MoeColArgs* cols, int nc, int k, int n_embd, int n_ff, size_t down_off,
                   const void* hq, size_t hq_cs, const float* sh_out, float* out, size_t out_cs, cudaStream_t s, const float* nv_scale = nullptr);

/// The prompt path's LIGHTLY routed experts (a few rows each - MMQ's 64-128-token tiles would be mostly empty):
/// light[3 i .. 3 i + 2] = {slot, r0, nr} (device): expert i is the blob at base + slot * stride, its rows are
/// [r0, r0 + nr) of the sorted rows (token row_tok[r]), all inside [rlo, rhi).  x: the chunk's n_tok token rows
/// (n_embd floats).  Gate/up then swiglu (limit) then down, the down rows into out rows (ld floats apart; out also
/// holds the gate/up rows in between).  Scratch: xq for n_tok x n_embd q8_1, hq for (rhi - rlo) x n_ff q8_1.
/// ptrs (device, n entries): expert i's blob address instead of base + slot * stride (the iGPU lane's RAM tier).
/// False: a type without the light kernels (the caller keeps MMQ for them).
bool rows_experts(int gu_type, int d_type, const uint8_t* base, size_t stride, size_t down_off, const int* light, int n,
                  const int* row_tok, const float* x, int n_tok, int n_embd, int n_ff, float limit, int rlo, int rhi,
                  void* xq_scratch, void* hq_scratch, float* out, int ld, cudaStream_t s,
                  const unsigned long long* ptrs = nullptr);
/// rows_experts in the DECODE kernels' arithmetic (the exact combine, STRATA_GLM_LANE_XC): each (expert, token) pair's
/// rows the bits moe_gate_up / moe_down give that token - and the CPU lane's (lane_q81) - so the prompt's CPU share
/// does not change its numbers.  The same arguments; false: a type it does not take (rows_exact_type).
bool rows_exact_type(int ggml_type);
bool rows_experts_exact(int gu_type, int d_type, const uint8_t* base, size_t stride, size_t down_off, const int* light,
                        int n, const int* row_tok, const float* x, int n_tok, int n_embd, int n_ff, float limit, int rlo,
                        int rhi, void* xq_scratch, void* hq_scratch, float* out, int ld, cudaStream_t s,
                        const unsigned long long* ptrs = nullptr);

/// tab[keys[i]] = vals[i] for i < n (the expert-table edits the host queues between tokens).
void tab_update(unsigned long long* tab, const int* keys, const unsigned long long* vals, int n, cudaStream_t s);

/// The NextN block's input: cat = [rms(emb) * enorm, rms(h) * hnorm] (2 n values) as q8_1.  n <= 4096.
void mtp_in(const float* emb, const float* h, const float* enorm, const float* hnorm, float eps, int n, void* cat_q8_1,
            cudaStream_t s);
/// h += add (add may be null), then x = rms(h) * w and its q8_1 (a plain pre-norm residual step).  n <= 4096.
void rms_q8(float* h, const float* add, const float* w, float eps, int n, float* x, void* xq, cudaStream_t s);

/// argmax over n floats -> *out (device int).  Ties: lowest index.
void argmax(const float* x, int n, int* out, cudaStream_t s);

}  // namespace strata::kernels::glmf
