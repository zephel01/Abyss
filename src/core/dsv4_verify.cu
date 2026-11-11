// src/core/dsv4_verify.cu - deepseek4's verify window: T (1..kVerifyMax) tokens through the fast path at once, for
// speculative decoding (the engine's n-gram drafts, src/program/generate.cpp).
//
// The layer loop of fast_layers with T columns.  What is shared by the window:
//   * every dense GEMV (attention projections, q_b, wo_a / wo_b, the router and the shared expert's gate/up, the
//     output head) is one gf::mvn launch over the T columns - the weights are read once (DeepSeek V4 Flash's 6.7 GB
//     of them: 4.99 ms for one token, 5.10 for two on an RTX 5090);
//   * the CPU lane's requests: every token's route of a layer is issued before the first answer is awaited, so the
//     host works through the window's RAM-tier experts while the device runs the resident ones.
// What stays per token: the mHC reads, the attention's prep / compress / q_prep / indexer / attend (in position order:
// token t sees exactly positions <= p0 + t - a later token's ring rows and compressed rows are not visible to it), a
// route each, the resident experts' kernels and the CPU-lane combine.
// STRATA_DS4_SPEC_PIPE (default 1): the PIPELINED window instead - each token on its own stream through the
// single-token kernels, token t + 1 a stage behind token t: its attention of a layer runs while the CPU lane computes
// token t's RAM-tier experts of that layer, and token t's next layer runs while the lane does token t + 1's.  The
// dense weights are read once a token (no GEMV sharing), but the host's expert work - the window's largest cost -
// overlaps the device's instead of adding to it.  The order that keeps it exact: a token's attention of a layer
// waits for the previous token's (ring rows, compressor state), and the routes run in (layer, token) order (the
// request counter, the tier tables, the host's in-order answers).
// Rejected tokens are dropped by spec_rewind (the position only): the window ring has kSpecSlack spare rows, and the
// compressor rings / compressed caches are rewritten in position order before anything reads them.
// A layer split: the parts one after another (verify_part), as a token's - each part's last layer writes its columns'
// complete residuals, which cross to the next part through a pinned host buffer, the copies ordered by an event; the
// last part runs the head and keeps the logits and the argmax (verify_argmax / _sample / _logits read its buffers).
// deepseek41 (verify_part_v41, verify_attn_v41; the batched kind, one part): fast_dsv41_attn's attention with the
// dense GEMVs over the columns; per column its two lagged pre buffers, its index selection and candidate blocks (a
// later layer of the same column reads them), and its Engram rows - read for every column before the layers, the
// n-gram history advancing over the window's tokens.  A rejected column leaves nothing a later position reads: the
// window ring has kSpecSlack spare rows, a KV slot's rows are written at their position's row and become visible only
// past it, and a ratio-2 slot parks a pair's first token in the pair's own entry (kV41PendRing).
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/dsv41_fast.hpp"
#include "strata/kernels/dsv4_fast.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/qwen35_fast.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace ds = strata::kernels::dsv4;
namespace d41 = strata::kernels::dsv41;
namespace q35 = strata::kernels::q35;

namespace {
constexpr size_t kQ8 = 36;   // bytes of a q8_1 block (32 values)

ds::Rope layer_rope_v(const Glm5Geometry& g, bool comp) {
    if (!comp) return ds::make_rope(g.n_rot, g.rope_base, 1.0f, 0.0f, 0, 0.0f, 0.0f);
    return ds::make_rope(g.n_rot, g.comp_rope_base, 1.0f / g.yarn_factor, 1.0f, g.yarn_orig_ctx, g.yarn_beta_fast,
                         g.yarn_beta_slow);
}

// the window's buffers: [kVerifyMax] columns of each per-token activation, contiguous (column t at + t * width)
struct VerifyBuf {
    int TM = 0;
    float *R = nullptr;                       // [TM][2][hc * n_embd]: each token's residual pair
    float *x = nullptr, *mixer = nullptr, *ffn = nullptr, *pre = nullptr, *post = nullptr, *comb = nullptr;
    float* post_defer = nullptr;   // expert deferral: a column's FFN write weights of the layer whose part is deferred
    void* xq = nullptr;
    float *qr_raw = nullptr, *qr = nullptr, *kv_raw = nullptr, *iw = nullptr, *q = nullptr, *iq = nullptr;
    void* qr_q = nullptr;
    float *ckv[2] = {nullptr, nullptr}, *csc[2] = {nullptr, nullptr};
    float *o = nullptr, *oa = nullptr;
    void *oq = nullptr, *oaq = nullptr;
    float *rlog = nullptr, *sh_g = nullptr, *sh_u = nullptr, *sh_out = nullptr;
    void* sh_hq = nullptr;
    float *head_x = nullptr, *logits = nullptr, *emb = nullptr;
    void* head_xq = nullptr;
    int* tok = nullptr;                       // device argmax per column
    int* tok_h = nullptr;                     // pinned, host-mapped (tok_hd: its device address)
    int* tok_hd = nullptr;
    // a sampled window's draws (verify_sample): all T columns in one sampler launch at column 0 into tok_h[32..],
    // row t with counter c0 + t - the same Philox draws the column-at-a-time path made (sp.counter + 1 a column)
    uint64_t samp_c0 = 0;
    bool samp_ok = false;
    int* tok_in_h = nullptr;                  // the window's tokens, host-mapped for the embedding kernel
    int* tok_in_d = nullptr;
    float* emb_h = nullptr;                   // pinned [TM][n_embd]
    float* hop_h = nullptr;                   // a split: pinned [TM][hc * n_embd], the columns' residuals to the next part
    cudaEvent_t ev_hop = nullptr;
    std::vector<gf::MoeDev> md;               // a route plan per column (the shared tables, own plan arrays)
    gf::MoeColArgs* cols_d = nullptr;         // ... their plan / CPU-lane fields on the device (the column-batched experts)
    gf::MoeDev* md_d = nullptr;               // ... the columns' plans as set up (device: moe_wait_fetch_cols)
    unsigned int* route_sync = nullptr;       // [2] device, zero: moe_route_cols' ticket and turn counters
    std::vector<cudaGraphExec_t> q4g;         // qwen4exp: a layer's graph a window width ([layer][T], verify_part_q4x)
    std::vector<unsigned char> q4g_n;         // ... the windows seen before its capture (255: it did not capture)
    std::vector<uint64_t> q4g_sig;            // ... the route arguments it was captured with
    unsigned long long q4g_rep = 0, q4g_eager = 0, q4g_recap = 0;   // layers replayed / launched, re-captures
    unsigned long long q4g_pdl_ok = 0, q4g_pdl_no = 0;   // captures with PDL edges taken / refused (STRATA_Q4X_GRAPH_PDL)
    float* ffn_chk = nullptr;                 // STRATA_DS4_WIN_COLS_CHECK: the batched launch's FFN rows, then compared
    uint64_t chk_calls = 0, chk_cols = 0, chk_diff = 0;
    // per column, for the pipelined window's concurrent tokens: the mHC partials, the routed experts' h, the
    // indexer's scores / selection, the fetched experts' scratch slots (column 0: the token path's)
    float* hpart = nullptr;
    unsigned int* hcnt = nullptr;
    void* hq = nullptr;
    float* score = nullptr;
    int* sel = nullptr;
    void* scr = nullptr;
    size_t hpart_n = 0, hq_n = 0, score_n = 0;
    bool pipe = false;                        // the per-column scratch slots allocated: the pipelined window possible
    bool pipe_now = true;                     // this window pipelined (verify's choice, STRATA_DS4_SPEC_PIPE)
    // (the first part's) the kind kept on this machine, a width each (-1: still timing both): the kinds are close at
    // 3 columns and far apart at 6 (61 against 75 ms) - a choice made at one width does not hold at another
    int pipe_choice[Glm5Model::kVerifyMax + 1] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
    // ... the timed windows' ms of each kind, by width (windows of different widths are not comparable: DSpark's
    // vary 2..6 columns, and 3 + 3 of mixed widths once kept the batched window at 78 ms against 56)
    std::vector<double> ms_pipe[Glm5Model::kVerifyMax + 1], ms_batch[Glm5Model::kVerifyMax + 1];
    cudaStream_t st[Glm5Model::kVerifyMax] = {};   // column t's stream (st[0]: the compute stream)
    cudaEvent_t ev_attn[Glm5Model::kVerifyMax] = {}, ev_route[Glm5Model::kVerifyMax] = {},
                ev_fin[Glm5Model::kVerifyMax] = {}, ev_start = nullptr;
    cudaEvent_t done = nullptr;
    int T = 0;                                // the last window's size
    // column t's CPU-lane answer of the last layer, left to its next attention-side hc3 read (verify_moe, as fast_moe's
    // STRATA_GLM_HC_CPU) - and the paced promotion copies, which go out after that read
    bool hc_cpu[Glm5Model::kVerifyMax] = {}, bg_after[Glm5Model::kVerifyMax] = {};
    // the pipelined window's scratch slot sets (8 VRAM slots a set): set 0 the token path's, sets 1..n_scr the
    // window's own - a group of columns runs on one stream, so its columns share their group's set.  deepseek41 (the
    // batched window only): a set a column up to kScrSets + 1 columns, so their SSD reads stay apart until the
    // column-batched expert launches read them (V4's pipelined window keeps 3 sets)
    static constexpr int kScrSets = 5;
    unsigned long long* scr_set[kScrSets + 1] = {};
    int n_scr = 0;
    // deepseek41, a column each: its two pre buffers ([0, 4) the attention side's, [4, 8) the FFN side's - the lagged
    // collapse), its index scores (ctx + 1), block maxima, chosen blocks and candidate mask (fast_dsv41_attn's v41_*)
    float* v41_pre = nullptr;
    float *v41_score = nullptr, *v41_bscore = nullptr;
    int* v41_bsel = nullptr;
    unsigned char* v41_keep = nullptr;
    size_t v41_score_n = 0, v41_b_n = 0;
    // qwen35moe: the window's rows a delta-net layer (its conv inputs, its prepped conv rows, its gates: the rewind
    // replays the kept rows from them), the states before the window (every delta-net layer's state and conv state),
    // the attention and recurrence scratch over the columns, the shared expert's gate logit a column
    int q35_layers = 0;                       // delta-net layers in this part
    int64_t q35_p0 = -1;                      // the last window's first position (-1: none)
    int32_t q35_toks[Glm5Model::kVerifyMax] = {};   // ... its tokens (the drafter's rows take the next ones)
    float *q35_qkv = nullptr, *q35_rows = nullptr, *q35_alpha = nullptr, *q35_beta = nullptr;
    float *q35_snap = nullptr, *q35_csnap = nullptr;
    float *q35_z = nullptr, *q35_o = nullptr, *q35_y = nullptr, *q35_qf = nullptr, *q35_k = nullptr, *q35_v = nullptr,
          *q35_q = nullptr, *q35_ao = nullptr, *q35_shg = nullptr, *q35_part = nullptr;
    void *q35_yq = nullptr, *q35_aoq = nullptr;
    // qwen4exp, a column each: the hyper-connection's normed streams (and q8_1), low rank, gates and the two injects;
    // the PLE's embedding rows, keys, values, postops scratch (q4_ple_qn ends as the rows the conv history takes - the
    // rewind replays the kept ones onto the history kept before the window, q4_ple_snap); the QSA indexer's raw keys,
    // queries, step records and query positions (and their pinned source)
    float *q4_xn = nullptr, *q4_lo = nullptr, *q4_gate = nullptr, *q4_inj[2] = {nullptr, nullptr}, *q4_sc = nullptr;
    void *q4_xnq = nullptr, *q4_loq = nullptr, *q4_embq = nullptr;
    float *q4_emb = nullptr, *q4_key = nullptr, *q4_val = nullptr, *q4_qn = nullptr, *q4_gated = nullptr,
          *q4_g4 = nullptr, *q4_ple_snap = nullptr, *q4_iraw = nullptr, *q4_iq = nullptr;
    int32_t *q4_steps = nullptr, *q4_steps_h = nullptr;
    // ... and a QSA layer each: its raw-key tail before the window and the columns' raw keys (a rejected column past a
    // block's end overwrote tail slots the block's kept cells still need: the rewind puts the tail back and appends
    // the kept columns' keys again)
    int q4_nattn = 0;
    float *q4_tsnap = nullptr, *q4_rawk = nullptr;
    bool q4_ple_on = false;                   // the last window ran the PLE (its rows and snapshot are valid)
    // glm5-next (verify_part_glm), a column each: the KDA layers' gated output (q8_1); the DSA layers' raw latent,
    // indexer key / gate rows, q, the attention's q8_1, its cells and pool scores; the dense FFN's gate / up / h.
    // A KDA layer x a column: its inputs (q, k, v, f_a, g_a, beta: glm_ki floats) - the rewind replays the kept columns
    // from them through the token path's kda_prep / kda_rec onto the state kept after column 0 (kda_bak_)
    int glm_rec = 0;                          // KDA layers in this part
    std::vector<int> glm_li;                  // layer -> its KDA index (-1: a DSA layer)
    size_t glm_ki = 0, glm_run = 0;           // floats: a (layer, column)'s inputs; a layer's S + conv state
    float* glm_kin = nullptr;
    void* glm_gq = nullptr;
    size_t glm_gq_b = 0, glm_aq_b = 0, glm_score_n = 0;
    float *glm_kv = nullptr, *glm_ik = nullptr, *glm_ig = nullptr, *glm_q = nullptr, *glm_score = nullptr;
    void* glm_aq = nullptr;
    int* glm_cells = nullptr;
    int glm_nsel = 0;
    float *glm_dg = nullptr, *glm_du = nullptr;
    void* glm_dhq = nullptr;
    int64_t glm_p0 = -1;                      // the last window's first position (-1: none)
    int32_t glm_toks[Glm5Model::kVerifyMax] = {};
    // the checks (STRATA_GLM_WINDOW_CHECK / _REWIND_CHECK / _MVN_CHECK): the KDA states before and after the window, a
    // replay's scratch state, a mv() reference's outputs
    float *chk_pre = nullptr, *chk_post = nullptr, *chk_ref = nullptr, *mvn_ref = nullptr;
    size_t mvn_ref_n = 0;
    bool chk_pre_ok = false;
};
}  // namespace

bool Glm5Model::verify_ready() const {
    // every part of a split (the window runs part after part): the fast path, no draft block, prefetch or lookahead,
    // MoE layers; the head's weights on the last
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        const FastState* F = m->fast_;
        if (F != nullptr && m->g_.arch == 0) {
            // glm5-next: one part, the batched window (STRATA_GLM_WINDOW=1); its NextN block allowed (the drafter);
            // no prefetch / lookahead / deferral / refusal edit; no EXL3 layers (verify_moe runs the ggml experts)
            if (!m->glm_window() || F->max_pf > 0 || F->n_ahead > 0 || F->defer || m->split_next_ != nullptr ||
                m != this || m->next_port_ != nullptr || m->abl_loaded())
                return false;
            for (int il = m->l0_; il < m->l1_; ++il)
                if (F->L[(size_t) il].exl3 || F->L[(size_t) il].idx_q_b_x.q != nullptr) return false;
            continue;
        }
        if (F == nullptr || !(m->g_.dsv4() || m->g_.qwen35()) || (m->mtp_il_ >= 0 && !m->g_.qwen35()) || F->max_pf > 0 ||
            F->n_ahead > 0) return false;
        if ((m->g_.v41 || m->g_.qwen35()) && F->defer) return false;   // (no deferral)
        // (one part; qwen4exp's window crosses a split as its token does: verify_part_q4x's hop)
        if ((m->g_.v41 || (m->g_.qwen35() && !m->g_.q4x)) && (m->split_next_ != nullptr || m != this)) return false;
        for (int il = m->l0_; il < m->l1_; ++il)
            if (!F->L[(size_t) il].moe) return false;
    }
    const Glm5Model* tail = verify_tail();
    const auto it = tail->ws_map_.find("output.weight");
    return it != tail->ws_map_.end() && it->second.type != 0;
}

const Glm5Model* Glm5Model::verify_tail() const {
    const Glm5Model* t = this;
    while (t->split_next_) t = t->split_next_.get();
    return t;
}

bool Glm5Model::verify_alloc(std::string& err) {
    if (vb_ != nullptr) return true;
    const Glm5Geometry& g = g_;
    auto* V = new VerifyBuf();
    const int TM = kVerifyMax;
    V->TM = TM;
    const size_t E = (size_t) g.n_embd, D = (size_t) g.head_dim, HD = (size_t) g.n_head * D,
                 OL = (size_t) g.o_groups * g.o_lora, FF = (size_t) g.n_ff_exp * g.n_shared;
    bool ok = true;
    const auto f = [&](float*& p, size_t n) { ok = ok && cudaMalloc((void**) &p, n * TM * sizeof(float)) == cudaSuccess; };
    const auto b = [&](void*& p, size_t n) { ok = ok && cudaMalloc(&p, n * TM) == cudaSuccess; };
    ok = cudaMalloc((void**) &V->R, (size_t) TM * 2 * g.hc * E * sizeof(float)) == cudaSuccess;
    f(V->x, E); f(V->mixer, E); f(V->ffn, E); f(V->pre, 8); f(V->post, 8); f(V->comb, 16); f(V->post_defer, 8);
    b(V->xq, E / 32 * kQ8);
    f(V->qr_raw, g.q_lora); f(V->qr, g.q_lora); b(V->qr_q, (size_t) g.q_lora / 32 * kQ8);
    f(V->kv_raw, D); f(V->iw, g.idx_heads); f(V->q, HD); f(V->iq, (size_t) g.idx_heads * g.idx_key);
    for (int i = 0; i < 2; ++i) { f(V->ckv[i], 2 * D); f(V->csc[i], 2 * D); }
    f(V->o, HD); b(V->oq, HD / 32 * kQ8); f(V->oa, OL); b(V->oaq, OL / 32 * kQ8);
    f(V->rlog, g.n_expert); f(V->sh_g, FF); f(V->sh_u, FF); f(V->sh_out, E); b(V->sh_hq, FF / 32 * kQ8);
    f(V->head_x, E); b(V->head_xq, E / 32 * kQ8); f(V->logits, (size_t) g.n_vocab); f(V->emb, E);
    ok = ok && cudaMalloc((void**) &V->tok, 64 * sizeof(int)) == cudaSuccess;
    ok = ok && cudaHostAlloc((void**) &V->tok_h, 64 * sizeof(int), cudaHostAllocMapped) == cudaSuccess;
    ok = ok && cudaHostAlloc((void**) &V->tok_in_h, 64 * sizeof(int), cudaHostAllocMapped) == cudaSuccess;
    ok = ok && cudaHostGetDevicePointer((void**) &V->tok_hd, V->tok_h, 0) == cudaSuccess;
    ok = ok && cudaHostGetDevicePointer((void**) &V->tok_in_d, V->tok_in_h, 0) == cudaSuccess;
    ok = ok && cudaHostAlloc((void**) &V->emb_h, (size_t) TM * E * sizeof(float), cudaHostAllocDefault) == cudaSuccess;
    ok = ok && cudaEventCreateWithFlags(&V->done, cudaEventDisableTiming) == cudaSuccess;
    V->hpart_n = (size_t) (4 * E / 512) * 36 + 64;
    const int KS = std::max(8, g.n_exp_used);   // a column's route slots (8, or the top-k above it: qwen4exp 10)
    V->hq_n = (size_t) KS * g.n_ff_exp / 32 * kQ8;
    V->score_n = (size_t) std::max<int64_t>(64, max_ctx_ / 4 + 64);
    ok = ok && cudaMalloc((void**) &V->hpart, V->hpart_n * TM * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc((void**) &V->hcnt, (size_t) 64 * TM) == cudaSuccess;
    ok = ok && cudaMalloc(&V->hq, V->hq_n * TM) == cudaSuccess;
    ok = ok && cudaMalloc((void**) &V->score, V->score_n * TM * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc((void**) &V->sel, (size_t) g.idx_top_k * TM * sizeof(int)) == cudaSuccess;
    if (ok) cudaMemset(V->hcnt, 0, (size_t) 64 * TM);
    if (ok && g.qwen35()) {
        const size_t C = (size_t) g.ssm_conv_dim(), Vd = (size_t) g.ssm_inner, Sd = (size_t) g.ssm_state,
                     HV = (size_t) g.ssm_v_heads, KV = (size_t) g.attn_head_kv * D;
        int nrec = 0;
        for (int il = l0_; il < l1_; ++il) nrec += fast_->L[(size_t) il].recr ? 1 : 0;
        V->q35_layers = nrec;
        const size_t L = (size_t) std::max(nrec, 1);
        ok = cudaMalloc((void**) &V->q35_qkv, L * TM * C * sizeof(float)) == cudaSuccess &&
             cudaMalloc((void**) &V->q35_rows, L * TM * C * sizeof(float)) == cudaSuccess &&
             cudaMalloc((void**) &V->q35_alpha, L * TM * HV * sizeof(float)) == cudaSuccess &&
             cudaMalloc((void**) &V->q35_beta, L * TM * HV * sizeof(float)) == cudaSuccess &&
             cudaMalloc((void**) &V->q35_snap, L * Sd * HV * Sd * sizeof(float)) == cudaSuccess &&
             cudaMalloc((void**) &V->q35_csnap, L * C * (size_t) (g.d_conv - 1) * sizeof(float)) == cudaSuccess;
        f(V->q35_z, Vd); f(V->q35_o, Vd); f(V->q35_y, Vd); b(V->q35_yq, Vd / 32 * kQ8);
        f(V->q35_qf, 2 * HD); f(V->q35_k, KV); f(V->q35_v, KV); f(V->q35_q, HD); f(V->q35_ao, HD); b(V->q35_aoq, HD / 32 * kQ8);
        f(V->q35_shg, 1);
        ok = ok && cudaMalloc((void**) &V->q35_part,
                              q35::attn_window_floats(g.n_head, (int) D, (int) std::max<int64_t>(max_ctx_, 1), (int) TM) *
                                  sizeof(float)) == cudaSuccess;
    }
    if (ok && g.q4x) {
        const size_t HE = (size_t) g.hc * E, LR = (size_t) g.hc_lr, ID = (size_t) g.idx_key, IH = (size_t) g.idx_heads;
        f(V->q4_xn, HE); b(V->q4_xnq, HE / 32 * kQ8); f(V->q4_lo, LR); b(V->q4_loq, (LR + 31) / 32 * kQ8);
        f(V->q4_gate, HE); f(V->q4_inj[0], 4); f(V->q4_inj[1], 4); f(V->q4_sc, 4);
        f(V->q4_emb, E); b(V->q4_embq, E / 32 * kQ8); f(V->q4_key, HE); f(V->q4_val, E); f(V->q4_qn, HE);
        f(V->q4_gated, HE); f(V->q4_g4, 4); f(V->q4_iraw, ID); f(V->q4_iq, IH * ID);
        ok = ok && cudaMalloc((void**) &V->q4_ple_snap, (size_t) strata::kernels::NG_HIST * HE * sizeof(float)) == cudaSuccess &&
             cudaMalloc((void**) &V->q4_steps, (size_t) TM * (4 + IH) * sizeof(int32_t)) == cudaSuccess &&
             cudaHostAlloc((void**) &V->q4_steps_h, (size_t) TM * (4 + IH) * sizeof(int32_t), cudaHostAllocDefault) == cudaSuccess;
        int na = 0;
        for (int il = l0_; il < l1_; ++il) na += fast_->L[(size_t) il].q4_tail != nullptr ? 1 : 0;
        V->q4_nattn = na;
        const size_t tail_rows = (size_t) strata::kernels::qsa_real_shapes().idx_block - 1;
        if (na > 0)
            ok = ok && cudaMalloc((void**) &V->q4_tsnap, (size_t) na * tail_rows * ID * sizeof(float)) == cudaSuccess &&
                 cudaMalloc((void**) &V->q4_rawk, (size_t) na * TM * ID * sizeof(float)) == cudaSuccess;
    }
    if (ok && g.v41) {
        const size_t ctx = (size_t) std::max<int64_t>(max_ctx_, 1);
        V->v41_score_n = ctx + 1;
        V->v41_b_n = ctx / (size_t) g.cand_bs + 2;
        ok = cudaMalloc((void**) &V->v41_pre, (size_t) TM * 8 * sizeof(float)) == cudaSuccess &&
             cudaMalloc((void**) &V->v41_score, V->v41_score_n * TM * sizeof(float)) == cudaSuccess &&
             cudaMalloc((void**) &V->v41_bscore, V->v41_b_n * TM * sizeof(float)) == cudaSuccess &&
             cudaMalloc((void**) &V->v41_bsel, (size_t) g.cand_blocks * TM * sizeof(int)) == cudaSuccess &&
             cudaMalloc((void**) &V->v41_keep, V->v41_b_n * TM) == cudaSuccess;
        if (ok) cudaMemset(V->v41_keep, 0, V->v41_b_n * TM);
    }
    if (ok && g.arch == 0) {
        const size_t DI = (size_t) g.d_inner(), HDk = (size_t) g.kda_head_dim;
        V->glm_li.assign((size_t) g.n_layers + 1, -1);
        int nrec = 0;
        for (int il = l0_; il < l1_; ++il)
            if (fast_->L[(size_t) il].recr) V->glm_li[(size_t) il] = nrec++;
        V->glm_rec = nrec;
        V->glm_ki = (3 * DI + 2 * HDk + (size_t) g.n_head + 63) / 64 * 64;
        V->glm_run = DI * HDk + 3 * DI * (size_t) (g.d_conv - 1);
        V->glm_gq_b = DI / 32 * kQ8;
        V->glm_aq_b = (size_t) g.n_head * g.v_head / 32 * kQ8;
        V->glm_nsel = g.n_sel_max();
        V->glm_score_n = (size_t) std::max<int64_t>(64, max_ctx_ / std::max(1, g.idx_kpool) + 64);
        const size_t L = (size_t) std::max(nrec, 1);
        ok = cudaMalloc((void**) &V->glm_kin, L * TM * V->glm_ki * sizeof(float)) == cudaSuccess &&
             cudaMalloc((void**) &V->glm_cells, (size_t) V->glm_nsel * TM * sizeof(int)) == cudaSuccess &&
             cudaMalloc((void**) &V->glm_score, V->glm_score_n * TM * sizeof(float)) == cudaSuccess;
        b(V->glm_gq, V->glm_gq_b);
        f(V->glm_kv, (size_t) g.kv_lora); f(V->glm_ik, (size_t) g.idx_key); f(V->glm_ig, (size_t) g.idx_key);
        f(V->glm_q, (size_t) g.n_head * g.qk_nope); b(V->glm_aq, V->glm_aq_b);
        f(V->glm_dg, (size_t) g.n_ff_dense); f(V->glm_du, (size_t) g.n_ff_dense);
        b(V->glm_dhq, (size_t) g.n_ff_dense / 32 * kQ8);
        // the state after column 0 (the rewind's start): kda_bak_, the two-part decode's buffer (one layout)
        if (ok && nrec > 0 && kda_bak_ == nullptr)
            ok = cudaMalloc((void**) &kda_bak_, (size_t) nrec * V->glm_run * sizeof(float)) == cudaSuccess;
        std::fprintf(stderr, "glm verify: CUDA%d the window's buffers - %d KDA layers' replay inputs %.0f MB, the state "
                             "after column 0 %.0f MB\n", dev_, nrec,
                     (double) L * TM * V->glm_ki * 4 / 1048576.0, (double) nrec * V->glm_run * 4 / 1048576.0);
    }
    V->st[0] = fast_->cs;
    for (int t = 0; t < TM && ok; ++t) {
        if (t > 0) ok = cudaStreamCreateWithFlags(&V->st[t], cudaStreamNonBlocking) == cudaSuccess;
        for (cudaEvent_t* e : {&V->ev_attn[t], &V->ev_route[t], &V->ev_fin[t]})
            ok = ok && cudaEventCreateWithFlags(e, cudaEventDisableTiming) == cudaSuccess;
    }
    ok = ok && cudaEventCreateWithFlags(&V->ev_start, cudaEventDisableTiming) == cudaSuccess;
    // the per-column route plans: everything the route / wait / fetch / gate_up / down / cpu_wait kernels write per
    // route; the tables, the request counter, the rings, the scratch slots and the route counts stay shared
    V->md.assign((size_t) TM, fast_->md);
    for (int t = 0; t < TM && ok; ++t) {
        gf::MoeDev& m = V->md[(size_t) t];
        uint8_t* blk = nullptr;
        const size_t u = (size_t) KS * sizeof(unsigned long long);   // (64 bytes at 8 slots, as before)
        const size_t nb = 4 * u + 3 * 64 + E * sizeof(float);
        ok = cudaMalloc((void**) &blk, nb) == cudaSuccess;
        if (!ok) break;
        cudaMemset(blk, 0, nb);
        m.plan_ptr = (unsigned long long*) blk;
        m.fetch_src = (unsigned long long*) (blk + u);
        m.plan_w = (float*) (blk + 2 * u);
        m.plan_id = (int*) (blk + 3 * u);
        m.wait_seq = (unsigned int*) (blk + 4 * u);
        m.cpu_flag = (int*) (blk + 4 * u + 64);
        m.cpu_part = (float*) (blk + 4 * u + 192);
        if (fast_->md.cpu_seq != nullptr) m.cpu_seq = (unsigned int*) (blk + 4 * u + 128);
        m.xpart = nullptr;   // (the exact combine: a column's own fixed-point sum, STRATA_GLM_LANE_XC)
        if (fast_->xc && m.cpu_seq != nullptr) {
            const size_t xb = gf::xpart_floats(g.n_embd) * sizeof(float);
            ok = cudaMalloc((void**) &m.xpart, xb) == cudaSuccess;
            if (!ok) break;
            cudaMemset(m.xpart, 0, xb);
        }
        m.pf_src = fast_->pf_buf[0];
        m.pf_dst = fast_->pf_buf[0] + 8;
        m.pf_n = fast_->pf_n_buf[0];
    }
    if (ok) {   // the columns' plans for the experts in one launch a group (moe_gate_up_cols / moe_down_cols)
        std::vector<gf::MoeColArgs> ca((size_t) TM);
        for (int t = 0; t < TM; ++t) {
            const gf::MoeDev& m = V->md[(size_t) t];
            ca[(size_t) t] = gf::MoeColArgs{m.plan_ptr, m.plan_w, m.cpu_flag, m.cpu_part, m.plan_id, m.xpart, m.cpu_seq, m.cpu_ans};
        }
        ok = cudaMalloc((void**) &V->cols_d, ca.size() * sizeof(gf::MoeColArgs)) == cudaSuccess &&
             cudaMemcpy(V->cols_d, ca.data(), ca.size() * sizeof(gf::MoeColArgs), cudaMemcpyHostToDevice) == cudaSuccess;
        // (the fields the wait and the fetch read - the plan, the fetch sources, wait_seq, the response ring, the
        // tables - are fixed a column from here on)
        ok = ok && cudaMalloc((void**) &V->md_d, (size_t) TM * sizeof(gf::MoeDev)) == cudaSuccess &&
             cudaMemcpy(V->md_d, V->md.data(), (size_t) TM * sizeof(gf::MoeDev), cudaMemcpyHostToDevice) == cudaSuccess;
        ok = ok && cudaMalloc((void**) &V->route_sync, 64) == cudaSuccess && cudaMemset(V->route_sync, 0, 64) == cudaSuccess;
    }
    // the pipelined window's scratch slots (8 an extra column, an expert blob each): without them, the batched window
    size_t blob_max = 0;
    for (int il = l0_; il < l1_; ++il) blob_max = std::max(blob_max, fast_->L[(size_t) il].blob);
    const size_t sstride = (blob_max + 255u) & ~(size_t) 255u;
    // (sets for up to 4 groups: the 8-column window keeps 4 columns' worth, 192 MB a 8 MB blob)
    // (deepseek41, qwen35moe / qwen4exp: their windows are batched only - a set a column, up to kScrSets + 1 columns)
    const int NS = std::min(TM - 1, g.v41 || g.qwen35() || g.arch == 0 ? (int) VerifyBuf::kScrSets : 3);
    V->scr_set[0] = fast_->md.scratch;
    if (ok && sstride > 0 && cudaMalloc(&V->scr, (size_t) NS * KS * sstride) == cudaSuccess) {
        for (int k = 1; k <= NS && ok; ++k) {
            std::vector<unsigned long long> sp((size_t) KS);
            for (int i = 0; i < KS; ++i)
                sp[(size_t) i] = (unsigned long long) ((uint8_t*) V->scr + ((size_t) (k - 1) * KS + i) * sstride);
            unsigned long long* d = nullptr;
            const size_t spb = sp.size() * sizeof(unsigned long long);
            ok = cudaMalloc((void**) &d, spb) == cudaSuccess &&
                 cudaMemcpy(d, sp.data(), spb, cudaMemcpyHostToDevice) == cudaSuccess;
            V->scr_set[k] = d;
        }
        for (int t = 1; t < TM; ++t) V->md[(size_t) t].scratch = V->scr_set[std::min(t, NS)];
        V->n_scr = NS;
        V->pipe = ok && !g.v41 && g.arch != 0;   // (deepseek41, glm5-next: the batched window only)
    } else {
        cudaGetLastError();
        std::fprintf(stderr, "deepseek4 verify: no VRAM for the pipelined window's scratch slots (%.0f MB) - batched\n",
                     (double) NS * KS * sstride / 1048576.0);
    }
    if (!ok) {
        cudaGetLastError();
        err = "deepseek4 verify: the window's buffers did not allocate";
        delete V;
        return false;
    }
    vb_ = V;
    return true;
}

bool Glm5Model::verify_attn(int il, int64_t p0, int tb, int T, cudaStream_t s, std::string& err) {
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const int D = g.head_dim, r = Ly.ratio, E = g.n_embd;
    const ds::Rope rope = layer_rope_v(g, r != 0), crope = layer_rope_v(g, true);
    const int HD = g.n_head * D, OL = g.o_groups * g.o_lora;
    // the columns tb.. (the window's buffers from column tb on)
    const size_t q8e = (size_t) (E / 32) * kQ8;
    void* const xq = (uint8_t*) V->xq + tb * q8e;
    float* const x = V->x + (size_t) tb * E;
    float* const qr_raw = V->qr_raw + (size_t) tb * g.q_lora;
    float* const qr = V->qr + (size_t) tb * g.q_lora;
    void* const qr_qb = (uint8_t*) V->qr_q + (size_t) tb * (g.q_lora / 32) * kQ8;
    float* const kv_raw = V->kv_raw + (size_t) tb * D;
    float* const iwb = V->iw + (size_t) tb * g.idx_heads;
    float* const ckv0 = V->ckv[0] + (size_t) tb * 2 * D;
    float* const csc0 = V->csc[0] + (size_t) tb * 2 * D;
    float* const ckv1 = V->ckv[1] + (size_t) tb * 2 * D;
    float* const csc1 = V->csc[1] + (size_t) tb * 2 * D;
    float* const qb = V->q + (size_t) tb * HD;
    float* const iqb = V->iq + (size_t) tb * g.idx_heads * g.idx_key;
    float* const ob = V->o + (size_t) tb * HD;
    void* const oqb = (uint8_t*) V->oq + (size_t) tb * (HD / 32) * kQ8;
    float* const oab = V->oa + (size_t) tb * OL;
    void* const oaqb = (uint8_t*) V->oaq + (size_t) tb * (OL / 32) * kQ8;
    float* const mixer = V->mixer + (size_t) tb * E;
    // 1. the projections of x, every column at once
    gf::MvJob j[gf::kMaxMvJobs];
    int nj = 0;
    const int cw = r == 4 ? 2 * D : D;
    j[nj++] = {Ly.q_a.q, xq, x, qr_raw, nullptr, 1.0f, Ly.q_a.type, E, g.q_lora};
    j[nj++] = {Ly.kv.q, xq, x, kv_raw, nullptr, 1.0f, Ly.kv.type, E, D};
    if (r != 0) {
        j[nj++] = {Ly.comp_kv.q, xq, x, ckv0, nullptr, 1.0f, Ly.comp_kv.type, E, cw, 0, 2 * D};
        j[nj++] = {Ly.comp_gate.q, xq, x, csc0, nullptr, 1.0f, Ly.comp_gate.type, E, cw, 0, 2 * D};
    }
    if (r == 4) {
        j[nj++] = {Ly.idx_comp_kv.q, xq, x, ckv1, nullptr, 1.0f, Ly.idx_comp_kv.type, E, 2 * g.idx_key, 0, 2 * D};
        j[nj++] = {Ly.idx_comp_gate.q, xq, x, csc1, nullptr, 1.0f, Ly.idx_comp_gate.type, E, 2 * g.idx_key, 0, 2 * D};
        j[nj++] = {Ly.idx_proj_f, nullptr, x, iwb, nullptr, 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads)),
                   gf::kTypeF32, E, g.idx_heads};
    }
    if (!gf::mvn(j, nj, T, s)) { err = "deepseek4 verify: attention projections"; return false; }
    // 2. per token, in position order: norms, RoPE, the ring writes, a block end's compression
    for (int t = 0; t < T; ++t) {
        const int p = (int) (p0 + tb + t);
        ds::PrepArgs a;
        a.qr_raw = qr_raw + (size_t) t * g.q_lora;
        a.q_a_norm = Ly.q_a_norm;
        a.qr = qr + (size_t) t * g.q_lora;
        a.qr_q = (uint8_t*) qr_qb + (size_t) t * (g.q_lora / 32) * kQ8;
        a.q_lora = g.q_lora;
        a.kv_raw = kv_raw + (size_t) t * D;
        a.kv_norm = Ly.kv_a_norm;
        a.raw = Ly.raw;
        a.ring = ds_ring_;
        a.head_dim = D;
        a.n_rot = g.n_rot;
        a.rope = rope;
        a.comp_rope = crope;
        a.p = p;
        a.eps = g.norm_eps;
        if (r != 0) {
            ds::Comp& c = a.comp[a.n_comp];
            c.ring_kv = Ly.ring_kv;
            c.ring_sc = Ly.ring_sc;
            c.cache = Ly.comp_cache;
            c.ape = Ly.comp_ape;
            c.norm = Ly.comp_norm;
            c.D = D;
            c.ratio = r;
            c.overlap = r == 4;
            a.comp_kv[a.n_comp] = ckv0 + (size_t) t * 2 * D;
            a.comp_sc[a.n_comp] = csc0 + (size_t) t * 2 * D;
            ++a.n_comp;
        }
        if (r == 4) {
            ds::Comp& c = a.comp[a.n_comp];
            c.ring_kv = Ly.iring_kv;
            c.ring_sc = Ly.iring_sc;
            c.cache = Ly.idx_cache;
            c.ape = Ly.idx_comp_ape;
            c.norm = Ly.idx_comp_norm;
            c.D = g.idx_key;
            c.ratio = 4;
            c.overlap = 1;
            a.comp_kv[a.n_comp] = ckv1 + (size_t) t * 2 * D;
            a.comp_sc[a.n_comp] = csc1 + (size_t) t * 2 * D;
            ++a.n_comp;
        }
        ds::prep(a, s);
        if (r != 0 && (p + 1) % r == 0) ds::compress(a, s);
    }
    // 3. q_b (and the indexer's q), every column at once
    gf::MvJob j2[2];
    int nj2 = 0;
    j2[nj2++] = {Ly.q_b.q, qr_qb, qr, qb, nullptr, 1.0f, Ly.q_b.type, g.q_lora, HD};
    if (r == 4)
        j2[nj2++] = {Ly.idx_q_b_q.q, qr_qb, qr, iqb, nullptr, 1.0f, Ly.idx_q_b_q.type, g.q_lora,
                     g.idx_heads * g.idx_key};
    if (!gf::mvn(j2, nj2, T, s)) { err = "deepseek4 verify: q_b"; return false; }
    // 4. per token: the per-head norm + RoPE, the indexer's top-k, the attention
    for (int t = 0; t < T; ++t) {
        const int c = tb + t, p = (int) (p0 + c);
        float* q = qb + (size_t) t * HD;
        float* iq = iqb + (size_t) t * g.idx_heads * g.idx_key;
        ds::q_prep(q, g.n_head, D, g.n_rot, rope, r == 4 ? iq : nullptr, g.idx_heads, g.idx_key, crope, p, g.norm_eps,
                   s);
        const int n_raw = std::min(p + 1, g.n_swa);
        int n_comp = 0;
        const int* sel = nullptr;
        if (r != 0) {
            const int n_vis = (p + 1) / r;
            static const bool idx_all = getenv("STRATA_DS4_IDX_ALL") != nullptr && std::atoi(getenv("STRATA_DS4_IDX_ALL")) != 0;
            if (r == 4 && n_vis > 0 && (idx_all || n_vis > g.idx_top_k)) {   // (as the token path: all rows below top_k)
                int* cs = V->sel + (size_t) c * g.idx_top_k;
                ds::indexer(iq, Ly.idx_cache, iwb + (size_t) t * g.idx_heads, g.idx_heads, g.idx_key, n_vis,
                            g.idx_top_k, V->score + (size_t) c * V->score_n, cs, s);
                n_comp = std::min(n_vis, g.idx_top_k);
                sel = cs;
            } else {
                n_comp = n_vis;
            }
        }
        ds::attend(q, Ly.raw, ds_ring_, n_raw, Ly.comp_cache, sel, n_comp, Ly.sinks, g.n_head, D, g.n_rot, rope, p,
                   ob + (size_t) t * HD, s, c, (uint8_t*) oqb + (size_t) t * (HD / 32) * kQ8);
    }
    // 5. the grouped output LoRA, every column at once (a group's rows of column t at o + t * HD + gi * gin)
    const int gin = g.n_head / g.o_groups * D;
    const size_t wrow = gf::row_bytes(Ly.out_a.type, gin);
    if (wrow == 0 || g.o_groups > gf::kMaxMvJobs) { err = "deepseek4 verify: wo_a type"; return false; }
    gf::MvJob jo[gf::kMaxMvJobs];
    for (int gi = 0; gi < g.o_groups; ++gi)
        jo[gi] = {(const uint8_t*) Ly.out_a.q + (size_t) gi * g.o_lora * wrow,
                  (const uint8_t*) oqb + (size_t) gi * (gin / 32) * kQ8, ob + (size_t) gi * gin,
                  oab + (size_t) gi * g.o_lora, nullptr, 1.0f, Ly.out_a.type, gin, g.o_lora, HD / 32, OL};
    if (!gf::mvn(jo, g.o_groups, T, s)) { err = "deepseek4 verify: wo_a"; return false; }
    gf::quantize_q8_1(oab, oaqb, T * OL, s);
    gf::MvJob jb = {Ly.out_b.q, oaqb, oab, mixer, nullptr, 1.0f, Ly.out_b.type, OL, E};
    if (!gf::mvn(&jb, 1, T, s)) { err = "deepseek4 verify: wo_b"; return false; }
    return true;
}

// deepseek41's attention for columns tb.. (fast_dsv41_attn with the dense GEMVs over the columns): the projections,
// then a token at a time in position order the norms, RoPE and window ring rows, a KV source's slot row (its pending
// pair in the pair's own ring entry), q's RoPE, the index scores / candidate blocks / top-k of an index source (into the
// column's own selection: the layers after it read that), the attention; then the output LoRA over the columns
bool Glm5Model::verify_attn_v41(int il, int64_t p0, int tb, int T, cudaStream_t s, std::string& err) {
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const int D = g.head_dim, r = Ly.ratio, E = g.n_embd;
    const ds::Rope rope = layer_rope_v(g, r != 0), crope = layer_rope_v(g, true);
    const int HD = g.n_head * D, OL = g.o_groups * g.o_lora;
    const size_t q8e = (size_t) (E / 32) * kQ8;
    void* const xq = (uint8_t*) V->xq + tb * q8e;
    float* const x = V->x + (size_t) tb * E;
    float* const qr_raw = V->qr_raw + (size_t) tb * g.q_lora;
    float* const qr = V->qr + (size_t) tb * g.q_lora;
    void* const qr_qb = (uint8_t*) V->qr_q + (size_t) tb * (g.q_lora / 32) * kQ8;
    float* const kv_raw = V->kv_raw + (size_t) tb * D;
    float* const iwb = V->iw + (size_t) tb * g.idx_heads;
    float* const ckv0 = V->ckv[0] + (size_t) tb * 2 * D;
    float* const csc0 = V->csc[0] + (size_t) tb * 2 * D;
    float* const qb = V->q + (size_t) tb * HD;
    float* const iqb = V->iq + (size_t) tb * g.idx_heads * g.idx_key;
    float* const ob = V->o + (size_t) tb * HD;
    void* const oqb = (uint8_t*) V->oq + (size_t) tb * (HD / 32) * kQ8;
    float* const oab = V->oa + (size_t) tb * OL;
    void* const oaqb = (uint8_t*) V->oaq + (size_t) tb * (OL / 32) * kQ8;
    float* const mixer = V->mixer + (size_t) tb * E;
    // 1. the projections of x: q_a, kv, the compressor (KV sources), the indexer's weights (index sources)
    gf::MvJob j[gf::kMaxMvJobs];
    int nj = 0;
    j[nj++] = {Ly.q_a.q, xq, x, qr_raw, nullptr, 1.0f, Ly.q_a.type, E, g.q_lora};
    j[nj++] = {Ly.kv.q, xq, x, kv_raw, nullptr, 1.0f, Ly.kv.type, E, D};
    if (Ly.kv_src) {
        j[nj++] = {Ly.comp_kv.q, xq, x, ckv0, nullptr, 1.0f, Ly.comp_kv.type, E, D, 0, 2 * D};
        if (r == 2) j[nj++] = {Ly.comp_gate.q, xq, x, csc0, nullptr, 1.0f, Ly.comp_gate.type, E, D, 0, 2 * D};
    }
    if (Ly.idx_src)
        j[nj++] = {Ly.idx_proj_f, nullptr, x, iwb, nullptr, 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads)),
                   gf::kTypeF32, E, g.idx_heads};
    if (!gf::mvn(j, nj, T, s)) { err = "deepseek41 verify: attention projections"; return false; }
    // 2. per token, in position order: qr's norm, kv's norm + RoPE into the window ring, then the KV source's row
    for (int t = 0; t < T; ++t) {
        const int p = (int) (p0 + tb + t);
        ds::PrepArgs a;
        a.qr_raw = qr_raw + (size_t) t * g.q_lora;
        a.q_a_norm = Ly.q_a_norm;
        a.qr = qr + (size_t) t * g.q_lora;
        a.qr_q = (uint8_t*) qr_qb + (size_t) t * (g.q_lora / 32) * kQ8;
        a.q_lora = g.q_lora;
        a.kv_raw = kv_raw + (size_t) t * D;
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
            pa.v = ckv0 + (size_t) t * 2 * D;
            pa.g = r == 2 ? csc0 + (size_t) t * 2 * D : nullptr;
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
    }
    // 3. q_b (and the indexer's q), every column at once
    gf::MvJob j2[2];
    int nj2 = 0;
    j2[nj2++] = {Ly.q_b.q, qr_qb, qr, qb, nullptr, 1.0f, Ly.q_b.type, g.q_lora, HD};
    if (Ly.idx_src)
        j2[nj2++] = {Ly.idx_q_b_q.q, qr_qb, qr, iqb, nullptr, 1.0f, Ly.idx_q_b_q.type, g.q_lora,
                     g.idx_heads * g.idx_key};
    if (!gf::mvn(j2, nj2, T, s)) { err = "deepseek41 verify: q_b"; return false; }
    // 4. per token: RoPE (no per-head norm), the keys - the window, and the slot's visible rows (all of them up to
    // top_k; beyond, the column's top-k: this layer's own when it is an index source, its last index source's
    // otherwise) - and the attention
    for (int t = 0; t < T; ++t) {
        const int c = tb + t, p = (int) (p0 + c);
        float* q = qb + (size_t) t * HD;
        float* iq = iqb + (size_t) t * g.idx_heads * g.idx_key;
        d41::q_rope(q, g.n_head, D, g.n_rot, rope, Ly.idx_src ? iq : nullptr, g.idx_heads, g.idx_key, crope, p, s);
        const int n_raw = std::min(p + 1, g.n_swa);
        int n_comp = 0;
        const int* sel = nullptr;
        if (r != 0) {
            const int n_vis = (p + 1) / r;
            if (n_vis > g.idx_top_k) {
                int* cs = V->sel + (size_t) c * g.idx_top_k;
                if (Ly.idx_src) {
                    float* sc = V->v41_score + (size_t) c * V->v41_score_n;
                    unsigned char* keep = V->v41_keep + (size_t) c * V->v41_b_n;
                    ds::indexer_scores(iq, Ly.idx_cache, iwb + (size_t) t * g.idx_heads, g.idx_heads, n_vis, sc, s);
                    if (il == g.cand_layer && n_vis > g.cand_blocks * g.cand_bs)
                        d41::candidates(sc, n_vis, g.cand_bs, g.cand_blocks, V->v41_bscore + (size_t) c * V->v41_b_n,
                                        V->v41_bsel + (size_t) c * g.cand_blocks, keep, s);
                    if (il > g.cand_layer && n_vis > g.cand_blocks * g.cand_bs)
                        d41::candidate_mask(sc, n_vis, g.cand_bs, keep, s);
                    ds::indexer_select(sc, n_vis, g.idx_top_k, cs, s);
                }
                n_comp = g.idx_top_k;
                sel = cs;
            } else {
                n_comp = n_vis;
            }
        }
        ds::attend(q, Ly.raw, ds_ring_, n_raw, Ly.comp_cache, sel, n_comp, Ly.sinks, g.n_head, D, g.n_rot, rope, p,
                   ob + (size_t) t * HD, s, c, (uint8_t*) oqb + (size_t) t * (HD / 32) * kQ8);
    }
    // 5. the grouped output LoRA, every column at once (a group's rows of column t at o + t * HD + gi * gin)
    const int gin = g.n_head / g.o_groups * D;
    const size_t wrow = gf::row_bytes(Ly.out_a.type, gin);
    if (wrow == 0 || g.o_groups > gf::kMaxMvJobs) { err = "deepseek41 verify: wo_a type"; return false; }
    gf::MvJob jo[gf::kMaxMvJobs];
    for (int gi = 0; gi < g.o_groups; ++gi)
        jo[gi] = {(const uint8_t*) Ly.out_a.q + (size_t) gi * g.o_lora * wrow,
                  (const uint8_t*) oqb + (size_t) gi * (gin / 32) * kQ8, ob + (size_t) gi * gin,
                  oab + (size_t) gi * g.o_lora, nullptr, 1.0f, Ly.out_a.type, gin, g.o_lora, HD / 32, OL};
    if (!gf::mvn(jo, g.o_groups, T, s)) { err = "deepseek41 verify: wo_a"; return false; }
    gf::quantize_q8_1(oab, oaqb, T * OL, s);
    gf::MvJob jb = {Ly.out_b.q, oaqb, oab, mixer, nullptr, 1.0f, Ly.out_b.type, OL, E};
    if (!gf::mvn(&jb, 1, T, s)) { err = "deepseek41 verify: wo_b"; return false; }
    abl_apply(il, 0, mixer, T, E, s);
    return true;
}

// verify_moe's first launch: the router's logits (and, STRATA_GLM_SH_LATE=0, the shared expert's gate / up) of the
// columns tb .. tb + T - 1 - apart, so a layer's graph can carry it (verify_part_q4x)
bool Glm5Model::verify_moe_router(int il, int tb, int T, cudaStream_t s, std::string& err) {
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, FFs = g.n_ff_exp * g.n_shared;
    const size_t q8e = (size_t) (E / 32) * kQ8;
    gf::MvJob j[3];
    j[0] = {Ly.router, nullptr, V->x + (size_t) tb * E, V->rlog + (size_t) tb * g.n_expert, nullptr, 1.0f,
            Ly.router_type, E, g.n_expert};   // (as fast_moe: deepseek4's BF16, deepseek41's F32)
    j[1] = {Ly.sh_gate.q, (uint8_t*) V->xq + tb * q8e, nullptr, V->sh_g + (size_t) tb * FFs, nullptr, 1.0f,
            Ly.sh_gate.type, E, FFs};
    j[2] = {Ly.sh_up.q, (uint8_t*) V->xq + tb * q8e, nullptr, V->sh_u + (size_t) tb * FFs, nullptr, 1.0f,
            Ly.sh_up.type, E, FFs};
    const int nj = verify_sh_late() && FFs > 0 ? 1 : 3;
    if (!(g.arch == 0 ? glm_mvn(j, nj, T, s) : g.q4x ? q4x_mvn(j, nj, T, s) : gf::mvn(j, nj, T, s))) { err = "deepseek4 verify: router / shared expert"; return false; }
    return true;
}

void Glm5Model::verify_report() const {
    const auto* V = (const VerifyBuf*) vb_;
    if (V == nullptr || V->q4g_rep + V->q4g_eager == 0) return;
    size_t n = 0;
    for (auto e : V->q4g) n += e != nullptr;
    std::fprintf(stderr, "qwen4exp verify graphs CUDA%d: %llu layer segments replayed, %llu launched (%.1f%% replayed), %zu "
                         "graphs, %llu re-captured for new route arguments; captures with PDL edges %llu, refused %llu\n",
                 dev_, V->q4g_rep, V->q4g_eager, 100.0 * (double) V->q4g_rep / (double) (V->q4g_rep + V->q4g_eager), n,
                 V->q4g_recap, V->q4g_pdl_ok, V->q4g_pdl_no);
}

bool Glm5Model::verify_sh_late() {
    static const bool sh_late = [] {
        const char* v = getenv("STRATA_GLM_SH_LATE");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return sh_late;
}

bool Glm5Model::verify_moe(int il, int tb, int T, const std::vector<int32_t>& toks, cudaStream_t s,
                           cudaEvent_t wait_ev, cudaEvent_t done_ev, std::string& err, bool router_done, bool graphed) {
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, FFs = g.n_ff_exp * g.n_shared, k = g.n_exp_used;
    const size_t q8e = (size_t) (E / 32) * kQ8;
    gf::MvJob j[3];
    j[0] = {Ly.router, nullptr, V->x + (size_t) tb * E, V->rlog + (size_t) tb * g.n_expert, nullptr, 1.0f,
            Ly.router_type, E, g.n_expert};   // (as fast_moe: deepseek4's BF16, deepseek41's F32)
    j[1] = {Ly.sh_gate.q, (uint8_t*) V->xq + tb * q8e, nullptr, V->sh_g + (size_t) tb * FFs, nullptr, 1.0f,
            Ly.sh_gate.type, E, FFs};
    j[2] = {Ly.sh_up.q, (uint8_t*) V->xq + tb * q8e, nullptr, V->sh_u + (size_t) tb * FFs, nullptr, 1.0f,
            Ly.sh_up.type, E, FFs};
    // the shared expert's gate/up after the routes, as the token path does (fast_moe, STRATA_GLM_SH_LATE)
    const bool late = verify_sh_late() && FFs > 0;
    if (!router_done && !verify_moe_router(il, tb, T, s, err)) return false;
    const bool lane = F->cpu_plan != 0ull && F->cpu_fmt[(size_t) il].n_ff > 0;
    static const int promote_min = [] {
        const char* v = getenv("STRATA_GLM_PROMOTE_MIN");
        return v ? std::max(0, std::atoi(v)) : 0;
    }();
    static const int pf_rank = [] {
        const char* v = getenv("STRATA_GLM_PREFETCH_RANK");
        return v != nullptr ? std::max(1, std::atoi(v)) : 0;
    }();
    // every token's route first: the CPU lane has the whole window's RAM-tier experts queued before any wait
    if (wait_ev != nullptr) cudaStreamWaitEvent(s, wait_ev, 0);
    // the expert helper's experts go to the CPU lane's request too (it hands them to the helper GPU, #24)
    const uint8_t* helper_owned = F->expert_helper && F->expert_helper->active && !F->expert_helper->failed ?
                                  F->expert_helper->owned_d : nullptr;
    const unsigned int helper_promote = helper_owned ? expert_helper_promote() : 0u;
    for (int t = tb; t < tb + T; ++t) {
        V->md[(size_t) t].helper_owned = helper_owned;
        V->md[(size_t) t].helper_promote = helper_promote;
    }
    // STRATA_GLM_ROUTE_COLS (default on for qwen4exp): the columns' routes in one launch (gf::moe_route_cols: the same
    // plans in the same order; a launch a column took 48 x T launches a window, ~13 us each on a DGX Spark)
    static const int route_cols = [] {
        const char* v = getenv("STRATA_GLM_ROUTE_COLS");
        return v == nullptr ? -1 : std::atoi(v) != 0 ? 1 : 0;
    }();
    bool routed = false;
    if ((route_cols < 0 ? g.q4x : route_cols != 0) && T > 1 && (late || FFs == 0) && V->route_sync != nullptr &&
        !F->prof_on) {
        const int* hid[8] = {};
        for (int t = 0; t < T && t < 8; ++t)
            hid[t] = Ly.tid2eid != nullptr ? Ly.tid2eid + (size_t) toks[(size_t) (tb + t)] * k : nullptr;
        routed = gf::moe_route_cols(V->rlog + (size_t) tb * g.n_expert, (size_t) g.n_expert, Ly.router_bias, g.n_expert,
                                    k, g.w_scale, g.norm_w != 0, il, V->x + (size_t) tb * E, (size_t) E,
                                    V->md.data() + tb, T, V->route_sync, s, fast_skip_from(il),
                                    lane ? F->cpu_plan : 0ull, promote_min, pf_rank, g.gating, hid, fast_skip_w(il),
                                    F->tail_skip_disk ? 1 : 0);
        if (routed) F->expected += T;
    }
    for (int t = tb; t < tb + T && !routed; ++t) {
        gf::moe_route(V->rlog + (size_t) t * g.n_expert, Ly.router_bias, g.n_expert, k, g.w_scale, g.norm_w != 0, il,
                      V->x + (size_t) t * E, E, V->md[(size_t) t], late ? nullptr : V->sh_g + (size_t) t * FFs,
                      late ? nullptr : V->sh_u + (size_t) t * FFs, g.swiglu_shexp, late ? 0 : FFs,
                      (uint8_t*) V->sh_hq + (size_t) t * (FFs / 32) * kQ8,
                      s, nullptr, nullptr, 0, nullptr, nullptr, 0, fast_skip_from(il), lane ? F->cpu_plan : 0ull, promote_min, pf_rank,
                      g.gating, Ly.tid2eid != nullptr ? Ly.tid2eid + (size_t) toks[(size_t) t] * k : nullptr,
                      fast_skip_w(il), F->tail_skip_disk ? 1 : 0, nullptr, tb + T - 1 - t);   // (the columns after
                                    // this one: the CPU lane answers a batched window's columns together; the disk
                                    // tail skip as the token path's)
        ++F->expected;
    }
    if (late) {
        if (!(g.arch == 0 ? glm_mvn(j + 1, 2, T, s) : g.q4x ? q4x_mvn(j + 1, 2, T, s) : gf::mvn(j + 1, 2, T, s))) { err = "deepseek4 verify: shared expert"; return false; }
        gf::sh_swiglu(V->sh_g + (size_t) tb * FFs, V->sh_u + (size_t) tb * FFs, g.swiglu_shexp, T * FFs,
                      (uint8_t*) V->sh_hq + (size_t) tb * (FFs / 32) * kQ8, s);
    }
    if (F->prof_on) F->mark("vm_route_shexp");
    // ... then each token's resident experts (its own plan and h) ...
    static const bool wait_fetch = [] {   // STRATA_GLM_WAIT_FETCH, as fast_moe's
        const char* v = getenv("STRATA_GLM_WAIT_FETCH");
#if defined(STRATA_USE_HIP)
        return v == nullptr || std::atoi(v) != 0;
#else
        return v != nullptr && std::atoi(v) != 0;
#endif
    }();
    // STRATA_DS4_WIN_COLS (default 1): the columns' resident experts in one gate/up and one down launch (the same
    // blocks as a launch a column: the same sums) - a column's ~4 resident experts a layer left the GPU half idle
    static const bool win_cols = [] {
        const char* v = getenv("STRATA_DS4_WIN_COLS");
        return v == nullptr || std::atoi(v) != 0;
    }();
    // (deepseek41: its SSD reads land in the column's scratch set - a column a set up to 1 + n_scr columns; a longer
    // window keeps each column's experts right after its own fetch)
    const bool cols = win_cols && T > 1 && V->cols_d != nullptr && !F->prof_on && T <= 1 + V->n_scr;
    // (any model: a fetch with no spare left lands in its column's scratch set, so two columns sharing a set would
    // have the later fetch overwrite the earlier one's experts before the batched launches read them - qwen4exp's
    // 5-column MTP windows shared set 3 among columns 3-5 until 2026-10-31; a launch a column reads each in time)
    if (g.qwen35() && Ly.sh_gate_w != nullptr) {   // the shared expert's gate logit a column (sigmoid below)
        gf::MvJob jg = {Ly.sh_gate_w, nullptr, V->x + (size_t) tb * E, V->q35_shg + tb, nullptr, 1.0f, gf::kTypeF32, E, 1};
        if (!(g.q4x ? q4x_mvn(&jg, 1, T, s) : gf::mvn(&jg, 1, T, s))) { err = "qwen35 verify: the shared expert's gate"; return false; }
    }
    // STRATA_DS4_WIN_COLS_CHECK=1: every 16th group of columns also through a launch a column, compared bit by bit
    static const bool cols_chk = getenv("STRATA_DS4_WIN_COLS_CHECK") != nullptr && std::atoi(getenv("STRATA_DS4_WIN_COLS_CHECK")) != 0;
    const bool chk = cols && cols_chk && (V->chk_calls++ % 16) == 0;
    if (chk && V->ffn_chk == nullptr && cudaMalloc((void**) &V->ffn_chk, (size_t) kVerifyMax * E * sizeof(float)) != cudaSuccess) {
        err = "deepseek4 verify: the check buffer";
        return false;
    }
    // STRATA_GLM_FETCH_COLS (default 1): with the column-batched experts, the columns' waits and fetches in one
    // launch (moe_wait_fetch_cols: the same copies); 0: a moe_wait and a moe_fetch a column
    // (default on for qwen4exp only: DeepSeek V4/V4.1's SSD-streamed fetches share the SMs differently - not measured)
    static const int fetch_cols = [] {
        const char* v = getenv("STRATA_GLM_FETCH_COLS");
        return v == nullptr ? -1 : std::atoi(v) != 0 ? 1 : 0;
    }();
    const bool fcols = (fetch_cols < 0 ? g.q4x : fetch_cols != 0) && cols && V->md_d != nullptr && !F->prof_on;
    if (fcols) {
        if (F->stream_fetch)
            gf::moe_wait_fetch_cols(V->md_d + tb, T, E, k, Ly.blob, s, Ly.gu_bytes, Ly.dn_bytes, F->read_chunks,
                                    Ly.up_bytes);
        else
            gf::moe_wait_fetch_cols(V->md_d + tb, T, E, k, Ly.blob, s);
        if (done_ev != nullptr) cudaEventRecord(done_ev, s);
    }
    for (int t = tb; t < tb + T && !fcols; ++t) {
        gf::MoeDev md = V->md[(size_t) t];
        md.nv_scale = Ly.nv_scale;   // (NVFP4 experts; null otherwise)
        float* sh_out = V->sh_out + (size_t) t * E;
        void* hq = (uint8_t*) V->hq + (size_t) t * V->hq_n;
        if (wait_fetch && !F->prof_on) {   // (one launch: gf::moe_wait_fetch, as fast_moe; streamed as its)
            if (F->stream_fetch) gf::moe_wait_fetch(md, E, k, Ly.blob, s, Ly.gu_bytes, Ly.dn_bytes, F->read_chunks,
                                                    Ly.up_bytes);
            else gf::moe_wait_fetch(md, E, k, Ly.blob, s);
        } else {
            gf::moe_wait(md, E, s);
            if (F->prof_on) F->mark("vm_wait");
            if (F->stream_fetch) gf::moe_fetch(md, k, Ly.blob, s, Ly.gu_bytes, Ly.dn_bytes, F->read_chunks,
                                                    Ly.up_bytes);
            else gf::moe_fetch(md, k, Ly.blob, s);
            if (F->prof_on) F->mark("vm_fetch");
        }
        // the next route may find this one's promotions resident: it goes after their copies have landed
        if (done_ev != nullptr && t == tb + T - 1) cudaEventRecord(done_ev, s);
        if (cols) continue;   // (the batched launches below; the check runs these after them)
        gf::moe_gate_up(Ly.gu_type, md, k, E, g.n_ff_exp, g.swiglu_exp, (const uint8_t*) V->xq + (size_t) t * q8e,
                        hq, Ly.sh_down.q, Ly.sh_down.type, (const uint8_t*) V->sh_hq + (size_t) t * (FFs / 32) * kQ8,
                        FFs, sh_out, s, Ly.up_type);
        if (Ly.sh_down.q != nullptr) abl_apply(il, 1, sh_out, 1, E, s);
        if (g.qwen35() && Ly.sh_down.q != nullptr && Ly.sh_gate_w != nullptr)   // (as fast_moe)
            q35::scale_sigmoid(sh_out, E, V->q35_shg + t, s);
        gf::moe_down(Ly.d_type, md, k, E, g.n_ff_exp, Ly.down_off, hq, Ly.sh_down.q != nullptr ? sh_out : nullptr,
                     V->ffn + (size_t) t * E, s);
        if (F->prof_on) F->mark("vm_experts");
    }
    if (cols) {
        const size_t shq = (size_t) (FFs / 32) * kQ8;
        float* sh_out = V->sh_out + (size_t) tb * E;
        void* hq = (uint8_t*) V->hq + (size_t) tb * V->hq_n;
        gf::moe_gate_up_cols(Ly.gu_type, V->cols_d + tb, T, k, E, g.n_ff_exp, g.swiglu_exp,
                             (const uint8_t*) V->xq + (size_t) tb * q8e, q8e, hq, V->hq_n, Ly.sh_down.q, Ly.sh_down.type,
                             (const uint8_t*) V->sh_hq + (size_t) tb * shq, shq, FFs, sh_out, E, s, Ly.nv_scale,
                             Ly.up_type);
        if (Ly.sh_down.q != nullptr) abl_apply(il, 1, sh_out, T, E, s);
        if (g.qwen35() && Ly.sh_down.q != nullptr && Ly.sh_gate_w != nullptr)   // (the shared expert's gate a column)
            q35::scale_sigmoid_rows(sh_out, T, E, V->q35_shg + tb, s);
        gf::moe_down_cols(Ly.d_type, V->cols_d + tb, T, k, E, g.n_ff_exp, Ly.down_off, hq, V->hq_n,
                          Ly.sh_down.q != nullptr ? sh_out : nullptr, chk ? V->ffn_chk : V->ffn + (size_t) tb * E, E, s,
                          Ly.nv_scale);
        if (chk) {   // the same inputs through a launch a column into V->ffn, then the rows compared
            for (int t = tb; t < tb + T; ++t) {
                gf::MoeDev md = V->md[(size_t) t];
        md.nv_scale = Ly.nv_scale;   // (NVFP4 experts; null otherwise)
                float* sh_o = V->sh_out + (size_t) t * E;
                void* hq1 = (uint8_t*) V->hq + (size_t) t * V->hq_n;
                gf::moe_gate_up(Ly.gu_type, md, k, E, g.n_ff_exp, g.swiglu_exp, (const uint8_t*) V->xq + (size_t) t * q8e,
                                hq1, Ly.sh_down.q, Ly.sh_down.type, (const uint8_t*) V->sh_hq + (size_t) t * shq, FFs, sh_o, s,
                                Ly.up_type);
                if (Ly.sh_down.q != nullptr) abl_apply(il, 1, sh_o, 1, E, s);
                if (g.qwen35() && Ly.sh_down.q != nullptr && Ly.sh_gate_w != nullptr)
                    q35::scale_sigmoid(sh_o, E, V->q35_shg + t, s);
                gf::moe_down(Ly.d_type, md, k, E, g.n_ff_exp, Ly.down_off, hq1, Ly.sh_down.q != nullptr ? sh_o : nullptr,
                             V->ffn + (size_t) t * E, s);
            }
            std::vector<float> a((size_t) T * E), b((size_t) T * E);
            cudaMemcpyAsync(a.data(), V->ffn_chk, a.size() * sizeof(float), cudaMemcpyDeviceToHost, s);
            cudaMemcpyAsync(b.data(), V->ffn + (size_t) tb * E, b.size() * sizeof(float), cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
            for (int t = 0; t < T; ++t) {
                ++V->chk_cols;
                if (std::memcmp(a.data() + (size_t) t * E, b.data() + (size_t) t * E, (size_t) E * sizeof(float)) != 0) ++V->chk_diff;
            }
            if ((V->chk_calls & 1023) == 1)
                std::fprintf(stderr, "deepseek4 verify: column-batched experts checked - %llu columns, %llu differ\n",
                             (unsigned long long) V->chk_cols, (unsigned long long) V->chk_diff);
        }
    }
    // ... and the CPU lane's answers, in request order.  Deferred (F->defer, as fast_ffn): a column's part is
    // written into its streams after the next layer's attention (verify's hc_defer_add) with these write weights,
    // saved now; the trunk's (or this part's) last layer waits for its own
    static const bool hc_cpu_env = [] {   // STRATA_GLM_HC_CPU, as fast_moe's
        const char* v = getenv("STRATA_GLM_HC_CPU");
#if defined(STRATA_USE_HIP)
        return v == nullptr || std::atoi(v) != 0;
#else
        return v != nullptr && std::atoi(v) != 0;
#endif
    }();
    const bool hc_cpu = lane && hc_cpu_env && !F->xc && !F->defer && !Ly.mtp && !F->prof_on && il != l1_ - 1 && !abl_full() &&
                        !(g.v41 && F->L[(size_t) il + 1].eng >= 0) && gf::hc_fuses_cpu(E);   // (as fast_moe)
    if (F->defer && Ly.moe && !Ly.mtp && il != l1_ - 1) {
        if (lane)
            for (int t = tb; t < tb + T; ++t) gf::copy_floats(V->post_defer + (size_t) t * 8, V->post + (size_t) t * 8, 8, s);
    } else if (hc_cpu) {
        for (int t = tb; t < tb + T; ++t) V->hc_cpu[t] = true;
    } else if (lane) {
        // STRATA_GLM_CPU_WAIT_COLS=1/0: the columns' answers in one launch (the same adds) / a launch a column.
        // Default: on for qwen4exp only (measured there)
        static const int wait_cols = [] {
            const char* v = getenv("STRATA_GLM_CPU_WAIT_COLS");
            return v == nullptr ? -1 : std::atoi(v) != 0 ? 1 : 0;
        }();
        if ((wait_cols < 0 ? g.q4x : wait_cols != 0) && T > 1 && T <= 8 && V->md[(size_t) tb].cpu_seq != nullptr) {
            gf::CpuWaitCols c;
            c.n = T;
            c.k = k;
            for (int t = 0; t < T; ++t) {
                c.seq[t] = V->md[(size_t) (tb + t)].cpu_seq;
                c.out[t] = V->ffn + (size_t) (tb + t) * E;
                c.xpart[t] = V->md[(size_t) (tb + t)].xpart;
            }
            gf::moe_cpu_wait_cols(c, V->md[(size_t) tb].cpu_ans, E, s);
        } else {
            for (int t = tb; t < tb + T; ++t) gf::moe_cpu_wait(V->md[(size_t) t], E, V->ffn + (size_t) t * E, s, 1, k);
        }
        if (F->prof_on) F->mark("vm_cpu_wait");
    }
    abl_out(il, 1, V->ffn + (size_t) tb * E, T, (int) E, s);   // (complete: an edit keeps hc_cpu off)
    if (graphed && (hc_cpu || (F->defer && Ly.moe && !Ly.mtp && il != l1_ - 1))) {
        err = "a deferred or hc-fused CPU-lane answer";   // (host state the next layer reads: not in a graph)
        return false;
    }
    if (hc_cpu) V->bg_after[tb] = true;
    else if (!graphed) F->bg_drain(s, F->bg_pace);   // the paced promotions' next bytes, as fast_moe (graphed: the caller)
    return true;
}

bool Glm5Model::verify(const std::vector<int32_t>& toks, std::string& err) {
    const int T = (int) toks.size();
    if (T < 1 || T > kVerifyMax) {
        err = "deepseek4 verify: 1.." + std::to_string(kVerifyMax) + " tokens";
        return false;
    }
    if (!verify_ready()) {
        err = "deepseek4 verify: not available (deepseek4 fast path, no MTP / prefetch / lookahead)";
        return false;
    }
    if (pos_ + T > max_ctx_) {
        err = "deepseek4 verify: the window exceeds max_ctx";
        return false;
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (!m->verify_alloc(err)) return false;
    }
    cudaSetDevice(dev_);
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    // the window's kind: STRATA_DS4_SPEC_PIPE=1 pipelined, 0 batched (2: pipelined on one stream, debug), auto: the
    // faster on this machine - the first windows alternate, 3 of each, then the kind with the lower median time a token
    // stays.  Unset: auto on HIP, pipelined on NVIDIA (the timing is not checked on an NVIDIA card yet).  An RTX 5090 +
    // iGPU lane: pipelined (90 against 73 tok/s on swe-bench prompts).  An AI PRO R9700 + iGPU lane: batched (2-token
    // windows 30.6 against 44.5 ms, swe-bench 6 tasks 42 -> 46-50 tok/s) - HIP put every stream in one hardware queue,
    // so the pipelined window's tokens ran one after another, each reading the dense weights
    static const int pipe_env = [] {
        const char* v = getenv("STRATA_DS4_SPEC_PIPE");
        if (v != nullptr) return std::strcmp(v, "auto") == 0 ? -1 : std::atoi(v);
#if defined(STRATA_USE_HIP)
        return -1;
#else
        return 1;
#endif
    }();
    bool timed = pipe_env < 0 && V->pipe_choice[T] < 0 && T > 1 && !F->prof_on && !g.v41 && !g.qwen35() && g.arch != 0;
    const bool use_pipe = g.v41 || g.qwen35() || g.arch == 0 ? false   // (deepseek41, qwen35moe, glm5-next: batched)
                        : pipe_env >= 0 ? pipe_env != 0
                        : V->pipe_choice[T] >= 0 ? V->pipe_choice[T] != 0
                        : V->ms_pipe[T].size() <= V->ms_batch[T].size();   // (timing: alternate a width, pipelined first)
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        auto* VM = (VerifyBuf*) m->vb_;
        VM->pipe_now = use_pipe;
        timed = timed && VM->pipe;   // (no scratch slots: every window is batched)
    }
    dspark_settle();
    const int64_t p0 = pos_;
    pos_ += T;
    dspark_window(p0, T);   // (DSpark: the kept columns into its context at spec_rewind)
    const bool gp = F->gap_on && split_next_ == nullptr;   // (as the token path: one device's events)
    if (gp) {
        cudaEventRecord(F->ev_g0, s);
        if (F->gap_live)
            F->gap_host += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - F->t_ready).count();
    }
    // between windows (every device idle): finished promotions go live in the device tables
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (!m->fast_boundary(err)) return false;
    }
    cudaSetDevice(dev_);
    const auto t0 = std::chrono::steady_clock::now();
    // STRATA_DS4_WIN_TIMING=1 (investigation): the host's time a window - to the layers' first launch, the launches,
    // the wait for the device - as means every 32 windows
    static const bool win_timing = getenv("STRATA_DS4_WIN_TIMING") != nullptr && std::atoi(getenv("STRATA_DS4_WIN_TIMING")) != 0;
    static double wt_pre = 0.0, wt_eng = 0.0, wt_launch = 0.0, wt_wait = 0.0;
    static int wt_n = 0;
    auto wt_mark = std::chrono::steady_clock::now();
    const auto wt_lap = [&](double& acc) {
        const auto now = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double, std::milli>(now - wt_mark).count();
        wt_mark = now;
    };
    // ---- the embedding rows -> each token's 4 streams (the first part's columns)
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "deepseek4 verify: no embedding dequantizer";
        return false;
    }
    const size_t E = (size_t) g.n_embd, HCE = (size_t) g.hc * E;
    if (F->emb_tab_d != nullptr) {
        for (int t = 0; t < T; ++t) V->tok_in_h[t] = toks[(size_t) t];
        gf::embed_rows_q(pack_emb_type_, F->emb_tab_d, V->tok_in_d, T, g.n_embd, V->emb, s);
    } else {
        for (int t = 0; t < T; ++t)
            tt->to_float(pack_emb_src_ + (size_t) toks[(size_t) t] * ggml_row_size((ggml_type) pack_emb_type_, g.n_embd),
                         V->emb_h + (size_t) t * E, g.n_embd);
        cudaMemcpyAsync(V->emb, V->emb_h, (size_t) T * E * sizeof(float), cudaMemcpyHostToDevice, s);
    }
    abl_res(V->emb, T, (int) E, s);
    for (int t = 0; t < T; ++t) {
        if (g.q4x)   // qwen4exp: four streams a column, the columns contiguous (hc n_embd apart)
            gf::embed_streams(V->emb + (size_t) t * E, V->R + (size_t) t * HCE, g.n_embd, s);
        else if (g.qwen35())   // a plain residual: column t's first stream (embed_streams writes four)
            cudaMemcpyAsync(V->R + (size_t) t * 2 * HCE, V->emb + (size_t) t * E, E * sizeof(float),
                            cudaMemcpyDeviceToDevice, s);
        else
            gf::embed_streams(V->emb + (size_t) t * E, V->R + (size_t) t * 2 * HCE, g.n_embd, s);
    }
    if (g.v41) {   // every column's Engram rows (read while the embedding runs), layer 0's one-hot pre
        static const float onehot[4] = {1.0f, 0.0f, 0.0f, 0.0f};   // (as fast_token: layer 0 collapses stream 0)
        for (int t = 0; t < T; ++t)
            cudaMemcpyAsync(V->v41_pre + (size_t) t * 8 + 4, onehot, sizeof(onehot), cudaMemcpyHostToDevice, s);
        wt_lap(wt_pre);
        if (!dsv41_engram_window(toks.data(), p0, T, err)) return false;
        wt_lap(wt_eng);
    }
    if (gp) {
        cudaEventRecord(F->ev_gs, s);
        F->gap_emb += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    // ---- the parts in order (one part: this one's layers and the head; qwen4exp's split: this part's, which hands the
    //      columns to the next part's port)
    if (next_port_ != nullptr) {
        V->T = T;
        if (!verify_part_q4x(toks, p0, err)) return false;
    }
    for (Glm5Model* m = next_port_ != nullptr ? nullptr : this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        m->pos_ = pos_;
        ((VerifyBuf*) m->vb_)->T = T;
        if (!(g.arch == 0 ? m->verify_part_glm(toks, p0, err)
              : g.v41 ? m->verify_part_v41(toks, p0, err) : g.q4x ? m->verify_part_q4x(toks, p0, err)
              : g.qwen35() ? m->verify_part_q35(toks, p0, err)
                                                                       : m->verify_part(toks, p0, err))) return false;
    }
    wt_lap(wt_launch);
    cudaSetDevice(dev_);
    const Glm5Model* tail = verify_tail();
    auto* VT = (VerifyBuf*) tail->vb_;
    if (gp) cudaEventRecord(F->ev_ge[F->gap_par], s);
    { auto* VS = (VerifyBuf*) verify_tail()->vb_; if (VS) VS->samp_ok = false; }   // (a new window: no stale draws)
    // the drafter's expert-cache uploads (the misses its last draft found) while the window runs: its own slots and
    // table on its own stream, which the next draft waits for (ev_up) - generate.cpp's call after verify() was serial
    if (dsp_ != nullptr) dspark_upload();
    const auto tw = std::chrono::steady_clock::now();
    if (next_port_ != nullptr) {   // (the last part's argmax per column, through the port)
        if (!next_port_->window_wait(win_tok_, err)) return false;
        cudaSetDevice(dev_);
    } else {
        cudaError_t done;
        while ((done = cudaEventQuery(VT->done)) == cudaErrorNotReady) {
            if (F->token_spin) glmfast::cpu_relax(); else std::this_thread::yield();
        }
        if (!glmfast::cuda_ok(done, "deepseek4 verify completion", err)) return false;
    }
    if (win_timing) {
        wt_lap(wt_wait);
        if (++wt_n % 32 == 0)
            std::fprintf(stderr, "deepseek4 verify timing: %d windows, a window: %.2f ms to the layers (Engram rows %.2f), "
                                 "%.2f ms launching them, %.2f ms waiting for the device\n", wt_n, wt_pre / wt_n,
                         wt_eng / wt_n, wt_launch / wt_n, wt_wait / wt_n);
    }
    for (Glm5Model* m = this; m != nullptr; m = next_port_ != nullptr ? nullptr : m->split_next_.get())
        if (!m->fast_->route_ok(err)) return false;   // (the port's parts: in window_wait)
    if (g_.v41 && !dsv41_engram_ok(err)) return false;   // (a window's Engram row the reader never delivered)
    if (gf::launch_errors() > 0) {
        err = "deepseek4 verify: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return false;
    }
    const double wms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    F->ms += wms;
    if (timed) {   // (the window kind still being timed: 3 of each at one width decide)
        (use_pipe ? V->ms_pipe[T] : V->ms_batch[T]).push_back(wms);
        if (V->ms_pipe[T].size() >= 3 && V->ms_batch[T].size() >= 3) {
            const auto median = [](std::vector<double> v) {
                std::sort(v.begin(), v.end());
                return v[v.size() / 2];
            };
            const double mp = median(V->ms_pipe[T]), mb = median(V->ms_batch[T]);
            V->pipe_choice[T] = mp <= mb ? 1 : 0;
            std::fprintf(stderr, "deepseek4 verify: the %s window here at %d columns (%.1f ms against %.1f %s) - kept "
                                 "(STRATA_DS4_SPEC_PIPE=0 batched, 1 pipelined)\n",
                         V->pipe_choice[T] ? "pipelined" : "batched", T, std::min(mp, mb), std::max(mp, mb),
                         V->pipe_choice[T] ? "batched" : "pipelined");
        }
    }
    for (Glm5Model* m = this; m != nullptr; m = next_port_ != nullptr ? nullptr : m->split_next_.get())
        if (m->fast_->prof_on) {
            cudaSetDevice(m->dev_);
            m->fast_->collect();
        }
    if (next_port_ != nullptr) next_port_->collect();
    cudaSetDevice(dev_);
    last_tok_ = next_port_ != nullptr ? win_tok_[(size_t) T - 1] : VT->tok_h[T - 1];
    if (gp) fast_gap_mark(tw);
    if (g.arch == 0 && !glm_window_check(toks, p0, err)) return false;   // (STRATA_GLM_WINDOW_CHECK; off: nothing)
    return true;
}

bool Glm5Model::verify_part(const std::vector<int32_t>& toks, int64_t p0, std::string& err) {
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int T = V->T;
    const bool last = split_next_ == nullptr;
    const size_t E = (size_t) g.n_embd, HCE = (size_t) g.hc * E;
    std::vector<float*> Rc((size_t) T), Ro((size_t) T);
    for (int t = 0; t < T; ++t) {
        Rc[(size_t) t] = V->R + (size_t) t * 2 * HCE;
        Ro[(size_t) t] = Rc[(size_t) t] + HCE;
    }
    const auto hc_read = [&](int il, int t, const float* block_out, bool attn, cudaStream_t hs) {
        const auto& Ly = F->L[(size_t) il];
        gf::HcArgs h;
        h.block_out = block_out;
        h.R_old = Rc[(size_t) t];
        h.R_new = Ro[(size_t) t];
        h.post_in = V->post + (size_t) t * 8;
        h.comb_in = V->comb + (size_t) t * 16;
        h.pre = V->pre + (size_t) t * 8;
        h.post = V->post + (size_t) t * 8;
        h.comb = V->comb + (size_t) t * 16;
        h.w_fn = attn ? Ly.hc_attn_fn : Ly.hc_ffn_fn;
        h.w_scale = attn ? Ly.hc_attn_scale : Ly.hc_ffn_scale;
        h.w_base = attn ? Ly.hc_attn_base : Ly.hc_ffn_base;
        h.norm_w = attn ? Ly.attn_norm : Ly.ffn_norm;
        h.norm_eps = g.norm_eps;
        h.hc_eps = g.hc_eps;
        h.iters = g.sinkhorn_iters;
        h.n_embd = g.n_embd;
        h.x = V->x + (size_t) t * E;
        h.xq = (uint8_t*) V->xq + (size_t) t * (E / 32) * kQ8;
        h.part = V->hpart + (size_t) t * V->hpart_n;
        h.counter = V->hcnt + (size_t) t * 16;
        const bool cpu = attn && block_out != nullptr && V->hc_cpu[t];   // the column's last MoE answer, added here
        if (cpu) {
            h.cpu_seq = V->md[(size_t) t].cpu_seq;
            h.cpu_ans = V->md[(size_t) t].cpu_ans;
            V->hc_cpu[t] = false;
        }
        gf::hc(h, hs);
        if (block_out != nullptr) std::swap(Rc[(size_t) t], Ro[(size_t) t]);
        if (attn && V->bg_after[t]) {
            V->bg_after[t] = false;
            F->bg_drain(hs, F->bg_pace);
        }
    };
    // expert deferral (F->defer, as fast_layers): the layer before's CPU-lane part of column t into its streams after
    // this layer's attention, with the weights verify_moe saved (its own md: the last route is the layer before's)
    const auto defer_add = [&](int il, int t, cudaStream_t hs) {
        if (!F->defer || il <= g.dense_lead || il <= l0_ || !F->L[(size_t) il].moe) return;
        gf::hc_defer_add(V->md[(size_t) t], V->post_defer + (size_t) t * 8, g.n_embd, Rc[(size_t) t], hs);
    };
    // STRATA_GLM_PROF (as the token path's): the window's phases, marked v<T>_* so the two kinds stay apart
    // the pipelined window (verify's choice, V->pipe_now; not with PROF - its marks are one stream's)
    static const bool pipe_env1 = getenv("STRATA_DS4_SPEC_PIPE") != nullptr && std::atoi(getenv("STRATA_DS4_SPEC_PIPE")) == 2;
    const bool pipe = V->pipe_now && V->pipe && T > 1 && !F->prof_on;
    // STRATA_DS4_SPEC_GROUP=<n>: the pipelined window's columns a group; default two groups of ceil(T / 2) (the two
    // hardware queues), so a 2-column window keeps a column a group
    static const int grp_env = [] {
        const char* v = getenv("STRATA_DS4_SPEC_GROUP");
        return v != nullptr ? std::max(1, std::atoi(v)) : 0;
    }();
    int grp = !pipe ? T : grp_env > 0 ? std::min(grp_env, T) : (T + 1) / 2;
    while ((T + grp - 1) / grp > 1 + V->n_scr) ++grp;   // (a scratch set a group)
    const int n_grp = (T + grp - 1) / grp;
    // (a batched window: a set a column again - a pipelined window before left its groups' sets in md)
    for (int t = 0; t < T; ++t) V->md[(size_t) t].scratch = V->scr_set[pipe ? t / grp : std::min(t, V->n_scr)];
    const bool prof = F->prof_on;
    const std::string vt = "v" + std::to_string(T) + "_";
    static std::vector<std::string> names;   // (mark keeps the pointer)
    if (prof && names.empty())
        for (int n = 1; n <= kVerifyMax; ++n)
            for (const char* ph : {"hc_attn", "attn", "hc_ffn", "moe"}) names.push_back("v" + std::to_string(n) + "_" + ph);
    const auto mark = [&](int ph) { if (prof) F->mark(names[(size_t) (T - 1) * 4 + ph].c_str()); };
    if (prof) F->mark("v_start");
    for (int t = 0; t < kVerifyMax; ++t) V->hc_cpu[t] = V->bg_after[t] = false;
    if (pipe) {
        // group g of the columns (gs a group, the last one the rest) on stream g, after the embedding; per layer, its
        // attention after group g - 1's (ring rows, compressor state) and its routes after group g - 1's (group 0:
        // after the last group's of the layer before).  Inside a group the columns go together, as the batched
        // window's (the dense weights read once a group); across groups the host's experts of one group are computed
        // while the device runs another's attention.  gs = 1: a column a stream (the window before groups)
        cudaEventRecord(V->ev_start, s);
        for (int gi = 1; gi < n_grp; ++gi) cudaStreamWaitEvent(V->st[gi], V->ev_start, 0);
        for (int il = l0_; il < l1_; ++il)
            for (int gi = 0; gi < n_grp; ++gi) {
                const int c0 = gi * grp, nc = std::min(grp, T - c0);
                cudaStream_t st = pipe_env1 ? s : V->st[gi];
                if (gi > 0) cudaStreamWaitEvent(st, V->ev_attn[gi - 1], 0);
                for (int t = c0; t < c0 + nc; ++t) hc_read(il, t, il == l0_ ? nullptr : V->ffn + (size_t) t * E, true, st);
                if (!verify_attn(il, p0, c0, nc, st, err)) return false;
                cudaEventRecord(V->ev_attn[gi], st);
                for (int t = c0; t < c0 + nc; ++t) defer_add(il, t, st);
                if (dsp_ != nullptr && il > l0_)   // DSpark's features: the layer before's complete output
                    for (int t = c0; t < c0 + nc; ++t) dspark_capture(il - 1, Rc[(size_t) t], t, st);
                abl_out(il, 0, V->mixer + (size_t) c0 * E, nc, (int) E, st);
                for (int t = c0; t < c0 + nc; ++t) hc_read(il, t, V->mixer + (size_t) t * E, false, st);
                cudaEvent_t after = gi > 0 ? V->ev_route[gi - 1] : il > l0_ ? V->ev_route[n_grp - 1] : nullptr;
                if (!verify_moe(il, c0, nc, toks, st, after, V->ev_route[gi], err)) return false;
            }
    } else {
        for (int il = l0_; il < l1_; ++il) {
            for (int t = 0; t < T; ++t) hc_read(il, t, il == l0_ ? nullptr : V->ffn + (size_t) t * E, true, s);
            mark(0);
            if (!verify_attn(il, p0, 0, T, s, err)) return false;
            mark(1);
            for (int t = 0; t < T; ++t) defer_add(il, t, s);
            if (dsp_ != nullptr && il > l0_)
                for (int t = 0; t < T; ++t) dspark_capture(il - 1, Rc[(size_t) t], t, s);
            abl_out(il, 0, V->mixer, T, (int) E, s);
            for (int t = 0; t < T; ++t) hc_read(il, t, V->mixer + (size_t) t * E, false, s);
            mark(2);
            if (!verify_moe(il, 0, T, toks, s, nullptr, nullptr, err)) return false;
            mark(3);
        }
    }
    // (verify_moe leaves no answer to a read past the part's last layer; the check is for safety)
    for (int t = 0; t < T; ++t) {
        cudaStream_t st = pipe && !pipe_env1 ? V->st[t / grp] : s;
        if (V->hc_cpu[t]) {
            gf::moe_cpu_wait(V->md[(size_t) t], g.n_embd, V->ffn + (size_t) t * E, st);
            V->hc_cpu[t] = false;
        }
        if (V->bg_after[t]) {
            V->bg_after[t] = false;
            F->bg_drain(st, F->bg_pace);
        }
    }
    // the last layer's write half; a part before the last: the columns' residuals to the next part (as a token's hop);
    // the last: the head, the output rows (every column at once) and each column's argmax
    if (!last) {
        Glm5Model* B = split_next_.get();
        auto* VB = (VerifyBuf*) B->vb_;
        for (int t = 0; t < T; ++t) {
            cudaStream_t st = pipe && !pipe_env1 ? V->st[t / grp] : s;
            gf::hc_post(V->ffn + (size_t) t * E, Rc[(size_t) t], V->post + (size_t) t * 8, V->comb + (size_t) t * 16,
                        g.n_embd, Ro[(size_t) t], st);
            if (pipe && t / grp > 0) {
                cudaEventRecord(V->ev_fin[t / grp], st);
                cudaStreamWaitEvent(s, V->ev_fin[t / grp], 0);
            }
        }
        HopLink* L = hop_link_.get();
        const size_t pb = HCE * sizeof(float);
        if (!L->reserve(HopLink::kWindow, (size_t) T * pb, err)) return false;
        for (int t = 0; t < T; ++t)
            if (!L->put(HopLink::kWindow, (size_t) t * pb, Ro[(size_t) t], pb, s, err)) return false;
        if (!L->ship(HopLink::kWindow, (size_t) T * pb, s, err)) return false;
        cudaStream_t sb = B->fast_->cs;
        if (!L->arrive(HopLink::kWindow, sb, err)) return false;
        for (int t = 0; t < T; ++t)
            if (!L->get(HopLink::kWindow, (size_t) t * pb, VB->R + (size_t) t * 2 * HCE, pb, sb, err)) return false;
        cudaSetDevice(B->dev_);
        return true;
    }
    const auto fn = w16_.find("output_hc_fn.weight");
    const auto sc = w_.find("output_hc_scale.weight");
    const auto bs = w_.find("output_hc_base.weight");
    const auto nw = w_.find("output_norm.weight");
    if (fn == w16_.end() || sc == w_.end() || bs == w_.end() || nw == w_.end()) {
        err = "deepseek4 verify: the head's tensors";
        return false;
    }
    for (int t = 0; t < T; ++t) {
        cudaStream_t st = pipe && !pipe_env1 ? V->st[t / grp] : s;
        gf::hc_post(V->ffn + (size_t) t * E, Rc[(size_t) t], V->post + (size_t) t * 8, V->comb + (size_t) t * 16,
                    g.n_embd, Ro[(size_t) t], st);
        if (dsp_ != nullptr) dspark_capture(l1_ - 1, Ro[(size_t) t], t, st);
        ds::head_prep(Ro[(size_t) t], fn->second, sc->second, bs->second, g.hc_eps, nw->second, g.norm_eps, g.n_embd,
                      V->head_x + (size_t) t * E, (uint8_t*) V->head_xq + (size_t) t * (E / 32) * kQ8, st);
        if (pipe && t / grp > 0) {
            cudaEventRecord(V->ev_fin[t / grp], st);
            cudaStreamWaitEvent(s, V->ev_fin[t / grp], 0);
        }
    }
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, V->head_xq, V->head_x, V->logits, nullptr, 1.0f, ow.type, g.n_embd, g.n_vocab};
    if (gf::mvn_supported(ow.type)) {
        if (!gf::mvn(&o, 1, T, s)) { err = "deepseek4 verify: output head"; return false; }
    } else {
        for (int t = 0; t < T; ++t) {
            gf::MvJob ot = o;
            ot.xq = (const uint8_t*) V->head_xq + (size_t) t * (E / 32) * kQ8;
            ot.xf = V->head_x + (size_t) t * E;
            ot.y = V->logits + (size_t) t * g.n_vocab;
            if (!gf::mv(&ot, 1, s)) { err = "deepseek4 verify: output head (type " + std::to_string(ow.type) + ")"; return false; }
        }
    }
    if (F->zero_copy) {
        for (int t = 0; t < T; ++t) gf::argmax(V->logits + (size_t) t * g.n_vocab, g.n_vocab, V->tok_hd + t, s);
    } else {
        for (int t = 0; t < T; ++t) gf::argmax(V->logits + (size_t) t * g.n_vocab, g.n_vocab, V->tok + t, s);
        cudaMemcpyAsync(V->tok_h, V->tok, (size_t) T * sizeof(int), cudaMemcpyDeviceToHost, s);
    }
    cudaEventRecord(V->done, s);
    return true;
}

// deepseek41's window (one part, the batched kind): fast_layers' V4.1 layer loop with the columns side by side -
// an Engram layer writes the previous FFN alone, adds the column's rows, then reads without a write; the reads
// collapse with the previous sublayer's pre (each column's pair of buffers, layer 0's one-hot set by verify); then
// the last write half, the head's collapse with the last FFN's pre, the output rows of every column at once and each
// column's argmax
bool Glm5Model::verify_part_v41(const std::vector<int32_t>& toks, int64_t p0, std::string& err) {
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int T = V->T;
    const size_t E = (size_t) g.n_embd, HCE = (size_t) g.hc * E, q8e = (E / 32) * kQ8;
    std::vector<float*> Rc((size_t) T), Ro((size_t) T);
    for (int t = 0; t < T; ++t) {
        Rc[(size_t) t] = V->R + (size_t) t * 2 * HCE;
        Ro[(size_t) t] = Rc[(size_t) t] + HCE;
    }
    const auto hc_read = [&](int il, int t, const float* block_out, bool attn) {
        const auto& Ly = F->L[(size_t) il];
        float* pp = V->v41_pre + (size_t) t * 8;   // [0, 4) the attention side's pre, [4, 8) the FFN side's
        gf::HcArgs h;
        h.block_out = block_out;
        h.R_old = Rc[(size_t) t];
        h.R_new = Ro[(size_t) t];
        h.post_in = V->post + (size_t) t * 8;
        h.comb_in = V->comb + (size_t) t * 16;
        h.pre = attn ? pp : pp + 4;
        h.pre_use = attn ? pp + 4 : pp;
        h.post = V->post + (size_t) t * 8;
        h.comb = V->comb + (size_t) t * 16;
        h.w_fn = attn ? Ly.hc_attn_fn : Ly.hc_ffn_fn;
        h.w_scale = attn ? Ly.hc_attn_scale : Ly.hc_ffn_scale;
        h.w_base = attn ? Ly.hc_attn_base : Ly.hc_ffn_base;
        h.norm_w = attn ? Ly.attn_norm : Ly.ffn_norm;
        h.norm_eps = g.norm_eps;
        h.hc_eps = g.hc_eps;
        h.iters = g.sinkhorn_iters;
        h.n_embd = g.n_embd;
        h.x = V->x + (size_t) t * E;
        h.xq = (uint8_t*) V->xq + (size_t) t * q8e;
        h.part = V->hpart + (size_t) t * V->hpart_n;
        h.counter = V->hcnt + (size_t) t * 16;
        if (attn && block_out != nullptr && V->hc_cpu[t]) {   // the column's last MoE answer, added here
            h.cpu_seq = V->md[(size_t) t].cpu_seq;
            h.cpu_ans = V->md[(size_t) t].cpu_ans;
            V->hc_cpu[t] = false;
        }
        gf::hc(h, s);
        if (block_out != nullptr) std::swap(Rc[(size_t) t], Ro[(size_t) t]);
        if (attn && V->bg_after[t]) {
            V->bg_after[t] = false;
            F->bg_drain(s, F->bg_pace);
        }
    };
    // STRATA_GLM_PROF (as the token path's): the window's phases, marked v<T>_*
    const bool prof = F->prof_on;
    static std::vector<std::string> names;   // (mark keeps the pointer)
    if (prof && names.empty())
        for (int n = 1; n <= kVerifyMax; ++n)
            for (const char* ph : {"hc_attn", "attn", "hc_ffn", "moe", "head"}) names.push_back("v" + std::to_string(n) + "_" + ph);
    const auto mark = [&](int ph) { if (prof) F->mark(names[(size_t) (T - 1) * 5 + ph].c_str()); };
    if (prof) F->mark("v_start");
    for (int t = 0; t < kVerifyMax; ++t) V->hc_cpu[t] = V->bg_after[t] = false;
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        for (int t = 0; t < T; ++t) {
            bool wrote = il != l0_;
            if (Ly.eng >= 0) {
                if (il != l0_) {   // (verify_moe leaves no answer for a read before an Engram layer)
                    gf::hc_post(V->ffn + (size_t) t * E, Rc[(size_t) t], V->post + (size_t) t * 8,
                                V->comb + (size_t) t * 16, g.n_embd, Ro[(size_t) t], s);
                    std::swap(Rc[(size_t) t], Ro[(size_t) t]);
                    wrote = false;
                }
                if (!dsv41_engram_window_add(il, t, Rc[(size_t) t], err)) return false;
                abl_res(Rc[(size_t) t], g.hc, g.n_embd, s);   // (Engram's rows are a residual write too)
            }
            hc_read(il, t, wrote ? V->ffn + (size_t) t * E : nullptr, true);
        }
        mark(0);
        if (!verify_attn_v41(il, p0, 0, T, s, err)) return false;
        mark(1);
        if (dsp_ != nullptr && il > l0_)   // DSpark's features: the layer before's output, each column's
            for (int t = 0; t < T; ++t) dspark_capture(il - 1, Rc[(size_t) t], t, s);
        abl_out(il, 0, V->mixer, T, (int) E, s);
        for (int t = 0; t < T; ++t) hc_read(il, t, V->mixer + (size_t) t * E, false);
        mark(2);
        if (!verify_moe(il, 0, T, toks, s, nullptr, nullptr, err)) return false;
        mark(3);
    }
    for (int t = 0; t < T; ++t) {   // (verify_moe leaves no answer past the last layer; the check is for safety)
        if (V->hc_cpu[t]) {
            gf::moe_cpu_wait(V->md[(size_t) t], g.n_embd, V->ffn + (size_t) t * E, s);
            V->hc_cpu[t] = false;
        }
        if (V->bg_after[t]) {
            V->bg_after[t] = false;
            F->bg_drain(s, F->bg_pace);
        }
    }
    const auto nw = w_.find("output_norm.weight");
    if (nw == w_.end()) { err = "deepseek41 verify: output_norm.weight"; return false; }
    for (int t = 0; t < T; ++t) {
        gf::hc_post(V->ffn + (size_t) t * E, Rc[(size_t) t], V->post + (size_t) t * 8, V->comb + (size_t) t * 16,
                    g.n_embd, Ro[(size_t) t], s);
        // the last sublayer (layer n-1's FFN) wrote its pre into the column's FFN-side buffer
        d41::head_prep(Ro[(size_t) t], V->v41_pre + (size_t) t * 8 + 4, nw->second, g.norm_eps, g.n_embd,
                       V->head_x + (size_t) t * E, (uint8_t*) V->head_xq + (size_t) t * q8e, s);
    }
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, V->head_xq, V->head_x, V->logits, nullptr, 1.0f, ow.type, g.n_embd, g.n_vocab};
    if (gf::mvn_supported(ow.type)) {
        if (!gf::mvn(&o, 1, T, s)) { err = "deepseek41 verify: output head"; return false; }
    } else {
        for (int t = 0; t < T; ++t) {
            gf::MvJob ot = o;
            ot.xq = (const uint8_t*) V->head_xq + (size_t) t * q8e;
            ot.xf = V->head_x + (size_t) t * E;
            ot.y = V->logits + (size_t) t * g.n_vocab;
            if (!gf::mv(&ot, 1, s)) { err = "deepseek41 verify: output head (type " + std::to_string(ow.type) + ")"; return false; }
        }
    }
    if (F->zero_copy) {
        for (int t = 0; t < T; ++t) gf::argmax(V->logits + (size_t) t * g.n_vocab, g.n_vocab, V->tok_hd + t, s);
    } else {
        for (int t = 0; t < T; ++t) gf::argmax(V->logits + (size_t) t * g.n_vocab, g.n_vocab, V->tok + t, s);
        cudaMemcpyAsync(V->tok_h, V->tok, (size_t) T * sizeof(int), cudaMemcpyDeviceToHost, s);
    }
    mark(4);
    cudaEventRecord(V->done, s);
    return true;
}

// qwen35moe's window (one part, batched): the token path's layers over the T columns - the dense GEMVs one mvn launch
// over the columns, the plain residual's reads a column (rms_q8), the delta-net rows and the attention rows the prompt
// path's row kernels (the recurrence and the conv carry their state through the window's rows in order), the MoE
// verify_moe's.  The states before the window are kept, and every delta-net layer's conv inputs, prepped rows and
// gates: spec_rewind replays the kept rows from them (verify_rewind_q35).
bool Glm5Model::verify_part_q35(const std::vector<int32_t>& toks, int64_t p0, std::string& err) {
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int T = V->T, TM = V->TM;
    const size_t E = (size_t) g.n_embd, q8e = (E / 32) * kQ8;
    const int C = g.ssm_conv_dim(), Vd = g.ssm_inner, Sd = g.ssm_state, HK = g.ssm_k_heads, HV = g.ssm_v_heads;
    const int D = g.head_dim, HD = g.n_head * D, KV = g.attn_head_kv * D;
    const size_t state_n = (size_t) Sd * HV * Sd, conv_n = (size_t) C * (g.d_conv - 1);
    const auto Rt = [&](int t) { return V->R + (size_t) t * 2 * E; };   // (its pair's first half)
    const bool prof = F->prof_on;
    static std::vector<std::string> names;
    if (prof && names.empty())
        for (int n = 1; n <= kVerifyMax; ++n)
            for (const char* ph : {"norm", "mixer", "norm2", "moe", "head"}) names.push_back("v" + std::to_string(n) + "_" + ph);
    const auto mark = [&](int ph) { if (prof) F->mark(names[(size_t) (T - 1) * 5 + ph].c_str()); };
    if (prof) F->mark("v_start");
    V->q35_p0 = p0;
    for (int t = 0; t < T; ++t) V->q35_toks[t] = toks[(size_t) t];
    if (q35_mtp_on() && !q35_mtp_pend_commit(toks[0], err)) return false;   // (the drafter: the last position's row)
    int li = 0;
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (!Ly.recr) continue;
        cudaMemcpyAsync(V->q35_snap + (size_t) li * state_n, Ly.gdn_state, state_n * sizeof(float), cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(V->q35_csnap + (size_t) li * conv_n, Ly.gdn_conv, conv_n * sizeof(float), cudaMemcpyDeviceToDevice, s);
        ++li;
    }
    li = 0;
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        for (int t = 0; t < T; ++t)
            gf::rms_q8(Rt(t), il == l0_ ? nullptr : V->ffn + (size_t) t * E, Ly.attn_norm, g.norm_eps, (int) E,
                       V->x + (size_t) t * E, (uint8_t*) V->xq + (size_t) t * q8e, s);
        mark(0);
        if (Ly.recr) {
            float* qkv = V->q35_qkv + (size_t) li * TM * C;
            float* rows = V->q35_rows + (size_t) li * TM * C;
            float* alpha = V->q35_alpha + (size_t) li * TM * HV;
            float* beta = V->q35_beta + (size_t) li * TM * HV;
            gf::MvJob j[4];
            j[0] = {Ly.qkv.q, V->xq, V->x, qkv, nullptr, 1.0f, Ly.qkv.type, (int) E, C};
            j[1] = {Ly.zg.q, V->xq, V->x, V->q35_z, nullptr, 1.0f, Ly.zg.type, (int) E, Vd};
            j[2] = {Ly.alpha_w.q, V->xq, V->x, alpha, nullptr, 1.0f, Ly.alpha_w.type, (int) E, HV};
            j[3] = {Ly.beta_w.q, V->xq, V->x, beta, nullptr, 1.0f, Ly.beta_w.type, (int) E, HV};
            if (!gf::mvn(j, 4, T, s)) { err = "qwen35 verify: the delta-net projections"; return false; }
            q35::conv_rows(qkv, T, C, Ly.conv1d, g.d_conv, Ly.gdn_conv, rows, s);
            q35::gdn_prep_gates(rows, T, C, HK, Sd, g.norm_eps, alpha, beta, Ly.dt_bias, Ly.ssm_a, HV, s);
            q35::gdn_scan(Ly.gdn_state, rows, T, C, alpha, beta, Sd, HK, HV, V->q35_o, s);
            strata::kernels::gdn_out_norm_silu(V->q35_o, V->q35_z, Ly.ssm_norm, V->q35_y, (int64_t) T * HV, Sd, g.norm_eps,
                                               (void*) s);
            gf::quantize_q8_1(V->q35_y, V->q35_yq, T * Vd, s);
            gf::MvJob o = {Ly.ssm_o.q, V->q35_yq, V->q35_y, V->mixer, nullptr, 1.0f, Ly.ssm_o.type, Vd, (int) E};
            if (!gf::mvn(&o, 1, T, s)) { err = "qwen35 verify: ssm_out"; return false; }
            ++li;
        } else {
            gf::MvJob j[3];
            j[0] = {Ly.aq.q, V->xq, V->x, V->q35_qf, nullptr, 1.0f, Ly.aq.type, (int) E, 2 * HD};
            j[1] = {Ly.ak.q, V->xq, V->x, V->q35_k, nullptr, 1.0f, Ly.ak.type, (int) E, KV};
            j[2] = {Ly.av.q, V->xq, V->x, V->q35_v, nullptr, 1.0f, Ly.av.type, (int) E, KV};
            if (!gf::mvn(j, 3, T, s)) { err = "qwen35 verify: the attention projections"; return false; }
            q35::attn_prep_rows(V->q35_qf, V->q35_k, V->q35_v, Ly.q_norm, Ly.k_norm, g.norm_eps, g.n_head, g.attn_head_kv, D,
                                g.n_rot, g.rope_base, p0, T, Ly.kc, Ly.vc, V->q35_q, s, q35_kv_);
            q35::attn_window(V->q35_q, Ly.kc, Ly.vc, g.n_head, g.attn_head_kv, D, p0, T, 1.0f / std::sqrt((float) D),
                             V->q35_part, V->q35_ao, s, q35_kv_);
            q35::attn_gate_rows(V->q35_ao, V->q35_qf, T, g.n_head, D, s, q35_kv_);
            gf::quantize_q8_1(V->q35_ao, V->q35_aoq, T * HD, s);
            gf::MvJob o = {Ly.out.q, V->q35_aoq, V->q35_ao, V->mixer, nullptr, 1.0f, Ly.out.type, HD, (int) E};
            if (!gf::mvn(&o, 1, T, s)) { err = "qwen35 verify: attn_output"; return false; }
        }
        mark(1);
        abl_out(il, 0, V->mixer, T, (int) E, s);
        for (int t = 0; t < T; ++t)
            gf::rms_q8(Rt(t), V->mixer + (size_t) t * E, Ly.post_norm, g.norm_eps, (int) E, V->x + (size_t) t * E,
                       (uint8_t*) V->xq + (size_t) t * q8e, s);
        mark(2);
        if (!verify_moe(il, 0, T, toks, s, nullptr, nullptr, err)) return false;
        mark(3);
    }
    const auto nw = w_.find("output_norm.weight");
    if (nw == w_.end()) { err = "qwen35 verify: output_norm.weight"; return false; }
    for (int t = 0; t < T; ++t)
        gf::rms_q8(Rt(t), V->ffn + (size_t) t * E, nw->second, g.norm_eps, (int) E, V->head_x + (size_t) t * E,
                   (uint8_t*) V->head_xq + (size_t) t * q8e, s);
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, V->head_xq, V->head_x, V->logits, nullptr, 1.0f, ow.type, (int) E, g.n_vocab};
    if (gf::mvn_supported(ow.type)) {
        if (!gf::mvn(&o, 1, T, s)) { err = "qwen35 verify: output head"; return false; }
    } else {
        for (int t = 0; t < T; ++t) {
            gf::MvJob ot = o;
            ot.xq = (const uint8_t*) V->head_xq + (size_t) t * q8e;
            ot.xf = V->head_x + (size_t) t * E;
            ot.y = V->logits + (size_t) t * g.n_vocab;
            if (!gf::mv(&ot, 1, s)) { err = "qwen35 verify: output head (type " + std::to_string(ow.type) + ")"; return false; }
        }
    }
    if (F->zero_copy) {
        for (int t = 0; t < T; ++t) gf::argmax(V->logits + (size_t) t * g.n_vocab, g.n_vocab, V->tok_hd + t, s);
    } else {
        for (int t = 0; t < T; ++t) gf::argmax(V->logits + (size_t) t * g.n_vocab, g.n_vocab, V->tok + t, s);
        cudaMemcpyAsync(V->tok_h, V->tok, (size_t) T * sizeof(int), cudaMemcpyDeviceToHost, s);
    }
    mark(4);
    cudaEventRecord(V->done, s);
    return true;
}

// qwen4exp's hyper-connection over the window's T columns (R: [T][hc][n_embd], the columns contiguous) - the token
// path's q4x_mix a column at a time in one launch each: R += out . 2 sigmoid(inj_prev / hc) (out null: none), the
// streams normed, inject (inject non-null), lo = silu(down . xn / hc), gate = up . lo, x = mean_c xn sigmoid(gate)
bool Glm5Model::q4x_mix_cols(float* R, int T, const float* norm, const WSlot& down, const WSlot& up, const WSlot* inject,
                             const float* out, const float* inj_prev, float* inj_out, float* x, void* xq, std::string& err) {
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = fast_->cs;
    const int E = g.n_embd, HE = g.hc * E, LR = g.hc_lr;
    // the five launches below in two (q35::hc4_fused_cols: the same bits as CUDA's five - hc4_fused_parity, T 1..8),
    // as the token path's q4x_mix.  RTX 5090, Qwen3.8 ISTA IQ3_S: a window's 97 mixes 2.6 -> 1.8 ms (2 columns),
    // 5.5 -> 2.6 ms (5) - mvn's T columns of xn did not fit in L1.  STRATA_Q4X_HC_FUSE=0: the five; 2: the token
    // path's fused, the windows' five (A/B).  Off by default with HIP, as the token path's
    static const bool fuse = [] {
        const char* v = getenv("STRATA_Q4X_HC_FUSE");
#if defined(STRATA_USE_HIP)
        return v != nullptr && std::atoi(v) == 1;
#else
        return v == nullptr || (std::atoi(v) != 0 && std::atoi(v) != 2);
#endif
    }();
    if (fuse && q35::hc4_fused_cols_ok(E, g.hc, LR, T, down.type, up.type, inject != nullptr ? inject->type : -1)) {
        q35::hc4_fused_cols(R, out, inj_prev, g.hc, norm, g.norm_eps, E, g.hc, LR, T, down.q,
                            inject != nullptr ? inject->q : nullptr, up.q, V->q4_lo, inj_out, V->q4_sc, V->q4_xn,
                            V->q4_gate, x, xq, s, inject != nullptr && inject->type == gf::kTypeF32, down.type, V->q4_xnq);
        return true;
    }
    q35::hc4_combine_norm(R, out, inj_prev, g.hc, norm, g.norm_eps, E, g.hc, T, V->q4_xn, s, V->q4_xnq);
    gf::MvJob j[2];
    j[0] = {down.q, V->q4_xnq, V->q4_xn, V->q4_lo, nullptr, 1.0f, down.type, HE, LR};
    if (inject != nullptr) j[1] = {inject->q, V->q4_xnq, V->q4_xn, inj_out, nullptr, 1.0f, inject->type, HE, g.hc};
    if (!q4x_mvn(j, inject != nullptr ? 2 : 1, T, s)) { err = "qwen4exp verify: the hyper-connection's down / inject"; return false; }
    q35::silu_scale(V->q4_lo, T * LR, 1.0f / (float) g.hc, s, V->q4_loq);
    gf::MvJob u = {up.q, V->q4_loq, V->q4_lo, V->q4_gate, nullptr, 1.0f, up.type, LR, HE};
    if (!q4x_mvn(&u, 1, T, s)) { err = "qwen4exp verify: the hyper-connection's up"; return false; }
    q35::hc4_mix(V->q4_xn, V->q4_gate, E, g.hc, T, x, s, xq);
    return true;
}

// qwen4exp's window (one part, batched): fast_q4x_layers over the T columns - the hyper-connections a launch over the
// columns (q4x_mix_cols), layer 1's PLE for the window's rows in one table read (its conv history kept before the
// window and its rows for the rewind), the delta-net layers verify_part_q35's (their rows and states kept the same
// way), the attention layers' indexer keys a column, each column's attention dense up to 2,051 cells and over its own
// selection past them (fast_qwen35_attn's), the MoE verify_moe's, the output hyper-connection and the head.  Every
// dense GEMV is glm_mvn (gf::mvn_mv: mv()'s sums a column, the weights read once): gf::mvn ran the small Q8_0 matrices
// (hc down, the shared expert, attn k / v) one warp a row where the token path's mv() splits them four warps a row,
// so greedy answers with drafts parted from spec=0 at tokens 7-30 (2026-11-10; tools/spec_exact_check.py)
void q4x_trace(cudaStream_t s, int64_t p, int il, const char* name, const float* dev, int64_t n);   // (qwen35_fast_path.cu)
bool Glm5Model::verify_part_q4x(const std::vector<int32_t>& toks, int64_t p0, std::string& err) {
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    namespace k = strata::kernels;
    const int T = V->T, TM = V->TM;
    const int E = g.n_embd, HE = g.hc * E;
    const int C = g.ssm_conv_dim(), Vd = g.ssm_inner, Sd = g.ssm_state, HK = g.ssm_k_heads, HV = g.ssm_v_heads;
    const int D = g.head_dim, HD = g.n_head * D, KV = g.attn_head_kv * D, ID = g.idx_key, IH = g.idx_heads;
    const size_t state_n = (size_t) Sd * HV * Sd, conv_n = (size_t) C * (g.d_conv - 1);
    const k::QsaShapes sh = k::qsa_real_shapes();
    V->q35_p0 = p0;
    for (int t = 0; t < T; ++t) V->q35_toks[t] = toks[(size_t) t];
    const bool prof = F->prof_on;   // STRATA_GLM_PROF: the window's phases as q4w_*
    // STRATA_GLM_GRAPH=0/1: the per-layer graphs below off / on (default: on with CUDA, off with HIP - not measured)
    static const bool q4g_moe = [] {
        const char* v = getenv("STRATA_GLM_GRAPH_MOE");
        return v == nullptr || std::atoi(v) != 0;
    }();
    // STRATA_Q4X_GRAPH_PDL=0/1: the graphs below captured with PDL's programmatic edges (pdl.hpp's PdlCapture: the
    // launches that carry the attribute outside a capture carry it inside); default on where PDL is on, except an
    // integrated GPU (not measured there).  A refused capture is taken again without the edges
    static const bool q4g_pdl = [] {
        const char* v = getenv("STRATA_Q4X_GRAPH_PDL");
        if (v != nullptr) return std::atoi(v) != 0;
#if defined(STRATA_USE_HIP)
        return false;
#else
        int dev = 0, integ = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&integ, cudaDevAttrIntegrated, dev);
        return integ == 0;
#endif
    }();
    static const bool q4g_on = [] {
        const char* v = getenv("STRATA_GLM_GRAPH");
#if defined(STRATA_USE_HIP)
        return v != nullptr && std::atoi(v) != 0;
#else
        return v == nullptr || std::atoi(v) != 0;
#endif
    }();
    if (prof) F->mark("q4w_start");
    if (q35_mtp_on() && !q35_mtp_pend_commit(toks[0], err)) return false;   // (the drafter: the last position's row)
    if (prof) F->mark("q4w_mtp_commit");
    int li = 0;
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (!Ly.recr) continue;
        cudaMemcpyAsync(V->q35_snap + (size_t) li * state_n, Ly.gdn_state, state_n * sizeof(float), cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(V->q35_csnap + (size_t) li * conv_n, Ly.gdn_conv, conv_n * sizeof(float), cudaMemcpyDeviceToDevice, s);
        ++li;
    }
    // the window's PLE rows: hashed on the host (ple_toks carries the history), queued as one request (the reader's
    // worker reads them) while the device runs layer 0, collected before layer 1's launches (STRATA_Q4X_PLE_SYNC=1:
    // collected here).  The drafter's steps prefetched the rows of the window's tokens it knew (q35_mtp_draft): those
    // come from the row cache
    static const bool ple_sync = getenv("STRATA_Q4X_PLE_SYNC") != nullptr && std::atoi(getenv("STRATA_Q4X_PLE_SYNC")) != 0;
    bool ple_wait = false;
    struct Collect {   // (an early return still collects the batch: one batch in flight a table)
        k::PleTable* t;
        bool& on;
        float* dst;
        ~Collect() {
            std::string e;
            if (on) (void) t->collect_batch(dst, e);
            if (t != nullptr) t->reap_prefetches();
        }
    } ple_join{F->ple.get(), ple_wait, F->ple_pf_h};
    V->q4_ple_on = F->ple != nullptr && l0_ <= F->ple_il && F->ple_il < l1_;
    if (V->q4_ple_on) {
        if (F->ple_issued) {
            if (!F->ple->collect(F->ple_emb_h, err)) return false;
            F->ple_issued = false;
        }
        const size_t nt = F->ple_toks.size();
        if (p0 < 0 || (size_t) (p0 + T) > nt) { err = "qwen4exp verify: the PLE history is shorter than the context"; return false; }
        for (int t = 0; t < T; ++t) F->ple_toks[(size_t) (p0 + t)] = toks[(size_t) t];
        int32_t prev[2 * Glm5Model::kVerifyMax];
        for (int t = 0; t < T; ++t) {
            const int64_t p = p0 + t;
            prev[2 * t] = p >= 2 ? F->ple_toks[(size_t) (p - 2)] : -1;
            prev[2 * t + 1] = p >= 1 ? F->ple_toks[(size_t) (p - 1)] : -1;
        }
        static const k::PleConsts consts = k::ple_artifact_consts();
        uint32_t rows[k::PLE_N_HEADS * Glm5Model::kVerifyMax];
        k::ngram_rows(toks.data(), prev, T, consts, rows);
        if (F->ple_pf_n < (size_t) TM * E) {
            if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "qwen4exp verify PLE buffer", err)) return false;
            if (F->ple_pf_h != nullptr) cudaFreeHost(F->ple_pf_h);
            F->ple_pf_n = 0;
            if (cudaHostAlloc((void**) &F->ple_pf_h, (size_t) TM * E * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
                F->ple_pf_h = nullptr;
                err = "qwen4exp verify: the PLE rows did not allocate";
                return false;
            }
            F->ple_pf_n = (size_t) TM * E;
        }
        // (the buffer's last H2D copy, an earlier window's, finished: the previous window was waited for)
        ple_join.dst = F->ple_pf_h;
        if (!F->ple->issue_batch(rows, (size_t) T)) { err = "qwen4exp verify: the PLE rows could not be issued"; return false; }
        ple_wait = true;
        if (ple_sync) {
            ple_wait = false;
            if (!F->ple->collect_batch(F->ple_pf_h, err)) { err = "qwen4exp verify: PLE: " + err; return false; }
        }
        cudaMemcpyAsync(V->q4_ple_snap, F->ple_hist, (size_t) k::NG_HIST * HE * sizeof(float), cudaMemcpyDeviceToDevice, s);
    }
    // the columns' QSA step records and query positions
    for (int t = 0; t < T; ++t) {
        k::qsa_step_fill(V->q4_steps_h + (size_t) t * 4, p0 + t, sh);
        for (int h = 0; h < IH; ++h) V->q4_steps_h[(size_t) T * 4 + (size_t) t * IH + h] = (int32_t) (p0 + t);
    }
    cudaMemcpyAsync(V->q4_steps, V->q4_steps_h, (size_t) T * (4 + IH) * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    const int32_t* ipos = V->q4_steps + (size_t) T * 4;
    const size_t tail_n = (size_t) (sh.idx_block - 1) * ID;
    {   // every QSA layer's raw-key tail as before the window (verify_rewind_q35)
        int la = 0;
        for (int il = l0_; il < l1_; ++il) {
            const auto& Ly = F->L[(size_t) il];
            if (Ly.q4_tail == nullptr) continue;
            cudaMemcpyAsync(V->q4_tsnap + (size_t) la * tail_n, Ly.q4_tail, tail_n * sizeof(float), cudaMemcpyDeviceToDevice, s);
            ++la;
        }
    }
    if (prof) F->mark("q4w_ple_read");
    li = 0;
    int la = 0;   // (the QSA layers so far)
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        const float* prev_out = il == l0_ ? nullptr : V->ffn;
        if (V->q4_ple_on && il == F->ple_il) {
            if (prev_out != nullptr)
                q35::hc4_combine_norm(V->R, prev_out, V->q4_inj[1], g.hc, Ly.hc4[0].norm, g.norm_eps, E, g.hc, T, V->q4_xn, s);
            prev_out = nullptr;
            if (ple_wait) {
                ple_wait = false;
                if (!F->ple->collect_batch(F->ple_pf_h, err)) { err = "qwen4exp verify: PLE: " + err; return false; }
            }
            F->ple->reap_prefetches();
            cudaMemcpyAsync(V->q4_emb, F->ple_pf_h, (size_t) T * E * sizeof(float), cudaMemcpyHostToDevice, s);
            gf::quantize_q8_1(V->q4_emb, V->q4_embq, T * E, s);
            gf::MvJob j[2];
            j[0] = {F->ple_key_w.q, V->q4_embq, V->q4_emb, V->q4_key, nullptr, 1.0f, F->ple_key_w.type, E, HE};
            j[1] = {F->ple_val_w.q, V->q4_embq, V->q4_emb, V->q4_val, nullptr, 1.0f, F->ple_val_w.type, E, E};
            if (!q4x_mvn(j, 2, T, s)) { err = "qwen4exp verify: the PLE key / value"; return false; }
            k::native_ple_postops_batch(V->q4_key, V->R, V->q4_val, F->ple_hist, F->ple_w, V->q4_qn, V->q4_gated, V->q4_g4,
                                        T, (void*) s);
            abl_res(V->R, T * g.hc, (int) E, s);   // (the PLE's rows are a residual write too)
            if (prof) F->mark("q4w_ple");
        }
        // the layer's same-every-window part - the hyper-connections, the delta-net block and the MoE's router -
        // as a graph a (layer, window width) (STRATA_GLM_GRAPH, stage 1): captured at its second window, replayed
        // after; the attention layers (their positions are arguments), the PLE layer and the MoE's dispatch run as
        // launches.  The same kernels, the same arguments: the same bits
        const auto seg = [&](bool with_router) -> bool {
            if (!q4x_mix_cols(V->R, T, Ly.hc4[0].norm, Ly.hc4[0].down, Ly.hc4[0].up, &Ly.hc4[0].inject, prev_out,
                              V->q4_inj[1], V->q4_inj[0], V->x, V->xq, err))
                return false;
            if (dfl_ != nullptr) dflash_capture(il, V->R, T, (size_t) HE, false, s);   // (DFlash 2's features a column)
            for (int t = 0; t < T; ++t) {
                q4x_trace(s, p0 + t, il, "a_R", V->R + (size_t) t * HE, HE);
                q4x_trace(s, p0 + t, il, "b_x", V->x + (size_t) t * E, E);
            }
            if (prof) F->mark("q4w_hc_attn");
            if (Ly.recr) {
                q4x_trace(s, p0, il, "a0_state", Ly.gdn_state, (int64_t) state_n);
                q4x_trace(s, p0, il, "a1_conv", Ly.gdn_conv, (int64_t) conv_n);
                float* qkv = V->q35_qkv + (size_t) li * TM * C;
                float* rws = V->q35_rows + (size_t) li * TM * C;
                float* alpha = V->q35_alpha + (size_t) li * TM * HV;
                float* beta = V->q35_beta + (size_t) li * TM * HV;
                gf::MvJob j[4];
                j[0] = {Ly.qkv.q, V->xq, V->x, qkv, nullptr, 1.0f, Ly.qkv.type, E, C};
                j[1] = {Ly.zg.q, V->xq, V->x, V->q35_z, nullptr, 1.0f, Ly.zg.type, E, Vd};
                j[2] = {Ly.alpha_w.q, V->xq, V->x, alpha, nullptr, 1.0f, Ly.alpha_w.type, E, HV};
                j[3] = {Ly.beta_w.q, V->xq, V->x, beta, nullptr, 1.0f, Ly.beta_w.type, E, HV};
                if (!q4x_mvn(j, 4, T, s)) { err = "qwen4exp verify: the delta-net projections"; return false; }
                if (prof) F->mark("q4w_gdn_mv");
                q35::conv_rows(qkv, T, C, Ly.conv1d, g.d_conv, Ly.gdn_conv, rws, s);
                q35::gdn_prep_gates(rws, T, C, HK, Sd, g.norm_eps, alpha, beta, Ly.dt_bias, Ly.ssm_a, HV, s);
                if (prof) F->mark("q4w_gdn_conv");
                // the recurrence, the output norm and y's q8_1 in one launch (q35::gdn_core after the prep above - the rows,
                // gates and betas stay for the rewind's replay; the same bits, gdn_core_parity).  STRATA_Q4X_GDN_FUSE=0: the
                // three launches.  Off by default with HIP, as the token path's
                static const bool gdn_fuse = [] {
                    const char* v = getenv("STRATA_Q4X_GDN_FUSE");
    #if defined(STRATA_USE_HIP)
                    return v != nullptr && std::atoi(v) != 0;
    #else
                    return v == nullptr || std::atoi(v) != 0;
    #endif
                }();
                if (gdn_fuse && Sd == 128) {
                    q35::gdn_core(Ly.gdn_state, rws, T, C, alpha, beta, Sd, HK, HV, false, g.norm_eps, nullptr, nullptr,
                                  V->q35_z, Ly.ssm_norm, g.norm_eps, V->q35_y, V->q35_yq, s);
                    if (prof) F->mark("q4w_gdn_scan");
                } else {
                    q35::gdn_scan(Ly.gdn_state, rws, T, C, alpha, beta, Sd, HK, HV, V->q35_o, s);
                    if (prof) F->mark("q4w_gdn_scan");
                    k::gdn_out_norm(V->q35_o, V->q35_z, Ly.ssm_norm, V->q35_y, (int64_t) T * HV, Sd, g.norm_eps, (void*) s);
                    gf::quantize_q8_1(V->q35_y, V->q35_yq, T * Vd, s);
                }
                gf::MvJob o = {Ly.ssm_o.q, V->q35_yq, V->q35_y, V->mixer, nullptr, 1.0f, Ly.ssm_o.type, Vd, E};
                if (!q4x_mvn(&o, 1, T, s)) { err = "qwen4exp verify: ssm_out"; return false; }
                if (prof) F->mark("q4w_gdn");
                ++li;
            } else {
                gf::MvJob j[3];
                j[0] = {Ly.aq.q, V->xq, V->x, V->q35_qf, nullptr, 1.0f, Ly.aq.type, E, 2 * HD};
                j[1] = {Ly.ak.q, V->xq, V->x, V->q35_k, nullptr, 1.0f, Ly.ak.type, E, KV};
                j[2] = {Ly.av.q, V->xq, V->x, V->q35_v, nullptr, 1.0f, Ly.av.type, E, KV};
                if (!q4x_mvn(j, 3, T, s)) { err = "qwen4exp verify: the attention projections"; return false; }
                q35::attn_prep_rows(V->q35_qf, V->q35_k, V->q35_v, Ly.q_norm, Ly.k_norm, g.norm_eps, g.n_head, g.attn_head_kv, D,
                                    g.n_rot, g.rope_base, p0, T, Ly.kc, Ly.vc, V->q35_q, s);
                const float scale = 1.0f / std::sqrt((float) D);
                // STRATA_Q4X_QSA_DECODE (on for CUDA, off on HIP): the QSA layer's keys and attention for the window's
                // columns at once (below); the same bits
    #if defined(STRATA_USE_HIP)
                static const bool rows_attn = [] { const char* v = getenv("STRATA_Q4X_QSA_DECODE"); return v != nullptr && std::atoi(v) != 0; }();
    #else
                static const bool rows_attn = [] { const char* v = getenv("STRATA_Q4X_QSA_DECODE"); return v == nullptr || std::atoi(v) != 0; }();
    #endif
                if (Ly.q4_pooled != nullptr) {
                    // the indexer's raw keys (kept a layer for the rewind) and queries (BF16 rows against the float
                    // columns), the block keys a column
                    float* rawk = Ly.q4_tail != nullptr ? V->q4_rawk + (size_t) la++ * TM * ID : V->q4_iraw;
                    gf::MvJob ij[2];
                    ij[0] = {Ly.q4_ik.q, nullptr, V->x, rawk, nullptr, 1.0f, Ly.q4_ik.type, E, ID};
                    ij[1] = {Ly.q4_iq.q, nullptr, V->x, V->q4_iq, nullptr, 1.0f, Ly.q4_iq.type, E, IH * ID};
                    if (!q4x_mvn(ij, 2, T, s)) { err = "qwen4exp verify: the indexer projections"; return false; }
                    const k::QsaIndexerBuffers ib{Ly.q4_tail, Ly.q4_dead, Ly.q4_pooled, Ly.q4_bpos};
                    if (rows_attn)   // (the window's keys in two launches, the same buffers)
                        k::indexer_key_append_rows(rawk, ID, p0, T, 0, Ly.q4_ikn, g.norm_eps, ib, sh, F->q4_cos, F->q4_sin,
                                                   (void*) s);
                    else
                        for (int t = 0; t < T; ++t)
                            k::indexer_key_append(rawk + (size_t) t * ID, V->q4_steps + (size_t) t * 4 + k::kStepPos, 0,
                                                  Ly.q4_ikn, g.norm_eps, ib, sh, F->q4_cos, F->q4_sin, (void*) s);
                    bool any = false;
                    for (int t = 0; t < T && !any; ++t) any = k::qsa_selection_width(p0 + t + 1, sh) < p0 + t + 1;
                    if (any) {
                        k::rms_norm_weighted(V->q4_iq, Ly.q4_iqn, (int64_t) T * IH, ID, g.norm_eps, (void*) s);
                        k::rope_neox_apply(V->q4_iq, V->q4_iq, (int64_t) T * IH, ID, g.n_rot, F->q4_cos, F->q4_sin, ipos, (void*) s);
                    }
                }
                // the window's columns in one launch when they are all past 2,051 cells (one selection call for them all,
                // then each column's cells read in place) or all below (every cell); a window across the bound: a column
                // at a time
                bool cols_done = false;
                if (rows_attn && Ly.q4_pooled != nullptr && F->q4_pf_part != nullptr && T <= FastState::kQ4Grp) {
                    const bool sel_all = k::qsa_selection_width(p0 + 1, sh) < p0 + 1;
                    const bool dense_all = k::qsa_selection_width(p0 + T, sh) == p0 + T;
                    if (sel_all) {
                        k::qsa_block_scores(Ly.q4_pooled, Ly.q4_dead, V->q4_iq, V->q4_steps, T, F->q4_max_blocks, sh,
                                            F->q4_pf_scores, (void*) s);
                        k::qsa_block_topk(F->q4_pf_scores, V->q4_steps, T, F->q4_max_blocks, F->q4_cap, sh, F->q4_pf_ids,
                                          (void*) s);
                        cols_done = q35::attn_decode_rows(V->q35_q, Ly.kc, Ly.vc, F->q4_pf_ids, F->q4_cap, V->q4_steps, T,
                                                          F->q4_cap, g.n_head, g.attn_head_kv, D, scale, F->q4_pf_part,
                                                          V->q35_ao, s);
                    } else if (dense_all) {
                        cols_done = q35::attn_decode_rows(V->q35_q, Ly.kc, Ly.vc, nullptr, 0, V->q4_steps, T, (int) (p0 + T),
                                                          g.n_head, g.attn_head_kv, D, scale, F->q4_pf_part, V->q35_ao, s);
                    }
                }
                for (int t = 0; t < T && !cols_done; ++t) {
                    const int64_t p = p0 + t;
                    const int w = Ly.q4_pooled != nullptr ? (int) k::qsa_selection_width(p + 1, sh) : (int) (p + 1);
                    float* ao = V->q35_ao + (size_t) t * HD;
                    const float* q = V->q35_q + (size_t) t * HD;
                    if (w < p + 1) {
                        const int32_t* st = V->q4_steps + (size_t) t * 4;
                        k::qsa_block_scores(Ly.q4_pooled, Ly.q4_dead, V->q4_iq + (size_t) t * IH * ID, st, 1, F->q4_max_blocks, sh,
                                            F->q4_scores, (void*) s);
                        k::qsa_block_topk(F->q4_scores, st, 1, F->q4_max_blocks, F->q4_cap, sh, F->q4_ids, (void*) s);
                        q35::gather_rows(Ly.kc, Ly.vc, F->q4_ids, w, KV, F->q4_ks, F->q4_vs, s);
                        q35::attn_decode(q, F->q4_ks, F->q4_vs, g.n_head, g.attn_head_kv, D, w, scale, F->q35_part, ao, s);
                    } else {
                        q35::attn_decode(q, Ly.kc, Ly.vc, g.n_head, g.attn_head_kv, D, (int) (p + 1), scale, F->q35_part, ao, s);
                    }
                }
                q35::attn_gate_rows(V->q35_ao, V->q35_qf, T, g.n_head, D, s);
                gf::quantize_q8_1(V->q35_ao, V->q35_aoq, T * HD, s);
                gf::MvJob o = {Ly.out.q, V->q35_aoq, V->q35_ao, V->mixer, nullptr, 1.0f, Ly.out.type, HD, E};
                if (!q4x_mvn(&o, 1, T, s)) { err = "qwen4exp verify: attn_output"; return false; }
                if (prof) F->mark("q4w_attn");
            }
            abl_out(il, 0, V->mixer, T, E, s);
            for (int t = 0; t < T; ++t) q4x_trace(s, p0 + t, il, "c_mixer", V->mixer + (size_t) t * E, E);
            if (!q4x_mix_cols(V->R, T, Ly.hc4[1].norm, Ly.hc4[1].down, Ly.hc4[1].up, &Ly.hc4[1].inject, V->mixer,
                              V->q4_inj[0], V->q4_inj[1], V->x, V->xq, err))
                return false;
            if (prof) F->mark("q4w_hc_ffn");
            for (int t = 0; t < T; ++t) q4x_trace(s, p0 + t, il, "d_x2", V->x + (size_t) t * E, E);
            return !with_router || verify_moe_router(il, 0, T, s, err);
        };
        // stage 2 (STRATA_GLM_GRAPH_MOE, default on with the graphs): the MoE's dispatch in the same graph - the routes,
        // the waits and fetches, the experts and the CPU lane's wait all read their plans from device memory; what the
        // host decides per window stays outside (the request count, the paced background copies) and what the routes
        // take by value (the CPU lane's split, the helper, the tail skip) is the graph's key: another key re-captures
        const uint64_t sig = [&] {
            uint64_t h = F->cpu_plan * 0x9E3779B97F4A7C15ull;
            const auto mix = [&h](uint64_t v) { h = (h ^ v) * 0x100000001B3ull; };
            mix((uint64_t) (uintptr_t) (F->expert_helper && F->expert_helper->active && !F->expert_helper->failed
                                            ? F->expert_helper->owned_d : nullptr));
            mix((uint64_t) F->tail_skip);
            uint32_t w = 0;
            std::memcpy(&w, &F->tail_skip_w, sizeof w);
            mix(w);
            mix(F->tail_skip_disk ? 1u : 0u);
            return h;
        }();
        const auto seg_moe = [&]() -> bool {
            return seg(true) && verify_moe(il, 0, T, toks, s, nullptr, nullptr, err, true, true);
        };
        bool router_done = false;
        if (q4g_on && Ly.recr && il != l0_ && !(V->q4_ple_on && il == F->ple_il) && !prof) {
            const size_t gk = (size_t) il * (kVerifyMax + 1) + (size_t) T;
            if (V->q4g.empty()) {
                V->q4g.assign((size_t) g.n_layers * (kVerifyMax + 1), nullptr);
                V->q4g_n.assign(V->q4g.size(), 0);
                V->q4g_sig.assign(V->q4g.size(), 0);
            }
            cudaGraphExec_t& ex = V->q4g[gk];
            if (ex != nullptr && q4g_moe && V->q4g_sig[gk] != sig) {   // (a new key: captured again now)
                cudaGraphExecDestroy(ex);
                ex = nullptr;
                ++V->q4g_recap;
            }
            const uint64_t exp0 = F->expected;
            if (ex == nullptr && V->q4g_n[gk] < 255 && ++V->q4g_n[gk] >= 2) {
                const int li0 = li, la0 = la;
                // with PDL edges first (q4g_pdl), then once more without them if that capture or its instantiation
                // was refused
                for (int pass = q4g_pdl ? 0 : 1; pass < 2 && ex == nullptr; ++pass) {
                    li = li0;
                    la = la0;
                    F->expected = exp0;
                    err.clear();
                    cudaGraph_t gr = nullptr;
                    bool cok = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
                    if (cok) {
                        const k::PdlCapture pc(pass == 0);
                        cok = q4g_moe ? seg_moe() : seg(true);
                        const cudaError_t e = cudaStreamEndCapture(s, &gr);
                        cok = cok && e == cudaSuccess && gr != nullptr;
                    }
                    if (cok && cudaGraphInstantiate(&ex, gr, 0) != cudaSuccess) ex = nullptr;
                    if (!cok) ex = nullptr;
                    if (gr != nullptr) cudaGraphDestroy(gr);
                    cudaGetLastError();
                    if (pass == 0) ++(ex != nullptr ? V->q4g_pdl_ok : V->q4g_pdl_no);
                }
                li = li0;
                la = la0;
                F->expected = exp0;   // (the capture ran no route: counted again below)
                V->q4g_sig[gk] = sig;
                if (ex == nullptr) {   // (this layer and width stay launches)
                    V->q4g_n[gk] = 255;
                    std::fprintf(stderr, "qwen4exp verify: layer %d's graph (%d columns) did not capture%s%s - launches\n",
                                 il, T, err.empty() ? "" : ": ", err.c_str());
                    err.clear();
                }
            }
            if (ex != nullptr) {
                if (!glmfast::cuda_ok(cudaGraphLaunch(ex, s), "qwen4exp verify graph", err)) return false;
                ++li;   // (the segment's delta-net layer)
                router_done = true;
                ++V->q4g_rep;
            }
        }
        if (!router_done) {
            if (!seg(false)) return false;
            ++V->q4g_eager;
        }
        if (router_done && q4g_moe) {   // (the graph ran the MoE too: its T routes, then the paced copies)
            F->expected += (uint64_t) T;
            F->bg_drain(s, F->bg_pace);
        } else if (!verify_moe(il, 0, T, toks, s, nullptr, nullptr, err, router_done)) {
            return false;
        }
        if (prof) F->mark("q4w_moe");
        for (int t = 0; t < T; ++t) q4x_trace(s, p0 + t, il, "e_ffn", V->ffn + (size_t) t * E, E);
    }
    if (split_next_) {
        // a part before the last: the last FFN combined into the columns' streams, which cross over to the next part
        // through the pinned hop buffer (as a token's hop; its first layer reads combined residuals)
        q35::hc4_combine_norm(V->R, V->ffn, V->q4_inj[1], g.hc, F->L[(size_t) (l1_ - 1)].hc4[0].norm, g.norm_eps, E,
                              g.hc, T, V->q4_xn, s);
        if (!hop_link_->send(HopLink::kWindow, V->R, (size_t) T * HE * sizeof(float), s, err)) return false;
        if (prof) F->mark("q4w_hop");
        return next_port_->window(toks, p0, pos_, err);
    }
    // the output hyper-connection (the last FFN combined first) and the head
    if (!q4x_mix_cols(V->R, T, F->q4_head[0], F->q4_head_w[0], F->q4_head_w[1], nullptr, V->ffn, V->q4_inj[1], nullptr,
                      V->head_x, V->head_xq, err))
        return false;
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, V->head_xq, V->head_x, V->logits, nullptr, 1.0f, ow.type, E, g.n_vocab};
    if (gf::mvn_supported(ow.type)) {
        if (!q4x_mvn(&o, 1, T, s)) { err = "qwen4exp verify: output head"; return false; }
    } else {
        const size_t q8e = (size_t) (E / 32) * kQ8;
        for (int t = 0; t < T; ++t) {
            gf::MvJob ot = o;
            ot.xq = (const uint8_t*) V->head_xq + (size_t) t * q8e;
            ot.xf = V->head_x + (size_t) t * E;
            ot.y = V->logits + (size_t) t * g.n_vocab;
            if (!gf::mv(&ot, 1, s)) { err = "qwen4exp verify: output head (type " + std::to_string(ow.type) + ")"; return false; }
        }
    }
    if (F->zero_copy) {
        for (int t = 0; t < T; ++t) gf::argmax(V->logits + (size_t) t * g.n_vocab, g.n_vocab, V->tok_hd + t, s);
    } else {
        for (int t = 0; t < T; ++t) gf::argmax(V->logits + (size_t) t * g.n_vocab, g.n_vocab, V->tok + t, s);
        cudaMemcpyAsync(V->tok_h, V->tok, (size_t) T * sizeof(int), cudaMemcpyDeviceToHost, s);
    }
    if (prof) F->mark("q4w_head");
    cudaEventRecord(V->done, s);
    return true;
}

// the window's tail dropped (positions >= pos): every delta-net layer's state and conv state back to before the
// window, then the kept rows replayed from the window's saved rows and gates (the attention caches need nothing: a
// row past pos is rewritten before it is read)
void Glm5Model::verify_rewind_q35(int64_t pos) {
    auto* V = (VerifyBuf*) vb_;
    FastState* F = fast_;
    if (V == nullptr || F == nullptr || V->q35_p0 < 0) return;
    const int64_t p0 = V->q35_p0;
    const int kept = (int) (pos - p0), T = V->T, TM = V->TM;
    V->q35_p0 = -1;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    // the drafter's cache rows of the kept columns whose next token is known (the window's own); the last kept
    // column's waits for the token after it
    if (q35_mtp_on() && kept >= 1) {
        // (qwen4exp: the columns' final residuals)
        const size_t E = g.q4x ? (size_t) g.hc * g.n_embd : (size_t) g.n_embd;
        const float* h = g.q4x ? V->R : V->head_x;
        std::string err;
        if (kept >= 2 && !q35_mtp_rows(p0, kept - 1, h, E, V->q35_toks + 1, err))
            std::fprintf(stderr, "qwen35 mtp: the window's rows: %s\n", err.c_str());
        q35_mtp_pend_set(p0 + kept - 1, h + (size_t) (kept - 1) * E);
    }
    if (kept >= T) return;   // (every column kept: the states are after the window already)
    if (g.q4x && V->q4_nattn > 0 && V->q4_tsnap != nullptr) {
        // qwen4exp's QSA layers: the raw-key tail back to before the window, then the kept columns' keys again (a
        // dropped column in the next block wrote over slots of cells the kept block still pools)
        namespace k = strata::kernels;
        const k::QsaShapes sh = k::qsa_real_shapes();
        const size_t ID = (size_t) g.idx_key, tail_n = (size_t) (sh.idx_block - 1) * ID;
        int la = 0;
        for (int il = l0_; il < l1_; ++il) {
            const auto& Ly = F->L[(size_t) il];
            if (Ly.q4_tail == nullptr) continue;
            cudaMemcpyAsync(Ly.q4_tail, V->q4_tsnap + (size_t) la * tail_n, tail_n * sizeof(float), cudaMemcpyDeviceToDevice, s);
            if (Ly.q4_pooled != nullptr) {
                const k::QsaIndexerBuffers ib{Ly.q4_tail, Ly.q4_dead, Ly.q4_pooled, Ly.q4_bpos};
                for (int t = 0; t < kept; ++t)
                    k::indexer_key_append(V->q4_rawk + ((size_t) la * TM + t) * ID, V->q4_steps + (size_t) t * 4 + k::kStepPos,
                                          0, Ly.q4_ikn, g.norm_eps, ib, sh, F->q4_cos, F->q4_sin, (void*) s);
            }
            ++la;
        }
    }
    if (g.q4x && V->q4_ple_on && F->ple_hist != nullptr) {   // qwen4exp: the PLE conv history after the kept columns
        const size_t HE = (size_t) g.hc * g.n_embd;
        cudaMemcpyAsync(F->ple_hist, V->q4_ple_snap, (size_t) strata::kernels::NG_HIST * HE * sizeof(float),
                        cudaMemcpyDeviceToDevice, s);
        for (int t = 0; t < kept; ++t) strata::kernels::ple_history_advance(F->ple_hist, V->q4_qn + (size_t) t * HE, (void*) s);
    }
    const int C = g.ssm_conv_dim(), Sd = g.ssm_state, HK = g.ssm_k_heads, HV = g.ssm_v_heads;
    const size_t state_n = (size_t) Sd * HV * Sd, conv_n = (size_t) C * (g.d_conv - 1);
    int li = 0;
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (!Ly.recr) continue;
        cudaMemcpyAsync(Ly.gdn_state, V->q35_snap + (size_t) li * state_n, state_n * sizeof(float), cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(Ly.gdn_conv, V->q35_csnap + (size_t) li * conv_n, conv_n * sizeof(float), cudaMemcpyDeviceToDevice, s);
        if (kept > 0) {
            q35::conv_state_rows(V->q35_qkv + (size_t) li * TM * C, kept, C, g.d_conv, Ly.gdn_conv, s);
            q35::gdn_scan(Ly.gdn_state, V->q35_rows + (size_t) li * TM * C, kept, C, V->q35_alpha + (size_t) li * TM * HV,
                          V->q35_beta + (size_t) li * TM * HV, Sd, HK, HV, V->q35_o, s);
        }
        ++li;
    }
}

// ---------------------------------------------------------------- glm5-next (GLM-5.3-Flash): the batched window
// STRATA_GLM_WINDOW: verify windows on one part - the speculative decode's NextN drafter (the model's own
// blk.<n_layers>) or the n-gram drafts. Default: on for a CUDA integrated GPU with unified memory (DGX Spark / GB10:
// +14-16% decode, the same tokens), off elsewhere (a discrete card: the CPU lane's share makes it break-even or slower);
// =1 / =0 force it.  Exactness: every kernel of a column is the token path's (hc, kda_prep /
// kda_rec, dsa_prep / score / select / mla, swiglu_q8, hc_post, head_prep, the routes and experts' kernels) except the
// dense GEMVs, which are mvn_mv - mv()'s sums a column, the weights read once (STRATA_GLM_MVN_CHECK shows it) - and the
// column-batched experts (STRATA_DS4_WIN_COLS_CHECK).
bool Glm5Model::glm_window_env() {
    static const bool on = [] {
        const char* v = getenv("STRATA_GLM_WINDOW");
        if (v != nullptr) return std::atoi(v) != 0;
#if defined(STRATA_USE_HIP) || defined(_WIN32)
        return false;
#else
        const char* uv = getenv("STRATA_GLM_UMA");   // (the unified-memory policy off: the discrete defaults)
        if (uv != nullptr && std::atoi(uv) == 0) return false;
        int integrated = 0, dev = 0;   // (the model's device is current when the load asks)
        if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        return integrated != 0;
#endif
    }();
    return on;
}

namespace {
int glm_env_int(const char* name) {
    const char* v = getenv(name);
    return v != nullptr ? std::atoi(v) : 0;
}
}  // namespace

// STRATA_GLM_KDA_OOP (default 1): the window's column 0 writes its KDA state straight into kda_bak_ and column 1
// reads it from there (kda_rec out of place - the same sums), and a rewind replays from kda_bak_ the same way: no
// copy of S after column 0 nor back before a replay (2 x 146 MB of copies a window on GLM-5.3-Flash).  0: the copies
static bool glm_kda_oop() {
    static const bool on = [] {
        const char* v = getenv("STRATA_GLM_KDA_OOP");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

// one KDA layer's state (S then the conv rows, contiguous) <-> its slot li of kda_bak_ (or bak)
void Glm5Model::kda_copy_layer(int il, int li, bool restore, float* bak) {
    const Glm5Geometry& g = g_;
    const size_t run = (size_t) g.d_inner() * g.kda_head_dim + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
    float* live = state_ + kda_S_[(size_t) il];
    float* b = (bak != nullptr ? bak : kda_bak_) + (size_t) li * run;
    cudaMemcpyAsync(restore ? live : b, restore ? b : live, run * sizeof(float), cudaMemcpyDeviceToDevice, fast_->cs);
}

// the window's GEMVs: mvn_mv; STRATA_GLM_MVN_CHECK=1: every 16th call also as a mv() a column into a scratch copy of
// the outputs, compared bit by bit (a count every 64 checks and at the first differences)
// qwen4exp's verify window (issue #42): glm_mvn gives each column mv()'s sums, so greedy answers with drafts equal the
// token path's (v1.3.125) - on CUDA in one launch that reads the weights once (gf::mvn_mv's typed kernels), on HIP as a
// mv() a column (HIP's mv fuses small calls of mixed types; mvn_mv does not reproduce that): every dense weight and the
// output head read once per column, an R9700 on Qwen3.8 + MTP 87 -> 67 tok/s.  STRATA_Q4X_VERIFY_EXACT=1 / 0 forces
// either (STRATA_DETERMINISTIC sets 1); default 1 on CUDA, 0 on HIP (gf::mvn: the weights read once, as before v1.3.125
// - drafts can then part from spec=0 late in an answer, as they did there)
bool Glm5Model::q4x_mvn(const void* jobs, int n, int T, cudaStream_t s) {
    static const bool exact = [] {
        const char* v = getenv("STRATA_Q4X_VERIFY_EXACT");
#if defined(STRATA_USE_HIP)
        return v != nullptr && std::atoi(v) != 0;
#else
        return v == nullptr || std::atoi(v) != 0;
#endif
    }();
    return exact ? glm_mvn(jobs, n, T, s) : gf::mvn((const gf::MvJob*) jobs, n, T, s);
}

bool Glm5Model::glm_mvn(const void* jobs, int n, int T, cudaStream_t s) {
    const auto* j = (const gf::MvJob*) jobs;
    if (!gf::mvn_mv(j, n, T, s)) return false;
    static const bool chk = glm_env_int("STRATA_GLM_MVN_CHECK") != 0;
    if (!chk || T < 2) return true;
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;   // (a layer graph being captured: no host check)
    if (cudaStreamIsCapturing(s, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) return true;
    static unsigned long long calls = 0, checked = 0, cols = 0, diff = 0;
    if ((calls++ % 16) != 0) return true;
    auto* V = (VerifyBuf*) vb_;
    size_t need = 0;
    for (int i = 0; i < n; ++i) need += (size_t) T * j[i].n_out;
    if (V->mvn_ref_n < need) {
        if (V->mvn_ref) cudaFree(V->mvn_ref);
        V->mvn_ref = nullptr;
        V->mvn_ref_n = 0;
        if (cudaMalloc((void**) &V->mvn_ref, need * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            return true;
        }
        V->mvn_ref_n = need;
    }
    size_t off = 0;
    std::vector<size_t> offs((size_t) n);
    for (int c = 0; c < T; ++c) {
        gf::MvJob sub[gf::kMaxMvJobs];
        off = 0;
        for (int i = 0; i < n; ++i) {
            const gf::MvJob& J = j[i];
            sub[i] = J;
            const bool f = J.type == gf::kTypeBF16 || J.type == gf::kTypeF32;
            const size_t xs = J.xs ? (size_t) J.xs : (size_t) (f ? J.n_in : J.n_in / 32);
            if (f) {
                if (J.xf != nullptr) sub[i].xf = J.xf + (size_t) c * xs;
            } else {
                if (J.xq != nullptr) sub[i].xq = (const uint8_t*) J.xq + (size_t) c * xs * kQ8;
                if (J.xf != nullptr) sub[i].xf = J.xf + (size_t) c * J.n_in;
            }
            offs[(size_t) i] = off;
            sub[i].y = V->mvn_ref + off + (size_t) c * J.n_out;
            sub[i].xs = sub[i].ys = 0;
            off += (size_t) T * J.n_out;
        }
        if (!gf::mv(sub, n, s)) return false;
    }
    std::vector<float> a, b;
    cudaStreamSynchronize(s);
    for (int i = 0; i < n; ++i) {
        const gf::MvJob& J = j[i];
        const size_t ys = J.ys ? (size_t) J.ys : (size_t) J.n_out;
        a.resize((size_t) J.n_out);
        b.resize((size_t) J.n_out);
        for (int c = 0; c < T; ++c) {
            cudaMemcpy(a.data(), J.y + (size_t) c * ys, a.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), V->mvn_ref + offs[(size_t) i] + (size_t) c * J.n_out, b.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
            ++cols;
            if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
                ++diff;
                if (diff <= 8)
                    std::fprintf(stderr, "glm mvn check: DIFFERENT - job %d of %d (type %d, %d x %d), column %d of %d\n", i,
                                 n, J.type, J.n_out, J.n_in, c, T);
            }
        }
    }
    if ((++checked & 63) == 1)
        std::fprintf(stderr, "glm mvn check: %llu calls checked, %llu columns, %llu differ\n", checked, cols, diff);
    return true;
}

bool Glm5Model::verify_part_glm(const std::vector<int32_t>& toks, int64_t p0, std::string& err) {
    FastState* F = fast_;
    auto* V = (VerifyBuf*) vb_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int T = V->T, TM = V->TM;
    const size_t E = (size_t) g.n_embd, HCE = (size_t) g.hc * E, q8e = (E / 32) * kQ8;
    const int DI = g.d_inner(), HDk = g.kda_head_dim, KI = (int) V->glm_ki;
    std::vector<float*> Rc((size_t) T), Ro((size_t) T);
    for (int t = 0; t < T; ++t) {
        Rc[(size_t) t] = V->R + (size_t) t * 2 * HCE;
        Ro[(size_t) t] = Rc[(size_t) t] + HCE;
    }
    V->glm_p0 = p0;
    for (int t = 0; t < T; ++t) V->glm_toks[t] = toks[(size_t) t];
    if (q35_mtp_on() && !q35_mtp_pend_commit(toks[0], err)) return false;   // (the drafter: the last position's row)
    // the checks' copy of the KDA states before the window
    static const bool chk_pre = glm_env_int("STRATA_GLM_WINDOW_CHECK") != 0 || glm_env_int("STRATA_GLM_REWIND_CHECK") != 0;
    V->chk_pre_ok = false;
    if (chk_pre && V->glm_rec > 0) {
        if (V->chk_pre == nullptr &&
            cudaMalloc((void**) &V->chk_pre, (size_t) V->glm_rec * V->glm_run * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            V->chk_pre = nullptr;
        }
        if (V->chk_pre != nullptr) {
            for (int il = l0_; il < l1_; ++il)
                if (V->glm_li[(size_t) il] >= 0) kda_copy_layer(il, V->glm_li[(size_t) il], false, V->chk_pre);
            V->chk_pre_ok = true;
        }
    }
    const auto hc_read = [&](int il, int t, const float* block_out, bool attn) {
        const auto& Ly = F->L[(size_t) il];
        gf::HcArgs h;
        h.block_out = block_out;
        h.R_old = Rc[(size_t) t];
        h.R_new = Ro[(size_t) t];
        h.post_in = V->post + (size_t) t * 8;
        h.comb_in = V->comb + (size_t) t * 16;
        h.pre = V->pre + (size_t) t * 8;
        h.post = V->post + (size_t) t * 8;
        h.comb = V->comb + (size_t) t * 16;
        h.w_fn = attn ? Ly.hc_attn_fn : Ly.hc_ffn_fn;
        h.w_fn_q8 = attn ? Ly.hc_attn_fn8 : Ly.hc_ffn_fn8;
        h.w_scale = attn ? Ly.hc_attn_scale : Ly.hc_ffn_scale;
        h.w_base = attn ? Ly.hc_attn_base : Ly.hc_ffn_base;
        h.norm_w = attn ? Ly.attn_norm : Ly.ffn_norm;
        h.norm_eps = g.norm_eps;
        h.hc_eps = g.hc_eps;
        h.iters = g.sinkhorn_iters;
        h.n_embd = g.n_embd;
        h.x = V->x + (size_t) t * E;
        h.xq = (uint8_t*) V->xq + (size_t) t * q8e;
        h.part = V->hpart + (size_t) t * V->hpart_n;
        h.counter = V->hcnt + (size_t) t * 16;
        if (attn && block_out != nullptr && V->hc_cpu[t]) {   // the column's last MoE answer, added here
            h.cpu_seq = V->md[(size_t) t].cpu_seq;
            h.cpu_ans = V->md[(size_t) t].cpu_ans;
            V->hc_cpu[t] = false;
        }
        gf::hc(h, s);
        if (block_out != nullptr) std::swap(Rc[(size_t) t], Ro[(size_t) t]);
        if (attn && V->bg_after[t]) {
            V->bg_after[t] = false;
            F->bg_drain(s, F->bg_pace);
        }
    };
    const bool prof = F->prof_on;
    static std::vector<std::string> names;
    if (prof && names.empty())
        for (int n = 1; n <= kVerifyMax; ++n)
            for (const char* ph : {"hc_attn", "attn", "hc_ffn", "moe", "head"}) names.push_back("v" + std::to_string(n) + "_" + ph);
    const auto mark = [&](int ph) { if (prof) F->mark(names[(size_t) (T - 1) * 5 + ph].c_str()); };
    if (prof) F->mark("v_start");
    for (int t = 0; t < kVerifyMax; ++t) V->hc_cpu[t] = V->bg_after[t] = false;
    const float prescale = 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads));
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        for (int t = 0; t < T; ++t) hc_read(il, t, il != l0_ ? V->ffn + (size_t) t * E : nullptr, true);
        mark(0);
        if (Ly.recr) {
            // ---- KDA: the projections over the columns (into the replay inputs), then column after column the token
            //      path's conv + recurrence on the live state; the state after column 0 kept
            const int li = V->glm_li[(size_t) il];
            float* kin = V->glm_kin + (size_t) li * TM * V->glm_ki;
            const auto bq = [&](const uint16_t* w16, const void* w8, int off, int n_out) {
                return w8 != nullptr ? gf::MvJob{w8, V->xq, V->x, kin + off, nullptr, 1.0f, 8, (int) E, n_out, 0, KI}
                                     : gf::MvJob{w16, nullptr, V->x, kin + off, nullptr, 1.0f, gf::kTypeBF16, (int) E, n_out, 0, KI};
            };
            gf::MvJob j[6];
            j[0] = {Ly.q.q, V->xq, V->x, kin, nullptr, 1.0f, Ly.q.type, (int) E, DI, 0, KI};
            j[1] = {Ly.k.q, V->xq, V->x, kin + DI, nullptr, 1.0f, Ly.k.type, (int) E, DI, 0, KI};
            j[2] = {Ly.v.q, V->xq, V->x, kin + 2 * DI, nullptr, 1.0f, Ly.v.type, (int) E, DI, 0, KI};
            j[3] = bq(Ly.f_a, Ly.f_a8, 3 * DI, HDk);
            j[4] = bq(Ly.g_a, Ly.g_a8, 3 * DI + HDk, HDk);
            j[5] = bq(Ly.beta, Ly.beta8, 3 * DI + 2 * HDk, g.n_head);
            if (!glm_mvn(j, 6, T, s)) { err = "glm verify: kda projections"; return false; }
            for (int t = 0; t < T; ++t) {
                float* in = kin + (size_t) t * KI;
                gf::KdaPrepArgs k;
                for (int i = 0; i < 3; ++i) {
                    k.proj[i] = in + (size_t) i * DI;
                    k.conv_w[i] = Ly.conv[i];
                    k.out[i] = F->conv[i];
                }
                k.conv_state = state_ + kda_conv_[(size_t) il];
                k.fa = in + 3 * DI;
                k.ga = in + 3 * DI + HDk;
                k.f_b = Ly.f_b;
                k.g_b = Ly.g_b;
                k.f_b_q8 = Ly.f_b8;
                k.g_b_q8 = Ly.g_b8;
                k.dt_bias = Ly.dt_bias;
                k.ssm_a = Ly.ssm_a;
                k.lower_bound = g.kda_lb;
                k.g1 = F->g1;
                k.g2 = F->g2;
                k.n_head = g.n_head;
                k.head_dim = HDk;
                k.d_conv = g.d_conv;
                gf::kda_prep(k, s);
                float* const live = state_ + kda_S_[(size_t) il];
                float* const bak = kda_bak_ + (size_t) li * V->glm_run;   // (S, then the conv rows)
                // (out of place: column 0 live -> bak, column 1 bak -> live, the rest in place)
                const bool oop = glm_kda_oop() && T > 1 && t < 2;
                gf::kda_rec(F->conv[0], F->conv[1], F->conv[2], F->g1, in + 3 * DI + 2 * HDk, oop && t == 0 ? bak : live,
                            F->g2, Ly.ssm_norm, g.norm_eps, g.n_head, HDk, (uint8_t*) V->glm_gq + (size_t) t * V->glm_gq_b, s,
                            oop ? (t == 0 ? live : bak) : nullptr);
                if (t == 0) {   // (column 0 is always kept: the rewind starts here)
                    if (oop) {
                        const size_t sn = (size_t) DI * HDk;
                        cudaMemcpyAsync(bak + sn, live + sn, (V->glm_run - sn) * sizeof(float), cudaMemcpyDeviceToDevice, s);
                    } else {
                        kda_copy_layer(il, li, false);
                    }
                }
            }
            gf::MvJob o = {Ly.out.q, V->glm_gq, nullptr, V->mixer, nullptr, 1.0f, Ly.out.type, DI, (int) E};
            if (!glm_mvn(&o, 1, T, s)) { err = "glm verify: kda output"; return false; }
        } else {
            // ---- MLA + DSA: the projections over the columns, the rows in position order (latent, indexer key / gate,
            //      a completed pool's key), q_b, then a column at a time the pool scores, the selection, the attention
            gf::MvJob j[5];
            j[0] = {Ly.q_a.q, V->xq, V->x, V->qr_raw, nullptr, 1.0f, Ly.q_a.type, (int) E, g.q_lora};
            j[1] = {Ly.kv_a.q, V->xq, V->x, V->glm_kv, nullptr, 1.0f, Ly.kv_a.type, (int) E, g.kv_lora};
            j[2] = Ly.idx_k8 != nullptr ? gf::MvJob{Ly.idx_k8, V->xq, V->x, V->glm_ik, nullptr, 1.0f, 8, (int) E, g.idx_key}
                                        : gf::MvJob{Ly.idx_k, nullptr, V->x, V->glm_ik, nullptr, 1.0f, gf::kTypeBF16, (int) E, g.idx_key};
            j[3] = Ly.idx_gate8 != nullptr
                       ? gf::MvJob{Ly.idx_gate8, V->xq, V->x, V->glm_ig, nullptr, 1.0f, 8, (int) E, g.idx_key}
                       : gf::MvJob{Ly.idx_gate, nullptr, V->x, V->glm_ig, nullptr, 1.0f, gf::kTypeBF16, (int) E, g.idx_key};
            j[4] = {Ly.idx_proj, nullptr, V->x, V->iw, nullptr, prescale, gf::kTypeBF16, (int) E, g.idx_heads};
            if (!glm_mvn(j, 5, T, s)) { err = "glm verify: dsa projections"; return false; }
            for (int t = 0; t < T; ++t) {
                gf::DsaPrepArgs d;
                d.qr_raw = V->qr_raw + (size_t) t * g.q_lora;
                d.q_a_norm = Ly.q_a_norm;
                d.qr = V->qr + (size_t) t * g.q_lora;
                d.qr_q = (uint8_t*) V->qr_q + (size_t) t * (g.q_lora / 32) * kQ8;
                d.q_lora = g.q_lora;
                d.kv_raw = V->glm_kv + (size_t) t * g.kv_lora;
                d.kv_norm = Ly.kv_a_norm;
                d.lat = (uint16_t*) (state_ + dsa_lat_[(size_t) il]);
                d.kv_lora = g.kv_lora;
                d.lat_q8 = lat_q8_;
                d.ik_raw = V->glm_ik + (size_t) t * g.idx_key;
                d.k_norm_w = Ly.k_norm_w;
                d.k_norm_b = Ly.k_norm_b;
                d.ik_cache = state_ + dsa_ik_[(size_t) il];
                d.ig_raw = V->glm_ig + (size_t) t * g.idx_key;
                d.ig_cache = state_ + dsa_ig_[(size_t) il];
                d.ape = Ly.ape;
                d.pooled = state_ + dsa_pool_[(size_t) il];
                d.idx_key = g.idx_key;
                d.kpool = g.idx_kpool;
                d.ring = ik_ring_;
                d.p = (int) (p0 + t);
                d.eps = g.norm_eps;
                gf::dsa_prep(d, s);
            }
            const int IQ = g.idx_heads * g.idx_key;
            gf::MvJob j2[2];
            j2[0] = {Ly.q_b.q, V->qr_q, V->qr, V->glm_q, nullptr, 1.0f, Ly.q_b.type, g.q_lora, g.n_head * g.qk_nope};
            j2[1] = Ly.idx_q_b8 != nullptr ? gf::MvJob{Ly.idx_q_b8, V->qr_q, V->qr, V->iq, nullptr, 1.0f, 8, g.q_lora, IQ}
                                           : gf::MvJob{Ly.idx_q_b, nullptr, V->qr, V->iq, nullptr, 1.0f, gf::kTypeBF16, g.q_lora, IQ};
            if (!glm_mvn(j2, 2, T, s)) { err = "glm verify: dsa q"; return false; }
            // STRATA_GLM_MLA_COLS (default 1): the absorbed k_b / v_b GEMVs of the columns in one launch each
            // (gf::mla_cols: each weight read once, every column's sums mla()'s); 0: mla() a column
            static const bool mla_cols_on = [] {
                const char* v = getenv("STRATA_GLM_MLA_COLS");
                return v == nullptr || std::atoi(v) != 0;
            }();
            int n_sels[kVerifyMax];
            for (int t = 0; t < T; ++t) {
                const int64_t p = p0 + t;
                const int pool_done = (int) ((p + 1) / g.idx_kpool);
                float* sc = V->glm_score + (size_t) t * V->glm_score_n;
                int* cells = V->glm_cells + (size_t) t * V->glm_nsel;
                if (pool_done > 0)
                    gf::dsa_score(V->iq + (size_t) t * IQ, state_ + dsa_pool_[(size_t) il], V->iw + (size_t) t * g.idx_heads,
                                  g.idx_key, g.idx_heads, pool_done, sc, s);
                const int top_pools = std::min(g.top_pools_max(), pool_done);
                const int n_sel = g.idx_kpool * top_pools + (g.idx_select_tail ? g.idx_kpool - 1 : 0);
                n_sels[t] = n_sel;
                gf::dsa_select(sc, pool_done, g.idx_kpool, top_pools, n_sel, (int) p, cells, s);
            }
            const bool batched = mla_cols_on && V->glm_aq_b % kQ8 == 0 &&
                                 gf::mla_cols(T, V->glm_q, (size_t) g.n_head * g.qk_nope, Ly.k_b, Ly.v_b,
                                              (const uint16_t*) (state_ + dsa_lat_[(size_t) il]), V->glm_cells,
                                              (size_t) V->glm_nsel, n_sels, g.n_head, g.qk_nope, g.kv_lora, g.v_head,
                                              V->glm_aq, V->glm_aq_b / kQ8, s, lat_q8_, Ly.k_b8, Ly.v_b8);
            for (int t = 0; t < T && !batched; ++t)
                gf::mla(V->glm_q + (size_t) t * g.n_head * g.qk_nope, Ly.k_b, Ly.v_b,
                        (const uint16_t*) (state_ + dsa_lat_[(size_t) il]), V->glm_cells + (size_t) t * V->glm_nsel,
                        n_sels[t], g.n_head, g.qk_nope, g.kv_lora, g.v_head, (uint8_t*) V->glm_aq + (size_t) t * V->glm_aq_b,
                        s, lat_q8_, Ly.k_b8, Ly.v_b8);
            gf::MvJob o = {Ly.out.q, V->glm_aq, nullptr, V->mixer, nullptr, 1.0f, Ly.out.type, g.n_head * g.v_head, (int) E};
            if (!glm_mvn(&o, 1, T, s)) { err = "glm verify: dsa output"; return false; }
        }
        mark(1);
        for (int t = 0; t < T; ++t) hc_read(il, t, V->mixer + (size_t) t * E, false);
        mark(2);
        if (!Ly.moe) {
            gf::MvJob jd[2];
            jd[0] = {Ly.ffn_gate.q, V->xq, V->x, V->glm_dg, nullptr, 1.0f, Ly.ffn_gate.type, (int) E, g.n_ff_dense};
            jd[1] = {Ly.ffn_up.q, V->xq, V->x, V->glm_du, nullptr, 1.0f, Ly.ffn_up.type, (int) E, g.n_ff_dense};
            if (!glm_mvn(jd, 2, T, s)) { err = "glm verify: dense ffn"; return false; }
            for (int t = 0; t < T; ++t)
                gf::swiglu_q8(V->glm_dg + (size_t) t * g.n_ff_dense, V->glm_du + (size_t) t * g.n_ff_dense, g.swiglu_shexp,
                              g.n_ff_dense, (uint8_t*) V->glm_dhq + (size_t) t * (g.n_ff_dense / 32) * kQ8, s);
            gf::MvJob dn = {Ly.ffn_down.q, V->glm_dhq, nullptr, V->ffn, nullptr, 1.0f, Ly.ffn_down.type, g.n_ff_dense,
                            (int) E};
            if (!glm_mvn(&dn, 1, T, s)) { err = "glm verify: dense ffn down"; return false; }
        } else if (!verify_moe(il, 0, T, toks, s, nullptr, nullptr, err)) {
            return false;
        }
        mark(3);
    }
    for (int t = 0; t < T; ++t) {   // (verify_moe leaves no answer past the last layer; the check is for safety)
        if (V->hc_cpu[t]) {
            gf::moe_cpu_wait(V->md[(size_t) t], g.n_embd, V->ffn + (size_t) t * E, s);
            V->hc_cpu[t] = false;
        }
        if (V->bg_after[t]) {
            V->bg_after[t] = false;
            F->bg_drain(s, F->bg_pace);
        }
    }
    const auto nw = w_.find("output_norm.weight");
    if (nw == w_.end()) { err = "glm verify: output_norm.weight"; return false; }
    for (int t = 0; t < T; ++t) {
        gf::hc_post(V->ffn + (size_t) t * E, Rc[(size_t) t], V->post + (size_t) t * 8, V->comb + (size_t) t * 16, g.n_embd,
                    Ro[(size_t) t], s);
        gf::head_prep(Ro[(size_t) t], nw->second, g.norm_eps, g.n_embd, V->head_x + (size_t) t * E,
                      (uint8_t*) V->head_xq + (size_t) t * q8e, s);
    }
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, V->head_xq, V->head_x, V->logits, nullptr, 1.0f, ow.type, g.n_embd, g.n_vocab};
    if (!glm_mvn(&o, 1, T, s)) { err = "glm verify: output head (type " + std::to_string(ow.type) + ")"; return false; }
    if (F->zero_copy) {
        for (int t = 0; t < T; ++t) gf::argmax(V->logits + (size_t) t * g.n_vocab, g.n_vocab, V->tok_hd + t, s);
    } else {
        for (int t = 0; t < T; ++t) gf::argmax(V->logits + (size_t) t * g.n_vocab, g.n_vocab, V->tok + t, s);
        cudaMemcpyAsync(V->tok_h, V->tok, (size_t) T * sizeof(int), cudaMemcpyDeviceToHost, s);
    }
    mark(4);
    cudaEventRecord(V->done, s);
    return true;
}

namespace {
// a KDA layer's columns c0 .. c1 - 1 replayed from the window's saved inputs onto the state at S (S, then the conv rows)
void glm_kda_replay(const Glm5Geometry& g, const float* kin, int KI, int c0, int c1, float* S,
                    const float* const conv_w[3], const uint16_t* f_b, const uint16_t* g_b, const void* f_b8,
                    const void* g_b8, const float* dt_bias, const float* ssm_a, const float* ssm_norm, float* const conv[3],
                    float* g1, float* g2, void* gq, cudaStream_t s, const float* S_in = nullptr) {
    // (S_in: column c0 reads its state from there - out of place, kda_rec's state_in - the conv rows are S's own)
    const int DI = g.d_inner(), HDk = g.kda_head_dim;
    for (int t = c0; t < c1; ++t) {
        const float* in = kin + (size_t) t * KI;
        gf::KdaPrepArgs k;
        for (int i = 0; i < 3; ++i) {
            k.proj[i] = in + (size_t) i * DI;
            k.conv_w[i] = conv_w[i];
            k.out[i] = conv[i];
        }
        k.conv_state = S + (size_t) DI * HDk;
        k.fa = in + 3 * DI;
        k.ga = in + 3 * DI + HDk;
        k.f_b = f_b;
        k.g_b = g_b;
        k.f_b_q8 = f_b8;
        k.g_b_q8 = g_b8;
        k.dt_bias = dt_bias;
        k.ssm_a = ssm_a;
        k.lower_bound = g.kda_lb;
        k.g1 = g1;
        k.g2 = g2;
        k.n_head = g.n_head;
        k.head_dim = HDk;
        k.d_conv = g.d_conv;
        gf::kda_prep(k, s);
        gf::kda_rec(conv[0], conv[1], conv[2], g1, in + 3 * DI + 2 * HDk, S, g2, ssm_norm, g.norm_eps, g.n_head, HDk, gq, s,
                    t == c0 ? S_in : nullptr);
    }
}
}  // namespace

// the window's tail dropped (positions >= pos): the drafter's rows of the kept columns; every KDA layer back to its
// state after column 0, then columns 1 .. kept - 1 replayed (the DSA caches need nothing: a row past pos is rewritten
// before anything reads it, and a pool a dropped column completed is recomputed when its last position runs again)
void Glm5Model::verify_rewind_glm(int64_t pos) {
    auto* V = (VerifyBuf*) vb_;
    FastState* F = fast_;
    if (V == nullptr || F == nullptr || V->glm_p0 < 0) return;
    const int64_t p0 = V->glm_p0;
    const int kept = (int) (pos - p0), T = V->T, TM = V->TM;
    V->glm_p0 = -1;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const size_t E = (size_t) g.n_embd;
    if (q35_mtp_on() && kept >= 1) {
        std::string err;
        if (kept >= 2 && !q35_mtp_rows(p0, kept - 1, V->head_x, E, V->glm_toks + 1, err))
            std::fprintf(stderr, "glm mtp: the window's rows: %s\n", err.c_str());
        q35_mtp_pend_set(p0 + kept - 1, V->head_x + (size_t) (kept - 1) * E);
    }
    if (kept >= T || kept < 1) return;   // (every column kept: the states are after the window already)
    const int KI = (int) V->glm_ki;
    for (int il = l0_; il < l1_; ++il) {
        const int li = V->glm_li[(size_t) il];
        if (li < 0) continue;
        const auto& Ly = F->L[(size_t) il];
        // (out of place: only the conv rows back, then the first replayed column reads S from kda_bak_)
        const bool oop = glm_kda_oop() && kept > 1;
        float* const live = state_ + kda_S_[(size_t) il];
        float* const bak = kda_bak_ + (size_t) li * V->glm_run;
        if (oop) {
            const size_t sn = (size_t) g.d_inner() * g.kda_head_dim;
            cudaMemcpyAsync(live + sn, bak + sn, (V->glm_run - sn) * sizeof(float), cudaMemcpyDeviceToDevice, s);
        } else {
            kda_copy_layer(il, li, true);
        }
        if (kept > 1)
            glm_kda_replay(g, V->glm_kin + (size_t) li * TM * V->glm_ki, KI, 1, kept, live, Ly.conv,
                           Ly.f_b, Ly.g_b, Ly.f_b8, Ly.g_b8, Ly.dt_bias, Ly.ssm_a, Ly.ssm_norm, F->conv, F->g1, F->g2,
                           V->glm_gq, s, oop ? bak : nullptr);
    }
    // STRATA_GLM_REWIND_CHECK=1: the state before the window replayed through columns 0 .. kept - 1 (scratch) against the
    // live state, bit by bit - the kept count, the rows and the layers the rewind used
    static const bool rchk = glm_env_int("STRATA_GLM_REWIND_CHECK") != 0;
    if (!rchk || !V->chk_pre_ok) return;
    const size_t run = V->glm_run;
    if (V->chk_ref == nullptr && cudaMalloc((void**) &V->chk_ref, run * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        V->chk_ref = nullptr;
        return;
    }
    static unsigned long long n_chk = 0, n_layers = 0, n_diff = 0, by_kept[Glm5Model::kVerifyMax + 1] = {};
    std::vector<float> a(run), b(run);
    int bad = 0;
    for (int il = l0_; il < l1_; ++il) {
        const int li = V->glm_li[(size_t) il];
        if (li < 0) continue;
        const auto& Ly = F->L[(size_t) il];
        cudaMemcpyAsync(V->chk_ref, V->chk_pre + (size_t) li * run, run * sizeof(float), cudaMemcpyDeviceToDevice, s);
        glm_kda_replay(g, V->glm_kin + (size_t) li * TM * V->glm_ki, KI, 0, kept, V->chk_ref, Ly.conv, Ly.f_b, Ly.g_b,
                       Ly.f_b8, Ly.g_b8, Ly.dt_bias, Ly.ssm_a, Ly.ssm_norm, F->conv, F->g1, F->g2, V->glm_gq, s);
        cudaMemcpyAsync(a.data(), V->chk_ref, run * sizeof(float), cudaMemcpyDeviceToHost, s);
        cudaMemcpyAsync(b.data(), state_ + kda_S_[(size_t) il], run * sizeof(float), cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        ++n_layers;
        if (std::memcmp(a.data(), b.data(), run * sizeof(float)) != 0) {
            ++n_diff;
            ++bad;
        }
    }
    ++n_chk;
    ++by_kept[kept];
    std::fprintf(stderr, "glm rewind check: %llu rewinds (kept 1..7: %llu %llu %llu %llu %llu %llu %llu), %llu layer "
                             "states compared, %llu differ%s\n", n_chk, by_kept[1], by_kept[2], by_kept[3], by_kept[4],
                     by_kept[5], by_kept[6], by_kept[7], n_layers, n_diff, bad > 0 ? " (this one)" : "");
}

// STRATA_GLM_WINDOW_CHECK=1: after each window, the token path from the state before it on column 0's token - its
// logits against the window's column 0 and its KDA states against the window's state after column 0, bit by bit; =2:
// every column's token (the logits of each column, and the states after the window).  Then the window's state again.
bool Glm5Model::glm_window_check(const std::vector<int32_t>& toks, int64_t p0, std::string& err) {
    static const int mode = glm_env_int("STRATA_GLM_WINDOW_CHECK");
    auto* V = (VerifyBuf*) vb_;
    if (mode <= 0 || V == nullptr || !V->chk_pre_ok) return true;
    const Glm5Geometry& g = g_;
    FastState* F = fast_;
    cudaStream_t s = F->cs;
    const int T = V->T, n = mode >= 2 ? T : 1;
    const size_t run = V->glm_run, NV = (size_t) g.n_vocab;
    if (V->chk_post == nullptr &&
        cudaMalloc((void**) &V->chk_post, (size_t) V->glm_rec * run * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        V->chk_post = nullptr;
        return true;
    }
    std::vector<float> wl((size_t) n * NV), tl(NV);
    cudaMemcpyAsync(wl.data(), V->logits, wl.size() * sizeof(float), cudaMemcpyDeviceToHost, s);
    std::vector<int> wtok(V->tok_h, V->tok_h + T);
    for (int il = l0_; il < l1_; ++il)
        if (V->glm_li[(size_t) il] >= 0) kda_copy_layer(il, V->glm_li[(size_t) il], false, V->chk_post);
    for (int il = l0_; il < l1_; ++il)
        if (V->glm_li[(size_t) il] >= 0) kda_copy_layer(il, V->glm_li[(size_t) il], true, V->chk_pre);
    const int64_t pend = q35_pend_pos_, done = q35_row_done_, pos_after = pos_;
    const int last = last_tok_;
    q35_pend_pos_ = -1;
    pos_ = p0;
    static unsigned long long n_win = 0, n_cols = 0, n_ldiff = 0, n_tdiff = 0, n_sdiff = 0, n_states = 0;
    int bad = 0;
    const auto cmp_states = [&](float* ref) {
        std::vector<float> a(run), b(run);
        for (int il = l0_; il < l1_; ++il) {
            const int li = V->glm_li[(size_t) il];
            if (li < 0) continue;
            cudaMemcpyAsync(a.data(), state_ + kda_S_[(size_t) il], run * sizeof(float), cudaMemcpyDeviceToHost, s);
            cudaMemcpyAsync(b.data(), ref + (size_t) li * run, run * sizeof(float), cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
            ++n_states;
            if (std::memcmp(a.data(), b.data(), run * sizeof(float)) != 0) {
                ++n_sdiff;
                ++bad;
            }
        }
    };
    // the DSA rows the window wrote (latent, indexer key / gate, the pools it closed), to compare with the re-run's
    const auto dsa_rows = [&](std::vector<uint8_t>& out) {
        out.clear();
        cudaStreamSynchronize(s);
        const size_t lat_b = lat_q8_ ? (size_t) gf::lat8_rec_bytes(g.kv_lora) : (size_t) g.kv_lora * 2;
        const size_t kb = (size_t) g.idx_key * sizeof(float);
        for (int il = l0_; il < l1_; ++il) {
            if (V->glm_li[(size_t) il] >= 0) continue;
            for (int t = 0; t < n; ++t) {
                const int64_t p = p0 + t;
                const size_t o = out.size();
                out.resize(o + lat_b + 2 * kb + ((p + 1) % g.idx_kpool == 0 ? kb : 0));
                cudaMemcpy(out.data() + o, (const uint8_t*) (state_ + dsa_lat_[(size_t) il]) + (size_t) p * lat_b, lat_b,
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(out.data() + o + lat_b, state_ + dsa_ik_[(size_t) il] + (size_t) g.idx_key * (p % ik_ring_), kb,
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(out.data() + o + lat_b + kb, state_ + dsa_ig_[(size_t) il] + (size_t) g.idx_key * (p % ik_ring_), kb,
                           cudaMemcpyDeviceToHost);
                if ((p + 1) % g.idx_kpool == 0)
                    cudaMemcpy(out.data() + o + lat_b + 2 * kb,
                               state_ + dsa_pool_[(size_t) il] + (size_t) g.idx_key * ((p + 1) / g.idx_kpool - 1), kb,
                               cudaMemcpyDeviceToHost);
            }
        }
    };
    std::vector<uint8_t> rows_w, rows_t;
    dsa_rows(rows_w);
    static unsigned long long n_rdiff = 0;
    for (int t = 0; t < n; ++t) {
        if (!fast_token(toks[(size_t) t], err)) return false;
        cudaMemcpy(tl.data(), sc_ + sc_logits, NV * sizeof(float), cudaMemcpyDeviceToHost);
        ++n_cols;
        if (std::memcmp(tl.data(), wl.data() + (size_t) t * NV, NV * sizeof(float)) != 0) {
            ++n_ldiff;
            ++bad;
            if (n_ldiff <= 8) {
                double md = 0.0;
                for (size_t i = 0; i < NV; ++i) md = std::max(md, (double) std::fabs(tl[i] - wl[(size_t) t * NV + i]));
                std::fprintf(stderr, "glm window check: column %d of %d at position %lld: logits DIFFER (max |d| %.3g)%s\n",
                             t, T, (long long) (p0 + t), md, last_tok_ != wtok[(size_t) t] ? ", argmax too" : "");
            }
        }
        if (last_tok_ != wtok[(size_t) t]) ++n_tdiff;
        if (t == 0) cmp_states(kda_bak_);
    }
    if (n == T) cmp_states(V->chk_post);
    dsa_rows(rows_t);
    if (rows_t != rows_w) {
        ++n_rdiff;
        ++bad;
    }
    for (int il = l0_; il < l1_; ++il)
        if (V->glm_li[(size_t) il] >= 0) kda_copy_layer(il, V->glm_li[(size_t) il], true, V->chk_post);
    cudaStreamSynchronize(s);
    pos_ = pos_after;
    q35_pend_pos_ = pend;
    q35_row_done_ = done;
    last_tok_ = last;
    ++n_win;
    static unsigned long long by_mod[4] = {}, by_w[Glm5Model::kVerifyMax + 1] = {};
    ++by_mod[p0 & 3];
    ++by_w[T];
    std::fprintf(stderr, "glm window check: %llu windows (p0 %% 4 = 0..3: %llu %llu %llu %llu; widths 2..6: %llu %llu %llu %llu "
                         "%llu), %llu columns re-run on the token path - logits differ in %llu (argmax in %llu), KDA states "
                         "compared %llu, differ %llu; DSA rows differ in %llu windows%s\n", n_win, by_mod[0], by_mod[1], by_mod[2], by_mod[3], by_w[2], by_w[3],
                 by_w[4], by_w[5], by_w[6], n_cols, n_ldiff, n_tdiff, n_states, n_sdiff, n_rdiff, bad > 0 ? " (this one)" : "");
    return true;
}

int Glm5Model::verify_argmax(int i) const {
    if (next_port_ != nullptr) return i >= 0 && (size_t) i < win_tok_.size() ? win_tok_[(size_t) i] : -1;
    const auto* V = (const VerifyBuf*) verify_tail()->vb_;
    return V != nullptr && i >= 0 && i < V->T ? V->tok_h[i] : -1;
}

bool Glm5Model::verify_logits(int i, std::vector<float>& out, std::string& err) {
    if (next_port_ != nullptr) {
        const bool ok = next_port_->window_logits(i, out, err);
        cudaSetDevice(dev_);
        return ok;
    }
    const Glm5Model* tail = verify_tail();
    auto* V = (VerifyBuf*) tail->vb_;
    if (V == nullptr || i < 0 || i >= V->T) {
        err = "deepseek4 verify: no such column";
        return false;
    }
    cudaSetDevice(tail->dev_);
    out.resize((size_t) g_.n_vocab);
    const bool ok = cudaMemcpy(out.data(), V->logits + (size_t) i * g_.n_vocab, out.size() * sizeof(float),
                               cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaSetDevice(dev_);
    if (!ok) {
        err = "deepseek4 verify: logits copy";
        return false;
    }
    return true;
}

int Glm5Model::verify_sample(int i, strata::kernels::SamplerParams& sp, std::string& err) {
    if (next_port_ != nullptr) {
        if (sp.greedy) return verify_argmax(i);
        const int tok = next_port_->window_sample(i, sp, err);
        cudaSetDevice(dev_);
        return tok;
    }
    Glm5Model* tail = this;
    while (tail->split_next_) tail = tail->split_next_.get();
    auto* V = (VerifyBuf*) tail->vb_;
    if (V == nullptr || i < 0 || i >= V->T) {
        err = "deepseek4 verify: no such column";
        return -1;
    }
    if (sp.greedy) {
        // the end-of-turn guard reads the sampler's row (end_guard_pick): the column's row there, as a sampled pick
        // leaves it - else the guard saw another position's logits and let the pick stand
        if (guard_ids_ != nullptr) {
            cudaSetDevice(tail->dev_);
            cudaMemcpyAsync(tail->sc_ + tail->sc_logits, V->logits + (size_t) i * g_.n_vocab,
                            (size_t) g_.n_vocab * sizeof(float), cudaMemcpyDeviceToDevice, tail->fast_->cs);
            cudaStreamSynchronize(tail->fast_->cs);
            cudaSetDevice(dev_);
        }
        return V->tok_h[i];
    }
    // all columns at once (STRATA_GLM_SAMPLE_COLS=0: one column a call, the old way): one launch and one wait a window
    // instead of a copy, a launch and a wait per column
    static const bool cols = [] { const char* v = getenv("STRATA_GLM_SAMPLE_COLS"); return !(v && v[0] == '0'); }();
    if (cols && V->T <= 32) {
        if (i == 0) {
            V->samp_ok = false;
            cudaSetDevice(tail->dev_);
            FastState* TF = tail->fast_;
            if (TF->zero_copy) {
                strata::kernels::sample_tokens(V->logits, V->T, (int) g_.n_vocab, nullptr, 0, sp, V->tok_hd + 32, TF->cs);
            } else {
                strata::kernels::sample_tokens(V->logits, V->T, (int) g_.n_vocab, nullptr, 0, sp, V->tok + 32, TF->cs);
                cudaMemcpyAsync(V->tok_h + 32, V->tok + 32, (size_t) V->T * sizeof(int), cudaMemcpyDeviceToHost, TF->cs);
            }
            const bool ok = glmfast::cuda_ok(cudaStreamSynchronize(TF->cs), "glm sampler (window)", err) &&
                            TF->route_ok(err) && (!g_.v41 || tail->dsv41_engram_ok(err));
            cudaSetDevice(dev_);
            if (!ok) return -1;
            V->samp_c0 = sp.counter;
            V->samp_ok = true;
        }
        if (V->samp_ok && sp.counter == V->samp_c0 + (uint64_t) i) {
            // STRATA_GLM_SAMPLE_COLS_CHECK=1: each column drawn the old way as well (the same logits, the same counter)
            // and compared - a test, not for use (it is the old cost plus the new)
            static const bool chk = [] { const char* v = getenv("STRATA_GLM_SAMPLE_COLS_CHECK"); return v && v[0] == '1'; }();
            const int got = V->tok_h[32 + i];
            // the column's row where a sampled pick leaves it: the end-of-turn guard (end_guard_pick) reads it after
            // this pick (a copy on the stream, no wait)
            cudaSetDevice(tail->dev_);
            cudaMemcpyAsync(tail->sc_ + tail->sc_logits, V->logits + (size_t) i * g_.n_vocab,
                            (size_t) g_.n_vocab * sizeof(float), cudaMemcpyDeviceToDevice, tail->fast_->cs);
            cudaSetDevice(dev_);
            if (chk) {
                cudaSetDevice(tail->dev_);
                const int ref = tail->fast_sample(sp, err);
                cudaSetDevice(dev_);
                static long long n_chk = 0, n_bad = 0;
                ++n_chk;
                if (ref != got) ++n_bad;
                if (ref != got || (n_chk & 255) == 0)
                    std::fprintf(stderr, "glm sample cols check: %lld columns, %lld differ%s\n", n_chk, n_bad,
                                 ref != got ? " (this one)" : "");
            }
            return got;
        }
    }
    cudaSetDevice(tail->dev_);
    cudaMemcpyAsync(tail->sc_ + tail->sc_logits, V->logits + (size_t) i * g_.n_vocab,
                    (size_t) g_.n_vocab * sizeof(float), cudaMemcpyDeviceToDevice, tail->fast_->cs);
    const int tok = tail->fast_sample(sp, err);
    cudaSetDevice(dev_);
    return tok;
}


// ---------------------------------------------------------------- a later part's side of the window requests
// (qwen4exp's splits: part_port.hpp.)  The columns' rows from `in`, this part's layers (verify_part_q4x hands them on
// to the next part's port, the last part runs the head), then the last part's argmax per column.
bool Glm5Model::part_window(HopLink* in, const std::vector<int32_t>& toks, int64_t p0, int64_t pos, std::string& err) {
    auto* V = (VerifyBuf*) vb_;
    if (V == nullptr && !verify_alloc(err)) return false;
    V = (VerifyBuf*) vb_;
    const int T = (int) toks.size();
    if (!in->recv(HopLink::kWindow, V->R, (size_t) T * g_.hc * g_.n_embd * sizeof(float), fast_->cs, err)) return false;
    cudaSetDevice(dev_);
    pos_ = pos;
    V->T = T;
    return verify_part_q4x(toks, p0, err);
}

bool Glm5Model::part_window_wait(std::vector<int32_t>& argmax, std::string& err) {
    FastState* F = fast_;
    if (next_port_ != nullptr) return next_port_->window_wait(argmax, err) && F->route_ok(err);
    auto* V = (VerifyBuf*) vb_;
    cudaSetDevice(dev_);
    cudaError_t done;
    while ((done = cudaEventQuery(V->done)) == cudaErrorNotReady) {
        if (F->token_spin) glmfast::cpu_relax(); else std::this_thread::yield();
    }
    if (!glmfast::cuda_ok(done, "qwen4exp window completion", err) || !F->route_ok(err)) return false;
    argmax.assign(V->tok_h, V->tok_h + V->T);
    return true;
}

int Glm5Model::part_window_sample(int i, strata::kernels::SamplerParams& sp, std::string& err) {
    if (next_port_ != nullptr) return next_port_->window_sample(i, sp, err);
    auto* V = (VerifyBuf*) vb_;
    if (V == nullptr || i < 0 || i >= V->T) {
        err = "qwen4exp window: no such column";
        return -1;
    }
    if (sp.greedy) return V->tok_h[i];
    cudaSetDevice(dev_);
    cudaMemcpyAsync(sc_ + sc_logits, V->logits + (size_t) i * g_.n_vocab, (size_t) g_.n_vocab * sizeof(float),
                    cudaMemcpyDeviceToDevice, fast_->cs);
    return fast_sample(sp, err);
}

bool Glm5Model::part_window_logits(int i, std::vector<float>& out, std::string& err) {
    if (next_port_ != nullptr) return next_port_->window_logits(i, out, err);
    auto* V = (VerifyBuf*) vb_;
    if (V == nullptr || i < 0 || i >= V->T) {
        err = "qwen4exp window: no such column";
        return false;
    }
    cudaSetDevice(dev_);
    out.resize((size_t) g_.n_vocab);
    if (cudaMemcpy(out.data(), V->logits + (size_t) i * g_.n_vocab, out.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "qwen4exp window: logits copy";
        return false;
    }
    return true;
}

}  // namespace strata::core
