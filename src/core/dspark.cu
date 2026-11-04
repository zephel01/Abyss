// src/core/dspark.cu - DSpark, DeepSeek V4 Flash 0731's block drafter (bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF,
// arch deepseek4-dspark), loaded with STRATA_DS4_DSPARK=<the GGUF>.  Three DSV4 stages read the target's layer 40, 41
// and 42 outputs and propose the next 5 tokens of a verify window at once.  The arithmetic (worklog/2026-10-21_dspark.md,
// taken from antirez/ds4 and llama.cpp's dflash; the numpy reference is Abyss-tools bench/dspark/dspark_probe.py):
//   * the context: g[p] = rms(main_proj [h40; h41; h42]) * main_norm, h_L the mean of layer L's 4 output streams at
//     position p; each stage's K = V row rope_p(rms(attn_kv g[p]) * attn_kv_a_norm), kept in a ring of kRing
//     positions (FP16, as the target's window keys); a draft reads the last kCtx of them;
//   * the block: [y, noise x 4] (y the pending token, the noise token's embedding after it) at positions pos ..
//     pos + 4 through the stages - the mHC read / write halves, attention over the context and the whole block (no
//     mask), the MoE (sqrt-softplus router + bias, 6 of 256 experts, weights x 1.5, the shared expert; the SwiGLU's
//     clamp 10) - then the hc head, the norm and the TARGET's output head (row i proposes position pos + i + 1);
//   * the Markov chain: draft i = argmax(logits_i + markov_w2 . markov_w1[draft i - 1]) (draft -1 = y), its
//     confidence sigmoid(conf_proj . [h_i; markov_w1[draft i - 1]]) (h_i before the norm).  The decode stops the
//     window at the first draft below the threshold.
// The context follows the sequence by itself: the token path, the verify window (the kept columns, at spec_rewind)
// and the prompt path (a chunk's last kCtx rows) leave their layer 40-42 outputs, dspark_commit turns them into ring
// rows.  The routed experts (2.1 GB a stage, Q2_K) live in a VRAM cache, STRATA_DS4_DSPARK_EXPERTS a stage (default
// all 256): an expert without a slot is left out of its row (the kept weights renormalized - a draft only proposes,
// the window decides), and after a draft the most used missing ones are copied in (dspark_upload, while the window
// runs), the least used out.
//
// DeepSeek V4.1 Flash 1017's drafter (Lucebox/DeepSeek-V4.1-Flash-1017-DSpark, arch deepseek41-dflash-draft: the
// official mtp.0..2 block, Q4_K dense, Q2_K experts), the same STRATA_DS4_DSPARK on a V4.1 target.  The same scheme
// with the V4.1 layer's arithmetic (the numpy reference is Abyss-tools bench/v41dsp/v41_probe.py, llama.cpp's
// src/models/dflash.cpp graph_dsv4 its source):
//   * the features are layers 36, 37 and 38's outputs (the GGUF's dflash.capture_layer_ids), g = rms(dflash.fc
//     [h36; h37; h38]) * dflash.hidden_norm, rms eps 1e-20 (the drafter's own keys) everywhere;
//   * a stage is a V4.1 layer: no per-head q norm (d41::q_rope), the reads collapse with the PREVIOUS sublayer's pre
//     (stage 0's attention: the one-hot of stream 0), 128 experts top-3 (ff 2304, q_lora 1280 - the drafter's keys);
//   * no hc head: the head collapses with the last FFN's pre (d41::head_prep) and the drafter's output_norm;
//   * the Markov w2 rows are F16 (dspark_markov_f16), the confidence has a bias.
// Its experts (3.9 GB at 128 a stage) default to 48 / 40 / 24 slots for the stages (1.24 GB; the probe's uses: 90-97%,
// 97-100% and 100% of the routes - the later stages pick from fewer).
#include "glm_fast_state.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/dsv41_fast.hpp"
#include "strata/kernels/dsv4_fast.hpp"
#include "strata/kernels/glm_fast.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace ds = strata::kernels::dsv4;
namespace d41 = strata::kernels::dsv41;

namespace {
constexpr size_t kQ8 = 36;      // bytes of a q8_1 block (32 values)
constexpr int kB = 5;           // the block: y + 4 noise rows -> 5 drafts
constexpr int kRing = 256;      // the context ring's rows (positions mod kRing): kCtx + kB < kRing
constexpr int kCtx = 128;       // the context a draft reads
constexpr int kStages = 3;
constexpr int kL0 = 40;         // the target layers whose outputs are the features (40, 41, 42; V4.1: the GGUF's list)
constexpr int kQ2K = 10, kQ8_0 = 8, kF16 = 1, kF32 = 0;
constexpr float kEps = 1e-6f, kClamp = 10.0f, kWScale = 1.5f;

struct DspStage {
    WSlot q_a, q_b, kv, out_a, out_b, sh_gate, sh_up, sh_down;   // Q8_0, on the device
    float *q_a_norm = nullptr, *kv_a_norm = nullptr, *sinks = nullptr, *attn_norm = nullptr, *ffn_norm = nullptr;
    float *gate_inp = nullptr, *probs_b = nullptr;
    float *hc_attn_scale = nullptr, *hc_attn_base = nullptr, *hc_ffn_scale = nullptr, *hc_ffn_base = nullptr;
    uint16_t *hc_attn_fn = nullptr, *hc_ffn_fn = nullptr;       // BF16 [24][4 n_embd]
    uint16_t* ring = nullptr;                                     // [kRing][head_dim] FP16 K = V rows
    unsigned long long* tab = nullptr;                            // [n_expert] the expert's VRAM slot (0: none)
    uint8_t* slots = nullptr;                                     // [n_slots] blobs [gate | up | down]
    const uint8_t *src_g = nullptr, *src_u = nullptr, *src_d = nullptr;   // the GGUF's expert tensors (mapped)
    std::vector<int> slot_of, expert_in;                          // expert -> slot, slot -> expert (-1: none)
    std::vector<float> score;                                     // decayed use counts (the cache's policy)
    int n_slots = 0;
};

struct DspUp {
    int s, e, slot, evict;
};

struct DSparkState {
    std::unique_ptr<strata::GgufFile> gguf;
    DspStage st[kStages];
    WSlot main_proj, markov_w2;                                   // (V4.1: dflash.fc; markov_w2 unused)
    float *main_norm = nullptr, *conf_proj = nullptr, *hc_head_scale = nullptr, *hc_head_base = nullptr,
          *norm = nullptr;
    uint16_t *markov_w1 = nullptr, *hc_head_fn = nullptr;
    // the drafter's geometry (V4: the target's; V4.1: the GGUF's keys) and its numbers
    bool v41 = false;
    int cap[3] = {kL0, kL0 + 1, kL0 + 2};                         // the target layers whose outputs are the features
    int ne = 0, k = 0, ff = 0, nh = 0, ql = 0, og = 0, olr = 0, rot = 0, iters = 20, etype = kQ2K;
    float eps = kEps, hc_eps = 1e-6f, wscale = kWScale, clamp = kClamp;
    uint16_t* markov_w2h = nullptr;                               // V4.1: F16 rows [V][256]
    float* conf_b = nullptr;                                      // V4.1: the confidence's bias (1 float)
    float* pre_init = nullptr;                                    // V4.1: [kB][8] a row's pre pair, the FFN half one-hot
    FILE* dump = nullptr;                                         // STRATA_DS4_DSPARK_DUMP (the probe's comparison)
    int n_slots = 256, n_up = 4;                                  // (n_slots: the stages' largest)
    size_t blob = 0, gu_rb = 0, dn_rb = 0, down_off = 0, dense_bytes = 0;
    ds::Rope rope;
    // features, [3][n_embd] a row: the token's / the window's columns (feat) and a prompt chunk's last kCtx rows
    float *feat = nullptr, *feat_p = nullptr;
    void* featq = nullptr;
    float *gpre = nullptr, *gv = nullptr, *kvc = nullptr;         // the commit: main_proj's rows, g, the stages' kv
    void* gq = nullptr;
    // the draft ([kB] rows)
    float *emb_noise = nullptr, *emb_y = nullptr, *R = nullptr, *x = nullptr, *pre = nullptr, *post = nullptr,
          *comb = nullptr, *hpart = nullptr, *qr_raw = nullptr, *qr = nullptr, *kv_raw = nullptr, *q = nullptr,
          *o = nullptr, *oa = nullptr, *mixer = nullptr, *rlog = nullptr, *sh_g = nullptr, *sh_u = nullptr,
          *sh_out = nullptr, *ffn = nullptr, *head_h = nullptr, *head_x = nullptr, *logits = nullptr,
          *biased = nullptr, *conf = nullptr;
    void *xq = nullptr, *qr_q = nullptr, *oq = nullptr, *oaq = nullptr, *sh_hq = nullptr, *hq = nullptr,
         *head_xq = nullptr;
    unsigned int* hcnt = nullptr;
    size_t hpart_n = 0, hq_n = 0;
    unsigned long long* plan_ptr = nullptr;                       // [kB][8]
    float* plan_w = nullptr;
    int* zero = nullptr;                                          // the plans' cpu_flag (always 0)
    float* cpu_part = nullptr;
    int *ids_d = nullptr, *dtok = nullptr, *iota = nullptr;       // [3][kB][k] routes; [kB + 1] the chain; [2 kRing]
    std::vector<gf::MoeDev> md;
    gf::MoeColArgs* cols_d = nullptr;                             // ... the rows' plans for the experts in one launch
    // pinned host: the token in (host-mapped), the results, the cache's staging and table values
    int *y_h = nullptr, *y_d = nullptr, *res_tok = nullptr, *res_ids = nullptr;
    float *res_conf = nullptr, *emb_h = nullptr;
    uint8_t* stage_h = nullptr;
    unsigned long long* tabv_h = nullptr;
    cudaStream_t up = nullptr;
    cudaEvent_t ev_up = nullptr;
    std::vector<DspUp> ups;
    std::vector<void*> allocs;                                    // every device allocation (freed by dspark_destroy)
    int launch_err0 = 0;
    // the context: positions [ctx_lo, ctx_end) have ring rows; a verify window's columns wait for spec_rewind
    int64_t ctx_lo = 0, ctx_end = 0, win_p0 = -1;
    int win_T = 0;
    // statistics; STRATA_DS4_DSPARK_PROF=1: the device time of a draft's phases (events: start, each stage's attention
    // and MoE, the head, the Markov chain)
    uint64_t drafts = 0, uses = 0, misses = 0, uploads = 0, commits = 0;
    double ms_draft = 0.0, ms_up = 0.0;   // (ms_up: dspark_upload's host side - the staging copies)
    bool prof = false;
    static constexpr int kPh = 2 * kStages + 2;
    cudaEvent_t ev[kPh + 1] = {};
    double ms_ph[kPh] = {};
};

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);   // NaN stays NaN
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// the feature slot (0..2) of target layer il's output, -1 when it is none
int cap_slot(const DSparkState* D, int il) {
    for (int k = 0; k < 3; ++k)
        if (D->cap[k] == il) return k;
    return -1;
}

// the stages' slot counts for the report: "n" when they agree, else "a/b/c"
std::string slots_str(const DSparkState* D) {
    std::string r = std::to_string(D->st[0].n_slots);
    bool same = true;
    for (int s = 1; s < kStages; ++s) same = same && D->st[s].n_slots == D->st[0].n_slots;
    for (int s = 1; !same && s < kStages; ++s) r += "/" + std::to_string(D->st[s].n_slots);
    return r;
}
}  // namespace

bool Glm5Model::dspark_load(std::string& err) {
    const char* path = getenv("STRATA_DS4_DSPARK");
    if (path == nullptr || !path[0]) return true;
    const Glm5Geometry& g = g_;
    const auto refuse = [&](const std::string& why) {
        std::fprintf(stderr, "dspark: not loaded - %s\n", why.c_str());
        dspark_destroy();
        return true;   // the target runs on without the drafter
    };
    if (!g.dsv4() || n_parts_ != 1 || l0_ != 0 || l1_ != g.n_layers || mtp_il_ >= 0)
        return refuse("DeepSeek V4 / V4.1 Flash on one device only (no split, no MTP)");
    if (g.v41 ? (g.n_embd > 5120 || g.n_embd % 256 != 0 || g.head_dim != 512)
              : (g.n_layers <= kL0 + 2 || g.n_embd > 4096 || g.head_dim != 512 || g.n_expert > 512 || g.n_exp_used > 8))
        return refuse("the target's geometry (n_layers " + std::to_string(g.n_layers) + ", n_embd " +
                      std::to_string(g.n_embd) + ")");
    auto* D = new DSparkState();
    dsp_ = D;
    try {
        D->gguf = std::make_unique<strata::GgufFile>(path);
    } catch (const std::exception& e) {
        return refuse(std::string("cannot open ") + path + ": " + e.what());
    }
    const strata::GgufFile& G = *D->gguf;
    const auto num = [&](const std::string& key, double def) {
        const strata::MetaValue* v = G.get(key);
        return v != nullptr && v->is_num() ? v->num() : def;
    };
    D->v41 = g.v41;
    const strata::MetaValue* arch = G.get("general.architecture");
    const std::string A = "deepseek41-dflash-draft.";   // (V4.1's keys)
    int noise = -1;
    if (!g.v41) {
        if (arch == nullptr || arch->s != "deepseek4-dspark") return refuse("not a deepseek4-dspark GGUF");
        const strata::MetaValue* tl = G.get("deepseek4.dspark.target_layers");
        bool layers_ok = tl != nullptr && tl->items.size() == 3;
        for (int i = 0; layers_ok && i < 3; ++i) layers_ok = (int) tl->items[(size_t) i].num() == kL0 + i;
        if (!layers_ok || (int) num("deepseek4.dspark.layer_count", 0) != kStages ||
            (int) num("deepseek4.dspark.block_size", 0) != kB || (int) num("deepseek4.dspark.markov_rank", 0) != 256 ||
            (int) num("deepseek4.dspark.expert_count", 0) != g.n_expert)
            return refuse("the drafter's layout (3 stages over layers 40-42, block 5, Markov rank 256, the target's "
                          "expert count)");
        noise = (int) num("deepseek4.dspark.noise_token_id", -1);
        D->ne = g.n_expert;
        D->k = g.n_exp_used;
        D->ff = g.n_ff_exp;
        D->nh = g.n_head;
        D->ql = g.q_lora;
        D->og = g.o_groups;
        D->olr = g.o_lora;
        D->rot = g.n_rot;
        D->hc_eps = g.hc_eps;
        D->iters = g.sinkhorn_iters;
    } else {
        if (arch == nullptr || arch->s != "deepseek41-dflash-draft")
            return refuse("not a deepseek41-dflash-draft GGUF (DeepSeek V4.1 Flash's drafter)");
        const auto key = [&](const char* k, double def) { return num(A + k, def); };
        const strata::MetaValue* cl = G.get(A + "dflash.capture_layer_ids");
        bool layers_ok = cl != nullptr && cl->items.size() == 3;
        for (int i = 0; layers_ok && i < 3; ++i) {
            D->cap[i] = (int) cl->items[(size_t) i].num();
            // (a feature is the residual the next layer reads: captured inside the layer loop)
            layers_ok = D->cap[i] >= 0 && D->cap[i] + 1 < g.n_layers;
        }
        const strata::MetaValue* gate = G.get(A + "expert_gating_func");
        D->ne = (int) key("expert_count", 0);
        D->k = (int) key("expert_used_count", 0);
        D->ff = (int) key("expert_feed_forward_length", 0);
        D->nh = (int) key("attention.head_count", 0);
        D->ql = (int) key("attention.q_lora_rank", 0);
        D->og = (int) key("attention.output_group_count", 0);
        D->olr = (int) key("attention.output_lora_rank", 0);
        D->rot = (int) key("rope.dimension_count", 0);
        D->eps = (float) key("attention.layer_norm_rms_epsilon", g.norm_eps);
        D->hc_eps = (float) key("hyper_connection.epsilon", g.hc_eps);
        D->iters = (int) key("hyper_connection.sinkhorn_iterations", g.sinkhorn_iters);
        D->wscale = (float) key("expert_weights_scale", kWScale);
        D->clamp = (float) key("swiglu_clamp_exp", kClamp);
        noise = (int) key("dflash.mask_token_id", -1);
        if (!layers_ok || (int) key("block_count", 0) != kStages || (int) key("embedding_length", 0) != g.n_embd ||
            (int) key("vocab_size", 0) != g.n_vocab || (int) key("attention.key_length", 0) != g.head_dim ||
            (int) key("attention.value_length", 0) != g.head_dim || (int) key("attention.head_count_kv", 0) != 1 ||
            (int) key("attention.sliding_window", 0) != kCtx || (int) key("dflash.block_size", 0) != kB ||
            (int) key("dflash.head_hc_enabled", 1) != 0 || (int) key("dflash.dspark.enabled", 0) != 1 ||
            (int) key("dflash.dspark.markov_rank", 0) != 256 ||
            (int) key("dflash.dspark.confidence_dim", 0) != g.n_embd + 256 ||
            (int) key("hyper_connection.count", 0) != g.hc || (int) key("expert_shared_count", 0) != 1 ||
            (int) key("hash_layer_count", 0) != 0 || gate == nullptr || gate->s != "sqrtsoftplus")
            return refuse("the drafter's layout (3 stages, block 5, a 128-row window, Markov rank 256, no hc head, "
                          "the target's width / vocabulary / head size, one shared expert, sqrt-softplus routing)");
        if (D->ne < 8 || D->ne > 512 || D->k < 1 || D->k > 8 || D->ff % 256 != 0 || D->ql % 256 != 0 || D->nh < 1 ||
            D->og < 1 || D->og > gf::kMaxMvJobs || D->nh % D->og != 0 || D->olr % 256 != 0 || D->rot < 2 ||
            D->rot % 2 != 0 || D->rot > g.head_dim)
            return refuse("the drafter's geometry (experts " + std::to_string(D->ne) + " top-" + std::to_string(D->k) +
                          ", ff " + std::to_string(D->ff) + ", q_lora " + std::to_string(D->ql) + ")");
    }
    if (noise < 0 || noise >= g.n_vocab) return refuse("the noise token id");
    const int E = g.n_embd, NE = D->ne, K = D->k, FF = D->ff, V = g.n_vocab;
    const int HD = D->nh * g.head_dim, OL = D->og * D->olr, QL = D->ql, DH = g.head_dim;
    std::string why;
    // type < 0: any quantized type the GEMVs take (V4.1's Q4_K dense rows, its Q2_K experts)
    const auto find = [&](const std::string& n, int type, std::initializer_list<uint64_t> shape) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = G.find(n);
        if (t == nullptr) { if (why.empty()) why = n + " missing"; return nullptr; }
        bool ok = (type < 0 || (int) t->type == type) && t->shape.size() == shape.size();
        size_t i = 0;
        for (uint64_t d : shape) ok = ok && t->shape[i++] == d;
        if (!ok && why.empty()) why = n + " has another type or shape";
        return ok ? t : nullptr;
    };
    bool ok = true;
    const auto dev = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (!ok || cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; cudaGetLastError(); return nullptr; }
        D->allocs.push_back(p);
        D->dense_bytes += bytes;
        return p;
    };
    const auto raw = [&](const std::string& n, int type, int64_t n_in, int64_t n_out, WSlot& w) {
        const strata::TensorInfo* t = find(n, type, {(uint64_t) n_in, (uint64_t) n_out});
        if (t == nullptr) { ok = false; return; }
        if (type < 0 && (t->type == (uint32_t) kF32 || t->type == (uint32_t) kF16 || !gf::mvn_supported((int) t->type) ||
                         gf::row_bytes((int) t->type, n_in) == 0)) {
            if (why.empty()) why = n + ": the GEMVs do not take its type " + std::to_string(t->type);
            ok = false;
            return;
        }
        type = (int) t->type;
        const size_t bytes = gf::row_bytes(type, n_in) * (size_t) n_out;
        void* p = dev(bytes);
        if (p == nullptr) return;
        ok = cudaMemcpy(p, G.tensor_data(*t), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        w = WSlot{nullptr, type, p, n_in, n_out};
    };
    const auto f32 = [&](const std::string& n, std::initializer_list<uint64_t> shape) -> float* {
        const strata::TensorInfo* t = find(n, kF32, shape);
        if (t == nullptr) { ok = false; return nullptr; }
        float* p = (float*) dev(t->elements() * sizeof(float));
        if (p != nullptr)
            ok = cudaMemcpy(p, G.tensor_data(*t), t->elements() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
        return p;
    };
    // a F32 (V4) or F16 (V4.1) tensor as host floats
    const auto host = [&](const std::string& n, int type, std::initializer_list<uint64_t> shape, std::vector<float>& h) {
        const strata::TensorInfo* t = find(n, type, shape);
        if (t == nullptr) { ok = false; return false; }
        h.resize((size_t) t->elements());
        if (type == kF16) {
            const ggml_fp16_t* src = (const ggml_fp16_t*) G.tensor_data(*t);
            for (size_t i = 0; i < h.size(); ++i) h[i] = ggml_fp16_to_fp32(src[i]);
        } else {
            std::memcpy(h.data(), G.tensor_data(*t), h.size() * sizeof(float));
        }
        return true;
    };
    const auto bf16 = [&](const std::string& n, std::initializer_list<uint64_t> shape, int type = kF32) -> uint16_t* {
        std::vector<float> src;
        if (!host(n, type, shape, src)) return nullptr;
        std::vector<uint16_t> h(src.size());
        for (size_t i = 0; i < h.size(); ++i) h[i] = to_bf16(src[i]);
        uint16_t* p = (uint16_t*) dev(h.size() * 2);
        if (p != nullptr) ok = cudaMemcpy(p, h.data(), h.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess;
        return p;
    };
    const auto f16 = [&](const std::string& n, std::initializer_list<uint64_t> shape) -> uint16_t* {   // as stored
        const strata::TensorInfo* t = find(n, kF16, shape);
        if (t == nullptr) { ok = false; return nullptr; }
        uint16_t* p = (uint16_t*) dev(t->elements() * 2);
        if (p != nullptr) ok = cudaMemcpy(p, G.tensor_data(*t), t->elements() * 2, cudaMemcpyHostToDevice) == cudaSuccess;
        return p;
    };
    const uint64_t uE = (uint64_t) E, u4E = 4 * uE;
    const int dq = D->v41 ? -1 : kQ8_0;   // the dense rows' type (V4.1: Q4_K, whatever the GEMVs take)
    const int hct = D->v41 ? kF16 : kF32;   // the hc fns' (BF16 on the device)
    const strata::TensorInfo* te[kStages][3] = {};
    for (int s = 0; s < kStages && ok; ++s) {
        DspStage& S = D->st[s];
        const std::string P = (D->v41 ? "blk." : "dspark.") + std::to_string(s) + ".";
        raw(P + "attn_q_a.weight", dq, E, QL, S.q_a);
        raw(P + "attn_q_b.weight", dq, QL, HD, S.q_b);
        raw(P + "attn_kv.weight", dq, E, DH, S.kv);
        raw(P + "attn_output_a.weight", dq, (int64_t) HD / D->og, OL, S.out_a);
        raw(P + "attn_output_b.weight", dq, OL, E, S.out_b);
        raw(P + "ffn_gate_shexp.weight", dq, E, FF, S.sh_gate);
        raw(P + "ffn_up_shexp.weight", dq, E, FF, S.sh_up);
        raw(P + "ffn_down_shexp.weight", dq, FF, E, S.sh_down);
        S.q_a_norm = f32(P + "attn_q_a_norm.weight", {(uint64_t) QL});
        S.kv_a_norm = f32(P + "attn_kv_a_norm.weight", {(uint64_t) DH});
        S.sinks = f32(P + "attn_sinks.weight", {(uint64_t) D->nh});
        S.attn_norm = f32(P + "attn_norm.weight", {uE});
        S.ffn_norm = f32(P + "ffn_norm.weight", {uE});
        S.gate_inp = f32(P + "ffn_gate_inp.weight", {uE, (uint64_t) NE});
        S.probs_b = f32(P + "exp_probs_b.bias", {(uint64_t) NE});
        S.hc_attn_fn = bf16(P + "hc_attn_fn.weight", {u4E, 24}, hct);
        S.hc_attn_scale = f32(P + "hc_attn_scale.weight", {3});
        S.hc_attn_base = f32(P + "hc_attn_base.weight", {24});
        S.hc_ffn_fn = bf16(P + "hc_ffn_fn.weight", {u4E, 24}, hct);
        S.hc_ffn_scale = f32(P + "hc_ffn_scale.weight", {3});
        S.hc_ffn_base = f32(P + "hc_ffn_base.weight", {24});
        const int xt = D->v41 ? -1 : kQ2K;
        te[s][0] = find(P + "ffn_gate_exps.weight", xt, {uE, (uint64_t) FF, (uint64_t) NE});
        te[s][1] = find(P + "ffn_up_exps.weight", xt, {uE, (uint64_t) FF, (uint64_t) NE});
        te[s][2] = find(P + "ffn_down_exps.weight", xt, {(uint64_t) FF, uE, (uint64_t) NE});
        if (te[s][0] == nullptr || te[s][1] == nullptr || te[s][2] == nullptr) { ok = false; break; }
        if (s == 0) D->etype = (int) te[0][0]->type;
        for (int i = 0; i < 3 && ok; ++i)
            if ((int) te[s][i]->type != D->etype) {
                why = P + "'s experts are not of one type";
                ok = false;
            }
        S.src_g = G.tensor_data(*te[s][0]);
        S.src_u = G.tensor_data(*te[s][1]);
        S.src_d = G.tensor_data(*te[s][2]);
        S.ring = (uint16_t*) dev((size_t) kRing * DH * sizeof(uint16_t));
        S.tab = (unsigned long long*) dev((size_t) NE * sizeof(unsigned long long));
    }
    if (!D->v41) {
        raw("dspark.main_proj.weight", kQ8_0, 3 * E, E, D->main_proj);
        raw("dspark.markov_w2.weight", kQ8_0, 256, V, D->markov_w2);
        D->main_norm = f32("dspark.main_norm.weight", {uE});
        D->conf_proj = f32("dspark.conf_proj.weight", {uE + 256, 1});
        D->hc_head_fn = bf16("dspark.hc_head_fn.weight", {u4E, 4});
        D->hc_head_scale = f32("dspark.hc_head_scale.weight", {1});
        D->hc_head_base = f32("dspark.hc_head_base.weight", {4});
        D->norm = f32("dspark.norm.weight", {uE});
        if (ok) {
            const strata::TensorInfo* t = find("dspark.markov_w1.weight", kF16, {256, (uint64_t) V});
            if (t == nullptr) ok = false;
            D->markov_w1 = (uint16_t*) dev((size_t) V * 256 * 2);
            if (ok) ok = cudaMemcpy(D->markov_w1, G.tensor_data(*t), (size_t) V * 256 * 2, cudaMemcpyHostToDevice) == cudaSuccess;
        }
    } else {
        raw("dflash.fc.weight", -1, 3 * E, E, D->main_proj);
        D->main_norm = f32("dflash.hidden_norm.weight", {uE});
        D->norm = f32("output_norm.weight", {uE});
        if (ok) D->markov_w1 = f16("dflash.dspark.markov.w1", {256, (uint64_t) V});
        if (ok) D->markov_w2h = f16("dflash.dspark.markov.w2", {256, (uint64_t) V});
        std::vector<float> cp;
        if (ok && host("dflash.dspark.confidence.weight", kF16, {uE + 256}, cp)) {
            D->conf_proj = (float*) dev(cp.size() * sizeof(float));
            if (ok) ok = cudaMemcpy(D->conf_proj, cp.data(), cp.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
        }
        D->conf_b = f32("dflash.dspark.confidence.bias", {1});
    }
    if (!ok) return refuse(why.empty() ? "the weights did not upload (VRAM?)" : why);
    if (!gf::moe_supported(D->etype) || (!D->v41 && !gf::mvn_supported(kQ8_0)))
        return refuse("the experts' type " + std::to_string(D->etype) + " / the Q8_0 kernels");
    // the noise token's embedding, once
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) return refuse("the target's embedding");
    {
        std::vector<float> e((size_t) E);
        tt->to_float(pack_emb_src_ + (size_t) noise * ggml_row_size((ggml_type) pack_emb_type_, E), e.data(), E);
        D->emb_noise = (float*) dev((size_t) E * sizeof(float));
        if (ok) ok = cudaMemcpy(D->emb_noise, e.data(), (size_t) E * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    // the buffers
    const auto fb = [&](float*& p, size_t n) { p = (float*) dev(n * sizeof(float)); };
    const auto bb = [&](void*& p, size_t n) { p = dev(n); };
    const size_t HCE = 4 * (size_t) E;
    fb(D->feat, (size_t) kVerifyMax * 3 * E);
    fb(D->feat_p, (size_t) kCtx * 3 * E);
    bb(D->featq, (size_t) kCtx * 3 * E / 32 * kQ8);
    fb(D->gpre, (size_t) kCtx * E);
    fb(D->gv, (size_t) kCtx * E);
    bb(D->gq, (size_t) kCtx * E / 32 * kQ8);
    fb(D->kvc, (size_t) kStages * kCtx * DH);
    fb(D->emb_y, E);
    fb(D->R, (size_t) kB * 2 * HCE);
    fb(D->x, (size_t) kB * E);
    bb(D->xq, (size_t) kB * E / 32 * kQ8);
    fb(D->pre, kB * 8);
    fb(D->post, kB * 8);
    fb(D->comb, kB * 16);
    D->hpart_n = (size_t) (4 * E / 512) * 36 + 64;
    fb(D->hpart, D->hpart_n * kB);
    D->hcnt = (unsigned int*) dev((size_t) kB * 64);
    fb(D->qr_raw, (size_t) kB * QL);
    fb(D->qr, (size_t) kB * QL);
    bb(D->qr_q, (size_t) kB * QL / 32 * kQ8);
    fb(D->kv_raw, (size_t) kB * DH);
    fb(D->q, (size_t) kB * HD);
    fb(D->o, (size_t) kB * HD);
    bb(D->oq, (size_t) kB * HD / 32 * kQ8);
    fb(D->oa, (size_t) kB * OL);
    bb(D->oaq, (size_t) kB * OL / 32 * kQ8);
    fb(D->mixer, (size_t) kB * E);
    fb(D->rlog, (size_t) kB * NE);
    fb(D->sh_g, (size_t) kB * FF);
    fb(D->sh_u, (size_t) kB * FF);
    bb(D->sh_hq, (size_t) kB * FF / 32 * kQ8);
    D->hq_n = (size_t) 8 * FF / 32 * kQ8;
    bb(D->hq, D->hq_n * kB);
    fb(D->sh_out, (size_t) kB * E);
    fb(D->ffn, (size_t) kB * E);
    fb(D->head_h, (size_t) kB * E);
    fb(D->head_x, (size_t) kB * E);
    bb(D->head_xq, (size_t) kB * E / 32 * kQ8);
    fb(D->logits, (size_t) kB * V);
    fb(D->biased, (size_t) V);
    fb(D->conf, 8);
    D->plan_ptr = (unsigned long long*) dev((size_t) kB * 8 * sizeof(unsigned long long));
    fb(D->plan_w, kB * 8);
    D->zero = (int*) dev(64);
    fb(D->cpu_part, E);
    D->ids_d = (int*) dev((size_t) kStages * kB * 8 * sizeof(int));
    D->dtok = (int*) dev(16 * sizeof(int));
    D->iota = (int*) dev(2 * kRing * sizeof(int));
    if (D->v41) fb(D->pre_init, kB * 8);
    if (!ok) return refuse("the buffers did not allocate (VRAM?)");
    if (D->v41) {   // a row's pre pair: [0, 4) the attention side's, [4, 8) the FFN side's - stage 0 reads the one-hot
        std::vector<float> pi((size_t) kB * 8, 0.0f);
        for (int i = 0; i < kB; ++i) pi[(size_t) i * 8 + 4] = 1.0f;
        cudaMemcpy(D->pre_init, pi.data(), pi.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemset(D->hcnt, 0, (size_t) kB * 64);
    cudaMemset(D->zero, 0, 64);
    cudaMemset(D->plan_ptr, 0, (size_t) kB * 8 * sizeof(unsigned long long));
    {
        std::vector<int> io(2 * kRing);
        for (int i = 0; i < 2 * kRing; ++i) io[(size_t) i] = i % kRing;
        cudaMemcpy(D->iota, io.data(), io.size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    D->md.assign(kB, gf::MoeDev{});
    {
        gf::MoeColArgs ca[kB];
        for (int i = 0; i < kB; ++i) {
            gf::MoeDev& m = D->md[(size_t) i];
            m.plan_ptr = D->plan_ptr + i * 8;
            m.plan_w = D->plan_w + i * 8;
            m.cpu_flag = D->zero;
            m.cpu_part = D->cpu_part;
            ca[i] = gf::MoeColArgs{m.plan_ptr, m.plan_w, m.cpu_flag, m.cpu_part};
        }
        D->cols_d = (gf::MoeColArgs*) dev(sizeof ca);
        if (ok) ok = cudaMemcpy(D->cols_d, ca, sizeof ca, cudaMemcpyHostToDevice) == cudaSuccess;
        if (!ok) return refuse("the buffers did not allocate (VRAM?)");
    }
    // the routed experts: a VRAM slot cache a stage; STRATA_DS4_DSPARK_EXPERTS=<n> or <n0>,<n1>,<n2> (a stage each;
    // default all - V4.1's 48,40,24)
    D->gu_rb = gf::row_bytes(D->etype, E);
    D->dn_rb = gf::row_bytes(D->etype, FF);
    D->down_off = 2 * (size_t) FF * D->gu_rb;
    D->blob = (D->down_off + (size_t) E * D->dn_rb + 255) & ~(size_t) 255;
    {
        int want[kStages] = {NE, NE, NE};
        if (D->v41) {
            want[0] = 48;
            want[1] = 40;
            want[2] = 24;
        }
        if (const char* v = getenv("STRATA_DS4_DSPARK_EXPERTS")) {
            int n = 0;
            for (const char* q = v; *q && n < kStages;) {
                char* e = nullptr;
                const long x = std::strtol(q, &e, 10);
                if (e == q) break;
                want[n++] = (int) x;
                q = *e == ',' ? e + 1 : e;
            }
            for (int s = std::max(1, n); n > 0 && s < kStages; ++s) want[s] = want[n - 1];   // (one number: every stage)
        }
        D->n_slots = 0;
        for (int s = 0; s < kStages; ++s) {
            D->st[s].n_slots = std::max(8, std::min(NE, want[s]));
            D->n_slots = std::max(D->n_slots, D->st[s].n_slots);
        }
    }
    if (const char* v = getenv("STRATA_DS4_DSPARK_UPLOADS")) D->n_up = std::max(0, std::min(8, std::atoi(v)));
    const auto blob_of = [&](const DspStage& S, int e, uint8_t* dst, cudaStream_t cs, bool async) {
        const size_t gu = (size_t) FF * D->gu_rb, dn = (size_t) E * D->dn_rb;
        const uint8_t* src[3] = {S.src_g + (size_t) e * gu, S.src_u + (size_t) e * gu, S.src_d + (size_t) e * dn};
        const size_t off[3] = {0, gu, D->down_off}, len[3] = {gu, gu, dn};
        for (int i = 0; i < 3; ++i) {
            if (async) cudaMemcpyAsync(dst + off[i], src[i], len[i], cudaMemcpyHostToDevice, cs);
            else if (cudaMemcpy(dst + off[i], src[i], len[i], cudaMemcpyHostToDevice) != cudaSuccess) return false;
        }
        return true;
    };
    std::vector<unsigned long long> tab((size_t) NE);
    for (int s = 0; s < kStages && ok; ++s) {
        DspStage& S = D->st[s];
        S.slots = (uint8_t*) dev((size_t) S.n_slots * D->blob);
        if (!ok) break;
        S.slot_of.assign((size_t) NE, -1);
        S.expert_in.assign((size_t) S.n_slots, -1);
        S.score.assign((size_t) NE, 0.0f);
        std::fill(tab.begin(), tab.end(), 0ull);
        for (int e = 0; e < S.n_slots && ok; ++e) {   // (the first n_slots experts until the uses say otherwise)
            uint8_t* dst = S.slots + (size_t) e * D->blob;
            ok = blob_of(S, e, dst, nullptr, false);
            S.slot_of[(size_t) e] = e;
            S.expert_in[(size_t) e] = e;
            tab[(size_t) e] = (unsigned long long) dst;
        }
        if (ok) ok = cudaMemcpy(S.tab, tab.data(), tab.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (!ok) return refuse("the expert cache did not allocate (" + slots_str(D) + " a stage)");
    // pinned host buffers, the upload stream
    const size_t n_stage = (size_t) std::max(1, D->n_up) * kStages;
    ok = cudaHostAlloc((void**) &D->y_h, 64, cudaHostAllocMapped) == cudaSuccess &&
         cudaHostGetDevicePointer((void**) &D->y_d, D->y_h, 0) == cudaSuccess &&
         cudaHostAlloc((void**) &D->res_tok, 64, cudaHostAllocDefault) == cudaSuccess &&
         cudaHostAlloc((void**) &D->res_conf, 64, cudaHostAllocDefault) == cudaSuccess &&
         cudaHostAlloc((void**) &D->res_ids, (size_t) kStages * kB * 8 * sizeof(int), cudaHostAllocDefault) == cudaSuccess &&
         cudaHostAlloc((void**) &D->emb_h, (size_t) E * sizeof(float), cudaHostAllocDefault) == cudaSuccess &&
         cudaHostAlloc((void**) &D->tabv_h, 2 * n_stage * sizeof(unsigned long long), cudaHostAllocDefault) == cudaSuccess &&
         ((D->st[0].n_slots >= NE && D->st[1].n_slots >= NE && D->st[2].n_slots >= NE) ||
          cudaHostAlloc((void**) &D->stage_h, n_stage * D->blob, cudaHostAllocDefault) == cudaSuccess) &&
         cudaStreamCreateWithFlags(&D->up, cudaStreamNonBlocking) == cudaSuccess &&
         cudaEventCreateWithFlags(&D->ev_up, cudaEventDisableTiming) == cudaSuccess;
    if (!ok) return refuse("the pinned buffers");
    cudaEventRecord(D->ev_up, D->up);
    D->launch_err0 = gf::launch_errors();
    D->prof = getenv("STRATA_DS4_DSPARK_PROF") != nullptr && std::atoi(getenv("STRATA_DS4_DSPARK_PROF")) != 0;
    for (int i = 0; D->prof && i <= DSparkState::kPh; ++i) cudaEventCreate(&D->ev[i]);
    // (the drafter's RoPE: base 10000, no YaRN)
    D->rope = ds::make_rope(D->rot, D->v41 ? (float) num(A + "rope.freq_base", 10000.0) : 10000.0f, 1.0f, 0.0f, 0, 0.0f,
                            0.0f);
    if (const char* v = getenv("STRATA_DS4_DSPARK_DUMP"); v != nullptr && v[0]) D->dump = std::fopen(v, "w");
    size_t exp_bytes = 0;
    for (int s = 0; s < kStages; ++s) exp_bytes += (size_t) D->st[s].n_slots * D->blob;
    std::fprintf(stderr, "dspark: %s - %d stages over target layers %d-%d, %d drafts a step; %.2f GB on CUDA%d "
                         "(dense %.2f, experts %s of %d a stage %.2f)\n",
                 path, kStages, D->cap[0], D->cap[2], kB, (double) (D->dense_bytes) / 1073741824.0, dev_,
                 (double) (D->dense_bytes - exp_bytes) / 1073741824.0, slots_str(D).c_str(), NE,
                 (double) exp_bytes / 1073741824.0);
    return true;
}

void Glm5Model::dspark_destroy() {
    auto* D = (DSparkState*) dsp_;
    if (D == nullptr) return;
    dsp_ = nullptr;
    if (D->up) cudaStreamSynchronize(D->up);
    for (cudaEvent_t e : D->ev)
        if (e) cudaEventDestroy(e);
    for (void* p : D->allocs) cudaFree(p);
    if (D->dump) std::fclose(D->dump);
    if (D->up) cudaStreamDestroy(D->up);
    if (D->ev_up) cudaEventDestroy(D->ev_up);
    for (void* p : {(void*) D->y_h, (void*) D->res_tok, (void*) D->res_conf, (void*) D->res_ids, (void*) D->emb_h,
                    (void*) D->tabv_h, (void*) D->stage_h})
        if (p) cudaFreeHost(p);
    delete D;
}

size_t Glm5Model::dspark_vram_bytes() const {
    const auto* D = (const DSparkState*) dsp_;
    return D != nullptr ? D->dense_bytes : 0;
}

void Glm5Model::dspark_report() const {
    const auto* D = (const DSparkState*) dsp_;
    if (D == nullptr || D->drafts == 0) return;
    std::fprintf(stderr, "dspark: %llu drafts, %.2f ms each (host to results); routed experts %llu, %.1f%% without a "
                         "slot (%s a stage), %llu uploads (%.1f ms on the host); %llu context commits\n",
                 (unsigned long long) D->drafts, D->ms_draft / (double) D->drafts, (unsigned long long) D->uses,
                 100.0 * (double) D->misses / (double) std::max<uint64_t>(1, D->uses), slots_str(D).c_str(),
                 (unsigned long long) D->uploads, D->ms_up, (unsigned long long) D->commits);
    if (D->prof) {
        static const char* names[DSparkState::kPh] = {"embed", "s0 attn", "s0 moe", "s1 attn", "s1 moe", "s2 attn",
                                                     "s2 moe", "head+markov"};
        std::fprintf(stderr, "dspark: device ms a draft -");
        double t = 0.0;
        for (int i = 0; i < DSparkState::kPh; ++i) {
            std::fprintf(stderr, " %s %.3f", names[i], D->ms_ph[i] / (double) D->drafts);
            t += D->ms_ph[i];
        }
        std::fprintf(stderr, " | total %.3f\n", t / (double) D->drafts);
    }
}

void Glm5Model::dspark_reset() {
    auto* D = (DSparkState*) dsp_;
    if (D == nullptr) return;
    D->ctx_lo = D->ctx_end = pos_;
    D->win_p0 = -1;
}

void Glm5Model::dspark_capture(int il_out, const float* R, int col, cudaStream_t s) {
    auto* D = (DSparkState*) dsp_;
    if (D == nullptr) return;
    const int k = cap_slot(D, il_out);
    if (k < 0 || col < 0 || col >= kVerifyMax) return;
    const int E = g_.n_embd;
    ds::stream_mean(R, 1, 0, E, D->feat + ((size_t) col * 3 + k) * E, 0, s);
}

void Glm5Model::dspark_capture_rows(int il_out, const float* R, int T, cudaStream_t s) {
    auto* D = (DSparkState*) dsp_;
    if (D == nullptr) return;
    const int k = cap_slot(D, il_out);
    if (k < 0 || T <= 0) return;
    const int E = g_.n_embd, n = std::min(T, kCtx);
    ds::stream_mean(R + (size_t) (T - n) * 4 * E, n, (size_t) 4 * E, E, D->feat_p + (size_t) k * E, (size_t) 3 * E, s);
}

bool Glm5Model::dspark_commit(int64_t p0, int n, bool prompt, cudaStream_t s) {
    auto* D = (DSparkState*) dsp_;
    if (D == nullptr || n <= 0) return true;
    const Glm5Geometry& g = g_;
    const int E = g.n_embd, DH = g.head_dim;
    if (n > kCtx) {   // (only the last kCtx rows can be read)
        p0 += n - kCtx;
        n = kCtx;
    }
    const float* feat = prompt ? D->feat_p : D->feat;   // (the prompt rows: the chunk's last n, from row 0)
    gf::quantize_q8_1(feat, D->featq, n * 3 * E, s);
    gf::MvJob j = {D->main_proj.q, D->featq, feat, D->gpre, nullptr, 1.0f, D->main_proj.type, 3 * E, E};
    if (!gf::mvn(&j, 1, n, s)) return false;
    if (D->v41)   // (5120 wide: past rms_q8's 4096)
        d41::rms_rows(D->gpre, n, D->main_norm, D->eps, E, D->gv, D->gq, s);
    else
        for (int c = 0; c < n; ++c)
            gf::rms_q8(D->gpre + (size_t) c * E, nullptr, D->main_norm, kEps, E, D->gv + (size_t) c * E,
                       (uint8_t*) D->gq + (size_t) c * (E / 32) * kQ8, s);
    gf::MvJob jk[kStages];
    for (int st = 0; st < kStages; ++st)
        jk[st] = {D->st[st].kv.q, D->gq, D->gv, D->kvc + (size_t) st * kCtx * DH, nullptr, 1.0f, D->st[st].kv.type, E, DH};
    if (!gf::mvn(jk, kStages, n, s)) return false;
    for (int st = 0; st < kStages; ++st)
        ds::kv_rows(D->kvc + (size_t) st * kCtx * DH, n, D->st[st].kv_a_norm, DH, D->rot, D->rope, (int) p0,
                    D->st[st].ring, kRing, D->eps, s);
    if (p0 > D->ctx_end || p0 < D->ctx_lo) D->ctx_lo = p0;   // (a gap: the context starts over)
    D->ctx_end = p0 + n;
    ++D->commits;
    return true;
}

void Glm5Model::dspark_settle() {
    auto* D = (DSparkState*) dsp_;
    if (D == nullptr || D->win_p0 < 0) return;
    const int64_t p0 = D->win_p0;
    const int n = (int) std::max<int64_t>(0, std::min<int64_t>(D->win_T, pos_ - p0));
    D->win_p0 = -1;
    dspark_commit(p0, n, false, fast_->cs);
}

void Glm5Model::dspark_window(int64_t p0, int T) {
    auto* D = (DSparkState*) dsp_;
    if (D == nullptr) return;
    D->win_p0 = p0;
    D->win_T = T;
}

void Glm5Model::spec_rewind(int64_t pos) {
    if (g_.qwen35() && next_port_ != nullptr) {   // (qwen4exp's split: the later parts through the port)
        verify_rewind_q35(pos);
        next_port_->rewind(pos);
        cudaSetDevice(dev_);
    } else if (g_.qwen35()) {   // (the delta-net and conv states: not rings - replayed; each part of a split its own layers)
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
            if (m != this) cudaSetDevice(m->dev_);
            m->verify_rewind_q35(pos);
            m->pos_ = pos;
        }
        cudaSetDevice(dev_);
    }
    pos_ = pos;
    dspark_settle();
}

void Glm5Model::part_rewind(int64_t pos) {
    cudaSetDevice(dev_);
    verify_rewind_q35(pos);
    pos_ = pos;
    if (next_port_ != nullptr) next_port_->rewind(pos);
}

void Glm5Model::dspark_upload() {
    auto* D = (DSparkState*) dsp_;
    if (D == nullptr || D->ups.empty()) return;
    const int E = g_.n_embd, FF = D->ff;
    const auto t0 = std::chrono::steady_clock::now();
    cudaEventSynchronize(D->ev_up);   // (the staging buffers of the last uploads; long done)
    const size_t gu = (size_t) FF * D->gu_rb, dn = (size_t) E * D->dn_rb;
    for (size_t i = 0; i < D->ups.size(); ++i) {
        const DspUp& u = D->ups[i];
        DspStage& S = D->st[u.s];
        uint8_t* slot = S.slots + (size_t) u.slot * D->blob;
        uint8_t* stg = D->stage_h + i * D->blob;
        // the evicted expert's table entry off first, then the bytes, then the new entry
        D->tabv_h[2 * i] = 0ull;
        D->tabv_h[2 * i + 1] = (unsigned long long) slot;
        if (u.evict >= 0)
            cudaMemcpyAsync(S.tab + u.evict, &D->tabv_h[2 * i], sizeof(unsigned long long), cudaMemcpyHostToDevice, D->up);
        std::memcpy(stg, S.src_g + (size_t) u.e * gu, gu);
        std::memcpy(stg + gu, S.src_u + (size_t) u.e * gu, gu);
        std::memcpy(stg + D->down_off, S.src_d + (size_t) u.e * dn, dn);
        cudaMemcpyAsync(slot, stg, D->down_off + dn, cudaMemcpyHostToDevice, D->up);
        cudaMemcpyAsync(S.tab + u.e, &D->tabv_h[2 * i + 1], sizeof(unsigned long long), cudaMemcpyHostToDevice, D->up);
        ++D->uploads;
    }
    D->ups.clear();
    cudaEventRecord(D->ev_up, D->up);
    D->ms_up += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

bool Glm5Model::dspark_draft(int32_t y, int32_t* drafts, float* conf, std::string& err) {
    if (dsp_ == nullptr && q35_mtp_on()) return q35_mtp_draft(y, drafts, conf, err);   // (qwen35moe's NextN block)
    if (dsp_ == nullptr && next_port_ != nullptr && next_port_->mtp_on()) {   // (qwen4exp's layer: the last part)
        const bool ok = next_port_->mtp_draft(y, drafts, conf, err);
        cudaSetDevice(dev_);
        return ok;
    }
    auto* D = (DSparkState*) dsp_;
    if (D == nullptr) { err = "dspark: not loaded"; return false; }
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto t0 = std::chrono::steady_clock::now();
    dspark_settle();
    dspark_upload();   // (a draft before the last one's uploads went out)
    const int64_t pos = pos_;
    const int E = g.n_embd, NE = D->ne, K = D->k, FF = D->ff, V = g.n_vocab, DH = g.head_dim;
    const int HD = D->nh * DH, OL = D->og * D->olr, QL = D->ql;
    const size_t HCE = 4 * (size_t) E, q8e = (size_t) (E / 32) * kQ8;
    // the context rows [lo, pos) when the ring holds every one of them, else none
    int64_t lo = std::max<int64_t>(D->ctx_lo, pos - kCtx);
    int n_ctx = D->ctx_end == pos ? (int) (pos - lo) : 0;
    if (n_ctx <= 0) {
        n_ctx = 0;
        lo = pos;
    }
    cudaStreamWaitEvent(s, D->ev_up, 0);   // the cache's last copies and table edits
    int ph = 0;
    const auto mark = [&] { if (D->prof) cudaEventRecord(D->ev[ph++], s); };
    mark();
    // the rows' streams: y's embedding, then the noise token's
    D->y_h[0] = y;
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (F->emb_tab_d != nullptr) {
        gf::embed_rows_q(pack_emb_type_, F->emb_tab_d, D->y_d, 1, E, D->emb_y, s);
    } else {
        tt->to_float(pack_emb_src_ + (size_t) y * ggml_row_size((ggml_type) pack_emb_type_, E), D->emb_h, E);
        cudaMemcpyAsync(D->emb_y, D->emb_h, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s);
    }
    cudaMemcpyAsync(D->dtok, D->y_h, sizeof(int), cudaMemcpyHostToDevice, s);
    float* Rc[kB];
    float* Ro[kB];
    for (int i = 0; i < kB; ++i) {
        Rc[i] = D->R + (size_t) i * 2 * HCE;
        Ro[i] = Rc[i] + HCE;
        gf::embed_streams(i == 0 ? D->emb_y : D->emb_noise, Rc[i], E, s);
    }
    if (D->v41) gf::copy_floats(D->pre, D->pre_init, kB * 8, s);   // (the reads' lagged pre: stage 0 the one-hot)
    const auto hc_read = [&](const DspStage& S, int i, const float* block_out, bool attn) {
        gf::HcArgs h;
        h.block_out = block_out;
        h.R_old = Rc[i];
        h.R_new = Ro[i];
        h.post_in = D->post + i * 8;
        h.comb_in = D->comb + i * 16;
        h.pre = D->pre + i * 8;
        h.post = D->post + i * 8;
        h.comb = D->comb + i * 16;
        h.w_fn = attn ? S.hc_attn_fn : S.hc_ffn_fn;
        h.w_scale = attn ? S.hc_attn_scale : S.hc_ffn_scale;
        h.w_base = attn ? S.hc_attn_base : S.hc_ffn_base;
        h.norm_w = attn ? S.attn_norm : S.ffn_norm;
        h.norm_eps = D->eps;
        h.hc_eps = D->hc_eps;
        h.iters = D->iters;
        h.n_embd = E;
        if (D->v41) {   // the collapse with the previous sublayer's pre: [0, 4) the attention side's, [4, 8) the FFN's
            h.pre = D->pre + i * 8 + (attn ? 0 : 4);
            h.pre_use = D->pre + i * 8 + (attn ? 4 : 0);
        }
        h.x = D->x + (size_t) i * E;
        h.xq = (uint8_t*) D->xq + (size_t) i * q8e;
        h.part = D->hpart + (size_t) i * D->hpart_n;
        h.counter = D->hcnt + (size_t) i * 16;
        gf::hc(h, s);
        if (block_out != nullptr) std::swap(Rc[i], Ro[i]);
    };
    for (int st = 0; st < kStages; ++st) {
        const DspStage& S = D->st[st];
        mark();
        // ---- attention: the block's rows over the context and the whole block
        for (int i = 0; i < kB; ++i) hc_read(S, i, st == 0 ? nullptr : D->ffn + (size_t) i * E, true);
        gf::MvJob j[2] = {{S.q_a.q, D->xq, D->x, D->qr_raw, nullptr, 1.0f, S.q_a.type, E, QL},
                          {S.kv.q, D->xq, D->x, D->kv_raw, nullptr, 1.0f, S.kv.type, E, DH}};
        if (!gf::mvn(j, 2, kB, s)) { err = "dspark: q_a / kv"; return false; }
        for (int i = 0; i < kB; ++i) {
            ds::PrepArgs a;
            a.qr_raw = D->qr_raw + (size_t) i * QL;
            a.q_a_norm = S.q_a_norm;
            a.qr = D->qr + (size_t) i * QL;
            a.qr_q = (uint8_t*) D->qr_q + (size_t) i * (QL / 32) * kQ8;
            a.q_lora = QL;
            a.kv_raw = D->kv_raw + (size_t) i * DH;
            a.kv_norm = S.kv_a_norm;
            a.raw = S.ring;   // (row (pos + i) % kRing: past the context's rows, rewritten by later commits)
            a.ring = kRing;
            a.head_dim = DH;
            a.n_rot = D->rot;
            a.rope = D->rope;
            a.comp_rope = D->rope;
            a.p = (int) (pos + i);
            a.eps = D->eps;
            ds::prep(a, s);
        }
        gf::MvJob jq = {S.q_b.q, D->qr_q, D->qr, D->q, nullptr, 1.0f, S.q_b.type, QL, HD};
        if (!gf::mvn(&jq, 1, kB, s)) { err = "dspark: q_b"; return false; }
        for (int i = 0; i < kB; ++i) {
            float* q = D->q + (size_t) i * HD;
            if (D->v41)   // (no per-head q norm)
                d41::q_rope(q, D->nh, DH, D->rot, D->rope, nullptr, 0, 0, D->rope, (int) (pos + i), s);
            else
                ds::q_prep(q, D->nh, DH, D->rot, D->rope, nullptr, 0, 0, D->rope, (int) (pos + i), kEps, s);
            ds::attend(q, S.ring, kRing, 0, S.ring, D->iota + (lo % kRing), n_ctx + kB, S.sinks, D->nh, DH, D->rot,
                       D->rope, (int) (pos + i), D->o + (size_t) i * HD, s, 0,
                       (uint8_t*) D->oq + (size_t) i * (HD / 32) * kQ8);
        }
        const int gin = D->nh / D->og * DH;
        const size_t wrow = gf::row_bytes(S.out_a.type, gin);
        gf::MvJob jo[gf::kMaxMvJobs];
        for (int gi = 0; gi < D->og; ++gi)
            jo[gi] = {(const uint8_t*) S.out_a.q + (size_t) gi * D->olr * wrow,
                      (const uint8_t*) D->oq + (size_t) gi * (gin / 32) * kQ8, D->o + (size_t) gi * gin,
                      D->oa + (size_t) gi * D->olr, nullptr, 1.0f, S.out_a.type, gin, D->olr, HD / 32, OL};
        if (!gf::mvn(jo, D->og, kB, s)) { err = "dspark: wo_a"; return false; }
        gf::quantize_q8_1(D->oa, D->oaq, kB * OL, s);
        gf::MvJob jb = {S.out_b.q, D->oaq, D->oa, D->mixer, nullptr, 1.0f, S.out_b.type, OL, E};
        if (!gf::mvn(&jb, 1, kB, s)) { err = "dspark: wo_b"; return false; }
        // ---- the MoE
        mark();
        for (int i = 0; i < kB; ++i) hc_read(S, i, D->mixer + (size_t) i * E, false);
        gf::MvJob jm[3] = {{S.gate_inp, nullptr, D->x, D->rlog, nullptr, 1.0f, gf::kTypeF32, E, NE},
                           {S.sh_gate.q, D->xq, nullptr, D->sh_g, nullptr, 1.0f, S.sh_gate.type, E, FF},
                           {S.sh_up.q, D->xq, nullptr, D->sh_u, nullptr, 1.0f, S.sh_up.type, E, FF}};
        if (!gf::mvn(jm, 3, kB, s)) { err = "dspark: router / shared expert"; return false; }
        if (!ds::dspark_route(D->rlog, S.probs_b, NE, K, D->wscale, S.tab, kB, D->plan_ptr, D->plan_w,
                              D->ids_d + (size_t) st * kB * K, s)) {
            err = "dspark: router";
            return false;
        }
        gf::sh_swiglu(D->sh_g, D->sh_u, D->clamp, kB * FF, D->sh_hq, s);
        // the block's rows in one gate/up and one down launch (moe_gate_up_cols: a row's plan each)
        gf::moe_gate_up_cols(D->etype, D->cols_d, kB, K, E, FF, D->clamp, D->xq, q8e, D->hq, D->hq_n, S.sh_down.q,
                             S.sh_down.type, D->sh_hq, (size_t) (FF / 32) * kQ8, FF, D->sh_out, E, s);
        gf::moe_down_cols(D->etype, D->cols_d, kB, K, E, FF, D->down_off, D->hq, D->hq_n, D->sh_out, D->ffn, E, s);
    }
    // ---- the head: the last write half, the hc head, the target's output rows, the Markov chain
    mark();
    for (int i = 0; i < kB; ++i) {
        gf::hc_post(D->ffn + (size_t) i * E, Rc[i], D->post + i * 8, D->comb + i * 16, E, Ro[i], s);
        if (D->v41)   // (no hc head: the last FFN's pre)
            d41::head_prep(Ro[i], D->pre + i * 8 + 4, D->norm, D->eps, E, D->head_x + (size_t) i * E,
                           (uint8_t*) D->head_xq + (size_t) i * q8e, s, D->head_h + (size_t) i * E);
        else
            ds::dspark_head(Ro[i], D->hc_head_fn, D->hc_head_scale, D->hc_head_base, g.hc_eps, D->norm, kEps, E,
                            D->head_h + (size_t) i * E, D->head_x + (size_t) i * E,
                            (uint8_t*) D->head_xq + (size_t) i * q8e, s);
    }
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, D->head_xq, D->head_x, D->logits, nullptr, 1.0f, ow.type, E, V};
    if (gf::mvn_supported(ow.type)) {
        if (!gf::mvn(&o, 1, kB, s)) { err = "dspark: output head"; return false; }
    } else {
        for (int i = 0; i < kB; ++i) {
            gf::MvJob ot = o;
            ot.xq = (const uint8_t*) D->head_xq + (size_t) i * q8e;
            ot.xf = D->head_x + (size_t) i * E;
            ot.y = D->logits + (size_t) i * V;
            if (!gf::mv(&ot, 1, s)) { err = "dspark: output head (type " + std::to_string(ow.type) + ")"; return false; }
        }
    }
    for (int i = 0; i < kB; ++i) {
        if (D->v41)
            ds::dspark_markov_f16(D->logits + (size_t) i * V, D->markov_w2h, D->markov_w1, V, D->dtok + i, D->conf_proj,
                                  D->conf_b, D->head_h + (size_t) i * E, E, D->biased, D->conf + i, s);
        else
            ds::dspark_markov(D->logits + (size_t) i * V, D->markov_w2.q, D->markov_w1, V, D->dtok + i, D->conf_proj,
                              D->head_h + (size_t) i * E, E, D->biased, D->conf + i, s);
        gf::argmax(D->biased, V, D->dtok + i + 1, s);
    }
    mark();
    cudaMemcpyAsync(D->res_tok, D->dtok + 1, kB * sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(D->res_conf, D->conf, kB * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(D->res_ids, D->ids_d, (size_t) kStages * kB * K * sizeof(int), cudaMemcpyDeviceToHost, s);
    if (cudaStreamSynchronize(s) != cudaSuccess || gf::launch_errors() != D->launch_err0) {
        err = std::string("dspark: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    for (int i = 0; i < kB; ++i) {
        drafts[i] = D->res_tok[i];
        conf[i] = D->res_conf[i];
    }
    // STRATA_DS4_DSPARK_DUMP=<file> (debug): a line a draft - the position (y's), y, the context rows, the 5 drafts,
    // their confidences, then each stage's routes (row-major, k a row) - for bench/v41dsp/v41_probe.py --compare
    if (D->dump != nullptr) {
        std::fprintf(D->dump, "%lld %d %d |", (long long) pos, y, n_ctx);
        for (int i = 0; i < kB; ++i) std::fprintf(D->dump, " %d", D->res_tok[i]);
        std::fprintf(D->dump, " |");
        for (int i = 0; i < kB; ++i) std::fprintf(D->dump, " %.6f", D->res_conf[i]);
        for (int st = 0; st < kStages; ++st) {
            std::fprintf(D->dump, " |");
            for (int i = 0; i < kB * K; ++i) std::fprintf(D->dump, " %d", D->res_ids[(size_t) st * kB * K + i]);
        }
        std::fprintf(D->dump, "\n");
        std::fflush(D->dump);
    }
    for (int i = 0; D->prof && i < DSparkState::kPh; ++i) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, D->ev[i], D->ev[i + 1]);
        D->ms_ph[i] += ms;
    }
    // ---- the cache: decayed use counts; this draft's missing experts with the most uses in, the coldest out
    for (int st = 0; st < kStages; ++st) {
        DspStage& S = D->st[st];
        for (float& v : S.score) v *= 0.95f;
        std::vector<int> miss;
        for (int i = 0; i < kB * K; ++i) {
            const int e = D->res_ids[(size_t) st * kB * K + i];
            if (e < 0 || e >= NE) continue;
            S.score[(size_t) e] += 1.0f;
            ++D->uses;
            if (S.slot_of[(size_t) e] < 0) {
                ++D->misses;
                if (std::find(miss.begin(), miss.end(), e) == miss.end()) miss.push_back(e);
            }
        }
        if (S.n_slots >= NE || D->n_up == 0) continue;
        std::sort(miss.begin(), miss.end(), [&](int a, int b) { return S.score[(size_t) a] > S.score[(size_t) b]; });
        int n = 0;
        for (int e : miss) {
            if (n >= D->n_up) break;
            int slot = -1, cold = -1;
            for (int k = 0; k < S.n_slots; ++k) {
                const int c = S.expert_in[(size_t) k];
                if (c < 0) { slot = k; cold = -1; break; }
                if (slot < 0 || S.score[(size_t) c] < S.score[(size_t) cold]) { slot = k; cold = c; }
            }
            if (slot < 0 || (cold >= 0 && S.score[(size_t) cold] >= S.score[(size_t) e])) break;
            if (cold >= 0) S.slot_of[(size_t) cold] = -1;
            S.slot_of[(size_t) e] = slot;
            S.expert_in[(size_t) slot] = e;
            D->ups.push_back({st, e, slot, cold});
            ++n;
        }
    }
    ++D->drafts;
    D->ms_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace strata::core
