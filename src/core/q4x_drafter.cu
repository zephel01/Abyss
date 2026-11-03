// SPDX-License-Identifier: MIT
//
// qwen4exp's MTP draft layer on the fast path.  The model's GGUFs carry no draft block; its weights are the runtime
// files tools/mtp_rt.py writes (dense.bin / dense.txt: Q8_0 projections, BF16 hyper-connection and router rows, F32
// norms with the "+1" applied; experts.bin: the 512 routed experts in the Q2_0 planar blob; draft_vocab.bin: the
// head's token subset), the same ones the reference path's MtpDrafter (src/core/mtp.cpp) runs.  The layer
// (MtpDrafter::record_forward): e2 = fc_embedding(rms(the next token's embedding)), h2 = fc_hidden a stream of
// rms(the trunk's final residual, 4 x n_embd as one row), R = h2 + e2 a stream; the attention hyper-connection, k / v
// into its own cache, dense attention over its cells, the MLP hyper-connection, the router, the 512 experts resident
// (the reference path's grouped Q2_0 product), the shared expert, the combine; the final mixer and the main head.
// Here the projections and hyper-connections are the fast path's kernels (mvn / hc4_combine_norm / hc4_mix /
// attn_decode), the experts the reference path's.  A draft step's own residual feeds the next chained step.
// STRATA_Q4X_MTP=<rt dir> loads it (with the decode's speculative path it drafts like qwen35moe's NextN block).

#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/qwen35_fast.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace q35 = strata::kernels::q35;

struct Q4xDrafter {
    struct W {
        int type = 0;   // 8 Q8_0, kTypeBF16, 0 F32
        const void* p = nullptr;
        int n_out = 0, n_in = 0;
    };
    std::map<std::string, W> w;
    void* dense = nullptr;
    uint8_t* experts = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr;
    void* arena = nullptr;
    int nb = 0;   // rows a call at most (the prompt's cache rows; a draft step is one)
    float *emb, *e2, *hn, *h2, *R, *xn, *lo, *gate, *inj, *inj2, *x, *k, *v, *qf, *q, *ao, *bo, *rl, *rw, *shg, *shu,
        *sh, *g1, *parts, *y, *hx, *logits, *part, *pend;
    void *xq, *hq;
    int32_t *ids, *grp_start, *counts, *hit_dst, *hit_slot, *tok;
    unsigned long long* grp_ptr;
    uint8_t* hit_xq;
    float* hit_xs;
    void* hit_scratch;
    int32_t* tok_h = nullptr;   // pinned: the rows' next tokens
    // the head: the draft vocabulary's rows of output.weight (or the whole head)
    void* head = nullptr;
    int head_type = 0;
    int64_t nv = 0;
    int32_t* dvocab = nullptr;
    uint64_t vram = 0;
};

namespace {
bool read_all(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    out.resize((size_t) f.tellg());
    f.seekg(0);
    return (bool) f.read((char*) out.data(), (std::streamsize) out.size());
}
}  // namespace

bool Glm5Model::q4x_mtp_load(const std::string& dir, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    if (!g.q4x || F == nullptr || l0_ != 0 || l1_ != g.n_layers) {
        err = "q4x mtp: one device's whole qwen4exp model only";
        return false;
    }
    const int E = g.n_embd, HE = g.hc * E, LR = g.hc_lr, D = g.head_dim, NH = g.n_head, HD = NH * D,
              KV = g.attn_head_kv * D, K = g.n_exp_used, FF = g.n_ff_exp;
    // the drafter's own 512 experts and router (experts.bin): the main model may keep fewer (a pruned build - the
    // drafter only proposes tokens, the main model checks them)
    constexpr int NE = 512;
    if (g.n_expert > NE || K != 10 || FF != 640 || E != 2560 || g.hc != 4) {
        err = "q4x mtp: the runtime files are for 512 experts, top-10, n_ff 640, 4 x 2560 streams";
        return false;
    }
    auto* M = new Q4xDrafter();
    q4d_ = M;
    // ---- the dense weights
    std::vector<uint8_t> blob;
    {
        std::ifstream idx(dir + "/dense.txt");
        if (!idx || !read_all(dir + "/dense.bin", blob)) {
            err = "q4x mtp: cannot read " + dir + "/dense.txt / dense.bin (tools/mtp_rt.py writes them)";
            return false;
        }
        if (cudaMalloc(&M->dense, blob.size()) != cudaSuccess) {
            err = "q4x mtp: the dense weights do not fit";
            return false;
        }
        cudaMemcpy(M->dense, blob.data(), blob.size(), cudaMemcpyHostToDevice);
        M->vram += blob.size();
        std::string line;
        while (std::getline(idx, line)) {
            std::istringstream ls(line);
            std::string name, kind;
            long long rows = 0, cols = 0, off = 0, bytes = 0;
            if (!(ls >> name >> kind >> rows >> cols >> off >> bytes)) continue;
            if ((size_t) (off + bytes) > blob.size()) {
                err = "q4x mtp: dense.txt points past dense.bin (" + name + ")";
                return false;
            }
            Q4xDrafter::W t;
            t.type = kind == "q8_0" ? 8 : kind == "bf16" ? gf::kTypeBF16 : 0;
            t.p = (const uint8_t*) M->dense + off;
            t.n_out = (int) rows;
            t.n_in = (int) cols;
            M->w[name] = t;
        }
    }
    const char* need[] = {"fc_embedding.weight", "fc_hidden.weight", "pre_fc_norm_embedding.weight",
                          "pre_fc_norm_hidden.weight", "attn_hyper_connection.hc_norm.weight",
                          "attn_hyper_connection.input_mix_weight_down.weight",
                          "attn_hyper_connection.input_mix_weight_up.weight",
                          "attn_hyper_connection.block_inject_weight.weight", "mlp_hyper_connection.hc_norm.weight",
                          "mlp_hyper_connection.input_mix_weight_down.weight",
                          "mlp_hyper_connection.input_mix_weight_up.weight",
                          "mlp_hyper_connection.block_inject_weight.weight", "hyper_connection_mixer.hc_norm.weight",
                          "hyper_connection_mixer.input_mix_weight_down.weight",
                          "hyper_connection_mixer.input_mix_weight_up.weight", "self_attn.q_proj.weight",
                          "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                          "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate.weight",
                          "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
                          "mlp.shared_expert.down_proj.weight", "mlp.shared_expert_gate.weight"};
    for (const char* n : need)
        if (M->w.find(n) == M->w.end()) {
            err = std::string("q4x mtp: dense.txt has no ") + n;
            return false;
        }
    // ---- the experts, resident
    {
        const size_t bytes = (size_t) NE * strata::kernels::cpu::BLOB;
        FILE* f = std::fopen((dir + "/experts.bin").c_str(), "rb");
        std::vector<uint8_t> eb(bytes);
        const bool rd = f != nullptr && std::fread(eb.data(), 1, bytes, f) == bytes;
        if (f) std::fclose(f);
        if (!rd) {
            err = "q4x mtp: experts.bin is missing or short";
            return false;
        }
        if (cudaMalloc((void**) &M->experts, bytes) != cudaSuccess) {
            err = "q4x mtp: the 512 experts do not fit";
            return false;
        }
        cudaMemcpy(M->experts, eb.data(), bytes, cudaMemcpyHostToDevice);
        M->vram += bytes;
    }
    // ---- the head: the draft vocabulary's rows (STRATA_Q4X_MTP_FULLHEAD=1: every token)
    const WSlot& ow = ws_map_.at("output.weight");
    M->head_type = ow.type;
    M->head = (void*) (ow.type == 0 ? (const void*) ow.f32 : ow.q);
    M->nv = g.n_vocab;
    {
        std::vector<uint8_t> raw;
        if (getenv("STRATA_Q4X_MTP_FULLHEAD") == nullptr && read_all(dir + "/draft_vocab.bin", raw) && raw.size() >= 4 &&
            raw.size() % 4 == 0) {
            const int64_t n = (int64_t) (raw.size() / 4);
            const size_t rb = ow.type == 0 ? (size_t) E * 4 : ggml_row_size((ggml_type) ow.type, E);
            std::vector<uint8_t> full((size_t) g.n_vocab * rb), sub((size_t) n * rb);
            cudaMemcpy(full.data(), M->head, full.size(), cudaMemcpyDeviceToHost);
            const int32_t* id = (const int32_t*) raw.data();
            bool ok = true;
            for (int64_t i = 0; i < n && ok; ++i) {
                ok = id[i] >= 0 && id[i] < g.n_vocab;
                if (ok) std::memcpy(sub.data() + (size_t) i * rb, full.data() + (size_t) id[i] * rb, rb);
            }
            if (ok && cudaMalloc(&M->head, sub.size()) == cudaSuccess &&
                cudaMalloc((void**) &M->dvocab, raw.size()) == cudaSuccess) {
                cudaMemcpy(M->head, sub.data(), sub.size(), cudaMemcpyHostToDevice);
                cudaMemcpy(M->dvocab, raw.data(), raw.size(), cudaMemcpyHostToDevice);
                M->nv = n;
                M->vram += sub.size() + raw.size();
            } else {
                M->head = (void*) (ow.type == 0 ? (const void*) ow.f32 : ow.q);
                M->dvocab = nullptr;
            }
        }
    }
    // ---- its cache and scratch: nb rows a call (the prompt's cache rows in groups of nb)
    const int ctx = (int) std::max<int64_t>(max_ctx_, 1);
    const int nb = M->nb = 64;
    const size_t part = q35::attn_part_floats(NH, D, ctx);
    const auto al = [](size_t b) { return (b + 255) & ~(size_t) 255; };
    const size_t q8 = 36;
    size_t bytes = 0;
    std::vector<std::pair<void**, size_t>> take;
    const auto T = [&](void** p, size_t b) { take.push_back({p, b}); bytes += al(b); };
    T((void**) &M->kc, (size_t) ctx * KV * 2);
    T((void**) &M->vc, (size_t) ctx * KV * 2);
    T((void**) &M->emb, (size_t) nb * E * 4);
    T((void**) &M->e2, (size_t) nb * E * 4);
    T((void**) &M->hn, (size_t) nb * HE * 4);
    T((void**) &M->h2, (size_t) nb * HE * 4);
    T((void**) &M->R, (size_t) nb * HE * 4);
    T((void**) &M->xn, (size_t) nb * HE * 4);
    T((void**) &M->lo, (size_t) nb * LR * 4);
    T((void**) &M->gate, (size_t) nb * HE * 4);
    T((void**) &M->inj, (size_t) nb * 4 * 4);
    T((void**) &M->inj2, (size_t) nb * 4 * 4);
    T((void**) &M->x, (size_t) nb * E * 4);
    T((void**) &M->k, (size_t) nb * KV * 4);
    T((void**) &M->v, (size_t) nb * KV * 4);
    T((void**) &M->qf, (size_t) 2 * HD * 4);
    T((void**) &M->q, (size_t) HD * 4);
    T((void**) &M->ao, (size_t) HD * 4);
    T((void**) &M->bo, (size_t) E * 4);
    T((void**) &M->rl, (size_t) NE * 4);
    T((void**) &M->rw, (size_t) K * 4);
    T((void**) &M->shg, (size_t) FF * 4);
    T((void**) &M->shu, (size_t) FF * 4);
    T((void**) &M->sh, (size_t) E * 4);
    T((void**) &M->g1, 64);
    T((void**) &M->parts, (size_t) K * E * 4);
    T((void**) &M->y, (size_t) E * 4);
    T((void**) &M->hx, (size_t) E * 4);
    T((void**) &M->logits, (size_t) g.n_vocab * 4);
    T((void**) &M->part, part * 4);
    T((void**) &M->pend, (size_t) HE * 4);
    T(&M->xq, (size_t) nb * HE / 32 * q8);
    T(&M->hq, (size_t) FF / 32 * q8);
    T((void**) &M->ids, (size_t) K * 4);
    T((void**) &M->grp_start, (size_t) (K + 1) * 4);
    T((void**) &M->counts, 16);
    T((void**) &M->hit_dst, (size_t) K * 4);
    T((void**) &M->hit_slot, (size_t) K * 4);
    T((void**) &M->tok, (size_t) (nb + 8) * 4);
    T((void**) &M->grp_ptr, (size_t) K * 8);
    T((void**) &M->hit_xq, (size_t) (E / 32) * 34);
    T((void**) &M->hit_xs, (size_t) (E / 32) * 4);
    T(&M->hit_scratch, (size_t) strata::kernels::moe_hit_grouped_scratch_bytes(K, E, FF));
    if (cudaMalloc(&M->arena, bytes) != cudaSuccess) {
        cudaGetLastError();
        err = "q4x mtp: its cache and buffers (" + std::to_string(bytes >> 20) + " MB) do not fit";
        return false;
    }
    cudaMemset(M->arena, 0, bytes);
    uint8_t* at = (uint8_t*) M->arena;
    for (auto& t : take) {
        *t.first = at;
        at += al(t.second);
    }
    M->vram += bytes;
    if (cudaHostAlloc((void**) &M->tok_h, (size_t) (nb + 8) * 4, cudaHostAllocDefault) != cudaSuccess) {
        err = "q4x mtp: the token staging did not allocate";
        return false;
    }
    // the trunk's pending row (q35_mtp_pend_set): the final residual, 4 streams
    F->q35_pend_h = M->pend;
    // the chain's tokens and confidences (q35_mtp_draft's)
    if (F->q35_mt_tok == nullptr) {
        if (cudaMalloc((void**) &F->q35_mt_tok, 64) != cudaSuccess ||
            cudaHostAlloc((void**) &F->q35_mt_conf_h, 256, cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &F->q35_mt_conf_d, F->q35_mt_conf_h, 0) != cudaSuccess) {
            err = "q4x mtp: the chain's outputs did not allocate";
            return false;
        }
    }
    std::fprintf(stderr, "q4x mtp: CUDA%d the draft layer from %s (%.0f MB, head over %lld tokens)\n", dev_, dir.c_str(),
                 (double) M->vram / 1048576.0, (long long) M->nv);
    return true;
}

// the draft layer over n rows at positions p0 .. p0 + n - 1 from the residual rows Rin (r_stride floats apart) and the
// rows' next tokens (tok, on the device): full = false writes the rows' K / V alone (n <= nb); full (n == 1) runs the
// whole layer, leaves its residual in M->R and its draft (and the draft's probability) in tok_out / conf_out
bool Glm5Model::q4x_mtp_forward(int64_t p0, int n, const float* Rin, size_t r_stride, const int* tok, bool full,
                                float* conf_out, int* tok_out, std::string& err) {
    auto* M = (Q4xDrafter*) q4d_;
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int E = g.n_embd, HE = g.hc * E, LR = g.hc_lr, D = g.head_dim, NH = g.n_head, HD = NH * D,
              NKV = g.attn_head_kv, KV = NKV * D, K = g.n_exp_used, FF = g.n_ff_exp;
    const float eps = g.norm_eps;
    if (n < 1 || n > M->nb || (full && n != 1)) {
        err = "q4x mtp: " + std::to_string(n) + " rows";
        return false;
    }
    const auto& W = M->w;
    const auto job = [&](const char* name, const void* xq, const float* xf, float* y) {
        const Q4xDrafter::W& t = W.at(name);
        gf::MvJob j{};
        j.w = t.p;
        j.xq = t.type == 8 ? xq : nullptr;
        j.xf = xf;
        j.y = y;
        j.type = t.type;
        j.n_in = t.n_in;
        j.n_out = t.n_out;
        return j;
    };
    const auto f32 = [&](const char* name) { return (const float*) W.at(name).p; };
    // a hyper-connection over n rows (q4x_mix's): R += out . 2 sigmoid(inj_prev / hc) when out, the streams normed,
    // inject (when inj_out), lo = silu(down . xn / hc), gate = up . lo, xo = mean_c xn sigmoid(gate)
    const auto mix = [&](const char* pfx, bool inject, const float* out, const float* inj_prev, float* inj_out,
                         float* xo) {
        const std::string P(pfx);
        q35::hc4_combine_norm(M->R, out, inj_prev, g.hc, f32((P + ".hc_norm.weight").c_str()), eps, E, g.hc, n, M->xn, s);
        gf::MvJob j[2];
        j[0] = job((P + ".input_mix_weight_down.weight").c_str(), nullptr, M->xn, M->lo);
        if (inject) j[1] = job((P + ".block_inject_weight.weight").c_str(), nullptr, M->xn, inj_out);
        if (!gf::mvn(j, inject ? 2 : 1, n, s)) return false;
        q35::silu_scale(M->lo, n * LR, 1.0f / (float) g.hc, s);
        gf::MvJob u = job((P + ".input_mix_weight_up.weight").c_str(), nullptr, M->lo, M->gate);
        if (!gf::mvn(&u, 1, n, s)) return false;
        q35::hc4_mix(M->xn, M->gate, E, g.hc, n, xo, s);
        return true;
    };
    // ---- the two input branches
    if (!gf::embed_rows_q(F->emb_tab_d ? pack_emb_type_ : 8 /* Q8_0 */, F->emb_tab_d ? F->emb_tab_d : F->emb_q8_d, tok, n,
                          E, M->emb, s)) {
        err = "q4x mtp: the embedding rows";
        return false;
    }
    strata::kernels::rms_norm_weighted(M->emb, f32("pre_fc_norm_embedding.weight"), n, E, eps, (void*) s);
    gf::quantize_q8_1(M->emb, M->xq, n * E, s);
    gf::MvJob fe = job("fc_embedding.weight", M->xq, M->emb, M->e2);
    if (!gf::mvn(&fe, 1, n, s)) { err = "q4x mtp: fc_embedding"; return false; }
    cudaMemcpy2DAsync(M->hn, (size_t) HE * 4, Rin, (r_stride ? r_stride : (size_t) HE) * 4, (size_t) HE * 4, (size_t) n,
                      cudaMemcpyDeviceToDevice, s);
    strata::kernels::rms_norm_weighted(M->hn, f32("pre_fc_norm_hidden.weight"), n, HE, eps, (void*) s);
    gf::quantize_q8_1(M->hn, M->xq, n * HE, s);
    gf::MvJob fh = job("fc_hidden.weight", M->xq, M->hn, M->h2);
    if (!gf::mvn(&fh, 1, n * g.hc, s)) { err = "q4x mtp: fc_hidden"; return false; }
    strata::kernels::add_streams_broadcast(M->h2, M->e2, M->R, E, g.hc, n, (void*) s);
    // ---- the attention: k / v into the cache (all rows), then for a draft step q and the cells up to its own
    if (!mix("attn_hyper_connection", true, nullptr, nullptr, M->inj, M->x)) { err = "q4x mtp: the attention mix"; return false; }
    gf::quantize_q8_1(M->x, M->xq, n * E, s);
    gf::MvJob kv[3];
    kv[0] = job("self_attn.k_proj.weight", M->xq, M->x, M->k);
    kv[1] = job("self_attn.v_proj.weight", M->xq, M->x, M->v);
    if (!full) {
        if (!gf::mvn(kv, 2, n, s)) { err = "q4x mtp: k / v"; return false; }
        q35::attn_kv_rows(M->k, M->v, f32("self_attn.k_norm.weight"), eps, NKV, D, g.n_rot, g.rope_base, p0, n, M->kc,
                          M->vc, s);
        return true;
    }
    kv[2] = job("self_attn.q_proj.weight", M->xq, M->x, M->qf);
    if (!gf::mv(kv, 3, s)) { err = "q4x mtp: q / k / v"; return false; }
    q35::attn_prep(M->qf, M->k, M->v, f32("self_attn.q_norm.weight"), f32("self_attn.k_norm.weight"), eps, NH, NKV, D,
                   g.n_rot, g.rope_base, (int) p0, M->kc, M->vc, M->q, s);
    q35::attn_decode(M->q, M->kc, M->vc, NH, NKV, D, (int) (p0 + 1), 1.0f / std::sqrt((float) D), M->part, M->ao, s);
    q35::attn_gate(M->ao, M->qf, NH, D, s);
    gf::quantize_q8_1(M->ao, M->xq, HD, s);
    gf::MvJob o = job("self_attn.o_proj.weight", M->xq, M->ao, M->bo);
    if (!gf::mv(&o, 1, s)) { err = "q4x mtp: o_proj"; return false; }
    // ---- the MoE: router, the resident experts, the shared expert, the combine
    if (!mix("mlp_hyper_connection", true, M->bo, M->inj, M->inj2, M->x)) { err = "q4x mtp: the MLP mix"; return false; }
    gf::MvJob rj[2];
    rj[0] = job("mlp.gate.weight", nullptr, M->x, M->rl);
    rj[1] = job("mlp.shared_expert_gate.weight", nullptr, M->x, M->g1);
    if (!gf::mv(rj, 2, s)) { err = "q4x mtp: the router"; return false; }
    strata::kernels::native_router_top10(M->rl, M->ids, M->rw, (void*) s);
    strata::kernels::moe_group_resident(M->ids, K, K, M->experts, (int64_t) strata::kernels::cpu::BLOB, M->grp_ptr,
                                        M->grp_start, M->counts, M->hit_dst, M->hit_slot, (void*) s);
    strata::kernels::quantize_q8_0_scaled(M->x, M->hit_xq, M->hit_xs, E, (void*) s);
    strata::kernels::moe_grouped_s2(M->grp_ptr, M->grp_start, M->counts, M->hit_dst, M->hit_slot, K, K, M->hit_xq,
                                    M->hit_xs, M->hit_scratch, M->parts, (void*) s);
    gf::quantize_q8_1(M->x, M->xq, E, s);
    gf::MvJob sj[2];
    sj[0] = job("mlp.shared_expert.gate_proj.weight", M->xq, M->x, M->shg);
    sj[1] = job("mlp.shared_expert.up_proj.weight", M->xq, M->x, M->shu);
    if (!gf::mv(sj, 2, s)) { err = "q4x mtp: the shared expert's gate / up"; return false; }
    gf::swiglu_q8(M->shg, M->shu, INFINITY, FF, M->hq, s);
    gf::MvJob sd = job("mlp.shared_expert.down_proj.weight", M->hq, nullptr, M->sh);
    if (!gf::mv(&sd, 1, s)) { err = "q4x mtp: the shared expert's down"; return false; }
    q35::scale_sigmoid(M->sh, E, M->g1, s);
    strata::kernels::moe_combine(M->parts, M->rw, M->sh, M->y, E, K, (void*) s);
    // ---- the final mixer (the MoE written first: M->R is then the chain's next residual) and the head
    if (!mix("hyper_connection_mixer", false, M->y, M->inj2, nullptr, M->hx)) { err = "q4x mtp: the final mixer"; return false; }
    gf::quantize_q8_1(M->hx, M->xq, E, s);
    gf::MvJob h{};
    h.w = M->head;
    h.xq = M->xq;
    h.xf = M->hx;
    h.y = M->logits;
    h.type = M->head_type;
    h.n_in = E;
    h.n_out = (int) M->nv;
    if (!gf::mv(&h, 1, s)) { err = "q4x mtp: the head"; return false; }
    gf::argmax(M->logits, (int) M->nv, tok_out, s);
    q35::top_prob(M->logits, (int) M->nv, tok_out, conf_out, s);
    if (M->dvocab != nullptr) strata::kernels::map_ids(tok_out, M->dvocab, 1, (void*) s);
    return true;
}

// the cache rows of positions p0 .. p0 + n - 1 (any n: groups of nb) from the residual rows and the next tokens (host)
bool Glm5Model::q4x_mtp_rows(int64_t p0, int n, const float* Rin, size_t r_stride, const int32_t* next, std::string& err) {
    auto* M = (Q4xDrafter*) q4d_;
    cudaStream_t s = fast_->cs;
    const size_t rs = r_stride ? r_stride : (size_t) g_.hc * g_.n_embd;
    for (int c = 0; c < n; c += M->nb) {
        const int m = std::min(M->nb, n - c);
        if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "q4x mtp rows", err)) return false;   // (tok_h's last copy)
        std::memcpy(M->tok_h, next + c, (size_t) m * 4);
        cudaMemcpyAsync(M->tok, M->tok_h, (size_t) m * 4, cudaMemcpyHostToDevice, s);
        if (!q4x_mtp_forward(p0 + c, m, Rin + (size_t) c * rs, rs, M->tok, false, nullptr, nullptr, err)) return false;
    }
    q35_row_done_ = std::max(q35_row_done_, p0 + n - 1);
    return true;
}

const float* Glm5Model::q4x_mtp_residual() const { return q4d_ != nullptr ? ((const Q4xDrafter*) q4d_)->R : nullptr; }

}  // namespace strata::core
