// src/core/ernie_fast_path.cu - baidu ERNIE-5.5-Flash-Preview on the deepseek4 fast decode path (Glm5Model members).
//
// An ERNIE layer is a DeepSeek V4 Flash layer with these differences (worklog 2026-11-25_ernie55-port-survey.md
// section 1; tools/ernie55_convert.py writes them as deepseek4.ernie.* keys and these tensors):
//   * the HCA layers (compress ratio 128): V4's attention (dsv4_fast_path.cu) plus the value heads' mix VHA on its
//     output (ernie_hca_vha: o_h <- sum_g M[h][g] o_g, M = I + V U^T folded here at load) and an output gate on wo_a's
//     output (ernie_hca_gate: sigmoid(qr . attn_gate), rounded to BF16);
//   * six MLA + DSA layers (fast_ernie_mla): q through a 1024 LoRA (q_a, its norm, q_b: 64 heads of 192 nope + 64
//     rope), one 512 latent + a 64-dim rope key a position (kv_a_mqa, its norm), kv_b absorbed into the query and the
//     output, plain RoPE (base 8e6, adjacent pairs), sinks, VHA, the gate, attn_output.  The DSA indexer picks the top
//     2048 of the positions older than the window only past er_dense_ctx() positions - the engine refuses longer
//     contexts until it is implemented;
//   * the mHC pre's RMS factor 1 / (sqrt(mean) + eps) (HcArgs::eps_out);
//   * the FFN (fast_ernie_ffn): layer 0 a dense SiTU FFN (14336); the others the latent MoE - two routers
//     (s = sigmoid(l0) + sigmoid(l1), selected by s + bias, weighted s / sum s times the expert's learned scale, which
//     rides the experts' down scale MoeDev::nv_scale), fc1 (4096 -> 2048), the 10 routed experts in the latent through
//     the expert tiers (MXFP4, SiTU), RMSNorm, fc2 (2048 -> 4096), plus the shared SiTU expert on x;
//   * MultiMax on the logits (ernie_multimax).
// Not here yet (refused where they would start): the prompt path, verify windows, DSpark, the CPU / iGPU lanes, layer
// splits, the DSA indexer (contexts past er_dense_ctx()).
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/dsv4_fast.hpp"
#include "strata/kernels/ernie_fast.hpp"
#include "strata/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace ek = strata::kernels::ernie;

struct Ernie {
    struct Lay {
        WSlot gate;                      // attn_gate: q latent (q_lora) -> the gate (HCA o_groups x o_lora, MLA heads x v)
        float* vha = nullptr;            // M = I + V U^T [heads][heads] (device, owned)
        // MLA layers
        WSlot q_a, q_b, kv_a, out;
        const uint16_t* kvb = nullptr;   // attn_kv_b BF16 [heads (nope + v)][kv_lora]
        const float *q_a_norm = nullptr, *kv_a_norm = nullptr, *sinks = nullptr;
        float* cache = nullptr;          // [ctx][kv_lora + rope] F32 (owned)
        // MoE layers
        const uint16_t* router1 = nullptr;
        WSlot lat_down, lat_up;
        const float* lat_norm = nullptr;
    };
    std::vector<Lay> L;
    int ctx = 0;
    float *rlog1 = nullptr, *xl = nullptr, *yl = nullptr, *ylr = nullptr, *sh_h = nullptr, *dh = nullptr, *glog = nullptr;
    void* xlq = nullptr;
    float *qr_raw = nullptr, *qc = nullptr, *kva = nullptr, *q = nullptr, *qlat = nullptr, *score = nullptr,
          *olat = nullptr, *o = nullptr;
    const float *mm_r = nullptr, *mm_t = nullptr;
    std::vector<void*> owned;
    bool dump = false;
    int gate_bf16 = 1;   // (STRATA_ERNIE_GATE_BF16=0, debug: the gate's sigmoid unrounded - the float64 reference's dumps)
};

namespace {
void* dev_alloc(Ernie* E, size_t bytes, std::string& err, const char* what) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        cudaGetLastError();
        err = std::string("ernie: ") + what + " (" + std::to_string(bytes >> 20) + " MB) did not allocate";
        return nullptr;
    }
    cudaMemset(p, 0, bytes);
    E->owned.push_back(p);
    return p;
}

// STRATA_ERNIE_DUMP=<dir> (debug): n floats of device memory appended to <dir>/<name>.f32
void er_dump(const char* name, const float* d, size_t n, cudaStream_t s) {
    static const char* dir = getenv("STRATA_ERNIE_DUMP");
    if (dir == nullptr || !dir[0]) return;
    cudaStreamSynchronize(s);
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (FILE* f = std::fopen((std::string(dir) + "/" + name + ".f32").c_str(), "ab")) {
        std::fwrite(h.data(), sizeof(float), n, f);
        std::fclose(f);
    }
}
}  // namespace

bool Glm5Model::ernie_setup(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    if (n_parts_ != 1 || l0_ != 0 || l1_ != g.n_layers || split_next_ != nullptr) {
        err = "ernie: a layer split is not implemented yet (one GPU)";
        return false;
    }
    if (max_ctx_ > g.er_dense_ctx()) {
        err = "ernie: the MLA layers' DSA indexer is not implemented yet - --max-context up to " +
              std::to_string(g.er_dense_ctx()) + " (asked " + std::to_string((long long) max_ctx_) + ")";
        return false;
    }
    Ernie* E = new Ernie();
    F->ernie = E;
    E->ctx = (int) std::max<int64_t>(max_ctx_, 1);
    E->dump = getenv("STRATA_ERNIE_DUMP") != nullptr;
    E->gate_bf16 = g.er_gate_bf16 ? 1 : 0;
    if (const char* v = getenv("STRATA_ERNIE_GATE_BF16")) E->gate_bf16 = std::atoi(v) != 0 ? 1 : 0;
    E->L.resize((size_t) g.n_layers);
    std::string missing;
    const int H = g.n_head, R = g.er_vha_rank;
    const int LW = g.er_mla_kvlora + g.er_mla_rope;   // an MLA cache row
    for (int il = 0; il < g.n_layers; ++il) {
        auto& Ly = F->L[(size_t) il];
        auto& El = E->L[(size_t) il];
        const std::string P = "blk." + std::to_string(il) + ".";
        const auto f32 = [&](const char* n) -> const float* {
            auto it = w_.find(P + n);
            if (it == w_.end()) { missing += P + n + " (f32) "; return nullptr; }
            return it->second;
        };
        const auto b16 = [&](const char* n) -> const uint16_t* {
            auto it = w16_.find(P + n);
            if (it == w16_.end()) { missing += P + n + " (bf16) "; return nullptr; }
            return it->second;
        };
        const auto q = [&](const char* n) -> WSlot {
            auto it = ws_map_.find(P + n);
            if (it == ws_map_.end() || it->second.type == 0 || !gf::mv_supported(it->second.type)) {
                missing += P + n + " (gemv) ";
                return WSlot{};
            }
            return it->second;
        };
        El.gate = q("attn_gate.weight");
        // VHA: M = I + V U^T (U, V: [heads][rank] F32), folded in double
        const float* du = f32("attn_vha_u.weight");
        const float* dv = f32("attn_vha_v.weight");
        if (du != nullptr && dv != nullptr) {
            std::vector<float> U((size_t) H * R), V((size_t) H * R), M((size_t) H * H);
            cudaMemcpy(U.data(), du, U.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(V.data(), dv, V.size() * 4, cudaMemcpyDeviceToHost);
            for (int h = 0; h < H; ++h)
                for (int gg = 0; gg < H; ++gg) {
                    double a = h == gg ? 1.0 : 0.0;
                    for (int r = 0; r < R; ++r) a += (double) V[(size_t) h * R + r] * (double) U[(size_t) gg * R + r];
                    M[(size_t) h * H + gg] = (float) a;
                }
            El.vha = (float*) dev_alloc(E, M.size() * 4, err, "the VHA matrices");
            if (El.vha == nullptr) return false;
            cudaMemcpy(El.vha, M.data(), M.size() * 4, cudaMemcpyHostToDevice);
        }
        if (g.is_er_mla(il)) {
            El.q_a = q("attn_q_a.weight");
            El.q_b = q("attn_q_b.weight");
            El.kv_a = q("attn_kv_a_mqa.weight");
            El.out = q("attn_output.weight");
            El.kvb = b16("attn_kv_b.weight");
            El.q_a_norm = f32("attn_q_a_norm.weight");
            El.kv_a_norm = f32("attn_kv_a_norm.weight");
            El.sinks = f32("attn_sinks.weight");
            El.cache = (float*) dev_alloc(E, (size_t) E->ctx * LW * 4, err, "an MLA cache");
            if (El.cache == nullptr) return false;
        }
        if (il >= g.dense_lead) {
            El.router1 = b16("ffn_gate_inp_1.weight");
            El.lat_down = q("ffn_latent_down.weight");
            El.lat_up = q("ffn_latent_up.weight");
            El.lat_norm = f32("ffn_latent_norm.weight");
            // the learned per-expert scale: the experts' down scale (MoeDev::nv_scale [2 e + 1], the route weight's
            // factor in moe_down), the gate/up one 1 (an exact product)
            const float* sc = f32("ffn_exp_scale.weight");
            if (sc != nullptr) {
                if (Ly.nv_scale != nullptr) {
                    missing += P + "ffn_exp_scale beside NVFP4 expert scales ";
                } else {
                    std::vector<float> hs((size_t) g.n_expert), nv((size_t) 2 * g.n_expert, 1.0f);
                    cudaMemcpy(hs.data(), sc, hs.size() * 4, cudaMemcpyDeviceToHost);
                    for (int e = 0; e < g.n_expert; ++e) nv[(size_t) 2 * e + 1] = hs[(size_t) e];
                    Ly.nv_scale = (float*) dev_alloc(E, nv.size() * 4, err, "the expert scales");
                    if (Ly.nv_scale == nullptr) return false;
                    cudaMemcpy(Ly.nv_scale, nv.data(), nv.size() * 4, cudaMemcpyHostToDevice);
                }
            }
            if (Ly.router == nullptr || Ly.router_type != gf::kTypeBF16 || Ly.sh_gate.type != gf::kTypeBF16 ||
                Ly.sh_up.type != gf::kTypeBF16 || Ly.sh_down.type != gf::kTypeBF16)
                missing += P + "router / shared expert (BF16 expected) ";
        } else if (Ly.ffn_gate.type != gf::kTypeBF16 || Ly.ffn_up.type != gf::kTypeBF16 || Ly.ffn_down.type != gf::kTypeBF16) {
            missing += P + "dense FFN (BF16 expected) ";
        }
    }
    {
        const auto r = w_.find("output_multimax_r.weight"), t = w_.find("output_multimax_t.weight");
        if (g.er_multimax && (r == w_.end() || t == w_.end())) missing += "output_multimax_r / _t ";
        else if (g.er_multimax) {
            E->mm_r = r->second;
            E->mm_t = t->second;
        }
    }
    if (!missing.empty()) {
        err = "ernie: unresolved weights: " + missing.substr(0, 1500);
        return false;
    }
    const size_t X = (size_t) g.exp_dim(), FS = (size_t) g.n_ff_exp * g.n_shared, MH = (size_t) g.n_head;
    const size_t gate_n = std::max((size_t) g.o_groups * g.o_lora, MH * g.er_mla_v);
    const auto fl = [&](size_t n, const char* what) { return (float*) dev_alloc(E, n * 4, err, what); };
    if (!(E->rlog1 = fl(g.n_expert, "scratch")) || !(E->xl = fl(X, "scratch")) || !(E->yl = fl(X, "scratch")) ||
        !(E->ylr = fl(X, "scratch")) || !(E->sh_h = fl(FS, "scratch")) || !(E->dh = fl(g.n_ff_dense, "scratch")) ||
        !(E->glog = fl(gate_n, "scratch")) || !(E->qr_raw = fl(g.er_mla_qlora, "scratch")) ||
        !(E->qc = fl(g.er_mla_qlora, "scratch")) || !(E->kva = fl(LW, "scratch")) ||
        !(E->q = fl(MH * (g.er_mla_nope + g.er_mla_rope), "scratch")) || !(E->qlat = fl(MH * g.er_mla_kvlora, "scratch")) ||
        !(E->score = fl(MH * (size_t) E->ctx, "scratch")) || !(E->olat = fl(MH * g.er_mla_kvlora, "scratch")) ||
        !(E->o = fl(MH * g.er_mla_v, "scratch")))
        return false;
    if (!(E->xlq = dev_alloc(E, X / 32 * 36, err, "scratch"))) return false;
    int n_mla = 0;
    for (int il = 0; il < g.n_layers; ++il) n_mla += g.is_er_mla(il) ? 1 : 0;
    std::fprintf(stderr, "ernie: CUDA%d %d MLA layers (F32 caches %.2f GB for %d tokens), latent MoE %d -> %d, %d of %d "
                         "experts, VHA rank %d, MultiMax %s; token path only (no prompt path, verify window, DSpark, "
                         "lanes; DSA indexer: contexts up to %d)\n",
                 dev_, n_mla, (double) n_mla * E->ctx * LW * 4 / 1073741824.0, E->ctx, g.n_embd, g.exp_dim(),
                 g.n_exp_used, g.n_expert, g.er_vha_rank, g.er_multimax ? "on" : "off", g.er_dense_ctx());
    return true;
}

void Glm5Model::ernie_free() {
    if (fast_ == nullptr || fast_->ernie == nullptr) return;
    Ernie* E = fast_->ernie;
    for (int il = 0; il < (int) fast_->L.size() && il < g_.n_layers; ++il)
        if (il >= g_.dense_lead) fast_->L[(size_t) il].nv_scale = nullptr;   // (owned below)
    for (void* p : E->owned) cudaFree(p);
    delete E;
    fast_->ernie = nullptr;
}

// HCA: the value heads' mix on the attention's de-roped output (the mix and the inverse RoPE commute: the same rotation
// on every head), then its q8_1 again (wo_a's input)
bool Glm5Model::ernie_hca_vha(int il, std::string& err) {
    FastState* F = fast_;
    Ernie* E = F->ernie;
    const Glm5Geometry& g = g_;
    ek::vha(F->ds_o, E->L[(size_t) il].vha, g.n_head, g.head_dim, F->cs);
    gf::quantize_q8_1(F->ds_o, F->ds_oq, g.n_head * g.head_dim, F->cs);
    if (ek::launch_errors() > 0) { err = "ernie: vha"; return false; }
    return true;
}

// HCA: wo_a's output (o_groups x o_lora) times sigmoid(qr . attn_gate) (BF16-rounded)
bool Glm5Model::ernie_hca_gate(int il, std::string& err) {
    FastState* F = fast_;
    Ernie* E = F->ernie;
    const Glm5Geometry& g = g_;
    const auto& El = E->L[(size_t) il];
    const int n = g.o_groups * g.o_lora;
    gf::MvJob j = {El.gate.q, nullptr, F->qr, E->glog, nullptr, 1.0f, El.gate.type, g.q_lora, n};
    if (!gf::mv(&j, 1, F->cs)) { err = "ernie: the HCA gate"; return false; }
    ek::gate_mul(F->ds_oa, E->glog, n, E->gate_bf16, F->cs);
    return true;
}

bool Glm5Model::fast_ernie_mla(int il, int64_t p64, std::string& err) {
    FastState* F = fast_;
    Ernie* E = F->ernie;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& El = E->L[(size_t) il];
    const int p = (int) p64, H = g.n_head, NO = g.er_mla_nope, RO = g.er_mla_rope, V = g.er_mla_v, LK = g.er_mla_kvlora;
    const int QL = g.er_mla_qlora;
    if (p + 1 > E->ctx || p + 1 > g.er_dense_ctx()) {
        err = "ernie: position " + std::to_string(p) + " is past the MLA cache (" + std::to_string(E->ctx) +
              " tokens; the DSA indexer is not implemented yet)";
        return false;
    }
    gf::MvJob j[2];
    j[0] = {El.q_a.q, nullptr, F->x, E->qr_raw, nullptr, 1.0f, El.q_a.type, g.n_embd, QL};
    j[1] = {El.kv_a.q, nullptr, F->x, E->kva, nullptr, 1.0f, El.kv_a.type, g.n_embd, LK + RO};
    if (!gf::mv(j, 2, s)) { err = "ernie: MLA q_a / kv_a"; return false; }
    ek::rms(E->qr_raw, El.q_a_norm, E->qc, QL, g.norm_eps, s);
    float* row = El.cache + (size_t) p * (LK + RO);
    ek::mla_kv(E->kva, El.kv_a_norm, row, LK, RO, p, g.er_mla_rope_base, g.norm_eps, s);
    gf::MvJob jq = {El.q_b.q, nullptr, E->qc, E->q, nullptr, 1.0f, El.q_b.type, QL, H * (NO + RO)};
    if (!gf::mv(&jq, 1, s)) { err = "ernie: MLA q_b"; return false; }
    ek::mla_q_rope(E->q, H, NO, RO, p, g.er_mla_rope_base, s);
    ek::mla_absorb(E->q, El.kvb, E->qlat, H, NO, RO, V, LK, s);
    ek::mla_attend(E->qlat, E->q, El.cache, p + 1, El.sinks, 1.0f / std::sqrt((float) (NO + RO)), E->score, E->olat, H,
                   NO, RO, LK, s);
    ek::mla_v(E->olat, El.kvb, E->o, H, NO, V, LK, s);
    ek::vha(E->o, El.vha, H, V, s);
    gf::MvJob jg = {El.gate.q, nullptr, E->qc, E->glog, nullptr, 1.0f, El.gate.type, QL, H * V};
    if (!gf::mv(&jg, 1, s)) { err = "ernie: MLA gate"; return false; }
    ek::gate_mul(E->o, E->glog, H * V, E->gate_bf16, s);
    gf::MvJob jo = {El.out.q, nullptr, E->o, F->mixer, nullptr, 1.0f, El.out.type, H * V, g.n_embd};
    if (!gf::mv(&jo, 1, s)) { err = "ernie: MLA output"; return false; }
    if (E->dump) er_dump(("mla_o-" + std::to_string(il)).c_str(), E->o, (size_t) H * V, s);
    if (ek::launch_errors() > 0) { err = "ernie: MLA kernels"; return false; }
    return true;
}

bool Glm5Model::fast_ernie_ffn(int il, std::string& err) {
    FastState* F = fast_;
    Ernie* E = F->ernie;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const auto& El = E->L[(size_t) il];
    if (!Ly.moe) {   // the dense SiTU FFN
        gf::MvJob j[2];
        j[0] = {Ly.ffn_gate.q, nullptr, F->x, F->dg, nullptr, 1.0f, Ly.ffn_gate.type, g.n_embd, g.n_ff_dense};
        j[1] = {Ly.ffn_up.q, nullptr, F->x, F->du, nullptr, 1.0f, Ly.ffn_up.type, g.n_embd, g.n_ff_dense};
        if (!gf::mv(j, 2, s)) { err = "ernie: dense ffn"; return false; }
        ek::situ(F->dg, F->du, E->dh, g.n_ff_dense, s);
        gf::MvJob dn = {Ly.ffn_down.q, nullptr, E->dh, F->ffn, nullptr, 1.0f, Ly.ffn_down.type, g.n_ff_dense, g.n_embd};
        if (!gf::mv(&dn, 1, s)) { err = "ernie: dense ffn down"; return false; }
        return true;
    }
    const int X = g.exp_dim(), FS = g.n_ff_exp * g.n_shared, NE = g.n_expert;
    // the two routers, fc1 into the latent, the shared expert's gate / up (all on x, BF16)
    gf::MvJob j[5];
    j[0] = {Ly.router, nullptr, F->x, F->rlog, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, NE};
    j[1] = {El.router1, nullptr, F->x, E->rlog1, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, NE};
    j[2] = {El.lat_down.q, nullptr, F->x, E->xl, nullptr, 1.0f, El.lat_down.type, g.n_embd, X};
    j[3] = {Ly.sh_gate.q, nullptr, F->x, F->sh_g, nullptr, 1.0f, Ly.sh_gate.type, g.n_embd, FS};
    j[4] = {Ly.sh_up.q, nullptr, F->x, F->sh_u, nullptr, 1.0f, Ly.sh_up.type, g.n_embd, FS};
    if (!gf::mv(j, 5, s)) { err = "ernie: routers / fc1 / shared expert"; return false; }
    ek::router2(F->rlog, E->rlog1, F->rlog, NE, s);
    gf::quantize_q8_1(E->xl, E->xlq, X, s);
    if (E->dump) {
        er_dump(("rs-" + std::to_string(il)).c_str(), F->rlog, (size_t) NE, s);
        er_dump(("xl-" + std::to_string(il)).c_str(), E->xl, (size_t) X, s);
    }
    gf::MoeDev md = F->md;
    md.nv_scale = Ly.nv_scale;   // (the learned per-expert scale on the down weights)
    md.pf_src = F->pf_buf[il & 1];
    md.pf_dst = F->pf_buf[il & 1] + 8;
    md.pf_n = F->pf_n_buf[il & 1];
    static const int promote_min = [] {
        const char* v = getenv("STRATA_GLM_PROMOTE_MIN");
        return v ? std::max(0, std::atoi(v)) : 0;
    }();
    gf::moe_route(F->rlog, Ly.router_bias, NE, g.n_exp_used, g.w_scale, g.norm_w != 0, il, E->xl, X, md, nullptr,
                  nullptr, g.swiglu_shexp, 0, F->sh_hq, s, nullptr, nullptr, 0, nullptr, nullptr, 0, fast_skip_from(il),
                  0ull, promote_min, 0, g.gating, nullptr, fast_skip_w(il), F->tail_skip_disk ? 1 : 0, nullptr);
    if (F->prof_on) F->mark("moe_route");
    ++F->expected;
    if (F->bg_moe > 0 && !F->kda_graph_on) F->bg_drain(s, F->bg_moe, F->bg_moe_mask);
    // the shared expert while the fetches run: SiTU, its down (BF16 on the float activation) -> sh_out
    ek::situ(F->sh_g, F->sh_u, E->sh_h, FS, s);
    gf::MvJob sd = {Ly.sh_down.q, nullptr, E->sh_h, F->sh_out, nullptr, 1.0f, Ly.sh_down.type, FS, g.n_embd};
    if (!gf::mv(&sd, 1, s)) { err = "ernie: shared expert down"; return false; }
    gf::moe_wait(md, X, s);
    if (F->stream_fetch)
        gf::moe_fetch(md, g.n_exp_used, Ly.blob, s, Ly.gu_bytes, Ly.dn_bytes, F->read_chunks, Ly.up_bytes);
    else
        gf::moe_fetch(md, g.n_exp_used, Ly.blob, s);
    gf::moe_gate_up(Ly.gu_type, md, g.n_exp_used, X, g.n_ff_exp, g.lim_exp(il), E->xlq, F->hq, nullptr, 0, F->sh_hq, 0,
                    F->sh_out, s, Ly.up_type);
    gf::moe_down(Ly.d_type, md, g.n_exp_used, X, g.n_ff_exp, Ly.down_off, F->hq, nullptr, E->yl, s);
    if (F->prof_on) F->mark("moe_down");
    // back to n_embd: fc2 of the normed routed sum, plus the shared expert
    ek::rms(E->yl, El.lat_norm, E->ylr, X, g.norm_eps, s);
    gf::MvJob up = {El.lat_up.q, nullptr, E->ylr, F->ffn, F->sh_out, 1.0f, El.lat_up.type, X, g.n_embd};
    if (!gf::mv(&up, 1, s)) { err = "ernie: fc2"; return false; }
    if (E->dump) {
        er_dump(("yl-" + std::to_string(il)).c_str(), E->yl, (size_t) X, s);
        er_dump(("shx-" + std::to_string(il)).c_str(), F->sh_out, (size_t) g.n_embd, s);
    }
    if (!F->kda_graph_on) F->bg_drain(s, F->bg_pace, F->bg_attn_mask);
    if (ek::launch_errors() > 0) { err = "ernie: ffn kernels"; return false; }
    return true;
}

void Glm5Model::ernie_multimax(float* logits, cudaStream_t s) {
    const Ernie* E = fast_ != nullptr ? fast_->ernie : nullptr;
    if (E == nullptr || E->mm_r == nullptr) return;
    ek::multimax(logits, g_.n_vocab, E->mm_r, E->mm_t, s);
}

}  // namespace strata::core
