// src/core/raon_fast_path.cu - KRAFTON Raon-1 (GGUF arch raon1) on the qwen35moe fast path (Glm5Model members).  The
// model's math (its own modeling code; the GGUF's tensors as the loader maps them - recipes/raon1's mapcheck):
//   the residual: 4 streams, the embedding copied into each
//   a site (attention, then FFN): Raon-mHC (kernels/raon_fast.hpp) - the branch reads rms(sum h_pre[n] R[n]) * norm,
//   its output goes back as R[i] = sum_j H[i][j] R[j] + h_post[i] * out
//   RDA layer   q, k, v = silu(conv4(W x)) (q, k L2-normed); the decay g1 = lb * sigmoid(exp(A_log) * (f_b f_a x + dt));
//               beta = sigmoid(b x), v *= sigmoid(b x + w x) / sigmoid(b x) (w: ssm_rda); the gated delta rule (glm_fast's
//               kda_rec) with the output rms(o) * o_norm * sigmoid(g x); attn_output
//   MLA layer   q = q_b rms(q_a x) [heads][128 nope | 64 pe]; kv = kv_a x [512 latent | 64 pe], the latent rms-normed; no
//               rotary (the pe dims as they are: an identity table into ling_fast's mla_prep); attention over every
//               position in the latent space, scale 1/sqrt(192); o[h][v] = (v_b ctx_h)[v] * sigmoid(attn_gate x)[h, v]
//   FFN         layer 0 a SITU-GLU of n_ff_dense; then the MoE (fast_moe: the sigmoid router + exp_probs_b, normalised,
//               a SITU shared expert, SITU experts) + raon_alpha * the steering vector on raon_steer's layers
//   head        rms(mean of the streams) * output_norm (glm_fast's head_prep)
// The state rides qwen35_alloc as ling's does: the RDA state / conv rows, an MLA layer's cache rows of 576 FP16.
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_batch.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/ling_fast.hpp"
#include "strata/kernels/raon_fast.hpp"
#include "strata/qtypes.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace lk = strata::kernels::ling;
namespace rk = strata::kernels::raon;

namespace {
size_t al4(size_t floats) { return (floats + 63u) & ~(size_t) 63u; }   // (256-byte steps)
constexpr int kFaRank = 128;
}  // namespace

void Glm5Model::raon_bind(int il, std::string& missing) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    auto& Ly = F->L[(size_t) il];
    const std::string P = "blk." + std::to_string(il) + ".";
    const auto f32 = [&](const char* n) -> const float* {
        auto it = w_.find(P + n);
        if (it == w_.end()) { missing += P + n + " (f32) "; return nullptr; }
        return it->second;
    };
    const auto q = [&](const char* n, int64_t n_in, int64_t n_out) -> WSlot {
        const auto it = ws_map_.find(P + n);
        if (it == ws_map_.end() || it->second.type == 0) {
            if (const auto b = w16_.find(P + n); b != w16_.end()) return WSlot{nullptr, gf::kTypeBF16, b->second, n_in, n_out};
            if (const auto f = w_.find(P + n); f != w_.end()) return WSlot{f->second, gf::kTypeF32, f->second, n_in, n_out};
            missing += P + n + " (quant) ";
            return WSlot{};
        }
        if (!gf::mv_supported(it->second.type)) {
            missing += P + n + " (type " + std::to_string(it->second.type) + ") ";
            return WSlot{};
        }
        if ((it->second.n_in != 0 && it->second.n_in != n_in) || (it->second.n_out != 0 && it->second.n_out != n_out))
            missing += P + n + " (shape " + std::to_string(it->second.n_in) + "x" + std::to_string(it->second.n_out) +
                       ", expected " + std::to_string(n_in) + "x" + std::to_string(n_out) + ") ";
        return it->second;
    };
    // a 3-D absorbed projection (k_b / v_b): decoded once to F32 rows on the device
    const auto rows32 = [&](const char* n, int64_t n_in, int64_t n_out) -> const float* {
        const size_t count = (size_t) n_in * (size_t) n_out;
        std::vector<float> host;
        if (const auto f = w_.find(P + n); f != w_.end()) return f->second;
        if (const auto it = ws_map_.find(P + n); it != ws_map_.end() && it->second.type != 0) {
            const int t = it->second.type;
            const strata::qt::Traits* tt = strata::qt::traits(t);
            const size_t rb = strata::qt::row_size(t, n_in);
            if (tt != nullptr && tt->to_float != nullptr && rb > 0) {
                std::vector<uint8_t> bytes(rb * (size_t) n_out);
                if (cudaMemcpy(bytes.data(), it->second.q, bytes.size(), cudaMemcpyDeviceToHost) == cudaSuccess) {
                    host.resize(count);
                    for (int64_t r = 0; r < n_out; ++r)
                        tt->to_float(bytes.data() + (size_t) r * rb, host.data() + (size_t) r * n_in, n_in);
                }
            }
        } else if (const auto b = w16_.find(P + n); b != w16_.end()) {
            std::vector<uint16_t> h16(count);
            if (cudaMemcpy(h16.data(), b->second, count * 2, cudaMemcpyDeviceToHost) == cudaSuccess) {
                host.resize(count);
                for (size_t i = 0; i < count; ++i) host[i] = strata::qt::bf16_to_fp32(h16[i]);
            }
        }
        void* d = nullptr;
        if (!host.empty() && cudaMalloc(&d, count * sizeof(float)) == cudaSuccess) raon_owned_.push_back(d);
        if (host.empty() || d == nullptr ||
            cudaMemcpy(d, host.data(), count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaGetLastError();
            missing += P + n + " (" + std::to_string(n_in) + "x" + std::to_string(n_out) + ", a type with a host decoder) ";
            return nullptr;
        }
        return (const float*) d;
    };
    const int E = g.n_embd, H = g.n_head, DI = g.d_inner();
    Ly.attn_norm = f32("attn_norm.weight");
    Ly.post_norm = f32("ffn_norm.weight");
    const char* site[2] = {"hc_attn_", "hc_ffn_"};
    for (int k = 0; k < 2; ++k) {
        const std::string b = site[k];
        const auto it = w16_.find(P + b + "fn.weight");
        if (it == w16_.end()) missing += P + b + "fn.weight (BF16) ";
        else Ly.rn_hc_fn[k] = it->second;
        Ly.rn_hc_base[k] = f32((b + "base.weight").c_str());
        Ly.rn_hc_scale[k] = f32((b + "scale.weight").c_str());
    }
    if (il < (int) g.raon_steer.size() && g.raon_steer[(size_t) il]) Ly.rn_steer = f32("ffn_effort_steer.weight");
    if (Ly.recr) {
        Ly.q = q("attn_q.weight", E, DI);
        Ly.k = q("attn_k.weight", E, DI);
        Ly.v = q("attn_v.weight", E, DI);
        Ly.lg_g = q("ssm_g.weight", E, DI);
        Ly.lg_beta = q("ssm_beta.weight", E, H);
        Ly.rn_rda = q("ssm_rda.weight", E, H);
        Ly.rn_fa = q("ssm_f_a.weight", E, kFaRank);
        Ly.rn_fb = q("ssm_f_b.weight", kFaRank, DI);
        Ly.conv[0] = f32("ssm_conv1d_q.weight");
        Ly.conv[1] = f32("ssm_conv1d_k.weight");
        Ly.conv[2] = f32("ssm_conv1d_v.weight");
        Ly.dt_bias = f32("ssm_dt.bias");
        Ly.ssm_a = f32("ssm_a");   // (-exp(A_log): GLM's sign, for glm_batch's recurrence)
        // (ling_fast's kda_prep takes +exp(A_log): a negated copy)
        if (Ly.ssm_a != nullptr) {
            std::vector<float> a((size_t) H);
            void* d = nullptr;
            if (cudaMemcpy(a.data(), Ly.ssm_a, a.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMalloc(&d, a.size() * 4) == cudaSuccess) {
                raon_owned_.push_back(d);
                for (float& v : a) v = -v;
                if (cudaMemcpy(d, a.data(), a.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess) Ly.rn_apos = (const float*) d;
            }
            cudaGetLastError();
            if (Ly.rn_apos == nullptr) missing += P + "ssm_a (+exp copy) ";
        }
        Ly.ssm_norm = f32("ssm_norm.weight");
        Ly.out = q("attn_output.weight", DI, E);
    } else {
        Ly.rn_qa = q("attn_q_a.weight", E, g.q_lora);
        Ly.rn_qan = f32("attn_q_a_norm.weight");
        Ly.rn_qb = q("attn_q_b.weight", g.q_lora, (int64_t) H * lk::kQHead);
        Ly.lg_kva = q("attn_kv_a_mqa.weight", E, lk::kRow);
        Ly.lg_gate = q("attn_gate.weight", E, (int64_t) H * lk::kV);
        Ly.lg_kvn = f32("attn_kv_a_norm.weight");
        Ly.lg_kb = rows32("attn_k_b.weight", lk::kNope, (int64_t) lk::kLat * H);
        Ly.lg_vb = rows32("attn_v_b.weight", lk::kLat, (int64_t) lk::kV * H);
        Ly.out = q("attn_output.weight", (int64_t) H * lk::kV, E);
    }
}

// the scratch: f_a's output | rda's logits | q_a's output | its norm | the norm's q8_1 | the gate | the mHC z | the two
// sites' coefficients | the identity rotary table
namespace {
struct RaonCarve {
    size_t fa, rda, qa, qan, qanq, gate, z, coef0, coef1, cs, total;
};
RaonCarve raon_carve(const Glm5Geometry& g) {
    RaonCarve c{};
    size_t at = 0;
    const auto take = [&](size_t f) { const size_t o = at; at += al4(f); return o; };
    c.fa = take(kFaRank);
    c.rda = take((size_t) g.n_head);
    c.qa = take((size_t) g.q_lora);
    c.qan = take((size_t) g.q_lora);
    c.qanq = take((size_t) g.q_lora / 32 * 9);   // (36-byte blocks = 9 floats)
    c.gate = take((size_t) g.n_head * lk::kV);
    c.z = take(32);
    c.coef0 = take(32);
    c.coef1 = take(32);
    c.cs = take(64);
    c.total = at;
    return c;
}
}  // namespace

bool Glm5Model::raon_alloc(std::string& err) {
    const Glm5Geometry& g = g_;
    if (l0_ != 0 || l1_ != g.n_layers) {
        err = "raon: a layer split is not supported yet (one GPU holds every layer)";
        return false;
    }
    if (g.q_lora % 32 != 0) {
        err = "raon: q_lora_rank must be a multiple of 32";
        return false;
    }
    const int64_t ctx = std::max<int64_t>(max_ctx_, 1);
    const size_t sc = lk::mla_sc_floats(ctx, g.n_head), part = lk::mla_part_floats(ctx, g.n_head);
    const RaonCarve c = raon_carve(g);
    if (cudaMalloc((void**) &raon_sc_, sc * 4) != cudaSuccess || cudaMalloc((void**) &raon_part_, part * 4) != cudaSuccess ||
        cudaMalloc((void**) &raon_buf_, c.total * 4) != cudaSuccess) {
        cudaGetLastError();
        err = "raon: the attention scratch (" + std::to_string((sc + part) * 4 >> 20) + " MB) did not allocate";
        return false;
    }
    cudaMemset(raon_buf_, 0, c.total * 4);
    // the identity rotary table (cos 1, sin 0): the pe dims pass through mla_prep unrotated, exactly
    {
        float2 h[lk::kRot / 2];
        for (float2& v : h) v = make_float2(1.0f, 0.0f);
        if (cudaMemcpy(raon_buf_ + c.cs, h, sizeof h, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaGetLastError();
            err = "raon: the identity rotary table did not upload";
            return false;
        }
    }
    int n_mla = 0, n_steer = 0;
    for (int il = l0_; il < l1_; ++il) {
        n_mla += !fast_->L[(size_t) il].recr;
        n_steer += fast_->L[(size_t) il].rn_steer != nullptr;
    }
    raon_prompt({});
    std::fprintf(stderr, "raon: CUDA%d %d RDA layers, %d MLA layers (latent caches %.2f GB, attention scratch %.0f MB for "
                         "%lld tokens), 4 mHC streams, effort steering on %d layers (alpha %.3g)\n",
                 dev_, (l1_ - l0_) - n_mla, n_mla, (double) n_mla * (double) ctx * lk::kRow * 2 / 1073741824.0,
                 (double) (sc + part) * 4 / 1048576.0, (long long) ctx, n_steer, (double) raon_alpha_);
    return true;
}

bool Glm5Model::raon_prompt_coef(int T, std::string& err) {
    if (T <= raon_pc_rows_) return true;
    if (raon_pc_ != nullptr) {
        cudaStreamSynchronize(fast_->cs);
        cudaFree(raon_pc_);
        raon_pc_ = nullptr;
        raon_pc_rows_ = 0;
    }
    if (cudaMalloc((void**) &raon_pc_, (size_t) T * (24 + 24 + rk::kZRow) * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        raon_pc_ = nullptr;
        err = "raon: the prompt's mHC coefficients (" + std::to_string(T) + " rows) did not allocate";
        return false;
    }
    raon_pc_rows_ = T;
    return true;
}

void Glm5Model::raon_prompt_read(int il, int site, int T, float* R, float* x, uint16_t* x16, cudaStream_t s) {
    const Glm5Geometry& g = g_;
    const auto& Ly = fast_->L[(size_t) il];
    rk::mhc_pre(R, Ly.rn_hc_fn[site], Ly.rn_hc_base[site], Ly.rn_hc_scale[site], site == 0 ? Ly.attn_norm : Ly.post_norm,
                g.raon_hc_eps, g.norm_eps, g.raon_hc_post, g.n_embd, raon_pc_ + (size_t) 48 * T,
                raon_pc_ + (size_t) (site == 0 ? 0 : 24) * T, x, s, T);
    strata::kernels::glmb::f32_to_f16(x, x16, (int64_t) T * g.n_embd, s);
}

void Glm5Model::raon_free() {
    if (raon_pc_ != nullptr) cudaFree(raon_pc_);
    raon_pc_ = nullptr;
    raon_pc_rows_ = 0;
    for (void* p : raon_owned_) cudaFree(p);
    raon_owned_.clear();
    if (raon_sc_ != nullptr) cudaFree(raon_sc_);
    if (raon_part_ != nullptr) cudaFree(raon_part_);
    if (raon_buf_ != nullptr) cudaFree(raon_buf_);
    raon_sc_ = raon_part_ = raon_buf_ = nullptr;
}

// STRATA_RAON_EFFORT: a name (low / medium / high / xhigh / max, as the GGUF lists them; none = thinking off's 0.1),
// a number, or auto (unset): the model code's rule - thinking off (<think></think> adjacent in the prompt) 0.1, else the
// default effort's alpha
void Glm5Model::raon_prompt(const std::vector<int64_t>& ids, const std::string& effort) {
    const Glm5Geometry& g = g_;
    if (!g.raon) return;
    constexpr float kNoThinking = 0.1f;
    const char* v = getenv("STRATA_RAON_EFFORT");
    const std::string e = !effort.empty() ? effort : v != nullptr ? v : "";
    if (!e.empty() && e != "auto") {
        if (e == "none") { raon_alpha_ = kNoThinking; return; }
        for (size_t i = 0; i < g.raon_effort_names.size() && i < g.raon_effort_alpha.size(); ++i)
            if (g.raon_effort_names[i] == e) { raon_alpha_ = g.raon_effort_alpha[i]; return; }
        char* end = nullptr;
        const double a = std::strtod(e.c_str(), &end);
        if (end != e.c_str() && *end == '\0') { raon_alpha_ = (float) a; return; }
        static bool warned = false;
        if (!warned) std::fprintf(stderr, "raon: effort %s is not an effort name or a number - auto\n", e.c_str());
        warned = true;
    }
    // (the tokenizer's <think> / </think>: found by text in the vocabulary would need the tokenizer here - the GGUF's
    // ids, Raon-1's 259213 / 259214, are checked against the vocabulary size)
    constexpr int64_t kThink = 259213, kThinkEnd = 259214;
    // (the LAST <think>: the generation prompt's - a past turn without reasoning is <think></think> too)
    bool off = false;
    if (g.n_vocab > kThinkEnd)
        for (size_t i = ids.size(); i-- > 0;)
            if (ids[i] == kThink) { off = i + 1 < ids.size() && ids[i + 1] == kThinkEnd; break; }
    raon_alpha_ = off ? kNoThinking : g.raon_alpha;
}

static void raon_dump(cudaStream_t s, const char* name, int il, const float* dev, int64_t n) {
    static const char* dir = getenv("GLM_CB_DIR");
    if (dir == nullptr) return;
    cudaStreamSynchronize(s);
    std::vector<float> h((size_t) n);
    cudaMemcpy(h.data(), dev, (size_t) n * 4, cudaMemcpyDeviceToHost);
    const std::string p = std::string(dir) + "/" + name + "-" + std::to_string(il) + ".bin";
    if (FILE* f = std::fopen(p.c_str(), "ab")) {
        std::fwrite(h.data(), 4, h.size(), f);
        std::fclose(f);
    }
}

bool Glm5Model::raon_mhc(int il, int site, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const RaonCarve c = raon_carve(g);
    const float* nw = site == 0 ? Ly.attn_norm : Ly.post_norm;
    rk::mhc_pre(state_, Ly.rn_hc_fn[site], Ly.rn_hc_base[site], Ly.rn_hc_scale[site], nw, g.raon_hc_eps, g.norm_eps,
                g.raon_hc_post, g.n_embd, raon_buf_ + c.z, raon_buf_ + (site == 0 ? c.coef0 : c.coef1), F->x, F->cs);
    gf::quantize_q8_1(F->x, F->xq, g.n_embd, F->cs);
    if (rk::launch_errors() != 0) { err = "raon: an mHC kernel launch failed"; return false; }
    return true;
}

bool Glm5Model::fast_raon_rda(int il, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, H = g.n_head, DI = g.d_inner();
    const RaonCarve c = raon_carve(g);
    float* proj = F->q35_qkv;   // q | k | v
    float* fa = raon_buf_ + c.fa;
    float* wr = raon_buf_ + c.rda;
    gf::MvJob j[7];
    j[0] = {Ly.q.q, F->xq, F->x, proj, nullptr, 1.0f, Ly.q.type, E, DI};
    j[1] = {Ly.k.q, F->xq, F->x, proj + DI, nullptr, 1.0f, Ly.k.type, E, DI};
    j[2] = {Ly.v.q, F->xq, F->x, proj + 2 * DI, nullptr, 1.0f, Ly.v.type, E, DI};
    j[3] = {Ly.lg_g.q, F->xq, F->x, F->q35_o, nullptr, 1.0f, Ly.lg_g.type, E, DI};
    j[4] = {Ly.lg_beta.q, F->xq, F->x, F->q35_gate, nullptr, 1.0f, Ly.lg_beta.type, E, H};
    j[5] = {Ly.rn_rda.q, F->xq, F->x, wr, nullptr, 1.0f, Ly.rn_rda.type, E, H};
    j[6] = {Ly.rn_fa.q, F->xq, F->x, fa, nullptr, 1.0f, Ly.rn_fa.type, E, kFaRank};
    if (!gf::mv(j, 7, s)) { err = "raon: the RDA projections"; return false; }
    gf::MvJob fb = {Ly.rn_fb.q, nullptr, fa, F->q35_z, nullptr, 1.0f, Ly.rn_fb.type, kFaRank, DI};
    if (!gf::mv(&fb, 1, s)) { err = "raon: the RDA decay's f_b"; return false; }
    if (F->prof_on) F->mark("raon_rda_mv");
    const float* cw[3] = {Ly.conv[0], Ly.conv[1], Ly.conv[2]};
    lk::kda_prep(proj, cw, Ly.gdn_conv, F->q35_h, F->q35_z, Ly.dt_bias, Ly.rn_apos, g.kda_lb, F->q35_y, H, g.kda_head_dim,
                 g.d_conv, s);
    rk::rda_ratio(F->q35_h + 2 * DI, F->q35_gate, wr, H, g.kda_head_dim, s);
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    if (dumps) {
        raon_dump(s, "rda_q", il, F->q35_h, DI);
        raon_dump(s, "rda_k", il, F->q35_h + DI, DI);
        raon_dump(s, "rda_v", il, F->q35_h + 2 * DI, DI);
        raon_dump(s, "rda_decay", il, F->q35_y, DI);
    }
    gf::kda_rec(F->q35_h, F->q35_h + DI, F->q35_h + 2 * DI, F->q35_y, F->q35_gate, Ly.gdn_state, F->q35_o, Ly.ssm_norm,
                g.norm_eps, H, g.kda_head_dim, F->q35_yq, s);
    if (F->prof_on) F->mark("raon_rda_rec");
    gf::MvJob o = {Ly.out.q, F->q35_yq, nullptr, F->mixer, nullptr, 1.0f, Ly.out.type, DI, E};
    if (!gf::mv(&o, 1, s)) { err = "raon: the RDA output"; return false; }
    if (F->prof_on) F->mark("raon_rda_out");
    if (dumps) raon_dump(s, "rda_out", il, F->mixer, E);
    return true;
}

bool Glm5Model::fast_raon_mla(int il, int64_t p, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, H = g.n_head, HV = H * lk::kV, QL = g.q_lora;
    const RaonCarve c = raon_carve(g);
    float* qa = raon_buf_ + c.qa;
    float* qan = raon_buf_ + c.qan;
    void* qanq = (void*) (raon_buf_ + c.qanq);
    float* gate = raon_buf_ + c.gate;
    gf::MvJob j[3];
    j[0] = {Ly.rn_qa.q, F->xq, F->x, qa, nullptr, 1.0f, Ly.rn_qa.type, E, QL};
    j[1] = {Ly.lg_kva.q, F->xq, F->x, F->q35_k, nullptr, 1.0f, Ly.lg_kva.type, E, lk::kRow};
    j[2] = {Ly.lg_gate.q, F->xq, F->x, gate, nullptr, 1.0f, Ly.lg_gate.type, E, HV};
    if (!gf::mv(j, 3, s)) { err = "raon: the MLA projections"; return false; }
    gf::rms_q8(qa, nullptr, Ly.rn_qan, g.raon_lat_eps, QL, qan, qanq, s);
    gf::MvJob qb = {Ly.rn_qb.q, qanq, qan, F->q35_qf, nullptr, 1.0f, Ly.rn_qb.type, QL, H * lk::kQHead};
    if (!gf::mv(&qb, 1, s)) { err = "raon: the MLA q_b"; return false; }
    if (F->prof_on) F->mark("raon_mla_mv");
    const float2* cs = (const float2*) (raon_buf_ + c.cs);
    lk::mla_prep(F->q35_qf, H, F->q35_k, Ly.lg_kvn, g.raon_lat_eps, cs, Ly.kc + (size_t) p * lk::kRow, s);
    lk::mla_absorb(F->q35_qf, Ly.lg_kb, H, F->q35_q, s);
    lk::mla_attend(F->q35_q, F->q35_qf, Ly.kc, p + 1, 1.0f / std::sqrt((float) lk::kQHead), H, raon_sc_, raon_part_,
                   F->q35_ao, s);
    rk::mla_out_gated(F->q35_ao, Ly.lg_vb, gate, H, F->q35_o, s);
    if (F->prof_on) F->mark("raon_mla_attend");
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    if (dumps) raon_dump(s, "mla_gated", il, F->q35_o, HV);
    gf::quantize_q8_1(F->q35_o, F->q35_aoq, HV, s);
    gf::MvJob o = {Ly.out.q, F->q35_aoq, F->q35_o, F->mixer, nullptr, 1.0f, Ly.out.type, HV, E};
    if (!gf::mv(&o, 1, s)) { err = "raon: the MLA output"; return false; }
    if (F->prof_on) F->mark("raon_mla_out");
    if (dumps) raon_dump(s, "mla_out", il, F->mixer, E);
    if (lk::launch_errors() != 0 || rk::launch_errors() != 0) { err = "raon: a kernel launch failed"; return false; }
    return true;
}

bool Glm5Model::fast_raon_layers(int64_t p, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    bool pf_pending = false;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    const int E = g.n_embd;
    const RaonCarve c = raon_carve(g);
    if (F->prof_on) F->mark("start");
    if (dumps) raon_dump(s, "inp_streams", 0, state_, 4 * E);
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (!raon_mhc(il, 0, err)) return false;
        if (F->prof_on) F->mark("raon_mhc");
        if (dumps) raon_dump(s, "attn_norm", il, F->x, E);
        if (!(Ly.recr ? fast_raon_rda(il, err) : fast_raon_mla(il, p, err))) return false;
        rk::mhc_post(state_, F->mixer, raon_buf_ + c.coef0, E, s);
        if (!raon_mhc(il, 1, err)) return false;
        if (F->prof_on) F->mark("raon_mhc");
        if (dumps) raon_dump(s, "ffn_norm", il, F->x, E);
        if (!Ly.moe) {
            if (!fast_dense_ffn(il, err)) return false;
        } else if (!fast_moe(il, pf_pending, err)) {
            return false;
        }
        if (Ly.rn_steer != nullptr && raon_alpha_ != 0.0f) rk::axpy(F->ffn, Ly.rn_steer, raon_alpha_, E, s);
        if (dumps) raon_dump(s, "ffn_out", il, F->ffn, E);
        rk::mhc_post(state_, F->ffn, raon_buf_ + c.coef1, E, s);
        if (dumps) raon_dump(s, "l_out", il, state_, 4 * E);
    }
    const auto nw = w_.find("output_norm.weight");
    if (nw == w_.end()) { err = "raon: output_norm.weight"; return false; }
    gf::head_prep(state_, nw->second, g.norm_eps, E, F->head_x, F->head_xq, s);
    if (F->prof_on) F->mark("raon_head");
    if (dumps) raon_dump(s, "result_norm", 0, F->head_x, E);
    if (rk::launch_errors() != 0) { err = "raon: a kernel launch failed"; return false; }
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { err = std::string("raon: ") + cudaGetErrorString(e); return false; } }
    return true;
}

}  // namespace strata::core
