// src/core/giga_fast_path.cu - ai-sage GigaChat4 (GGUF arch gigachat4) on the qwen35moe fast path (Glm5Model
// members): the one-token decode.  The math is the model's own modeling_gigachat4.py (the GGUF keeps the model's
// tensors as they are: its value-head order, the MLA alphas outside the weights - the float64 reference
// recipes/giga4/ref.py settled both on the file):
//   the residual: one stream R (state_), plain adds; every norm a GatedNorm GN (kernels/giga_fast.hpp):
//               y = rms(x) * w, GN(x) = y * giga_gate_w * sigmoid(P_out silu(P_in y))
//   a layer     x = GN_attn(R); m = the mixer(x); R += GN_post_attention(m)
//               x = GN_ffn(R);  f = the FFN(x);   R += GN_post_ffw(f)
//   GDN layer   qwen35's gated delta-net (qkv conv4 + SiLU, q / k L2-normed, the decay softplus(alpha + dt) * ssm_a,
//               beta's sigmoid, the recurrence qwen35_fast's gdn_scan) with value head v on key head v / 2 (the
//               model's repeat_interleave; qwen35 GGUFs are reordered to v % 16) and the closing norm's gate
//               giga_gdn_gate * sigmoid(z); ssm_out
//   MLA layer   q = q_b (alpha_q GN_qa(q_a x)) [heads][128 nope | 64 rope]; kv = kv_a x [512 latent | 64 rope], the
//               latent alpha_kv GN_kva(.), alpha = sqrt(n_embd / rank) (use_mla_scaling_factor); the rope dims roped
//               (interleaved pairs, rope_base with YaRN's blended frequencies); attention in the latent space (q_abs =
//               k_b q_nope), scale 1/sqrt(192) x the YaRN mscale^2; o[h][v] = (v_b ctx_h)[v] * sigmoid(attn_gate x)[h, v];
//               attn_output
//   DSA         the MLA layer's indexer: a key rope(rms(idx_k x) * k_norm) [128] a position, the query idx_q_b of the
//               alpha_q-scaled q latent [64][128], the head weights idx_proj x; score(s) = sum_h w_h relu(iq_h . key_s)
//               (the model's positive factors left out: the order alone is read); up to idx_top_k positions every one
//               is attended (no scores computed), past that the idx_top_k best
//   FFN         layer 0 dense (fast_dense_ffn), then the MoE (fast_moe: the sigmoid router + exp_probs_b, normalised,
//               x w_scale; the ungated shared expert; the SwiGLU limit)
//   head        GN_output(R) -> head_x / head_xq (fast_token's output rows, then the logit softcap)
// The state rides qwen35_alloc: the delta-net state / conv rows, an MLA layer's cache rows of 576 FP16 (kc); the
// indexer's keys (FP16 [128] a position and MLA layer) live in giga_buf_ - they are NOT in the qwen35 snapshots (slots
// stay off for gigachat4 until they are).
// Weights: giga_bind.cu.  Dumps (GLM_CB_DIR): inp_embd-0, l_out-<il> (R after the layer), attn_norm / mixer /
// attn_post / ffn_norm / ffn_out / ffn_post per layer, result_norm-0 - a position's floats appended a call
// (GLM_CB_FROM=<position>: from that position on; past idx_top_k also dsa_score-<il> / dsa_sel-<il>).
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/giga_fast.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/ling_fast.hpp"
#include "strata/kernels/qwen35_fast.hpp"
#include "strata/kernels/raon_fast.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace lk = strata::kernels::ling;
namespace rk = strata::kernels::raon;
namespace gk = strata::kernels::giga;
namespace q35 = strata::kernels::q35;

namespace {
size_t al4(size_t floats) { return (floats + 63u) & ~(size_t) 63u; }   // (256-byte steps)

constexpr int kNormForm = 6;   // (giga_fast.hpp: the gate reads rms(x) * w, SiLU between the projections, gate_w * sigmoid)

// the scratch: q_a's output | its norm | the norm's q8_1 | the MLA gate | the normed latent | the rotary table | the
// pair frequencies (doubles) | the delta-net's expanded q / k / v | the indexer's raw key, head weights, query, scores,
// chosen positions, their gathered cache rows | the indexer's keys (n_mla x ctx rows of 128 FP16)
struct GigaCarve {
    size_t qa, qan, qanq, gate, lat, cs, freq, exp, ikraw, iw, iq, score, sel, comp, keys, key_layer, total;
};
GigaCarve giga_carve(const Glm5Geometry& g, int64_t ctx) {
    GigaCarve c{};
    size_t at = 0;
    const auto take = [&](size_t f) { const size_t o = at; at += al4(f); return o; };
    c.qa = take((size_t) g.q_lora);
    c.qan = take((size_t) g.q_lora);
    c.qanq = take((size_t) g.q_lora / 32 * 9);   // (36-byte blocks = 9 floats)
    c.gate = take((size_t) g.n_head * (size_t) g.ling_v);
    c.lat = take((size_t) g.kv_lora);
    c.cs = take(64);
    c.freq = take(64);
    c.exp = take((size_t) 3 * g.ssm_inner);
    c.ikraw = take((size_t) g.idx_key);
    c.iw = take((size_t) g.idx_heads);
    c.iq = take((size_t) g.idx_heads * (size_t) g.idx_key);
    c.score = take((size_t) ctx);
    c.sel = take((size_t) g.idx_top_k);
    c.comp = take((size_t) g.idx_top_k * gk::kRow / 2);
    int n_mla = 0;
    for (int il = 0; il < g.n_layers; ++il) n_mla += g.n_head_kv[(size_t) il] != 0;
    c.key_layer = al4((size_t) ctx * (size_t) g.idx_key / 2);   // (a layer's keys, in floats)
    c.keys = take(c.key_layer * (size_t) n_mla);
    c.total = at;
    return c;
}

// (GLM_CB_FROM=<position>: the dumps start there - a long prompt's last positions alone)
int64_t giga_dump_pos = 0;
void giga_dump(cudaStream_t s, const char* name, int il, const float* dev, int64_t n) {
    static const char* dir = getenv("GLM_CB_DIR");
    static const int64_t from = getenv("GLM_CB_FROM") != nullptr ? std::atoll(getenv("GLM_CB_FROM")) : 0;
    if (dir == nullptr || giga_dump_pos < from) return;
    cudaStreamSynchronize(s);
    std::vector<float> h((size_t) n);
    cudaMemcpy(h.data(), dev, (size_t) n * 4, cudaMemcpyDeviceToHost);
    const std::string p = std::string(dir) + "/" + name + "-" + std::to_string(il) + ".bin";
    if (FILE* f = std::fopen(p.c_str(), "ab")) {
        std::fwrite(h.data(), 4, h.size(), f);
        std::fclose(f);
    }
}

// the YaRN factor on the attention's logits / cos, sin: yarn_log_mul * ln(factor) + 1
float yarn_mscale(const Glm5Geometry& g) {
    return g.yarn_factor > 1.0f ? g.yarn_log_mul * std::log(g.yarn_factor) + 1.0f : 1.0f;
}
}  // namespace

bool Glm5Model::giga_alloc(std::string& err) {
    const Glm5Geometry& g = g_;
    if (l0_ != 0 || l1_ != g.n_layers) {
        err = "gigachat4: a layer split is not supported yet (one GPU holds every layer)";
        return false;
    }
    if (g.q_lora % 32 != 0 || g.kv_lora != gk::kLat || g.ling_rope != gk::kRot || g.ling_qk_nope != gk::kNope ||
        g.ling_v != lk::kV || g.giga_rank > gk::kRankMax || g.ssm_state != 128 || g.idx_key != gk::kIdxKey ||
        g.idx_heads != 64 || g.idx_top_k < 1 || g.ssm_v_heads % g.ssm_k_heads != 0) {
        err = "gigachat4: the kernels take q_lora a multiple of 32, kv_lora 512, rope 64, qk_nope 128, v 128, a norm gate "
              "rank <= 16, ssm state 128, 64 indexer heads of 128";
        return false;
    }
    std::string missing;
    if (!giga_bind_head(missing)) {
        err = "gigachat4: the head's GatedNorm is missing: " + missing;
        return false;
    }
    const int64_t ctx = std::max<int64_t>(max_ctx_, 1);
    const size_t sc = lk::mla_sc_floats(ctx, g.n_head), part = lk::mla_part_floats(ctx, g.n_head);
    const GigaCarve c = giga_carve(g, ctx);
    void *b = nullptr, *d_sc = nullptr, *d_part = nullptr;
    const bool ok = cudaMalloc(&b, c.total * 4) == cudaSuccess && cudaMalloc(&d_sc, sc * 4) == cudaSuccess &&
                    cudaMalloc(&d_part, part * 4) == cudaSuccess;
    for (void* p : {b, d_sc, d_part})
        if (p != nullptr) giga_owned_.push_back(p);   // (giga_free's)
    if (!ok) {
        cudaGetLastError();
        err = "gigachat4: the attention scratch and the indexer's keys (" + std::to_string((sc + part + c.total) * 4 >> 20) +
              " MB) did not allocate";
        return false;
    }
    giga_buf_ = (float*) b;
    giga_sc_ = (float*) d_sc;
    giga_part_ = (float*) d_part;
    cudaMemset(giga_buf_, 0, c.total * 4);
    // the rope pairs' frequencies (the model's YaRN rotary: the interpolated and the plain frequencies blended by a
    // linear ramp between the correction dims of beta_fast and beta_slow)
    {
        const int dim = gk::kRot, half = dim / 2;
        const double base = (double) g.rope_base, factor = (double) g.yarn_factor, kPi = 3.14159265358979323846;
        double f[gk::kRot / 2];
        const auto corr = [&](double n_rot) {
            return (double) dim * std::log((double) g.yarn_orig_ctx / (n_rot * 2.0 * kPi)) / (2.0 * std::log(base));
        };
        double low = std::max(std::floor(corr((double) g.yarn_beta_fast)), 0.0);
        double high = std::min(std::ceil(corr((double) g.yarn_beta_slow)), (double) (dim - 1));
        if (low == high) high += 0.001;
        for (int j = 0; j < half; ++j) {
            const double extra = std::pow(base, -2.0 * (double) j / (double) dim);
            if (factor <= 1.0) {
                f[j] = extra;
                continue;
            }
            const double ramp = std::min(1.0, std::max(0.0, ((double) j - low) / (high - low)));
            const double mask = 1.0 - ramp;   // (1: the plain frequency)
            f[j] = extra / factor * (1.0 - mask) + extra * mask;
        }
        if (cudaMemcpy(giga_buf_ + c.freq, f, sizeof f, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaGetLastError();
            err = "gigachat4: the rope frequencies did not upload";
            return false;
        }
    }
    int n_mla = 0;
    for (int il = l0_; il < l1_; ++il) n_mla += !fast_->L[(size_t) il].recr;
    std::fprintf(stderr, "gigachat4: CUDA%d %d GDN layers, %d MLA layers (latent caches %.2f GB, the indexer's keys %.2f GB, "
                         "attention scratch %.0f MB for %lld tokens; every position attended up to %d, then the "
                         "indexer's %d best)\n",
                 dev_, (l1_ - l0_) - n_mla, n_mla, (double) n_mla * (double) ctx * gk::kRow * 2 / 1073741824.0,
                 (double) c.key_layer * n_mla * 4 / 1073741824.0, (double) (sc + part) * 4 / 1048576.0, (long long) ctx,
                 g.idx_top_k, g.idx_top_k);
    return true;
}

// (the prompt path's - giga_prefill.cu - view of the carve: an MLA layer's indexer keys, the rope frequencies)
uint16_t* Glm5Model::giga_keys(int mla) const {
    const GigaCarve c = giga_carve(g_, std::max<int64_t>(max_ctx_, 1));
    return (uint16_t*) (giga_buf_ + c.keys + (size_t) mla * c.key_layer);
}
const double* Glm5Model::giga_freq() const {
    return (const double*) (giga_buf_ + giga_carve(g_, std::max<int64_t>(max_ctx_, 1)).freq);
}

namespace {
// GN over n floats: h += add, y = GN(h)
template <class GN>
inline void gnorm(const GN& n, const Glm5Geometry& g, float* h, const float* add, int len,
                  float* y, cudaStream_t s) {
    gk::gated_norm(h, add, n.w, n.up /* n -> rank */, n.down /* rank -> n */, g.giga_gate_w, g.norm_eps, len, g.giga_rank, kNormForm, y, s);
}
}  // namespace

// the gated delta-net layer: F->x / xq -> F->mixer
template <class FS>
static bool giga_gdn(FS* F, const Glm5Geometry& g, int il, float* buf, const GigaCarve& c, std::string& err) {
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, C = g.ssm_conv_dim(), V = g.ssm_inner, S = g.ssm_state, HK = g.ssm_k_heads, HV = g.ssm_v_heads;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    gf::MvJob j[4];
    j[0] = {Ly.qkv.q, F->xq, F->x, F->q35_qkv, nullptr, 1.0f, Ly.qkv.type, E, C};
    j[1] = {Ly.zg.q, F->xq, F->x, F->q35_z, nullptr, 1.0f, Ly.zg.type, E, V};
    j[2] = {Ly.alpha_w.q, F->xq, F->x, F->q35_ab, nullptr, 1.0f, Ly.alpha_w.type, E, HV};
    j[3] = {Ly.beta_w.q, F->xq, F->x, F->q35_ab + HV, nullptr, 1.0f, Ly.beta_w.type, E, HV};
    if (!gf::mv(j, 4, s)) { err = "gigachat4: the delta-net projections"; return false; }
    if (F->prof_on) F->mark("giga_gdn_mv");
    if (dumps) {
        giga_dump(s, "gdn_qkv", il, F->q35_qkv, C);
        giga_dump(s, "gdn_z", il, F->q35_z, V);
        giga_dump(s, "gdn_alpha", il, F->q35_ab, HV);
        giga_dump(s, "gdn_beta", il, F->q35_ab + HV, HV);
    }
    // the conv over [state | x] + SiLU; q / k L2-normed, q / sqrt(S); the gates over alpha / beta
    q35::conv_silu_step(Ly.gdn_conv, F->q35_qkv, Ly.conv1d, C, g.d_conv, F->q35_h, s);
    q35::gdn_prep_gates(F->q35_h, 1, C, HK, S, g.norm_eps, F->q35_ab, F->q35_ab + HV, Ly.dt_bias, Ly.ssm_a, HV, s);
    if (dumps) {
        giga_dump(s, "gdn_conv", il, F->q35_h, C);
        giga_dump(s, "gdn_decay", il, F->q35_ab, HV);
        giga_dump(s, "gdn_beta_sigmoid", il, F->q35_ab + HV, HV);
    }
    if (HK != HV) {   // (value head v on key head v / (HV / HK): q / k written once a value head)
        gk::gdn_expand(F->q35_h, buf + c.exp, HK, HV, S, s);
        q35::gdn_scan(Ly.gdn_state, buf + c.exp, 1, 3 * V, F->q35_ab, F->q35_ab + HV, S, HV, HV, F->q35_o, s);
    } else {
        q35::gdn_scan(Ly.gdn_state, F->q35_h, 1, C, F->q35_ab, F->q35_ab + HV, S, HK, HV, F->q35_o, s);
    }
    if (dumps) giga_dump(s, "gdn_o", il, F->q35_o, V);
    gk::gdn_out(F->q35_o, F->q35_z, Ly.ssm_norm, F->q35_y, HV, S, g.norm_eps,
                g.giga_gdn_gate, false, s);
    if (F->prof_on) F->mark("giga_gdn");
    if (dumps) giga_dump(s, "gdn_y", il, F->q35_y, V);
    gf::quantize_q8_1(F->q35_y, F->q35_yq, V, s);
    gf::MvJob o = {Ly.ssm_o.q, F->q35_yq, F->q35_y, F->mixer, nullptr, 1.0f, Ly.ssm_o.type, V, E};
    if (!gf::mv(&o, 1, s)) { err = "gigachat4: ssm_out"; return false; }
    if (F->prof_on) F->mark("giga_gdn_out");
    return true;
}

// the MLA layer at position p: F->x / xq -> F->mixer.  mla: the layer's place among the MLA layers (its indexer keys)
template <class FS>
static bool giga_mla(FS* F, const Glm5Geometry& g, int il, int mla, int64_t p, float* buf, const GigaCarve& c, float* sc,
                     float* part, std::string& err) {
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, H = g.n_head, HV = H * lk::kV, QL = g.q_lora, IH = g.idx_heads, K = g.idx_top_k;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    const bool pick = p + 1 > (int64_t) K;   // (more positions than the indexer keeps)
    float* qa = buf + c.qa;
    float* qan = buf + c.qan;
    void* qanq = (void*) (buf + c.qanq);
    float* gate = buf + c.gate;
    float* lat = buf + c.lat;
    float2* cs = (float2*) (buf + c.cs);
    float* ikraw = buf + c.ikraw;
    float* iw = buf + c.iw;
    float* iq = buf + c.iq;
    uint16_t* keys = (uint16_t*) (buf + c.keys + (size_t) mla * c.key_layer);
    gf::MvJob j[5];
    j[0] = {Ly.rn_qa.q, F->xq, F->x, qa, nullptr, 1.0f, Ly.rn_qa.type, E, QL};
    j[1] = {Ly.lg_kva.q, F->xq, F->x, F->q35_k, nullptr, 1.0f, Ly.lg_kva.type, E, gk::kRow};
    j[2] = {Ly.lg_gate.q, F->xq, F->x, gate, nullptr, 1.0f, Ly.lg_gate.type, E, HV};
    j[3] = {Ly.gi_ik.q, F->xq, F->x, ikraw, nullptr, 1.0f, Ly.gi_ik.type, E, gk::kIdxKey};
    j[4] = {Ly.gi_iproj.q, F->xq, F->x, iw, nullptr, 1.0f, Ly.gi_iproj.type, E, IH};
    if (!gf::mv(j, pick ? 5 : 4, s)) { err = "gigachat4: the MLA projections"; return false; }
    gnorm(Ly.gn_qa, g, qa, nullptr, QL, qan, s);
    gk::scale(qan, std::sqrt((float) E / (float) QL), QL, s);   // (alpha_q; the indexer reads this latent too)
    gf::quantize_q8_1(qan, qanq, QL, s);
    gf::MvJob qb[2];
    qb[0] = {Ly.rn_qb.q, qanq, qan, F->q35_qf, nullptr, 1.0f, Ly.rn_qb.type, QL, H * gk::kQHead};
    qb[1] = {Ly.gi_iqb.q, qanq, qan, iq, nullptr, 1.0f, Ly.gi_iqb.type, QL, IH * gk::kIdxKey};
    if (!gf::mv(qb, pick ? 2 : 1, s)) { err = "gigachat4: the MLA q_b"; return false; }
    if (F->prof_on) F->mark("giga_mla_mv");
    gnorm(Ly.gn_kva, g, F->q35_k, nullptr, gk::kLat, lat, s);
    gk::scale(lat, std::sqrt((float) E / (float) gk::kLat), gk::kLat, s);   // (alpha_kv)
    if (dumps) {
        giga_dump(s, "mla_q", il, F->q35_qf, (int64_t) H * gk::kQHead);
        giga_dump(s, "mla_lat", il, lat, gk::kLat);
        giga_dump(s, "mla_krot", il, F->q35_k + gk::kLat, gk::kRot);
    }
    gk::rope_table(p, (const double*) (buf + c.freq), 1.0f, cs, s);
    gk::mla_prep(F->q35_qf, H, lat, F->q35_k + gk::kLat, cs, Ly.kc + (size_t) p * gk::kRow, true, s);
    gk::idx_prep(pick ? iq : nullptr, IH, ikraw, Ly.gi_ikn, g.norm_eps, cs, keys + (size_t) p * gk::kIdxKey, s);
    lk::mla_absorb(F->q35_qf, Ly.lg_kb, H, F->q35_q, s);
    const float ms = yarn_mscale(g);
    const float scale = ms * ms / std::sqrt((float) gk::kQHead);
    if (pick) {   // the indexer's K best of the p + 1 positions, their cache rows side by side
        int* sel = (int*) (buf + c.sel);
        uint16_t* comp = (uint16_t*) (buf + c.comp);
        gk::idx_score(iq, iw, keys, p + 1, 1.0f, buf + c.score, s);
        gk::idx_topk(buf + c.score, p + 1, K, sel, s);
        gk::gather_rows(Ly.kc, sel, K, comp, s);
        if (F->prof_on) F->mark("giga_dsa");
        if (dumps) {
            giga_dump(s, "dsa_score", il, buf + c.score, p + 1);
            giga_dump(s, "dsa_sel", il, (const float*) sel, K);   // (int32 positions)
        }
        lk::mla_attend(F->q35_q, F->q35_qf, comp, K, scale, H, sc, part, F->q35_ao, s);
    } else {
        lk::mla_attend(F->q35_q, F->q35_qf, Ly.kc, p + 1, scale, H, sc, part, F->q35_ao, s);
    }
    rk::mla_out_gated(F->q35_ao, Ly.lg_vb, gate, H, F->q35_o, s);
    if (F->prof_on) F->mark("giga_mla_attend");
    if (dumps) giga_dump(s, "mla_gated", il, F->q35_o, HV);
    gf::quantize_q8_1(F->q35_o, F->q35_aoq, HV, s);
    gf::MvJob o = {Ly.out.q, F->q35_aoq, F->q35_o, F->mixer, nullptr, 1.0f, Ly.out.type, HV, E};
    if (!gf::mv(&o, 1, s)) { err = "gigachat4: the MLA output"; return false; }
    if (F->prof_on) F->mark("giga_mla_out");
    return true;
}

bool Glm5Model::fast_giga_layers(int64_t p, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    bool pf_pending = false;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    const int E = g.n_embd;
    if (giga_buf_ == nullptr) { err = "gigachat4: the scratch is not allocated"; return false; }
    const GigaCarve c = giga_carve(g, std::max<int64_t>(max_ctx_, 1));
    if (p >= std::max<int64_t>(max_ctx_, 1)) { err = "gigachat4: position " + std::to_string(p) + " is past the context"; return false; }
    int mla = 0;
    giga_dump_pos = p;
    if (F->prof_on) F->mark("start");
    if (dumps) giga_dump(s, "inp_embd", 0, state_, E);
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        // R += the last FFN's (post-normed) output, x = GN_attn(R)
        gnorm(Ly.gn[0], g, state_, il == l0_ ? nullptr : F->ffn, E, F->x, s);
        gf::quantize_q8_1(F->x, F->xq, E, s);
        if (F->prof_on) F->mark("giga_norm");
        if (dumps) {
            if (il > l0_) giga_dump(s, "l_out", il - 1, state_, E);
            giga_dump(s, "attn_norm", il, F->x, E);
        }
        if (!(Ly.recr ? giga_gdn(F, g, il, giga_buf_, c, err)
                      : giga_mla(F, g, il, mla++, p, giga_buf_, c, giga_sc_, giga_part_, err)))
            return false;
        if (dumps) giga_dump(s, "mixer", il, F->mixer, E);
        gnorm(Ly.gn[1], g, F->mixer, nullptr, E, F->mixer, s);
        if (dumps) giga_dump(s, "attn_post", il, F->mixer, E);
        gnorm(Ly.gn[2], g, state_, F->mixer, E, F->x, s);
        gf::quantize_q8_1(F->x, F->xq, E, s);
        if (F->prof_on) F->mark("giga_norm");
        if (dumps) giga_dump(s, "ffn_norm", il, F->x, E);
        if (!Ly.moe) {
            if (!fast_dense_ffn(il, err)) return false;
        } else if (!fast_moe(il, pf_pending, err)) {
            return false;
        }
        if (dumps) giga_dump(s, "ffn_out", il, F->ffn, E);
        gnorm(Ly.gn[3], g, F->ffn, nullptr, E, F->ffn, s);
        if (dumps) giga_dump(s, "ffn_post", il, F->ffn, E);
    }
    // the head: R += the last FFN's, x = GN_output(R)
    FastState::Layer::GatedNorm on;
    on.w = giga_out_w_;
    on.down = giga_out_down_;
    on.up = giga_out_up_;
    gnorm(on, g, state_, F->ffn, E, F->head_x, s);
    gf::quantize_q8_1(F->head_x, F->head_xq, E, s);
    if (F->prof_on) F->mark("giga_head");
    if (dumps) {
        giga_dump(s, "l_out", l1_ - 1, state_, E);
        giga_dump(s, "result_norm", 0, F->head_x, E);
    }
    if (gk::launch_errors() != 0 || lk::launch_errors() != 0 || rk::launch_errors() != 0) {
        err = "gigachat4: a kernel launch failed";
        return false;
    }
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { err = std::string("gigachat4: ") + cudaGetErrorString(e); return false; } }
    return true;
}

}  // namespace strata::core
