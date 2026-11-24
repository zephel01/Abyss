// src/core/ling_fast_path.cu - inclusionAI Ling-3.x (GGUF arch bailingmoe3) on the qwen35moe fast path (Glm5Model
// members).  The model (llama.cpp src/models/bailingmoe3.cpp reads it the same way; the math is the model's own
// modeling_bailing_moe_v3*.py):
//   x = rms(h) * attn_norm
//   KDA layer   q, k, v = silu(conv4(W x)) (q, k L2-normed); f = f_a x, g = g_a x, beta = sigmoid(b x);
//               g1 = lb * sigmoid((f + dt_bias) * exp(A_log)); the gated delta recurrence (glm_fast's kda_rec) with
//               the output rms(o) * o_norm * sigmoid(g); attn_output
//   MLA layer   q = W_q x [heads][128 nope | 64 rope]; kv = W_kva x [512 latent | 64 rope]; the latent rms-normed, the
//               rope dims NORM-roped (interleaved pairs, rope_base); attention over every position in the latent space
//               (q_abs = k_b q_nope), scale 1/sqrt(192); o_h = (v_b ctx_h) * sigmoid(attn_gate x)_h; attn_output
//   h += mixer; x = rms(h) * ffn_norm; the dense FFN (the first dense_lead layers) or the MoE (fast_moe: the sigmoid
//   router + exp_probs_b, group-limited, normalised, x 2.5; the ungated shared expert; per-layer SwiGLU limits)
// The state rides qwen35_alloc: the delta-net state / conv rows are the KDA ones (Glm5Geometry::ling sets the sizes),
// an MLA layer's KV rows (kc) its cache rows of 576 FP16.  Kernels: kernels/ling_fast.cu, glm_fast's kda_rec.
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/ling_fast.hpp"
#include "strata/qtypes.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace lk = strata::kernels::ling;

void Glm5Model::ling_bind(int il, std::string& missing) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    auto& Ly = F->L[(size_t) il];
    const std::string P = "blk." + std::to_string(il) + ".";
    const auto f32 = [&](const char* n) -> const float* {
        auto it = w_.find(P + n);
        if (it == w_.end()) { missing += P + n + " (f32) "; return nullptr; }
        return it->second;
    };
    // a projection: as stored (a quantized type the GEMV serves), else its F32 / BF16 rows
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
        if (!host.empty() && cudaMalloc(&d, count * sizeof(float)) == cudaSuccess) ling_owned_.push_back(d);
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
    if (Ly.recr) {
        Ly.q = q("attn_q.weight", E, DI);
        Ly.k = q("attn_k.weight", E, DI);
        Ly.v = q("attn_v.weight", E, DI);
        Ly.lg_f = q("ssm_f_a.weight", E, DI);
        Ly.lg_g = q("ssm_g_a.weight", E, DI);
        Ly.lg_beta = q("ssm_beta.weight", E, H);
        Ly.conv[0] = f32("ssm_conv1d_q.weight");
        Ly.conv[1] = f32("ssm_conv1d_k.weight");
        Ly.conv[2] = f32("ssm_conv1d_v.weight");
        Ly.dt_bias = f32("ssm_dt.bias");
        Ly.ssm_a = f32("ssm_a");
        Ly.ssm_norm = f32("ssm_norm.weight");
        Ly.out = q("attn_output.weight", DI, E);
    } else {
        Ly.lg_wq = q("attn_q.weight", E, (int64_t) H * lk::kQHead);
        Ly.lg_kva = q("attn_kv_a_mqa.weight", E, lk::kRow);
        Ly.lg_gate = q("attn_gate.weight", E, H);
        Ly.lg_kvn = f32("attn_kv_a_norm.weight");
        Ly.lg_kb = rows32("attn_k_b.weight", lk::kNope, (int64_t) lk::kLat * H);
        Ly.lg_vb = rows32("attn_v_b.weight", lk::kLat, (int64_t) lk::kV * H);
        Ly.out = q("attn_output.weight", (int64_t) H * lk::kV, E);
    }
}

bool Glm5Model::ling_alloc(std::string& err) {
    const Glm5Geometry& g = g_;
    const int64_t ctx = std::max<int64_t>(max_ctx_, 1);
    const size_t sc = lk::mla_sc_floats(ctx, g.n_head), part = lk::mla_part_floats(ctx, g.n_head);
    if (cudaMalloc((void**) &ling_sc_, sc * 4) != cudaSuccess || cudaMalloc((void**) &ling_part_, part * 4) != cudaSuccess ||
        cudaMalloc((void**) &ling_bias_, ((size_t) g.n_expert + 64) * 4) != cudaSuccess) {
        cudaGetLastError();
        err = "ling: the attention scratch (" + std::to_string((sc + part) * 4 >> 20) + " MB) did not allocate";
        return false;
    }
    int n_mla = 0;
    for (int il = l0_; il < l1_; ++il) n_mla += !fast_->L[(size_t) il].recr;
    std::fprintf(stderr, "ling: CUDA%d %d KDA layers, %d MLA layers (latent caches %.2f GB, attention scratch %.0f MB for "
                         "%lld tokens), %d groups of experts keep %d\n",
                 dev_, (l1_ - l0_) - n_mla, n_mla, (double) n_mla * (double) ctx * lk::kRow * 2 / 1073741824.0,
                 (double) (sc + part) * 4 / 1048576.0, (long long) ctx, g.ling_group, g.ling_topk_group);
    return true;
}

void Glm5Model::ling_free() {
    for (void* p : ling_owned_) cudaFree(p);
    ling_owned_.clear();
    if (ling_sc_ != nullptr) cudaFree(ling_sc_);
    if (ling_part_ != nullptr) cudaFree(ling_part_);
    if (ling_bias_ != nullptr) cudaFree(ling_bias_);
    ling_sc_ = ling_part_ = ling_bias_ = nullptr;
}

const float* Glm5Model::ling_group_bias(int il, const float* logits, const float* bias, cudaStream_t s) {
    (void) il;
    const Glm5Geometry& g = g_;
    lk::group_bias(logits, bias, g.n_expert, g.ling_group, g.ling_topk_group, ling_bias_, s);
    return ling_bias_;
}

static void ling_dump(cudaStream_t s, const char* name, int il, const float* dev, int64_t n) {
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

bool Glm5Model::fast_ling_kda(int il, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, H = g.n_head, DI = g.d_inner();
    float* proj = F->q35_qkv;   // q | k | v
    gf::MvJob j[6];
    j[0] = {Ly.q.q, F->xq, F->x, proj, nullptr, 1.0f, Ly.q.type, E, DI};
    j[1] = {Ly.k.q, F->xq, F->x, proj + DI, nullptr, 1.0f, Ly.k.type, E, DI};
    j[2] = {Ly.v.q, F->xq, F->x, proj + 2 * DI, nullptr, 1.0f, Ly.v.type, E, DI};
    j[3] = {Ly.lg_f.q, F->xq, F->x, F->q35_z, nullptr, 1.0f, Ly.lg_f.type, E, DI};
    j[4] = {Ly.lg_g.q, F->xq, F->x, F->q35_o, nullptr, 1.0f, Ly.lg_g.type, E, DI};
    j[5] = {Ly.lg_beta.q, F->xq, F->x, F->q35_gate, nullptr, 1.0f, Ly.lg_beta.type, E, H};
    if (!gf::mv(j, 6, s)) { err = "ling: the KDA projections"; return false; }
    if (F->prof_on) F->mark("ling_kda_mv");
    const float* cw[3] = {Ly.conv[0], Ly.conv[1], Ly.conv[2]};
    lk::kda_prep(proj, cw, Ly.gdn_conv, F->q35_h, F->q35_z, Ly.dt_bias, Ly.ssm_a, g.kda_lb, F->q35_y, H, g.kda_head_dim,
                 g.d_conv, s);
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    if (dumps) {
        ling_dump(s, "kda_q", il, F->q35_h, DI);
        ling_dump(s, "kda_k", il, F->q35_h + DI, DI);
        ling_dump(s, "kda_v", il, F->q35_h + 2 * DI, DI);
        ling_dump(s, "kda_gate", il, F->q35_y, DI);
    }
    gf::kda_rec(F->q35_h, F->q35_h + DI, F->q35_h + 2 * DI, F->q35_y, F->q35_gate, Ly.gdn_state, F->q35_o, Ly.ssm_norm,
                g.norm_eps, H, g.kda_head_dim, F->q35_yq, s);
    if (F->prof_on) F->mark("ling_kda_rec");
    gf::MvJob o = {Ly.out.q, F->q35_yq, nullptr, F->mixer, nullptr, 1.0f, Ly.out.type, DI, E};
    if (!gf::mv(&o, 1, s)) { err = "ling: the KDA output"; return false; }
    if (F->prof_on) F->mark("ling_kda_out");
    if (dumps) ling_dump(s, "kda_out", il, F->mixer, E);
    return true;
}

bool Glm5Model::fast_ling_mla(int il, int64_t p, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, H = g.n_head, HV = H * lk::kV;
    gf::MvJob j[3];
    j[0] = {Ly.lg_wq.q, F->xq, F->x, F->q35_qf, nullptr, 1.0f, Ly.lg_wq.type, E, H * lk::kQHead};
    j[1] = {Ly.lg_kva.q, F->xq, F->x, F->q35_k, nullptr, 1.0f, Ly.lg_kva.type, E, lk::kRow};
    j[2] = {Ly.lg_gate.q, F->xq, F->x, F->q35_v, nullptr, 1.0f, Ly.lg_gate.type, E, H};
    if (!gf::mv(j, 3, s)) { err = "ling: the MLA projections"; return false; }
    if (F->prof_on) F->mark("ling_mla_mv");
    float2* cs = (float2*) (ling_bias_ + g.n_expert);
    lk::rope_table(p, (double) g.rope_base, cs, s);
    lk::mla_prep(F->q35_qf, H, F->q35_k, Ly.lg_kvn, g.norm_eps, cs, Ly.kc + (size_t) p * lk::kRow, s);
    lk::mla_absorb(F->q35_qf, Ly.lg_kb, H, F->q35_q, s);
    lk::mla_attend(F->q35_q, F->q35_qf, Ly.kc, p + 1, 1.0f / std::sqrt((float) lk::kQHead), H, ling_sc_, ling_part_,
                   F->q35_ao, s);
    lk::mla_out(F->q35_ao, Ly.lg_vb, F->q35_v, H, F->q35_o, s);
    if (F->prof_on) F->mark("ling_mla_attend");
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    if (dumps) ling_dump(s, "mla_gated", il, F->q35_o, HV);
    gf::quantize_q8_1(F->q35_o, F->q35_aoq, HV, s);
    gf::MvJob o = {Ly.out.q, F->q35_aoq, F->q35_o, F->mixer, nullptr, 1.0f, Ly.out.type, HV, E};
    if (!gf::mv(&o, 1, s)) { err = "ling: the MLA output"; return false; }
    if (F->prof_on) F->mark("ling_mla_out");
    if (dumps) ling_dump(s, "mla_out", il, F->mixer, E);
    if (lk::launch_errors() != 0) { err = "ling: a kernel launch failed"; return false; }
    return true;
}

}  // namespace strata::core
