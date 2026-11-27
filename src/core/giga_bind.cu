// src/core/giga_bind.cu - ai-sage GigaChat4 (GGUF arch gigachat4): a layer's weights for the qwen35moe fast path
// (Glm5Model::giga_bind, from qwen35_bind) and the head's GatedNorm (giga_bind_head, from giga_alloc).  The fields are
// FastState::Layer's giga block (glm_fast_state.hpp); the token path that reads them is giga_fast_path.cu.
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/qtypes.hpp"

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;

void Glm5Model::giga_bind(int il, std::string& missing) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    auto& Ly = F->L[(size_t) il];
    const std::string P = "blk." + std::to_string(il) + ".";
    const auto f32 = [&](const std::string& n) -> const float* {
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
    // a 3-D absorbed projection (k_b / v_b): decoded once to F32 rows on the device - rows of n_in, n_out rows, in the
    // GGUF's order (k_b [head][512 rows][128], v_b [head][128 rows][512])
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
        }
        void* d = nullptr;
        if (!host.empty() && cudaMalloc(&d, count * sizeof(float)) == cudaSuccess) giga_owned_.push_back(d);
        if (host.empty() || d == nullptr ||
            cudaMemcpy(d, host.data(), count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaGetLastError();
            missing += P + n + " (" + std::to_string(n_in) + "x" + std::to_string(n_out) + ", a type with a host decoder) ";
            return nullptr;
        }
        return (const float*) d;
    };
    const auto gnorm = [&](const std::string& base) {
        FastState::Layer::GatedNorm n;
        n.w = f32(base + ".weight");
        n.down = f32(base + "_gate_down.weight");
        n.up = f32(base + "_gate_up.weight");
        return n;
    };
    const int E = g.n_embd, H = g.n_head;
    Ly.gn[0] = gnorm("attn_norm");
    Ly.gn[1] = gnorm("post_attention_norm");
    Ly.gn[2] = gnorm("ffn_norm");
    Ly.gn[3] = gnorm("post_ffw_norm");
    Ly.attn_norm = Ly.gn[0].w;   // (the shared code's names for the pre-norms)
    Ly.post_norm = Ly.gn[2].w;
    if (Ly.recr) {
        const int C = g.ssm_conv_dim(), V = g.ssm_inner;
        Ly.qkv = q("attn_qkv.weight", E, C);
        Ly.zg = q("attn_gate.weight", E, V);
        Ly.ssm_o = q("ssm_out.weight", V, E);
        Ly.alpha_w = q("ssm_alpha.weight", E, g.ssm_v_heads);
        Ly.beta_w = q("ssm_beta.weight", E, g.ssm_v_heads);
        Ly.conv1d = f32("ssm_conv1d.weight");
        Ly.dt_bias = f32("ssm_dt.bias");
        Ly.ssm_a = f32("ssm_a");
        Ly.ssm_norm = f32("ssm_norm.weight");
    } else {
        const int nope = g.ling_qk_nope, rope = g.ling_rope, vd = g.ling_v, lat = g.kv_lora;
        Ly.rn_qa = q("attn_q_a.weight", E, g.q_lora);
        Ly.gn_qa = gnorm("attn_q_a_norm");
        Ly.rn_qan = Ly.gn_qa.w;
        Ly.rn_qb = q("attn_q_b.weight", g.q_lora, (int64_t) H * (nope + rope));
        Ly.lg_kva = q("attn_kv_a_mqa.weight", E, lat + rope);
        Ly.gn_kva = gnorm("attn_kv_a_norm");
        Ly.lg_kvn = Ly.gn_kva.w;
        Ly.lg_gate = q("attn_gate.weight", E, (int64_t) H * vd);
        Ly.lg_kb = rows32("attn_k_b.weight", nope, (int64_t) lat * H);
        Ly.lg_vb = rows32("attn_v_b.weight", lat, (int64_t) vd * H);
        Ly.out = q("attn_output.weight", (int64_t) H * vd, E);
        Ly.gi_iqb = q("indexer.attn_q_b.weight", g.q_lora, (int64_t) g.idx_heads * g.idx_key);
        Ly.gi_ik = q("indexer.attn_k.weight", E, g.idx_key);
        Ly.gi_ikn = f32("indexer.k_norm.weight");
        Ly.gi_iproj = q("indexer.proj.weight", E, g.idx_heads);
    }
}

bool Glm5Model::giga_bind_head(std::string& missing) {
    const auto f = [&](const char* n) -> const float* {
        const auto it = w_.find(n);
        if (it == w_.end()) { missing += std::string(n) + " (f32) "; return nullptr; }
        return it->second;
    };
    giga_out_w_ = f("output_norm.weight");
    giga_out_down_ = f("output_norm_gate_down.weight");
    giga_out_up_ = f("output_norm_gate_up.weight");
    return giga_out_w_ != nullptr && giga_out_down_ != nullptr && giga_out_up_ != nullptr;
}

void Glm5Model::giga_free() {
    for (void* p : giga_owned_) cudaFree(p);
    giga_owned_.clear();
    giga_buf_ = giga_sc_ = giga_part_ = nullptr;
    giga_psc_ = nullptr;
    giga_psc_floats_ = 0;
}

}  // namespace strata::core
