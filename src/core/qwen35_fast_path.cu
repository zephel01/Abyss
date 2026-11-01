// src/core/qwen35_fast_path.cu - the qwen35moe (Qwen3.5-MoE: Ornith-1.5-122B) pieces of the fast decode path
// (Glm5Model members).  A layer: x = rms(R) * attn_norm -> the mixer (a gated delta-net layer: gdn.hpp's recurrence
// with SiLU on the output gate; or a full-attention layer: GQA over an FP16 KV cache, q / k norms, NEOX RoPE on the
// first n_rot dims, a sigmoid gate from the q projection) -> R += mixer -> x = rms(R) * post_attention_norm -> the
// MoE (fast_moe: a softmax top-k router, the routed experts through the tiers and the lanes, the shared expert behind
// its sigmoid gate) -> R += ffn.  No hyper-connections: the residual R is state_ (n_embd floats).  The math follows
// llama.cpp's src/models/qwen35moe.cpp and delta-net-base.cpp.
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/core/knobs.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/qwen35_fast.hpp"
#include "strata/kernels/rope.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace q35 = strata::kernels::q35;

namespace {
size_t al(size_t b) { return (b + 255u) & ~(size_t) 255u; }
}  // namespace

void Glm5Model::qwen35_bind(int il, std::string& missing) {
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
        auto it = ws_map_.find(P + n);
        if ((it == ws_map_.end() || it->second.type == 0) && g.q4x) {
            // ISTA-DASLab's GSQ-RCO Qwen3.8 keeps the hyper-connections and ssm_alpha / beta in BF16 (the pack's BF16
            // arena): the GEMVs take BF16 rows with the float activation every job here passes beside its q8_1
            const auto b = w16_.find(P + n);
            if (b != w16_.end()) return WSlot{nullptr, gf::kTypeBF16, b->second, n_in, n_out};
            const auto f = w_.find(P + n);   // (a small BF16 matrix - the injects - is widened to F32 at load)
            if (f != w_.end()) return WSlot{f->second, gf::kTypeF32, f->second, n_in, n_out};
        }
        if (it == ws_map_.end() || it->second.type == 0 || !gf::mv_supported(it->second.type)) {
            missing += P + n + " (quant) ";
            return WSlot{};
        }
        if ((it->second.n_in != 0 && it->second.n_in != n_in) || (it->second.n_out != 0 && it->second.n_out != n_out))
            missing += P + n + " (shape " + std::to_string(it->second.n_in) + "x" + std::to_string(it->second.n_out) +
                       ", expected " + std::to_string(n_in) + "x" + std::to_string(n_out) + ") ";
        return it->second;
    };
    const int E = g.n_embd, C = g.ssm_conv_dim(), V = g.ssm_inner, D = g.head_dim;
    if (g.q4x) {   // no norms: the hyper-connections' mixers
        const int64_t HE = (int64_t) g.hc * E;
        const char* pre[2] = {"hc_attn_", "hc_ffn_"};
        for (int h = 0; h < 2; ++h) {
            const std::string b = pre[h];
            Ly.hc4[h].norm = f32((b + "norm.weight").c_str());
            Ly.hc4[h].down = q((b + "down.weight").c_str(), HE, g.hc_lr);
            Ly.hc4[h].up = q((b + "up.weight").c_str(), g.hc_lr, HE);
            Ly.hc4[h].inject = q((b + "inject.weight").c_str(), HE, g.hc);
        }
    } else {
        Ly.attn_norm = f32("attn_norm.weight");
        Ly.post_norm = f32("post_attention_norm.weight");
    }
    if (Ly.recr) {
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
        Ly.aq = q("attn_q.weight", E, (int64_t) g.n_head * 2 * D);
        Ly.ak = q("attn_k.weight", E, (int64_t) g.attn_head_kv * D);
        Ly.av = q("attn_v.weight", E, (int64_t) g.attn_head_kv * D);
        Ly.out = q("attn_output.weight", (int64_t) g.n_head * D, E);
        Ly.q_norm = f32("attn_q_norm.weight");
        Ly.k_norm = f32("attn_k_norm.weight");
        if (g.q4x) {   // the QSA indexer: BF16 projections
            const auto b16 = [&](const char* n, int64_t n_out) -> WSlot {
                const auto it = w16_.find(P + n);
                if (it == w16_.end()) { missing += P + n + " (bf16) "; return WSlot{}; }
                return WSlot{nullptr, gf::kTypeBF16, it->second, E, n_out};
            };
            Ly.q4_ik = b16("indexer.k_proj.weight", g.idx_key);
            Ly.q4_iq = b16("indexer.q_proj.weight", (int64_t) g.idx_heads * g.idx_key);
            Ly.q4_ikn = f32("indexer.k_norm.weight");
            Ly.q4_iqn = f32("indexer.q_norm.weight");
        }
    }
    if (Ly.moe && g.n_shared > 0) Ly.sh_gate_w = f32("ffn_gate_inp_shexp.weight");
    if (Ly.mtp) {   // the NextN block: eh_proj in (2 n_embd -> n_embd) and its two norms; the head's norm the trunk's
        Ly.eh = q("nextn.eh_proj.weight", 2 * E, E);
        Ly.enorm = f32("nextn.enorm.weight");
        Ly.hnorm = f32("nextn.hnorm.weight");
    }
}

bool Glm5Model::qwen35_alloc(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    if (g.q4x && l1_ == g.n_layers) {   // the output mixer's weights (the head's part)
        const auto nw = w_.find("output_hc_norm.weight");
        const auto dw = ws_map_.find("output_hc_down.weight"), uw = ws_map_.find("output_hc_up.weight");
        const auto db = w16_.find("output_hc_down.weight"), ub = w16_.find("output_hc_up.weight");   // (ISTA: BF16)
        const int64_t HE = (int64_t) g.hc * g.n_embd;
        if (nw == w_.end() || (dw == ws_map_.end() && db == w16_.end()) || (uw == ws_map_.end() && ub == w16_.end()) ||
            (dw != ws_map_.end() && !gf::mv_supported(dw->second.type)) ||
            (uw != ws_map_.end() && !gf::mv_supported(uw->second.type))) {
            err = "qwen4exp: output_hc_norm / output_hc_down / output_hc_up (served natively) are missing";
            return false;
        }
        F->q4_head[0] = nw->second;
        F->q4_head_w[0] = dw != ws_map_.end() ? dw->second : WSlot{nullptr, gf::kTypeBF16, db->second, HE, g.hc_lr};
        F->q4_head_w[1] = uw != ws_map_.end() ? uw->second : WSlot{nullptr, gf::kTypeBF16, ub->second, g.hc_lr, HE};
    }
    const size_t S = (size_t) g.ssm_state, HV = (size_t) g.ssm_v_heads, C = (size_t) g.ssm_conv_dim(), V = (size_t) g.ssm_inner;
    const size_t D = (size_t) g.head_dim, HD = (size_t) g.n_head * D, KV = (size_t) g.attn_head_kv * D;
    const size_t ctx = (size_t) std::max<int64_t>(max_ctx_, 1);
    const size_t state_b = al(S * HV * S * 4), conv_b = al(C * (size_t) (g.d_conv - 1) * 4), cache_b = al(ctx * KV * 2);
    int n_rec = 0, n_attn = 0;
    for (int il = l0_; il < lt_; ++il) (F->L[(size_t) il].recr ? n_rec : n_attn) += 1;   // (lt_: the NextN block too)
    const size_t part = q35::attn_part_floats(g.n_head, g.head_dim, (int) ctx);
    size_t bytes = (size_t) n_rec * (state_b + conv_b) + (size_t) n_attn * 2 * cache_b;
    bytes += al(C * 4) + al(V * 4) + al(C * 4) + al(2 * HV * 4) + al(HV * 4) + al(V * 4) + al(V * 4) + al(V / 32 * 36) +
             al(HD * 2 * 4) + al(KV * 4) + al(KV * 4) + al(HD * 4) + al(HD * 4) + al(HD / 32 * 36) + al(part * 4) + 256;
    const size_t E = (size_t) g.n_embd, TM = (size_t) kVerifyMax;
    const size_t HE = (size_t) g.hc * E, LR = (size_t) g.hc_lr;
    if (g.q4x)   // the hyper-connections' activations
        bytes += al(HE * 4) + al(HE / 32 * 36) + al(LR * 4) + al(LR / 32 * 36) + al(HE * 4) + 2 * al(64);
    const bool mtp = mtp_il_ >= 0;
    if (mtp)   // the drafter: the pending hidden, <= kVerifyMax rows of scratch, the chain's tokens and confidences
        bytes += al(E * 4) + al(TM * E * 4) * 3 + al(TM * (E / 32) * 36) + al(TM * (2 * E / 32) * 36) + al(TM * KV * 4) * 2 +
                 al(TM * 4) * 2;
    if (cudaMalloc(&qwen35_state_, bytes) != cudaSuccess) {
        cudaGetLastError();
        err = "qwen35moe: the states and caches (" + std::to_string(bytes >> 20) + " MB for " + std::to_string(ctx) +
              " tokens) did not allocate";
        return false;
    }
    cudaMemset(qwen35_state_, 0, bytes);
    uint8_t* at = (uint8_t*) qwen35_state_;
    const auto take = [&](size_t b) {
        uint8_t* p = at;
        at += al(b);
        return p;
    };
    for (int il = l0_; il < lt_; ++il) {
        auto& Ly = F->L[(size_t) il];
        if (Ly.recr) {
            Ly.gdn_state = (float*) take(S * HV * S * 4);
            Ly.gdn_conv = (float*) take(C * (size_t) (g.d_conv - 1) * 4);
        } else {
            Ly.kc = (uint16_t*) take(ctx * KV * 2);
            Ly.vc = (uint16_t*) take(ctx * KV * 2);
        }
    }
    F->q35_qkv = (float*) take(C * 4);
    F->q35_z = (float*) take(V * 4);
    F->q35_h = (float*) take(C * 4);
    F->q35_ab = (float*) take(2 * HV * 4);
    F->q35_gate = (float*) take(HV * 4);
    F->q35_o = (float*) take(V * 4);
    F->q35_y = (float*) take(V * 4);
    F->q35_yq = take(V / 32 * 36);
    F->q35_qf = (float*) take(HD * 2 * 4);
    F->q35_k = (float*) take(KV * 4);
    F->q35_v = (float*) take(KV * 4);
    F->q35_q = (float*) take(HD * 4);
    F->q35_ao = (float*) take(HD * 4);
    F->q35_aoq = take(HD / 32 * 36);
    F->q35_part = (float*) take(part * 4);
    F->q35_shg = (float*) take(256);
    if (g.q4x) {
        F->q4_xn = (float*) take(HE * 4);
        F->q4_xnq = take(HE / 32 * 36);
        F->q4_lo = (float*) take(LR * 4);
        F->q4_loq = take(LR / 32 * 36);
        F->q4_gate = (float*) take(HE * 4);
        F->q4_sc = (float*) take(64);
        F->q4_inj[0] = (float*) take(64);
        F->q4_inj[1] = (float*) take(64);
    }
    if (mtp) {
        F->q35_pend_h = (float*) take(E * 4);
        F->q35_mt_emb = (float*) take(TM * E * 4);
        F->q35_mt_hid = (float*) take(TM * E * 4);
        F->q35_mt_x = (float*) take(TM * E * 4);
        F->q35_mt_xq = take(TM * (E / 32) * 36);
        F->q35_mt_cat = take(TM * (2 * E / 32) * 36);
        F->q35_mt_k = (float*) take(TM * KV * 4);
        F->q35_mt_v = (float*) take(TM * KV * 4);
        F->q35_mt_conf = (float*) take(TM * 4);
        F->q35_mt_tok = (int*) take(TM * 4);
        if (cudaHostAlloc((void**) &F->q35_mt_conf_h, 256, cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &F->q35_mt_conf_d, F->q35_mt_conf_h, 0) != cudaSuccess) {
            err = "qwen35moe: the drafter's host-mapped word did not allocate";
            return false;
        }
    }
    std::fprintf(stderr, "qwen35moe: CUDA%d %d delta-net layers (states %.0f MB), %d attention layers (KV caches %.2f GB "
                         "for %lld tokens)\n", dev_, n_rec, (double) n_rec * (state_b + conv_b) / 1048576.0, n_attn,
                 (double) n_attn * 2 * cache_b / 1073741824.0, (long long) ctx);
    if (g.q4x && l0_ <= 1 && 1 < l1_ && !q4x_ple_setup(err)) return false;
    if (g.q4x && !q4x_qsa_setup(err)) return false;
    if (g.q4x && q4d_ == nullptr && getenv("STRATA_Q4X_MTP") != nullptr && getenv("STRATA_Q4X_MTP")[0] != '\0' &&
        getenv("STRATA_GLM_NO_MTP") == nullptr && !q4x_mtp_load(getenv("STRATA_Q4X_MTP"), err))
        return false;
    return true;
}

// qwen4exp's QSA (llama.cpp build_qsa_top_k / build_attn_qsa): every attention layer keeps the indexer's block keys
// (4 cells pooled, normed, rotated at the block's first cell: the old path's indexer_key_append); past 2,051 cells a
// token attends to the 512 best-scored blocks plus the incomplete tail (qsa_block_scores / qsa_block_topk: the cells
// ascending, ties to the lowest), gathered out of the KV cache; up to 2,051 the selection is every cell (dense)
bool Glm5Model::q4x_qsa_setup(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    namespace k = strata::kernels;
    const k::QsaShapes sh = k::qsa_real_shapes();
    if (g.n_head != sh.n_head || g.attn_head_kv != sh.n_head_kv || g.head_dim != sh.head_dim || g.n_rot != sh.n_rot ||
        g.idx_heads != sh.idx_n_head || g.idx_key != sh.idx_dim || g.idx_top_k != sh.idx_top_k) {
        err = "qwen4exp: the QSA kernels take Qwen3.8-Flash-Next's attention geometry";
        return false;
    }
    const int64_t ctx = std::max<int64_t>(max_ctx_, 1);
    F->q4_max_blocks = (int) (ctx / sh.idx_block + 2);
    F->q4_cap = (int) k::qsa_selection_width(k::kTopkMaxCells, sh);
    const size_t ID = (size_t) sh.idx_dim, half = (size_t) sh.n_rot / 2, row = (size_t) g.attn_head_kv * g.head_dim;
    bool ok = true;
    const auto dm = [&](auto** p, size_t b) { ok = ok && cudaMalloc((void**) p, b) == cudaSuccess; };
    for (int il = l0_; il < l1_; ++il) {
        auto& Ly = F->L[(size_t) il];
        if (Ly.recr) continue;
        dm(&Ly.q4_tail, (size_t) (sh.idx_block - 1) * ID * 4);
        dm(&Ly.q4_dead, ID * 4);
        dm(&Ly.q4_pooled, (size_t) F->q4_max_blocks * ID * 4);
        dm(&Ly.q4_bpos, 64);
    }
    dm(&F->q4_cos, (size_t) ctx * half * 4);
    dm(&F->q4_sin, (size_t) ctx * half * 4);
    dm(&F->q4_iraw, ID * 4);
    dm(&F->q4_iq, (size_t) sh.idx_n_head * ID * 4);
    dm(&F->q4_scores, (size_t) F->q4_max_blocks * 4);
    dm(&F->q4_ids, (size_t) F->q4_cap * 4);
    dm(&F->q4_ks, (size_t) F->q4_cap * row * 2);
    dm(&F->q4_vs, (size_t) F->q4_cap * row * 2);
    dm(&F->q4_step, 64);
    dm(&F->q4_pos, 256);
    dm(&F->q4_pf_scores, (size_t) FastState::kQ4Grp * F->q4_max_blocks * 4);
    dm(&F->q4_pf_ids, (size_t) FastState::kQ4Grp * F->q4_cap * 4);
#if !defined(STRATA_USE_HIP)
    dm(&F->q4_pf_part, q35::attn_rows_part_floats(FastState::kQ4Grp, g.n_head, g.head_dim, F->q4_cap) * 4);
#endif
    ok = ok && cudaHostAlloc((void**) &F->q4_step_h, 64, cudaHostAllocDefault) == cudaSuccess &&
         cudaHostAlloc((void**) &F->q4_pos_h, 256, cudaHostAllocDefault) == cudaSuccess;
    if (!ok) {
        cudaGetLastError();
        err = "qwen4exp: the QSA indexer's buffers did not allocate";
        return false;
    }
    {   // the RoPE table in float64 on the host (the old path's, rope.hpp)
        std::vector<float> c((size_t) ctx * half), sn((size_t) ctx * half);
        k::build_rope_table((int) sh.n_rot, (double) g.rope_base, (int) ctx, c.data(), sn.data());
        cudaMemcpy(F->q4_cos, c.data(), c.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(F->q4_sin, sn.data(), sn.size() * 4, cudaMemcpyHostToDevice);
    }
    F->q4_step_at = -1;
    return true;
}

// ---------------------------------------------------------------- qwen4exp's per-layer embedding (layer 1)
// llama.cpp src/models/qwen4exp.cpp build_inp_ple / build_ple: the token and the two before it hash (ngram_rows, the
// EOS cut) into 16 rows of the 28.8 GB IQ4_NL table (PleTable: direct SSD reads, issued as soon as the token is known,
// collected before layer 1), key = ple_key . emb and value = ple_value . emb on the device, then the gate, the dilated
// conv over the normalized rows' history and R += gated + silu(conv) - native_ple_postops (the old path's, matched to
// llama.cpp) - and the history advances a row
bool Glm5Model::q4x_ple_setup(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    namespace k = strata::kernels;
    const int E = g.n_embd, HE = g.hc * E;
    if (E != k::NG_N_EMBD || HE != k::NG_HC_DIM) {
        err = "qwen4exp: the PLE kernels take n_embd 2560 and 4 streams";
        return false;
    }
    const auto wq = [&](const char* n, int64_t n_in, int64_t n_out, WSlot& out) {
        const auto it = ws_map_.find(n);
        if (it == ws_map_.end() || it->second.type == 0) {   // ISTA-DASLab's GSQ-RCO: key / value in BF16
            const auto b = w16_.find(n);
            if (b == w16_.end()) return false;
            out = WSlot{nullptr, gf::kTypeBF16, b->second, n_in, n_out};
            return true;
        }
        if (it == ws_map_.end() || !gf::mv_supported(it->second.type) || it->second.n_in != n_in || it->second.n_out != n_out)
            return false;
        out = it->second;
        return true;
    };
    const auto wf = [&](const char* n) -> const float* {
        const auto it = w_.find(n);
        return it == w_.end() ? nullptr : it->second;
    };
    const float* conv = wf("blk.1.ple_conv1d.weight");
    F->ple_w.norm_key = wf("blk.1.ple_norm_key.weight");
    F->ple_w.norm_query = wf("blk.1.ple_norm_query.weight");
    F->ple_w.norm_conv = wf("blk.1.ple_norm_conv.weight");
    if (!wq("blk.1.ple_key.weight", E, HE, F->ple_key_w) || !wq("blk.1.ple_value.weight", E, E, F->ple_val_w) ||
        conv == nullptr || F->ple_w.norm_key == nullptr || F->ple_w.norm_query == nullptr || F->ple_w.norm_conv == nullptr) {
        err = "qwen4exp: blk.1.ple_* (key / value served natively, the norms and the conv) are missing - pack with "
              "tools/iq_pack.py --fast";
        return false;
    }
    if (ple_gguf_.empty()) {
        err = "qwen4exp: no shard holds per_layer_token_embd (STRATA_Q4X_PLE_GGUF names the GGUF that does)";
        return false;
    }
    const size_t n_conv = (size_t) k::PLE_CONV_KERNEL * HE;
    bool ok = cudaHostAlloc((void**) &F->ple_emb_h, (size_t) E * 4, cudaHostAllocDefault) == cudaSuccess;
    const auto dm = [&](auto** p, size_t b) { ok = ok && cudaMalloc((void**) p, b) == cudaSuccess; };
    dm(&F->ple_emb, (size_t) E * 4);
    dm(&F->ple_embq, (size_t) E / 32 * 36);
    dm(&F->ple_kp, (size_t) HE * 4);
    dm(&F->ple_val, (size_t) E * 4);
    dm(&F->ple_hist, (size_t) k::NG_HIST * HE * 4);
    dm(&F->ple_key, (size_t) HE * 4);
    dm(&F->ple_norm, (size_t) HE * 4);
    dm(&F->ple_gated, (size_t) HE * 4);
    dm(&F->ple_conv, (size_t) HE * 4);
    dm(&F->ple_gate, 64);
    dm(&F->ple_conv16, n_conv * 2);
    if (!ok) {
        cudaGetLastError();
        err = "qwen4exp: the PLE buffers did not allocate";
        return false;
    }
    cudaMemset(F->ple_hist, 0, (size_t) k::NG_HIST * HE * 4);
    {   // the conv taps as F16 (the pack widened the GGUF's F16 to F32: exact both ways)
        std::vector<float> c32(n_conv);
        std::vector<ggml_fp16_t> c16(n_conv);
        cudaMemcpy(c32.data(), conv, n_conv * 4, cudaMemcpyDeviceToHost);
        ggml_fp32_to_fp16_row(c32.data(), c16.data(), (int64_t) n_conv);
        cudaMemcpy(F->ple_conv16, c16.data(), n_conv * 2, cudaMemcpyHostToDevice);
    }
    F->ple_w.conv1d_f16 = F->ple_conv16;
    F->ple.reset(new k::PleTable());
    k::PleIoOptions io;
    if (!F->ple->open(ple_gguf_, err, io)) {
        err = "qwen4exp: the PLE table (" + ple_gguf_ + "): " + err;
        F->ple.reset();
        return false;
    }
    F->ple_toks.assign((size_t) std::max<int64_t>(max_ctx_, 1), -1);
    F->ple_il = 1;
    std::fprintf(stderr, "qwen4exp: CUDA%d PLE table %llu rows from %s (direct reads)\n", dev_,
                 (unsigned long long) F->ple->rows(), ple_gguf_.c_str());
    return true;
}

bool Glm5Model::q4x_ple_issue(int32_t token, int64_t p, std::string& err) {
    FastState* F = fast_;
    namespace k = strata::kernels;
    if (F->ple == nullptr) return true;
    if (F->ple_issued) {   // (a token whose layers never ran: its rows are dropped)
        if (!F->ple->collect(F->ple_emb_h, err)) return false;
        F->ple_issued = false;
    }
    const size_t n = F->ple_toks.size();
    if (p >= 0 && (size_t) p < n) F->ple_toks[(size_t) p] = token;
    const int32_t prev[2] = {p >= 2 && (size_t) p - 2 < n ? F->ple_toks[(size_t) p - 2] : -1,
                             p >= 1 && (size_t) p - 1 < n ? F->ple_toks[(size_t) p - 1] : -1};
    static const k::PleConsts consts = k::ple_artifact_consts();
    uint32_t rows[k::PLE_N_HEADS];
    k::ngram_rows(&token, prev, 1, consts, rows);
    if (!F->ple->issue(rows)) {
        err = "qwen4exp: the PLE rows could not be issued";
        return false;
    }
    F->ple_issued = true;
    return true;
}

bool Glm5Model::q4x_ple_apply(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    namespace k = strata::kernels;
    if (F->ple == nullptr) return true;
    if (!F->ple_issued) {
        err = "qwen4exp: layer 1 ran without its PLE rows";
        return false;
    }
    if (!F->ple->collect(F->ple_emb_h, err)) { err = "qwen4exp: PLE: " + err; return false; }
    F->ple_issued = false;
    const int E = g.n_embd, HE = g.hc * E;
    cudaMemcpyAsync(F->ple_emb, F->ple_emb_h, (size_t) E * 4, cudaMemcpyHostToDevice, s);
    gf::quantize_q8_1(F->ple_emb, F->ple_embq, E, s);
    gf::MvJob j[2];
    j[0] = {F->ple_key_w.q, F->ple_embq, F->ple_emb, F->ple_kp, nullptr, 1.0f, F->ple_key_w.type, E, HE};
    j[1] = {F->ple_val_w.q, F->ple_embq, F->ple_emb, F->ple_val, nullptr, 1.0f, F->ple_val_w.type, E, E};
    if (!gf::mv(j, 2, s)) { err = "qwen4exp: the PLE key / value"; return false; }
    k::NativePlePostopsBuffers b;
    b.key = F->ple_key;
    b.query = F->ple_norm;   // (may equal normalized)
    b.gate = F->ple_gate;
    b.gated = F->ple_gated;
    b.normalized = F->ple_norm;
    b.conv = F->ple_conv;
    b.result = state_;       // (may equal hidden)
    k::native_ple_postops(F->ple_kp, state_, F->ple_val, F->ple_hist, F->ple_w, b, (void*) s);
    k::ple_history_advance(F->ple_hist, F->ple_norm, (void*) s);
    return true;
}

void Glm5Model::qwen35_reset() {
    FastState* F = fast_;
    if (F == nullptr || qwen35_state_ == nullptr) return;
    const Glm5Geometry& g = g_;
    const size_t S = (size_t) g.ssm_state, HV = (size_t) g.ssm_v_heads, C = (size_t) g.ssm_conv_dim();
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (!Ly.recr) continue;
        cudaMemsetAsync(Ly.gdn_state, 0, S * HV * S * 4, F->cs);
        cudaMemsetAsync(Ly.gdn_conv, 0, C * (size_t) (g.d_conv - 1) * 4, F->cs);
    }
    if (F->ple_hist != nullptr)
        cudaMemsetAsync(F->ple_hist, 0, (size_t) strata::kernels::NG_HIST * g.hc * g.n_embd * 4, F->cs);
    q35_pend_pos_ = q35_row_done_ = -1;
    cudaStreamSynchronize(F->cs);
}

bool Glm5Model::q35_snap_copy(bool restore) {
    FastState* F = fast_;
    if (F == nullptr || qwen35_state_ == nullptr) return false;
    const Glm5Geometry& g = g_;
    const size_t S = (size_t) g.ssm_state, HV = (size_t) g.ssm_v_heads, C = (size_t) g.ssm_conv_dim();
    const size_t state_b = S * HV * S * 4, conv_b = C * (size_t) (g.d_conv - 1) * 4;
    const size_t pend_b = F->q35_pend_h != nullptr ? (size_t) (g.q4x ? g.hc : 1) * g.n_embd * 4 : 0;
    const size_t ple_b = F->ple_hist != nullptr ? (size_t) strata::kernels::NG_HIST * g.hc * g.n_embd * 4 : 0;
    // (qwen4exp: every attention layer's QSA raw tail - the cells of the block being filled; a position restored inside
    // a block finds its earlier cells' keys there, not the later positions' that overwrote them)
    const size_t tail_b = (size_t) 3 * g.idx_key * sizeof(float);
    size_t total = pend_b + ple_b;
    for (int il = l0_; il < l1_; ++il) {
        if (F->L[(size_t) il].recr) total += state_b + conv_b;
        else if (F->L[(size_t) il].q4_tail != nullptr) total += tail_b;
    }
    if (q35_snap_h_ == nullptr) {
        if (restore) return false;
        if (cudaHostAlloc(&q35_snap_h_, total, cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            q35_snap_h_ = nullptr;
            return false;
        }
    }
    uint8_t* at = (uint8_t*) q35_snap_h_;
    const auto run = [&](void* d, size_t b) {
        cudaMemcpyAsync(restore ? d : at, restore ? at : d, b, restore ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost,
                        F->cs);
        at += b;
    };
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (!Ly.recr) {
            if (Ly.q4_tail != nullptr) run(Ly.q4_tail, tail_b);
            continue;
        }
        run(Ly.gdn_state, state_b);
        run(Ly.gdn_conv, conv_b);
    }
    if (pend_b > 0) run(F->q35_pend_h, pend_b);
    if (ple_b > 0) run(F->ple_hist, ple_b);   // (qwen4exp: the PLE conv's history)
    if (restore) {
        q35_pend_pos_ = q35_snap_pend_;
        q35_row_done_ = q35_snap_done_;
    } else {
        q35_snap_pend_ = q35_pend_pos_;
        q35_snap_done_ = q35_row_done_;
    }
    return cudaStreamSynchronize(F->cs) == cudaSuccess;
}

// ---------------------------------------------------------------- the NextN block as the drafter
// The block (llama.cpp's graph_mtp): h = the trunk's output_norm'd hidden at a position, e = the NEXT token's
// embedding -> eh_proj([enorm(e) | hnorm(h)]) -> an attention layer (its own KV cache, a row a position) -> the MoE ->
// output_norm -> the trunk's head: the logits of the token after the next.  A cache row needs only the position's
// (h, next token) through eh_proj and k / v, so a position waits (q35_pend_pos_) until its next token arrives
// (fast_token, a window's first token, or a draft), then its row is written; a draft step at the pending position
// writes the row itself.  Chained drafts feed the block its own normed output as h and its argmax as the token.
bool Glm5Model::q35_mtp_on() const {
    if (g_.q4x)
        return q4d_ != nullptr && fast_ != nullptr && fast_->q35_pend_h != nullptr &&
               (fast_->emb_tab_d != nullptr || fast_->emb_q8_d != nullptr);
    return g_.qwen35() && mtp_il_ >= 0 && fast_ != nullptr && (fast_->emb_tab_d != nullptr || fast_->emb_q8_d != nullptr) && fast_->q35_pend_h != nullptr;
}

void Glm5Model::q35_mtp_pend_set(int64_t p, const float* h) {
    FastState* F = fast_;
    cudaMemcpyAsync(F->q35_pend_h, h, (size_t) (g_.q4x ? g_.hc : 1) * g_.n_embd * sizeof(float), cudaMemcpyDeviceToDevice,
                    F->cs);
    q35_pend_pos_ = p;
}

bool Glm5Model::q35_mtp_pend_commit(int32_t next, std::string& err) {
    if (q35_pend_pos_ < 0 || q35_pend_pos_ <= q35_row_done_) {   // (none, or a draft step wrote it)
        q35_pend_pos_ = -1;
        return true;
    }
    const int64_t p = q35_pend_pos_;
    q35_pend_pos_ = -1;
    return q35_mtp_rows(p, 1, fast_->q35_pend_h, 0, &next, err);
}

// the cache rows of positions p0 .. p0 + n - 1 (n <= kVerifyMax) from their normed hiddens (h_stride floats apart) and
// their next tokens: eh_proj over the rows in one mvn launch, the attention norm a row, k / v in one launch
bool Glm5Model::q35_mtp_rows(int64_t p0, int n, const float* h_rows, size_t h_stride, const int32_t* next, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    if (n <= 0) return true;
    if (g.q4x) return q4x_mtp_rows(p0, n, h_rows, h_stride, next, err);
    if (n > kVerifyMax) { err = "qwen35 mtp: " + std::to_string(n) + " rows (at most " + std::to_string(kVerifyMax) + ")"; return false; }
    const int il = mtp_il_, E = g.n_embd, D = g.head_dim, KV = g.attn_head_kv * D;
    const size_t q8e = (size_t) (E / 32) * 36, cat_b = (size_t) (2 * E / 32) * 36;
    const auto& Ly = F->L[(size_t) il];
    if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "qwen35 mtp rows", err)) return false;   // (tok_in_h may feed an earlier copy)
    for (int t = 0; t < n; ++t) F->tok_in_h[t] = next[t];
    if (!gf::embed_rows_q(F->emb_tab_d ? pack_emb_type_ : 8 /* Q8_0 */, F->emb_tab_d ? F->emb_tab_d : F->emb_q8_d, F->tok_in_d, n, E, F->q35_mt_emb, s)) { err = "qwen35 mtp: the embedding rows"; return false; }
    for (int t = 0; t < n; ++t)
        gf::mtp_in(F->q35_mt_emb + (size_t) t * E, h_rows + (size_t) t * h_stride, Ly.enorm, Ly.hnorm, g.norm_eps, E,
                   (uint8_t*) F->q35_mt_cat + (size_t) t * cat_b, s);
    gf::MvJob eh = {Ly.eh.q, F->q35_mt_cat, nullptr, F->q35_mt_hid, nullptr, 1.0f, Ly.eh.type, 2 * E, E};
    if (!gf::mvn(&eh, 1, n, s)) { err = "qwen35 mtp: eh_proj rows"; return false; }
    for (int t = 0; t < n; ++t)
        gf::rms_q8(F->q35_mt_hid + (size_t) t * E, nullptr, Ly.attn_norm, g.norm_eps, E, F->q35_mt_x + (size_t) t * E,
                   (uint8_t*) F->q35_mt_xq + (size_t) t * q8e, s);
    gf::MvJob j[2];
    j[0] = {Ly.ak.q, F->q35_mt_xq, F->q35_mt_x, F->q35_mt_k, nullptr, 1.0f, Ly.ak.type, E, KV};
    j[1] = {Ly.av.q, F->q35_mt_xq, F->q35_mt_x, F->q35_mt_v, nullptr, 1.0f, Ly.av.type, E, KV};
    if (!gf::mvn(j, 2, n, s)) { err = "qwen35 mtp: k / v rows"; return false; }
    q35::attn_kv_rows(F->q35_mt_k, F->q35_mt_v, Ly.k_norm, g.norm_eps, g.attn_head_kv, D, g.n_rot, g.rope_base, p0, n,
                      Ly.kc, Ly.vc, s);
    q35_row_done_ = std::max(q35_row_done_, p0 + n - 1);
    return true;
}

// one draft step at position p: the block whole (its row written on the way), the head, the argmax (tok_out, on the
// device) and its probability (conf_out); F->head_x holds the block's normed output for a chained step
bool Glm5Model::q35_mtp_step(int64_t p, const int* tok_dev, const float* h, float* conf_out, int* tok_out, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    if (g.q4x) return q4x_mtp_forward(p, 1, h, 0, tok_dev, true, conf_out, tok_out, err);
    const int il = mtp_il_, E = g.n_embd;
    const auto& Ly = F->L[(size_t) il];
    if (!gf::embed_rows_q(F->emb_tab_d ? pack_emb_type_ : 8 /* Q8_0 */, F->emb_tab_d ? F->emb_tab_d : F->emb_q8_d, tok_dev, 1, E, F->emb, s)) { err = "qwen35 mtp: the embedding"; return false; }
    gf::mtp_in(F->emb, h, Ly.enorm, Ly.hnorm, g.norm_eps, E, F->mtp_catq, s);
    gf::MvJob eh = {Ly.eh.q, F->mtp_catq, nullptr, F->mtp_h, nullptr, 1.0f, Ly.eh.type, 2 * E, E};
    if (!gf::mv(&eh, 1, s)) { err = "qwen35 mtp: eh_proj"; return false; }
    gf::rms_q8(F->mtp_h, nullptr, Ly.attn_norm, g.norm_eps, E, F->x, F->xq, s);
    if (!fast_qwen35_attn(il, p, err)) return false;
    gf::rms_q8(F->mtp_h, F->mixer, Ly.post_norm, g.norm_eps, E, F->x, F->xq, s);
    bool pf_pending = false;
    if (!fast_moe(il, pf_pending, err)) return false;
    gf::rms_q8(F->mtp_h, F->ffn, w_.at("output_norm.weight"), g.norm_eps, E, F->head_x, F->head_xq, s);
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, F->head_xq, F->head_x, F->mtp_logits, nullptr, 1.0f, ow.type, E, g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!gf::mv(&o, 1, s)) { err = "qwen35 mtp: the head"; return false; }
    gf::argmax(F->mtp_logits, g.n_vocab, F->mtp_tok, s);
    q35::top_prob(F->mtp_logits, g.n_vocab, F->mtp_tok, conf_out, s);
    cudaMemcpyAsync(tok_out, F->mtp_tok, sizeof(int), cudaMemcpyDeviceToDevice, s);
    return true;
}

// the drafts after y (the token at position(), pending): up to STRATA_Q35_MTP_K (1..5, default 3) chained steps from
// the pending position, each the block over the previous step's argmax and normed output; a draft's confidence is
// its softmax probability, read after every step (host-mapped) - the chain stops at the first below
// STRATA_Q35_MTP_MIN (default STRATA_DS4_DSPARK_THR, else the decode's 0.5 for this drafter: it drops the rest anyway)
bool Glm5Model::q35_mtp_draft(int32_t y, int32_t* drafts, float* conf, std::string& err) {
    FastState* F = fast_;
    for (int k = 0; k < 5; ++k) {
        drafts[k] = -1;
        conf[k] = 0.0f;
    }
    if (q35_pend_pos_ < 0) return true;   // (no position to draft from)
    // (qwen4exp's draft layer: 4 and 0.75 - prose 59.1 -> 60.5-64.1 tok/s against 3 and 0.5, code the same 81)
    const int kmax0 = [this] {
        const char* v = getenv("STRATA_Q35_MTP_K");
        return v != nullptr ? std::max(1, std::min(5, std::atoi(v))) : g_.q4x ? 4 : 3;
    }();
    // every step writes the KV row at its position: the chain stops where the context ends (the caller's `room` bounds
    // the accepted drafts, not the rows tried; the server's CTX_SLACK of 8 hid the overrun, the CLI had none)
    const int kmax = strata::core::draft_kmax(kmax0, q35_pend_pos_, max_ctx_);
    if (kmax <= 0) {
        q35_pend_pos_ = -1;   // (at the context's end: no draft, and no row for the pending position either)
        return true;
    }
    const float stop_below = [this] {
        const char* v = getenv("STRATA_Q35_MTP_MIN");
        if (v == nullptr) v = getenv("STRATA_DS4_DSPARK_THR");
        return v != nullptr ? (float) std::atof(v) : g_.q4x ? 0.75f : 0.5f;
    }();
    cudaStream_t s = F->cs;
    const int64_t p = q35_pend_pos_;
    if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "qwen35 mtp draft", err)) return false;   // (tok_in_h)
    F->tok_in_h[0] = y;
    // qwen4exp's PLE: the verify window [y, drafts..] at position() reads each column's 16 table rows before its layer
    // 1 (an SSD read each, ~0.3 ms in the engine).  The rows of y and of each draft are prefetched into the row cache
    // as soon as the token is known - a draft step later (~1 ms) they are there; the window reads only the last
    // draft's.  STRATA_Q4X_PLE_PREFETCH=0: off
    static const bool ple_pf_env = [] {
        const char* v = getenv("STRATA_Q4X_PLE_PREFETCH");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const bool ple_pf = ple_pf_env && g_.q4x && F->ple != nullptr && l0_ <= F->ple_il && F->ple_il < l1_ &&
                        (size_t) (pos_ + kmax + 1) <= F->ple_toks.size();
    // (ple_toks is filled only on qwen4exp: read it only when the prefetch runs - qwen35moe's NextN drafter
    // (Ornith, Qwen3.5/3.8-35B) read the empty vector here and crashed at its first draft, v1.3.87 - v1.3.91)
    int32_t pf_tok[8] = {ple_pf && pos_ >= 2 ? F->ple_toks[(size_t) (pos_ - 2)] : -1,
                         ple_pf && pos_ >= 1 ? F->ple_toks[(size_t) (pos_ - 1)] : -1, y, -1, -1, -1, -1, -1};
    const auto ple_prefetch = [&](int t) {   // column t's rows: its token pf_tok[t + 2] after pf_tok[t], pf_tok[t + 1]
        namespace kn = strata::kernels;
        static const kn::PleConsts consts = kn::ple_artifact_consts();
        uint32_t rows[kn::PLE_N_HEADS];
        kn::ngram_rows(&pf_tok[t + 2], &pf_tok[t], 1, consts, rows);
        F->ple->prefetch(rows, kn::PLE_N_HEADS);
    };
    if (ple_pf) ple_prefetch(0);
    int n = 0;
    int toks[5];
    for (int k = 0; k < kmax; ++k) {
        if (!q35_mtp_step(p + k, k == 0 ? F->tok_in_d : g_.q4x ? F->q35_mt_tok + (k - 1) : F->mtp_tok,
                          k == 0 ? F->q35_pend_h : g_.q4x ? q4x_mtp_residual() : F->head_x,
                          F->q35_mt_conf_d, F->q35_mt_tok + k, err))
            return false;
        if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "qwen35 mtp draft step", err)) return false;
        conf[k] = F->q35_mt_conf_h[0];
        n = k + 1;
        if (conf[k] < stop_below) break;
        if (ple_pf && k + 1 < kmax) {   // (a kept draft with a step after it: the window may hold it)
            cudaMemcpy(&toks[k], F->q35_mt_tok + k, sizeof(int), cudaMemcpyDeviceToHost);
            pf_tok[k + 3] = toks[k];
            ple_prefetch(k + 1);
        }
    }
    cudaMemcpy(toks, F->q35_mt_tok, (size_t) n * sizeof(int), cudaMemcpyDeviceToHost);
    if (!F->route_ok(err)) return false;
    for (int k = 0; k < n; ++k) drafts[k] = toks[k];
    q35_row_done_ = std::max(q35_row_done_, p);   // (the step at p wrote its row from the real h and y)
    q35_pend_pos_ = -1;
    return true;
}

// GLM_CB_DIR seam dumps under llama.cpp's cb names (<name>-<layer>.f32; with GLM_CB_POS=1 <name>-<layer>@<pos>.f32,
// else every position overwrites): synchronous, debug only - the diff script bisects the two engines by op
static int64_t q35_dump_pos = 0;
static void q35_dump(cudaStream_t s, const char* name, int il, const float* dev, int64_t n) {
    static const char* dir = getenv("GLM_CB_DIR");
    static const bool per_pos = getenv("GLM_CB_POS") != nullptr;
    if (dir == nullptr || !dir[0]) return;
    cudaStreamSynchronize(s);
    std::vector<float> buf((size_t) n);
    cudaMemcpy(buf.data(), dev, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost);
    const std::string fn = std::string(dir) + "/" + name + "-" + std::to_string(il) +
                           (per_pos ? "@" + std::to_string(q35_dump_pos) : "") + ".f32";
    if (FILE* f = std::fopen(fn.c_str(), "wb")) {
        std::fwrite(buf.data(), 4, (size_t) n, f);
        std::fclose(f);
    }
}

bool Glm5Model::fast_qwen35_gdn(int il, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, C = g.ssm_conv_dim(), V = g.ssm_inner, S = g.ssm_state, HK = g.ssm_k_heads, HV = g.ssm_v_heads;
    // 1. the projections: qkv (C), z (V), alpha and beta (h_v each) in one launch
    gf::MvJob j[4];
    j[0] = {Ly.qkv.q, F->xq, F->x, F->q35_qkv, nullptr, 1.0f, Ly.qkv.type, E, C};
    j[1] = {Ly.zg.q, F->xq, F->x, F->q35_z, nullptr, 1.0f, Ly.zg.type, E, V};
    j[2] = {Ly.alpha_w.q, F->xq, F->x, F->q35_ab, nullptr, 1.0f, Ly.alpha_w.type, E, HV};
    j[3] = {Ly.beta_w.q, F->xq, F->x, F->q35_ab + HV, nullptr, 1.0f, Ly.beta_w.type, E, HV};
    if (!gf::mv(j, 4, s)) { err = "qwen35moe: the delta-net projections"; return false; }
    if (F->prof_on) F->mark("q35_gdn_mv");
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    if (dumps) {
        q35_dump(s, "linear_attn_qkv_mixed", il, F->q35_qkv, C);
        q35_dump(s, "z", il, F->q35_z, V);
        q35_dump(s, "alpha", il, F->q35_ab, HV);
        q35_dump(s, "beta", il, F->q35_ab + HV, HV);
    }
    // 2. the conv over [state | x] (the state slides), SiLU on the whole output, q / k L2-normed, q scaled by 1/sqrt(S)
    q35::conv_silu_step(Ly.gdn_conv, F->q35_qkv, Ly.conv1d, C, g.d_conv, F->q35_h, s);
    if (dumps) {   // (llama.cpp's q_conv / k_conv are the views before the L2 norm, *_predelta after it)
        q35_dump(s, "conv_output_silu", il, F->q35_h, C);
        q35_dump(s, "q_conv", il, F->q35_h, (int64_t) HK * S);
        q35_dump(s, "k_conv", il, F->q35_h + (size_t) HK * S, (int64_t) HK * S);
        q35_dump(s, "v_conv", il, F->q35_h + (size_t) 2 * HK * S, (int64_t) HV * S);
    }
    static const bool old_step = getenv("STRATA_Q35_GDN_STEP") != nullptr;
    // qwen4exp: the heads' L2 norm, the gates, the recurrence, the output norm and y's q8_1 in one launch
    // (q35::gdn_core, the same bits as the launches below - gdn_core_parity).  STRATA_Q4X_GDN_FUSE=0: those launches.
    // Off by default with HIP (unmeasured there): STRATA_Q4X_GDN_FUSE=1 there
    static const bool gdn_fuse = [] {
        const char* v = getenv("STRATA_Q4X_GDN_FUSE");
#if defined(STRATA_USE_HIP)
        return v != nullptr && std::atoi(v) != 0;
#else
        return v == nullptr || std::atoi(v) != 0;
#endif
    }();
    if (g.q4x && gdn_fuse && !old_step && !dumps && S == 128) {
        q35::gdn_core(Ly.gdn_state, F->q35_h, 1, C, F->q35_ab, F->q35_ab + HV, S, HK, HV, true, g.norm_eps, Ly.dt_bias,
                      Ly.ssm_a, F->q35_z, Ly.ssm_norm, g.norm_eps, F->q35_y, F->q35_yq, s);
        if (F->prof_on) F->mark("q35_gdn");
    } else {
        // the q and k heads L2-normed, q scaled, the gates (softplus(alpha + dt) * A, sigmoid(beta)) in one launch
        // (the gate lands over alpha: F->q35_ab [0, HV), beta's sigmoid over beta: [HV, 2 HV))
        q35::gdn_prep_gates(F->q35_h, 1, C, HK, S, g.norm_eps, F->q35_ab, F->q35_ab + HV, Ly.dt_bias, Ly.ssm_a, HV, s);
        if (dumps) {
            q35_dump(s, "q_conv_predelta", il, F->q35_h, (int64_t) HK * S);
            q35_dump(s, "k_conv_predelta", il, F->q35_h + (size_t) HK * S, (int64_t) HK * S);
        }
        // 3. the recurrence (the state in place), the gated norm, the output projection
        if (dumps) {
            q35_dump(s, "gate", il, F->q35_ab, HV);
            q35_dump(s, "beta_sigmoid", il, F->q35_ab + HV, HV);
        }
        // the recurrence: the prompt path's scan over one row (a block a value head, the state column in registers: the
        // state read and written once, 8 MB a layer) - gdn.hpp's two-pass kernel behind STRATA_Q35_GDN_STEP=1 (the A/B)
        if (old_step) {
            strata::kernels::GdnShapes sh;
            sh.S = S;
            sh.h_k = HK;
            sh.h_v = HV;
            strata::kernels::gdn_step(Ly.gdn_state, F->q35_h, F->q35_h + (size_t) HK * S,
                                      F->q35_h + (size_t) 2 * HK * S, F->q35_ab, F->q35_ab + HV, F->q35_o, sh,
                                      (void*) s);
        } else {
            q35::gdn_scan(Ly.gdn_state, F->q35_h, 1, C, F->q35_ab, F->q35_ab + HV, S, HK, HV, F->q35_o, s);
        }
        if (dumps) q35_dump(s, "gdn_o", il, F->q35_o, V);
        if (g.q4x)   // (qwen4exp: a sigmoid output gate)
            strata::kernels::gdn_out_norm(F->q35_o, F->q35_z, Ly.ssm_norm, F->q35_y, HV, S, g.norm_eps, (void*) s);
        else
            strata::kernels::gdn_out_norm_silu(F->q35_o, F->q35_z, Ly.ssm_norm, F->q35_y, HV, S, g.norm_eps, (void*) s);
        if (F->prof_on) F->mark("q35_gdn");
        if (dumps) q35_dump(s, "final_output", il, F->q35_y, V);
        gf::quantize_q8_1(F->q35_y, F->q35_yq, V, s);
    }
    gf::MvJob o = {Ly.ssm_o.q, F->q35_yq, F->q35_y, F->mixer, nullptr, 1.0f, Ly.ssm_o.type, V, E};
    if (!gf::mv(&o, 1, s)) { err = "qwen35moe: ssm_out"; return false; }
    if (F->prof_on) F->mark("q35_gdn_out");
    if (dumps) q35_dump(s, "linear_attn_out", il, F->mixer, E);
    return true;
}

bool Glm5Model::fast_qwen35_attn(int il, int64_t p64, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, D = g.head_dim, HD = g.n_head * D, KV = g.attn_head_kv * D, p = (int) p64;
    gf::MvJob j[5];
    j[0] = {Ly.aq.q, F->xq, F->x, F->q35_qf, nullptr, 1.0f, Ly.aq.type, E, 2 * HD};
    j[1] = {Ly.ak.q, F->xq, F->x, F->q35_k, nullptr, 1.0f, Ly.ak.type, E, KV};
    j[2] = {Ly.av.q, F->xq, F->x, F->q35_v, nullptr, 1.0f, Ly.av.type, E, KV};
    const bool qsa = g.q4x && Ly.q4_pooled != nullptr;
    if (qsa) {   // (qwen4exp: the indexer's raw key and queries from the same input)
        j[3] = {Ly.q4_ik.q, F->xq, F->x, F->q4_iraw, nullptr, 1.0f, Ly.q4_ik.type, E, g.idx_key};
        j[4] = {Ly.q4_iq.q, F->xq, F->x, F->q4_iq, nullptr, 1.0f, Ly.q4_iq.type, E, g.idx_heads * g.idx_key};
    }
    if (!gf::mv(j, qsa ? 5 : 3, s)) { err = "qwen35moe: the attention projections"; return false; }
    if (F->prof_on) F->mark("q35_attn_mv");
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    if (dumps) {
        q35_dump(s, "Qcur_full", il, F->q35_qf, 2 * HD);
        q35_dump(s, "Kcur_raw", il, F->q35_k, KV);
        q35_dump(s, "Vcur", il, F->q35_v, KV);
    }
    q35::attn_prep(F->q35_qf, F->q35_k, F->q35_v, Ly.q_norm, Ly.k_norm, g.norm_eps, g.n_head, g.attn_head_kv, D, g.n_rot,
                   g.rope_base, p, Ly.kc, Ly.vc, F->q35_q, s);
    namespace k = strata::kernels;
    int width = p + 1;
    // STRATA_Q4X_QSA_DECODE (on for CUDA, off on HIP): the QSA layer's attention reads its cells in place, a few query
    // heads a block (q35::attn_decode_rows), not a gathered copy and attn_decode; the same bits
#if defined(STRATA_USE_HIP)
    static const bool rows_attn = [] { const char* v = getenv("STRATA_Q4X_QSA_DECODE"); return v != nullptr && std::atoi(v) != 0; }();
#else
    static const bool rows_attn = [] { const char* v = getenv("STRATA_Q4X_QSA_DECODE"); return v == nullptr || std::atoi(v) != 0; }();
#endif
    bool done = false;
    if (qsa) {
        const k::QsaShapes sh = k::qsa_real_shapes();
        if (F->q4_step_at != p64) {   // the step record and the rows' positions, once a token
            k::qsa_step_fill(F->q4_step_h, p64, sh);
            for (int h = 0; h < 64; ++h) F->q4_pos_h[h] = p;
            cudaMemcpyAsync(F->q4_step, F->q4_step_h, k::qsa_step_bytes(), cudaMemcpyHostToDevice, s);
            cudaMemcpyAsync(F->q4_pos, F->q4_pos_h, 256, cudaMemcpyHostToDevice, s);
            F->q4_step_at = p64;
        }
        const k::QsaIndexerBuffers ib{Ly.q4_tail, Ly.q4_dead, Ly.q4_pooled, Ly.q4_bpos};
        k::indexer_key_append(F->q4_iraw, F->q4_step + k::kStepPos, 0, Ly.q4_ikn, g.norm_eps, ib, sh, F->q4_cos,
                              F->q4_sin, (void*) s);
        static const bool dense = getenv("STRATA_Q4X_DENSE") != nullptr;   // (A/B: every cell, as below 2,051)
        width = dense ? p + 1 : (int) k::qsa_selection_width(p64 + 1, sh);
        if (width < p + 1) {   // past 2,051 cells: the indexer's selection, gathered
            k::rms_norm_weighted(F->q4_iq, Ly.q4_iqn, g.idx_heads, g.idx_key, g.norm_eps, (void*) s);
            k::rope_neox_apply(F->q4_iq, F->q4_iq, g.idx_heads, g.idx_key, g.n_rot, F->q4_cos, F->q4_sin, F->q4_pos,
                               (void*) s);
            k::qsa_block_scores(Ly.q4_pooled, Ly.q4_dead, F->q4_iq, F->q4_step, 1, F->q4_max_blocks, sh, F->q4_scores,
                                (void*) s);
            k::qsa_block_topk(F->q4_scores, F->q4_step, 1, F->q4_max_blocks, F->q4_cap, sh, F->q4_ids, (void*) s);
            if (!(rows_attn && q35::attn_decode_rows(F->q35_q, Ly.kc, Ly.vc, F->q4_ids, F->q4_cap, F->q4_step, 1, width,
                                                     g.n_head, g.attn_head_kv, D, 1.0f / std::sqrt((float) D),
                                                     F->q35_part, F->q35_ao, s)))
                q35::gather_rows(Ly.kc, Ly.vc, F->q4_ids, width, KV, F->q4_ks, F->q4_vs, s);
            else
                done = true;
        } else if (rows_attn && !dense &&
                   q35::attn_decode_rows(F->q35_q, Ly.kc, Ly.vc, nullptr, 0, F->q4_step, 1, width, g.n_head,
                                         g.attn_head_kv, D, 1.0f / std::sqrt((float) D), F->q35_part, F->q35_ao, s)) {
            done = true;   // (the dense row: every cell, read in place the same way)
        }
    }
    if (!done && width < p + 1)
        q35::attn_decode(F->q35_q, F->q4_ks, F->q4_vs, g.n_head, g.attn_head_kv, D, width, 1.0f / std::sqrt((float) D),
                         F->q35_part, F->q35_ao, s);
    else if (!done)
        q35::attn_decode(F->q35_q, Ly.kc, Ly.vc, g.n_head, g.attn_head_kv, D, p + 1, 1.0f / std::sqrt((float) D),
                         F->q35_part, F->q35_ao, s);
    if (dumps) {
        q35_dump(s, "Qcur", il, F->q35_q, HD);
        q35_dump(s, "attn_pregate", il, F->q35_ao, HD);
    }
    q35::attn_gate(F->q35_ao, F->q35_qf, g.n_head, D, s);
    if (F->prof_on) F->mark("q35_attend");
    if (dumps) q35_dump(s, "attn_gated", il, F->q35_ao, HD);
    gf::quantize_q8_1(F->q35_ao, F->q35_aoq, HD, s);
    gf::MvJob o = {Ly.out.q, F->q35_aoq, F->q35_ao, F->mixer, nullptr, 1.0f, Ly.out.type, HD, E};
    if (!gf::mv(&o, 1, s)) { err = "qwen35moe: attn_output"; return false; }
    if (F->prof_on) F->mark("q35_attn_out");
    if (dumps) q35_dump(s, "attn_output", il, F->mixer, E);
    return true;
}

// qwen4exp: build_hc_mix after the combine - R[c] += out * 2 sigmoid(inj_prev[c] / hc) (out == nullptr: none), xn =
// the grouped norm, lo = silu(down . xn / hc), gate = up . lo, x = mean_c xn[c] sigmoid(gate[c]) (and its q8_1),
// inj_out = inject . xn (nullptr: the head's mixer has none)
bool Glm5Model::q4x_mix(const float* norm, const WSlot& down, const WSlot& up, const WSlot* inject, const float* out,
                        const float* inj_prev, float* inj_out, float* x, void* xq, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int E = g.n_embd, HE = g.hc * E, LR = g.hc_lr;
    // the five launches below in two (q35::hc4_fused: the same bits as CUDA's five - hc4_fused_parity).  RTX 5090,
    // Qwen3.8 ISTA IQ3_S: the mixes 2.98 -> 2.21 ms a token (STRATA_GLM_PROF), token by token ~+8%.
    // STRATA_Q4X_HC_FUSE=0: the five.  Off by default with HIP (unmeasured there, and HIP's mv takes down + inject in
    // one mixed launch, 4 waves a row - another order of sums than the fused kernels'): STRATA_Q4X_HC_FUSE=1 there
    static const bool fuse = [] {
        const char* v = getenv("STRATA_Q4X_HC_FUSE");
#if defined(STRATA_USE_HIP)
        return v != nullptr && std::atoi(v) != 0;
#else
        return v == nullptr || std::atoi(v) != 0;
#endif
    }();
    const bool fused = fuse && xq != nullptr &&
                       q35::hc4_fused_ok(E, g.hc, LR, down.type, up.type, inject != nullptr ? inject->type : -1);
    if (fused) {
        q35::hc4_fused(state_, out, inj_prev, norm, g.norm_eps, E, g.hc, LR, down.q, inject != nullptr ? inject->q : nullptr,
                       up.q, F->q4_lo, inj_out, F->q4_sc, F->q4_xn, F->q4_gate, x, xq, s,
                       inject != nullptr && inject->type == gf::kTypeF32);
        return true;
    }
    return q4x_mix5(norm, down, up, inject, out, inj_prev, inj_out, x, xq, err);
}

// q4x_mix in five launches (the reference hc4_fused matches)
bool Glm5Model::q4x_mix5(const float* norm, const WSlot& down, const WSlot& up, const WSlot* inject, const float* out,
                         const float* inj_prev, float* inj_out, float* x, void* xq, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int E = g.n_embd, HE = g.hc * E, LR = g.hc_lr;
    q35::hc4_combine_norm(state_, out, inj_prev, g.hc, norm, g.norm_eps, E, g.hc, 1, F->q4_xn, s, F->q4_xnq);
    gf::MvJob j[2];
    j[0] = {down.q, F->q4_xnq, F->q4_xn, F->q4_lo, nullptr, 1.0f, down.type, HE, LR};
    if (inject != nullptr) j[1] = {inject->q, F->q4_xnq, F->q4_xn, inj_out, nullptr, 1.0f, inject->type, HE, g.hc};
    if (!gf::mv(j, inject != nullptr ? 2 : 1, s)) { err = "qwen4exp: the hyper-connection's down / inject"; return false; }
    q35::silu_scale(F->q4_lo, LR, 1.0f / (float) g.hc, s, F->q4_loq);
    gf::MvJob u = {up.q, F->q4_loq, F->q4_lo, F->q4_gate, nullptr, 1.0f, up.type, LR, HE};
    if (!gf::mv(&u, 1, s)) { err = "qwen4exp: the hyper-connection's up"; return false; }
    q35::hc4_mix(F->q4_xn, F->q4_gate, E, g.hc, 1, x, s, xq);
    return true;
}

bool Glm5Model::fast_qwen35_layers(int64_t p, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    bool pf_pending = false;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    q35_dump_pos = p;
    if (g.q4x) return fast_q4x_layers(p, err);
    if (F->prof_on) F->mark("start");
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        // the residual R (state_) takes the last FFN's output, then x = rms(R) * attn_norm (and its q8_1)
        gf::rms_q8(state_, il == l0_ ? nullptr : F->ffn, Ly.attn_norm, g.norm_eps, g.n_embd, F->x, F->xq, s);
        abl_cap(il, state_, 1, s);
        if (F->prof_on) F->mark("q35_norm");
        if (dumps) {
            if (il > l0_) q35_dump(s, "l_out", il - 1, state_, g.n_embd);
            else q35_dump(s, "inp_embd", il, state_, g.n_embd);
            q35_dump(s, "attn_norm", il, F->x, g.n_embd);
        }
        if (Ly.recr) {
            if (!fast_qwen35_gdn(il, err)) return false;
        } else {
            if (!fast_qwen35_attn(il, p, err)) return false;
        }
        abl_out(il, 0, F->mixer, 1, g.n_embd, s);
        gf::rms_q8(state_, F->mixer, Ly.post_norm, g.norm_eps, g.n_embd, F->x, F->xq, s);
        if (F->prof_on) F->mark("q35_norm");
        if (dumps) {
            q35_dump(s, "attn_residual", il, state_, g.n_embd);
            q35_dump(s, "attn_post_norm", il, F->x, g.n_embd);
        }
        if (!Ly.moe) { err = "qwen35moe: a dense FFN layer (not supported)"; return false; }
        if (!fast_moe(il, pf_pending, err)) return false;
        abl_out(il, 1, F->ffn, 1, g.n_embd, s);
        if (dumps) {
            q35_dump(s, "ffn_shexp_gated", il, F->sh_out, g.n_embd);
            q35_dump(s, "ffn_out", il, F->ffn, g.n_embd);
        }
    }
    // the head: R += the last FFN, x = rms(R) * output_norm -> head_x / head_xq (fast_token runs the output rows)
    const auto nw = w_.find("output_norm.weight");
    if (nw == w_.end()) { err = "qwen35moe: output_norm.weight"; return false; }
    gf::rms_q8(state_, F->ffn, nw->second, g.norm_eps, g.n_embd, F->head_x, F->head_xq, s);
    if (l1_ == g.n_layers) abl_cap(g.n_layers, state_, 1, s);
    if (F->prof_on) F->mark("q35_head");
    if (dumps) {
        q35_dump(s, "l_out", l1_ - 1, state_, g.n_embd);
        q35_dump(s, "result_norm", l1_ - 1, F->head_x, g.n_embd);
    }
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { err = std::string("qwen35moe: ") + cudaGetErrorString(e); return false; } }
    return true;
}

// qwen4exp's layers (llama.cpp src/models/qwen4exp.cpp graph): the residual state_ is hc streams; a layer is
// mix(attn) -> the mixer -> combine -> mix(ffn) -> the MoE -> combine (the FFN's combine runs inside the next mix);
// the head is the output mixer (no output norm)
bool Glm5Model::fast_q4x_layers(int64_t p, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    bool pf_pending = false;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    if (F->prof_on) F->mark("start");
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        bool combined = il == l0_;
        if (il == F->ple_il) {   // the last FFN's combine, then the PLE into the streams, then the mixer reads R
            if (!combined)
                q35::hc4_combine_norm(state_, F->ffn, F->q4_inj[1], g.hc, Ly.hc4[0].norm, g.norm_eps, g.n_embd, g.hc, 1,
                                      F->q4_xn, s);
            combined = true;
            if (!q4x_ple_apply(err)) return false;
            abl_res(state_, g.hc, g.n_embd, s);   // (the PLE's rows are a residual write too, as V4.1's Engram)
            if (F->prof_on) F->mark("q4x_ple");
            if (dumps) q35_dump(s, "ple_out", il, state_, (int64_t) g.hc * g.n_embd);
        }
        if (!q4x_mix(Ly.hc4[0].norm, Ly.hc4[0].down, Ly.hc4[0].up, &Ly.hc4[0].inject, combined ? nullptr : F->ffn,
                     F->q4_inj[1], F->q4_inj[0], F->x, F->xq, err))
            return false;
        abl_cap(il, state_, g.hc, s);   // (--abl-build: the residual entering the layer, the last FFN combined)
        if (F->prof_on) F->mark("q4x_hc");
        if (dumps) {
            q35_dump(s, "hc_norm", il, F->q4_xn, (int64_t) g.hc * g.n_embd);
            q35_dump(s, "hc_gate", il, F->q4_gate, (int64_t) g.hc * g.n_embd);
            q35_dump(s, "hc_mixed", il, F->x, g.n_embd);
            q35_dump(s, "hc_inject", il, F->q4_inj[0], g.hc);
        }
        if (Ly.recr) {
            if (!fast_qwen35_gdn(il, err)) return false;
        } else {
            if (!fast_qwen35_attn(il, p, err)) return false;
        }
        abl_out(il, 0, F->mixer, 1, g.n_embd, s);
        if (!q4x_mix(Ly.hc4[1].norm, Ly.hc4[1].down, Ly.hc4[1].up, &Ly.hc4[1].inject, F->mixer, F->q4_inj[0],
                     F->q4_inj[1], F->x, F->xq, err))
            return false;
        if (F->prof_on) F->mark("q4x_hc");
        if (dumps) {   // (the FFN mixer's names a second time in llama.cpp's graph: cb_diff.py reads <name>2)
            q35_dump(s, "hc_combine", il, state_, (int64_t) g.hc * g.n_embd);
            q35_dump(s, "hc_norm2", il, F->q4_xn, (int64_t) g.hc * g.n_embd);
            q35_dump(s, "hc_gate2", il, F->q4_gate, (int64_t) g.hc * g.n_embd);
            q35_dump(s, "hc_mixed2", il, F->x, g.n_embd);
            q35_dump(s, "hc_inject2", il, F->q4_inj[1], g.hc);
        }
        if (!Ly.moe) { err = "qwen4exp: a dense FFN layer (not supported)"; return false; }
        if (!fast_moe(il, pf_pending, err)) return false;
        abl_out(il, 1, F->ffn, 1, g.n_embd, s);
        if (dumps) q35_dump(s, "ffn_out", il, F->ffn, g.n_embd);
    }
    // the head: the last FFN's combine, then the output mixer -> head_x / head_xq
    if (!q4x_mix(F->q4_head[0], F->q4_head_w[0], F->q4_head_w[1], nullptr, F->ffn, F->q4_inj[1], nullptr, F->head_x,
                 F->head_xq, err))
        return false;
    if (l1_ == g.n_layers) abl_cap(g.n_layers, state_, g.hc, s);
    if (F->prof_on) F->mark("q35_head");
    if (dumps) {
        q35_dump(s, "l_last", l1_ - 1, state_, (int64_t) g.hc * g.n_embd);
        q35_dump(s, "result_norm", l1_ - 1, F->head_x, g.n_embd);
    }
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { err = std::string("qwen4exp: ") + cudaGetErrorString(e); return false; } }
    return true;
}

}  // namespace strata::core
