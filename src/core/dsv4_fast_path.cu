// src/core/dsv4_fast_path.cu - the deepseek4 (DeepSeek V4 Flash) pieces of the fast decode path (Glm5Model members).
//
// A deepseek4 layer is the glm5-next fast path's layer with another attention mixer: the mHC read/write halves, the
// router + expert tiers + shared expert and the GEMVs are the same kernels (glm_fast.hpp).  This file holds what
// differs (docs/deepseek4/SPEC.md):
//   * dsv4_alloc: per layer the raw window ring, the compressed caches and the compressors' rings (FP16 caches,
//     F32 rings - llama.cpp's defaults);
//   * dsv4_bind: the per-layer weights;
//   * fast_dsv4_attn: q / kv / compressor projections -> norms, RoPE, ring writes -> (block ends) compression ->
//     q_b and the indexer's q -> per-head norm + RoPE -> (CSA) the indexer's top-k -> attention with sinks ->
//     de-rope -> the grouped output LoRA (wo_a per group of heads, then wo_b);
//   * dsv4_head: the gated hyper-connection head.
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/dsv4_fast.hpp"
#include "strata/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace ds = strata::kernels::dsv4;

namespace {
// the RoPE of a layer kind (llama.cpp's deepseek4 graph: compressed layers use the YaRN-scaled base 160000, the
// window-only layers plain RoPE at freq_base)
ds::Rope layer_rope(const Glm5Geometry& g, bool comp) {
    if (!comp) return ds::make_rope(g.n_rot, g.rope_base, 1.0f, 0.0f, 0, 0.0f, 0.0f);
    return ds::make_rope(g.n_rot, g.comp_rope_base, 1.0f / g.yarn_factor, 1.0f, g.yarn_orig_ctx, g.yarn_beta_fast,
                         g.yarn_beta_slow);
}
size_t al(size_t b) { return (b + 255u) & ~(size_t) 255u; }

void dsv4_dump(cudaStream_t s, const std::string& name, const float* dev, int64_t n) {
    static const char* dir = getenv("GLM_CB_DIR");
    if (!dir || !dir[0]) return;
    cudaStreamSynchronize(s);
    std::vector<float> buf((size_t) n);
    cudaMemcpy(buf.data(), dev, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost);
    if (FILE* f = std::fopen((std::string(dir) + "/" + name + ".f32").c_str(), "wb")) {
        std::fwrite(buf.data(), 4, (size_t) n, f);
        std::fclose(f);
    }
}
}  // namespace

bool Glm5Model::dsv4_bind(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    std::string missing;
    for (int il = l0_; il < l1_; ++il) {
        auto& Ly = F->L[(size_t) il];
        const std::string P = "blk." + std::to_string(il) + ".";
        const auto f32 = [&](const char* n) -> const float* {
            auto it = w_.find(P + n);
            if (it == w_.end()) { missing += P + n + " (f32) "; return nullptr; }
            return it->second;
        };
        const auto q = [&](const char* n) -> WSlot {
            auto it = ws_map_.find(P + n);
            if (it == ws_map_.end() || it->second.type == 0 || !gf::mv_supported(it->second.type)) {
                missing += P + n + " (quant) ";
                return WSlot{};
            }
            return it->second;
        };
        Ly.ratio = g.ratio(il);
        Ly.q_a = q("attn_q_a.weight");
        Ly.q_b = q("attn_q_b.weight");
        Ly.kv = q("attn_kv.weight");
        Ly.out_a = q("attn_output_a.weight");
        Ly.out_b = q("attn_output_b.weight");
        Ly.q_a_norm = f32("attn_q_a_norm.weight");
        Ly.kv_a_norm = f32("attn_kv_a_norm.weight");
        Ly.sinks = f32("attn_sinks.weight");
        if (Ly.ratio != 0) {
            Ly.comp_kv = q("attn_compressor_kv.weight");
            Ly.comp_gate = q("attn_compressor_gate.weight");
            Ly.comp_ape = f32("attn_compressor_ape.weight");
            Ly.comp_norm = f32("attn_compressor_norm.weight");
        }
        if (Ly.ratio == 4) {
            Ly.idx_q_b_q = q("indexer.attn_q_b.weight");
            Ly.idx_comp_kv = q("indexer_compressor_kv.weight");
            Ly.idx_comp_gate = q("indexer_compressor_gate.weight");
            Ly.idx_comp_ape = f32("indexer_compressor_ape.weight");
            Ly.idx_comp_norm = f32("indexer_compressor_norm.weight");
            Ly.idx_proj_f = f32("indexer.proj.weight");
        }
        if (il < g.hash_layers) {
            const auto it = wi32_.find(P + "ffn_gate_tid2eid.weight");
            if (it == wi32_.end()) missing += P + "ffn_gate_tid2eid.weight (i32) ";
            else Ly.tid2eid = it->second;
        }
    }
    if (!missing.empty()) {
        err = "deepseek4: unresolved weights: " + missing.substr(0, 600);
        return false;
    }
    return true;
}

bool Glm5Model::dsv4_alloc(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const size_t D = (size_t) g.head_dim, ctx = (size_t) std::max<int64_t>(max_ctx_, 1);
    // the caches: per layer the window ring; CSA: ctx/4 compressed rows + ctx/4 indexer keys and two rings of 8
    // (+ ds::kCompSlack spare rows each); HCA: ctx/128 compressed rows and a ring of 128 (+ the same)
    size_t bytes = 0;
    const auto layer_bytes = [&](int il) {
        const int r = g.ratio(il);
        size_t b = al((size_t) (g.n_swa + kSpecSlack) * D * 2);
        if (r == 4) {
            b += al((ctx / 4 + 1) * D * 2) + al((ctx / 4 + 1) * (size_t) g.idx_key * 2);
            b += 2 * al((size_t) ds::ring_rows(8) * 2 * D * 4) + 2 * al((size_t) ds::ring_rows(8) * 2 * g.idx_key * 4);
        } else if (r == 128) {
            b += al((ctx / 128 + 1) * D * 2) + 2 * al((size_t) ds::ring_rows(128) * D * 4);
        }
        return b;
    };
    ds_ring_ = g.n_swa + kSpecSlack;
    for (int il = l0_; il < l1_; ++il) bytes += layer_bytes(il);
    // the activations: the compressors' projections (2 x 2D), the attention output and its q8_1, wo_a's output and q8_1,
    // the indexer's selection
    const size_t HD = (size_t) g.n_head * D, OL = (size_t) g.o_groups * g.o_lora;
    bytes += 4 * al(2 * D * 4) + al(HD * 4) + al(HD / 32 * 36) + al(OL * 4) + al(OL / 32 * 36) +
             al((size_t) g.idx_top_k * 4);
    if (cudaMalloc(&dsv4_state_, bytes) != cudaSuccess) {
        cudaGetLastError();
        err = "deepseek4: the caches (" + std::to_string(bytes >> 20) + " MB for " + std::to_string(ctx) +
              " tokens) did not allocate";
        return false;
    }
    dsv4_state_bytes_ = bytes;
    cudaMemset(dsv4_state_, 0, bytes);
    uint8_t* at = (uint8_t*) dsv4_state_;
    const auto take = [&](size_t b) {
        uint8_t* p = at;
        at += al(b);
        return p;
    };
    for (int il = l0_; il < l1_; ++il) {
        auto& Ly = F->L[(size_t) il];
        const int r = g.ratio(il);
        Ly.raw = (uint16_t*) take((size_t) ds_ring_ * D * 2);
        if (r == 4) {
            Ly.comp_cache = (uint16_t*) take((ctx / 4 + 1) * D * 2);
            Ly.idx_cache = (uint16_t*) take((ctx / 4 + 1) * (size_t) g.idx_key * 2);
            Ly.ring_kv = (float*) take((size_t) ds::ring_rows(8) * 2 * D * 4);
            Ly.ring_sc = (float*) take((size_t) ds::ring_rows(8) * 2 * D * 4);
            Ly.iring_kv = (float*) take((size_t) ds::ring_rows(8) * 2 * g.idx_key * 4);
            Ly.iring_sc = (float*) take((size_t) ds::ring_rows(8) * 2 * g.idx_key * 4);
        } else if (r == 128) {
            Ly.comp_cache = (uint16_t*) take((ctx / 128 + 1) * D * 2);
            Ly.ring_kv = (float*) take((size_t) ds::ring_rows(128) * D * 4);
            Ly.ring_sc = (float*) take((size_t) ds::ring_rows(128) * D * 4);
        }
    }
    for (int i = 0; i < 2; ++i) {
        F->ds_ckv[i] = (float*) take(2 * D * 4);
        F->ds_csc[i] = (float*) take(2 * D * 4);
    }
    F->ds_o = (float*) take(HD * 4);
    F->ds_oq = take(HD / 32 * 36);
    F->ds_oa = (float*) take(OL * 4);
    F->ds_oaq = take(OL / 32 * 36);
    F->ds_sel = (int*) take((size_t) g.idx_top_k * 4);
    std::fprintf(stderr, "deepseek4: CUDA%d attention caches %.2f GB for %lld tokens\n", dev_,
                 (double) bytes / 1073741824.0, (long long) ctx);
    return true;
}

bool Glm5Model::fast_dsv4_attn(int il, int64_t p64, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    const auto& Ly = F->L[(size_t) il];
    const int p = (int) p64;
    const int D = g.head_dim, r = Ly.ratio;
    const ds::Rope rope = layer_rope(g, r != 0), crope = layer_rope(g, true);
    const std::string Ls = std::to_string(il);

    // 1. the projections of x: q_a, kv, the compressors (kv and gate), the indexer's weights
    gf::MvJob j[gf::kMaxMvJobs];
    int nj = 0;
    j[nj++] = {Ly.q_a.q, F->xq, F->x, F->qr_raw, nullptr, 1.0f, Ly.q_a.type, g.n_embd, g.q_lora};
    j[nj++] = {Ly.kv.q, F->xq, F->x, F->kv_raw, nullptr, 1.0f, Ly.kv.type, g.n_embd, D};
    const int cw = r == 4 ? 2 * D : D;   // CSA overlaps two windows: twice the width
    if (r != 0) {
        j[nj++] = {Ly.comp_kv.q, F->xq, F->x, F->ds_ckv[0], nullptr, 1.0f, Ly.comp_kv.type, g.n_embd, cw};
        j[nj++] = {Ly.comp_gate.q, F->xq, F->x, F->ds_csc[0], nullptr, 1.0f, Ly.comp_gate.type, g.n_embd, cw};
    }
    if (r == 4) {
        j[nj++] = {Ly.idx_comp_kv.q, F->xq, F->x, F->ds_ckv[1], nullptr, 1.0f, Ly.idx_comp_kv.type, g.n_embd,
                   2 * g.idx_key};
        j[nj++] = {Ly.idx_comp_gate.q, F->xq, F->x, F->ds_csc[1], nullptr, 1.0f, Ly.idx_comp_gate.type, g.n_embd,
                   2 * g.idx_key};
        j[nj++] = {Ly.idx_proj_f, nullptr, F->x, F->iw, nullptr, 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads)),
                   gf::kTypeF32, g.n_embd, g.idx_heads};
    }
    if (!gf::mv(j, nj, s)) { err = "deepseek4: attention projections"; return false; }
    if (F->prof_on) F->mark("ds_proj_mv");

    // 2. norms, RoPE, the window ring and the compressor rings; at a block's end, its compressed rows
    ds::PrepArgs a;
    a.qr_raw = F->qr_raw;
    a.q_a_norm = Ly.q_a_norm;
    a.qr = F->qr;
    a.qr_q = F->qr_q;
    a.q_lora = g.q_lora;
    a.kv_raw = F->kv_raw;
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
        a.comp_kv[a.n_comp] = F->ds_ckv[0];
        a.comp_sc[a.n_comp] = F->ds_csc[0];
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
        a.comp_kv[a.n_comp] = F->ds_ckv[1];
        a.comp_sc[a.n_comp] = F->ds_csc[1];
        ++a.n_comp;
    }
    ds::prep(a, s);
    if (r != 0 && (p + 1) % r == 0) ds::compress(a, s);   // every compressor of the layer ends its block together
    if (F->prof_on) F->mark("ds_prep");

    // 3. q_b (and the indexer's q) from the normed q_a, then the per-head norm and RoPE
    gf::MvJob j2[2];
    int nj2 = 0;
    j2[nj2++] = {Ly.q_b.q, F->qr_q, F->qr, F->q, nullptr, 1.0f, Ly.q_b.type, g.q_lora, g.n_head * D};
    if (r == 4)
        j2[nj2++] = {Ly.idx_q_b_q.q, F->qr_q, F->qr, F->iq, nullptr, 1.0f, Ly.idx_q_b_q.type, g.q_lora,
                     g.idx_heads * g.idx_key};
    if (!gf::mv(j2, nj2, s)) { err = "deepseek4: q_b"; return false; }
    ds::q_prep(F->q, g.n_head, D, g.n_rot, rope, r == 4 ? F->iq : nullptr, g.idx_heads, g.idx_key, crope, p,
               g.norm_eps, s);
    if (F->prof_on) F->mark("ds_q");
    if (dumps) {
        dsv4_dump(s, "qr_norm-" + Ls, F->qr, g.q_lora);
        dsv4_dump(s, "q-" + Ls, F->q, (int64_t) g.n_head * D);
    }

    // 4. the keys: the window, and the compressed rows (HCA: every visible one; CSA: the indexer's top-k)
    const int n_raw = std::min(p + 1, g.n_swa);
    int n_comp = 0;
    const int* sel = nullptr;
    if (r != 0) {
        const int n_vis = (p + 1) / r;
        // the indexer's top-k only once more rows are visible than it keeps: below that every row is selected, and
        // the attention reads them in position order instead (the same set; STRATA_DS4_IDX_ALL=1: the old way)
        static const bool idx_all = getenv("STRATA_DS4_IDX_ALL") != nullptr && std::atoi(getenv("STRATA_DS4_IDX_ALL")) != 0;
        if (r == 4 && n_vis > 0 && (idx_all || n_vis > g.idx_top_k)) {
            ds::indexer(F->iq, Ly.idx_cache, F->iw, g.idx_heads, g.idx_key, n_vis, g.idx_top_k, F->score, F->ds_sel, s);
            if (F->prof_on) F->mark("ds_index");
            n_comp = std::min(n_vis, g.idx_top_k);
            sel = F->ds_sel;
        } else {
            n_comp = n_vis;
        }
    }
    ds::attend(F->q, Ly.raw, ds_ring_, n_raw, Ly.comp_cache, sel, n_comp, Ly.sinks, g.n_head, D, g.n_rot, rope, p,
               F->ds_o, s, 0, F->ds_oq);   // (and its q8_1, wo_a's input)
    if (F->prof_on) F->mark("ds_attend");
    if (dumps) dsv4_dump(s, "attn_derope-" + Ls, F->ds_o, (int64_t) g.n_head * D);

    // 5. the grouped output LoRA: group gi (heads gi*H/G ..) through rows [gi*o_lora, (gi+1)*o_lora) of wo_a, then wo_b
    const int gin = g.n_head / g.o_groups * D;   // a group's input width
    const size_t wrow = gf::row_bytes(Ly.out_a.type, gin);
    if (wrow == 0 || g.o_groups > gf::kMaxMvJobs) { err = "deepseek4: wo_a type"; return false; }
    gf::MvJob jo[gf::kMaxMvJobs];
    for (int gi = 0; gi < g.o_groups; ++gi)
        jo[gi] = {(const uint8_t*) Ly.out_a.q + (size_t) gi * g.o_lora * wrow,
                  (const uint8_t*) F->ds_oq + (size_t) gi * (gin / 32) * 36, F->ds_o + (size_t) gi * gin,
                  F->ds_oa + (size_t) gi * g.o_lora, nullptr, 1.0f, Ly.out_a.type, gin, g.o_lora};
    if (!gf::mv(jo, g.o_groups, s)) { err = "deepseek4: wo_a"; return false; }
    gf::quantize_q8_1(F->ds_oa, F->ds_oaq, g.o_groups * g.o_lora, s);
    gf::MvJob ob = {Ly.out_b.q, F->ds_oaq, F->ds_oa, F->mixer, nullptr, 1.0f, Ly.out_b.type, g.o_groups * g.o_lora,
                    g.n_embd};
    if (!gf::mv(&ob, 1, s)) { err = "deepseek4: wo_b"; return false; }
    if (F->prof_on) F->mark("ds_out_mv");
    return true;
}

bool Glm5Model::dsv4_snap_copy(bool restore) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const size_t D = (size_t) g.head_dim;
    struct Run {
        void* p;
        size_t bytes;
    };
    std::vector<Run> runs;
    if (g.v41) {
        // deepseek41: what later positions OVERWRITE - each layer's window ring and the four KV slots' pending even
        // row (ratio 2: a snapshot at an odd position holds half a pair).  The slots' compressed / index-key rows are
        // written at their position's row only, and Engram's history is kept per position (V41Engram::hist), so the
        // position alone restores those; the HC lag and the index selection are per token.
        for (int il = l0_; il < l1_; ++il) runs.push_back({F->L[(size_t) il].raw, (size_t) ds_ring_ * D * 2});
        for (int sl = 0; sl < 4; ++sl)
            if (F->v41_pend[sl] != nullptr) runs.push_back({F->v41_pend[sl], FastState::kV41PendRing * 2 * D * 4});
    }
    for (int il = l0_; il < l1_ && !g.v41; ++il) {
        const auto& Ly = F->L[(size_t) il];
        runs.push_back({Ly.raw, (size_t) ds_ring_ * D * 2});
        if (Ly.ratio == 4) {
            runs.push_back({Ly.ring_kv, (size_t) ds::ring_rows(8) * 2 * D * 4});
            runs.push_back({Ly.ring_sc, (size_t) ds::ring_rows(8) * 2 * D * 4});
            runs.push_back({Ly.iring_kv, (size_t) ds::ring_rows(8) * 2 * g.idx_key * 4});
            runs.push_back({Ly.iring_sc, (size_t) ds::ring_rows(8) * 2 * g.idx_key * 4});
        } else if (Ly.ratio == 128) {
            runs.push_back({Ly.ring_kv, (size_t) ds::ring_rows(128) * D * 4});
            runs.push_back({Ly.ring_sc, (size_t) ds::ring_rows(128) * D * 4});
        }
    }
    size_t total = 0;
    for (const auto& r : runs) total += r.bytes;
    if (dsv4_snap_ == nullptr && cudaMalloc(&dsv4_snap_, total) != cudaSuccess) {
        cudaGetLastError();
        dsv4_snap_ = nullptr;
        return false;
    }
    uint8_t* at = (uint8_t*) dsv4_snap_;
    for (const auto& r : runs) {
        cudaMemcpyAsync(restore ? r.p : at, restore ? at : r.p, r.bytes, cudaMemcpyDeviceToDevice, F->cs);
        at += r.bytes;
    }
    return true;
}

bool Glm5Model::dsv4_head(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const auto fn = w16_.find("output_hc_fn.weight");
    const auto sc = w_.find("output_hc_scale.weight");
    const auto bs = w_.find("output_hc_base.weight");
    const auto nw = w_.find("output_norm.weight");
    if (fn == w16_.end() || sc == w_.end() || bs == w_.end() || nw == w_.end()) {
        err = "deepseek4: the head's tensors (output_hc_fn BF16, output_hc_scale, output_hc_base, output_norm)";
        return false;
    }
    ds::head_prep(state_, fn->second, sc->second, bs->second, g.hc_eps, nw->second, g.norm_eps, g.n_embd, F->head_x,
                  F->head_xq, F->cs);
    return true;
}

}  // namespace strata::core
