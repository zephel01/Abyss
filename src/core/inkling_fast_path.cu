// src/core/inkling_fast_path.cu - Inkling 2 Mini (GGUF arch inkling2_loop) on the qwen35moe fast path (Glm5Model
// members).  The model: 24 physical layers; layers [ink_lo, ink_hi) run ink_passes times with the same weights, every
// pass reading the prelude's output again and its own KV / conv state (48 layer applications a token: 48 "slots").
//
// A layer application (physical il, position ip in the pass's order, slot sl):
//   inputs   the residual h, or (il > 0) the depth attention over the blocks + h with the layer's attn_res_query (one
//            query on a window layer, three - q / k / v - on a full one); in a later pass each input goes through its
//            controller with the previous pass's message (msg = depth attention over that pass's bank)
//   norms    window: attn_norm (once, or a time per input when the messages made them differ); full: q / k / v norms
//   attn     q, k (+ conv), v (+ conv), r and the sink logit from the inputs; full layers: per-head dynamic conv taps
//            from attn_{k,v}_conv_proj; q / k / r L2-clipped (q, r by learned scales), full layers' q / r times the
//            log-scaled position; v rms-normed; softmax(q.k / sqrt(128) + rel bias) with the sink, the sigmoid gate,
//            attn_output, its post conv
//   residual at a block's first layer (ip % ink_block == 0) h joins the blocks and the residual restarts from the
//            attention's output; else h += it
//   ffn      the depth attention with ffn_res_query (+ the ffn controller), ffn_norm, the MoE (fast_moe: sigmoid router
//            + exp_probs_b, 8 of 128, SITU), its post conv, h += it
// The head: the depth attention over the blocks + h with output_res_query, output_norm.  The math follows the model's
// definition as llama.cpp's src/models/inkling2.cpp / inkling2-loop.cpp read it; the kernels are kernels/inkling_fast.cu.
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/inkling_fast.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace ik = strata::kernels::ink;

namespace {
size_t ink_al(size_t b) { return (b + 255u) & ~(size_t) 255u; }
constexpr int kCtrl = 4;   // the loop controllers: q, k, v, ffn
}  // namespace

struct Ink {
    struct Layer {
        bool swa = false;
        WSlot q, k, v, o, gate, rel, sink, kcp, vcp;
        const float *attn_norm = nullptr, *qn = nullptr, *kn = nullptr, *vn = nullptr;
        const float *q_clip = nullptr, *rel_bias = nullptr, *rel_clip = nullptr;
        const float *k_conv = nullptr, *v_conv = nullptr, *post_conv = nullptr, *ffn_conv = nullptr;
        const float *attn_rq = nullptr, *ffn_rq = nullptr, *ffn_norm = nullptr;
        // the loop's (core layers): the message queries [kCtrl][E], the controllers' down (2E -> rank), gain and
        // forget (rank -> E) - F32 rows
        const float* msg_q = nullptr;
        const float *down[kCtrl] = {}, *gain[kCtrl] = {}, *forget[kCtrl] = {};
    };
    std::vector<Layer> L;
    struct Slot {
        uint16_t *kc = nullptr, *vc = nullptr;
        int ring = 0;
        float* conv = nullptr;   // [K | V | attn post | ffn post] channels x 3
    };
    std::vector<Slot> S;
    size_t conv_floats = 0;      // a slot's conv state
    const float *tok_norm = nullptr, *out_norm = nullptr, *out_rq = nullptr;
    // the residuals: h and the blocks live in these (a block's buffer is never written after it joins)
    static constexpr int kBufs = 12;
    float* hb[kBufs] = {};
    // scratch
    float *mixo = nullptr, *msgs = nullptr, *cat = nullptr, *lo = nullptr, *cg = nullptr, *cf = nullptr;
    float *cin[3] = {}, *xn[3] = {};
    void* xq[3] = {};
    float *qf = nullptr, *kf = nullptr, *vf = nullptr, *rf = nullptr, *sk = nullptr, *gl = nullptr, *kd = nullptr,
          *vd = nullptr, *ao = nullptr, *tmp = nullptr, *part = nullptr;
    void* aoq = nullptr;
    void* base = nullptr;
    size_t bytes = 0;
    std::vector<float*> owned;   // (unsloth's looped layout: the gathered message queries)
};

// ---------------------------------------------------------------- weights
void Glm5Model::ink_bind(int il, std::string& missing) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    if (F->ink == nullptr) {
        F->ink = new Ink();
        F->ink->L.assign((size_t) g.n_layers, Ink::Layer{});
    }
    Ink& K = *F->ink;
    auto& Ly = K.L[(size_t) il];
    const std::string P = "blk." + std::to_string(il) + ".";
    const auto f32 = [&](const char* n, bool need = true) -> const float* {
        auto it = w_.find(P + n);
        if (it == w_.end()) {
            if (need) missing += P + n + " (f32) ";
            return nullptr;
        }
        return it->second;
    };
    const auto q = [&](const char* n, int64_t n_in, int64_t n_out) -> WSlot {
        auto it = ws_map_.find(P + n);
        if (it == ws_map_.end() || it->second.type == 0) {   // (an F32 projection: the F32 GEMV)
            const auto f = w_.find(P + n);
            if (f != w_.end()) return WSlot{f->second, gf::kTypeF32, f->second, n_in, n_out};
            const auto h = w16_.find(P + n);   // (BF16 rows the pack keeps BF16)
            if (h != w16_.end()) return WSlot{nullptr, gf::kTypeBF16, h->second, n_in, n_out};
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
    const int E = g.n_embd, D = g.head_dim, H = g.n_head, KV = g.attn_head_kv * D;
    Ly.swa = g.ink_swa[(size_t) il] != 0;
    Ly.q = q("attn_q.weight", E, (int64_t) H * D);
    Ly.k = q("attn_k.weight", E, KV);
    Ly.v = q("attn_v.weight", E, KV);
    Ly.o = q("attn_output.weight", (int64_t) H * D, E);
    Ly.gate = q("attn_gate.weight", E, (int64_t) H * D);
    Ly.rel = q("attn_rel.weight", E, (int64_t) H * ik::kRel);
    Ly.sink = q("attn_sink_proj.weight", E, H);
    Ly.q_clip = f32("attn_q_clip.weight");
    Ly.rel_bias = f32("attn_rel_bias.weight");
    Ly.rel_clip = f32("attn_rel_clip.weight");
    Ly.k_conv = f32("attn_k_conv.weight");
    Ly.v_conv = f32("attn_v_conv.weight");
    Ly.post_conv = f32("attn_post_conv.weight");
    Ly.ffn_conv = f32("ffn_post_conv.weight");
    if (Ly.swa) {
        Ly.attn_norm = f32("attn_norm.weight");
    } else {
        Ly.qn = f32("attn_q_inp_norm.weight");
        Ly.kn = f32("attn_k_inp_norm.weight");
        Ly.vn = f32("attn_v_inp_norm.weight");
        Ly.kcp = q("attn_k_conv_proj.weight", E, (int64_t) ik::kConv * g.attn_head_kv);
        Ly.vcp = q("attn_v_conv_proj.weight", E, (int64_t) ik::kConv * g.attn_head_kv);
    }
    if (il > 0) Ly.attn_rq = f32("attn_res_query.weight");
    Ly.ffn_rq = f32("ffn_res_query.weight");
    Ly.ffn_norm = f32("ffn_norm.weight");
    if (g.ink_passes > 1 && il >= g.ink_lo && il < g.ink_hi) {
        // unsloth's own looped layout (arch inkling2, num_loops; the file mainline llama.cpp does not read) holds the
        // same tensors under other names (checked byte for byte against its Mainline_Support copy): the controllers
        // loop_{attn,k,v,mlp}_{down,gain,forget}, and the message queries as two global tensors - core layer lo + i's
        // rows (q, k, v, ffn) are loop_res_query[2i], loop_kv_res_query[2i], loop_kv_res_query[2i + 1],
        // loop_res_query[2i + 1] (gathered into a buffer of its own here)
        const bool root = w_.count(P + "loop_msg_query.weight") == 0 && w_.count("loop_res_query.weight") != 0;
        const char* nm[kCtrl] = {"q", "k", "v", "ffn"};
        const char* rn[kCtrl] = {"attn", "k", "v", "mlp"};
        for (int c = 0; c < kCtrl; ++c) {
            const std::string b = root ? std::string("loop_") + rn[c] + "_" : std::string("loop_") + nm[c] + "_ctrl_";
            Ly.down[c] = f32((b + "down.weight").c_str());
            Ly.gain[c] = f32((b + "gain.weight").c_str());
            Ly.forget[c] = f32((b + "forget.weight").c_str());
        }
        if (!root) {
            Ly.msg_q = f32("loop_msg_query.weight");
        } else {
            const float* lr = w_.at("loop_res_query.weight");
            const auto kv = w_.find("loop_kv_res_query.weight");
            const size_t E = (size_t) g.n_embd, i = (size_t) (il - g.ink_lo);
            float* d = nullptr;
            if (kv == w_.end() || cudaMalloc(&d, 4 * E * 4) != cudaSuccess) {
                cudaGetLastError();
                missing += "loop_kv_res_query.weight / the message queries' buffer ";
            } else {
                const float* src[4] = {lr + 2 * i * E, kv->second + 2 * i * E, kv->second + (2 * i + 1) * E,
                                       lr + (2 * i + 1) * E};
                for (int r = 0; r < 4; ++r) cudaMemcpy(d + r * E, src[r], E * 4, cudaMemcpyDeviceToDevice);
                K.owned.push_back(d);
                Ly.msg_q = d;
            }
        }
    }
    if (il == 0) {
        const auto gl = [&](const char* n) -> const float* {
            auto it = w_.find(n);
            if (it == w_.end()) {
                missing += std::string(n) + " (f32) ";
                return nullptr;
            }
            return it->second;
        };
        K.tok_norm = gl("token_embd_norm.weight");
        K.out_norm = gl("output_norm.weight");
        K.out_rq = gl("output_res_query.weight");
    }
}

// ---------------------------------------------------------------- the slots and the scratch
bool Glm5Model::ink_alloc(size_t& bytes_out, bool place, uint8_t*& at_out, std::string& err) {
    (void) bytes_out;
    (void) place;
    (void) at_out;
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    if (F->ink == nullptr) {
        err = "inkling: the weights were not bound";
        return false;
    }
    Ink& K = *F->ink;
    if (l0_ != 0 || l1_ != g.n_layers) {
        err = "inkling: a layer split across GPUs is not supported";
        return false;
    }
    const size_t E = (size_t) g.n_embd, D = (size_t) g.head_dim, H = (size_t) g.n_head, KVh = (size_t) g.attn_head_kv;
    const size_t KV = KVh * D, HD = H * D;
    const size_t ctx = (size_t) std::max<int64_t>(max_ctx_, 1);
    const int NS = g.ink_slots();
    K.conv_floats = (2 * KV + 2 * E) * 3;
    const auto ring_of = [&](int il) -> size_t {
        return K.L[(size_t) il].swa ? std::min(ctx, (size_t) g.n_swa + (size_t) Glm5Geometry::kLagSlice) : ctx;
    };
    size_t bytes = 0, kv_bytes = 0;
    for (int sl = 0; sl < NS; ++sl) {
        const size_t r = ring_of(g.ink_phys(sl));
        kv_bytes += 2 * ink_al(r * KV * 2);
        bytes += 2 * ink_al(r * KV * 2) + ink_al(K.conv_floats * 4);
    }
    const size_t T = 1;
    const size_t part = ik::attn_part_floats((int) T, (int) H, ik::kMaxSplits);
    bytes += Ink::kBufs * ink_al(E * 4) + ink_al(4 * E * 4) * 2 + ink_al(6 * E * 4) + ink_al(3 * 256 * 4) + 2 * ink_al(3 * E * 4) +
             ink_al(3 * E * 4) + 3 * ink_al(E * 4) + 3 * ink_al(2 * E / 32 * 36) + ink_al(HD * 4) * 2 + ink_al(KV * 4) * 2 +
             ink_al(H * ik::kRel * 4) + ink_al(H * 4) + ink_al(KVh * ik::kConv * 4) * 2 + ink_al(HD * 4) + ink_al(E * 4) +
             ink_al(part * 4) + ink_al(HD / 32 * 36) + 256;
    if (cudaMalloc(&K.base, bytes) != cudaSuccess) {
        cudaGetLastError();
        err = "inkling: the slots and scratch (" + std::to_string(bytes >> 20) + " MB for " + std::to_string(ctx) +
              " tokens) did not allocate";
        return false;
    }
    cudaMemset(K.base, 0, bytes);
    K.bytes = bytes;
    uint8_t* at = (uint8_t*) K.base;
    const auto take = [&](size_t b) {
        uint8_t* p = at;
        at += ink_al(b);
        return p;
    };
    K.S.assign((size_t) NS, Ink::Slot{});
    for (int sl = 0; sl < NS; ++sl) {
        auto& S = K.S[(size_t) sl];
        S.ring = (int) ring_of(g.ink_phys(sl));
        S.kc = (uint16_t*) take((size_t) S.ring * KV * 2);
        S.vc = (uint16_t*) take((size_t) S.ring * KV * 2);
        S.conv = (float*) take(K.conv_floats * 4);
    }
    for (int i = 0; i < Ink::kBufs; ++i) K.hb[i] = (float*) take(E * 4);
    K.mixo = (float*) take(4 * E * 4);
    K.msgs = (float*) take(4 * E * 4);
    K.cat = (float*) take(3 * 2 * E * 4);
    K.lo = (float*) take(3 * 256 * 4);
    K.cg = (float*) take(3 * E * 4);
    K.cf = (float*) take(3 * E * 4);
    {
        float* cin = (float*) take(3 * E * 4);   // (the three controlled inputs, rows E apart)
        for (int j = 0; j < 3; ++j) {
            K.cin[j] = cin + (size_t) j * E;
            K.xn[j] = (float*) take(E * 4);
            K.xq[j] = take(2 * E / 32 * 36);
        }
    }
    K.qf = (float*) take(HD * 4);
    K.gl = (float*) take(HD * 4);
    K.kf = (float*) take(KV * 4);
    K.vf = (float*) take(KV * 4);
    K.rf = (float*) take(H * ik::kRel * 4);
    K.sk = (float*) take(H * 4);
    K.kd = (float*) take(KVh * ik::kConv * 4);
    K.vd = (float*) take(KVh * ik::kConv * 4);
    K.ao = (float*) take(HD * 4);
    K.tmp = (float*) take(E * 4);
    K.part = (float*) take(part * 4);
    K.aoq = take(HD / 32 * 36);
    qwen35_state_ = K.base;   // (qwen35_reset / the snapshot see a fast-path state)
    std::fprintf(stderr, "inkling: CUDA%d %d layer applications a token (%d physical, layers %d-%d x %d), KV %.2f GB for "
                         "%lld tokens (window layers' rings %lld rows)\n",
                 dev_, NS, g.n_layers, g.ink_lo, g.ink_hi - 1, g.ink_passes, (double) kv_bytes / 1073741824.0,
                 (long long) ctx, (long long) std::min(ctx, (size_t) g.n_swa + (size_t) Glm5Geometry::kLagSlice));
    return true;
}

void Glm5Model::ink_free() {
    if (fast_ == nullptr || fast_->ink == nullptr) return;
    if (fast_->ink->base != nullptr) cudaFree(fast_->ink->base);
    for (float* p : fast_->ink->owned) cudaFree(p);
    if (qwen35_state_ == fast_->ink->base) qwen35_state_ = nullptr;
    delete fast_->ink;
    fast_->ink = nullptr;
}

void Glm5Model::ink_reset() {
    FastState* F = fast_;
    if (F == nullptr || F->ink == nullptr) return;
    for (auto& S : F->ink->S) cudaMemsetAsync(S.conv, 0, F->ink->conv_floats * 4, F->cs);
}

void Glm5Model::ink_snap_runs(std::vector<std::pair<void*, size_t>>& out) const {
    const FastState* F = fast_;
    if (F == nullptr || F->ink == nullptr) return;
    const size_t KV = (size_t) g_.attn_head_kv * g_.head_dim;
    for (size_t sl = 0; sl < F->ink->S.size(); ++sl) {
        const auto& S = F->ink->S[sl];
        out.emplace_back(S.conv, F->ink->conv_floats * 4);
        if (F->ink->L[(size_t) g_.ink_phys((int) sl)].swa) {   // (a window layer's ring: rows a later position overwrites)
            out.emplace_back(S.kc, (size_t) S.ring * KV * 2);
            out.emplace_back(S.vc, (size_t) S.ring * KV * 2);
        }
    }
}

// ---------------------------------------------------------------- the token path
namespace {
struct Blocks {
    const float* p[ik::kMaxStates];
    int n = 0;
};
}  // namespace

bool Glm5Model::fast_ink_layers(int64_t p, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    Ink& K = *F->ink;
    cudaStream_t s = F->cs;
    const int E = g.n_embd, D = g.head_dim, H = g.n_head, Hkv = g.attn_head_kv, KV = Hkv * D, HD = H * D;
    const int R = g.ink_rank;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    (void) dumps;
    if (F->prof_on) F->mark("start");
    if (p == 0) ink_reset();   // (a new sequence: the conv states start at zero; the caches' rows are written first)
    int nb = 0;                // the next free residual buffer
    const auto fresh = [&]() -> float* { return K.hb[nb++ % Ink::kBufs]; };
    // the embedding row (fast_token left it in state_), rms-normed with token_embd_norm
    float* h = fresh();
    ik::rms_rows(state_, E, K.tok_norm, g.norm_eps, E, 1, h, E, s);
    Blocks blocks;
    const auto states = [&](const Blocks& b, const float* cur) {
        ik::States st;
        for (int i = 0; i < b.n; ++i) st.p[i] = b.p[i];
        st.p[b.n] = cur;
        st.n = b.n + 1;
        st.stride = E;
        return st;
    };
    // the controller c of physical layer il: out = control(cur, msg) (cat -> down -> tanh(gain), tanh(forget))
    // controllers c0 .. c0 + n - 1 of physical layer il on n rows at once (cur rows cs apart - 0: one row for all -,
    // msg rows E apart, out rows E apart): out = control(cur, msg) (cat -> down -> tanh(gain), tanh(forget))
    const auto control = [&](int il, int c0, int n, const float* cur, int64_t cs, const float* msg, float* out) -> bool {
        const auto& Ly = K.L[(size_t) il];
        ik::ctrl_cat(cur, cs, msg, E, E, n, K.cat, s);
        gf::MvJob j[8];
        for (int c = 0; c < n; ++c)
            j[c] = {Ly.down[c0 + c], nullptr, K.cat + (size_t) c * 2 * E, K.lo + (size_t) c * R, nullptr, 1.0f,
                    gf::kTypeF32, 2 * E, R};
        if (!gf::mv(j, n, s)) { err = "inkling: a controller's down"; return false; }
        for (int c = 0; c < n; ++c) {
            j[2 * c] = {Ly.gain[c0 + c], nullptr, K.lo + (size_t) c * R, K.cg + (size_t) c * E, nullptr, 1.0f,
                        gf::kTypeF32, R, E};
            j[2 * c + 1] = {Ly.forget[c0 + c], nullptr, K.lo + (size_t) c * R, K.cf + (size_t) c * E, nullptr, 1.0f,
                            gf::kTypeF32, R, E};
        }
        if (!gf::mv(j, 2 * n, s)) { err = "inkling: a controller's gain / forget"; return false; }
        ik::ctrl_apply(cur, cs, msg, E, K.cg, K.cf, E, n, out, E, s);
        return true;
    };
    bool pf_pending = false;
    // one layer application
    const auto layer = [&](int il, int ip, int sl, const float* msgs) -> bool {
        const auto& Ly = K.L[(size_t) il];
        auto& S = K.S[(size_t) sl];
        const bool swa = Ly.swa;
        // ---- the attention's inputs
        const float* in[3] = {h, h, h};
        if (Ly.attn_rq != nullptr) {
            const int nq = swa ? 1 : 3;
            ik::mix(states(blocks, h), Ly.attn_rq, nq, E, 1, K.mixo, s);
            for (int j = 0; j < 3; ++j) in[j] = K.mixo + (size_t) (swa ? 0 : j) * E;
        }
        if (msgs != nullptr) {   // (in[] is h for all, or the mix's rows: one row, or three E apart)
            const int64_t cs = in[1] == in[0] ? 0 : E;
            if (cs != 0 && (in[1] != in[0] + E || in[2] != in[0] + 2 * E)) { err = "inkling: the inputs' rows"; return false; }
            if (!control(il, 0, 3, in[0], cs, msgs, K.cin[0])) return false;
            for (int j = 0; j < 3; ++j) in[j] = K.cin[j];
        }
        const int nx = swa && msgs == nullptr ? 1 : 3;
        {
            const float* gm[3] = {swa ? Ly.attn_norm : Ly.qn, swa ? Ly.attn_norm : Ly.kn, swa ? Ly.attn_norm : Ly.vn};
            for (int j = 0; j < nx; ++j)
                gf::rms_q8(const_cast<float*>(in[j]), nullptr, gm[j], g.norm_eps, E, K.xn[j], K.xq[j], s);
        }
        const int i1 = nx == 1 ? 0 : 1, i2 = nx == 1 ? 0 : 2;
        // ---- the projections
        gf::MvJob j[8];
        int nj = 0;
        j[nj++] = {Ly.q.q, K.xq[0], K.xn[0], K.qf, nullptr, 1.0f, Ly.q.type, E, HD};
        j[nj++] = {Ly.gate.q, K.xq[0], K.xn[0], K.gl, nullptr, 1.0f, Ly.gate.type, E, HD};
        j[nj++] = {Ly.rel.q, K.xq[0], K.xn[0], K.rf, nullptr, 1.0f, Ly.rel.type, E, H * ik::kRel};
        j[nj++] = {Ly.sink.q, K.xq[0], K.xn[0], K.sk, nullptr, 1.0f, Ly.sink.type, E, H};
        j[nj++] = {Ly.k.q, K.xq[i1], K.xn[i1], K.kf, nullptr, 1.0f, Ly.k.type, E, KV};
        j[nj++] = {Ly.v.q, K.xq[i2], K.xn[i2], K.vf, nullptr, 1.0f, Ly.v.type, E, KV};
        if (!swa) {
            j[nj++] = {Ly.kcp.q, K.xq[i1], K.xn[i1], K.kd, nullptr, 1.0f, Ly.kcp.type, E, ik::kConv * Hkv};
            j[nj++] = {Ly.vcp.q, K.xq[i2], K.xn[i2], K.vd, nullptr, 1.0f, Ly.vcp.type, E, ik::kConv * Hkv};
        }
        if (!gf::mv(j, nj, s)) { err = "inkling: the attention projections"; return false; }
        if (F->prof_on) F->mark("ink_proj");
        // ---- conv, clips, the caches
        float* conv = S.conv;
        ik::conv_rows(K.kf, KV, Ly.k_conv, swa ? nullptr : K.kd, ik::kConv * Hkv, D, conv, KV, 1, K.kf, KV, s);
        ik::conv_rows(K.vf, KV, Ly.v_conv, swa ? nullptr : K.vd, ik::kConv * Hkv, D, conv + (size_t) KV * 3, KV, 1, K.vf,
                      KV, s);
        const float lf = swa ? 0.0f : g.ink_log_floor;
        ik::clip_heads(K.qf, HD, Ly.q_clip, D, H, g.ink_qk_clip, p, lf, g.ink_log_alpha, 1, s);
        ik::clip_heads(K.kf, KV, nullptr, D, Hkv, g.ink_qk_clip, p, 0.0f, 0.0f, 1, s);
        ik::clip_heads(K.rf, H * ik::kRel, Ly.rel_clip, ik::kRel, H, g.ink_rel_clip, p, lf, g.ink_log_alpha, 1, s);
        ik::kv_store(K.kf, KV, K.vf, KV, g.norm_eps, Hkv, p, 1, S.ring, S.kc, S.vc, s);
        // ---- attention, gate, output, post conv
        ik::attn(K.qf, HD, K.rf, H * ik::kRel, K.sk, H, Ly.rel_bias, swa ? g.n_swa : g.ink_extent, S.kc, S.vc, S.ring, H,
                 Hkv, p, 1, swa ? g.n_swa : 0, 1.0f / std::sqrt((float) D), K.part, ik::kMaxSplits, K.ao, HD, s);
        ik::gate_sigmoid(K.ao, HD, K.gl, HD, HD, 1, s);
        gf::quantize_q8_1(K.ao, K.aoq, HD, s);
        gf::MvJob o = {Ly.o.q, K.aoq, K.ao, K.tmp, nullptr, 1.0f, Ly.o.type, HD, E};
        if (!gf::mv(&o, 1, s)) { err = "inkling: attn_output"; return false; }
        if (F->prof_on) F->mark("ink_attn");
        const bool starts = ip % g.ink_block == 0;
        float* conv_post = conv + (size_t) 2 * KV * 3;
        if (starts) {   // the residual h joins the blocks; the attention's output starts the next one
            if (blocks.n + 1 >= ik::kMaxStates) { err = "inkling: too many residual blocks"; return false; }
            blocks.p[blocks.n++] = h;
            h = fresh();
            ik::conv_rows(K.tmp, E, Ly.post_conv, nullptr, 0, E, conv_post, E, 1, h, E, s);
        } else {
            ik::conv_rows(K.tmp, E, Ly.post_conv, nullptr, 0, E, conv_post, E, 1, K.tmp, E, s);
            ik::add_rows(h, E, K.tmp, E, E, 1, s);
        }
        // ---- the FFN
        ik::mix(states(blocks, h), Ly.ffn_rq, 1, E, 1, K.mixo, s);
        const float* fin = K.mixo;
        if (msgs != nullptr) {
            if (!control(il, 3, 1, fin, E, msgs + (size_t) 3 * E, K.cin[0])) return false;
            fin = K.cin[0];
        }
        gf::rms_q8(const_cast<float*>(fin), nullptr, Ly.ffn_norm, g.norm_eps, E, F->x, F->xq, s);
        if (!fast_moe(il, pf_pending, err)) return false;
        ik::conv_rows(F->ffn, E, Ly.ffn_conv, nullptr, 0, E, conv + (size_t) (2 * KV + E) * 3, E, 1, K.tmp, E, s);
        ik::add_rows(h, E, K.tmp, E, E, 1, s);
        return true;
    };
    const int lo = g.ink_passes > 1 ? g.ink_lo : g.n_layers, hi = g.ink_passes > 1 ? g.ink_hi : g.n_layers;
    for (int il = 0; il < lo; ++il)
        if (!layer(il, il, il, nullptr)) return false;
    if (lo < hi) {
        float* prelude = h;
        const Blocks prelude_blocks = blocks;
        const float* bank_blocks[ik::kMaxStates];
        int bank_n = 0;
        const float* bank_out = nullptr;
        for (int pass = 0; pass < g.ink_passes; ++pass) {
            if (pass > 0) {
                // the previous pass's bank: its blocks after the core's first, and its output
                bank_n = 0;
                for (int i = prelude_blocks.n + 1; i < blocks.n; ++i) bank_blocks[bank_n++] = blocks.p[i];
                bank_out = h;
                h = prelude;
                blocks = prelude_blocks;
            }
            for (int ip = lo; ip < hi; ++ip) {
                const int sl = pass == 0 ? ip : g.n_layers + (pass - 1) * (hi - lo) + (ip - lo);
                const float* msgs = nullptr;
                if (pass > 0) {
                    ik::States st;
                    for (int i = 0; i < bank_n; ++i) st.p[i] = bank_blocks[i];
                    st.p[bank_n] = bank_out;
                    st.n = bank_n + 1;
                    st.stride = E;
                    ik::mix(st, K.L[(size_t) ip].msg_q, kCtrl, E, 1, K.msgs, s);
                    msgs = K.msgs;
                }
                if (!layer(ip, ip, sl, msgs)) return false;
            }
        }
    }
    for (int il = hi; il < g.n_layers; ++il)
        if (!layer(il, il, il, nullptr)) return false;
    // the head: the depth attention with output_res_query, output_norm -> head_x / head_xq
    ik::mix(states(blocks, h), K.out_rq, 1, E, 1, K.mixo, s);
    gf::rms_q8(K.mixo, nullptr, K.out_norm, g.norm_eps, E, F->head_x, F->head_xq, s);
    if (F->prof_on) F->mark("q35_head");
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("inkling: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

}  // namespace strata::core
