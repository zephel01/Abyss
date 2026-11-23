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
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/glm_batch.hpp"
#include "strata/kernels/inkling_fast.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/core/graph.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace gb = strata::kernels::glmb;
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
        // (STRATA_INK_CTL_BF16: bf[c][0 down / 1 gain / 2 forget] - that tensor's rows now BF16, packed into the first
        // half of its F32 arena span at the same pointer: ink_ctl_pack)
        bool bf[kCtrl][3] = {};
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
    // the verify window's (ink_window, allocated at its first use): the same scratch over kVerifyMax columns, every
    // slot's window inputs to the conv states ([part][column][channels], parts K, V, attn post, ffn post: conv_commit)
    struct Win {
        void* base = nullptr;
        float* hb[kBufs] = {};
        float *mixo = nullptr, *msgs = nullptr, *cat = nullptr, *lo = nullptr, *cg = nullptr, *cf = nullptr;
        float *cin[3] = {}, *xn[3] = {};
        void* xq[3] = {};
        float *qf = nullptr, *gl = nullptr, *kf = nullptr, *vf = nullptr, *rf = nullptr, *sk = nullptr, *kd = nullptr,
              *vd = nullptr, *ao = nullptr, *tmp = nullptr, *part = nullptr;
        void* aoq = nullptr;
        float* mixp = nullptr;       // (mix's partials: the window's own - a graph keeps the address)
        float* save = nullptr;       // [slot][...]
        size_t slot_floats = 0;
        float** states_d = nullptr;  // the slots' conv states (device pointers)
    };
    Win* win = nullptr;
    // the token path in one CUDA graph (STRATA_INK_GRAPH): the position from pos_d; a graph per (full layers' attention
    // splits, window layers', the CPU-lane plan, the tail skip) - the kernel arguments that change between tokens
    int64_t* pos_d = nullptr;
    float* mixp = nullptr;   // (mix_tok's partials)
    // the attention's small F32 projections (rel, sink, the conv taps) beside its Q8 ones: a side stream, fork / join
    cudaStream_t side = nullptr;
    cudaEvent_t ev_fork = nullptr, ev_join = nullptr;
    struct TokGraph {
        CapturedGraph g;
        uint64_t requests = 0;   // (the routes it posts: FastState::expected)
    };
    std::map<std::tuple<int, int, unsigned long long, int, float, int>, TokGraph> graphs;   // (+ the lane: its caches)
    uint64_t graph_caps = 0, graph_runs = 0;
    // the verify window in one CUDA graph too (STRATA_INK_WIN_GRAPH): a graph per (width, the last column's full /
    // window layers' attention splits, the CPU-lane plan, the tail skip, the routes' by-value settings); the first
    // position from pos_d.  bad: its capture failed - that key's windows stay launches
    struct WinGraph {
        CapturedGraph g;
        uint64_t requests = 0;
        bool bad = false;
        int seen = 0;   // (windows of this key launched before its capture)
    };
    std::map<std::tuple<int, int, int, unsigned long long, int, float, uint64_t, int>, WinGraph> wgraphs;   // (+ batch)
    uint64_t wgraph_caps = 0, wgraph_runs = 0, wgraph_eager = 0, wgraph_bad = 0;
    uint64_t graph_inkd = 0;   // (the DSpark taps' accumulators' generation the graphs were captured with)
    void* base = nullptr;
    size_t bytes = 0;
    // the batched decode (--parallel N, STRATA_INK_PARALLEL): N lanes, each one conversation's slots (lane 0's in base,
    // the others' in their own allocation) and, while another lane is the model's, its sequence state (the position,
    // the snapshot's buffer and position, the last argmax) - ink_lane_use swaps them in
    struct Lane {
        std::vector<Slot> S;
        void* mem = nullptr;   // (lanes 1..: the slots' allocation)
        int64_t pos = 0, snap_pos = -1, snap_pend = -1, snap_done = -1, pend_pos = -1, row_done = -1;
        void* snap_h = nullptr;
        int last_tok = -1;
    };
    std::vector<Lane> lanes;
    int lane = 0;
    // the batch window's table (ink_batch_step: a row a lane): pos[8], then per slot kc[8], vc[8], conv[8] - on the device
    // (the window's graph reads it at these addresses) and its pinned host copy
    void* bt_d = nullptr;
    void* bt_h = nullptr;
    int64_t* bt_pos = nullptr;
    uint16_t** bt_kc = nullptr;
    uint16_t** bt_vc = nullptr;
    float** bt_conv = nullptr;
    bool batch = false;          // (ink_window: the batch mode - each column its own lane)
    int64_t batch_pmax = 0;      // (the batch's largest position: the attention's grid)
    std::vector<float*> owned;   // (unsloth's looped layout: the gathered message queries)
    std::set<const void*> packed;   // (the controller tensors ink_ctl_pack turned BF16: never twice)
    int ctl_packed = 0, ctl_kept = 0;
    // the batched prompt's cursor through a chunk's applications (ink_prompt_*): the token path's bookkeeping, over
    // the chunk's rows (each residual buffer T rows of E)
    struct Cursor {
        float* hb[kBufs] = {};
        float* fin = nullptr;    // the last application's head rows
        int nb = 0;
        float* h = nullptr;      // the residual
        float* hin = nullptr;    // (an application's attention input, before its block starts)
        const float* bl[ik::kMaxStates] = {};
        int bn = 0;
        float* prelude = nullptr;
        const float* pbl[ik::kMaxStates] = {};
        int pbn = 0;
        const float* bank[ik::kMaxStates] = {};
        int bank_n = 0;
        const float* bank_out = nullptr;
    } pc;
};

// ---------------------------------------------------------------- weights
namespace {
// STRATA_INK_CTL_BF16: the loop controllers' rows as BF16 (ink_ctl_pack).  Unset: on with CUDA, off with HIP (its
// numbers not checked there yet); 0 keeps the F32 rows, 1 packs them on HIP too
bool ink_ctl_bf16_on() {
    static const bool on = [] {
        const char* v = getenv("STRATA_INK_CTL_BF16");
#if defined(STRATA_USE_HIP)
        return v != nullptr && std::atoi(v) != 0;
#else
        return v == nullptr || std::atoi(v) != 0;
#endif
    }();
    return on;
}
// a controller tensor's GEMV shape: r 0 down (2E -> rank), 1 gain / 2 forget (rank -> E)
void ctl_shape(int r, int E, int R, int& n_in, int& n_out) {
    n_in = r == 0 ? 2 * E : R;
    n_out = r == 0 ? R : E;
}
// the rows a block ctl_mv gets: the down's 64 rows over 32 blocks (one warp a row either way: the same sums)
int ctl_rpb(int n_out) { return n_out <= 256 ? 2 : 8; }
// a controller GEMV call: the BF16 tensors' jobs through ik::ctl_mv, the F32 ones through f32 (mv / glm_mvn) - each
// launch only when it has jobs.  bf[i]: job i's rows are BF16
template <class F32Go>
bool ctl_go(const gf::MvJob* j, const bool* bf, int n, int nc, cudaStream_t s, F32Go&& f32) {
    gf::MvJob jf[gf::kMaxMvJobs];
    ik::CtlJob jb[ik::kMaxCtlJobs];
    int nf = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        if (bf[i]) jb[nb++] = ik::CtlJob{(const uint16_t*) j[i].w, j[i].xf, j[i].y, j[i].n_in, j[i].n_out};
        else jf[nf++] = j[i];
    }
    // STRATA_INK_CTL_SPLIT=2/4/8 (opt-in, another sum order): the down's rows (the calls of <= 256-row jobs) that many
    // warps a row
    static const int split = [] { const char* v = getenv("STRATA_INK_CTL_SPLIT"); return v == nullptr ? 1 : std::atoi(v); }();
    if (nb > 0 && !ik::ctl_mv(jb, nb, nc, ctl_rpb(jb[0].n_out), s, jb[0].n_out <= 256 ? split : 1)) return false;
    if (nf > 0 && !f32(jf, nf)) return false;
    return true;
}
}  // namespace

// The controllers of physical layer il to BF16 where every value is one (STRATA_INK_CTL_BF16): each tensor's high
// halves packed into the first half of its own F32 span (the arena keeps its size; the GEMVs read half the bytes),
// a tensor with any non-zero low half left F32.  STRATA_INK_CTL_CHECK=1: each packed tensor's GEMV (1 and 3 columns)
// against the F32 one on the same random inputs first, bit for bit, said in the log
static void ink_ctl_pack(Ink& K, int il, int E, int R) {
    auto& Ly = K.L[(size_t) il];
    static const bool chk = [] { const char* v = getenv("STRATA_INK_CTL_CHECK"); return v != nullptr && std::atoi(v) != 0; }();
    const size_t most = (size_t) 2 * E * R;
    uint16_t* tmp = nullptr;
    int* bad = nullptr;
    float *xs = nullptr, *ya = nullptr, *yb = nullptr;
    if (cudaMalloc(&tmp, most * 2) != cudaSuccess || cudaMalloc(&bad, sizeof(int)) != cudaSuccess ||
        (chk && (cudaMalloc(&xs, 3 * (size_t) 2 * E * 4) != cudaSuccess || cudaMalloc(&ya, 3 * (size_t) E * 4) != cudaSuccess ||
                 cudaMalloc(&yb, 3 * (size_t) E * 4) != cudaSuccess))) {
        cudaGetLastError();
        std::fprintf(stderr, "inkling: blk.%d controllers: no scratch for the BF16 rows - kept F32\n", il);
        cudaFree(tmp); cudaFree(bad); cudaFree(xs); cudaFree(ya); cudaFree(yb);
        return;
    }
    if (chk) {   // (inputs: a fixed LCG in [-1, 1))
        std::vector<float> hx(3 * (size_t) 2 * E);
        uint32_t st = 12345u + (uint32_t) il;
        for (float& v : hx) {
            st = st * 1664525u + 1013904223u;
            v = (float) (st >> 8) / 8388608.0f - 1.0f;
        }
        cudaMemcpy(xs, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
    }
    int kept = 0, packed = 0, same = 0, diff = 0;
    for (int c = 0; c < kCtrl; ++c)
        for (int r = 0; r < 3; ++r) {
            const float* w = r == 0 ? Ly.down[c] : r == 1 ? Ly.gain[c] : Ly.forget[c];
            if (w == nullptr || Ly.bf[c][r]) continue;
            if (K.packed.count(w) != 0) {   // (a tensor another layer shares: packed already)
                Ly.bf[c][r] = true;
                continue;
            }
            int n_in = 0, n_out = 0;
            ctl_shape(r, E, R, n_in, n_out);
            const size_t n = (size_t) n_in * n_out;
            int hb = 0;
            cudaMemset(bad, 0, sizeof(int));
            ik::ctl_pack(w, tmp, (int64_t) n, bad, 0);
            if (cudaMemcpy(&hb, bad, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess || hb != 0) {
                cudaGetLastError();
                ++kept;
                continue;
            }
            if (chk) {
                for (int nc = 1; nc <= 3; nc += 2) {
                    gf::MvJob jf{w, nullptr, xs, ya, nullptr, 1.0f, gf::kTypeF32, n_in, n_out};
                    const ik::CtlJob jb{tmp, xs, yb, n_in, n_out};
                    cudaMemset(ya, 0xff, 3 * (size_t) E * 4);
                    cudaMemset(yb, 0x7f, 3 * (size_t) E * 4);
                    const bool ok = (nc == 1 ? gf::mv(&jf, 1, 0) : gf::mvn_mv(&jf, 1, nc, 0)) &&
                                    ik::ctl_mv(&jb, 1, nc, ctl_rpb(n_out), 0);
                    std::vector<uint32_t> a((size_t) nc * n_out), b2((size_t) nc * n_out);
                    cudaMemcpy(a.data(), ya, a.size() * 4, cudaMemcpyDeviceToHost);
                    cudaMemcpy(b2.data(), yb, b2.size() * 4, cudaMemcpyDeviceToHost);
                    if (ok && a == b2) ++same;
                    else ++diff;
                }
            }
            cudaMemcpy((void*) w, tmp, n * 2, cudaMemcpyDeviceToDevice);
            Ly.bf[c][r] = true;
            K.packed.insert(w);
            ++packed;
        }
    cudaDeviceSynchronize();
    cudaFree(tmp); cudaFree(bad); cudaFree(xs); cudaFree(ya); cudaFree(yb);
    K.ctl_packed += packed;
    K.ctl_kept += kept;
    if (chk)
        std::fprintf(stderr, "inkling: blk.%d controllers' BF16 GEMVs against the F32 ones: %d same, %d differ\n", il, same, diff);
}

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
        if (ink_ctl_bf16_on() && missing.empty()) ink_ctl_pack(K, il, g.n_embd, g.ink_rank);
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
             ink_al(part * 4) + ink_al(HD / 32 * 36) + ink_al(64) + ink_al(ik::mix_tok_floats() * 4) + 256;
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
    K.pos_d = (int64_t*) take(64);
    K.mixp = (float*) take(ik::mix_tok_floats() * 4);
    if (cudaStreamCreateWithFlags(&K.side, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&K.ev_fork, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&K.ev_join, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        K.side = nullptr;   // (no fork: the projections in one call)
    }
    qwen35_state_ = K.base;   // (qwen35_reset / the snapshot see a fast-path state)
    // ---- the batched decode's lanes (--parallel N / auto, STRATA_INK_PARALLEL): lane 0 the slots above, each further
    // lane the same slots in its own allocation, before the expert pool takes what is left.  auto: as many as leave
    // every expert in VRAM with 3 GB to spare (the prompt path's buffers, the verify window, the graphs), at most 8
    {
        size_t lane_b = 0;
        for (int sl = 0; sl < NS; ++sl)
            lane_b += 2 * ink_al((size_t) K.S[(size_t) sl].ring * KV * 2) + ink_al(K.conv_floats * 4);
        const int want = ink_parallel_env();
        int n = want;
        size_t expert_b = 0;
        for (int il = l0_; il < l1_; ++il) {
            const auto& Ly = F->L[(size_t) il];
            if (Ly.moe)
                expert_b += (size_t) g.n_expert * glmfast::expert_stride(Ly.blob, Ly.gu_type, Ly.d_type, g.n_ff_exp, Ly.up_type);
        }
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const size_t spare = (size_t) 3 << 30;
        const int fit = free_b > expert_b + spare ? (int) std::min<size_t>(64, (free_b - expert_b - spare) / std::max<size_t>(1, lane_b)) : 0;
        if (want < 0) n = std::max(1, fit);
        n = std::max(1, std::min(n, kVerifyMax));
        K.lanes.assign(1, Ink::Lane{});
        K.lanes[0].S = K.S;
        for (int j = 1; j < n; ++j) {
            Ink::Lane L;
            if (cudaMalloc(&L.mem, lane_b) != cudaSuccess) {
                cudaGetLastError();
                std::fprintf(stderr, "inkling: CUDA%d only %d of %d parallel lanes allocated (%.2f GB each)\n", dev_, j, n,
                             (double) lane_b / 1073741824.0);
                break;
            }
            cudaMemset(L.mem, 0, lane_b);
            uint8_t* la = (uint8_t*) L.mem;
            const auto ltake = [&](size_t b) { uint8_t* q = la; la += ink_al(b); return q; };
            L.S.assign((size_t) NS, Ink::Slot{});
            for (int sl = 0; sl < NS; ++sl) {
                auto& S = L.S[(size_t) sl];
                S.ring = K.S[(size_t) sl].ring;
                S.kc = (uint16_t*) ltake((size_t) S.ring * KV * 2);
                S.vc = (uint16_t*) ltake((size_t) S.ring * KV * 2);
                S.conv = (float*) ltake(K.conv_floats * 4);
            }
            K.lanes.push_back(std::move(L));
        }
        ink_par_ = (int) K.lanes.size();
        if (want != 1 || ink_par_ > 1)
            std::fprintf(stderr, "inkling: CUDA%d %d parallel lane%s (%s; %.2f GB of caches each for %lld tokens; every "
                                 "expert in VRAM leaves room for %d)%s\n", dev_, ink_par_, ink_par_ == 1 ? "" : "s",
                         want < 0 ? "auto" : "--parallel", (double) lane_b / 1073741824.0, (long long) ctx, fit,
                         want > fit ? " - the expert pool will not hold every expert" : "");
    }
    if (K.ctl_packed + K.ctl_kept > 0)
        std::fprintf(stderr, "inkling: CUDA%d the loop controllers' rows BF16 (STRATA_INK_CTL_BF16): %d tensors, %d kept F32 "
                             "(a value not BF16)\n", dev_, K.ctl_packed, K.ctl_kept);
    std::fprintf(stderr, "inkling: CUDA%d %d layer applications a token (%d physical, layers %d-%d x %d), KV %.2f GB for "
                         "%lld tokens (window layers' rings %lld rows)\n",
                 dev_, NS, g.n_layers, g.ink_lo, g.ink_hi - 1, g.ink_passes, (double) kv_bytes / 1073741824.0,
                 (long long) ctx, (long long) std::min(ctx, (size_t) g.n_swa + (size_t) Glm5Geometry::kLagSlice));
    return true;
}

void Glm5Model::ink_free() {
    if (fast_ == nullptr || fast_->ink == nullptr) return;
    if (!fast_->ink->graphs.empty() || fast_->ink->graph_caps > 0)
        std::fprintf(stderr, "inkling: CUDA%d token graphs: %llu captured, %llu runs\n", dev_,
                     (unsigned long long) fast_->ink->graph_caps, (unsigned long long) fast_->ink->graph_runs);
    if (fast_->ink->wgraph_caps + fast_->ink->wgraph_eager > 0)
        std::fprintf(stderr, "inkling: CUDA%d window graphs: %llu captured (%llu refused), %llu runs, %llu windows launched\n",
                     dev_, (unsigned long long) fast_->ink->wgraph_caps, (unsigned long long) fast_->ink->wgraph_bad,
                     (unsigned long long) fast_->ink->wgraph_runs, (unsigned long long) fast_->ink->wgraph_eager);
    fast_->ink->graphs.clear();   // (before the buffers they read)
    fast_->ink->wgraphs.clear();
    if (fast_->ink->ev_fork != nullptr) cudaEventDestroy(fast_->ink->ev_fork);
    if (fast_->ink->ev_join != nullptr) cudaEventDestroy(fast_->ink->ev_join);
    if (fast_->ink->side != nullptr) cudaStreamDestroy(fast_->ink->side);
    if (fast_->ink->win != nullptr) {
        if (fast_->ink->win->base != nullptr) cudaFree(fast_->ink->win->base);
        delete fast_->ink->win;
    }
    if (fast_->ink->base != nullptr) cudaFree(fast_->ink->base);
    for (size_t j = 0; j < fast_->ink->lanes.size(); ++j) {   // (the lanes: their slots; the snapshots not live here)
        auto& L = fast_->ink->lanes[j];
        if (L.mem != nullptr) cudaFree(L.mem);
        if ((int) j != fast_->ink->lane && L.snap_h != nullptr && L.snap_h != q35_snap_h_) cudaFreeHost(L.snap_h);
    }
    if (fast_->ink->bt_d != nullptr) cudaFree(fast_->ink->bt_d);
    if (fast_->ink->bt_h != nullptr) cudaFreeHost(fast_->ink->bt_h);
    for (float* p : fast_->ink->owned) cudaFree(p);
    if (qwen35_state_ == fast_->ink->base) qwen35_state_ = nullptr;
    delete fast_->ink;
    fast_->ink = nullptr;
}

void Glm5Model::ink_reset() {
    FastState* F = fast_;
    ink_win_p0_ = -1;   // (a window pending: dropped with the sequence)
    if (F == nullptr || F->ink == nullptr) return;
    for (auto& S : F->ink->S) cudaMemsetAsync(S.conv, 0, F->ink->conv_floats * 4, F->cs);
}

void Glm5Model::ink_snap_runs(std::vector<std::pair<void*, size_t>>& out, std::vector<size_t>* used, int64_t pos) const {
    const FastState* F = fast_;
    if (F == nullptr || F->ink == nullptr) return;
    const size_t KV = (size_t) g_.attn_head_kv * g_.head_dim;
    for (size_t sl = 0; sl < F->ink->S.size(); ++sl) {
        const auto& S = F->ink->S[sl];
        out.emplace_back(S.conv, F->ink->conv_floats * 4);
        if (used) used->push_back(out.back().second);
        if (F->ink->L[(size_t) g_.ink_phys((int) sl)].swa) {   // (a window layer's ring: rows a later position overwrites)
            out.emplace_back(S.kc, (size_t) S.ring * KV * 2);
            out.emplace_back(S.vc, (size_t) S.ring * KV * 2);
            // (the rows a conversation at `pos` has written: position p is row p % ring, so below a full ring only the
            // first pos rows hold anything - a short prompt's snapshot moves those, not the whole 3,072-row ring;
            // the rows past them are written again before any attention reads them)
            if (used) {
                const size_t rows = pos < 0 ? (size_t) S.ring : (size_t) std::min<int64_t>(pos, S.ring);
                used->push_back(rows * KV * 2);
                used->push_back(rows * KV * 2);
            }
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
    // STRATA_INK_FUSE=0: the layer's small kernels one by one (the fused ones compute the same bits)
    static const bool fuse = [] { const char* v = getenv("STRATA_INK_FUSE"); return v == nullptr || std::atoi(v) != 0; }();
    if (F->prof_on) F->mark("start");
    if (p == 0) ink_reset();   // (a new sequence: the conv states start at zero; the caches' rows are written first)
    else ink_win_commit(p);    // (a verify window not settled by spec_rewind: its columns up to p kept)
    inkd_settle(p);            // (the DSpark drafter's: the same columns into its context before this token's taps)
    // STRATA_INK_GRAPH=0: the token's launches one by one.  Default: the whole token (48 layer applications and
    // the head's mix) replayed from one CUDA graph - the same kernels in the same order (the same bits), the position
    // read from a device word.  Not with the profile marks, dumps, prefetch / lookahead, the expert helper or images
    static const bool graph_env = [] { const char* v = getenv("STRATA_INK_GRAPH"); return v == nullptr || std::atoi(v) != 0; }();
    const bool graph = graph_env && fuse && D == ik::kHeadDim && E % 32 == 0 && E <= 4096 && K.pos_d != nullptr &&
                       !F->prof_on && !dumps && F->max_pf == 0 && F->n_ahead == 0 && !F->cur_img &&
                       !(F->expert_helper && F->expert_helper->active) && split_next_ == nullptr &&
                       next_port_ == nullptr && getenv("GLM_DUMP_H") == nullptr;
    const int64_t* pdv = graph ? K.pos_d : nullptr;
    inkd_tok_begin(p, pdv);   // (the drafter's rows deferred: this token's row - from the graph's position word)
    const auto run = [&]() -> bool {
        int nb = 0;                // the next free residual buffer
        int visit = 0;             // (the layer applications in order: the DSpark drafter's taps 5 / 23 / 41)
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
        // a controller's two GEMV launches (fused: ctrl_cat came with the mix, ctrl_apply comes with the norms); the
        // rows BF16 where ink_ctl_pack packed them (ctl_go: the same sums as the F32 GEMV's)
        const auto ctrl_mv = [&](int il, int c0, int n) -> bool {
            const auto& Ly = K.L[(size_t) il];
            const auto f32 = [&](const gf::MvJob* jj, int m) { return gf::mv(jj, m, s); };
            gf::MvJob j[8];
            bool bf[8] = {};
            for (int c = 0; c < n; ++c) {
                j[c] = {Ly.down[c0 + c], nullptr, K.cat + (size_t) c * 2 * E, K.lo + (size_t) c * R, nullptr, 1.0f,
                        gf::kTypeF32, 2 * E, R};
                bf[c] = Ly.bf[c0 + c][0];
            }
            if (!ctl_go(j, bf, n, 1, s, f32)) { err = "inkling: a controller's down"; return false; }
            for (int c = 0; c < n; ++c) {
                j[2 * c] = {Ly.gain[c0 + c], nullptr, K.lo + (size_t) c * R, K.cg + (size_t) c * E, nullptr, 1.0f,
                            gf::kTypeF32, R, E};
                j[2 * c + 1] = {Ly.forget[c0 + c], nullptr, K.lo + (size_t) c * R, K.cf + (size_t) c * E, nullptr, 1.0f,
                                gf::kTypeF32, R, E};
                bf[2 * c] = Ly.bf[c0 + c][1];
                bf[2 * c + 1] = Ly.bf[c0 + c][2];
            }
            if (!ctl_go(j, bf, 2 * n, 1, s, f32)) { err = "inkling: a controller's gain / forget"; return false; }
            return true;
        };
        // the controller c of physical layer il: out = control(cur, msg) (cat -> down -> tanh(gain), tanh(forget))
        // controllers c0 .. c0 + n - 1 of physical layer il on n rows at once (cur rows cs apart - 0: one row for all -,
        // msg rows E apart, out rows E apart): out = control(cur, msg) (cat -> down -> tanh(gain), tanh(forget))
        const auto control = [&](int il, int c0, int n, const float* cur, int64_t cs, const float* msg, float* out) -> bool {
            ik::ctrl_cat(cur, cs, msg, E, E, n, K.cat, s);
            if (!ctrl_mv(il, c0, n)) return false;
            ik::ctrl_apply(cur, cs, msg, E, K.cg, K.cf, E, n, out, E, s);
            return true;
        };

        // fused: the FFN's post conv + residual add rides the next mix that reads h (pend: its operands)
        ik::MixTok pend;
        const auto take_pend = [&](ik::MixTok& mt) {
            if (pend.pre_in == nullptr) return;
            mt.pre_in = pend.pre_in;
            mt.pre_w = pend.pre_w;
            mt.pre_state = pend.pre_state;
            mt.pre_tmp = pend.pre_tmp;
            mt.pre_out = pend.pre_out;
            mt.pre_add = true;
            pend.pre_in = nullptr;
        };
        const auto flush_pend = [&] {   // (a reader of h without a fused mix: the two launches)
            if (pend.pre_in == nullptr) return;
            ik::conv_rows(pend.pre_in, E, pend.pre_w, nullptr, 0, E, pend.pre_state, E, 1, pend.pre_tmp, E, s);
            ik::add_rows(pend.pre_out, E, pend.pre_tmp, E, E, 1, s);
            pend.pre_in = nullptr;
        };
        const bool ftok = fuse && D == ik::kHeadDim && E % 32 == 0 && E <= 4096;
        bool pf_pending = false;
        // one layer application
        const auto layer = [&](int il, int ip, int sl, const float* msgs) -> bool {
            const auto& Ly = K.L[(size_t) il];
            auto& S = K.S[(size_t) sl];
            const bool swa = Ly.swa;
            if (inkd_ != nullptr && (visit == 5 || visit == 23 || visit == 41)) {
                flush_pend();   // (the tap reads the residual entering the application: the deferred add first)
                ink_tap(visit, h, E, 1, nullptr, false, s, true);
            }
            ++visit;
            const float* gm[3] = {swa ? Ly.attn_norm : Ly.qn, swa ? Ly.attn_norm : Ly.kn, swa ? Ly.attn_norm : Ly.vn};
            // ---- the attention's inputs
            const float* in[3] = {h, h, h};
            const bool fin_tok = ftok && Ly.attn_rq != nullptr;
            if (!fin_tok) flush_pend();
            if (fin_tok) {   // the mix + (ctrl_cat, the controllers' GEMVs, ctrl_apply + the norms | the norms)
                ik::MixTok mt;
                mt.part = K.mixp;
                take_pend(mt);
                mt.st = states(blocks, h);
                mt.query = Ly.attn_rq;
                mt.nq = swa ? 1 : 3;
                mt.out = K.mixo;
                mt.eps = g.norm_eps;
                if (msgs != nullptr) {
                    mt.n_cat = 3;
                    mt.cat_cs = swa ? 0 : E;
                    mt.msg = msgs;
                    mt.cat = K.cat;
                    ik::mix_tok(mt, E, s);
                    if (!ctrl_mv(il, 0, 3)) return false;
                    ik::CtrlTok ct;
                    ct.cur = K.mixo;
                    ct.cs = swa ? 0 : E;
                    ct.msg = msgs;
                    ct.g = K.cg;
                    ct.f = K.cf;
                    ct.out = K.cin[0];
                    ct.n = 3;
                    ct.eps = g.norm_eps;
                    for (int j = 0; j < 3; ++j) {
                        ct.gamma[j] = gm[j];
                        ct.x[j] = K.xn[j];
                        ct.xq[j] = K.xq[j];
                    }
                    ik::ctrl_tok(ct, E, s);
                } else {
                    mt.n_norm = swa ? 1 : 3;
                    mt.norm_rs = swa ? 0 : 1;
                    for (int j = 0; j < mt.n_norm; ++j) {
                        mt.gamma[j] = gm[j];
                        mt.x[j] = K.xn[j];
                        mt.xq[j] = K.xq[j];
                    }
                    ik::mix_tok(mt, E, s);
                }
            } else if (Ly.attn_rq != nullptr) {
                const int nq = swa ? 1 : 3;
                ik::mix(states(blocks, h), Ly.attn_rq, nq, E, 1, K.mixo, s);
                for (int j = 0; j < 3; ++j) in[j] = K.mixo + (size_t) (swa ? 0 : j) * E;
            }
            if (!fin_tok && msgs != nullptr) {   // (in[] is h for all, or the mix's rows: one row, or three E apart)
                const int64_t cs = in[1] == in[0] ? 0 : E;
                if (cs != 0 && (in[1] != in[0] + E || in[2] != in[0] + 2 * E)) { err = "inkling: the inputs' rows"; return false; }
                if (!control(il, 0, 3, in[0], cs, msgs, K.cin[0])) return false;
                for (int j = 0; j < 3; ++j) in[j] = K.cin[j];
            }
            const int nx = swa && msgs == nullptr ? 1 : 3;
            if (!fin_tok)
                for (int j = 0; j < nx; ++j)
                    gf::rms_q8(const_cast<float*>(in[j]), nullptr, gm[j], g.norm_eps, E, K.xn[j], K.xq[j], s);
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
            gf::MvJob jq[8], jf[8];
            int nq8 = 0, nf32 = 0;
            for (int i = 0; i < nj; ++i) (j[i].type == gf::kTypeF32 ? jf[nf32++] : jq[nq8++]) = j[i];
            if (ftok && K.side != nullptr && !F->prof_on && nq8 > 0 && nf32 > 0) {
                // (the F32 rows - rel, sink, the conv taps - on the side stream while the Q8 ones run: the same
                // kernels a row, apart)
                cudaEventRecord(K.ev_fork, s);
                cudaStreamWaitEvent(K.side, K.ev_fork, 0);
                if (!gf::mv(jf, nf32, K.side)) { err = "inkling: the attention projections (F32)"; return false; }
                cudaEventRecord(K.ev_join, K.side);
                if (!gf::mv(jq, nq8, s)) { err = "inkling: the attention projections"; return false; }
                cudaStreamWaitEvent(s, K.ev_join, 0);
            } else if (!gf::mv(j, nj, s)) {
                err = "inkling: the attention projections";
                return false;
            }
            if (F->prof_on) F->mark("ink_proj");
            // ---- conv, clips, the caches
            float* conv = S.conv;
            const float lf = swa ? 0.0f : g.ink_log_floor;
            if (fuse && D == ik::kHeadDim) {   // (one launch, the same arithmetic as the six below)
                ik::PostProj pp{K.qf, K.kf, K.vf, K.rf, conv, swa ? nullptr : K.kd, swa ? nullptr : K.vd, Ly.k_conv, Ly.v_conv,
                                Ly.q_clip, Ly.rel_clip, (int64_t) ik::kConv * Hkv, KV, HD, H, Hkv, S.ring, p, g.ink_qk_clip,
                                g.ink_rel_clip, lf, g.ink_log_alpha, g.norm_eps, S.kc, S.vc};
                pp.pd = pdv;
                ik::post_proj(pp, s);
            } else {
                ik::conv_rows(K.kf, KV, Ly.k_conv, swa ? nullptr : K.kd, ik::kConv * Hkv, D, conv, KV, 1, K.kf, KV, s);
                ik::conv_rows(K.vf, KV, Ly.v_conv, swa ? nullptr : K.vd, ik::kConv * Hkv, D, conv + (size_t) KV * 3, KV, 1,
                              K.vf, KV, s);
                ik::clip_heads(K.qf, HD, Ly.q_clip, D, H, g.ink_qk_clip, p, lf, g.ink_log_alpha, 1, s);
                ik::clip_heads(K.kf, KV, nullptr, D, Hkv, g.ink_qk_clip, p, 0.0f, 0.0f, 1, s);
                ik::clip_heads(K.rf, H * ik::kRel, Ly.rel_clip, ik::kRel, H, g.ink_rel_clip, p, lf, g.ink_log_alpha, 1, s);
                ik::kv_store(K.kf, KV, K.vf, KV, g.norm_eps, Hkv, p, 1, S.ring, S.kc, S.vc, s);
            }
            // ---- attention, gate, output, post conv
            if (ftok) {   // (the combine, the gate and the q8_1 in one launch)
                ik::attn_tok(K.qf, K.rf, K.sk, Ly.rel_bias, swa ? g.n_swa : g.ink_extent, S.kc, S.vc, S.ring, H, Hkv, p,
                             pdv, swa ? g.n_swa : 0, 1.0f / std::sqrt((float) D), K.part, ik::kMaxSplits, K.gl, K.ao,
                             K.aoq, s);
            } else {
                ik::attn(K.qf, HD, K.rf, H * ik::kRel, K.sk, H, Ly.rel_bias, swa ? g.n_swa : g.ink_extent, S.kc, S.vc, S.ring,
                         H, Hkv, p, 1, swa ? g.n_swa : 0, 1.0f / std::sqrt((float) D), K.part, ik::kMaxSplits, K.ao, HD, s);
                ik::gate_sigmoid(K.ao, HD, K.gl, HD, HD, 1, s);
                gf::quantize_q8_1(K.ao, K.aoq, HD, s);
            }
            gf::MvJob o = {Ly.o.q, K.aoq, K.ao, K.tmp, nullptr, 1.0f, Ly.o.type, HD, E};
            if (!gf::mv(&o, 1, s)) { err = "inkling: attn_output"; return false; }
            if (F->prof_on) F->mark("ink_attn");
            const bool starts = ip % g.ink_block == 0;
            float* conv_post = conv + (size_t) 2 * KV * 3;
            if (ftok) {   // the post conv (+ the add) and the FFN's mix + (ctrl_cat .. | the norm) in one launch
                ik::MixTok mt;
                mt.part = K.mixp;
                mt.pre_in = K.tmp;
                mt.pre_w = Ly.post_conv;
                mt.pre_state = conv_post;
                mt.pre_tmp = K.tmp;
                if (starts) {
                    if (blocks.n + 1 >= ik::kMaxStates) { err = "inkling: too many residual blocks"; return false; }
                    blocks.p[blocks.n++] = h;
                    h = fresh();
                }
                mt.pre_out = h;
                mt.pre_add = !starts;
                mt.st = states(blocks, h);
                mt.query = Ly.ffn_rq;
                mt.nq = 1;
                mt.out = K.mixo;
                mt.eps = g.norm_eps;
                if (msgs != nullptr) {
                    mt.n_cat = 1;
                    mt.cat_cs = E;
                    mt.msg = msgs + (size_t) 3 * E;
                    mt.cat = K.cat;
                    ik::mix_tok(mt, E, s);
                    if (!ctrl_mv(il, 3, 1)) return false;
                    ik::CtrlTok ct;
                    ct.cur = K.mixo;
                    ct.cs = E;
                    ct.msg = msgs + (size_t) 3 * E;
                    ct.g = K.cg;
                    ct.f = K.cf;
                    ct.out = K.cin[0];
                    ct.n = 1;
                    ct.eps = g.norm_eps;
                    ct.gamma[0] = Ly.ffn_norm;
                    ct.x[0] = F->x;
                    ct.xq[0] = F->xq;
                    ik::ctrl_tok(ct, E, s);
                } else {
                    mt.n_norm = 1;
                    mt.gamma[0] = Ly.ffn_norm;
                    mt.x[0] = F->x;
                    mt.xq[0] = F->xq;
                    ik::mix_tok(mt, E, s);
                }
                if (!fast_moe(il, pf_pending, err)) return false;
                pend.pre_in = F->ffn;
                pend.pre_w = Ly.ffn_conv;
                pend.pre_state = conv + (size_t) (2 * KV + E) * 3;
                pend.pre_tmp = K.tmp;
                pend.pre_out = h;
                return true;
            }
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
                        if (ftok) {
                            ik::MixTok mt;
                            mt.part = K.mixp;
                            take_pend(mt);
                            mt.st = st;
                            mt.query = K.L[(size_t) ip].msg_q;
                            mt.nq = kCtrl;
                            mt.out = K.msgs;
                            ik::mix_tok(mt, E, s);
                        } else {
                            ik::mix(st, K.L[(size_t) ip].msg_q, kCtrl, E, 1, K.msgs, s);
                        }
                        msgs = K.msgs;
                    }
                    if (!layer(ip, ip, sl, msgs)) return false;
                }
            }
        }
        for (int il = hi; il < g.n_layers; ++il)
            if (!layer(il, il, il, nullptr)) return false;
        // the head: the depth attention with output_res_query, output_norm -> head_x / head_xq
        if (ftok) {
            ik::MixTok mt;
            mt.part = K.mixp;
            take_pend(mt);
            mt.st = states(blocks, h);
            mt.query = K.out_rq;
            mt.nq = 1;
            mt.out = K.mixo;
            mt.eps = g.norm_eps;
            mt.n_norm = 1;
            mt.gamma[0] = K.out_norm;
            mt.x[0] = F->head_x;
            mt.xq[0] = F->head_xq;
            ik::mix_tok(mt, E, s);
        } else {
            ik::mix(states(blocks, h), K.out_rq, 1, E, 1, K.mixo, s);
            gf::rms_q8(K.mixo, nullptr, K.out_norm, g.norm_eps, E, F->head_x, F->head_xq, s);
        }
        if (inkd_ != nullptr) ink_tap(48, F->head_x, E, 1, nullptr, false, s, true);   // (the drafter's tap 48)
        if (F->prof_on) F->mark("q35_head");
        return true;
    };
    if (!graph) {
        if (!run()) return false;
    } else {
        // (no paced tier copies from inside the layers - they would join the capture: the boundary issues them at
        // once, as under the KDA graphs, and whatever is queued goes now, behind the work so far)
        F->kda_graph_on = true;
        F->bg_drain(s, SIZE_MAX);
        if (F->copy2 != nullptr) F->bg_drain(s, SIZE_MAX, 3, F->copy2);
        ik::set_pos(K.pos_d, p, s);
        const auto key = std::make_tuple(ik::attn_splits(p, 0, ik::kMaxSplits), ik::attn_splits(p, g.n_swa, ik::kMaxSplits),
                                         F->cpu_plan, fast_skip_from(0), fast_skip_w(0), K.lane);
        if (inkd_acc_gen() != K.graph_inkd) {   // (the drafter's tap buffers moved: the graphs write the old ones)
            K.graphs.clear();
            K.wgraphs.clear();
            K.graph_inkd = inkd_acc_gen();
        }
        auto it = K.graphs.find(key);
        if (it == K.graphs.end()) {
            if (K.graphs.size() >= 96) K.graphs.clear();
            Ink::TokGraph tg;
            const uint64_t e0 = F->expected;
            if (!tg.g.begin(s, err)) return false;
            bool ok;
            {   // (PDL on: the graph keeps the launches' programmatic edges)
                strata::kernels::PdlCapture pc(strata::kernels::pdl_enabled());
                ok = run();
            }
            const cudaError_t ce = cudaGetLastError();
            if (!ok || ce != cudaSuccess) {
                cudaGraph_t junk = nullptr;
                cudaStreamEndCapture(s, &junk);
                if (junk) cudaGraphDestroy(junk);
                F->expected = e0;
                if (err.empty()) err = std::string("inkling: the token graph's capture: ") + cudaGetErrorString(ce);
                return false;
            }
            if (!tg.g.end(s, err)) {
                F->expected = e0;
                return false;
            }
            tg.requests = F->expected - e0;
            F->expected = e0;
            it = K.graphs.emplace(key, std::move(tg)).first;
            ++K.graph_caps;
            graph_cap_ = true;
        }
        if (!it->second.g.launch(s, err)) return false;
        inkd_tok_taps();   // (the taps inside the graph: their rows' bookkeeping on the host)
        F->expected += it->second.requests;
        ++K.graph_runs;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("inkling: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------- the verify window
bool Glm5Model::ink_draft_env() {
    static const bool on = [] {
        const char* v = getenv("STRATA_INK_DRAFT");
        return v != nullptr && std::atoi(v) != 0;
    }();
    return on;
}

void Glm5Model::ink_win_commit(int64_t pos) {
    FastState* F = fast_;
    if (ink_win_p0_ < 0 || F == nullptr || F->ink == nullptr || F->ink->win == nullptr) {
        ink_win_p0_ = -1;
        return;
    }
    const int n = (int) std::max<int64_t>(0, std::min<int64_t>(pos - ink_win_p0_, ink_win_T_));
    ink_win_p0_ = -1;
    Ink::Win& W = *F->ink->win;
    const int KV = g_.attn_head_kv * g_.head_dim;
    ik::conv_commit(W.states_d, (int) F->ink->S.size(), W.save, W.slot_floats, KV, g_.n_embd, kVerifyMax, n, F->cs);
}

// ---------------------------------------------------------------- the batched decode (--parallel)
int Glm5Model::ink_parallel_env() {
    static const int n = [] {
        const char* v = getenv("STRATA_INK_PARALLEL");
        if (v == nullptr || *v == 0) return 1;
        if (std::strcmp(v, "auto") == 0) return -1;
        return std::max(1, std::atoi(v));
    }();
    return n;
}

int64_t Glm5Model::ink_lane_pos(int j) const {
    const Ink* K = fast_ != nullptr ? fast_->ink : nullptr;
    if (K == nullptr || j < 0 || j >= (int) K->lanes.size()) return -1;
    return j == K->lane ? pos_ : K->lanes[(size_t) j].pos;
}

int Glm5Model::ink_lane() const {
    const Ink* K = fast_ != nullptr ? fast_->ink : nullptr;
    return K != nullptr ? K->lane : 0;
}

// the live lane's sequence state back into its Lane (the model's fields hold it while it is the model's)
static void ink_lane_park(Ink& K, int64_t pos, int64_t snap_pos, void* snap_h, int64_t snap_pend, int64_t snap_done,
                          int64_t pend_pos, int64_t row_done, int last_tok) {
    auto& L = K.lanes[(size_t) K.lane];
    L.S = K.S;
    L.pos = pos;
    L.snap_pos = snap_pos;
    L.snap_h = snap_h;
    L.snap_pend = snap_pend;
    L.snap_done = snap_done;
    L.pend_pos = pend_pos;
    L.row_done = row_done;
    L.last_tok = last_tok;
}

bool Glm5Model::ink_lane_use(int j, std::string& err) {
    FastState* F = fast_;
    Ink* K = F != nullptr ? F->ink : nullptr;
    if (K == nullptr || j < 0 || j >= (int) K->lanes.size()) {
        err = "inkling: no lane " + std::to_string(j);
        return false;
    }
    if (j == K->lane) return true;
    if (ink_win_p0_ >= 0) ink_win_commit(pos_);   // (a window pending on the old lane: settled there)
    cudaStreamSynchronize(F->cs);
    ink_lane_park(*K, pos_, snap_pos_, q35_snap_h_, q35_snap_pend_, q35_snap_done_, q35_pend_pos_, q35_row_done_, last_tok_);
    const auto& L = K->lanes[(size_t) j];
    K->S = L.S;
    pos_ = L.pos;
    snap_pos_ = L.snap_pos;
    q35_snap_h_ = L.snap_h;
    q35_snap_pend_ = L.snap_pend;
    q35_snap_done_ = L.snap_done;
    q35_pend_pos_ = L.pend_pos;
    q35_row_done_ = L.row_done;
    last_tok_ = L.last_tok;
    K->lane = j;
    if (K->win != nullptr) {   // (the verify window's conv state table: this lane's)
        std::vector<float*> st(K->S.size());
        for (size_t sl = 0; sl < st.size(); ++sl) st[sl] = K->S[sl].conv;
        cudaMemcpy(K->win->states_d, st.data(), st.size() * sizeof(float*), cudaMemcpyHostToDevice);
    }
    return true;
}

bool Glm5Model::ink_batch_step(const int* lanes, const int32_t* toks, int n, std::string& err) {
    FastState* F = fast_;
    Ink* K = F != nullptr ? F->ink : nullptr;
    if (K == nullptr || n < 2 || n > kVerifyMax || n > (int) K->lanes.size()) {
        err = "inkling batch: 2.." + std::to_string(std::min<int>(kVerifyMax, K ? (int) K->lanes.size() : 0)) + " lanes";
        return false;
    }
    if (inkd_ != nullptr) {
        err = "inkling batch: not with the DSpark drafter";
        return false;
    }
    const int TM = kVerifyMax, NS = (int) K->S.size();
    if (ink_win_p0_ >= 0) ink_win_commit(pos_);   // (a window pending on the live lane: settled before it is parked)
    if (K->bt_d == nullptr) {
        const size_t b = 64 + (size_t) 3 * NS * TM * sizeof(void*);
        if (cudaMalloc(&K->bt_d, b) != cudaSuccess || cudaHostAlloc(&K->bt_h, b, cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            err = "inkling batch: the batch table did not allocate";
            return false;
        }
        cudaMemset(K->bt_d, 0, b);
        std::memset(K->bt_h, 0, b);
        uint8_t* d = (uint8_t*) K->bt_d;
        K->bt_pos = (int64_t*) d;
        K->bt_kc = (uint16_t**) (d + 64);
        K->bt_vc = K->bt_kc + (size_t) NS * TM;
        K->bt_conv = (float**) (K->bt_vc + (size_t) NS * TM);
    }
    ink_lane_park(*K, pos_, snap_pos_, q35_snap_h_, q35_snap_pend_, q35_snap_done_, q35_pend_pos_, q35_row_done_, last_tok_);
    uint8_t* h = (uint8_t*) K->bt_h;
    int64_t* hpos = (int64_t*) h;
    uint16_t** hkc = (uint16_t**) (h + 64);
    uint16_t** hvc = hkc + (size_t) NS * TM;
    float** hconv = (float**) (hvc + (size_t) NS * TM);
    int64_t pmax = 0;
    for (int t = 0; t < n; ++t) {
        const int l = lanes[t];
        if (l < 0 || l >= (int) K->lanes.size()) { err = "inkling batch: lane " + std::to_string(l); return false; }
        for (int u = 0; u < t; ++u)
            if (lanes[u] == l) { err = "inkling batch: lane " + std::to_string(l) + " twice"; return false; }
        const auto& L = K->lanes[(size_t) l];
        if (L.pos + 1 > max_ctx_) { err = "inkling batch: lane " + std::to_string(l) + " is at the context's end"; return false; }
        hpos[t] = L.pos;
        pmax = std::max(pmax, L.pos);
        for (int sl = 0; sl < NS; ++sl) {
            hkc[(size_t) sl * TM + t] = L.S[(size_t) sl].kc;
            hvc[(size_t) sl * TM + t] = L.S[(size_t) sl].vc;
            hconv[(size_t) sl * TM + t] = L.S[(size_t) sl].conv;
        }
    }
    cudaMemcpyAsync(K->bt_d, K->bt_h, 64 + (size_t) 3 * NS * TM * sizeof(void*), cudaMemcpyHostToDevice, F->cs);
    K->batch = true;
    K->batch_pmax = pmax;
    const int64_t live = pos_;
    pos_ = 0;   // (verify() reads the window from pos_: the batch's positions are the table's)
    const bool ok = verify(std::vector<int32_t>(toks, toks + n), err);
    K->batch = false;
    pos_ = live;
    last_tok_ = K->lanes[(size_t) K->lane].last_tok;
    if (!ok) return false;
    for (int t = 0; t < n; ++t) {
        auto& L = K->lanes[(size_t) lanes[t]];
        L.pos += 1;
        if (lanes[t] == K->lane) pos_ = L.pos;
    }
    return true;
}

bool Glm5Model::ink_window(const std::vector<int32_t>& toks, int64_t p0, const InkWinIo& io, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    Ink& K = *F->ink;
    cudaStream_t s = F->cs;
    const int T = (int) toks.size(), TM = kVerifyMax;
    const int E = g.n_embd, D = g.head_dim, H = g.n_head, Hkv = g.attn_head_kv, KV = Hkv * D, HD = H * D;
    const int R = g.ink_rank;
    const int NS = (int) K.S.size();
    const size_t q8e = (size_t) E / 32 * 36, q8h = (size_t) HD / 32 * 36;
    if (D != ik::kHeadDim) { err = "inkling verify: head_dim"; return false; }
    // STRATA_INK_FUSE=0 here too: the window's conv / clips / cache rows as six launches (the same bits)
    static const bool win_fuse = [] { const char* v = getenv("STRATA_INK_FUSE"); return v == nullptr || std::atoi(v) != 0; }();
    if (K.win == nullptr) {   // ---- the window's buffers
        auto* W = new Ink::Win();
        const size_t slot_floats = (size_t) TM * (2 * KV + 2 * E);
        size_t bytes = 0;
        const auto add = [&](size_t b) { bytes += ink_al(b); };
        for (int i = 0; i < Ink::kBufs; ++i) add((size_t) TM * E * 4);
        add((size_t) TM * 3 * E * 4); add((size_t) TM * 4 * E * 4); add((size_t) 3 * TM * 2 * E * 4);
        add((size_t) 3 * TM * 256 * 4); add((size_t) 3 * TM * E * 4); add((size_t) 3 * TM * E * 4);
        for (int j = 0; j < 3; ++j) { add((size_t) TM * E * 4); add((size_t) TM * E * 4); add((size_t) TM * q8e); }
        add((size_t) TM * HD * 4); add((size_t) TM * HD * 4); add((size_t) TM * KV * 4); add((size_t) TM * KV * 4);
        add((size_t) TM * H * ik::kRel * 4); add((size_t) TM * H * 4); add((size_t) TM * Hkv * ik::kConv * 4);
        add((size_t) TM * Hkv * ik::kConv * 4); add((size_t) TM * HD * 4); add((size_t) TM * E * 4);
        add(ik::attn_part_floats(TM, H, ik::kMaxSplits) * 4); add((size_t) TM * q8h);
        add((size_t) NS * slot_floats * 4); add((size_t) NS * sizeof(float*)); add(ik::mix_part_floats(TM, E) * 4);
        if (R > 256) { delete W; err = "inkling verify: the controllers' rank"; return false; }
        if (cudaMalloc(&W->base, bytes) != cudaSuccess) {
            cudaGetLastError();
            delete W;
            err = "inkling verify: the window's buffers (" + std::to_string(bytes >> 20) + " MB)";
            return false;
        }
        cudaMemset(W->base, 0, bytes);
        uint8_t* at = (uint8_t*) W->base;
        const auto take = [&](size_t b) { uint8_t* p = at; at += ink_al(b); return p; };
        for (int i = 0; i < Ink::kBufs; ++i) W->hb[i] = (float*) take((size_t) TM * E * 4);
        W->mixo = (float*) take((size_t) TM * 3 * E * 4);
        W->msgs = (float*) take((size_t) TM * 4 * E * 4);
        W->cat = (float*) take((size_t) 3 * TM * 2 * E * 4);
        W->lo = (float*) take((size_t) 3 * TM * 256 * 4);
        W->cg = (float*) take((size_t) 3 * TM * E * 4);
        W->cf = (float*) take((size_t) 3 * TM * E * 4);
        for (int j = 0; j < 3; ++j) {
            W->cin[j] = (float*) take((size_t) TM * E * 4);
            W->xn[j] = (float*) take((size_t) TM * E * 4);
            W->xq[j] = take((size_t) TM * q8e);
        }
        W->qf = (float*) take((size_t) TM * HD * 4);
        W->gl = (float*) take((size_t) TM * HD * 4);
        W->kf = (float*) take((size_t) TM * KV * 4);
        W->vf = (float*) take((size_t) TM * KV * 4);
        W->rf = (float*) take((size_t) TM * H * ik::kRel * 4);
        W->sk = (float*) take((size_t) TM * H * 4);
        W->kd = (float*) take((size_t) TM * Hkv * ik::kConv * 4);
        W->vd = (float*) take((size_t) TM * Hkv * ik::kConv * 4);
        W->ao = (float*) take((size_t) TM * HD * 4);
        W->tmp = (float*) take((size_t) TM * E * 4);
        W->part = (float*) take(ik::attn_part_floats(TM, H, ik::kMaxSplits) * 4);
        W->aoq = take((size_t) TM * q8h);
        W->save = (float*) take((size_t) NS * slot_floats * 4);
        W->slot_floats = slot_floats;
        W->states_d = (float**) take((size_t) NS * sizeof(float*));
        W->mixp = (float*) take(ik::mix_part_floats(TM, E) * 4);
        std::vector<float*> st((size_t) NS);
        for (int sl = 0; sl < NS; ++sl) st[(size_t) sl] = K.S[(size_t) sl].conv;
        cudaMemcpy(W->states_d, st.data(), st.size() * sizeof(float*), cudaMemcpyHostToDevice);
        K.win = W;
        std::fprintf(stderr, "inkling: verify window CUDA%d, up to %d columns, %.1f MB\n", dev_, TM,
                     (double) bytes / 1048576.0);
    }
    Ink::Win& W = *K.win;
    if (T < 1 || T > TM) { err = "inkling verify: the window's width"; return false; }
    // the batch mode (ink_batch_step): each column its own lane's - its position, caches and conv states from the batch
    // table, every state updated as the token path updates it (nothing to commit later), no drafter taps
    const bool bt = K.batch;
    if (bt && (!win_fuse || inkd_ != nullptr || K.bt_d == nullptr)) {
        err = "inkling batch: needs STRATA_INK_FUSE on and no drafter";
        return false;
    }
    if (!bt) {
        ink_win_commit(p0);   // (an earlier window not settled: its columns before p0 kept)
        if (!inkd_flush()) { err = "ink dspark: the deferred rows' commit"; return false; }
        ink_win_p0_ = p0;
        ink_win_T_ = T;
    }
    const int64_t p_last = bt ? K.batch_pmax : p0 + T - 1;   // (the attention's grid: the largest row position)
    const auto rows_of = [&](int sl) {   // (the batch table's rows of slot sl)
        ik::BatchRows b;
        b.pos = K.bt_pos;
        b.kc = K.bt_kc + (size_t) sl * TM;
        b.vc = K.bt_vc + (size_t) sl * TM;
        b.conv = K.bt_conv + (size_t) sl * TM;
        return b;
    };
    const auto slot_save = [&](int sl, int part) {   // (part 0 K, 1 V, 2 attn post, 3 ffn post)
        const size_t off = part == 0 ? 0 : part == 1 ? (size_t) KV : part == 2 ? (size_t) 2 * KV : (size_t) 2 * KV + E;
        return W.save + (size_t) sl * W.slot_floats + (size_t) TM * off;
    };
    // STRATA_INK_WIN_GRAPH=0: the window's launches one by one.  Default: the window's layers and the head (io.tail)
    // replayed from one CUDA graph a (width, the attention's split counts, the routes' settings) - the same kernels in
    // the same order (the same bits), the first position read from a device word (pos_d, as the token graph's).  Not
    // with the profile marks, dumps, STRATA_INK_FUSE=0, prefetch / lookahead, the expert helper, a hash-routed layer
    // or the experts' column check
    static const bool wg_env = [] { const char* v = getenv("STRATA_INK_WIN_GRAPH"); return v == nullptr || std::atoi(v) != 0; }();
    static const bool wg_dumps = getenv("GLM_CB_DIR") != nullptr || getenv("GLM_DUMP_H") != nullptr ||
                                 (getenv("STRATA_DS4_WIN_COLS_CHECK") != nullptr && std::atoi(getenv("STRATA_DS4_WIN_COLS_CHECK")) != 0);
    bool hashed = false;
    for (const auto& Lf : F->L) hashed = hashed || Lf.tid2eid != nullptr;
    const bool wgraph = wg_env && !wg_dumps && !hashed && win_fuse && K.pos_d != nullptr && !F->prof_on && F->max_pf == 0 &&
                        F->n_ahead == 0 && !F->cur_img && !(F->expert_helper && F->expert_helper->active) &&
                        split_next_ == nullptr && next_port_ == nullptr;
    const int64_t* pdv = wgraph ? K.pos_d : nullptr;
    const auto run = [&]() -> bool {
        int nb = 0;
        const auto fresh = [&]() -> float* { return W.hb[nb++ % Ink::kBufs]; };
        float* h = fresh();
        ik::rms_rows(io.emb, E, K.tok_norm, g.norm_eps, E, T, h, E, s);
        int visit = 0;   // (the drafter's taps, as the token path's)
        Blocks blocks;
        const auto states = [&](const Blocks& b, const float* cur) {
            ik::States st;
            for (int i = 0; i < b.n; ++i) st.p[i] = b.p[i];
            st.p[b.n] = cur;
            st.n = b.n + 1;
            st.stride = E;
            return st;
        };
        // the controllers c0 .. c0 + n - 1 of physical layer il on T columns: controller c's current rows at cur[c]
        // (rows cs apart), its message rows at msg + c E (rows ms apart), out[c] [T][E] - the token path's control(): its
        // mv() calls' jobs (n downs, then 2n gains / forgets) with the columns through glm_mvn
        const auto control = [&](int il, int c0, int n, const float* const* cur, int64_t cs, const float* msg, int64_t ms,
                                 float* const* out) -> bool {
            const auto& Ly = K.L[(size_t) il];
            ik::CtrlRows cr;
            for (int c = 0; c < n; ++c) {
                cr.cur[c] = cur[c];
                cr.out[c] = out[c];
            }
            cr.cs = cs;
            cr.msg = msg;
            cr.ms = ms;
            cr.cat = W.cat;
            cr.cat_c = (int64_t) TM * 2 * E;
            cr.g = W.cg;
            cr.f = W.cf;
            cr.gf_c = (int64_t) TM * E;
            ik::ctrl_cat_n(cr, n, E, T, s);
            const auto f32 = [&](const gf::MvJob* jj, int m) { return glm_mvn(jj, m, T, s); };
            gf::MvJob j[8];
            bool bf[8] = {};
            for (int c = 0; c < n; ++c) {
                j[c] = {Ly.down[c0 + c], nullptr, W.cat + (size_t) c * TM * 2 * E, W.lo + (size_t) c * TM * R, nullptr, 1.0f,
                        gf::kTypeF32, 2 * E, R};
                bf[c] = Ly.bf[c0 + c][0];
            }
            if (!ctl_go(j, bf, n, T, s, f32)) { err = "inkling verify: a controller's down"; return false; }
            for (int c = 0; c < n; ++c) {
                j[2 * c] = {Ly.gain[c0 + c], nullptr, W.lo + (size_t) c * TM * R, W.cg + (size_t) c * TM * E, nullptr, 1.0f,
                            gf::kTypeF32, R, E};
                j[2 * c + 1] = {Ly.forget[c0 + c], nullptr, W.lo + (size_t) c * TM * R, W.cf + (size_t) c * TM * E, nullptr,
                                1.0f, gf::kTypeF32, R, E};
                bf[2 * c] = Ly.bf[c0 + c][1];
                bf[2 * c + 1] = Ly.bf[c0 + c][2];
            }
            if (!ctl_go(j, bf, 2 * n, T, s, f32)) { err = "inkling verify: a controller's gain / forget"; return false; }
            ik::ctrl_apply_n(cr, n, E, T, s);
            return true;
        };
        // fused (STRATA_INK_FUSE, as the token path's): a mix and its norms in two launches over the rows (mix_tok's
        // rows), the post conv + add in the mix after it, the FFN's post conv + add (pend) in the next mix that reads h
        const bool wfuse = win_fuse && E % 32 == 0 && E <= 4096 && T <= 8;
        ik::MixTok pend;
        const auto take_pend = [&](ik::MixTok& mt) {
            if (pend.pre_in == nullptr) return;
            mt.pre_in = pend.pre_in;
            mt.pre_w = pend.pre_w;
            mt.pre_state = pend.pre_state;
            mt.pre_tmp = pend.pre_tmp;
            mt.pre_out = pend.pre_out;
            mt.pre_add = true;
            mt.pre_rows = true;
            mt.pre_bs = pend.pre_bs;
            mt.pre_bso = pend.pre_bso;
            pend.pre_in = nullptr;
        };
        const auto flush_pend = [&] {   // (a reader of h without a fused mix: the two launches)
            if (pend.pre_in == nullptr) return;
            if (pend.pre_bs != nullptr)
                ik::conv_batch(pend.pre_in, E, pend.pre_w, pend.pre_bs, pend.pre_bso, E, T, pend.pre_tmp, E, s);
            else
                ik::conv_win(pend.pre_in, E, pend.pre_w, nullptr, 0, E, pend.pre_state, E, T, pend.pre_tmp, E, s);
            ik::add_rows(pend.pre_out, E, pend.pre_tmp, E, E, T, s);
            pend.pre_in = nullptr;
        };
        const auto layer = [&](int il, int ip, int sl, const float* msgs) -> bool {
            const auto& Ly = K.L[(size_t) il];
            auto& S = K.S[(size_t) sl];
            const bool swa = Ly.swa;
            if (inkd_ != nullptr && (visit == 5 || visit == 23 || visit == 41)) {
                flush_pend();   // (the tap reads the residual entering the application)
                ink_tap(visit, h, E, T, nullptr, false, s);
            }
            ++visit;
            // ---- the attention's inputs: in[j] rows ins apart
            const float* in[3] = {h, h, h};
            int64_t ins = E;
            const bool fin_mix = wfuse && Ly.attn_rq != nullptr && msgs == nullptr;
            const float* gm0[3] = {swa ? Ly.attn_norm : Ly.qn, swa ? Ly.attn_norm : Ly.kn, swa ? Ly.attn_norm : Ly.vn};
            if (!fin_mix) flush_pend();
            if (fin_mix) {   // (the mix and the norms in two launches)
                ik::MixTok mt;
                mt.part = W.mixp;
                take_pend(mt);
                mt.st = states(blocks, h);
                mt.query = Ly.attn_rq;
                mt.nq = swa ? 1 : 3;
                mt.out = W.mixo;
                mt.out_rs = (int64_t) mt.nq * E;
                mt.rows = T;
                mt.eps = g.norm_eps;
                mt.n_norm = swa ? 1 : 3;
                mt.norm_rs = swa ? 0 : 1;
                for (int j = 0; j < mt.n_norm; ++j) {
                    mt.gamma[j] = gm0[j];
                    mt.x[j] = W.xn[j];
                    mt.xq[j] = W.xq[j];
                }
                ik::mix_tok(mt, E, s);
            } else if (Ly.attn_rq != nullptr) {
                const int nq = swa ? 1 : 3;
                ik::mix(states(blocks, h), Ly.attn_rq, nq, E, T, W.mixo, s, W.mixp);
                for (int j = 0; j < 3; ++j) in[j] = W.mixo + (size_t) (swa ? 0 : j) * E;
                ins = (int64_t) nq * E;
            }
            if (msgs != nullptr) {
                float* const outs[3] = {W.cin[0], W.cin[1], W.cin[2]};
                if (!control(il, 0, 3, in, ins, msgs, 4 * (int64_t) E, outs)) return false;
                for (int j = 0; j < 3; ++j) in[j] = W.cin[j];
                ins = E;
            }
            const int nx = swa && msgs == nullptr ? 1 : 3;
            if (!fin_mix) {
                const float* const* gm = gm0;
                for (int j = 0; j < nx; ++j)
                    gf::rms_q8_rows(const_cast<float*>(in[j]), ins, nullptr, 0, gm[j], g.norm_eps, E, W.xn[j], E, W.xq[j],
                                    E / 32, T, s);
            }
            const int i1 = nx == 1 ? 0 : 1, i2 = nx == 1 ? 0 : 2;
            // ---- the projections (k / v straight into the slot's window inputs)
            float* kraw = slot_save(sl, 0);
            float* vraw = slot_save(sl, 1);
            gf::MvJob j[8];
            int nj = 0;
            j[nj++] = {Ly.q.q, W.xq[0], W.xn[0], W.qf, nullptr, 1.0f, Ly.q.type, E, HD};
            j[nj++] = {Ly.gate.q, W.xq[0], W.xn[0], W.gl, nullptr, 1.0f, Ly.gate.type, E, HD};
            j[nj++] = {Ly.rel.q, W.xq[0], W.xn[0], W.rf, nullptr, 1.0f, Ly.rel.type, E, H * ik::kRel};
            j[nj++] = {Ly.sink.q, W.xq[0], W.xn[0], W.sk, nullptr, 1.0f, Ly.sink.type, E, H};
            j[nj++] = {Ly.k.q, W.xq[i1], W.xn[i1], kraw, nullptr, 1.0f, Ly.k.type, E, KV};
            j[nj++] = {Ly.v.q, W.xq[i2], W.xn[i2], vraw, nullptr, 1.0f, Ly.v.type, E, KV};
            if (!swa) {
                j[nj++] = {Ly.kcp.q, W.xq[i1], W.xn[i1], W.kd, nullptr, 1.0f, Ly.kcp.type, E, ik::kConv * Hkv};
                j[nj++] = {Ly.vcp.q, W.xq[i2], W.xn[i2], W.vd, nullptr, 1.0f, Ly.vcp.type, E, ik::kConv * Hkv};
            }
            if (!glm_mvn(j, nj, T, s)) { err = "inkling verify: the attention projections"; return false; }
            // ---- conv (the state read, left), clips, the caches' rows p0 .. p0 + T - 1
            float* conv = S.conv;
            const float lf = swa ? 0.0f : g.ink_log_floor;
            if (win_fuse) {   // (one launch, the same functions as the six below)
                ik::PostProj pp{W.qf, W.kf, W.vf, W.rf, conv, swa ? nullptr : W.kd, swa ? nullptr : W.vd, Ly.k_conv, Ly.v_conv,
                                Ly.q_clip, Ly.rel_clip, (int64_t) ik::kConv * Hkv, KV, HD, H, Hkv, S.ring, p0, g.ink_qk_clip,
                                g.ink_rel_clip, lf, g.ink_log_alpha, g.norm_eps, S.kc, S.vc};
                pp.pd = pdv;
                if (bt) ik::post_proj_batch(pp, rows_of(sl), kraw, vraw, T, s);
                else ik::post_proj_rows(pp, kraw, vraw, T, s);
            } else {
                ik::conv_win(kraw, KV, Ly.k_conv, swa ? nullptr : W.kd, ik::kConv * Hkv, D, conv, KV, T, W.kf, KV, s);
                ik::conv_win(vraw, KV, Ly.v_conv, swa ? nullptr : W.vd, ik::kConv * Hkv, D, conv + (size_t) KV * 3, KV, T, W.vf,
                             KV, s);
                ik::clip_heads(W.qf, HD, Ly.q_clip, D, H, g.ink_qk_clip, p0, lf, g.ink_log_alpha, T, s);
                ik::clip_heads(W.kf, KV, nullptr, D, Hkv, g.ink_qk_clip, p0, 0.0f, 0.0f, T, s);
                ik::clip_heads(W.rf, H * ik::kRel, Ly.rel_clip, ik::kRel, H, g.ink_rel_clip, p0, lf, g.ink_log_alpha, T, s);
                ik::kv_store(W.kf, KV, W.vf, KV, g.norm_eps, Hkv, p0, T, S.ring, S.kc, S.vc, s);
            }
            // ---- attention (a row's keys up to its own position, its own split count), gate, output, post conv
            const ik::BatchRows br = bt ? rows_of(sl) : ik::BatchRows();
            ik::attn(W.qf, HD, W.rf, H * ik::kRel, W.sk, H, Ly.rel_bias, swa ? g.n_swa : g.ink_extent, S.kc, S.vc, S.ring, H,
                     Hkv, bt ? p_last - (T - 1) : p0, T, swa ? g.n_swa : 0, 1.0f / std::sqrt((float) D), W.part,
                     ik::kMaxSplits, W.ao, HD, s, true, bt ? nullptr : pdv, bt ? &br : nullptr);
            ik::gate_sigmoid(W.ao, HD, W.gl, HD, HD, T, s);
            gf::quantize_q8_1(W.ao, W.aoq, T * HD, s);
            float* praw = slot_save(sl, 2);
            gf::MvJob o = {Ly.o.q, W.aoq, W.ao, praw, nullptr, 1.0f, Ly.o.type, HD, E};
            if (!glm_mvn(&o, 1, T, s)) { err = "inkling verify: attn_output"; return false; }
            const bool starts = ip % g.ink_block == 0;
            float* conv_post = conv + (size_t) 2 * KV * 3;
            if (wfuse && msgs == nullptr) {   // the post conv (+ the add), the FFN's mix and its norm in two launches
                ik::MixTok mt;
                mt.part = W.mixp;
                mt.pre_in = praw;
                mt.pre_w = Ly.post_conv;
                mt.pre_state = conv_post;
                mt.pre_tmp = W.tmp;
                mt.pre_rows = true;
                if (bt) {
                    mt.pre_bs = K.bt_conv + (size_t) sl * TM;
                    mt.pre_bso = (int64_t) 2 * KV * 3;
                }
                if (starts) {
                    if (blocks.n + 1 >= ik::kMaxStates) { err = "inkling verify: too many residual blocks"; return false; }
                    blocks.p[blocks.n++] = h;
                    h = fresh();
                }
                mt.pre_out = h;
                mt.pre_add = !starts;
                mt.st = states(blocks, h);
                mt.query = Ly.ffn_rq;
                mt.nq = 1;
                mt.out = W.mixo;
                mt.out_rs = E;
                mt.rows = T;
                mt.eps = g.norm_eps;
                mt.n_norm = 1;
                mt.gamma[0] = Ly.ffn_norm;
                mt.x[0] = io.x;
                mt.xq[0] = io.xq;
                ik::mix_tok(mt, E, s);
                if (!verify_moe(il, 0, T, toks, s, nullptr, nullptr, err)) return false;
                float* fraw = slot_save(sl, 3);
                cudaMemcpyAsync(fraw, io.ffn, (size_t) T * E * sizeof(float), cudaMemcpyDeviceToDevice, s);
                pend.pre_in = fraw;
                pend.pre_w = Ly.ffn_conv;
                pend.pre_state = conv + (size_t) (2 * KV + E) * 3;
                pend.pre_tmp = W.tmp;
                pend.pre_out = h;
                pend.pre_bs = bt ? K.bt_conv + (size_t) sl * TM : nullptr;
                pend.pre_bso = (int64_t) (2 * KV + E) * 3;
                return true;
            }
            if (starts) {
                if (blocks.n + 1 >= ik::kMaxStates) { err = "inkling verify: too many residual blocks"; return false; }
                blocks.p[blocks.n++] = h;
                h = fresh();
                if (bt) ik::conv_batch(praw, E, Ly.post_conv, K.bt_conv + (size_t) sl * TM, (int64_t) 2 * KV * 3, E, T, h, E, s);
                else ik::conv_win(praw, E, Ly.post_conv, nullptr, 0, E, conv_post, E, T, h, E, s);
            } else {
                if (bt) ik::conv_batch(praw, E, Ly.post_conv, K.bt_conv + (size_t) sl * TM, (int64_t) 2 * KV * 3, E, T, W.tmp, E, s);
                else ik::conv_win(praw, E, Ly.post_conv, nullptr, 0, E, conv_post, E, T, W.tmp, E, s);
                ik::add_rows(h, E, W.tmp, E, E, T, s);
            }
            // ---- the FFN
            ik::mix(states(blocks, h), Ly.ffn_rq, 1, E, T, W.mixo, s, W.mixp);
            const float* fin = W.mixo;
            if (msgs != nullptr) {
                const float* cur[1] = {fin};
                float* const outs[1] = {W.cin[0]};
                if (!control(il, 3, 1, cur, E, msgs + (size_t) 3 * E, 4 * (int64_t) E, outs)) return false;
                fin = W.cin[0];
            }
            gf::rms_q8_rows(const_cast<float*>(fin), E, nullptr, 0, Ly.ffn_norm, g.norm_eps, E, io.x, E, io.xq, E / 32, T, s);
            if (!verify_moe(il, 0, T, toks, s, nullptr, nullptr, err)) return false;
            float* fraw = slot_save(sl, 3);
            cudaMemcpyAsync(fraw, io.ffn, (size_t) T * E * sizeof(float), cudaMemcpyDeviceToDevice, s);
            pend.pre_in = fraw;   // (the token path's order: the next reader of h adds it)
            pend.pre_w = Ly.ffn_conv;
            pend.pre_state = conv + (size_t) (2 * KV + E) * 3;
            pend.pre_tmp = W.tmp;
            pend.pre_out = h;
            pend.pre_bs = bt ? K.bt_conv + (size_t) sl * TM : nullptr;
            pend.pre_bso = (int64_t) (2 * KV + E) * 3;
            if (!wfuse) flush_pend();
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
                        flush_pend();   // (bank_out may be the residual the last FFN adds to)
                        ik::mix(st, K.L[(size_t) ip].msg_q, kCtrl, E, T, W.msgs, s, W.mixp);
                        msgs = W.msgs;
                    }
                    if (!layer(ip, ip, sl, msgs)) return false;
                }
            }
        }
        for (int il = hi; il < g.n_layers; ++il)
            if (!layer(il, il, il, nullptr)) return false;
        if (wfuse) {   // the head's mix + output_norm in two launches (the last FFN's add first)
            ik::MixTok mt;
            mt.part = W.mixp;
            take_pend(mt);
            mt.st = states(blocks, h);
            mt.query = K.out_rq;
            mt.nq = 1;
            mt.out = W.mixo;
            mt.out_rs = E;
            mt.rows = T;
            mt.eps = g.norm_eps;
            mt.n_norm = 1;
            mt.gamma[0] = K.out_norm;
            mt.x[0] = io.head_x;
            mt.xq[0] = io.head_xq;
            ik::mix_tok(mt, E, s);
        } else {
            flush_pend();
            ik::mix(states(blocks, h), K.out_rq, 1, E, T, W.mixo, s, W.mixp);
            gf::rms_q8_rows(W.mixo, E, nullptr, 0, K.out_norm, g.norm_eps, E, io.head_x, E, io.head_xq, E / 32, T, s);
        }
        if (inkd_ != nullptr) ink_tap(48, io.head_x, E, T, nullptr, false, s);
        if (io.tail && !io.tail(err)) return false;
        return true;
    };
    if (!wgraph) {
        ++K.wgraph_eager;
        if (!run()) return false;
    } else {
        // (as the token graph: no paced tier copies from inside the layers - whatever is queued goes now)
        F->kda_graph_on = true;
        F->bg_drain(s, SIZE_MAX);
        if (F->copy2 != nullptr) F->bg_drain(s, SIZE_MAX, 3, F->copy2);
        ik::set_pos(K.pos_d, p0, s);   // (the graph's position; a refused key's launches read it too)
        if (inkd_acc_gen() != K.graph_inkd) {   // (the drafter's tap buffers moved: the graphs write the old ones)
            K.graphs.clear();
            K.wgraphs.clear();
            K.graph_inkd = inkd_acc_gen();
        }
        // (the routes' by-value settings, as verify_part_q4x's key: the helper's table, the disk tail skip)
        const uint64_t sig = (uint64_t) (uintptr_t) (F->expert_helper && F->expert_helper->active &&
                                                     !F->expert_helper->failed ? F->expert_helper->owned_d : nullptr) * 2u +
                             (F->tail_skip_disk ? 1u : 0u);
        const int64_t p1 = p_last;
        const auto key = std::make_tuple(T, ik::attn_rows_grid(p1, 0, ik::kMaxSplits), ik::attn_rows_grid(p1, g.n_swa, ik::kMaxSplits),
                                         F->cpu_plan, fast_skip_from(0), fast_skip_w(0), sig, bt ? 1 : 0);
        // (a key's first windows as launches: a capture costs ~13 ms - RTX 5090, ~1,450 nodes - and a replay saves
        // ~0.6, so a width seen a few times only is not captured)
        // STRATA_INK_WIN_CAP_AFTER=<n> (default 6): the windows of a key launched before its capture
        static const int kCapAfter = [] {
            const char* v = getenv("STRATA_INK_WIN_CAP_AFTER");
            return v != nullptr ? std::max(0, std::atoi(v)) : 6;
        }();
        auto it = K.wgraphs.find(key);
        if (it == K.wgraphs.end()) {
            if (K.wgraphs.size() >= 128) K.wgraphs.clear();
            it = K.wgraphs.emplace(key, Ink::WinGraph()).first;
        }
        if (!it->second.bad && !it->second.g.valid() && ++it->second.seen <= kCapAfter) {
            ++K.wgraph_eager;
            if (!run()) return false;
        } else if (!it->second.bad && !it->second.g.valid()) {
            Ink::WinGraph& wg = it->second;
            const uint64_t e0 = F->expected;
            bool ok = wg.g.begin(s, err);
            if (ok) {
                {   // (PDL on: the graph keeps the launches' programmatic edges)
                    strata::kernels::PdlCapture pc(strata::kernels::pdl_enabled());
                    ok = run();
                }
                const cudaError_t ce = cudaGetLastError();
                if (!ok || ce != cudaSuccess) {
                    cudaGraph_t junk = nullptr;
                    cudaStreamEndCapture(s, &junk);
                    if (junk) cudaGraphDestroy(junk);
                    cudaGetLastError();
                    ok = false;
                } else {
                    ok = wg.g.end(s, err);
                    cudaGetLastError();
                }
            } else {
                cudaGetLastError();
            }
            wg.requests = F->expected - e0;
            F->expected = e0;
            if (!ok) {   // (this key's windows: launches - the same kernels)
                std::fprintf(stderr, "inkling: the window graph (%d columns) did not capture%s%s - launches\n", T,
                             err.empty() ? "" : ": ", err.c_str());
                err.clear();
                wg.bad = true;
                ++K.wgraph_bad;
            } else {
                ++K.wgraph_caps;
            }
            graph_cap_ = true;
        }
        if (it->second.seen <= kCapAfter && !it->second.g.valid() && !it->second.bad) {
            // (launched above)
        } else if (it->second.bad) {
            ++K.wgraph_eager;
            if (!run()) return false;
        } else {
            if (!it->second.g.launch(s, err)) return false;
            inkd_tok_taps(T);   // (the taps inside the graph: their rows' bookkeeping on the host)
            F->expected += it->second.requests;
            ++K.wgraph_runs;
        }
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("inkling verify: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------- the batched prompt path
namespace {

// the attention's sub-batch (STRATA_INK_PREFILL_SUB, 64 .. 2048: a window ring's rows stay unread); 2048 with the
// tensor-core attention (its blocks fill the GPU from 2048 rows: 8K prompt +8%, 32K +11% against 512), else 512
int ink_sub() {
    static const int v = [] {
        const char* e = getenv("STRATA_INK_PREFILL_SUB");
        return e != nullptr ? std::clamp(std::atoi(e), 64, 2048) : ik::attn_prompt_on() ? 2048 : 512;
    }();
    return v;
}
constexpr int kPromptSplits = 4;   // the prompt attention's key ranges a row (the rows fill the GPU)

struct Cv {   // 256-byte aligned carving from a base (base null: sizing only)
    uint8_t* base = nullptr;
    size_t off = 0;
    template <class Tp>
    Tp* take(size_t n) {
        Tp* p = base != nullptr ? (Tp*) (base + off) : nullptr;
        off += (n * sizeof(Tp) + 255u) & ~(size_t) 255u;
        return p;
    }
};
// a sub-batch's scratch
struct Sub {
    float *mixo, *msgs, *cin[3], *cat, *lo, *cg, *cf, *xn[3];
    uint16_t* x16[3];
    float *qf, *gl, *kf, *vf, *kc, *vc, *rf, *sk, *kd, *vd, *ao, *tmp, *tmp2, *part;
    uint16_t* ao16;
};
Sub carve_sub(Cv& c, const Glm5Geometry& g, size_t tn) {
    const size_t E = (size_t) g.n_embd, HD = (size_t) g.n_head * g.head_dim, KV = (size_t) g.attn_head_kv * g.head_dim;
    const size_t H = (size_t) g.n_head, Hkv = (size_t) g.attn_head_kv, R = (size_t) g.ink_rank;
    Sub b{};
    b.mixo = c.take<float>(tn * 3 * E);
    b.msgs = c.take<float>(tn * 4 * E);
    for (int j = 0; j < 3; ++j) {
        b.cin[j] = c.take<float>(tn * E);
        b.xn[j] = c.take<float>(tn * E);
        b.x16[j] = c.take<uint16_t>(tn * E);
    }
    b.cat = c.take<float>(tn * 2 * E);
    b.lo = c.take<float>(tn * R);
    b.cg = c.take<float>(tn * E);
    b.cf = c.take<float>(tn * E);
    b.qf = c.take<float>(tn * HD);
    b.gl = c.take<float>(tn * HD);
    b.kf = c.take<float>(tn * KV);
    b.vf = c.take<float>(tn * KV);
    b.kc = c.take<float>(tn * KV);
    b.vc = c.take<float>(tn * KV);
    b.rf = c.take<float>(tn * H * ik::kRel);
    b.sk = c.take<float>(tn * H);
    b.kd = c.take<float>(tn * Hkv * ik::kConv);
    b.vd = c.take<float>(tn * Hkv * ik::kConv);
    b.ao = c.take<float>(tn * HD);
    b.ao16 = c.take<uint16_t>(tn * HD);
    b.tmp = c.take<float>(tn * E);
    b.tmp2 = c.take<float>(tn * E);
    b.part = c.take<float>(ik::attn_part_floats((int) tn, (int) H, kPromptSplits));
    return b;
}
bool blas_ok(cublasStatus_t st, const char* what, std::string& err) {
    if (st == CUBLAS_STATUS_SUCCESS) return true;
    err = std::string("inkling prompt: ") + what + ": cuBLAS status " + std::to_string((int) st);
    return false;
}
// (i-quants too: unsloth's UD-IQ3_XXS / UD-IQ2_XXS store the last layers' attention as IQ3_XXS; rows of whole
// 256-value super-blocks)
bool f16_type(int t) {
    return t == gf::kTypeBF16 || t == gf::kTypeF32 || strata::kernels::dequant_bf16_supported(t) ||
           strata::kernels::iq_supported(t);
}
// Y[t][N] (ldy) = X16[t][Kd] . W[N][Kd]^T: W widened / dequantized to FP16 in slices that fit the scratch, the tensor
// cores' product with F32 accumulation (the prompt path's hgemm_q)
bool gemm16(const Glm5Model::Hy4PromptIo& io, const WSlot& w, int N, int Kd, const uint16_t* X16, int ldx, float* Y,
            int ldy, int Tn, cudaStream_t s, std::string& err) {
    if (Tn <= 0) return true;
    const float one = 1.0f, zero = 0.0f;
    const int rows = (int) std::max<int64_t>(1, std::min<int64_t>(N, io.w16_elems / Kd));
    for (int r0 = 0; r0 < N; r0 += rows) {
        const int n = std::min(rows, N - r0);
        if (w.type == gf::kTypeBF16)
            gb::bf16_to_f16((const uint16_t*) w.q + (size_t) r0 * Kd, io.w16, (int64_t) n * Kd, s);
        else if (w.type == gf::kTypeF32)
            gb::f32_to_f16((const float*) w.q + (size_t) r0 * Kd, io.w16, (int64_t) n * Kd, s);
        else if (!strata::kernels::dequant_bf16_supported(w.type))   // (an i-quant: f16_type)
            strata::kernels::iq_dequant_f16(w.type, (const uint8_t*) w.q + (size_t) r0 * strata::kernels::iq_row_bytes(w.type, Kd),
                                            (int64_t) n * Kd, io.w16, s);
        else
            strata::kernels::dequant_f16(w.type, w.q, r0, n, Kd, io.w16, s);
        if (!blas_ok(cublasGemmEx((cublasHandle_t) io.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, Tn, Kd, &one, io.w16,
                                  CUDA_R_16F, Kd, X16, CUDA_R_16F, ldx, &zero, Y + r0, CUDA_R_32F, ldy,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                     "gemm f16", err))
            return false;
    }
    return true;
}
// Y[t][N] = X[t][Kd] (ldx) . W[N][Kd]^T in F32 (the controllers' F32 rows; bf: W holds them as BF16 - widened back
// to the same F32 values in the weight scratch first, so the SGEMM is the one it was)
bool gemm32(const Glm5Model::Hy4PromptIo& io, const float* W, bool bf, int N, int Kd, const float* X, int ldx, float* Y,
            int Tn, cudaStream_t s, std::string& err) {
    if (Tn <= 0) return true;
    const float one = 1.0f, zero = 0.0f;
    if (bf) {
        if (io.w16 == nullptr || io.w16_elems < 2 * (int64_t) N * Kd) {
            err = "inkling prompt: the weight scratch is smaller than a controller's rows";
            return false;
        }
        ik::ctl_unpack((const uint16_t*) W, (float*) io.w16, (int64_t) N * Kd, s);
        W = (const float*) io.w16;
    }
    return blas_ok(cublasSgemm((cublasHandle_t) io.blas, CUBLAS_OP_T, CUBLAS_OP_N, N, Tn, Kd, &one, W, Kd, X, ldx, &zero,
                               Y, N),
                   "sgemm", err);
}
ik::States rows_of(const float* const* p, int n, const float* cur, int64_t t0, int64_t E) {
    ik::States st;
    for (int i = 0; i < n; ++i) st.p[i] = p[i] + t0 * E;
    st.p[n] = cur + t0 * E;
    st.n = n + 1;
    st.stride = E;
    return st;
}

}  // namespace

bool Glm5Model::ink_prompt_ok(std::string& why) const {
    const char* v = getenv("STRATA_INK_PREFILL");
    if (v != nullptr && std::atoi(v) == 0) {
        why = "STRATA_INK_PREFILL=0";
        return false;
    }
    const FastState* F = fast_;
    if (F == nullptr || F->ink == nullptr) {
        why = "inkling: no token path state";
        return false;
    }
    const Glm5Geometry& g = g_;
    if (g.head_dim != ik::kHeadDim || g.n_head % g.attn_head_kv != 0) {
        why = "inkling: head geometry";
        return false;
    }
    std::string bad;
    const auto ck = [&](const WSlot& w, int il, const char* nm) {
        if (w.q == nullptr || !f16_type(w.type))
            bad += "blk." + std::to_string(il) + "." + nm + " (type " + std::to_string(w.type) + ") ";
    };
    for (int il = 0; il < g.n_layers; ++il) {
        const auto& Ly = F->ink->L[(size_t) il];
        ck(Ly.q, il, "attn_q");
        ck(Ly.k, il, "attn_k");
        ck(Ly.v, il, "attn_v");
        ck(Ly.o, il, "attn_output");
        ck(Ly.gate, il, "attn_gate");
        ck(Ly.rel, il, "attn_rel");
        ck(Ly.sink, il, "attn_sink_proj");
        if (!Ly.swa) {
            ck(Ly.kcp, il, "attn_k_conv_proj");
            ck(Ly.vcp, il, "attn_v_conv_proj");
        }
        const auto& FL = F->L[(size_t) il];
        if (FL.router_type != gf::kTypeF32 && FL.router_type != gf::kTypeBF16)
            bad += "blk." + std::to_string(il) + ".ffn_gate_inp (type " + std::to_string(FL.router_type) + ") ";
    }
    if (!bad.empty()) {
        why = "inkling: weights without an FP16 decode: " + bad.substr(0, 300);
        return false;
    }
    return true;
}

std::pair<size_t, size_t> Glm5Model::ink_prompt_bytes(size_t T) const {
    const Glm5Geometry& g = g_;
    const size_t E = (size_t) g.n_embd;
    Cv k;
    for (int i = 0; i < Ink::kBufs; ++i) k.take<float>(T * E);
    k.take<float>(T * E);   // fin
    Cv c;
    carve_sub(c, g, std::min<size_t>(T, (size_t) ink_sub()));
    return {k.off, c.off};
}

const float* Glm5Model::ink_prompt_final() const {
    return fast_ != nullptr && fast_->ink != nullptr ? fast_->ink->pc.fin : nullptr;
}

bool Glm5Model::ink_prompt_embed(int T, const Hy4PromptIo& io, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    Ink& K = *F->ink;
    const size_t E = (size_t) g.n_embd;
    auto& C = K.pc;
    Cv k{io.keep};
    for (int i = 0; i < Ink::kBufs; ++i) C.hb[i] = k.take<float>((size_t) T * E);
    C.fin = k.take<float>((size_t) T * E);
    C.nb = 0;
    C.bn = 0;
    C.pbn = 0;
    C.bank_n = 0;
    C.bank_out = nullptr;
    C.prelude = nullptr;
    C.h = C.hb[C.nb++ % Ink::kBufs];
    if (pos_ == 0) ink_reset();   // (a new sequence: the conv states start at zero)
    // the embedding rows (prefill_run's, in io.x), rms-normed with token_embd_norm
    ik::rms_rows(io.x, (int64_t) E, K.tok_norm, g.norm_eps, (int) E, T, C.h, (int64_t) E, F->cs);
    (void) err;
    return true;
}

bool Glm5Model::ink_prompt_layer(int ia, int64_t p0, int T, const Hy4PromptIo& io, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    Ink& K = *F->ink;
    auto& C = K.pc;
    cudaStream_t s = F->cs;
    const int E = g.n_embd, D = g.head_dim, H = g.n_head, Hkv = g.attn_head_kv, KV = Hkv * D, HD = H * D;
    const int R = g.ink_rank;
    const int il = g.ink_app_phys(ia), pass = g.ink_app_pass(ia), sl = g.ink_app_slot(ia);
    const auto& Ly = K.L[(size_t) il];
    auto& S = K.S[(size_t) sl];
    const bool swa = Ly.swa;
    const bool looped = g.ink_passes > 1;
    // ---- the passes' bookkeeping (the token path's): the prelude before the core, each later pass from it again
    if (looped && il == g.ink_lo) {
        if (pass == 0) {
            C.prelude = C.h;
            C.pbn = C.bn;
            for (int i = 0; i < C.bn; ++i) C.pbl[i] = C.bl[i];
        } else {
            C.bank_n = 0;
            for (int i = C.pbn + 1; i < C.bn; ++i) C.bank[C.bank_n++] = C.bl[i];
            C.bank_out = C.h;
            C.h = C.prelude;
            C.bn = C.pbn;
            for (int i = 0; i < C.pbn; ++i) C.bl[i] = C.pbl[i];
        }
    }
    const bool has_msgs = pass > 0;
    const bool starts = il % g.ink_block == 0;
    float* hin = C.h;
    if (inkd_ != nullptr) ink_tap(ia, hin, E, T, nullptr, true, s);   // (the DSpark drafter's taps 5 / 23 / 41)
    const float* ab[ik::kMaxStates];   // the attention mix's blocks
    const int an = C.bn;
    for (int i = 0; i < an; ++i) ab[i] = C.bl[i];
    if (starts) {
        if (C.bn + 1 >= ik::kMaxStates) {
            err = "inkling prompt: too many residual blocks";
            return false;
        }
        C.bl[C.bn++] = hin;
        C.h = C.hb[C.nb++ % Ink::kBufs];
    }
    float* hout = C.h;
    C.hin = hin;
    const int tn_max = std::min(T, ink_sub());
    Cv cv{io.uni};
    const Sub B = carve_sub(cv, g, (size_t) tn_max);
    if (cv.off > io.uni_bytes) {
        err = "inkling prompt: the scratch";
        return false;
    }
    for (int t0 = 0; t0 < T; t0 += tn_max) {
        const int tn = std::min(tn_max, T - t0);
        const int64_t pt = p0 + t0;
        // ---- the attention's inputs
        const float* in[3] = {hin + (size_t) t0 * E, hin + (size_t) t0 * E, hin + (size_t) t0 * E};
        int64_t ins = E;
        const int nx = swa && !has_msgs ? 1 : 3;
        const float* gm[3] = {swa ? Ly.attn_norm : Ly.qn, swa ? Ly.attn_norm : Ly.kn, swa ? Ly.attn_norm : Ly.vn};
        bool normed = false;   // (the norms and the FP16 copies done with the mix: ik::mix_norm)
        if (Ly.attn_rq != nullptr) {
            const int nq = swa ? 1 : 3;
            void* x16[3] = {B.x16[0], B.x16[1], B.x16[2]};
            if (!has_msgs)
                normed = ik::mix_norm(rows_of(ab, an, hin, t0, E), Ly.attn_rq, nq, E, tn, B.mixo, nx, !swa, gm,
                                      g.norm_eps, B.xn, x16, s);
            if (!normed) ik::mix(rows_of(ab, an, hin, t0, E), Ly.attn_rq, nq, E, tn, B.mixo, s);
            for (int j = 0; j < 3; ++j) in[j] = B.mixo + (size_t) (swa ? 0 : j) * E;
            ins = (int64_t) nq * E;
        }
        const auto control = [&](int c, const float* cur, int64_t cs, const float* msg, int64_t ms, float* out) -> bool {
            ik::ctrl_cat(cur, cs, msg, ms, E, tn, B.cat, s);
            if (!gemm32(io, Ly.down[c], Ly.bf[c][0], R, 2 * E, B.cat, 2 * E, B.lo, tn, s, err) ||
                !gemm32(io, Ly.gain[c], Ly.bf[c][1], E, R, B.lo, R, B.cg, tn, s, err) ||
                !gemm32(io, Ly.forget[c], Ly.bf[c][2], E, R, B.lo, R, B.cf, tn, s, err))
                return false;
            ik::ctrl_apply(cur, cs, msg, ms, B.cg, B.cf, E, tn, out, E, s);
            return true;
        };
        if (has_msgs) {
            ik::mix(rows_of(C.bank, C.bank_n, C.bank_out, t0, E), Ly.msg_q, kCtrl, E, tn, B.msgs, s);
            for (int j = 0; j < 3; ++j)
                if (!control(j, in[j], ins, B.msgs + (size_t) j * E, 4 * (int64_t) E, B.cin[j])) return false;
            for (int j = 0; j < 3; ++j) in[j] = B.cin[j];
            ins = E;
        }
        for (int j = 0; j < nx && !normed; ++j) {
            ik::rms_rows(in[j], ins, gm[j], g.norm_eps, E, tn, B.xn[j], E, s);
            gb::f32_to_f16(B.xn[j], B.x16[j], (int64_t) tn * E, s);
        }
        const int i1 = nx == 1 ? 0 : 1, i2 = nx == 1 ? 0 : 2;
        if (io.mark) io.mark(io.mark_ctx, "ink_mix", s);
        // ---- the projections
        if (!gemm16(io, Ly.q, HD, E, B.x16[0], E, B.qf, HD, tn, s, err) ||
            !gemm16(io, Ly.gate, HD, E, B.x16[0], E, B.gl, HD, tn, s, err) ||
            !gemm16(io, Ly.rel, H * ik::kRel, E, B.x16[0], E, B.rf, H * ik::kRel, tn, s, err) ||
            !gemm16(io, Ly.sink, H, E, B.x16[0], E, B.sk, H, tn, s, err) ||
            !gemm16(io, Ly.k, KV, E, B.x16[i1], E, B.kf, KV, tn, s, err) ||
            !gemm16(io, Ly.v, KV, E, B.x16[i2], E, B.vf, KV, tn, s, err))
            return false;
        if (!swa && (!gemm16(io, Ly.kcp, ik::kConv * Hkv, E, B.x16[i1], E, B.kd, ik::kConv * Hkv, tn, s, err) ||
                     !gemm16(io, Ly.vcp, ik::kConv * Hkv, E, B.x16[i2], E, B.vd, ik::kConv * Hkv, tn, s, err)))
            return false;
        if (io.mark) io.mark(io.mark_ctx, "ink_proj", s);
        // ---- conv, clips, the caches
        float* conv = S.conv;
        ik::conv_rows_par(B.kf, KV, Ly.k_conv, swa ? nullptr : B.kd, ik::kConv * Hkv, D, conv, KV, tn, B.kc, KV, s);
        ik::conv_rows_par(B.vf, KV, Ly.v_conv, swa ? nullptr : B.vd, ik::kConv * Hkv, D, conv + (size_t) KV * 3, KV, tn,
                          B.vc, KV, s);
        const float lf = swa ? 0.0f : g.ink_log_floor;
        ik::clip_heads(B.qf, HD, Ly.q_clip, D, H, g.ink_qk_clip, pt, lf, g.ink_log_alpha, tn, s);
        ik::clip_heads(B.kc, KV, nullptr, D, Hkv, g.ink_qk_clip, pt, 0.0f, 0.0f, tn, s);
        ik::clip_heads(B.rf, H * ik::kRel, Ly.rel_clip, ik::kRel, H, g.ink_rel_clip, pt, lf, g.ink_log_alpha, tn, s);
        ik::kv_store(B.kc, KV, B.vc, KV, g.norm_eps, Hkv, pt, tn, S.ring, S.kc, S.vc, s);
        if (io.mark) io.mark(io.mark_ctx, "ink_conv", s);
        // ---- attention, gate, output, post conv
        if (!ik::attn_prompt(B.qf, HD, B.rf, H * ik::kRel, B.sk, H, Ly.rel_bias, swa ? g.n_swa : g.ink_extent, S.kc,
                             S.vc, S.ring, H, Hkv, pt, tn, swa ? g.n_swa : 0, 1.0f / std::sqrt((float) D), B.part,
                             kPromptSplits, B.ao, HD, s))
            ik::attn(B.qf, HD, B.rf, H * ik::kRel, B.sk, H, Ly.rel_bias, swa ? g.n_swa : g.ink_extent, S.kc, S.vc,
                     S.ring, H, Hkv, pt, tn, swa ? g.n_swa : 0, 1.0f / std::sqrt((float) D), B.part, kPromptSplits, B.ao,
                     HD, s);
        if (io.mark) io.mark(io.mark_ctx, "ink_attn_core", s);
        ik::gate_sigmoid(B.ao, HD, B.gl, HD, HD, tn, s);
        gb::f32_to_f16(B.ao, B.ao16, (int64_t) tn * HD, s);
        if (!gemm16(io, Ly.o, E, HD, B.ao16, HD, B.tmp, E, tn, s, err)) return false;
        if (io.mark) io.mark(io.mark_ctx, "ink_attn", s);
        float* conv_post = conv + (size_t) 2 * KV * 3;
        // ---- the post conv (+ the add; the attention's output starts the next block) and the FFN's input
        ik::PromptPre pre;
        pre.in = B.tmp;
        pre.w = Ly.post_conv;
        pre.state = conv_post;
        pre.out = hout + (size_t) t0 * E;
        pre.tmp = B.tmp2;
        pre.add = !starts;
        float* xo = io.x + (size_t) t0 * E;
        if (!has_msgs && ik::mix_norm(rows_of(C.bl, C.bn, hout, t0, E), Ly.ffn_rq, 1, E, tn, B.mixo, 1, false,
                                      &Ly.ffn_norm, g.norm_eps, &xo, nullptr, s, &pre))
            continue;
        ik::mix(rows_of(C.bl, C.bn, hout, t0, E), Ly.ffn_rq, 1, E, tn, B.mixo, s, nullptr, &pre);
        const float* fin = B.mixo;
        if (has_msgs) {
            if (!control(3, fin, E, B.msgs + (size_t) 3 * E, 4 * (int64_t) E, B.cin[0])) return false;
            fin = B.cin[0];
        }
        ik::rms_rows(fin, E, Ly.ffn_norm, g.norm_eps, E, tn, xo, E, s);
    }
    gb::f32_to_f16(io.x, io.x16, (int64_t) T * E, s);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("inkling prompt: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool Glm5Model::ink_prompt_ffn_out(int ia, int T, const Hy4PromptIo& io, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    Ink& K = *F->ink;
    auto& C = K.pc;
    cudaStream_t s = F->cs;
    const int E = g.n_embd, KV = g.attn_head_kv * g.head_dim;
    const int il = g.ink_app_phys(ia), sl = g.ink_app_slot(ia);
    const auto& Ly = K.L[(size_t) il];
    float* conv = K.S[(size_t) sl].conv;
    // (x, the experts' input, is free now: the post conv's rows)
    ik::conv_rows_par(io.ffn, E, Ly.ffn_conv, nullptr, 0, E, conv + (size_t) (2 * KV + E) * 3, E, T, io.x, E, s);
    ik::add_rows(C.h, E, io.x, E, E, T, s);
    if (ia + 1 == g.ink_slots())   // the head's rows: the depth attention with output_res_query
        ik::mix(rows_of(C.bl, C.bn, C.h, 0, E), K.out_rq, 1, E, T, C.fin, s);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("inkling prompt ffn: ") + cudaGetErrorString(e);
        return false;
    }
    return true;
}

}  // namespace strata::core
