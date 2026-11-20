// src/core/hy4_fast_path.cu - Tencent Hy4-flash (GGUF arch hy_v4_flash) on the fast decode path (Glm5Model members).
//
// The semantics are Tencent's (the HYV4Flash* modules of the transformers port, the checkpoint's config.json); ref/hy4.py
// is the numpy yardstick on the same GGUF weights.  A token:
//   * the embedding row (hy_dim) copied into 4 streams R; the Engram rows of the 4 Engram layers hashed from the token
//     and the two before it and read from the GGUF's second shard (FP8 rows) while the first layers run
//   * a layer: [Engram add] -> attn hyper-connection (8 mixes, pre / post sigmoid gates) -> attn_norm -> gated MLA
//     (a shared 512 latent + a 64-dim rope key, absorbed k_b / v_b, sinks, a sigmoid gate per head; window layers a
//     512-row ring, the others the whole context and past 2048 keys the compressed indexer's cells) -> attn_output ->
//     post_attention_norm -> R_c += post_c * a -> ffn hyper-connection -> ffn_norm -> layer 0 a dense SwiGLU, the others
//     the latent MoE (a sigmoid router on x; the routed experts on ffn_routed_down x, 1536 wide, through the expert
//     tiers exactly as every other model's - Glm5Geometry::n_embd is that width; ffn_routed_up back; the shared expert
//     on x) -> post_ffw_norm -> R_c += post_c * f
//   * the head: a 4-mix collapse of the streams, output_norm, output
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#include <array>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"

// the Engram table's file: opened once, read at 64-bit offsets from the read pool's threads (Windows: an overlapped
// handle, so the threads' reads run side by side; elsewhere pread)
namespace {
#ifdef _WIN32
using EngFile = HANDLE;
const EngFile kEngNone = INVALID_HANDLE_VALUE;
EngFile eng_open(const std::string& path) {
    return CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                       FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED, nullptr);
}
bool eng_pread(EngFile f, uint8_t* dst, size_t n, uint64_t at) {
    struct Ev {
        HANDLE ev = CreateEventA(nullptr, TRUE, FALSE, nullptr);
        ~Ev() { if (ev != nullptr) CloseHandle(ev); }
    };
    static thread_local Ev ev;
    if (ev.ev == nullptr) return false;
    size_t done = 0;
    while (done < n) {
        OVERLAPPED ov{};
        ov.Offset = (DWORD) (at + done);
        ov.OffsetHigh = (DWORD) ((at + done) >> 32);
        ov.hEvent = ev.ev;
        DWORD r = 0;
        if (!ReadFile(f, dst + done, (DWORD) (n - done), nullptr, &ov) && GetLastError() != ERROR_IO_PENDING) return false;
        if (!GetOverlappedResult(f, &ov, &r, TRUE) || r == 0) return false;
        done += (size_t) r;
    }
    return true;
}
#else
using EngFile = int;
const EngFile kEngNone = -1;
EngFile eng_open(const std::string& path) { return ::open(path.c_str(), O_RDONLY | O_CLOEXEC); }
bool eng_pread(EngFile f, uint8_t* dst, size_t n, uint64_t at) {
    size_t done = 0;
    while (done < n) {
        const ssize_t r = ::pread(f, dst + done, n - done, (off_t) (at + done));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        done += (size_t) r;
    }
    return true;
}
#endif
}   // namespace
#include "strata/kernels/hy4_fast.hpp"
#include "strata/qtypes.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace hk = strata::kernels::hy4;

namespace {

// a few threads for the Engram rows' reads (64 reads of 96 bytes a token, random over a 63 GB table)
struct ReadPool {
    std::mutex m;
    std::condition_variable cv, done_cv;
    std::function<void(int)> fn;
    int n = 0, left = 0;
    std::atomic<int> next{0};
    uint64_t gen = 0;
    bool quit = false;
    std::vector<std::thread> th;
    explicit ReadPool(int threads) {
        for (int i = 0; i < threads; ++i)
            th.emplace_back([this] {
                uint64_t seen = 0;
                for (;;) {
                    std::unique_lock<std::mutex> l(m);
                    cv.wait(l, [&] { return gen != seen || quit; });
                    if (quit) return;
                    seen = gen;
                    l.unlock();
                    drain();
                }
            });
    }
    ~ReadPool() {
        {
            std::lock_guard<std::mutex> l(m);
            quit = true;
        }
        cv.notify_all();
        for (auto& t : th) t.join();
    }
    void drain() {
        int done = 0;
        for (int j; (j = next.fetch_add(1)) < n;) {
            fn(j);
            ++done;
        }
        if (done > 0) {
            std::lock_guard<std::mutex> l(m);
            left -= done;
            if (left == 0) done_cv.notify_all();
        }
    }
    void run(int jobs, std::function<void(int)> f) {
        if (jobs <= 0) return;
        {
            std::lock_guard<std::mutex> l(m);
            fn = std::move(f);
            n = jobs;
            left = jobs;
            next = 0;
            ++gen;
        }
        cv.notify_all();
        drain();
        std::unique_lock<std::mutex> l(m);
        done_cv.wait(l, [&] { return left == 0; });
    }
};

struct Hy4Layer {
    int heads = 0;
    bool swa = false, moe = false;
    const float *hc_fn[2] = {}, *hc_scale[2] = {}, *hc_base[2] = {};
    const float *attn_norm = nullptr, *ffn_norm = nullptr, *post_attn = nullptr, *post_ffw = nullptr;
    WSlot q_a, q_b, kv_a, gate, out;
    const float *q_a_norm = nullptr, *kv_a_norm = nullptr, *sinks = nullptr;
    const void *k_b = nullptr, *v_b = nullptr;
    bool kb_f32 = false, vb_f32 = false;   // decoded to float rows at load (a GGUF with other k_b / v_b types)
    // the indexer (the non-window layers)
    WSlot ik, iqb, iproj;
    const float *ik_norm = nullptr, *cv = nullptr, *cg = nullptr, *ape = nullptr, *cnorm = nullptr;
    // the FFN
    WSlot router, rdown, rup, shg, shu, shd, dg, du, dd;
    const float* bias = nullptr;
    // Engram
    int eng = -1;
    WSlot ekv;
    const float *ekv_b = nullptr, *egate = nullptr, *elnorm = nullptr, *escale = nullptr, *epg = nullptr;
    // caches
    uint16_t* cache = nullptr;   // rows x 576 FP16
    int ring = 0;                // rows (a window layer: n_swa; else the context)
    float* tk4 = nullptr;        // the indexer's last 4 token keys (4 x 128)
    float* pooled = nullptr;     // the closed groups' keys (ctx / 4 x 128)
};

struct Hy4State {
    std::vector<Hy4Layer> L;
    void* arena = nullptr;
    size_t arena_bytes = 0;
    // activations
    float *x = nullptr, *mix = nullptr, *post = nullptr, *qr = nullptr, *q = nullptr, *kv = nullptr, *gate = nullptr,
          *qabs = nullptr, *ctx = nullptr, *o = nullptr, *a = nullptr, *f = nullptr, *iq = nullptr, *iw = nullptr,
          *tk = nullptr, *score = nullptr, *sg = nullptr, *su = nullptr, *sh = nullptr, *r_out = nullptr,
          *dgv = nullptr, *duv = nullptr, *e = nullptr, *ekv = nullptr, *erow = nullptr, *head_x = nullptr,
          *emb = nullptr;
    void *xq = nullptr, *qrq = nullptr, *oq = nullptr, *shq = nullptr, *ffq = nullptr, *dhq = nullptr, *eq = nullptr,
         *head_xq = nullptr;
    int *cells = nullptr, *n_cells = nullptr;
    uint8_t* erow_raw = nullptr;   // device: the 64 rows' FP8 codes
    // host
    float* emb_h = nullptr;        // pinned
    uint8_t* erow_h = nullptr;     // pinned 64 rows of eng_rb bytes
    float* erow_fh = nullptr;      // pinned 64 x eng_dim floats (a table that is not FP8: decoded on the host)
    // Engram (pack/engram.txt)
    EngFile eng_fd = kEngNone;
    uint64_t eng_table = 0, eng_rows = 0;
    int eng_dim = 96, eng_ng = 3, eng_hpn = 8, eng_pad = 120002;
    int eng_type = -1;              // the table's ggml type (-1: FP8 E4M3 codes, decoded on the GPU)
    int eng_rb = 96;                // bytes a row
    std::vector<int> eng_layers;
    std::vector<std::array<int64_t, 3>> eng_mult;   // per Engram layer
    std::vector<uint64_t> eng_off, eng_size;        // 64 heads
    std::vector<uint32_t> tmap;
    std::vector<int32_t> hist;                      // the tokens by position (the n-grams)
    std::vector<char> img_pos;                      // an image row at that position (no n-gram)
    bool eng_on = false;
    float eng_eps = 0.0078125f, eng_row_eps = 1e-12f;
    bool eng_vfirst = false;
    std::unique_ptr<ReadPool> pool;
    // the head
    const float *ohc_fn = nullptr, *ohc_scale = nullptr, *ohc_base = nullptr, *onorm = nullptr;
    int max_ctx = 0;
    // GLM_CB_DIR-style dumps (HY4_DUMP_DIR): per position, appended
    std::string dump_dir;
};

size_t al(size_t b) { return (b + 255u) & ~(size_t) 255u; }
size_t q8b(int64_t n) { return (size_t) (n / 32) * 36; }

Hy4State* S(void* p) { return static_cast<Hy4State*>(p); }

}  // namespace

const float* Glm5Model::hy4_head_x() const { return S(hy4_)->head_x; }
const void* Glm5Model::hy4_head_xq() const { return S(hy4_)->head_xq; }

void Glm5Model::hy4_reset() {
    if (hy4_ == nullptr) return;
    Hy4State* H = S(hy4_);
    H->hist.clear();
    H->img_pos.clear();
}

bool Glm5Model::hy4_setup(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    if (n_parts_ > 1 || l0_ != 0 || l1_ != g.n_layers) {
        err = "hy4: layer splits are not supported yet (one GPU)";
        return false;
    }
    auto* H = new Hy4State();
    hy4_ = H;
    H->max_ctx = (int) max_ctx_;
    const int D = g.hy_dim, E = g.n_embd;
    std::string missing;
    const auto f32 = [&](const std::string& n) -> const float* {
        auto it = w_.find(n);
        if (it == w_.end()) {
            missing += n + " (f32) ";
            return nullptr;
        }
        return it->second;
    };
    const auto f32m = [&](const std::string& n, int64_t n_in, int64_t n_out) -> WSlot {
        const float* p = f32(n);
        return WSlot{p, gf::kTypeF32, p, n_in, n_out};
    };
    const auto q = [&](const std::string& n, int64_t n_in, int64_t n_out) -> WSlot {
        auto it = ws_map_.find(n);
        // (a projection the GGUF keeps in F32 / F16 - widened to F32 by the pack; an F32 row of the index is type 0)
        if (it == ws_map_.end() || it->second.type == 0)
            if (auto f = w_.find(n); f != w_.end()) return WSlot{f->second, gf::kTypeF32, f->second, n_in, n_out};
        if (it == ws_map_.end() || it->second.type == 0 || !gf::mv_supported(it->second.type)) {
            missing += n + " (quant) ";
            return WSlot{};
        }
        if (it->second.n_in != n_in || it->second.n_out != n_out)
            missing += n + " (shape " + std::to_string(it->second.n_in) + "x" + std::to_string(it->second.n_out) +
                       ", expected " + std::to_string(n_in) + "x" + std::to_string(n_out) + ") ";
        return it->second;
    };
    // k_b / v_b: as stored when they are the types the attention kernels read (Q8_0 / Q6_K, Maya's GGUFs), else
    // decoded once to float rows of n_in (another converter's types: any ggml type with a host decoder, BF16 kept as
    // such by the pack, or F32)
    const auto raw = [&](const std::string& n, int type, int64_t n_in, int64_t n_out, bool& as_f32) -> const void* {
        as_f32 = false;
        auto it = ws_map_.find(n);
        if (it != ws_map_.end() && it->second.type == type && it->second.n_in == n_in && it->second.n_out == n_out)
            return it->second.q;
        const size_t count = (size_t) n_in * (size_t) n_out;
        std::vector<float> host;
        if (auto f = w_.find(n); f != w_.end() && it == ws_map_.end()) {
            as_f32 = true;
            return f->second;
        }
        if (it != ws_map_.end() && it->second.n_in == n_in && it->second.n_out == n_out) {
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
        } else if (auto b = w16_.find(n); b != w16_.end()) {
            std::vector<uint16_t> h16(count);
            if (cudaMemcpy(h16.data(), b->second, count * 2, cudaMemcpyDeviceToHost) == cudaSuccess) {
                host.resize(count);
                for (size_t i = 0; i < count; ++i) host[i] = strata::qt::bf16_to_fp32(h16[i]);
            }
        }
        void* d = nullptr;
        if (host.empty() || cudaMalloc(&d, count * sizeof(float)) != cudaSuccess ||
            cudaMemcpy(d, host.data(), count * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            missing += n + " (" + std::to_string(n_in) + "x" + std::to_string(n_out) + ", Q" +
                       (type == 8 ? "8_0" : "6_K") + " or a type with a host decoder) ";
            return nullptr;
        }
        as_f32 = true;
        return d;
    };
    H->L.assign((size_t) g.n_layers, Hy4Layer{});
    for (int il = 0; il < g.n_layers; ++il) {
        Hy4Layer& Y = H->L[(size_t) il];
        const std::string P = "blk." + std::to_string(il) + ".";
        const int nh = g.lag_heads[(size_t) il];
        Y.heads = nh;
        Y.swa = g.lag_swa[(size_t) il] != 0;
        Y.moe = il >= g.dense_lead;
        for (int k = 0; k < 2; ++k) {
            const std::string side = k == 0 ? "hc_attn_" : "hc_ffn_";
            Y.hc_fn[k] = f32(P + side + "fn.weight");
            Y.hc_scale[k] = f32(P + side + "scale.weight");
            Y.hc_base[k] = f32(P + side + "base.weight");
        }
        Y.attn_norm = f32(P + "attn_norm.weight");
        Y.ffn_norm = f32(P + "ffn_norm.weight");
        Y.post_attn = f32(P + "post_attention_norm.weight");
        Y.post_ffw = f32(P + "post_ffw_norm.weight");
        Y.q_a = q(P + "attn_q_a.weight", D, g.q_lora);
        Y.q_a_norm = f32(P + "attn_q_a_norm.weight");
        Y.q_b = q(P + "attn_q_b.weight", g.q_lora, (int64_t) nh * g.head_dim);
        Y.kv_a = q(P + "attn_kv_a_mqa.weight", D, g.kv_lora + g.n_rot);
        Y.kv_a_norm = f32(P + "attn_kv_a_norm.weight");
        Y.gate = q(P + "attn_gate.weight", D, (int64_t) nh * g.v_head);
        Y.out = q(P + "attn_output.weight", (int64_t) nh * g.v_head, D);
        Y.sinks = f32(P + "attn_sinks.weight");
        Y.k_b = raw(P + "attn_k_b.weight", 8, g.qk_nope, (int64_t) g.kv_lora * nh, Y.kb_f32);      // Q8_0
        Y.v_b = raw(P + "attn_v_b.weight", 14, g.kv_lora, (int64_t) g.v_head * nh, Y.vb_f32);      // Q6_K
        if (!Y.swa) {
            Y.ik = q(P + "indexer.attn_k.weight", D, g.idx_key);         // F32 (Maya) or quantized (nace-ai: Q8_0)
            Y.ik_norm = f32(P + "indexer.k_norm.weight");
            Y.iqb = q(P + "indexer.attn_q_b.weight", g.q_lora, (int64_t) g.idx_heads * g.idx_key);
            Y.iproj = f32m(P + "indexer.proj.weight", D, g.idx_heads);
            Y.cv = f32(P + "indexer_compressor_value.weight");
            Y.cg = f32(P + "indexer_compressor_gate.weight");
            Y.ape = f32(P + "indexer_compressor_ape.weight");
            Y.cnorm = f32(P + "indexer_compressor_norm.weight");
        }
        if (Y.moe) {
            Y.router = f32m(P + "ffn_gate_inp.weight", D, g.n_expert);
            Y.bias = f32(P + "exp_probs_b.bias");
            Y.rdown = q(P + "ffn_routed_down.weight", D, E);
            Y.rup = q(P + "ffn_routed_up.weight", E, D);
            Y.shg = q(P + "ffn_gate_shexp.weight", D, g.n_ff_shexp);
            Y.shu = q(P + "ffn_up_shexp.weight", D, g.n_ff_shexp);
            Y.shd = q(P + "ffn_down_shexp.weight", g.n_ff_shexp, D);
        } else {
            Y.dg = q(P + "ffn_gate.weight", D, g.n_ff_dense);
            Y.du = q(P + "ffn_up.weight", D, g.n_ff_dense);
            Y.dd = q(P + "ffn_down.weight", g.n_ff_dense, D);
        }
        const auto ei = std::find(g.hy_eng.begin(), g.hy_eng.end(), il);
        if (ei != g.hy_eng.end()) {
            Y.eng = (int) (ei - g.hy_eng.begin());
            Y.ekv = q(P + "engram_kv.weight", 16 * 96, (int64_t) (g.hc + 1) * D);
            Y.ekv_b = f32(P + "engram_kv_b.weight");
            Y.egate = f32(P + "engram_gate.weight");
            Y.elnorm = f32(P + "engram_lookup_norm.weight");
            Y.escale = f32(P + "engram_hc_scale.weight");
            Y.epg = f32(P + "engram_pre_gamma.weight");
        }
    }
    H->ohc_fn = f32("output_hc_fn.weight");
    H->ohc_scale = f32("output_hc_scale.weight");
    H->ohc_base = f32("output_hc_base.weight");
    H->onorm = f32("output_norm.weight");
    if (!missing.empty()) {
        err = "hy4: unresolved weights: " + missing.substr(0, 4000);
        return false;
    }
    if (pack_emb_src_ == nullptr || strata::qt::traits((strata::qt::Type) pack_emb_type_) == nullptr) {
        err = "hy4: no token_embd mapping";
        return false;
    }

    // ---- the caches and the activations: one arena
    const int64_t ctx = max_ctx_;
    const int NH = g.n_head;
    size_t need = 0;
    for (int il = 0; il < g.n_layers; ++il) {
        Hy4Layer& Y = H->L[(size_t) il];
        Y.ring = Y.swa ? (int) std::min<int64_t>(ctx, g.n_swa) : (int) ctx;
        need += al((size_t) Y.ring * hk::kRow * 2);
        if (!Y.swa) need += al(4 * 128 * 4) + al((size_t) (ctx / 4 + 1) * 128 * 4);
    }
    const int64_t fl[] = {D, 16, 8, g.q_lora, (int64_t) NH * g.head_dim, g.kv_lora + g.n_rot, (int64_t) NH * g.v_head,
                          (int64_t) NH * g.kv_lora, (int64_t) NH * g.kv_lora, (int64_t) NH * g.v_head, D, D,
                          (int64_t) g.idx_heads * g.idx_key, 32, 128, ctx / 4 + 8, g.n_ff_shexp, g.n_ff_shexp,
                          D /* sh: the shared expert's output */, D, g.n_ff_dense, g.n_ff_dense, 16 * 96, (int64_t) (g.hc + 1) * D, 64 * 96, D, D};
    for (int64_t n : fl) need += al((size_t) n * 4);
    const int64_t qn[] = {D, g.q_lora, (int64_t) NH * g.v_head, g.n_ff_shexp, E, g.n_ff_dense, 16 * 96, D};
    for (int64_t n : qn) need += al(q8b(n));
    need += al((size_t) hk::kMaxCells * 4) + al(4) + al(64 * 96);
    if (cudaMalloc(&H->arena, need) != cudaSuccess) {
        cudaGetLastError();
        err = "hy4: the cache / activation arena (" + std::to_string(need >> 20) + " MB) did not allocate";
        return false;
    }
    cudaMemset(H->arena, 0, need);
    H->arena_bytes = need;
    uint8_t* at = (uint8_t*) H->arena;
    const auto take = [&](size_t b) -> void* {
        void* p = at;
        at += al(b);
        return p;
    };
    for (int il = 0; il < g.n_layers; ++il) {
        Hy4Layer& Y = H->L[(size_t) il];
        Y.cache = (uint16_t*) take((size_t) Y.ring * hk::kRow * 2);
        if (!Y.swa) {
            Y.tk4 = (float*) take(4 * 128 * 4);
            Y.pooled = (float*) take((size_t) (ctx / 4 + 1) * 128 * 4);
        }
    }
    float** fp[] = {&H->x, &H->iw /*16*/, &H->mix /*8*/, &H->qr, &H->q, &H->kv, &H->gate, &H->qabs, &H->ctx, &H->o,
                    &H->a, &H->f, &H->iq, &H->post /*32*/, &H->tk, &H->score, &H->sg, &H->su, &H->sh, &H->r_out,
                    &H->dgv, &H->duv, &H->e, &H->ekv, &H->erow, &H->head_x, &H->emb};
    for (size_t i = 0; i < sizeof(fl) / sizeof(fl[0]); ++i) *fp[i] = (float*) take((size_t) fl[i] * 4);
    void** qp[] = {&H->xq, &H->qrq, &H->oq, &H->shq, &H->ffq, &H->dhq, &H->eq, &H->head_xq};
    for (size_t i = 0; i < sizeof(qn) / sizeof(qn[0]); ++i) *qp[i] = take(q8b(qn[i]));
    H->cells = (int*) take((size_t) hk::kMaxCells * 4);
    H->n_cells = (int*) take(4);
    H->erow_raw = (uint8_t*) take(64 * 96);
    if (cudaHostAlloc((void**) &H->emb_h, (size_t) D * 4, cudaHostAllocPortable) != cudaSuccess ||
        cudaHostAlloc((void**) &H->erow_h, 64 * 96, cudaHostAllocPortable) != cudaSuccess) {
        err = "hy4: pinned buffers did not allocate";
        return false;
    }
    std::fprintf(stderr, "hy4: CUDA%d caches + activations %.2f GB (context %lld; %d window layers of %d rows)\n", dev_,
                 (double) need / 1073741824.0, (long long) ctx,
                 (int) std::count(g.lag_swa.begin(), g.lag_swa.end(), (char) 1), g.n_swa);

    // ---- Engram: pack/engram.txt (tools/iq_pack.py write_hy4_engram) and the table's shard
    if (!g.hy_eng.empty()) {
        std::ifstream ef(pack_dir_ + "/engram.txt");
        if (!ef) {
            err = "hy4: the pack has no engram.txt (re-pack with tools/iq_pack.py)";
            return false;
        }
        std::string line, shard;
        H->eng_mult.assign(g.hy_eng.size(), {0, 0, 0});
        while (std::getline(ef, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string k;
            ss >> k;
            if (k == "format") {
                // "<type> <values a row> [<bytes a row>]": FP8 E4M3 codes (Maya's GGUFs) or a ggml type the host
                // decodes (llama.cpp-fork GGUFs keep the table in IQ4_NL; F16 / BF16 / F32 / Q8_0 as stored)
                std::string fmt;
                int rb = -1;
                ss >> fmt >> H->eng_dim >> rb;
                const std::pair<const char*, int> kinds[] = {{"fp8_e4m3", -1}, {"iq4_nl", strata::qt::IQ4_NL},
                                                             {"q8_0", strata::qt::Q8_0}, {"f16", strata::qt::F16},
                                                             {"bf16", strata::qt::BF16}, {"f32", strata::qt::F32}};
                bool known = false;
                for (const auto& kd : kinds)
                    if (fmt == kd.first) {
                        H->eng_type = kd.second;
                        known = true;
                    }
                H->eng_rb = H->eng_type < 0 ? H->eng_dim
                                            : (int) strata::qt::row_size(H->eng_type, H->eng_dim);
                if (!known || (rb >= 0 && rb != H->eng_rb) || H->eng_rb <= 0) {
                    err = "hy4: engram.txt row format " + fmt + " (" + std::to_string(rb) +
                          " bytes a row) is not one this engine reads (fp8_e4m3, iq4_nl, q8_0, f16, bf16, f32)";
                    return false;
                }
            } else if (k == "table") {
                ss >> H->eng_table >> H->eng_rows >> shard;
            } else if (k == "layers") {
                int v;
                while (ss >> v) H->eng_layers.push_back(v);
            } else if (k == "ngram") {
                ss >> H->eng_ng >> H->eng_hpn;
            } else if (k == "pad") {
                ss >> H->eng_pad;
            } else if (k == "mult") {
                int il;
                ss >> il;
                const auto it = std::find(g.hy_eng.begin(), g.hy_eng.end(), il);
                if (it == g.hy_eng.end()) continue;
                auto& m = H->eng_mult[(size_t) (it - g.hy_eng.begin())];
                for (int j = 0; j < 3; ++j) ss >> m[(size_t) j];
            } else if (k == "head") {
                int hidx;
                uint64_t o, sz;
                ss >> hidx >> o >> sz;
                if (hidx >= 0) {
                    if ((size_t) hidx >= H->eng_off.size()) {
                        H->eng_off.resize((size_t) hidx + 1, 0);
                        H->eng_size.resize((size_t) hidx + 1, 1);
                    }
                    H->eng_off[(size_t) hidx] = o;
                    H->eng_size[(size_t) hidx] = sz;
                }
            }
        }
        const int heads = H->eng_hpn * (H->eng_ng - 1);
        if (H->eng_dim != 96 || H->eng_ng != 3 || H->eng_hpn != 8 || heads != 16 ||
            H->eng_off.size() != (size_t) heads * g.hy_eng.size() || H->eng_layers != g.hy_eng || shard.empty()) {
            err = "hy4: engram.txt does not match the model (96-wide rows, 3-grams, 8 heads an order, the GGUF's layers)";
            return false;
        }
        std::ifstream tm(pack_dir_ + "/engram_token_map.u32", std::ios::binary | std::ios::ate);
        if (!tm) {
            err = "hy4: the pack has no engram_token_map.u32";
            return false;
        }
        const size_t nb = (size_t) tm.tellg();
        H->tmap.resize(nb / 4);
        tm.seekg(0);
        tm.read((char*) H->tmap.data(), (std::streamsize) nb);
        const std::string path = pack_dir_ + "/../" + shard;
        H->eng_fd = eng_open(path);
        if (H->eng_fd == kEngNone) {
            err = "hy4: cannot open the Engram table's shard " + path;
            return false;
        }
        const char* ee = getenv("STRATA_HY4_ENG_EPS");
        if (ee != nullptr && ee[0]) H->eng_eps = (float) std::atof(ee);
        const char* re = getenv("STRATA_HY4_ENG_ROW_EPS");
        if (re != nullptr && re[0]) H->eng_row_eps = (float) std::atof(re);
        const char* vf = getenv("STRATA_HY4_ENG_VFIRST");
        H->eng_vfirst = vf != nullptr && std::atoi(vf) != 0;
        const char* off = getenv("STRATA_HY4_ENGRAM");
        H->eng_on = off == nullptr || std::atoi(off) != 0;
        H->pool.reset(new ReadPool(8));
        if (H->eng_rb > 96) {   // (the pinned row buffer was sized for FP8 rows)
            cudaFreeHost(H->erow_h);
            H->erow_h = nullptr;
            if (cudaHostAlloc((void**) &H->erow_h, (size_t) 64 * H->eng_rb, cudaHostAllocPortable) != cudaSuccess) {
                err = "hy4: pinned Engram rows did not allocate";
                return false;
            }
        }
        if (H->eng_type >= 0 &&
            cudaHostAlloc((void**) &H->erow_fh, (size_t) 64 * H->eng_dim * sizeof(float), cudaHostAllocPortable) !=
                cudaSuccess) {
            err = "hy4: pinned Engram rows did not allocate";
            return false;
        }
        std::fprintf(stderr, "hy4: Engram on layers %d %d %d %d from %s (%llu rows of %d %s values)%s\n", g.hy_eng[0],
                     g.hy_eng.size() > 1 ? g.hy_eng[1] : -1, g.hy_eng.size() > 2 ? g.hy_eng[2] : -1,
                     g.hy_eng.size() > 3 ? g.hy_eng[3] : -1, shard.c_str(), (unsigned long long) H->eng_rows,
                     H->eng_dim, H->eng_type < 0 ? "FP8" : strata::qt::type_name(H->eng_type),
                     H->eng_on ? "" : " - OFF (STRATA_HY4_ENGRAM=0)");
    }
    if (const char* dd = getenv("HY4_DUMP_DIR"); dd != nullptr && dd[0]) H->dump_dir = dd;
    (void) F;
    return true;
}

bool Glm5Model::hy4_embed(int32_t token, int64_t p, const float* img, std::string& err) {
    FastState* F = fast_;
    Hy4State* H = S(hy4_);
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int D = g.hy_dim;
    if (img != nullptr) {
        std::memcpy(H->emb_h, img, (size_t) D * sizeof(float));
    } else {
        const strata::qt::Traits* tt = strata::qt::traits((strata::qt::Type) pack_emb_type_);
        tt->to_float(pack_emb_src_ + (size_t) token * strata::qt::row_size((strata::qt::Type) pack_emb_type_, D), H->emb_h, D);
    }
    cudaMemcpyAsync(H->emb, H->emb_h, (size_t) D * sizeof(float), cudaMemcpyHostToDevice, s);
    for (int c = 0; c < g.hc; ++c)
        cudaMemcpyAsync(state_ + (size_t) c * D, H->emb, (size_t) D * sizeof(float), cudaMemcpyDeviceToDevice, s);
    // the n-gram history
    if ((int64_t) H->hist.size() < p + 1) {
        H->hist.resize((size_t) p + 1, H->eng_pad);
        H->img_pos.resize((size_t) p + 1, 0);
    }
    H->hist[(size_t) p] = token;
    H->img_pos[(size_t) p] = img != nullptr;
    if (H->eng_fd == kEngNone || !H->eng_on) return true;
    // the 64 rows: per Engram layer, 8 heads of the 2-gram and 8 of the 3-gram
    const auto cid = [&](int64_t q) -> int64_t {
        const int32_t t = q < 0 ? H->eng_pad : H->hist[(size_t) q];
        const int32_t tt = (t >= 0 && (size_t) t < H->tmap.size()) ? t : H->eng_pad;
        return (int64_t) H->tmap[(size_t) tt];
    };
    const int64_t c0 = cid(p), c1 = cid(p - 1), c2 = cid(p - 2);
    uint64_t rows[64];
    const int NL = (int) g.hy_eng.size();
    for (int li = 0; li < NL; ++li) {
        const auto& m = H->eng_mult[(size_t) li];
        const uint64_t m0 = (uint64_t) (c0 * m[0]), m1 = (uint64_t) (c1 * m[1]), m2 = (uint64_t) (c2 * m[2]);
        const uint64_t mix[2] = {m0 ^ m1, m0 ^ m1 ^ m2};
        for (int n = 0; n < 2; ++n)
            for (int j = 0; j < 8; ++j) {
                const int h = n * 8 + j, gh = 16 * li + h;
                rows[16 * li + h] = H->eng_off[(size_t) gh] + mix[n] % H->eng_size[(size_t) gh];
            }
    }
    const int n_rows = 16 * NL;
    bool bad = false;
    H->pool->run(n_rows, [&](int k) {
        const uint64_t at = H->eng_table + rows[k] * (uint64_t) H->eng_rb;
        uint8_t* dst = H->erow_h + (size_t) k * H->eng_rb;
        if (!eng_pread(H->eng_fd, dst, (size_t) H->eng_rb, at)) {
            bad = true;
            return;
        }
        if (H->eng_type >= 0)
            strata::qt::traits(H->eng_type)->to_float(dst, H->erow_fh + (size_t) k * H->eng_dim, H->eng_dim);
    });
    if (bad) {
        err = "hy4: an Engram row read failed";
        return false;
    }
    if (H->eng_type < 0) {
        cudaMemcpyAsync(H->erow_raw, H->erow_h, (size_t) n_rows * H->eng_dim, cudaMemcpyHostToDevice, s);
        hk::e4m3_to_f32(H->erow_raw, H->erow, n_rows * H->eng_dim, s);
    } else {
        cudaMemcpyAsync(H->erow, H->erow_fh, (size_t) n_rows * H->eng_dim * sizeof(float), cudaMemcpyHostToDevice, s);
    }
    // (the pinned rows are read by the copy above before the next token rewrites them: fast_token waits for each
    // token's completion)
    return true;
}

namespace { void hy4_dump(const std::string& dir, const std::string& name, const float* d, size_t n, cudaStream_t s); }

bool Glm5Model::fast_hy4_moe(int il, std::string& err) {
    FastState* F = fast_;
    Hy4State* H = S(hy4_);
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const Hy4Layer& Y = H->L[(size_t) il];
    const auto& Ly = F->L[(size_t) il];
    const int D = g.hy_dim, E = g.n_embd, FS = g.n_ff_shexp;
    // the router (F32, on x), the routed down-projection (x -> the experts' 1536), the shared gate / up
    gf::MvJob j[4];
    j[0] = {Y.router.q, nullptr, H->x, F->rlog, nullptr, 1.0f, gf::kTypeF32, D, g.n_expert};
    j[1] = {Y.rdown.q, H->xq, H->x, F->x, nullptr, 1.0f, Y.rdown.type, D, E};
    j[2] = {Y.shg.q, H->xq, H->x, H->sg, nullptr, 1.0f, Y.shg.type, D, FS};
    j[3] = {Y.shu.q, H->xq, H->x, H->su, nullptr, 1.0f, Y.shu.type, D, FS};
    if (!gf::mv(j, 4, s)) {
        err = "hy4: router / routed_down / shared expert";
        return false;
    }
    if (!H->dump_dir.empty()) {
        hy4_dump(H->dump_dir, "rlog-" + std::to_string(il), F->rlog, (size_t) g.n_expert, s);
        hy4_dump(H->dump_dir, "rlat-" + std::to_string(il), F->x, (size_t) E, s);
    }
    gf::quantize_q8_1(F->x, F->xq, E, s);
    gf::MoeDev md = F->md;
    md.nv_scale = Ly.nv_scale;
    if (F->expert_helper && F->expert_helper->active && !F->expert_helper->failed) {
        md.helper_owned = F->expert_helper->owned_d;
        md.helper_promote = expert_helper_promote();
    }
    md.pf_src = F->pf_buf[il & 1];
    md.pf_dst = F->pf_buf[il & 1] + 8;
    md.pf_n = F->pf_n_buf[il & 1];
    const bool lane = F->cpu_plan != 0ull && F->cpu_fmt[(size_t) il].n_ff > 0;
    static const int promote_min = [] {
        const char* v = getenv("STRATA_GLM_PROMOTE_MIN");
        return v ? std::max(0, std::atoi(v)) : 0;
    }();
    const bool dlane = lane && F->disk_lane;
    gf::moe_route(F->rlog, Y.bias, g.n_expert, g.n_exp_used, g.w_scale, g.norm_w != 0, il, F->x, E, md, nullptr, nullptr,
                  g.swiglu_shexp, 0, F->sh_hq, s, nullptr, nullptr, 0, nullptr, nullptr, 0, fast_skip_from(il),
                  lane ? F->cpu_plan : 0ull, dlane ? -1 : promote_min, 0, g.gating, nullptr, fast_skip_w(il),
                  F->tail_skip_disk ? 1 : 0, lane && F->igpu_experts_q != nullptr ? F->xq : nullptr);
    ++F->expected;
    if (F->bg_moe > 0) F->bg_drain(s, F->bg_moe, F->bg_moe_mask);
    // the shared expert while the lane and the fetches run (no clamp on it)
    hk::swiglu(H->sg, H->su, FS, H->sg, s);
    gf::quantize_q8_1(H->sg, H->shq, FS, s);
    gf::moe_wait(md, E, s);
    if (F->stream_fetch)
        gf::moe_fetch(md, g.n_exp_used, Ly.blob, s, Ly.gu_bytes, Ly.dn_bytes, F->read_chunks, Ly.up_bytes);
    else
        gf::moe_fetch(md, g.n_exp_used, Ly.blob, s);
    gf::moe_gate_up(Ly.gu_type, md, g.n_exp_used, E, g.n_ff_exp, g.swiglu_exp, F->xq, F->hq, nullptr, 0, F->sh_hq, 0,
                    F->sh_out, s, Ly.up_type);
    gf::moe_down(Ly.d_type, md, g.n_exp_used, E, g.n_ff_exp, Ly.down_off, F->hq, nullptr, F->ffn, s);
    if (lane) gf::moe_cpu_wait(md, E, F->ffn, s, 1, g.n_exp_used);
    if (!H->dump_dir.empty()) hy4_dump(H->dump_dir, "rsum-" + std::to_string(il), F->ffn, (size_t) E, s);
    // back to hy_dim: routed_up (the routed sum) + the shared expert's down, then post_ffw_norm
    gf::quantize_q8_1(F->ffn, H->ffq, E, s);
    gf::MvJob u[2];
    u[0] = {Y.rup.q, H->ffq, F->ffn, H->r_out, nullptr, 1.0f, Y.rup.type, E, D};
    u[1] = {Y.shd.q, H->shq, H->sg, H->sh, nullptr, 1.0f, Y.shd.type, FS, D};
    if (!gf::mv(u, 2, s)) {
        err = "hy4: routed_up / shared down";
        return false;
    }
    if (!H->dump_dir.empty()) {
        hy4_dump(H->dump_dir, "rup-" + std::to_string(il), H->r_out, (size_t) D, s);
        hy4_dump(H->dump_dir, "shx-" + std::to_string(il), H->sh, (size_t) D, s);
    }
    hk::rms(H->r_out, Y.post_ffw, g.norm_eps, D, H->f, s, H->sh);
    if (!F->kda_graph_on) F->bg_drain(s, F->bg_pace, F->bg_attn_mask);
    return true;
}

namespace {
// HY4_DUMP_DIR: append n floats of device memory to <dir>/<name>.f32
void hy4_dump(const std::string& dir, const std::string& name, const float* d, size_t n, cudaStream_t s) {
    if (dir.empty()) return;
    cudaStreamSynchronize(s);
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (FILE* f = std::fopen((dir + "/" + name + ".f32").c_str(), "ab")) {
        std::fwrite(h.data(), sizeof(float), n, f);
        std::fclose(f);
    }
}
}  // namespace

bool Glm5Model::fast_hy4_layers(int64_t p, std::string& err) {
    FastState* F = fast_;
    Hy4State* H = S(hy4_);
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int D = g.hy_dim, HC = g.hc;
    float* R = state_;
    const std::string& dd = H->dump_dir;
    // the cell's rotary position: the image request's (t, h, w) table, else the cell itself
    const std::vector<int32_t>& mt = mrope_h_;
    const auto rp = [&mt](int64_t c) -> strata::kernels::hy4::Pos3 {
        if ((size_t) c * 3 + 2 >= mt.size()) return c;
        return {mt[(size_t) c * 3], mt[(size_t) c * 3 + 1], mt[(size_t) c * 3 + 2]};
    };
    if (p >= H->max_ctx) {
        err = "hy4: the context is full";
        return false;
    }
    const bool eng_live = H->eng_fd != kEngNone && H->eng_on;
    const int32_t tok = H->hist[(size_t) p];
    const bool eng_masked = H->img_pos[(size_t) p] || tok == H->eng_pad || tok == 120120 || tok == 120124;
    for (int il = 0; il < g.n_layers; ++il) {
        const Hy4Layer& Y = H->L[(size_t) il];
        const int nh = Y.heads;
        // ---- Engram
        if (Y.eng >= 0 && eng_live && !eng_masked) {
            hk::eng_rows(H->erow + (size_t) Y.eng * 16 * 96, Y.elnorm, 16, 96, H->eng_row_eps, H->e, s);
            gf::quantize_q8_1(H->e, H->eq, 16 * 96, s);
            const gf::MvJob ej = {Y.ekv.q, H->eq, H->e, H->ekv, Y.ekv_b, 1.0f, Y.ekv.type, 16 * 96, (HC + 1) * D};
            if (!gf::mv(&ej, 1, s)) {
                err = "hy4: engram_kv";
                return false;
            }
            hk::eng_add(H->ekv, R, Y.egate, Y.escale, Y.epg, HC, D, H->eng_eps, H->eng_vfirst, s);
            hy4_dump(dd, "engram-" + std::to_string(il), R, (size_t) HC * D, s);
        }
        // ---- attention
        hk::hc_mix(R, Y.hc_fn[0], HC, D, 2 * HC, g.norm_eps, H->mix, s);
        hk::hc_pre(H->mix, Y.hc_scale[0], Y.hc_base[0], R, HC, D, g.hc_eps, g.hc_mag, Y.attn_norm, g.norm_eps, H->x,
                   H->post, s);
        gf::quantize_q8_1(H->x, H->xq, D, s);
        hy4_dump(dd, "attn_norm-" + std::to_string(il), H->x, D, s);
        {
            gf::MvJob j[5];
            int nj = 0;
            j[nj++] = {Y.q_a.q, H->xq, H->x, H->qr, nullptr, 1.0f, Y.q_a.type, D, g.q_lora};
            j[nj++] = {Y.kv_a.q, H->xq, H->x, H->kv, nullptr, 1.0f, Y.kv_a.type, D, g.kv_lora + g.n_rot};
            j[nj++] = {Y.gate.q, H->xq, H->x, H->gate, nullptr, 1.0f, Y.gate.type, D, nh * g.v_head};
            if (!Y.swa) {
                j[nj++] = {Y.ik.q, H->xq, H->x, H->tk, nullptr, 1.0f, Y.ik.type, D, g.idx_key};
                j[nj++] = {Y.iproj.q, nullptr, H->x, H->iw, nullptr, 1.0f, gf::kTypeF32, D, g.idx_heads};
            }
            if (!gf::mv(j, nj, s)) {
                err = "hy4: q_a / kv_a / gate";
                return false;
            }
        }
        hk::rms(H->qr, Y.q_a_norm, g.norm_eps, g.q_lora, H->qr, s);
        gf::quantize_q8_1(H->qr, H->qrq, g.q_lora, s);
        {
            gf::MvJob j[2];
            int nj = 0;
            j[nj++] = {Y.q_b.q, H->qrq, H->qr, H->q, nullptr, 1.0f, Y.q_b.type, g.q_lora, nh * g.head_dim};
            if (!Y.swa)
                j[nj++] = {Y.iqb.q, H->qrq, H->qr, H->iq, nullptr, 1.0f, Y.iqb.type, g.q_lora, g.idx_heads * g.idx_key};
            if (!gf::mv(j, nj, s)) {
                err = "hy4: q_b";
                return false;
            }
        }
        const float base = Y.swa ? g.rope_base_swa : g.rope_base;
        hk::mla_prep(H->q, nh, H->kv, Y.kv_a_norm, g.norm_eps, rp(p), base, Y.cache + (size_t) (p % Y.ring) * hk::kRow, s);
        hk::mla_absorb(H->q, Y.k_b, nh, H->qabs, s, Y.kb_f32);
        const float scale = 1.0f / std::sqrt((float) g.head_dim);
        if (Y.swa) {
            const int64_t lo = std::max<int64_t>(0, p - g.n_swa + 1);
            hk::mla_attend(H->qabs, H->q, Y.cache, Y.ring, nullptr, 0, lo, p, Y.sinks, scale, nh, H->ctx, s);
        } else {
            // the indexer's token key into its slot; a closed group's pooled key
            hk::rms(H->tk, Y.ik_norm, g.norm_eps, g.idx_key, Y.tk4 + (size_t) (p % 4) * 128, s);
            if (p % 4 == 3)
                hk::idx_pool(Y.tk4, Y.cv, Y.cg, Y.ape, Y.cnorm, g.norm_eps, rp(p - 3), g.rope_base, Y.pooled + (size_t) (p / 4) * 128, s);
            if (p + 1 <= g.idx_top_k) {
                hk::mla_attend(H->qabs, H->q, Y.cache, Y.ring, nullptr, 0, 0, p, Y.sinks, scale, nh, H->ctx, s);
            } else {
                const int own = (int) (p / 4);
                hk::idx_query(H->iq, g.idx_heads, H->iw, rp(p), g.rope_base, s);
                hk::idx_score(H->iq, H->iw, g.idx_heads, Y.pooled, own, H->score, s);
                const int top = g.idx_top_k / 4 - 1;
                hk::idx_select(H->score, own, top, p, H->cells, H->n_cells, s);
                hk::mla_attend(H->qabs, H->q, Y.cache, Y.ring, H->cells, std::min(top, own) * 4 + 4, 0, 0, Y.sinks,
                               scale, nh, H->ctx, s);
            }
        }
        hk::mla_out(H->ctx, Y.v_b, H->gate, nh, H->o, s, Y.vb_f32);
        gf::quantize_q8_1(H->o, H->oq, nh * g.v_head, s);
        {
            const gf::MvJob oj = {Y.out.q, H->oq, H->o, H->a, nullptr, 1.0f, Y.out.type, nh * g.v_head, D};
            if (!gf::mv(&oj, 1, s)) {
                err = "hy4: attn_output";
                return false;
            }
        }
        hk::hc_add(R, H->post, H->a, Y.post_attn, g.norm_eps, HC, D, s);
        hy4_dump(dd, "attn_out-" + std::to_string(il), H->a, D, s);
        // ---- the FFN
        hk::hc_mix(R, Y.hc_fn[1], HC, D, 2 * HC, g.norm_eps, H->mix, s);
        hk::hc_pre(H->mix, Y.hc_scale[1], Y.hc_base[1], R, HC, D, g.hc_eps, g.hc_mag, Y.ffn_norm, g.norm_eps, H->x,
                   H->post, s);
        gf::quantize_q8_1(H->x, H->xq, D, s);
        hy4_dump(dd, "ffn_norm-" + std::to_string(il), H->x, D, s);
        if (Y.moe) {
            if (!fast_hy4_moe(il, err)) return false;
        } else {
            gf::MvJob j[2];
            j[0] = {Y.dg.q, H->xq, H->x, H->dgv, nullptr, 1.0f, Y.dg.type, D, g.n_ff_dense};
            j[1] = {Y.du.q, H->xq, H->x, H->duv, nullptr, 1.0f, Y.du.type, D, g.n_ff_dense};
            if (!gf::mv(j, 2, s)) {
                err = "hy4: the dense FFN";
                return false;
            }
            hk::swiglu(H->dgv, H->duv, g.n_ff_dense, H->dgv, s);
            gf::quantize_q8_1(H->dgv, H->dhq, g.n_ff_dense, s);
            const gf::MvJob dj = {Y.dd.q, H->dhq, H->dgv, H->r_out, nullptr, 1.0f, Y.dd.type, g.n_ff_dense, D};
            if (!gf::mv(&dj, 1, s)) {
                err = "hy4: the dense FFN down";
                return false;
            }
            hk::rms(H->r_out, Y.post_ffw, g.norm_eps, D, H->f, s);
        }
        hy4_dump(dd, "ffn_out-" + std::to_string(il), H->f, D, s);
        hk::hc_add(R, H->post, H->f, nullptr, g.norm_eps, HC, D, s);
        hy4_dump(dd, "l_out-" + std::to_string(il), R, (size_t) HC * D, s);
    }
    return true;
}

bool Glm5Model::hy4_head(std::string& err) {
    FastState* F = fast_;
    Hy4State* H = S(hy4_);
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int D = g.hy_dim;
    hk::hc_mix(state_, H->ohc_fn, g.hc, D, g.hc, g.norm_eps, H->mix, s);
    hk::hc_pre(H->mix, H->ohc_scale, H->ohc_base, state_, g.hc, D, g.hc_eps, 0.0f, H->onorm, g.norm_eps, H->head_x,
               nullptr, s);
    gf::quantize_q8_1(H->head_x, H->head_xq, D, s);
    hy4_dump(H->dump_dir, "result_norm", H->head_x, D, s);
    (void) err;
    return true;
}

}  // namespace strata::core
