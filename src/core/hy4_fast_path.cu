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

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
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
void eng_close(EngFile f) {
    if (f != kEngNone) CloseHandle(f);
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
void eng_close(EngFile f) {
    if (f != kEngNone) ::close(f);
}
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
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/glm_batch.hpp"
#include "strata/kernels/hy4_fast.hpp"
#include "strata/qtypes.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace gb = strata::kernels::glmb;
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
    // the jobs to the threads; the caller does not take part (wait() for them - before the next start / run)
    void start(int jobs, std::function<void(int)> f) {
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
    }
    void wait() {
        std::unique_lock<std::mutex> l(m);
        done_cv.wait(l, [&] { return left == 0; });
    }
    void run(int jobs, std::function<void(int)> f) {
        if (jobs <= 0) return;
        start(jobs, std::move(f));
        drain();
        wait();
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
    // the Engram gate norms' (value, key, stream) epsilon: FP32's machine epsilon - the reference's RMSNorm(eps=None)
    // adds that for BF16 inputs too. Up to v1.3.198 it was bf16's 2^-7, which shrank the small layer-1/11 streams in the
    // gate's query and opened the gates 2-8x; STRATA_HY4_ENG_EPS=0.0078125 brings the old value back.
    float eng_eps = 1.1920929e-07f, eng_row_eps = 1e-12f;
    bool eng_vfirst = false;
    std::unique_ptr<ReadPool> pool;
    // STRATA_HY4_ENG_ASYNC (default on): a token's Engram reads run on the pool while the GPU computes layer 0 - each
    // Engram layer waits for its own 16 rows (eng_done) and uploads them just before it uses them; the same values
    bool eng_async = true;
    bool eng_pending = false;            // this token's reads were started and not all waited for
    std::atomic<int> eng_done[16];       // per Engram layer: rows read
    std::atomic<bool> eng_bad{false};
    uint64_t eng_sel[256] = {};          // the rows being read (row index in the table)
    // the head
    const float *ohc_fn = nullptr, *ohc_scale = nullptr, *ohc_base = nullptr, *onorm = nullptr;
    int max_ctx = 0;
    // GLM_CB_DIR-style dumps (HY4_DUMP_DIR): per position, appended
    std::string dump_dir;
    std::vector<void*> owned;                       // device buffers hy4_setup allocated (k_b / v_b decoded at load)
    // the prompt path: a chunk's Engram rows (pinned, [Engram layer][row][16 heads] rows of eng_rb bytes; a table
    // that is not FP8: decoded floats beside them) and the readers that fetch them
    uint8_t* pf_rows_h = nullptr;
    size_t pf_rows_n = 0;
    float* pf_rowsf_h = nullptr;
    size_t pf_rowsf_n = 0;
    std::unique_ptr<ReadPool> pf_pool;
    double pf_read_ms = 0.0;
    uint64_t pf_reads = 0;
    // STRATA_HY4_PREFILL_ENG_ASYNC (default on): a prompt chunk's rows read on the pool while its layers run, Engram
    // layer by Engram layer (the first one's rows first); each Engram layer waits for its own (pf_done)
    bool pf_async = true;
    bool pf_pending = false;               // reads started for the chunk, the pool not waited for yet
    std::atomic<int> pf_done[16];          // per Engram layer: the chunk's tokens read
    std::atomic<bool> pf_bad{false};
    std::vector<unsigned char> pf_skip;    // the chunk's skips, kept while the reads run
    int pf_T = 0;
    double tok_read_ms = 0.0;   // the token path's Engram reads (host time; STRATA_GLM_PROF / STRATA_GLM_TIMING report it)
    uint64_t tok_reads = 0;
    uint8_t* snap = nullptr;   // the conversation snapshot (hy4_snap_copy): every window ring, every tk4
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
    if (H->pf_pending) {   // (a prompt chunk's Engram reads read the history)
        H->pf_pool->wait();
        H->pf_pending = false;
    }
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
        if (!host.empty() && cudaMalloc(&d, count * sizeof(float)) == cudaSuccess) H->owned.push_back(d);
        if (host.empty() || d == nullptr ||
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
        const char* pa = getenv("STRATA_HY4_PREFILL_ENG_ASYNC");
        H->pf_async = (pa == nullptr || std::atoi(pa) != 0) && g.hy_eng.size() <= 16;
        const char* ea = getenv("STRATA_HY4_ENG_ASYNC");
        H->eng_async = (ea == nullptr || std::atoi(ea) != 0) && g.hy_eng.size() <= 4;
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
    if (H->pf_pending) {   // (a prompt chunk's Engram reads: done before the history grows)
        H->pf_pool->wait();
        H->pf_pending = false;
    }
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
    if (H->eng_pending) {   // (a token whose layers stopped early: its reads end before the buffers are reused)
        H->pool->wait();
        H->eng_pending = false;
    }
    if (H->eng_async) {
        // the reads in the background; fast_hy4_layers' Engram layers wait for their rows (hy4_eng_upload)
        for (int k = 0; k < n_rows; ++k) H->eng_sel[k] = rows[k];
        for (int li = 0; li < NL; ++li) H->eng_done[li].store(0, std::memory_order_relaxed);
        H->eng_bad.store(false, std::memory_order_relaxed);
        H->eng_pending = true;
        H->pool->start(n_rows, [H](int k) {
            const uint64_t at = H->eng_table + H->eng_sel[k] * (uint64_t) H->eng_rb;
            uint8_t* dst = H->erow_h + (size_t) k * H->eng_rb;
            if (!eng_pread(H->eng_fd, dst, (size_t) H->eng_rb, at))
                H->eng_bad.store(true, std::memory_order_relaxed);
            else if (H->eng_type >= 0)
                strata::qt::traits(H->eng_type)->to_float(dst, H->erow_fh + (size_t) k * H->eng_dim, H->eng_dim);
            H->eng_done[k / 16].fetch_add(1, std::memory_order_release);
        });
        return true;
    }
    bool bad = false;
    const auto tr0 = std::chrono::steady_clock::now();
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
    H->tok_read_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr0).count();
    ++H->tok_reads;
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

namespace {
void hy4_dump(const std::string& dir, const std::string& name, const float* d, size_t n, cudaStream_t s);

// STRATA_HY4_ENG_ASYNC: Engram layer li's 16 rows, once read, up to the device (the slice of the synchronous path's one
// copy and conversion: the same bytes, the same values).  The host waits here, the GPU runs the layers before it
bool hy4_eng_upload(Hy4State* H, int li, cudaStream_t s, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    while (H->eng_done[li].load(std::memory_order_acquire) < 16) std::this_thread::yield();
    H->tok_read_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (H->eng_bad.load(std::memory_order_relaxed)) {
        err = "hy4: an Engram row read failed";
        return false;
    }
    const size_t off = (size_t) li * 16 * H->eng_dim;
    if (H->eng_type < 0) {
        cudaMemcpyAsync(H->erow_raw + off, H->erow_h + off, (size_t) 16 * H->eng_dim, cudaMemcpyHostToDevice, s);
        hk::e4m3_to_f32(H->erow_raw + off, H->erow + off, 16 * H->eng_dim, s);
    } else {
        cudaMemcpyAsync(H->erow + off, H->erow_fh + off, (size_t) 16 * H->eng_dim * sizeof(float), cudaMemcpyHostToDevice, s);
    }
    return true;
}
}  // namespace

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
    if (F->prof_on) F->mark("router_shexp_mv");
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
    if (F->prof_on) F->mark("moe_route");
    ++F->expected;
    if (F->bg_moe > 0) F->bg_drain(s, F->bg_moe, F->bg_moe_mask);
    // the shared expert while the lane and the fetches run (no clamp on it)
    hk::swiglu(H->sg, H->su, FS, H->sg, s);
    gf::quantize_q8_1(H->sg, H->shq, FS, s);
    if (F->prof_on) F->mark("shexp");
    gf::moe_wait(md, E, s);
    if (F->prof_on) F->mark("moe_wait");
    if (F->stream_fetch)
        gf::moe_fetch(md, g.n_exp_used, Ly.blob, s, Ly.gu_bytes, Ly.dn_bytes, F->read_chunks, Ly.up_bytes);
    else
        gf::moe_fetch(md, g.n_exp_used, Ly.blob, s);
    if (F->prof_on) F->mark("moe_fetch");
    gf::moe_gate_up(Ly.gu_type, md, g.n_exp_used, E, g.n_ff_exp, g.swiglu_exp, F->xq, F->hq, nullptr, 0, F->sh_hq, 0,
                    F->sh_out, s, Ly.up_type);
    if (F->prof_on) F->mark("moe_gate_up");
    gf::moe_down(Ly.d_type, md, g.n_exp_used, E, g.n_ff_exp, Ly.down_off, F->hq, nullptr, F->ffn, s);
    if (F->prof_on) F->mark("moe_down");
    if (lane) gf::moe_cpu_wait(md, E, F->ffn, s, 1, g.n_exp_used);
    if (F->prof_on && lane) F->mark("moe_cpu_wait");
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
    if (F->prof_on) F->mark("rup_shd_mv");
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
    // STRATA_HY4_DUMP_CACHE=<file> + STRATA_HY4_DUMP_CACHE_AT=<p> (debug): every layer's cache rows of positions
    // [0, p) as FP16 (a window layer's ring as it stands), when the token path reaches p - the prompt path's against
    // the token path's
    static const char* cdump = getenv("STRATA_HY4_DUMP_CACHE");
    static const int64_t cdump_at = getenv("STRATA_HY4_DUMP_CACHE_AT") ? std::atoll(getenv("STRATA_HY4_DUMP_CACHE_AT")) : -1;
    if (cdump != nullptr && p == cdump_at) {
        cudaStreamSynchronize(s);
        if (FILE* f = std::fopen(cdump, "wb")) {
            for (const Hy4Layer& Y : H->L) {
                const int64_t n = Y.swa ? std::min<int64_t>(p, Y.ring) : p;
                std::vector<uint16_t> h((size_t) Y.ring * hk::kRow);
                cudaMemcpy(h.data(), Y.cache, h.size() * 2, cudaMemcpyDeviceToHost);
                for (int64_t j = p - n; j < p; ++j) std::fwrite(h.data() + (size_t) (j % Y.ring) * hk::kRow, 2, hk::kRow, f);
            }
            std::fclose(f);
        }
    }
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
    const bool pm = F->prof_on;   // (STRATA_GLM_PROF: the stages' events)
    if (pm) F->mark("start");
    const int32_t tok = H->hist[(size_t) p];
    const bool eng_masked = H->img_pos[(size_t) p] || tok == H->eng_pad || tok == 120120 || tok == 120124;
    for (int il = 0; il < g.n_layers; ++il) {
        const Hy4Layer& Y = H->L[(size_t) il];
        const int nh = Y.heads;
        // ---- Engram
        if (Y.eng >= 0 && eng_live && !eng_masked) {
            if (H->eng_pending && !hy4_eng_upload(H, Y.eng, s, err)) return false;
            hk::eng_rows(H->erow + (size_t) Y.eng * 16 * 96, Y.elnorm, 16, 96, H->eng_row_eps, H->e, s);
            gf::quantize_q8_1(H->e, H->eq, 16 * 96, s);
            const gf::MvJob ej = {Y.ekv.q, H->eq, H->e, H->ekv, Y.ekv_b, 1.0f, Y.ekv.type, 16 * 96, (HC + 1) * D};
            if (!gf::mv(&ej, 1, s)) {
                err = "hy4: engram_kv";
                return false;
            }
            hk::eng_add(H->ekv, R, Y.egate, Y.escale, Y.epg, HC, D, H->eng_eps, H->eng_vfirst, s);
            hy4_dump(dd, "engram-" + std::to_string(il), R, (size_t) HC * D, s);
            if (pm) F->mark("engram");
        }
        // ---- attention
        hk::hc_mix(R, Y.hc_fn[0], HC, D, 2 * HC, g.norm_eps, H->mix, s);
        hk::hc_pre(H->mix, Y.hc_scale[0], Y.hc_base[0], R, HC, D, g.hc_eps, g.hc_mag, Y.attn_norm, g.norm_eps, H->x,
                   H->post, s);
        gf::quantize_q8_1(H->x, H->xq, D, s);
        hy4_dump(dd, "attn_norm-" + std::to_string(il), H->x, D, s);
        if (pm) F->mark("hc_attn");
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
        if (pm) F->mark("attn_proj_mv");
        const float base = Y.swa ? g.rope_base_swa : g.rope_base;
        hk::mla_prep(H->q, nh, H->kv, Y.kv_a_norm, g.norm_eps, rp(p), base, Y.cache + (size_t) (p % Y.ring) * hk::kRow, s);
        hk::mla_absorb(H->q, Y.k_b, nh, H->qabs, s, Y.kb_f32);
        if (pm) F->mark("mla_prep_absorb");
        const float scale = 1.0f / std::sqrt((float) g.head_dim);
        if (Y.swa) {
            const int64_t lo = std::max<int64_t>(0, p - g.n_swa + 1);
            hk::mla_attend(H->qabs, H->q, Y.cache, Y.ring, nullptr, 0, lo, p, Y.sinks, scale, nh, H->ctx, s);
            if (pm) F->mark("attn_win");
        } else {
            // the indexer's token key into its slot; a closed group's pooled key
            hk::rms(H->tk, Y.ik_norm, g.norm_eps, g.idx_key, Y.tk4 + (size_t) (p % 4) * 128, s);
            if (p % 4 == 3)
                hk::idx_pool(Y.tk4, Y.cv, Y.cg, Y.ape, Y.cnorm, g.norm_eps, rp(p - 3), g.rope_base, Y.pooled + (size_t) (p / 4) * 128, s);
            if (pm) F->mark("idx_key");
            if (p + 1 <= g.idx_top_k) {
                hk::mla_attend(H->qabs, H->q, Y.cache, Y.ring, nullptr, 0, 0, p, Y.sinks, scale, nh, H->ctx, s);
                if (pm) F->mark("attn_dense");
            } else {
                const int own = (int) (p / 4);
                hk::idx_query(H->iq, g.idx_heads, H->iw, rp(p), g.rope_base, s);
                hk::idx_score(H->iq, H->iw, g.idx_heads, Y.pooled, own, H->score, s);
                if (pm) F->mark("idx_score");
                const int top = g.idx_top_k / 4 - 1;
                hk::idx_select(H->score, own, top, p, H->cells, H->n_cells, s);
                if (pm) F->mark("idx_select");
                hk::mla_attend(H->qabs, H->q, Y.cache, Y.ring, H->cells, std::min(top, own) * 4 + 4, 0, 0, Y.sinks,
                               scale, nh, H->ctx, s);
                if (pm) F->mark("attn_sel");
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
        if (pm) F->mark("attn_out");
        // ---- the FFN
        hk::hc_mix(R, Y.hc_fn[1], HC, D, 2 * HC, g.norm_eps, H->mix, s);
        hk::hc_pre(H->mix, Y.hc_scale[1], Y.hc_base[1], R, HC, D, g.hc_eps, g.hc_mag, Y.ffn_norm, g.norm_eps, H->x,
                   H->post, s);
        gf::quantize_q8_1(H->x, H->xq, D, s);
        hy4_dump(dd, "ffn_norm-" + std::to_string(il), H->x, D, s);
        if (pm) F->mark("hc_ffn");
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
            if (pm) F->mark("dense_ffn");
        }
        hy4_dump(dd, "ffn_out-" + std::to_string(il), H->f, D, s);
        hk::hc_add(R, H->post, H->f, nullptr, g.norm_eps, HC, D, s);
        hy4_dump(dd, "l_out-" + std::to_string(il), R, (size_t) HC * D, s);
        if (pm) F->mark("hc_add");
    }
    if (H->eng_pending) {   // (a masked token's reads, unused: done before the buffers are reused)
        H->pool->wait();
        H->eng_pending = false;
        if (H->eng_bad.load()) {
            err = "hy4: an Engram row read failed";
            return false;
        }
        ++H->tok_reads;
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

// ================================================================ the conversation snapshot
// What a position's restore needs beyond the position: the window layers' rings (their rows past the snapshot overwrite
// rows it still reads) and the full layers' open indexer group (tk4).  The full layers' cache rows and pooled keys are
// append-only - rewritten as the new tokens arrive.  On the compute stream.
bool Glm5Model::hy4_snap_copy(bool restore) {
    if (hy4_ == nullptr) return false;
    Hy4State* H = S(hy4_);
    size_t need = 0;
    for (const Hy4Layer& Y : H->L)
        need += Y.swa ? (size_t) Y.ring * hk::kRow * sizeof(uint16_t) : (Y.tk4 != nullptr ? 4 * 128 * sizeof(float) : 0);
    if (H->snap == nullptr) {
        if (restore) return false;
        if (cudaMalloc((void**) &H->snap, need) != cudaSuccess) {
            cudaGetLastError();
            H->snap = nullptr;
            return false;
        }
    }
    size_t off = 0;
    for (const Hy4Layer& Y : H->L) {
        void* live = Y.swa ? (void*) Y.cache : (void*) Y.tk4;
        const size_t n = Y.swa ? (size_t) Y.ring * hk::kRow * sizeof(uint16_t) : (Y.tk4 != nullptr ? 4 * 128 * sizeof(float) : 0);
        if (n == 0 || live == nullptr) continue;
        cudaMemcpyAsync(restore ? live : (void*) (H->snap + off), restore ? (const void*) (H->snap + off) : live, n,
                        cudaMemcpyDeviceToDevice, fast_->cs);
        off += n;
    }
    return true;
}

// ================================================================ teardown
void Glm5Model::hy4_destroy() {
    if (hy4_ == nullptr) return;
    Hy4State* H = S(hy4_);
    if (H->eng_pending && H->pool) H->pool->wait();
    if (H->pf_pending && H->pf_pool) H->pf_pool->wait();
    if ((fast_ != nullptr && (fast_->prof_on || fast_->timing)) && (H->tok_reads > 0 || H->pf_reads > 0))
        std::fprintf(stderr, "hy4 engram reads: tokens %llu, the host waited %.1f ms for their rows (%.3f ms a token; %s), "
                             "prompt %llu rows, the host waited %.1f ms for them (%s)\n",
                     (unsigned long long) H->tok_reads, H->tok_read_ms,
                     H->tok_read_ms / (double) std::max<uint64_t>(1, H->tok_reads),
                     H->eng_async ? "async: past what layer 0 hid" : "sync: the GPU idle meanwhile", (unsigned long long) H->pf_reads,
                     H->pf_read_ms, H->pf_async ? "async: past what the layers before each Engram layer hid" : "sync");
    H->pool.reset();
    H->pf_pool.reset();
    eng_close(H->eng_fd);   // (the Engram table's shard: opened once in hy4_setup)
    H->eng_fd = kEngNone;
    if (H->arena != nullptr) cudaFree(H->arena);
    for (void* d : H->owned) cudaFree(d);
    if (H->snap != nullptr) cudaFree(H->snap);
    for (void* h : {(void*) H->emb_h, (void*) H->erow_h, (void*) H->erow_fh, (void*) H->pf_rows_h, (void*) H->pf_rowsf_h})
        if (h != nullptr) cudaFreeHost(h);
    delete H;
    hy4_ = nullptr;
}

// ================================================================ the batched prompt path
//
// glm_prefill.cu's prefill_half runs a chunk of T tokens a layer at a time; for hy4 it calls, per layer:
//   hy4_prompt_layer: [the Engram adds] -> the attention's hyper-connection read (every row) -> the attention in
//     sub-batches (the projections as FP16 GEMMs on dequantized weights - the prompt path's arithmetic -, the cache
//     rows, the indexer's keys / pools / selection, the attention over each row's keys with the sinks, the absorbed
//     products as batched GEMMs per head, the gate, the output) -> the streams' add -> the FFN's read; layer 0 its
//     dense SwiGLU whole, a MoE layer the latent (routed_down x) into the prompt path's x / x16
//   hy4_prompt_router / hy4_prompt_shared: the router's logits and the shared expert (on x, the residual's width)
//   (the routed experts: the prompt path's own grouping, MMQ products, staging and CPU lane, on the latent)
//   hy4_prompt_ffn_out: routed_up of the routed sum + the shared expert, post_ffw_norm, the streams' add
// The kernels are the token path's arithmetic row by row (hy4_fast.hpp's _rows variants); the GEMMs' FP16 rounding of
// the activations and their summing order are what differs (the token path's GEMVs read q8_1 activations).
namespace {

int hy4_sub() {   // the attention's / the dense FFN's sub-batch (STRATA_HY4_PREFILL_SUB, 64 .. 4096)
    static const int v = [] {
        const char* e = getenv("STRATA_HY4_PREFILL_SUB");
        return e != nullptr ? std::clamp(std::atoi(e), 64, 4096) : 512;
    }();
    return v;
}

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

// the chunk's rows across the layers
struct Keep {
    float *R, *x, *post, *mix, *sh;
    uint16_t* x16;
    hk::Pos3* pos;          // positions p0 - 3 .. p0 + T - 1 (the indexer's pools read 3 before the chunk)
    unsigned char* skip;    // the rows Engram adds nothing to
};
Keep carve_keep(Cv& c, const Glm5Geometry& g, size_t T) {
    const size_t D = (size_t) g.hy_dim, HC = (size_t) g.hc;
    Keep k{};
    k.R = c.take<float>(T * HC * D);
    k.x = c.take<float>(T * D);
    k.x16 = c.take<uint16_t>(T * D);
    k.post = c.take<float>(T * HC);
    k.mix = c.take<float>(T * 2 * HC);
    k.sh = c.take<float>(T * D);
    k.pos = c.take<hk::Pos3>(T + 3);
    k.skip = c.take<unsigned char>(T);
    return k;
}
// a sub-batch of the attention (n_head heads at most)
struct Attn {
    float *qr, *kv, *gate, *q, *qabs, *ctx, *a, *tk, *iw, *iq, *score;
    uint16_t *qr16, *q16, *ctx16, *o16, *kbuf;
    float* o;
    int *cells, *n_cells;
    int score_ld;
};
Attn carve_attn(Cv& c, const Glm5Geometry& g, size_t tn, int64_t max_ctx) {
    const size_t NH = (size_t) g.n_head, D = (size_t) g.hy_dim;
    Attn b{};
    b.qr = c.take<float>(tn * g.q_lora);
    b.qr16 = c.take<uint16_t>(tn * g.q_lora);
    b.kv = c.take<float>(tn * hk::kRow);
    b.gate = c.take<float>(tn * NH * hk::kV);
    b.q = c.take<float>(tn * NH * (hk::kNope + hk::kRot));
    b.o = b.q;   // (q is dead once the attention ran)
    b.q16 = c.take<uint16_t>(tn * NH * hk::kNope);
    b.qabs = c.take<float>(tn * NH * hk::kLat);
    b.ctx16 = (uint16_t*) b.qabs;   // (q_abs is dead once the attention ran)
    b.ctx = c.take<float>(tn * NH * hk::kLat);
    b.o16 = c.take<uint16_t>(tn * NH * hk::kV);
    b.a = c.take<float>(tn * D);
    b.tk = c.take<float>(tn * g.idx_key);
    b.iw = c.take<float>(tn * std::max(1, g.idx_heads));
    b.iq = c.take<float>(tn * std::max(1, g.idx_heads) * g.idx_key);
    b.score_ld = (int) (max_ctx / 4 + 8);
    b.score = c.take<float>(tn * (size_t) b.score_ld);
    b.cells = c.take<int>(tn * (size_t) hk::kMaxCells);
    b.n_cells = c.take<int>(tn);
    b.kbuf = c.take<uint16_t>((tn + (size_t) g.n_swa) * hk::kRow);
    return b;
}
// the dense FFN's / the routed output's / an Engram layer's sub-batch scratch
struct Ffn {
    float *dg, *du, *r;
    uint16_t* h16;
};
Ffn carve_ffn(Cv& c, const Glm5Geometry& g, size_t tn) {
    Ffn b{};
    b.dg = c.take<float>(tn * g.n_ff_dense);
    b.du = c.take<float>(tn * g.n_ff_dense);
    b.h16 = c.take<uint16_t>(tn * std::max<size_t>(g.n_ff_dense, g.n_embd));
    b.r = c.take<float>(tn * g.hy_dim);
    return b;
}
struct Eng {
    uint8_t* raw;
    float *rows, *e, *kv;
    uint16_t* e16;
};
Eng carve_eng(Cv& c, const Glm5Geometry& g, size_t tn) {
    Eng b{};
    b.raw = c.take<uint8_t>(tn * 16 * 96);
    b.rows = c.take<float>(tn * 16 * 96);
    b.e = c.take<float>(tn * 16 * 96);
    b.e16 = c.take<uint16_t>(tn * 16 * 96);
    b.kv = c.take<float>(tn * (size_t) (g.hc + 1) * g.hy_dim);
    return b;
}

bool blas_ok(cublasStatus_t st, const char* what, std::string& err) {
    if (st == CUBLAS_STATUS_SUCCESS) return true;
    err = std::string("hy4 prompt: ") + what + ": cuBLAS status " + std::to_string((int) st);
    return false;
}
// the weight types the prompt's FP16 products read (dequant_f16, or BF16 / F32 widened / narrowed)
bool f16_type(int t) { return t == gf::kTypeBF16 || t == gf::kTypeF32 || strata::kernels::dequant_bf16_supported(t); }
// rows [r0, r0 + n) of a weight (n_in = Kd) as FP16 into dst
void to_f16(const WSlot& w, int64_t r0, int64_t n, int64_t Kd, uint16_t* dst, cudaStream_t s) {
    if (w.type == gf::kTypeBF16)
        gb::bf16_to_f16((const uint16_t*) w.q + (size_t) r0 * Kd, dst, n * Kd, s);
    else if (w.type == gf::kTypeF32)
        gb::f32_to_f16((const float*) w.q + (size_t) r0 * Kd, dst, n * Kd, s);
    else
        strata::kernels::dequant_f16(w.type, w.q, r0, n, Kd, dst, s);
}
// Y[t][N] (row stride ldy) = X16[t][Kd] (ldx) . W[N][Kd]^T; beta 1 adds to Y.  W dequantized to FP16 in slices of
// rows that fit the scratch, the product on the tensor cores with F32 accumulation (the prompt path's hgemm_q)
bool gemm16(const Glm5Model::Hy4PromptIo& io, const WSlot& w, int N, int Kd, const uint16_t* X16, int ldx, float* Y,
            int ldy, int Tn, float beta, cudaStream_t s, std::string& err) {
    if (Tn <= 0) return true;
    const float one = 1.0f;
    const int rows = (int) std::max<int64_t>(1, std::min<int64_t>(N, io.w16_elems / Kd));
    for (int r0 = 0; r0 < N; r0 += rows) {
        const int n = std::min(rows, N - r0);
        to_f16(w, r0, n, Kd, io.w16, s);
        if (!blas_ok(cublasGemmEx((cublasHandle_t) io.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, Tn, Kd, &one, io.w16,
                                  CUDA_R_16F, Kd, X16, CUDA_R_16F, ldx, &beta, Y + r0, CUDA_R_32F, ldy,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                     "gemm f16", err))
            return false;
    }
    return true;
}
// Y[t][N] = X[t][Kd] . W[N][Kd]^T in F32 (the indexer's and the router's F32 weights, as the token path keeps them)
bool gemm32(const Glm5Model::Hy4PromptIo& io, const float* W, int N, int Kd, const float* X, float* Y, int Tn,
            std::string& err) {
    if (Tn <= 0) return true;
    const float one = 1.0f, zero = 0.0f;
    return blas_ok(cublasSgemm((cublasHandle_t) io.blas, CUBLAS_OP_T, CUBLAS_OP_N, N, Tn, Kd, &one, W, Kd, X, Kd, &zero,
                               Y, N),
                   "sgemm", err);
}
// per head h < nh: Y_h[t][M] (ldy = nh * M, head offset h * M) = X_h[t][Kd] (ldx = nh * Kd) . B_h[M][Kd]^T with B_h
// rows [h * M, (h + 1) * M) of the weight (k_b: M 512, Kd 192; v_b: M 256, Kd 512) - FP16 in the scratch, as many heads
// a pass as fit, one strided batched product a pass
bool head_gemm(const Glm5Model::Hy4PromptIo& io, const void* w, bool f32, int wtype, int nh, int M, int Kd,
               const uint16_t* X16, float* Y, int Tn, cudaStream_t s, std::string& err) {
    if (Tn <= 0) return true;
    const float one = 1.0f, zero = 0.0f;
    const int per = (int) std::max<int64_t>(1, std::min<int64_t>(nh, io.w16_elems / ((int64_t) M * Kd)));
    for (int h0 = 0; h0 < nh; h0 += per) {
        const int nhp = std::min(per, nh - h0);
        if (f32)
            gb::f32_to_f16((const float*) w + (size_t) h0 * M * Kd, io.w16, (int64_t) nhp * M * Kd, s);
        else
            strata::kernels::dequant_f16(wtype, w, (int64_t) h0 * M, (int64_t) nhp * M, Kd, io.w16, s);
        if (!blas_ok(cublasGemmStridedBatchedEx((cublasHandle_t) io.blas, CUBLAS_OP_T, CUBLAS_OP_N, M, Tn, Kd, &one,
                                                io.w16, CUDA_R_16F, Kd, (long long) M * Kd, X16 + (size_t) h0 * Kd,
                                                CUDA_R_16F, nh * Kd, (long long) Kd, &zero, Y + (size_t) h0 * M,
                                                CUDA_R_32F, nh * M, (long long) M, nhp, CUBLAS_COMPUTE_32F,
                                                CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                     "head gemm", err))
            return false;
    }
    return true;
}

}  // namespace

std::pair<size_t, size_t> Glm5Model::hy4_prompt_bytes(size_t T) const {
    const Glm5Geometry& g = g_;
    Cv k;
    carve_keep(k, g, T);
    const size_t tn = std::min<size_t>(T, (size_t) hy4_sub());
    size_t uni = 0;
    {
        Cv c;
        carve_attn(c, g, tn, max_ctx_);
        uni = std::max(uni, c.off);
    }
    {
        Cv c;
        carve_ffn(c, g, tn);
        uni = std::max(uni, c.off);
    }
    {
        Cv c;
        carve_eng(c, g, tn);
        uni = std::max(uni, c.off);
    }
    return {k.off, uni};
}

bool Glm5Model::hy4_prompt_ok(std::string& why) const {
    const char* v = getenv("STRATA_HY4_PREFILL");
    if (v != nullptr && std::atoi(v) == 0) {
        why = "STRATA_HY4_PREFILL=0";
        return false;
    }
    if (hy4_ == nullptr) {
        why = "hy4: no token path state";
        return false;
    }
    const Hy4State* H = S(hy4_);
    const Glm5Geometry& g = g_;
    if (g.n_head % 16 != 0) {
        why = "hy4: " + std::to_string(g.n_head) + " attention heads (the prompt kernels take multiples of 16)";
        return false;
    }
    std::string bad;
    const auto ck = [&](const WSlot& w, int il, const char* nm) {
        if (w.q == nullptr || !f16_type(w.type))
            bad += "blk." + std::to_string(il) + "." + nm + " (type " + std::to_string(w.type) + ") ";
    };
    for (int il = 0; il < g.n_layers; ++il) {
        const Hy4Layer& Y = H->L[(size_t) il];
        if (Y.heads % 16 != 0) bad += "blk." + std::to_string(il) + " heads " + std::to_string(Y.heads) + " ";
        ck(Y.q_a, il, "attn_q_a");
        ck(Y.q_b, il, "attn_q_b");
        ck(Y.kv_a, il, "attn_kv_a_mqa");
        ck(Y.gate, il, "attn_gate");
        ck(Y.out, il, "attn_output");
        if (!Y.kb_f32 && !strata::kernels::dequant_bf16_supported(8)) bad += "attn_k_b (Q8_0) ";
        if (!Y.vb_f32 && !strata::kernels::dequant_bf16_supported(14)) bad += "attn_v_b (Q6_K) ";
        if (Y.eng >= 0) ck(Y.ekv, il, "engram_kv");
        if (Y.moe) {
            ck(Y.rdown, il, "ffn_routed_down");
            ck(Y.rup, il, "ffn_routed_up");
            ck(Y.shg, il, "ffn_gate_shexp");
            ck(Y.shu, il, "ffn_up_shexp");
            ck(Y.shd, il, "ffn_down_shexp");
        } else {
            ck(Y.dg, il, "ffn_gate");
            ck(Y.du, il, "ffn_up");
            ck(Y.dd, il, "ffn_down");
        }
    }
    if (!bad.empty()) {
        why = "hy4: weights without an FP16 decode: " + bad.substr(0, 300);
        return false;
    }
    return true;
}

// the chunk's embedding rows into the 4 streams, its n-gram history, positions and Engram rows (read now, every row of
// the 4 Engram layers, by more threads than a token's 8)
namespace {
// a prompt chunk's Engram rows into pinned memory: [Engram layer][row][16 heads] rows of eng_rb bytes (+ floats for a
// table that is not FP8), as hy4_embed hashes them (H->hist holds the chunk's tokens already, H->pf_skip its skips).
// A job is one (Engram layer, token), Engram layer major: async (STRATA_HY4_PREFILL_ENG_ASYNC) the pool reads them
// while the layers run and hy4_prompt_layer waits for its layer's (hy4_eng_wait); else they are read here
bool hy4_eng_chunk(Hy4State* H, const Glm5Geometry& g, int64_t p0, int T, bool async, std::string& err) {
    const int NL = (int) g.hy_eng.size();
    const size_t n_rows = (size_t) NL * T * 16;
    if (H->pf_rows_n < n_rows * H->eng_rb) {
        if (H->pf_rows_h != nullptr) cudaFreeHost(H->pf_rows_h);
        H->pf_rows_h = nullptr;
        H->pf_rows_n = 0;
        if (cudaHostAlloc((void**) &H->pf_rows_h, n_rows * H->eng_rb, cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            H->pf_rows_h = nullptr;
            err = "hy4 prompt: the chunk's pinned Engram rows did not allocate";
            return false;
        }
        H->pf_rows_n = n_rows * H->eng_rb;
    }
    if (H->eng_type >= 0 && H->pf_rowsf_n < n_rows * 96) {
        if (H->pf_rowsf_h != nullptr) cudaFreeHost(H->pf_rowsf_h);
        H->pf_rowsf_h = nullptr;
        H->pf_rowsf_n = 0;
        if (cudaHostAlloc((void**) &H->pf_rowsf_h, n_rows * 96 * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            H->pf_rowsf_h = nullptr;
            err = "hy4 prompt: the chunk's pinned Engram rows did not allocate";
            return false;
        }
        H->pf_rowsf_n = n_rows * 96;
    }
    // STRATA_HY4_PREFILL_READERS=<n>: the threads that read a chunk's rows (a token's 64 rows take 8)
    static const int readers = [] {
        const char* v = getenv("STRATA_HY4_PREFILL_READERS");
        return v != nullptr ? std::clamp(std::atoi(v), 1, 256) : 32;
    }();
    if (H->pf_pool == nullptr) H->pf_pool.reset(new ReadPool(readers - 1));   // (the calling thread reads too, sync)
    uint8_t* const rows_h = H->pf_rows_h;
    float* const rowsf_h = H->pf_rowsf_h;
    const unsigned char* skip = H->pf_skip.data();
    const auto job = [H, p0, T, skip, rows_h, rowsf_h](int j) {
        const int li = j / T, t = j - li * T;
        if (!skip[(size_t) t] && !H->pf_bad.load(std::memory_order_relaxed)) {
            const auto cid = [&](int64_t q) -> int64_t {
                const int32_t tk = q < 0 ? H->eng_pad : H->hist[(size_t) q];
                const int32_t tt2 = (tk >= 0 && (size_t) tk < H->tmap.size()) ? tk : H->eng_pad;
                return (int64_t) H->tmap[(size_t) tt2];
            };
            const int64_t p = p0 + t;
            const int64_t c0 = cid(p), c1 = cid(p - 1), c2 = cid(p - 2);
            const auto& m = H->eng_mult[(size_t) li];
            const uint64_t m0 = (uint64_t) (c0 * m[0]), m1 = (uint64_t) (c1 * m[1]), m2 = (uint64_t) (c2 * m[2]);
            const uint64_t mix[2] = {m0 ^ m1, m0 ^ m1 ^ m2};
            bool ok = true;
            for (int nn = 0; nn < 2 && ok; ++nn)
                for (int jj = 0; jj < 8; ++jj) {
                    const int h = nn * 8 + jj, gh = 16 * li + h;
                    const uint64_t r = H->eng_off[(size_t) gh] + mix[nn] % H->eng_size[(size_t) gh];
                    const size_t k = ((size_t) li * T + t) * 16 + h;
                    uint8_t* dst = rows_h + k * H->eng_rb;
                    if (!eng_pread(H->eng_fd, dst, (size_t) H->eng_rb, H->eng_table + r * (uint64_t) H->eng_rb)) {
                        H->pf_bad = true;
                        ok = false;
                        break;
                    }
                    if (H->eng_type >= 0) strata::qt::traits(H->eng_type)->to_float(dst, rowsf_h + k * 96, H->eng_dim);
                }
        }
        H->pf_done[li].fetch_add(1, std::memory_order_release);
    };
    H->pf_bad.store(false);
    for (int li = 0; li < NL; ++li) H->pf_done[li].store(0, std::memory_order_relaxed);
    H->pf_T = T;
    if (async) {
        H->pf_pending = true;
        H->pf_pool->start(NL * T, job);
        return true;
    }
    H->pf_pool->run(NL * T, job);
    if (H->pf_bad.load()) {
        err = "hy4 prompt: an Engram row read failed";
        return false;
    }
    return true;
}

// STRATA_HY4_PREFILL_ENG_ASYNC: Engram layer li's rows of the chunk, before hy4_prompt_layer uses them
bool hy4_eng_wait(Hy4State* H, int li, std::string& err) {
    if (!H->pf_pending) return true;
    const auto t0 = std::chrono::steady_clock::now();
    while (H->pf_done[li].load(std::memory_order_acquire) < H->pf_T) std::this_thread::yield();
    H->pf_read_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (H->pf_bad.load()) {
        err = "hy4 prompt: an Engram row read failed";
        return false;
    }
    return true;
}
}  // namespace

bool Glm5Model::hy4_prompt_embed(const int32_t* toks, int64_t p0, int T, const Hy4PromptIo& io, std::string& err) {
    FastState* F = fast_;
    Hy4State* H = S(hy4_);
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int D = g.hy_dim, HC = g.hc;
    Cv kc{io.keep};
    const Keep K = carve_keep(kc, g, (size_t) T);
    if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "hy4 prompt embedding", err)) return false;   // (the staging's reuse)
    if (H->pf_pending) {   // (the chunk before's reads: done before the history and the rows are rewritten)
        H->pf_pool->wait();
        H->pf_pending = false;
    }
    // the history, the positions and the rows Engram skips
    if ((int64_t) H->hist.size() < p0 + T) {
        H->hist.resize((size_t) (p0 + T), H->eng_pad);
        H->img_pos.resize((size_t) (p0 + T), 0);
    }
    std::vector<unsigned char> skip((size_t) T, 0);
    for (int t = 0; t < T; ++t) {
        const int64_t p = p0 + t;
        H->hist[(size_t) p] = toks[t];
        H->img_pos[(size_t) p] = image_row(p) != nullptr;
        const int32_t tok = toks[t];
        skip[(size_t) t] = H->img_pos[(size_t) p] || tok == H->eng_pad || tok == 120120 || tok == 120124;
    }
    const std::vector<int32_t>& mt = mrope_h_;
    std::vector<hk::Pos3> pos;
    pos.reserve((size_t) T + 3);
    for (int64_t c = p0 - 3; c < p0 + T; ++c) {
        const int64_t cc = std::max<int64_t>(0, c);
        if ((size_t) cc * 3 + 2 < mt.size())
            pos.emplace_back((int64_t) mt[(size_t) cc * 3], (int64_t) mt[(size_t) cc * 3 + 1], (int64_t) mt[(size_t) cc * 3 + 2]);
        else
            pos.emplace_back(cc);
    }
    // (pageable sources: the copies complete before the calls return)
    cudaMemcpyAsync(K.pos, pos.data(), pos.size() * sizeof(hk::Pos3), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(K.skip, skip.data(), skip.size(), cudaMemcpyHostToDevice, s);
    // the embedding rows through the pinned staging, as many rows a pass as it holds
    const strata::qt::Traits* tt = strata::qt::traits((strata::qt::Type) pack_emb_type_);
    const size_t row_b = strata::qt::row_size((strata::qt::Type) pack_emb_type_, D);
    const int per = (int) std::max<size_t>(1, io.emb_floats / (size_t) D);
    for (int t0 = 0; t0 < T; t0 += per) {
        const int n = std::min(per, T - t0);
        const auto row = [&](int t) {
            float* dst = io.emb_h + (size_t) (t - t0) * D;
            if (const float* img = image_row(p0 + t))
                std::memcpy(dst, img, (size_t) D * sizeof(float));
            else
                tt->to_float(pack_emb_src_ + (size_t) toks[t] * row_b, dst, D);
        };
        if (F->workers != nullptr && n >= 256) {
            const int nj = 8;
            F->workers->run(nj, [&](int job) { for (int t = t0 + job; t < t0 + n; t += nj) row(t); });
        } else {
            for (int t = t0; t < t0 + n; ++t) row(t);
        }
        for (int c = 0; c < HC; ++c)
            cudaMemcpy2DAsync(K.R + (size_t) t0 * HC * D + (size_t) c * D, (size_t) HC * D * sizeof(float), io.emb_h,
                              (size_t) D * sizeof(float), (size_t) D * sizeof(float), (size_t) n, cudaMemcpyHostToDevice,
                              s);
        if (t0 + n < T && !glmfast::cuda_ok(cudaStreamSynchronize(s), "hy4 prompt embedding", err)) return false;
    }
    if (H->eng_fd == kEngNone || !H->eng_on || g.hy_eng.empty()) return true;
    // the Engram rows of every row and Engram layer: [layer][row][16 heads] (the token path's hashes, hy4_embed)
    const auto tp = std::chrono::steady_clock::now();
    H->pf_skip.swap(skip);
    if (!hy4_eng_chunk(H, g, p0, T, H->pf_async, err)) return false;
    if (!H->pf_async)
        H->pf_read_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp).count();
    H->pf_reads += (size_t) g.hy_eng.size() * T * 16;
    return true;
}

bool Glm5Model::hy4_prompt_layer(int il, int64_t p0, int T, bool skip_output, const Hy4PromptIo& io, std::string& err) {
    FastState* F = fast_;
    Hy4State* H = S(hy4_);
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const Hy4Layer& Y = H->L[(size_t) il];
    const int D = g.hy_dim, HC = g.hc, nh = Y.heads, E = g.n_embd;
    Cv kc{io.keep};
    const Keep K = carve_keep(kc, g, (size_t) T);
    const int sub = std::min(T, hy4_sub());
    const auto M = [&](const char* nm) {   // (STRATA_GLM_PREFILL_PROF)
        if (io.mark != nullptr) io.mark(io.mark_ctx, nm, (void*) s);
    };
    M("hy4_pre");
    // ---- Engram: the rows of this layer, normed per head, engram_kv as a GEMM, the gated adds into the streams
    if (Y.eng >= 0 && H->eng_fd != kEngNone && H->eng_on && H->pf_rows_h != nullptr) {
        if (!hy4_eng_wait(H, Y.eng, err)) return false;
        for (int t0 = 0; t0 < T; t0 += sub) {
            const int tn = std::min(sub, T - t0);
            Cv c{io.uni};
            const Eng B = carve_eng(c, g, (size_t) tn);
            const size_t k0 = ((size_t) Y.eng * T + t0) * 16;
            if (H->eng_type < 0) {
                cudaMemcpyAsync(B.raw, H->pf_rows_h + k0 * H->eng_rb, (size_t) tn * 16 * 96, cudaMemcpyHostToDevice, s);
                hk::e4m3_to_f32(B.raw, B.rows, tn * 16 * 96, s);
            } else {
                cudaMemcpyAsync(B.rows, H->pf_rowsf_h + k0 * 96, (size_t) tn * 16 * 96 * sizeof(float),
                                cudaMemcpyHostToDevice, s);
            }
            hk::eng_rows_rows(B.rows, Y.elnorm, 16, 96, H->eng_row_eps, B.e, tn, s);
            gb::f32_to_f16(B.e, B.e16, (int64_t) tn * 16 * 96, s);
            if (!gemm16(io, Y.ekv, (HC + 1) * D, 16 * 96, B.e16, 16 * 96, B.kv, (HC + 1) * D, tn, 0.0f, s, err))
                return false;
            hk::add_bias_rows(B.kv, Y.ekv_b, (HC + 1) * D, tn, s);
            hk::eng_add_rows(B.kv, K.R + (size_t) t0 * HC * D, Y.egate, Y.escale, Y.epg, HC, D, H->eng_eps,
                             H->eng_vfirst, K.skip + t0, tn, s);
        }
        M("hy4_engram");
    }
    // ---- the attention: the streams' read for every row, then sub-batches
    hk::hc_mix_rows(K.R, Y.hc_fn[0], HC, D, 2 * HC, g.norm_eps, K.mix, T, s);
    hk::hc_pre_rows(K.mix, 2 * HC, Y.hc_scale[0], Y.hc_base[0], K.R, HC, D, g.hc_eps, g.hc_mag, Y.attn_norm, g.norm_eps,
                    K.x, K.post, T, s);
    gb::f32_to_f16(K.x, K.x16, (int64_t) T * D, s);
    M("hy4_hc");
    const float base = Y.swa ? g.rope_base_swa : g.rope_base;
    const float scale = 1.0f / std::sqrt((float) g.head_dim);
    const int qd = nh * (hk::kNope + hk::kRot);
    for (int t0 = 0; t0 < T; t0 += sub) {
        const int tn = std::min(sub, T - t0);
        const int64_t pt0 = p0 + t0;
        Cv c{io.uni};
        const Attn A = carve_attn(c, g, (size_t) tn, max_ctx_);
        const uint16_t* x16 = K.x16 + (size_t) t0 * D;
        const float* xf = K.x + (size_t) t0 * D;
        if (!gemm16(io, Y.q_a, g.q_lora, D, x16, D, A.qr, g.q_lora, tn, 0.0f, s, err) ||
            !gemm16(io, Y.kv_a, hk::kRow, D, x16, D, A.kv, hk::kRow, tn, 0.0f, s, err))
            return false;
        if (!skip_output) {
            if (!gemm16(io, Y.gate, nh * hk::kV, D, x16, D, A.gate, nh * hk::kV, tn, 0.0f, s, err)) return false;
            hk::rms_rows(A.qr, Y.q_a_norm, g.norm_eps, g.q_lora, A.qr, tn, s);
            gb::f32_to_f16(A.qr, A.qr16, (int64_t) tn * g.q_lora, s);
            if (!gemm16(io, Y.q_b, qd, g.q_lora, A.qr16, g.q_lora, A.q, qd, tn, 0.0f, s, err)) return false;
        }
        M("hy4_proj");
        // the cache rows: a window layer's into a linear buffer of the window before the sub-batch and its rows (the
        // ring's last positions first), back into the ring after the attention; the others' into the cache itself
        const int64_t kb0 = Y.swa ? std::max<int64_t>(0, pt0 - (g.n_swa - 1)) : 0;
        constexpr int kLinear = 1 << 30;
        uint16_t* cache = Y.swa ? A.kbuf : Y.cache;
        const int64_t row_base = Y.swa ? kb0 : 0;
        const int ring = Y.swa ? kLinear : Y.ring;
        if (Y.swa) hk::ring_copy(Y.cache, 0, Y.ring, A.kbuf, kb0, kLinear, kb0, (int) (pt0 - kb0), s);
        hk::mla_prep_rows(A.q, skip_output ? 0 : nh, A.kv, Y.kv_a_norm, g.norm_eps, K.pos + 3 + t0, pt0, base, cache,
                          row_base, ring, tn, s);
        M("hy4_prep");
        bool indexed = false;
        if (!Y.swa) {
            // the indexer: the rows' token keys, the groups they close pooled, the selection where it applies
            if (!gemm32(io, Y.ik.q ? (const float*) Y.ik.q : Y.ik.f32, g.idx_key, D, xf, A.tk, tn, err)) return false;
            hk::rms_rows(A.tk, Y.ik_norm, g.norm_eps, g.idx_key, A.tk, tn, s);
            hk::idx_pool_rows(A.tk, pt0, tn, Y.tk4, Y.cv, Y.cg, Y.ape, Y.cnorm, g.norm_eps, K.pos + t0, g.rope_base,
                              Y.pooled, s);
            M("hy4_ix_key");
            indexed = !skip_output && pt0 + tn > g.idx_top_k;   // (a row past the dense span)
            if (indexed) {
                if (!gemm32(io, (const float*) Y.iproj.q, g.idx_heads, D, xf, A.iw, tn, err) ||
                    !gemm32(io, (const float*) Y.iqb.q, g.idx_heads * g.idx_key, g.q_lora, A.qr, A.iq, tn, err))
                    return false;
                hk::idx_query_rows(A.iq, g.idx_heads, A.iw, K.pos + 3 + t0, g.rope_base, tn, s);
                M("hy4_ix_q");
                gb::dsa_score(A.iq, Y.pooled, A.iw, g.idx_key, g.idx_heads, (int) pt0, 4, tn, (int) ((pt0 + tn) / 4),
                              A.score, A.score_ld, s);
                M("hy4_ix_score");
                hk::idx_select_rows(A.score, A.score_ld, pt0, g.idx_top_k, g.idx_top_k / 4 - 1, A.cells, hk::kMaxCells,
                                    A.n_cells, tn, s);
                M("hy4_ix_select");
            }
        }
        if (skip_output) {
            if (Y.swa)
                hk::ring_copy(A.kbuf, kb0, kLinear, Y.cache, 0, Y.ring, std::max(kb0, pt0 + tn - Y.ring),
                              (int) (pt0 + tn - std::max(kb0, pt0 + tn - Y.ring)), s);
            continue;
        }
        // q_abs = k_b . q_nope per head; the attention; o = v_b . ctx per head, times sigmoid(gate); the output
        hk::q_nope_f16(A.q, nh, tn, A.q16, s);
        if (!head_gemm(io, Y.k_b, Y.kb_f32, 8, nh, hk::kLat, hk::kNope, A.q16, A.qabs, tn, s, err)) return false;
        M("hy4_qabs");
        hk::mla_attend_rows(A.qabs, A.q, cache, row_base, ring, indexed ? A.cells : nullptr, A.n_cells, hk::kMaxCells,
                            pt0, Y.swa ? g.n_swa : 0, Y.sinks, scale, nh, tn, A.ctx, s);
        M(Y.swa ? "hy4_attn_win" : indexed ? "hy4_attn_sel" : "hy4_attn_dense");
        if (Y.swa)
            hk::ring_copy(A.kbuf, kb0, kLinear, Y.cache, 0, Y.ring, std::max(kb0, pt0 + tn - Y.ring),
                          (int) (pt0 + tn - std::max(kb0, pt0 + tn - Y.ring)), s);
        gb::f32_to_f16(A.ctx, A.ctx16, (int64_t) tn * nh * hk::kLat, s);
        if (!head_gemm(io, Y.v_b, Y.vb_f32, 14, nh, hk::kV, hk::kLat, A.ctx16, A.o, tn, s, err)) return false;
        hk::gate_rows(A.o, A.gate, (int64_t) tn * nh * hk::kV, A.o16, s);
        if (!gemm16(io, Y.out, D, nh * hk::kV, A.o16, nh * hk::kV, A.a, D, tn, 0.0f, s, err)) return false;
        hk::hc_add_rows(K.R + (size_t) t0 * HC * D, K.post + (size_t) t0 * HC, A.a, Y.post_attn, g.norm_eps, HC, D, tn, s);
        M("hy4_attn_out");
    }
    if (skip_output) return true;
    // ---- the FFN's read
    hk::hc_mix_rows(K.R, Y.hc_fn[1], HC, D, 2 * HC, g.norm_eps, K.mix, T, s);
    hk::hc_pre_rows(K.mix, 2 * HC, Y.hc_scale[1], Y.hc_base[1], K.R, HC, D, g.hc_eps, g.hc_mag, Y.ffn_norm, g.norm_eps,
                    K.x, K.post, T, s);
    gb::f32_to_f16(K.x, K.x16, (int64_t) T * D, s);
    M("hy4_hc");
    if (Y.moe) {   // the latent: the experts' input, for the prompt path's MoE
        if (!gemm16(io, Y.rdown, E, D, K.x16, D, io.x, E, T, 0.0f, s, err)) return false;
        gb::f32_to_f16(io.x, io.x16, (int64_t) T * E, s);
        M("hy4_rdown");
        return true;
    }
    // layer 0's dense SwiGLU (no clamp), post_ffw_norm, the add
    const int NF = g.n_ff_dense;
    for (int t0 = 0; t0 < T; t0 += sub) {
        const int tn = std::min(sub, T - t0);
        Cv c{io.uni};
        const Ffn B = carve_ffn(c, g, (size_t) tn);
        const uint16_t* x16 = K.x16 + (size_t) t0 * D;
        if (!gemm16(io, Y.dg, NF, D, x16, D, B.dg, NF, tn, 0.0f, s, err) ||
            !gemm16(io, Y.du, NF, D, x16, D, B.du, NF, tn, 0.0f, s, err))
            return false;
        gb::swiglu_f16(B.dg, B.du, B.h16, (int64_t) tn * NF, INFINITY, s);
        if (!gemm16(io, Y.dd, D, NF, B.h16, NF, B.r, D, tn, 0.0f, s, err)) return false;
        hk::rms_rows(B.r, Y.post_ffw, g.norm_eps, D, B.r, tn, s);
        hk::hc_add_rows(K.R + (size_t) t0 * HC * D, K.post + (size_t) t0 * HC, B.r, nullptr, g.norm_eps, HC, D, tn, s);
    }
    M("hy4_dense");
    return true;
}

bool Glm5Model::hy4_prompt_router(int il, int T, float* logits, const Hy4PromptIo& io, std::string& err) {
    Hy4State* H = S(hy4_);
    const Glm5Geometry& g = g_;
    const Hy4Layer& Y = H->L[(size_t) il];
    Cv kc{io.keep};
    const Keep K = carve_keep(kc, g, (size_t) T);
#if defined(STRATA_USE_HIP)
    (void) err;
    gb::router_f32((const float*) Y.router.q, g.n_expert, K.x, T, g.hy_dim, logits, fast_->cs);
    return true;
#else
    return gemm32(io, (const float*) Y.router.q, g.n_expert, g.hy_dim, K.x, logits, T, err);
#endif
}

bool Glm5Model::hy4_prompt_shared(int il, int T, float* sg, float* su, uint16_t* sh16, int sub, const Hy4PromptIo& io,
                                  std::string& err) {
    Hy4State* H = S(hy4_);
    const Glm5Geometry& g = g_;
    cudaStream_t s = fast_->cs;
    const Hy4Layer& Y = H->L[(size_t) il];
    const int D = g.hy_dim, FS = g.n_ff_shexp;
    Cv kc{io.keep};
    const Keep K = carve_keep(kc, g, (size_t) T);
    for (int t0 = 0; t0 < T; t0 += sub) {
        const int tn = std::min(sub, T - t0);
        const uint16_t* x16 = K.x16 + (size_t) t0 * D;
        if (!gemm16(io, Y.shg, FS, D, x16, D, sg, FS, tn, 0.0f, s, err) ||
            !gemm16(io, Y.shu, FS, D, x16, D, su, FS, tn, 0.0f, s, err))
            return false;
        gb::swiglu_f16(sg, su, sh16, (int64_t) tn * FS, INFINITY, s);   // (the shared expert: no clamp)
        if (!gemm16(io, Y.shd, D, FS, sh16, FS, K.sh + (size_t) t0 * D, D, tn, 0.0f, s, err)) return false;
    }
    // (the routed rows are added into ffn: it starts at zero - the shared expert has the residual's width)
    cudaMemsetAsync(io.ffn, 0, (size_t) T * g.n_embd * sizeof(float), s);
    return true;
}

bool Glm5Model::hy4_prompt_ffn_out(int il, int T, const Hy4PromptIo& io, std::string& err) {
    Hy4State* H = S(hy4_);
    const Glm5Geometry& g = g_;
    cudaStream_t s = fast_->cs;
    const Hy4Layer& Y = H->L[(size_t) il];
    const int D = g.hy_dim, HC = g.hc, E = g.n_embd;
    Cv kc{io.keep};
    const Keep K = carve_keep(kc, g, (size_t) T);
    const int sub = std::min(T, hy4_sub());
    for (int t0 = 0; t0 < T; t0 += sub) {
        const int tn = std::min(sub, T - t0);
        Cv c{io.uni};
        const Ffn B = carve_ffn(c, g, (size_t) tn);
        gb::f32_to_f16(io.ffn + (size_t) t0 * E, B.h16, (int64_t) tn * E, s);
        if (!gemm16(io, Y.rup, D, E, B.h16, E, B.r, D, tn, 0.0f, s, err)) return false;
        hk::rms_rows(B.r, Y.post_ffw, g.norm_eps, D, B.r, tn, s, K.sh + (size_t) t0 * D);
        hk::hc_add_rows(K.R + (size_t) t0 * HC * D, K.post + (size_t) t0 * HC, B.r, nullptr, g.norm_eps, HC, D, tn, s);
    }
    if (hk::launch_errors() > 0) {
        err = "hy4 prompt: " + std::to_string(hk::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return false;
    }
    return true;
}

}  // namespace strata::core
