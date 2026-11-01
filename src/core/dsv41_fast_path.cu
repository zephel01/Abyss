// src/core/dsv41_fast_path.cu - the deepseek41 (DeepSeek V4.1 Flash) pieces of the fast decode path (Glm5Model
// members).  A V4.1 layer is a deepseek4 layer (dsv4_fast_path.cu) with these differences (Abyss-tools/worklog/
// 2026-10-18_deepseek41-spec.md):
//   * the compressed rows live in 4 KV slots, written by the KV-source layers (2, 8, 14, 20: rows of 2 tokens pooled
//     by a per-channel gate, or 1 token) with their index keys; the other compressed layers read their slot;
//   * the index sources (the KV sources and 24, 28, 32, 36) compute the top-k, the layers after them reuse it; above
//     the candidate-source layer the scores are masked to its candidate blocks;
//   * no per-head q norm, no hash routing, an F32 router, rms eps 1e-20;
//   * Engram (layers 1 and 14): rows of two n-gram-hashed tables, read from the GGUF per token, projected and gated
//     into the 4 streams before the layer's hyper-connection read;
//   * the hyper-connections collapse with the previous sublayer's pre (glm_fast_path.cu fast_layers), and the head
//     with the last FFN's.
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/core/knobs.hpp"
#include "strata/kernels/dsv41_fast.hpp"
#include "strata/kernels/dsv4_fast.hpp"
#include "strata/kernels/glm_fast.hpp"

#include "ggml.h"
#include <cuda_runtime.h>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <thread>
#include <mutex>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace ds = strata::kernels::dsv4;
namespace d41 = strata::kernels::dsv41;

namespace {
ds::Rope layer_rope(const Glm5Geometry& g, bool comp) {
    if (!comp) return ds::make_rope(g.n_rot, g.rope_base, 1.0f, 0.0f, 0, 0.0f, 0.0f);
    return ds::make_rope(g.n_rot, g.comp_rope_base, 1.0f / g.yarn_factor, 1.0f, g.yarn_orig_ctx, g.yarn_beta_fast,
                         g.yarn_beta_slow);
}
size_t al(size_t b) { return (b + 255u) & ~(size_t) 255u; }

constexpr int kEngCols = 24, kEngDim = 256, kEngRowMax = 272, kEngDead = -1;
// an image row's Engram "row number": no table row, the row decodes to zeros (llama.cpp's engram_keep = 0: the layer's
// add is x += g * val with val = engram_kv(rows) = 0, so the stream is unchanged)
constexpr uint64_t kEngRowZero = ~0ull;
constexpr int kV41Blk = 256;   // the prompt path's Engram rows a block

// the Engram reads' own threads (the fast path's workers are the MoE service's: the token's rows are read before
// layer 0, a prompt chunk's while the service is idle - either way not on the workers' callers): run(n, fn) calls
// fn(0..n-1) across them and the caller
struct EngPool {
    std::mutex m;
    std::condition_variable cv, done_cv;
    std::function<void(int)> fn;
    int n = 0, left = 0;
    std::atomic<int> next{0};
    uint64_t gen = 0;
    std::vector<std::thread> th;
    explicit EngPool(int threads) {
        for (int i = 0; i < threads; ++i)
            th.emplace_back([this] {
                uint64_t seen = 0;
                for (;;) {
                    std::unique_lock<std::mutex> l(m);
                    cv.wait(l, [&] { return gen != seen; });
                    seen = gen;
                    l.unlock();
                    drain();
                }
            });
        for (auto& t : th) t.detach();   // (lives as long as the process; idle threads sleep on cv)
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

// the Engram tables (pack/engram.txt, engram_token_map.u32 - tools/iq_pack.py write_engram) and the n-gram history
struct V41Engram {
    int gtype = -1;            // llama.cpp's layout: a ggml row type (Q8_0 / Q2_K ...) decoded by ggml; ds4: e4m3 + e8m0 rows (264)
    int row_bytes = 264;
#ifdef _WIN32
    HANDLE h[2] = {INVALID_HANDLE_VALUE, INVALID_HANDLE_VALUE};   // the tables' files, overlapped
#else
    int fd[2] = {-1, -1};
#endif
    uint64_t off[2] = {0, 0}, rows[2] = {0, 0};
    int layer[2] = {-1, -1};
    uint32_t primes[2][kEngCols] = {};
    uint64_t mult[2][4] = {};
    uint32_t pad = 2;
    std::vector<uint32_t> tmap;
    std::vector<int32_t> hist;   // the compressed id of the token at each position (the n-grams read the three before -
                                 // a rewind to any position stays exact); kEngDead past a sequence start
    std::vector<uint64_t> chunk; // the prompt path's chunk: 2 x 24 row numbers a token
    uint32_t lut[256][256];   // (scale, code) -> the BF16 value's F32 bits (ds4_engram.c init_engram_bf16); 0xffffffff
                              // marks an invalid pair
    // the RAW ROW CACHE (Xolotl's: the n-grams repeat - a few common ones hold most rows): direct-mapped, a row's raw
    // bytes by (table, row); STRATA_DS41_ENGRAM_CACHE_MB (default 256, ~1M rows; 0: none).  The page cache keeps a
    // 4 KB page for each 264-byte row, 15x the room
    std::vector<uint64_t> tag;   // (table << 63 | row) + 1, 0: empty
    std::vector<uint8_t> cache;
    uint64_t cmask = 0;
    std::mutex stripe[256];
    std::atomic<uint64_t> hits{0}, misses{0};
    std::unique_ptr<EngPool> pool;
    // the token's rows in the BACKGROUND (STRATA_DS41_ENGRAM_ASYNC, default on): a decode token's 48 row reads used to
    // hold the host before layer 0's launch while the GPU idled (the token is known only once the last one ended);
    // now one thread reads table 0's 24 rows, flags them, then table 1's, and the device waits at layers 1 / 14
    std::mutex am;
    std::condition_variable acv;
    bool a_have = false, a_busy = false, a_bad = false;
    std::string a_err;
    uint64_t a_rows[2 * kEngCols] = {};
    float* a_dst = nullptr;
    unsigned int* a_flag = nullptr;
    unsigned int a_seq = 0;
    bool a_started = false;   // the worker, started once (detached like the pools: it lives as long as the process)
    // a prompt chunk's rows PREFETCHED into the row cache (table 0's while layer 0 runs, then table 1's for layer 14)
    // (STRATA_DS41_ENGRAM_PREFETCH, default on; needs the row cache): its own few threads, stopped at the next chunk or
    // token - the rows are read the same way, so the prompt's sums are unchanged
    std::unique_ptr<EngPool> ppool;
    std::thread p_th;
    std::atomic<bool> p_stop{false};
    std::atomic<uint64_t> p_reads{0};
};

float e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = e ? std::ldexp((float) (8 + m), e - 10) : std::ldexp((float) m, -9);
    return b & 128 ? -v : v;
}

bool engram_open(const std::string& pack, V41Engram& E, std::string& err) {
    std::ifstream f(pack + "/engram.txt");
    if (!f) { err = "deepseek41: " + pack + "/engram.txt is missing (re-pack with tools/iq_pack.py)"; return false; }
    std::string line;
    int tables = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        std::string key;
        is >> key;
        if (key == "format") {
            std::string name;
            is >> name >> E.row_bytes;
            if (name == "ggml") {   // format ggml <type> <row bytes>
                E.gtype = E.row_bytes;
                is >> E.row_bytes;
            }
            const ggml_type_traits* tt = E.gtype >= 0 && E.gtype < GGML_TYPE_COUNT ? ggml_get_type_traits((ggml_type) E.gtype) : nullptr;
            if ((E.gtype >= 0 && (tt == nullptr || tt->to_float == nullptr || E.row_bytes <= 0 || E.row_bytes > kEngRowMax ||
                                  ggml_row_size((ggml_type) E.gtype, kEngDim) != (size_t) E.row_bytes)) ||
                (E.gtype < 0 && (name != "row264" || E.row_bytes != 264))) {
                err = "deepseek41: engram.txt: unknown row format " + name;
                return false;
            }
        } else if (key == "table") {
            int k = -1, il = -1;
            uint64_t off = 0, rows = 0;
            is >> k >> il >> off >> rows;
            std::string path;
            std::getline(is, path);
            path.erase(0, path.find_first_not_of(' '));
            if (k < 0 || k > 1 || path.empty()) { err = "deepseek41: engram.txt: bad table line"; return false; }
#ifdef _WIN32
            // overlapped: Windows runs the reads of a synchronous handle one at a time, and the pool reads from 16
            // threads (issue #6)
            E.h[k] = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED, nullptr);
            if (E.h[k] == INVALID_HANDLE_VALUE) { err = "deepseek41: cannot open the Engram table in " + path; return false; }
#else
            E.fd[k] = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
            if (E.fd[k] < 0) { err = "deepseek41: cannot open the Engram table in " + path; return false; }
            posix_fadvise(E.fd[k], 0, 0, POSIX_FADV_RANDOM);   // (264-byte rows at random: no readahead)
#endif
            E.off[k] = off;
            E.rows[k] = rows;
            E.layer[k] = il;
            ++tables;
        } else if (key == "primes") {
            for (int k = 0; k < 2; ++k)
                for (int c = 0; c < kEngCols; ++c) is >> E.primes[k][c];
        } else if (key == "multipliers") {
            for (int k = 0; k < 2; ++k)
                for (int j = 0; j < 4; ++j) is >> E.mult[k][j];
        } else if (key == "pad") {
            is >> E.pad;
        }
    }
    if (tables != 2) { err = "deepseek41: engram.txt names " + std::to_string(tables) + " tables, not 2"; return false; }
    for (int k = 0; k < 2; ++k) {
        uint64_t sum = 0;
        for (int c = 0; c < kEngCols; ++c) sum += E.primes[k][c];
        if (sum > E.rows[k]) { err = "deepseek41: the Engram buckets run past the table's rows"; return false; }
    }
    std::ifstream t(pack + "/engram_token_map.u32", std::ios::binary | std::ios::ate);
    if (!t) { err = "deepseek41: " + pack + "/engram_token_map.u32 is missing"; return false; }
    const std::streamsize n = t.tellg();
    t.seekg(0);
    E.tmap.resize((size_t) n / 4);
    t.read((char*) E.tmap.data(), (std::streamsize) E.tmap.size() * 4);
    for (unsigned s = 0; s < 256; ++s)
        for (unsigned c = 0; c < 256; ++c) {
            uint32_t r = 0xffffffffu;
            if ((c & 127u) != 127u && s != 255u) {
                const float v = std::ldexp(e4m3((uint8_t) c), (int) s - 127);
                uint32_t bits;
                std::memcpy(&bits, &v, 4);
                bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;   // BF16, nearest even
                if ((bits & 0x7f800000u) != 0x7f800000u) r = bits;
            }
            E.lut[s][c] = r;
        }
    return true;
}
}  // namespace

bool Glm5Model::dsv41_bind(std::string& err) {
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
        // a quantized slot, or a BF16 tensor the packer converted (served by the BF16 GEMV)
        const auto q = [&](const char* n, int64_t n_in, int64_t n_out) -> WSlot {
            auto it = ws_map_.find(P + n);
            if (it != ws_map_.end() && it->second.type != 0 && gf::mv_supported(it->second.type)) return it->second;
            const auto b = w16_.find(P + n);
            if (b != w16_.end()) return WSlot{nullptr, gf::kTypeBF16, b->second, n_in, n_out};
            missing += P + n + " (quant / bf16) ";
            return WSlot{};
        };
        const auto has = [&](const char* n) {
            return ws_map_.count(P + n) != 0 || w16_.count(P + n) != 0 || w_.count(P + n) != 0;
        };
        // the roles come from the GGUF's lists (ds4) or ds4's fixed ones (llama.cpp's converter): the tensors must agree
        if (has("attn_compressor_kv.weight") != (g.kv_src[(size_t) il] != 0) ||
            has("indexer.attn_q_b.weight") != (g.idx_src[(size_t) il] != 0)) {
            err = "deepseek41: layer " + std::to_string(il) + "'s compressor / indexer tensors disagree with its role "
                  "(KV source " + std::to_string((int) g.kv_src[(size_t) il]) + ", index source " +
                  std::to_string((int) g.idx_src[(size_t) il]) + ")";
            return false;
        }
        Ly.ratio = g.ratio(il);
        Ly.slot = g.kv_slot[(size_t) il];
        Ly.kv_src = g.kv_src[(size_t) il] != 0;
        Ly.idx_src = g.idx_src[(size_t) il] != 0;
        Ly.eng = g.eng_slot[(size_t) il];
        Ly.q_a = q("attn_q_a.weight", g.n_embd, g.q_lora);
        Ly.q_b = q("attn_q_b.weight", g.q_lora, (int64_t) g.n_head * g.head_dim);
        Ly.kv = q("attn_kv.weight", g.n_embd, g.head_dim);
        Ly.out_a = q("attn_output_a.weight", (int64_t) g.n_head / g.o_groups * g.head_dim, (int64_t) g.o_groups * g.o_lora);
        Ly.out_b = q("attn_output_b.weight", (int64_t) g.o_groups * g.o_lora, g.n_embd);
        Ly.q_a_norm = f32("attn_q_a_norm.weight");
        Ly.kv_a_norm = f32("attn_kv_a_norm.weight");
        Ly.sinks = f32("attn_sinks.weight");
        if (Ly.kv_src) {
            Ly.comp_kv = q("attn_compressor_kv.weight", g.n_embd, g.head_dim);
            if (Ly.ratio == 2) Ly.comp_gate = q("attn_compressor_gate.weight", g.n_embd, g.head_dim);
            Ly.comp_norm = f32("attn_compressor_norm.weight");
            Ly.idx_k_w = q("indexer.attn_k.weight", g.head_dim, g.idx_key);
            Ly.idx_k_norm = f32("indexer.k_norm.weight");
            if (Ly.idx_k_w.type != gf::kTypeBF16) missing += P + "indexer.attn_k.weight (must be BF16) ";
        }
        if (Ly.idx_src) {
            Ly.idx_q_b_q = q("indexer.attn_q_b.weight", g.q_lora, (int64_t) g.idx_heads * g.idx_key);
            Ly.idx_proj_f = f32("indexer.proj.weight");
        }
        if (Ly.eng >= 0) {   // (ds4's names, then llama.cpp's)
            const bool l = !has("engram_kv.weight");
            Ly.eng_kv = q(l ? "engram_wkv.weight" : "engram_kv.weight", kEngCols * kEngDim, 5 * (int64_t) g.n_embd);
            if (!l && has("engram_gate.weight")) {
                // llama.cpp PR 30817 (unsloth): the q and k norm weights folded into one gate (q * k); the key's
                // weight is then ones (dsv41_alloc)
                Ly.eng_qn = f32("engram_gate.weight");
                Ly.eng_kn = nullptr;
            } else {
                Ly.eng_qn = f32(l ? "engram_q.weight" : "engram_q_norm.weight");
                Ly.eng_kn = f32(l ? "engram_k.weight" : "engram_k_norm.weight");
            }
        }
    }
    if (!missing.empty()) {
        err = "deepseek41: unresolved weights: " + missing.substr(0, 800);
        return false;
    }
    if (n_parts_ > 1 || l0_ != 0 || l1_ != g.n_layers) {
        err = "deepseek41: layer splits are not implemented";
        return false;
    }
    auto* E = new V41Engram();
    if (!engram_open(pack_dir_, *E, err)) {
        delete E;
        return false;
    }
    if (E->tmap.size() < (size_t) g.n_vocab) {
        delete E;
        err = "deepseek41: the Engram token map is shorter than the vocabulary";
        return false;
    }
    for (int k = 0; k < 2; ++k)
        if (E->layer[k] < 0 || E->layer[k] >= g.n_layers || g.eng_slot[(size_t) E->layer[k]] != k) {
            delete E;
            err = "deepseek41: engram.txt's layers disagree with deepseek41.engram.layer_ids";
            return false;
        }
    E->hist.assign((size_t) max_ctx_ + 1, kEngDead);
    {   // the raw row cache (a power of two of rows) and the read threads
        const uint64_t mb = (uint64_t) strata::core::env_num("STRATA_DS41_ENGRAM_CACHE_MB", 256, 0, 65536);
        uint64_t n = 1;
        while (n * 2 * (uint64_t) E->row_bytes <= mb << 20) n *= 2;
        if (mb > 0 && n >= 1024) {
            E->tag.assign(n, 0ull);
            E->cache.resize(n * (uint64_t) E->row_bytes);
            E->cmask = n - 1;
        }
        // (the reader threads live as long as the process: a bound, so a typo does not start hundreds)
        E->pool = std::make_unique<EngPool>((int) strata::core::env_num("STRATA_DS41_ENGRAM_THREADS", 15, 0, 64));
        std::fprintf(stderr, "deepseek41: Engram row cache %.0f MB (%llu rows), %zu read threads\n",
                     (double) E->cache.size() / 1048576.0, (unsigned long long) (E->cmask ? E->cmask + 1 : 0),
                     E->pool->th.size() + 1);
    }
    v41_eng_ = E;
    return true;
}

bool Glm5Model::dsv41_alloc(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const size_t D = (size_t) g.head_dim, K = (size_t) g.idx_key, ctx = (size_t) std::max<int64_t>(max_ctx_, 1);
    ds_ring_ = g.n_swa + kSpecSlack;
    // the slots' row counts: ctx / ratio + 1 (ds4.c: ctx/2+1 for the ratio-2 slots, ctx+1 for the ratio-1 one)
    size_t slot_rows[4] = {0, 0, 0, 0};
    int slot_ratio[4] = {0, 0, 0, 0};
    for (int il = 0; il < g.n_layers; ++il)
        if (g.kv_src[(size_t) il]) {
            const int s = g.kv_slot[(size_t) il];
            if (s < 0 || s > 3) { err = "deepseek41: more than 4 KV sources"; return false; }
            slot_ratio[s] = g.ratio(il);
            slot_rows[s] = ctx / (size_t) g.ratio(il) + 1;
        }
    const size_t B = ctx / (size_t) g.cand_bs + 2;
    size_t bytes = 0;
    bytes += (size_t) g.n_layers * al((size_t) ds_ring_ * D * 2);
    for (int s = 0; s < 4; ++s)
        if (slot_rows[s]) bytes += al(slot_rows[s] * D * 2) + al(slot_rows[s] * K * 2) + al(FastState::kV41PendRing * 2 * D * 4);
    const size_t HD = (size_t) g.n_head * D, OL = (size_t) g.o_groups * g.o_lora;
    bytes += 2 * al(D * 4) + al(HD * 4) + al(HD / 32 * 36) + al(OL * 4) + al(OL / 32 * 36) + al((size_t) g.idx_top_k * 4);
    bytes += 2 * al(4 * 4) + al((ctx + 1) * 4) + al(B * 4) + al((size_t) g.cand_blocks * 4) + al(B);
    bytes += al((size_t) 2 * kEngCols * kEngDim * 4) + al((size_t) 5 * g.n_embd * 4) + al((size_t) 2 * kEngCols * kEngDim / 32 * 36) +
             al((size_t) kV41Blk * kEngCols * kEngDim * 4) + al((size_t) g.hc * g.n_embd * 4);
    // exp_probs_b_vl (the image rows' router bias, n_layers x n_expert F32): STRATA_GLM_VL_BIAS=<file>, else
    // <pack>/exp_probs_b_vl.f32 (tools/iq_pack.py writes it when the GGUF has it); none: an image request is refused
    std::vector<float> vl_h;
    {
        std::string vp;
        if (const char* e = getenv("STRATA_GLM_VL_BIAS"); e != nullptr && e[0]) vp = e;
        else if (!pack_dir_.empty() && std::filesystem::exists(pack_dir_ + "/exp_probs_b_vl.f32")) vp = pack_dir_ + "/exp_probs_b_vl.f32";
        if (!vp.empty()) {
            const size_t want = (size_t) g.n_layers * (size_t) g.n_expert * 4;
            std::ifstream f(vp, std::ios::binary | std::ios::ate);
            if (!f) { err = "deepseek41: cannot open the image router bias " + vp; return false; }
            const size_t sz = (size_t) f.tellg();
            if (sz != want) {
                err = "deepseek41: the image router bias " + vp + " is " + std::to_string(sz) + " bytes, expected " +
                      std::to_string(want) + " (" + std::to_string(g.n_layers) + " layers x " + std::to_string(g.n_expert) +
                      " experts x 4)";
                return false;
            }
            vl_h.resize(want / 4);
            f.seekg(0);
            f.read((char*) vl_h.data(), (std::streamsize) want);
            if (!f) { err = "deepseek41: short read of the image router bias " + vp; return false; }
            for (float v : vl_h)
                if (!std::isfinite(v)) { err = "deepseek41: the image router bias " + vp + " has a non-finite value"; return false; }
            bytes += al(want);
        }
    }
    if (cudaMalloc(&dsv41_state_, bytes) != cudaSuccess) {
        cudaGetLastError();
        err = "deepseek41: the caches (" + std::to_string(bytes >> 20) + " MB for " + std::to_string(ctx) +
              " tokens) did not allocate";
        return false;
    }
    cudaMemset(dsv41_state_, 0, bytes);
    uint8_t* at = (uint8_t*) dsv41_state_;
    const auto take = [&](size_t b) {
        uint8_t* p = at;
        at += al(b);
        return p;
    };
    for (int il = 0; il < g.n_layers; ++il) F->L[(size_t) il].raw = (uint16_t*) take((size_t) ds_ring_ * D * 2);
    for (int s = 0; s < 4; ++s)
        if (slot_rows[s]) {
            F->v41_comp[s] = (uint16_t*) take(slot_rows[s] * D * 2);
            F->v41_idx[s] = (uint16_t*) take(slot_rows[s] * K * 2);
            F->v41_pend[s] = (float*) take(FastState::kV41PendRing * 2 * D * 4);
        }
    for (int il = 0; il < g.n_layers; ++il) {
        auto& Ly = F->L[(size_t) il];
        if (Ly.slot >= 0) {
            Ly.comp_cache = F->v41_comp[Ly.slot];
            Ly.idx_cache = F->v41_idx[Ly.slot];
        }
    }
    F->ds_ckv[0] = (float*) take(D * 4);
    F->ds_csc[0] = (float*) take(D * 4);
    F->ds_o = (float*) take(HD * 4);
    F->ds_oq = take(HD / 32 * 36);
    F->ds_oa = (float*) take(OL * 4);
    F->ds_oaq = take(OL / 32 * 36);
    F->ds_sel = (int*) take((size_t) g.idx_top_k * 4);
    F->v41_pre[0] = (float*) take(4 * 4);
    F->v41_pre[1] = (float*) take(4 * 4);
    F->v41_score = (float*) take((ctx + 1) * 4);
    F->v41_bscore = (float*) take(B * 4);
    F->v41_bsel = (int*) take((size_t) g.cand_blocks * 4);
    F->v41_keep = (unsigned char*) take(B);
    F->v41_eng = (float*) take((size_t) 2 * kEngCols * kEngDim * 4);
    F->v41_eng_kv = (float*) take((size_t) 5 * g.n_embd * 4);
    F->v41_eng_q = take((size_t) 2 * kEngCols * kEngDim / 32 * 36);
    F->v41_pblk = (float*) take((size_t) kV41Blk * kEngCols * kEngDim * 4);
    {   // a folded Engram gate's key weight (dsv41_bind): ones
        float* ones = (float*) take((size_t) g.hc * g.n_embd * 4);
        const std::vector<float> h((size_t) g.hc * g.n_embd, 1.0f);
        cudaMemcpy(ones, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
        for (int il = l0_; il < l1_; ++il)
            if (F->L[(size_t) il].eng >= 0 && F->L[(size_t) il].eng_kn == nullptr) F->L[(size_t) il].eng_kn = ones;
    }
    if (!vl_h.empty()) {
        float* vl_d = (float*) take(vl_h.size() * 4);
        cudaMemcpy(vl_d, vl_h.data(), vl_h.size() * 4, cudaMemcpyHostToDevice);
        for (int il = l0_; il < l1_; ++il) {   // (a hash-routed layer has no bias to swap)
            auto& Ly = F->L[(size_t) il];
            if (il < g.n_layers && Ly.router_bias != nullptr) Ly.router_bias_vl = vl_d + (size_t) il * g.n_expert;
        }
        vl_bias_ok_ = true;
    }
    if (cudaMallocHost(&F->v41_pblk_h, (size_t) kV41Blk * kEngCols * kEngDim * 4) != cudaSuccess) {
        cudaGetLastError();
        err = "deepseek41: the prompt path's Engram staging buffer did not allocate";
        return false;
    }
    if (cudaMallocHost(&F->v41_eng_h, (size_t) 2 * kEngCols * kEngDim * 4) != cudaSuccess) {
        cudaGetLastError();
        err = "deepseek41: the Engram staging buffer did not allocate";
        return false;
    }
    if (cudaMallocHost(&F->v41_eng_flag, 2 * sizeof(unsigned int)) != cudaSuccess) {
        cudaGetLastError();
        F->v41_eng_flag = nullptr;   // (the rows are then read before layer 0, as before)
    } else {
        F->v41_eng_flag[0] = F->v41_eng_flag[1] = 0u;
    }
    std::fprintf(stderr, "deepseek41: CUDA%d attention caches %.2f GB for %lld tokens (KV slots of ratio %d/%d/%d/%d)\n",
                 dev_, (double) bytes / 1073741824.0, (long long) ctx, slot_ratio[0], slot_ratio[1], slot_ratio[2],
                 slot_ratio[3]);
    return true;
}

void Glm5Model::dsv41_engram_report() const {
    if (const auto* E = (const V41Engram*) v41_eng_) {
        const uint64_t h = E->hits.load(), m = E->misses.load();
        std::fprintf(stderr, "deepseek41 Engram CUDA%d: %llu rows, %.1f%% from the row cache, %llu read (+%llu prefetched)\n",
                     dev_, (unsigned long long) (h + m), 100.0 * (double) h / (double) std::max<uint64_t>(1, h + m),
                     (unsigned long long) m, (unsigned long long) E->p_reads.load());
    }
}

void Glm5Model::dsv41_reset() {
    if (auto* E = (V41Engram*) v41_eng_) std::fill(E->hist.begin(), E->hist.end(), kEngDead);
}

namespace {
// the 48 row numbers (2 tables x 24 heads) of the token at position p; records its compressed id at p
// img: position p is an image row - its id is recorded DEAD (the n-grams of the text after it stop at the image, as
// llama.cpp's cid() of an embedding token) and its rows are the zero row (no Engram contribution)
bool engram_rows(V41Engram& E, int32_t tok, int64_t p, uint64_t* rows, std::string& err, bool img = false) {
    if (tok < 0 || (size_t) tok >= E.tmap.size()) { err = "deepseek41: Engram: token out of range"; return false; }
    if (p < 0 || (size_t) p >= E.hist.size()) { err = "deepseek41: Engram: position past the context"; return false; }
    const int32_t cur = img ? kEngDead : (int32_t) E.tmap[(size_t) tok];
    E.hist[(size_t) p] = cur;
    if (img) {
        for (int j = 0; j < 2 * kEngCols; ++j) rows[j] = kEngRowZero;
        return true;
    }
    uint32_t ids[4];
    bool blocked = false;
    for (int j = 0; j < 4; ++j) {
        const int32_t id = j == 0 ? cur : p - j >= 0 ? E.hist[(size_t) (p - j)] : kEngDead;
        blocked |= id == kEngDead;
        ids[j] = blocked ? E.pad : (uint32_t) id;
    }
    for (int k = 0; k < 2; ++k) {
        uint64_t h = (uint64_t) ids[0] * E.mult[k][0];
        uint64_t off = 0;
        for (int j = 1; j < 4; ++j) {
            h ^= (uint64_t) ids[j] * E.mult[k][j];
            for (int hd = 0; hd < 8; ++hd) {
                const int col = (j - 1) * 8 + hd;
                const uint32_t prime = E.primes[k][col];
                rows[k * kEngCols + col] = h % prime + off;
                off += prime;
            }
        }
    }
    return true;
}

// n bytes of table k's file at byte at (64-bit: the tables sit past 100 GB) -> dst
static bool engram_read(V41Engram& E, int k, uint64_t at, uint8_t* dst, size_t n) {
    size_t done = 0;
#ifdef _WIN32
    // each read waits on this thread's own event; closed when its thread ends
    struct Ev {
        HANDLE ev = CreateEventA(nullptr, TRUE, FALSE, nullptr);
        ~Ev() { if (ev != nullptr) CloseHandle(ev); }
    };
    static thread_local Ev ev;
    if (ev.ev == nullptr) return false;
    while (done < n) {
        OVERLAPPED ov{};
        ov.Offset = (DWORD) (at + done);
        ov.OffsetHigh = (DWORD) ((at + done) >> 32);
        ov.hEvent = ev.ev;
        DWORD r = 0;
        if (!ReadFile(E.h[k], dst + done, (DWORD) (n - done), nullptr, &ov) && GetLastError() != ERROR_IO_PENDING)
            return false;
        if (!GetOverlappedResult(E.h[k], &ov, &r, TRUE) || r == 0) return false;
        done += (size_t) r;
    }
#else
    while (done < n) {
        const ssize_t r = pread(E.fd[k], dst + done, n - done, (off_t) (at + done));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        done += (size_t) r;
    }
#endif
    return true;
}

// table k's 24 rows (row numbers rows[0..23]) -> dst, 24 x 256 floats
// one row of table k -> dst (256 floats), through the raw row cache
bool engram_row(V41Engram& E, int k, uint64_t row, float* dst, std::string& err) {
    if (row == kEngRowZero) {   // an image row's: nothing to read
        std::memset(dst, 0, kEngDim * sizeof(float));
        return true;
    }
    uint8_t raw[kEngRowMax];
    const int RB = E.row_bytes;
    bool hit = false;
    const uint64_t key = (((uint64_t) k << 63) | row) + 1;
    uint64_t slot = 0;
    if (E.cmask != 0) {
        slot = (row * 0x9E3779B97F4A7C15ull ^ (uint64_t) k) & E.cmask;
        std::lock_guard<std::mutex> l(E.stripe[slot & 255]);
        if (E.tag[slot] == key) {
            std::memcpy(raw, E.cache.data() + slot * (uint64_t) RB, (size_t) RB);
            hit = true;
        }
    }
    if (hit) {
        E.hits.fetch_add(1, std::memory_order_relaxed);
    } else {
        if (!engram_read(E, k, E.off[k] + row * (uint64_t) RB, raw, (size_t) RB)) {
            err = "deepseek41: Engram row read failed";
            return false;
        }
        E.misses.fetch_add(1, std::memory_order_relaxed);
        if (E.cmask != 0) {
            std::lock_guard<std::mutex> l(E.stripe[slot & 255]);
            E.tag[slot] = key;
            std::memcpy(E.cache.data() + slot * (uint64_t) RB, raw, (size_t) RB);
        }
    }
    if (E.gtype >= 0) {   // llama.cpp's get_rows: ggml's dequantizer, F32 (no BF16 rounding)
        ggml_get_type_traits((ggml_type) E.gtype)->to_float(raw, dst, kEngDim);
        return true;
    }
    uint32_t* du = (uint32_t*) dst;
    for (int i = 0; i < kEngDim; ++i) {
        const uint32_t v = E.lut[raw[kEngDim + i / 32]][raw[i]];
        if (v == 0xffffffffu) { err = "deepseek41: an invalid Engram row (NaN code or scale)"; return false; }
        du[i] = v;
    }
    return true;
}

bool engram_rows_par(V41Engram& E, int n, const uint64_t* rows, int kk, const int* ks, float* dst, std::string& err);

// the background reader's loop (started on the first async token; lives as long as the process, as the pool does)
void engram_async_loop(V41Engram* E) {
    for (;;) {
        std::unique_lock<std::mutex> l(E->am);
        E->acv.wait(l, [&] { return E->a_have; });
        E->a_have = false;
        uint64_t rows[2 * kEngCols];
        std::memcpy(rows, E->a_rows, sizeof rows);
        float* dst = E->a_dst;
        unsigned int* flag = E->a_flag;
        const unsigned int seq = E->a_seq;
        l.unlock();
        std::string e;
        bool ok = true;
        for (int k = 0; k < 2; ++k) {
            if (ok && !engram_rows_par(*E, kEngCols, rows + (size_t) k * kEngCols, k, nullptr,
                                       dst + (size_t) k * kEngCols * kEngDim, e))
                ok = false;
            std::atomic_thread_fence(std::memory_order_release);
            // even on a failure: the layer's kernel spins on this flag (engram_wait) and must not hang the GPU; the
            // failure reaches the host at the token's end (dsv41_engram_ok) and the token's answer is an error
            std::atomic_ref<unsigned int>(flag[k]).store(seq, std::memory_order_release);
        }
        l.lock();
        if (!ok && !E->a_bad) { E->a_bad = true; E->a_err = e; }
        E->a_busy = false;
        E->acv.notify_all();
    }
}

// a row's raw bytes into the row cache unless there (the prefetch: no decode, no error - a failed read stays a miss)
void engram_cache_row(V41Engram& E, int k, uint64_t row) {
    if (row == kEngRowZero) return;
    uint8_t raw[kEngRowMax];
    const int RB = E.row_bytes;
    const uint64_t key = (((uint64_t) k << 63) | row) + 1;
    const uint64_t slot = (row * 0x9E3779B97F4A7C15ull ^ (uint64_t) k) & E.cmask;
    {
        std::lock_guard<std::mutex> l(E.stripe[slot & 255]);
        if (E.tag[slot] == key) return;
    }
    if (!engram_read(E, k, E.off[k] + row * (uint64_t) RB, raw, (size_t) RB)) return;
    E.p_reads.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> l(E.stripe[slot & 255]);
    E.tag[slot] = key;
    std::memcpy(E.cache.data() + slot * (uint64_t) RB, raw, (size_t) RB);
}

void engram_pf_stop(V41Engram& E) {
    if (!E.p_th.joinable()) return;
    E.p_stop.store(true, std::memory_order_relaxed);
    E.p_th.join();
    E.p_stop.store(false, std::memory_order_relaxed);
}

// waits for the background reader to be idle; its failure, if any, is this call's - and this call's only: the flag
// is cleared here, so one transient read error (an EIO, a drive's hiccup) fails the token it belonged to and not
// every V4.1 request after it (before, a_bad stayed set for the life of the process)
bool engram_sync(V41Engram& E, std::string& err) {
    std::unique_lock<std::mutex> l(E.am);
    E.acv.wait(l, [&] { return !E.a_busy; });
    if (E.a_bad) {
        err = E.a_err.empty() ? "deepseek41: Engram row read failed" : E.a_err;
        E.a_bad = false;
        E.a_err.clear();
        return false;
    }
    return true;
}

// n rows (table of each: k[j] or kk for all) -> dst rows, across the Engram pool
bool engram_rows_par(V41Engram& E, int n, const uint64_t* rows, int kk, const int* ks, float* dst, std::string& err) {
    std::atomic<bool> bad{false};
    std::mutex mu;
    E.pool->run(n, [&](int j) {
        std::string e;
        if (!engram_row(E, ks != nullptr ? ks[j] : kk, rows[j], dst + (size_t) j * kEngDim, e)) {
            std::lock_guard<std::mutex> l(mu);
            err = e;
            bad = true;
        }
    });
    return !bad;
}
}  // namespace

bool Glm5Model::dsv41_engram_token(int32_t tok, std::string& err) {
    FastState* F = fast_;
    auto* E = (V41Engram*) v41_eng_;
    if (E == nullptr) { err = "deepseek41: Engram is not open"; return false; }
    static const bool async = [] {
        const char* v = getenv("STRATA_DS41_ENGRAM_ASYNC");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (!engram_sync(*E, err)) return false;
    engram_pf_stop(*E);
    uint64_t rows[2 * kEngCols];
    if (!engram_rows(*E, tok, pos_ - 1, rows, err, is_image_pos(pos_ - 1))) return false;   // (pos_ has moved past this token already)
    F->v41_eng_async = async && F->v41_eng_flag != nullptr;
    if (F->v41_eng_async) {
        if (++F->v41_eng_seq == 0u) F->v41_eng_seq = 1u;
        std::lock_guard<std::mutex> l(E->am);
        std::memcpy(E->a_rows, rows, sizeof rows);
        E->a_dst = F->v41_eng_h;
        E->a_flag = F->v41_eng_flag;
        E->a_seq = F->v41_eng_seq;
        E->a_have = true;
        E->a_busy = true;
        if (!E->a_started) {   // (a detached std::thread is never joinable: testing that started one a token, #20)
            E->a_started = true;
            std::thread(engram_async_loop, E).detach();
        }
        E->acv.notify_all();
        return true;
    }
    int ks[2 * kEngCols];
    for (int j = 0; j < 2 * kEngCols; ++j) ks[j] = j / kEngCols;
    if (!engram_rows_par(*E, 2 * kEngCols, rows, 0, ks, F->v41_eng_h, err)) return false;
    cudaMemcpyAsync(F->v41_eng, F->v41_eng_h, (size_t) 2 * kEngCols * kEngDim * 4, cudaMemcpyHostToDevice, F->cs);
    return true;
}

bool Glm5Model::dsv41_engram_hash(const int32_t* toks, int64_t p0, int T, std::string& err) {
    auto* E = (V41Engram*) v41_eng_;
    if (E == nullptr) { err = "deepseek41: Engram is not open"; return false; }
    if (!engram_sync(*E, err)) return false;   // (the pool is the background reader's too)
    engram_pf_stop(*E);
    E->chunk.resize((size_t) T * 2 * kEngCols);
    for (int t = 0; t < T; ++t)
        if (!engram_rows(*E, toks[t], p0 + t, E->chunk.data() + (size_t) t * 2 * kEngCols, err, is_image_pos(p0 + t)))
            return false;
    static const int pf_threads = [] {
        const char* v = getenv("STRATA_DS41_ENGRAM_PREFETCH");
        return v == nullptr ? 8 : std::max(0, std::min(16, std::atoi(v)));   // (1 = on: 8 threads; 0 = off)
    }();
    // table 0's rows (layer 1 reads them right after layer 0), then table 1's (layer 14's), as many tokens as half
    // of the direct-mapped cache holds without crowding it out
    const int tn = (int) std::min<uint64_t>((uint64_t) T, (E->cmask + 1) / 4 / kEngCols);
    if (pf_threads > 0 && E->cmask != 0 && tn > 0) {
        if (!E->ppool) E->ppool = std::make_unique<EngPool>(pf_threads == 1 ? 7 : pf_threads - 1);
        E->p_th = std::thread([E, tn] {
            E->ppool->run(2 * tn * kEngCols, [E, tn](int j) {
                if (E->p_stop.load(std::memory_order_relaxed)) return;
                const int k = j / (tn * kEngCols), t = (j / kEngCols) % tn, c = j % kEngCols;
                engram_cache_row(*E, k, E->chunk[(size_t) t * 2 * kEngCols + (size_t) k * kEngCols + c]);
            });
        });
    }
    return true;
}

bool Glm5Model::dsv41_engram_prompt(int il, int T, float* R, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    auto* E = (V41Engram*) v41_eng_;
    const int k = F->L[(size_t) il].eng;
    if (E == nullptr || k < 0 || E->chunk.size() < (size_t) T * 2 * kEngCols) { err = "deepseek41: no Engram rows"; return false; }
    // blocks of kV41Blk tokens: their rows read by the workers into the pinned block, up, then one add a token
    const size_t per = (size_t) kEngCols * kEngDim;
    for (int b0 = 0; b0 < T; b0 += kV41Blk) {
        const int bn = std::min(kV41Blk, T - b0);
        if (!glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "deepseek41 Engram block", err)) return false;
        std::vector<uint64_t> rr((size_t) bn * kEngCols);
        for (int t = 0; t < bn; ++t)
            for (int c = 0; c < kEngCols; ++c)
                rr[(size_t) t * kEngCols + c] = E->chunk[(size_t) (b0 + t) * 2 * kEngCols + (size_t) k * kEngCols + c];
        if (!engram_rows_par(*E, bn * kEngCols, rr.data(), k, nullptr, F->v41_pblk_h, err)) return false;
        cudaMemcpyAsync(F->v41_pblk, F->v41_pblk_h, (size_t) bn * per * 4, cudaMemcpyHostToDevice, F->cs);
        if (prefill_v41_engram_gemm()) {
            if (!prefill_v41_engram_block(il, bn, (int) per, F->v41_pblk, R + (size_t) b0 * 4 * g.n_embd, err)) return false;
            continue;
        }
        for (int t = 0; t < bn; ++t)
            if (!dsv41_engram_add(il, R + (size_t) (b0 + t) * 4 * g.n_embd, err, F->v41_pblk + (size_t) t * per))
                return false;
    }
    return true;
}

bool Glm5Model::dsv41_engram_window(const int32_t* toks, int64_t p0, int T, std::string& err) {
    FastState* F = fast_;
    auto* E = (V41Engram*) v41_eng_;
    if (E == nullptr) { err = "deepseek41: Engram is not open"; return false; }
    if (T < 1 || T * 2 > kV41Blk) { err = "deepseek41: a verify window of " + std::to_string(T) + " columns"; return false; }
    if (!engram_sync(*E, err)) return false;   // (the pool is the background reader's too)
    engram_pf_stop(*E);
    // column t's table k: rows (t * 2 + k) * 24 .. of the prompt path's block buffers (pinned, then the device's)
    std::vector<uint64_t> rows((size_t) T * 2 * kEngCols);
    std::vector<int> ks(rows.size());
    for (int t = 0; t < T; ++t) {
        if (!engram_rows(*E, toks[t], p0 + t, rows.data() + (size_t) t * 2 * kEngCols, err, is_image_pos(p0 + t)))
            return false;
        for (int j = 0; j < 2 * kEngCols; ++j) ks[(size_t) t * 2 * kEngCols + j] = j / kEngCols;
    }
    if (!engram_rows_par(*E, (int) rows.size(), rows.data(), 0, ks.data(), F->v41_pblk_h, err)) return false;
    cudaMemcpyAsync(F->v41_pblk, F->v41_pblk_h, rows.size() * kEngDim * sizeof(float), cudaMemcpyHostToDevice, F->cs);
    return true;
}

bool Glm5Model::dsv41_engram_window_add(int il, int t, float* R, std::string& err) {
    FastState* F = fast_;
    const int k = F->L[(size_t) il].eng;
    if (k < 0) { err = "deepseek41: layer " + std::to_string(il) + " has no Engram table"; return false; }
    return dsv41_engram_add(il, R, err, F->v41_pblk + (size_t) (t * 2 + k) * kEngCols * kEngDim);
}

// after a token's layers: the background reader's failure for this token, if any (its rows were never written, the
// layer computed on stale ones) - the token's step fails with it, the reader's flag is clear again for the next one
bool Glm5Model::dsv41_engram_ok(std::string& err) {
    auto* E = (V41Engram*) v41_eng_;
    if (E == nullptr || fast_ == nullptr || !fast_->v41_eng_async) return true;
    return engram_sync(*E, err);
}

bool Glm5Model::dsv41_engram_add(int il, float* R, std::string& err, const float* rows) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    if (rows == nullptr) {   // a decode token's rows
        rows = F->v41_eng + (size_t) Ly.eng * kEngCols * kEngDim;
        if (F->v41_eng_async)
            d41::engram_wait(F->v41_eng_flag + Ly.eng, F->v41_eng_seq, F->v41_eng_h + (size_t) Ly.eng * kEngCols * kEngDim,
                             (float*) rows, kEngCols * kEngDim, F->cs);
    }
    const void* rq = nullptr;   // (a quantized engram_kv - mrfakename's Q2_K - takes the rows as Q8_1)
    if (Ly.eng_kv.type != gf::kTypeBF16 && Ly.eng_kv.type != gf::kTypeF32) {
        rq = (const uint8_t*) F->v41_eng_q + (size_t) Ly.eng * kEngCols * kEngDim / 32 * 36;
        gf::quantize_q8_1(rows, (void*) rq, kEngCols * kEngDim, F->cs);
    }
    gf::MvJob j = {Ly.eng_kv.q, rq, rows, F->v41_eng_kv, nullptr, 1.0f, Ly.eng_kv.type, kEngCols * kEngDim, 5 * g.n_embd};
    if (!gf::mv(&j, 1, F->cs)) { err = "deepseek41: engram_kv"; return false; }
    d41::engram_add(R, F->v41_eng_kv, Ly.eng_qn, Ly.eng_kn, g.n_embd, g.norm_eps, F->cs);
    if (F->prof_on) F->mark("v41_engram");
    return true;
}

bool Glm5Model::fast_dsv41_attn(int il, int64_t p64, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int p = (int) p64;
    const int D = g.head_dim, r = Ly.ratio;
    const ds::Rope rope = layer_rope(g, r != 0), crope = layer_rope(g, true);

    // 1. the projections of x: q_a, kv, the compressor (KV sources), the indexer's weights (index sources)
    gf::MvJob j[gf::kMaxMvJobs];
    int nj = 0;
    j[nj++] = {Ly.q_a.q, F->xq, F->x, F->qr_raw, nullptr, 1.0f, Ly.q_a.type, g.n_embd, g.q_lora};
    j[nj++] = {Ly.kv.q, F->xq, F->x, F->kv_raw, nullptr, 1.0f, Ly.kv.type, g.n_embd, D};
    if (Ly.kv_src) {
        j[nj++] = {Ly.comp_kv.q, F->xq, F->x, F->ds_ckv[0], nullptr, 1.0f, Ly.comp_kv.type, g.n_embd, D};
        if (r == 2) j[nj++] = {Ly.comp_gate.q, F->xq, F->x, F->ds_csc[0], nullptr, 1.0f, Ly.comp_gate.type, g.n_embd, D};
    }
    if (Ly.idx_src)
        j[nj++] = {Ly.idx_proj_f, nullptr, F->x, F->iw, nullptr, 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads)),
                   gf::kTypeF32, g.n_embd, g.idx_heads};
    if (!gf::mv(j, nj, s)) { err = "deepseek41: attention projections"; return false; }
    if (F->prof_on) F->mark("ds_proj_mv");

    // 2. qr's norm, kv's norm + RoPE into the window ring (deepseek4's prep, no compressor), then the KV source's row
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
    ds::prep(a, s);
    if (Ly.kv_src) {
        d41::PublishArgs pa;
        pa.v = F->ds_ckv[0];
        pa.g = r == 2 ? F->ds_csc[0] : nullptr;
        pa.pend = F->v41_pend[Ly.slot] + (size_t) ((p / 2) % FastState::kV41PendRing) * 2 * D;   // (the pair's entry)
        pa.norm = Ly.comp_norm;
        pa.idx_k = (const uint16_t*) Ly.idx_k_w.q;
        pa.idx_k_norm = Ly.idx_k_norm;
        pa.comp_cache = Ly.comp_cache;
        pa.idx_cache = Ly.idx_cache;
        pa.ratio = r;
        pa.p = p;
        pa.D = D;
        pa.idx_dim = g.idx_key;
        pa.n_rot = g.n_rot;
        pa.rope = crope;
        pa.eps = g.norm_eps;
        d41::publish(pa, s);
    }
    if (F->prof_on) F->mark("ds_prep");

    // 3. q_b (and the indexer's q) from the normed q_a, RoPE (no per-head norm)
    gf::MvJob j2[2];
    int nj2 = 0;
    j2[nj2++] = {Ly.q_b.q, F->qr_q, F->qr, F->q, nullptr, 1.0f, Ly.q_b.type, g.q_lora, g.n_head * D};
    if (Ly.idx_src)
        j2[nj2++] = {Ly.idx_q_b_q.q, F->qr_q, F->qr, F->iq, nullptr, 1.0f, Ly.idx_q_b_q.type, g.q_lora,
                     g.idx_heads * g.idx_key};
    if (!gf::mv(j2, nj2, s)) { err = "deepseek41: q_b"; return false; }
    d41::q_rope(F->q, g.n_head, D, g.n_rot, rope, Ly.idx_src ? F->iq : nullptr, g.idx_heads, g.idx_key, crope, p, s);
    if (F->prof_on) F->mark("ds_q");

    // 4. the keys: the window, and the slot's visible rows (all of them up to top_k; beyond, the top-k - this layer's
    // own when it is an index source, the last index source's otherwise: ds_sel is only written by index sources)
    const int n_raw = std::min(p + 1, g.n_swa);
    int* const sel_buf = pf_sel_ != nullptr ? pf_sel_ : F->ds_sel;   // (the prompt path: the row's own)
    unsigned char* const keep_buf = pf_keep_ != nullptr ? pf_keep_ : F->v41_keep;
    int n_comp = 0;
    const int* sel = nullptr;
    if (r != 0) {
        const int n_vis = (p + 1) / r;
        if (n_vis > g.idx_top_k) {
            if (Ly.idx_src) {
                ds::indexer_scores(F->iq, Ly.idx_cache, F->iw, g.idx_heads, n_vis, F->v41_score, s);
                if (il == g.cand_layer && n_vis > g.cand_blocks * g.cand_bs)
                    d41::candidates(F->v41_score, n_vis, g.cand_bs, g.cand_blocks, F->v41_bscore, F->v41_bsel,
                                    keep_buf, s);
                if (il > g.cand_layer && n_vis > g.cand_blocks * g.cand_bs)
                    d41::candidate_mask(F->v41_score, n_vis, g.cand_bs, keep_buf, s);
                ds::indexer_select(F->v41_score, n_vis, g.idx_top_k, sel_buf, s);
                if (F->prof_on) F->mark("ds_index");
            }
            n_comp = g.idx_top_k;
            sel = sel_buf;
        } else {
            n_comp = n_vis;
        }
    }
    ds::attend(F->q, Ly.raw, ds_ring_, n_raw, Ly.comp_cache, sel, n_comp, Ly.sinks, g.n_head, D, g.n_rot, rope, p,
               F->ds_o, s, 0, F->ds_oq);
    if (F->prof_on) F->mark("ds_attend");

    // 5. the grouped output LoRA (deepseek4's)
    const int gin = g.n_head / g.o_groups * D;
    const size_t wrow = gf::row_bytes(Ly.out_a.type, gin);
    if (wrow == 0 || g.o_groups > gf::kMaxMvJobs) { err = "deepseek41: wo_a type"; return false; }
    gf::MvJob jo[gf::kMaxMvJobs];
    for (int gi = 0; gi < g.o_groups; ++gi)
        jo[gi] = {(const uint8_t*) Ly.out_a.q + (size_t) gi * g.o_lora * wrow,
                  (const uint8_t*) F->ds_oq + (size_t) gi * (gin / 32) * 36, F->ds_o + (size_t) gi * gin,
                  F->ds_oa + (size_t) gi * g.o_lora, nullptr, 1.0f, Ly.out_a.type, gin, g.o_lora};
    if (!gf::mv(jo, g.o_groups, s)) { err = "deepseek41: wo_a"; return false; }
    gf::quantize_q8_1(F->ds_oa, F->ds_oaq, g.o_groups * g.o_lora, s);
    gf::MvJob ob = {Ly.out_b.q, F->ds_oaq, F->ds_oa, F->mixer, nullptr, 1.0f, Ly.out_b.type, g.o_groups * g.o_lora,
                    g.n_embd};
    if (!gf::mv(&ob, 1, s)) { err = "deepseek41: wo_b"; return false; }
    abl_apply(il, 0, F->mixer, 1, g.n_embd, s);
    if (F->prof_on) F->mark("ds_out_mv");
    return true;
}

bool Glm5Model::dsv41_head(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const auto nw = w_.find("output_norm.weight");
    if (nw == w_.end()) { err = "deepseek41: output_norm.weight"; return false; }
    // the last sublayer (layer n-1's FFN, sublayer 2n - 1) wrote its pre into v41_pre[1]
    d41::head_prep(state_, F->v41_pre[1], nw->second, g.norm_eps, g.n_embd, F->head_x, F->head_xq, F->cs);
    return true;
}

}  // namespace strata::core
