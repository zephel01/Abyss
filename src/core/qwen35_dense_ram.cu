// src/core/qwen35_dense_ram.cu - the dense qwen35 (Qwen3.8-27B) on a GPU smaller than its file.
//
// A layer of Qwen3.8-27B UD-Q4_K_XL is ~270 MB, ~160 MB of it the FFN (gate / up 17408 x 5120, down 5120 x 17408).
// Everything but the trunk's FFN stays in VRAM - the attention and delta-net weights and their state, the NextN block,
// the head (the embedding is read on the host already) - and as many layers' FFN as the budget leaves; the other
// layers' FFN lives in pinned RAM:
//   - decode and verify windows: the CPU computes those layers' FFN (lane_q81 rows: the device's q8_1 arithmetic) on
//     a pool of threads splitting the rows; only the columns' q8_1 x (5.8 KB a column) goes up and the FFN output
//     (20 KB) comes back, through host-mapped memory - a kernel posts the request, the service thread answers, a kernel
//     waits for the answer in stream order.  Streaming a layer's FFN over PCIe instead would take ~25 ms a layer on
//     Gen4 x4 against ~1.7 ms on the CPU (94 GB/s on a Ryzen AI Max+ 395, 16 threads).
//   - the prompt path: the RAM layers' FFN is copied into one of two VRAM staging buffers on a copy stream while the
//     layers before it compute, and multiplied there with MMQ as a VRAM layer's - the RAM layers are spread evenly over
//     the depth so each copy has a few layers' compute to hide under.
// The budget: STRATA_GLM_VRAM_GB (a card of that size, what this process already holds counted) or the free VRAM,
// less STRATA_Q35D_RESERVE_MB (512).  STRATA_Q35D_GPU_LAYERS=<n> caps the VRAM layers (tests); STRATA_Q35D_RAM=0 keeps
// every FFN in VRAM (v1.3.138's load: a card too small stops at the load).  STRATA_Q35D_THREADS: the CPU threads.
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/cpu/lane_q81.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/qwen35_fast.hpp"
#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

namespace kc = strata::kernels::cpu;
namespace q35 = strata::kernels::q35;

namespace {
size_t al256(size_t b) { return (b + 255u) & ~(size_t) 255u; }

// MemAvailable, bytes (0: unknown)
int64_t ram_available() {
#ifdef __linux__
    std::ifstream f("/proc/meminfo");
    std::string k;
    int64_t v = 0;
    std::string unit;
    while (f >> k >> v >> unit)
        if (k == "MemAvailable:") return v * 1024;
    return 0;
#elif defined(_WIN32)
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    return GlobalMemoryStatusEx(&m) ? (int64_t) m.ullAvailPhys : 0;
#else
    return 0;
#endif
}

// one thread a physical core (Linux: distinct package/core pairs this process may run on), else half the logical CPUs
int physical_core_count() {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        std::vector<std::pair<int, int>> seen;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (!CPU_ISSET(c, &set)) continue;
            const std::string b = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
            int pk = -1, co = -1;
            std::ifstream(b + "physical_package_id") >> pk;
            std::ifstream(b + "core_id") >> co;
            if (co < 0) { seen.clear(); break; }
            if (std::find(seen.begin(), seen.end(), std::make_pair(pk, co)) == seen.end()) seen.push_back({pk, co});
        }
        if (!seen.empty()) return (int) seen.size();
    }
#endif
    const int n = (int) std::thread::hardware_concurrency();
    return n > 1 ? n / 2 : 1;
}
}  // namespace

struct Glm5Model::Q35dRam {
    // load_pack's record: per trunk layer, the three FFN tensors in the mapping
    struct Ffn {
        const uint8_t* src[3] = {nullptr, nullptr, nullptr};
        size_t bytes[3] = {0, 0, 0};
        int type[3] = {0, 0, 0};
        size_t row[3] = {0, 0, 0};   // bytes a row (gate / up: n_embd values, down: n_ff)
        // the split (q35d_place): FF rows [0, kg) on the GPU - gate / up rows and down's first kg columns, repacked as
        // [n_embd][kg] - and [kg, n_ff) on the CPU (down's other columns as [n_embd][n_ff - kg]); kg = n_ff: all in
        // VRAM, 0: all in RAM
        int kg = -1;
        uint8_t* dev[3] = {nullptr, nullptr, nullptr};
        uint8_t* host[3] = {nullptr, nullptr, nullptr};
        size_t drow_g = 0, drow_c = 0;   // down's row bytes in the two parts
        bool ram = false;                // a CPU part (kg < n_ff)
        bool lent = false;               // all in VRAM in the prompt path's scratch between prompts (the loan)
        size_t total() const { return al256(bytes[0]) + al256(bytes[1]) + al256(bytes[2]); }
        // the bytes of a part of k FF rows (gate / up rows, down columns), as laid out
        size_t part(int k, int n_embd) const {
            return k <= 0 ? 0
                          : al256((size_t) k * row[0]) + al256((size_t) k * row[1]) +
                                al256((size_t) n_embd * ggml_row_size((ggml_type) type[2], k));
        }
    };
    std::vector<Ffn> L;
    bool placed = false;
    void* dev = nullptr;          // the VRAM layers' FFN
    void* pinned = nullptr;       // the RAM layers' FFN
    size_t dev_bytes = 0, pinned_bytes = 0;
    std::vector<int> ram_layers;  // ascending
    std::vector<int> prompt_layers;   // the RAM and the loaned layers, ascending (the prompt path stages them)
    bool evicted = false;         // the loaned layers out of VRAM (a prompt runs)
    void* stage_block = nullptr;  // the two staging buffers in one allocation
    size_t loan_bytes = 0;
    // ---- the CPU service: the request / answer host-mapped, a thread answering on a pool
    void* req_h = nullptr;        // [16 uint32 header | the columns' q8_1 x]
    void* ans_h = nullptr;        // [16 uint32 header | the columns' FFN output]
    void* req_d = nullptr;
    void* ans_d = nullptr;
    unsigned int seq = 0;         // the last request enqueued (the device's order)
    std::unique_ptr<glmfast::Workers> pool;
    std::thread svc;
    std::atomic<bool> quit{false};
    int threads = 0, n_embd = 0, n_ff = 0;
    std::vector<uint8_t> act, hq;   // kVerifyCols columns of the lane's activations
    std::vector<float> ff;
    std::atomic<uint64_t> calls{0}, cols{0}, us{0};
    double ms_gu = 0, ms_q = 0, ms_dn = 0;   // (the service thread's phases)
    // ---- the prompt path's staging (two buffers, a copy stream)
    uint8_t* stage[2] = {nullptr, nullptr};
    size_t stage_bytes = 0;
    int holds[2] = {-1, -1};      // the layer a buffer holds (or is being filled with)
    int cur = 0;                  // the buffer the last staged layer used
    cudaEvent_t ready[2] = {nullptr, nullptr}, freed[2] = {nullptr, nullptr};
    cudaStream_t cs = nullptr;
    uint64_t staged = 0, stage_hits = 0, restores = 0;
    static constexpr int kCols = 8;   // the widest window the service takes (kVerifyMax)
    void serve();
    void compute(int il, int nt);
};

bool Glm5Model::q35d_defer_on() const {
    if (!g_.q35_dense() || !fast_mode_) return false;
    const char* v = getenv("STRATA_Q35D_RAM");
    return v == nullptr || std::atoi(v) != 0;
}

void Glm5Model::q35d_defer(int il, int role, const uint8_t* src, size_t bytes, int type) {
    if (q35d_ == nullptr) {
        q35d_ = new Q35dRam();
        q35d_->L.resize((size_t) g_.n_layers);
        q35d_->n_embd = g_.n_embd;
        q35d_->n_ff = g_.n_ff_dense;
    }
    if (il < 0 || il >= (int) q35d_->L.size() || role < 0 || role > 2) return;
    auto& f = q35d_->L[(size_t) il];
    f.src[role] = src;
    f.bytes[role] = bytes;
    f.type[role] = type;
    f.row[role] = role == 2 ? bytes / (size_t) g_.n_embd : bytes / (size_t) g_.n_ff_dense;
}

int Glm5Model::q35d_gpu_rows(int il) const {
    return q35d_ == nullptr || il < 0 || il >= (int) q35d_->L.size() || q35d_->L[(size_t) il].kg < 0
               ? g_.n_ff_dense
               : q35d_->L[(size_t) il].kg;
}

bool Glm5Model::q35d_ram(int il) const {
    return q35d_ != nullptr && il >= 0 && il < (int) q35d_->L.size() && q35d_->L[(size_t) il].ram;
}

bool Glm5Model::q35d_prompt_ram(int il) const {
    return q35d_ != nullptr && il >= 0 && il < (int) q35d_->L.size() &&
           (q35d_->L[(size_t) il].ram || q35d_->L[(size_t) il].lent);
}

void Glm5Model::q35d_evict() {
    Q35dRam* Q = q35d_;
    if (Q == nullptr || Q->loan_bytes == 0 || Q->evicted) return;
    cudaStreamSynchronize(fast_->cs);   // (the last token's reads of the loaned layers)
    Q->evicted = true;
    Q->holds[0] = Q->holds[1] = -1;
}

bool Glm5Model::q35d_restore(std::string& err) {
    Q35dRam* Q = q35d_;
    if (Q == nullptr || !Q->evicted) return true;
    // the staging copies first (the last layer's prefetch of the next chunk's first may still run), then the loaned
    // layers whole from their pinned copies - on the token stream, so the next token reads them after
    if (Q->cs != nullptr && !glmfast::cuda_ok(cudaStreamSynchronize(Q->cs), "qwen35 dense: staging copies", err)) return false;
    for (auto& f : Q->L) {
        if (!f.lent) continue;
        for (int r = 0; r < 3; ++r)
            if (!glmfast::cuda_ok(cudaMemcpyAsync(f.dev[r], f.host[r], f.bytes[r], cudaMemcpyHostToDevice, fast_->cs),
                                  "qwen35 dense: the loaned layers back", err))
                return false;
    }
    Q->evicted = false;
    Q->holds[0] = Q->holds[1] = -1;
    ++Q->restores;
    return true;
}

size_t Glm5Model::q35d_pending() const {
    if (q35d_ == nullptr || q35d_->placed) return 0;
    size_t b = 0;
    for (const auto& f : q35d_->L) b += f.total();
    return b;
}

// the CPU side of a request: the columns' x (the device's q8_1 blocks) -> gate / up rows and the SwiGLU (the pool,
// rows split) -> h as q8_1 a column -> down rows (the pool) into the answer
void Glm5Model::Q35dRam::compute(int il, int nt) {
    const Ffn& f = L[(size_t) il];
    const int E = n_embd, FC = n_ff - std::max(0, f.kg);   // (the CPU part's FF rows)
    const size_t q8e = (size_t) E / 32 * 36, ab = kc::lane_q81_act_bytes(E), hb = kc::lane_q81_act_bytes(FC);
    const uint8_t* xq = (const uint8_t*) req_h + 64;
    for (int t = 0; t < nt; ++t) kc::lane_q81_from_blocks(xq + (size_t) t * q8e, E, act.data() + (size_t) t * ab);
    const void* a[kCols];
    float* o[kCols];
    for (int t = 0; t < nt; ++t) {
        a[t] = act.data() + (size_t) t * ab;
        o[t] = ff.data() + (size_t) t * FC;
    }
    const int jobs = std::max(1, pool->active() * 4);
    const auto t0 = std::chrono::steady_clock::now();
    pool->run(jobs, [&](int j) {
        const int r0 = (int) ((int64_t) FC * j / jobs), r1 = (int) ((int64_t) FC * (j + 1) / jobs);
        kc::lane_q81_gu_rows(f.type[0], f.type[1], f.host[0], f.row[0], f.host[1], f.row[1], E, a, nt, o, r0, r1,
                             std::numeric_limits<float>::infinity(), 1.0f);
    });
    const auto t1 = std::chrono::steady_clock::now();
    if (nt == 1) {
        kc::lane_q81_quant(ff.data(), FC, hq.data());
    } else {
        pool->run(nt, [&](int t) { kc::lane_q81_quant(ff.data() + (size_t) t * FC, FC, hq.data() + (size_t) t * hb); });
    }
    const void* h[kCols];
    float* out[kCols];
    const auto t2 = std::chrono::steady_clock::now();
    float* ans = (float*) ((uint8_t*) ans_h + 64);
    for (int t = 0; t < nt; ++t) {
        h[t] = hq.data() + (size_t) t * hb;
        out[t] = ans + (size_t) t * E;
    }
    pool->run(jobs, [&](int j) {
        const int r0 = (int) ((int64_t) E * j / jobs), r1 = (int) ((int64_t) E * (j + 1) / jobs);
        kc::lane_q81_rows(f.type[2], f.host[2], f.drow_c, FC, h, nt, out, r0, r1);
    });
    const auto t3 = std::chrono::steady_clock::now();
    ms_gu += std::chrono::duration<double, std::milli>(t1 - t0).count();
    ms_q += std::chrono::duration<double, std::milli>(t2 - t1).count();
    ms_dn += std::chrono::duration<double, std::milli>(t3 - t2).count();
}

void Glm5Model::Q35dRam::serve() {
    volatile unsigned int* rq = (volatile unsigned int*) req_h;
    std::atomic<unsigned int>* aseq = (std::atomic<unsigned int>*) ans_h;
    unsigned int last = 0;
    auto idle_since = std::chrono::steady_clock::now();
    int spins = 0;
    while (!quit.load(std::memory_order_relaxed)) {
        const unsigned int s = rq[0];
        if (s == last) {
            glmfast::cpu_relax();
            // spin ~20 ms after the last request (a token's RAM layers come ~2 ms apart), then poll every 100 us
            if ((++spins & 1023) == 0 && std::chrono::steady_clock::now() - idle_since > std::chrono::milliseconds(20))
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        const int il = (int) rq[1], nt = (int) rq[2];
        const auto t0 = std::chrono::steady_clock::now();
        if (il >= 0 && il < (int) L.size() && L[(size_t) il].ram && nt >= 1 && nt <= kCols) compute(il, nt);
        std::atomic_thread_fence(std::memory_order_release);
        aseq->store(s, std::memory_order_release);
        last = s;
        idle_since = std::chrono::steady_clock::now();
        spins = 0;
        calls.fetch_add(1, std::memory_order_relaxed);
        cols.fetch_add((uint64_t) nt, std::memory_order_relaxed);
        us.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(idle_since - t0).count(),
                     std::memory_order_relaxed);
    }
}

// fast_setup (the dense branch, after the prompt path's buffers): the verify window's buffers, then the budget - the
// FFN of every trunk layer that fits in VRAM there, the rest (spread evenly over the depth) in pinned RAM with the
// CPU service and, with a prompt path, its two staging buffers
bool Glm5Model::q35d_place(std::string& err) {
    Q35dRam* Q = q35d_;
    if (Q == nullptr || Q->placed) return true;
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const int NL = g.n_layers;
    const double G = 1073741824.0;
    for (int il = 0; il < NL; ++il)
        for (int r = 0; r < 3; ++r)
            if (Q->L[(size_t) il].src[r] == nullptr) {
                err = "qwen35 dense: blk." + std::to_string(il) + "'s FFN is not in the pack";
                return false;
            }
    // the verify window's buffers first (they were allocated at the first window: a card the FFN filled had none left
    // - rel138's 20 GB check)
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const size_t free_pre_verify = free_b;
    if (mtp_il_ >= 0 && !verify_alloc(err)) return false;
    cudaMemGetInfo(&free_b, &total_b);
    const size_t verify_b = free_pre_verify > free_b ? free_pre_verify - free_b : 0, in_use0 = total_b - free_b;
    const size_t cap = vram_cap();
    const size_t used = total_b - free_b - std::min(vram_base_, total_b - free_b);
    size_t room = cap > 0 ? (cap > used ? std::min(cap - used, free_b) : 0) : free_b;
    const size_t reserve = (size_t) std::max(0.0, std::atof(getenv("STRATA_Q35D_RESERVE_MB") ? getenv("STRATA_Q35D_RESERVE_MB") : "128")) << 20;
    room = room > reserve ? room - reserve : 0;
    size_t all = 0, max_layer = 0;
    for (const auto& f : Q->L) {
        all += f.total();
        max_layer = std::max(max_layer, f.total());
    }
    const int FF = g.n_ff_dense, E = g.n_embd;
    int gpu_cap = NL;
    if (const char* v = getenv("STRATA_Q35D_GPU_LAYERS")) gpu_cap = std::max(0, std::min(NL, std::atoi(v)));
    const char* rv = getenv("STRATA_Q35D_RAM");
    const bool ram_ok = rv == nullptr || std::atoi(rv) != 0;
    // STRATA_Q35D_SPLIT=0: whole layers' FFN on the CPU (the A/B); else the CPU takes the last rows of n layers' FFN
    const char* sv = getenv("STRATA_Q35D_SPLIT");
    const bool split = sv == nullptr || std::atoi(sv) != 0;
    // the candidates for CPU rows (all but the loaned layers), n of them spread evenly (floor((i + 0.5) * nc / n))
    std::vector<int> cand;
    const auto spread = [&](int n) {
        std::vector<char> r((size_t) NL, 0);
        const int nc = (int) cand.size();
        n = std::min(n, nc);
        for (int i = 0; i < n; ++i) r[(size_t) cand[(size_t) std::min(nc - 1, (int) (((double) i + 0.5) * nc / n))]] = 1;
        return r;
    };
    // the VRAM n spread layers leave with c of their FF rows on the CPU (the loaned layers' VRAM is the prompt's)
    const auto gpu_need = [&](const std::vector<char>& set, int c) {
        size_t b = 0;
        for (int il = 0; il < NL; ++il)
            if (!Q->L[(size_t) il].lent) b += set[(size_t) il] ? Q->L[(size_t) il].part(FF - c, E) : Q->L[(size_t) il].total();
        return b;
    };
    std::vector<int> kg((size_t) NL, FF);
    int n_cpu = 0, c_rows = 0;
    if (all > room || gpu_cap < NL) {
        if (!ram_ok) {
            char b[160];
            std::snprintf(b, sizeof b, "qwen35 dense: the FFN weights (%.1f GB) do not fit in VRAM and STRATA_Q35D_RAM=0 "
                                       "keeps them there", (double) all / G);
            err = b;
            return false;
        }
        const size_t stage = pf_ != nullptr ? 2 * al256(max_layer) : 0;
        const size_t room2 = room > stage ? room - stage : 0;
        // the staging buffers now (one allocation), then the loan: whole layers' FFN in the prompt path's scratch and
        // the staging buffers between prompts (STRATA_Q35D_LOAN=0: none) - first fit, the layers spread over the depth
        if (pf_ != nullptr) {
            if (cudaMalloc(&Q->stage_block, stage) != cudaSuccess) {
                cudaGetLastError();
                err = "qwen35 dense: the prompt path's FFN staging buffers (" + std::to_string(stage >> 20) + " MB) did not allocate";
                return false;
            }
            Q->stage_bytes = al256(max_layer);
            const char* lv = getenv("STRATA_Q35D_LOAN");
            if (lv == nullptr || std::atoi(lv) != 0) {
                auto blocks = prefill_loan_blocks();
                blocks.push_back({(uint8_t*) Q->stage_block, stage});
                std::vector<size_t> used(blocks.size(), 0);
                const int step = 7;   // (a stride coprime to 64: spread over the depth)
                for (int i = 0; i < NL; ++i) {
                    const int il = (int) (((int64_t) i * step + 3) % NL);
                    auto& f = Q->L[(size_t) il];
                    for (size_t b = 0; b < blocks.size(); ++b) {
                        if (used[b] + f.total() > blocks[b].second) continue;
                        uint8_t* d = blocks[b].first + used[b];
                        for (int r = 0; r < 3; ++r) {
                            f.dev[r] = d;
                            d += al256(f.bytes[r]);
                        }
                        used[b] += f.total();
                        f.lent = true;
                        Q->loan_bytes += f.total();
                        break;
                    }
                }
            }
        }
        for (int il = 0; il < NL; ++il)
            if (!Q->L[(size_t) il].lent) cand.push_back(il);
        // the fewest CPU rows (a multiple of 256) on n layers that leave the rest within room2; FF + 256 when even all of
        // them on the CPU does not
        const auto rows_for = [&](int n) {
            const auto set = spread(n);
            for (int c = 256; c <= FF; c += 256)
                if (gpu_need(set, c) <= room2) return c;
            return FF + 256;
        };
        // the split: the most layers whose CPU part keeps at least STRATA_Q35D_SPLIT_MIN rows (2048: the GPU's rows of a
        // layer then take less time than the CPU's, so they are hidden under them) - each such layer's GPU part runs
        // while the CPU computes its part; whole layers (STRATA_Q35D_SPLIT=0 or STRATA_Q35D_GPU_LAYERS): the fewest
        int c_min = 2048;
        if (const char* v = getenv("STRATA_Q35D_SPLIT_MIN")) c_min = std::max(256, std::min(FF, std::atoi(v) / 256 * 256));
        if (split && gpu_cap == NL) {
            int n = std::max(1, (int) cand.size()), c = rows_for(n);
            if (const char* v = getenv("STRATA_Q35D_SPLIT_N")) {   // (tests: n layers, their fewest rows)
                n = std::max(1, std::min(NL, std::atoi(v)));
                c = rows_for(n);
            } else {
                while (c < c_min && n > 1) {
                    const int c2 = rows_for(n - 1);
                    if (c2 > FF) break;
                    --n;
                    c = c2;
                }
            }
            if (c > FF) c = FF;   // (all of every layer: the card is too small for more; the check below)
            n_cpu = n;
            c_rows = c;
        } else {
            int n = std::max(1, NL - gpu_cap);
            for (; n < NL && rows_for(n) > FF; ++n) {
            }
            n_cpu = n;
            c_rows = FF;
        }
        const auto set = spread(n_cpu);
        for (int il = 0; il < NL; ++il)
            if (set[(size_t) il]) kg[(size_t) il] = FF - c_rows;
    }
    size_t dev_b = 0, ram_b = 0;
    for (int il = 0; il < NL; ++il) {
        auto& f = Q->L[(size_t) il];
        f.kg = f.lent ? FF : kg[(size_t) il];
        f.ram = f.kg < FF;
        f.drow_g = ggml_row_size((ggml_type) f.type[2], f.kg);
        f.drow_c = f.row[2] - f.drow_g;
        if (f.lent) {
            ram_b += f.total();   // (its whole pinned copy: the prompt stages it, a restore uploads it)
        } else {
            dev_b += f.part(f.kg, E);
            ram_b += f.part(FF - f.kg, E);
        }
        if (f.ram) Q->ram_layers.push_back(il);
        if (f.ram || f.lent) Q->prompt_layers.push_back(il);
    }
    const int m = (int) Q->ram_layers.size();
    // RAM: what the pinned copy takes must leave the host its headroom (STRATA_GLM_RAM_HEADROOM_GB, 6)
    if (ram_b > 0) {
        const double head = std::atof(getenv("STRATA_GLM_RAM_HEADROOM_GB") ? getenv("STRATA_GLM_RAM_HEADROOM_GB") : "6");
        const int64_t av = ram_available();
        if (av > 0 && (double) av < (double) ram_b + head * G) {
            char b[256];
            std::snprintf(b, sizeof b, "qwen35 dense: %.1f GB of the FFN must stay in RAM on this GPU, but only %.1f GB is "
                                       "available (less a %.0f GB headroom, STRATA_GLM_RAM_HEADROOM_GB)",
                          (double) ram_b / G, (double) av / G, head);
            err = b;
            return false;
        }
    }
    const auto set_slot = [&](int il, int r, const void* p) {
        static const char* nm[3] = {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"};
        const std::string n = "blk." + std::to_string(il) + "." + nm[r];
        auto it = ws_map_.find(n);
        if (it != ws_map_.end()) it->second.q = p;
        auto& Ly = F->L[(size_t) il];
        WSlot& w = r == 0 ? Ly.ffn_gate : r == 1 ? Ly.ffn_up : Ly.ffn_down;
        w.q = p;
    };
    for (int il : Q->ram_layers)
        for (int r = 0; r < 3; ++r)
            if (!kc::lane_q81_type(Q->L[(size_t) il].type[r])) {
                err = "qwen35 dense: blk." + std::to_string(il) + "'s FFN type " + std::to_string(Q->L[(size_t) il].type[r]) +
                      " has no CPU rows";
                return false;
            }
    // VRAM: one region for the GPU parts, from the mapping (down's first kg columns of each row: a 2D copy)
    if (dev_b > 0) {
        if (cudaMalloc(&Q->dev, dev_b) != cudaSuccess) {
            cudaGetLastError();
            err = "qwen35 dense: the FFN's VRAM part (" + std::to_string(dev_b >> 20) + " MB) did not allocate";
            return false;
        }
        Q->dev_bytes = dev_b;
        size_t at = 0;
        for (int il = 0; il < NL; ++il) {
            auto& f = Q->L[(size_t) il];
            if (f.kg <= 0 || f.lent) continue;
            for (int r = 0; r < 3; ++r) {
                uint8_t* d = (uint8_t*) Q->dev + at;
                cudaError_t e;
                if (r < 2) {
                    e = cudaMemcpy(d, f.src[r], (size_t) f.kg * f.row[r], cudaMemcpyHostToDevice);
                    at += al256((size_t) f.kg * f.row[r]);
                } else {
                    e = f.kg == FF ? cudaMemcpy(d, f.src[2], f.bytes[2], cudaMemcpyHostToDevice)
                                   : cudaMemcpy2D(d, f.drow_g, f.src[2], f.row[2], f.drow_g, (size_t) E, cudaMemcpyHostToDevice);
                    at += al256((size_t) E * f.drow_g);
                }
                if (!glmfast::cuda_ok(e, "qwen35 dense: FFN upload", err)) return false;
                f.dev[r] = d;
                set_slot(il, r, d);
            }
        }
    }
    if (ram_b > 0) {
        if (cudaHostAlloc(&Q->pinned, ram_b, cudaHostAllocPortable) != cudaSuccess) {
            cudaGetLastError();
            err = "qwen35 dense: " + std::to_string(ram_b >> 20) + " MB of pinned RAM for the FFN did not allocate";
            return false;
        }
        Q->pinned_bytes = ram_b;
        // the CPU parts from the mapping on a few threads (a cold page cache reads the file here): gate / up rows
        // [kg, n_ff), down's last n_ff - kg columns of each row
        struct Job { uint8_t* dst; const uint8_t* src; size_t len, rows, dpitch, spitch; };
        std::vector<Job> jobs;
        size_t at = 0;
        for (int il : Q->prompt_layers) {
            auto& f = Q->L[(size_t) il];
            if (f.lent) {   // (whole, and in VRAM from the mapping)
                for (int r = 0; r < 3; ++r) {
                    f.host[r] = (uint8_t*) Q->pinned + at;
                    jobs.push_back({f.host[r], f.src[r], f.bytes[r], 1, 0, 0});
                    at += al256(f.bytes[r]);
                    if (!glmfast::cuda_ok(cudaMemcpy(f.dev[r], f.src[r], f.bytes[r], cudaMemcpyHostToDevice),
                                          "qwen35 dense: a loaned layer's upload", err))
                        return false;
                    set_slot(il, r, f.dev[r]);
                }
                continue;
            }
            const size_t fc = (size_t) (FF - f.kg);
            for (int r = 0; r < 3; ++r) {
                f.host[r] = (uint8_t*) Q->pinned + at;
                if (r < 2) {
                    jobs.push_back({f.host[r], f.src[r] + (size_t) f.kg * f.row[r], fc * f.row[r], 1, 0, 0});
                    at += al256(fc * f.row[r]);
                } else {
                    for (size_t r0 = 0; r0 < (size_t) E; r0 += 512)   // (in pieces: the threads share a layer)
                        jobs.push_back({f.host[2] + r0 * f.drow_c, f.src[2] + r0 * f.row[2] + f.drow_g, f.drow_c,
                                        std::min<size_t>(512, (size_t) E - r0), f.drow_c, f.row[2]});
                    at += al256((size_t) E * f.drow_c);
                }
                // (a layer all on the CPU: the slot points at RAM - the device never reads it; pinned memory is mapped
                // under unified addressing, so a kernel that did would read it over PCIe, not fault)
                if (f.kg <= 0) set_slot(il, r, f.host[r]);
            }
        }
        std::atomic<size_t> next{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 8; ++t)
            th.emplace_back([&] {
                for (size_t i; (i = next.fetch_add(1)) < jobs.size();) {
                    const Job& j = jobs[i];
                    if (j.rows == 1) {
                        std::memcpy(j.dst, j.src, j.len);
                    } else {
                        for (size_t r = 0; r < j.rows; ++r) std::memcpy(j.dst + r * j.dpitch, j.src + r * j.spitch, j.len);
                    }
                }
            });
        for (auto& t : th) t.join();
        // the service: request / answer mapped, the pool, its thread
        const size_t q8e = (size_t) g.n_embd / 32 * 36;
        if (cudaHostAlloc(&Q->req_h, 64 + Q35dRam::kCols * q8e, cudaHostAllocMapped) != cudaSuccess ||
            cudaHostAlloc(&Q->ans_h, 64 + (size_t) Q35dRam::kCols * g.n_embd * sizeof(float), cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer(&Q->req_d, Q->req_h, 0) != cudaSuccess ||
            cudaHostGetDevicePointer(&Q->ans_d, Q->ans_h, 0) != cudaSuccess) {
            err = "qwen35 dense: the CPU FFN's mapped buffers did not allocate";
            return false;
        }
        std::memset(Q->req_h, 0, 64);
        std::memset(Q->ans_h, 0, 64);
        // one core short of all of them (the service thread spins on the request; 16-core Ryzen AI Max+ 395: 15
        // threads 22.2 tok/s, 16 21.0)
        int threads = std::max(1, physical_core_count() - 1);
        if (const char* v = getenv("STRATA_Q35D_THREADS")) threads = std::max(1, std::min(256, std::atoi(v)));
        Q->threads = threads;
        Q->act.assign((size_t) Q35dRam::kCols * kc::lane_q81_act_bytes(g.n_embd), 0);
        Q->hq.assign((size_t) Q35dRam::kCols * kc::lane_q81_act_bytes(g.n_ff_dense), 0);
        Q->ff.assign((size_t) Q35dRam::kCols * g.n_ff_dense, 0.0f);
        // STRATA_Q35D_PIN=<CPU list> ("0-15"): the pool's workers on those CPUs (tests)
        std::vector<int> pin;
        if (const char* pv = getenv("STRATA_Q35D_PIN")) {
            for (const char* q = pv; *q;) {
                char* e = nullptr;
                const long a = std::strtol(q, &e, 10);
                if (e == q) break;
                long b2 = a;
                q = e;
                if (*q == '-') { b2 = std::strtol(q + 1, &e, 10); q = e; }
                for (long c = a; c <= b2 && c < 4096; ++c) pin.push_back((int) c);
                if (*q == ',') ++q; else if (*q) break;
            }
        }
        Q->pool.reset(new glmfast::Workers(threads - 1, 20000, pin));
        Q->svc = std::thread([Q] { Q->serve(); });
        // the prompt path's staging buffers
        if (Q->stage_block != nullptr) {
            const size_t sb = Q->stage_bytes;
            Q->stage[0] = (uint8_t*) Q->stage_block;
            Q->stage[1] = (uint8_t*) Q->stage_block + sb;
            bool ok = cudaStreamCreateWithFlags(&Q->cs, cudaStreamNonBlocking) == cudaSuccess;
            for (int b = 0; b < 2 && ok; ++b)
                ok = cudaEventCreateWithFlags(&Q->ready[b], cudaEventDisableTiming) == cudaSuccess &&
                     cudaEventCreateWithFlags(&Q->freed[b], cudaEventDisableTiming) == cudaSuccess;
            if (!ok) {
                cudaGetLastError();
                err = "qwen35 dense: the prompt path's staging events did not create";
                return false;
            }
        }
    }
    Q->placed = true;
    cudaMemGetInfo(&free_b, &total_b);
    if (getenv("STRATA_GLM_TIMING") != nullptr)
        std::fprintf(stderr, "glm fast: CUDA%d dense qwen35 placement: %.2f GB in use before it (the verify window %.2f "
                             "GB of it), room %.2f GB after a %zu MB reserve, FFN %.2f GB in all\n", dev_,
                     (double) in_use0 / G, (double) verify_b / G, (double) room / G, reserve >> 20, (double) all / G);
    char extra[160] = "";
    if (Q->stage_bytes > 0)
        std::snprintf(extra, sizeof extra, ", prompt staging 2 x %zu MB; %d layers' FFN (%.2f GB) in the prompt path's "
                                           "scratch between prompts", Q->stage_bytes >> 20,
                      (int) (Q->prompt_layers.size() - Q->ram_layers.size()), (double) Q->loan_bytes / G);
    char capb[64] = "";
    if (cap > 0) std::snprintf(capb, sizeof capb, " (cap %.1f GB)", (double) cap / G);
    if (m == 0)
        std::fprintf(stderr, "glm fast: CUDA%d dense qwen35: the FFN of all %d layers in VRAM (%.2f GB); VRAM %.2f GB in use%s\n",
                     dev_, NL, (double) dev_b / G, (double) (total_b - free_b) / G, capb);
    else
        std::fprintf(stderr, "glm fast: CUDA%d dense qwen35: the FFN %.2f GB in VRAM, %.2f GB in RAM (%d layers' last %d of "
                             "%d rows, pinned; %d CPU threads)%s; VRAM %.2f GB in use%s\n", dev_, (double) dev_b / G,
                     (double) ram_b / G, m, c_rows, FF, Q->threads, extra, (double) (total_b - free_b) / G, capb);
    return true;
}

bool Glm5Model::q35d_cpu_ffn(int il, const void* xq, int nt, float* out, cudaStream_t s, std::string& err,
                             const std::function<bool(int)>& gpu) {
    Q35dRam* Q = q35d_;
    if (Q == nullptr || Q->req_d == nullptr || nt < 1 || nt > Q35dRam::kCols) {
        err = "qwen35 dense: a RAM layer's FFN without the CPU service";
        return false;
    }
    const size_t q8e = (size_t) g_.n_embd / 32 * 36;
    ++Q->seq;
    if (Q->seq == 0) ++Q->seq;   // (0 is the empty header)
    q35::dffn_post(xq, (size_t) nt * q8e, Q->req_d, Q->seq, il, nt, s);
    const int kg = std::max(0, Q->L[(size_t) il].kg);
    if (kg > 0) {
        if (!gpu) {
            err = "qwen35 dense: a split layer's FFN without its GPU part";
            return false;
        }
        if (!gpu(kg)) return false;   // (runs while the CPU computes its rows)
    }
    q35::dffn_wait(Q->ans_d, Q->seq, out, nt * g_.n_embd, s, kg > 0);
    return true;
}

// the prompt path: layer il's FFN in staging buffer k % 2 (k: its place among the RAM layers), the next RAM layer's
// copy started into the other one - the weights never change, so a buffer that already holds a layer is reused
bool Glm5Model::q35d_stage(int il, cudaStream_t s, WSlot* w, std::string& err) {
    Q35dRam* Q = q35d_;
    if (Q == nullptr || Q->stage[0] == nullptr) {
        err = "qwen35 dense: a RAM layer's FFN in the prompt path without the staging buffers";
        return false;
    }
    const int n = (int) Q->prompt_layers.size();
    const int k = (int) (std::lower_bound(Q->prompt_layers.begin(), Q->prompt_layers.end(), il) - Q->prompt_layers.begin());
    if (k >= n || Q->prompt_layers[(size_t) k] != il) {
        err = "qwen35 dense: blk." + std::to_string(il) + " is not a RAM layer";
        return false;
    }
    // buffer b: the one that holds il already, else the one the last layer did not use (its copy waits for that
    // one's products before); then the next RAM layer's copy into the other (the last of a chunk wraps to the first:
    // the next chunk's)
    const auto issue = [&](int b, int l) {
        if (Q->holds[b] == l) {
            ++Q->stage_hits;
            return;
        }
        cudaStreamWaitEvent(Q->cs, Q->freed[b], 0);
        // the whole layer as it was in the file: gate / up rows [0, kg) from VRAM, [kg, n_ff) from RAM; down's
        // columns the same way, row by row (2D copies)
        const auto& f = Q->L[(size_t) l];
        const int kg = f.lent ? 0 : std::max(0, f.kg), E = Q->n_embd;   // (a loaned layer: its whole pinned copy)
        uint8_t* d = Q->stage[b];
        for (int r = 0; r < 2; ++r) {
            const size_t g0 = (size_t) kg * f.row[r];
            if (kg > 0) cudaMemcpyAsync(d, f.dev[r], g0, cudaMemcpyDeviceToDevice, Q->cs);
            cudaMemcpyAsync(d + g0, f.host[r], f.bytes[r] - g0, cudaMemcpyHostToDevice, Q->cs);
            d += al256(f.bytes[r]);
        }
        if (kg > 0) {
            cudaMemcpy2DAsync(d, f.row[2], f.dev[2], f.drow_g, f.drow_g, (size_t) E, cudaMemcpyDeviceToDevice, Q->cs);
            cudaMemcpy2DAsync(d + f.drow_g, f.row[2], f.host[2], f.drow_c, f.drow_c, (size_t) E, cudaMemcpyHostToDevice, Q->cs);
        } else {
            cudaMemcpyAsync(d, f.host[2], f.bytes[2], cudaMemcpyHostToDevice, Q->cs);
        }
        cudaEventRecord(Q->ready[b], Q->cs);
        Q->holds[b] = l;
        ++Q->staged;
    };
    int b = Q->holds[Q->cur ^ 1] == il ? Q->cur ^ 1 : Q->holds[Q->cur] == il ? Q->cur : Q->cur ^ 1;
    issue(b, il);
    if (n >= 2) issue(b ^ 1, Q->prompt_layers[(size_t) ((k + 1) % n)]);
    Q->cur = b;
    cudaStreamWaitEvent(s, Q->ready[b], 0);
    const auto& f = Q->L[(size_t) il];
    size_t at = 0;
    for (int r = 0; r < 3; ++r) {
        w[r].q = Q->stage[b] + at;
        w[r].type = f.type[r];
        at += al256(f.bytes[r]);
    }
    return true;
}

void Glm5Model::q35d_unstage(int il, cudaStream_t s) {
    Q35dRam* Q = q35d_;
    if (Q == nullptr) return;
    for (int b = 0; b < 2; ++b)
        if (Q->holds[b] == il) cudaEventRecord(Q->freed[b], s);
}

void Glm5Model::q35d_destroy() {
    Q35dRam* Q = q35d_;
    if (Q == nullptr) return;
    Q->quit.store(true);
    if (Q->svc.joinable()) Q->svc.join();
    if (const uint64_t c = Q->calls.load(); c > 0)
        std::fprintf(stderr, "glm fast: CUDA%d dense qwen35 CPU FFN: %llu calls of %.2f columns, %.3f ms each; prompt "
                             "staging %llu copies (%llu reused), %llu loan restores\n", dev_, (unsigned long long) c,
                     (double) Q->cols.load() / (double) c, (double) Q->us.load() / 1000.0 / (double) c,
                     (unsigned long long) Q->staged, (unsigned long long) Q->stage_hits, (unsigned long long) Q->restores);
    if (const uint64_t c = Q->calls.load(); c > 0 && getenv("STRATA_GLM_TIMING") != nullptr)
        std::fprintf(stderr, "glm fast: CUDA%d dense qwen35 CPU FFN phases a call: gate/up %.3f ms, h %.3f, down %.3f\n",
                     dev_, Q->ms_gu / (double) c, Q->ms_q / (double) c, Q->ms_dn / (double) c);
    Q->pool.reset();
    if (Q->cs) cudaStreamSynchronize(Q->cs);
    for (int b = 0; b < 2; ++b) {
        Q->stage[b] = nullptr;
        if (Q->ready[b]) cudaEventDestroy(Q->ready[b]);
        if (Q->freed[b]) cudaEventDestroy(Q->freed[b]);
    }
    if (Q->cs) cudaStreamDestroy(Q->cs);
    if (Q->stage_block) cudaFree(Q->stage_block);
    if (Q->req_h) cudaFreeHost(Q->req_h);
    if (Q->ans_h) cudaFreeHost(Q->ans_h);
    if (Q->pinned) cudaFreeHost(Q->pinned);
    if (Q->dev) cudaFree(Q->dev);
    delete Q;
    q35d_ = nullptr;
}

}  // namespace strata::core
