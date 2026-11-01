// src/core/glm_fast_state.hpp - the fast path's private state (Glm5Model::FastState) and host helpers, shared by
// src/core/glm_fast_path.cu (the one-token decode path) and src/core/glm_prefill.cu (the batched prompt path).
// Not a public header: glm_model.cu also uses it to reset request-local routing errors.
#pragma once

#include "strata/core/glm_model.hpp"
#include "strata/igpu/igpu_lane.h"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "glm_memory.hpp"
#include "glm_expert_helper.hpp"
#include "glm_layer_graphs.hpp"

#include "ggml.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>   // (pin_thread's SetThreadAffinityMask)
#endif

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace strata::core {
namespace glmfast {
inline bool cuda_ok(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}
inline bool wait_event(cudaEvent_t ev, const char* what, std::string& err) {
    cudaError_t e;
    while ((e = cudaEventQuery(ev)) == cudaErrorNotReady) std::this_thread::yield();
    return cuda_ok(e, what, err);
}

namespace gf = strata::kernels::glmf;

inline void cpu_relax() {
#if defined(__x86_64__) && !defined(_WIN32)
    asm volatile("pause" ::: "memory");
#endif
}

// a thread's CPUs: `cpus` (empty: left to the OS).  Linux, and Windows within its first processor group (CPUs 0-63:
// SetThreadAffinityMask - issue #4, a single-GPU i7-14700K on Windows 11); elsewhere a no-op.
inline void pin_thread(std::thread::native_handle_type h, const std::vector<int>& cpus) {
#ifdef __linux__
    if (cpus.empty()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus)
        if (c >= 0 && c < CPU_SETSIZE) CPU_SET(c, &set);
    pthread_setaffinity_np(h, sizeof set, &set);
#elif defined(_WIN32)
    if (cpus.empty()) return;
    DWORD_PTR mask = 0;
    for (int c : cpus)
        if (c >= 0 && c < 64) mask |= (DWORD_PTR) 1 << c;
    if (mask != 0) {
        SetThreadAffinityMask((HANDLE) h, mask);
        THREAD_POWER_THROTTLING_STATE throttling{};
        throttling.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        throttling.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        throttling.StateMask = 0;
        SetThreadInformation((HANDLE) h, ThreadPowerThrottling, &throttling, sizeof(throttling));
        SetThreadPriority((HANDLE) h, THREAD_PRIORITY_ABOVE_NORMAL);
    }
#else
    (void) h;
    (void) cpus;
#endif
}

// a fixed pool of worker threads: run(n, fn) calls fn(0..n-1) across the workers AND the caller and returns
// when all finished.  Jobs are claimed lock-free; a worker spins ~300 us for the next batch before it sleeps,
// because one miss request runs its batches back to back (reads, gate/up, down) and a futex wake per batch
// was most of the CPU path's latency.
class Workers {
public:
    // spin_us: how long an idle worker spins for the next batch before it sleeps; cpus: the CPUs the workers run on
    // (empty: any)
    explicit Workers(int n, int spin_us = 300, const std::vector<int>& cpus = {}) : spin_us_(spin_us), active_(n + 1) {
        for (int i = 0; i < n; ++i) {
            th_.emplace_back([this, i] { loop(i); });
            pin_thread(th_.back().native_handle(), cpus);
        }
    }
    ~Workers() {
        quit_.store(true);
        {
            std::lock_guard<std::mutex> lk(mu_);
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    void run(int n, const std::function<void(int)>& fn) {
        if (n <= 0) return;
        std::lock_guard<std::mutex> one(run_mu_);   // a pool a split's parts share: one batch at a time
        const uint64_t g = (gen_.load(std::memory_order_relaxed) + 1) & 0xffffffffu;
        // the ticket packs (generation, next index): a straggler of an older batch can never claim (or burn)
        // an index of this one - its CAS fails on the generation.  The old batch's ticket is CLOSED before total_
        // changes: a straggler that read the spent ticket (old generation, index n_old) and then this batch's larger
        // total_ would otherwise claim index n_old of the old batch - and run this batch's job with it, done_ counting
        // it too, so run() could return while a job still ran (a crash once the caller's frame was gone)
        ticket_.store((g << 32) | 0x7fffffffu, std::memory_order_seq_cst);   // (an index no batch reaches)
        fn_ = &fn;
        total_.store(n, std::memory_order_release);
        done_.store(0, std::memory_order_relaxed);
        ticket_.store(g << 32, std::memory_order_release);
        gen_.store(g, std::memory_order_release);
        if (sleeping_.load(std::memory_order_acquire) > 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_.notify_all();
        }
        work(g);
        while (done_.load(std::memory_order_acquire) < n) cpu_relax();
    }
    int size() const { return (int) th_.size() + 1; }
    // only the caller and the first n - 1 workers take jobs (fewer threads without a restart: setup's calibration);
    // a batch's job count should be active() for one job a thread
    void set_active(int n) { active_.store(std::max(1, std::min(n, size())), std::memory_order_release); }
    int active() const { return active_.load(std::memory_order_acquire); }

private:
    void work(uint64_t g) {
        for (;;) {
            uint64_t t = ticket_.load(std::memory_order_acquire);
            if ((t >> 32) != g) return;
            const int n = total_.load(std::memory_order_acquire);
            if ((int) (t & 0xffffffffu) >= n) return;
            if (!ticket_.compare_exchange_weak(t, t + 1, std::memory_order_acq_rel)) continue;
            (*fn_)((int) (t & 0xffffffffu));
            done_.fetch_add(1, std::memory_order_acq_rel);
        }
    }
    void loop(int id) {
        uint64_t seen = gen_.load();
        for (;;) {
            const auto t0 = std::chrono::steady_clock::now();
            int spins = 0;
            // a worker set_active left out sleeps at once: spinning, it held a CPU the active ones and the disk
            // readers wanted, so the setup's calibration timed fewer threads on a CPU still full of spinners
            // (24-core Ultra 9 275HX, no SMT, Windows: a decode's disk wait 16.8 ms with 24 threads, 8.1 ms with 16)
            const bool idle = id + 1 >= active_.load(std::memory_order_acquire);
            while (gen_.load(std::memory_order_acquire) == seen) {
                if (quit_.load(std::memory_order_relaxed)) return;
                cpu_relax();
                if (idle || ((++spins & 255) == 0 &&
                             std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_us_))) {
                    std::unique_lock<std::mutex> lk(mu_);
                    sleeping_.fetch_add(1);
                    cv_.wait(lk, [&] { return quit_.load() || gen_.load() != seen; });
                    sleeping_.fetch_sub(1);
                    break;
                }
            }
            if (quit_.load()) return;
            seen = gen_.load(std::memory_order_acquire);
            if (id + 1 < active_.load(std::memory_order_acquire)) work(seen);
        }
    }
    int spin_us_;
    std::vector<std::thread> th_;
    std::mutex mu_, run_mu_;
    std::condition_variable cv_;
    const std::function<void(int)>* volatile fn_ = nullptr;
    std::atomic<int> total_{0}, done_{0}, sleeping_{0}, active_;
    std::atomic<uint64_t> ticket_{0}, gen_{0};
    std::atomic<bool> quit_{false};
};

// [off, off + len) of a shard into dst (O_DIRECT when the shard has a direct fd; see glm_fast_path.cu)
void read_slice(const Glm5Model::Shard& sh, uint64_t off, size_t len, uint8_t* dst);

// A layer's VRAM slot stride: the expert blob ([gate | up | down]) rounded up so that both matrices' block sizes and
// 16 bytes divide it - llama.cpp's MMQ addresses expert e of a pool partition as base + e * stride in WHOLE blocks,
// so the prompt path multiplies the resident experts where they lie.
inline size_t expert_stride(size_t blob, int gu_type, int d_type) {
    // an EXL3 blob: every piece is a multiple of 256 bytes, no ggml blocks to keep whole (the prompt path does not
    // multiply these with MMQ)
    if (gu_type == strata::kernels::exl3::kTypeEXL3 || d_type == strata::kernels::exl3::kTypeEXL3)
        return (blob + 255) / 256 * 256;
    size_t a = 16;
    const size_t b1 = ggml_type_size((ggml_type) gu_type), b2 = ggml_type_size((ggml_type) d_type);
    if (b1 > 0) a = std::lcm(a, b1);
    if (b2 > 0) a = std::lcm(a, b2);
    return (blob + a - 1) / a * a;
}

}  // namespace glmfast

namespace gf = strata::kernels::glmf;

struct Glm5Model::FastState {
    bool unified_memory = false;   // set only for integrated HIP devices on Linux; CUDA and Windows keep the discrete policy
    cudaStream_t cs = nullptr, copy = nullptr, ps = nullptr;   // ps: the prefetch side stream
    cudaEvent_t ev_hop = nullptr, ev_done = nullptr, ev_pred = nullptr, ev_pf = nullptr, ev_pf_prev = nullptr;
    // ROUTE TAIL SKIP (STRATA_GLM_TAIL_SKIP=<r>): a trunk route's experts at rank >= r that are not in VRAM are left out
    // (no PCIe pull, no CPU lane, no disk wait); the others keep the router's weights.  8: off.  tail_skips counts them.
    // STRATA_GLM_TAIL_SKIP_W=<share>: also (or instead) those whose share of the route's probability is below it.
    int tail_skip = gf::kMaxK;   // (glmf::kMaxK: off)
    float tail_skip_w = 0.0f;
    bool tail_skip_disk = false;   // STRATA_GLM_TAIL_SKIP_DISK=1: ... only the disk-only ones
    std::atomic<uint64_t> tail_skips{0};
    // EXPERT DEFERRAL (STRATA_GLM_DEFER=1, lossy, off by default): the CPU lane's part of a trunk layer's MoE is not
    // waited for at the layer's end; after the NEXT layer's attention it is written into the residual streams with
    // its own layer's write weights (hc_defer_add).  The device never idles for the lane but at the last trunk
    // layer (its own part, once a token); the draft layer and the verify window keep the in-order sum
    bool defer = false;
    float* post_defer = nullptr;   // device, 8 floats: the FFN write weights of the layer whose part is deferred
    int max_pf = 0;   // STRATA_GLM_PREFETCH_N: measured a net LOSS on Mercury (the window between routes is shorter than one fetch)
    std::atomic<uint64_t> prefetches{0};
    unsigned long long* pf_buf[2] = {nullptr, nullptr};   // [src 8 | dst 8] per layer parity
    int* pf_n_buf[2] = {nullptr, nullptr};
    void* arena = nullptr;
    size_t arena_bytes = 0;
    // activations (device)
    float *x = nullptr, *mixer = nullptr, *ffn = nullptr, *pre = nullptr, *post = nullptr, *comb = nullptr;
    float* part = nullptr;
    unsigned int* counter = nullptr;
    void* xq = nullptr;
    float* proj[3] = {nullptr, nullptr, nullptr};
    float* conv[3] = {nullptr, nullptr, nullptr};
    float *fa = nullptr, *ga = nullptr, *beta = nullptr, *g1 = nullptr, *g2 = nullptr;
    void* gated_q = nullptr;
    float *qr_raw = nullptr, *qr = nullptr, *kv_raw = nullptr, *ik_raw = nullptr, *ig_raw = nullptr, *iw = nullptr;
    float *q = nullptr, *iq = nullptr, *score = nullptr;
    int* cells = nullptr;
    void *qr_q = nullptr, *attn_q = nullptr;
    float *rlog = nullptr, *plog = nullptr, *sh_g = nullptr, *sh_u = nullptr;
    float* sh_out = nullptr;   // the shared expert's down (n_embd), computed alongside the routed gate/up
    // prediction quality (the next layer's router on this layer's FFN input)
    std::vector<std::array<int, 8>> pred_of;   // per layer: the prediction made one layer earlier
    std::atomic<uint64_t> pred_n{0}, pred_overlap{0}, pred_miss{0}, pred_miss_hit{0};
    // ---- LOOKAHEAD (STRATA_GLM_AHEAD=<n layers>, 0 off): every route also names the top-k of the next n layers' routers
    //      on its FFN input; the disk-only experts among them are read into the RAM tier by the reader threads while
    //      the layers in between compute, so their own route finds them on the host (a short wait or none)
    int n_ahead = 0;
    bool ahead_read = false;                   // STRATA_GLM_AHEAD_READ=0: predict and count only
    // STRATA_GLM_RAM_FILL=<n> (default 2 with the disk tail skip, else 0): a boundary's background reads of the most
    // routed disk-only experts into the RAM tier (issue #8: the tail skip never reads the experts it leaves out, so
    // they never leave the disk); one is read over a colder RAM expert when its count beats that one's by
    // STRATA_GLM_RAM_FILL_MARGIN (4)
    int ram_fill = 0;
    uint32_t ram_fill_margin = 4;
    uint64_t fill_issued = 0, fill_evict = 0, fill_passed = 0;
    bool disk_lane = false;                    // STRATA_GLM_DISK_LANE=1: disk misses computed by the lane in RAM
    bool disk_adapt = false;                   // STRATA_GLM_DISK_ADAPT=1: ... evicting at the miss (arena-adapt)
    bool stream_fetch = true;                  // STRATA_GLM_STREAM_FETCH=0: the fetch after the whole read, as before
    int read_chunks = 8;                       // STRATA_GLM_READ_CHUNKS: a part's O_DIRECT reads (also the stream's bits)
    float* alog = nullptr;                     // kAhead x n_expert logits (device)
    std::vector<std::array<std::array<short, gf::kMaxK>, gf::kAhead>> ah_pred;   // per layer: the predictions d+1 layers early
    struct AheadJob {
        int rclass, rslot, il, e;
        uint64_t tag;
        bool fill = false;                     // a RAM fill: read in chunks, each waiting for the routes' reads to end
    };
    std::atomic<int> disk_busy{0};             // a route's own disk reads in flight (the fills yield to them)
    std::mutex ah_mu;                          // the job queue (never held with mu)
    std::condition_variable ah_cv;
    std::vector<AheadJob> ah_q;
    std::vector<std::thread> ah_th;
    bool ah_quit = false;
    std::atomic<int> ah_inflight{0};
    // per RAM class and slot, for kRLoad slots: tag << 2 | phase (0 queued, 1 reading, 2 landed) - a job reads only while
    // the tag it carries is current (a slot freed and reused under it gets a new tag)
    std::vector<std::unique_ptr<std::atomic<uint64_t>[]>> rload;
    uint64_t ah_tag = 0;
    // stats (the service thread)
    uint64_t ah_routes = 0, ah_ov[gf::kAhead] = {0, 0, 0, 0}, ah_npred[gf::kAhead] = {0, 0, 0, 0};
    uint64_t ah_disk = 0, ah_disk_cov[gf::kAhead] = {0, 0, 0, 0}, ah_disk_any = 0;
    uint64_t ah_issued = 0, ah_used = 0, ah_waited = 0, ah_stolen = 0, ah_full = 0, ah_nofree = 0;
    std::atomic<uint64_t> ah_wait_us{0};
    void *sh_hq = nullptr, *hq = nullptr;
    float *dg = nullptr, *du = nullptr;
    void* dhq = nullptr;
    float* head_x = nullptr;
    void* head_xq = nullptr;
    float* emb = nullptr;
    int* tok = nullptr;
    // the NextN block (the half that carries it): its residual, eh_proj's q8_1 input, its logits and token
    float* mtp_h = nullptr;
    void* mtp_catq = nullptr;
    float* mtp_logits = nullptr;
    int* mtp_tok = nullptr;
    int* mtp_tok_h = nullptr;    // pinned
    cudaEvent_t ev_mtp = nullptr;
    // routing
    gf::MoeDev md;
    unsigned long long* tab = nullptr;
    gf::MoeRequest* ring_h = nullptr;
    gf::MoeResponse* resp_h = nullptr;
    float* emb_h = nullptr;     // pinned
    // ZERO-COPY TOKEN BOUNDARY (STRATA_GLM_ZERO_COPY=0 turns it off): the small copies at a token's start and end
    // (the table edits, the embedding row, the argmax / sampler result) went through the copy engines, behind any
    // 8 MB expert move in flight there (0.5-0.9 ms of GPU idle a token on an RTX 5090 at Gen4 x4 with the default
    // promotions).  Instead the kernels read the host-mapped edit sets and a pinned copy of token_embd, and write
    // the token into host-mapped memory.
    bool zero_copy = false;
    int* tok_hd = nullptr;          // tok_h's device address (then tok_h is host-mapped)
    int* tok_in_h = nullptr;        // the token(s) going in, host-mapped: the embedding kernel reads them
    int* tok_in_d = nullptr;
    uint8_t* emb_tab_h = nullptr;   // pinned copy of token_embd (Q8_0), host-mapped
    uint8_t* emb_tab_d = nullptr;
    uint8_t* emb_q8_h = nullptr;    // Qwen3.5's NextN drafter when token_embd did not pin: a Q8_0 copy, host-mapped
    uint8_t* emb_q8_d = nullptr;
    int* upd_key_hd[2] = {nullptr, nullptr};   // the edit sets' device addresses
    unsigned long long* upd_val_hd[2] = {nullptr, nullptr};
    float* hop_h = nullptr;     // pinned, hc * n_embd
    int* tok_h = nullptr;       // pinned
    // per layer: the resolved weights
    struct Layer {
        bool recr = false, moe = false, mtp = false;
        WSlot eh;                                                   // the NextN block: eh_proj and its norms
        const float *enorm = nullptr, *hnorm = nullptr, *shnorm = nullptr;
        const uint16_t *hc_attn_fn = nullptr, *hc_ffn_fn = nullptr;
        const float *hc_attn_scale = nullptr, *hc_attn_base = nullptr, *hc_ffn_scale = nullptr, *hc_ffn_base = nullptr;
        const float *attn_norm = nullptr, *ffn_norm = nullptr;
        WSlot q, k, v, out;
        const uint16_t *f_a = nullptr, *g_a = nullptr, *f_b = nullptr, *g_b = nullptr, *beta = nullptr;
        const float *conv[3] = {nullptr, nullptr, nullptr};
        const float *dt_bias = nullptr, *ssm_a = nullptr, *ssm_norm = nullptr;
        WSlot q_a, q_b, kv_a;
        const float *q_a_norm = nullptr, *kv_a_norm = nullptr, *k_norm_w = nullptr, *k_norm_b = nullptr, *ape = nullptr;
        const uint16_t *idx_k = nullptr, *idx_gate = nullptr, *idx_q_b = nullptr, *idx_proj = nullptr;
        const uint16_t *k_b = nullptr, *v_b = nullptr;
        const uint16_t* router = nullptr;
        int router_type = gf::kTypeBF16;                            // deepseek41: kTypeF32 (router then points at F32)
        const float* router_bias = nullptr;
        const float* router_bias_vl = nullptr;                      // deepseek41: exp_probs_b_vl (an image position's bias)
        WSlot sh_gate, sh_up, sh_down;
        WSlot ffn_gate, ffn_up, ffn_down;
        int gu_type = 0, d_type = 0;
        size_t blob = 0, down_off = 0, gu_bytes = 0, dn_bytes = 0;
        // ---- deepseek4 (dsv4_fast_path.cu): the attention mixer's weights and this layer's caches
        int ratio = 0;                                              // 0 window only, 4 CSA, 128 HCA
        WSlot kv, out_a, out_b, comp_kv, comp_gate, idx_comp_kv, idx_comp_gate, idx_q_b_q;
        const float *sinks = nullptr, *comp_ape = nullptr, *comp_norm = nullptr, *idx_comp_ape = nullptr,
                    *idx_comp_norm = nullptr, *idx_proj_f = nullptr;
        const int* tid2eid = nullptr;                               // hash-routed layers: [vocab][k] expert ids
        uint16_t* raw = nullptr;                                    // the window ring (FP16 rows of head_dim)
        uint16_t* comp_cache = nullptr;                             // compressed rows (CSA / HCA)
        uint16_t* idx_cache = nullptr;                              // the indexer's compressed keys (CSA)
        float *ring_kv = nullptr, *ring_sc = nullptr, *iring_kv = nullptr, *iring_sc = nullptr;
        // ---- deepseek41 (dsv41_fast_path.cu): the KV-source / index-source roles and Engram
        int slot = -1;                                              // the compressed cache read (-1: window only)
        bool kv_src = false, idx_src = false;
        int eng = -1;                                               // Engram table (-1: none)
        WSlot idx_k_w, eng_kv;                                      // indexer.attn_k (BF16), engram_kv (BF16)
        const float *idx_k_norm = nullptr, *eng_qn = nullptr, *eng_kn = nullptr;
        // ---- qwen35moe (qwen35_fast_path.cu)
        WSlot qkv, zg, ssm_o, alpha_w, beta_w;                      // GDN: attn_qkv, attn_gate, ssm_out, ssm_alpha, ssm_beta
        const float* conv1d = nullptr;                              // ssm_conv1d, [C][d_conv] (GGUF: ne0 = d_conv)
        WSlot aq, ak, av;                                           // attention: attn_q (q | gate a head), attn_k, attn_v
        const float *q_norm = nullptr, *k_norm = nullptr, *post_norm = nullptr, *sh_gate_w = nullptr;
        float *gdn_state = nullptr, *gdn_conv = nullptr;            // (S x h_v x S), (C x (d_conv - 1))
        uint16_t *kc = nullptr, *vc = nullptr;                      // the KV cache, FP16 [ctx][attn_head_kv][head_dim]
        // ---- qwen4exp (q4x, qwen35_fast_path.cu): the hyper-connections' mixers, [0] the attention's, [1] the FFN's
        struct Hc4 {
            const float* norm = nullptr;                            // the grouped norm's gamma [hc][n_embd]
            WSlot down, up, inject;                                 // hc*n_embd -> hc_lr, hc_lr -> hc*n_embd, -> hc
        } hc4[2];
        // qwen4exp's QSA indexer (an attention layer): k_proj / q_proj (BF16), their norms, and the block keys -
        // the raw tail, the spare slot's key, the pooled keys (normed, rotated; the old path's indexer_key_append)
        WSlot q4_ik, q4_iq;
        const float *q4_ikn = nullptr, *q4_iqn = nullptr;
        float *q4_tail = nullptr, *q4_dead = nullptr, *q4_pooled = nullptr;
        int32_t* q4_bpos = nullptr;
        // EXL3 (an exl3.txt pack): the routed experts' blob layout, and the indexer's q_b when it is an EXL3 matrix
        bool exl3 = false;
        strata::kernels::exl3::ExpertLayout xl;
        WSlot idx_q_b_x;
    };
    std::vector<Layer> L;
    // ---- deepseek4 activations (dsv4_fast_path.cu)
    float *ds_ckv[2] = {nullptr, nullptr}, *ds_csc[2] = {nullptr, nullptr};   // the compressors' projections
    float *ds_o = nullptr, *ds_oa = nullptr;                    // the attention output (n_head x head_dim), wo_a's
    void *ds_oq = nullptr, *ds_oaq = nullptr;                   // ... their q8_1
    int* ds_sel = nullptr;                                      // the indexer's chosen rows
    // ---- deepseek41 (dsv41_fast_path.cu)
    uint16_t *v41_comp[4] = {}, *v41_idx[4] = {};               // per KV slot: compressed rows (512) and index keys
    // per KV slot (ratio 2): the parked even token (v | g), a ring of kV41PendRing entries by the pair's index (p / 2):
    // a verify window's rejected tokens park theirs in other entries, so the pair a rewind left half done still finds
    // its first half
    static constexpr int kV41PendRing = 8;
    float* v41_pend[4] = {};
    float* v41_pre[2] = {};                                     // the hyper-connections' pre, alternating sublayers
    float *v41_score = nullptr, *v41_bscore = nullptr;          // the index scores (ctx), the block maxima (ctx / 8)
    int* v41_bsel = nullptr;                                    // the selected candidate blocks
    unsigned char* v41_keep = nullptr;                          // the candidate mask of this token (ctx / 8)
    float *v41_eng = nullptr, *v41_eng_h = nullptr;             // Engram rows (2 x 24 x 256 f32, device / pinned)
    unsigned int* v41_eng_flag = nullptr;                        // pinned: a table's rows landed for token v41_eng_seq
    unsigned int v41_eng_seq = 0;
    bool v41_eng_async = false;                                  // this token's rows come from the background reader
    float* v41_eng_kv = nullptr;                                // engram_kv's output (5 x n_embd)
    void* v41_eng_q = nullptr;                                  // the Engram rows as Q8_1 (a quantized engram_kv)
    float *v41_pblk = nullptr, *v41_pblk_h = nullptr;           // the prompt path's block of rows (256 tokens' 24 x 256)
    bool cur_img = false;                                       // the token in flight is an image row (deepseek41)
    int32_t cur_tok = 0;                                        // the token in flight (hash routing)
    // ---- qwen35moe activations (qwen35_fast_path.cu)
    float *q35_qkv = nullptr, *q35_z = nullptr, *q35_h = nullptr, *q35_ab = nullptr, *q35_gate = nullptr,
          *q35_o = nullptr, *q35_y = nullptr, *q35_qf = nullptr, *q35_k = nullptr, *q35_v = nullptr, *q35_q = nullptr,
          *q35_ao = nullptr, *q35_part = nullptr, *q35_shg = nullptr;
    void *q35_yq = nullptr, *q35_aoq = nullptr;
    // ---- qwen4exp activations: the normed streams, the low rank, the gate, the injections ([0] attention, [1] FFN:
    // each lives until its combine, the FFN's into the next layer)
    float *q4_xn = nullptr, *q4_lo = nullptr, *q4_gate = nullptr, *q4_inj[2] = {nullptr, nullptr};
    void *q4_xnq = nullptr, *q4_loq = nullptr;
    float* q4_sc = nullptr;   // (q35::hc4_fused's stream scales)
    const float* q4_head[1] = {nullptr};                         // output_hc_norm
    // qwen4exp's per-layer embedding (layer 1, qwen35_fast_path.cu q4x_ple_*): the table on the SSD, the tokens by
    // position (the n-gram hash reads the two before), the gathered rows (pinned / device / q8_1), the projections,
    // the normalized rows' history (9 rows a channel, row-fastest) and the postops' scratch
    std::unique_ptr<strata::kernels::PleTable> ple;
    std::vector<int32_t> ple_toks;
    bool ple_issued = false;
    int ple_il = -1;                                             // the PLE layer (-1: none on this part)
    float *ple_emb_h = nullptr, *ple_emb = nullptr, *ple_kp = nullptr, *ple_val = nullptr, *ple_hist = nullptr;
    float *ple_key = nullptr, *ple_norm = nullptr, *ple_gated = nullptr, *ple_conv = nullptr, *ple_gate = nullptr;
    void* ple_embq = nullptr;
    uint16_t* ple_conv16 = nullptr;
    WSlot ple_key_w, ple_val_w;
    strata::kernels::PleWeights ple_w;
    // qwen4exp's QSA (sparse attention past 2,051 cells): the RoPE table, the step record (pos, n_kv, n_bid, width)
    // and the rows' positions (device; host-mapped sources), the indexer's raw key / queries, the block scores, the
    // selected cells and their gathered K / V
    float *q4_cos = nullptr, *q4_sin = nullptr, *q4_iraw = nullptr, *q4_iq = nullptr, *q4_scores = nullptr;
    int32_t *q4_step = nullptr, *q4_pos = nullptr, *q4_step_h = nullptr, *q4_pos_h = nullptr, *q4_ids = nullptr;
    uint16_t *q4_ks = nullptr, *q4_vs = nullptr;
    int64_t q4_step_at = -1;                                     // the position the step record holds
    int q4_max_blocks = 0, q4_cap = 0;
    // ... and the prompt path's: kQ4Grp rows' block scores and selections, and the chunk's PLE rows (pinned)
    static constexpr int kQ4Grp = 64;
    float* q4_pf_scores = nullptr;
    int32_t* q4_pf_ids = nullptr;
    float* q4_pf_part = nullptr;                                 // ... their attention partials (CUDA)
    float* ple_pf_h = nullptr;
    size_t ple_pf_n = 0;
    WSlot q4_head_w[2];                                          // output_hc_down, output_hc_up
    // ---- the VRAM tier: per-layer partitions.  A slot is FREE, RESIDENT (in tab), a SPARE (handed to the
    //      device, which may fill it during a token and mark it resident itself) or DRAINING (its old expert
    //      is being demoted to the RAM tier by a D2H; it becomes free when that copy lands).
    // kLanding: a background promotion's copy is on its way into the slot (fast_boundary step 5)
    enum : char { kFree = 0, kResident = 1, kSpare = 2, kDraining = 3, kLent = 4, kLanding = 5 };
    // A layer's slots [0, n_main) are at base, [n_main, n) at xbase: every layer's tail lies in ONE region (xpool),
    // which the prompt path borrows for its buffers while a prompt runs (those slots are kLent then) - and an
    // on-demand vision encoder while it encodes (the region is freed; xbase is null until it comes back).
    struct LayerPool {
        uint8_t* base = nullptr;
        uint8_t* xbase = nullptr;
        size_t stride = 0;
        int n = 0, n_main = 0;
        uint8_t* slot_ptr(int s) const {
            return s < n_main ? base + (size_t) s * stride : xbase + (size_t) (s - n_main) * stride;
        }
        int slot_index(unsigned long long p) const {
            const unsigned long long b0 = (unsigned long long) base, x0 = (unsigned long long) xbase;
            if (p >= b0 && p < b0 + (unsigned long long) n_main * stride) return (int) ((p - b0) / stride);
            if (xbase != nullptr && p >= x0 && p < x0 + (unsigned long long) (n - n_main) * stride)
                return n_main + (int) ((p - x0) / stride);
            return -1;
        }
        std::vector<int> key;        // slot -> expert, -1 none
        std::vector<char> st;
        std::vector<uint64_t> tick;  // recency (LFU tie-break)
        int spare[gf::kSpares] = {-1, -1, -1};   // the device's spare table, mirrored (slot or -1)
        uint64_t evictions = 0;
    };
    std::vector<LayerPool> lp;   // indexed by absolute layer
    std::vector<int> slot_of;    // il * n_expert + e -> slot in lp[il], -1 absent
    std::vector<uint32_t> cnt;   // il * n_expert + e -> LFU count (aged), over both tiers
    // ... and the long memory: routes per key across sessions (expert_usage.txt), halved when one passes 2^30 -
    // the warm-up fills the tiers in its order (this user's experts first)
    std::vector<uint32_t> usage;
    uint64_t cnt_events = 0;
    uint8_t* pool = nullptr;
    // the main slots' further allocations, where one did not allocate (each holds whole layers: lp[il].base)
    std::vector<uint8_t*> pool_more;
    size_t pool_bytes = 0;
    int64_t pool_slots = 0;
    // the lendable tail (every layer's slots [n_main, n)) is its own allocation: an on-demand vision encoder can
    // have that memory while it runs (vision_lend frees it, vision_reclaim allocates it again)
    uint8_t* xpool = nullptr;
    size_t xpool_bytes = 0;
    // ---- the RAM tier: pinned host slots per blob size class, holding experts that are NOT in VRAM
    //      (exclusive tiers).  rtab (on the device) points at HOLDING slots only.
    // kRPin: still the expert's live copy, being promoted (never a victim until the copy landed)
    enum : char { kRFree = 0, kRHold = 1, kRRelease = 2, kRDemote = 3, kRNew = 4, kRLoad = 5, kRPin = 6 };
    struct RamClass {
        size_t stride = 0;
        uint8_t* base = nullptr;
        int n = 0;
        bool registered = false;   // numa_pinned's (mmap + cudaHostRegister), not cudaHostAlloc's
        std::vector<int> key;   // slot -> il * n_expert + e, -1 none
        std::vector<uint64_t> tick;   // the route clock when a route last used it (or it arrived): eviction's recency
        std::vector<char> st;   // kRFree / kRHold / kRRelease (promoted: free at the boundary) /
                                // kRDemote (a D2H landing) / kRNew (a disk read this token: rtab at the boundary) /
                                // kRLoad (a LOOKAHEAD read, maybe still in flight: rload says; rtab once it landed)
    };
    std::vector<RamClass> rc;
    std::vector<int> layer_rc;   // layer -> RAM class
    std::vector<int> ram_of;     // key -> slot in its class, -1
    // STRATA_GLM_TIER_DIAG=1 (debug): how each expert last left the tiers (0 never held, 1 dropped from VRAM, 2 evicted
    // from RAM, 3 dropped by the prompt path's lending), and the disk reads by that cause
    std::vector<uint8_t> left;
    uint64_t diag_disk[4] = {0, 0, 0, 0}, diag_drop = 0, diag_ram_evict = 0, diag_lend = 0, diag_lend_park = 0, diag_b = 0;
    uint64_t diag_resident_skip = 0;   // RAM-resident mode: spare refills skipped (no free RAM slot, no drop)
    uint64_t diag_dup = 0, diag_adopt = 0;   // the boundary's clean-up: duplicate RAM copies freed, lost ones adopted
    size_t ram_bytes = 0;
    // ---- demotions VRAM -> RAM (copy stream, D2H) of the victims that make room for spares
    struct Drain {
        int il, vslot, key, rclass, rslot;
        cudaEvent_t ev;
    };
    std::vector<Drain> draining;
    // ---- background moves between the tiers (fast_boundary step 5): a promotion (up, RAM -> VRAM) or a demotion
    //      (VRAM -> RAM), both on the copy stream; the expert stays live where it was until its copy landed
    struct BgMove {
        int il, vslot, key, rclass, rslot;
        bool up;
        cudaEvent_t ev;
        bool queued = false;   // a paced promotion whose last piece is not issued yet (its event not recorded)
    };
    std::vector<BgMove> bg;
    bool bg_hold = false;   // lend_tail: no new moves while the pool's tail is lent
    // ---- paced promotions (STRATA_GLM_BG_PACE_KB, default 1024; 0: the boundary issues each whole copy).  The
    //      copies a boundary issued at once (the refill after a prompt: 24 x 7 MB, ~22 ms of a Gen4 x4 link) held
    //      up the next token's first zero-copy reads (the table edits, the embedding row) until all of them had
    //      landed: the GPU idled ~20 ms a token.  Instead each MoE layer, once its CPU-lane answer is in, lets up to
    //      bg_pace bytes go: they cross while the next layer's mixer runs, which reads no host memory
    struct BgPiece {
        uint8_t* dst;
        const uint8_t* src;
        size_t n, done;
        cudaEvent_t ev;   // the move's: recorded after its last piece
        cudaMemcpyKind kind;   // a promotion (H2D) or a swap's demotion (D2H: posted writes, which the GPU's zero-copy
                               // read requests may not pass - they held the token's reads up the same way)
    };
    std::vector<BgPiece> bg_q;   // in issue order
    size_t bg_q_at = 0;          // the first piece not fully issued
    size_t bg_pace = 0;
    cudaEvent_t ev_bgq = nullptr;
    // up to `budget` bytes of the queued copies onto the copy stream, after the work queued on `s` so far (s null:
    // at once)
    void bg_drain(cudaStream_t s, size_t budget) {
        if (bg_q_at == bg_q.size()) return;
        if (s != nullptr) {
            cudaEventRecord(ev_bgq, s);
            cudaStreamWaitEvent(copy, ev_bgq, 0);
        }
        while (bg_q_at < bg_q.size() && budget > 0) {
            BgPiece& p = bg_q[bg_q_at];
            const size_t n = p.n - p.done < budget ? p.n - p.done : budget;
            cudaMemcpyAsync(p.dst + p.done, p.src + p.done, n, p.kind, copy);
            p.done += n;
            budget -= n;
            if (p.done < p.n) break;
            cudaEventRecord(p.ev, copy);
            for (auto& m : bg)
                if (m.ev == p.ev) m.queued = false;
            ++bg_q_at;
        }
        if (bg_q_at == bg_q.size()) {
            bg_q.clear();
            bg_q_at = 0;
        }
    }
    volatile int* route_error_h = nullptr;   // in the mapped ring allocation, visible after compute completion
    uint64_t bg_up = 0, bg_down = 0;
    std::vector<cudaEvent_t> ev_free;
    int* upd_key_h[2] = {nullptr, nullptr};            // two sets: a flush records an event on its set and the next
    unsigned long long* upd_val_h[2] = {nullptr, nullptr};   // boundary waits on it before refilling that set (no
    int* upd_key_d[2] = {nullptr, nullptr};            // stream sync after the update kernel)
    unsigned long long* upd_val_d[2] = {nullptr, nullptr};
    cudaEvent_t upd_ev[2] = {nullptr, nullptr};
    int upd_buf = 0;
    static constexpr int kMaxUpd = 8192;
    // a batch holds each table key ONCE (its last value): the update kernel writes the batch in parallel, so a key
    // edited twice in one boundary (made live, then freed as the coldest) had no defined winner
    std::vector<int> upd_at;          // table key -> its index in the current batch, valid when upd_gen[key] == upd_g
    std::vector<uint32_t> upd_gen;
    uint32_t upd_g = 0;
    int n_keys = 0;
    uint8_t* scratch = nullptr;
    std::vector<uint8_t*> stage;   // pinned: disk reads when no RAM slot is free
    std::mutex mu;   // the service thread vs the between-token boundary
    std::unique_ptr<glmfast::Workers> workers;   // parallel disk reads
    // ---- the CPU LANE (STRATA_GLM_CPU_LANE=<threads>, 0 off): a route's RAM-tier experts split between the device's
    //      PCIe pulls and this pool, which computes its share from the pinned blobs meanwhile; the device adds the
    //      weighted sum (cpu_ans_h) before its down combine.  cpu_plan: how many of f RAM-tier experts go to the host.
    //      A split of 3+ GPUs shares one pool across its parts (fast_cpu_lane_setup).
    std::unique_ptr<GlmExpertHelper> expert_helper;
    std::shared_ptr<glmfast::Workers> cpu_pool;
    int cpu_node = -1;                         // the NUMA node the pool is pinned to (-1: unpinned) ...
    std::vector<int> cpu_pin;                  // ... and its CPUs (the service thread runs there too)
    double cpu_c_ms = 0.0, cpu_p_ms = 0.0;   // the lane's calibration: an expert on the CPU, one over PCIe
    double cpu_ps_ms = 0.0;                  // ... one in a stream of copies (the prompt's staging)
    unsigned long long cpu_plan = 0;
    unsigned long long cpu_plan_start = 0;   // ... the plan the engine started with (set_pcie_share(-1) restores it)
    // STRATA_GLM_PCIE_SHARE_LATE (issue #14): a request starts on cpu_plan_start - its PCIe copies refill the GPU's
    // pool after the prompt - and moves to cpu_plan_late after late_after boundaries (0: off)
    unsigned long long cpu_plan_late = 0;
    int late_after = 0, late_n = 0;
    bool late_done = true;
    // ... or, with ":settle", once per session instead of per request: the VRAM hit rate is measured over windows of
    // settle_win boundaries; the late plan starts when a window gains less than settle_eps on the one before (the
    // pool has settled) and the start plan comes back when one falls settle_drop below that level (a new topic)
    bool settle = false, settle_late = false;
    int settle_win = 128, settle_n = 0;
    double settle_eps = 0.01, settle_drop = 0.05, settle_prev = -1.0, settle_level = 0.0;
    uint64_t settle_h0 = 0, settle_t0 = 0, settle_moves = 0;
    gf::CpuAnswer* cpu_ans_h = nullptr;                      // host-mapped
    unsigned int* cpu_seq_d = nullptr;
    unsigned int* dcnt_d = nullptr;                          // the device's route counts (the coldest go to the host)
    std::vector<strata::kernels::cpu::NativeFmt> cpu_fmt;   // per layer; n_ff == 0: the lane skips that layer
    std::vector<uint8_t> cpu_act, cpu_hq;                    // x's activation quant; 8 h quants
    std::vector<float> cpu_ff, cpu_dn;                       // 8 x n_ff; 8 x n_embd
    // fast_cpu_experts_multi's: a window's columns' activations, and its (expert, column) pairs' h, h quants and downs
    std::vector<uint8_t> cpu_act_m, cpu_hq_m;
    std::vector<float> cpu_ff_m, cpu_dn_m;
    std::atomic<uint64_t> cpu_multi_calls{0}, cpu_multi_cols{0}, cpu_multi_uses{0}, cpu_multi_experts{0};
    std::vector<float> cpu_x3;                               // EXL3: per expert gate / up inputs, h, down input, out
    std::atomic<uint64_t> cpu_experts{0}, cpu_routes{0}, cpu_us{0};
    std::atomic<uint64_t> dlane_experts{0}, dlane_us{0};   // the disk lane's experts and its time
    uint64_t dlane_evict = 0;                              // ... and its evictions at the miss (service thread)
    // ---- the iGPU LANE (STRATA_GLM_IGPU=1, src/igpu/igpu_lane.cpp): the lane's calls computed by an APU's integrated
    //      GPU from the registered RAM tier instead of the pool's threads; a call it cannot take stays on the CPU
    void* igpu_lib = nullptr;
    abyss_igpu_experts_fn igpu_experts = nullptr;
    abyss_igpu_experts_q_fn igpu_experts_q = nullptr;   // (optional) x's q8_1 from the request instead of x
    abyss_igpu_experts_start_fn igpu_start = nullptr;   // (optional) the lane call in two halves: a route's lane
    abyss_igpu_experts_finish_fn igpu_finish = nullptr; // experts started before its disk reads
    abyss_igpu_experts_x3_fn igpu_x3 = nullptr;   // an EXL3 layer's (abyss_igpu_experts_x3; absent: the CPU takes them)
    abyss_igpu_x3_start_fn igpu_x3s = nullptr;     // ... in two halves, the CPU threads computing a share between them
    abyss_igpu_x3_finish_fn igpu_x3f = nullptr;
    float x3_frac = 0.0f;                          // the CPU's share of an EXL3 route's lane experts (0: one of the two)
    std::vector<float> x3_gout;                    // the iGPU's share's sum
    abyss_igpu_poke_fn igpu_poke = nullptr;   // the service thread's keep-alive while a decode runs
    // the prompt path's host experts on the iGPU (abyss_igpu_rows; STRATA_GLM_PREFILL_IGPU=0: the CPU threads)
    abyss_igpu_rows_fn igpu_rows = nullptr;
    abyss_igpu_register_fn igpu_reg = nullptr;
    abyss_igpu_unregister_fn igpu_unreg = nullptr;
    std::atomic<uint64_t> igpu_calls{0}, igpu_falls{0}, igpu_early{0};   // (early: started before disk reads)
    abyss_igpu_experts_cols_fn igpu_cols = nullptr;   // (optional) a verify window's columns in one lane call (STRATA_GLM_IGPU_COLS)
    std::atomic<uint64_t> igpu_cols_calls{0}, igpu_cols_cols{0}, igpu_cols_falls{0};
    std::thread svc;
    std::atomic<bool> quit{false};
    uint64_t clock = 0;
    uint64_t expected = 0;               // requests this half's device has issued (one per MoE layer per token)
    std::atomic<uint64_t> processed{0};  // ... and the service thread has consumed
    // stats (written by the service thread, read by anyone)
    std::atomic<uint64_t> hits{0}, misses{0}, miss_layers{0}, ram_hits{0}, disk_reads{0}, promotions{0};
    std::atomic<uint64_t> scratch_uses{0}, demotions{0}, drops{0};
    std::atomic<uint64_t> miss_us{0}, disk_us{0};
    uint64_t tokens = 0;
    double ms = 0;
    // ---- STRATA_GLM_GAP_PROF=1: where the time between two tokens goes.  Host laps (ms, summed): from a token's
    //      readiness to the next call (sampling, the drafter, emitting), the boundary's stream sync / service-thread
    //      wait / table scan, the embedding row, the spin on the token; the GPU's own idle gap by timing events (the
    //      previous token's last launch -> this token's first); the sampler's round trips.  gap_report() prints them.
    bool gap_on = false, gap_live = false;
    cudaEvent_t ev_gs = nullptr, ev_ge[2] = {nullptr, nullptr}, ev_g0 = nullptr;
    int gap_par = 0;
    std::chrono::steady_clock::time_point t_ready{};
    double gap_host = 0, gap_bsync = 0, gap_bwait = 0, gap_bscan = 0, gap_emb = 0, gap_spin = 0, gap_gpu = 0,
           gap_sample = 0, gap_busy = 0, gap_notice = 0;
    bool token_spin = false;   // STRATA_GLM_TOKEN_SPIN=1: the wait for the token polls without yielding (an experiment)
    uint64_t gap_n = 0, gap_gpu_n = 0, gap_sample_n = 0;
    bool timing = false;
    bool ram_resident = false;   // STRATA_GLM_RAM_RESIDENT=1 / --glm-ram-resident: the RAM tier holds EVERY expert
                                 // VRAM does not, nothing ever drops to disk; the start fails if the tier is small
    int ram_slack = 0;           // extra RAM-tier slots per MoE layer (STRATA_GLM_RAM_SLACK; default 16 resident)
    // ---- STRATA_GLM_PROF=1: a cudaEvent after every launch, the gaps summed per kernel name (debug only)
    bool prof_on = false;
    // ---- the refusal-direction edit (src/core/glm_abl.cu; STRATA_GLM_ABL / --uncensor): Q [n_embd][abl_r] on the
    // device, the strengths [layer][2] on the host - ABL1 (an abliteration's edit): [attention, shared expert]; ABL2
    // (abl_full, --abl-build's): [attention, the whole FFN] and abl_emb on the embedding (every residual write)
    float* abl_q = nullptr;
    int abl_r = 0;
    std::vector<float> abl_s;
    bool abl_full = false;
    float abl_emb = 0.0f;
    int* abl_on = nullptr;      // device int: the per-request switch (1 on), read by the kernel
    float* abl_cap = nullptr;   // --abl-build: [n_layers + 1][n_embd], the residual entering each layer
    bool abl_cap_on = false;
    // qwen35moe's NextN drafter: the pending position's normed hidden, the row kernels' scratch (<= kVerifyMax rows),
    // the chain's tokens and confidences
    float *q35_pend_h = nullptr, *q35_mt_emb = nullptr, *q35_mt_hid = nullptr, *q35_mt_x = nullptr, *q35_mt_k = nullptr,
          *q35_mt_v = nullptr, *q35_mt_conf = nullptr;
    void *q35_mt_cat = nullptr, *q35_mt_xq = nullptr;
    int* q35_mt_tok = nullptr;
    float *q35_mt_conf_h = nullptr, *q35_mt_conf_d = nullptr;   // ... a step's confidence, host-mapped (the chain stops early)
    bool kda_graph_on = false;   // experimental, single-device ordinary decode only
    // the CPU lane's answer of the last MoE, left to the next layer's attention-side hc3 read (fast_moe, HIP:
    // STRATA_GLM_HC_CPU) - with the paced promotion copies, which go out after that read as they did after the wait
    bool hc_cpu_on = false, bg_after_hc = false;
    const unsigned int* hc_cpu_seq = nullptr;
    const void* hc_cpu_ans = nullptr;
    glmfast::LayerGraphs kda_graphs;
    std::vector<cudaEvent_t> pev;
    std::vector<const char*> pname;
    size_t pn = 0;
    std::map<std::string, double> pacc;
    uint64_t ptokens = 0;
    void mark(const char* nm) {
        if (pn >= pev.size()) {
            cudaEvent_t e = nullptr;
            cudaEventCreate(&e);
            pev.push_back(e);
            pname.push_back(nm);
        }
        cudaEventRecord(pev[pn], cs);
        pname[pn] = nm;
        ++pn;
    }
    uint64_t pseen = 0, pskip = 0;   // STRATA_GLM_PROF=<n>: the first n tokens (the cold tiers) are not counted
    void collect() {
        if (pseen++ < pskip) {
            pn = 0;
            return;
        }
        for (size_t i = 1; i < pn; ++i) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, pev[i - 1], pev[i]) == cudaSuccess) pacc[pname[i]] += ms;
        }
        if (pn > 0) ++ptokens;
        pn = 0;
    }
    size_t tab_key(int key) const { return (size_t) key; }
    size_t rtab_key(int key) const { return (size_t) n_keys + (size_t) key; }
    size_t spare_key(int il, int j) const { return 2 * (size_t) n_keys + (size_t) il * gf::kSpares + (size_t) j; }
    cudaEvent_t get_event() {
        if (!ev_free.empty()) {
            cudaEvent_t e = ev_free.back();
            ev_free.pop_back();
            return e;
        }
        cudaEvent_t e = nullptr;
        cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        return e;
    }
    bool route_ok(std::string& err) const {
        const int e = route_error_h ? *route_error_h : 0;
        if (e == 0) return true;
        const char* kind = (e % 4 == 1) ? "decode" : (e % 4 == 2) ? "predicted" : "lookahead";
        err = "glm router: layer " + std::to_string(e / 4) + " " + kind + " invalid expert ID or non-finite score";
        return false;
    }
};


}  // namespace strata::core
