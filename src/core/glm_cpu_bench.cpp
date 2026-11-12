// src/core/glm_cpu_bench.cpp - how fast does this CPU run one GLM routed expert (ggml-cpu's dots)?
//
//   glm_cpu_bench <pack_dir> <layer> [threads...]
//
// Reads the first 64 experts of the layer straight from its shard (native_experts.txt gives the offsets) - ~450 MB,
// far past the L3, so every expert comes from DRAM like a real RAM-tier miss - then times gate/up (2 x n_ff rows of
// n_embd) and down (n_embd rows of n_ff) on each thread count given, rows split evenly over a persistent spinning
// pool (no thread start per expert) - the shape of a CPU miss path beside the PCIe fetches.
#include "strata/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

// nt - 1 spinning helpers + the caller; run(fn) calls fn(t, nt) once on every thread and returns when all are done
class Spin {
public:
    explicit Spin(int nt) : nt_(nt) {
        for (int t = 1; t < nt; ++t)
            th_.emplace_back([this, t] {
                uint64_t seen = 0;
                for (;;) {
                    uint64_t g;
                    while ((g = gen_.load(std::memory_order_acquire)) == seen) {
                    }
                    if (quit_.load(std::memory_order_relaxed)) return;
                    seen = g;
                    (*fn_)(t, nt_);
                    done_.fetch_add(1, std::memory_order_acq_rel);
                }
            });
    }
    ~Spin() {
        quit_.store(true);
        gen_.fetch_add(1, std::memory_order_acq_rel);
        for (auto& t : th_) t.join();
    }
    void run(const std::function<void(int, int)>& fn) {
        fn_ = &fn;
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_acq_rel);
        fn(0, nt_);
        while (done_.load(std::memory_order_acquire) < nt_ - 1) {
        }
    }

private:
    int nt_;
    std::vector<std::thread> th_;
    const std::function<void(int, int)>* fn_ = nullptr;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> quit_{false};
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <pack_dir> <layer> [threads...]\n", argv[0]);
        return 2;
    }
    const std::string pack = argv[1];
    const int layer = std::atoi(argv[2]);
    std::ifstream ne(pack + "/native_experts.txt");
    std::string line, shard0;
    int gu = -1, dt = -1;
    uint64_t go = 0, uo = 0, dof = 0;
    std::string shard;
    while (std::getline(ne, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const std::string key = "absolute offsets in ";
            const size_t a = line.find(key);
            if (a != std::string::npos) {
                const size_t b = line.find(',', a);
                shard0 = line.substr(a + key.size(), b == std::string::npos ? std::string::npos : b - a - key.size());
            }
            continue;
        }
        std::istringstream ss(line);
        int l = -1;
        uint64_t off = 0, blob = 0;
        ss >> l >> gu >> dt >> off >> blob >> go >> uo >> dof;
        std::string w;
        ss >> w;
        if (l == layer) {
            shard = w.empty() ? shard0 : w;   // a v3 row without a shard lies in shard 1 (a single-file model)
            break;
        }
    }
    if (shard.empty()) {
        std::fprintf(stderr, "layer %d not found (or a straddle row)\n", layer);
        return 1;
    }
    // GLM_BENCH_EMBD / GLM_BENCH_FF: another geometry (default DeepSeek V4's 4096 / 2048); GLM_BENCH_NEXP: how many
    // experts are cycled (64 = from DRAM; 1-2 = from the CPU's caches, what a warmed expert would cost)
    const int n_embd = getenv("GLM_BENCH_EMBD") ? std::atoi(getenv("GLM_BENCH_EMBD")) : 4096;
    const int n_ff = getenv("GLM_BENCH_FF") ? std::atoi(getenv("GLM_BENCH_FF")) : 2048;
    const int n_exp = getenv("GLM_BENCH_NEXP") ? std::max(1, std::atoi(getenv("GLM_BENCH_NEXP"))) : 64;
    strata::kernels::cpu::NativeFmt f;
    std::string err;
    if (!strata::kernels::cpu::native_fmt(gu, dt, n_embd, n_ff, f, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const std::string path = pack + "/../" + shard;
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "open %s failed\n", path.c_str());
        return 1;
    }
    // expert e's gate rows sit at go + e * gub (the GGUF's [expert][row] order), up and down likewise
    const size_t gub = f.gu_row * (size_t) n_ff, dnb = f.d_row * (size_t) n_embd, eb = 2 * gub + dnb;
    std::vector<uint8_t> blobs(eb * n_exp);
    for (int e = 0; e < n_exp; ++e) {
        uint8_t* b = blobs.data() + eb * e;
        if (pread(fd, b, gub, (off_t) (go + e * gub)) != (ssize_t) gub ||
            pread(fd, b + gub, gub, (off_t) (uo + e * gub)) != (ssize_t) gub ||
            pread(fd, b + 2 * gub, dnb, (off_t) (dof + e * dnb)) != (ssize_t) dnb) {
            std::fprintf(stderr, "short read\n");
            return 1;
        }
    }
    std::vector<float> x(n_embd);
    for (int i = 0; i < n_embd; ++i) x[i] = std::sin(0.37f * i) * 0.5f;
    std::vector<uint8_t> act(strata::kernels::cpu::kNativeActBytes), hq(strata::kernels::cpu::kNativeHBytes);
    std::vector<float> ff(n_ff), dn(n_embd);
    strata::kernels::cpu::native_quant_act(f, x.data(), act.data());
    std::vector<int> counts;
    for (int i = 3; i < argc; ++i) counts.push_back(std::atoi(argv[i]));
    if (counts.empty()) counts = {1, 6, 12};
    std::printf("layer %d gu_type %d d_type %d: expert %.2f MB (gate/up %.2f, down %.2f), %d experts cycled from DRAM\n",
                layer, gu, dt, (double) eb / 1e6, 2.0 * gub / 1e6, dnb / 1e6, n_exp);
    const int warm = getenv("GLM_BENCH_WARM") ? std::atoi(getenv("GLM_BENCH_WARM")) : 0;
    std::atomic<uint64_t> sink{0};
    for (int nt : counts) {
        Spin pool(nt);
        const int reps = 4 * n_exp;
        double t_gu = 0, t_dn = 0, best = 1e9;
        for (int rep = 0; rep < reps; ++rep) {
            const uint8_t* b = blobs.data() + eb * (size_t) (rep % n_exp);
            // GLM_BENCH_WARM=1: before the timed call, each thread reads (into its caches) the rows it will compute;
            // 2: the rows of the thread n/2 away (another CCX on a 2-CCX part) - what a warm by the wrong thread gives
            if (warm != 0) {
                pool.run([&](int t, int n) {
                    const int tw = warm == 2 ? (t + n / 2) % n : warm == 3 ? (t * 5 + 3) % n : t;
                    const size_t g0 = f.gu_row * (size_t) (n_ff * tw / n), g1 = f.gu_row * (size_t) (n_ff * (tw + 1) / n);
                    const size_t d0 = f.d_row * (size_t) (n_embd * tw / n), d1 = f.d_row * (size_t) (n_embd * (tw + 1) / n);
                    uint64_t acc = 0;
                    for (size_t o = g0; o < g1; o += 64) acc += *(const volatile uint8_t*) (b + o) + *(const volatile uint8_t*) (b + gub + o);
                    for (size_t o = d0; o < d1; o += 64) acc += *(const volatile uint8_t*) (b + 2 * gub + o);
                    sink.fetch_add(acc, std::memory_order_relaxed);
                });
            }
            const auto t0 = std::chrono::steady_clock::now();
            pool.run([&](int t, int n) {
                const int r0 = n_ff * t / n, r1 = n_ff * (t + 1) / n;
                const void* a[1] = {act.data()};
                float* o[1] = {ff.data()};
                strata::kernels::cpu::native_gu_rows_split(f, b, b + gub, a, 1, o, r0, r1, 10.0f);
            });
            const auto t1 = std::chrono::steady_clock::now();
            strata::kernels::cpu::native_quant_h(f, ff.data(), hq.data());
            pool.run([&](int t, int n) {
                const int r0 = n_embd * t / n, r1 = n_embd * (t + 1) / n;
                const void* h[1] = {hq.data()};
                float* o[1] = {dn.data()};
                strata::kernels::cpu::native_down_rows_split(f, b + 2 * gub, h, 1, o, r0, r1);
            });
            const auto t2 = std::chrono::steady_clock::now();
            if (rep < n_exp) continue;   // the first lap warms the pool and the clocks
            const double g = std::chrono::duration<double, std::micro>(t1 - t0).count();
            const double d = std::chrono::duration<double, std::micro>(t2 - t1).count();
            t_gu += g;
            t_dn += d;
            best = std::min(best, g + d);
        }
        const double n = reps - n_exp;
        std::printf("threads %2d: gate/up %7.1f us (%.2f GB/s)  down %7.1f us (%.2f GB/s)  expert %7.1f us mean, %7.1f best\n",
                    nt, t_gu / n, 2.0 * gub / (t_gu / n) / 1e3, t_dn / n, dnb / (t_dn / n) / 1e3, (t_gu + t_dn) / n, best);
    }
    return 0;
}
