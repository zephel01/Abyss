// abyss_gpu_tune: measures, on this GPU and without a model, the kernel choices the engine makes per GPU (glm_fast.cu's
// mv_tune) and prints which is fastest here - `abyss --gpu-tune` runs it, writes the result into the installed
// configs' env and prints a report to paste into a GitHub issue (the defaults table grows from those).
//
//   abyss_gpu_tune [--device N] [--json PATH] [--rounds N]
//
// Checks now: the device (name, architecture, compute units, VRAM, a device-to-device copy's bandwidth) and the Q6_K
// dense GEMV split (STRATA_GLM_MV_Q6K_SPLIT: 4 warps a row up to 4,096 blocks, or a warp a row) over the row counts
// the models' Q6_K projections have (1,024-32,768 rows of 4,096 / 5,120); and (a report only) the Q4_K / Q5_K / Q8_0
// row split at 1 / 2 / 4 warps a row.  Random weights: only the time is read.
#include "strata/kernels/glm_fast.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace gf = strata::kernels::glmf;

namespace {

constexpr int kQ6K = 14;                 // ggml's Q6_K
constexpr size_t kQ6KBlock = 210;        // 256 weights: ql[128] qh[64] scales[16] d (fp16)

bool ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "abyss_gpu_tune: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// the median of `rounds` timings of `calls` back-to-back GEMVs, in microseconds a call.  Each call reads the next
// copy of the weights in a pool far larger than any cache (an R9700's 64 MB Infinity Cache served one buffer read
// again at 1,050 GB/s - a model's decode reads its weights cold)
double time_mv(gf::MvJob j, const uint8_t* pool, size_t pool_bytes, size_t wbytes, int calls, int rounds,
               cudaStream_t s) {
    const size_t stride = (wbytes + 4095) & ~(size_t) 4095;
    const size_t copies = std::max<size_t>(1, pool_bytes / stride);
    size_t at = 0;
    auto one = [&] {
        j.w = pool + (at++ % copies) * stride;
        gf::mv(&j, 1, s);
    };
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < 8; ++i) one();   // (warm-up)
    std::vector<double> t;
    for (int r = 0; r < rounds; ++r) {
        cudaEventRecord(a, s);
        for (int i = 0; i < calls; ++i) one();
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        t.push_back(1000.0 * ms / calls);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    int dev = 0, rounds = 7;
    std::string json;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) dev = std::atoi(argv[++i]);
        else if (a == "--json" && i + 1 < argc) json = argv[++i];
        else if (a == "--rounds" && i + 1 < argc) rounds = std::max(3, std::atoi(argv[++i]));
        else {
            std::fprintf(stderr, "usage: abyss_gpu_tune [--device N] [--json PATH] [--rounds N]\n");
            return 2;
        }
    }
    if (!ok(cudaSetDevice(dev), "cudaSetDevice")) return 1;
    cudaDeviceProp prop{};
    if (!ok(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties")) return 1;
    char arch[64];
#if defined(STRATA_USE_HIP)
    std::snprintf(arch, sizeof arch, "%s", prop.gcnArchName);
    if (char* c = std::strchr(arch, ':')) *c = 0;
    const char* backend = "hip";
#else
    std::snprintf(arch, sizeof arch, "sm_%d%d", prop.major, prop.minor);
    const char* backend = "cuda";
#endif
    char pci[32];
    std::snprintf(pci, sizeof pci, "%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    const double vram_gb = (double) prop.totalGlobalMem / 1073741824.0;
    std::printf("device %d: %s (%s, %s, %d CUs/SMs, %.1f GB, PCI %s)\n", dev, prop.name, arch, backend,
                prop.multiProcessorCount, vram_gb, pci);
    cudaStream_t s;
    if (!ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream")) return 1;

    // 1. a device-to-device copy of 512 MB: its bandwidth (read + write), a rough check of the card's memory
    double copy_gbs = 0.0;
    {
        const size_t n = (size_t) 512 << 20;
        void *x = nullptr, *y = nullptr;
        if (ok(cudaMalloc(&x, n), "copy buffer") && ok(cudaMalloc(&y, n), "copy buffer")) {
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            cudaMemcpyAsync(y, x, n, cudaMemcpyDeviceToDevice, s);
            cudaEventRecord(a, s);
            for (int i = 0; i < 10; ++i) cudaMemcpyAsync(y, x, n, cudaMemcpyDeviceToDevice, s);
            cudaEventRecord(b, s);
            cudaEventSynchronize(b);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, a, b);
            copy_gbs = 2.0 * 10.0 * (double) n / (ms / 1000.0) / 1e9;
            cudaEventDestroy(a);
            cudaEventDestroy(b);
        }
        cudaFree(x);
        cudaFree(y);
        std::printf("memory: a device-to-device copy at %.0f GB/s (read + write)\n", copy_gbs);
    }

    // 2. the Q6_K GEMV split
    const bool chosen = gf::mv_q6k_split_choice();
    struct Shape { int n_out, n_in; };
    const Shape shapes[] = {{1024, 4096}, {2048, 5120}, {4096, 4096}, {5120, 8192}, {8192, 5120},
                            {16384, 4096}, {32768, 5120}};
    size_t wmax = 0;
    int imax = 0, omax = 0;
    for (const Shape& sh : shapes) {
        wmax = std::max(wmax, (size_t) sh.n_out * (sh.n_in / 256) * kQ6KBlock);
        imax = std::max(imax, sh.n_in);
        omax = std::max(omax, sh.n_out);
    }
    std::vector<uint8_t> hw(wmax);
    std::mt19937 rng(1234);
    for (auto& v : hw) v = (uint8_t) rng();
    for (size_t b = 0; b + kQ6KBlock <= wmax; b += kQ6KBlock) {   // a finite scale: d = 2^-10 (fp16 0x1400)
        hw[b + kQ6KBlock - 2] = 0x00;
        hw[b + kQ6KBlock - 1] = 0x14;
    }
    std::vector<float> hx(imax);
    for (auto& v : hx) v = (float) ((int) (rng() % 2001) - 1000) / 1000.0f;
    const size_t pool = std::max(wmax * 4, (size_t) 1 << 30);   // (1 GB, or 4 copies of the largest)
    void *w = nullptr, *xq = nullptr;
    float *x = nullptr, *y = nullptr;
    if (!ok(cudaMalloc(&w, pool), "weights") || !ok(cudaMalloc(&x, imax * sizeof(float)), "x") ||
        !ok(cudaMalloc(&xq, (size_t) imax / 32 * 36), "xq") || !ok(cudaMalloc(&y, omax * sizeof(float)), "y"))
        return 1;
    for (size_t o = 0; o + wmax <= pool; o += wmax) cudaMemcpy((uint8_t*) w + o, hw.data(), wmax, cudaMemcpyHostToDevice);
    cudaMemcpy(x, hx.data(), imax * sizeof(float), cudaMemcpyHostToDevice);
    gf::quantize_q8_1(x, xq, imax, s);
    cudaStreamSynchronize(s);
    std::printf("\nQ6_K dense GEMV, a call (us), the median of %d rounds:\n", rounds);
    std::printf("  %7s x %-6s %10s %10s\n", "rows", "cols", "4 warps", "1 warp");
    double tot[2] = {0.0, 0.0};
    std::string rows_json;
    for (const Shape& sh : shapes) {
        gf::MvJob j;
        j.xq = xq;
        j.y = y;
        j.type = kQ6K;
        j.n_in = sh.n_in;
        j.n_out = sh.n_out;
        const int calls = std::max(20, (int) (200000000.0 / ((double) sh.n_out * sh.n_in)));
        double t[2];
        for (int k = 0; k < 2; ++k) {
            gf::mv_force_q6k_split(k == 0 ? 1 : 0);
            t[k] = time_mv(j, (const uint8_t*) w, pool, (size_t) sh.n_out * (sh.n_in / 256) * kQ6KBlock, calls, rounds, s);
            tot[k] += t[k];
        }
        gf::mv_force_q6k_split(-1);
        const double gb = (double) sh.n_out * (sh.n_in / 256) * kQ6KBlock / 1e9;
        std::printf("  %7d x %-6d %10.1f %10.1f   (%.0f / %.0f GB/s)\n", sh.n_out, sh.n_in, t[0], t[1],
                    gb / (t[0] * 1e-6), gb / (t[1] * 1e-6));
        char row[128];
        std::snprintf(row, sizeof row, "%s{\"rows\": %d, \"cols\": %d, \"split4_us\": %.2f, \"split1_us\": %.2f}",
                      rows_json.empty() ? "" : ", ", sh.n_out, sh.n_in, t[0], t[1]);
        rows_json += row;
    }
    if (!ok(cudaGetLastError(), "the GEMVs")) return 1;
    // 3. (2026-11-06) the other dense types' row split - 1 / 2 / 4 warps a row for every launch (the engine's rule:
    // 4 under 256 blocks, else 1; STRATA_GLM_MV_SPLIT forces one) - a report for the per-SM-count table (48-SM Spark
    // vs 170-SM 5090): no env is written from it yet
    std::string split_json;
    {
        struct TypeInfo { int type; const char* name; int qk; size_t bytes; size_t dpos; };
        const TypeInfo types[] = {{12, "Q4_K", 256, 144, 0}, {13, "Q5_K", 256, 176, 0}, {8, "Q8_0", 32, 34, 0}};
        std::printf("\ndense GEMV row split (us a call, 1 / 2 / 4 warps a row; the engine: 4 under 256 blocks):\n");
        for (const TypeInfo& ti : types) {
            std::vector<uint8_t> tw(wmax);
            for (auto& v : tw) v = (uint8_t) rng();
            for (size_t b = 0; b + ti.bytes <= wmax; b += ti.bytes) {   // finite scales (2^-10), dmin too for K types
                tw[b + ti.dpos] = 0x00;
                tw[b + ti.dpos + 1] = 0x14;
                if (ti.qk == 256) { tw[b + 2] = 0x00; tw[b + 3] = 0x14; }
            }
            for (size_t o = 0; o + wmax <= pool; o += wmax) cudaMemcpy((uint8_t*) w + o, tw.data(), wmax, cudaMemcpyHostToDevice);
            double tt[3] = {0, 0, 0};
            for (const Shape& sh : shapes) {
                gf::MvJob j;
                j.xq = xq;
                j.y = y;
                j.type = ti.type;
                j.n_in = sh.n_in;
                j.n_out = sh.n_out;
                const size_t wb = (size_t) sh.n_out * (sh.n_in / ti.qk) * ti.bytes;
                if (wb > wmax) continue;
                const int calls = std::max(20, (int) (200000000.0 / ((double) sh.n_out * sh.n_in)));
                double t[3];
                const int sp[3] = {1, 2, 4};
                for (int k = 0; k < 3; ++k) {
                    gf::mv_force_split(sp[k]);
                    t[k] = time_mv(j, (const uint8_t*) w, pool, wb, calls, rounds, s);
                    tt[k] += t[k];
                }
                gf::mv_force_split(-1);
                std::printf("  %-5s %7d x %-6d %9.1f %9.1f %9.1f\n", ti.name, sh.n_out, sh.n_in, t[0], t[1], t[2]);
            }
            const int bi = tt[1] < tt[0] && tt[1] <= tt[2] ? 1 : tt[2] < tt[0] && tt[2] < tt[1] ? 2 : 0;
            std::printf("  %-5s total %21.1f %9.1f %9.1f   (best: %d warp%s a row)\n", ti.name, tt[0], tt[1], tt[2],
                        1 << bi, bi ? "s" : "");
            char row[160];
            std::snprintf(row, sizeof row, "%s\"%s\": {\"split1_us\": %.1f, \"split2_us\": %.1f, \"split4_us\": %.1f}",
                          split_json.empty() ? "" : ", ", ti.name, tt[0], tt[1], tt[2]);
            split_json += row;
        }
        if (!ok(cudaGetLastError(), "the split GEMVs")) return 1;
    }
    // a choice is changed only for a clear win (2% of the total) - else the table's stays
    const bool best = tot[0] < tot[1] * 0.98 ? true : tot[1] < tot[0] * 0.98 ? false : chosen;
    std::printf("  total %37.1f %10.1f\n", tot[0], tot[1]);
    std::printf("\nQ6_K split: the engine's choice here %s; measured best %s%s\n",
                chosen ? "4 warps a row" : "a warp a row", best ? "4 warps a row" : "a warp a row",
                best == chosen ? " (no change)"
                               : (best ? " -> STRATA_GLM_MV_Q6K_SPLIT=1" : " -> STRATA_GLM_MV_Q6K_SPLIT=0"));
    if (!json.empty()) {
        if (FILE* f = std::fopen(json.c_str(), "w")) {
            std::fprintf(f,
                         "{\"device\": %d, \"name\": \"%s\", \"arch\": \"%s\", \"backend\": \"%s\", \"cus\": %d, "
                         "\"vram_gb\": %.1f, \"pci\": \"%s\", \"copy_gbs\": %.0f,\n \"q6k\": {\"choice_split4\": %s, "
                         "\"best_split4\": %s, \"total_split4_us\": %.1f, \"total_split1_us\": %.1f, \"rows\": [%s]},\n"
                         " \"split\": {%s},\n"
                         " \"env\": {%s}}\n",
                         dev, prop.name, arch, backend, prop.multiProcessorCount, vram_gb, pci, copy_gbs,
                         chosen ? "true" : "false", best ? "true" : "false", tot[0], tot[1], rows_json.c_str(), split_json.c_str(),
                         best == chosen ? "" : (best ? "\"STRATA_GLM_MV_Q6K_SPLIT\": \"1\"" : "\"STRATA_GLM_MV_Q6K_SPLIT\": \"0\""));
            std::fclose(f);
        }
    }
    cudaFree(w);
    cudaFree(x);
    cudaFree(xq);
    cudaFree(y);
    cudaStreamDestroy(s);
    return 0;
}
