// tests/glm_uma_direct_test.cu - STRATA_GLM_UMA_DIRECT (MoeDev::ram_direct): the decode's expert step with its RAM-tier
// experts read IN PLACE from pinned host memory, against the same experts in device memory and against the copy path
// (the route plans a fetch into a scratch slot, moe_wait_fetch copies the blob, the kernels read the copy).
//
// The real kernels (moe_route -> moe_wait_fetch -> moe_gate_up -> moe_down) on synthetic Qwen3.8-shaped experts
// (n_embd 2560, n_ff 640, k 10, gate/up IQ3_XXS, down IQ4_NL - the 16G64G pack's types), no model:
//   A. VRAM:   every expert in the table (tab), cudaMalloc'd
//   B. DIRECT: none in tab, all in the RAM-tier table (rtab, cudaHostAlloc'd), ram_direct = 1 - read in place
//   C. COPY:   the same rtab, ram_direct = 0 - the route plans fetches into scratch slots and moe_wait_fetch copies
// The layer outputs must be BITWISE equal (the same kernels over the same bytes), and B's plan must name the host
// blobs (no fetch source, no request masks).
//
//   glm_uma_direct_test            the parity checks (ctest; 64 experts)
//   glm_uma_direct_test --bench    + timing: 512 experts (1.1 GB, past every cache), 400 routes of random logits,
//                                  ms per layer step for A / B / C (an integrated GPU: B should match A)
#include "strata/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace gf = strata::kernels::glmf;

namespace {
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm_uma_direct_test: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "glm_uma_direct_test: FAILED - %s\n", what.c_str());
        std::exit(1);
    }
}
template <class T> T* dev(size_t n) {
    void* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T) + 256), "cudaMalloc");
    check(cudaMemset(p, 0, n * sizeof(T) + 256), "cudaMemset");
    return (T*) p;
}
// host-mapped, and its device pointer
template <class T> T* mapped(size_t n, T** dptr) {
    void* p = nullptr;
    check(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
    std::memset(p, 0, n * sizeof(T));
    check(cudaHostGetDevicePointer((void**) dptr, p, 0), "cudaHostGetDevicePointer");
    return (T*) p;
}

constexpr int kIQ3XXS = 18, kIQ4NL = 20;   // ggml types
constexpr size_t kIQ3XXSBlock = 98, kIQ4NLBlock = 18;   // bytes (256 / 32 elements; fp16 d first)

// n bytes of one weight matrix: random codes, every block's fp16 scale a small positive value (finite sums)
void fill_matrix(uint8_t* p, size_t bytes, size_t block, std::mt19937& rng) {
    for (size_t i = 0; i < bytes; ++i) p[i] = (uint8_t) rng();
    std::uniform_int_distribution<int> sc(0x1C00, 0x2400);   // fp16 ~0.004 .. 0.016
    for (size_t b = 0; b + block <= bytes; b += block) {
        const uint16_t h = (uint16_t) sc(rng);
        std::memcpy(p + b, &h, 2);
    }
}

struct Run {
    gf::MoeDev d;
    unsigned long long* tab = nullptr;   // device: [tab n][rtab n][spares kSpares]
    int n = 0;
};

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const int E = 2560, F = 640, K = 10, NE = bench ? 512 : 64;
    const size_t gu_row = (size_t) E / 256 * kIQ3XXSBlock, d_row = (size_t) F / 32 * kIQ4NLBlock;
    const size_t gu_bytes = 2 * gu_row * F, down_off = gu_bytes, blob = gu_bytes + d_row * E;
    const size_t stride = (blob + 255) & ~(size_t) 255;
    const int k_slots = std::max(8, K);
    cudaStream_t s = nullptr;
    check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    int devn = 0, integrated = 0;
    cudaGetDevice(&devn);
    cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, devn);
    std::printf("glm_uma_direct_test: %d experts of %.2f MB (IQ3_XXS gate/up, IQ4_NL down, %d x %d), k %d, %s GPU\n",
                NE, blob / 1e6, E, F, K, integrated ? "integrated" : "discrete");

    // the experts: one host image, copied to device memory (A) and to pinned host memory (B, C)
    std::mt19937 rng(20261105);
    std::vector<uint8_t> img(stride * NE, 0);
    for (int e = 0; e < NE; ++e) {
        uint8_t* b = img.data() + stride * e;
        fill_matrix(b, gu_bytes, kIQ3XXSBlock, rng);
        fill_matrix(b + down_off, d_row * E, kIQ4NLBlock, rng);
    }
    uint8_t* blobs_d = dev<uint8_t>(img.size());
    check(cudaMemcpy(blobs_d, img.data(), img.size(), cudaMemcpyHostToDevice), "blobs H2D");
    uint8_t* blobs_h = nullptr;
    check(cudaHostAlloc((void**) &blobs_h, img.size(), cudaHostAllocPortable), "pinned blobs");
    std::memcpy(blobs_h, img.data(), img.size());
    // (the engine's rtab holds the tier's host pointers as they are - UVA; cudaHostGetDevicePointer agrees)
    unsigned long long blobs_hd = 0;
    check(cudaHostGetDevicePointer((void**) &blobs_hd, blobs_h, 0), "blobs device pointer");

    // the shared per-route buffers
    uint8_t* scratch = dev<uint8_t>(stride * k_slots);
    std::vector<unsigned long long> sp((size_t) k_slots);
    for (int i = 0; i < k_slots; ++i) sp[(size_t) i] = (unsigned long long) (scratch + stride * i);
    auto* ring_h = (gf::MoeRequest*) nullptr;
    gf::MoeRequest* ring_d = nullptr;
    ring_h = mapped<gf::MoeRequest>(gf::kRingSize, &ring_d);
    gf::MoeResponse* resp_d = nullptr;
    mapped<gf::MoeResponse>(gf::kCpuAnswers, &resp_d);
    float* x = dev<float>(E);
    uint8_t* xq = dev<uint8_t>((size_t) E / 32 * 36);
    uint8_t* hq = dev<uint8_t>((size_t) k_slots * F / 32 * 36);
    float* out = dev<float>(E);
    const int NT = bench ? 400 : 24;   // routes (tokens)
    float* logits = dev<float>((size_t) NT * NE);
    {
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> h((size_t) NT * NE), xv(E);
        for (auto& v : h) v = 3.0f * nd(rng);
        for (auto& v : xv) v = nd(rng);
        check(cudaMemcpy(logits, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "logits");
        check(cudaMemcpy(x, xv.data(), xv.size() * 4, cudaMemcpyHostToDevice), "x");
    }
    gf::quantize_q8_1(x, xq, E, s);

    auto make = [&](int mode) {   // 0 VRAM, 1 DIRECT, 2 COPY
        Run r;
        r.n = NE;
        r.tab = dev<unsigned long long>((size_t) 2 * NE + gf::kSpares);
        std::vector<unsigned long long> t((size_t) 2 * NE + gf::kSpares, 0ull);
        for (int e = 0; e < NE; ++e) {
            if (mode == 0) t[(size_t) e] = (unsigned long long) (blobs_d + stride * e);
            else t[(size_t) NE + e] = blobs_hd + stride * e;
        }
        check(cudaMemcpy(r.tab, t.data(), t.size() * 8, cudaMemcpyHostToDevice), "tab");
        gf::MoeDev& d = r.d;
        d.tab = r.tab;
        d.n_keys = NE;
        d.scratch = dev<unsigned long long>((size_t) k_slots);
        check(cudaMemcpy(d.scratch, sp.data(), sp.size() * 8, cudaMemcpyHostToDevice), "scratch");
        d.plan_ptr = dev<unsigned long long>((size_t) k_slots);
        d.plan_w = dev<float>((size_t) k_slots);
        d.plan_id = dev<int>((size_t) k_slots);
        d.fetch_src = dev<unsigned long long>((size_t) k_slots);
        d.pf_src = dev<unsigned long long>(gf::kMaxK);
        d.pf_dst = dev<unsigned long long>(gf::kMaxK);
        d.pf_n = dev<int>(1);
        d.seq = dev<unsigned int>(16);
        d.wait_seq = dev<unsigned int>(16);
        d.ring = ring_d;
        d.resp = resp_d;
        d.cpu_part = dev<float>(E);
        d.cpu_flag = dev<int>(16);
        d.ram_direct = mode == 1 ? 1 : 0;
        return r;
    };
    Run run[3] = {make(0), make(1), make(2)};
    const char* name[3] = {"VRAM  ", "DIRECT", "COPY  "};
    // one layer step: route (no tail skip, no CPU lane, no prefetch), the fetch, the expert kernels
    auto step = [&](Run& r, int t) {
        gf::moe_route(logits + (size_t) t * NE, nullptr, NE, K, 1.0f, true, 0, x, E, r.d, nullptr, nullptr, 0.0f, 0, nullptr, s,
                      nullptr, nullptr, 0, nullptr, nullptr, 0, /*skip_from*/ 64, 0ull, /*promote_min*/ 0, 0, /*gating*/ 2);
        gf::moe_wait_fetch(r.d, E, K, blob, s);
        gf::moe_gate_up(kIQ3XXS, r.d, K, E, F, 1e30f, xq, hq, nullptr, 0, nullptr, 0, nullptr, s);
        gf::moe_down(kIQ4NL, r.d, K, E, F, down_off, hq, nullptr, out, s);
    };

    // ---- parity: every route, the three outputs bit for bit; DIRECT's plan names the host blobs
    std::vector<float> o[3];
    int fetched_copy = 0;
    for (int t = 0; t < NT; ++t) {
        for (int m = 0; m < 3; ++m) {
            step(run[m], t);
            o[m].assign(E, 0.f);
            check(cudaMemcpyAsync(o[m].data(), out, E * 4, cudaMemcpyDeviceToHost, s), "out D2H");
            check(cudaStreamSynchronize(s), "step");
            std::vector<unsigned long long> pp((size_t) K), fs((size_t) K);
            check(cudaMemcpy(pp.data(), run[m].d.plan_ptr, K * 8, cudaMemcpyDeviceToHost), "plan");
            check(cudaMemcpy(fs.data(), run[m].d.fetch_src, K * 8, cudaMemcpyDeviceToHost), "fetch_src");
            unsigned seq = 0;
            check(cudaMemcpy(&seq, run[m].d.seq, 4, cudaMemcpyDeviceToHost), "seq");
            const gf::MoeRequest& rq = ring_h[seq % gf::kRingSize];
            for (int i = 0; i < K; ++i) {
                const unsigned long long p = pp[(size_t) i];
                if (m == 0) require(p >= (unsigned long long) blobs_d && p < (unsigned long long) (blobs_d + stride * NE), "VRAM plan");
                if (m == 1) {
                    require(p >= blobs_hd && p < blobs_hd + stride * NE && (p - blobs_hd) % stride == 0,
                            "DIRECT plan entry " + std::to_string(i) + " is not a RAM-tier blob");
                    require(fs[(size_t) i] == 0ull, "DIRECT plan has a fetch source");
                }
                if (m == 2) {
                    require(std::find(sp.begin(), sp.end(), p) != sp.end(), "COPY plan is not a scratch slot");
                    fetched_copy += fs[(size_t) i] != 0ull;
                }
            }
            if (m == 1) require(rq.fetch_mask == 0u && rq.miss_mask == 0u && rq.promo_mask == 0u && rq.cpu_mask == 0u &&
                                rq.skip_mask == 0u, "DIRECT request masks are not empty");
        }
        for (int m = 1; m < 3; ++m)
            for (int r = 0; r < E; ++r) {
                uint32_t a, b;
                std::memcpy(&a, &o[0][(size_t) r], 4);
                std::memcpy(&b, &o[m][(size_t) r], 4);
                require(a == b, std::string(name[m]) + " differs from VRAM at route " + std::to_string(t) + " row " +
                                    std::to_string(r) + ": " + std::to_string(o[m][(size_t) r]) + " vs " +
                                    std::to_string(o[0][(size_t) r]));
            }
        require(std::isfinite(o[0][0]) && std::isfinite(o[0][E - 1]), "non-finite outputs");
    }
    require(fetched_copy == NT * K, "COPY did not fetch every expert");
    std::printf("parity: %d routes x %d experts, DIRECT and COPY bitwise equal to VRAM; DIRECT plans read the host "
                "blobs in place (no fetch, empty request masks)\n", NT, K);
    if (!bench) {
        std::printf("glm_uma_direct_test: OK\n");
        return 0;
    }

    // ---- timing: the layer step per mode, NT distinct routes (a route's 10 blobs are 21.8 MB; 512 experts 1.1 GB)
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int rep = 0; rep < 2; ++rep)
        for (int m = 0; m < 3; ++m) {
            for (int t = 0; t < 8; ++t) step(run[m], t);   // warm
            cudaEventRecord(e0, s);
            for (int t = 0; t < NT; ++t) step(run[m], t);
            cudaEventRecord(e1, s);
            check(cudaEventSynchronize(e1), "timing");
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            const double per = ms / NT;
            std::printf("  %s %.1f us a layer step (%.0f GB/s of expert bytes)%s\n", name[m], per * 1e3,
                        (double) K * blob / (per * 1e-3) / 1e9, rep == 0 ? "" : " (2nd)");
        }
    // the expert kernels alone (no route / fetch): their read speed from each memory
    for (int m = 0; m < 2; ++m) {
        step(run[m], 0);
        cudaEventRecord(e0, s);
        for (int t = 0; t < NT; ++t) {
            gf::moe_gate_up(kIQ3XXS, run[m].d, K, E, F, 1e30f, xq, hq, nullptr, 0, nullptr, 0, nullptr, s);
            gf::moe_down(kIQ4NL, run[m].d, K, E, F, down_off, hq, nullptr, out, s);
        }
        cudaEventRecord(e1, s);
        check(cudaEventSynchronize(e1), "timing");
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        std::printf("  %s gate_up + down alone, one route repeated: %.1f us\n", name[m], ms / NT * 1e3);
    }
    std::printf("glm_uma_direct_test: OK\n");
    return 0;
}
