// src/kernels/qsa_topk_parity.cpp - qwen4exp's QSA block top-k: the register kernel's digit search by 256 threads
// (qsa_block_topk_variant par = true) against thread 0's walk (par = false, the v1.3.130 kernel) and the reference
// kernel (qsa_block_topk_ref).  GPU, synthetic, no model.  Contexts from just past the dense window to ~130K cells
// (multiples of the block and not), score patterns: random, few distinct values (many ties), all equal, +-0, huge, the
// tail block's +1e9; one query and a batch of queries.  The selected ids must be identical; then a token's top-k is
// timed both ways.  Also the token head's argmax over 128 blocks (gf::argmax_mb) against argmax, and the indexer
// query rope a thread a pair (rope_neox_apply_pairs) against a thread a row.  Usage: qsa_topk_parity [reps=200]
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/rope.hpp"

#include <cmath>
#include <cstring>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
    const k::QsaShapes sh = k::qsa_real_shapes();
    const int64_t R = sh.idx_block;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, sh);
    const int64_t max_ctx = 131072 + 7;
    const int64_t max_blocks = max_ctx / R + 2;
    std::mt19937 rng(1234);
    float* d_sc = nullptr;
    int32_t *d_st = nullptr, *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    const int max_nq = 8;
    ck(cudaMalloc(&d_sc, (size_t) max_nq * max_blocks * 4), "malloc");
    ck(cudaMalloc(&d_st, (size_t) max_nq * k::kStepCount * 4), "malloc");
    ck(cudaMalloc(&d_a, (size_t) max_nq * cap * 4), "malloc");
    ck(cudaMalloc(&d_b, (size_t) max_nq * cap * 4), "malloc");
    ck(cudaMalloc(&d_c, (size_t) max_nq * cap * 4), "malloc");
    const int64_t ctxs[] = {2052, 2053, 2055, 2056, 2100, 3000, 4097, 4608, 4611, 8192, 8195, 16384, 32768, 32771,
                            65536, 100003, 131072, 131075};
    int64_t cases = 0, bad = 0;
    for (const int64_t ctx : ctxs) {
        for (int pat = 0; pat < 7; ++pat) {
            for (const int nq : {1, 3, 8}) {
                std::vector<int32_t> st((size_t) nq * k::kStepCount);
                for (int i = 0; i < nq; ++i) k::qsa_step_fill(st.data() + i * k::kStepCount, ctx - nq + i, sh);
                std::vector<float> sc((size_t) nq * max_blocks, 0.0f);
                std::uniform_real_distribution<float> u(0.0f, 50.0f);
                std::uniform_int_distribution<int> few(0, 3);
                for (size_t i = 0; i < sc.size(); ++i) {
                    float v;
                    switch (pat) {
                    case 0: v = u(rng); break;                                  // random
                    case 1: v = (float) few(rng); break;                        // 4 values: ties everywhere
                    case 2: v = 7.0f; break;                                    // all equal
                    case 3: v = (rng() & 1) ? 0.0f : -0.0f; break;              // +-0
                    case 4: v = (rng() % 50 == 0) ? 1e30f : u(rng) * 1e-30f; break;   // huge / tiny
                    case 5: v = (float) (rng() % 1000) * 0.125f; break;         // exact, many ties
                    default: v = u(rng) + ((rng() % 7 == 0) ? 1e9f : 0.0f);     // many "tails"
                    }
                    sc[i] = v;
                }
                for (int i = 0; i < nq; ++i) {   // the tail block's +1e9 as block_scores writes it
                    const int64_t nkv = st[(size_t) i * k::kStepCount + k::kStepNKv];
                    const int64_t nb = st[(size_t) i * k::kStepCount + k::kStepNBid];
                    if (nkv % R != 0) sc[(size_t) i * max_blocks + nb] += 1e9f;
                }
                ck(cudaMemcpy(d_sc, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "up");
                ck(cudaMemcpy(d_st, st.data(), st.size() * 4, cudaMemcpyHostToDevice), "up");
                ck(cudaMemset(d_a, 0xff, (size_t) nq * cap * 4), "set");
                ck(cudaMemset(d_b, 0xfe, (size_t) nq * cap * 4), "set");
                ck(cudaMemset(d_c, 0xfd, (size_t) nq * cap * 4), "set");
                k::qsa_block_topk_variant(d_sc, d_st, nq, max_blocks, cap, sh, d_a, nullptr, true);
                k::qsa_block_topk_variant(d_sc, d_st, nq, max_blocks, cap, sh, d_b, nullptr, false);
                k::qsa_block_topk_ref(d_sc, d_st, nq, max_blocks, cap, sh, d_c, nullptr);
                ck(cudaDeviceSynchronize(), "run");
                std::vector<int32_t> a((size_t) nq * cap), b(a.size()), c(a.size());
                ck(cudaMemcpy(a.data(), d_a, a.size() * 4, cudaMemcpyDeviceToHost), "down");
                ck(cudaMemcpy(b.data(), d_b, b.size() * 4, cudaMemcpyDeviceToHost), "down");
                ck(cudaMemcpy(c.data(), d_c, c.size() * 4, cudaMemcpyDeviceToHost), "down");
                for (int i = 0; i < nq; ++i) {
                    const int64_t w = st[(size_t) i * k::kStepCount + k::kStepWidth];
                    bool same = true;
                    for (int64_t j = 0; j < w; ++j) {
                        const size_t x = (size_t) i * cap + j;
                        same = same && a[x] == b[x] && a[x] == c[x];
                    }
                    ++cases;
                    if (!same) {
                        ++bad;
                        if (bad <= 10) std::printf("DIFF ctx %lld pattern %d nq %d query %d\n", (long long) ctx, pat, nq, i);
                    }
                }
            }
        }
    }
    std::printf("qsa top-k: %lld / %lld selections identical (par, walk, reference)\n", (long long) (cases - bad),
                (long long) cases);
    // a token's top-k (one query) at a few contexts, both digit searches
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (const int64_t ctx : {4608LL, 32768LL, 131072LL}) {
        std::vector<int32_t> st(k::kStepCount);
        k::qsa_step_fill(st.data(), ctx - 1, sh);
        std::vector<float> sc((size_t) max_blocks);
        std::uniform_real_distribution<float> u(0.0f, 50.0f);
        for (float& v : sc) v = u(rng);
        ck(cudaMemcpy(d_sc, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "up");
        ck(cudaMemcpy(d_st, st.data(), st.size() * 4, cudaMemcpyHostToDevice), "up");
        float ms[2];
        for (int v = 0; v < 2; ++v) {
            k::qsa_block_topk_variant(d_sc, d_st, 1, max_blocks, cap, sh, d_a, nullptr, v == 0);
            cudaEventRecord(e0);
            for (int r = 0; r < reps; ++r) k::qsa_block_topk_variant(d_sc, d_st, 1, max_blocks, cap, sh, d_a, nullptr, v == 0);
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "time");
            cudaEventElapsedTime(&ms[v], e0, e1);
        }
        std::printf("ctx %lld: one query's top-k %.2f us (walk) -> %.2f us (par)\n", (long long) ctx,
                    1000.0 * ms[1] / reps, 1000.0 * ms[0] / reps);
    }
    // ---- argmax_mb vs argmax: random, ties (the lowest index), -inf, +-0, NaN, the max in the tail / first / last
    {
        namespace gf = strata::kernels::glmf;
        const int n = 248320;
        float* dx = nullptr;
        int *d1 = nullptr, *d2 = nullptr;
        void* scr = nullptr;
        ck(cudaMalloc(&dx, (size_t) (n + 8) * 4), "malloc");
        ck(cudaMalloc(&d1, 4), "malloc");
        ck(cudaMalloc(&d2, 4), "malloc");
        ck(cudaMalloc(&scr, gf::argmax_mb_scratch_bytes()), "malloc");
        ck(cudaMemset(scr, 0, gf::argmax_mb_scratch_bytes()), "set");
        int am_bad = 0, am_n = 0;
        for (int pat = 0; pat < 12; ++pat)
            for (int rep = 0; rep < 20; ++rep) {
                const int nn = pat == 11 ? n - 3 : n;   // (a vocabulary not a multiple of 4)
                std::vector<float> x((size_t) nn);
                std::uniform_real_distribution<float> u(-20.0f, 20.0f);
                for (float& v : x) v = u(rng);
                const int at = (int) (rng() % (uint32_t) nn);
                switch (pat) {
                case 1: for (float& v : x) v = std::floor(v / 8.0f); break;            // a handful of values: ties
                case 2: for (float& v : x) v = 3.0f; break;                            // all equal: index 0
                case 3: for (float& v : x) v = (rng() & 1) ? 0.0f : -0.0f; break;      // +-0
                case 4: for (float& v : x) v = -INFINITY; break;                       // all -inf
                case 5: x[(size_t) at] = NAN; x[(size_t) (at / 2)] = 100.0f; break;    // NaN never taken
                case 6: x[0] = 1e30f; break;
                case 7: x[(size_t) nn - 1] = 1e30f; break;
                case 8: x[(size_t) at] = 50.0f; x[(size_t) (rng() % (uint32_t) nn)] = 50.0f; break;   // two maxima
                case 9: for (size_t i = 0; i < x.size(); i += 7) x[i] = NAN; break;
                case 10: for (float& v : x) v = (float) (rng() % 3); break;
                default: break;
                }
                ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "up");
                gf::argmax(dx, nn, d1, nullptr);
                gf::argmax_mb(dx, nn, d2, scr, nullptr);
                int a = -1, b = -2;
                ck(cudaMemcpy(&a, d1, 4, cudaMemcpyDeviceToHost), "down");
                ck(cudaMemcpy(&b, d2, 4, cudaMemcpyDeviceToHost), "down");
                ++am_n;
                if (a != b) {
                    ++am_bad;
                    if (am_bad <= 5) std::printf("argmax DIFF pattern %d: %d vs %d\n", pat, a, b);
                }
            }
        float ms[2];
        for (int v = 0; v < 2; ++v) {
            cudaEventRecord(e0);
            for (int r = 0; r < reps; ++r) {
                if (v == 0) gf::argmax(dx, n, d1, nullptr);
                else gf::argmax_mb(dx, n, d2, scr, nullptr);
            }
            cudaEventRecord(e1);
            ck(cudaEventSynchronize(e1), "time");
            cudaEventElapsedTime(&ms[v], e0, e1);
        }
        std::printf("argmax_mb: %d / %d indices identical; %d entries %.2f us (one block) -> %.2f us (128 blocks)\n",
                    am_n - am_bad, am_n, n, 1000.0 * ms[0] / reps, 1000.0 * ms[1] / reps);
        bad += am_bad;
    }
    // ---- rope a thread a pair vs a thread a row (the indexer's queries: in place), several shapes
    {
        int rp_bad = 0, rp_n = 0;
        for (const int hd : {128, 256})
            for (const int nrot : {64, 128, hd})
                for (const int rows : {1, 7, 32, 64}) {
                    if (nrot > hd) continue;
                    const int max_pos = 70000;
                    std::vector<float> x((size_t) rows * hd), cs((size_t) max_pos * (nrot / 2)), sn(cs.size());
                    std::normal_distribution<float> nd(0.0f, 2.0f);
                    for (float& v : x) v = nd(rng);
                    for (size_t i = 0; i < cs.size(); ++i) { cs[i] = std::cos(0.001f * (float) i); sn[i] = std::sin(0.001f * (float) i); }
                    std::vector<int> pos((size_t) rows);
                    for (int& p : pos) p = (int) (rng() % (uint32_t) max_pos);
                    float *dxa, *dxb, *dc, *ds;
                    int* dp;
                    ck(cudaMalloc(&dxa, x.size() * 4), "m"); ck(cudaMalloc(&dxb, x.size() * 4), "m");
                    ck(cudaMalloc(&dc, cs.size() * 4), "m"); ck(cudaMalloc(&ds, sn.size() * 4), "m");
                    ck(cudaMalloc(&dp, pos.size() * 4), "m");
                    ck(cudaMemcpy(dxa, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "up");
                    ck(cudaMemcpy(dxb, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "up");
                    ck(cudaMemcpy(dc, cs.data(), cs.size() * 4, cudaMemcpyHostToDevice), "up");
                    ck(cudaMemcpy(ds, sn.data(), sn.size() * 4, cudaMemcpyHostToDevice), "up");
                    ck(cudaMemcpy(dp, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice), "up");
                    k::rope_neox_apply(dxa, dxa, rows, hd, nrot, dc, ds, dp, nullptr);
                    k::rope_neox_apply_pairs(dxb, dxb, rows, hd, nrot, dc, ds, dp, nullptr);
                    std::vector<float> a(x.size()), b(x.size());
                    ck(cudaMemcpy(a.data(), dxa, a.size() * 4, cudaMemcpyDeviceToHost), "down");
                    ck(cudaMemcpy(b.data(), dxb, b.size() * 4, cudaMemcpyDeviceToHost), "down");
                    ++rp_n;
                    if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0) {
                        ++rp_bad;
                        std::printf("rope DIFF head_dim %d n_rot %d rows %d\n", hd, nrot, rows);
                    }
                    cudaFree(dxa); cudaFree(dxb); cudaFree(dc); cudaFree(ds); cudaFree(dp);
                }
        std::printf("rope pairs: %d / %d shapes the same bytes\n", rp_n - rp_bad, rp_n);
        bad += rp_bad;
    }
    return bad == 0 ? 0 : 1;
}
