// tests/dsa_topk_parity.cu - the O(n) DSA pool selection (src/kernels/cuda/dsa_topk.cuh) against the rank order the
// O(n^2) loop defined (score descending, ties by the lower index), on random scores, heavy ties (the ReLU sums are
// often exactly 0), one value everywhere and negative scores, n from 1 to 250k pools; and the old and new kernels'
// times at the context sizes that matter.  Then the several-block version (select_top_pools_mb) against the one-block
// kernel's cells, every grid size, the same shapes plus ties straddling the cut, many equal keys and ReLU-like scores,
// and both kernels' times.
//   nvcc -O3 -std=c++17 -arch=sm_70 -I src/kernels/cuda tests/dsa_topk_parity.cu -o build/dsa_topk_parity
#include "dsa_topk.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

__global__ void new_kernel(const float* sc, int n, int top, int kpool, int* cells) {
    strata::kernels::dsa::select_top_pools(sc, n, top, kpool, cells);
}
__global__ void __launch_bounds__(1024) mb_kernel(const float* sc, int n, int top, int kpool, int* cells,
                                                  strata::kernels::dsa::MbScratch* ws) {
    strata::kernels::dsa::select_top_pools_mb(sc, n, top, kpool, cells, ws);
}

// the loop it replaces (glm_fast.cu before 2026-10-07), scores from global memory
__global__ void old_kernel(const float* sc, int n_vis, int top, int kpool, int* cells) {
    for (int p = threadIdx.x; p < n_vis; p += blockDim.x) {
        const float v = sc[p];
        int rank = 0;
        for (int q = 0; q < n_vis; ++q) {
            const float u = sc[q];
            rank += u > v ? 1 : 0;
            rank += (q < p && u == v) ? 1 : 0;
        }
        if (rank < top)
            for (int m = 0; m < kpool; ++m) cells[rank * kpool + m] = p * kpool + m;
    }
}

static std::vector<int> reference(const std::vector<float>& s, int top, int kpool) {
    std::vector<int> idx(s.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return s[a] > s[b]; });   // ties keep index order
    std::vector<int> cells((size_t) top * kpool);
    for (int r = 0; r < top; ++r)
        for (int m = 0; m < kpool; ++m) cells[(size_t) r * kpool + m] = idx[r] * kpool + m;
    return cells;
}

int main() {
    std::mt19937 rng(7);
    // DeepSeek V4's indexer (512 pools of 4 cells) and V4.1's candidate blocks (up to 2048 blocks, 1 a pool: the
    // selection past 16K tokens that wrote out of range before 2026-10-19), with a width between
    const int configs[][2] = {{4, 512}, {1, 1025}, {1, 2048}};   // {kpool, top_max}
    const int sizes[] = {1, 2, 7, 511, 512, 513, 1000, 1024, 1025, 2000, 2047, 2048, 2049, 7500, 12288, 12289, 16000,
                         65536, 250000};
    const char* kinds[] = {"random", "ties", "constant", "negative"};
    float* d_s = nullptr;
    int* d_c = nullptr;
    cudaMalloc(&d_s, 250000 * sizeof(float));
    cudaMalloc(&d_c, 2048 * 4 * sizeof(int));
    int bad = 0, cases = 0;
    for (const auto& cfg : configs)
      for (int n : sizes)
        for (int kind = 0; kind < 4; ++kind) {
            const int kpool = cfg[0], top_max = cfg[1];
            std::vector<float> s(n);
            std::uniform_real_distribution<float> u(0.0f, 10.0f);
            std::uniform_int_distribution<int> q(0, 6);
            for (int i = 0; i < n; ++i)
                s[i] = kind == 0 ? u(rng) : kind == 1 ? (q(rng) < 4 ? 0.0f : 0.25f * (float) q(rng))
                     : kind == 2 ? 1.5f : -u(rng);
            const int top = std::min(top_max, n);
            cudaMemcpy(d_s, s.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d_c, 0xff, top_max * kpool * sizeof(int));
            new_kernel<<<1, 1024>>>(d_s, n, top, kpool, d_c);
            std::vector<int> got((size_t) top * kpool);
            cudaMemcpy(got.data(), d_c, got.size() * sizeof(int), cudaMemcpyDeviceToHost);
            const cudaError_t e = cudaGetLastError();
            const bool ok = e == cudaSuccess && got == reference(s, top, kpool);
            ++cases;
            bad += !ok;
            if (!ok) std::printf("MISMATCH top %d kpool %d n=%d %s (%s)\n", top_max, kpool, n, kinds[kind], cudaGetErrorString(e));
        }
    std::printf("parity: %d of %d cases match the rank order\n", cases - bad, cases);
    // time at the context sizes of a chat: pools = context / 4
    const int kpool = 4, top_max = 512;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int ctx : {8192, 32768, 65536, 131072, 262144}) {
        const int n = ctx / kpool;
        std::vector<float> s(n);
        std::uniform_real_distribution<float> u(0.0f, 10.0f);
        for (float& v : s) v = u(rng);
        cudaMemcpy(d_s, s.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        float t_new = 0, t_old = 0;
        for (int rep = 0; rep < 2; ++rep) {   // the first launch warms up
            cudaEventRecord(a);
            for (int i = 0; i < 10; ++i) new_kernel<<<1, 1024>>>(d_s, n, top_max, kpool, d_c);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            cudaEventElapsedTime(&t_new, a, b);
        }
        const int reps_old = ctx <= 65536 ? 3 : 1;
        cudaEventRecord(a);
        for (int i = 0; i < reps_old; ++i) old_kernel<<<1, 1024>>>(d_s, n, top_max, kpool, d_c);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        cudaEventElapsedTime(&t_old, a, b);
        std::printf("context %6d (%5d pools): old %8.3f ms, new %6.3f ms per DSA layer per token\n", ctx, n,
                    t_old / reps_old, t_new / 10);
    }
    // ---- the several-block version: the one-block kernel's cells for every grid size
    strata::kernels::dsa::MbScratch* ws = nullptr;
    cudaMalloc(&ws, sizeof(*ws));
    cudaMemset(ws, 0, sizeof(*ws));
    const int nmax = 1 << 20;
    float* d_m = nullptr;
    int *d_a = nullptr, *d_b = nullptr;
    cudaMalloc(&d_m, nmax * sizeof(float));
    cudaMalloc(&d_a, 2048 * 4 * sizeof(int));
    cudaMalloc(&d_b, 2048 * 4 * sizeof(int));
    int mb_bad = 0, mb_cases = 0, nsm = 0, occ = 0;
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mb_kernel, 1024, 0);
    const int gres = nsm * occ;
    std::printf("several blocks: %d SMs / CUs x %d resident blocks of 1024 threads\n", nsm, occ);
    const int mb_sizes[] = {2, 7, 512, 513, 2048, 2049, 4095, 8192, 8193, 12289, 16384, 32768, 32771, 65536, 100003,
                            131072, 262144, 1 << 20};
    const int grids[] = {1, 2, 3, 5, 8, 13, 16, 32, 64};
    const char* mkinds[] = {"random", "ties", "constant", "negative", "relu", "straddle", "fewpos"};
    for (const auto& cfg : configs)
      for (int n : mb_sizes)
        for (int kind = 0; kind < 7; ++kind) {
            const int kpool = cfg[0], top_max = cfg[1];
            std::vector<float> s(n);
            std::uniform_real_distribution<float> u(0.0f, 10.0f);
            std::uniform_int_distribution<int> q(0, 6);
            std::exponential_distribution<float> ex(1.0f);
            const int top = std::min(top_max, n);
            for (int i = 0; i < n; ++i) {
                switch (kind) {
                    case 0: s[i] = u(rng); break;
                    case 1: s[i] = q(rng) < 4 ? 0.0f : 0.25f * (float) q(rng); break;
                    case 2: s[i] = 1.5f; break;
                    case 3: s[i] = -u(rng); break;
                    case 4: s[i] = q(rng) < 3 ? 0.0f : ex(rng); break;        // ReLU sums: many exact zeros
                    case 5: s[i] = u(rng) + 20.0f; break;                    // set below: a tie run across the cut
                    default: s[i] = (i % 997 == 0) ? u(rng) : 0.0f; break;   // fewer positives than top
                }
            }
            if (kind == 5 && n > top) {   // top / 2 clear winners, then 3 x top equal keys spread over every slice
                std::vector<int> idx(n);
                std::iota(idx.begin(), idx.end(), 0);
                std::shuffle(idx.begin(), idx.end(), rng);
                for (int i = 0; i < n; ++i) s[i] = u(rng);
                for (int i = 0; i < top / 2; ++i) s[idx[i]] = 100.0f + (float) i;
                for (int i = top / 2; i < std::min(n, top / 2 + 3 * top); ++i) s[idx[i]] = 50.0f;
            }
            cudaMemcpy(d_m, s.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d_a, 0xff, top_max * kpool * sizeof(int));
            new_kernel<<<1, 1024>>>(d_m, n, top, kpool, d_a);
            std::vector<int> want((size_t) top * kpool), got((size_t) top * kpool);
            cudaMemcpy(want.data(), d_a, want.size() * sizeof(int), cudaMemcpyDeviceToHost);
            for (int G : grids) {
                if (G > gres) continue;   // (the barrier needs every block resident at once)
                cudaMemset(d_b, 0xff, top_max * kpool * sizeof(int));
                mb_kernel<<<G, 1024>>>(d_m, n, top, kpool, d_b, ws);
                cudaMemcpy(got.data(), d_b, got.size() * sizeof(int), cudaMemcpyDeviceToHost);
                const cudaError_t e = cudaGetLastError();
                const bool ok = e == cudaSuccess && got == want;
                ++mb_cases;
                mb_bad += !ok;
                if (!ok)
                    std::printf("MB MISMATCH top %d kpool %d n=%d %s G=%d (%s)\n", top_max, kpool, n, mkinds[kind], G,
                                cudaGetErrorString(e));
            }
        }
    // the scratch is left clean (a second round on the same scratch gives the same cells: covered above, every case
    // reuses it); check it is all zero
    {
        std::vector<unsigned char> h(sizeof(*ws));
        cudaMemcpy(h.data(), ws, h.size(), cudaMemcpyDeviceToHost);
        const size_t off = offsetof(strata::kernels::dsa::MbScratch, win);
        const size_t off2 = offsetof(strata::kernels::dsa::MbScratch, bar);
        bool clean = true;
        for (size_t i = 0; i < off; ++i) clean = clean && h[i] == 0;
        for (size_t i = off2; i < h.size(); ++i) clean = clean && h[i] == 0;
        if (!clean) { std::printf("MB scratch not clean after the calls\n"); ++mb_bad; }
    }
    std::printf("several blocks: %d of %d cases match the one-block kernel\n", mb_cases - mb_bad, mb_cases);
    // times: one block vs several (kernel launches back to back, events around 50)
    for (const auto& tk : {std::pair<int, int>{512, 1}, std::pair<int, int>{2048, 1}}) {
        const int top = tk.first;
        for (int n : {4096, 8192, 12288, 16384, 32768, 65536, 131072, 262144}) {
            std::vector<float> s(n);
            std::exponential_distribution<float> ex(1.0f);
            std::uniform_int_distribution<int> q(0, 6);
            for (float& v : s) v = q(rng) < 2 ? 0.0f : ex(rng);
            cudaMemcpy(d_m, s.data(), n * sizeof(float), cudaMemcpyHostToDevice);
            const auto time = [&](auto&& launch) {
                float t = 0;
                for (int rep = 0; rep < 2; ++rep) {
                    cudaEventRecord(a);
                    for (int i = 0; i < 50; ++i) launch();
                    cudaEventRecord(b);
                    cudaEventSynchronize(b);
                    cudaEventElapsedTime(&t, a, b);
                }
                return 1000.0f * t / 50;
            };
            const float t1 = time([&] { new_kernel<<<1, 1024>>>(d_m, n, top, 1, d_a); });
            std::printf("top %4d n %6d: one block %6.1f us |", top, n, t1);
            for (int slice : {1024, 2048, 4096}) {
                const int G = std::min(std::min(64, gres / 2), std::max(2, (n + slice - 1) / slice));
                const float t2 = time([&] { mb_kernel<<<G, 1024>>>(d_m, n, top, 1, d_b, ws); });
                std::printf(" slice %4d G %2d %6.1f us |", slice, G, t2);
            }
            std::printf("\n");
        }
    }
    return (bad || mb_bad) ? 1 : 0;
}
