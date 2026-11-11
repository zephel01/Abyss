// src/kernels/topk_parity.cpp - the route kernel's top-k over 8 warps (block_topk512) against warp 0's (warp_topk):
// the same ids for k = 1..16 over random scores, many ties, -0 / +0, -inf runs, E 64..512 (GPU, synthetic).
// Usage: topk_parity [patterns=20000]
#include "strata/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace gf = strata::kernels::glmf;

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 20000;
    std::mt19937 rng(11);
    std::vector<float> sel((size_t) n * 512);
    std::vector<int> Es(n);
    const int Ech[] = {512, 384, 256, 160, 128, 64, 500, 333};
    for (int b = 0; b < n; ++b) {
        Es[b] = Ech[rng() % 8];
        const int mode = b % 6;
        for (int e = 0; e < 512; ++e) {
            float v;
            switch (mode) {
                case 0: v = std::normal_distribution<float>(0.0f, 1.0f)(rng); break;                 // distinct
                case 1: v = (float) (rng() % 7) * 0.25f; break;                                      // many ties
                case 2: v = (rng() % 3 == 0) ? -INFINITY : (float) (rng() % 5); break;                // -inf runs
                case 3: v = (rng() % 2) ? 0.0f : -0.0f; break;                                         // signed zeros
                case 4: v = (e < 5) ? (float) (rng() % 3) : -INFINITY; break;                          // fewer than k
                default: v = 1.0f / (1.0f + std::exp(-std::normal_distribution<float>(0.0f, 3.0f)(rng)));   // sigmoid
            }
            sel[(size_t) b * 512 + e] = v;
        }
    }
    float* sd; int* ed; int* bad;
    cudaMalloc(&sd, sel.size() * 4); cudaMalloc(&ed, n * 4); cudaMalloc(&bad, 4);
    cudaMemcpy(sd, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ed, Es.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemset(bad, 0, 4);
    if (gf::route_topk_check(sd, ed, n, bad, nullptr) != 0) { std::fprintf(stderr, "launch failed\n"); return 2; }
    int h = -1;
    if (cudaMemcpy(&h, bad, 4, cudaMemcpyDeviceToHost) != cudaSuccess) { std::fprintf(stderr, "run failed\n"); return 2; }
    unsigned long long* cy;
    cudaMalloc(&cy, 16);
    for (int k : {8, 10}) {
        gf::route_topk_time(sd, 512, k, 2000, cy, nullptr);
        unsigned long long c[2] = {0, 0};
        cudaMemcpy(c, cy, 16, cudaMemcpyDeviceToHost);
        std::printf("k %d of 512: warp_topk %llu cycles, block_topk512 %llu\n", k, c[0], c[1]);
    }
    std::printf(h == 0 ? "topk parity: OK (%d patterns x k 1..16, the same ids)\n" : "topk parity: %d ids differ\n", h == 0 ? n : h);
    return h == 0 ? 0 : 1;
}
