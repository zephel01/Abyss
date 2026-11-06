// embed_rows_parity.cpp - the GPU embedding gathers (embed_rows_q: Q8_0, Q4_K, Q5_K, Q6_K) against ggml's to_float
// on the host, bit for bit, over random rows (random bytes, finite fp16 scales).  Exit 77 without a GPU.
#include "strata/kernels/glm_fast.hpp"
#include "ggml.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("embed_rows_parity: no GPU (skip)\n"); return 77; }
    const int E = 4096, R = 64, T = 8;
    std::mt19937 rng(11);
    int bad_all = 0;
    for (const int type : {8, 12, 13, 14}) {
        const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) type);
        const size_t rb = ggml_row_size((ggml_type) type, E);
        std::vector<uint8_t> tab(rb * R);
        for (auto& b : tab) b = (uint8_t) rng();
        // every fp16 in the block headers made a small finite value (random bits give inf / nan)
        const size_t bb = ggml_type_size((ggml_type) type), nb = rb / bb;
        for (size_t i = 0; i < (size_t) R * nb; ++i) {
            uint8_t* blk = tab.data() + i * bb;
            const size_t off = type == 14 ? bb - 2 : 0;   // Q6_K: d at the end; Q8_0 / Q4_K / Q5_K: d (and dmin) first
            const int nh = (type == 12 || type == 13) ? 2 : 1;
            for (int h = 0; h < nh; ++h) {
                const uint16_t v = (uint16_t) (0x2000 + (rng() & 0x0FFF)) | (uint16_t) ((rng() & 1) << 15);
                std::memcpy(blk + off + 2 * h, &v, 2);
            }
        }
        std::vector<int> toks(T);
        for (auto& t : toks) t = (int) (rng() % R);
        void* dtab = nullptr; int* dtok = nullptr; float* dout = nullptr;
        cudaMalloc(&dtab, tab.size()); cudaMalloc(&dtok, T * sizeof(int)); cudaMalloc(&dout, (size_t) T * E * sizeof(float));
        cudaMemcpy(dtab, tab.data(), tab.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dtok, toks.data(), T * sizeof(int), cudaMemcpyHostToDevice);
        if (!strata::kernels::glmf::embed_rows_q(type, dtab, dtok, T, E, dout, nullptr)) {
            std::printf("type %d: not supported\n", type); ++bad_all; continue;
        }
        std::vector<float> got((size_t) T * E), want(E);
        cudaMemcpy(got.data(), dout, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int t = 0; t < T; ++t) {
            tt->to_float(tab.data() + (size_t) toks[t] * rb, want.data(), E);
            for (int e = 0; e < E; ++e)
                if (std::memcmp(&got[(size_t) t * E + e], &want[e], 4) != 0 && bad++ < 3)
                    std::printf("type %d row %d elem %d: %.9g vs %.9g\n", type, t, e, got[(size_t) t * E + e], want[e]);
        }
        std::printf("embed_rows_parity: %s %s (%d of %d differ)\n", ggml_type_name((ggml_type) type), bad ? "FAIL" : "ok",
                    bad, T * E);
        bad_all += bad;
        cudaFree(dtab); cudaFree(dtok); cudaFree(dout);
    }
    return bad_all ? 1 : 0;
}
