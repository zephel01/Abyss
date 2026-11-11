// src/kernels/hc4_fused_parity.cpp - qwen4exp's hyper-connection mix: q35::hc4_fused (two launches) against the five
// launches it replaces (hc4_combine_norm, gf::mv down + inject, silu_scale, gf::mv up, hc4_mix) - GPU, synthetic, no
// model.  Qwen3.8-Flash-Next's shapes (4 streams x 2560, low rank 320, BF16 down / up with F32 or BF16 inject, and
// Q8_0 down / up with F32 inject - unsloth UD 3.1's; the window's five through mvn_mv, the engine's glm_mvn), random
// residuals, weights and norms, with and without the combine's `out` and the inject.  Every output must be the same
// bytes (R, inject, xn, gate, the mix and its q8_1); then both are timed over a token's 97 mixes.
// Usage: hc4_fused_parity [reps=200]
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/qwen35_fast.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace gf = strata::kernels::glmf;
namespace q35 = strata::kernels::q35;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 256), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <typename T> std::vector<uint8_t> down(const T* d, size_t n) {
    std::vector<uint8_t> h(n * sizeof(T));
    ck(cudaMemcpy(h.data(), d, h.size(), cudaMemcpyDeviceToHost), "download");
    return h;
}
uint16_t bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) (u >> 16); }
uint16_t f16(float f) {   // (round to nearest, normal range only - the test's scales)
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t sg = (u >> 16) & 0x8000u;
    const int ex = (int) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t m = u & 0x7fffffu;
    if (ex <= 0) return (uint16_t) sg;
    uint32_t h = sg | ((uint32_t) ex << 10) | (m >> 13);
    if ((m & 0x1fffu) > 0x1000u || ((m & 0x1fffu) == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}
}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
    const int E = 2560, HC = 4, HE = HC * E, LR = 320;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const auto rnd = [&](size_t n, float sc) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * sc; return v; };
    const auto rbf = [&](size_t n, float sc) { std::vector<uint16_t> v(n); for (auto& x : v) x = bf16(nd(rng) * sc); return v; };
    const std::vector<float> R0 = rnd(HE, 1.0f), outv = rnd(E, 0.5f), injp = rnd(HC, 1.0f), gam = rnd(HE, 0.3f);
    // Q8_0 rows: 34-byte blocks, an FP16 scale and 32 int8
    const auto rq8 = [&](size_t rows, size_t n, float sc) {
        std::vector<uint8_t> v(rows * (n / 32) * 34);
        for (size_t b = 0; b < rows * (n / 32); ++b) {
            float x[32], amax = 0.0f;
            for (float& q : x) { q = nd(rng) * sc; amax = std::max(amax, std::fabs(q)); }
            const float d = amax / 127.0f;
            const uint16_t h = f16(d);
            std::memcpy(&v[b * 34], &h, 2);
            for (int i = 0; i < 32; ++i) v[b * 34 + 2 + i] = (uint8_t) (int8_t) std::lround(d > 0 ? x[i] / d : 0.0f);
        }
        return v;
    };
    uint16_t* dWb = up(rbf((size_t) LR * HE, 0.02f));
    uint16_t* uWb = up(rbf((size_t) HE * LR, 0.05f));
    uint8_t* dWq = up(rq8(LR, HE, 0.02f));
    uint8_t* uWq = up(rq8(HE, LR, 0.05f));
    int wt = gf::kTypeBF16;   // (the five's and the two's down / up: wt's pair)
    const void *dW = dWb, *uW = uWb;
    const auto set_wt = [&](int t) { wt = t; dW = t == 8 ? (const void*) dWq : (const void*) dWb; uW = t == 8 ? (const void*) uWq : (const void*) uWb; };
    float* iWf = up(rnd((size_t) HC * HE, 0.01f));
    float* iWf0 = iWf;
    uint16_t* iWb = up(rbf((size_t) HC * HE, 0.01f));
    float *R = up(R0), *out = up(outv), *inj_prev = up(injp), *gamma = up(gam);
    float *xn, *lo, *gate, *x, *inj, *sc;
    void *xnq, *loq, *xq;
    ck(cudaMalloc(&xn, HE * 4), "m"); ck(cudaMalloc(&lo, 4096), "m"); ck(cudaMalloc(&gate, HE * 4), "m");
    ck(cudaMalloc(&x, E * 4), "m"); ck(cudaMalloc(&inj, 64), "m"); ck(cudaMalloc(&sc, 64), "m");
    ck(cudaMalloc(&xnq, HE / 32 * 36), "m"); ck(cudaMalloc(&loq, LR / 32 * 36), "m"); ck(cudaMalloc(&xq, E / 32 * 36), "m");
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    const float eps = 1e-6f;
    const auto five = [&](bool with_out, int itype) {
        const void* iw = itype == 0 ? (const void*) iWf : (const void*) iWb;
        q35::hc4_combine_norm(R, with_out ? out : nullptr, inj_prev, HC, gamma, eps, E, HC, 1, xn, s, xnq);
        gf::MvJob j[2];
        j[0] = {dW, xnq, xn, lo, nullptr, 1.0f, wt, HE, LR};
        j[1] = {iw, xnq, xn, inj, nullptr, 1.0f, itype, HE, HC};
        gf::mv(j, itype >= 0 ? 2 : 1, s);
        q35::silu_scale(lo, LR, 1.0f / (float) HC, s, loq);
        gf::MvJob u = {uW, loq, lo, gate, nullptr, 1.0f, wt, LR, HE};
        gf::mv(&u, 1, s);
        q35::hc4_mix(xn, gate, E, HC, 1, x, s, xq);
    };
    const auto two = [&](bool with_out, int itype) {
        const void* iw = itype < 0 ? nullptr : itype == 0 ? (const void*) iWf : (const void*) iWb;
        q35::hc4_fused(R, with_out ? out : nullptr, inj_prev, gamma, eps, E, HC, LR, dW, iw, uW, lo, inj, sc, xn, gate, x,
                       xq, s, itype == 0, wt, xnq);
    };
    if (!q35::hc4_fused_ok(E, HC, LR, gf::kTypeBF16, gf::kTypeBF16, gf::kTypeF32)) { std::fprintf(stderr, "shapes\n"); return 2; }
    if (!q35::hc4_fused_ok(E, HC, LR, 8, 8, gf::kTypeF32)) { std::fprintf(stderr, "shapes (Q8_0)\n"); return 2; }
    int bad = 0;
    for (int wtt : {gf::kTypeBF16, 8}) {
    set_wt(wtt);
    for (int with_out = 0; with_out < 2; ++with_out)
        for (int itype : {gf::kTypeF32, gf::kTypeBF16, -1}) {
            if (wt == 8 && itype == gf::kTypeBF16) continue;
            std::vector<uint8_t> res[2][6];
            for (int v = 0; v < 2; ++v) {
                ck(cudaMemcpy(R, R0.data(), HE * 4, cudaMemcpyHostToDevice), "R");
                ck(cudaMemset(inj, 0, 64), "inj");
                v == 0 ? five(with_out, itype) : two(with_out, itype);
                ck(cudaStreamSynchronize(s), "run");
                res[v][0] = down(R, HE); res[v][1] = down(inj, HC); res[v][2] = down(xn, HE);
                res[v][3] = down(gate, HE); res[v][4] = down(x, E); res[v][5] = down((uint8_t*) xq, E / 32 * 36);
            }
            const char* names[6] = {"R", "inject", "xn", "gate", "mix", "mix q8_1"};
            for (int k = 0; k < 6; ++k)
                if (res[0][k] != res[1][k]) {
                    std::printf("w %d out=%d inject type %d: %s differs\n", wt, with_out, itype, names[k]);
                    ++bad;
                }
        }
    }
    // the window's T columns: q4x_mix_cols' five launches (mvn_mv) against hc4_fused_cols, T = 1..8
    const int TM = 8;
    std::vector<float> RW0 = rnd((size_t) TM * HE, 1.0f), outw = rnd((size_t) TM * E, 0.5f), injw = rnd((size_t) TM * HC, 1.0f);
    float *RW = up(RW0), *outW = up(outw), *injpW = up(injw);
    float *xnW, *loW, *gateW, *xW, *injW, *scW;
    void *xnqW, *loqW, *xqW;
    ck(cudaMalloc(&xnW, (size_t) TM * HE * 4), "m"); ck(cudaMalloc(&loW, (size_t) TM * LR * 4), "m");
    ck(cudaMalloc(&gateW, (size_t) TM * HE * 4), "m"); ck(cudaMalloc(&xW, (size_t) TM * E * 4), "m");
    ck(cudaMalloc(&injW, TM * 64), "m"); ck(cudaMalloc(&scW, TM * 64), "m");
    ck(cudaMalloc(&xnqW, (size_t) TM * HE / 32 * 36), "m"); ck(cudaMalloc(&loqW, (size_t) TM * LR / 32 * 36), "m");
    ck(cudaMalloc(&xqW, (size_t) TM * E / 32 * 36), "m");
    const auto five_c = [&](int T, bool with_out, int itype) {
        const void* iw = itype == 0 ? (const void*) iWf : (const void*) iWb;
        q35::hc4_combine_norm(RW, with_out ? outW : nullptr, injpW, HC, gamma, eps, E, HC, T, xnW, s, xnqW);
        gf::MvJob j[2];
        j[0] = {dW, xnqW, xnW, loW, nullptr, 1.0f, wt, HE, LR};
        j[1] = {iw, xnqW, xnW, injW, nullptr, 1.0f, itype, HE, HC};
        gf::mvn_mv(j, itype >= 0 ? 2 : 1, T, s);
        q35::silu_scale(loW, T * LR, 1.0f / (float) HC, s, loqW);
        gf::MvJob u = {uW, loqW, loW, gateW, nullptr, 1.0f, wt, LR, HE};
        gf::mvn_mv(&u, 1, T, s);
        q35::hc4_mix(xnW, gateW, E, HC, T, xW, s, xqW);
    };
    const auto two_c = [&](int T, bool with_out, int itype) {
        const void* iw = itype < 0 ? nullptr : itype == 0 ? (const void*) iWf : (const void*) iWb;
        q35::hc4_fused_cols(RW, with_out ? outW : nullptr, injpW, HC, gamma, eps, E, HC, LR, T, dW, iw, uW, loW, injW,
                            scW, xnW, gateW, xW, xqW, s, itype == 0, wt, xnqW);
    };
    for (int wtt : {gf::kTypeBF16, 8}) {
    set_wt(wtt);
    for (int T = 1; T <= TM; ++T)
        for (int with_out = 0; with_out < 2; ++with_out)
            for (int itype : {gf::kTypeF32, gf::kTypeBF16, -1}) {
                if (wt == 8 && itype == gf::kTypeBF16) continue;
                std::vector<uint8_t> res[2][6];
                for (int v = 0; v < 2; ++v) {
                    ck(cudaMemcpy(RW, RW0.data(), (size_t) TM * HE * 4, cudaMemcpyHostToDevice), "R");
                    ck(cudaMemset(injW, 0, TM * 64), "inj");
                    v == 0 ? five_c(T, with_out, itype) : two_c(T, with_out, itype);
                    ck(cudaStreamSynchronize(s), "run");
                    res[v][0] = down(RW, (size_t) T * HE); res[v][1] = down(injW, (size_t) T * HC);
                    res[v][2] = down(xnW, (size_t) T * HE); res[v][3] = down(gateW, (size_t) T * HE);
                    res[v][4] = down(xW, (size_t) T * E); res[v][5] = down((uint8_t*) xqW, (size_t) T * E / 32 * 36);
                }
                const char* names[6] = {"R", "inject", "xn", "gate", "mix", "mix q8_1"};
                for (int k = 0; k < 6; ++k)
                    if (res[0][k] != res[1][k]) {
                        std::printf("w %d T=%d out=%d inject type %d: %s differs\n", wt, T, with_out, itype, names[k]);
                        ++bad;
                    }
            }
    }
    // a token's 97 mixes, each way, each mix its own weights (read from memory, not L2, as in the engine)
    for (int wtt : {gf::kTypeBF16, 8}) {
    set_wt(wtt);
    const size_t db = wt == 8 ? (size_t) LR * (HE / 32) * 34 : (size_t) LR * HE * 2;
    const size_t ub = wt == 8 ? (size_t) HE * (LR / 32) * 34 : (size_t) LR * HE * 2;
    std::vector<void*> dWs(97), uWs(97);
    std::vector<float*> iWs(97);
    for (int m = 0; m < 97; ++m) {
        ck(cudaMalloc(&iWs[m], (size_t) HC * HE * 4), "m");
        ck(cudaMemcpy(iWs[m], iWf0, (size_t) HC * HE * 4, cudaMemcpyDeviceToDevice), "c");
        ck(cudaMalloc(&dWs[m], db), "m");
        ck(cudaMalloc(&uWs[m], ub), "m");
        ck(cudaMemcpy(dWs[m], dW, db, cudaMemcpyDeviceToDevice), "c");
        ck(cudaMemcpy(uWs[m], uW, ub, cudaMemcpyDeviceToDevice), "c");
    }
    const char* wn = wt == 8 ? "Q8_0" : "BF16";
    for (int v = 0; v < 2; ++v) {
        ck(cudaStreamSynchronize(s), "sync");
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            for (int m = 0; m < 97; ++m) {
                dW = dWs[m];
                uW = uWs[m];
                iWf = iWs[m];
                v == 0 ? five(true, gf::kTypeF32) : two(true, gf::kTypeF32);
            }
        ck(cudaStreamSynchronize(s), "sync");
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("%s %s: %.3f ms a token's 97 mixes (%.1f us a mix)\n", wn, v == 0 ? "five launches" : "hc4_fused", ms,
                    ms * 1000.0 / 97.0);
    }
    for (int T = 2; T <= 6; ++T)
        for (int v = 0; v < 2; ++v) {
            ck(cudaStreamSynchronize(s), "sync");
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r)
                for (int m = 0; m < 97; ++m) {
                    dW = dWs[m];
                    uW = uWs[m];
                    iWf = iWs[m];
                    v == 0 ? five_c(T, true, gf::kTypeF32) : two_c(T, true, gf::kTypeF32);
                }
            ck(cudaStreamSynchronize(s), "sync");
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
            std::printf("%s T=%d %s: %.3f ms a window's 97 mixes (%.1f us a mix)\n", wn, T,
                        v == 0 ? "five launches (mvn_mv)" : "hc4_fused_cols", ms, ms * 1000.0 / 97.0);
        }
    for (int m = 0; m < 97; ++m) { cudaFree(dWs[m]); cudaFree(uWs[m]); cudaFree(iWs[m]); }
    iWf = iWf0;
    }
    std::printf(bad == 0 ? "hc4_fused parity: OK (every output the same bytes)\n" : "hc4_fused parity: %d outputs differ\n", bad);
    return bad == 0 ? 0 : 1;
}
