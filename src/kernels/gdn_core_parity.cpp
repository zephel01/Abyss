// src/kernels/gdn_core_parity.cpp - qwen35 / qwen4exp's delta-net middle: q35::gdn_core (one launch) against the launches
// it replaces - token path: gdn_prep_gates + gdn_scan + gdn_out_norm + quantize_q8_1 (prep in the kernel); window path:
// gdn_scan + gdn_out_norm + quantize_q8_1 after gdn_prep_gates (T rows).  GPU, synthetic, no model: Qwen3.8's shapes
// (S 128, 16 k heads, 48 v heads; 16 / 32 too), random rows, gates, states and norms.  The state, y and y's q8_1 must be the same
// bytes; then both are timed over a token's 36 delta-net layers.
// Also q35::gdn_mid (alpha / beta + conv + gdn_core in one launch) against those launches (run_mid).
// Usage: gdn_core_parity [reps=200]
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/qwen35_fast.hpp"

#include <cuda_runtime.h>

#include <chrono>
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
float* up(const std::vector<float>& h) {
    float* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(float) + 256), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice), "upload");
    return d;
}
std::vector<uint8_t> down(const void* d, size_t bytes) {
    std::vector<uint8_t> h(bytes);
    ck(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost), "download");
    return h;
}
}  // namespace

int run(int HK, int HV, int reps) {
    std::printf("h_k %d, h_v %d\n", HK, HV);
    const int S = 128, C = 2 * HK * S + HV * S, V = HV * S, TM = 8;
    const float eps = 1e-6f;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const auto rnd = [&](size_t n, float sc, float off = 0.0f) {
        std::vector<float> v(n);
        for (auto& x : v) x = nd(rng) * sc + off;
        return v;
    };
    const std::vector<float> h0 = rnd((size_t) TM * C, 1.0f), al0 = rnd((size_t) TM * HV, 1.0f),
                             be0 = rnd((size_t) TM * HV, 1.0f), st0 = rnd((size_t) S * HV * S, 0.05f),
                             z0 = rnd((size_t) TM * V, 1.0f), dt0 = rnd(HV, 0.5f), A0 = rnd(HV, 0.3f, -1.0f),
                             nrm0 = rnd(S, 0.2f, 1.0f);
    float *h = up(h0), *al = up(al0), *be = up(be0), *st = up(st0), *st0d = st, *z = up(z0), *dt = up(dt0), *A = up(A0), *nrm = up(nrm0);
    float *o, *y;
    void* yq;
    ck(cudaMalloc(&o, (size_t) TM * V * 4), "m");
    ck(cudaMalloc(&y, (size_t) TM * V * 4), "m");
    ck(cudaMalloc(&yq, (size_t) TM * V / 32 * 36), "m");
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    const auto reset = [&] {
        ck(cudaMemcpy(h, h0.data(), h0.size() * 4, cudaMemcpyHostToDevice), "h");
        ck(cudaMemcpy(al, al0.data(), al0.size() * 4, cudaMemcpyHostToDevice), "a");
        ck(cudaMemcpy(be, be0.data(), be0.size() * 4, cudaMemcpyHostToDevice), "b");
        ck(cudaMemcpy(st, st0.data(), st0.size() * 4, cudaMemcpyHostToDevice), "s");
        ck(cudaMemset(y, 0, (size_t) TM * V * 4), "y");
        ck(cudaMemset(yq, 0, (size_t) TM * V / 32 * 36), "yq");
    };
    // token path: prep in the kernel (T = 1), or (window) prep first then the kernel, T rows
    // (the out norm's eps differs from the L2 norm's on purpose: the kernel must not mix them up)
    const float out_eps = 1e-5f;
    const auto old_way = [&](int T, bool token) {
        q35::gdn_prep_gates(h, T, C, HK, S, eps, al, be, dt, A, HV, s);
        q35::gdn_scan(st, h, T, C, al, be, S, HK, HV, o, s);
        strata::kernels::gdn_out_norm(o, z, nrm, y, (int64_t) T * HV, S, out_eps, (void*) s);
        gf::quantize_q8_1(y, yq, T * V, s);
        (void) token;
    };
    const auto new_way = [&](int T, bool token) {
        if (!token) q35::gdn_prep_gates(h, T, C, HK, S, eps, al, be, dt, A, HV, s);
        q35::gdn_core(st, h, T, C, al, be, S, HK, HV, token, eps, dt, A, z, nrm, out_eps, y, yq, s);
    };
    int bad = 0;
    for (int token = 1; token >= 0; --token)
        for (int T = 1; T <= TM; ++T) {   // (prep in the kernel: the token path's T 1, more rows checked too)
            std::vector<uint8_t> r[2][3];
            for (int v = 0; v < 2; ++v) {
                reset();
                v == 0 ? old_way(T, token) : new_way(T, token);
                ck(cudaStreamSynchronize(s), "run");
                r[v][0] = down(st, st0.size() * 4);
                r[v][1] = down(y, (size_t) T * V * 4);
                r[v][2] = down(yq, (size_t) T * V / 32 * 36);
            }
            const char* names[3] = {"state", "y", "y q8_1"};
            for (int k = 0; k < 3; ++k)
                if (r[0][k] != r[1][k]) {
                    std::printf("%s T=%d: %s differs\n", token ? "token" : "window", T, names[k]);
                    ++bad;
                }
        }
    // a token's 36 delta-net layers, each its own state (36 x 8 MB: from memory, as in the engine)
    std::vector<float*> sts(36);
    for (auto& p : sts) {
        ck(cudaMalloc(&p, st0.size() * 4), "m");
        ck(cudaMemcpy(p, st0.data(), st0.size() * 4, cudaMemcpyHostToDevice), "c");
    }
    for (int v = 0; v < 2; ++v) {
        ck(cudaStreamSynchronize(s), "sync");
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            for (int l = 0; l < 36; ++l) {
                st = sts[(size_t) l];
                v == 0 ? old_way(1, true) : new_way(1, true);
            }
        ck(cudaStreamSynchronize(s), "sync");
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("%s: %.3f ms a token's 36 layers (%.1f us a layer)\n", v == 0 ? "four launches" : "gdn_core", ms,
                    ms * 1000.0 / 36.0);
    }
    for (auto p : sts) cudaFree(p);
    for (float* p : {h, al, be, st0d, z, dt, A, nrm, o, y}) cudaFree(p);
    cudaFree(yq);
    cudaStreamDestroy(s);
    return bad;
}

// the token path's middle after the qkv / z GEMV: alpha / beta through gf::mv (F32 or BF16 rows against x), the conv +
// SiLU (conv_silu_step), gdn_core with prep - against q35::gdn_mid (one launch, a cluster of h_v / h_k blocks).  The
// delta state, the conv state, y and y's q8_1 must be the same bytes; then both timed over 36 layers
int run_mid(int HK, int HV, bool bf16, int reps) {
    const int S = 128, C = 2 * HK * S + HV * S, V = HV * S, E = 2560, dc = 4;
    std::printf("gdn_mid: h_k %d, h_v %d, alpha / beta %s\n", HK, HV, bf16 ? "BF16" : "F32");
    if (!q35::gdn_mid_ok(S, HK, HV, dc, E)) {
        std::printf("  not supported on this device: skipped\n");
        return 0;
    }
    const float eps = 1e-6f, out_eps = 1e-5f;
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const auto rnd = [&](size_t n, float sc, float off = 0.0f) {
        std::vector<float> v(n);
        for (auto& x : v) x = nd(rng) * sc + off;
        return v;
    };
    const std::vector<float> qkv0 = rnd(C, 1.0f), x0 = rnd(E, 1.0f), wa0 = rnd((size_t) HV * E, 0.02f),
                             wb0 = rnd((size_t) HV * E, 0.02f), cw0 = rnd((size_t) C * dc, 0.3f),
                             cs0 = rnd((size_t) C * (dc - 1), 1.0f), st0 = rnd((size_t) S * HV * S, 0.05f),
                             z0 = rnd(V, 1.0f), dt0 = rnd(HV, 0.5f), A0 = rnd(HV, 0.3f, -1.0f), nrm0 = rnd(S, 0.2f, 1.0f);
    std::vector<uint16_t> wa16(wa0.size()), wb16(wb0.size());
    for (size_t i = 0; i < wa0.size(); ++i) {
        uint32_t u, v;
        std::memcpy(&u, &wa0[i], 4);
        std::memcpy(&v, &wb0[i], 4);
        wa16[i] = (uint16_t) (u >> 16);
        wb16[i] = (uint16_t) (v >> 16);
    }
    float *qkv = up(qkv0), *x = up(x0), *cw = up(cw0), *cs = up(cs0), *st = up(st0), *z = up(z0), *dt = up(dt0),
          *A = up(A0), *nrm = up(nrm0);
    void *wa, *wb;
    if (bf16) {
        ck(cudaMalloc(&wa, wa16.size() * 2), "m");
        ck(cudaMalloc(&wb, wb16.size() * 2), "m");
        ck(cudaMemcpy(wa, wa16.data(), wa16.size() * 2, cudaMemcpyHostToDevice), "c");
        ck(cudaMemcpy(wb, wb16.data(), wb16.size() * 2, cudaMemcpyHostToDevice), "c");
    } else {
        wa = up(wa0);
        wb = up(wb0);
    }
    float *h, *ab, *y;
    void *yq, *xq;
    ck(cudaMalloc(&h, (size_t) C * 4), "m");
    ck(cudaMalloc(&ab, (size_t) 2 * HV * 4), "m");
    ck(cudaMalloc(&y, (size_t) V * 4), "m");
    ck(cudaMalloc(&yq, (size_t) V / 32 * 36), "m");
    ck(cudaMalloc(&xq, (size_t) E / 32 * 36), "m");
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    const int ty = bf16 ? gf::kTypeBF16 : gf::kTypeF32;
    const auto old_way = [&](float* stp, float* csp) {
        gf::MvJob j[2];
        j[0] = {wa, xq, x, ab, nullptr, 1.0f, ty, E, HV};
        j[1] = {wb, xq, x, ab + HV, nullptr, 1.0f, ty, E, HV};
        gf::mv(j, 2, s);
        q35::conv_silu_step(csp, qkv, cw, C, dc, h, s);
        q35::gdn_core(stp, h, 1, C, ab, ab + HV, S, HK, HV, true, eps, dt, A, z, nrm, out_eps, y, yq, s);
    };
    const auto new_way = [&](float* stp, float* csp) {
        q35::gdn_mid(stp, csp, qkv, cw, dc, x, wa, wb, bf16, E, S, HK, HV, eps, dt, A, z, nrm, out_eps, nullptr, nullptr,
                     y, yq, nullptr, 0, s);
    };
    int bad = 0;
    for (int step = 1; step <= 3; ++step) {   // (steps in a row: the slid conv state feeds the next)
        std::vector<uint8_t> r[2][4];
        for (int v = 0; v < 2; ++v) {
            ck(cudaMemcpy(st, st0.data(), st0.size() * 4, cudaMemcpyHostToDevice), "s");
            ck(cudaMemcpy(cs, cs0.data(), cs0.size() * 4, cudaMemcpyHostToDevice), "cs");
            ck(cudaMemset(y, 0, (size_t) V * 4), "y");
            ck(cudaMemset(yq, 0, (size_t) V / 32 * 36), "yq");
            for (int k = 0; k < step; ++k) v == 0 ? old_way(st, cs) : new_way(st, cs);
            ck(cudaStreamSynchronize(s), "run");
            r[v][0] = down(st, st0.size() * 4);
            r[v][1] = down(cs, cs0.size() * 4);
            r[v][2] = down(y, (size_t) V * 4);
            r[v][3] = down(yq, (size_t) V / 32 * 36);
        }
        const char* names[4] = {"state", "conv state", "y", "y q8_1"};
        for (int k = 0; k < 4; ++k)
            if (r[0][k] != r[1][k]) {
                std::printf("  step %d: %s differs\n", step, names[k]);
                ++bad;
            }
    }
    std::vector<float*> sts(36), css(36);
    for (int l = 0; l < 36; ++l) {
        sts[(size_t) l] = up(st0);
        css[(size_t) l] = up(cs0);
    }
    for (int v = 0; v < 2; ++v) {
        ck(cudaStreamSynchronize(s), "sync");
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r)
            for (int l = 0; l < 36; ++l) v == 0 ? old_way(sts[(size_t) l], css[(size_t) l]) : new_way(sts[(size_t) l], css[(size_t) l]);
        ck(cudaStreamSynchronize(s), "sync");
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("  %s: %.3f ms a token's 36 layers (%.1f us a layer)\n", v == 0 ? "mv + conv + gdn_core" : "gdn_mid",
                    ms, ms * 1000.0 / 36.0);
    }
    for (int l = 0; l < 36; ++l) {
        cudaFree(sts[(size_t) l]);
        cudaFree(css[(size_t) l]);
    }
    for (void* p : {(void*) qkv, (void*) x, (void*) cw, (void*) cs, (void*) st, (void*) z, (void*) dt, (void*) A,
                    (void*) nrm, wa, wb, (void*) h, (void*) ab, (void*) y, yq, xq})
        cudaFree(p);
    cudaStreamDestroy(s);
    return bad;
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 200;
    int bad = 0;
    bad += run(16, 48, reps);   // Qwen3.8-Flash-Next (qwen4exp)
    bad += run(16, 32, reps);
    bad += run_mid(16, 48, false, reps);   // (UD 3.1: F32 alpha / beta)
    bad += run_mid(16, 48, true, reps);    // (GSQ-RCO: BF16)
    bad += run_mid(16, 32, false, reps);
    std::printf(bad == 0 ? "gdn_core parity: OK (every output the same bytes)\n" : "gdn_core parity: %d outputs differ\n", bad);
    return bad == 0 ? 0 : 1;
}
