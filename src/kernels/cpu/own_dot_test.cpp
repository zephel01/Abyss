// src/kernels/cpu/own_dot_test.cpp - Strata's own GGUF arithmetic against ggml's, bit for bit:
//   * fp32 -> fp16 for every one of the 2^32 float bit patterns, fp16 -> fp32 for all 65536 halves;
//   * every type's row dequantizer (qtypes.cpp) on random blocks, and the Q8_0 reference quantizer;
//   * the CPU experts' activation quantizers (Q8_0, Q8_1, Q8_K) and dot products (own_dot.cpp: the portable and the
//     AVX2 build) for every expert type, on random rows and several lengths (odd block counts too).
// Without ggml in the build (STRATA_GGML=OFF) it checks the AVX2 build against the portable one only.
//   own_dot_test [--bench]   --bench: one-row dot timings (L2-resident rows), own vs ggml-cpu, per type
#include "strata/kernels/cpu/own_dot.hpp"
#include "strata/qtypes.hpp"
#include "ggml_cpu_oracle.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace kc = strata::kernels::cpu;
namespace qt = strata::qt;

namespace {
uint64_t st = 0x9E3779B97F4A7C15ull;
uint64_t rnd() {
    st ^= st << 13;
    st ^= st >> 7;
    st ^= st << 17;
    return st;
}
void put16(uint8_t* p, float v) {
    const uint16_t h = qt::fp32_to_fp16(v);
    std::memcpy(p, &h, 2);
}
float small() { return 0.002f + 0.001f * (float) (rnd() % 16); }

// random blocks of a type with every fp16 scale a small normal value (no NaN / infinity to compare)
std::vector<uint8_t> blocks(int t, int64_t n) {
    const size_t bs = qt::type_size(t), nb = (size_t) (n / qt::blck_size(t));
    std::vector<uint8_t> w(bs * nb);
    for (auto& b : w) b = (uint8_t) rnd();
    for (size_t i = 0; i < nb; ++i) {
        uint8_t* b = w.data() + i * bs;
        switch (t) {
            case qt::F16: for (size_t k = 0; k < bs; k += 2) put16(b + k, ((float) (rnd() % 2001) - 1000.f) * 1e-3f); break;
            case qt::BF16: case qt::F32: break;
            case qt::Q4_1: case qt::Q5_1: case qt::Q4_K: case qt::Q5_K: put16(b, small()); put16(b + 2, small() * 0.5f); break;
            case qt::Q2_K: put16(b + 80, small()); put16(b + 82, small() * 0.5f); break;
            case qt::Q3_K: put16(b + 108, small()); break;
            case qt::Q6_K: put16(b + 208, small()); break;
            case qt::IQ1_M: b[55] = (uint8_t) ((b[55] & 0x0F) | 0x10); break;   // the scale's top bits: a small value
            case qt::MXFP4: b[0] = (uint8_t) (110 + rnd() % 20); break;
            case qt::NVFP4: case qt::TQ2_0: break;
            case qt::Q8_K: { float d = small(); std::memcpy(b, &d, 4); break; }
            default: put16(b, small()); break;
        }
        if (t == qt::TQ2_0) put16(b + bs - 2, small());
    }
    if (t == qt::BF16)
        for (size_t k = 0; k < w.size(); k += 2) { const uint16_t h = qt::fp32_to_bf16(((float) (rnd() % 2001) - 1000.f) * 1e-3f); std::memcpy(&w[k], &h, 2); }
    if (t == qt::F32)
        for (size_t k = 0; k < w.size(); k += 4) { const float f = ((float) (rnd() % 2001) - 1000.f) * 1e-3f; std::memcpy(&w[k], &f, 4); }
    return w;
}

#if defined(__x86_64__) || defined(_M_X64)
constexpr bool kOracleExact = true;    // the own x86 builds follow ggml-cpu's x86 order: bit for bit
#else
constexpr bool kOracleExact = false;   // elsewhere ggml-cpu uses its own SIMD order (NEON): compare values
#endif

int fails = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++fails;
        std::printf("FAIL %s\n", what.c_str());
    }
}

const int kTypes[] = {2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 39, 40, 42};
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    const bool g = kc::ggml_cpu_have();
    std::printf("own_dot_test: oracle %s\n", !g ? "none (AVX2 build vs portable build only)"
                                             : kOracleExact ? "ggml (bit for bit)" : "ggml (values, summation-order bound)");
    if (g && !bench) {
        // fp16, both ways, every pattern
        uint64_t bad = 0;
        for (uint64_t u = 0; u <= 0xFFFFFFFFull; ++u) {
            const uint32_t b = (uint32_t) u;
            float f;
            std::memcpy(&f, &b, 4);
            if (qt::fp32_to_fp16(f) != kc::ggml_base_fp32_to_fp16(f)) ++bad;
        }
        check(bad == 0, "fp32_to_fp16 (" + std::to_string(bad) + " patterns differ)");
        {
            std::vector<uint16_t> h(65536);
            for (int i = 0; i < 65536; ++i) h[(size_t) i] = (uint16_t) i;
            std::vector<float> a(65536), b(65536);
            for (int i = 0; i < 65536; ++i) a[(size_t) i] = qt::fp16_to_fp32(h[(size_t) i]);
            kc::ggml_base_to_float(qt::F16, h.data(), b.data(), 65536);
            check(std::memcmp(a.data(), b.data(), a.size() * 4) == 0, "fp16_to_fp32");
        }
        // the type table: ids, block geometry and names equal ggml's (ggml's own count included)
        for (int t = 0; t < qt::COUNT; ++t) {
            int64_t b = 0;
            size_t sz = 0;
            const char* nm = nullptr;
            if (!kc::ggml_base_type_info(t, &b, &sz, &nm)) { check(false, "ggml type " + std::to_string(t)); continue; }
            if (b == 0 && qt::blck_size(t) == 0) continue;   // an id neither uses
            check(b == qt::blck_size(t) && sz == qt::type_size(t) && std::string(nm) == qt::type_name(t),
                  "type table " + std::to_string(t) + " (ggml " + nm + " " + std::to_string(b) + "/" + std::to_string(sz) +
                      ", own " + qt::type_name(t) + " " + std::to_string(qt::blck_size(t)) + "/" + std::to_string(qt::type_size(t)) + ")");
        }
        {
            int64_t b;
            size_t sz;
            const char* nm;
            check(!kc::ggml_base_type_info(qt::COUNT, &b, &sz, &nm), "ggml has more types than qtypes (COUNT)");
        }
        // dequantizers
        for (int t = 0; t < qt::COUNT; ++t) {
            const qt::Traits* tr = qt::traits(t);
            if (tr == nullptr || tr->to_float == nullptr || t == qt::Q8_K || t == qt::F32) continue;
            const int64_t n = qt::blck_size(t) * 64;
            const std::vector<uint8_t> w = blocks(t, n);
            std::vector<float> a((size_t) n), b((size_t) n);
            tr->to_float(w.data(), a.data(), n);
            const bool have = kc::ggml_base_to_float(t, w.data(), b.data(), n);
            check(have && std::memcmp(a.data(), b.data(), a.size() * 4) == 0, std::string("to_float ") + tr->name);
        }
        // the Q8_0 reference quantizer
        {
            std::vector<float> x(4096);
            for (auto& v : x) v = ((float) (rnd() % 20001) - 10000.f) * 1e-4f;
            std::vector<uint8_t> a(qt::row_size(qt::Q8_0, 4096)), b(a.size());
            qt::traits(qt::Q8_0)->from_float_ref(x.data(), a.data(), 4096);
            kc::ggml_base_q8_0_ref(x.data(), b.data(), 4096);
            check(a == b, "q8_0 from_float_ref");
        }
    }
    // activation quantizers and dot products
    int checked = 0;
    const int reps = std::getenv("OWN_DOT_REPS") ? std::atoi(std::getenv("OWN_DOT_REPS")) : 20;
    for (int t : kTypes) {
        const int at = kc::own_vec_dot_type(t);
        if (g) check(at == kc::ggml_cpu_vec_dot_type(t), std::string("vec_dot_type ") + qt::type_name(t));
        const int64_t bl = qt::blck_size(t);
        for (int64_t n : {bl * 1, bl * 3, bl * 8, (int64_t) 4096, (int64_t) 6144}) {
            if (n % bl != 0 || n % qt::blck_size(at) != 0) continue;
            for (int rep = 0; rep < (bench ? 0 : reps); ++rep) {
                const std::vector<uint8_t> w = blocks(t, n);
                std::vector<float> x((size_t) n);
                const float sc = rep % 4 == 0 ? 1e-3f : 1.0f;
                for (auto& v : x) v = ((float) (rnd() % 2001) - 1000.0f) * sc;
                if (rep == 1) x[(size_t) (rnd() % (uint64_t) n)] *= 50.0f;   // one outlier
                std::vector<uint8_t> aq(qt::row_size(at, n)), ag(aq.size());
                kc::own_quant_act(at, x.data(), aq.data(), n);
                const float s1 = kc::own_dot(t, (int) n, w.data(), aq.data());
                const float s2 = kc::own_dot_fast(t, (int) n, w.data(), aq.data());
                if (std::memcmp(&s1, &s2, 4) != 0 && fails < 5) std::printf("  n %lld portable %.9g avx2 %.9g\n", (long long) n, (double) s1, (double) s2);
                check(std::memcmp(&s1, &s2, 4) == 0, std::string("avx2 vs portable ") + qt::type_name(t));
                if (g && !kOracleExact) {
                    // aarch64 (no own SIMD build): ggml-cpu's NEON dots and quantizers sum in another order, so bits
                    // differ by design (and ggml stays the default there, STRATA_CPU_GGML).  Check the value instead:
                    // the same activations into both dots, within a summation-order bound of sum |w x|.
                    const qt::Traits* tw = qt::traits(t);
                    std::vector<float> wf((size_t) n);
                    tw->to_float(w.data(), wf.data(), n);
                    double l1 = 0.0;
                    for (int64_t k = 0; k < n; ++k) l1 += std::fabs((double) wf[(size_t) k] * (double) x[(size_t) k]);
                    const float sg = kc::ggml_cpu_dot(t, (int) n, w.data(), aq.data());
                    const double tol = 2e-4 * l1 + 1e-6;
                    if (std::fabs((double) s1 - (double) sg) > tol && fails < 40)
                        std::printf("  %s n %lld rep %d: own %.9g ggml %.9g (tol %.3g)\n", qt::type_name(t), (long long) n,
                                    rep, (double) s1, (double) sg, tol);
                    check(std::fabs((double) s1 - (double) sg) <= tol,
                          std::string("dot (value) ") + qt::type_name(t) + " n " + std::to_string(n));
                } else if (g) {
                    kc::ggml_cpu_quant(at, x.data(), ag.data(), n);
                    check(aq == ag, std::string("activation ") + qt::type_name(at) + " for " + qt::type_name(t) + " n " +
                                        std::to_string(n));
                    const float sg = kc::ggml_cpu_dot(t, (int) n, w.data(), ag.data());
                    if (std::memcmp(&s1, &sg, 4) != 0 && fails < 40)
                        std::printf("  %s n %lld rep %d: own %.9g ggml %.9g\n", qt::type_name(t), (long long) n, rep,
                                    (double) s1, (double) sg);
                    check(std::memcmp(&s1, &sg, 4) == 0, std::string("dot ") + qt::type_name(t) + " n " + std::to_string(n));
                }
                ++checked;
            }
        }
    }
    if (bench) {
        // one row of 4096 values (6144 for 64- or 256-value types it fits), 64 rows L2-resident, ns a row
        std::printf("%-8s %10s %10s %10s\n", "type", "own ns", "ggml ns", "own/ggml");
        for (int t : kTypes) {
            const int at = kc::own_vec_dot_type(t);
            const int n = 4096, rows = 64;
            std::vector<uint8_t> w;
            for (int r = 0; r < rows; ++r) {
                const std::vector<uint8_t> b = blocks(t, n);
                w.insert(w.end(), b.begin(), b.end());
            }
            const size_t rb = qt::row_size(t, n);
            std::vector<float> x(n);
            for (auto& v : x) v = ((float) (rnd() % 2001) - 1000.0f) * 1e-3f;
            std::vector<uint8_t> a(qt::row_size(at, n));
            kc::own_quant_act(at, x.data(), a.data(), n);
            const auto time = [&](auto&& f) {
                volatile float sink = 0;
                double best = 1e30;
                for (int k = 0; k < 7; ++k) {
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int it = 0; it < 20; ++it)
                        for (int r = 0; r < rows; ++r) sink = sink + f(w.data() + (size_t) r * rb);
                    best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / (20.0 * rows));
                }
                return best;
            };
            const double to = time([&](const uint8_t* p) { return kc::own_dot_fast(t, n, p, a.data()); });
            const double tg = g ? time([&](const uint8_t* p) { return kc::ggml_cpu_dot(t, n, p, a.data()); }) : 0.0;
            std::printf("%-8s %10.1f %10.1f %10.2f\n", qt::type_name(t), to, tg, tg > 0 ? to / tg : 0.0);
        }
        return 0;
    }
    std::printf("own_dot_test: %d dot cases, %d failures\n", checked, fails);
    return fails == 0 ? 0 : 1;
}
