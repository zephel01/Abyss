// src/kernels/cpu/lane_q81_test.cpp - the CPU lane's device-arithmetic rows (lane_q81.hpp), every type the device's
// expert kernels take, on synthetic rows:
//   1. lane_q81_quant against an emulation of glm_fast's q8_1_store_warp (written here again), byte for byte;
//   2. each vector form (the dispatch, AVX2, VBMI where the CPU has it) and the portable rows against the reference
//      (the device's dot functions transcribed, calls summed in double): relative L2 <= 2e-6 - float rounding;
//   3. a window's tokens through one call bit for bit the same as each token alone;
//   4. for the record: ggml-cpu's dots (the lane's old arithmetic: Q8_K activations) against the same reference.
#include "strata/kernels/cpu/lane_q81.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "lane_q81_impl.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace kc = strata::kernels::cpu;

namespace {

uint64_t st = 0x9E3779B97F4A7C15ull;
uint64_t rnd() {
    st ^= st << 13;
    st ^= st >> 7;
    st ^= st << 17;
    return st;
}
float frnd() { return (float) ((rnd() >> 11) * (1.0 / 9007199254740992.0)); }   // [0, 1)

struct Ty {
    int t, qk;
    size_t bytes;
    const char* name;
};
const Ty kTypes[] = {
    {16, 256, 66, "IQ2_XXS"}, {17, 256, 74, "IQ2_XS"}, {22, 256, 82, "IQ2_S"},  {18, 256, 98, "IQ3_XXS"},
    {21, 256, 110, "IQ3_S"},  {19, 256, 50, "IQ1_S"},  {29, 256, 56, "IQ1_M"},  {23, 256, 136, "IQ4_XS"},
    {20, 32, 18, "IQ4_NL"},   {39, 32, 17, "MXFP4"},   {40, 64, 36, "NVFP4"},   {8, 32, 34, "Q8_0"},
    {6, 32, 22, "Q5_0"},      {7, 32, 24, "Q5_1"},     {42, 64, 18, "Q2_0"},    {10, 256, 84, "Q2_K"},
    {11, 256, 110, "Q3_K"},   {12, 256, 144, "Q4_K"},  {13, 256, 176, "Q5_K"},  {14, 256, 210, "Q6_K"},
};

void put_h(uint8_t* p, float v) {
    const uint16_t h = kc::q81::f2h(v);
    std::memcpy(p, &h, 2);
}
// random codes, sane scales
void make_rows(const Ty& T, int n, int rows, std::vector<uint8_t>& w) {
    const size_t rb = (size_t) n / T.qk * T.bytes;
    w.resize(rb * rows);
    for (auto& b : w) b = (uint8_t) rnd();
    for (size_t o = 0; o < w.size(); o += T.bytes) {
        uint8_t* b = &w[o];
        const float d = 0.002f + 0.001f * (float) (rnd() % 16);
        switch (T.t) {
            case 7: put_h(b, d); put_h(b + 2, 0.001f * (float) (rnd() % 16)); break;
            case 39: b[0] = (uint8_t) (118 + rnd() % 12); break;
            case 40: break;
            case 10: put_h(b + 80, d); put_h(b + 82, 0.001f + 0.0005f * (float) (rnd() % 16)); break;
            case 11: put_h(b + 108, d); break;
            case 12: case 13: put_h(b, d); put_h(b + 2, 0.001f + 0.0005f * (float) (rnd() % 16)); break;
            case 14: put_h(b + 208, d); break;
            case 29: {
                const uint16_t h = kc::q81::f2h(d);
                uint16_t sc[4];
                std::memcpy(sc, b + 48, 8);
                for (int k = 0; k < 4; ++k) sc[k] = (uint16_t) ((sc[k] & 0x0FFF) | (((h >> (4 * k)) & 0xF) << 12));
                std::memcpy(b + 48, sc, 8);
                break;
            }
            default: put_h(b, d); break;
        }
    }
}

// glm_fast's q8_1_store_warp, once more: 32 lanes, warp_max / warp_sum by xor shuffles
void dev_q8_1(const float* x, int n, uint8_t* out) {
    for (int b = 0; b < n / 32; ++b, out += 36) {
        float mx[32], sm[32];
        for (int l = 0; l < 32; ++l) {
            mx[l] = std::fabs(x[32 * b + l]);
            sm[l] = x[32 * b + l];
        }
        for (int o = 16; o > 0; o >>= 1) {
            float nm[32], ns[32];
            for (int l = 0; l < 32; ++l) {
                nm[l] = std::fmax(mx[l], mx[l ^ o]);
                ns[l] = sm[l] + sm[l ^ o];
            }
            std::memcpy(mx, nm, sizeof mx);
            std::memcpy(sm, ns, sizeof sm);
        }
        const float amax = mx[0], sum = sm[0];
        const float d = std::fmin(amax / 127.0f, 65504.0f);
        for (int l = 0; l < 32; ++l)
            out[4 + l] = (uint8_t) (amax == 0.0f ? 0 : (int8_t) std::fmax(-127.0f, std::fmin(127.0f, std::round(x[32 * b + l] / d))));
        const uint16_t hd = kc::q81::f2h(d), hs = kc::q81::f2h(std::fmax(-65504.0f, std::fmin(65504.0f, sum)));
        std::memcpy(out, &hd, 2);
        std::memcpy(out + 2, &hs, 2);
    }
}

void make_x(int n, std::vector<float>& x) {
    x.resize((size_t) n);
    for (int i = 0; i < n; ++i) {
        float v = (frnd() * 2.0f - 1.0f);
        if (rnd() % 97 == 0) v *= 12.0f;
        if (rnd() % 31 == 0) v = std::round(v * 64.0f) / 64.0f;   // (ties for roundf)
        x[(size_t) i] = v;
    }
    if (n >= 64)
        for (int i = 32; i < 64; ++i) x[(size_t) i] = 0.0f;   // an all-zero block
}

bool check_quant() {
    bool ok = true;
    for (int rep = 0; rep < 50 && ok; ++rep) {
        const int n = 4096;
        std::vector<float> x;
        make_x(n, x);
        if (rep == 1)
            for (int i = 0; i < 32; ++i) x[(size_t) i] = 127.0f * (float) (i - 16) / 31.0f + 0.5f;
        std::vector<uint8_t> a(kc::lane_q81_act_bytes(n)), b(kc::lane_q81_act_bytes(n)), blk((size_t) n / 32 * 36);
        kc::lane_q81_quant(x.data(), n, a.data());
        dev_q8_1(x.data(), n, blk.data());
        kc::lane_q81_from_blocks(blk.data(), n, b.data());
        if (std::memcmp(a.data(), b.data(), a.size()) != 0) {
            size_t i = 0;
            while (a[i] == b[i]) ++i;
            std::printf("FAIL quant: byte %zu of %zu differs (rep %d)\n", i, a.size(), rep);
            ok = false;
        }
    }
    if (ok) std::printf("ok   quant: 50 activations of 4096 byte for byte as q8_1_store_warp\n");
    return ok;
}

using RowsFn = void (*)(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int);

double rel_l2(const std::vector<float>& got, const std::vector<double>& ref) {
    double e = 0, r = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        e += ((double) got[i] - ref[i]) * ((double) got[i] - ref[i]);
        r += ref[i] * ref[i];
    }
    return std::sqrt(e / std::max(r, 1e-300));
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    if (bench) {
        // lane_q81_test --bench: one thread, rows of a 4096 x 2048 gate/up from L2/L3-sized sets, ns a 256 values a token
        ggml_cpu_init();
        const int n = 4096, rows = 512;
        for (const Ty& T : kTypes) {
            std::vector<uint8_t> w;
            make_rows(T, n, rows, w);
            const size_t rb = (size_t) n / T.qk * T.bytes;
            for (int nt : {1, 4}) {
                std::vector<std::vector<uint8_t>> a((size_t) nt);
                std::vector<std::vector<uint8_t>> q8k((size_t) nt);
                std::vector<const void*> ap((size_t) nt), kp((size_t) nt);
                std::vector<std::vector<float>> o((size_t) nt, std::vector<float>(rows));
                std::vector<float*> op((size_t) nt);
                const ggml_type at = ggml_get_type_traits_cpu((ggml_type) T.t)->vec_dot_type;
                for (int k = 0; k < nt; ++k) {
                    std::vector<float> x;
                    make_x(n, x);
                    a[(size_t) k].resize(kc::lane_q81_act_bytes(n));
                    kc::lane_q81_quant(x.data(), n, a[(size_t) k].data());
                    ap[(size_t) k] = a[(size_t) k].data();
                    q8k[(size_t) k].resize(ggml_row_size(at, n));
                    ggml_get_type_traits_cpu(at)->from_float(x.data(), q8k[(size_t) k].data(), n);
                    kp[(size_t) k] = q8k[(size_t) k].data();
                    op[(size_t) k] = o[(size_t) k].data();
                }
                const auto time = [&](auto fn) {
                    double best = 1e30;
                    for (int rep = 0; rep < 7; ++rep) {
                        const auto t0 = std::chrono::steady_clock::now();
                        fn();
                        best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count());
                    }
                    return best / ((double) rows * n / 256 * nt);
                };
                const double tn = time([&] { kc::lane_q81_rows(T.t, w.data(), rb, n, ap.data(), nt, op.data(), 0, rows); });
                const double t2 = !kc::lane_q81_avx2_type(T.t) ? 0.0 : time([&] {
                    kc::lane_q81_avx2_rows(T.t, w.data(), rb, n, ap.data(), nt, op.data(), 0, rows); });
                const double t3 = !kc::lane_q81_avx2_type(T.t) || !kc::cpu_avxvnni_ok() ? 0.0 : time([&] {
                    kc::lane_q81_vnni_rows(T.t, w.data(), rb, n, ap.data(), nt, op.data(), 0, rows); });
                const double tv = !(kc::cpu_avx512_ok() && kc::lane_q81_vbmi_type(T.t)) ? 0.0 : time([&] {
                    kc::lane_q81_vbmi_rows(T.t, w.data(), rb, n, ap.data(), nt, op.data(), 0, rows); });
                // the lane's old rows: native_expert's split down rows for this type (its exact multi-token / one-token
                // kernels where it has them, else ggml's dot) on ggml's activation
                kc::NativeFmt f;
                std::string err;
                double to = 0.0;
                if (kc::native_fmt(T.t, T.t, T.t, n, 256, f, err))   // (gate and up the same rows: half the time a row)
                    to = 0.5 * time([&] { kc::native_gu_rows_split(f, w.data(), w.data(), kp.data(), nt, op.data(), 0, rows); });
                else
                    std::printf("(%s)\n", err.c_str());
                std::printf("%-8s %d token%s: lane_q81 %6.2f ns (avx2 %6.2f, vnni %6.2f, vbmi %6.2f)  old lane rows %6.2f ns a 256 values a token\n",
                            T.name, nt, nt > 1 ? "s" : " ", tn, t2, t3, tv, to);
            }
        }
        return 0;
    }

    if (argc > 1 && std::string(argv[1]) == "--mt") {
        // lane_q81_test --mt <type> <n> <rows> <threads> [experts]: rows of `experts` matrices (n x rows each) from DRAM,
        // the rows split over the threads as the lane splits a call; old rows (ggml's activation) vs lane_q81
        ggml_cpu_init();
        const int t = std::atoi(argv[2]), n = std::atoi(argv[3]), rows = std::atoi(argv[4]), th = std::atoi(argv[5]);
        const int ne = argc > 6 ? std::atoi(argv[6]) : 128;
        const Ty* T = nullptr;
        for (const Ty& x : kTypes)
            if (x.t == t) T = &x;
        const size_t rb = (size_t) n / T->qk * T->bytes, mb = rb * rows;
        std::vector<uint8_t> one;
        make_rows(*T, n, rows, one);
        std::vector<uint8_t> all(mb * ne);
        for (int e = 0; e < ne; ++e) std::memcpy(all.data() + mb * e, one.data(), mb);
        std::vector<float> x;
        make_x(n, x);
        std::vector<uint8_t> a(kc::lane_q81_act_bytes(n));
        kc::lane_q81_quant(x.data(), n, a.data());
        const ggml_type at = ggml_get_type_traits_cpu((ggml_type) t)->vec_dot_type;
        std::vector<uint8_t> k8(ggml_row_size(at, n));
        ggml_get_type_traits_cpu(at)->from_float(x.data(), k8.data(), n);
        kc::NativeFmt f;   // (only what the down rows read: their type, length and bytes)
        f.gu_type = f.up_type = f.d_type = t;
        f.n_ff = n;
        f.d_row = rb;
        const bool have_old = ggml_get_type_traits_cpu((ggml_type) t)->vec_dot != nullptr;
        std::vector<float> out((size_t) rows * th);
        for (int mode = 0; mode < 2; ++mode) {
            if (mode == 0 && !have_old) continue;
            std::atomic<int> go{0}, done{0};
            std::atomic<bool> quit{false};
            int e_cur = 0;
            std::vector<std::thread> pool;
            const auto job = [&](int k) {
                const uint8_t* m = all.data() + mb * (size_t) e_cur;
                const int r0 = rows * k / th, r1 = rows * (k + 1) / th;
                float* o = out.data() + (size_t) rows * k;
                if (mode == 0) {
                    const void* ap[1] = {k8.data()};
                    float* op[1] = {o};
                    kc::native_down_rows_split(f, m, ap, 1, op, r0, r1);
                } else {
                    const void* ap[1] = {a.data()};
                    float* op[1] = {o};
                    kc::lane_q81_rows(t, m, rb, n, ap, 1, op, r0, r1);
                }
            };
            for (int k = 1; k < th; ++k)
                pool.emplace_back([&, k] {
                    int seen = 0;
                    for (;;) {
                        int g;
                        while ((g = go.load(std::memory_order_acquire)) == seen) {}
                        if (quit.load()) return;
                        seen = g;
                        job(k);
                        done.fetch_add(1);
                    }
                });
            double tot = 0;
            int cnt = 0;
            for (int rep = 0; rep < 4 * ne; ++rep) {
                e_cur = rep % ne;
                done.store(0);
                const auto t0 = std::chrono::steady_clock::now();
                go.fetch_add(1, std::memory_order_release);
                job(0);
                while (done.load() < th - 1) {}
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                if (rep >= ne) {
                    tot += us;
                    ++cnt;
                }
            }
            quit.store(true);
            go.fetch_add(1);
            for (auto& p : pool) p.join();
            std::printf("%-8s n %d rows %d threads %d %s: %7.1f us a matrix, %6.1f GB/s\n", T->name, n, rows, th,
                        mode == 0 ? "old rows" : "lane_q81", tot / cnt, mb / (tot / cnt) / 1e3);
        }
        return 0;
    }
    ggml_cpu_init();
    if (kc::cpu_avx512_ok()) kc::iq512v_set_lane_kernels(true);   // (Qwen3.8's lane decodes; the STRATA_* envs pick the older)
    bool ok = check_quant();
    std::printf("avx2 %d  avx512 (VNNI+VBMI) %d\n", (int) kc::cpu_avx2_ok(), (int) kc::cpu_avx512_ok());
    for (const Ty& T : kTypes) {
        for (int n : {4096, 640, 2560}) {
            if (n % T.qk != 0) continue;
            const int rows = 24, nt = 5;
            std::vector<uint8_t> w;
            make_rows(T, n, rows, w);
            const size_t rb = (size_t) n / T.qk * T.bytes;
            std::vector<std::vector<uint8_t>> a((size_t) nt);
            std::vector<std::vector<float>> xs((size_t) nt);
            std::vector<const void*> ap((size_t) nt);
            for (int k = 0; k < nt; ++k) {
                make_x(n, xs[(size_t) k]);
                a[(size_t) k].resize(kc::lane_q81_act_bytes(n));
                kc::lane_q81_quant(xs[(size_t) k].data(), n, a[(size_t) k].data());
                ap[(size_t) k] = a[(size_t) k].data();
            }
            std::vector<double> ref((size_t) nt * rows);
            for (int k = 0; k < nt; ++k)
                for (int r = 0; r < rows; ++r)
                    ref[(size_t) k * rows + r] = kc::lane_q81_ref_dot(T.t, w.data() + rb * r, n, ap[(size_t) k]);
            const auto run = [&](RowsFn fn, int ntk, int k0, std::vector<float>& got) {
                std::vector<float*> op((size_t) ntk);
                for (int k = 0; k < ntk; ++k) op[(size_t) k] = got.data() + (size_t) (k0 + k) * rows;
                // (in two row ranges, as the lane's jobs cut them)
                fn(T.t, w.data(), rb, n, ap.data() + k0, ntk, op.data(), 0, 7);
                fn(T.t, w.data(), rb, n, ap.data() + k0, ntk, op.data(), 7, rows);
            };
            struct Form {
                const char* name;
                RowsFn fn;
                bool have;
            };
            const Form forms[] = {
                {"rows", kc::lane_q81_rows, true},
                {"scalar", kc::lane_q81_rows_scalar, true},
                {"avx2", kc::lane_q81_avx2_rows, kc::cpu_avx2_ok() && kc::lane_q81_avx2_type(T.t)},
#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_MSC_VER)
                {"vnni", kc::lane_q81_vnni_rows, kc::cpu_avx2_ok() && kc::lane_q81_avx2_type(T.t) && kc::cpu_avxvnni_ok()},
#endif
                {"vbmi", kc::lane_q81_vbmi_rows, kc::cpu_avx512_ok() && kc::lane_q81_vbmi_type(T.t)},
            };
            std::string line;
            // the device-order forms (AVX2 / VNNI / VBMI where both keep the device's float order) must give the
            // same bits: the CPU lane's exact combine takes either
            const bool dev_both = T.t == 16 || T.t == 17 || T.t == 18 || T.t == 20 || T.t == 22;
            std::vector<float> first_dev;
            for (const Form& F : forms) {
                if (!F.have) continue;
                std::vector<float> multi((size_t) nt * rows), single((size_t) nt * rows);
                run(F.fn, nt, 0, multi);
                if (dev_both && std::strcmp(F.name, "rows") != 0 && std::strcmp(F.name, "scalar") != 0) {
                    if (first_dev.empty()) {
                        first_dev = multi;
                    } else if (std::memcmp(first_dev.data(), multi.data(), multi.size() * sizeof(float)) != 0) {
                        line += std::string(" (") + F.name + " != the first device-order form: FAIL)";
                        ok = false;
                    }
                }
                for (int k = 0; k < nt; ++k) run(F.fn, 1, k, single);
                const double e = rel_l2(multi, ref);
                const bool same = std::memcmp(multi.data(), single.data(), multi.size() * sizeof(float)) == 0;
                char buf[96];
                std::snprintf(buf, sizeof buf, " %s %.1e%s", F.name, e, same ? "" : " (window != one at a time)");
                line += buf;
                if (!(e <= 2e-6) || !same) ok = false;
            }
            // the old arithmetic (ggml-cpu's dot against its own activation format) against the device reference
            const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu((ggml_type) T.t);
            if (tr != nullptr && tr->vec_dot != nullptr && n % ggml_blck_size(tr->vec_dot_type) == 0) {
                std::vector<float> old((size_t) nt * rows);
                std::vector<uint8_t> qa(ggml_row_size(tr->vec_dot_type, n));
                for (int k = 0; k < nt; ++k) {
                    ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float(xs[(size_t) k].data(), qa.data(), n);
                    for (int r = 0; r < rows; ++r)
                        tr->vec_dot(n, &old[(size_t) k * rows + r], 0, w.data() + rb * r, 0, qa.data(), 0, 1);
                }
                char buf[64];
                std::snprintf(buf, sizeof buf, "  | ggml (%s) %.1e", ggml_type_name(tr->vec_dot_type), rel_l2(old, ref));
                line += buf;
            }
            std::printf("%s %-8s n %4d:%s\n", line.find("FAIL") == std::string::npos ? "    " : "", T.name, n, line.c_str());
        }
    }
    std::printf(ok ? "lane_q81_test: OK\n" : "lane_q81_test: FAILED\n");
    return ok ? 0 : 1;
}
