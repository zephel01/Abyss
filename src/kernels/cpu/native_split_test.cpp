// src/kernels/cpu/native_split_test.cpp - GLM's split-base expert rows (a GGUF's own gate/up/down tensors, the CPU
// expert lane) against the contiguous [gate|up|down] blob path, and GLM's clamped SwiGLU.  Synthetic, CPU only.
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "native_split_test: %s\n", what);
        std::exit(1);
    }
}

int main() {
    namespace cpu = strata::kernels::cpu;
    cpu::NativeFmt glm, f;
    std::string err;
    // GLM-5.3-Flash's experts (n_embd 4096, n_ff 2048, IQ1_S gate/up, Q2_K down) fit the pool's activation buffers
    check(cpu::native_fmt(GGML_TYPE_IQ1_S, GGML_TYPE_Q2_K, 4096, 2048, glm, err), "GLM geometry");
    check(cpu::native_fmt(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 32, 32, f, err), "small format");
    std::vector<unsigned char> blob(f.bytes), aq(f.act_bytes), hq(f.h_bytes);
    std::vector<float> weights(32 * 32, 1.0f), x(32, 1.0f), h(32), ordinary(32), out(32), separate(32);
    const auto quant = [&](const std::vector<float>& w, size_t off) {
        ggml_quantize_chunk(GGML_TYPE_Q8_0, w.data(), blob.data() + off, 0, 32, 32, nullptr);
    };
    quant(weights, 0);   // gate: every dot 32
    for (auto& w : weights) w = 2.0f;
    quant(weights, f.up_off);   // up: 64
    for (auto& w : weights) w = 1.0f;
    quant(weights, f.down_off);
    cpu::native_quant_act(f, x.data(), aq.data());
    const void* act[] = {aq.data()};
    float* result[] = {h.data()};
    // limit 10: the gate is capped at 10 and the up clamped to [-10, 10]
    cpu::native_gu_rows_split(f, blob.data(), blob.data() + f.up_off, act, 1, result, 0, 32, 10.0f);
    const float expected = 100.0f / (1.0f + std::exp(-10.0f));
    for (float v : h) check(std::abs(v - expected) < 1e-4f, "clamped SwiGLU");
    cpu::native_quant_h(f, h.data(), hq.data());
    const void* down_act[] = {hq.data()};
    float* down_out[] = {out.data()};
    float* down_split[] = {separate.data()};
    cpu::native_down_rows(f, blob.data(), down_act, 1, down_out, 0, 32);
    cpu::native_down_rows_split(f, blob.data() + f.down_off, down_act, 1, down_split, 0, 32);
    check(out == separate, "split down equals contiguous down");
    float* plain_out[] = {ordinary.data()};
    cpu::native_gu_rows(f, blob.data(), act, 1, plain_out, 0, 32);
    cpu::native_gu_rows_split(f, blob.data(), blob.data() + f.up_off, act, 1, result, 0, 32,
                              std::numeric_limits<float>::infinity());
    check(h == ordinary, "unclamped split equals contiguous gate/up");

    // a mixed layer (unsloth's UD-IQ1_S Qwen3.8 layer 18: IQ2_S gate, IQ2_XXS up, Q2_0 down): each role at its own
    // type and row size - the blob path, the split path and the multi-token rows all equal per-role ggml dots
    {
        const int E = 512, FF = 64;
        cpu::NativeFmt m;
        check(cpu::native_fmt(GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_0, E, FF, m, err), "mixed format");
        check(m.mixed() && m.gu_row != m.up_row, "mixed rows");
        check(m.down_off == m.up_off + m.up_row * FF && m.bytes == m.down_off + m.d_row * E, "mixed blob layout");
        cpu::NativeFmt same;
        check(cpu::native_fmt(GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_S, GGML_TYPE_Q2_0, E, FF, same, err) && !same.mixed(),
              "same types are not mixed");
        check(!cpu::native_fmt(GGML_TYPE_IQ2_S, GGML_TYPE_Q8_0, GGML_TYPE_Q2_0, E, FF, same, err),
              "gate/up of different activation formats refused");
        int gt = 0, ut = 0;
        check(cpu::parse_gu_field("22,16", gt, ut) && gt == 22 && ut == 16, "parse G,U");
        check(cpu::parse_gu_field("17", gt, ut) && gt == 17 && ut == 17, "parse T");
        check(!cpu::parse_gu_field("17,", gt, ut) && !cpu::parse_gu_field("x", gt, ut), "parse junk");
        // quantize random rows (the i-quants need ggml's init)
        ggml_quantize_init(GGML_TYPE_IQ2_S);
        ggml_quantize_init(GGML_TYPE_IQ2_XXS);
        std::vector<unsigned char> mb(m.bytes);
        uint32_t st = 12345u;
        const auto rnd = [&st] {
            st = st * 1664525u + 1013904223u;
            return ((float) (st >> 8) / 16777216.0f - 0.5f) * 0.1f;
        };
        std::vector<float> wg((size_t) FF * E), wu((size_t) FF * E), wd((size_t) E * FF);
        for (auto& v : wg) v = rnd();
        for (auto& v : wu) v = rnd();
        for (auto& v : wd) v = rnd();
        const std::vector<float> imat((size_t) E, 1.0f);   // (IQ2_XXS needs an importance matrix: a flat one)
        ggml_quantize_chunk(GGML_TYPE_IQ2_S, wg.data(), mb.data(), 0, FF, E, imat.data());
        ggml_quantize_chunk(GGML_TYPE_IQ2_XXS, wu.data(), mb.data() + m.up_off, 0, FF, E, imat.data());
        ggml_quantize_chunk(GGML_TYPE_Q2_0, wd.data(), mb.data() + m.down_off, 0, E, FF, nullptr);
        const int NT = 3;
        std::vector<std::vector<unsigned char>> a(NT, std::vector<unsigned char>(m.act_bytes));
        std::vector<const void*> ap(NT);
        for (int t = 0; t < NT; ++t) {
            std::vector<float> xt(E);
            for (auto& v : xt) v = rnd() * 20.0f;
            cpu::native_quant_act(m, xt.data(), a[t].data());
            ap[t] = a[t].data();
        }
        const auto* tg = ggml_get_type_traits_cpu(GGML_TYPE_IQ2_S);
        const auto* tu = ggml_get_type_traits_cpu(GGML_TYPE_IQ2_XXS);
        for (int nt : {1, NT}) {
            std::vector<std::vector<float>> o1(nt, std::vector<float>(FF)), o2(nt, std::vector<float>(FF));
            std::vector<float*> p1(nt), p2(nt);
            for (int t = 0; t < nt; ++t) {
                p1[t] = o1[t].data();
                p2[t] = o2[t].data();
            }
            cpu::native_gu_rows_split(m, mb.data(), mb.data() + m.up_off, ap.data(), nt, p1.data(), 0, FF,
                                      std::numeric_limits<float>::infinity());
            cpu::native_gu_rows(m, mb.data(), ap.data(), nt, p2.data(), 0, FF);
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < FF; ++r) {
                    float g = 0.f, u = 0.f;
                    tg->vec_dot(E, &g, 0, mb.data() + (size_t) r * m.gu_row, 0, ap[t], 0, 1);
                    tu->vec_dot(E, &u, 0, mb.data() + m.up_off + (size_t) r * m.up_row, 0, ap[t], 0, 1);
                    const float ref = (g / (1.f + std::exp(-g))) * u;
                    check(o1[t][r] == ref, "mixed split gate/up equals per-role ggml dots");
                    check(std::abs(o2[t][r] - ref) <= 1e-6f * (1.0f + std::abs(ref)), "mixed blob gate/up");
                }
        }
    }
    // a Q5_0 down layer (unsloth's UD 3.1 UD-IQ2_XS Qwen3.8: one layer's 640-wide down experts): Q8_0 activations,
    // 22-byte blocks; the blob path and the split path equal ggml's own dots, one token and three
    {
        const int64_t E = 2560, FF = 640;
        cpu::NativeFmt q;
        check(cpu::native_fmt(GGML_TYPE_IQ2_S, GGML_TYPE_Q5_0, E, FF, q, err), "Q5_0 down format");
        check(q.d_act == GGML_TYPE_Q8_0 && q.d_row == (size_t) FF / 32 * 22, "Q5_0 down rows");
        std::vector<unsigned char> qb(q.bytes);
        uint32_t st = 777u;
        const auto rnd = [&st] {
            st = st * 1664525u + 1013904223u;
            return ((float) (st >> 8) / 16777216.0f - 0.5f) * 0.1f;
        };
        std::vector<float> wd((size_t) E * FF);
        for (auto& v : wd) v = rnd();
        ggml_quantize_chunk(GGML_TYPE_Q5_0, wd.data(), qb.data() + q.down_off, 0, E, FF, nullptr);
        const int NT = 3;
        std::vector<std::vector<unsigned char>> hb(NT, std::vector<unsigned char>(q.h_bytes));
        std::vector<const void*> hp(NT);
        for (int t = 0; t < NT; ++t) {
            std::vector<float> ht(FF);
            for (auto& v : ht) v = rnd() * 20.0f;
            cpu::native_quant_h(q, ht.data(), hb[t].data());
            hp[t] = hb[t].data();
        }
        const auto* td = ggml_get_type_traits_cpu(GGML_TYPE_Q5_0);
        for (int nt : {1, NT}) {
            std::vector<std::vector<float>> o1(nt, std::vector<float>(E)), o2(nt, std::vector<float>(E));
            std::vector<float*> p1(nt), p2(nt);
            for (int t = 0; t < nt; ++t) {
                p1[t] = o1[t].data();
                p2[t] = o2[t].data();
            }
            cpu::native_down_rows(q, qb.data(), hp.data(), nt, p1.data(), 0, (int) E);
            cpu::native_down_rows_split(q, qb.data() + q.down_off, hp.data(), nt, p2.data(), 0, (int) E);
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < E; ++r) {
                    float ref = 0.f;
                    td->vec_dot(FF, &ref, 0, qb.data() + q.down_off + (size_t) r * q.d_row, 0, hp[t], 0, 1);
                    check(o1[t][r] == ref && o2[t][r] == ref, "Q5_0 down equals ggml's dots");
                }
        }
    }
    std::puts("native_split_test: ok");
    return 0;
}
