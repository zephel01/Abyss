// tests/lane_exact_test.cu - the CPU lane's experts against the device's, BIT FOR BIT.
//
// The decode's expert kernels (moe_route -> moe_wait_fetch -> moe_gate_up -> moe_down) on synthetic Qwen3.8-shaped
// experts (n_embd 2560, n_ff 640, k 10, gate/up IQ3_XXS, down IQ4_NL - the 16G64G pack's types; then IQ3_S gate/up,
// its one IQ3_S layer), every expert in VRAM.  The same experts through the CPU lane's exact rows (lane_q81, the
// device order: lane_q81_gu_rows / lane_q81_rows) must give:
//   1. each expert's swiglu activation as the device's q8_1 blocks (moe_gate_up's hq) - byte for byte;
//   2. the layer's output (moe_down with MoeDev::xpart: the experts' weighted rows summed in fixed point) - from the
//      CPU's per-expert down rows summed as the lane does it (fx_term), bit for bit;
//   3. the prompt's exact light rows (rows_experts_exact, several tokens an expert) - the CPU share's lane rows.
// So an expert gives the same bits whoever computes it (the CPU lane's share follows wall-clock timing).  Also printed:
// the fixed-point sum's and the float chain's largest error against a double sum.
//   lane_exact_test            the checks (ctest)
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/cpu/lane_q81.hpp"

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
namespace kc = strata::kernels::cpu;

namespace {
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "lane_exact_test: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
template <class T> T* dev(size_t n) {
    void* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T) + 256), "cudaMalloc");
    check(cudaMemset(p, 0, n * sizeof(T) + 256), "cudaMemset");
    return (T*) p;
}
template <class T> T* mapped(size_t n, T** dptr) {
    void* p = nullptr;
    check(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
    std::memset(p, 0, n * sizeof(T));
    check(cudaHostGetDevicePointer((void**) dptr, p, 0), "cudaHostGetDevicePointer");
    return (T*) p;
}
void fill_matrix(uint8_t* p, size_t bytes, size_t block, std::mt19937& rng) {
    for (size_t i = 0; i < bytes; ++i) p[i] = (uint8_t) rng();
    std::uniform_int_distribution<int> sc(0x1C00, 0x2400);   // fp16 ~0.004 .. 0.016
    for (size_t b = 0; b + block <= bytes; b += block) {
        const uint16_t h = (uint16_t) sc(rng);
        std::memcpy(p + b, &h, 2);
    }
}
uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct TypeCase { int gu, d; size_t gu_block, gu_qk, d_block, d_qk; const char* name; };

int run_case(const TypeCase& tc, float limit) {
    const int E = 2560, F = 640, K = 10, NE = 48;
    const size_t gu_row = (size_t) E / tc.gu_qk * tc.gu_block, d_row = (size_t) F / tc.d_qk * tc.d_block;
    const size_t gu_bytes = 2 * gu_row * F, down_off = gu_bytes, blob = gu_bytes + d_row * E;
    const size_t stride = (blob + 255) & ~(size_t) 255;
    const int k_slots = std::max(8, K);
    cudaStream_t s = nullptr;
    check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    std::mt19937 rng(20261110);
    std::vector<uint8_t> img(stride * NE, 0);
    for (int e = 0; e < NE; ++e) {
        uint8_t* b = img.data() + stride * e;
        fill_matrix(b, gu_bytes, tc.gu_block, rng);
        fill_matrix(b + down_off, d_row * E, tc.d_block, rng);
    }
    uint8_t* blobs_d = dev<uint8_t>(img.size());
    check(cudaMemcpy(blobs_d, img.data(), img.size(), cudaMemcpyHostToDevice), "blobs H2D");
    uint8_t* scratch = dev<uint8_t>(stride * k_slots);
    std::vector<unsigned long long> sp((size_t) k_slots);
    for (int i = 0; i < k_slots; ++i) sp[(size_t) i] = (unsigned long long) (scratch + stride * i);
    gf::MoeRequest* ring_d = nullptr;
    mapped<gf::MoeRequest>(gf::kRingSize, &ring_d);
    gf::MoeResponse* resp_d = nullptr;
    mapped<gf::MoeResponse>(gf::kCpuAnswers, &resp_d);
    float* x = dev<float>(E);
    uint8_t* xq = dev<uint8_t>((size_t) E / 32 * 36);
    uint8_t* hq = dev<uint8_t>((size_t) k_slots * F / 32 * 36);
    float* out = dev<float>(E);
    const int NT = 24;
    float* logits = dev<float>((size_t) NT * NE);
    std::vector<float> lg((size_t) NT * NE);
    {
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : lg) v = 3.0f * nd(rng);
        check(cudaMemcpy(logits, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice), "logits");
    }
    unsigned long long* tab = dev<unsigned long long>((size_t) 2 * NE + gf::kSpares);
    {
        std::vector<unsigned long long> t((size_t) 2 * NE + gf::kSpares, 0ull);
        for (int e = 0; e < NE; ++e) t[(size_t) e] = (unsigned long long) (blobs_d + stride * e);
        check(cudaMemcpy(tab, t.data(), t.size() * 8, cudaMemcpyHostToDevice), "tab");
    }
    gf::MoeDev d;
    d.tab = tab;
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
    d.xpart = dev<float>(gf::xpart_floats(E));   // (the exact combine: the fixed-point sum)

    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xv(E);
    std::vector<uint8_t> xqh((size_t) E / 32 * 36), hqh((size_t) k_slots * F / 32 * 36);
    std::vector<uint8_t> act(kc::lane_q81_act_bytes(E)), hact_cpu(kc::lane_q81_act_bytes(F)),
        hact_gpu(kc::lane_q81_act_bytes(F));
    std::vector<float> ff((size_t) F), dn((size_t) K * E), o((size_t) E), w((size_t) K);
    std::vector<int> ids((size_t) K);
    long hq_bad = 0, hq_tot = 0, out_bad = 0, out_tot = 0, worst_ulp = 0;
    double err_fx = 0.0, err_chain = 0.0, vmax = 0.0;
    long fx_bad = 0;   // (the largest error against the double sum: fixed point, the float chain)
    for (int t = 0; t < NT; ++t) {
        for (auto& v : xv) v = (t & 1 ? 8.0f : 1.0f) * nd(rng);
        check(cudaMemcpy(x, xv.data(), xv.size() * 4, cudaMemcpyHostToDevice), "x");
        gf::quantize_q8_1(x, xq, E, s);
        gf::moe_route(logits + (size_t) t * NE, nullptr, NE, K, 1.0f, true, 0, x, E, d, nullptr, nullptr, 0.0f, 0,
                      nullptr, s, nullptr, nullptr, 0, nullptr, nullptr, 0, /*skip_from*/ 64, 0ull, /*promote_min*/ 0,
                      0, /*gating*/ 2);
        gf::moe_wait_fetch(d, E, K, blob, s);
        gf::moe_gate_up(tc.gu, d, K, E, F, limit, xq, hq, nullptr, 0, nullptr, 0, nullptr, s);
        gf::moe_down(tc.d, d, K, E, F, down_off, hq, nullptr, out, s);
        check(cudaMemcpyAsync(o.data(), out, E * 4, cudaMemcpyDeviceToHost, s), "out");
        check(cudaMemcpyAsync(xqh.data(), xq, xqh.size(), cudaMemcpyDeviceToHost, s), "xq");
        check(cudaMemcpyAsync(hqh.data(), hq, hqh.size(), cudaMemcpyDeviceToHost, s), "hq");
        check(cudaMemcpyAsync(w.data(), d.plan_w, K * 4, cudaMemcpyDeviceToHost, s), "w");
        check(cudaMemcpyAsync(ids.data(), d.plan_id, K * 4, cudaMemcpyDeviceToHost, s), "ids");
        check(cudaStreamSynchronize(s), "step");
        kc::lane_q81_from_blocks(xqh.data(), E, act.data());
        const void* a[1] = {act.data()};
        for (int i = 0; i < K; ++i) {
            const uint8_t* b = img.data() + stride * (size_t) ids[(size_t) i];
            float* fo[1] = {ff.data()};
            kc::lane_q81_gu_rows(tc.gu, tc.gu, b, gu_row, b + gu_row * F, gu_row, E, a, 1, fo, 0, F, limit, 1.0f);
            kc::lane_q81_quant(ff.data(), F, hact_cpu.data());
            kc::lane_q81_from_blocks(hqh.data() + (size_t) i * (F / 32) * 36, F, hact_gpu.data());
            ++hq_tot;
            if (std::memcmp(hact_cpu.data(), hact_gpu.data(), hact_cpu.size()) != 0) {
                ++hq_bad;
                if (hq_bad <= 3) {
                    int nq = 0;
                    const int8_t* qa = (const int8_t*) hact_cpu.data();
                    const int8_t* qb = (const int8_t*) hact_gpu.data();
                    for (int j = 0; j < F; ++j) nq += qa[j] != qb[j];
                    std::printf("  %s route %d slot %d: hq differs (%d of %d codes)\n", tc.name, t, i, nq, F);
                }
            }
            // the down rows from the DEVICE's hq (so step 2 checks the down rows and the combine alone)
            const void* h[1] = {hact_gpu.data()};
            float* dno[1] = {dn.data() + (size_t) i * E};
            kc::lane_q81_rows(tc.d, b + down_off, d_row, F, h, 1, dno, 0, E);
        }
        std::vector<long long> fxl((size_t) E);   // (the lane's own sum, kc::lane_q81_fx_sum: the vector form)
        {
            const float* rows[gf::kMaxK];
            for (int i = 0; i < K; ++i) rows[i] = dn.data() + (size_t) i * E;
            kc::lane_q81_fx_sum(K, w.data(), rows, 0, E, fxl.data(), gf::kFxShift);
        }
        for (int r = 0; r < E; ++r) {
            // the exact combine (glm_fast fx_term): each weighted row to int64 units of 2^-kFxShift, summed, one rounding
            long long sx = 0;
            double ref = 0.0, chain = 0.0;
            float fc = 0.0f;
            for (int i = 0; i < K; ++i) {
                const float sv = (w[(size_t) i] * dn[(size_t) i * E + r]) * (float) (1ll << gf::kFxShift);
                sx = std::fabs(sv) < 0x1p58f && sx != gf::kFxBad ? sx + std::llrint(sv) : gf::kFxBad;
                ref += (double) w[(size_t) i] * dn[(size_t) i * E + r];
                fc = std::fma(w[(size_t) i], dn[(size_t) i * E + r], fc);
            }
            if (fxl[(size_t) r] != sx) {
                ++out_bad;
                if (out_bad <= 3) std::printf("  %s route %d row %d: lane_q81_fx_sum %lld vs %lld\n", tc.name, t, r,
                                              fxl[(size_t) r], sx);
            }
            const float m = sx == gf::kFxBad ? NAN : (float) sx * (1.0f / (float) (1ll << gf::kFxShift));
            chain = fc;
            if (sx == gf::kFxBad) ++fx_bad;
            vmax = std::max(vmax, std::fabs(ref));
            err_fx = std::max(err_fx, std::fabs((double) m - ref));
            err_chain = std::max(err_chain, std::fabs(chain - ref));
            ++out_tot;
            if (bits(m) != bits(o[(size_t) r])) {
                ++out_bad;
                worst_ulp = std::max(worst_ulp, (long) std::labs((long) bits(m) - (long) bits(o[(size_t) r])));
                if (out_bad <= 3) std::printf("  %s route %d row %d: out %.9g (cpu) vs %.9g (gpu)\n", tc.name, t, r, m, o[(size_t) r]);
            }
        }
    }
    // 3. the prompt's exact light rows (gf::rows_experts_exact): NL experts, 1..9 rows each over NTOK prompt tokens,
    // against the CPU share's lane rows (the same experts as columns: gate/up, the q8_1 of h, down)
    long px_bad = 0, px_tot = 0;
    {
        const int NL = 12, NTOK = 40;
        std::vector<int> light, row_tok;
        int rows = 0;
        std::uniform_int_distribution<int> cnt_d(1, 9), tok_d(0, NTOK - 1);
        for (int i = 0; i < NL; ++i) {
            const int c = cnt_d(rng);
            light.push_back(i * 3 % NE);
            light.push_back(rows);
            light.push_back(c);
            for (int t = 0; t < c; ++t) row_tok.push_back(tok_d(rng));
            rows += c;
        }
        std::vector<float> xs((size_t) NTOK * E);
        for (auto& v : xs) v = nd(rng);
        int* light_d = dev<int>(light.size());
        int* tok_dv = dev<int>(row_tok.size());
        float* x_d = dev<float>(xs.size());
        uint8_t* xq_d = dev<uint8_t>((size_t) NTOK * E / 32 * 36);
        uint8_t* hq_d = dev<uint8_t>((size_t) rows * F / 32 * 36);
        float* o_d = dev<float>((size_t) rows * E);
        check(cudaMemcpy(light_d, light.data(), light.size() * 4, cudaMemcpyHostToDevice), "light");
        check(cudaMemcpy(tok_dv, row_tok.data(), row_tok.size() * 4, cudaMemcpyHostToDevice), "row_tok");
        check(cudaMemcpy(x_d, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice), "xs");
        check(cudaDeviceSynchronize(), "sync");   // (dev()'s memsets run on the legacy stream: done before s reads)
        if (!gf::rows_experts_exact(tc.gu, tc.d, blobs_d, stride, down_off, light_d, NL, tok_dv, x_d, NTOK, E, F, limit,
                                    0, rows, xq_d, hq_d, o_d, E, s)) {
            std::printf("  %s: rows_experts_exact refused the types\n", tc.name);
            return 1;
        }
        if (getenv("LXT_V")) {
            std::vector<uint8_t> xqg((size_t) NTOK * E / 32 * 36), xa1(kc::lane_q81_act_bytes(E)), xa2(kc::lane_q81_act_bytes(E));
            check(cudaMemcpy(xqg.data(), xq_d, xqg.size(), cudaMemcpyDeviceToHost), "xq dbg");
            for (int t = 0; t < NTOK; ++t) {
                kc::lane_q81_from_blocks(xqg.data() + (size_t) t * E / 32 * 36, E, xa1.data());
                kc::lane_q81_quant(xs.data() + (size_t) t * E, E, xa2.data());
                if (std::memcmp(xa1.data(), xa2.data(), xa1.size())) std::printf("    xq token %d differs\n", t);
            }
        }
        std::vector<float> og((size_t) rows * E);
        check(cudaMemcpyAsync(og.data(), o_d, og.size() * 4, cudaMemcpyDeviceToHost, s), "light out");
        check(cudaStreamSynchronize(s), "light");
        const size_t ab = kc::lane_q81_act_bytes(E), hb = kc::lane_q81_act_bytes(F);
        std::vector<uint8_t> xa((size_t) rows * ab), ha((size_t) rows * hb);
        std::vector<float> fr((size_t) rows * F), orow((size_t) rows * E);
        for (int r = 0; r < rows; ++r) kc::lane_q81_quant(xs.data() + (size_t) row_tok[(size_t) r] * E, E, xa.data() + r * ab);
        for (int i = 0; i < NL; ++i) {
            const uint8_t* b = img.data() + stride * (size_t) light[3 * i];
            const int r0 = light[3 * i + 1], c = light[3 * i + 2];
            std::vector<const void*> av((size_t) c), hv((size_t) c);
            std::vector<float*> fv((size_t) c), ov((size_t) c);
            for (int t = 0; t < c; ++t) {
                av[(size_t) t] = xa.data() + (size_t) (r0 + t) * ab;
                fv[(size_t) t] = fr.data() + (size_t) (r0 + t) * F;
                hv[(size_t) t] = ha.data() + (size_t) (r0 + t) * hb;
                ov[(size_t) t] = orow.data() + (size_t) (r0 + t) * E;
            }
            kc::lane_q81_gu_rows(tc.gu, tc.gu, b, gu_row, b + gu_row * F, gu_row, E, av.data(), c, fv.data(), 0, F, limit, 1.0f);
            for (int t = 0; t < c; ++t) kc::lane_q81_quant(fv[(size_t) t], F, ha.data() + (size_t) (r0 + t) * hb);
            kc::lane_q81_rows(tc.d, b + down_off, d_row, F, hv.data(), c, ov.data(), 0, E);
        }
        for (int r = 0; r < rows; ++r) {
            int nb = 0;
            for (int c = 0; c < E; ++c) nb += bits(og[(size_t) r * E + c]) != bits(orow[(size_t) r * E + c]);
            if (nb && getenv("LXT_V")) std::printf("    row %d (tok %d): %d differ\n", r, row_tok[(size_t) r], nb);
        }
        if (getenv("LXT_V")) for (int i = 0; i < NL; ++i) std::printf("    light %d: slot %d r0 %d n %d\n", i, light[3*i], light[3*i+1], light[3*i+2]);
        for (size_t q = 0; q < og.size(); ++q) {
            ++px_tot;
            if (bits(og[q]) != bits(orow[q])) {
                if (++px_bad <= 3)
                    std::printf("  %s prompt light row %zu col %zu: %.9g (cpu) vs %.9g (gpu)\n", tc.name, q / E, q % E,
                                orow[q], og[q]);
            }
        }
    }
    std::printf("lane_exact_test %s prompt light rows: %ld of %ld values differ\n", tc.name, px_bad, px_tot);
    std::printf("lane_exact_test %s (limit %g): hq %ld of %ld experts differ; out %ld of %ld rows differ (worst %ld ulp); "
                "largest error vs the double sum: fixed point %.3g, float chain %.3g (largest row %.3g, %ld out of range)\n",
                tc.name, limit, hq_bad, hq_tot, out_bad, out_tot, worst_ulp, err_fx, err_chain, vmax, fx_bad);
    return hq_bad == 0 && out_bad == 0 && fx_bad == 0 && px_bad == 0 ? 0 : 1;
}
}  // namespace

int main() {
#if defined(STRATA_USE_HIP)
    std::printf("lane_exact_test: skipped (HIP: the RDNA expert kernels sum in another order)\n");
    return 0;
#endif
    if (!kc::lane_q81_dev_order(18) || !kc::lane_q81_dev_order(20) || !kc::lane_q81_dev_order(21)) {
        std::printf("lane_exact_test: skipped (this CPU's lane rows do not keep the device's float order)\n");
        return 0;
    }
    int bad = 0;
    const TypeCase c18 = {18, 20, 98, 256, 18, 32, "IQ3_XXS/IQ4_NL"};
    const TypeCase c21 = {21, 20, 110, 256, 18, 32, "IQ3_S/IQ4_NL"};
    bad |= run_case(c18, 1e30f);
    bad |= run_case(c18, 7.0f);
    bad |= run_case(c21, 1e30f);
    std::printf(bad ? "lane_exact_test: FAILED\n" : "lane_exact_test: OK\n");
    return bad;
}
