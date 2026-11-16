// tests/moe_own_test.cu - the prompt path's own int8 tensor-core grouped GEMM (strata::prefill::own) against a scalar
// reference on the host, and against the MMQ path where the build has it.
//
//   moe_own_test                    every covered type: random blocks (every codebook entry, sign and scale appears),
//                                   experts of 0 .. 131 rows (every tile shape), rows that are not 16-byte aligned,
//                                   a partial row tile; the activation quantizer; gate/up + SwiGLU and down
//   moe_own_test --dump <file>      a set of real expert tensors the engine wrote (STRATA_PROMPT_OWN_DUMP=<file>):
//                                   the same checks on them
//   moe_own_test --bench [type_gu type_d n_embd n_ff n_exp rows_per_expert]
//                                   kernel time, own against MMQ (rows per expert: exponential around the mean)
//
// The reference: the weights through strata::qt's to_float, the activations rounded as the kernel's quantizer
// rounds them, the sum in double.  The kernel sums a row in FP32 (32 values a step): a result may differ from the
// reference by a few FP32 rounding steps of the row's absolute sum.
#include "strata/prefill/moe_own.hpp"
#include "strata/qtypes.hpp"
#if defined(STRATA_TEST_HAVE_MMQ)
#include "strata/prefill/moe_mmq.hpp"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace own = strata::prefill::own;
namespace qt = strata::qt;

namespace {
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "moe_own_test: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
template <class T> T* dev(size_t n) {
    void* p = nullptr;
    check(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    return (T*) p;
}
template <class T> T* up(const std::vector<T>& h) {
    T* d = dev<T>(h.size());
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <class T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

// Random blocks of `type`: every stored field uniform, the scales in a range that keeps the values ordinary
void random_rows(int type, uint8_t* dst, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bs = qt::type_size(type);
    const int64_t nb = rows * cols / qt::blck_size(type);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.01f, 0.08f);
    for (int64_t b = 0; b < nb; ++b) {
        uint8_t* p = dst + b * bs;
        for (size_t i = 0; i < bs; ++i) p[i] = (uint8_t) byte(rng);
        if (type == qt::MXFP4) {
            p[0] = (uint8_t) (119 + byte(rng) % 6);
        } else {   // (the i-quants: an fp16 scale first)
            const uint16_t d = qt::fp32_to_fp16(sc(rng));
            std::memcpy(p, &d, 2);
        }
    }
}

struct Case {
    int gu_type = 0, d_type = 0;
    int n_embd = 0, n_ff = 0, n_exp = 0;
    size_t up_off = 0, down_off = 0, stride = 0;
    std::vector<uint8_t> w;          // n_exp blobs: [gate | up | down], `stride` apart
    std::vector<int32_t> bounds;     // n_exp + 1
    std::vector<float> x;            // rows x n_embd
    float limit = INFINITY;
    int rows() const { return bounds.back() - bounds.front(); }
};

// the kernel's activation rounding, on the host
void quant_ref(const float* x, int64_t cols, std::vector<int8_t>& q, std::vector<float>& d) {
    q.resize((size_t) cols);
    d.resize((size_t) cols / 32);
    for (int64_t kb = 0; kb < cols / 32; ++kb) {
        float amax = 0.0f;
        for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(x[kb * 32 + k]));
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (int k = 0; k < 32; ++k) q[(size_t) (kb * 32 + k)] = (int8_t) std::nearbyintf(x[kb * 32 + k] * inv);
        d[(size_t) kb] = amax / 127.0f;
    }
}

struct Stat {
    double worst = 0;   // max |got - ref| / (FP32 eps * the row's absolute sum)
    double rel = 0;     // max |got - ref| / rms(ref)
    int64_t n = 0;
};

// out = W[rows_w, cols] (type) x (q * d) in double, and the absolute sum
void ref_product(int type, const uint8_t* w, int rows_w, int cols, const std::vector<int8_t>& q, const std::vector<float>& d,
                 std::vector<double>& out, std::vector<double>& mag) {
    std::vector<float> wf((size_t) cols);
    out.assign((size_t) rows_w, 0.0);
    mag.assign((size_t) rows_w, 0.0);
    const size_t rb = qt::row_size(type, cols);
    for (int r = 0; r < rows_w; ++r) {
        qt::traits(type)->to_float(w + (size_t) r * rb, wf.data(), cols);
        double s = 0, m = 0;
        for (int k = 0; k < cols; ++k) {
            const double t = (double) wf[(size_t) k] * ((double) q[(size_t) k] * (double) d[(size_t) k / 32]);
            s += t;
            m += std::fabs(t);
        }
        out[(size_t) r] = s;
        mag[(size_t) r] = m;
    }
}

void accumulate(Stat& st, const float* got, const std::vector<double>& ref, const std::vector<double>& mag) {
    double ss = 0;
    for (double v : ref) ss += v * v;
    const double rms = std::sqrt(ss / (double) std::max<size_t>(1, ref.size())) + 1e-30;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double err = std::fabs((double) got[i] - ref[i]);
        st.worst = std::max(st.worst, err / (1.1920929e-7 * (mag[i] + 1e-30)));
        st.rel = std::max(st.rel, err / rms);
        ++st.n;
    }
}

double silu_mul(double g, double u, double limit) {
    g = std::min(g, limit);
    u = std::min(std::max(u, -limit), limit);
    return g / (1.0 + std::exp(-g)) * u;
}

// The whole case: quantize, gate/up + SwiGLU, quantize, down - each stage against the reference of its own inputs.
// -> false on a mismatch
bool run_case(const Case& c, const char* name, bool verbose, int ref_rows = 1 << 30) {
    const int E = c.n_embd, FF = c.n_ff, rows = c.rows(), row0 = c.bounds.front();
    const cudaStream_t s = nullptr;
    uint8_t* w = up(c.w);
    int32_t* bounds = up(c.bounds);
    float* x = up(c.x);
    uint8_t* xq = dev<uint8_t>(own::act_bytes(rows, E));
    uint8_t* hq = dev<uint8_t>(own::act_bytes(rows, FF));
    float* h = dev<float>((size_t) rows * FF);
    float* out = dev<float>((size_t) rows * E);
    check(cudaMemset(h, 0xFF, (size_t) rows * FF * sizeof(float)), "memset");
    check(cudaMemset(out, 0xFF, (size_t) rows * E * sizeof(float)), "memset");
    own::Context ctx;
    own::Set set;
    set.w = w;
    set.stride = c.stride;
    set.n = c.n_exp;
    set.bounds = bounds;
    set.host_bounds = c.bounds.data();
    set.row0 = row0;
    set.rows = rows;
    set.xq = xq;
    set.dst = h;
    set.ld = FF;
    own::quantize(x, nullptr, xq, E, E, rows, s);
    ctx.gate_up(set, c.gu_type, FF, E, c.up_off, c.limit, s);
    own::quantize(h, nullptr, hq, FF, FF, rows, s);
    set.xq = hq;
    set.dst = out;
    set.ld = E;
    ctx.down(set, c.d_type, E, FF, c.down_off, s);
    check(cudaDeviceSynchronize(), "sync");
    const std::vector<uint8_t> xq_h = down(xq, own::act_bytes(rows, E)), hq_h = down(hq, own::act_bytes(rows, FF));
    const std::vector<float> h_h = down(h, (size_t) rows * FF), out_h = down(out, (size_t) rows * E);

    bool ok = true;
    int64_t q_bad = 0;
    Stat sg, sd;
    std::vector<int8_t> q;
    std::vector<float> d;
    std::vector<double> rg, mg, ru, mu, rd, md, rh, mh;
    for (int e = 0; e < c.n_exp; ++e) {
        const uint8_t* blob = c.w.data() + (size_t) e * c.stride;
        for (int b = c.bounds[(size_t) e]; b < c.bounds[(size_t) e + 1]; ++b) {
            const int r = b - row0;
            // (a large case: the host reference for ref_rows rows spread over the set, each expert's first and last)
            if (rows > ref_rows && r % (rows / ref_rows) != 0 && b != c.bounds[(size_t) e] && b + 1 != c.bounds[(size_t) e + 1])
                continue;
            // the quantizer, bit for bit
            quant_ref(c.x.data() + (size_t) r * E, E, q, d);
            if (std::memcmp(q.data(), xq_h.data() + (size_t) r * E, (size_t) E) != 0 ||
                std::memcmp(d.data(), xq_h.data() + (size_t) rows * E + (size_t) r * (E / 32) * 4, (size_t) (E / 32) * 4) != 0)
                ++q_bad;
            ref_product(c.gu_type, blob, FF, E, q, d, rg, mg);
            ref_product(c.gu_type, blob + c.up_off, FF, E, q, d, ru, mu);
            rh.resize((size_t) FF);
            mh.resize((size_t) FF);
            for (int k = 0; k < FF; ++k) {
                rh[(size_t) k] = silu_mul(rg[(size_t) k], ru[(size_t) k], c.limit);
                // (the SwiGLU's condition: both inputs' errors reach h)
                mh[(size_t) k] = (mg[(size_t) k] * (std::fabs(ru[(size_t) k]) + 1e-3) + mu[(size_t) k] * (std::fabs(rg[(size_t) k]) + 1e-3)) * 1.2 +
                                 std::fabs(rh[(size_t) k]) * 8;
            }
            accumulate(sg, h_h.data() + (size_t) r * FF, rh, mh);
            // down from the kernel's own h (its quantized form as the device wrote it)
            quant_ref(h_h.data() + (size_t) r * FF, FF, q, d);
            if (std::memcmp(q.data(), hq_h.data() + (size_t) r * FF, (size_t) FF) != 0) ++q_bad;
            ref_product(c.d_type, blob + c.down_off, E, FF, q, d, rd, md);
            accumulate(sd, out_h.data() + (size_t) r * E, rd, md);
        }
    }
    // a few FP32 rounding steps a sum of K / 32 terms: 64 is far above them and far below any wrong value
    if (q_bad != 0 || !(sg.worst < 64) || !(sd.worst < 64)) ok = false;
    if (verbose || !ok)
        std::printf("%-28s %s  rows %d  quantizer mismatches %lld | gate/up+swiglu: %lld values, worst %.2f eps of the "
                    "sum, %.2e of rms | down: %lld values, worst %.2f eps, %.2e of rms\n",
                    name, ok ? "ok  " : "FAIL", rows, (long long) q_bad, (long long) sg.n, sg.worst, sg.rel, (long long) sd.n,
                    sd.worst, sd.rel);

#if defined(STRATA_TEST_HAVE_MMQ)
    // against the MMQ path (its own quantizer, its own sums): informational, the two agree to the rounding of the
    // activations
    namespace mmq = strata::prefill::mmq;
    if (mmq::built() && mmq::supported(c.gu_type) && mmq::supported(c.d_type) && E % 512 == 0 && FF % 512 == 0 && rows > 0) {
        mmq::Context mc;
        int max_rows = 0;
        for (int e = 0; e < c.n_exp; ++e) max_rows = std::max(max_rows, c.bounds[(size_t) e + 1] - c.bounds[(size_t) e]);
        const int64_t n_ids = rows + mmq::ids_pad(max_rows) + row0;
        int32_t* ids = dev<int32_t>((size_t) n_ids);
        mmq::iota(ids, n_ids, s);
        uint8_t* mxq = dev<uint8_t>(mmq::q8_bytes(rows, E));
        uint8_t* mhq = dev<uint8_t>(mmq::q8_bytes(rows, FF));
        float* gu = dev<float>((size_t) rows * 2 * FF);
        float* mo = dev<float>((size_t) rows * E);
        float* xs = dev<float>((size_t) rows + 512);
        mmq::quantize(x, nullptr, mxq, c.gu_type, E, E, rows, s, xs);
        mmq::Product p;
        p.w = w;
        p.type = c.gu_type;
        p.w_rows = 2 * FF;
        p.w_cols = E;
        p.expert_bytes = c.stride;
        p.n = c.n_exp;
        p.xq = mxq - (size_t) row0 * mmq::q8_block_bytes();
        p.bounds = bounds;
        p.ids = ids;
        p.total_rows = rows;
        p.max_rows = max_rows;
        p.dst = gu - (size_t) row0 * 2 * FF;
        p.ld_dst = 2 * FF;
        p.y_scale = xs - row0;
        if (mmq::glu_supported(c.gu_type, FF) && c.up_off == mmq::matrix_bytes(c.gu_type, FF, E)) {
            mc.run_glu(p, c.limit, s);
            mmq::quantize(gu, nullptr, mhq, c.d_type, FF, 2 * FF, rows, s, xs);
            p.w = w + c.down_off;
            p.type = c.d_type;
            p.w_rows = E;
            p.w_cols = FF;
            p.xq = mhq - (size_t) row0 * mmq::q8_block_bytes();
            p.dst = mo - (size_t) row0 * E;
            p.ld_dst = E;
            mc.run(p, s);
            check(cudaDeviceSynchronize(), "sync");
            const std::vector<float> gu_h = down(gu, (size_t) rows * 2 * FF), mo_h = down(mo, (size_t) rows * E);
            double dh = 0, nh = 0, dd = 0, nd = 0;
            for (int r = 0; r < rows; ++r) {
                for (int k = 0; k < FF; ++k) {
                    const double a = gu_h[(size_t) r * 2 * FF + k], b = h_h[(size_t) r * FF + k];
                    dh += (a - b) * (a - b);
                    nh += a * a;
                }
                for (int k = 0; k < E; ++k) {
                    const double a = mo_h[(size_t) r * E + k], b = out_h[(size_t) r * E + k];
                    dd += (a - b) * (a - b);
                    nd += a * a;
                }
            }
            const double eh = std::sqrt(dh / (nh + 1e-30)), ed = std::sqrt(dd / (nd + 1e-30));
            if (verbose) std::printf("%-28s       vs MMQ: h differs %.2e of its norm, the output %.2e\n", name, eh, ed);
            // (MXFP4 on Blackwell: MMQ rounds the activations to FP4, not int8 - its output is ~12% of the norm away
            // from the int8 one; the reference above is what the own kernel is held to)
            const bool fp4 = c.gu_type == qt::MXFP4 || c.d_type == qt::MXFP4;
            if (!fp4 && (!(eh < 2e-2) || !(ed < 5e-2))) {
                std::printf("%-28s FAIL  vs MMQ: h differs %.2e of its norm, the output %.2e\n", name, eh, ed);
                ok = false;
            }
        }
        cudaFree(ids); cudaFree(mxq); cudaFree(mhq); cudaFree(gu); cudaFree(mo); cudaFree(xs);
    }
#endif
    cudaFree(w); cudaFree(bounds); cudaFree(x); cudaFree(xq); cudaFree(hq); cudaFree(h); cudaFree(out);
    return ok;
}

Case synthetic(int gu_type, int d_type, int n_embd, int n_ff, const std::vector<int>& counts, int row0, float limit,
               std::mt19937& rng) {
    Case c;
    c.gu_type = gu_type;
    c.d_type = d_type;
    c.n_embd = n_embd;
    c.n_ff = n_ff;
    c.n_exp = (int) counts.size();
    c.limit = limit;
    c.up_off = qt::row_size(gu_type, n_embd) * (size_t) n_ff;
    c.down_off = 2 * c.up_off;
    const size_t blob = c.down_off + qt::row_size(d_type, n_ff) * (size_t) n_embd;
    // (the experts lie whole blocks of both types apart: MMQ addresses expert e in blocks; rows stay unaligned)
    const size_t al = std::lcm<size_t>(std::lcm<size_t>(16, qt::type_size(gu_type)), qt::type_size(d_type));
    c.stride = (blob + al - 1) / al * al + al;
    c.w.assign(c.stride * (size_t) c.n_exp, 0);
    c.bounds.push_back(row0);
    for (int e = 0; e < c.n_exp; ++e) {
        uint8_t* b = c.w.data() + (size_t) e * c.stride;
        random_rows(gu_type, b, 2 * n_ff, n_embd, rng);
        random_rows(d_type, b + c.down_off, n_embd, n_ff, rng);
        c.bounds.push_back(c.bounds.back() + counts[(size_t) e]);
    }
    std::normal_distribution<float> nd(0.f, 1.0f);
    c.x.resize((size_t) c.rows() * n_embd);
    for (auto& v : c.x) v = nd(rng);
    if (c.rows() > 2) {   // a zero stretch, a zero row, a block of one large value
        std::fill(c.x.begin() + 64, c.x.begin() + 96, 0.0f);
        std::fill(c.x.begin() + n_embd, c.x.begin() + 2 * n_embd, 0.0f);
        c.x[(size_t) 2 * n_embd + 5] = 300.0f;
    }
    return c;
}

// STRATA_PROMPT_OWN_DUMP's file: int32 {magic 'OWN1', gu_type, d_type, n_embd, n_ff, n_exp, rows, row0}, float limit,
// uint64 {up_off, down_off, stride}, bounds, weights, x rows
bool load_dump(const char* path, Case& c) {
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return false;
    int32_t hd[8];
    uint64_t off[3];
    bool ok = std::fread(hd, 4, 8, f) == 8 && hd[0] == 0x314E574F && std::fread(&c.limit, 4, 1, f) == 1 &&
              std::fread(off, 8, 3, f) == 3;
    if (ok) {
        c.gu_type = hd[1]; c.d_type = hd[2]; c.n_embd = hd[3]; c.n_ff = hd[4]; c.n_exp = hd[5];
        c.up_off = off[0]; c.down_off = off[1]; c.stride = off[2];
        c.bounds.resize((size_t) c.n_exp + 1);
        c.w.resize(c.stride * (size_t) c.n_exp);
        c.x.resize((size_t) hd[6] * c.n_embd);
        ok = std::fread(c.bounds.data(), 4, c.bounds.size(), f) == c.bounds.size() &&
             std::fread(c.w.data(), 1, c.w.size(), f) == c.w.size() &&
             std::fread(c.x.data(), 4, c.x.size(), f) == c.x.size();
    }
    std::fclose(f);
    return ok;
}

int bench_case(const Case& c);
int bench(int gu_type, int d_type, int E, int FF, int n_exp, double mean_rows) {
    std::mt19937 rng(7);
    std::exponential_distribution<double> ed(1.0 / mean_rows);
    std::vector<int> counts((size_t) n_exp);
    for (auto& v : counts) v = (int) std::min(8.0 * mean_rows, ed(rng));
    return bench_case(synthetic(gu_type, d_type, E, FF, counts, 0, INFINITY, rng));
}
// kernel times of the case's set, own against MMQ (the best of 7)
int bench_case(const Case& c) {
    const int gu_type = c.gu_type, d_type = c.d_type, E = c.n_embd, FF = c.n_ff, n_exp = c.n_exp, row0 = c.bounds.front();
    const int rows = c.rows();
    int max_rows = 0;
    for (int e = 0; e < n_exp; ++e) max_rows = std::max(max_rows, c.bounds[(size_t) e + 1] - c.bounds[(size_t) e]);
    const cudaStream_t s = nullptr;
    uint8_t* w = up(c.w);
    int32_t* bounds = up(c.bounds);
    float* x = up(c.x);
    uint8_t* xq = dev<uint8_t>(std::max(own::act_bytes(rows, E) * 2, (size_t) rows * FF * 4));
    uint8_t* hq = dev<uint8_t>(own::act_bytes(rows, FF) * 2);
    float* gu = dev<float>((size_t) rows * 2 * FF);
    float* out = dev<float>((size_t) rows * E);
    const auto time = [&](const char* nm, const auto& fn) {
        double best = 1e30;
        for (int it = 0; it < 7; ++it) {
            check(cudaDeviceSynchronize(), "sync");
            const auto t0 = std::chrono::steady_clock::now();
            fn();
            check(cudaDeviceSynchronize(), "sync");
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::printf("  %-22s %8.3f ms\n", nm, best);
        return best;
    };
    std::printf("bench: %s / %s, n_embd %d n_ff %d, %d experts, %d rows (max %d)\n", qt::type_name(gu_type),
                qt::type_name(d_type), E, FF, n_exp, rows, max_rows);
    own::Context ctx;
    own::Set set;
    set.w = w; set.stride = c.stride; set.n = n_exp; set.bounds = bounds; set.host_bounds = c.bounds.data(); set.rows = rows;
    set.row0 = row0;
    double o_tot = 0, m_tot = 0;
    o_tot += time("own quantize x", [&] { own::quantize(x, nullptr, xq, E, E, rows, s); });
    o_tot += time("own gate/up+swiglu", [&] { set.xq = xq; set.dst = gu; set.ld = 2 * FF; ctx.gate_up(set, gu_type, FF, E, c.up_off, INFINITY, s); });
    o_tot += time("own quantize h", [&] { own::quantize(gu, nullptr, hq, FF, 2 * FF, rows, s); });
    o_tot += time("own down", [&] { set.xq = hq; set.dst = out; set.ld = E; ctx.down(set, d_type, E, FF, c.down_off, s); });
#if defined(STRATA_TEST_HAVE_MMQ)
    namespace mmq = strata::prefill::mmq;
    if (mmq::built() && mmq::supported(gu_type) && mmq::supported(d_type)) {
        const bool glu = mmq::glu_supported(gu_type, FF);
        mmq::Context mc;
        const int64_t n_ids = rows + mmq::ids_pad(max_rows) + row0;
        int32_t* ids = dev<int32_t>((size_t) n_ids);
        mmq::iota(ids, n_ids, s);
        uint8_t* mxq = dev<uint8_t>(mmq::q8_bytes(rows, E));
        uint8_t* mhq = dev<uint8_t>(mmq::q8_bytes(rows, FF));
        float* xs = dev<float>((size_t) rows + 512);
        mmq::Product p;
        p.w = w; p.type = gu_type; p.w_rows = 2 * FF; p.w_cols = E; p.expert_bytes = c.stride; p.n = n_exp;
        p.xq = mxq - (size_t) row0 * mmq::q8_block_bytes();
        p.bounds = bounds; p.ids = ids; p.total_rows = rows; p.max_rows = max_rows; p.dst = gu - (size_t) row0 * 2 * FF;
        p.ld_dst = 2 * FF; p.y_scale = xs - row0;
        mmq::Product d = p;
        d.w = w + c.down_off; d.type = d_type; d.w_rows = E; d.w_cols = FF; d.xq = mhq - (size_t) row0 * mmq::q8_block_bytes();
        d.dst = out - (size_t) row0 * E; d.ld_dst = E;
        m_tot += time("mmq quantize x", [&] { mmq::quantize(x, nullptr, mxq, gu_type, E, E, rows, s, xs); });
        if (glu) m_tot += time("mmq gate/up+swiglu", [&] { mc.run_glu(p, INFINITY, s); });
        else m_tot += time("mmq gate/up, swiglu", [&] { mc.run(p, s); mmq::swiglu(gu, (float*) xq, rows, FF, false, s); });
        m_tot += time("mmq quantize h", [&] { mmq::quantize(gu, nullptr, mhq, d_type, FF, 2 * FF, rows, s, xs); });
        m_tot += time("mmq down", [&] { mc.run(d, s); });
        std::printf("  total: own %.3f ms, mmq %.3f ms (own / mmq = %.3f)\n", o_tot, m_tot, o_tot / m_tot);
    }
#endif
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (!own::built()) {
        std::printf("moe_own_test: this build or device has no own prompt kernels - skipped\n");
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--bench") {
        if (argc >= 8) return bench(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]),
                                    std::atoi(argv[6]), std::atof(argv[7]));
        for (double mean : {6.0, 114.0, 910.0}) {
            bench(qt::IQ2_XXS, qt::IQ3_XXS, 4096, 2048, mean > 500 ? 96 : 288, mean);
            bench(qt::IQ2_XS, qt::IQ3_XXS, 4096, 2048, mean > 500 ? 96 : 288, mean);
            bench(qt::MXFP4, qt::MXFP4, 4096, 2048, mean > 500 ? 96 : 288, mean);
        }
        return 0;
    }
    if (argc >= 3 && std::string(argv[1]) == "--dump") {
        int bad = 0;
        for (int i = 2; i < argc; ++i) {
            Case c;
            if (!load_dump(argv[i], c)) {
                std::fprintf(stderr, "moe_own_test: cannot read %s\n", argv[i]);
                return 2;
            }
            std::printf("%s: %s / %s, n_embd %d n_ff %d, %d experts\n", argv[i], qt::type_name(c.gu_type),
                        qt::type_name(c.d_type), c.n_embd, c.n_ff, c.n_exp);
            if (!own::usable(c.gu_type, c.n_ff, c.n_embd) || !own::usable(c.d_type, c.n_embd, c.n_ff)) {
                std::printf("  not covered by the own kernels - skipped\n");
                continue;
            }
            if (!run_case(c, "real tensors", true, 64)) ++bad;
            bench_case(c);
        }
        return bad ? 1 : 0;
    }
    std::mt19937 rng(20261116);
    const int types[] = {qt::IQ2_XXS, qt::IQ2_XS, qt::IQ2_S, qt::IQ3_XXS, qt::MXFP4};
    int bad = 0;
    for (int gt : types)
        for (int dt : types) {
            if (gt != dt && !(gt == qt::IQ2_XXS || dt == qt::IQ3_XXS)) continue;   // (the pairs the packs have, and each alone)
            char name[64];
            // every tile shape (1, 2, 4, 8 groups; a second tile), an empty expert; bounds that do not start at 0
            std::snprintf(name, sizeof name, "%s/%s 1024x512", qt::type_name(gt), qt::type_name(dt));
            if (!run_case(synthetic(gt, dt, 1024, 512, {0, 3, 9, 20, 70, 131, 1, 64, 0, 8}, 5, INFINITY, rng), name, true)) ++bad;
            // a clamped SwiGLU, a smaller shape
            std::snprintf(name, sizeof name, "%s/%s 512x256 limit", qt::type_name(gt), qt::type_name(dt));
            if (!run_case(synthetic(gt, dt, 512, 256, {33, 0, 17}, 0, 0.75f, rng), name, true)) ++bad;
        }
    std::printf(bad ? "moe_own_test: %d FAILED\n" : "moe_own_test: all passed\n", bad);
    return bad ? 1 : 0;
}
