// tests/hip/own_gemm_test.cu - the HIP build's own GEMM (hip_gemm.hpp) against a double-precision reference of the
// same rounded inputs: 16-bit and F32 inputs, F32 and FP16 outputs, beta 0 / 1, strided batches, unaligned leading
// dimensions, tails in every dimension.  Also against hipBLAS (`--lib`), and timed (`--bench`, the prompt path's
// shapes, own vs hipBLAS; STRATA_HIP_GEMM_PORTABLE=1 times the plain kernel).
#include "strata/hip_compat/hip_gemm.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace hg = strata::hipgemm;

namespace {

#define HK(x)                                                                                    \
    do {                                                                                         \
        hipError_t e_ = (x);                                                                     \
        if (e_ != hipSuccess) {                                                                  \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }
float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
uint16_t f2bf(float f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}
float bf2f(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Case {
    int m, n, k, lda_pad, ldb_pad, ldc_pad, batch;
    hipDataType in, out;
    float alpha, beta;
};

const char* tn(hipDataType t) { return t == HIP_R_16F ? "f16" : t == HIP_R_16BF ? "bf16" : "f32"; }
size_t esz(hipDataType t) { return t == HIP_R_32F ? 4 : 2; }

// fills host values already rounded to the type, and their bits
void fill(std::vector<float>& v, std::vector<uint8_t>& bits, hipDataType t, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    bits.resize(v.size() * esz(t));
    for (size_t i = 0; i < v.size(); ++i) {
        float x = d(rng);
        if (t == HIP_R_16F) {
            const uint16_t h = f2h(x);
            x = h2f(h);
            std::memcpy(&bits[i * 2], &h, 2);
        } else if (t == HIP_R_16BF) {
            const uint16_t h = f2bf(x);
            x = bf2f(h);
            std::memcpy(&bits[i * 2], &h, 2);
        } else {
            std::memcpy(&bits[i * 4], &x, 4);
        }
        v[i] = x;
    }
}

bool run_case(const Case& c, std::mt19937& rng, bool lib, hipblasHandle_t hb) {
    const int lda = c.k + c.lda_pad, ldb = c.k + c.ldb_pad, ldc = c.m + c.ldc_pad;
    const long long sA = (long long) lda * c.m + (c.lda_pad ? 3 : 0), sB = (long long) ldb * c.n + (c.ldb_pad ? 5 : 0),
                    sC = (long long) ldc * c.n + 7;
    std::vector<float> A((size_t) sA * c.batch), B((size_t) sB * c.batch), C0((size_t) sC * c.batch);
    std::vector<uint8_t> Ab, Bb, Cb;
    fill(A, Ab, c.in, rng);
    fill(B, Bb, c.in, rng);
    fill(C0, Cb, c.out, rng);
    void *dA, *dB, *dC, *dL = nullptr;
    HK(hipMalloc(&dA, Ab.size()));
    HK(hipMalloc(&dB, Bb.size()));
    HK(hipMalloc(&dC, Cb.size()));
    HK(hipMemcpy(dA, Ab.data(), Ab.size(), hipMemcpyHostToDevice));
    HK(hipMemcpy(dB, Bb.data(), Bb.size(), hipMemcpyHostToDevice));
    HK(hipMemcpy(dC, Cb.data(), Cb.size(), hipMemcpyHostToDevice));
    if (!hg::own_gemm(nullptr, c.m, c.n, c.k, c.alpha, dA, c.in, lda, sA, dB, c.in, ldb, sB, c.beta, dC, c.out, ldc,
                      sC, c.batch)) {
        std::printf("FAIL not covered\n");
        return false;
    }
    HK(hipDeviceSynchronize());
    std::vector<uint8_t> Ob(Cb.size()), Lb;
    HK(hipMemcpy(Ob.data(), dC, Ob.size(), hipMemcpyDeviceToHost));
    if (lib) {
        HK(hipMalloc(&dL, Cb.size()));
        HK(hipMemcpy(dL, Cb.data(), Cb.size(), hipMemcpyHostToDevice));
        const hipblasStatus_t st = hipblasGemmStridedBatchedEx(
            hb, HIPBLAS_OP_T, HIPBLAS_OP_N, c.m, c.n, c.k, &c.alpha, dA, c.in, lda, sA, dB, c.in, ldb, sB, &c.beta, dL,
            c.out, ldc, sC, c.batch, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
        if (st != HIPBLAS_STATUS_SUCCESS) std::printf("(hipBLAS status %d) ", (int) st);
        HK(hipDeviceSynchronize());
        Lb.resize(Cb.size());
        HK(hipMemcpy(Lb.data(), dL, Lb.size(), hipMemcpyDeviceToHost));
    }
    const auto outv = [&](const std::vector<uint8_t>& b, size_t i) -> double {
        if (c.out == HIP_R_32F) {
            float f;
            std::memcpy(&f, &b[i * 4], 4);
            return f;
        }
        uint16_t h;
        std::memcpy(&h, &b[i * 2], 2);
        return h2f(h);
    };
    double worst = 0.0, lib_worst = 0.0;
    bool ok = true, untouched = true;
    for (int z = 0; z < c.batch; ++z)
        for (int j = 0; j < c.n; ++j) {
            for (int i = 0; i < c.m; ++i) {
                double s = 0.0, sa = 0.0;
                const float* a = &A[(size_t) z * sA + (size_t) i * lda];
                const float* b = &B[(size_t) z * sB + (size_t) j * ldb];
                for (int kk = 0; kk < c.k; ++kk) {
                    s += (double) a[kk] * b[kk];
                    sa += std::fabs((double) a[kk] * b[kk]);
                }
                const size_t ci = (size_t) z * sC + (size_t) j * ldc + i;
                const double want = c.alpha * s + c.beta * (double) C0[ci];
                const double scale = std::fabs(c.alpha) * sa + std::fabs(c.beta * C0[ci]) + 1e-6;
                const double acc_tol = 2e-6 * scale + 1e-6 * std::sqrt((double) c.k + 1) * scale;
                const double tol = c.out == HIP_R_16F ? 6e-4 * std::fabs(want) + acc_tol + 1e-7 : acc_tol;
                const double got = outv(Ob, ci), err = std::fabs(got - want);
                if (!(err <= tol)) ok = false;
                worst = std::max(worst, err / scale);
                if (lib) lib_worst = std::max(lib_worst, std::fabs(outv(Lb, ci) - want) / scale);
            }
            for (int i = c.m; i < ldc; ++i) {   // the padding must stay as it was
                const size_t ci = (size_t) z * sC + (size_t) j * ldc + i;
                if (std::memcmp(&Ob[ci * esz(c.out)], &Cb[ci * esz(c.out)], esz(c.out)) != 0) untouched = false;
            }
        }
    std::printf("%s m %5d n %5d k %5d lda+%d ldb+%d ldc+%d batch %d %s->%s alpha %.3g beta %.3g: worst %.2e%s%s %s\n",
                ok && untouched ? "ok  " : "FAIL", c.m, c.n, c.k, c.lda_pad, c.ldb_pad, c.ldc_pad, c.batch, tn(c.in),
                tn(c.out), c.alpha, c.beta, worst, lib ? " (hipBLAS " : "",
                lib ? (std::to_string(lib_worst).substr(0, 10) + ")").c_str() : "", untouched ? "" : "PADDING WRITTEN");
    HK(hipFree(dA));
    HK(hipFree(dB));
    HK(hipFree(dC));
    if (dL) HK(hipFree(dL));
    return ok && untouched;
}

void bench(hipblasHandle_t hb, bool lib, bool small) {
    struct S { int m, n, k; hipDataType in; };
    const S big_shapes[] = {
        {8192, 8192, 8192, HIP_R_16F}, {8192, 8192, 8192, HIP_R_16BF}, {8192, 8192, 8192, HIP_R_32F}};
    // few tokens: where a WMMA tile of 128 tokens is mostly idle
    const S small_shapes[] = {
        {4096, 1, 4096, HIP_R_16F},   {4096, 4, 4096, HIP_R_16F},   {4096, 8, 4096, HIP_R_16F},
        {4096, 16, 4096, HIP_R_16F},  {4096, 32, 4096, HIP_R_16F},  {4096, 48, 4096, HIP_R_16F},
        {4096, 64, 4096, HIP_R_16F},  {4096, 96, 4096, HIP_R_16F},  {4096, 128, 4096, HIP_R_16F},
        {4096, 256, 4096, HIP_R_16F}, {12288, 16, 4096, HIP_R_16F}, {12288, 64, 4096, HIP_R_16F},
        {512, 16, 4096, HIP_R_16F},   {512, 64, 4096, HIP_R_16F}};
    const S std_shapes[] = {
        {2048, 2048, 2048, HIP_R_16F}, {4096, 2048, 4096, HIP_R_16F}, {12288, 2048, 4096, HIP_R_16F},
        {4096, 512, 4096, HIP_R_16F},  {512, 2048, 4096, HIP_R_16F},  {4096, 4096, 1536, HIP_R_16F},
        {2048, 2048, 2048, HIP_R_16BF}, {64, 2048, 4096, HIP_R_32F},  {256, 2048, 3072, HIP_R_32F},
        {2048, 2048, 2048, HIP_R_32F},
    };
    std::vector<S> shapes;
    if (small) shapes.assign(std::begin(small_shapes), std::end(small_shapes));
    else {
        shapes.assign(std::begin(std_shapes), std::end(std_shapes));
        shapes.insert(shapes.end(), std::begin(big_shapes), std::end(big_shapes));
    }
    for (const S& s : shapes) {
        void *dA, *dB, *dC;
        HK(hipMalloc(&dA, (size_t) s.m * s.k * esz(s.in)));
        HK(hipMalloc(&dB, (size_t) s.n * s.k * esz(s.in)));
        HK(hipMalloc(&dC, (size_t) s.m * s.n * 4));
        HK(hipMemset(dA, 0x11, (size_t) s.m * s.k * esz(s.in)));
        HK(hipMemset(dB, 0x22, (size_t) s.n * s.k * esz(s.in)));
        const float one = 1.0f, zero = 0.0f;
        const auto time = [&](auto&& f) {
            for (int w = 0; w < 3; ++w) f();   // (fresh buffers on an APU settle over the first calls)
            HK(hipDeviceSynchronize());
            const int reps = 10;
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) f();
            HK(hipDeviceSynchronize());
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
        };
        const double t_own = time([&] {
            hg::own_gemm(nullptr, s.m, s.n, s.k, 1.0f, dA, s.in, s.k, 0, dB, s.in, s.k, 0, 0.0f, dC, HIP_R_32F, s.m, 0, 1);
        });
        double t_lib = 0.0;
        if (lib)
            t_lib = time([&] {
                hipblasGemmEx(hb, HIPBLAS_OP_T, HIPBLAS_OP_N, s.m, s.n, s.k, &one, dA, s.in, s.k, dB, s.in, s.k, &zero, dC,
                              HIP_R_32F, s.m, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
            });
        const double fl = 2.0 * s.m * s.n * s.k;
        std::printf("bench %-4s m %5d n %5d k %5d: own %8.3f ms %6.2f TFLOPS", tn(s.in), s.m, s.n, s.k, t_own * 1e3,
                    fl / t_own * 1e-12);
        if (lib) std::printf(" | hipBLAS %7.3f ms %6.2f TFLOPS", t_lib * 1e3, fl / t_lib * 1e-12);
        std::printf("\n");
        HK(hipFree(dA));
        HK(hipFree(dB));
        HK(hipFree(dC));
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool lib = false, do_bench = false, small = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--lib")) lib = true;
        if (!std::strcmp(argv[i], "--bench")) do_bench = true;
        if (!std::strcmp(argv[i], "--small")) do_bench = small = true;
    }
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
        std::printf("no HIP device - skipped\n");
        return 77;
    }
    std::printf("kernel for 16-bit inputs on HIP0: %s\n", hg::own_kernel(0));
    hipblasHandle_t hb = nullptr;
    if ((lib || do_bench) && hipblasCreate(&hb) != HIPBLAS_STATUS_SUCCESS) {
        std::printf("hipblasCreate failed - no library comparison\n");
        lib = false;
    }
    if (do_bench) {
        bench(hb, hb != nullptr, small);
        if (hb) hipblasDestroy(hb);
        return 0;
    }
    std::mt19937 rng(1234);
    std::vector<Case> cases;
    for (hipDataType in : {HIP_R_16F, HIP_R_16BF, HIP_R_32F}) {
        const hipDataType outs[2] = {HIP_R_32F, HIP_R_16F};
        for (int oi = 0; oi < (in == HIP_R_32F ? 1 : 2); ++oi) {
            const hipDataType out = outs[oi];
            cases.push_back({128, 128, 64, 0, 0, 0, 1, in, out, 1.0f, 0.0f});
            cases.push_back({300, 77, 200, 0, 0, 5, 1, in, out, 1.0f, 1.0f});
            cases.push_back({1, 1, 1, 0, 0, 0, 1, in, out, 1.0f, 0.0f});
            cases.push_back({17, 33, 9, 3, 1, 2, 1, in, out, 0.5f, 0.0f});   // unaligned leading dimensions
            cases.push_back({512, 300, 4096, 0, 0, 0, 1, in, out, 1.0f, 0.0f});
            cases.push_back({257, 129, 1000, 0, 8, 0, 3, in, out, 0.125f, 1.0f});   // batched
            cases.push_back({64, 1100, 576, 0, 0, 0, 2, in, out, 1.0f, 0.0f});
            cases.push_back({200, 50, 0, 0, 0, 0, 1, in, out, 1.0f, 1.0f});   // k = 0: C = beta C
        }
    }
    int bad = 0;
    for (const Case& c : cases) bad += run_case(c, rng, lib, hb) ? 0 : 1;
    if (hb) hipblasDestroy(hb);
    std::printf("%s: %d of %zu cases failed\n", bad ? "FAIL" : "PASS", bad, cases.size());
    return bad ? 1 : 0;
}
