// tests/dsv4_idx_parity.cu - the lightning indexer's tensor-core scores (idx_score_mma_kernel<H>,
// idx_score_batch_mma_kernel<H>, src/kernels/cuda/dsv4_fast.cu) against the FP32 kernels and a double reference, for
// 32 heads (deepseek41) and 64 (deepseek4): the largest score error relative to the row's largest |score|, how many of
// the FP32 kernel's top 512 rows the tensor-core scores keep, and both kernels' times at the context sizes that matter.
// HIP (gfx11 / gfx12): the WMMA kernels (idx_score_wmma_kernel<H>, idx_score_batch_wmma2_kernel<H>) in their place.
// The .cu is included whole (its kernels are in an unnamed namespace).
//   nvcc -O3 -std=c++17 -arch=sm_120a -I include -I src/kernels/cuda -I third_party/ggml tests/dsv4_idx_parity.cu \
//        -o build/dsv4_idx_parity
//   hipcc -O3 -std=c++20 --offload-arch=gfx1151 -DSTRATA_USE_HIP=1 -D__HIP_PLATFORM_AMD__=1 -I include \
//        -I include/strata/hip_compat -include include/strata/hip_compat/cuda_runtime.h -I src/kernels/cuda \
//        -I third_party/ggml -x hip tests/dsv4_idx_parity.cu -o build/dsv4_idx_parity_hip
#include "../src/kernels/cuda/dsv4_fast.cu"

#include <random>

using namespace strata::kernels::dsv4;

namespace {

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(1);
    }
}

// the share of a's top `top` rows that are in b's top `top` (scores descending, ties by the lower index)
double top_overlap(const std::vector<float>& a, const std::vector<float>& b, int n, int top) {
    top = std::min(top, n);
    auto order = [&](const std::vector<float>& s) {
        std::vector<int> o(n);
        std::iota(o.begin(), o.end(), 0);
        std::partial_sort(o.begin(), o.begin() + top, o.end(),
                          [&](int x, int y) { return s[x] > s[y] || (s[x] == s[y] && x < y); });
        o.resize(top);
        std::sort(o.begin(), o.end());
        return o;
    };
    const std::vector<int> oa = order(a), ob = order(b);
    std::vector<int> both;
    std::set_intersection(oa.begin(), oa.end(), ob.begin(), ob.end(), std::back_inserter(both));
    return (double) both.size() / top;
}

struct Data {
    int H, n;
    std::vector<float> q, w;    // T x H x 128, T x H
    std::vector<uint16_t> k;    // n x 128 (FP16)
};
Data make(int H, int n, int T, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    Data d{H, n, std::vector<float>((size_t) T * H * 128), std::vector<float>((size_t) T * H),
           std::vector<uint16_t>((size_t) n * 128)};
    for (auto& x : d.q) x = 0.15f * nd(rng);
    for (auto& x : d.w) x = 0.08f * nd(rng);
    for (auto& x : d.k) x = __half_as_ushort(__float2half(nd(rng)));
    return d;
}
double ref_score(const Data& d, int t, int j) {
    double acc = 0.0;
    for (int h = 0; h < d.H; ++h) {
        double s = 0.0;
        for (int c = 0; c < 128; ++c)
            s += (double) d.q[((size_t) t * d.H + h) * 128 + c] * (double) __half2float(__ushort_as_half(d.k[(size_t) j * 128 + c]));
        acc += std::max(s, 0.0) * d.w[(size_t) t * d.H + h];
    }
    return acc;
}

template <int H>
void decode_case(int n, int iters) {
    Data d = make(H, n, 1, 1234u + n + H);
    float *q, *w, *s0, *s1;
    uint16_t* k;
    ck(cudaMalloc(&q, d.q.size() * 4), "m");
    ck(cudaMalloc(&w, d.w.size() * 4), "m");
    ck(cudaMalloc(&k, d.k.size() * 2), "m");
    ck(cudaMalloc(&s0, (size_t) n * 4), "m");
    ck(cudaMalloc(&s1, (size_t) n * 4), "m");
    cudaMemcpy(q, d.q.data(), d.q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w, d.w.data(), d.w.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(k, d.k.data(), d.k.size() * 2, cudaMemcpyHostToDevice);
    const size_t sm = (size_t) (H * 128 + H) * sizeof(float);
    auto run0 = [&] { idx_score_kernel<<<(n + IDX_WARPS - 1) / IDX_WARPS, IDX_WARPS * 32, sm>>>(q, k, w, H, n, s0); };
#if defined(STRATA_USE_HIP)
    auto run1 = [&] {
        idx_score_wmma_kernel<H><<<(n + IW_WAVES * 16 - 1) / (IW_WAVES * 16), IW_WAVES * 32>>>(q, k, w, n, s1);
    };
#else
    auto run1 = [&] {
        idx_score_mma_kernel<H><<<(n + IDXM_WARPS * 16 - 1) / (IDXM_WARPS * 16), IDXM_WARPS * 32>>>(q, k, w, n, s1);
    };
#endif
    run0();
    run1();
    ck(cudaDeviceSynchronize(), "run");
    std::vector<float> a(n), b(n);
    cudaMemcpy(a.data(), s0, (size_t) n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), s1, (size_t) n * 4, cudaMemcpyDeviceToHost);
    double mx = 0.0, e0 = 0.0, e1 = 0.0, e01 = 0.0;
    for (int j = 0; j < n; j += std::max(1, n / 2048)) {   // the double reference on a sample of rows
        const double r = ref_score(d, 0, j);
        mx = std::max(mx, std::fabs(r));
        e0 = std::max(e0, std::fabs(a[j] - r));
        e1 = std::max(e1, std::fabs(b[j] - r));
    }
    for (int j = 0; j < n; ++j) e01 = std::max(e01, (double) std::fabs(a[j] - b[j]));
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    float t0 = 0, t1 = 0;
    cudaEventRecord(ev0);
    for (int i = 0; i < iters; ++i) run0();
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&t0, ev0, ev1);
    cudaEventRecord(ev0);
    for (int i = 0; i < iters; ++i) run1();
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&t1, ev0, ev1);
    std::printf("decode H %d rows %6d | err vs double: fp32 %.2e mma %.2e (of max %.3f) | fp32 vs mma %.2e | top512 "
                "kept %.4f | fp32 %7.1f us  mma %6.1f us\n",
                H, n, e0 / mx, e1 / mx, mx, e01 / mx, top_overlap(a, b, n, 512), 1000.0f * t0 / iters,
                1000.0f * t1 / iters);
    cudaFree(q); cudaFree(w); cudaFree(k); cudaFree(s0); cudaFree(s1);
}

// T tokens at positions p0.. (ratio 4): every token's visible rows compared, -inf / unwritten rows past n_vis ignored
template <int H>
void batch_case(int p0, int T, int iters) {
    const int ratio = 4, ld = (p0 + T) / ratio;
    Data d = make(H, ld, T, 99u + p0 + T + H);
    float *q, *w, *s0, *s1;
    uint16_t* k;
    ck(cudaMalloc(&q, d.q.size() * 4), "m");
    ck(cudaMalloc(&w, d.w.size() * 4), "m");
    ck(cudaMalloc(&k, d.k.size() * 2), "m");
    ck(cudaMalloc(&s0, (size_t) T * ld * 4), "m");
    ck(cudaMalloc(&s1, (size_t) T * ld * 4), "m");
    cudaMemcpy(q, d.q.data(), d.q.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(w, d.w.data(), d.w.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(k, d.k.data(), d.k.size() * 2, cudaMemcpyHostToDevice);
#if defined(STRATA_USE_HIP)
    float* q2;   // (the WMMA kernel reads Q converted in place)
    ck(cudaMalloc(&q2, d.q.size() * 4), "m");
    cudaMemcpy(q2, d.q.data(), d.q.size() * 4, cudaMemcpyHostToDevice);
    iwb_prep_kernel<<<(unsigned) ((T * H + 7) / 8), 256>>>(q2, T * H);
#else
    if (H == 64)
        ck(cudaFuncSetAttribute(idx_score_batch_mma_kernel<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, IBM_SMEM),
           "attr");
#endif
    auto run0 = [&] {
        const dim3 g((unsigned) ((ld + IRT - 1) / IRT), (unsigned) ((T + IQT - 1) / IQT));
        idx_score_batch_kernel<<<g, 256>>>(q, k, w, H, ratio, p0, T, s0, ld);
    };
#if defined(STRATA_USE_HIP)
    auto run1 = [&] {
        const dim3 g((unsigned) ((ld + IWB2_ROWS - 1) / IWB2_ROWS), (unsigned) ((T + IWB2_TT - 1) / IWB2_TT));
        idx_score_batch_wmma2_kernel<H><<<g, IWB2_WAVES * 32>>>(q2, k, w, ratio, p0, T, s1, ld);
    };
#else
    auto run1 = [&] {
        const dim3 g((unsigned) ((ld + IBM_ROWS - 1) / IBM_ROWS), (unsigned) ((T + IBM_TT - 1) / IBM_TT));
        idx_score_batch_mma_kernel<H><<<g, IBM_WARPS * 32, H == 64 ? IBM_SMEM : IBM_SMEM / 2>>>(q, k, w, ratio, p0, T,
                                                                                               s1, ld);
    };
#endif
    run0();
    run1();
    ck(cudaDeviceSynchronize(), "run");
    std::vector<float> a((size_t) T * ld), b((size_t) T * ld);
    cudaMemcpy(a.data(), s0, a.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), s1, b.size() * 4, cudaMemcpyDeviceToHost);
    double worst = 0.0, kept = 1.0;
    for (int t = 0; t < T; ++t) {
        const int n_vis = (p0 + t + 1) / ratio;
        if (n_vis <= 0) continue;
        double mx = 1e-30, e = 0.0;
        for (int j = 0; j < n_vis; ++j) {
            mx = std::max(mx, (double) std::fabs(a[(size_t) t * ld + j]));
            e = std::max(e, (double) std::fabs(a[(size_t) t * ld + j] - b[(size_t) t * ld + j]));
        }
        worst = std::max(worst, e / mx);
        if (t % 97 == 0 || t == T - 1) {
            std::vector<float> ra(a.begin() + (size_t) t * ld, a.begin() + (size_t) t * ld + n_vis);
            std::vector<float> rb(b.begin() + (size_t) t * ld, b.begin() + (size_t) t * ld + n_vis);
            kept = std::min(kept, top_overlap(ra, rb, n_vis, 512));
        }
    }
    // the double reference on the last token
    double mx = 0.0, e0 = 0.0, e1 = 0.0;
    const int tl = T - 1, nl = (p0 + T) / ratio;
    for (int j = 0; j < nl; j += std::max(1, nl / 1024)) {
        const double r = ref_score(d, tl, j);
        mx = std::max(mx, std::fabs(r));
        e0 = std::max(e0, std::fabs(a[(size_t) tl * ld + j] - r));
        e1 = std::max(e1, std::fabs(b[(size_t) tl * ld + j] - r));
    }
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    float t0 = 0, t1 = 0;
    cudaEventRecord(ev0);
    for (int i = 0; i < iters; ++i) run0();
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&t0, ev0, ev1);
    cudaEventRecord(ev0);
    for (int i = 0; i < iters; ++i) run1();
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&t1, ev0, ev1);
    std::printf("batch  H %d p0 %6d T %4d | last token err vs double: fp32 %.2e mma %.2e | fp32 vs mma worst %.2e | "
                "top512 kept min %.4f | fp32 %8.2f ms  mma %7.2f ms\n",
                H, p0, T, e0 / mx, e1 / mx, worst, kept, t0 / iters, t1 / iters);
    cudaFree(q); cudaFree(w); cudaFree(k); cudaFree(s0); cudaFree(s1);
#if defined(STRATA_USE_HIP)
    cudaFree(q2);
#endif
}

}  // namespace

int main() {
    for (int n : {100, 2048, 4096, 8192, 16384, 32768, 65536, 131072}) decode_case<32>(n, 200);
    for (int n : {2048, 32768}) decode_case<64>(n, 200);
    batch_case<32>(0, 1000, 3);
    batch_case<32>(16384, 4096, 3);
    batch_case<32>(126976, 4096, 3);
    batch_case<64>(126976, 4096, 3);
    return 0;
}
