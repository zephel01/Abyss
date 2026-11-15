// tests/dsv4_attn_parity.cu - the prompt path's tensor-core attention: attend_batch_mma2_kernel (32 heads a block,
// cp.async stages, src/kernels/cuda/dsv4_fast.cu) against attend_batch_mma_kernel (16 heads a block): the outputs must
// be bit-identical; both kernels' times at the shapes deepseek4 / deepseek41 prompts take (64 heads, head_dim 512,
// window 128, top_k 512 of the compressed rows, or every visible one). The .cu is included whole (its kernels are in
// an unnamed namespace).
//   nvcc -O3 -std=c++17 -arch=sm_120a -I include -I src/kernels/cuda -I third_party/ggml tests/dsv4_attn_parity.cu \
//        -o build/dsv4_attn_parity
#include "../src/kernels/cuda/dsv4_fast.cu"

#include <cstring>
#include <random>

using namespace strata::kernels::dsv4;

namespace {

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Case {
    const char* name;
    int ratio;   // as attend_batch takes it: 4 / 128 (deepseek4), -1 / -2 (deepseek41)
    int p0, T;
};

void run(const Case& c, unsigned seed) {
    const int H = 64, D = 512, n_rot = 64, n_swa = 128, top_k = 512;
    const int n_hist = std::min(c.p0, n_swa - 1);
    const int last = c.p0 + c.T - 1;
    const int div = c.ratio < 0 ? -c.ratio : c.ratio == 0 ? 1 : c.ratio;
    const int n_comp = std::max(1, (last + 1) / div);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> q((size_t) c.T * H * D), sinks(H);
    for (auto& x : q) x = 0.08f * nd(rng);
    for (auto& x : sinks) x = nd(rng);
    std::vector<uint16_t> keys((size_t) (n_hist + c.T) * D), comp((size_t) n_comp * D);
    for (auto& x : keys) x = __half_as_ushort(__float2half(nd(rng)));
    for (auto& x : comp) x = __half_as_ushort(__float2half(nd(rng)));
    // a token's selection: top_k sorted rows of its visible ones, half near the end (as the indexer tends to pick)
    std::vector<int> sel((size_t) c.T * top_k, 0);
    for (int t = 0; t < c.T; ++t) {
        const int vis = (c.p0 + t + 1) / div;
        if (c.ratio == 0 || c.ratio == 128 || vis <= top_k) continue;
        std::vector<int> pick;
        std::vector<char> used(vis, 0);
        while ((int) pick.size() < top_k) {
            int r = (pick.size() & 1) ? (int) (rng() % (unsigned) vis) : vis - 1 - (int) (rng() % (unsigned) std::min(vis, 4096));
            if (!used[r]) {
                used[r] = 1;
                pick.push_back(r);
            }
        }
        std::sort(pick.begin(), pick.end());
        std::memcpy(&sel[(size_t) t * top_k], pick.data(), top_k * sizeof(int));
    }
    float *dq, *dsink, *o1, *o2;
    uint16_t *dk, *dc;
    int* dsel;
    ck(cudaMalloc(&dq, q.size() * 4), "m");
    ck(cudaMalloc(&dsink, H * 4), "m");
    ck(cudaMalloc(&o1, q.size() * 4), "m");
    ck(cudaMalloc(&o2, q.size() * 4), "m");
    ck(cudaMalloc(&dk, keys.size() * 2), "m");
    ck(cudaMalloc(&dc, comp.size() * 2), "m");
    ck(cudaMalloc(&dsel, sel.size() * 4), "m");
    ck(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "c");
    ck(cudaMemcpy(dsink, sinks.data(), H * 4, cudaMemcpyHostToDevice), "c");
    ck(cudaMemcpy(dk, keys.data(), keys.size() * 2, cudaMemcpyHostToDevice), "c");
    ck(cudaMemcpy(dc, comp.data(), comp.size() * 2, cudaMemcpyHostToDevice), "c");
    ck(cudaMemcpy(dsel, sel.data(), sel.size() * 4, cudaMemcpyHostToDevice), "c");
    ck(cudaMemset(o1, 0xff, q.size() * 4), "s");
    ck(cudaMemset(o2, 0xff, q.size() * 4), "s");
    const Rope rope = make_rope(n_rot, 10000.0f, 1.0f, 0.0f, 0, 0.0f, 0.0f);
    const float scale = 1.0f / sqrtf((float) D);
    ck(cudaFuncSetAttribute(attend_batch_mma2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) AB2_SMEM), "attr");
    auto old_k = [&] {
        attend_batch_mma_kernel<<<dim3((unsigned) c.T, (unsigned) (H / ABM_H)), 128>>>(
            dq, dk, n_hist, n_swa, dc, dsel, top_k, top_k, c.ratio, dsink, H, n_rot, rope, c.p0, scale, o1);
    };
    auto new_k = [&] {
        attend_batch_mma2_kernel<<<dim3((unsigned) (H / AB2_HEADS), (unsigned) c.T), 256, AB2_SMEM>>>(
            dq, dk, n_hist, n_swa, dc, dsel, top_k, top_k, c.ratio, dsink, H, n_rot, rope, c.p0, scale, o2);
    };
    old_k();
    new_k();
    ck(cudaDeviceSynchronize(), "run");
    std::vector<float> a(q.size()), b(q.size());
    ck(cudaMemcpy(a.data(), o1, a.size() * 4, cudaMemcpyDeviceToHost), "c");
    ck(cudaMemcpy(b.data(), o2, b.size() * 4, cudaMemcpyDeviceToHost), "c");
    size_t diff = 0;
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], 4) != 0) {
            ++diff;
            worst = std::max(worst, (double) std::fabs(a[i] - b[i]));
        }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    auto time = [&](auto&& f) {
        f();
        cudaEventRecord(e0);
        const int reps = 10;
        for (int r = 0; r < reps; ++r) f();
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), "t");
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms / reps;
    };
    const float t1 = time(old_k), t2 = time(new_k);
    std::printf("%-26s p0 %6d T %4d | %zu of %zu floats differ (max %.3g) | old %.3f ms  new %.3f ms (%.2fx)\n", c.name,
                c.p0, c.T, diff, a.size(), worst, t1, t2, t1 / t2);
    cudaFree(dq);
    cudaFree(dsink);
    cudaFree(o1);
    cudaFree(o2);
    cudaFree(dk);
    cudaFree(dc);
    cudaFree(dsel);
}

}   // namespace

int main() {
    const Case cases[] = {
        {"v41 ratio -2 (top_k)", -2, 126976, 2048}, {"v41 ratio -1 (top_k)", -1, 126976, 2048},
        {"v41 ratio -2 early", -2, 0, 2048},        {"v41 ratio -1 32K", -1, 30720, 2048},
        {"v4 ratio 4 (top_k)", 4, 126976, 2048},    {"v4 ratio 128 (all)", 128, 126976, 2048},
        {"v41 ratio 0 (window)", 0, 126976, 2048},  {"v41 ratio -2 tail", -2, 130000, 620},
        {"v41 ratio -2 short", -2, 5, 37},
    };
    setvbuf(stdout, nullptr, _IONBF, 0);
    unsigned seed = 1;
    for (const Case& c : cases) run(c, seed++);
    return 0;
}
