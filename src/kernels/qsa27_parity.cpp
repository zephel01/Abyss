// src/kernels/qsa27_parity.cpp - Qwen3.8-27B QSA's indexer and selection (qsa27.hpp) against a host reference written
// from the model card's PyTorch reference (Qwen3.8-27B-QSA-BF16 indexer.py / modeling_qsa.py), in double: a cell's key
// normed before pooling, the block's mean normed again and rotated at its first cell, the queries normed and rotated at
// their position, s[b] = sum_h relu(q[h] . K[b]); a query attends to the top 512 complete blocks (ties to the lower
// block) and its own block's cells up to p.  GPU, synthetic, no model: the keys appended in uneven chunks (the tail
// carried across them), queries just past the dense window to 20K cells.  Checks the pooled keys (max error), and that
// the selected cells are the reference's - a block that differs must sit at the top-512 boundary within 1e-5 of the
// threshold score.  Usage: qsa27_parity
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa27.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <set>
#include <vector>

namespace k = strata::kernels;
namespace q27 = strata::kernels::qsa27;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
constexpr int D = q27::kDim, H = q27::kHeads, R = q27::kBlock, ROT = q27::kRot;
constexpr double kTheta = 1e7, kEps = 1e-6;

void rms(std::vector<double>& x, const float* w) {
    double ss = 0;
    for (double v : x) ss += v * v;
    const double inv = 1.0 / std::sqrt(ss / D + kEps);
    for (int d = 0; d < D; ++d) x[(size_t) d] = x[(size_t) d] * inv * w[d];
}
void rope(std::vector<double>& x, int64_t pos) {
    for (int i = 0; i < ROT / 2; ++i) {
        const double ang = (double) pos * std::pow(kTheta, -2.0 * i / ROT);
        const double a = x[(size_t) i], b = x[(size_t) i + ROT / 2];
        x[(size_t) i] = a * std::cos(ang) - b * std::sin(ang);
        x[(size_t) i + ROT / 2] = a * std::sin(ang) + b * std::cos(ang);
    }
}
}  // namespace

int main() {
    const int top_k = 2048;
    const k::QsaShapes sh = q27::shapes(top_k);
    const int64_t N = 20011;                       // cells
    const int64_t max_blocks = N / R + 2, cap = q27::cap(top_k);
    std::mt19937 rng(27);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> wd(0.8f, 1.3f);
    std::vector<float> raw((size_t) N * D), wpre(D), wblk(D), wq(D);
    for (auto& v : raw) v = nd(rng);
    // a shared direction makes the scores spread (relu of mostly-positive dots), as trained keys do
    std::vector<float> dir(D);
    for (auto& v : dir) v = nd(rng);
    for (int64_t c = 0; c < N; ++c) {
        const float a = 0.7f * std::sin(0.001f * (float) c) + 0.3f * nd(rng);
        for (int d = 0; d < D; ++d) raw[(size_t) c * D + d] += a * dir[(size_t) d];
    }
    for (int d = 0; d < D; ++d) { wpre[(size_t) d] = wd(rng); wblk[(size_t) d] = wd(rng); wq[(size_t) d] = wd(rng); }

    // ---- host reference: block keys
    const int64_t NB = N / R;
    std::vector<std::vector<double>> K((size_t) NB, std::vector<double>(D));
    for (int64_t b = 0; b < NB; ++b) {
        std::vector<double> m(D, 0.0);
        for (int i = 0; i < R; ++i) {
            std::vector<double> x(D);
            for (int d = 0; d < D; ++d) x[(size_t) d] = raw[(size_t) (b * R + i) * D + d];
            rms(x, wpre.data());
            for (int d = 0; d < D; ++d) m[(size_t) d] += x[(size_t) d];
        }
        for (auto& v : m) v /= R;
        rms(m, wblk.data());
        rope(m, b * R);
        K[(size_t) b] = m;
    }

    // ---- device: keys appended in uneven chunks
    float *d_raw, *d_wpre, *d_wblk, *d_wq, *d_tail, *d_pool, *d_cos, *d_sin, *d_dead, *d_q, *d_sc;
    int32_t *d_st, *d_st2, *d_ids, *d_pos;
    ck(cudaMalloc(&d_raw, raw.size() * 4), "m");
    ck(cudaMalloc(&d_wpre, D * 4), "m"); ck(cudaMalloc(&d_wblk, D * 4), "m"); ck(cudaMalloc(&d_wq, D * 4), "m");
    ck(cudaMalloc(&d_tail, 3 * D * 4), "m"); ck(cudaMalloc(&d_pool, (size_t) max_blocks * D * 4), "m");
    ck(cudaMalloc(&d_dead, D * 4), "m"); ck(cudaMemset(d_dead, 0, D * 4), "m");
    std::vector<float> ct((size_t) N * ROT / 2), st_((size_t) N * ROT / 2);
    k::build_rope_table(ROT, kTheta, (int) N, ct.data(), st_.data());
    ck(cudaMalloc(&d_cos, ct.size() * 4), "m"); ck(cudaMalloc(&d_sin, st_.size() * 4), "m");
    ck(cudaMemcpy(d_cos, ct.data(), ct.size() * 4, cudaMemcpyHostToDevice), "c");
    ck(cudaMemcpy(d_sin, st_.data(), st_.size() * 4, cudaMemcpyHostToDevice), "c");
    ck(cudaMemcpy(d_raw, raw.data(), raw.size() * 4, cudaMemcpyHostToDevice), "c");
    ck(cudaMemcpy(d_wpre, wpre.data(), D * 4, cudaMemcpyHostToDevice), "c");
    ck(cudaMemcpy(d_wblk, wblk.data(), D * 4, cudaMemcpyHostToDevice), "c");
    ck(cudaMemcpy(d_wq, wq.data(), D * 4, cudaMemcpyHostToDevice), "c");
    {
        const int chunks[] = {1, 1, 2, 3, 5, 7, 64, 513, 1, 2048, 3, 4, 4, 1000};
        int64_t at = 0;
        int ci = 0;
        while (at < N) {
            const int n = (int) std::min<int64_t>(chunks[ci++ % (int) (sizeof chunks / sizeof chunks[0])], N - at);
            q27::key_append_rows(d_raw + (size_t) at * D, D, at, n, d_wpre, d_wblk, (float) kEps, d_tail, d_pool, d_cos,
                                 d_sin, nullptr);
            at += n;
        }
    }
    std::vector<float> pool((size_t) NB * D);
    ck(cudaMemcpy(pool.data(), d_pool, pool.size() * 4, cudaMemcpyDeviceToHost), "c");
    double kerr = 0, kmax = 0;
    for (int64_t b = 0; b < NB; ++b)
        for (int d = 0; d < D; ++d) {
            kerr = std::max(kerr, std::fabs((double) pool[(size_t) b * D + d] - K[(size_t) b][(size_t) d]));
            kmax = std::max(kmax, std::fabs(K[(size_t) b][(size_t) d]));
        }
    std::printf("pooled keys: %lld blocks, max abs error %.3g (max |K| %.3g)\n", (long long) NB, kerr, kmax);
    bool ok = kerr < 1e-4 * std::max(1.0, kmax);

    // ---- queries
    const int64_t ps[] = {0, 5, 2047, 2049, 2050, 2051, 2052, 2054, 2055, 2056, 3000, 4095, 4096, 4099, 8191, 8192,
                          12345, 16383, 19999, N - 1};
    const int nq = (int) (sizeof ps / sizeof ps[0]);
    std::vector<float> qh((size_t) nq * H * D);
    for (auto& v : qh) v = nd(rng) + 0.5f;
    std::vector<int32_t> pos_h((size_t) nq * H), steps((size_t) nq * 4);
    for (int i = 0; i < nq; ++i) {
        for (int h = 0; h < H; ++h) pos_h[(size_t) i * H + h] = (int32_t) ps[i];
        q27::step_fill(steps.data() + (size_t) i * 4, ps[i], top_k);
    }
    ck(cudaMalloc(&d_q, qh.size() * 4), "m"); ck(cudaMalloc(&d_pos, pos_h.size() * 4), "m");
    ck(cudaMalloc(&d_st, steps.size() * 4), "m"); ck(cudaMalloc(&d_st2, steps.size() * 4), "m");
    ck(cudaMalloc(&d_sc, (size_t) nq * max_blocks * 4), "m"); ck(cudaMalloc(&d_ids, (size_t) nq * cap * 4), "m");
    ck(cudaMemcpy(d_q, qh.data(), qh.size() * 4, cudaMemcpyHostToDevice), "c");
    ck(cudaMemcpy(d_pos, pos_h.data(), pos_h.size() * 4, cudaMemcpyHostToDevice), "c");
    ck(cudaMemcpy(d_st, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "c");
    k::rms_norm_weighted(d_q, d_wq, (int64_t) nq * H, D, (float) kEps, nullptr);
    k::rope_neox_apply(d_q, d_q, (int64_t) nq * H, D, ROT, d_cos, d_sin, d_pos, nullptr);
    k::qsa_block_scores(d_pool, d_dead, d_q, d_st, nq, max_blocks, sh, d_sc, nullptr);
    k::qsa_block_topk(d_sc, d_st, nq, max_blocks, cap, sh, d_ids, nullptr);
    q27::own_block(d_ids, cap, d_st, d_st2, nq, nullptr);
    std::vector<int32_t> ids((size_t) nq * cap), st2((size_t) nq * 4);
    ck(cudaMemcpy(ids.data(), d_ids, ids.size() * 4, cudaMemcpyDeviceToHost), "c");
    ck(cudaMemcpy(st2.data(), d_st2, st2.size() * 4, cudaMemcpyDeviceToHost), "c");
    int bad = 0, edge = 0, appended = 0, own_in = 0;
    for (int i = 0; i < nq; ++i) {
        const int64_t p = ps[i], n_kv = p + 1, nbv = n_kv / R;
        std::vector<double> q[H];
        for (int h = 0; h < H; ++h) {
            q[h].assign(D, 0.0);
            for (int d = 0; d < D; ++d) q[h][(size_t) d] = qh[((size_t) i * H + h) * D + d];
            rms(q[h], wq.data());
            rope(q[h], p);
        }
        std::vector<double> s((size_t) nbv);
        for (int64_t b = 0; b < nbv; ++b) {
            double t = 0;
            for (int h = 0; h < H; ++h) {
                double dot = 0;
                for (int d = 0; d < D; ++d) dot += q[h][(size_t) d] * K[(size_t) b][(size_t) d];
                t += dot > 0 ? dot : 0;
            }
            s[(size_t) b] = t;
        }
        const int64_t kb = std::min<int64_t>(nbv, top_k / R);
        std::vector<int64_t> ord((size_t) nbv);
        std::iota(ord.begin(), ord.end(), 0);
        std::stable_sort(ord.begin(), ord.end(), [&](int64_t a, int64_t b) { return s[(size_t) a] > s[(size_t) b]; });
        std::set<int64_t> ref;
        for (int64_t j = 0; j < kb; ++j)
            for (int c = 0; c < R; ++c) ref.insert(ord[(size_t) j] * R + c);
        for (int64_t c = (p / R) * R; c <= p; ++c) ref.insert(c);   // the own block's cells up to p
        const double thr = kb > 0 ? s[(size_t) ord[(size_t) kb - 1]] : 0.0;
        const int w = st2[(size_t) i * 4 + 3];
        if (n_kv % R == 0 && n_kv > top_k + R - 1) (w == top_k + R ? appended : own_in) += 1;
        std::set<int64_t> got;
        bool asc = true;
        for (int j = 0; j < w; ++j) {
            const int32_t c = ids[(size_t) i * cap + j];
            if (j > 0 && c <= ids[(size_t) i * cap + j - 1]) asc = false;
            got.insert(c);
        }
        if (!asc || (int) got.size() != w) {
            std::printf("p %lld: the ids are not ascending and unique\n", (long long) p);
            ++bad;
            continue;
        }
        if (got == ref) continue;
        // the differences: whole blocks at the selection boundary only
        bool near = true;
        std::set<int64_t> diff;
        for (int64_t c : ref) if (!got.count(c)) diff.insert(c / R);
        for (int64_t c : got) if (!ref.count(c)) diff.insert(c / R);
        for (int64_t b : diff)
            if (b >= nbv || std::fabs(s[(size_t) b] - thr) > 1e-5 * std::max(1.0, std::fabs(thr))) near = false;
        if (near) {
            ++edge;
            continue;
        }
        std::printf("p %lld: selection differs (%zu ref cells, %d got, %zu blocks differ)\n", (long long) p,
                    ref.size(), w, diff.size());
        ++bad;
    }
    std::printf("selections: %d queries, %d exact, %d at a near-tie boundary, %d wrong (complete own block: %d added "
                "by own_block, %d already in the top-k)\n", nq, nq - bad - edge, edge, bad, appended, own_in);
    ok = ok && appended > 0 && own_in > 0;   // (both branches of the union seen)
    ok = ok && bad == 0;
    std::printf("%s\n", ok ? "qsa27_parity: PASS" : "qsa27_parity: FAIL");
    return ok ? 0 : 1;
}
