// tests/spec_sampling_test.cpp - speculative sampling's column rule (spec_sampling.hpp) gives tokens distributed as
// the target's p, whatever q the draft came from, and keeps drafts at sum min(p, q).  Host only, no GPU.
#include "strata/core/spec_sampling.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {
struct Dist {
    std::vector<int> ids;
    std::vector<float> p;
};

Dist make(uint64_t seed, int n, int off, float sharp) {
    Dist d;
    double s = 0.0;
    std::vector<double> w;
    for (int i = 0; i < n; ++i) {
        const double x = std::exp(sharp * strata::core::spec_uniform(seed, (uint64_t) i, 7));
        w.push_back(x);
        s += x;
    }
    for (int i = 0; i < n; ++i) {
        d.ids.push_back(off + i);
        d.p.push_back((float) (w[(size_t) i] / s));
    }
    return d;
}

int draw(const Dist& d, double u) {
    double c = 0.0;
    for (size_t i = 0; i < d.ids.size(); ++i) {
        c += d.p[i];
        if (u < c) return d.ids[i];
    }
    return d.ids.back();
}

int check(const char* name, const Dist& P, const Dist& Q, bool point, uint64_t seed) {
    const int N = 400000, V = 64;
    std::vector<double> got(V, 0.0);
    int kept_n = 0;
    for (int t = 0; t < N; ++t) {
        const int d = point ? 3 : draw(Q, strata::core::spec_uniform(seed, (uint64_t) t, 1));
        bool kept = false;
        const int tok = point ? strata::core::spec_column(d, nullptr, nullptr, 0, P.ids.data(), P.p.data(), (int) P.ids.size(),
                                                          strata::core::spec_uniform(seed, (uint64_t) t, 2),
                                                          strata::core::spec_uniform(seed, (uint64_t) t, 3), kept)
                              : strata::core::spec_column(d, Q.ids.data(), Q.p.data(), (int) Q.ids.size(), P.ids.data(),
                                                          P.p.data(), (int) P.ids.size(),
                                                          strata::core::spec_uniform(seed, (uint64_t) t, 2),
                                                          strata::core::spec_uniform(seed, (uint64_t) t, 3), kept);
        if (kept != (tok == d)) {
            std::printf("%s: kept flag disagrees with the token\n", name);
            return 1;
        }
        kept_n += kept;
        got[(size_t) tok] += 1.0;
    }
    double tv = 0.0, overlap = 0.0;
    for (int v = 0; v < V; ++v) {
        const double pv = strata::core::spec_prob_of(v, P.ids.data(), P.p.data(), (int) P.ids.size());
        const double qv = point ? (v == 3 ? 1.0 : 0.0) : strata::core::spec_prob_of(v, Q.ids.data(), Q.p.data(), (int) Q.ids.size());
        tv += std::fabs(got[(size_t) v] / N - pv);
        overlap += std::fmin(pv, qv);
    }
    tv *= 0.5;
    const double acc = (double) kept_n / N;
    std::printf("%s: total variation %.4f, kept %.4f (expected %.4f)\n", name, tv, acc, overlap);
    return tv < 0.006 && std::fabs(acc - overlap) < 0.006 ? 0 : 1;
}
}  // namespace

int main() {
    int bad = 0;
    const Dist P = make(11, 20, 0, 3.0f);
    bad += check("q close to p", P, make(11, 20, 0, 2.5f), false, 1);
    bad += check("q unlike p", P, make(29, 20, 0, 4.0f), false, 2);
    bad += check("q partly outside p", P, make(5, 24, 10, 3.0f), false, 3);
    bad += check("point mass (copy draft)", P, Dist{}, true, 4);
    std::printf(bad ? "spec_sampling_test: FAILED\n" : "spec_sampling_test: ok\n");
    return bad ? 1 : 0;
}
