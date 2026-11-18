// include/strata/core/spec_sampling.hpp - speculative sampling's accept / resample step (host side).
//
// A sampled request's verify window used to keep a draft only when the column's own draw equalled it: a draft is
// then kept with probability p(d), the target's probability of that one token - at temperature 1 that falls fast
// along the window.  When the drafter SAMPLES its draft d from its own distribution q (after the request's top_k /
// top_p / min_p / temperature, the same transforms as the target's p), the rule
//
//     keep d with probability min(1, p(d) / q(d));  else draw from the residual max(0, p - q), normalized
//
// gives every column a token distributed exactly as p (the standard speculative-sampling argument: P(x) =
// q(x) min(1, p(x)/q(x)) + (1 - sum_y min(p(y), q(y))) * max(0, p(x) - q(x)) / sum_y max(0, p(y) - q(y)) = p(x)),
// and keeps a draft with probability sum_x min(p(x), q(x)) - near 1 when the drafter agrees with the target.
// The output distribution is the target's; the exact tokens for a given seed are not the old rule's.  Opt-in:
// STRATA_SPEC_SAMPLING=1 for requests without a seed, =2 for seeded ones too (generate.cpp has the measurement).
//
// p and q are the sampler's kept lists (sample_tokens_dist: ids in order, probabilities summing to 1, at most 64).
#pragma once

#include <cstdint>

namespace strata::core {

/// A uniform double in [0, 1) from (seed, counter, tag): splitmix64 over the three, 53 bits.  The accept test and
/// the residual draw use different tags, so they are independent of each other and of the device's Philox draws.
inline double spec_uniform(uint64_t seed, uint64_t counter, uint64_t tag) {
    uint64_t z = seed * 0x9E3779B97F4A7C15ull ^ (counter + 0x632BE59BD9B4E019ull) * 0xBF58476D1CE4E5B9ull ^
                 (tag + 1) * 0x94D049BB133111EBull;
    for (int i = 0; i < 2; ++i) {
        z += 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
    }
    return (double) (z >> 11) * (1.0 / 9007199254740992.0);
}

inline double spec_prob_of(int tok, const int* ids, const float* pr, int n) {
    for (int i = 0; i < n; ++i)
        if (ids[i] == tok) return (double) pr[i];
    return 0.0;
}

/// One column: the draft `d` (drawn from q), the target's p.  Returns the column's token and sets `kept` when it is
/// d.  `u_keep`, `u_res`: two independent uniforms in [0, 1).  p's support is where the residual lives (p is 0
/// elsewhere); a residual that rounds to nothing (p <= q everywhere on p's support, which a rejection makes all but
/// impossible) draws from p itself.
inline int spec_column(int d, const int* q_ids, const float* q_p, int q_n, const int* p_ids, const float* p_p, int p_n,
                       double u_keep, double u_res, bool& kept) {
    const double pd = spec_prob_of(d, p_ids, p_p, p_n);
    const double qd = spec_prob_of(d, q_ids, q_p, q_n);
    const bool point = qd <= 0.0;   // d is not in q's list (a copy / n-gram draft): q is a point mass at d
    kept = point ? u_keep < pd : u_keep * qd < pd;
    if (kept) return d;
    const auto resid = [&](int i) -> double {
        if (point) return p_ids[i] != d ? (double) p_p[i] : 0.0;
        const double r = (double) p_p[i] - spec_prob_of(p_ids[i], q_ids, q_p, q_n);
        return r > 0.0 ? r : 0.0;
    };
    double rsum = 0.0;
    for (int i = 0; i < p_n; ++i) rsum += resid(i);
    if (!(rsum > 1e-12)) {   // (nothing left: draw from p)
        double cum = 0.0;
        for (int i = 0; i < p_n; ++i) {
            cum += p_p[i];
            if (u_res < cum) return p_ids[i];
        }
        return p_n > 0 ? p_ids[p_n - 1] : d;
    }
    const double target = u_res * rsum;
    double cum = 0.0;
    int last = d;
    for (int i = 0; i < p_n; ++i) {
        const double r = resid(i);
        if (r <= 0.0) continue;
        last = p_ids[i];
        cum += r;
        if (target < cum) return p_ids[i];
    }
    return last;
}

}  // namespace strata::core
