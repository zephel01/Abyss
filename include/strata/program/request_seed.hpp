// include/strata/program/request_seed.hpp - the seed a served request samples with.
//
// A GEN line with seed=N uses N (the same answer for the same seed).  Without one, the glm-pack engine used the
// engine's own --seed (0 unless given) and counter 0 for every request, so a sampled request without a seed gave
// the same answer every time (a client's "regenerate" changed nothing).  Now such a request gets a fresh seed: the
// clock mixed with a per-process count (two requests in the same clock tick still differ).  An engine started with
// an explicit --seed keeps using it for requests without one, as before.
#pragma once

#include <chrono>
#include <cstdint>

namespace strata::program {

inline uint64_t seed_mix(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/// `seeded`/`given`: the request's seed=; `engine_seed`/`engine_given`: --seed; `count`: this process's counter of
/// fresh seeds (advanced on every fresh one); `now`: a clock reading.
inline uint64_t request_seed(bool seeded, uint64_t given, bool engine_given, uint64_t engine_seed, uint64_t& count,
                             uint64_t now) {
    if (seeded) return given;
    if (engine_given) return engine_seed;
    const uint64_t s = seed_mix(now ^ seed_mix(++count));
    return s != 0 ? s : 1;
}

inline uint64_t request_seed_now() {
    return (uint64_t) std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

}  // namespace strata::program
