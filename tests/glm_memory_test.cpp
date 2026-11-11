// CPU-only policy checks: no HIP/CUDA headers, device queries or allocations.
#include "../src/core/glm_memory.hpp"

#undef NDEBUG   // the checks are asserts: keep them in Release builds
#include <cassert>
#include <cmath>
#include <cstdio>

int main() {
    using strata::core::glmfast::expert_pool_budget;
    using strata::core::glmfast::minimal_ram_tier;
    using strata::core::glmfast::full_unified_pool;
    using strata::core::glmfast::warm_pool_slots;
    constexpr size_t G = size_t{1} << 30;
    constexpr size_t experts = 80 * G + 174 * (size_t{1} << 20);

    // A 128 GiB APU with 110 GiB available after setup: the 112 GiB runtime
    // aperture (and even a tiny reported free value) cannot change the OS budget.
    assert(expert_pool_budget(true, 112 * G, 110 * G, G, 16 * G, experts) == experts);
    assert(expert_pool_budget(true, G, 110 * G, G, 16 * G, experts) == experts);
    // Memory pressure leaves 53 GiB; missing MemAvailable or less than headroom
    // leaves zero rather than wrapping or trusting HIP's much larger free figure.
    assert(expert_pool_budget(true, 112 * G, 70 * G, G, 16 * G, experts) == 53 * G);
    assert(expert_pool_budget(true, 112 * G, 0, G, 16 * G, experts) == 0);
    assert(expert_pool_budget(true, 112 * G, 16 * G, G, 16 * G, experts) == 0);
    assert(expert_pool_budget(true, 112 * G, 17 * G, G, 16 * G, experts) == 0);
    // User headroom/reserve and the discrete formula retain their meaning.
    assert(expert_pool_budget(true, 112 * G, 70 * G, 2 * G, 8 * G, experts) == 60 * G);
    assert(expert_pool_budget(false, 12 * G, 110 * G, 3 * G, 16 * G, experts) == 9 * G);
    assert(expert_pool_budget(false, G, 110 * G, 3 * G, 16 * G, experts) == 0);
    assert(minimal_ram_tier(true, true, false));
    assert(!minimal_ram_tier(true, true, true));  // STRATA_GLM_RAM_GB / supplied budget wins
    assert(!minimal_ram_tier(true, false, false));
    assert(!minimal_ram_tier(false, true, false));
    // Warm every expert when it fits, and never evict residents to make spares.
    // Smaller APU pools and all discrete pools retain the existing spare quota.
    assert(warm_pool_slots(true, 288, 288, 3) == 288);
    assert(full_unified_pool(true, 288, 288));
    assert(warm_pool_slots(true, 192, 288, 3) == 189);
    assert(!full_unified_pool(true, 192, 288));
    assert(warm_pool_slots(false, 288, 288, 3) == 285);
    assert(!full_unified_pool(false, 288, 288));

    // Unified memory (CUDA's GB10 too): ONE budget, MemAvailable less the headroom, shared by the pool and the tier.
    using strata::core::glmfast::uma_budget;
    using strata::core::glmfast::uma_ram_tier;
    using strata::core::glmfast::uma_direct_on;
    using strata::core::glmfast::uma_headroom_default_gb;
    assert(uma_budget(118 * G, 8 * G) == 110 * G);
    assert(uma_budget(8 * G, 8 * G) == 0 && uma_budget(3 * G, 8 * G) == 0 && uma_budget(0, 8 * G) == 0);
    // GB10, 118 GB available, Qwen3.8's 53 GB of experts: the pool takes them all, the tier at most the rest
    {
        const size_t budget = uma_budget(118 * G, 8 * G), q38 = 53 * G;
        const size_t pool = expert_pool_budget(true, 50 * G, 118 * G, G, 8 * G, q38);
        assert(pool == q38);
        assert(pool + uma_ram_tier(budget, pool) == budget);
    }
    // a model larger than the budget: the pool takes budget - reserve, the tier only the reserve's worth - never
    // pool + tier above the budget, whatever MemAvailable reads after the pool allocated
    for (size_t avail : {20 * G, 64 * G, 118 * G, 128 * G})
        for (size_t experts : {10 * G, 80 * G, 200 * G}) {
            const size_t budget = uma_budget(avail, 8 * G);
            const size_t pool = expert_pool_budget(true, G, avail, G, 8 * G, experts);
            assert(pool <= budget);
            assert(pool + uma_ram_tier(budget, pool) <= budget);
            assert(uma_ram_tier(budget, pool) == budget - pool);
        }
    assert(uma_ram_tier(10 * G, 12 * G) == 0);   // an explicit pool (STRATA_GLM_VRAM_GB) past the budget: no tier
    // STRATA_GLM_UMA_DIRECT: on exactly for an integrated device unless set; "0" off; any other value forces it on
    assert(uma_direct_on(true, nullptr) && !uma_direct_on(false, nullptr));
    assert(uma_direct_on(true, "") && !uma_direct_on(false, ""));
    assert(!uma_direct_on(true, "0") && uma_direct_on(false, "1") && uma_direct_on(true, "1"));
    assert(uma_headroom_default_gb(true) == 16.0 && uma_headroom_default_gb(false) == 8.0);
    // another process holding the shared memory (Spark, 2026-11-06): about 5 GB of others with nothing else running,
    // about 95 GB with a second engine left running; only a shortfall the others cause stops the start
    {
        using strata::core::glmfast::uma_others_gb;
        using strata::core::glmfast::uma_others_block;
        const int64_t T = (int64_t) (121.6 * (double) G);
        const double alone = uma_others_gb(T, (int64_t) (109.7 * (double) G), (int64_t) (6.5 * (double) G));
        const double second = uma_others_gb(T, (int64_t) (20.1 * (double) G), (int64_t) (6.5 * (double) G));
        assert(alone > 4.0 && alone < 7.0 && second > 90.0 && second < 100.0);
        assert(!uma_others_block(alone, true));      // V4.1 bigger than the Spark, nothing else running: runs
        assert(uma_others_block(second, true));      // a second engine and the experts short: stops
        assert(!uma_others_block(second, false));    // a second engine but the experts still fit: runs
        assert(uma_others_gb(0, 0, 0) == 0.0 && uma_others_gb(T, 0, 0) == 0.0);   // MemAvailable unknown: never stops
    }
    // The discrete-card RAM tier cap: the larger of MemTotal x STRATA_GLM_RAM_FRAC and MemTotal less the keep, the
    // keep only from 100 GiB of RAM (unless STRATA_GLM_RAM_KEEP_GB is set: keep_min_total 0).
    {
        using strata::core::glmfast::ram_tier_cap;
        const int64_t g = int64_t{1} << 30;
        const auto gib = [&](double x) { return (int64_t) (x * (double) g); };
        const auto near = [&](int64_t v, double want) { return std::fabs((double) v / (double) g - want) < 0.051; };
        bool k = true;
        // the four RAM sizes: 64 / 96 GB boxes keep 73% byte for byte, 128 / 256 GB boxes keep 24 GB
        assert(ram_tier_cap(gib(62.5), 0.73, 24.0, true, &k) == (int64_t) ((double) gib(62.5) * 0.73) && !k);
        assert(near(ram_tier_cap(gib(62.5), 0.73, 24.0, true, &k), 45.6));
        assert(ram_tier_cap(gib(94), 0.73, 24.0, true, &k) == (int64_t) ((double) gib(94) * 0.73) && !k);
        assert(near(ram_tier_cap(gib(94), 0.73, 24.0, true, &k), 68.6));
        assert(ram_tier_cap(gib(123.5), 0.73, 24.0, true, &k) == gib(123.5) - 24 * g && k);
        assert(near(ram_tier_cap(gib(123.5), 0.73, 24.0, true, &k), 99.5));
        assert(ram_tier_cap(gib(251), 0.73, 24.0, true, &k) == 227 * g && k);
        // this box's MemTotal (123.5 GiB)
        const int64_t box = 129458420LL * 1024;
        assert(ram_tier_cap(box, 0.73, 24.0, true, &k) == box - 24 * g && k);
        // an explicit STRATA_GLM_RAM_FRAC alone: the share, as before
        assert(ram_tier_cap(box, 0.73, 24.0, false, &k) == (int64_t) ((double) box * 0.73) && !k);
        // the threshold: just below 100 GiB the share, at 100 GiB the keep (76 > 73)
        assert(ram_tier_cap(100 * g - 1, 0.73, 24.0, true, &k) == (int64_t) ((double) (100 * g - 1) * 0.73) && !k);
        assert(ram_tier_cap(100 * g, 0.73, 24.0, true, &k) == 76 * g && k);
        // STRATA_GLM_RAM_KEEP_GB set: the keep at any size (94 GiB less 24 = 70 > 68.6)
        assert(ram_tier_cap(gib(94), 0.73, 24.0, true, &k, 0) == 70 * g && k);
        assert(!(ram_tier_cap(88 * g, 0.73, 24.0, true, &k, 0) > 0 && k));   // the crossover sits at 24 / 0.27 = 88.9 GiB
        // a keep larger than the RAM leaves the share; MemTotal unknown: no cap
        assert(ram_tier_cap(32 * g, 0.73, 64.0, true, &k, 0) == (int64_t) (32.0 * 0.73 * (double) g) && !k);
        assert(ram_tier_cap(0, 0.73, 24.0, true, &k) == 0 && !k);
    }
    std::puts("GLM memory policy checks passed (CPU only)");
}
