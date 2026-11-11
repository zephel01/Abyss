#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace strata::core::glmfast {

// All inputs are bytes. On an APU the GPU pool and pinned host allocations spend
// the same physical memory; the runtime's GTT/carve-out size is not free RAM.
inline size_t expert_pool_budget(bool unified, size_t device_free, size_t ram_available,
                                size_t reserve, size_t ram_headroom, size_t expert_bytes) {
    if (!unified) return device_free > reserve ? device_free - reserve : 0;
    const size_t room = ram_available > ram_headroom ? ram_available - ram_headroom : 0;
    return std::min(expert_bytes, room > reserve ? room - reserve : 0);
}

inline bool minimal_ram_tier(bool unified, bool all_experts_fit, bool explicit_budget) {
    return unified && all_experts_fit && !explicit_budget;
}

inline bool full_unified_pool(bool unified, int slots, int experts) {
    return unified && slots >= experts;
}

inline int warm_pool_slots(bool unified, int slots, int experts, int spares) {
    return full_unified_pool(unified, slots, experts) ? experts : std::max(0, slots - spares);
}

// ---- unified memory (an integrated GPU: AMD APUs, the DGX Spark's GB10).  The device pool and the pinned RAM tier
// are the SAME physical memory, so one budget serves both: MemAvailable (page cache counts as available - the CUDA
// runtime's cudaMemGetInfo counts it as used on GB10, and cudaMalloc reclaims it) less a headroom for the OS, the
// server and the page cache the shards' reads go through.

// the budget the pool and the RAM tier share (0 when MemAvailable is unknown or below the headroom)
inline size_t uma_budget(size_t mem_available, size_t headroom) {
    return mem_available > headroom ? mem_available - headroom : 0;
}

// what the RAM tier may still pin once the pool took pool_bytes of that budget: pool + tier never exceed it
inline size_t uma_ram_tier(size_t budget, size_t pool_bytes) { return budget > pool_bytes ? budget - pool_bytes : 0; }

// the default headroom, GB (STRATA_GLM_UMA_HEADROOM_GB, else STRATA_GLM_RAM_HEADROOM_GB, sets it): 16 on HIP (the
// AMD setup's configs were written for it), 8 on CUDA - a DGX Spark: the OS and the CUDA context (~3 GB), the server
// and the tokenizer (~1-2 GB), and the rest for the page cache that the pack's Engram / PLE tables and the disk reads
// go through.  Everything this process allocates BEFORE the pool (dense weights, KV, prompt buffers) is already out of
// MemAvailable when the pool is priced, so it is not in the headroom.
inline double uma_headroom_default_gb(bool hip) { return hip ? 16.0 : 8.0; }

// the shared memory other processes hold, GB: MemTotal - MemAvailable - what this process already allocated (page
// cache is in MemAvailable, so it does not count).  A DGX Spark with nothing else running: about 5 GB (the OS, the
// server); with a second GLM engine left running: about 95 GB
inline double uma_others_gb(int64_t mem_total, int64_t mem_available, int64_t held) {
    const int64_t o = mem_total - mem_available - held;
    return mem_total > 0 && mem_available > 0 && o > 0 ? (double) o / 1073741824.0 : 0.0;
}

// stop instead of starting: other processes hold more than STRATA_GLM_UMA_OTHERS_GB (default 16) AND the experts do
// not fit because of it.  A model that is bigger than the machine (V4.1 on one Spark) on a machine with nothing else
// running is not stopped: there "others" is small
inline bool uma_others_block(double others_gb, bool experts_short) {
    const char* v = std::getenv("STRATA_GLM_UMA_OTHERS_GB");
    const double lim = v != nullptr && v[0] != '\0' ? std::atof(v) : 16.0;
    return experts_short && lim > 0.0 && others_gb > lim;
}

// ---- the RAM tier's cap on Linux with a discrete card (no STRATA_GLM_RAM_GB).  A share of MemTotal alone
// (STRATA_GLM_RAM_FRAC, 0.73) was set on a 94 GB machine (67-68 GB pinned ran for days, 73.9 thrashed: the page cache
// the shards and Engram tables go through, the server and the desktop need ~20-26 GB whatever the RAM's size), so on
// bigger machines it left more free than that need: 34 GB on a 123 GB one (88.7 GB pinned; 100 decoded DeepSeek V4.1
// 30.2 -> 35.8 tok/s, 110 was OOM-killed at load).  So on machines of at least `keep_min_total` bytes of RAM (100 GiB
// by default: a 94 GiB one would gain 1.4 GB over the share, where 69 GB was "sometimes slow") the cap is the larger
// of the share and MemTotal less a fixed keep (STRATA_GLM_RAM_KEEP_GB, 24); below it the share alone, byte for byte
// as before.  An explicit STRATA_GLM_RAM_FRAC (without STRATA_GLM_RAM_KEEP_GB) keeps the share alone, as before.
// Bytes; 0 when MemTotal is unknown (no cap).  *by_keep says which bound won
constexpr int64_t kRamKeepMinTotal = int64_t{100} << 30;
inline int64_t ram_tier_cap(int64_t mem_total, double frac, double keep_gb, bool use_keep, bool* by_keep = nullptr,
                            int64_t keep_min_total = kRamKeepMinTotal) {
    if (by_keep != nullptr) *by_keep = false;
    if (mem_total <= 0) return 0;
    const int64_t by_frac = (int64_t) ((double) mem_total * frac);
    if (!use_keep || mem_total < keep_min_total) return by_frac;
    const int64_t by_k = std::max<int64_t>(0, mem_total - (int64_t) (keep_gb * 1073741824.0));
    if (by_k > by_frac) {
        if (by_keep != nullptr) *by_keep = true;
        return by_k;
    }
    return by_frac;
}

// STRATA_GLM_UMA_DIRECT: the device reads RAM-tier experts IN PLACE (their pinned host address in the plan) instead
// of copying them into a VRAM slot first.  Unset: `integrated` (fast_setup passes false - the copy mode is the default
// everywhere); "0" off; any other value on (a discrete card too: for tests - its reads then cross PCIe every token)
inline bool uma_direct_on(bool integrated, const char* env) {
    return env != nullptr && env[0] != '\0' ? std::atoi(env) != 0 : integrated;
}

}  // namespace strata::core::glmfast
