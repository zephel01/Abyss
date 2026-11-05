#pragma once

#include <algorithm>
#include <cstddef>
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

// STRATA_GLM_UMA_DIRECT: the device reads RAM-tier experts IN PLACE (their pinned host address in the plan) instead
// of copying them into a VRAM slot first.  Unset: `integrated` (fast_setup passes false - the copy mode is the default
// everywhere); "0" off; any other value on (a discrete card too: for tests - its reads then cross PCIe every token)
inline bool uma_direct_on(bool integrated, const char* env) {
    return env != nullptr && env[0] != '\0' ? std::atoi(env) != 0 : integrated;
}

}  // namespace strata::core::glmfast
