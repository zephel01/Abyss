// src/platform/memory_test.cpp - plan v0.3 P0.1: lock_resident on a 256 MiB region (CPU only, no GPU, no model).
#include "strata/platform/memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(__linux__)
#include <sys/resource.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

int main() {
    const uint64_t bytes = 256ull << 20;
    void* p = std::malloc(bytes);
    if (p == nullptr) return 2;
    std::memset(p, 1, bytes);
    const strata::platform::LockResult r = strata::platform::lock_resident(p, bytes);
    std::printf("lock_resident: ok=%d locked=%llu MiB (%s)\n", (int) r.ok, (unsigned long long) (r.locked_bytes >> 20),
                r.note.c_str());
    int fail = !(r.ok && r.locked_bytes == bytes);
#if defined(__linux__)
    // a lock refused because RLIMIT_MEMLOCK (ulimit -l) is below the region is the environment, not lock_resident:
    // SKIPPED (77), with the limit.  (Root with CAP_IPC_LOCK ignores the limit - such a refusal still fails.)
    rlimit rl{};
    if (fail && !r.ok && getrlimit(RLIMIT_MEMLOCK, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < bytes) {
        std::printf("platform_memory_test: SKIPPED - RLIMIT_MEMLOCK %llu KiB < %llu MiB (raise ulimit -l)\n",
                    (unsigned long long) (rl.rlim_cur >> 10), (unsigned long long) (bytes >> 20));
        std::free(p);
        return 77;
    }
#endif
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc);
    std::printf("working set %llu MiB\n", (unsigned long long) (pmc.WorkingSetSize >> 20));
    fail |= pmc.WorkingSetSize < bytes;
#endif
    strata::platform::unlock_resident(p, bytes);
    std::free(p);
    std::printf("platform_memory_test: %s\n", fail ? "FAILED" : "OK");
    return fail;
}
