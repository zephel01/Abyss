// include/strata/platform/memory.hpp - plan v0.3 P0.1/P1: keep a large host region resident.
//
// `cudaHostRegister` refuses the 31.6 GiB expert arena on Windows, and an unlocked arena is trimmed under memory
// pressure (the CPU pool then swung 2x between runs). With the n-gram table out of RAM there is headroom to lock
// it instead: raise the process's minimum working set by the region's size, then VirtualLock it (Windows needs
// only SeIncreaseWorkingSetPrivilege, which ordinary accounts hold). Linux: mlock.
#pragma once

#include <cstdint>
#include <string>

namespace strata::platform {

struct LockResult {
    bool ok = false;
    uint64_t locked_bytes = 0;   ///< may be less than requested; the rest stays pageable
    std::string note;            ///< what was done or why it failed, for the startup print
};

/// Lock [p, p + bytes) into physical memory. Partial success is reported, not hidden.
LockResult lock_resident(void* p, uint64_t bytes);

/// Undo lock_resident for the same region (best effort).
void unlock_resident(void* p, uint64_t bytes);

// ---- Windows commit headroom ------------------------------------------------------------------------------------
// Pinned RAM (and, under WDDM, device memory) is charged to the commit: RAM + page file.  GlobalMemoryStatusEx's
// ullAvailPageFile is the commit left at the page file's CURRENT size, but a system-managed page file, or one set with
// initial < maximum, grows when the commit runs out - so on a PC with an 8 GB / 32 GB page file and 40 GB of RAM a
// 24 GiB tier was cut to 17 GiB (the idea is Strata's, issue #5742 there; the code is ours).  The arithmetic is here,
// host-only and testable on Linux; the query is Windows-only (memory.cpp).

inline uint64_t add_saturating(uint64_t a, uint64_t b) { return a > UINT64_MAX - b ? UINT64_MAX : a + b; }

/// The most a system-managed page file grows to: 3 x RAM or 4 GB, whichever is larger, limited to 1/8 of its volume
/// (Microsoft Learn, "How to determine the appropriate page file size for 64-bit versions of Windows", the
/// system-managed table).  volume_total 0 = unknown: no volume limit.
inline uint64_t managed_page_file_max(uint64_t ram_total, uint64_t volume_total) {
    const uint64_t four_gb = 4ull << 30;
    uint64_t m = ram_total > UINT64_MAX / 3 ? UINT64_MAX : ram_total * 3;
    if (m < four_gb) m = four_gb;
    if (volume_total > 0 && m > volume_total / 8) m = volume_total / 8;
    return m;
}

/// What one page file can still grow: its maximum less its current size (0 for a fixed-size file, or one already at
/// or past its maximum), and never more than `room` (the volume's free space left for it).
inline uint64_t page_file_growth(uint64_t current, uint64_t maximum, uint64_t room) {
    const uint64_t g = maximum > current ? maximum - current : 0;
    return g < room ? g : room;
}

#if defined(_WIN32)
/// The commit the page files can still add by growing, in bytes (summed, saturating; see above).  0 when the page
/// files cannot be read (or a listed one is missing), and with STRATA_PAGEFILE_GROWTH=0.  Re-read at most every 2 s.
uint64_t page_file_growth_bytes(uint64_t ram_total);

/// GlobalMemoryStatusEx's ullAvailPageFile plus page_file_growth_bytes(ram_total): the commit left counting what the
/// page files can still grow; exactly ullAvailPageFile when that is unknown or STRATA_PAGEFILE_GROWTH=0.
inline uint64_t available_commit(uint64_t avail_page_file, uint64_t ram_total) {
    return add_saturating(avail_page_file, page_file_growth_bytes(ram_total));
}
#endif

}  // namespace strata::platform
