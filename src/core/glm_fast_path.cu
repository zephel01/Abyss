// src/core/glm_fast_path.cu - the glm5-next FAST decode path (Glm5Model members).
//
// The same model as Glm5Model::step_layers (the correctness-first reference, STRATA_GLM_SLOW=1), run
// as ~11 fused launches per layer (strata/kernels/glm_fast.hpp) on a private non-blocking stream per
// device, with NO host synchronisation inside a token except where an expert is missing:
//
//   * every weight pointer is resolved once at load (no string lookups per layer per token);
//   * the router runs on the device and looks its experts up in a DEVICE table (layer x expert ->
//     VRAM slot pointer); an all-resident layer never waits for the host;
//   * every route is published to a host-mapped ring; a per-device SERVICE thread keeps the LFU
//     counts and, when a route has misses, reads the missing experts from the shards (pread into
//     pinned staging, in parallel), DMAs them into victim slots OF THAT LAYER on a copy stream and
//     answers through host-mapped memory - the device spins in a one-block wait kernel meanwhile.
//
// Victims come only from the requesting layer's own slot partition, so no kernel can be reading a
// slot while it is overwritten: the device is parked in that layer's wait kernel, the layer's
// resident experts are excluded, and every other layer's slots are untouched.
#include "glm_fast_state.hpp"
#include "shard_io.hpp"
#include "strata/core/knobs.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/qwen35_fast.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/exl3_cpu.hpp"
#include "strata/kernels/cpu/exl3_ref.hpp"

#include "ggml.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#if !defined(STRATA_USE_HIP)
#include <nvtx3/nvToolsExt.h>
#endif

#include <algorithm>
#include <numeric>
#include <cctype>
#include <cerrno>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <mutex>
#include <thread>
#include <vector>
#include <bit>
#ifndef _WIN32
#include <unistd.h>
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#ifdef __linux__
#include <dlfcn.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#endif

namespace gf = strata::kernels::glmf;

// Keeps the GPU out of its idle P-state for ns nanoseconds.  A consumer card at P8 drops its PCIe link (Gen1 on an
// RTX 4070 Ti SUPER / 5070 Ti), and copies alone do not wake it, so a link timed at idle reads 3-8x slower than it
// runs during decode (Tesla cards keep the link up, so this never showed there).
static __global__ void glm_link_wake(unsigned long long ns) {
#if defined(STRATA_USE_HIP)
    // no PTX %globaltimer on AMD: wall_clock64() is the constant 100 MHz counter (10 ns a tick) on gfx11 / gfx12, and
    // a wake-up spin needs no exact timing (from @boxwrench, #24)
    const long long t0 = wall_clock64();
    const long long ticks = (long long) (ns / 10ull);
    while (wall_clock64() - t0 < ticks) {
    }
#else
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
#endif
}

namespace strata::core {

using glmfast::cpu_relax;
using glmfast::Workers;

// The NUMA nodes this machine has (sysfs "online" list), and the most of this process's CPUs on any one of them;
// {} / 0 when unknown or on a single node.
static std::vector<int> numa_nodes() {
    std::vector<int> nodes;
#ifdef __linux__
    std::string list;
    std::ifstream("/sys/devices/system/node/online") >> list;
    std::stringstream ss(list);
    std::string r;
    while (std::getline(ss, r, ',')) {
        const size_t d = r.find('-');
        const int a = std::atoi(r.c_str()), b = d == std::string::npos ? a : std::atoi(r.c_str() + d + 1);
        for (int n = a; n <= b && n < 256; ++n) nodes.push_back(n);
    }
#endif
    return nodes;
}

// this process's CPUs on each NUMA node (sysfs cpulist), node by node; {} when unknown or on a single node
static std::vector<std::vector<int>> numa_node_cpu_lists() {
    std::vector<std::vector<int>> out;
#ifdef __linux__
    const auto nodes = numa_nodes();
    if (nodes.size() < 2) return out;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return out;
    for (int n : nodes) {
        std::string list;
        std::ifstream("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist") >> list;
        std::stringstream ss(list);
        std::string r;
        std::vector<int> cpus;
        while (std::getline(ss, r, ',')) {
            const size_t d = r.find('-');
            const int a = std::atoi(r.c_str()), b = d == std::string::npos ? a : std::atoi(r.c_str() + d + 1);
            for (int c = a; c <= b && c < CPU_SETSIZE; ++c)
                if (CPU_ISSET(c, &set)) cpus.push_back(c);
        }
        if (!cpus.empty()) out.push_back(std::move(cpus));
    }
#endif
    return out;
}

// this process's CPUs (its affinity mask) by NUMA node: {node, CPUs}, node by node - one group (node -1) without NUMA
static std::vector<std::pair<int, std::vector<int>>> cpu_groups() {
    std::vector<std::pair<int, std::vector<int>>> out;
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return out;
    for (int n : numa_nodes()) {
        std::string list;
        std::ifstream("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist") >> list;
        std::stringstream ss(list);
        std::string r;
        std::vector<int> cpus;
        while (std::getline(ss, r, ',')) {
            const size_t d = r.find('-');
            const int a = std::atoi(r.c_str()), b = d == std::string::npos ? a : std::atoi(r.c_str() + d + 1);
            for (int c = a; c <= b && c < CPU_SETSIZE; ++c)
                if (CPU_ISSET(c, &set)) cpus.push_back(c);
        }
        if (!cpus.empty()) out.push_back({n, std::move(cpus)});
    }
    if (out.empty()) {
        std::vector<int> cpus;
        for (int c = 0; c < CPU_SETSIZE; ++c)
            if (CPU_ISSET(c, &set)) cpus.push_back(c);
        if (!cpus.empty()) out.push_back({-1, std::move(cpus)});
    }
#endif
    return out;
}

// a CPU's physical core (package, core id): its SMT siblings share it
static std::pair<int, int> cpu_core(int cpu) {
    int pkg = cpu, core = cpu;
#ifdef __linux__
    const std::string t = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    std::ifstream(t + "physical_package_id") >> pkg;
    std::ifstream(t + "core_id") >> core;
#endif
    return {pkg, core};
}

// the NUMA node a GPU hangs off (sysfs, by its PCI address), -1 when unknown (a VM, or no NUMA)
static int gpu_numa_node(int dev) {
    int node = -1;
#ifdef __linux__
    char bus[32] = {0};
    if (cudaDeviceGetPCIBusId(bus, (int) sizeof bus, dev) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    std::string b(bus);
    for (char& ch : b) ch = (char) std::tolower((unsigned char) ch);
    std::ifstream("/sys/bus/pci/devices/" + b + "/numa_node") >> node;
#else
    (void) dev;
#endif
    return node;
}

// Part `part` of a split across `n` GPUs (`gpu_node`: each part's GPU's NUMA node): the CPUs its pool runs on - no
// other part's, on any machine - with the node it is on (-1: none known) and how many of them to leave free.  Whole
// NUMA nodes while there are as many as parts: each part on its GPU's node when the GPUs hang off distinct nodes,
// else part i on node i.  With fewer nodes than parts (one socket), the parts on a node share it out in runs of whole
// physical cores (SMT siblings together).  The 4 CPUs a node keeps free (the main thread's event wait, the service
// and warm-up threads, as one GPU's pool leaves) are shared by its parts.  {} with nothing to give.
static std::vector<int> part_cpus(int part, int n, const std::vector<int>& gpu_node, int& node, int& spare) {
    const auto groups = cpu_groups();
    node = -1;
    spare = 4;
    if (groups.empty() || n < 1 || part < 0 || part >= n) return {};
    const int G = (int) groups.size();
    std::vector<int> grp((size_t) n);
    bool by_gpu = n <= G && (int) gpu_node.size() == n;
    std::vector<bool> taken((size_t) G, false);
    for (int i = 0; by_gpu && i < n; ++i) {
        int at = -1;
        for (int j = 0; j < G; ++j)
            if (groups[(size_t) j].first >= 0 && groups[(size_t) j].first == gpu_node[(size_t) i]) at = j;
        by_gpu = at >= 0 && !taken[(size_t) at];
        if (by_gpu) {
            taken[(size_t) at] = true;
            grp[(size_t) i] = at;
        }
    }
    if (!by_gpu)
        for (int i = 0; i < n; ++i) grp[(size_t) i] = i % G;
    int k = 0, slot = 0;   // the parts on this part's node, and this part's place among them
    for (int i = 0; i < n; ++i)
        if (grp[(size_t) i] == grp[(size_t) part]) {
            if (i == part) slot = k;
            ++k;
        }
    const auto& g = groups[(size_t) grp[(size_t) part]];
    node = g.first;
    spare = (4 + k - 1) / k;
    if (k == 1) return g.second;
    std::vector<std::pair<std::pair<int, int>, int>> by_core;
    for (int c : g.second) by_core.push_back({cpu_core(c), c});
    std::sort(by_core.begin(), by_core.end());
    std::vector<std::vector<int>> cores;
    for (size_t i = 0; i < by_core.size(); ++i) {
        if (i == 0 || by_core[i].first != by_core[i - 1].first) cores.emplace_back();
        cores.back().push_back(by_core[i].second);
    }
    std::vector<int> out;
    const size_t a = cores.size() * (size_t) slot / (size_t) k, b = cores.size() * (size_t) (slot + 1) / (size_t) k;
    for (size_t j = a; j < b; ++j) out.insert(out.end(), cores[j].begin(), cores[j].end());
    std::sort(out.begin(), out.end());
    return out;
}

static int numa_node_cpus() {
    int best = 0;
    for (const auto& c : numa_node_cpu_lists()) best = std::max(best, (int) c.size());
    return best;
}

// The RAM tier's pinned memory spread page by page over every NUMA node: mmap + mbind(MPOL_INTERLEAVE) (+ transparent
// huge pages), then cudaHostRegister - the pages are placed when the registration faults them in, so the policy
// holds.  cudaHostAlloc's pages come from the driver node by node whatever the thread's policy (a 112 GB tier: 57.6 GB
// on one node, 72.7 GB on the other, each expert on ONE of them), and the CPU lane reads every expert with all its
// threads through that node's memory controllers only (2-socket Xeon, Q3_K experts, 44 threads: 216 us an expert
// node-local, 152 us interleaved).  Mapped, so the device reads its host pointers as it does cudaHostAlloc's.
// nullptr: one node, STRATA_GLM_NUMA=0, no host-pointer access for registered memory, or a failure.
// STRATA_GLM_RAM_THP (default on, 0 = off): this path on ONE node too, with MADV_HUGEPAGE on as much of the tier as
// the free 2 MB blocks cover - 512x fewer pages for the APU driver to re-validate when its MMU notifier fires: the
// iGPU lane's stalls of 15-24 s (7 in 8 x 1K tokens, "svm_range_restore_work hogged CPU" in the kernel log) were
// gone with 47.5 of a 60 GB tier in huge pages, and the lane registered the tier in 255 ms instead of 390 (RTX 5090 +
// Radeon 8060S, DeepSeek V4.1, 2026-10-18).  The caller drops the shards' page cache first so the 2 MB blocks exist.
// Registered memory's pages cannot be swapped either way.
static bool ram_thp() {
    static const bool on = [] {
        const char* v = getenv("STRATA_GLM_RAM_THP");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

static void* numa_pinned(size_t bytes) {
#ifdef __linux__
    const char* v = getenv("STRATA_GLM_NUMA");
    const auto nodes = numa_nodes();
    if ((v != nullptr && std::atoi(v) == 0) || (nodes.size() < 2 && !ram_thp())) return nullptr;
    int dev = 0, ok = 0;
    cudaGetDevice(&dev);
    if (cudaDeviceGetAttribute(&ok, cudaDevAttrCanUseHostPointerForRegisteredMem, dev) != cudaSuccess || !ok) {
        cudaGetLastError();
        return nullptr;
    }
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    unsigned long mask[4] = {0, 0, 0, 0};
    for (int n : nodes) mask[n / 64] |= 1ul << (n % 64);
    // huge pages only while free 2 MB blocks cover the tier (/proc/buddyinfo, orders 9 and up): otherwise every 2 MB
    // fault of a MADV_HUGEPAGE region compacts memory on the spot (transparent_hugepage/defrag "madvise") and mostly
    // fails - a split's second card, pinning its tier after the first card's had taken most of the RAM and the page
    // cache the rest (0.28 GB left in 2 MB blocks), sat in the kernel for over 10 minutes at ~10 MB/s.  4 KB pages
    // reclaim the cache as they fault in (the CPU lane reads them 6-8% slower than huge ones)
    size_t huge_free = 0;
    bool huge_known = false;
    if (FILE* bf = std::fopen("/proc/buddyinfo", "r")) {
        char line[512];
        while (std::fgets(line, sizeof line, bf)) {
            const char* z = std::strstr(line, "zone");
            if (z == nullptr) continue;
            z += 4;
            while (*z == ' ') ++z;
            while (*z != ' ' && *z != '\0') ++z;   // the zone's name
            unsigned long long c[16] = {};
            int k = 0;
            char* end = nullptr;
            for (const char* q = z; k < 16; ++k, q = end) {
                c[k] = std::strtoull(q, &end, 10);
                if (end == q) break;
            }
            for (int o = 9; o < k; ++o) huge_free += (size_t) c[o] << (o + 12);
            huge_known = true;
        }
        std::fclose(bf);
    }
    // STRATA_GLM_RAM_THP: huge pages for the part of the tier the free blocks cover (less 1 GB), 4 KB pages for the
    // rest - asking for the whole tier made the faults compact for the remainder, the memory pressure that caused sat
    // over systemd-oomd's limit (50% for 20 s) and the engine was killed (60 GB tier, 45-50 GB in free blocks, 2026-10-18)
    size_t thp_bytes = 0;
    if (!huge_known || huge_free >= bytes + ((size_t) 1 << 30)) thp_bytes = bytes;
    else if (ram_thp() && huge_free > ((size_t) 1 << 30))
        thp_bytes = std::min(bytes, (huge_free - ((size_t) 1 << 30)) & ~(((size_t) 2 << 20) - 1));
    if (thp_bytes > 0) madvise(p, thp_bytes, MADV_HUGEPAGE);
    // (the registration faults the pages in, one thread: faulting them from 16 threads first started ~40 s sooner but
    // left fewer huge pages - the CPU lane read its experts 6-8% slower, 2-socket Xeon 6152)
    if ((nodes.size() >= 2 &&
         syscall(SYS_mbind, p, bytes, 3 /* MPOL_INTERLEAVE */, mask, (unsigned long) (sizeof mask * 8), 0u) != 0) ||
        cudaHostRegister(p, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
        cudaGetLastError();
        munmap(p, bytes);
        return nullptr;
    }
    if (ram_thp()) {   // what the faults gave: the process's huge pages before and after this tier
        long long huge_kb = 0;
        if (FILE* sm = std::fopen("/proc/self/smaps_rollup", "r")) {
            char line[256];
            while (std::fgets(line, sizeof line, sm))
                if (std::sscanf(line, "AnonHugePages: %lld kB", &huge_kb) == 1) break;
            std::fclose(sm);
        }
        std::fprintf(stderr, "glm fast: RAM tier %.1f GB registered from an mmap, MADV_HUGEPAGE on %.1f GB of it; the "
                             "process now holds %.1f GB in transparent huge pages (free 2 MB blocks before: %.1f GB)\n",
                     (double) bytes / 1073741824.0, (double) thp_bytes / 1073741824.0, (double) huge_kb / 1048576.0,
                     (double) huge_free / 1073741824.0);
    }
    return p;
#else
    (void) bytes;
    return nullptr;
#endif
}

// the machine's RAM: MemTotal (Linux; 0 when unknown)
static int64_t mem_total() {
    long long kb = 0;
#ifdef __linux__
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f))
            if (std::sscanf(line, "MemTotal: %lld kB", &kb) == 1) break;
        std::fclose(f);
    }
#endif
    return (int64_t) kb * 1024;
}

// the RAM free now: MemAvailable (Windows: the smaller of the free RAM and the free commit, which pinning is charged to);
// 0 when unknown
static int64_t avail_ram_now() {
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (int64_t) std::min(ms.ullAvailPhys, ms.ullAvailPageFile) : 0;
#else
    long long kb = 0;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f))
            if (std::sscanf(line, "MemAvailable: %lld kB", &kb) == 1) break;
        std::fclose(f);
    }
    return (int64_t) kb * 1024;
#endif
}

static void numa_pinned_free(void* p, size_t bytes) {
#ifdef __linux__
    cudaHostUnregister(p);
    munmap(p, bytes);
#else
    (void) p;
    (void) bytes;
#endif
}

namespace glmfast {
// the VRAM a discrete card keeps free beside the expert pool (STRATA_GLM_RESERVE_MB): ~700 MB for the driver, the
// sampler and kernel launches' local memory; 1.5 GB on HIP, where the first prompt's hipBLASLt GEMMs load their code
// objects and take ~0.7 GB more (an R9700 at --max-context 131072 had 0.6 GB left: every prompt GEMM failed to load
// and fell back, an 8K prompt 1,300 s).  CUDA too from v1.3.82 on cards over 12 GB: an RTX 5090 running DeepSeek V4.1
// at --max-context 65536 ran out in the attention's and the DSpark drafter's MMQ buffers with 700 kept.  A card of
// 12 GB or less (or STRATA_GLM_VRAM_GB of that) keeps ~700 MB: its pool is a few slots a layer (issue #23)
constexpr int kVramReserveMb = 1536;
inline int default_reserve_mb(size_t total_b) {
#if defined(STRATA_USE_HIP)
    (void) total_b;
    return kVramReserveMb;
#else
    const double cap = strata::core::env_num("STRATA_GLM_VRAM_GB", 0.0, 0.0, 65536.0) * 1073741824.0;
    const double card = cap > 0.0 && cap < (double) total_b ? cap : (double) total_b;
    return card > 12.5 * 1073741824.0 ? kVramReserveMb : 700;
#endif
}

// RAM left free beside the pinned expert tier (STRATA_GLM_RAM_HEADROOM_GB): 6 GB with a discrete card, 16 on a
// unified-memory APU, whose pool is that RAM - the one place both tier sizings read it from (the AMD setup writes 16
// into its configs on purpose)
inline double ram_headroom_gb(bool unified) {
    return strata::core::env_num("STRATA_GLM_RAM_HEADROOM_GB", unified ? 16.0 : 6.0, 0.0, 4096.0);
}
// the VRAM kept free, from the build's default or STRATA_GLM_RESERVE_MB (never negative: it was cast to size_t)
inline size_t vram_reserve_bytes(int default_mb) {
    return (size_t) strata::core::env_num("STRATA_GLM_RESERVE_MB", default_mb, 0.0, 1048576.0) << 20;
}
// STRATA_GLM_POOL_GB / STRATA_GLM_VRAM_GB as bytes, or 0 when unset
inline size_t env_gib(const char* name) {
    return (size_t) (strata::core::env_num(name, 0.0, 0.0, 65536.0) * 1073741824.0);
}

// The 2nd SSD (STRATA_GLM_MIRROR, see pack_shard_mmap), the O_DIRECT bounce buffers, the fallbacks and their counters
// live in shard_io.cpp (host-only, tested by tools/test_shard_io.py).

// [off, off + len) of a shard into dst: O_DIRECT through an aligned per-thread bounce buffer when the shard has a
// direct fd (no page cache), a buffered pread otherwise, the mapping as the last resort
void read_slice(const Glm5Model::Shard& sh, uint64_t off, size_t len, uint8_t* dst) {
#ifndef _WIN32
    namespace sio = strata::core::shard_io;
    if (sh.fd_direct >= 0 && sio::read_direct(sh.fd_direct, sh.fd_mirror, off, len, dst)) return;
    if (sio::read_buffered(sh.fd, sh.base, off, len, dst)) return;
    // the views went after the load (pack_release_views) and the file could not be read: nothing to fall back to
    std::fprintf(stderr, "glm fast: an expert read of %s at %llu failed (%s) - STRATA_GLM_KEEP_MAP=1 keeps the mapping "
                         "to fall back to\n", sh.path.c_str(), (unsigned long long) off, std::strerror(errno));
    std::abort();
#else
    // Windows: the unbuffered handle the same way (sector-aligned offset, size and buffer), the mapping otherwise.
    // The handle is overlapped (reads from many threads run at once): each read waits on this thread's own event
    if (sh.h_direct != nullptr) {
        // freed and closed when its thread ends
        struct Bounce {
            uint8_t* p = nullptr;
            size_t cap = 0;
            HANDLE ev = CreateEventA(nullptr, TRUE, FALSE, nullptr);
            ~Bounce() {
                _aligned_free(p);
                if (ev != nullptr) CloseHandle(ev);
            }
        };
        static thread_local Bounce bb;
        uint8_t*& bounce = bb.p;
        size_t& cap = bb.cap;
        const HANDLE done_ev = bb.ev;
        const uint64_t a0 = off & ~(uint64_t) 4095, a1 = (off + len + 4095) & ~(uint64_t) 4095;
        const size_t need = (size_t) (a1 - a0);
        if (cap < need) {
            _aligned_free(bounce);
            bounce = (uint8_t*) _aligned_malloc(need, 4096);
            cap = bounce ? need : 0;
        }
        if (bounce != nullptr && done_ev != nullptr) {
            size_t got = 0;
            while (got < need) {
                OVERLAPPED ov{};
                const uint64_t at = a0 + got;
                ov.Offset = (DWORD) at;
                ov.OffsetHigh = (DWORD) (at >> 32);
                ov.hEvent = done_ev;
                const DWORD want = (DWORD) std::min<size_t>(need - got, (size_t) 1 << 30);
                DWORD r = 0;
                if (!ReadFile((HANDLE) sh.h_direct, bounce + got, want, nullptr, &ov) &&
                    GetLastError() != ERROR_IO_PENDING) break;   // EOF: the last aligned read is short
                if (!GetOverlappedResult((HANDLE) sh.h_direct, &ov, &r, TRUE) || r == 0) break;
                got += (size_t) r;
            }
            if (got >= (size_t) (off - a0) + len) {
                std::memcpy(dst, bounce + (off - a0), len);
                return;
            }
        }
    }
    if (sh.base == nullptr) {   // the views went after the load (pack_release_views): nothing to fall back to
        std::fprintf(stderr, "glm fast: an expert read of %s failed (error %lu) - STRATA_GLM_KEEP_MAP=1 keeps the "
                             "mapping to fall back to\n", sh.path.c_str(), (unsigned long) GetLastError());
        std::abort();
    }
    std::memcpy(dst, sh.base + off, len);
#endif
}

}  // namespace glmfast

using glmfast::read_slice;

// the route's slots the buffers hold: 8 as before (the same layout for every top-8-or-less model), the top-k above that
static_assert(gf::kMaxK == 16, "fast_ahead_route's declaration (glm_model.hpp) spells kMaxK as 16");
static int k_slots(const Glm5Geometry& g) { return std::max(8, g.n_exp_used); }
// the CPU lane's split table covers f = 0..plan_f(g) RAM-tier experts a route (4 bits each in 64: at most 15)
static int plan_f(const Glm5Geometry& g) { return std::min(15, k_slots(g)); }

static float* fa_take(uint8_t*& at, size_t floats) {
    float* p = (float*) at;
    at += (floats * sizeof(float) + 255u) & ~(size_t) 255u;
    return p;
}
static void* fa_take_b(uint8_t*& at, size_t bytes) {
    void* p = at;
    at += (bytes + 255u) & ~(size_t) 255u;
    return p;
}

bool Glm5Model::fast_setup(std::string& err) {
    cudaSetDevice(dev_);
    auto* F = new FastState();
    fast_ = F;
    // Not on Windows: there an APU's GPU memory is a fixed carve-out (AMD's Variable Graphics Memory, or the BIOS)
    // that Windows does not count as RAM, so it is sized like a discrete card - the pool from what HIP reports free,
    // the RAM tier from the free RAM and commit (below)
#if defined(STRATA_USE_HIP) && !defined(_WIN32)
    int integrated = 0;
    if (cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev_) == cudaSuccess)
        F->unified_memory = integrated != 0;
    else
        cudaGetLastError();
#endif
    const Glm5Geometry& g = g_;
    F->timing = getenv("STRATA_GLM_TIMING") != nullptr;
    F->prof_on = getenv("STRATA_GLM_PROF") != nullptr;
    F->gap_on = getenv("STRATA_GLM_GAP_PROF") != nullptr && std::atoi(getenv("STRATA_GLM_GAP_PROF")) != 0;
    F->token_spin = getenv("STRATA_GLM_TOKEN_SPIN") != nullptr && std::atoi(getenv("STRATA_GLM_TOKEN_SPIN")) != 0;
    if (F->prof_on) F->pskip = (uint64_t) std::max(0, std::atoi(getenv("STRATA_GLM_PROF")));
    if (const char* pn = getenv("STRATA_GLM_PREFETCH_N")) F->max_pf = std::max(0, std::min(gf::kSpares - 1, std::atoi(pn)));
    if (const char* bp = getenv("STRATA_GLM_BG_PACE_KB")) F->bg_pace = (size_t) std::max(0, std::atoi(bp)) << 10;
    else F->bg_pace = (size_t) 1024 << 10;   // (FastState::bg_drain)
    // RAM-resident mode (--glm-ram-resident / STRATA_GLM_RAM_RESIDENT=1): the RAM tier's per-class cap grows by
    // ram_slack slots per MoE layer so it holds EVERY expert VRAM does not; nothing is ever dropped to disk and the
    // start fails when the tier cannot hold them all - the disk is the pack's home, never a read source at runtime
    F->ram_resident = getenv("STRATA_GLM_RAM_RESIDENT") != nullptr && std::atoi(getenv("STRATA_GLM_RAM_RESIDENT")) != 0;
    if (const char* rs = getenv("STRATA_GLM_RAM_SLACK")) F->ram_slack = std::max(0, std::atoi(rs));
    else if (F->ram_resident) F->ram_slack = 16;
    if (cudaStreamCreateWithFlags(&F->cs, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&F->copy, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&F->ps, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&F->ev_hop, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&F->ev_pred, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&F->ev_pf, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&F->ev_pf_prev, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&F->ev_done, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&F->ev_bgq, cudaEventDisableTiming) != cudaSuccess) {
        err = "glm fast: streams/events did not create";
        return false;
    }
    if (F->gap_on && (cudaEventCreate(&F->ev_gs) != cudaSuccess || cudaEventCreate(&F->ev_ge[0]) != cudaSuccess ||
                      cudaEventCreate(&F->ev_ge[1]) != cudaSuccess || cudaEventCreate(&F->ev_g0) != cudaSuccess)) {
        err = "glm fast: gap profile events did not create";
        return false;
    }
    // STRATA_GLM_ABL=<file> (--uncensor <file>): the refusal-direction edit at run time (src/core/glm_abl.cu)
    if (const char* ap = getenv("STRATA_GLM_ABL"); ap != nullptr && ap[0]) {
        if (!abl_load(ap, err)) return false;
        // STRATA_GLM_ABL_ON=0: loaded but switched off (what a request with abl=0 runs; the check that off = no file)
        if (const char* on = getenv("STRATA_GLM_ABL_ON"); on != nullptr && on[0] == '0') abl_enable(false);
    }
    // an EXL3 pack: its kernels' scratch on the compute stream now (a graph capture cannot allocate it)
    if (exl3_ && !strata::kernels::exl3::prepare(F->cs)) {
        err = "glm fast: the EXL3 scratch did not allocate";
        return false;
    }

    // ---- the weights, resolved once
    const int NL = g.n_layers + 1;   // every per-layer table: the trunk + the NextN block's slot
    F->L.assign((size_t) NL, FastState::Layer{});
    std::string missing;
    for (int il = l0_; il < lt_; ++il) {
        const bool is_mtp = il == mtp_il_;
        auto& Ly = F->L[(size_t) il];
        const std::string P = "blk." + std::to_string(il) + ".";
        const auto f32 = [&](const char* n) -> const float* {
            auto it = w_.find(P + n);
            if (it == w_.end()) { missing += P + n + " (f32) "; return nullptr; }
            return it->second;
        };
        const auto b16 = [&](const char* n) -> const uint16_t* {
            auto it = w16_.find(P + n);
            if (it == w16_.end()) { missing += P + n + " (bf16) "; return nullptr; }
            return it->second;
        };
        const auto q = [&](const char* n) -> WSlot {
            auto it = ws_map_.find(P + n);
            if (it == ws_map_.end() || it->second.type == 0 || !gf::mv_supported(it->second.type)) {
                missing += P + n + " (quant) ";
                return WSlot{};
            }
            return it->second;
        };
        // a projection a GGUF keeps unquantized (BF16 - e.g. attn_kv_a_mqa in some community GGUFs) where Maya's
        // quants quantize it: the BF16 copy; the jobs that take these pass the float activation as well
        const auto q_or_b16 = [&](const char* n, int64_t n_in, int64_t n_out) -> WSlot {
            const auto it = ws_map_.find(P + n);
            if (it == ws_map_.end() || it->second.type == 0) {
                const auto b = w16_.find(P + n);
                if (b != w16_.end()) return WSlot{nullptr, gf::kTypeBF16, b->second, n_in, n_out};
            }
            return q(n);
        };
        Ly.recr = is_mtp ? false : g.is_recr(il);
        Ly.moe = il >= g.dense_lead;
        Ly.mtp = is_mtp;
        if (g.qwen35()) {
            qwen35_bind(il, missing);   // (qwen35_fast_path.cu: no hyper-connections, its own mixers)
        } else {
        if (is_mtp) {
            // no hyper-connections: eh_proj in, plain residuals, the shared head's norm out
            Ly.eh = q("nextn.eh_proj.weight");
            Ly.enorm = f32("nextn.enorm.weight");
            Ly.hnorm = f32("nextn.hnorm.weight");
            Ly.shnorm = f32("nextn.shared_head_norm.weight");
        } else {
            Ly.hc_attn_fn = b16("hc_attn_fn.weight");
            Ly.hc_ffn_fn = b16("hc_ffn_fn.weight");
            Ly.hc_attn_scale = f32("hc_attn_scale.weight");
            Ly.hc_attn_base = f32("hc_attn_base.weight");
            Ly.hc_ffn_scale = f32("hc_ffn_scale.weight");
            Ly.hc_ffn_base = f32("hc_ffn_base.weight");
        }
        Ly.attn_norm = f32("attn_norm.weight");
        Ly.ffn_norm = f32("ffn_norm.weight");
        if (g.dsv4()) {
            // the attention mixer's weights: dsv4_bind (dsv4_fast_path.cu)
        } else if (Ly.recr) {
            Ly.q = q("attn_q.weight");
            Ly.k = q("attn_k.weight");
            Ly.v = q("attn_v.weight");
            Ly.f_a = b16("ssm_f_a.weight");
            Ly.g_a = b16("ssm_g_a.weight");
            Ly.f_b = b16("ssm_f_b.weight");
            Ly.g_b = b16("ssm_g_b.weight");
            Ly.beta = b16("ssm_beta.weight");
            Ly.conv[0] = f32("ssm_conv1d_q.weight");
            Ly.conv[1] = f32("ssm_conv1d_k.weight");
            Ly.conv[2] = f32("ssm_conv1d_v.weight");
            Ly.dt_bias = f32("ssm_dt.bias");
            Ly.ssm_a = f32("ssm_a");
            Ly.ssm_norm = f32("ssm_norm.weight");
        } else {
            Ly.q_a = q_or_b16("attn_q_a.weight", g.n_embd, g.q_lora);
            Ly.q_b = q("attn_q_b.weight");
            Ly.kv_a = q_or_b16("attn_kv_a_mqa.weight", g.n_embd, g.kv_lora);
            Ly.q_a_norm = f32("attn_q_a_norm.weight");
            Ly.kv_a_norm = f32("attn_kv_a_norm.weight");
            Ly.k_norm_w = f32("indexer.k_norm.weight");
            Ly.k_norm_b = f32("indexer.k_norm.bias");
            Ly.ape = f32("indexer_compressor_ape.weight");
            Ly.idx_k = b16("indexer.attn_k.weight");
            Ly.idx_gate = b16("indexer_compressor_gate.weight");
            if (const auto xi = ws_map_.find(P + "indexer.attn_q_b.weight");
                xi != ws_map_.end() && xi->second.type == strata::kernels::exl3::kTypeEXL3)
                Ly.idx_q_b_x = xi->second;   // an EXL3 pack quantizes it
            else
                Ly.idx_q_b = b16("indexer.attn_q_b.weight");
            Ly.idx_proj = b16("indexer.proj.weight");
            Ly.k_b = b16("attn_k_b.weight");
            Ly.v_b = b16("attn_v_b.weight");
        }
        if (!g.dsv4()) Ly.out = q("attn_output.weight");
        }   // (!qwen35)
        if (Ly.moe) {
            if ((g.v41 || g.qwen35()) && w16_.find(P + "ffn_gate_inp.weight") == w16_.end()) {   // (F32 routers, served as F32)
                Ly.router = (const uint16_t*) f32("ffn_gate_inp.weight");
                Ly.router_type = gf::kTypeF32;
            } else {
                Ly.router = b16("ffn_gate_inp.weight");
            }
            if (!(g.dsv4() && il < g.hash_layers) && !g.qwen35()) Ly.router_bias = f32("exp_probs_b.bias");   // (hash layers, qwen35moe: none)
            Ly.sh_gate = q("ffn_gate_shexp.weight");
            Ly.sh_up = q("ffn_up_shexp.weight");
            Ly.sh_down = q("ffn_down_shexp.weight");
            const auto& nl = pack_layers_[(size_t) il];
            if (nl.layer < 0) {
                missing += P + "native experts ";
            } else if (nl.x_cb >= 0) {
                // EXL3: the blob is [gate | up | down], a part [trellis | suh | svh]
                Ly.exl3 = true;
                Ly.gu_type = Ly.d_type = strata::kernels::exl3::kTypeEXL3;
                Ly.gu_bytes = nl.x_part(0);
                Ly.dn_bytes = nl.x_part(2);
                Ly.down_off = 2 * Ly.gu_bytes;
                Ly.blob = 2 * Ly.gu_bytes + Ly.dn_bytes;
                Ly.xl.cb = nl.x_cb;
                for (int r = 0; r < 3; ++r) {
                    const size_t base = r == 0 ? 0 : r == 1 ? Ly.gu_bytes : Ly.down_off;
                    Ly.xl.trellis[r] = base;
                    Ly.xl.suh[r] = base + nl.x_piece(r, 0);
                    Ly.xl.svh[r] = base + nl.x_piece(r, 0) + nl.x_piece(r, 1);
                    Ly.xl.K[r] = nl.x_K[r];
                }
            } else {
                Ly.gu_type = nl.fmt.gu_type;
                Ly.d_type = nl.fmt.d_type;
                Ly.gu_bytes = (size_t) nl.fmt.gu_row * (size_t) g.n_ff_exp;
                Ly.dn_bytes = (size_t) nl.fmt.d_row * (size_t) g.n_embd;
                Ly.down_off = 2 * Ly.gu_bytes;
                Ly.blob = 2 * Ly.gu_bytes + Ly.dn_bytes;
                if (!gf::moe_supported(Ly.gu_type) || !gf::moe_supported(Ly.d_type) ||
                    gf::row_bytes(Ly.gu_type, g.n_embd) != nl.fmt.gu_row ||
                    gf::row_bytes(Ly.d_type, g.n_ff_exp) != nl.fmt.d_row)
                    missing += P + "expert types " + std::to_string(Ly.gu_type) + "/" + std::to_string(Ly.d_type) + " ";
            }
        } else {
            Ly.ffn_gate = q("ffn_gate.weight");
            Ly.ffn_up = q("ffn_up.weight");
            Ly.ffn_down = q("ffn_down.weight");
        }
    }
    if (!missing.empty()) {
        err = "glm fast: unresolved weights: " + missing.substr(0, 4000);
        return false;
    }
    if (g.dsv4() && !g.v41 && (!dsv4_bind(err) || !dsv4_alloc(err))) return false;
    if (g.v41 && (!dsv41_bind(err) || !dsv41_alloc(err))) return false;
    if (g.qwen35() && !qwen35_alloc(err)) return false;

    // ---- the activation arena
    const int64_t E = g.n_embd, DI = g.d_inner(), FF = g.n_ff_exp * g.n_shared;
    const int64_t max_pools = std::max<int64_t>(1, max_ctx_ / g.idx_kpool);
    const size_t q8 = 36;   // bytes per q8_1 block
    size_t need = 0;
    {
        // generous upper bound, carved below
        need = (size_t) (4 * E + 64 + 32 * 36 + 128 + 6 * DI + 4 * 128 + 2 * DI + 2 * g.q_lora + 2 * g.kv_lora +
                         4 * g.idx_key + 64 + (int64_t) g.n_head * g.qk_nope + (int64_t) g.idx_heads * g.idx_key +
                         max_pools + g.n_sel_max() + 2 * g.n_expert + 2 * FF + 2 * g.n_ff_dense + 2 * E + 64) *
                   sizeof(float);
        need += (size_t) (E + DI + g.q_lora + (int64_t) g.n_head * g.v_head + FF + (int64_t) k_slots(g) * g.n_ff_exp + g.n_ff_dense + E) /
                32 * q8;
        need += 64 * 256 + (size_t) (2 * NL * g.n_expert + NL * gf::kSpares) * sizeof(unsigned long long) +
                8 * 16 + 6 * 256 + gf::kMaxEmbd * 4 + 4 * 256 + (size_t) g.n_expert * 4 + 256 + 4 * 256;
        need += (size_t) gf::kAhead * g.n_expert * sizeof(float) + 256 + (size_t) E * sizeof(float) + 256;
    }
    if (cudaMalloc(&F->arena, need) != cudaSuccess) {
        err = "glm fast: the activation arena did not allocate";
        return false;
    }
    cudaMemset(F->arena, 0, need);
    F->arena_bytes = need;
    uint8_t* at = (uint8_t*) F->arena;
    F->x = fa_take(at, E);
    F->mixer = fa_take(at, E);
    F->ffn = fa_take(at, E);
    F->pre = fa_take(at, 8);
    F->post = fa_take(at, 8);
    F->post_defer = fa_take(at, 8);
    F->comb = fa_take(at, 16);
    F->part = fa_take(at, (size_t) (4 * E / 512) * 36 + 64);   // 25 partials per block + 10 Gram + a spare row
    F->counter = (unsigned int*) fa_take_b(at, 64);
    F->xq = fa_take_b(at, (size_t) E / 32 * q8);
    for (int i = 0; i < 3; ++i) F->proj[i] = fa_take(at, DI);
    for (int i = 0; i < 3; ++i) F->conv[i] = fa_take(at, DI);
    F->fa = fa_take(at, g.kda_head_dim);
    F->ga = fa_take(at, g.kda_head_dim);
    F->beta = fa_take(at, g.n_head);
    F->g1 = fa_take(at, DI);
    F->g2 = fa_take(at, DI);
    F->gated_q = fa_take_b(at, (size_t) DI / 32 * q8);
    F->qr_raw = fa_take(at, g.q_lora);
    F->qr = fa_take(at, g.q_lora);
    F->qr_q = fa_take_b(at, (size_t) g.q_lora / 32 * q8);
    F->kv_raw = fa_take(at, g.kv_lora);
    F->ik_raw = fa_take(at, g.idx_key);
    F->ig_raw = fa_take(at, g.idx_key);
    F->iw = fa_take(at, g.idx_heads);
    F->q = fa_take(at, (size_t) g.n_head * g.qk_nope);
    F->iq = fa_take(at, (size_t) g.idx_heads * g.idx_key);
    F->score = fa_take(at, (size_t) max_pools);
    F->cells = (int*) fa_take(at, (size_t) g.n_sel_max());
    F->attn_q = fa_take_b(at, (size_t) g.n_head * g.v_head / 32 * q8);
    F->rlog = fa_take(at, g.n_expert);
    F->plog = fa_take(at, g.n_expert);
    F->alog = fa_take(at, (size_t) gf::kAhead * g.n_expert);
    F->sh_out = fa_take(at, E);
    F->sh_g = fa_take(at, FF);
    F->sh_u = fa_take(at, FF);
    F->sh_hq = fa_take_b(at, (size_t) FF / 32 * q8);
    F->hq = fa_take_b(at, (size_t) k_slots(g) * g.n_ff_exp / 32 * q8);
    F->dg = fa_take(at, g.n_ff_dense);
    F->du = fa_take(at, g.n_ff_dense);
    F->dhq = fa_take_b(at, (size_t) g.n_ff_dense / 32 * q8);
    F->head_x = fa_take(at, E);
    F->head_xq = fa_take_b(at, (size_t) E / 32 * q8);
    F->emb = fa_take(at, E);
    F->tok = (int*) fa_take_b(at, 64);
    F->n_keys = NL * g.n_expert;
    F->tab = (unsigned long long*) fa_take_b(
        at, (size_t) (2 * F->n_keys + NL * gf::kSpares) * sizeof(unsigned long long));
    F->md.tab = F->tab;
    F->md.n_keys = F->n_keys;
    F->md.scratch = (unsigned long long*) fa_take_b(at, k_slots(g) * sizeof(unsigned long long));
    F->md.fetch_src = (unsigned long long*) fa_take_b(at, k_slots(g) * sizeof(unsigned long long));
    for (int b2 = 0; b2 < 2; ++b2) {
        F->pf_buf[b2] = (unsigned long long*) fa_take_b(at, 16 * sizeof(unsigned long long));
        F->pf_n_buf[b2] = (int*) fa_take_b(at, 64);
    }
    F->md.plan_ptr = (unsigned long long*) fa_take_b(at, k_slots(g) * sizeof(unsigned long long));
    F->md.plan_w = fa_take(at, k_slots(g));
    F->md.plan_id = (int*) fa_take(at, k_slots(g));
    F->md.seq = (unsigned int*) fa_take_b(at, 64);
    F->md.wait_seq = (unsigned int*) fa_take_b(at, 64);
    F->md.cpu_part = fa_take(at, E);
    F->md.cpu_flag = (int*) fa_take_b(at, 64);
    if ((size_t) (at - (uint8_t*) F->arena) > need) {
        err = "glm fast: the activation arena overflowed its estimate";
        return false;
    }
    if (mtp_il_ >= 0) {
        if (cudaMalloc(&F->mtp_h, (size_t) g.n_embd * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&F->mtp_catq, (size_t) 2 * g.n_embd / 32 * 36) != cudaSuccess ||
            cudaMalloc(&F->mtp_logits, (size_t) g.n_vocab * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&F->mtp_tok, 64) != cudaSuccess ||
            cudaHostAlloc((void**) &F->mtp_tok_h, 64, cudaHostAllocDefault) != cudaSuccess ||
            cudaEventCreateWithFlags(&F->ev_mtp, cudaEventDisableTiming) != cudaSuccess) {
            err = "glm fast: the NextN block's buffers did not allocate";
            return false;
        }
    }

    // ---- host-mapped routing ring + response, pinned staging for the embedding / hop / token
    void* hp = nullptr;
    if (cudaHostAlloc(&hp, sizeof(gf::MoeRequest) * gf::kRingSize + sizeof(int), cudaHostAllocMapped) != cudaSuccess) {
        err = "glm fast: the routing ring did not allocate";
        return false;
    }
    std::memset(hp, 0, sizeof(gf::MoeRequest) * gf::kRingSize + sizeof(int));
    F->ring_h = (gf::MoeRequest*) hp;
    void* dp = nullptr;
    cudaHostGetDevicePointer(&dp, hp, 0);
    F->md.ring = dp;
    F->route_error_h = (volatile int*) (F->ring_h + gf::kRingSize);
    F->md.route_error = (volatile int*) ((gf::MoeRequest*) dp + gf::kRingSize);
    if (cudaMalloc((void**) &F->md.route_error_d, 64) != cudaSuccess ||
        cudaMemset(F->md.route_error_d, 0, 64) != cudaSuccess) {
        err = "glm fast: the route error flag did not allocate";
        return false;
    }
    if (cudaHostAlloc(&hp, gf::kCpuAnswers * sizeof(gf::MoeResponse), cudaHostAllocMapped) != cudaSuccess) {
        err = "glm fast: the routing response did not allocate";
        return false;
    }
    std::memset(hp, 0, gf::kCpuAnswers * sizeof(gf::MoeResponse));
    F->resp_h = (gf::MoeResponse*) hp;
    cudaHostGetDevicePointer(&dp, hp, 0);
    F->md.resp = dp;
    if (cudaHostAlloc((void**) &F->emb_h, (size_t) E * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &F->hop_h, (size_t) g.hc * E * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &F->tok_h, 64, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostAlloc((void**) &F->tok_in_h, 64 * sizeof(int), cudaHostAllocMapped) != cudaSuccess) {
        err = "glm fast: pinned staging did not allocate";
        return false;
    }
    F->zero_copy = getenv("STRATA_GLM_ZERO_COPY") == nullptr || std::atoi(getenv("STRATA_GLM_ZERO_COPY")) != 0;
    if (cudaHostGetDevicePointer((void**) &F->tok_hd, F->tok_h, 0) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &F->tok_in_d, F->tok_in_h, 0) != cudaSuccess) {
        err = "glm fast: host-mapped token buffers have no device address";
        return false;
    }
    std::memset(F->tok_h, 0, 64);
    std::memset(F->tok_in_h, 0, 64 * sizeof(int));
    // the pinned copy of token_embd (Q8_0 / Q5_K - the other types keep the host dequantizer and its copy)
    if (F->zero_copy && pack_emb_src_ != nullptr && gf::embed_rows_q_supported(pack_emb_type_, (int) g.n_embd)) {
        const size_t bytes = (size_t) g.n_vocab * ggml_row_size((ggml_type) pack_emb_type_, g.n_embd);
        if (cudaHostAlloc((void**) &F->emb_tab_h, bytes, cudaHostAllocMapped) == cudaSuccess) {
            if (cudaHostGetDevicePointer((void**) &F->emb_tab_d, F->emb_tab_h, 0) == cudaSuccess) {
                std::memcpy(F->emb_tab_h, pack_emb_src_, bytes);
            } else {
                cudaFreeHost(F->emb_tab_h);
                F->emb_tab_h = nullptr;
            }
        } else {
            cudaGetLastError();
            F->emb_tab_h = nullptr;
        }
        if (F->emb_tab_h == nullptr)
            std::fprintf(stderr, "glm fast: token_embd (%.0f MB) did not pin: the embedding row goes through a copy\n",
                         (double) bytes / 1048576.0);
        else
            std::fprintf(stderr, "glm fast: CUDA%d zero-copy token boundary: token_embd pinned (%.0f MB), the table "
                                 "edits and the token are read and written host-mapped\n", dev_, (double) bytes / 1048576.0);
    }
    // the NextN drafter reads its token's embedding on the device: a table the GPU cannot read (IQ3_S, IQ4_XS, ...;
    // or STRATA_GLM_ZERO_COPY=0) gets a Q8_0 copy for the drafter only - the trunk keeps its exact host row
    // (qwen4exp's draft layer, STRATA_Q4X_MTP, the same)
    const char* q4m = getenv("STRATA_Q4X_MTP");
    const bool q4_draft = g.q4x && q4m != nullptr && q4m[0] != '\0' && getenv("STRATA_GLM_NO_MTP") == nullptr;
    if (g.qwen35() && (mtp_il_ >= 0 || q4_draft) && F->emb_tab_d == nullptr && pack_emb_src_ != nullptr) {
        const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
        const ggml_type_traits* tq = ggml_get_type_traits(GGML_TYPE_Q8_0);
        const int E = (int) g.n_embd;
        const size_t src_rb = ggml_row_size((ggml_type) pack_emb_type_, E), q8_rb = ggml_row_size(GGML_TYPE_Q8_0, E);
        const size_t bytes = (size_t) g.n_vocab * q8_rb;
        if (tt == nullptr || tt->to_float == nullptr || tq->from_float_ref == nullptr || E % 32 != 0) {
            std::fprintf(stderr, "glm fast: the NextN drafter is off: token_embd (%s) has no row reader\n",
                         ggml_type_name((ggml_type) pack_emb_type_));
        } else if (cudaHostAlloc((void**) &F->emb_q8_h, bytes, cudaHostAllocMapped) != cudaSuccess ||
                   cudaHostGetDevicePointer((void**) &F->emb_q8_d, F->emb_q8_h, 0) != cudaSuccess) {
            cudaGetLastError();
            if (F->emb_q8_h) cudaFreeHost(F->emb_q8_h);
            F->emb_q8_h = F->emb_q8_d = nullptr;
            std::fprintf(stderr, "glm fast: the NextN drafter is off: its Q8_0 copy of token_embd (%.0f MB) did not pin\n",
                         (double) bytes / 1048576.0);
        } else {
            const int64_t V = (int64_t) g.n_vocab;
            const int nt = std::max(1, std::min(16, (int) std::thread::hardware_concurrency()));
            std::vector<std::thread> th;
            for (int k = 0; k < nt; ++k)
                th.emplace_back([&, k] {
                    std::vector<float> row((size_t) E);
                    for (int64_t r = V * k / nt; r < V * (k + 1) / nt; ++r) {
                        tt->to_float(pack_emb_src_ + (size_t) r * src_rb, row.data(), E);
                        tq->from_float_ref(row.data(), F->emb_q8_h + (size_t) r * q8_rb, E);
                    }
                });
            for (auto& t : th) t.join();
            std::fprintf(stderr, "glm fast: CUDA%d the NextN drafter reads a Q8_0 copy of token_embd (%s, %.0f MB pinned)\n",
                         dev_, ggml_type_name((ggml_type) pack_emb_type_), (double) bytes / 1048576.0);
        }
    }

    // ---- the expert pool: per-layer partitions (equal quota per layer within its size class)
    size_t blob_max = 0;
    int n_moe = 0;
    for (int il = l0_; il < lt_; ++il)
        if (F->L[(size_t) il].moe) {
            blob_max = std::max(blob_max, F->L[(size_t) il].blob);
            ++n_moe;
        }
    F->lp.assign((size_t) NL, FastState::LayerPool{});
    F->slot_of.assign((size_t) NL * g.n_expert, -1);
    if (n_moe > 0) {
        // the scratch slots: a fetched expert that found no spare lands here (one per routed entry)
        const size_t sstride = (blob_max + 255u) & ~(size_t) 255u;
        if (cudaMalloc(&F->scratch, (size_t) k_slots(g) * sstride) != cudaSuccess) {
            err = "glm fast: the scratch slots did not allocate";
            return false;
        }
        {
            std::vector<unsigned long long> sp((size_t) k_slots(g));
            for (int i = 0; i < k_slots(g); ++i) sp[(size_t) i] = (unsigned long long) (F->scratch + (size_t) i * sstride);
            cudaMemcpy(F->md.scratch, sp.data(), sp.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        }
        // the batched prompt path: its sizes (it borrows the pool's tail, see below); qwen35moe has none yet (its
        // prompts go through the token path)
        if (!prefill_setup(err)) return false;
        // DSpark (STRATA_DS4_DSPARK): the drafter's weights and expert cache before the pool takes what is left
        if (!dspark_load(err)) return false;
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        {
            const double G = 1073741824.0, used = (double) (total_b - free_b);
            const double known = (double) (w_arena_bytes_ + state_bytes_ + sc_bytes_ + F->arena_bytes + 8 * sstride);
            std::fprintf(stderr, "glm fast: CUDA%d VRAM before the expert pool: %.2f of %.2f GB in use - dense weights %.2f, "
                                 "state/KV (%lld ctx) %.2f, activations %.2f, other (context, MTP, scratch, prompt path) "
                                 "%.2f\n", dev_, used / G, (double) total_b / G, (double) w_arena_bytes_ / G,
                         (long long) max_ctx_, (double) state_bytes_ / G,
                         (double) (sc_bytes_ + F->arena_bytes + 8 * sstride) / G, (used - known) / G);
        }
        // headroom: the context's own growth is already allocated (state arena); see default_reserve_mb
        const size_t reserve = glmfast::vram_reserve_bytes(F->unified_memory ? 1024 : glmfast::default_reserve_mb(total_b));
        size_t avail = glmfast::expert_pool_budget(false, free_b, 0, reserve, 0, 0);
#if defined(STRATA_USE_HIP)
        if (F->unified_memory) {
            // Measure after dense/KV/scratch and pinned prompt staging are allocated.
            // Prompt device buffers borrow the pool, so their budget is not subtracted again.
            long long avail_kb = 0;
            if (FILE* mf = std::fopen("/proc/meminfo", "r")) {
                char line[256];
                while (std::fgets(line, sizeof line, mf))
                    if (std::sscanf(line, "MemAvailable: %lld kB", &avail_kb) == 1) break;
                std::fclose(mf);
            }
            const double head_gb = glmfast::ram_headroom_gb(true);
            size_t expert_bytes = 0;
            for (int il = l0_; il < lt_; ++il) {
                const auto& Ly = F->L[(size_t) il];
                if (Ly.moe) expert_bytes += (size_t) g.n_expert * glmfast::expert_stride(Ly.blob, Ly.gu_type, Ly.d_type);
            }
            avail = glmfast::expert_pool_budget(true, free_b, (size_t) std::max(0LL, avail_kb) * 1024,
                                                reserve, (size_t) (head_gb * 1073741824.0), expert_bytes);
            std::fprintf(stderr, "glm fast: HIP%d unified memory: MemAvailable %.2f GB, headroom %.2f GB + "
                                 "%zu MB reserve, experts need %.2f GB, pool budget %.2f GB\n", dev_,
                         (double) avail_kb / 1048576.0, head_gb, reserve >> 20,
                         (double) expert_bytes / 1073741824.0, (double) avail / 1073741824.0);
        }
#endif
        if (getenv("STRATA_GLM_POOL_GB")) avail = std::min(avail, glmfast::env_gib("STRATA_GLM_POOL_GB"));
        // STRATA_GLM_VRAM_GB=<n>: behave like a card with n GB - the cap counts everything this process already
        // holds on the device (context, dense weights, state, arenas), so the pool gets what such a card would have
        if (getenv("STRATA_GLM_VRAM_GB")) {
            const size_t cap = glmfast::env_gib("STRATA_GLM_VRAM_GB");
            const size_t used = total_b - free_b;
            const size_t room = cap > used + reserve ? cap - used - reserve : 0;
            std::fprintf(stderr, "glm fast: CUDA%d VRAM cap %.1f GB: %.2f GB already in use, %.2f GB left for experts\n",
                         dev_, (double) cap / 1073741824.0, (double) used / 1073741824.0, (double) room / 1073741824.0);
            avail = std::min(avail, room);
        }
        // a layer's slot stride: MMQ-addressable (glmfast::expert_stride), so the prompt path multiplies the pool
        // partition in place
        const auto slot_stride = [&](int il) {
            const auto& Ly = F->L[(size_t) il];
            return glmfast::expert_stride(Ly.blob, Ly.gu_type, Ly.d_type);
        };
        // every layer gets the same number of slots: avail / sum(blob over layers)
        size_t blob_sum = 0;
        for (int il = l0_; il < lt_; ++il)
            if (F->L[(size_t) il].moe) blob_sum += F->L[(size_t) il].blob;
        int per = (int) std::min<size_t>((size_t) g.n_expert, avail / std::max<size_t>(1, blob_sum));
        // keep a whole number of 256-byte-aligned blobs
        while (per > 0) {
            size_t tot = 0;
            for (int il = l0_; il < lt_; ++il)
                if (F->L[(size_t) il].moe) tot += (size_t) per * slot_stride(il);
            if (tot <= avail) break;
            --per;
        }
        size_t stride_sum = 0;
        for (int il = l0_; il < lt_; ++il)
            if (F->L[(size_t) il].moe) stride_sum += slot_stride(il);
        // THE FLOOR (issue #23): a route's experts need not all fit.  A miss lands in a spare slot or, with none free,
        // in its own scratch slot (one a routed entry: glm_fast.cu's route), and the CPU lane computes its share from
        // RAM - so the pool may run below top-k: the spares and one resident at least.  An 8 GB RTX A2000 ran
        // GLM-5.3-Flash Maya-S-v2 with 6 slots of a route's 8, KL 0 against itself.  Such a pool misses most of a
        // route and refills its spares every token - a lighter dense pack is the better fix (docs/SMALL-VRAM.md), and
        // the refusal says how much is short.  STRATA_GLM_POOL_FLOOR=<slots> moves the floor (1..)
        static const int pool_floor = [] {
            const char* v = getenv("STRATA_GLM_POOL_FLOOR");
            return v != nullptr ? std::max(1, std::atoi(v)) : gf::kSpares + 1;
        }();
        const auto short_mb = [&](int slots) {   // MB of VRAM short of `slots` a layer
            const size_t need = (size_t) slots * stride_sum;
            return (unsigned long long) ((need > avail ? need - avail : 0) + 1048575) >> 20;
        };
        if (per < pool_floor) {
            char buf[640];
            std::snprintf(buf, sizeof buf,
                          "glm fast: not enough VRAM for the expert pool: %d slots per layer, %d at least (%llu MB short; "
                          "a whole route's %d slots: %llu MB short) - a smaller --max-context, --kv q4_0 or a lighter "
                          "dense pack (tools/requant_dense.py IQ4_XS, docs/SMALL-VRAM.md) frees it",
                          per, pool_floor, short_mb(pool_floor), g.n_exp_used, short_mb(g.n_exp_used));
            err = buf;
            return false;
        }
        if (per < g.n_exp_used)
            std::fprintf(stderr, "glm fast: CUDA%d the expert pool holds %d slots a layer, below a route's %d: the rest "
                                 "of every route runs from the scratch slots and the CPU lane (%llu MB more VRAM would "
                                 "hold a route; docs/SMALL-VRAM.md)\n", dev_, per, g.n_exp_used, short_mb(g.n_exp_used));
        // the prompt path borrows the pool's TAIL: the last k slots of every layer, laid out as one region after all
        // the main segments - the prompt path's buffers while a prompt runs, expert slots the rest of the time
        int k_extra = 0;
        const auto tail_slots = [&](size_t bytes) {
            return (int) ((bytes + stride_sum - 1) / std::max<size_t>(1, stride_sum));
        };
        if (pf_ != nullptr) {
            k_extra = tail_slots(prefill_borrow_bytes());
            // (the prestage buffer is the part that can shrink: to what the pool can lend, else none)
            const int keep = prefill_keep();   // (prefill_setup: a route, the spares and 2 - or the small keep, #23)
            const int k_max = per - keep;
            if (k_extra > k_max && k_max > 0) k_extra = tail_slots(prefill_trim_prestage((size_t) k_max * stride_sum));
            // (a prompt runs alone - no decode beside it, unlike the vision encoder's lending below - so its main
            // slots need little more than one route's experts and the spares)
            if (per - k_extra < keep) {
                std::fprintf(stderr, "glm prefill: CUDA%d the pool (%d slots/layer) cannot lend %zu MB - token by token\n",
                             dev_, per, prefill_borrow_bytes() >> 20);
                prefill_destroy();
                k_extra = 0;
            }
        }
        // an on-demand vision encoder on the first GPU (the server sets STRATA_GLM_VISION_LEND_MB) borrows the same
        // tail while it encodes, so the tail there is the larger of the two needs
        vis_lend_ok_ = false;
        if (const char* v = getenv("STRATA_GLM_VISION_LEND_MB"); v != nullptr && dev_ == 0) {
            const size_t vb = (size_t) std::max(0LL, std::atoll(v)) << 20;
            if (vb > 0 && per - tail_slots(vb) >= g.n_exp_used + gf::kSpares + 8) {
                k_extra = std::max(k_extra, tail_slots(vb));
                vis_lend_ok_ = true;
            } else if (vb > 0) {
                std::fprintf(stderr, "glm fast: CUDA%d the pool (%d slots/layer) cannot lend %zu MB to the vision "
                                     "encoder\n", dev_, per, vb >> 20);
            }
        }
        // the slots PER LAYER: uniform, or - with the pack's expert_counts.txt (tools/glm_expert_prior.py) - by routing
        // share: every layer gets a floor, then each further slot goes to the layer whose next-most-routed expert serves
        // the largest share of its lookups (early layers spread their routing wide and gain the most).
        // STRATA_GLM_UNIFORM_SLOTS=1 keeps the uniform split (A/B).
        std::vector<int> nsl((size_t) NL, 0);
        for (int il = l0_; il < lt_; ++il)
            if (F->L[(size_t) il].moe) nsl[(size_t) il] = per;
        // When the APU can hold every expert, keep exactly that many slots in each
        // layer rather than spending the capped budget on uneven partitions/spares.
        if (getenv("STRATA_GLM_UNIFORM_SLOTS") == nullptr && !(F->unified_memory && per == g.n_expert)) {
            std::vector<std::vector<double>> share((size_t) NL);
            std::ifstream cf(expert_counts_path(pack_dir_));
            std::string line;
            while (std::getline(cf, line)) {
                std::istringstream ss(line);
                int il = -1;
                ss >> il;
                if (il < l0_ || il >= lt_ || !F->L[(size_t) il].moe) continue;
                std::vector<double> c;
                double v = 0, tot_c = 0;
                while (ss >> v) {
                    c.push_back(v);
                    tot_c += v;
                }
                if (tot_c <= 0) continue;
                std::sort(c.rbegin(), c.rend());
                for (double& x : c) x /= tot_c;
                share[(size_t) il] = std::move(c);
            }
            // a layer the counts do not cover (the NextN block) takes the mean profile of the others
            {
                std::vector<double> mean;
                int nm = 0;
                for (int il = l0_; il < lt_; ++il) {
                    const auto& sh = share[(size_t) il];
                    if (sh.empty()) continue;
                    if (mean.size() < sh.size()) mean.resize(sh.size(), 0.0);
                    for (size_t j = 0; j < sh.size(); ++j) mean[j] += sh[j];
                    ++nm;
                }
                if (nm > 0) {
                    for (double& x : mean) x /= nm;
                    for (int il = l0_; il < lt_; ++il)
                        if (F->L[(size_t) il].moe && share[(size_t) il].empty()) share[(size_t) il] = mean;
                }
            }
            bool all = true;
            for (int il = l0_; il < lt_; ++il)
                if (F->L[(size_t) il].moe && share[(size_t) il].empty()) all = false;
            if (all) {
                const int cap = g.n_expert + gf::kSpares;
                // the floor: a route's experts, the spares and the lendable tail, +16 when the budget has room for
                // it (a 24 GB card running the whole model has ~30 slots a layer: the +16 floor alone overran it)
                // (a pool below the floor - issue #23 - gives every layer what it has: per)
                const int need = g.n_exp_used + gf::kSpares + k_extra;
                const int fl = std::min({cap, per, std::max(need, std::min(per, need + 16))});
                size_t used = 0;
                for (int il = l0_; il < lt_; ++il)
                    if (F->L[(size_t) il].moe) {
                        nsl[(size_t) il] = fl;
                        used += (size_t) fl * slot_stride(il);
                    }
                // the value of layer l's next slot: the routing share of its next expert (spares past the 288th serve none)
                const auto gain = [&](int il) {
                    const auto& sh = share[(size_t) il];
                    const int j = nsl[(size_t) il] - gf::kSpares;
                    return (j >= 0 && j < (int) sh.size()) ? sh[(size_t) j] : 0.0;
                };
                for (;;) {
                    int best = -1;
                    double bg = -1.0;
                    for (int il = l0_; il < lt_; ++il) {
                        if (!F->L[(size_t) il].moe || nsl[(size_t) il] >= cap) continue;
                        if (used + slot_stride(il) > avail) continue;
                        const double gv = gain(il);
                        if (gv > bg) {
                            bg = gv;
                            best = il;
                        }
                    }
                    if (best < 0) break;
                    ++nsl[(size_t) best];
                    used += slot_stride(best);
                }
            }
        }
        size_t tot = 0, xtot = 0;
        int nmin = INT32_MAX, nmax = 0;
        for (int il = l0_; il < lt_; ++il)
            if (F->L[(size_t) il].moe) {
                tot += (size_t) nsl[(size_t) il] * slot_stride(il);
                xtot += (size_t) k_extra * slot_stride(il);
                nmin = std::min(nmin, nsl[(size_t) il]);
                nmax = std::max(nmax, nsl[(size_t) il]);
            }
        // the main slots in one allocation - or, where the driver caps a single allocation below them (Windows gives
        // a 160 GiB APU carve-out in blocks of at most ~96 GiB), in several, each holding whole layers: a layer's
        // slots are addressed from its own base (P.base), never across layers
        std::vector<uint8_t*> segs;
        std::vector<int> seg_of((size_t) NL, 0);
        const size_t main_bytes = tot - xtot;
        if (uint8_t* one = nullptr; cudaMalloc(&one, main_bytes) == cudaSuccess) {
            segs.push_back(one);
        } else {
            cudaGetLastError();
            size_t layer_max = 0;
            for (int il = l0_; il < lt_; ++il)
                if (F->L[(size_t) il].moe)
                    layer_max = std::max(layer_max, (size_t) (nsl[(size_t) il] - k_extra) * slot_stride(il));
            for (size_t cap = main_bytes / 2; segs.empty() && cap >= layer_max && layer_max > 0; cap /= 2) {
                std::vector<size_t> sizes(1, 0);
                for (int il = l0_; il < lt_; ++il) {
                    if (!F->L[(size_t) il].moe) continue;
                    const size_t lb = (size_t) (nsl[(size_t) il] - k_extra) * slot_stride(il);
                    if (sizes.back() > 0 && sizes.back() + lb > cap) sizes.push_back(0);
                    sizes.back() += lb;
                    seg_of[(size_t) il] = (int) sizes.size() - 1;
                }
                for (size_t sz : sizes) {
                    uint8_t* p = nullptr;
                    if (cudaMalloc(&p, sz) != cudaSuccess) {
                        cudaGetLastError();
                        for (uint8_t* q : segs) cudaFree(q);
                        segs.clear();
                        break;
                    }
                    segs.push_back(p);
                }
            }
            if (!segs.empty())
                std::fprintf(stderr, "glm fast: CUDA%d one allocation of %.2f GB did not allocate: the expert pool is in "
                                     "%zu\n", dev_, (double) main_bytes / 1073741824.0, segs.size());
        }
        if (segs.empty() || (xtot > 0 && cudaMalloc(&F->xpool, xtot) != cudaSuccess)) {
            for (uint8_t* q : segs) cudaFree(q);
            err = "glm fast: the expert pool (" + std::to_string(tot >> 20) + " MB) did not allocate";
            return false;
        }
        F->pool = segs[0];
        F->pool_more.assign(segs.begin() + 1, segs.end());
        F->pool_bytes = tot;
        F->xpool_bytes = xtot;
        uint8_t* b = F->pool;
        int cur_seg = 0;
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe) continue;
            auto& P = F->lp[(size_t) il];
            const int nl = nsl[(size_t) il];
            if (seg_of[(size_t) il] != cur_seg) b = segs[(size_t) (cur_seg = seg_of[(size_t) il])];
            P.stride = slot_stride(il);
            P.n = nl;
            P.n_main = nl - k_extra;
            P.base = b;
            P.key.assign((size_t) nl, -1);
            P.tick.assign((size_t) nl, 0);
            P.st.assign((size_t) nl, FastState::kFree);
            b += (size_t) P.n_main * P.stride;
            F->pool_slots += nl;
        }
        uint8_t* xregion = F->xpool;
        b = xregion;
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe) continue;
            auto& P = F->lp[(size_t) il];
            P.xbase = b;
            b += (size_t) k_extra * P.stride;
        }
        if (pf_ != nullptr && !prefill_bind(xregion, xtot, err)) return false;
        if (vis_lend_ok_)
            std::fprintf(stderr, "glm fast: CUDA%d the vision encoder borrows the pool's tail (%.2f GB) while it "
                                 "encodes\n", dev_, (double) xtot / 1073741824.0);
        F->cnt.assign((size_t) NL * g.n_expert, 0);
        F->usage.assign((size_t) NL * g.n_expert, 0);
        F->pred_of.assign((size_t) NL, std::array<int, 8>{-1, -1, -1, -1, -1, -1, -1, -1});
        std::fprintf(stderr, "glm fast: CUDA%d layers [%d,%d) expert pool %.2f GB, %d-%d slots/layer (%lld total)\n",
                     dev_, l0_, l1_, (double) tot / 1073741824.0, nmin, nmax, (long long) F->pool_slots);
        fast_tail_skip_setup();
        if (const char* dv = getenv("STRATA_GLM_DEFER"); dv != nullptr && std::atoi(dv) != 0 && !F->abl_full) {
            // (a layer split: each part defers within its own layers - its last layer waits for its own part, as
            // the trunk's last layer does, and the next part's first layer has nothing to add)
            F->defer = true;
            std::fprintf(stderr, "glm fast: CUDA%d expert deferral on (STRATA_GLM_DEFER): the CPU lane's part of a layer joins "
                                 "the residual one layer late%s\n", dev_,
                         n_parts_ > 1 ? " (a split: each part's last layer waits for its own)" : "");
        }
        if (getenv("STRATA_GLM_TIMING") != nullptr) {
            std::string sl;
            for (int il = l0_; il < lt_; ++il)
                if (F->L[(size_t) il].moe) sl += std::to_string(il) + ":" + std::to_string(nsl[(size_t) il]) + " ";
            std::fprintf(stderr, "glm fast: CUDA%d slots per layer %s\n", dev_, sl.c_str());
        }

        // the model's clean page cache goes first (once): the experts are read with O_DIRECT, so it never serves the
        // engine, and on a 2-socket host it fills one node - MPOL_INTERLEAVE then falls back instead of reclaiming
        // (49 GB of the GGUF cached on node 1 left a 100 GB tier 78% on node 0: the CPU lane read one node's
        // controllers, ~60 instead of ~92 GB/s).  Pages some process still maps stay.  STRATA_GLM_DROP_CACHE=0 keeps it.
#ifdef __linux__   // (numa_nodes() is empty elsewhere; posix_fadvise is POSIX)
        {
            static bool dropped = false;
            const char* dc = getenv("STRATA_GLM_DROP_CACHE");
            if (!dropped && (dc == nullptr || std::atoi(dc) != 0) && numa_nodes().size() >= 2) {
                dropped = true;
                for (const Shard& sh : pack_shards_)
                    if (sh.fd >= 0) posix_fadvise(sh.fd, 0, 0, POSIX_FADV_DONTNEED);
            }
        }
#endif
        // ---- the RAM tier: one pinned arena per blob size class, slots in proportion to the class's
        //      share of this half's expert bytes NOT already held by the VRAM pool
        F->layer_rc.assign((size_t) NL, -1);
        F->ram_of.assign((size_t) NL * g.n_expert, -1);
        F->left.assign((size_t) NL * g.n_expert, 0);
        std::vector<size_t> cls_stride;
        std::vector<double> cls_weight;
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe) continue;
            const size_t st = (F->L[(size_t) il].blob + 4095u) & ~(size_t) 4095u;
            int c = -1;
            for (size_t k = 0; k < cls_stride.size(); ++k)
                if (cls_stride[k] == st) c = (int) k;
            if (c < 0) {
                c = (int) cls_stride.size();
                cls_stride.push_back(st);
                cls_weight.push_back(0.0);
            }
            F->layer_rc[(size_t) il] = c;
            // (the experts its main slots cannot hold: the pool's lendable tail counts as not held - a prompt
            // borrows it, and the experts there go to RAM meanwhile, prefill_lend)
            cls_weight[(size_t) c] +=
                (double) std::max(2, g.n_expert - F->lp[(size_t) il].n_main + gf::kSpares + 2 + F->ram_slack) * (double) st;
        }
        double wsum = 0;
        for (double w : cls_weight) wsum += w;
        int64_t budget = ram_budget_;
        const bool staging_only = glmfast::minimal_ram_tier(F->unified_memory, nmin >= g.n_expert,
                                                           ram_budget_ >= 0 || getenv("STRATA_GLM_RAM_GB") != nullptr);
        if (staging_only) {
            // Decode misses and pool-tail lending still need landing slots. Keep
            // only the existing per-class floor; do not duplicate the expert cache.
            budget = 0;
            std::fprintf(stderr, "glm fast: HIP%d all experts fit in the GPU pool; RAM tier is staging only\n", dev_);
        }
        // the whole machine's budget is measured ONCE (the first half's pinning would shrink what the second
        // half sees): MemAvailable minus headroom for the OS, the server and the page cache the disk reads go
        // through; STRATA_GLM_RAM_GB pins it.  Each half takes its share of the MoE layers still unserved,
        // and whatever a half cannot use (its experts already fit) passes on to the next half.
        // The later parts' prompt path pins after this measurement (their token rows and landing ring, and the rows a
        // middle part hands on), and that came out of the headroom: one part's staging fits in it (two GPUs: 2x V100,
        // as v1.0.30), three parts' did not and froze a Linux desktop (#77).  What a third part on adds is set aside
        // here (estimated from this part's own sizes), and each later part measures again and leaves at least half
        // the headroom free.
        static int64_t total = -1, remaining = -1;
        static int remaining_layers = 0;
        const auto pinned_pf = prefill_pinned();
        const int later = n_parts_ - 1 - part_;   // the parts that start after this one
        const int64_t unseen_pf = (int64_t) std::max(0, later - 1) * (int64_t) pinned_pf.staging +
                                  (int64_t) std::max(0, later - 1) * (int64_t) pinned_pf.hop;
        const double head_gb = glmfast::ram_headroom_gb(F->unified_memory);
        const int64_t head_b = (int64_t) (head_gb * 1073741824.0);
        // the disk staging pinned after the tier (one set of max(8, k) experts per in-flight route, below)
        const int64_t stage_b = (int64_t) std::max(8, g.n_exp_used) * gf::kCpuAnswers *
                                (int64_t) ((blob_max + 255u) & ~(size_t) 255u);
        const bool fixed_ram = getenv("STRATA_GLM_RAM_GB") != nullptr;
        if (budget < 0) {
            if (total < 0) {
                if (const char* rg = getenv("STRATA_GLM_RAM_GB")) {
                    total = (int64_t) (std::atof(rg) * 1073741824.0);
                } else {
                    int64_t avail_kb = 0;
#ifdef _WIN32
                    MEMORYSTATUSEX ms{};
                    ms.dwLength = sizeof ms;
                    // pinned RAM is charged to the commit (RAM + page file), and under WDDM so is every allocation
                    // on the card - the pool above already took its share: the smaller of the two is what can pin
                    if (GlobalMemoryStatusEx(&ms)) {
                        avail_kb = (int64_t) (std::min(ms.ullAvailPhys, ms.ullAvailPageFile) >> 10);
                        // the commit, not the RAM, caps the tier: say so (issue #20 - a page file of 8 GB on a
                        // 128 GB PC left 27 GB of RAM free while the SSD served experts)
                        const double G = 1073741824.0, short_gb = ((double) ms.ullAvailPhys - (double) ms.ullAvailPageFile) / G;
                        if (short_gb > 2.0)
                            std::fprintf(stderr, "glm fast: WARNING - Windows' commit limit (RAM + page file) caps the RAM "
                                                 "tier %.0f GB below the free RAM (%.1f GB free, %.1f GB of commit left): "
                                                 "those experts are read from the SSD. Enlarge the page file by at "
                                                 "least %.0f GB (System > Advanced system settings > Performance > "
                                                 "Virtual memory) and restart Maya\n", short_gb,
                                         (double) ms.ullAvailPhys / G, (double) ms.ullAvailPageFile / G, short_gb + 4.0);
                    }
#else
                    if (FILE* mf = std::fopen("/proc/meminfo", "r")) {
                        char line[256];
                        while (std::fgets(line, sizeof line, mf))
                            if (std::sscanf(line, "MemAvailable: %lld kB", (long long*) &avail_kb) == 1) break;
                        std::fclose(mf);
                    }
#endif
                    total = std::max<int64_t>(0, avail_kb * 1024 - head_b - unseen_pf - stage_b);
                    // and at most 73% of the RAM (STRATA_GLM_RAM_FRAC; Linux, a discrete card): MemAvailable less
                    // 6 GB pinned 82 of 94 GB for DeepSeek V4.1 (its experts outgrow the RAM) and the disk staging
                    // then did not allocate; on that box 67-68 GB pinned ran for days and 73.9 GB thrashed (an 8K
                    // prompt 301 s instead of 11) - the page cache the Engram tables and the shards go through
                    // needs the room
#ifdef __linux__
                    // (a value of 1.0 or more would lift the cap - the thrash case itself: a discrete card's is held
                    // to 0.95; a unified-memory APU has no cap unless one is set)
                    const double frac = strata::core::env_num("STRATA_GLM_RAM_FRAC", F->unified_memory ? 1.0 : 0.73,
                                                              0.05, F->unified_memory ? 1.0 : 0.95);
                    if (const int64_t mt = mem_total(); mt > 0 && frac < 1.0)
                        total = std::min<int64_t>(total, std::max<int64_t>(0, (int64_t) ((double) mt * frac) - stage_b));
#endif
                }
                remaining = total;
                remaining_layers = g.n_layers - g.dense_lead + (getenv("STRATA_GLM_NO_MTP") ? 0 : g.nextn);
            }
            // the last half takes everything left (one device: the NextN block, counted above, is not loaded)
            budget = l1_ >= g.n_layers ? remaining
                                       : std::min<int64_t>(remaining, (int64_t) ((double) remaining * (double) n_moe /
                                                                                 (double) std::max(1, remaining_layers)));
            // a later part: no more than is free now, less what is still to come and half the headroom
            if (part_ > 0 && !fixed_ram) {
                const int64_t now = avail_ram_now();
                if (now > 0) budget = std::min<int64_t>(budget, std::max<int64_t>(0, now - head_b / 2 - unseen_pf));
            }
        }
        // DSpark's VRAM came out of the expert pool, and the tier would pin those experts instead: on a 94 GB machine
        // 73.9 GB pinned instead of 67.4 left 13 GB free, an 8K prompt took 301 s instead of 11 and the decode ran at
        // 1.6 tok/s (the same without the drafter and a pool as small).  The tier keeps its size without the drafter;
        // the coldest experts come from the disk (0.01 reads a token in the decode)
        if (dsp_ != nullptr && !fixed_ram && budget > 0) {
            const int64_t keep = std::max<int64_t>(0, (int64_t) wsum - (int64_t) dspark_vram_bytes());
            if (budget > keep) {
                std::fprintf(stderr, "glm fast: CUDA%d RAM tier without the %.2f GB the DSpark drafter took from the pool "
                                     "(those experts read from the disk)\n", dev_,
                             (double) dspark_vram_bytes() / 1073741824.0);
                budget = keep;
            }
        }
        F->rc.assign(cls_stride.size(), FastState::RamClass{});
        for (size_t c = 0; c < cls_stride.size() && wsum > 0; ++c) {
            auto& R = F->rc[c];
            R.stride = cls_stride[c];
            int64_t n = (int64_t) ((double) budget * cls_weight[c] / wsum / (double) R.stride);
            n = std::max<int64_t>(n, 16);   // a floor so a miss always has somewhere to land
            // never more than the experts of the class that the VRAM pool does not hold, plus the free slots
            // fast_boundary keeps per class for disk reads and reads ahead.  When the pool holds every expert (a big
            // card: 288 of 288 slots a layer, or a split across many GPUs) that is a few spares' worth - below the floor
            // for a class of one layer (the NextN block), which then never allocated and stopped the start ("did not
            // allocate")
            const int64_t cap = std::max<int64_t>(1, (int64_t) (cls_weight[c] / (double) R.stride)) + 16;
            n = std::min<int64_t>(n, cap);
            const int64_t least = std::min<int64_t>(16, cap);   // the floor, or the whole cap when that is smaller
            void* p = nullptr;
#ifdef __linux__
            if (ram_thp()) {   // the shards' page cache (the experts are read O_DIRECT; Engram has its row cache) ->
                for (auto& sh : pack_shards_)   // free 2 MB blocks for the tier's huge pages
                    if (sh.fd >= 0) posix_fadvise(sh.fd, 0, 0, POSIX_FADV_DONTNEED);
            }
#endif
            while (n >= least) {
                if ((p = numa_pinned((size_t) n * R.stride)) != nullptr) {   // (over every socket: numa_pinned)
                    R.registered = true;
                    break;
                }
                if (cudaHostAlloc(&p, (size_t) n * R.stride, cudaHostAllocPortable) == cudaSuccess) break;
                cudaGetLastError();
                p = nullptr;
                n = (n > least && n * 7 / 8 < least) ? least : n * 7 / 8;
            }
            if (p == nullptr) {
                err = "glm fast: the pinned RAM tier did not allocate";
                return false;
            }
            // (an APU whose pool holds every expert keeps a staging-only tier: no expert is outside VRAM to hold)
            if (F->ram_resident && !staging_only && n < cap) {
                err = "glm fast: --glm-ram-resident: the RAM tier cannot hold every non-VRAM expert (class " +
                      std::to_string(c) + ": " + std::to_string(n) + " of " + std::to_string(cap) +
                      " slots pinned; raise STRATA_GLM_RAM_GB, its headroom, or lower --max-context)";
                return false;
            }
            R.base = (uint8_t*) p;
            R.n = (int) n;
            R.key.assign((size_t) n, -1);
            R.st.assign((size_t) n, FastState::kRFree);
            R.tick.assign((size_t) n, 0);
            F->ram_bytes += (size_t) n * R.stride;
        }
        if (!staging_only && ram_budget_ < 0 && remaining >= 0) {
            remaining = std::max<int64_t>(0, remaining - (int64_t) F->ram_bytes);
            remaining_layers -= n_moe;
        }
        // pinned memory cannot be swapped out: say so when what is left free is short of the headroom
        if (!staging_only && !fixed_ram && ram_budget_ < 0) {
            const int64_t left = avail_ram_now();
            if (left > 0 && left - unseen_pf < head_b / 2)
                std::fprintf(stderr, "glm fast: WARNING - CUDA%d: %.1f GB of RAM free after its RAM tier, and the prompt "
                                     "path still pins up to %.1f GB: if the system swaps or stalls, set "
                                     "STRATA_GLM_RAM_GB lower\n", dev_, (double) left / 1073741824.0,
                             (double) unseen_pf / 1073741824.0);
        }
        int64_t ram_slots = 0;
        for (auto& R : F->rc) ram_slots += R.n;
        std::fprintf(stderr, "glm fast: CUDA%d RAM tier %.2f GB pinned, %lld slots\n", dev_,
                     (double) F->ram_bytes / 1073741824.0, (long long) ram_slots);
        if (F->ram_resident)
            std::fprintf(stderr, "glm fast: CUDA%d RAM-resident mode: every non-VRAM expert gets a slot (+%d slack/layer); disk evictions are OFF\n",
                         dev_, F->ram_slack);
        // the prompt path's lend cap (Strata's, from the share of the expert bytes held pinned), now that it is known
        if (pf_ != nullptr) {
            double held = (double) F->ram_bytes, all = 0;
            for (int il = l0_; il < lt_; ++il)
                if (F->L[(size_t) il].moe) {
                    const auto& P = F->lp[(size_t) il];
                    all += (double) g.n_expert * (double) P.stride;
                    held += (double) std::min(g.n_expert, P.n) * (double) P.stride;
                }
            prefill_settle(all > 0 ? std::min(1.0, held / all) : 1.0);
        }
        for (int ub = 0; ub < 2; ++ub)
            if (cudaHostAlloc((void**) &F->upd_key_h[ub], FastState::kMaxUpd * sizeof(int), cudaHostAllocMapped) != cudaSuccess ||
                cudaHostAlloc((void**) &F->upd_val_h[ub], FastState::kMaxUpd * sizeof(unsigned long long),
                              cudaHostAllocMapped) != cudaSuccess ||
                cudaHostGetDevicePointer((void**) &F->upd_key_hd[ub], F->upd_key_h[ub], 0) != cudaSuccess ||
                cudaHostGetDevicePointer((void**) &F->upd_val_hd[ub], F->upd_val_h[ub], 0) != cudaSuccess ||
                cudaMalloc((void**) &F->upd_key_d[ub], FastState::kMaxUpd * sizeof(int)) != cudaSuccess ||
                cudaMalloc((void**) &F->upd_val_d[ub], FastState::kMaxUpd * sizeof(unsigned long long)) != cudaSuccess ||
                cudaEventCreateWithFlags(&F->upd_ev[ub], cudaEventDisableTiming) != cudaSuccess) {
                err = "glm fast: the table update buffers did not allocate";
                return false;
            }
        F->upd_at.assign((size_t) (2 * F->n_keys + NL * gf::kSpares), 0);
        F->upd_gen.assign(F->upd_at.size(), 0u);
        // disk reads: three slices per expert in parallel; pinned staging for when the RAM tier has no free slot
        F->workers.reset(new Workers(8));
        // (fast_warm streams 8 at a time; the service's disk misses take a set of 8 per in-flight route, by its seq
        // modulo kCpuAnswers - a verify window has several routes of a layer answered before the device reads them)
        F->stage.assign((size_t) std::max(8, g.n_exp_used) * gf::kCpuAnswers, nullptr);
        for (auto& sbuf : F->stage)
            if (cudaHostAlloc((void**) &sbuf, sstride, cudaHostAllocPortable) != cudaSuccess) {
                err = "glm fast: pinned disk staging did not allocate";
                return false;
            }
        // LOOKAHEAD: the predictions per layer, a load state per RAM slot, and the reader threads
        if (const char* ah = getenv("STRATA_GLM_AHEAD")) F->n_ahead = std::max(0, std::min(gf::kAhead, std::atoi(ah)));
        if (g.n_expert > 512) F->n_ahead = 0;
        const char* ahr = getenv("STRATA_GLM_AHEAD_READ");
        F->ahead_read = F->n_ahead > 0 && (ahr == nullptr || std::atoi(ahr) != 0);
        const char* dl = getenv("STRATA_GLM_DISK_LANE");
        F->disk_lane = dl != nullptr && std::atoi(dl) != 0;
        const char* da = getenv("STRATA_GLM_DISK_ADAPT");
        F->disk_adapt = F->disk_lane && da != nullptr && std::atoi(da) != 0;
        const char* sf = getenv("STRATA_GLM_STREAM_FETCH");
        F->stream_fetch = sf == nullptr || std::atoi(sf) != 0;
        // at most 10: a streamed expert's ready word has a bit per part and chunk (3 x 10 = 30 of 32; 16 set bits
        // past 31 - the device copied chunks that had not landed, and the routes read garbage)
        const char* rc = getenv("STRATA_GLM_READ_CHUNKS");
        F->read_chunks = std::max(1, std::min(10, rc ? std::atoi(rc) : 8));
        std::array<std::array<short, gf::kMaxK>, gf::kAhead> none;
        for (auto& r : none) r.fill(-1);
        F->ah_pred.assign((size_t) NL, none);
        for (auto& R : F->rc) {
            F->rload.emplace_back(new std::atomic<uint64_t>[(size_t) R.n]);
            for (int s = 0; s < R.n; ++s) F->rload.back()[(size_t) s].store(0);
        }
        {   // the RAM fill (fast_boundary step 4b): on with the disk tail skip, which starves the experts it skips
            const char* rf = getenv("STRATA_GLM_RAM_FILL");
            F->ram_fill = rf != nullptr ? std::max(0, std::atoi(rf)) : F->tail_skip_disk ? 2 : 0;
            const char* rm = getenv("STRATA_GLM_RAM_FILL_MARGIN");
            F->ram_fill_margin = (uint32_t) (rm != nullptr ? std::max(0, std::atoi(rm)) : 4);
            if (F->ram_resident) F->ram_fill = 0;
            if (F->ram_fill > 0)
                std::fprintf(stderr, "glm fast: CUDA%d RAM fill: up to %d of the most routed disk-only experts a token "
                                     "read into the RAM tier (STRATA_GLM_RAM_FILL)\n", dev_, F->ram_fill);
        }
        if (F->ahead_read || F->ram_fill > 0)
            for (int i = 0; i < 2; ++i) F->ah_th.emplace_back([this] { fast_ahead_reader(); });
        const char* wv = getenv("STRATA_GLM_WARM");
        if ((wv == nullptr || std::atoi(wv) != 0) && !fast_warm(err)) return false;
        // RAM-resident mode is all-or-nothing: after the warm, every expert must be in VRAM or the RAM tier -
        // one that fits neither would be read from disk at runtime (STRATA_GLM_WARM=0 fills on demand: no check)
        if (F->ram_resident && (wv == nullptr || std::atoi(wv) != 0)) {
            int uncovered = 0, total = 0;
            for (int il = l0_; il < lt_; ++il) {
                if (!F->L[(size_t) il].moe) continue;
                total += g.n_expert;
                for (int e = 0; e < g.n_expert; ++e) {
                    const size_t key = (size_t) il * g.n_expert + e;
                    uncovered += F->slot_of[key] < 0 && F->ram_of[key] < 0;
                }
            }
            if (uncovered > 0) {
                err = "glm fast: --glm-ram-resident: " + std::to_string(uncovered) +
                      " experts fit neither tier after the warm - raise STRATA_GLM_RAM_GB or lower --max-context";
                return false;
            }
            std::fprintf(stderr, "glm fast: CUDA%d RAM-resident verified: all %d experts are in VRAM or pinned RAM\n",
                         dev_, total);
        }
        if (!fast_cpu_lane_setup(err)) return false;
        F->svc = std::thread([this] { fast_service(); });
        glmfast::pin_thread(F->svc.native_handle(), F->cpu_pin);   // (the CPU lane's last worker: on its pool's node)
    }
    // Lazily allocate the opt-in tier after Prefill, so it cannot change the primary pool or prompt buffers.
    if (const char* helper = getenv("STRATA_GLM_EXPERT_HELPER"); helper && std::strcmp(helper, "0") != 0) {
        char* end = nullptr;
        const long device = std::strtol(helper, &end, 10);
        const char* cap = getenv("STRATA_GLM_EXPERT_HELPER_GIB");
        char* cap_end = nullptr;
        const double gib = cap ? std::strtod(cap, &cap_end) : 13.0;
        const char* reason = nullptr;
#if defined(STRATA_USE_HIP)
        reason = "this prototype supports CUDA only";
#endif
        if (end == helper || *end || device < 0 || device > INT_MAX ||
            (cap && (cap_end == cap || *cap_end)) || !std::isfinite(gib) || gib < 0.25 || gib > 128)
            reason = "invalid device or VRAM budget";
        else if (!g.v41 || n_parts_ != 1 || l0_ != 0 || l1_ != g.n_layers || exl3_)
            reason = "requires an unsplit DeepSeek V4.1 native pack";
        else if (!F->cpu_pool || !F->cpu_plan || F->igpu_experts)
            reason = "requires the CPU lane without an iGPU lane";
        if (reason) std::fprintf(stderr, "expert helper disabled: %s; using the existing lane\n", reason);
        else F->expert_helper = std::make_unique<GlmExpertHelper>((int) device, (size_t) (gib * 1073741824.0));
    }
    return true;
}

void Glm5Model::expert_helper_decode_begin(bool speculative) {
    auto* F = fast_;
    if (!F || !F->expert_helper || F->expert_helper->failed) return;
    auto& H = *F->expert_helper;
    H.active = false;
    if (speculative || (getenv("STRATA_GLM_KDA_GRAPH") && std::atoi(getenv("STRATA_GLM_KDA_GRAPH")) != 0)) {
        std::fprintf(stderr, "expert helper inactive: speculative decoding or layer graphs; using the CPU lane\n");
        return;
    }
    std::string err;
    if (!H.opened && !H.open(dev_, g_.n_embd, g_.n_ff_exp, g_.n_layers, g_.n_expert, err)) {
        H.disable();
        std::fprintf(stderr, "%s; using the CPU lane\n", err.c_str());
        F->expert_helper.reset(); // No primary route has read its ownership table yet.
        return;
    }
    // Keep the RAM sources stable until every helper upload completes.
    if (!fast_boundary(err)) {
        H.disable();
        std::fprintf(stderr, "expert helper disabled: %s; using the CPU lane\n", err.c_str());
        return;
    }
    std::lock_guard<std::mutex> lock(F->mu);
    std::vector<std::vector<int>> ranked((size_t) g_.n_layers);
    std::vector<size_t> blobs((size_t) g_.n_layers, 0);
    for (int il = 0; il < g_.n_layers; ++il) {
        const auto& layer = F->L[(size_t) il];
        if (!layer.moe || layer.exl3 || F->cpu_fmt[(size_t) il].n_ff <= 0 ||
            !gf::moe_supported(layer.gu_type) || !gf::moe_supported(layer.d_type)) continue;
        blobs[(size_t) il] = layer.blob;
        const int cls = F->layer_rc[(size_t) il];
        if (cls < 0) continue;
        const auto& ram = F->rc[(size_t) cls];
        for (int e = 0; e < g_.n_expert; ++e) {
            const int key = il * g_.n_expert + e, slot = F->ram_of[(size_t) key];
            if (F->slot_of[(size_t) key] < 0 && slot >= 0 && ram.st[(size_t) slot] == FastState::kRHold)
                ranked[(size_t) il].push_back(key);
        }
        auto& rank = ranked[(size_t) il];
        std::stable_sort(rank.begin(), rank.end(), [&](int a, int b) {
            return F->usage[(size_t) a] != F->usage[(size_t) b] ?
                F->usage[(size_t) a] > F->usage[(size_t) b] : a < b;
        });
    }
    const auto source = [&](int key) -> const uint8_t* {
        const int il = key / g_.n_expert, slot = F->ram_of[(size_t) key], cls = F->layer_rc[(size_t) il];
        if (slot < 0 || cls < 0) return nullptr;
        const auto& ram = F->rc[(size_t) cls];
        return ram.st[(size_t) slot] == FastState::kRHold ? ram.base + (size_t) slot * ram.stride : nullptr;
    };
    if (!H.prepare(ranked, blobs, source, err)) {
        H.disable(); // Retires any queued upload before the RAM tier may change again.
        std::fprintf(stderr, "%s; using the CPU lane\n", err.c_str());
        F->expert_helper.reset();
        return;
    }
    H.report("prepared");
}

void Glm5Model::expert_helper_decode_end() {
    auto* F = fast_;
    if (!F || !F->expert_helper) return;
    auto& H = *F->expert_helper;
    if (!H.active && !H.failed) return;
    // Drain published routes before changing ownership or freeing a failed helper's buffers.
    std::string err;
    const bool drained = fast_boundary(err);
    H.active = false;
    if (H.failed) H.disable();
    if (drained) {
        H.report("request_end");
        if (H.failed) F->expert_helper.reset();
    }
}

// ---------------------------------------------------------------- the CPU lane
// ne RAM-tier experts of layer il (blob[i]: [gate | up | down] as in the tiers) on the CPU pool: out = sum_i w[i] *
// expert_i(x), in i order.  Gate/up rows then down rows, each split in row chunks across the pool and the caller.
// ---- the iGPU LANE (STRATA_GLM_IGPU, default auto): libabyss_igpu.so (beside the executable, or STRATA_GLM_IGPU_LIB) opened on
//      the first HIP device - an APU's integrated GPU - and the RAM tier registered with it, so the lane's calls read
//      the pinned slots in place (Ryzen AI Max+ 395: ~240 GB/s against the CPU threads' ~75).  Any failure: the CPU lane.
void Glm5Model::fast_igpu_setup() {
#ifdef __linux__
    FastState* F = fast_;
    // unset: on when the library and a HIP device are there (and the calibration finds it faster than the CPU); 0 off;
    // 1 on even when slower.  Asked for (set) and unavailable: the reason is printed; auto and unavailable: silent
    const char* on = getenv("STRATA_GLM_IGPU");
    // (a layer split's parts each open it: the library is one lane for the process - opened once, every part's RAM
    // tier registered with it, one call at a time)
    if ((on != nullptr && std::atoi(on) == 0) || F->igpu_lib != nullptr) return;
    const bool asked = on != nullptr;
    std::string path;
    if (const char* lp = getenv("STRATA_GLM_IGPU_LIB")) {
        path = lp;
    } else {
        char b[4096];
        const ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1);
        if (n > 0) {
            b[n] = 0;
            path = b;
            path = path.substr(0, path.rfind('/') + 1);
        }
        path += "libabyss_igpu.so";
    }
    if (!asked && access(path.c_str(), R_OK) != 0) return;   // not built here
    // HIP reads CUDA_VISIBLE_DEVICES too when HIP_VISIBLE_DEVICES is unset: the server's "1" (the second NVIDIA card)
    // or "1,0" (a split) hid the iGPU, HIP's device 0.  STRATA_GLM_IGPU_DEVICE: another HIP device (default 0).
    // A HIP engine's runtime is the lane's, up already: the server lists the iGPU after the engine's GPU(s) and the
    // lane takes the integrated one
#if !defined(STRATA_USE_HIP)
    if (getenv("HIP_VISIBLE_DEVICES") == nullptr && getenv("ROCR_VISIBLE_DEVICES") == nullptr) {
        const char* d = getenv("STRATA_GLM_IGPU_DEVICE");
        setenv("HIP_VISIBLE_DEVICES", d != nullptr && d[0] ? d : "0", 0);
    }
#else
    {   // the engine on the integrated GPU itself (an APU alone): no second GPU for a lane
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, dev_) == cudaSuccess && p.integrated) return;
    }
    {   // no integrated GPU among the visible devices (one discrete Radeon, its APU's iGPU off or none): the library
        // fell back to device 0, the engine's own card - the lane then ran there, its calls 27-62 ms late (issue #11)
        int n = 0;
        bool igpu = false;
        cudaDeviceProp p{};
        cudaGetDeviceCount(&n);
        for (int i = 0; i < n && !igpu; ++i) igpu = i != dev_ && cudaGetDeviceProperties(&p, i) == cudaSuccess && p.integrated;
        if (!igpu) {
            if (asked)
                std::fprintf(stderr, "glm fast: CUDA%d iGPU lane off (no integrated GPU among the visible HIP devices) "
                                     "- the CPU lane computes\n", dev_);
            return;
        }
    }
#endif
    void* lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    const auto fail = [&](const std::string& why) {
        std::fprintf(stderr, "glm fast: CUDA%d iGPU lane off (%s) - the CPU lane computes\n", dev_, why.c_str());
        if (lib != nullptr) dlclose(lib);
        // the lane's failed call left its error as the thread's last one: the next launch_check counted it, and the
        // count is never reset - every request then ended "kernel launch(es) failed" (ROCm 7.2.1, issue #11)
        (void) cudaGetLastError();
    };
    if (lib == nullptr) return fail(std::string("dlopen ") + path + ": " + dlerror());
    auto ver = (abyss_igpu_version_fn) dlsym(lib, "abyss_igpu_version");
    auto op = (abyss_igpu_open_fn) dlsym(lib, "abyss_igpu_open");
    auto rg = (abyss_igpu_register_fn) dlsym(lib, "abyss_igpu_register");
    auto ex = (abyss_igpu_experts_fn) dlsym(lib, "abyss_igpu_experts");
    // (optional) the same with x's q8_1 from the request: the lane skips its quantize; STRATA_GLM_IGPU_XQ=0: never
    static const bool xq_on = [] { const char* v = getenv("STRATA_GLM_IGPU_XQ"); return v == nullptr || std::atoi(v) != 0; }();
    auto exq = xq_on ? (abyss_igpu_experts_q_fn) dlsym(lib, "abyss_igpu_experts_q") : nullptr;
    // (optional) the two halves: a route's lane experts start before its disk reads; STRATA_GLM_IGPU_EARLY=0: never
    static const bool early_on = [] { const char* v = getenv("STRATA_GLM_IGPU_EARLY"); return v == nullptr || std::atoi(v) != 0; }();
    auto exs = early_on ? (abyss_igpu_experts_start_fn) dlsym(lib, "abyss_igpu_experts_start") : nullptr;
    auto exf = early_on ? (abyss_igpu_experts_finish_fn) dlsym(lib, "abyss_igpu_experts_finish") : nullptr;
    // (optional) a verify window's columns in one call (STRATA_GLM_IGPU_COLS=0: a call a column, as before).  The
    // window's columns are queued for it like the CPU lane's multi call (the service thread, lane_flush)
    static const bool cols_on = [] { const char* v = getenv("STRATA_GLM_IGPU_COLS"); return v == nullptr || std::atoi(v) != 0; }();
    auto exc = cols_on ? (abyss_igpu_experts_cols_fn) dlsym(lib, "abyss_igpu_experts_cols") : nullptr;
    if (ver == nullptr || op == nullptr || rg == nullptr || ex == nullptr || ver() != ABYSS_IGPU_API)
        return fail("interface mismatch: rebuild libabyss_igpu.so");
    char e[256] = {0};
    if (op(g_.n_embd, g_.n_ff_exp, e, (int) sizeof e) != 0) return fail(e);
    const auto t0 = std::chrono::steady_clock::now();
    // Registered in pieces of whole experts, ~STRATA_GLM_IGPU_CHUNK_MB (256) each: the kernel driver keeps a host
    // range in step with the CPU's page tables (it does not pin it), and any change to one of its pages - the kernel
    // compacting memory around it, for one - stops the iGPU's queues until the WHOLE range is faulted in again: with
    // one 69 GB range a decode token waited 10+ s about once a request on a fragmented host (the kernel log:
    // "svm_range_restore_work / amdgpu_amdkfd_restore_userptr_worker hogged CPU"), and registering it took 0.4-13 s
    // against ~0.4-1.3 s in 283 pieces.  0: one range a RAM class, as before.
    static const size_t chunk_mb = [] {
        const char* v = getenv("STRATA_GLM_IGPU_CHUNK_MB");
        return v != nullptr ? (size_t) std::max(0, std::atoi(v)) : (size_t) 256;
    }();
    size_t bytes = 0, pieces = 0;
    for (const auto& R : F->rc) {
        if (R.base == nullptr || R.n <= 0) continue;
        const size_t per = chunk_mb == 0 ? (size_t) R.n : std::max<size_t>(1, (chunk_mb << 20) / R.stride);
        for (size_t s0 = 0; s0 < (size_t) R.n; s0 += per) {
            const size_t ns = std::min(per, (size_t) R.n - s0);
            if (rg(R.base + s0 * R.stride, ns * R.stride, e, (int) sizeof e) != 0) {
                if (auto cl = (abyss_igpu_close_fn) dlsym(lib, "abyss_igpu_close")) cl();
                return fail(e);
            }
            ++pieces;
        }
        bytes += (size_t) R.n * R.stride;
    }
    // the request ring (x) too: the lane then reads it in place (the answer slots: in fast_cpu_lane_setup)
    if (rg(F->ring_h, sizeof(gf::MoeRequest) * gf::kRingSize, e, (int) sizeof e) != 0)
        std::fprintf(stderr, "glm fast: CUDA%d iGPU lane: the ring did not register (%s) - x copied instead\n", dev_, e);
    F->igpu_lib = lib;
    F->igpu_experts = ex;
    F->igpu_experts_q = exq;
    F->igpu_cols = exc;
    F->igpu_start = exs != nullptr && exf != nullptr ? exs : nullptr;
    F->igpu_finish = exs != nullptr && exf != nullptr ? exf : nullptr;
    F->igpu_poke = (abyss_igpu_poke_fn) dlsym(lib, "abyss_igpu_poke");
    // STRATA_GLM_IGPU_POKE=0 (investigation): no keep-alive kernel between routes
    if (const char* pk = getenv("STRATA_GLM_IGPU_POKE"))
        if (std::atoi(pk) == 0) F->igpu_poke = nullptr;
    F->igpu_rows = (abyss_igpu_rows_fn) dlsym(lib, "abyss_igpu_rows");
    F->igpu_x3 = (abyss_igpu_experts_x3_fn) dlsym(lib, "abyss_igpu_experts_x3");
    F->igpu_x3s = (abyss_igpu_x3_start_fn) dlsym(lib, "abyss_igpu_x3_start");
    F->igpu_x3f = (abyss_igpu_x3_finish_fn) dlsym(lib, "abyss_igpu_x3_finish");
    if (F->igpu_x3f == nullptr) F->igpu_x3s = nullptr;
    F->igpu_reg = rg;
    F->igpu_unreg = (abyss_igpu_unregister_fn) dlsym(lib, "abyss_igpu_unregister");
    if (F->igpu_unreg == nullptr) F->igpu_rows = nullptr;
    std::fprintf(stderr, "glm fast: CUDA%d iGPU lane on (%s): the RAM tier's %.2f GB registered (%zu pieces) in %.0f ms\n",
                 dev_, path.c_str(), (double) bytes / 1073741824.0, pieces,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
#endif
}

// EXL3 on the CPU lane, one (token, expert) entry's rows: the gate and up inputs xh = H(x * suh) and their 128-value
// block sums, the down input and its block sums, the expert's output, and the byte-sum products of the split-K pieces:
// gate/up in pieces of kX3GuIn inputs, down in pieces of kX3DnIn = 128 (one gate/up band), each over every column (the
// tile rows in memory order: exl3_cpu_rows_acc).  The pieces are fixed, so an entry's answer does not depend on how a
// pool cuts the work (Maya e62ff8a / #188 by @needmorevram)
namespace {
constexpr int kX3GuIn = 256, kX3DnIn = 128;
struct X3Rows {
    float* p;
    int E, FF;
    static size_t floats(int E, int FF) {
        return 3 * (size_t) E + FF + 2 * (size_t) (E / kX3GuIn) * FF + (size_t) (FF / kX3DnIn) * E +
               2 * (size_t) (E / 128) + (size_t) FF / 128;
    }
    float* x(int r) const { return p + (size_t) r * E; }
    float* xd() const { return p + 2 * (size_t) E; }
    float* dn() const { return p + 2 * (size_t) E + FF; }
    float* gu(int r, int q) const { return p + 3 * (size_t) E + FF + ((size_t) r * (E / kX3GuIn) + q) * FF; }
    float* dp(int q) const { return gu(2, 0) + (size_t) q * E; }
    float* xsb(int r) const { return dp(FF / kX3DnIn) + (size_t) r * (E / 128); }
    float* xds() const { return xsb(2); }
};

// matrix r (0 gate, 1 up, 2 down) of an EXL3 expert blob
strata::kernels::cpu::Exl3Part x3_part(const uint8_t* blob, const strata::kernels::exl3::ExpertLayout& xl, int r,
                                       int E, int FF) {
    strata::kernels::cpu::Exl3Part p;
    p.trellis = (const uint32_t*) (blob + xl.trellis[r]);
    p.suh = (const uint16_t*) (blob + xl.suh[r]);
    p.svh = (const uint16_t*) (blob + xl.svh[r]);
    p.k = r < 2 ? E : FF;
    p.n = r < 2 ? FF : E;
    p.K = xl.K[r];
    p.cb = xl.cb;
    return p;
}

// piece q of matrix r (0 gate, 1 up) for the entries R[0..n) of one expert: their inputs' blocks of that piece (x[i]:
// entry i's token input) - once a piece, before any of its columns' runs
void x3_gate_up_prep(const X3Rows* R, const float* const* x, int n, const strata::kernels::cpu::Exl3Part& p, int r,
                     int q) {
    namespace kc = strata::kernels::cpu;
    for (int i = 0; i < n; ++i)
        for (int b = q * (kX3GuIn / 128); b < (q + 1) * (kX3GuIn / 128); ++b)
            R[i].xsb(r)[b] = kc::exl3_cpu_prepare_block(p.suh + b * 128, x[i] + b * 128, R[i].x(r) + b * 128);
}

// ... then the piece's byte-sum products over the columns [c0, c1), the tile rows decoded once for all n
void x3_gate_up_rows(const X3Rows* R, int n, const strata::kernels::cpu::Exl3Part& p, int r, int q, int c0, int c1) {
    constexpr int kN = 16;   // (entries of one expert a call: 1 on the decode lane)
    const float* xh[kN];
    float* acc[kN];
    for (int i = 0; i < n; ++i) {
        xh[i] = R[i].x(r);
        acc[i] = R[i].gu(r, q);
    }
    strata::kernels::cpu::exl3_cpu_rows_acc_cols_n(p, xh, n, q * (kX3GuIn / 16), (q + 1) * (kX3GuIn / 16), c0, c1, acc);
}

// band b of the gate and up outputs from their pieces, the clamped swiglu, and that block of the down input
void x3_gate_up_block(const X3Rows& R, const strata::kernels::cpu::Exl3Part* pt, int b, float lim) {
    namespace kc = strata::kernels::cpu;
    float y[2][128], a[128];
    for (int r = 0; r < 2; ++r) {
        for (int j = 0; j < 128; ++j) a[j] = 0.0f;
        for (int q = 0; q < R.E / kX3GuIn; ++q) {
            const float* aq = R.gu(r, q) + b * 128;
            for (int j = 0; j < 128; ++j) a[j] += aq[j];
        }
        double xs = 0.0;   // (exl3_cpu_prepare's sum)
        for (int k = 0; k < R.E / 128; ++k) xs += R.xsb(r)[k];
        kc::exl3_cpu_band_out(pt[r], a, (float) xs, b * 128, y[r]);
    }
    float hb[128];
    for (int j = 0; j < 128; ++j) {
        float gv = y[0][j], uv = y[1][j];
        if (lim > 0.0f) {
            gv = std::min(gv, lim);
            uv = std::min(std::max(uv, -lim), lim);
        }
        hb[j] = gv / (1.0f + std::exp(-gv)) * uv;
    }
    R.xds()[b] = kc::exl3_cpu_prepare_block(pt[2].suh + b * 128, hb, R.xd() + (size_t) b * 128);
}

// down piece q's input (gate/up band q) for the entries R[0..n) of one expert - once a piece, before its columns' runs
void x3_down_prep(const X3Rows* R, int n, const strata::kernels::cpu::Exl3Part* pt, int q, float lim) {
    static_assert(kX3DnIn == 128, "a down piece's input is one gate/up band");
    for (int i = 0; i < n; ++i) x3_gate_up_block(R[i], pt, q, lim);
}

// ... then the piece's byte-sum products over the columns [c0, c1), decoded once for all n
void x3_down_rows(const X3Rows* R, int n, const strata::kernels::cpu::Exl3Part& pd, int q, int c0, int c1) {
    constexpr int kN = 16;   // (entries of one expert a call: 1 on the decode lane)
    const float* xh[kN];
    float* acc[kN];
    for (int i = 0; i < n; ++i) {
        xh[i] = R[i].xd();
        acc[i] = R[i].dp(q);
    }
    strata::kernels::cpu::exl3_cpu_rows_acc_cols_n(pd, xh, n, q * (kX3DnIn / 16), (q + 1) * (kX3DnIn / 16), c0, c1, acc);
}

// band b of the down output from its pieces
void x3_down_block(const X3Rows& R, const strata::kernels::cpu::Exl3Part& pd, int b) {
    float a[128];
    for (int j = 0; j < 128; ++j) a[j] = 0.0f;
    for (int q = 0; q < R.FF / kX3DnIn; ++q) {
        const float* aq = R.dp(q) + b * 128;
        for (int j = 0; j < 128; ++j) a[j] += aq[j];
    }
    float s = 0.0f;
    for (int j = 0; j < R.FF / 128; ++j) s += R.xds()[j];
    strata::kernels::cpu::exl3_cpu_band_out(pd, a, s, b * 128, R.dn() + (size_t) b * 128);
}

// n experts through their gate/up pieces, down pieces and output bands as ONE batch of jobs in that order: expert i's
// down pieces wait for its own gate/up columns, its bands for its down columns, never for another expert's - so the
// phases overlap, where a batch each waited for its slowest job (40 threads, ~6 RAM-tier experts: the down pieces' ~100
// jobs took three rounds for two and a third rounds' work).  A piece is cut by columns too, into as many runs as one
// round of the pool's threads holds (a lone expert's 16 down pieces left 20 of 36 threads idle for half its time): the
// first run of a piece prepares its input (gu_prep / dn_prep), the others wait for that; a down piece needs only the
// gate/up run holding its band, a band only the down runs holding its columns.  Each column's sums are the uncut
// piece's, so the answers do not depend on the cut.  A job waits only for jobs claimed before it, which run, so the
// pool cannot deadlock
template <class GP, class GR, class DP, class DR, class BandOut>
void x3_jobs(glmfast::Workers& pool, int n, int E, int FF, const GP& gu_prep, const GR& gu_rows, const DP& dn_prep,
             const DR& dn_rows, const BandOut& out) {
    constexpr int kRuns = 16;
    const int GQ = E / kX3GuIn, DQ = FF / kX3DnIn, DB = E / 128, GB = FF / 128;
    const int T = std::max(1, pool.active());
    const int CG = std::max(1, std::min({kRuns, GB, T / std::max(1, 2 * GQ * n)}));
    const int CD = std::max(1, std::min({kRuns, DB, T / std::max(1, DQ * n)}));
    // run c of C over B bands holds the bands [B c / C, B (c + 1) / C); band k is in run ((k + 1) C - 1) / B
    const auto run_of = [](int k, int C, int B) { return (int) (((int64_t) (k + 1) * C - 1) / B); };
    // the counters (a run's finished pieces) and flags (a piece's input prepared): the caller's, reused call to call
    static thread_local std::unique_ptr<std::atomic<int>[]> flags;
    static thread_local size_t n_flags = 0;
    const size_t per = (size_t) (CG + CD + 2 * GQ + DQ), need = (size_t) n * per;
    if (n_flags < need) {
        flags.reset(new std::atomic<int>[need]);
        n_flags = need;
    }
    for (size_t k = 0; k < need; ++k) flags[k].store(0, std::memory_order_relaxed);
    std::atomic<int>* const fl = flags.get();
    const auto c_gu = [&](int i, int c) -> std::atomic<int>& { return fl[(size_t) i * per + c]; };
    const auto c_dn = [&](int i, int c) -> std::atomic<int>& { return fl[(size_t) i * per + CG + c]; };
    const auto f_gu = [&](int i, int r, int q) -> std::atomic<int>& { return fl[(size_t) i * per + CG + CD + r * GQ + q]; };
    const auto f_dn = [&](int i, int q) -> std::atomic<int>& { return fl[(size_t) i * per + CG + CD + 2 * GQ + q]; };
    const auto spin = [](const std::atomic<int>& v, int want) {
        while (v.load(std::memory_order_acquire) < want) glmfast::cpu_relax();
    };
    const int a = n * 2 * GQ * CG, b = a + n * DQ * CD;
    pool.run(b + n * DB, [&](int job) {
        if (job < a) {
            const int c = job % CG, q = job / CG % GQ, r = job / CG / GQ % 2, i = job / CG / GQ / 2;
            if (c == 0) {
                gu_prep(i, r, q);
                f_gu(i, r, q).store(1, std::memory_order_release);
            } else {
                spin(f_gu(i, r, q), 1);
            }
            gu_rows(i, r, q, 128 * (GB * c / CG), 128 * (GB * (c + 1) / CG));
            c_gu(i, c).fetch_add(1, std::memory_order_release);
        } else if (job < b) {
            const int c = (job - a) % CD, q = (job - a) / CD % DQ, i = (job - a) / CD / DQ;
            if (c == 0) {
                // gate/up band q: the gate/up run holding its columns, gate and up, every piece
                spin(c_gu(i, run_of(q, CG, GB)), 2 * GQ);
                dn_prep(i, q);
                f_dn(i, q).store(1, std::memory_order_release);
            } else {
                spin(f_dn(i, q), 1);
            }
            dn_rows(i, q, 128 * (DB * c / CD), 128 * (DB * (c + 1) / CD));
            c_dn(i, c).fetch_add(1, std::memory_order_release);
        } else {
            const int i = (job - b) / DB, k = (job - b) % DB;
            spin(c_dn(i, run_of(k, CD, DB)), DQ);
            out(i, k);
        }
    });
}
}  // namespace

void Glm5Model::fast_cpu_experts(int il, int ne, const uint8_t* const* blob, const float* w, const float* x, float* out,
                                 const void* xq) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const auto& nf = F->cpu_fmt[(size_t) il];
    namespace kc = strata::kernels::cpu;
    const int n_ff = g.n_ff_exp, n_embd = g.n_embd;
    // an EXL3 layer's experts on the CPU threads (bl[0..nx): the blobs, wv their weights): o = their weighted sum
    const auto x3_cpu = [&](int nx, const uint8_t* const* bl, const float* wv, float* o) {
        // EXL3 (mul1), in split-K pieces of 128-256 K weights (X3Rows), one batch of jobs (x3_jobs): per expert,
        // matrix and piece of the inputs, those inputs' blocks and the piece's byte-sum products over every column;
        // per expert and down piece, its gate/up band finished (the clamped swiglu, that block of the down input) and
        // the piece's products; per expert and band of 128 outputs, the down rows; then the weighted sum in i order
        const int E = n_embd, FF = n_ff;
        const size_t per = X3Rows::floats(E, FF);
        float* x3 = F->cpu_x3.data();
        const auto rows = [&](int i) { return X3Rows{x3 + (size_t) i * per, E, FF}; };
        const auto part = [&](int i, int r) { return x3_part(bl[i], Ly.xl, r, E, FF); };
        x3_jobs(*F->cpu_pool, nx, E, FF, [&](int i, int r, int q) {
            const X3Rows R = rows(i);
            x3_gate_up_prep(&R, &x, 1, part(i, r), r, q);
        }, [&](int i, int r, int q, int c0, int c1) {
            const X3Rows R = rows(i);
            x3_gate_up_rows(&R, 1, part(i, r), r, q, c0, c1);
        }, [&](int i, int q) {
            const X3Rows R = rows(i);
            const kc::Exl3Part pt[3] = {part(i, 0), part(i, 1), part(i, 2)};
            x3_down_prep(&R, 1, pt, q, g.swiglu_exp);
        }, [&](int i, int q, int c0, int c1) {
            const X3Rows R = rows(i);
            x3_down_rows(&R, 1, part(i, 2), q, c0, c1);
        }, [&](int i, int b) { x3_down_block(rows(i), part(i, 2), b); });
        for (int r = 0; r < E; ++r) {
            float m = 0.0f;
            for (int i = 0; i < nx; ++i) m += wv[i] * rows(i).dn()[r];
            o[r] = m;
        }
    };
    // EXL3 on the iGPU AND the CPU threads at once (F->x3_frac > 0, set by the lane's start-up timing): the CPU's share
    // of the route (the first experts) while the iGPU computes the rest - EXL3's trellis decode costs both about the
    // same (Radeon 8060S 0.088 ms an expert, 16 Zen 5 threads 0.118), where a ggml type's is 2.5x apart.  A late or
    // failed iGPU call: its experts on the CPU threads after (the same answer)
    if (Ly.exl3 && F->x3_frac > 0.0f && F->igpu_experts != nullptr && F->igpu_x3s != nullptr && ne >= 2) {
        const int nc = std::max(1, std::min(ne - 1, (int) std::lround(ne * F->x3_frac)));
        const size_t off9[9] = {Ly.xl.suh[0], Ly.xl.suh[1], Ly.xl.suh[2], Ly.xl.svh[0], Ly.xl.svh[1],
                                Ly.xl.svh[2], Ly.xl.trellis[0], Ly.xl.trellis[1], Ly.xl.trellis[2]};
        if (F->igpu_x3s(Ly.xl.K, Ly.xl.cb, off9, n_embd, n_ff, g.swiglu_exp, ne - nc, (const void* const*) (blob + nc),
                        w + nc, x) == 0) {
            x3_cpu(nc, blob, w, out);
            float* go = F->x3_gout.data();
            const int rc = F->igpu_x3f(go, n_embd);
            if (rc != 0) {
                if (rc < 0) {
                    std::fprintf(stderr, "glm fast: CUDA%d iGPU lane failed - the CPU lane takes over\n", dev_);
                    F->igpu_experts = nullptr;
                }
                F->igpu_falls.fetch_add(1, std::memory_order_relaxed);
                x3_cpu(ne - nc, blob + nc, w + nc, go);
            } else {
                F->igpu_calls.fetch_add(1, std::memory_order_relaxed);
            }
            for (int r = 0; r < n_embd; ++r) out[r] += go[r];
            return;
        }
    }
    // the iGPU lane: the same sum from the registered RAM tier (an EXL3 layer through abyss_igpu_experts_x3, when the
    // library has it - else the CPU threads below)
    if (F->igpu_experts != nullptr && (!Ly.exl3 || F->igpu_x3 != nullptr)) {
        const auto ti = std::chrono::steady_clock::now();
        int rc;
        if (Ly.exl3) {
            const size_t off9[9] = {Ly.xl.suh[0], Ly.xl.suh[1], Ly.xl.suh[2], Ly.xl.svh[0], Ly.xl.svh[1],
                                    Ly.xl.svh[2], Ly.xl.trellis[0], Ly.xl.trellis[1], Ly.xl.trellis[2]};
            rc = F->igpu_x3(Ly.xl.K, Ly.xl.cb, off9, n_embd, n_ff, g.swiglu_exp, ne, (const void* const*) blob, w, x,
                            out);
        } else {
            rc = xq != nullptr && F->igpu_experts_q != nullptr
                     ? F->igpu_experts_q(Ly.gu_type, Ly.d_type, n_embd, n_ff, Ly.down_off, g.swiglu_exp, ne,
                                         (const void* const*) blob, w, xq, out)
                     : F->igpu_experts(Ly.gu_type, Ly.d_type, n_embd, n_ff, Ly.down_off, g.swiglu_exp, ne,
                                       (const void* const*) blob, w, x, out);
        }
        // a call is ~0.1-0.3 ms; one that waits for the kernel driver to fault a registered range in again (the stall
        // STRATA_GLM_IGPU_CHUNK_MB bounds) says so, at most once a second
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ti).count();
        if (ms > 500.0) {
            static std::atomic<int64_t> last{0};
            const int64_t now_s = std::chrono::duration_cast<std::chrono::seconds>(ti.time_since_epoch()).count();
            if (last.exchange(now_s) != now_s)
                std::fprintf(stderr, "glm fast: CUDA%d iGPU lane: a call took %.0f ms (the kernel driver re-validating "
                                     "registered RAM?)\n", dev_, ms);
        }
        if (rc == 0) {
            F->igpu_calls.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (rc < 0) {
            std::fprintf(stderr, "glm fast: CUDA%d iGPU lane failed - the CPU lane takes over\n", dev_);
            F->igpu_experts = nullptr;
        }
        F->igpu_falls.fetch_add(1, std::memory_order_relaxed);
    }
    // one job per pool thread: a contiguous run of the call's rows, the experts' gate/up rows end to end (then the
    // down rows likewise), the weighted sum after.  Each thread streams its share in one piece - 2-socket Xeon, 5
    // Q3_K experts from DRAM, 40 threads: 0.154 ms an expert vs 0.201 with 32/64-row jobs claimed in turn (the short
    // streams held the memory system to ~55 of the ~92 GB/s one socket reads; a pure read of the same jobs did too).
    // The run is cut in contiguous pieces claimed in turn, about 48 in all (STRATA_GLM_CPU_SPLIT=<pieces a thread>
    // overrides): with few threads a thread the disk readers or the service thread delay no longer holds up the whole
    // call, with many each still streams its share in one piece.  1x V100, 6-core i5, Maya-S decode: one piece a thread
    // 14.3 tok/s (the lane 34 ms a token), 2: 15.7, 4: 16.8, 8: 17.1 (21 ms; v1.0.11's 64-row jobs 16.8); the 2-socket
    // Xeon above, 40 threads: one piece each
    static const int kSplit = [] {
        const char* v = getenv("STRATA_GLM_CPU_SPLIT");
        return v ? std::max(1, std::min(64, std::atoi(v))) : 0;
    }();
    if (Ly.exl3) {
        x3_cpu(ne, blob, w, out);
        return;
    }
    const int threads = F->cpu_pool->active();
    const int T = threads * (kSplit > 0 ? kSplit : std::max(1, std::min(8, 48 / std::max(1, threads))));
    kc::native_quant_act(nf, x, F->cpu_act.data());
    const int gu_tot = ne * n_ff;
    F->cpu_pool->run(T, [&](int job) {
        const void* a[1] = {F->cpu_act.data()};
        const int q1 = (int) ((int64_t) gu_tot * (job + 1) / T);
        for (int q = (int) ((int64_t) gu_tot * job / T); q < q1;) {
            const int i = q / n_ff, r0 = q % n_ff, r1 = std::min(n_ff, r0 + (q1 - q));
            float* o[1] = {F->cpu_ff.data() + (size_t) i * n_ff};
            kc::native_gu_rows_split(nf, blob[i], blob[i] + Ly.gu_bytes, a, 1, o, r0, r1, g.swiglu_exp);
            q += r1 - r0;
        }
    });
    for (int i = 0; i < ne; ++i)
        kc::native_quant_h(nf, F->cpu_ff.data() + (size_t) i * n_ff, F->cpu_hq.data() + (size_t) i * kc::kNativeHBytes);
    const int dn_tot = ne * n_embd;
    F->cpu_pool->run(T, [&](int job) {
        const int q1 = (int) ((int64_t) dn_tot * (job + 1) / T);
        for (int q = (int) ((int64_t) dn_tot * job / T); q < q1;) {
            const int i = q / n_embd, r0 = q % n_embd, r1 = std::min(n_embd, r0 + (q1 - q));
            const void* h[1] = {F->cpu_hq.data() + (size_t) i * kc::kNativeHBytes};
            float* o[1] = {F->cpu_dn.data() + (size_t) i * n_embd};
            kc::native_down_rows_split(nf, blob[i] + Ly.down_off, h, 1, o, r0, r1);
            q += r1 - r0;
        }
    });
    for (int r = 0; r < n_embd; ++r) {
        float m = 0.0f;
        for (int i = 0; i < ne; ++i) m += w[i] * F->cpu_dn[(size_t) i * n_embd + r];
        out[r] = m;
    }
    // STRATA_GLM_CPU_LANE_VERIFY=1 (debug, slow): the same experts in double precision from ggml's dequantised rows;
    // the relative L2 distance of the lane's sum, accumulated and printed every 16 calls
    static const bool verify = getenv("STRATA_GLM_CPU_LANE_VERIFY") != nullptr;
    if (verify && !Ly.exl3) {
        static double e2 = 0, n2 = 0, worst = 0;
        static int calls = 0;
        const ggml_type_traits* tg = ggml_get_type_traits((ggml_type) Ly.gu_type);
        const ggml_type_traits* td = ggml_get_type_traits((ggml_type) Ly.d_type);
        std::vector<float> row((size_t) std::max(n_embd, n_ff));
        std::vector<double> hr((size_t) n_ff), ref((size_t) n_embd, 0.0);
        const double lim = g.swiglu_exp;
        for (int i = 0; i < ne; ++i) {
            for (int r = 0; r < n_ff; ++r) {
                double gs = 0.0, us = 0.0;
                tg->to_float(blob[i] + (size_t) r * nf.gu_row, row.data(), n_embd);
                for (int c = 0; c < n_embd; ++c) gs += (double) row[(size_t) c] * x[c];
                tg->to_float(blob[i] + Ly.gu_bytes + (size_t) r * nf.gu_row, row.data(), n_embd);
                for (int c = 0; c < n_embd; ++c) us += (double) row[(size_t) c] * x[c];
                gs = std::min(gs, lim);
                us = std::min(std::max(us, -lim), lim);
                hr[(size_t) r] = gs / (1.0 + std::exp(-gs)) * us;
            }
            for (int r = 0; r < n_embd; ++r) {
                double s = 0.0;
                td->to_float(blob[i] + Ly.down_off + (size_t) r * nf.d_row, row.data(), n_ff);
                for (int j = 0; j < n_ff; ++j) s += (double) row[(size_t) j] * hr[(size_t) j];
                ref[(size_t) r] += w[i] * s;
            }
        }
        double a = 0, b = 0;
        for (int r = 0; r < n_embd; ++r) {
            a += (out[r] - ref[(size_t) r]) * (out[r] - ref[(size_t) r]);
            b += ref[(size_t) r] * ref[(size_t) r];
        }
        e2 += a;
        n2 += b;
        worst = std::max(worst, std::sqrt(a / std::max(b, 1e-30)));
        if (++calls % 16 == 0)
            std::fprintf(stderr, "cpu lane verify (%d calls, layer %d): relative L2 error %.5f, worst call %.5f\n", calls, il,
                         std::sqrt(e2 / n2), worst);
    }
}

// A batched verify window's columns of a layer on the CPU threads at once (the service thread queues them,
// MoeRequest::win_more): the union of the columns' experts, each through the multi-token rows for the columns that
// route it - its weights read from RAM once - then every column's weighted sum in its own order.  Each (expert,
// column) product is the one-column call's bit for bit (the multi-token rows are ggml's dot per token, like the
// one-token rows), and so is each sum.  An EXL3 layer or the iGPU lane: one call a column
void Glm5Model::fast_cpu_experts_multi(int il, int n, const int* ne, const uint8_t* const* blob, const float* w,
                                       const float* const* x, float* const* out, const void* const* xq) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const int n_ff = g.n_ff_exp, n_embd = g.n_embd;
    // STRATA_GLM_CPU_MULTI_CHECK=1 (debug, slow): every column through the one-column call too, compared bit for bit
    static const bool check = getenv("STRATA_GLM_CPU_MULTI_CHECK") != nullptr;
    const auto check_cols = [&](const char* who) {
        static std::vector<float> ref;
        ref.resize((size_t) n_embd);
        static uint64_t cols = 0, bad = 0;
        for (int t = 0; t < n; ++t) {
            fast_cpu_experts(il, ne[t], blob + gf::kMaxK * t, w + gf::kMaxK * t, x[t], ref.data(), xq ? xq[t] : nullptr);
            ++cols;
            if (std::memcmp(ref.data(), out[t], (size_t) n_embd * sizeof(float)) != 0) ++bad;
        }
        if (cols % 512 < (uint64_t) n)
            std::fprintf(stderr, "%s multi check: %llu columns, %llu differ from the one-column call\n", who,
                         (unsigned long long) cols, (unsigned long long) bad);
    };
    // the iGPU lane's columns call (abyss_igpu_experts_cols): the window's columns in one call; 1 (busy, a blob it
    // cannot see, ...): a call a column below; < 0: the lane is gone, the CPU takes over as fast_cpu_experts does
    if (n >= 2 && !Ly.exl3 && F->igpu_experts != nullptr && F->igpu_cols != nullptr && n <= kVerifyMax) {
        const int rc = F->igpu_cols(Ly.gu_type, Ly.d_type, n_embd, n_ff, Ly.down_off, g.swiglu_exp, n, ne,
                                    (const void* const*) blob, w, x, xq, out);
        if (rc == 0) {
            F->igpu_calls.fetch_add((uint64_t) n, std::memory_order_relaxed);
            F->igpu_cols_calls.fetch_add(1, std::memory_order_relaxed);
            F->igpu_cols_cols.fetch_add((uint64_t) n, std::memory_order_relaxed);
            if (check) check_cols("igpu");
            return;
        }
        if (rc < 0) {
            std::fprintf(stderr, "glm fast: CUDA%d iGPU lane failed (columns call) - the CPU lane takes over\n", dev_);
            F->igpu_experts = nullptr;
        }
        F->igpu_cols_falls.fetch_add(1, std::memory_order_relaxed);
    }
    if (n <= 1 || Ly.exl3 || F->igpu_experts != nullptr) {
        for (int t = 0; t < n; ++t)
            fast_cpu_experts(il, ne[t], blob + gf::kMaxK * t, w + gf::kMaxK * t, x[t], out[t], xq ? xq[t] : nullptr);
        return;
    }
    namespace kc = strata::kernels::cpu;
    const auto& nf = F->cpu_fmt[(size_t) il];
    constexpr int kCols = kVerifyMax, kPairs = gf::kMaxK * kVerifyMax;   // (a column's entries at stride kMaxK)
    // the union: expert u's blob, its columns, the (expert, column) pair each of them is; pair p's buffers at p
    int nu = 0, np = 0;
    const uint8_t* ub[kPairs];
    int ucnt[kPairs], ucol[kPairs][kCols], upair[kPairs][kCols], pair_of[kCols][gf::kMaxK];
    for (int t = 0; t < n; ++t)
        for (int i = 0; i < ne[t]; ++i) {
            const uint8_t* b = blob[gf::kMaxK * t + i];
            int u = 0;
            while (u < nu && ub[u] != b) ++u;
            if (u == nu) {
                ub[nu] = b;
                ucnt[nu++] = 0;
            }
            ucol[u][ucnt[u]] = t;
            upair[u][ucnt[u]++] = np;
            pair_of[t][i] = np++;
        }
    if (F->cpu_act_m.size() < (size_t) kCols * kc::kNativeActBytes) {
        F->cpu_act_m.resize((size_t) kCols * kc::kNativeActBytes);
        F->cpu_hq_m.resize((size_t) kPairs * kc::kNativeHBytes);
        F->cpu_ff_m.resize((size_t) kPairs * n_ff);
        F->cpu_dn_m.resize((size_t) kPairs * n_embd);
    }
    uint8_t* act = F->cpu_act_m.data();
    for (int t = 0; t < n; ++t) kc::native_quant_act(nf, x[t], act + (size_t) t * kc::kNativeActBytes);
    static const int kSplit = [] {
        const char* v = getenv("STRATA_GLM_CPU_SPLIT");
        return v ? std::max(1, std::min(64, std::atoi(v))) : 0;
    }();
    const int threads = F->cpu_pool->active();
    const int T = threads * (kSplit > 0 ? kSplit : std::max(1, std::min(8, 48 / std::max(1, threads))));
    // gate/up: the union's rows end to end, a contiguous run a job (as fast_cpu_experts), every column of an expert
    const int gu_tot = nu * n_ff;
    F->cpu_pool->run(T, [&](int job) {
        const int q1 = (int) ((int64_t) gu_tot * (job + 1) / T);
        for (int q = (int) ((int64_t) gu_tot * job / T); q < q1;) {
            const int u = q / n_ff, r0 = q % n_ff, r1 = std::min(n_ff, r0 + (q1 - q));
            const void* a[kCols];
            float* o[kCols];
            for (int k = 0; k < ucnt[u]; ++k) {
                a[k] = act + (size_t) ucol[u][k] * kc::kNativeActBytes;
                o[k] = F->cpu_ff_m.data() + (size_t) upair[u][k] * n_ff;
            }
            kc::native_gu_rows_split(nf, ub[u], ub[u] + Ly.gu_bytes, a, ucnt[u], o, r0, r1, g.swiglu_exp);
            q += r1 - r0;
        }
    });
    for (int p = 0; p < np; ++p)
        kc::native_quant_h(nf, F->cpu_ff_m.data() + (size_t) p * n_ff, F->cpu_hq_m.data() + (size_t) p * kc::kNativeHBytes);
    const int dn_tot = nu * n_embd;
    F->cpu_pool->run(T, [&](int job) {
        const int q1 = (int) ((int64_t) dn_tot * (job + 1) / T);
        for (int q = (int) ((int64_t) dn_tot * job / T); q < q1;) {
            const int u = q / n_embd, r0 = q % n_embd, r1 = std::min(n_embd, r0 + (q1 - q));
            const void* h[kCols];
            float* o[kCols];
            for (int k = 0; k < ucnt[u]; ++k) {
                h[k] = F->cpu_hq_m.data() + (size_t) upair[u][k] * kc::kNativeHBytes;
                o[k] = F->cpu_dn_m.data() + (size_t) upair[u][k] * n_embd;
            }
            kc::native_down_rows_split(nf, ub[u] + Ly.down_off, h, ucnt[u], o, r0, r1);
            q += r1 - r0;
        }
    });
    for (int t = 0; t < n; ++t)
        for (int r = 0; r < n_embd; ++r) {
            float m = 0.0f;
            for (int i = 0; i < ne[t]; ++i) m += w[gf::kMaxK * t + i] * F->cpu_dn_m[(size_t) pair_of[t][i] * n_embd + r];
            out[t][r] = m;
        }
    if (check) check_cols("cpu");
    F->cpu_multi_calls.fetch_add(1, std::memory_order_relaxed);
    F->cpu_multi_cols.fetch_add((uint64_t) n, std::memory_order_relaxed);
    F->cpu_multi_uses.fetch_add((uint64_t) np, std::memory_order_relaxed);
    F->cpu_multi_experts.fetch_add((uint64_t) nu, std::memory_order_relaxed);
}

// The CPU lane's pool a split's parts share (fast_cpu_lane_setup): made by the first part, found by the later ones -
// one model a process; a reload, or another count, makes a new one.
static std::shared_ptr<glmfast::Workers> shared_cpu_pool(int threads) {
    static std::mutex mu;
    static std::weak_ptr<glmfast::Workers> held;
    std::lock_guard<std::mutex> lk(mu);
    std::shared_ptr<glmfast::Workers> p = held.lock();
    if (p == nullptr || p->size() != threads) {
        p = std::make_shared<glmfast::Workers>(threads - 1, 20000);
        held = p;
    }
    return p;
}

// This process's physical cores (distinct package/core pairs among the CPUs it may run on); 0 when unknown.
static int physical_cores() {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return 0;
    std::vector<std::pair<int, int>> seen;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &set)) continue;
        const std::string b = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
        int pk = -1, co = -1;
        std::ifstream(b + "physical_package_id") >> pk;
        std::ifstream(b + "core_id") >> co;
        if (co < 0) return 0;
        if (std::find(seen.begin(), seen.end(), std::make_pair(pk, co)) == seen.end()) seen.push_back({pk, co});
    }
    return (int) seen.size();
#elif defined(_WIN32)
    // Windows: the processor cores (each one entry, whatever its SMT threads)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return 0;
    std::vector<uint8_t> buf(len);
    auto* info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) buf.data();
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) return 0;
    int cores = 0;
    for (DWORD at = 0; at < len;) {
        auto* e = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) (buf.data() + at);
        if (e->Relationship == RelationProcessorCore) ++cores;
        at += e->Size;
    }
    return cores;
#else
    return 0;
#endif
}

// The CPU LANE, on by default: one thread per physical core (split evenly across the parts of a layer split);
// STRATA_GLM_CPU_LANE=<threads> sets the count, 0 turns it off; STRATA_GLM_CPU_LANE<n>=<threads> sets one GPU's own
// (CUDA<n>: a slow link wants more threads than a fast one).  Measures this machine once - one expert on the CPU
// pool vs one over PCIe - and derives the split: of f RAM-tier experts in a route, the k the host computes so that
// the slower of the two lanes finishes first (STRATA_GLM_CPU_PLAN=<digits for f = 0..8> overrides).  A CPU slower
// than the PCIe link at every f leaves the lane off.
bool Glm5Model::fast_cpu_lane_setup(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const char* lv = getenv(("STRATA_GLM_CPU_LANE" + std::to_string(dev_)).c_str());
    const bool own_count = lv != nullptr;   // this card's own count: a pool of its own
    if (lv == nullptr) lv = getenv("STRATA_GLM_CPU_LANE");
    // a split of 3+ GPUs runs its parts one after another - a decode token visits the cards in turn, and the pipelined
    // speculative decode, which runs two at once, is a two-part one - so ONE pool serves every part, sized as one
    // GPU's (Strata: one expert pool for the process).  A pool per part, cores / parts each, left most of the CPU idle -
    // and as idle workers spin 20 ms after a batch, more threads a part would have spun against the card at work.
    // Maya-L on 9 GPUs, 88 vCPUs, fresh prompts (95.8% VRAM hits): 9 threads a card 0.74 ms a RAM-tier expert, 25.7
    // tok/s; one pool of 40 0.32 ms, 30.7 tok/s; an 8K prompt, whose chunks take the pool in turn, 1265 -> 1466 tok/s.
    // STRATA_GLM_CPU_SHARED=0: a pool per part (A/B); 1: one pool for a two-GPU split too.
    const char* shv = getenv("STRATA_GLM_CPU_SHARED");
    const bool shared = n_parts_ >= 2 && !own_count &&
                        (shv != nullptr && shv[0] != '\0' ? std::atoi(shv) != 0 : n_parts_ >= 3);
    int threads = 0;
    if (lv != nullptr) {
        threads = std::max(0, std::min(64, std::atoi(lv)));
    } else {
        int cores = physical_cores();
        if (cores <= 0) cores = (int) std::thread::hardware_concurrency() / 2;
        threads = cores;
        // cores without SMT siblings (Intel's Arrow Lake, E-core parts): a thread a core spins on EVERY CPU, and the
        // disk readers, the service and the main thread wait for a time slice - two thirds of them leave the rest free
        // (24-core Ultra 9 275HX, Windows, 33 GB RAM tier, the same 36.6 disk reads a token: 24 threads 2.1 tok/s and
        // a disk wait 16.8 ms, 20 3.5 tok/s, 16 3.8 tok/s and 8.1 ms, 12 3.8 tok/s; shard views released: 24 threads
        // 5.7 tok/s, 16 7.6)
        if (const int logical = (int) std::thread::hardware_concurrency(); logical > 0 && cores >= logical)
            threads = logical * 2 / 3;
        // one GPU: at most one NUMA node's worth, less 4 - a spinning pool on both sockets syncs across them, and the
        // CPUs left free keep the main thread's event wait, the service and warm-up threads off the spinning ones
        // (2-socket Xeon, 88 vCPUs, Q3_K experts: 88 threads 531 us an expert, 44 216 us, 44 with the RAM tier
        // interleaved 152 us - ~85 of the ~105 GB/s this host reads; a pool per socket, tried, did not add to that)
        if (n_parts_ == 1 || shared)
            if (const int node = numa_node_cpus(); node > 0 && threads > node - 4) threads = node - 4;
        if (shared)
            threads = std::min(threads, cores - (n_parts_ - 1));   // a core for each other part's service thread
        else
            threads /= std::max(1, n_parts_);   // a two-part split's parts can run at once: one pool each
        if (threads < 2) threads = 0;   // one core is the service thread's
    }
    // a split whose parts have a pool each (two GPUs, or STRATA_GLM_CPU_SHARED=0): each part's pool on CPUs of its own
    // (part_cpus: whole NUMA nodes while there are enough, runs of whole cores of a shared one otherwise), at most those
    // CPUs less the part's share of the 4 a node keeps free, like one GPU's - the shared pool is one GPU's, unpinned.  Unpinned, the parts' pools (half the vCPUs each) shared the cores: a part works while the other's idle
    // workers spin, and the 2-socket Xeon's 3090 + 5060 Ti split took 0.28 ms an expert where one GPU's pool took 0.15
    // (and with MTP drafting both parts run at once).  The RAM tier stays interleaved (one socket reading only its own
    // memory: 54 GB/s, interleaved 77).  By default only on a host with two NUMA nodes or more, where it measured
    // faster; on one node the unpinned pools sharing every core were (two V100s, one 14-core Xeon: 25.8 tok/s against
    // 24.7 pinned).  STRATA_GLM_CPU_PIN=1 / 0 pins / unpins a split's pools on any host.
    // STRATA_GLM_CPU_PIN=<CPU list> ("0-15", "0,2,4-7"): the pool's workers on those CPUs, ONE GPU too, and no more
    // threads than the list has - the hybrid CPUs' P-cores (an i7-14700K's 16 P-core threads: 31.26 tok/s against
    // 30.41 unpinned, issue #4).  Only the workers: the GPU runtime's and the service thread stay free (the whole
    // process on the 8 physical P-cores collapsed to 4.97 tok/s).
    std::vector<int> pin_cpus;
    int pin_node = -1;
    const char* pin_env = getenv("STRATA_GLM_CPU_PIN");
    std::vector<int> pin_list;
    if (pin_env != nullptr && (std::strchr(pin_env, '-') != nullptr || std::strchr(pin_env, ',') != nullptr ||
                               std::atoi(pin_env) > 1)) {
        for (const char* q = pin_env; *q;) {
            char* e = nullptr;
            const long a = std::strtol(q, &e, 10);
            if (e == q) break;
            long b = a;
            q = e;
            if (*q == '-') {
                b = std::strtol(q + 1, &e, 10);
                q = e;
            }
            for (long c = a; c <= b && c < 4096; ++c)
                if (c >= 0) pin_list.push_back((int) c);
            if (*q == ',') ++q;
            else if (*q) break;
        }
        std::sort(pin_list.begin(), pin_list.end());
        pin_list.erase(std::unique(pin_list.begin(), pin_list.end()), pin_list.end());
    }
    const bool pin = !pin_list.empty() ? false
                     : pin_env != nullptr && pin_env[0] ? std::atoi(pin_env) != 0 : numa_nodes().size() >= 2;
    if (!pin_list.empty() && !shared && threads > 0) {
        pin_cpus = pin_list;
        threads = std::min(threads, std::max(2, (int) pin_list.size()));
    }
    if (n_parts_ >= 2 && !shared && threads > 0 && pin) {
        std::vector<int> gpu_node;
        for (int d : split_devs_) gpu_node.push_back(gpu_numa_node(d));
        int spare = 4;
        pin_cpus = part_cpus(part_, n_parts_, gpu_node, pin_node, spare);
        // a thread a physical core, as unpinned - their SMT siblings are the CPUs left free (one CCD of a 5900X a
        // part: 10 threads on its 6 cores 10.88 tok/s, 6 threads 11.45, #77); no siblings: those CPUs less the spare
        if (!pin_cpus.empty() && lv == nullptr) {
            std::vector<std::pair<int, int>> pc;
            for (int c : pin_cpus) pc.push_back(cpu_core(c));
            std::sort(pc.begin(), pc.end());
            const int n_cores = (int) (std::unique(pc.begin(), pc.end()) - pc.begin());
            threads = n_cores < (int) pin_cpus.size() ? std::max(2, n_cores)
                                                      : std::max(2, (int) pin_cpus.size() - spare);
        }
    }
    if (threads <= 0 || g.n_embd > gf::kMaxEmbd || g.n_exp_used > gf::kMaxK) return true;
    namespace kc = strata::kernels::cpu;
    F->cpu_fmt.assign(F->L.size(), kc::NativeFmt{});
    int il_cal = -1;
    for (int il = l0_; il < lt_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (!Ly.moe || Ly.mtp) continue;
        if (Ly.exl3) {
            // EXL3: the CPU kernel takes mul1 and mcg (+ E4M3: exlr8) - 3inst stays on the PCIe lane; n_ff marks the
            // layer
            if (Ly.xl.cb >= 1 && Ly.xl.cb <= 3) {
                auto& f = F->cpu_fmt[(size_t) il];
                f.gu_type = f.d_type = strata::kernels::exl3::kTypeEXL3;
                f.n_embd = g.n_embd;
                f.n_ff = g.n_ff_exp;
                if (il_cal < 0) il_cal = il;
            }
            continue;
        }
        std::string e2;
        kc::NativeFmt f;
        if (kc::native_fmt(Ly.gu_type, Ly.d_type, g.n_embd, g.n_ff_exp, f, e2) && f.h_bytes <= kc::kNativeHBytes)
            F->cpu_fmt[(size_t) il] = f;
        if (il_cal < 0 && F->cpu_fmt[(size_t) il].n_ff > 0) il_cal = il;
    }
    if (il_cal < 0) return true;
    fast_igpu_setup();
    if (const char* ck = getenv("STRATA_GLM_CPU_LANE_CHECK"))   // the check's layer (below)
        if (std::atoi(ck) > 0 && std::atoi(ck) < (int) F->cpu_fmt.size() && F->cpu_fmt[(size_t) std::atoi(ck)].n_ff > 0)
            il_cal = std::atoi(ck);
    // a calibration blob: a RAM-tier expert of that layer (the warm-up filled the tier; a class holds every layer of
    // its blob size)
    const auto& R = F->rc[(size_t) F->layer_rc[(size_t) il_cal]];
    // (up to 4 of them: a route's RAM-tier experts come several to a call, and one alone leaves most of the pool idle -
    // timed alone, a 40-thread pool's expert took 0.26 ms where a decode's calls ran them at 0.16, and the split
    // sent the 8th of 8 over PCIe, 2.1 ms on the critical path)
    // Each call takes the next 4 of up to 64 (~480 MB of Maya-S24's 7.5 MB experts): a decode reads its RAM-tier
    // experts from DRAM, and 4 timed again and again stay in the CPU's L3 - a 7950X (64 MB L3) timed 0.097 ms an
    // expert where decode ran them at ~0.16 (DRAM-bound, 47 of its 57 GB/s), and the split kept 4 and 5 of a layer's
    // RAM-tier experts on the CPU: 25.1 / 25.3 tok/s against 26.9 / 27.5 with one of them over PCIe
    constexpr int kCalSets = 16;
    std::vector<const uint8_t*> cal_all;
    for (int s = 0; s < R.n && (int) cal_all.size() < 4 * kCalSets; ++s)
        if (R.st[(size_t) s] == FastState::kRHold && R.key[(size_t) s] / g.n_expert == il_cal)
            cal_all.push_back(R.base + (size_t) s * R.stride);
    if (cal_all.empty()) return true;   // nothing in RAM: the lane would never run
    const uint8_t* cal = cal_all[0];
    const int n_cal = (int) std::min<size_t>(4, cal_all.size());
    const int n_sets = (int) cal_all.size() / n_cal;   // (fewer than 64 held: the sets repeat sooner)
    auto cal_set = [&](int i) { return cal_all.data() + (size_t) (i % n_sets) * n_cal; };
    void* hp = nullptr;
    if (cudaHostAlloc(&hp, gf::kCpuAnswers * sizeof(gf::CpuAnswer), cudaHostAllocMapped) != cudaSuccess ||
        cudaMalloc((void**) &F->cpu_seq_d, 64) != cudaSuccess) {
        err = "glm fast: the CPU lane's buffers did not allocate";
        return false;
    }
    std::memset(hp, 0, gf::kCpuAnswers * sizeof(gf::CpuAnswer));
    cudaMemset(F->cpu_seq_d, 0, 64);
    F->cpu_ans_h = (gf::CpuAnswer*) hp;
    if (F->igpu_reg != nullptr) {   // the iGPU lane writes its sums into these slots in place
        char e[256] = {0};
        if (F->igpu_reg(hp, gf::kCpuAnswers * sizeof(gf::CpuAnswer), e, (int) sizeof e) != 0)
            std::fprintf(stderr, "glm fast: CUDA%d iGPU lane: the answers did not register (%s) - copied instead\n", dev_, e);
    }
    void* dp = nullptr;
    cudaHostGetDevicePointer(&dp, hp, 0);
    F->cpu_act.assign(kc::kNativeActBytes, 0);
    F->cpu_hq.assign((size_t) k_slots(g) * kc::kNativeHBytes, 0);
    F->cpu_ff.assign((size_t) k_slots(g) * g.n_ff_exp, 0.0f);
    F->cpu_dn.assign((size_t) k_slots(g) * g.n_embd, 0.0f);
    if (exl3_)   // (fast_cpu_experts' EXL3 layout, per expert)
        F->cpu_x3.assign((size_t) k_slots(g) * X3Rows::floats(g.n_embd, g.n_ff_exp), 0.0f);
    // the service thread is the pool's last worker; idle workers spin 20 ms (a decode's routes come ~1 ms apart)
    if (shared) {
        F->cpu_pool = shared_cpu_pool(threads);
    } else {
        F->cpu_pool.reset(new glmfast::Workers(threads - 1, 20000, pin_cpus));
        F->cpu_node = pin_node;
        F->cpu_pin = pin_cpus;
    }
    // ---- calibration, each lane alone: the CPU after 100 ms of the same work (an idle CPU's clocks take tens of ms to
    //      ramp up - a decode keeps them up), then the MEDIAN of 48 runs.  The mean of 16 swung with a few slow runs
    //      (the warm-up's last copies, a preempted worker): a Ryzen AI Max+ 395 timed an expert anywhere from 0.092 to
    //      0.159 ms on one model, and the slow readings sent the 8th of 8 RAM-tier experts over PCIe Gen4 x4 - a
    //      decode ~10% slower until the next start
    std::vector<float> x((size_t) g.n_embd), out((size_t) g.n_embd);
    for (int i = 0; i < g.n_embd; ++i) x[(size_t) i] = 0.01f * (float) ((i * 37) % 101 - 50);
    const float w4[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    int set = 0;
    constexpr int kCalReps = 48;
    const bool cal_verbose = getenv("STRATA_GLM_CPU_CAL_VERBOSE") != nullptr;
    double cal_max = 0.0;   // the last timing's slowest run, an expert's share
    const auto time_lane = [&](const char* who) {
        for (const auto tw = std::chrono::steady_clock::now();
             std::chrono::steady_clock::now() - tw < std::chrono::milliseconds(100);)
            fast_cpu_experts(il_cal, n_cal, cal_set(set++), w4, x.data(), out.data());
        std::vector<double> c_runs;
        c_runs.reserve(kCalReps);
        for (int rep = 0; rep < kCalReps; ++rep) {
            const uint8_t* const* cals = cal_set(set++);
            const auto t0 = std::chrono::steady_clock::now();
            fast_cpu_experts(il_cal, n_cal, cals, w4, x.data(), out.data());
            c_runs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(c_runs.begin(), c_runs.end());
        const double mean = std::accumulate(c_runs.begin(), c_runs.end(), 0.0) / kCalReps / n_cal;
        const double med = 0.5 * (c_runs[kCalReps / 2 - 1] + c_runs[kCalReps / 2]) / n_cal;   // an expert's share
        cal_max = c_runs.back() / n_cal;
        if (cal_verbose)
            std::fprintf(stderr, "glm fast: CUDA%d %s lane calibration: an expert min %.3f / median %.3f / mean %.3f / "
                                 "max %.3f ms (%d runs of %d)\n", dev_, who, c_runs.front() / n_cal, med, mean,
                         c_runs.back() / n_cal, kCalReps, n_cal);
        return med;
    };
    // the CPU's own time (the prompt path stages experts to the CPU threads by it, whatever computes the decode's lane),
    // then the iGPU lane's: it keeps the decode's lane only when it is the faster (STRATA_GLM_IGPU=1: always)
    const abyss_igpu_experts_fn igpu = F->igpu_experts;
    F->igpu_experts = nullptr;
    const double cpu_ms = time_lane("CPU");
    double c_ms = cpu_ms;
    if (igpu != nullptr) {
        F->igpu_experts = igpu;
        // A kernel-driver stall (see fast_cpu_experts) while the lane is timed sends its calls to the CPU threads past
        // the 20 ms deadline, and the median then times the CPU: v1.3.13, a call held 11 s right after the start ->
        // "an expert 0.129 ms vs 0.124 on the CPU" and the whole session decoded on the CPU lane, ~45 tok/s against
        // ~75.  A timing counts only when every call ran on the iGPU and none took 500 ms; otherwise wait until the
        // lane takes 16 calls in a row again and time it again, up to STRATA_GLM_IGPU_CAL_WAIT_S (60) in all.  No clean
        // timing by then: the iGPU keeps the lane (each stalled call goes to the CPU threads anyway)
        static const double wait_s = [] {
            const char* v = getenv("STRATA_GLM_IGPU_CAL_WAIT_S");
            return v != nullptr ? std::max(0.0, std::atof(v)) : 60.0;
        }();
        const auto tw0 = std::chrono::steady_clock::now();
        const auto waited = [&] {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count();
        };
        double ig_ms = 0.0;
        bool clean = false;
        for (;;) {
            const uint64_t f0 = F->igpu_falls.load();
            ig_ms = time_lane("iGPU");
            const uint64_t falls = F->igpu_falls.load() - f0;
            clean = falls == 0 && cal_max * n_cal < 500.0;
            if (clean || F->igpu_experts == nullptr || waited() >= wait_s) break;
            std::fprintf(stderr, "glm fast: CUDA%d iGPU lane: the timing hit a stall (%llu calls went to the CPU, the "
                                 "slowest %.0f ms) - timing it again once it takes calls\n",
                         dev_, (unsigned long long) falls, cal_max * n_cal);
            for (int run = 0; run < 16 && F->igpu_experts != nullptr && waited() < wait_s;) {
                const uint64_t c0 = F->igpu_calls.load();
                fast_cpu_experts(il_cal, n_cal, cal_set(set++), w4, x.data(), out.data());
                if (F->igpu_calls.load() > c0) {
                    ++run;
                } else {
                    run = 0;
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            }
        }
        const char* on = getenv("STRATA_GLM_IGPU");
        const bool failed = F->igpu_experts == nullptr;
        const bool keep = !failed && (!clean || ig_ms < cpu_ms || (on != nullptr && std::atoi(on) == 1));
        if (clean || failed)
            std::fprintf(stderr, "glm fast: CUDA%d iGPU lane: an expert %.3f ms vs %.3f on the CPU's %d threads -> %s\n",
                         dev_, ig_ms, cpu_ms, threads, keep ? "the iGPU computes the lane" : "the CPU (faster here)");
        else
            std::fprintf(stderr, "glm fast: CUDA%d iGPU lane: no clean timing in %.0f s (the kernel driver stalling it) "
                                 "- the iGPU computes the lane, a stalled call goes to the CPU's %d threads\n",
                         dev_, waited(), threads);
        if (keep && clean) c_ms = ig_ms;   // (not clean: the split below is planned on the CPU's time)
        else if (!keep) F->igpu_experts = nullptr;
        // an EXL3 pack, opt-in: the iGPU and the CPU threads share each route (STRATA_GLM_X3_SPLIT=<the CPU's fraction>,
        // or "auto" by their speeds).  Off by default: on a Ryzen AI Max+ 395 the two share the package's power and
        // heat, and GLM-5.3-Flash sc_2.10bpw's glm_ids4k ran 22.7 ms a token on the iGPU alone against 23.5 (CPU 46%
        // on 8 threads), 25.1 (25% on 4) and 31.6-32.6 (16 threads)
        const char* xs = getenv("STRATA_GLM_X3_SPLIT");
        if (exl3_ && xs != nullptr && F->igpu_experts != nullptr && F->igpu_x3s != nullptr && clean) {
            const double fr = std::strcmp(xs, "auto") == 0 ? ig_ms / (ig_ms + cpu_ms) : std::atof(xs);
            F->x3_frac = (float) std::max(0.0, std::min(0.9, fr));
            if (F->x3_frac > 0.0f) {
                F->x3_gout.assign((size_t) g.n_embd, 0.0f);
                c_ms = ig_ms * cpu_ms / (ig_ms + cpu_ms);
                std::fprintf(stderr, "glm fast: CUDA%d EXL3 lane: the CPU threads take %.0f%% of each route's RAM-tier "
                                     "experts, the iGPU the rest (an expert %.3f ms together)\n", dev_,
                             100.0 * F->x3_frac, c_ms);
            }
        }
    }
    const size_t blob = F->L[(size_t) il_cal].blob;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    double p_ms = 0.0;
    // the link is timed with the GPU awake (see glm_link_wake): 400 ms for the clocks and the link to ramp up, then
    // 1.1 s of the kernel left for both measurements below
    cudaStream_t wake = nullptr;
    if (cudaStreamCreateWithFlags(&wake, cudaStreamNonBlocking) == cudaSuccess) {
        glm_link_wake<<<1, 1, 0, wake>>>(1500ull * 1000000ull);
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }
    for (int rep = 0; rep < 24; ++rep) {
        cudaEventRecord(e0, F->cs);
        cudaMemcpyAsync(F->scratch, cal, blob, cudaMemcpyHostToDevice, F->cs);
        cudaEventRecord(e1, F->cs);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        if (rep >= 8) p_ms += ms;
    }
    p_ms /= 16.0;
    // ... and the decode's own PCIe path: moe_fetch's SMs reading the host-mapped blob (zero copy), not the copy engine.
    // On PCIe Gen4 x4 (Ryzen AI Max+ 395 + RTX 5090) it is slower than the DMA copy above, and a split planned on the
    // DMA time sent the 8th of 8 RAM-tier experts over PCIe where the CPU alone was ~10% faster a token
    std::vector<float> z_runs;
    for (int rep = 0; rep < 24; ++rep) {
        cudaEventRecord(e0, F->cs);
        gf::zero_copy(cal, F->scratch, blob, F->cs);
        cudaEventRecord(e1, F->cs);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        if (rep >= 8 && cudaEventElapsedTime(&ms, e0, e1) == cudaSuccess && ms > 0.0f) z_runs.push_back(ms);
    }
    double pz_ms = p_ms;
    if (!z_runs.empty()) {
        std::sort(z_runs.begin(), z_runs.end());
        pz_ms = z_runs[z_runs.size() / 2];
    }
    if (getenv("STRATA_GLM_CPU_CAL_VERBOSE") != nullptr)
        std::fprintf(stderr, "glm fast: CUDA%d CPU lane calibration: an expert over PCIe %.3f ms by the copy engine, %.3f "
                             "ms the decode's way (SMs reading host memory)\n", dev_, p_ms, pz_ms);
    // ... and its streaming rate - copies back to back, the way the prompt path stages experts (a 3090 on PCIe 3.0 x8:
    // 2.09 ms for one copy, 1.57 a copy in a stream): the prompt's CPU / PCIe split plans with this one
    double ps_ms = p_ms;
    {
        cudaEventRecord(e0, F->cs);
        for (int rep = 0; rep < 8; ++rep) cudaMemcpyAsync(F->scratch, cal, blob, cudaMemcpyHostToDevice, F->cs);
        cudaEventRecord(e1, F->cs);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, e0, e1) == cudaSuccess && ms > 0.0f) ps_ms = std::min(p_ms, (double) ms / 8.0);
    }
    if (wake != nullptr) {
        cudaStreamSynchronize(wake);
        cudaStreamDestroy(wake);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    F->cpu_c_ms = cpu_ms;   // (the prompt path splits its staged experts by them too: the CPU's own)
    F->cpu_p_ms = p_ms;
    F->cpu_ps_ms = ps_ms;
    // the split: k of f to the host minimises max(PCIe (f - k) p, lane overhead + k c); ties keep the PCIe lane.  p is
    // the decode's own PCIe path (zero copy, above), the slower of the two timings
    const double over_ms = 0.04;
    const double pd_ms = std::max(p_ms, pz_ms);
    unsigned long long plan = 0;
    std::string tab;
    const char* pv = getenv("STRATA_GLM_CPU_PLAN");
    // STRATA_GLM_PCIE_SHARE=<0..1> (setup's calibration): the share of a route's RAM-tier experts that go over PCIe
    const char* sv = getenv("STRATA_GLM_PCIE_SHARE");
    const double share = sv != nullptr ? std::max(0.0, std::min(1.0, std::atof(sv))) : -1.0;
    for (int f = 0; f <= plan_f(g); ++f) {
        int k = 0;
        if (pv != nullptr) {
            k = f < (int) std::strlen(pv) ? std::max(0, std::min(f, pv[f] - '0')) : 0;
        } else if (share >= 0.0) {
            k = f - (int) std::lround(share * f);
        } else {
            double best = pd_ms * f;
            for (int kk = 1; kk <= f; ++kk) {
                const double t = std::max(pd_ms * (f - kk), over_ms + c_ms * kk);
                if (t < best - 1e-9) {
                    best = t;
                    k = kk;
                }
            }
        }
        plan |= (unsigned long long) k << (4 * f);
        tab += " " + std::to_string(k);
    }
    // (kept with STRATA_GLM_PCIE_SHARE_LATE: a request can start with the lane idle and move to it later)
    if (plan == 0ull && getenv("STRATA_GLM_PCIE_SHARE_LATE") == nullptr) {
        std::fprintf(stderr, "glm fast: CUDA%d CPU lane off: an expert %.3f ms on the CPU vs %.3f ms over PCIe\n", dev_,
                     c_ms, pd_ms);
        F->cpu_pool.reset();
        return true;
    }
    F->cpu_plan = plan;
    F->cpu_plan_start = plan;
    // STRATA_GLM_PCIE_SHARE_LATE=<share>[:<steps>] (issue #14): each request starts on the plan above and moves to this
    // share after <steps> token boundaries (default 256). A lower share copies fewer experts over PCIe, and so promotes
    // fewer into VRAM: set from the start it leaves the startup's pool in place (Ornith on a DDR4 desktop: share 0
    // from the start 21 tokens/s, the engine's own 24.7, share 0 once the own share had filled the pool 34.7)
    if (const char* lv = getenv("STRATA_GLM_PCIE_SHARE_LATE")) {
        const double ls = std::max(0.0, std::min(1.0, std::atof(lv)));
        const char* colon = std::strchr(lv, ':');
        for (int f = 0; f <= plan_f(g); ++f) F->cpu_plan_late |= (unsigned long long) (f - (int) std::lround(ls * f)) << (4 * f);
        // <share>:settle[:<window>[:<gain>[:<drop>]]]: the switch follows the pool, not the request (#14: the warm
        // state the --deep sweep measures builds over several answers, and the per-request switch never reached it)
        if (colon != nullptr && std::strncmp(colon + 1, "settle", 6) == 0) {
            F->settle = true;
            const char* p = colon + 7;
            double v[3] = {(double) F->settle_win, F->settle_eps * 100.0, F->settle_drop * 100.0};
            for (int i = 0; i < 3 && *p == ':'; ++i) {
                v[i] = std::atof(p + 1);
                p = std::strchr(p + 1, ':');
                if (p == nullptr) break;
            }
            F->settle_win = std::max(16, (int) v[0]);
            F->settle_eps = std::max(0.0, v[1]) / 100.0;
            F->settle_drop = std::max(0.0, v[2]) / 100.0;
            std::fprintf(stderr, "glm fast: CUDA%d CPU lane: share %.2f once the VRAM hit rate settles (windows of %d "
                                 "token steps, gain < %.1f points; back when it falls %.1f points) "
                                 "(STRATA_GLM_PCIE_SHARE_LATE)\n", dev_, ls, F->settle_win, F->settle_eps * 100.0,
                         F->settle_drop * 100.0);
        } else {
            F->late_after = colon != nullptr ? std::max(1, std::atoi(colon + 1)) : 256;
            F->late_done = false;
            std::fprintf(stderr, "glm fast: CUDA%d CPU lane: share %.2f after %d token steps of each request "
                                 "(STRATA_GLM_PCIE_SHARE_LATE)\n", dev_, ls, F->late_after);
        }
    }
    // THE LEAN (issue #23): the split above hides the CPU lane behind the PCIe lane, and so keeps one expert of a route
    // on the link whenever the CPU's whole route would take longer than that one copy.  On a box whose CPU does an
    // expert much faster than the link, that one copy is dearer than it looks: both lanes read the same DRAM, and the
    // copy's zero-copy kernel sits on the compute stream before the layer's experts (an RTX A2000 8 GB on Gen4 x8,
    // Ryzen 9 7940HX: 0.055 ms on the CPU vs 0.282 over PCIe; the split kept 7 of 8 on the CPU at 16.8 tok/s, every
    // one of them 18.5 - and an H100 reported 0.099 vs 0.448 the same way).  When the CPU is STRATA_GLM_LANE_LEAN (4)
    // times faster, every RAM-tier expert goes to the CPU - once the pool has settled (the ":settle" switch below,
    // issue #14: from the start, the lane's share alone would never refill the pool after a prompt), and back when
    // the hit rate falls.  0 keeps the split; STRATA_GLM_PCIE_SHARE / _LATE / CPU_PLAN set by hand win.  A card
    // whose split already sends every RAM-tier expert to the CPU (a Gen4 x4 / OCuLink link) is as before.
    static const double lean = [] {
        const char* v = getenv("STRATA_GLM_LANE_LEAN");
        return v != nullptr ? std::max(0.0, std::atof(v)) : 4.0;
    }();
    if (lean > 0.0 && pv == nullptr && sv == nullptr && getenv("STRATA_GLM_PCIE_SHARE_LATE") == nullptr &&
        c_ms * lean <= pd_ms) {
        unsigned long long all_cpu = 0ull;
        for (int f = 0; f <= plan_f(g); ++f) all_cpu |= (unsigned long long) f << (4 * f);
        if (plan != all_cpu) {
            F->cpu_plan_late = all_cpu;
            F->settle = true;
            std::fprintf(stderr, "glm fast: CUDA%d CPU lane: the CPU does an expert %.1fx faster than the link - every "
                                 "RAM-tier expert goes to the CPU once the pool settles (windows of %d token steps), the "
                                 "split below until then and when the VRAM hits fall (STRATA_GLM_LANE_LEAN=0 keeps the "
                                 "split)\n", dev_, pd_ms / std::max(1e-6, c_ms), F->settle_win);
        }
    }
    F->md.cpu_seq = F->cpu_seq_d;
    F->md.cpu_ans = dp;
    // the route counts that pick the coldest RAM-tier experts for the host, seeded with the tiers' LFU counts
    // (STRATA_GLM_CPU_COLD=0: the last ones in route order instead)
    const char* cv = getenv("STRATA_GLM_CPU_COLD");
    if ((cv == nullptr || std::atoi(cv) != 0) && F->cnt.size() == (size_t) F->n_keys) {
        if (cudaMalloc((void**) &F->dcnt_d, (size_t) F->n_keys * sizeof(unsigned int)) != cudaSuccess) {
            err = "glm fast: the CPU lane's route counts did not allocate";
            return false;
        }
        cudaMemcpy(F->dcnt_d, F->cnt.data(), (size_t) F->n_keys * sizeof(unsigned int), cudaMemcpyHostToDevice);
        F->md.dcnt = F->dcnt_d;
    }
    std::fprintf(stderr, "glm fast: CUDA%d CPU lane: %d threads%s, an expert %.3f ms on the CPU vs %.3f ms over PCIe -> "
                         "of 0..8 RAM-tier experts the CPU takes%s\n", dev_, threads,
                 shared ? " (one pool, shared by the split's GPUs)"
                 : pin_cpus.empty() ? ""
                 : (" on " + std::to_string(pin_cpus.size()) + " CPUs of its own" +
                    (pin_node >= 0 ? " (NUMA node " + std::to_string(pin_node) + ")" : std::string())).c_str(),
                 c_ms, pd_ms, tab.c_str());
    // STRATA_GLM_CPU_LANE_CHECK=1: the calibration expert on one normalised input three ways - the device's decode
    // kernels, the CPU lane, and a double-precision reference from ggml's dequantised rows - and their distances
    if (getenv("STRATA_GLM_CPU_LANE_CHECK") != nullptr && F->L[(size_t) il_cal].exl3) {
        // EXL3: the device's kernels, the CPU lane, and the scalar reference (exl3_ref, double sums)
        namespace x3 = strata::kernels::exl3;
        const int E = g.n_embd, FFE = g.n_ff_exp;
        const auto& Ly = F->L[(size_t) il_cal];
        std::vector<float> xc((size_t) E), og((size_t) E), oc((size_t) E);
        double ss = 0;
        for (int i = 0; i < E; ++i) {
            xc[(size_t) i] = std::sin(0.71f * i) + ((i % 97) == 0 ? 6.0f : 0.0f);
            ss += (double) xc[(size_t) i] * xc[(size_t) i];
        }
        for (auto& v : xc) v = (float) (v / std::sqrt(ss / E));
        float *dx = nullptr, *dout = nullptr;
        cudaMalloc(&dx, (size_t) E * 4);
        cudaMalloc(&dout, (size_t) E * 4);
        cudaMemcpy(dx, xc.data(), (size_t) E * 4, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(F->scratch, cal, blob, cudaMemcpyHostToDevice, F->cs);
        const unsigned long long sp = (unsigned long long) F->scratch;
        const float one = 1.0f;
        cudaMemcpyAsync(F->md.plan_ptr, &sp, 8, cudaMemcpyHostToDevice, F->cs);
        cudaMemcpyAsync(F->md.plan_w, &one, 4, cudaMemcpyHostToDevice, F->cs);
        cudaMemsetAsync(F->md.cpu_flag, 0, 4, F->cs);
        x3::MoeArgs xa;
        xa.plan_ptr = F->md.plan_ptr;
        xa.plan_w = F->md.plan_w;
        xa.cpu_part = F->md.cpu_part;
        xa.cpu_flag = F->md.cpu_flag;
        xa.lay = Ly.xl;
        xa.k = 1;
        xa.n_embd = E;
        xa.n_ff = FFE;
        xa.limit = g.swiglu_exp;
        x3::moe_gate_up(xa, dx, nullptr, F->cs);
        x3::moe_down(xa, nullptr, nullptr, dout, F->cs);
        cudaMemcpyAsync(og.data(), dout, (size_t) E * 4, cudaMemcpyDeviceToHost, F->cs);
        cudaStreamSynchronize(F->cs);
        fast_cpu_experts(il_cal, 1, &cal, &one, xc.data(), oc.data());
        // the reference
        const auto ref_mv = [&](int r, const float* in, float* o) {
            const int k = r < 2 ? E : FFE, n = r < 2 ? FFE : E;
            std::vector<float> wt((size_t) k * n);
            kc::exl3_inner((const uint32_t*) (cal + Ly.xl.trellis[r]), n / 16, k, n, Ly.xl.K[r], Ly.xl.cb, wt.data());
            kc::exl3_mv(wt.data(), (const uint16_t*) (cal + Ly.xl.suh[r]), (const uint16_t*) (cal + Ly.xl.svh[r]), k, n,
                        in, o);
        };
        std::vector<float> gr((size_t) FFE), ur((size_t) FFE), hr((size_t) FFE), orf((size_t) E);
        ref_mv(0, xc.data(), gr.data());
        ref_mv(1, xc.data(), ur.data());
        const float lim = g.swiglu_exp;
        for (int r = 0; r < FFE; ++r) {
            const float gs = std::min(gr[(size_t) r], lim), us = std::min(std::max(ur[(size_t) r], -lim), lim);
            hr[(size_t) r] = gs / (1.0f + std::exp(-gs)) * us;
        }
        ref_mv(2, hr.data(), orf.data());
        double nr = 0, eg = 0, ec = 0, egc = 0;
        for (int r = 0; r < E; ++r) {
            nr += (double) orf[(size_t) r] * orf[(size_t) r];
            eg += (double) (og[(size_t) r] - orf[(size_t) r]) * (og[(size_t) r] - orf[(size_t) r]);
            ec += (double) (oc[(size_t) r] - orf[(size_t) r]) * (oc[(size_t) r] - orf[(size_t) r]);
            egc += (double) (og[(size_t) r] - oc[(size_t) r]) * (og[(size_t) r] - oc[(size_t) r]);
        }
        std::fprintf(stderr, "glm fast: CPU lane check (layer %d, EXL3 K %d/%d/%d): relative L2 error vs the reference: "
                             "device %.5f, CPU lane %.5f; device vs CPU lane %.5f\n", il_cal, Ly.xl.K[0], Ly.xl.K[1],
                     Ly.xl.K[2], std::sqrt(eg / nr), std::sqrt(ec / nr), std::sqrt(egc / nr));
        cudaFree(dx);
        cudaFree(dout);
    } else if (getenv("STRATA_GLM_CPU_LANE_CHECK") != nullptr) {
        const int E = g.n_embd, FFE = g.n_ff_exp;
        const auto& Ly = F->L[(size_t) il_cal];
        const auto& nf = F->cpu_fmt[(size_t) il_cal];
        float *dh = nullptr, *dw = nullptr, *dx = nullptr, *dout = nullptr;
        void *dxq = nullptr, *dhq = nullptr;
        cudaMalloc(&dh, (size_t) E * 4);
        cudaMalloc(&dw, (size_t) E * 4);
        cudaMalloc(&dx, (size_t) E * 4);
        cudaMalloc(&dout, (size_t) E * 4);
        cudaMalloc(&dxq, (size_t) E / 32 * 36);
        cudaMalloc(&dhq, (size_t) 8 * FFE / 32 * 36);
        std::vector<float> h((size_t) E), ones((size_t) E, 1.0f), xc((size_t) E), og((size_t) E), oc((size_t) E);
        for (int i = 0; i < E; ++i) h[(size_t) i] = std::sin(0.71f * i) + ((i % 97) == 0 ? 6.0f : 0.0f);
        cudaMemcpy(dh, h.data(), (size_t) E * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, ones.data(), (size_t) E * 4, cudaMemcpyHostToDevice);
        gf::rms_q8(dh, nullptr, dw, g.norm_eps, E, dx, dxq, F->cs);
        cudaMemcpyAsync(F->scratch, cal, blob, cudaMemcpyHostToDevice, F->cs);
        const unsigned long long sp = (unsigned long long) F->scratch;
        const float one = 1.0f;
        cudaMemcpyAsync(F->md.plan_ptr, &sp, 8, cudaMemcpyHostToDevice, F->cs);
        cudaMemcpyAsync(F->md.plan_w, &one, 4, cudaMemcpyHostToDevice, F->cs);
        cudaMemsetAsync(F->md.cpu_flag, 0, 4, F->cs);
        gf::moe_gate_up(Ly.gu_type, F->md, 1, E, FFE, g.swiglu_exp, dxq, dhq, nullptr, 0, nullptr, 0, nullptr, F->cs);
        gf::moe_down(Ly.d_type, F->md, 1, E, FFE, Ly.down_off, dhq, nullptr, dout, F->cs);
        cudaMemcpyAsync(xc.data(), dx, (size_t) E * 4, cudaMemcpyDeviceToHost, F->cs);
        cudaMemcpyAsync(og.data(), dout, (size_t) E * 4, cudaMemcpyDeviceToHost, F->cs);
        cudaStreamSynchronize(F->cs);
        fast_cpu_experts(il_cal, 1, &cal, &one, xc.data(), oc.data());
        // the reference: rows dequantised by ggml, dots in double
        const ggml_type_traits* tg = ggml_get_type_traits((ggml_type) Ly.gu_type);
        const ggml_type_traits* td = ggml_get_type_traits((ggml_type) Ly.d_type);
        std::vector<float> row((size_t) std::max(E, FFE));
        std::vector<double> hr((size_t) FFE), orf((size_t) E);
        for (int r = 0; r < FFE; ++r) {
            double gs = 0.0, us = 0.0;
            tg->to_float(cal + (size_t) r * nf.gu_row, row.data(), E);
            for (int i = 0; i < E; ++i) gs += (double) row[(size_t) i] * xc[(size_t) i];
            tg->to_float(cal + Ly.gu_bytes + (size_t) r * nf.gu_row, row.data(), E);
            for (int i = 0; i < E; ++i) us += (double) row[(size_t) i] * xc[(size_t) i];
            const double lim = g.swiglu_exp;
            gs = std::min(gs, lim);
            us = std::min(std::max(us, -lim), lim);
            hr[(size_t) r] = gs / (1.0 + std::exp(-gs)) * us;
        }
        for (int r = 0; r < E; ++r) {
            double s = 0.0;
            td->to_float(cal + Ly.down_off + (size_t) r * nf.d_row, row.data(), FFE);
            for (int j = 0; j < FFE; ++j) s += (double) row[(size_t) j] * hr[(size_t) j];
            orf[(size_t) r] = s;
        }
        double nr = 0, eg = 0, ec = 0, egc = 0;
        for (int r = 0; r < E; ++r) {
            nr += orf[(size_t) r] * orf[(size_t) r];
            eg += (og[(size_t) r] - orf[(size_t) r]) * (og[(size_t) r] - orf[(size_t) r]);
            ec += (oc[(size_t) r] - orf[(size_t) r]) * (oc[(size_t) r] - orf[(size_t) r]);
            egc += (double) (og[(size_t) r] - oc[(size_t) r]) * (og[(size_t) r] - oc[(size_t) r]);
        }
        std::fprintf(stderr, "glm fast: CPU lane check (layer %d, types %d/%d): relative L2 error vs the double "
                             "reference: device %.5f, CPU lane %.5f; device vs CPU lane %.5f\n", il_cal, Ly.gu_type,
                     Ly.d_type, std::sqrt(eg / nr), std::sqrt(ec / nr), std::sqrt(egc / nr));
        cudaFree(dh);
        cudaFree(dw);
        cudaFree(dx);
        cudaFree(dout);
        cudaFree(dxq);
        cudaFree(dhq);
    }
    return true;
}

// ---------------------------------------------------------------- the usage profile
// Where this machine's expert usage is kept between sessions: STRATA_GLM_USAGE=<file> (0: none), else
// expert_usage.txt in the pack's folder.
std::string Glm5Model::usage_path() const {
    const char* u = getenv("STRATA_GLM_USAGE");
    if (u != nullptr) return std::string(u) == "0" ? std::string() : std::string(u);
    return pack_dir_.empty() ? std::string() : pack_dir_ + "/expert_usage.txt";
}

// Every half's long-memory routes ("layer e:count ..."), one file, written whole and renamed into place.
bool Glm5Model::save_usage() {
    const std::string up = usage_path();
    if (up.empty() || fast_ == nullptr) return false;
    std::string out = "# strata expert usage: routes per expert across sessions (the warm-up fills the tiers in this "
                      "order)\n";
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        FastState* F = m->fast_;
        if (F == nullptr || F->usage.empty()) continue;
        std::vector<uint32_t> u;
        {
            std::lock_guard<std::mutex> lk(F->mu);
            u = F->usage;
        }
        const int NE = m->g_.n_expert;
        for (int il = m->l0_; il < m->lt_; ++il) {
            std::string line = std::to_string(il);
            bool any = false;
            for (int e = 0; e < NE; ++e) {
                const uint32_t n = u[(size_t) il * NE + e];
                if (n == 0) continue;
                line += " " + std::to_string(e) + ":" + std::to_string(n);
                any = true;
            }
            if (any) out += line + "\n";
        }
    }
    const std::string tmp = up + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) return false;
        f << out;
        if (!f) return false;
    }
    return std::rename(tmp.c_str(), up.c_str()) == 0;
}

// ---------------------------------------------------------------- load-time warm-up
// VRAM and the pinned RAM tier together hold (nearly) every routed expert, so the whole model is streamed from the
// shards ONCE at load: each layer's most frequently routed experts (the pack's expert_prior.txt, written by
// tools/glm_expert_prior.py from routing traces; id order without it) fill its VRAM slots - all but the spares - and
// the rest fill the RAM tier (a few slots per class stay free for disk reads).  The first request then runs warm and
// the disk is touched again only for experts that fit nowhere.  STRATA_GLM_WARM=0 skips it (the tiers fill on demand).
bool Glm5Model::fast_warm(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const auto t0 = std::chrono::steady_clock::now();
    const int NL = g.n_layers + 1;
    std::vector<std::vector<int>> order((size_t) NL);
    {
        std::ifstream pf(pack_dir_ + "/expert_prior.txt");
        std::string line;
        while (std::getline(pf, line)) {
            std::istringstream ss(line);
            int il = -1, e = 0;
            ss >> il;
            if (il < 0 || il >= NL) continue;
            while (ss >> e)
                if (e >= 0 && e < g.n_expert) order[(size_t) il].push_back(e);
        }
    }
    // THIS user's experts first: the usage profile of earlier sessions (save_usage) leads each layer's order, its
    // counts become the long memory again, and the LFU counts start from them scaled to at most 32 (enough to keep
    // the profile's experts against one-off routes, little enough that a new task takes over within ~100 tokens;
    // a restart then hits ~75% instead of ~67% over the first 50 tokens on one V100, simulated)
    // the routing profile's counts per expert id (expert_counts.txt), for the blend below
    std::vector<std::vector<double>> pcount((size_t) NL);
    {
        std::ifstream cf(expert_counts_path(pack_dir_));
        std::string line;
        while (std::getline(cf, line)) {
            std::istringstream ss(line);
            int il = -1;
            double c = 0;
            ss >> il;
            if (il < 0 || il >= NL) continue;
            while (ss >> c) pcount[(size_t) il].push_back(c);
        }
    }
    // no expert_prior.txt (a pack with only the counts, or the shipped profile STRATA_GLM_PRIOR points at): each
    // layer's order from the counts, the way tools/glm_expert_prior.py writes it (most routed first, ties by id)
    {
        bool any = false;
        for (const auto& o : order) any = any || !o.empty();
        int n_cnt = 0;
        for (int il = 0; il < NL && !any; ++il) {
            const auto& c = pcount[(size_t) il];
            if (c.empty()) continue;
            std::vector<int> o((size_t) g.n_expert);
            std::iota(o.begin(), o.end(), 0);
            std::stable_sort(o.begin(), o.end(), [&](int a, int b) {
                const double ca = (size_t) a < c.size() ? c[(size_t) a] : 0.0, cb = (size_t) b < c.size() ? c[(size_t) b] : 0.0;
                return ca > cb;
            });
            order[(size_t) il].swap(o);
            ++n_cnt;
        }
        if (n_cnt > 0)
            std::fprintf(stderr, "glm fast: CUDA%d the warm-up follows the routing profile %s (%d layers)\n", dev_,
                         expert_counts_path(pack_dir_).c_str(), n_cnt);
    }
    static const double profile_w = [] {
        const char* v = getenv("STRATA_GLM_PROFILE_WEIGHT");
        return v != nullptr ? std::max(0.0, std::atof(v)) : 1.0;
    }();
    int n_blend = 0;
    const std::string up = usage_path();
    int n_prof = 0;
    std::vector<char> useeded((size_t) NL, 0);   // layers whose LFU counts the usage profile seeded
    if (!up.empty()) {
        std::ifstream uf(up);
        std::string line;
        while (std::getline(uf, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            int il = -1;
            ss >> il;
            if (il < l0_ || il >= lt_ || !F->L[(size_t) il].moe) continue;
            std::vector<std::pair<uint32_t, int>> v;
            std::string tok;
            while (ss >> tok) {
                const size_t c = tok.find(':');
                if (c == std::string::npos) continue;
                const int e = std::atoi(tok.substr(0, c).c_str());
                const uint32_t n = (uint32_t) std::strtoul(tok.substr(c + 1).c_str(), nullptr, 10);
                if (e >= 0 && e < g.n_expert && n > 0) v.emplace_back(n, e);
            }
            if (v.empty()) continue;
            std::sort(v.rbegin(), v.rend());
            std::vector<int> o;
            for (auto& p : v) {
                o.push_back(p.second);
                const size_t key = (size_t) il * g.n_expert + p.second;
                F->usage[key] = p.first;
                F->cnt[key] = (uint32_t) std::max<uint64_t>(1, (uint64_t) 32 * p.first / v.front().first);
            }
            // with the pack's routing profile (expert_counts.txt, tools/glm_expert_prior.py over a broad corpus): the
            // order is the blend of the two shares - this machine's history, and the profile, which corrects what the
            // history over-represents (one task, a benchmark) and ranks what it never routed
            if (!pcount[(size_t) il].empty() && profile_w > 0.0) {
                double ut = 0.0, pt = 0.0;
                for (auto& p : v) ut += p.first;
                for (double c : pcount[(size_t) il]) pt += c;
                if (ut > 0.0 && pt > 0.0) {
                    std::vector<double> score((size_t) g.n_expert, 0.0);
                    for (auto& p : v) score[(size_t) p.second] += p.first / ut;
                    for (size_t e = 0; e < pcount[(size_t) il].size() && e < score.size(); ++e)
                        score[e] += profile_w * pcount[(size_t) il][e] / pt;
                    o.resize((size_t) g.n_expert);
                    std::iota(o.begin(), o.end(), 0);
                    std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return score[(size_t) a] > score[(size_t) b]; });
                    ++n_blend;
                }
            }
            o.insert(o.end(), order[(size_t) il].begin(), order[(size_t) il].end());
            order[(size_t) il].swap(o);
            useeded[(size_t) il] = 1;
            ++n_prof;
        }
        if (n_prof > 0)
            std::fprintf(stderr, "glm fast: CUDA%d the tiers follow this machine's usage profile (%s, %d layers%s)\n", dev_,
                         up.c_str(), n_prof, n_blend > 0 ? ", blended with the pack's routing profile" : "");
    }
    // Without a usage profile every LFU count started at 0, and the eviction rules then broke every tie on slot
    // order and recency: the boundary's VRAM victim was DROPPED to disk instead of demoted (rc_min < bc never held
    // with both at 0), and the RAM victim was whichever landed first.  DeepSeek V4.1 (nazeshinjite Q2, seq0): 1,255
    // VRAM drops in 512 tokens, 31% of the disk reads in the first 64 tokens.  The pack's routing profile seeds the
    // counts the way the usage profile does (at most 32 a layer, relative to the layer's hottest expert).
    // STRATA_GLM_SEED_PRIOR=1 - measured WORSE on V4.1 (seq0 512: 53.6 -> 61.6 ms/token, RAM re-reads 450 -> 774):
    // the seeded counts protect the profile's experts against the text's own, as a session usage file did.  Off.
    static const bool seed_prior = [] {
        const char* v = getenv("STRATA_GLM_SEED_PRIOR");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (seed_prior) {
        int n_seed = 0;
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe || useeded[(size_t) il] || pcount[(size_t) il].empty()) continue;
            double mx = 0.0;
            for (double c : pcount[(size_t) il]) mx = std::max(mx, c);
            if (mx <= 0.0) continue;
            for (size_t e = 0; e < pcount[(size_t) il].size() && e < (size_t) g.n_expert; ++e)
                if (pcount[(size_t) il][e] > 0.0)
                    F->cnt[(size_t) il * g.n_expert + e] =
                        (uint32_t) std::max(1.0, std::floor(32.0 * pcount[(size_t) il][e] / mx));
            ++n_seed;
        }
        if (n_seed > 0)
            std::fprintf(stderr, "glm fast: CUDA%d LFU counts seeded from the pack's routing profile (%d layers)\n", dev_,
                         n_seed);
    }
    for (int il = 0; il < NL; ++il) {
        auto& o = order[(size_t) il];
        std::vector<char> seen((size_t) g.n_expert, 0);
        std::vector<int> clean;
        for (int e : o)
            if (!seen[(size_t) e]) {
                seen[(size_t) e] = 1;
                clean.push_back(e);
            }
        for (int e = 0; e < g.n_expert; ++e)
            if (!seen[(size_t) e]) clean.push_back(e);
        o.swap(clean);
    }
    struct Job {
        int il, e;
        uint8_t* dst;    // the RAM slot, or the VRAM slot (staged)
        bool vram;
    };
    std::vector<Job> jobs;
    const size_t N = (size_t) F->n_keys;
    std::vector<unsigned long long> th(2 * N + (size_t) NL * gf::kSpares, 0ull);
    std::vector<int> next((size_t) NL, 0);
    for (int il = l0_; il < lt_; ++il) {
        if (!F->L[(size_t) il].moe) continue;
        auto& P = F->lp[(size_t) il];
        const int nv = glmfast::warm_pool_slots(F->unified_memory, P.n, g.n_expert, gf::kSpares);
        for (int j = 0; j < nv; ++j) {
            const int e = order[(size_t) il][(size_t) j];
            const int key = il * g.n_expert + e;
            P.key[(size_t) j] = e;
            P.st[(size_t) j] = FastState::kResident;
            F->slot_of[(size_t) key] = j;
            th[(size_t) key] = (unsigned long long) P.slot_ptr(j);
            jobs.push_back(Job{il, e, P.slot_ptr(j), true});
        }
        next[(size_t) il] = nv;
    }
    // the RAM tier, round-robin over the layers so every layer gets its share of a class that cannot hold all
    std::vector<int> rfree(F->rc.size(), 0), rcur(F->rc.size(), 0);
    for (size_t c = 0; c < F->rc.size(); ++c) rfree[c] = F->rc[c].n;
    for (bool progress = true; progress;) {
        progress = false;
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe || next[(size_t) il] >= g.n_expert) continue;
            const int c = F->layer_rc[(size_t) il];
            auto& R = F->rc[(size_t) c];
            if (rfree[(size_t) c] <= (F->ram_resident ? 0 : 4)) continue;   // resident: no disk-read slots to keep
            while (rcur[(size_t) c] < R.n && R.st[(size_t) rcur[(size_t) c]] != FastState::kRFree) ++rcur[(size_t) c];
            if (rcur[(size_t) c] >= R.n) continue;
            const int s = rcur[(size_t) c]++;
            const int e = order[(size_t) il][(size_t) next[(size_t) il]++];
            const int key = il * g.n_expert + e;
            R.key[(size_t) s] = key;
            R.st[(size_t) s] = FastState::kRHold;
            F->ram_of[(size_t) key] = s;
            th[N + (size_t) key] = (unsigned long long) (R.base + (size_t) s * R.stride);
            jobs.push_back(Job{il, e, R.base + (size_t) s * R.stride, false});
            --rfree[(size_t) c];
            progress = true;
        }
    }
    // EXL3 experts in file order: the warm-up reads whole experts (fast_read_expert_x3), and in the order the
    // routes rank them the reads seek all over the model's files (a cold read from a disk at ~0.05 GB/s)
    if (exl3_)
        std::stable_sort(jobs.begin(), jobs.end(), [&](const Job& a, const Job& b) {
            const auto& na = pack_layers_[(size_t) a.il];
            const auto& nb2 = pack_layers_[(size_t) b.il];
            if (na.x_cb < 0 || nb2.x_cb < 0) return false;
            const size_t ia = (size_t) a.e * 9, ib = (size_t) b.e * 9;
            return std::make_pair(na.x_shard[ia], na.x_off[ia]) < std::make_pair(nb2.x_shard[ib], nb2.x_off[ib]);
        });
    // stream them: 8 experts per batch, 3 slices each in parallel; VRAM ones through the pinned staging
    size_t done_bytes = 0, total_bytes = 0;
    for (const auto& j : jobs) total_bytes += F->L[(size_t) j.il].blob;
    int last_pct = -1;
    for (size_t b = 0; b < jobs.size(); b += 8) {
        const int nb = (int) std::min<size_t>(8, jobs.size() - b);
        F->workers->run(nb * 3, [&](int job) {
            const Job& J = jobs[b + (size_t) (job / 3)];
            const int role = job % 3;
            const auto& Ly = F->L[(size_t) J.il];
            const auto& nl = pack_layers_[(size_t) J.il];
            if (nl.x_cb >= 0) {   // EXL3: the expert's nine pieces in as few reads as they allow (the first job of 3)
                if (role == 0) fast_read_expert_x3(J.il, J.e, J.vram ? F->stage[(size_t) (job / 3)] : J.dst);
                return;
            }
            const Shard& sh = pack_shards_[(size_t) (role == 0 ? nl.gate_shard : role == 1 ? nl.up_shard : nl.down_shard)];
            const uint64_t off = role == 0   ? nl.gate_off + (uint64_t) J.e * Ly.gu_bytes
                                 : role == 1 ? nl.up_off + (uint64_t) J.e * Ly.gu_bytes
                                             : nl.down_off + (uint64_t) J.e * Ly.dn_bytes;
            const size_t len = role == 2 ? Ly.dn_bytes : Ly.gu_bytes;
            uint8_t* base = J.vram ? F->stage[(size_t) (job / 3)] : J.dst;
            uint8_t* d = base + (role == 0 ? 0 : role == 1 ? Ly.gu_bytes : 2 * Ly.gu_bytes);
            read_slice(sh, off, len, d);
        });
        for (int k = 0; k < nb; ++k) {
            const Job& J = jobs[b + (size_t) k];
            if (J.vram)
                cudaMemcpyAsync(J.dst, F->stage[(size_t) k], F->L[(size_t) J.il].blob, cudaMemcpyHostToDevice, F->copy);
            done_bytes += F->L[(size_t) J.il].blob;
        }
        cudaStreamSynchronize(F->copy);
        const int pct = (int) (100.0 * (double) done_bytes / (double) std::max<size_t>(1, total_bytes));
        if (pct / 10 != last_pct / 10) {
            last_pct = pct;
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "glm fast: CUDA%d warming the expert tiers %d%% (%.1f of %.1f GB, %.0f s)\n", dev_, pct,
                         (double) done_bytes / 1e9, (double) total_bytes / 1e9, s);
        }
    }
    if (cudaMemcpy(F->tab, th.data(), th.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "glm fast: the warmed expert tables did not upload";
        return false;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "glm fast: CUDA%d tiers warm: %zu experts (%.1f GB) in %.0f s (%.2f GB/s)\n", dev_, jobs.size(),
                 (double) total_bytes / 1e9, s, (double) total_bytes / 1e9 / std::max(1e-9, s));
    return true;
}

void Glm5Model::fast_destroy() {
    FastState* F = fast_;
    if (F == nullptr) return;
    cudaSetDevice(dev_);
    dspark_report();
    dspark_destroy();
    prefill_destroy();
    F->quit.store(true);
    if (F->svc.joinable()) F->svc.join();
    if (F->expert_helper) {
        F->expert_helper->report("destroy");
        F->expert_helper.reset();
    }
    if (const uint64_t mc = F->cpu_multi_calls.load(); mc > 0)
        std::fprintf(stderr, "glm fast CPU lane CUDA%d: %llu verify-window calls of %.2f columns each, %.2f experts read "
                             "for %.2f uses\n", dev_, (unsigned long long) mc,
                     (double) F->cpu_multi_cols.load() / (double) mc, (double) F->cpu_multi_experts.load() / (double) mc,
                     (double) F->cpu_multi_uses.load() / (double) mc);
    {
        std::lock_guard<std::mutex> lk(F->ah_mu);
        F->ah_quit = true;
    }
    F->ah_cv.notify_all();
    for (auto& t : F->ah_th) t.join();
    F->workers.reset();
    F->cpu_pool.reset();
    if (F->cpu_ans_h) cudaFreeHost(F->cpu_ans_h);
    if (F->cpu_seq_d) cudaFree(F->cpu_seq_d);
    if (F->dcnt_d) cudaFree(F->dcnt_d);
    if (F->timing && F->ah_routes > 0) {
        std::string ov;
        for (int d = 0; d < F->n_ahead; ++d)
            ov += " " + std::to_string(d + 1) + ":" +
                  std::to_string((double) F->ah_ov[d] / (double) std::max<uint64_t>(1, F->ah_npred[d])).substr(0, 4);
        std::string cov;
        for (int d = 0; d < F->n_ahead; ++d)
            cov += " " + std::to_string(d + 1) + ":" +
                   std::to_string(100.0 * (double) F->ah_disk_cov[d] / (double) std::max<uint64_t>(1, F->ah_disk))
                       .substr(0, 4) + "%";
        std::fprintf(stderr,
                     "glm fast ahead CUDA%d: of 8 experts predicted d layers early%s | disk misses %llu, named d early%s, "
                     "by any %.1f%% | reads issued %llu, used %llu (waited %llu, %.1f ms; taken over %llu), queue full %llu, "
                     "no free slot %llu\n",
                     dev_, ov.c_str(), (unsigned long long) F->ah_disk, cov.c_str(),
                     100.0 * (double) F->ah_disk_any / (double) std::max<uint64_t>(1, F->ah_disk),
                     (unsigned long long) F->ah_issued, (unsigned long long) F->ah_used,
                     (unsigned long long) F->ah_waited, (double) F->ah_wait_us.load() / 1000.0,
                     (unsigned long long) F->ah_stolen, (unsigned long long) F->ah_full,
                     (unsigned long long) F->ah_nofree);
    }
    if (F->timing && F->tokens > 0)
        std::fprintf(stderr, "glm fast timing: %llu tokens, %.2f ms/token\n", (unsigned long long) F->tokens,
                     F->ms / (double) F->tokens);
    if (F->timing && (F->tail_skips.load() > 0 || F->fill_issued > 0))
        std::fprintf(stderr, "glm fast tail skip CUDA%d: %llu experts left out | RAM fill: %llu read into the RAM tier "
                             "(%llu over a colder one), %llu passed (no room)\n", dev_,
                     (unsigned long long) F->tail_skips.load(), (unsigned long long) F->fill_issued,
                     (unsigned long long) F->fill_evict, (unsigned long long) F->fill_passed);
    if (F->timing && F->hits.load() + F->misses.load() > 0)
        std::fprintf(stderr,
                     "glm fast tiers CUDA%d: vram hits %llu, ram fetches %llu, disk reads %llu (%.2f%% vram hit) | "
                     "promotions %llu (scratch %llu) demotions %llu drops %llu prefetches %llu | disk waits %llu, %.1f ms\n",
                     dev_, (unsigned long long) F->hits.load(), (unsigned long long) F->ram_hits.load(),
                     (unsigned long long) F->disk_reads.load(),
                     100.0 * (double) F->hits.load() / (double) std::max<uint64_t>(1, F->hits.load() + F->misses.load()),
                     (unsigned long long) F->promotions.load(), (unsigned long long) F->scratch_uses.load(),
                     (unsigned long long) F->demotions.load(), (unsigned long long) F->drops.load(),
                     (unsigned long long) F->prefetches.load(), (unsigned long long) F->miss_layers.load(),
                     (double) F->miss_us.load() / 1000.0);
        if (g_.v41) dsv41_engram_report();
        if (const uint64_t m0 = strata::core::shard_io::mirror_bytes[0].load(), m1 = strata::core::shard_io::mirror_bytes[1].load(); m1 > 0)
            std::fprintf(stderr, "glm fast mirror CUDA%d: %.2f GB from the shards' drive, %.2f GB from the mirror "
                                 "(%.1f%%)%s\n", dev_, (double) m0 / 1e9, (double) m1 / 1e9,
                         100.0 * (double) m1 / (double) (m0 + m1),
                         strata::core::shard_io::mirror_on() ? "" : " - the mirror was turned off by a read error");
        if (const uint64_t fb = strata::core::shard_io::fallback_buffered.load(), fm = strata::core::shard_io::fallback_mapped.load(); fb > 0)
            std::fprintf(stderr, "glm fast CUDA%d: %llu expert reads left the O_DIRECT path (buffered pread; %llu of them "
                                 "copied from the mapping) - an SSD or filesystem that refuses O_DIRECT reads, or short reads\n",
                         dev_, (unsigned long long) fb, (unsigned long long) fm);
        if (F->disk_lane)
            std::fprintf(stderr, "glm fast disk lane CUDA%d: %llu disk experts computed in RAM, %.1f ms; %llu evicted at "
                                 "the miss\n", dev_, (unsigned long long) F->dlane_experts.load(),
                         (double) F->dlane_us.load() / 1000.0, (unsigned long long) F->dlane_evict);
    if (F->timing && F->pred_n.load() > 0)
        std::fprintf(stderr, "glm fast predict CUDA%d: %.2f of 8 experts predicted one layer early; %.1f%% of the "
                             "non-resident ones (%llu)\n",
                     dev_, (double) F->pred_overlap.load() / (double) F->pred_n.load(),
                     100.0 * (double) F->pred_miss_hit.load() / (double) std::max<uint64_t>(1, F->pred_miss.load()),
                     (unsigned long long) F->pred_miss.load());
    if (F->prof_on && F->ptokens > 0) {
        std::vector<std::pair<double, std::string>> v;
        double tot = 0;
        for (auto& kv : F->pacc) {
            v.push_back({kv.second, kv.first});
            tot += kv.second;
        }
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "glm fast prof CUDA%d (%llu tokens, %.2f ms/token of GPU stream time):\n", dev_,
                     (unsigned long long) F->ptokens, tot / (double) F->ptokens);
        for (auto& e : v)
            std::fprintf(stderr, "  %-18s %7.3f ms/token  %5.1f%%\n", e.second.c_str(), e.first / (double) F->ptokens,
                         100.0 * e.first / tot);
    }
    for (auto e : F->pev) cudaEventDestroy(e);
    if (dsv4_snap_) {
        cudaFree(dsv4_snap_);
        dsv4_snap_ = nullptr;
    }
    if (q35_snap_h_) {
        cudaFreeHost(q35_snap_h_);
        q35_snap_h_ = nullptr;
    }
    if (dsv4_state_) {
        cudaFree(dsv4_state_);
        dsv4_state_ = nullptr;
    }
    if (snap_pool_) {
        cudaFree(snap_pool_);
        snap_pool_ = nullptr;
    }
    if (snap_) {
        cudaFree(snap_);
        snap_ = nullptr;
    }
    if (kda_bak_) {
        cudaFree(kda_bak_);
        kda_bak_ = nullptr;
    }
    for (int i = 0; i < 2; ++i) {
        if (spec_hop_h_[i]) cudaFreeHost(spec_hop_h_[i]);
        if (spec_ev_hop_[i]) cudaEventDestroy(spec_ev_hop_[i]);
        spec_hop_h_[i] = nullptr;
        spec_ev_hop_[i] = nullptr;
    }
    if (spec_steps_ > 0)
        std::fprintf(stderr, "glm spec: %llu speculative positions, %.1f%% accepted\n", (unsigned long long) spec_steps_,
                     100.0 * (double) spec_hits_ / (double) spec_steps_);
    if (F->cs) cudaStreamSynchronize(F->cs);
    if (F->copy) cudaStreamSynchronize(F->copy);
    if (F->kda_graphs.captures || F->kda_graphs.replays)
        std::fprintf(stderr, "glm KDA graphs: %llu captures, %llu replays, %llu invalidations\n",
                     (unsigned long long) F->kda_graphs.captures,
                     (unsigned long long) F->kda_graphs.replays,
                     (unsigned long long) F->kda_graphs.invalidations);
    F->kda_graphs.clear();   // before the graph's buffers and streams are freed
#ifdef __linux__
    if (F->igpu_lib != nullptr) {   // (unregisters the RAM tier before it is freed)
        if (F->igpu_falls.load() > 0 || F->igpu_early.load() > 0)
            std::fprintf(stderr, "glm fast: iGPU lane: %llu calls (%llu started before their route's disk reads), %llu "
                                 "left to the CPU\n", (unsigned long long) F->igpu_calls.load(),
                         (unsigned long long) F->igpu_early.load(), (unsigned long long) F->igpu_falls.load());
        if (const uint64_t cc = F->igpu_cols_calls.load(); cc > 0 || F->igpu_cols_falls.load() > 0)
            std::fprintf(stderr, "glm fast: iGPU lane: %llu verify-window calls of %.2f columns each, %llu windows a call a column\n",
                         (unsigned long long) cc, cc > 0 ? (double) F->igpu_cols_cols.load() / (double) cc : 0.0,
                         (unsigned long long) F->igpu_cols_falls.load());
        if (auto cl = (abyss_igpu_close_fn) dlsym(F->igpu_lib, "abyss_igpu_close")) cl();
        F->igpu_experts = nullptr;
        F->igpu_rows = nullptr;
        F->igpu_reg = nullptr;
        F->igpu_unreg = nullptr;
    }
#endif
    for (auto& R : F->rc)
        if (R.base && R.registered) numa_pinned_free(R.base, (size_t) R.n * R.stride);
        else if (R.base) cudaFreeHost(R.base);
    for (auto& d : F->draining) cudaEventDestroy(d.ev);
    for (auto& m : F->bg) cudaEventDestroy(m.ev);
    for (auto e : F->ev_free) cudaEventDestroy(e);
    for (auto* sb : F->stage)
        if (sb) cudaFreeHost(sb);
    for (int ub = 0; ub < 2; ++ub) {
        if (F->upd_key_h[ub]) cudaFreeHost(F->upd_key_h[ub]);
        if (F->upd_val_h[ub]) cudaFreeHost(F->upd_val_h[ub]);
        if (F->upd_key_d[ub]) cudaFree(F->upd_key_d[ub]);
        if (F->upd_val_d[ub]) cudaFree(F->upd_val_d[ub]);
        if (F->upd_ev[ub]) cudaEventDestroy(F->upd_ev[ub]);
    }
    if (F->ring_h) cudaFreeHost(F->ring_h);
    if (F->md.route_error_d) cudaFree(F->md.route_error_d);
    if (F->resp_h) cudaFreeHost(F->resp_h);
    if (F->emb_h) cudaFreeHost(F->emb_h);
    if (F->tok_in_h) cudaFreeHost(F->tok_in_h);
    if (F->emb_tab_h) cudaFreeHost(F->emb_tab_h);
    if (F->emb_q8_h) cudaFreeHost(F->emb_q8_h);
    if (F->hop_h) cudaFreeHost(F->hop_h);
    if (F->tok_h) cudaFreeHost(F->tok_h);
    if (F->scratch) cudaFree(F->scratch);
    if (F->mtp_h) cudaFree(F->mtp_h);
    if (F->mtp_catq) cudaFree(F->mtp_catq);
    if (F->mtp_logits) cudaFree(F->mtp_logits);
    if (F->mtp_tok) cudaFree(F->mtp_tok);
    if (F->mtp_tok_h) cudaFreeHost(F->mtp_tok_h);
    if (F->ev_mtp) cudaEventDestroy(F->ev_mtp);
    if (F->pool) cudaFree(F->pool);
    for (uint8_t* p : F->pool_more) cudaFree(p);
    if (F->xpool) cudaFree(F->xpool);
    if (F->arena) cudaFree(F->arena);
    if (F->ps) cudaStreamSynchronize(F->ps);
    if (F->ev_hop) cudaEventDestroy(F->ev_hop);
    if (F->ev_done) cudaEventDestroy(F->ev_done);
    if (F->ev_bgq) cudaEventDestroy(F->ev_bgq);
    if (F->ev_gs) cudaEventDestroy(F->ev_gs);
    if (F->ev_g0) cudaEventDestroy(F->ev_g0);
    for (cudaEvent_t e : F->ev_ge)
        if (e) cudaEventDestroy(e);
    if (F->ev_pred) cudaEventDestroy(F->ev_pred);
    if (F->ev_pf) cudaEventDestroy(F->ev_pf);
    if (F->ev_pf_prev) cudaEventDestroy(F->ev_pf_prev);
    if (F->cs) cudaStreamDestroy(F->cs);
    if (F->copy) cudaStreamDestroy(F->copy);
    if (F->ps) cudaStreamDestroy(F->ps);
    delete F;
    fast_ = nullptr;
}

// the CPU lane's split as a share of a route's RAM-tier experts that go over PCIe (the mean over f = 1..8)
static double plan_share(unsigned long long plan) {
    double s = 0.0;
    for (int f = 1; f <= 8; ++f) s += (double) (f - (int) ((plan >> (4 * f)) & 15ull)) / f;
    return s / 8.0;
}

double Glm5Model::pcie_share() const {
    return fast_ != nullptr && fast_->cpu_pool != nullptr ? plan_share(fast_->cpu_plan) : 1.0;
}

bool Glm5Model::set_pcie_share(double share) {
    bool any = false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        FastState* F = m->fast_;
        if (F == nullptr || F->cpu_pool == nullptr) continue;
        unsigned long long plan = F->settle && F->settle_late ? F->cpu_plan_late : F->cpu_plan_start;
        if (share >= 0.0) {
            plan = 0;
            const double s = std::min(1.0, share);
            for (int f = 0; f <= plan_f(m->g_); ++f) plan |= (unsigned long long) (f - (int) std::lround(s * f)) << (4 * f);
        }
        F->cpu_plan = plan;   // (the next route reads it; the boundary's background moves follow it too)
        // a request's own share holds for the whole request; the engine's starts the late switch's count again
        F->late_n = 0;
        F->late_done = share >= 0.0 || F->late_after == 0;
        any = true;
    }
    return any;
}

int Glm5Model::cpu_lane_threads() const {
    return fast_ != nullptr && fast_->cpu_pool != nullptr ? fast_->cpu_pool->active() : 0;
}

bool Glm5Model::set_cpu_lane_threads(int n) {
    bool any = false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        FastState* F = m->fast_;
        if (F == nullptr || F->cpu_pool == nullptr) continue;
        F->cpu_pool->set_active(n > 0 ? n : F->cpu_pool->size());
        any = true;
    }
    return any;
}

Glm5Model::FastStats Glm5Model::fast_stats() const {
    FastStats s;
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        const FastState* F = m->fast_;
        if (F == nullptr) continue;
        s.hits += F->hits.load();
        s.misses += F->misses.load();
        s.miss_layers += F->miss_layers.load();
        s.ram_hits += F->ram_hits.load();
        s.disk_reads += F->disk_reads.load();
        s.promotions += F->promotions.load();
        s.cpu_experts += F->cpu_experts.load();
        s.tail_skips += F->tail_skips.load();
        s.cpu_ms += (double) F->cpu_us.load() / 1000.0;
        s.miss_ms += (double) F->miss_us.load() / 1000.0;
        s.disk_ms += (double) F->disk_us.load() / 1000.0;
        s.pool_slots += F->pool_slots;
        s.pool_gb += (double) F->pool_bytes / 1073741824.0;
        s.ram_gb += (double) F->ram_bytes / 1073741824.0;
        for (const auto& P : F->lp)
            for (char c : P.st) s.pool_used += c == FastState::kResident;
        for (const auto& R : F->rc) {
            s.ram_slots += R.n;
            for (char c : R.st) s.ram_used += c == FastState::kRHold || c == FastState::kRNew;
        }
    }
    if (fast_ != nullptr) {
        s.tokens = fast_->tokens;
        s.ms = fast_->ms;
    }
    s.mirror_b0 = strata::core::shard_io::mirror_bytes[0].load(std::memory_order_relaxed);
    s.mirror_b1 = strata::core::shard_io::mirror_bytes[1].load(std::memory_order_relaxed);
    return s;
}

// ---------------------------------------------------------------- the service thread
//
// Consumes every route the device publishes, in order: LFU counts, the promotions the device made (an expert
// landed in a spare slot), and - the only case the device waits for - the experts that are on disk only: read
// into a free RAM-tier slot (or pinned staging) and answered with that source for the fetch kernel.
void Glm5Model::fast_service() {
    FastState* F = fast_;
    cudaSetDevice(dev_);
    const Glm5Geometry& g = g_;
    const int K = g.n_exp_used;
    unsigned int next = 1;
    // Idle (through a prompt, between requests) the thread polls the ring with sleeps instead of spinning: it held one
    // core per GPU at 100% while the server waited (2x V100: 200% CPU idle), a core a 6-core PC's prompt lane and the
    // system then lacked, and the server's stall watchdog could never see a stuck engine as idle.  A decode's routes
    // come ~1 ms apart (two GPUs: half a token between a card's routes), so it spins all through one.  After
    // STRATA_GLM_SERVICE_IDLE_MS (200; 0 = spin always) without a route it polls every 0.1 ms (Windows' timer: 1 ms),
    // after 10x that every 2 ms - the first route after a quiet spell is answered up to one sleep later
    static const int kIdleMs = [] {
        const char* v = getenv("STRATA_GLM_SERVICE_IDLE_MS");
        return v ? std::max(0, std::atoi(v)) : 200;
    }();
    auto last_route = std::chrono::steady_clock::now();
    auto igpu_route = last_route, igpu_poked = last_route;   // the iGPU lane's keep-alive (below)
    // ... for how long after a route: STRATA_GLM_IGPU_POKE_MS (500; it was 20).  DeepSeek V4.1 waits on the SSD
    // between routes for longer than 20 ms, and the idle iGPU then took 21-27 ms to answer its next call (the CPU lane
    // took over): V4.1 Q2 on an AI PRO R9700 + 8060S, the slowest held-out sequence, 433 such calls and 136 ms a token;
    // 500 ms: 3 calls, 65.6 ms (the same sequence with 2 hardware queues and 20 ms: 123 calls, 80.4 ms)
    static const auto poke_win = std::chrono::milliseconds([] {
        const char* v = getenv("STRATA_GLM_IGPU_POKE_MS");
        return v != nullptr ? std::max(0, std::atoi(v)) : 500;
    }());
    int idle = 0;                        // 0 spinning, 1 short sleeps, 2 long sleeps
    unsigned int polls = 0;
    // STRATA_GLM_SVC_STAT=1 (investigation): where this thread's time goes per route - from seeing the request to
    // the lane call (the bookkeeping), the lane call, from its return to the answer's publication - and how often a
    // spin iteration took > 20 us (the thread was inside something else, e.g. the keep-alive's driver call, when a
    // request could have landed).  Printed every 4,096 routes
    static const bool svc_stat = getenv("STRATA_GLM_SVC_STAT") != nullptr;
    struct SvcStat {
        uint64_t n = 0, pre_us = 0, pre_max = 0, lane_us = 0, lane_max = 0, post_us = 0, post_max = 0;
        uint64_t gaps = 0, gap_us = 0, gap_max = 0, iters = 0;
        uint64_t ne_n[9] = {}, ne_us[9] = {};   // the lane's calls by their expert count
    } ss;
    auto last_iter = std::chrono::steady_clock::now();
    auto t_seen = last_iter;
    const auto us_since = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
#if defined(_WIN32) && defined(STRATA_USE_HIP)
    auto last_kick = last_route;
    unsigned int kick_polls = 0;
#endif
    // a batched verify window's columns of a layer (MoeRequest::win_more), queued until its last column's route and
    // answered by one fast_cpu_experts_multi call (the CPU lane, not the iGPU lane); STRATA_GLM_CPU_MULTI=0: one call
    // a column, as before
    static const bool cpu_multi = [] {
        const char* v = getenv("STRATA_GLM_CPU_MULTI");
        return v == nullptr || std::atoi(v) != 0;
    }();
    struct WinCol {
        unsigned int seq;
        int ne;
        const uint8_t* blob[gf::kMaxK];
        float w[gf::kMaxK];
        const float* x;
        const void* xq;   // (the iGPU lane's columns call: x's q8_1 from the request, when it has it)
    };
    WinCol wq[kVerifyMax];
    int wq_n = 0, wq_layer = -1;
    const auto lane_flush = [&] {
        const auto t0 = std::chrono::steady_clock::now();
        int ne[kVerifyMax];
        const uint8_t* blob[gf::kMaxK * kVerifyMax];
        float w[gf::kMaxK * kVerifyMax];
        const float* x[kVerifyMax];
        const void* xqv[kVerifyMax];
        float* out[kVerifyMax];
        uint64_t n_exp = 0;
        bool xq_all = true;
        for (int t = 0; t < wq_n; ++t) {
            xqv[t] = wq[t].xq;
            xq_all = xq_all && wq[t].xq != nullptr;
            ne[t] = wq[t].ne;
            n_exp += (uint64_t) wq[t].ne;
            for (int i = 0; i < gf::kMaxK; ++i) {
                blob[gf::kMaxK * t + i] = i < wq[t].ne ? wq[t].blob[i] : nullptr;
                w[gf::kMaxK * t + i] = i < wq[t].ne ? wq[t].w[i] : 0.0f;
            }
            x[t] = wq[t].x;
            out[t] = (F->cpu_ans_h + (wq[t].seq % gf::kCpuAnswers))->part;
        }
        fast_cpu_experts_multi(wq_layer, wq_n, ne, blob, w, x, out, xq_all ? xqv : nullptr);
        const auto t1 = std::chrono::steady_clock::now();
        std::atomic_thread_fence(std::memory_order_release);
        for (int t = 0; t < wq_n; ++t) (F->cpu_ans_h + (wq[t].seq % gf::kCpuAnswers))->seq = wq[t].seq;
        F->cpu_experts.fetch_add(n_exp, std::memory_order_relaxed);
        F->cpu_routes.fetch_add((uint64_t) wq_n, std::memory_order_relaxed);
        F->cpu_us.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(),
                            std::memory_order_relaxed);
        wq_n = 0;
        wq_layer = -1;
    };
    while (!F->quit.load(std::memory_order_relaxed)) {
        gf::MoeRequest* rq = F->ring_h + (next % gf::kRingSize);
        bool lane_started = false;   // (EARLY LANE, below: this route's lane experts already on the iGPU)
        const unsigned int sq = rq->seq;
        if (sq != next) {
            // an entry AHEAD of us means the device lapped the ring (only non-waiting routes can be lapped: a
            // disk miss parks the device until it is answered) - resync instead of waiting forever
            if ((int) (sq - next) > 0 && sq % gf::kRingSize == next % gf::kRingSize) {
                F->processed.fetch_add((uint64_t) (sq - next), std::memory_order_release);
                next = sq;
            } else {
#if defined(_WIN32) && defined(STRATA_USE_HIP)
                // Windows' HIP runtime submits queued launches only when the host waits on the GPU (Linux submits
                // each): a route the main thread queued can sit there while this thread waits for it.  After 1 ms
                // without a route, submit what is queued - cudaStreamQuery does without waiting (not while a graph is
                // being captured on the stream).  Submitting every launch instead (GPU_FLUSH_ON_EXECUTION=1) cost
                // ~31 us a launch on a Radeon 8065S: decode 9.8 instead of 15.2 tokens/s
                if ((++kick_polls & 63u) == 0 || idle != 0) {
                    const auto now = std::chrono::steady_clock::now();
                    if (now - last_route > std::chrono::milliseconds(1) && now - last_kick > std::chrono::milliseconds(1)) {
                        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
                        if (cudaStreamIsCapturing(F->cs, &cap) == cudaSuccess && cap == cudaStreamCaptureStatusNone)
                            (void) cudaStreamQuery(F->cs);
                        (void) cudaGetLastError();
                        last_kick = now;
                    }
                }
#endif
                if (idle == 2) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                } else if (idle == 1) {
#ifdef _WIN32
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
#else
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
#endif
                    if ((++polls & 255u) == 0 &&
                        std::chrono::steady_clock::now() - last_route > std::chrono::milliseconds(10 * kIdleMs))
                        idle = 2;
                } else {
                    cpu_relax();
                    if (svc_stat) {
                        const auto now = std::chrono::steady_clock::now();
                        const uint64_t g = us_since(last_iter, now);
                        last_iter = now;
                        ++ss.iters;
                        if (g > 20) {
                            ++ss.gaps;
                            ss.gap_us += g;
                            ss.gap_max = std::max(ss.gap_max, g);
                        }
                    }
                    // the iGPU lane: a no-op kernel every 50 us within poke_win of a route keeps its clocks up
                    if (F->igpu_poke != nullptr && (polls & 63u) == 0) {
                        const auto now = std::chrono::steady_clock::now();
                        if (now - igpu_poked > std::chrono::microseconds(50) && now - igpu_route < poke_win) {
                            F->igpu_poke();
                            igpu_poked = now;
                        }
                    }
                    if (kIdleMs > 0 && (++polls & 4095u) == 0 &&
                        std::chrono::steady_clock::now() - last_route > std::chrono::milliseconds(kIdleMs))
                        idle = 1;
                }
                continue;
            }
        }
        if (svc_stat) t_seen = std::chrono::steady_clock::now();
        if (kIdleMs > 0) {
            idle = 0;
            last_route = std::chrono::steady_clock::now();
        }
        if (F->igpu_poke != nullptr) igpu_route = std::chrono::steady_clock::now();
        std::atomic_thread_fence(std::memory_order_acquire);
        if (rq->error != 0) {
            if (wq_n > 0) lane_flush();
            F->processed.fetch_add(1, std::memory_order_release);
            ++next;
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(F->mu);
            const int il = rq->layer;
            const unsigned int miss = rq->miss_mask, fetch = rq->fetch_mask, promo = rq->promo_mask;
            const unsigned int cpu = rq->cpu_mask;   // the CPU lane's experts: not in VRAM, computed below
            const unsigned int skip = rq->skip_mask;   // left out (the draft block's misses, the trunk's tail skip)
            if (skip != 0u && il != mtp_il_)
                F->tail_skips.fetch_add((uint64_t) std::popcount(skip), std::memory_order_relaxed);
            auto& P = F->lp[(size_t) il];
            auto& R = F->rc[(size_t) F->layer_rc[(size_t) il]];
            ++F->clock;
            int ids[gf::kMaxK];
            for (int i = 0; i < K; ++i) ids[i] = rq->ids[i];
            // STRATA_GLM_ROUTE_LOG=<prefix>: "layer e0 .. e7 tier-mask" per route into <prefix>.<device> (cache studies)
            static const char* rlog = getenv("STRATA_GLM_ROUTE_LOG");
            if (rlog != nullptr) {
                static thread_local FILE* rf = std::fopen((std::string(rlog) + "." + std::to_string(dev_)).c_str(), "w");
                if (rf != nullptr) {
                    std::fprintf(rf, "%d", il);
                    for (int i = 0; i < K; ++i) std::fprintf(rf, " %d", ids[i]);
                    std::fprintf(rf, " %u %u %u\n", fetch, miss, cpu);
                }
            }
            uint64_t nh = 0;
            for (int i = 0; i < K; ++i) {
                const int key = il * g.n_expert + ids[i];
                if (F->cnt[(size_t) key] < (1u << 24)) ++F->cnt[(size_t) key];
                if (++F->usage[(size_t) key] >= (1u << 30))
                    for (auto& u : F->usage) u >>= 1;
                if ((skip >> i) & 1u) continue;   // not computed: neither a hit nor a miss (its count above stays)
                const bool resident = (((miss | fetch | cpu) >> i) & 1u) == 0;
                if (resident) {
                    const int s = F->slot_of[(size_t) key];
                    if (s >= 0) P.tick[(size_t) s] = F->clock;
                    ++nh;
                } else {
                    const int rs0 = F->ram_of[(size_t) key];   // used from the RAM tier: recent there
                    if (rs0 >= 0) R.tick[(size_t) rs0] = F->clock;
                }
                if ((promo >> i) & 1u) {
                    // the device filled one of this layer's spares with it: resident from now on
                    const int s = P.slot_index(rq->promo_ptr[i]);
                    if (s >= 0 && s < P.n) {
                        P.st[(size_t) s] = FastState::kResident;
                        P.key[(size_t) s] = ids[i];
                        P.tick[(size_t) s] = F->clock;
                        F->slot_of[(size_t) key] = s;
                        for (int j = 0; j < gf::kSpares; ++j)
                            if (P.spare[j] == s) P.spare[j] = -1;
                        F->promotions.fetch_add(1, std::memory_order_relaxed);
                    }
                    // exclusive tiers: its RAM copy is released at the next boundary (the fetch reads it now)
                    const int rs = F->ram_of[(size_t) key];
                    if (rs >= 0 && R.st[(size_t) rs] == FastState::kRHold) R.st[(size_t) rs] = FastState::kRRelease;
                } else if (((fetch | miss) >> i) & 1u) {
                    F->scratch_uses.fetch_add(1, std::memory_order_relaxed);
                }
            }
            F->cnt_events += (uint64_t) K;
            if (F->cnt_events >= 32768) {
                F->cnt_events = 0;
                for (auto& c : F->cnt) c >>= 1;
            }
            F->hits.fetch_add(nh, std::memory_order_relaxed);
            {
                // prediction quality: how many of this route's experts (and of its non-resident ones) the previous
                // layer's prediction named
                const auto& pv = F->pred_of[(size_t) il];
                if (pv[0] >= 0) {
                    uint64_t ov = 0, mh = 0;
                    for (int i = 0; i < K; ++i) {
                        bool in = false;
                        for (int q2 = 0; q2 < K; ++q2) in |= pv[(size_t) q2] == ids[i];
                        ov += in;
                        if ((((fetch | miss) >> i) & 1u) && in) ++mh;
                    }
                    F->pred_n.fetch_add(1, std::memory_order_relaxed);
                    F->pred_overlap.fetch_add(ov, std::memory_order_relaxed);
                    F->pred_miss.fetch_add((uint64_t) std::popcount((unsigned) (fetch | miss)), std::memory_order_relaxed);
                    F->pred_miss_hit.fetch_add(mh, std::memory_order_relaxed);
                }
                if (il + 1 < (int) F->pred_of.size())
                    for (int i = 0; i < K; ++i) F->pred_of[(size_t) il + 1][(size_t) i] = rq->pred[i];
            }
            // the next layer's experts this route prefetched into that layer's spares: resident from now on
            const int npf = rq->pf_n;
            if (npf > 0 && il + 1 < (int) F->lp.size()) {
                auto& P1 = F->lp[(size_t) il + 1];
                auto& R1 = F->rc[(size_t) F->layer_rc[(size_t) il + 1]];
                for (int q2 = 0; q2 < npf && q2 < gf::kMaxK; ++q2) {
                    const int e = rq->pf_ids[q2];
                    const int key = (il + 1) * g.n_expert + e;
                    const int s = P1.slot_index(rq->pf_ptr[q2]);
                    if (s < 0 || s >= P1.n) continue;
                    P1.st[(size_t) s] = FastState::kResident;
                    P1.key[(size_t) s] = e;
                    P1.tick[(size_t) s] = F->clock;
                    F->slot_of[(size_t) key] = s;
                    for (int j = 0; j < gf::kSpares; ++j)
                        if (P1.spare[j] == s) P1.spare[j] = -1;
                    const int rs = F->ram_of[(size_t) key];
                    if (rs >= 0 && R1.st[(size_t) rs] == FastState::kRHold) R1.st[(size_t) rs] = FastState::kRRelease;
                    F->prefetches.fetch_add(1, std::memory_order_relaxed);
                }
            }
            F->ram_hits.fetch_add((uint64_t) std::popcount((unsigned) (fetch)), std::memory_order_relaxed);
            F->misses.fetch_add((uint64_t) std::popcount((unsigned) (fetch | miss | cpu)), std::memory_order_relaxed);
            if (F->n_ahead > 0) fast_ahead_route(il, ids, miss, rq->ahead);
            // EARLY LANE: a route with disk reads and lane experts starts the iGPU on those before its reads (they used
            // to wait for the reads, ~3 ms, and the device then for the lane after the last chunk); collected below
            const auto lane_start = [&] {
                if (lane_started || rq->cpu_mask == 0u || F->cpu_ans_h == nullptr || F->igpu_start == nullptr ||
                    F->igpu_experts == nullptr || F->L[(size_t) il].exl3)
                    return;
                const auto& Ly = F->L[(size_t) il];
                const void* bl[gf::kMaxK];
                float wl[gf::kMaxK];
                int ne = 0;
                for (int i = 0; i < g.n_exp_used; ++i)
                    if ((rq->cpu_mask >> i) & 1u) {
                        bl[ne] = (const void*) rq->cpu_src[i];
                        wl[ne++] = rq->w[i];
                    }
                lane_started = F->igpu_start(Ly.gu_type, Ly.d_type, g.n_embd, g.n_ff_exp, Ly.down_off, g.swiglu_exp, ne,
                                             bl, wl, rq->x,
                                             rq->xq_ok && F->igpu_experts_q != nullptr ? (const void*) rq->xq : nullptr) == 0;
            };
            if (miss != 0) {
                // ---- disk only: read each into a free RAM slot (it stays there unless it was promoted) or staging;
                //      one the LOOKAHEAD already read (or is reading) is answered from its RAM slot
                const auto t0 = std::chrono::steady_clock::now();
                const int cls = F->layer_rc[(size_t) il];
                int nm = 0, mi[gf::kMaxK], nr = 0, rd[gf::kMaxK], nw = 0, wt[gf::kMaxK], wm[gf::kMaxK], ns = 0, sl[gf::kMaxK], nev = 0, evk[gf::kMaxK];
                uint64_t stv[gf::kMaxK];
                uint8_t* dst[gf::kMaxK];
                for (int i = 0; i < K; ++i)
                    if ((miss >> i) & 1u) mi[nm++] = i;
                for (int m = 0; m < nm; ++m) {
                    const int key = il * g.n_expert + ids[mi[m]];
                    int rs = F->ram_of[(size_t) key];
                    if (rs >= 0 && (R.st[(size_t) rs] == FastState::kRLoad || R.st[(size_t) rs] == FastState::kRNew)) {
                        dst[m] = R.base + (size_t) rs * R.stride;
                        if (R.st[(size_t) rs] == FastState::kRLoad) {
                            auto& ld = F->rload[(size_t) cls][(size_t) rs];
                            uint64_t v = ld.load(std::memory_order_acquire);
                            if ((v & 3u) == 0u && ld.compare_exchange_strong(v, v | 1u, std::memory_order_acq_rel)) {
                                rd[nr++] = m;   // still queued: read it here (the reader skips it)
                                sl[ns] = rs;
                                stv[ns++] = (v & ~(uint64_t) 3u) | 2u;
                                ++F->ah_stolen;
                            } else if ((v & 3u) != 2u) {
                                wm[nw] = m;
                                wt[nw++] = rs;   // a reader has it in hand
                            }
                            ++F->ah_used;
                        }
                        if ((promo >> mi[m]) & 1u) {   // promoted: a pure staging use, freed at the boundary
                            R.st[(size_t) rs] = FastState::kRRelease;
                            R.key[(size_t) rs] = -1;
                            F->ram_of[(size_t) key] = -1;
                        } else {
                            R.st[(size_t) rs] = FastState::kRNew;
                        }
                        continue;
                    }
                    // Demoted at the last boundary: its RAM copy is still landing (kRDemote until the next boundary makes
                    // it live), so the table has no entry for it - but its bytes are one PCIe copy away, not a disk read
                    // away.  Wait for that drain's event and serve it from the slot.  (The boundary runs only once this
                    // thread has caught up, so F->draining is not being edited here.)  A route that promotes it leaves
                    // a duplicate that the tier clean-up frees.
                    if (rs >= 0 && R.st[(size_t) rs] == FastState::kRDemote && R.key[(size_t) rs] == key) {
                        bool draining = false;
                        for (const auto& d : F->draining)
                            if (d.rclass == cls && d.rslot == rs) {
                                cudaEventSynchronize(d.ev);
                                draining = true;
                                break;
                            }
                        if (draining) {
                            dst[m] = R.base + (size_t) rs * R.stride;
                            continue;
                        }
                    }
                    rd[nr++] = m;
                    rs = -1;
                    for (int s2 = 0; s2 < R.n; ++s2)
                        if (R.st[(size_t) s2] == FastState::kRFree) {
                            rs = s2;
                            break;
                        }
                    if (rs < 0 && F->disk_lane && F->disk_adapt) {
                        // ARENA-ADAPT (Xolotl; STRATA_GLM_DISK_ADAPT=1 - measured worse here, 2026-10-18: 11.6K -> 17.4K
                        // disk reads on seq0's 512 tokens): no free slot - evict the class's least used RAM expert not routed for
                        // STRATA_GLM_RAM_PROTECT tokens (the boundary's victim rule) and land the miss in its slot,
                        // instead of staging it (read again next time).  The device's rtab entry is cleared by this
                        // response, before any later route reads it; none of this route's experts is a candidate.
                        static const uint64_t protect = [] {
                            const char* v = getenv("STRATA_GLM_RAM_PROTECT");
                            return (uint64_t) (v ? std::max(0, std::atoi(v)) : 64) * 42u;
                        }();
                        uint32_t bc = UINT32_MAX;
                        for (int s2 = 0; s2 < R.n; ++s2) {
                            if (R.st[(size_t) s2] != FastState::kRHold || R.tick[(size_t) s2] + protect > F->clock)
                                continue;
                            const int k2 = R.key[(size_t) s2];
                            bool mine = false;
                            for (int i = 0; i < K; ++i) mine |= k2 == il * g.n_expert + ids[i];
                            const uint32_t c = F->cnt[(size_t) k2];
                            if (!mine && (c < bc || (c == bc && rs >= 0 && R.tick[(size_t) s2] < R.tick[(size_t) rs]))) {
                                bc = c;
                                rs = s2;
                            }
                        }
                        if (rs >= 0) {
                            const int k2 = R.key[(size_t) rs];
                            evk[nev++] = k2;
                            F->ram_of[(size_t) k2] = -1;
                            F->left[(size_t) k2] = 2;
                            ++F->diag_ram_evict;
                            ++F->dlane_evict;
                            R.st[(size_t) rs] = FastState::kRFree;
                            R.key[(size_t) rs] = -1;
                        }
                    }
                    if (rs >= 0) {
                        R.key[(size_t) rs] = key;
                        R.tick[(size_t) rs] = F->clock;
                        R.st[(size_t) rs] = ((promo >> mi[m]) & 1u) ? FastState::kRRelease : FastState::kRNew;
                        F->ram_of[(size_t) key] = ((promo >> mi[m]) & 1u) ? -1 : rs;
                        dst[m] = R.base + (size_t) rs * R.stride;
                        if ((promo >> mi[m]) & 1u) R.key[(size_t) rs] = -1;   // a pure staging use: freed at the boundary
                    } else {
                        dst[m] = F->stage[(size_t) (next % gf::kCpuAnswers) * std::max(8, g.n_exp_used) + m];
                    }
                }
                // each part in chunks across the workers: the O_DIRECT bounce copy of a whole part was ~0.4 ms of the
                // wait (Mercury: 3.6 -> 3.1 ms a disk wait at 8 chunks; STRATA_GLM_READ_CHUNKS overrides)
                const int rch = F->read_chunks;
                const bool dlane_now = F->disk_lane && promo == 0u && F->cpu_plan != 0ull &&
                                       F->cpu_fmt[(size_t) il].n_ff > 0;
                gf::MoeResponse* Rp = F->resp_h + (next % gf::kCpuAnswers);
                // STREAMED fetch: the response goes out now, before the reads; the reads below set ready[] bit by bit
                // and the device's fetch kernel copies each chunk as it lands.  The lookahead's in-flight slots go out
                // as streamed entries too and get all their bits once their reader is done - after this call's own
                // reads, which so run alongside them (they used to wait for them first: a lookahead read that was
                // late by most of a read then cost a read and a half).  STRATA_GLM_AHEAD_MERGE=0: the old order
                static const bool ah_merge = [] {
                    const char* v = getenv("STRATA_GLM_AHEAD_MERGE");
                    return v == nullptr || std::atoi(v) != 0;
                }();
                const bool stream = F->stream_fetch && !dlane_now && (nr > 0 || (ah_merge && nw > 0));
                if (stream) {
                    if (nw > 0 && !ah_merge) {
                        const auto tw = std::chrono::steady_clock::now();
                        for (int i = 0; i < nw; ++i)
                            while ((F->rload[(size_t) cls][(size_t) wt[i]].load(std::memory_order_acquire) & 3u) != 2u)
                                cpu_relax();
                        F->ah_waited += (uint64_t) nw;
                        F->ah_wait_us.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                                    std::chrono::steady_clock::now() - tw).count(),
                                                std::memory_order_relaxed);
                    }
                    Rp->ptr_mask = 0;
                    Rp->host_mask = 0;
                    Rp->stream_mask = 0;
                    Rp->n_upd = 0;
                    Rp->cpu = 0;
                    for (int v = 0; v < nev; ++v) {
                        Rp->upd_key[Rp->n_upd] = (int) F->rtab_key(evk[v]);
                        Rp->upd_val[Rp->n_upd++] = 0ull;
                    }
                    for (int m = 0; m < nm; ++m) {
                        Rp->ptr[mi[m]] = (unsigned long long) dst[m];
                        Rp->ptr_mask |= 1u << mi[m];
                        Rp->ready[mi[m]] = 0xffffffffu;   // (landed already: every chunk there)
                    }
                    for (int r2 = 0; r2 < nr; ++r2) {
                        Rp->ready[mi[rd[r2]]] = 0u;
                        Rp->stream_mask |= 1u << mi[rd[r2]];
                    }
                    if (ah_merge)
                        for (int i = 0; i < nw; ++i) {
                            Rp->ready[mi[wm[i]]] = 0u;
                            Rp->stream_mask |= 1u << mi[wm[i]];
                        }
                    std::atomic_thread_fence(std::memory_order_release);
                    Rp->seq = next;
                }
                if (nr > 0 && !dlane_now) lane_start();
                F->disk_busy.store(1, std::memory_order_release);   // (the RAM fill's reads wait)
                F->workers->run(nr * 3 * rch, [&](int job) {
                    const int m = rd[job / (3 * rch)], part = job % (3 * rch);
                    fast_read_part(il, ids[mi[m]], part / rch, dst[m], part % rch, rch);
                    if (stream) {   // the chunk is in the slot: tell the fetch kernel
                        std::atomic_thread_fence(std::memory_order_release);
                        std::atomic_ref<unsigned int>(Rp->ready[mi[m]]).fetch_or(1u << part, std::memory_order_release);
                    }
                });
                F->disk_busy.store(0, std::memory_order_release);
                for (int i = 0; i < ns; ++i) F->rload[(size_t) cls][(size_t) sl[i]].store(stv[i], std::memory_order_release);
                if (nw > 0 && stream && ah_merge) {   // the lookahead's in-flight slots: all their chunks once landed
                    const auto tw = std::chrono::steady_clock::now();
                    for (int i = 0; i < nw; ++i) {
                        while ((F->rload[(size_t) cls][(size_t) wt[i]].load(std::memory_order_acquire) & 3u) != 2u)
                            cpu_relax();
                        std::atomic_thread_fence(std::memory_order_release);
                        std::atomic_ref<unsigned int>(Rp->ready[mi[wm[i]]]).store(0xffffffffu, std::memory_order_release);
                    }
                    F->ah_waited += (uint64_t) nw;
                    F->ah_wait_us.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                                std::chrono::steady_clock::now() - tw).count(),
                                            std::memory_order_relaxed);
                }
                if (nw > 0 && !stream) {
                    const auto tw = std::chrono::steady_clock::now();
                    for (int i = 0; i < nw; ++i)
                        while ((F->rload[(size_t) cls][(size_t) wt[i]].load(std::memory_order_acquire) & 3u) != 2u)
                            cpu_relax();
                    F->ah_waited += (uint64_t) nw;
                    F->ah_wait_us.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                                std::chrono::steady_clock::now() - tw).count(),
                                            std::memory_order_relaxed);
                }
                if (!stream) {
                Rp->ptr_mask = 0;
                Rp->host_mask = 0;
                Rp->stream_mask = 0;
                Rp->n_upd = 0;
                Rp->cpu = 0;
                for (int v = 0; v < nev; ++v) {   // the arena-adapt evictions: their RAM-tier entries leave the table
                    Rp->upd_key[Rp->n_upd] = (int) F->rtab_key(evk[v]);
                    Rp->upd_val[Rp->n_upd++] = 0ull;
                }
                if (dlane_now) {
                    // the disk lane (the route claimed no spare): their weighted sum from the RAM landing, by the lane
                    const uint8_t* blob[gf::kMaxK];
                    float wl[gf::kMaxK];
                    for (int m = 0; m < nm; ++m) {
                        blob[m] = dst[m];
                        wl[m] = rq->w[mi[m]];
                        Rp->host_mask |= 1u << mi[m];
                    }
                    const auto tl = std::chrono::steady_clock::now();
                    fast_cpu_experts(il, nm, blob, wl, rq->x, Rp->cpu_part);
                    Rp->cpu = 1;
                    F->dlane_experts.fetch_add((uint64_t) nm, std::memory_order_relaxed);
                    F->dlane_us.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                              std::chrono::steady_clock::now() - tl).count(),
                                          std::memory_order_relaxed);
                } else {
                    for (int m = 0; m < nm; ++m) {
                        Rp->ptr[mi[m]] = (unsigned long long) dst[m];
                        Rp->ptr_mask |= 1u << mi[m];
                    }
                }
                std::atomic_thread_fence(std::memory_order_release);
                Rp->seq = next;
                }   // (!stream)
                const uint64_t us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - t0).count();
                F->disk_reads.fetch_add((uint64_t) nr, std::memory_order_relaxed);   // read here (not ahead)
                for (int r2 = 0; r2 < nr; ++r2) ++F->diag_disk[F->left[(size_t) il * g.n_expert + ids[mi[rd[r2]]]] & 3];
                F->disk_us.fetch_add(us, std::memory_order_relaxed);
                F->miss_us.fetch_add(us, std::memory_order_relaxed);
                F->miss_layers.fetch_add(1, std::memory_order_relaxed);
                static const bool trace = getenv("STRATA_GLM_SVC_TRACE") != nullptr;
                if (trace)
                    std::fprintf(stderr, "svc CUDA%d seq %u layer %d disk %d %.2f ms\n", dev_, next, il, nm,
                                 (double) us / 1000.0);
            }
        }
        // ---- the CPU LANE: this route's host experts from their RAM-tier blobs (they stay in RAM until the next
        //      boundary: the device waits for this answer before its down combine, so the token cannot end first).
        //      A batched verify window's column (win_more: that many more routes of the layer follow, the device
        //      waiting for none of them yet) is queued, and the window's last column answers them all
        if (wq_n > 0 && wq_layer != rq->layer) lane_flush();   // (a window cut short)
        if (rq->cpu_mask != 0u && F->cpu_ans_h != nullptr && cpu_multi && !lane_started &&
            (F->igpu_experts == nullptr || (F->igpu_cols != nullptr && !F->L[(size_t) rq->layer].exl3)) &&
            (rq->win_more > 0 || wq_n > 0) && wq_n < kVerifyMax) {
            WinCol& c = wq[wq_n++];
            wq_layer = rq->layer;
            c.seq = next;
            c.ne = 0;
            for (int i = 0; i < g.n_exp_used; ++i)
                if ((rq->cpu_mask >> i) & 1u) {
                    c.blob[c.ne] = (const uint8_t*) rq->cpu_src[i];
                    c.w[c.ne++] = rq->w[i];
                }
            c.x = rq->x;
            c.xq = rq->xq_ok ? (const void*) rq->xq : nullptr;
        } else if (rq->cpu_mask != 0u && F->cpu_ans_h != nullptr) {
            const auto t0 = std::chrono::steady_clock::now();
            const unsigned int cm = rq->cpu_mask;
            int ne = 0;
            const uint8_t* blob[gf::kMaxK];
            float w[gf::kMaxK];
            for (int i = 0; i < g.n_exp_used; ++i)
                if ((cm >> i) & 1u) {
                    blob[ne] = (const uint8_t*) rq->cpu_src[i];
                    w[ne++] = rq->w[i];
                }
            gf::CpuAnswer* ans = F->cpu_ans_h + (next % gf::kCpuAnswers);
            int rc = 1;
            if (lane_started) {
                rc = F->igpu_finish(ans->part, g.n_embd);
                if (rc == 0) {
                    F->igpu_calls.fetch_add(1, std::memory_order_relaxed);
                    F->igpu_early.fetch_add(1, std::memory_order_relaxed);
                } else {
                    if (rc < 0) {
                        std::fprintf(stderr, "glm fast: CUDA%d iGPU lane failed - the CPU lane takes over\n", dev_);
                        F->igpu_experts = nullptr;
                    }
                    F->igpu_falls.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (rc != 0) {
                auto* H = F->expert_helper.get();
                if (!H || !H->active || H->failed) {
                    fast_cpu_experts(rq->layer, ne, blob, w, rq->x, ans->part, rq->xq_ok ? (const void*) rq->xq : nullptr);
                } else {
                    int hn = 0, cn = 0, hi[8];
                    float hw[8], cw[8];
                    const uint8_t* cb[8];
                    for (int i = 0; i < g.n_exp_used; ++i) if ((cm >> i) & 1u) {
                        if (H->owns(rq->layer * g.n_expert + rq->ids[i])) {
                            hi[hn] = rq->ids[i]; hw[hn++] = rq->w[i];
                        } else {
                            cb[cn] = (const uint8_t*) rq->cpu_src[i]; cw[cn++] = rq->w[i];
                        }
                    }
                    bool ok = false;
                    std::string helper_err;
                    if (hn) {
                        const auto& L = F->L[(size_t) rq->layer];
                        ok = H->begin(rq->layer, hn, hi, hw, rq->x, L.gu_type, L.d_type, L.down_off,
                                      g.swiglu_exp, helper_err);
                    }
                    if (ok) {
                        if (cn) fast_cpu_experts(rq->layer, cn, cb, cw, rq->x, ans->part,
                                                rq->xq_ok ? (const void*) rq->xq : nullptr);
                        else std::fill(ans->part, ans->part + g.n_embd, 0.0f);
                        ok = H->finish(ans->part, helper_err);
                    }
                    if (!ok) {
                        if (hn) {
                            ++H->fallbacks;
                            H->disable();
                            std::fprintf(stderr, "%s; CPU fallback for this and subsequent requests\n", helper_err.c_str());
                        }
                        // Recompute the entire original route; a partial helper result is never published.
                        fast_cpu_experts(rq->layer, ne, blob, w, rq->x, ans->part,
                                         rq->xq_ok ? (const void*) rq->xq : nullptr);
                    }
                }
            }
            const auto t1 = std::chrono::steady_clock::now();
            std::atomic_thread_fence(std::memory_order_release);
            ans->seq = next;
            F->cpu_experts.fetch_add((uint64_t) ne, std::memory_order_relaxed);
            F->cpu_routes.fetch_add(1, std::memory_order_relaxed);
            F->cpu_us.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count(),
                                std::memory_order_relaxed);
            if (svc_stat) {
                const auto t2 = std::chrono::steady_clock::now();
                const uint64_t pre = us_since(t_seen, t0), lane = us_since(t0, t1), post = us_since(t1, t2);
                ++ss.n;
                ss.pre_us += pre; ss.pre_max = std::max(ss.pre_max, pre);
                ss.lane_us += lane; ss.lane_max = std::max(ss.lane_max, lane);
                ss.post_us += post; ss.post_max = std::max(ss.post_max, post);
                ss.ne_n[std::min(ne, 8)] += 1;
                ss.ne_us[std::min(ne, 8)] += lane;
                if (ss.n % 4096 == 0) {
                    std::fprintf(stderr, "svc stat CUDA%d: %llu routes | seen->lane %.1f us (max %llu) | lane %.1f us (max %llu) | "
                                         "lane->answer %.1f us (max %llu) | spin iterations %llu, > 20 us: %llu (%.1f%%), "
                                         "%.2f ms in all, max %llu us\n",
                                 dev_, (unsigned long long) ss.n, (double) ss.pre_us / (double) ss.n,
                                 (unsigned long long) ss.pre_max, (double) ss.lane_us / (double) ss.n,
                                 (unsigned long long) ss.lane_max, (double) ss.post_us / (double) ss.n,
                                 (unsigned long long) ss.post_max, (unsigned long long) ss.iters,
                                 (unsigned long long) ss.gaps, 100.0 * (double) ss.gaps / (double) std::max<uint64_t>(1, ss.iters),
                                 (double) ss.gap_us / 1000.0, (unsigned long long) ss.gap_max);
                    std::string by;
                    for (int k = 1; k <= 8; ++k)
                        if (ss.ne_n[k] > 0)
                            by += " " + std::to_string(k) + ": " + std::to_string(ss.ne_n[k]) + " x " +
                                  std::to_string((int) (ss.ne_us[k] / ss.ne_n[k])) + " us";
                    std::fprintf(stderr, "svc stat CUDA%d: the lane's calls by their experts:%s\n", dev_, by.c_str());
                    ss = SvcStat{};
                }
                last_iter = t2;
            }
        }
        if (wq_n > 0 && rq->win_more == 0) lane_flush();   // the window's last column: answer them all
        F->processed.fetch_add(1, std::memory_order_release);
        ++next;
    }
}

// a whole EXL3 expert into its blob: the nine pieces sorted by file offset, those less than 64 KB apart read as one
// range through a bounce buffer (the 3.05 bpw files keep an expert's three trellises together and its scales elsewhere:
// two reads instead of nine; files that keep the whole expert together: one)
void Glm5Model::fast_read_expert_x3(int il, int e, uint8_t* blob) {
    const auto& Ly = fast_->L[(size_t) il];
    const auto& nl = pack_layers_[(size_t) il];
    struct Piece {
        int sh;
        uint64_t off;
        size_t len;
        uint8_t* dst;
    } p[9];
    for (int r = 0; r < 3; ++r) {
        uint8_t* at = blob + (r == 0 ? 0 : r == 1 ? Ly.gu_bytes : Ly.down_off);
        for (int j = 0; j < 3; ++j) {
            const size_t i = (size_t) e * 9 + (size_t) r * 3 + (size_t) j;
            p[r * 3 + j] = Piece{nl.x_shard[i], nl.x_off[i], nl.x_piece(r, j), at};
            at += nl.x_piece(r, j);
        }
    }
    std::sort(p, p + 9, [](const Piece& a, const Piece& b) { return std::make_pair(a.sh, a.off) < std::make_pair(b.sh, b.off); });
    static thread_local std::vector<uint8_t> bounce;
    for (int a = 0; a < 9;) {
        int b = a + 1;
        uint64_t end = p[a].off + p[a].len;
        while (b < 9 && p[b].sh == p[a].sh && p[b].off <= end + (64u << 10)) {
            end = std::max<uint64_t>(end, p[b].off + p[b].len);
            ++b;
        }
        const Shard& sh = pack_shards_[(size_t) p[a].sh];
        if (b == a + 1) {
            read_slice(sh, p[a].off, p[a].len, p[a].dst);
        } else {
            bounce.resize((size_t) (end - p[a].off));
            read_slice(sh, p[a].off, (size_t) (end - p[a].off), bounce.data());
            for (int k = a; k < b; ++k) std::memcpy(p[k].dst, bounce.data() + (p[k].off - p[a].off), p[k].len);
        }
        a = b;
    }
}

// one part (0 gate, 1 up, 2 down) of expert e of layer il - or its chunk-th of n_chunks equal pieces - from the shards
// into its place in the blob at `blob`
void Glm5Model::fast_read_part(int il, int e, int role, uint8_t* blob, int chunk, int n_chunks) {
    const auto& Ly = fast_->L[(size_t) il];
    const auto& nl = pack_layers_[(size_t) il];
    if (nl.x_cb >= 0) {
        // EXL3: the part's three pieces (trellis, suh, svh) from wherever the safetensors keep them; a chunk is the
        // same byte range of the part [trellis | suh | svh] as a GGUF part's - the streamed fetch kernel copies a
        // chunk's range once its bit is set (chunks of the trellis alone, the scales with the first, had the last
        // chunk's range copied before its scales landed: SSD-read experts were wrong)
        const size_t base = role == 0 ? 0 : role == 1 ? Ly.gu_bytes : Ly.down_off;
        const size_t len = nl.x_part(role);
        const size_t piece = ((len + (size_t) n_chunks - 1) / (size_t) n_chunks + 4095) & ~(size_t) 4095;
        const size_t c0 = std::min(len, (size_t) chunk * piece), c1 = std::min(len, c0 + piece);
        size_t at = 0;
        for (int j = 0; j < 3; ++j) {
            const size_t i = (size_t) e * 9 + (size_t) role * 3 + (size_t) j;
            const size_t pl = nl.x_piece(role, j);
            const size_t a = std::max(c0, at), b = std::min(c1, at + pl);
            if (b > a) read_slice(pack_shards_[(size_t) nl.x_shard[i]], nl.x_off[i] + (a - at), b - a, blob + base + a);
            at += pl;
        }
        return;
    }
    const Shard& sh = pack_shards_[(size_t) (role == 0 ? nl.gate_shard : role == 1 ? nl.up_shard : nl.down_shard)];
    const uint64_t off = role == 0   ? nl.gate_off + (uint64_t) e * Ly.gu_bytes
                         : role == 1 ? nl.up_off + (uint64_t) e * Ly.gu_bytes
                                     : nl.down_off + (uint64_t) e * Ly.dn_bytes;
    const size_t len = role == 2 ? Ly.dn_bytes : Ly.gu_bytes;
    const size_t piece = ((len + (size_t) n_chunks - 1) / (size_t) n_chunks + 4095) & ~(size_t) 4095;
    const size_t c0 = std::min(len, (size_t) chunk * piece), c1 = std::min(len, c0 + piece);
    if (c1 > c0)
        read_slice(sh, off + c0, c1 - c0, blob + (role == 0 ? 0 : role == 1 ? Ly.gu_bytes : 2 * Ly.gu_bytes) + c0);
}

// ---- LOOKAHEAD, for the route of layer il (service thread, F->mu held): score the predictions made for this layer
//      d+1 layers earlier, keep this route's predictions for the layers after it, and queue a read of every
//      predicted expert that is on disk only (VRAM and RAM tiers both lack it) into a free RAM slot.  The slot is
//      kRLoad until it lands; the layer's own route then finds it through ram_of (the device still calls it a disk
//      miss: rtab learns of it at the next boundary).
void Glm5Model::fast_ahead_route(int il, const int* ids, unsigned int miss, const short (*ahead)[gf::kMaxK]) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const int K = g.n_exp_used;
    ++F->ah_routes;
    auto& pr = F->ah_pred[(size_t) il];
    bool cov[gf::kMaxK] = {};
    for (int d = 0; d < F->n_ahead; ++d) {
        if (pr[(size_t) d][0] < 0) continue;
        ++F->ah_npred[d];
        for (int i = 0; i < K; ++i) {
            bool in = false;
            for (int q = 0; q < (int) pr[(size_t) d].size(); ++q) in |= pr[(size_t) d][(size_t) q] == ids[i];
            F->ah_ov[d] += in;
            if (in && ((miss >> i) & 1u)) {
                ++F->ah_disk_cov[d];
                cov[i] = true;
            }
        }
        pr[(size_t) d].fill(-1);
    }
    for (int i = 0; i < K; ++i)
        if ((miss >> i) & 1u) {
            ++F->ah_disk;
            F->ah_disk_any += cov[i];
        }
    for (int d = 0; d < F->n_ahead; ++d) {
        const int la = il + 1 + d;
        if (ahead[d][0] < 0 || la >= (int) F->ah_pred.size()) continue;
        for (int i = 0; i < (int) F->ah_pred[(size_t) la][(size_t) d].size(); ++i) F->ah_pred[(size_t) la][(size_t) d][(size_t) i] = ahead[d][i];
    }
    if (!F->ahead_read) return;
    constexpr int kMaxInflight = 4;
    bool queued = false;
    for (int d = 0; d < F->n_ahead; ++d) {   // the nearest layer first: its reads are the most urgent
        const int la = il + 1 + d;
        if (ahead[d][0] < 0 || la >= (int) F->layer_rc.size() || F->layer_rc[(size_t) la] < 0) continue;
        const int cls = F->layer_rc[(size_t) la];
        auto& R = F->rc[(size_t) cls];
        // STRATA_GLM_AHEAD_TOP=<n>: read only the prediction's first n ranks (Xolotl: top-4 of a two-layers-ahead
        // prediction used 65% of its reads; all six used 22% here and the wasted reads held the SSD up)
        static const int ah_top = [] {
            const char* v = getenv("STRATA_GLM_AHEAD_TOP");
            return v != nullptr ? std::max(1, std::atoi(v)) : 8;
        }();
        for (int i = 0; i < std::min(K, ah_top); ++i) {
            const int e = ahead[d][i];
            if (e < 0) continue;
            const int key = la * g.n_expert + e;
            if (F->slot_of[(size_t) key] >= 0 || F->ram_of[(size_t) key] >= 0) continue;
            if (F->ah_inflight.load(std::memory_order_relaxed) >= kMaxInflight) {
                ++F->ah_full;
                continue;
            }
            int rs = -1;
            for (int s = 0; s < R.n; ++s)
                if (R.st[(size_t) s] == FastState::kRFree) {
                    rs = s;
                    break;
                }
            if (rs < 0) {
                ++F->ah_nofree;
                continue;
            }
            R.key[(size_t) rs] = key;
            R.st[(size_t) rs] = FastState::kRLoad;
            F->ram_of[(size_t) key] = rs;
            const uint64_t tag = ++F->ah_tag;
            F->rload[(size_t) cls][(size_t) rs].store(tag << 2, std::memory_order_release);
            F->ah_inflight.fetch_add(1, std::memory_order_relaxed);
            ++F->ah_issued;
            {
                std::lock_guard<std::mutex> lk(F->ah_mu);
                F->ah_q.push_back(FastState::AheadJob{cls, rs, la, e, tag});
            }
            queued = true;
        }
    }
    if (queued) F->ah_cv.notify_all();
}

// the LOOKAHEAD reader threads: one queued expert at a time into the RAM slot the service thread claimed for it - unless
// the route got there first and read it itself (the tag's phase moved on), or the slot was reused under a newer tag
void Glm5Model::fast_ahead_reader() {
    FastState* F = fast_;
    for (;;) {
        FastState::AheadJob j;
        {
            std::unique_lock<std::mutex> lk(F->ah_mu);
            F->ah_cv.wait(lk, [&] { return F->ah_quit || !F->ah_q.empty(); });
            if (F->ah_quit) return;
            j = F->ah_q.front();
            F->ah_q.erase(F->ah_q.begin());
        }
        auto& ld = F->rload[(size_t) j.rclass][(size_t) j.rslot];
        uint64_t want = j.tag << 2;
        if (ld.compare_exchange_strong(want, (j.tag << 2) | 1u, std::memory_order_acq_rel)) {
            uint8_t* blob = F->rc[(size_t) j.rclass].base + (size_t) j.rslot * F->rc[(size_t) j.rclass].stride;
            if (j.fill) {
                // a RAM fill reads behind the routes: a chunk at a time, each waiting until no route read is in flight
                // (the SSD serves one stream at a time at full speed; the route's reads are on the token's path)
                const int rch = F->read_chunks;
                for (int role = 0; role < 3; ++role)
                    for (int c = 0; c < rch; ++c) {
                        while (F->disk_busy.load(std::memory_order_acquire) != 0)
                            std::this_thread::sleep_for(std::chrono::microseconds(100));
                        fast_read_part(j.il, j.e, role, blob, c, rch);
                    }
            } else {
                for (int role = 0; role < 3; ++role) fast_read_part(j.il, j.e, role, blob);
            }
            ld.store((j.tag << 2) | 2u, std::memory_order_release);
        }
        F->ah_inflight.fetch_sub(1, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------- between tokens
//
// Every device idle, the service thread caught up: (1) RAM slots of promoted experts free up, disk reads of the
// last token become RAM residents; (2) demotions that landed make their RAM copy live and their VRAM slot free;
// (3) every layer's spares are refilled - from free slots, else by evicting the layer's least-used resident
// (demoted into the RAM tier when it beats the RAM tier's least-used, dropped otherwise); (4) the RAM tier keeps
// a few free slots for disk reads.  The table edits ride one small update kernel at the head of the next token.
bool Glm5Model::fast_boundary(std::string& err) {
    FastState* F = fast_;
    if (F == nullptr || F->lp.empty() || F->rc.empty()) return true;
    const auto tb0 = std::chrono::steady_clock::now();
    // Also catches an asynchronous failure before waiting for routes that may never have been published.
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm boundary compute", err) || !F->route_ok(err))
        return false;
    const Glm5Geometry& g = g_;
    const auto tb1 = std::chrono::steady_clock::now();
    while (F->processed.load(std::memory_order_acquire) < F->expected) cpu_relax();
    std::lock_guard<std::mutex> lk(F->mu);
    if (!F->late_done && ++F->late_n >= F->late_after) {   // (STRATA_GLM_PCIE_SHARE_LATE)
        F->cpu_plan = F->cpu_plan_late;
        F->late_done = true;
    }
    if (F->settle && ++F->settle_n >= F->settle_win) {   // (STRATA_GLM_PCIE_SHARE_LATE=<share>:settle)
        const uint64_t h = F->hits.load(std::memory_order_relaxed), t = h + F->misses.load(std::memory_order_relaxed);
        const double r = t > F->settle_t0 ? (double) (h - F->settle_h0) / (double) (t - F->settle_t0) : -1.0;
        F->settle_n = 0;
        F->settle_h0 = h;
        F->settle_t0 = t;
        if (r >= 0.0) {
            const bool own = F->cpu_plan == F->cpu_plan_start || F->cpu_plan == F->cpu_plan_late;
            if (!F->settle_late && F->settle_prev >= 0.0 && r - F->settle_prev < F->settle_eps) {
                F->settle_late = true;
                F->settle_level = r;
                if (own) F->cpu_plan = F->cpu_plan_late;   // (a request's own share stays)
                ++F->settle_moves;
                std::fprintf(stderr, "glm fast: CUDA%d CPU lane: the pool settled at %.1f%% VRAM hits - the late share "
                                     "(switch %llu)\n", dev_, 100.0 * r, (unsigned long long) F->settle_moves);
            } else if (F->settle_late && r < F->settle_level - F->settle_drop) {
                F->settle_late = false;
                if (own) F->cpu_plan = F->cpu_plan_start;
                ++F->settle_moves;
                std::fprintf(stderr, "glm fast: CUDA%d CPU lane: VRAM hits fell to %.1f%% (from %.1f%%) - the engine's "
                                     "share again (switch %llu)\n", dev_, 100.0 * r, 100.0 * F->settle_level,
                             (unsigned long long) F->settle_moves);
            }
            F->settle_prev = r;
        }
    }
    struct GapScan {   // the table scan's time, whichever way the function leaves
        FastState* F;
        std::chrono::steady_clock::time_point t;
        ~GapScan() {
            if (F) F->gap_bscan += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
        }
    } gap_scan{F->gap_on ? F : nullptr, std::chrono::steady_clock::now()};
    if (F->gap_on) {
        F->gap_bsync += std::chrono::duration<double, std::milli>(tb1 - tb0).count();
        F->gap_bwait += std::chrono::duration<double, std::milli>(gap_scan.t - tb1).count();
    }
    int nu = 0;
    bool ok = true;
    // the edits go through one of two pinned sets: a flush records an event on the stream after its copies, and a
    // set is refilled only once that event has passed (the previous boundary's set, in practice long done) - the
    // flush itself no longer waits for the stream (it was ~145 us a token with the GPU idle).
    // STRATA_GLM_BOUNDARY_SYNC=1: the stream sync after every flush, as before
    static const bool bsync = getenv("STRATA_GLM_BOUNDARY_SYNC") != nullptr && std::atoi(getenv("STRATA_GLM_BOUNDARY_SYNC")) != 0;
    int ub = F->upd_buf;
    int* kh = F->upd_key_h[ub];
    unsigned long long* vh = F->upd_val_h[ub];
    const auto take_set = [&] {
        ub = F->upd_buf;
        kh = F->upd_key_h[ub];
        vh = F->upd_val_h[ub];
        if (!bsync) ok = ok && glmfast::cuda_ok(cudaEventSynchronize(F->upd_ev[ub]), "glm table set reuse", err);
    };
    take_set();
    const auto flush = [&] {
        if (nu > 0 && ok) {
            if (F->zero_copy) {
                // the kernel reads the host-mapped set itself: no copy engine, nothing to queue behind
                gf::tab_update(F->tab, F->upd_key_hd[ub], F->upd_val_hd[ub], nu, F->cs);
                ok = glmfast::cuda_ok(cudaGetLastError(), "glm table update", err);
            } else {
                ok = glmfast::cuda_ok(cudaMemcpyAsync(F->upd_key_d[ub], kh, (size_t) nu * sizeof(int),
                                                      cudaMemcpyHostToDevice, F->cs), "glm table keys", err) &&
                     glmfast::cuda_ok(cudaMemcpyAsync(F->upd_val_d[ub], vh,
                                                      (size_t) nu * sizeof(unsigned long long), cudaMemcpyHostToDevice,
                                                      F->cs), "glm table values", err);
                if (ok) {
                    gf::tab_update(F->tab, F->upd_key_d[ub], F->upd_val_d[ub], nu, F->cs);
                    ok = glmfast::cuda_ok(cudaGetLastError(), "glm table update", err);
                }
            }
            if (bsync) {
                // Pinned sources belong to the DMA until THIS stream completes, including the final batch.
                const cudaError_t e = cudaStreamSynchronize(F->cs);
                if (ok) ok = glmfast::cuda_ok(e, "glm table update sync", err);
            } else {
                if (ok) ok = glmfast::cuda_ok(cudaEventRecord(F->upd_ev[ub], F->cs), "glm table update event", err);
                F->upd_buf ^= 1;
                take_set();
            }
        }
        nu = 0;
        ++F->upd_g;
    };
    ++F->upd_g;
    const auto upd = [&](size_t k, unsigned long long v) {
        if (!ok) return;
        if (F->upd_gen[k] == F->upd_g) {   // edited earlier in this batch: the last value is the one that counts
            vh[F->upd_at[k]] = v;
            return;
        }
        if (nu == FastState::kMaxUpd) {   // a full batch goes out first (the host buffers are reused after it ran)
            flush();
            if (!ok) return;
        }
        F->upd_gen[k] = F->upd_g;
        F->upd_at[k] = nu;
        kh[nu] = (int) k;
        vh[nu] = v;
        ++nu;
    };
    // the RAM tier's victim among its held experts: the least-used by the aged counts among those no route used in the
    // last `protect` routes (~64 tokens) - a newly hot expert keeps its place while its count catches up, instead of
    // going back to disk as the lowest count and being read again a moment later (real chats measured 7-11 disk reads
    // a token that way); all of them recent: the least recently used.  STRATA_GLM_RAM_EVICT=lfu: the plain least-used
    // (the old rule), =lru: the least recently used.  -1: none held.
    static const int ram_policy = [] {
        const char* v = getenv("STRATA_GLM_RAM_EVICT");
        return v == nullptr ? 0 : std::strcmp(v, "lfu") == 0 ? 1 : std::strcmp(v, "lru") == 0 ? 2 : 0;
    }();
    static const uint64_t protect = [] {
        const char* v = getenv("STRATA_GLM_RAM_PROTECT");
        return (uint64_t) (v ? std::max(0, std::atoi(v)) : 64) * 42u;
    }();
    const auto ram_victim = [&](const FastState::RamClass& R) -> int {
        int best = -1, oldest = -1;
        uint32_t bc = UINT32_MAX;
        uint64_t bt = UINT64_MAX;
        for (int s = 0; s < R.n; ++s) {
            if (R.st[(size_t) s] != FastState::kRHold) continue;
            const uint64_t t = R.tick[(size_t) s];
            if (t < bt) {
                bt = t;
                oldest = s;
            }
            if (ram_policy == 2) continue;
            if (ram_policy == 0 && t + protect > F->clock) continue;
            const uint32_t c = F->cnt[(size_t) R.key[(size_t) s]];
            if (c < bc || (c == bc && t < R.tick[(size_t) best])) {
                bc = c;
                best = s;
            }
        }
        return best >= 0 ? best : oldest;
    };
    static const bool tier_gc = [] {
        const char* v = getenv("STRATA_GLM_TIER_GC");
        return v == nullptr || std::atoi(v) != 0;
    }();
    // (1)
    for (size_t c = 0; c < F->rc.size(); ++c) {
        auto& R = F->rc[c];
        for (int s = 0; s < R.n; ++s) {
            if (R.st[(size_t) s] == FastState::kRLoad) {
                // a LOOKAHEAD read nobody asked for yet: in the RAM tier once it landed
                if ((F->rload[c][(size_t) s].load(std::memory_order_acquire) & 3u) == 2u) {
                    R.st[(size_t) s] = FastState::kRHold;
                    upd(F->rtab_key(R.key[(size_t) s]), (unsigned long long) (R.base + (size_t) s * R.stride));
                }
            } else if (R.st[(size_t) s] == FastState::kRRelease) {
                const int key = R.key[(size_t) s];
                if (key >= 0) {
                    upd(F->rtab_key(key), 0ull);
                    if (F->ram_of[(size_t) key] == s) F->ram_of[(size_t) key] = -1;
                }
                R.key[(size_t) s] = -1;
                R.st[(size_t) s] = FastState::kRFree;
            } else if (R.st[(size_t) s] == FastState::kRNew) {
                R.st[(size_t) s] = FastState::kRHold;
                upd(F->rtab_key(R.key[(size_t) s]), (unsigned long long) (R.base + (size_t) s * R.stride));
            } else if (R.st[(size_t) s] == FastState::kRHold && tier_gc) {
                // exclusive tiers, kept (STRATA_GLM_TIER_GC=0: not): a held copy the tables lost (a route promoted or re-read the expert while its
                // demotion was still landing, which then held a second copy nothing points at) or of an expert VRAM
                // holds is freed; one that is the expert's only copy is adopted back.  Such copies piled up to ~12%
                // of the experts in no tier at all, the RAM tier full of duplicates (chats: ~5 disk reads a token)
                const int key = R.key[(size_t) s];
                const int ro = key >= 0 ? F->ram_of[(size_t) key] : -1;
                if (key >= 0 && ro == s && F->slot_of[(size_t) key] < 0) continue;   // the normal case
                if (key >= 0 && ro < 0 && F->slot_of[(size_t) key] < 0) {
                    F->ram_of[(size_t) key] = s;
                    upd(F->rtab_key(key), (unsigned long long) (R.base + (size_t) s * R.stride));
                    ++F->diag_adopt;
                    continue;
                }
                if (key >= 0 && (ro == s || ro < 0)) {
                    upd(F->rtab_key(key), 0ull);
                    if (ro == s) F->ram_of[(size_t) key] = -1;
                }
                R.key[(size_t) s] = -1;
                R.st[(size_t) s] = FastState::kRFree;
                ++F->diag_dup;
            }
        }
    }
    // (2)
    std::vector<FastState::Drain> still;
    for (auto& d : F->draining) {
        const cudaError_t de = cudaEventQuery(d.ev);
        if (de != cudaSuccess && de != cudaErrorNotReady)
            return glmfast::cuda_ok(de, "glm demotion event", err);
        if (de == cudaErrorNotReady) {
            still.push_back(d);
            continue;
        }
        auto& R = F->rc[(size_t) d.rclass];
        if (R.st[(size_t) d.rslot] == FastState::kRDemote && R.key[(size_t) d.rslot] == d.key) {
            R.st[(size_t) d.rslot] = FastState::kRHold;
            upd(F->rtab_key(d.key), (unsigned long long) (R.base + (size_t) d.rslot * R.stride));
            if (F->ram_resident) {   // the deferred half of the move: forget the VRAM copy only now
                upd(F->tab_key((size_t) d.key), 0ull);
                F->slot_of[(size_t) d.key] = -1;
            }
        }
        char& vst = F->lp[(size_t) d.il].st[(size_t) d.vslot];
        if (vst == FastState::kDraining) vst = FastState::kFree;   // (a lent slot stays lent)
        F->ev_free.push_back(d.ev);
    }
    F->draining.swap(still);
    // (3)
    for (int il = l0_; il < lt_; ++il) {
        if (F->bg_hold) break;   // lending retires moves without starting spare-refill demotions
        auto& P = F->lp[(size_t) il];
        if (P.n == 0) continue;
        auto& R = F->rc[(size_t) F->layer_rc[(size_t) il]];
        for (int j = 0; j < gf::kSpares; ++j) {
            if (P.spare[j] >= 0) continue;
            int s = -1;
            for (int s2 = 0; s2 < P.n; ++s2)
                if (P.st[(size_t) s2] == FastState::kFree) {
                    s = s2;
                    break;
                }
            if (s >= 0) {
                P.st[(size_t) s] = FastState::kSpare;
                P.key[(size_t) s] = -1;
                P.spare[j] = s;
                upd(F->spare_key(il, j), (unsigned long long) P.slot_ptr(s));
                continue;
            }
            // A full unified pool needs spares only in empty slots (e.g. after
            // prompt lending). Never evict an expert just to reserve a spare.
            if (glmfast::full_unified_pool(F->unified_memory, P.n, g.n_expert)) break;
            // evict the least-used resident of this layer (recency breaks ties).  STRATA_GLM_VRAM_EVICT=lru: the
            // least recently used instead - route traces replayed through these rules (Maya-M, 3090 + 3060 split):
            // the aged counts keep experts that were frequent a while ago, the smaller half's pool most of all (VRAM
            // hits 55.7 -> 65.2 %, off-card experts a token 57.7 -> 45.4 on the 3060; 71.8 -> 64.6 on the 3090)
            static const bool vram_lru = [] {
                const char* ve = getenv("STRATA_GLM_VRAM_EVICT");
                return ve != nullptr && std::strcmp(ve, "lru") == 0;
            }();
            int v = -1;
            uint32_t bc = UINT32_MAX;
            uint64_t bt = UINT64_MAX;
            for (int s2 = 0; s2 < P.n; ++s2) {
                if (P.st[(size_t) s2] != FastState::kResident) continue;
                const uint32_t c = vram_lru ? 0u : F->cnt[(size_t) il * g.n_expert + P.key[(size_t) s2]];
                if (c < bc || (c == bc && P.tick[(size_t) s2] < bt)) {
                    bc = c;
                    bt = P.tick[(size_t) s2];
                    v = s2;
                }
            }
            if (v < 0) break;
            const int vkey = il * g.n_expert + P.key[(size_t) v];
            // demote it into the RAM tier: a free slot, else the RAM tier's victim (if colder than it)
            int rs = -1;
            uint32_t rc_min = UINT32_MAX;
            for (int s2 = 0; s2 < R.n; ++s2)
                if (R.st[(size_t) s2] == FastState::kRFree) {
                    rs = s2;
                    rc_min = 0;
                    break;
                }
            if (rs < 0 && F->ram_resident) {
                // no free RAM slot: keep the VRAM expert (nothing drops to disk); the spare is refilled later
                ++F->diag_resident_skip;
                break;
            }
            // resident mode keeps the VRAM table entry live until the demotion's copy lands (the slot is
            // kDraining, so nothing overwrites it): a route mid-flight still hits VRAM
            if (!F->ram_resident) {
                upd(F->tab_key(vkey), 0ull);
                F->slot_of[(size_t) vkey] = -1;
            }
            ++P.evictions;
            if (rs < 0) {
                rs = ram_victim(R);
                if (rs >= 0) rc_min = F->cnt[(size_t) R.key[(size_t) rs]];
            }
            // STRATA_GLM_DEMOTE_TIES=1: a tie demotes instead of dropping (the VRAM expert was resident, the RAM
            // victim is the tier's coldest).  Off: with the counts mostly 0 the demotions pushed RAM experts out that
            // came back from disk (V4.1 seq0: re-reads after a RAM eviction 450 -> 774)
            static const bool demote_ties = [] {
                const char* v = getenv("STRATA_GLM_DEMOTE_TIES");
                return v != nullptr && std::atoi(v) != 0;
            }();
            if (rs >= 0 && (R.st[(size_t) rs] == FastState::kRFree || rc_min < bc || (demote_ties && rc_min == bc))) {
                if (R.st[(size_t) rs] == FastState::kRHold) {
                    const int old = R.key[(size_t) rs];
                    F->left[(size_t) old] = 2;
                    ++F->diag_ram_evict;
                    upd(F->rtab_key(old), 0ull);
                    F->ram_of[(size_t) old] = -1;
                }
                R.key[(size_t) rs] = vkey;
                R.tick[(size_t) rs] = F->clock;
                R.st[(size_t) rs] = FastState::kRDemote;
                F->ram_of[(size_t) vkey] = rs;
                cudaMemcpyAsync(R.base + (size_t) rs * R.stride, P.slot_ptr(v),
                                F->L[(size_t) il].blob, cudaMemcpyDeviceToHost, F->copy);
                cudaEvent_t ev = F->get_event();
                cudaEventRecord(ev, F->copy);
                F->draining.push_back(FastState::Drain{il, v, vkey, F->layer_rc[(size_t) il], rs, ev});
                P.st[(size_t) v] = FastState::kDraining;
                P.key[(size_t) v] = -1;
                F->demotions.fetch_add(1, std::memory_order_relaxed);
            } else {
                // colder than everything in RAM: dropped (disk only); the slot can be a spare right away - the
                // table edit that forgets it runs before any route of the next token
                P.st[(size_t) v] = FastState::kSpare;
                P.key[(size_t) v] = -1;
                P.spare[j] = v;
                upd(F->spare_key(il, j), (unsigned long long) P.slot_ptr(v));
                F->drops.fetch_add(1, std::memory_order_relaxed);
                F->left[(size_t) vkey] = 1;
                ++F->diag_drop;
            }
        }
    }
    // (4) a few free RAM slots per class for the next token's disk reads (and its reads ahead); RAM-resident
    // mode plans no disk reads, so no residents are evicted to keep landing room either
    static const char* kf = getenv("STRATA_GLM_RAM_FREE");
    // (the disk lane lands every miss in a free RAM slot - ~25 a token here - so it keeps more free; 8 / 16 staged most
    // of them and read them again, 330-340 ms/token; with arena-adapt it evicts at the miss instead)
    const int keep_free = F->ram_resident ? 0 : kf ? std::max(1, std::atoi(kf)) :
                          F->disk_lane && !F->disk_adapt ? 32 : F->ahead_read ? 12 : 4;
    for (auto& R : F->rc) {
        int nfree = 0;
        for (char c : R.st) nfree += c == FastState::kRFree;
        while (nfree < keep_free) {
            const int best = ram_victim(R);
            if (best < 0) break;
            upd(F->rtab_key(R.key[(size_t) best]), 0ull);
            F->ram_of[(size_t) R.key[(size_t) best]] = -1;
            F->left[(size_t) R.key[(size_t) best]] = 2;
            ++F->diag_ram_evict;
            R.key[(size_t) best] = -1;
            R.st[(size_t) best] = FastState::kRFree;
            ++nfree;
        }
    }
    // (4b) the RAM FILL (STRATA_GLM_RAM_FILL, default 2 with the disk tail skip): the most routed disk-only experts
    //      read into the RAM tier in the background, up to ram_fill a boundary.  With the disk tail skip an expert the
    //      routes keep placing at rank 4-6 is counted but never read, so it never leaves the disk and is skipped at
    //      every route (issue #8: 16-25 skips a token through a whole session, the disk reads long since down to
    //      ~0.6).  A candidate takes a free slot past keep_free's reserve (the reserve is the routes'), else the class's
    //      victim when the candidate's count beats the victim's by ram_fill_margin; the lookahead's reader threads
    //      read it into the slot (kRLoad), and step (1) of a later boundary makes it live once it landed.  Nothing
    //      waits: the routes see it on the disk until then
    if (F->ram_fill > 0 && !F->bg_hold) {
        struct FillCand {
            uint32_t cnt;
            int il, e;
        };
        std::vector<FillCand> fc;
        const int NE = g.n_expert;
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe || F->layer_rc[(size_t) il] < 0 || F->L[(size_t) il].tid2eid != nullptr) continue;
            FillCand top[2] = {{0u, il, -1}, {0u, il, -1}};   // the layer's two most routed disk-only experts
            for (int e = 0; e < NE; ++e) {
                const size_t key = (size_t) il * NE + e;
                if (F->slot_of[key] >= 0 || F->ram_of[key] >= 0) continue;
                const uint32_t c = F->cnt[key];
                if (c <= top[1].cnt) continue;
                if (c > top[0].cnt) {
                    top[1] = top[0];
                    top[0] = FillCand{c, il, e};
                } else {
                    top[1] = FillCand{c, il, e};
                }
            }
            for (const auto& t : top)
                if (t.e >= 0 && t.cnt >= F->ram_fill_margin) fc.push_back(t);
        }
        std::sort(fc.begin(), fc.end(), [](const FillCand& a, const FillCand& b) { return a.cnt > b.cnt; });
        int issued = 0;
        bool queued = false;
        for (const auto& c : fc) {
            if (issued >= F->ram_fill || F->ah_inflight.load(std::memory_order_relaxed) >= 4) break;
            const int cls = F->layer_rc[(size_t) c.il];
            auto& R = F->rc[(size_t) cls];
            int nfree = 0, rs = -1;
            for (int s = 0; s < R.n; ++s)
                if (R.st[(size_t) s] == FastState::kRFree) {
                    if (rs < 0) rs = s;
                    ++nfree;
                }
            if (nfree <= keep_free) {   // the reserve is the routes': a colder RAM expert's slot instead, or nothing
                const int v = ram_victim(R);
                if (v < 0 || F->cnt[(size_t) R.key[(size_t) v]] + F->ram_fill_margin > c.cnt) {
                    ++F->fill_passed;
                    continue;
                }
                upd(F->rtab_key(R.key[(size_t) v]), 0ull);
                F->ram_of[(size_t) R.key[(size_t) v]] = -1;
                F->left[(size_t) R.key[(size_t) v]] = 2;
                ++F->diag_ram_evict;
                ++F->fill_evict;
                R.key[(size_t) v] = -1;
                R.st[(size_t) v] = FastState::kRFree;
                rs = v;
            }
            const int key = c.il * NE + c.e;
            R.key[(size_t) rs] = key;
            R.tick[(size_t) rs] = F->clock;
            R.st[(size_t) rs] = FastState::kRLoad;
            F->ram_of[(size_t) key] = rs;
            const uint64_t tag = ++F->ah_tag;
            F->rload[(size_t) cls][(size_t) rs].store(tag << 2, std::memory_order_release);
            F->ah_inflight.fetch_add(1, std::memory_order_relaxed);
            ++F->fill_issued;
            ++issued;
            {
                std::lock_guard<std::mutex> lk(F->ah_mu);
                F->ah_q.push_back(FastState::AheadJob{cls, rs, c.il, c.e, tag, true});
            }
            queued = true;
        }
        if (queued) F->ah_cv.notify_all();
    }
    // (5) background moves between the tiers, over the copy stream.  Where the CPU lane takes every RAM-tier expert
    //     (two sockets beside an x8 link: 0.18 ms an expert on the CPU, 2.1 over PCIe) no route fetches one into VRAM
    //     any more, so the pool kept what the warm-up loaded - minus what each prompt's lending moved out (one 3090:
    //     1134 -> 1022 residents after a prompt, 9-15% hits).  (a) Landed moves go live: a promoted expert is
    //     resident and its RAM slot free (exclusive tiers); a demoted one lives in RAM and its VRAM slot is free.
    //     (b) Up to bg_n new moves: a layer's most used RAM-only expert is promoted into a free slot; with none free,
    //     the layer's least used resident is demoted when a RAM-only expert is used bg_ratio times as much (the slot
    //     takes the promotion at a later boundary).  Either way the expert stays live where it was until its copy
    //     landed - no window in which a route would go to disk.  STRATA_GLM_PROMOTE=<moves a boundary>, 0: off.
    {
        const int NE = g.n_expert;
        std::vector<FastState::BgMove> keep;
        for (auto& m : F->bg) {
            if (m.queued) {   // (paced: pieces still to go, the event not recorded yet)
                keep.push_back(m);
                continue;
            }
            const cudaError_t me = cudaEventQuery(m.ev);
            if (me != cudaSuccess && me != cudaErrorNotReady)
                return glmfast::cuda_ok(me, "glm background event", err);
            if (me == cudaErrorNotReady) {
                keep.push_back(m);
                continue;
            }
            auto& P = F->lp[(size_t) m.il];
            auto& R = F->rc[(size_t) m.rclass];
            if (m.up) {
                P.st[(size_t) m.vslot] = FastState::kResident;
                P.key[(size_t) m.vslot] = m.key % NE;
                P.tick[(size_t) m.vslot] = F->clock;
                F->slot_of[(size_t) m.key] = m.vslot;
                upd(F->tab_key((size_t) m.key), (unsigned long long) P.slot_ptr(m.vslot));
                if (R.st[(size_t) m.rslot] == FastState::kRPin) {
                    R.st[(size_t) m.rslot] = FastState::kRFree;
                    R.key[(size_t) m.rslot] = -1;
                }
                if (F->ram_of[(size_t) m.key] == m.rslot) F->ram_of[(size_t) m.key] = -1;
                upd(F->rtab_key((size_t) m.key), 0ull);
                ++F->bg_up;
            } else {
                upd(F->tab_key((size_t) m.key), 0ull);
                if (F->slot_of[(size_t) m.key] == m.vslot) F->slot_of[(size_t) m.key] = -1;
                if (P.st[(size_t) m.vslot] == FastState::kDraining) {
                    P.st[(size_t) m.vslot] = FastState::kFree;
                    P.key[(size_t) m.vslot] = -1;
                }
                R.st[(size_t) m.rslot] = FastState::kRHold;
                R.tick[(size_t) m.rslot] = F->clock;
                F->ram_of[(size_t) m.key] = m.rslot;
                upd(F->rtab_key((size_t) m.key), (unsigned long long) (R.base + (size_t) m.rslot * R.stride));
                ++F->bg_down;
            }
            F->ev_free.push_back(m.ev);
        }
        F->bg.swap(keep);
        // Qwen3.8-Flash-Next (qwen4exp): 16 moves a boundary and swaps at 2 / 8 (below).  Its experts are ~2 MB (IQ3_S;
        // a copy ~0.3 ms at Gen4 x4, DeepSeek V4's 8 MB held the link 1.2 ms), and at 8 and 4 / 16 the pool barely
        // followed the text: RTX 5090 + Gen4 x4, ISTA IQ3_S, 4 greedy answers, VRAM hits 60-77% -> 68-89%, drafter-on
        // 88 -> 100 tok/s, token by token 79 -> 90 (with the shipped routing profile: 115 -> 118, 94 -> 98)
        const bool small_x = g.q4x;
        static const int bg_n = [small_x] {
            const char* v = getenv("STRATA_GLM_PROMOTE");
            return v != nullptr ? std::max(0, std::atoi(v)) : small_x ? 16 : 8;
        }();
        // (c) the REFILL: while layers have free VRAM slots - a prompt's lending ends with ~15 a layer free on a 24 GB
        //     card - each such layer promotes its hottest RAM-only experts into them, up to STRATA_GLM_PROMOTE_FILL a
        //     boundary in all (24: about a token's worth of the x8 link's copies; the CPU lane loses nothing to the DMA,
        //     a 3090 beside 2x Xeon 6152: 65.5 vs 68.1 GB/s with 6.25 GB/s of H2D beside it) instead of one a layer and
        //     bg_n in all (~80 tokens to refill)
        static const int bg_fill = [] {
            const char* v = getenv("STRATA_GLM_PROMOTE_FILL");
            return v != nullptr ? std::max(0, std::atoi(v)) : 24;
        }();
        const int bg_up_max = std::max(bg_n, bg_fill);
        // a swap (the layer's coldest resident out, a RAM-only expert in) when hot >= ratio * cold + margin
        // (STRATA_GLM_PROMOTE_RATIO / _MARGIN, 4 / 16).  The counts halve every ~130 tokens, so they are small and
        // noisy: at 2 / 4 DeepSeek V4 swapped ~2.7 experts a token, and each 8 MB copy held the PCIe link ~1.2 ms -
        // the route's and the CPU lane's host-mapped reads and writes waited behind it (16 -> ~100 us a route).
        // 4 / 16: ~0.5 a token, decode +3% (prose) / +9% (code) on an RTX 5090 at Gen4 x4.
        static const uint32_t bg_ratio = [small_x] {
            const char* v = getenv("STRATA_GLM_PROMOTE_RATIO");
            return v != nullptr ? (uint32_t) std::max(1, std::atoi(v)) : small_x ? 2u : 4u;
        }();
        static const uint32_t bg_margin = [small_x] {
            const char* v = getenv("STRATA_GLM_PROMOTE_MARGIN");
            return v != nullptr ? (uint32_t) std::max(0, std::atoi(v)) : small_x ? 8u : 16u;
        }();
        bool lane_all = F->cpu_plan != 0ull;   // the CPU lane takes every RAM-tier expert of a route
        for (int f = 1; f <= g.n_exp_used && lane_all; ++f) lane_all = (int) ((F->cpu_plan >> (4 * f)) & 15ull) == f;
        if (bg_n > 0 && lane_all && !F->bg_hold && (int) F->bg.size() < 2 * bg_up_max) {
            struct Cand {
                uint32_t hot, cold;
                int il, e, rs, free_slot, cold_slot;
            };
            std::vector<Cand> cands;
            for (int il = l0_; il < lt_; ++il) {
                if (!F->L[(size_t) il].moe || F->layer_rc[(size_t) il] < 0) continue;
                const auto& P = F->lp[(size_t) il];
                const auto& R = F->rc[(size_t) F->layer_rc[(size_t) il]];
                int cs = -1;
                uint32_t cold = UINT32_MAX;
                int frees[8], nfree = 0;   // (refill: up to 8 a layer a boundary)
                for (int s2 = 0; s2 < P.n; ++s2) {
                    if (P.st[(size_t) s2] == FastState::kFree) {
                        if (nfree < (bg_fill > 0 ? 8 : 1)) frees[nfree++] = s2;
                    } else if (P.st[(size_t) s2] == FastState::kResident && P.key[(size_t) s2] >= 0) {
                        const uint32_t c = F->cnt[(size_t) il * NE + P.key[(size_t) s2]];
                        if (c < cold) {
                            cold = c;
                            cs = s2;
                        }
                    }
                }
                // the layer's hottest RAM-only experts: one for each free slot (at least one: the demotion's candidate)
                const int want = std::max(1, nfree);
                std::array<Cand, 8> top{};
                int ntop = 0;
                for (int e = 0; e < NE; ++e) {
                    const size_t key = (size_t) il * NE + e;
                    if (F->slot_of[key] >= 0) continue;
                    const int rs = F->ram_of[key];
                    if (rs < 0 || R.st[(size_t) rs] != FastState::kRHold) continue;
                    const uint32_t h = F->cnt[key];
                    if (ntop == want && h <= top[(size_t) ntop - 1].hot) continue;
                    int at = ntop < want ? ntop++ : ntop - 1;
                    while (at > 0 && top[(size_t) at - 1].hot < h) {
                        top[(size_t) at] = top[(size_t) at - 1];
                        --at;
                    }
                    top[(size_t) at] = Cand{h, cold, il, e, rs, -1, -1};
                }
                for (int j = 0; j < ntop; ++j) {
                    Cand c = top[(size_t) j];
                    c.free_slot = j < nfree ? frees[j] : -1;
                    c.cold_slot = j == 0 ? cs : -1;
                    if (c.free_slot >= 0 || c.cold_slot >= 0) cands.push_back(c);
                }
            }
            std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.hot > b.hot; });
            int issued = 0, filled = 0;   // demotion moves (at most bg_n), promotions into free slots (bg_up_max)
            for (const auto& c : cands) {
                if (issued >= bg_n && filled >= bg_up_max) break;
                if (c.free_slot >= 0 ? filled >= bg_up_max : issued >= bg_n) continue;
                auto& P = F->lp[(size_t) c.il];
                const int rc = F->layer_rc[(size_t) c.il];
                auto& R = F->rc[(size_t) rc];
                if (c.free_slot >= 0) {
                    P.st[(size_t) c.free_slot] = FastState::kLanding;
                    R.st[(size_t) c.rs] = FastState::kRPin;
                    cudaEvent_t ev = F->get_event();
                    // paced: the MoE layers of the next tokens issue it piece by piece (FastState::bg_drain); not
                    // under the KDA graphs (a copy queued from inside a capture would join it)
                    const bool paced = F->bg_pace > 0 && !F->kda_graph_on;
                    if (paced) {
                        F->bg_q.push_back(FastState::BgPiece{P.slot_ptr(c.free_slot), R.base + (size_t) c.rs * R.stride,
                                                             F->L[(size_t) c.il].blob, 0, ev, cudaMemcpyHostToDevice});
                    } else {
                        cudaMemcpyAsync(P.slot_ptr(c.free_slot), R.base + (size_t) c.rs * R.stride,
                                        F->L[(size_t) c.il].blob, cudaMemcpyHostToDevice, F->copy);
                        cudaEventRecord(ev, F->copy);
                    }
                    F->bg.push_back(FastState::BgMove{c.il, c.free_slot, c.il * NE + c.e, rc, c.rs, true, ev, paced});
                    ++filled;
                } else if (c.cold_slot >= 0 && c.hot >= bg_ratio * std::max<uint32_t>(c.cold, 1u) + bg_margin) {
                    int rs = -1;
                    for (int s2 = 0; s2 < R.n && rs < 0; ++s2)
                        if (R.st[(size_t) s2] == FastState::kRFree) rs = s2;
                    if (rs < 0) continue;
                    const int vkey = c.il * NE + P.key[(size_t) c.cold_slot];
                    P.st[(size_t) c.cold_slot] = FastState::kDraining;   // (still live in VRAM until the copy landed)
                    R.st[(size_t) rs] = FastState::kRDemote;
                    R.key[(size_t) rs] = vkey;
                    cudaEvent_t ev = F->get_event();
                    const bool paced = F->bg_pace > 0 && !F->kda_graph_on;   // (as the promotions above)
                    if (paced) {
                        F->bg_q.push_back(FastState::BgPiece{R.base + (size_t) rs * R.stride, P.slot_ptr(c.cold_slot),
                                                             F->L[(size_t) c.il].blob, 0, ev, cudaMemcpyDeviceToHost});
                    } else {
                        cudaMemcpyAsync(R.base + (size_t) rs * R.stride, P.slot_ptr(c.cold_slot),
                                        F->L[(size_t) c.il].blob, cudaMemcpyDeviceToHost, F->copy);
                        cudaEventRecord(ev, F->copy);
                    }
                    F->bg.push_back(FastState::BgMove{c.il, c.cold_slot, vkey, rc, rs, false, ev, paced});
                    ++issued;
                }
            }
        }
    }
    flush();
    static const bool diag = getenv("STRATA_GLM_TIER_DIAG") != nullptr;
    if (diag && ++F->diag_b % 64 == 0)
        std::fprintf(stderr, "glm tier diag CUDA%d (%llu boundaries): VRAM drops %llu, RAM evictions %llu, lend drops %llu (to RAM %llu), background promotions %llu / demotions %llu, resident skips %llu | "
                             "disk reads of: never held %llu, dropped from VRAM %llu, evicted from RAM %llu, lend-dropped %llu\n",
                     dev_, (unsigned long long) F->diag_b, (unsigned long long) F->diag_drop,
                     (unsigned long long) F->diag_ram_evict, (unsigned long long) F->diag_lend,
                     (unsigned long long) F->diag_lend_park, (unsigned long long) F->bg_up,
                     (unsigned long long) F->bg_down, (unsigned long long) F->diag_resident_skip,
                     (unsigned long long) F->diag_disk[0], (unsigned long long) F->diag_disk[1],
                     (unsigned long long) F->diag_disk[2], (unsigned long long) F->diag_disk[3]);
    if (diag && F->diag_b % 64 == 0) {
        // per RAM class: slots, held, free, and how many of its layers' experts are in no tier at all
        for (size_t c = 0; c < F->rc.size(); ++c) {
            const auto& R = F->rc[c];
            int held = 0, nfree = 0, out = 0, layers = 0;
            for (char st : R.st) {
                held += st == FastState::kRHold;
                nfree += st == FastState::kRFree;
            }
            for (int il = l0_; il < lt_; ++il) {
                if (!F->L[(size_t) il].moe || F->layer_rc[(size_t) il] != (int) c) continue;
                ++layers;
                for (int e = 0; e < g.n_expert; ++e)
                    out += F->slot_of[(size_t) il * g.n_expert + e] < 0 && F->ram_of[(size_t) il * g.n_expert + e] < 0;
            }
            std::fprintf(stderr, "glm tier diag class %zu (%d layers, %.2f MB): %d slots, %d held, %d free, %d experts in "
                                 "no tier\n", c, layers, (double) R.stride / 1e6, R.n, held, nfree, out);
        }
        std::fprintf(stderr, "glm tier diag clean-up: %llu duplicate RAM copies freed, %llu lost copies adopted\n",
                     (unsigned long long) F->diag_dup, (unsigned long long) F->diag_adopt);
    }
    return ok;
}

// GLM_CB_DIR seam dumps (the reference path's names, so the same diff script bisects both): synchronous,
// debug only - every token overwrites, so a run of n tokens leaves position n-1's seams
static void fast_dump(cudaStream_t s, const std::string& name, const float* dev, int64_t n) {
    static const char* dir = getenv("GLM_CB_DIR");
    if (!dir || !dir[0]) return;
    cudaStreamSynchronize(s);
    std::vector<float> buf((size_t) n);
    cudaMemcpy(buf.data(), dev, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost);
    if (FILE* f = std::fopen((std::string(dir) + "/" + name + ".f32").c_str(), "wb")) {
        std::fwrite(buf.data(), 4, (size_t) n, f);
        std::fclose(f);
    }
}

// ---------------------------------------------------------------- the forward
// ---- one DSA (MLA + indexer) mixer: F->x / F->xq -> F->mixer at position p (the trunk's DSA layers and the NextN block)
bool Glm5Model::fast_dsa(int il, int64_t p, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    const auto& Ly = F->L[(size_t) il];
    const float prescale = 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads));
    gf::MvJob j[5];
    j[0] = {Ly.q_a.q, F->xq, F->x, F->qr_raw, nullptr, 1.0f, Ly.q_a.type, g.n_embd, g.q_lora};
    j[1] = {Ly.kv_a.q, F->xq, F->x, F->kv_raw, nullptr, 1.0f, Ly.kv_a.type, g.n_embd, g.kv_lora};
    j[2] = {Ly.idx_k, nullptr, F->x, F->ik_raw, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.idx_key};
    j[3] = {Ly.idx_gate, nullptr, F->x, F->ig_raw, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.idx_key};
    j[4] = {Ly.idx_proj, nullptr, F->x, F->iw, nullptr, prescale, gf::kTypeBF16, g.n_embd, g.idx_heads};
    if (!gf::mv(j, 5, s)) { err = "glm fast: dsa projections"; return false; }
    if (F->prof_on) F->mark("dsa_proj_mv");
    gf::DsaPrepArgs d;
    d.qr_raw = F->qr_raw;
    d.q_a_norm = Ly.q_a_norm;
    d.qr = F->qr;
    d.qr_q = F->qr_q;
    d.q_lora = g.q_lora;
    d.kv_raw = F->kv_raw;
    d.kv_norm = Ly.kv_a_norm;
    d.lat = (uint16_t*) (state_ + dsa_lat_[(size_t) il]);
    d.kv_lora = g.kv_lora;
    d.lat_q8 = lat_q8_;
    d.ik_raw = F->ik_raw;
    d.k_norm_w = Ly.k_norm_w;
    d.k_norm_b = Ly.k_norm_b;
    d.ik_cache = state_ + dsa_ik_[(size_t) il];
    d.ig_raw = F->ig_raw;
    d.ig_cache = state_ + dsa_ig_[(size_t) il];
    d.ape = Ly.ape;
    d.pooled = state_ + dsa_pool_[(size_t) il];
    d.idx_key = g.idx_key;
    d.kpool = g.idx_kpool;
    d.ring = ik_ring_;
    d.p = (int) p;
    d.eps = g.norm_eps;
    gf::dsa_prep(d, s);
    if (F->prof_on) F->mark("dsa_prep");
    gf::MvJob j2[2];
    j2[0] = {Ly.q_b.q, F->qr_q, F->qr, F->q, nullptr, 1.0f, Ly.q_b.type, g.q_lora, g.n_head * g.qk_nope};
    j2[1] = Ly.idx_q_b_x.q != nullptr
                ? gf::MvJob{Ly.idx_q_b_x.q, F->qr_q, F->qr, F->iq, nullptr, 1.0f, Ly.idx_q_b_x.type, g.q_lora,
                            g.idx_heads * g.idx_key}
                : gf::MvJob{Ly.idx_q_b, nullptr, F->qr, F->iq, nullptr, 1.0f, gf::kTypeBF16, g.q_lora,
                            g.idx_heads * g.idx_key};
    if (!gf::mv(j2, 2, s)) { err = "glm fast: dsa q"; return false; }
    if (F->prof_on) F->mark("dsa_q_mv");
    if (dumps) {
        fast_dump(s, "dsa_qr-" + std::to_string(il), F->qr, g.q_lora);
        fast_dump(s, "dsa_q-" + std::to_string(il), F->q, (int64_t) g.n_head * g.qk_nope);
        fast_dump(s, "dsa_iq-" + std::to_string(il), F->iq, (int64_t) g.idx_heads * g.idx_key);
    }
    const int pool_done = (int) ((p + 1) / g.idx_kpool);
    if (pool_done > 0)
        gf::dsa_score(F->iq, state_ + dsa_pool_[(size_t) il], F->iw, g.idx_key, g.idx_heads, pool_done,
                      F->score, s);
    const int top_pools = std::min(g.top_pools_max(), pool_done);
    const int n_sel = g.idx_kpool * top_pools + (g.idx_select_tail ? g.idx_kpool - 1 : 0);
    gf::dsa_select(F->score, pool_done, g.idx_kpool, top_pools, n_sel, (int) p, F->cells, s);
    if (F->prof_on) F->mark("dsa_score_select");
    gf::mla(F->q, Ly.k_b, Ly.v_b, (const uint16_t*) (state_ + dsa_lat_[(size_t) il]), F->cells, n_sel, g.n_head, g.qk_nope,
            g.kv_lora, g.v_head, F->attn_q, s, lat_q8_);
    if (F->prof_on) F->mark("mla");
    gf::MvJob o = {Ly.out.q, F->attn_q, nullptr, F->mixer, nullptr, 1.0f, Ly.out.type,
                   g.n_head * g.v_head, g.n_embd};
    if (!gf::mv(&o, 1, s)) { err = "glm fast: dsa output"; return false; }
    if (F->prof_on) F->mark("dsa_out_mv");
    return true;
}

// ---- one MoE FFN: F->x / F->xq -> F->ffn (routes, tiers, routed + shared experts)
// The NextN draft block leaves its experts that are not in VRAM out (no RAM pull, no disk wait): its draft only has
// to be a good guess - every token is the trunk's own - and its misses sat on the tail's critical path.
// STRATA_GLM_MTP_MISS=1 fetches them like the trunk; STRATA_GLM_MTP_KEEP=<n> fetches the missing ones among the
// route's n best-scored experts only (the route lists them in selection order) and leaves the rest out.  The default
// keeps 2: never slower than skipping all and drafts better (2x V100: 82-83% accepted against 76-81%, decode 28.0-28.2
// against 27.0-28.1 tok/s; RTX 3090 + 3060: 19.5 -> 20.6); KEEP=0 skips every missing one.
static int mtp_skip_from() {
    static const int v = [] {
        if (const char* k = getenv("STRATA_GLM_MTP_KEEP")) return std::max(0, std::min(gf::kMaxK, std::atoi(k)));
        return getenv("STRATA_GLM_MTP_MISS") != nullptr ? gf::kMaxK : 2;
    }();
    return v;
}

// ROUTE TAIL SKIP (Strata 0.1.42's STRATA_ROUTE_TAIL_SKIP, per route here): STRATA_GLM_TAIL_SKIP=<r> leaves a trunk
// route's experts from rank r on (0-based: r = 4 drops ranks 5 and 6 of deepseek4's 6) out when they are not in
// VRAM - no PCIe pull, no CPU lane, no disk wait; the route's other experts keep the router's weights.
// STRATA_GLM_TAIL_SKIP_W=<share> leaves out (also) those whose share of the route's probability is below it.  Not on
// the hash-routed layers (their ids come in hash order, not by score) nor the draft block (its own rule).  0 / >= k:
// off.  Off when the pool holds 90% or more of the experts (nothing to gain; STRATA_GLM_TAIL_SKIP_MAX_CACHED=<percent>
// moves that line).
void Glm5Model::fast_tail_skip_setup() {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    F->tail_skip = gf::kMaxK;   // (off: no route rank reaches it - a top-10 route's ranks 8 and 9 too)
    F->tail_skip_w = 0.0f;
    const char* v = getenv("STRATA_GLM_TAIL_SKIP");
    const char* vw = getenv("STRATA_GLM_TAIL_SKIP_W");
    int r = v != nullptr ? std::atoi(v) : 0;
    if (r <= 0 || r >= g.n_exp_used) r = gf::kMaxK;
    const float w = vw != nullptr ? std::max(0.0f, std::min(1.0f, (float) std::atof(vw))) : 0.0f;
    if (r == gf::kMaxK && w <= 0.0f) return;
    int64_t n = 0;
    for (int il = l0_; il < l1_; ++il)
        if (F->L[(size_t) il].moe) n += g.n_expert;
    const double held = n > 0 ? 100.0 * (double) F->pool_slots / (double) n : 100.0;
    const char* mc = getenv("STRATA_GLM_TAIL_SKIP_MAX_CACHED");
    const double max_cached = mc != nullptr ? std::atof(mc) : 90.0;
    if (held >= max_cached) {
        std::fprintf(stderr, "glm fast: CUDA%d route tail skip off: VRAM holds %.1f%% of the experts (>= %.0f%%)\n",
                     dev_, held, max_cached);
        return;
    }
    F->tail_skip = r;
    F->tail_skip_w = w;
    // STRATA_GLM_TAIL_SKIP_DISK=1: only the experts in NO tier (disk-only) are left out - a RAM-tier one costs the lane
    // ~0.1 ms, a disk one a ~4-10 ms read (Maya's STRATA_GLM_DISK_SKIP)
    const char* vd = getenv("STRATA_GLM_TAIL_SKIP_DISK");
    F->tail_skip_disk = vd != nullptr && std::atoi(vd) != 0;
    std::string rule;
    if (r < 8) rule = "at rank " + std::to_string(r + 1) + " of " + std::to_string(g.n_exp_used) + " and below";
    if (w > 0.0f) {
        char b[64];
        std::snprintf(b, sizeof b, "%sbelow %.3f of the route's weight", r < 8 ? " or " : "", (double) w);
        rule += b;
    }
    std::fprintf(stderr, "glm fast: CUDA%d route tail skip: experts %s %s are left out (VRAM holds %.1f%% of "
                         "the experts)\n", dev_, F->tail_skip_disk ? "only on disk" : "not in VRAM", rule.c_str(), held);
}

int Glm5Model::fast_skip_from(int il) const {
    if (il == mtp_il_) return mtp_skip_from();
    return fast_->L[(size_t) il].tid2eid != nullptr ? gf::kMaxK : fast_->tail_skip;
}

float Glm5Model::fast_skip_w(int il) const {
    return il == mtp_il_ || fast_->L[(size_t) il].tid2eid != nullptr ? 0.0f : fast_->tail_skip_w;
}

bool Glm5Model::fast_moe(int il, bool& pf_pending, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int FFs = g.n_ff_exp * g.n_shared;
    gf::MvJob j[gf::kMaxMvJobs];
    // The shared expert's gate/up GEMV runs after the route, not beside the router: the route - and with it the CPU /
    // iGPU lane's request - goes out a GEMV earlier (DS4's shared gate/up: 18 MB of Q8_0, ~12 us a layer), and the
    // shared expert then runs while the lane computes.  RTX 5090 + Radeon 8060S lane, DS4 1,024-token answers, two
    // A/B rounds: prose 79.0 -> 81.3 tok/s, code 78.3 -> 81.4; KL unchanged (the same sums, in another order).
    // STRATA_GLM_SH_LATE=0: beside the router, as before.  The shared gate/up take the F32 input too (xf): an EXL3
    // matrix reads it, the ggml types their q8_1
    static const bool sh_late = [] {
        const char* v = getenv("STRATA_GLM_SH_LATE");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const bool late = sh_late && FFs > 0;
    int nj = late ? 1 : 3;
    j[0] = {Ly.router, nullptr, F->x, F->rlog, nullptr, 1.0f, Ly.router_type, g.n_embd, g.n_expert};
    if (!late) {
        j[1] = {Ly.sh_gate.q, F->xq, F->x, F->sh_g, nullptr, 1.0f, Ly.sh_gate.type, g.n_embd, FFs};
        j[2] = {Ly.sh_up.q, F->xq, F->x, F->sh_u, nullptr, 1.0f, Ly.sh_up.type, g.n_embd, FFs};
    }
    // qwen35moe: the shared expert's gate logit (sigmoid(w . x) scales its output, below)
    if (g.qwen35() && Ly.sh_gate_w != nullptr)
        j[nj++] = {Ly.sh_gate_w, nullptr, F->x, F->q35_shg, nullptr, 1.0f, gf::kTypeF32, g.n_embd, 1};
    // the next layer's router on THIS layer's FFN input: its experts, one layer early
    // (only with prefetch on: the prediction costs a router GEMV and a second top-k per layer)
    const bool pred = F->max_pf > 0 && il + 1 < l1_ && F->L[(size_t) il + 1].moe;
    if (pred)
        j[nj++] = {F->L[(size_t) il + 1].router, nullptr, F->x, F->plog, nullptr, 1.0f, F->L[(size_t) il + 1].router_type, g.n_embd,
                   g.n_expert};
    // LOOKAHEAD: the next layers' routers on this input too (their disk-only experts are read ahead)
    int n_ah = 0;
    const float* ah_bias[gf::kAhead] = {nullptr, nullptr, nullptr, nullptr};
    for (int d = 0; d < F->n_ahead && il + 1 + d < l1_ && F->L[(size_t) il + 1 + d].moe && nj < gf::kMaxMvJobs; ++d) {
        const auto& La = F->L[(size_t) il + 1 + d];
        j[nj++] = {La.router, nullptr, F->x, F->alog + (size_t) d * g.n_expert, nullptr, 1.0f, La.router_type, g.n_embd,
                   g.n_expert};
        ah_bias[d] = F->cur_img && La.router_bias_vl ? La.router_bias_vl : La.router_bias;   // (an image row: exp_probs_b_vl)
        ++n_ah;
    }
    if (!gf::mv(j, nj, s)) { err = "glm fast: router / shared expert"; return false; }
    if (F->prof_on) F->mark("router_shexp_mv");
    // the prefetch list of THIS route lives in the buffer of this layer's parity (the side stream may still
    // read the previous layer's list while this route writes)
    gf::MoeDev md = F->md;
    if (F->expert_helper && F->expert_helper->active && !F->expert_helper->failed)
        md.helper_owned = F->expert_helper->owned_d;
    md.pf_src = F->pf_buf[il & 1];
    md.pf_dst = F->pf_buf[il & 1] + 8;
    md.pf_n = F->pf_n_buf[il & 1];
    // the CPU lane splits this layer's RAM-tier experts when the CPU has a dot product for its types
    const bool lane = F->cpu_plan != 0ull && F->cpu_fmt[(size_t) il].n_ff > 0;
    // STRATA_GLM_PROMOTE_MIN=<n>: a fetched expert is kept in VRAM only when its aged route count clears n.
    // Unset or <= 1: the old rule (keep it whenever a spare is free).
    static const int promote_min = [] {
        const char* v = getenv("STRATA_GLM_PROMOTE_MIN");
        return v ? std::max(0, std::atoi(v)) : 0;
    }();
    // STRATA_GLM_DISK_LANE=1: a disk-only expert is computed by the CPU / iGPU lane straight from the RAM slot it was
    // read into (it stays in the RAM tier) instead of being pulled over PCIe into a VRAM spare (-1 tells the route)
    const bool dlane = lane && F->disk_lane;
    // STRATA_GLM_PREFETCH_RANK=<n>: prefetch only the prediction's first n ranks (unset: any of its top k)
    static const int pf_rank = [] {
        const char* v = getenv("STRATA_GLM_PREFETCH_RANK");
        return v != nullptr ? std::max(1, std::atoi(v)) : 0;
    }();
    // deepseek41: an image row selects its experts with exp_probs_b_vl (the prediction and lookahead of the next layers too)
    const bool vl = F->cur_img && Ly.router_bias_vl != nullptr;
    gf::moe_route(F->rlog, vl ? Ly.router_bias_vl : Ly.router_bias, g.n_expert, g.n_exp_used, g.w_scale, g.norm_w != 0, il, F->x,
                  g.n_embd, md, late ? nullptr : F->sh_g, late ? nullptr : F->sh_u, g.swiglu_shexp, late ? 0 : FFs,
                  F->sh_hq, s,
                  pred ? F->plog : nullptr, pred ? (F->cur_img && F->L[(size_t) il + 1].router_bias_vl ? F->L[(size_t) il + 1].router_bias_vl
                                                                    : F->L[(size_t) il + 1].router_bias)
                       : nullptr,
                  pred ? F->max_pf : 0, n_ah > 0 ? F->alog : nullptr, ah_bias, n_ah, fast_skip_from(il),
                  lane ? F->cpu_plan : 0ull, dlane ? -1 : promote_min, pf_rank, g.gating,
                  Ly.tid2eid != nullptr ? Ly.tid2eid + (size_t) F->cur_tok * g.n_exp_used : nullptr, fast_skip_w(il),
                  F->tail_skip_disk ? 1 : 0, lane && !Ly.exl3 && F->igpu_experts_q != nullptr ? F->xq : nullptr);
    if (F->prof_on) F->mark("moe_route");
    ++F->expected;
    // the side stream copies the next layer's predicted experts while this layer computes.  STRATA_GLM_PREFETCH_AT:
    // when that copy starts - "route" (the default: at once, beside this layer's own PCIe fetch and CPU lane),
    // "fetch" (after this layer's own fetch: the link is free) or "cpu" (after this layer's CPU-lane answer: RAM is
    // free too; the copy then runs beside the next layer's attention).  The prefetch lists are double-buffered by
    // layer parity and the next layer waits for this copy before its experts run, so any of the three is safe.
    static const int pf_at = [] {
        const char* v = getenv("STRATA_GLM_PREFETCH_AT");
        return v == nullptr ? 0 : std::strcmp(v, "fetch") == 0 ? 1 : std::strcmp(v, "cpu") == 0 ? 2 : 0;
    }();
    const bool pf = pred && F->max_pf > 0;
    const auto issue_prefetch = [&] {
        cudaEventRecord(F->ev_pred, s);
        cudaStreamWaitEvent(F->ps, F->ev_pred, 0);
        gf::moe_prefetch(md, F->L[(size_t) il + 1].blob, F->ps);
        cudaEventRecord(F->ev_pf, F->ps);
    };
    if (pf && pf_at == 0) issue_prefetch();
    if (late) {
        gf::MvJob sj[2];
        sj[0] = {Ly.sh_gate.q, F->xq, F->x, F->sh_g, nullptr, 1.0f, Ly.sh_gate.type, g.n_embd, FFs};
        sj[1] = {Ly.sh_up.q, F->xq, F->x, F->sh_u, nullptr, 1.0f, Ly.sh_up.type, g.n_embd, FFs};
        if (!gf::mv(sj, 2, s)) { err = "glm fast: shared expert"; return false; }
        gf::sh_swiglu(F->sh_g, F->sh_u, g.swiglu_shexp, FFs, F->sh_hq, s);
        if (F->prof_on) F->mark("shexp_mv");
    }
    // disk-only experts (rare once warm) park the device until the host has read them; everything not
    // in VRAM is then pulled over PCIe into its slot, and all 8 run from VRAM.  STRATA_GLM_WAIT_FETCH (default on with
    // HIP; 0 off, 1 on CUDA too - not measured there): the wait and the copies in one launch (gf::moe_wait_fetch)
    static const bool wait_fetch = [] {
        const char* v = getenv("STRATA_GLM_WAIT_FETCH");
#if defined(STRATA_USE_HIP)
        return v == nullptr || std::atoi(v) != 0;
#else
        return v != nullptr && std::atoi(v) != 0;
#endif
    }();
    if (wait_fetch && !F->prof_on) {
        if (F->stream_fetch)
            gf::moe_wait_fetch(md, g.n_embd, g.n_exp_used, Ly.blob, s, Ly.gu_bytes, Ly.dn_bytes, F->read_chunks);
        else
            gf::moe_wait_fetch(md, g.n_embd, g.n_exp_used, Ly.blob, s);
    } else {
        gf::moe_wait(md, g.n_embd, s);
        if (F->prof_on) F->mark("moe_wait");
        if (F->stream_fetch) gf::moe_fetch(md, g.n_exp_used, Ly.blob, s, Ly.gu_bytes, Ly.dn_bytes, F->read_chunks);
        else gf::moe_fetch(md, g.n_exp_used, Ly.blob, s);
        if (F->prof_on) F->mark("moe_fetch");
    }
    if (pf && pf_at == 1) issue_prefetch();
    // this layer's own prefetched experts (issued one layer ago) must have landed before they are read
    if (pf_pending) cudaStreamWaitEvent(s, F->ev_pf_prev, 0);
    pf_pending = false;
    if (Ly.exl3) {
        // EXL3: the shared expert's down first (its h is moe_route's q8_1), then the routed experts' gate/up/swiglu
        // and down over the F32 FFN input, summed in plan order with the CPU lane's part and the shared expert
        if (Ly.sh_down.q != nullptr) {
            const gf::MvJob sd = {Ly.sh_down.q, F->sh_hq, nullptr, F->sh_out, nullptr, 1.0f, Ly.sh_down.type, FFs,
                                  g.n_embd};
            if (!gf::mv(&sd, 1, s)) { err = "glm fast: shared expert down"; return false; }
        }
        strata::kernels::exl3::MoeArgs xa;
        xa.plan_ptr = md.plan_ptr;
        xa.plan_w = md.plan_w;
        xa.cpu_part = md.cpu_part;
        xa.cpu_flag = md.cpu_flag;
        xa.lay = Ly.xl;
        xa.k = g.n_exp_used;
        xa.n_embd = g.n_embd;
        xa.n_ff = g.n_ff_exp;
        xa.limit = g.swiglu_exp;
        strata::kernels::exl3::moe_gate_up(xa, F->x, nullptr, s);
        if (F->prof_on) F->mark("moe_gate_up");
        strata::kernels::exl3::moe_down(xa, nullptr, Ly.sh_down.q != nullptr ? F->sh_out : nullptr, F->ffn, s);
        if (F->prof_on) F->mark("moe_down");
    } else {
    // (the shared expert's down rides the gate/up launch: sh_out)
    gf::moe_gate_up(Ly.gu_type, md, g.n_exp_used, g.n_embd, g.n_ff_exp, g.swiglu_exp, F->xq, F->hq, Ly.sh_down.q,
                    Ly.sh_down.type, F->sh_hq, FFs, F->sh_out, s);
    if (Ly.sh_down.q != nullptr) abl_apply(il, 1, F->sh_out, 1, g.n_embd, s);
    if (F->prof_on) F->mark("moe_gate_up");
    if (g.qwen35() && Ly.sh_down.q != nullptr && Ly.sh_gate_w != nullptr)   // the shared expert behind its sigmoid gate
        strata::kernels::q35::scale_sigmoid(F->sh_out, g.n_embd, F->q35_shg, s);
    gf::moe_down(Ly.d_type, md, g.n_exp_used, g.n_embd, g.n_ff_exp, Ly.down_off, F->hq,
                 Ly.sh_down.q != nullptr ? F->sh_out : nullptr, F->ffn, s);
    if (F->prof_on) F->mark("moe_down");
    }
    // the CPU lane's experts last: the device's own down rows ran while the CPU worked.  Deferred (F->defer): the
    // layer before's part is added here instead, this layer's by the next (the last trunk layer adds both; in a
    // layer split each part's last layer, so the hop carries a complete residual and the next part starts clean)
    // STRATA_GLM_HC_CPU (default on with HIP; 0 off, 1 on CUDA too - not measured there): deepseek4's answer is added
    // by the next layer's attention-side hc3 read instead (gf::HcArgs::cpu_seq) - a launch less a layer, and the
    // answer's 16 KB read by hc3's 32 blocks instead of the wait kernel's 8.  The same float add: bit-identical
    static const bool hc_cpu_env = [] {
        const char* v = getenv("STRATA_GLM_HC_CPU");
#if defined(STRATA_USE_HIP)
        return v == nullptr || std::atoi(v) != 0;
#else
        return v != nullptr && std::atoi(v) != 0;
#endif
    }();
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    // (deepseek41: not before an Engram layer - its separate hc_post writes the FFN's output before the next read)
    const bool hc_cpu = lane && hc_cpu_env && g.dsv4() && !F->defer && !Ly.mtp && !Ly.exl3 && !F->prof_on && !abl_full() &&
                        !dumps && il + 1 < l1_ && !(g.v41 && F->L[(size_t) il + 1].eng >= 0) && !(pf && pf_at == 2) &&
                        !F->kda_graph_on && gf::hc_fuses_cpu(g.n_embd);
    if (F->defer && !Ly.mtp) {
        if (il == l1_ - 1) {   // the trunk's (or this part's) last layer: nothing after it here to defer into
            gf::moe_cpu_wait(md, g.n_embd, F->ffn, s);
            if (F->prof_on) F->mark("moe_cpu_wait");
        } else if (lane) {            // its write weights, for the add after the next layer's attention
            gf::copy_floats(F->post_defer, F->post, 8, s);
        }
    } else if (hc_cpu) {
        F->hc_cpu_on = true;
        F->hc_cpu_seq = md.cpu_seq;
        F->hc_cpu_ans = md.cpu_ans;
    } else if (lane) {
        gf::moe_cpu_wait(md, g.n_embd, F->ffn, s);
        if (F->prof_on) F->mark("moe_cpu_wait");
    }
    if (pf && pf_at == 2) issue_prefetch();
    if (pf) {
        std::swap(F->ev_pf, F->ev_pf_prev);   // the next layer waits on THIS layer's prefetch
        pf_pending = true;
    }
    // the paced promotions' next bytes: across the link while the next layer's mixer runs (no zero-copy reads there)
    if (hc_cpu) F->bg_after_hc = true;
    else if (!F->kda_graph_on) F->bg_drain(s, F->bg_pace);
    return true;
}

// STRATA_DS4_DUMP_FEAT=<file> (debug, the DSpark probe): append the residual streams a layer of
// STRATA_DS4_DUMP_FEAT_LAYERS (default 40,41,42: the drafter's target layers) leaves - hc x n_embd floats a layer, in
// the list's order, every position the token path runs
void Glm5Model::fast_dump_feat(int il, const float* R) {
    static const char* feat_file = getenv("STRATA_DS4_DUMP_FEAT");
    if (feat_file == nullptr || !feat_file[0] || !g_.dsv4()) return;
    static const std::vector<int> feat_layers = [] {
        std::vector<int> v;
        const char* l = getenv("STRATA_DS4_DUMP_FEAT_LAYERS");
        for (const char* q = l != nullptr ? l : "40,41,42"; *q;) {
            char* e = nullptr;
            const long x = std::strtol(q, &e, 10);
            if (e == q) break;
            v.push_back((int) x);
            q = *e == ',' ? e + 1 : e;
        }
        return v;
    }();
    for (int fl : feat_layers)
        if (fl == il) {
            cudaStreamSynchronize(fast_->cs);
            std::vector<float> fb((size_t) g_.hc * g_.n_embd);
            cudaMemcpy(fb.data(), R, fb.size() * sizeof(float), cudaMemcpyDeviceToHost);
            if (FILE* f = std::fopen(feat_file, "ab")) {
                std::fwrite(fb.data(), sizeof(float), fb.size(), f);
                std::fclose(f);
            }
        }
}

bool Glm5Model::fast_layers(int64_t p, bool hop_in, std::string& err) {
    (void) hop_in;
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    bool pf_pending = false;
    float* Rc = state_;
    float* Ro = state_ + (int64_t) g.hc * g.n_embd;
    const char* graph_opt = getenv("STRATA_GLM_KDA_GRAPH");
    const bool graphs = graph_opt && std::atoi(graph_opt) != 0 && n_parts_ == 1 && !hop_in &&
                        mtp_il_ < 0 && !F->prof_on && !dumps && F->max_pf == 0 && F->n_ahead == 0 &&
                        getenv("GLM_DUMP_H") == nullptr;
    F->kda_graph_on = graphs;
    if (graphs) F->bg_drain(nullptr, SIZE_MAX);   // (paced copies are not issued from inside a capture)
    F->hc_cpu_on = F->bg_after_hc = false;
    if (g.qwen35()) return fast_qwen35_layers(p, err);   // (plain residuals: qwen35_fast_path.cu)
    if (F->prof_on) F->mark("start");
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        const auto body = [&]() -> bool {
        // deepseek41: Engram before the layer's read - the previous FFN's write alone first, then the add into the
        // streams, then a read without a write
        bool wrote = il != l0_;
        if (g.v41 && Ly.eng >= 0) {
            if (il != l0_) {
                gf::hc_post(F->ffn, Rc, F->post, F->comb, g.n_embd, Ro, s);
                std::swap(Rc, Ro);
                wrote = false;
            }
            if (!dsv41_engram_add(il, Rc, err)) return false;
            abl_res(Rc, g.hc, g.n_embd, s);   // (Engram's rows are a residual write too)
        }
        // ---- the attention-side read (fused with the previous FFN's write)
        gf::HcArgs h;
        h.block_out = wrote ? F->ffn : nullptr;
        h.R_old = Rc;
        h.R_new = Ro;
        h.post_in = F->post;
        h.comb_in = F->comb;
        h.pre = F->pre;
        h.post = F->post;
        h.comb = F->comb;
        h.w_fn = Ly.hc_attn_fn;
        h.w_scale = Ly.hc_attn_scale;
        h.w_base = Ly.hc_attn_base;
        h.norm_w = Ly.attn_norm;
        h.norm_eps = g.norm_eps;
        h.hc_eps = g.hc_eps;
        h.iters = g.sinkhorn_iters;
        h.n_embd = g.n_embd;
        h.x = F->x;
        h.xq = F->xq;
        h.part = F->part;
        h.counter = F->counter;
        if (g.v41) {   // the collapse with the previous sublayer's pre (layer 0: the one-hot fast_token sets)
            h.pre = F->v41_pre[0];
            h.pre_use = F->v41_pre[1];
        }
        if (F->hc_cpu_on && wrote) {   // the last MoE's CPU-lane answer, added by this read (fast_moe)
            h.cpu_seq = F->hc_cpu_seq;
            h.cpu_ans = F->hc_cpu_ans;
            F->hc_cpu_on = false;
        }
        gf::hc(h, s);
        h.cpu_seq = nullptr;
        h.cpu_ans = nullptr;
        if (F->bg_after_hc) {
            F->bg_after_hc = false;
            F->bg_drain(s, F->bg_pace);
        }
        if (F->prof_on) F->mark("hc_attn");
        if (wrote) std::swap(Rc, Ro);
        abl_cap(il, Rc, g.hc, s);
        if (il > l0_) fast_dump_feat(il - 1, Rc);   // (the layer before's output: the residual entering this one)
        if (dumps) {
            const std::string Ls = std::to_string(il);
            if (il != l0_) fast_dump(s, "l_out-" + std::to_string(il - 1), Rc, (int64_t) g.hc * g.n_embd);
            fast_dump(s, "attn_norm-" + Ls, F->x, g.n_embd);
        }

        if (g.v41) {
            if (!fast_dsv41_attn(il, p, err)) return false;
        } else if (g.dsv4()) {
            if (!fast_dsv4_attn(il, p, err)) return false;
        } else if (Ly.recr) {
            const int DI = g.d_inner();
            gf::MvJob j[6];
            j[0] = {Ly.q.q, F->xq, F->x, F->proj[0], nullptr, 1.0f, Ly.q.type, g.n_embd, DI};
            j[1] = {Ly.k.q, F->xq, F->x, F->proj[1], nullptr, 1.0f, Ly.k.type, g.n_embd, DI};
            j[2] = {Ly.v.q, F->xq, F->x, F->proj[2], nullptr, 1.0f, Ly.v.type, g.n_embd, DI};
            j[3] = {Ly.f_a, nullptr, F->x, F->fa, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.kda_head_dim};
            j[4] = {Ly.g_a, nullptr, F->x, F->ga, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.kda_head_dim};
            j[5] = {Ly.beta, nullptr, F->x, F->beta, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.n_head};
            if (!gf::mv(j, 6, s)) { err = "glm fast: kda projections"; return false; }
            if (F->prof_on) F->mark("kda_proj_mv");
            gf::KdaPrepArgs k;
            for (int i = 0; i < 3; ++i) {
                k.proj[i] = F->proj[i];
                k.conv_w[i] = Ly.conv[i];
                k.out[i] = F->conv[i];
            }
            k.conv_state = state_ + kda_conv_[(size_t) il];
            k.fa = F->fa;
            k.ga = F->ga;
            k.f_b = Ly.f_b;
            k.g_b = Ly.g_b;
            k.dt_bias = Ly.dt_bias;
            k.ssm_a = Ly.ssm_a;
            k.lower_bound = g.kda_lb;
            k.g1 = F->g1;
            k.g2 = F->g2;
            k.n_head = g.n_head;
            k.head_dim = g.kda_head_dim;
            k.d_conv = g.d_conv;
            gf::kda_prep(k, s);
            if (F->prof_on) F->mark("kda_prep");
            gf::kda_rec(F->conv[0], F->conv[1], F->conv[2], F->g1, F->beta, state_ + kda_S_[(size_t) il], F->g2,
                        Ly.ssm_norm, g.norm_eps, g.n_head, g.kda_head_dim, F->gated_q, s);
            if (F->prof_on) F->mark("kda_rec");
            gf::MvJob o = {Ly.out.q, F->gated_q, nullptr, F->mixer, nullptr, 1.0f, Ly.out.type, DI, g.n_embd};
            if (!gf::mv(&o, 1, s)) { err = "glm fast: kda output"; return false; }
            if (F->prof_on) F->mark("kda_out_mv");
        } else {
            if (!fast_dsa(il, p, err)) return false;
        }

        // expert deferral: the layer before's CPU-lane part into the streams now (its own write weights), so this
        // layer's router and experts see it; the attention above did not
        if (F->defer && Ly.moe && !Ly.mtp && il > g.dense_lead && il > l0_) {
            gf::hc_defer_add(F->md, F->post_defer, g.n_embd, Rc, s);
            if (F->prof_on) F->mark("hc_defer_add");
        }
        // DSpark's features: the layer before's complete output (its deferred part added above)
        if (dsp_ != nullptr && il > l0_) dspark_capture(il - 1, Rc, 0, s);
        // ---- the FFN-side read (fused with the mixer's write)
        abl_out(il, 0, F->mixer, 1, g.n_embd, s);
        h.block_out = F->mixer;
        h.R_old = Rc;
        h.R_new = Ro;
        h.w_fn = Ly.hc_ffn_fn;
        h.w_scale = Ly.hc_ffn_scale;
        h.w_base = Ly.hc_ffn_base;
        h.norm_w = Ly.ffn_norm;
        if (g.v41) {
            h.pre = F->v41_pre[1];
            h.pre_use = F->v41_pre[0];
        }
        gf::hc(h, s);
        if (F->prof_on) F->mark("hc_ffn");
        std::swap(Rc, Ro);
        if (dumps) {
            const std::string Ls = std::to_string(il);
            fast_dump(s, "hc_attn_post-" + Ls, Rc, (int64_t) g.hc * g.n_embd);
            fast_dump(s, "ffn_norm-" + Ls, F->x, g.n_embd);
            fast_dump(s, "mixer-" + Ls, F->mixer, g.n_embd);
        }

        if (!Ly.moe) {
            gf::MvJob j[2];
            j[0] = {Ly.ffn_gate.q, F->xq, F->x, F->dg, nullptr, 1.0f, Ly.ffn_gate.type, g.n_embd, g.n_ff_dense};
            j[1] = {Ly.ffn_up.q, F->xq, F->x, F->du, nullptr, 1.0f, Ly.ffn_up.type, g.n_embd, g.n_ff_dense};
            if (!gf::mv(j, 2, s)) { err = "glm fast: dense ffn"; return false; }
            if (F->prof_on) F->mark("dense_gu_mv");
            gf::swiglu_q8(F->dg, F->du, g.swiglu_shexp, g.n_ff_dense, F->dhq, s);
            if (F->prof_on) F->mark("dense_swiglu");
            gf::MvJob dn = {Ly.ffn_down.q, F->dhq, nullptr, F->ffn, nullptr, 1.0f, Ly.ffn_down.type, g.n_ff_dense,
                            g.n_embd};
            if (!gf::mv(&dn, 1, s)) { err = "glm fast: dense ffn down"; return false; }
            if (F->prof_on) F->mark("dense_down_mv");
        } else {
            if (!fast_moe(il, pf_pending, err)) return false;
        }
        abl_out(il, 1, F->ffn, 1, g.n_embd, s);   // (complete: an edit keeps the CPU lane's answer out of the read)
        if (dumps) fast_dump(s, "ffn_out-" + std::to_string(il), F->ffn, g.n_embd);
        return true;
        };
        if (graphs && Ly.recr) {
            bool replayed = false;
            const glmfast::LayerGraphs::Key key{F->cpu_plan, Rc, Ro};
            if (!F->kda_graphs.enqueue(il, key, s, F->expected, Ly.moe ? 1 : 0, body, replayed, err))
                return false;
            // Capture executes the host pointer swaps; replay only executes device
            // work. Maintain the identical residual convention in both cases.
            if (replayed) {
                if (il != l0_) std::swap(Rc, Ro);
                std::swap(Rc, Ro);
            }
        } else if (!body()) {
            return false;
        }
    }
    // the last layer's write half: R (in the OTHER buffer) = post x ffn + comb . R  (fast_moe leaves no answer to a
    // hc3 read past the part's last layer; the check is for safety)
    if (F->hc_cpu_on) {
        gf::MoeDev d;
        d.cpu_seq = (unsigned int*) F->hc_cpu_seq;
        d.cpu_ans = F->hc_cpu_ans;
        gf::moe_cpu_wait(d, g.n_embd, F->ffn, s);
        F->hc_cpu_on = false;
    }
    if (F->bg_after_hc) {
        F->bg_after_hc = false;
        F->bg_drain(s, F->bg_pace);
    }
    gf::hc_post(F->ffn, Rc, F->post, F->comb, g.n_embd, Ro, s);
    if (l1_ == g.n_layers) abl_cap(g.n_layers, Ro, g.hc, s);
    if (F->prof_on) F->mark("hc_post_end");
    fast_dump_feat(l1_ - 1, Ro);
    if (dsp_ != nullptr) dspark_capture(l1_ - 1, Ro, 0, s);
    // keep the convention "the residual lives at state_" for the hop and the head: copy back
    if (Ro != state_)
        cudaMemcpyAsync(state_, Ro, (size_t) g.hc * g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, s);
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { err = std::string("glm fast: ") + cudaGetErrorString(e); return false; } }
    return true;
}

bool Glm5Model::fast_token(int32_t token, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    dspark_settle();   // (a window's kept columns before this token's features take column 0)
    const int64_t p = pos_++;
    const bool gp = F->gap_on && split_next_ == nullptr;
    if (gp) {
        cudaEventRecord(F->ev_g0, F->cs);   // the first command of this token: when it reaches the GPU
        if (F->gap_live)
            F->gap_host += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - F->t_ready).count();
    }
    // between tokens (every device idle): finished promotions go live in the device tables
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (!m->fast_boundary(err)) return false;
    }
    cudaSetDevice(dev_);
    const auto te0 = std::chrono::steady_clock::now();
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        m->fast_->cur_tok = token;   // hash routing
        m->fast_->cur_img = is_image_pos(p);   // deepseek41: exp_probs_b_vl for an image row (false: no image rows set)
    }
    // ---- the embedding row (host-dequantized from the shard mapping) -> the 4 streams
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm fast: no embedding dequantizer";
        return false;
    }
    const float* img = image_row(p);
    if (F->emb_tab_d != nullptr && img == nullptr) {
        // the row from the pinned table, gathered by the kernel (the id read host-mapped): no copy engine
        F->tok_in_h[0] = token;
        gf::embed_rows_q(pack_emb_type_, F->emb_tab_d, F->tok_in_d, 1, g.n_embd, F->emb, F->cs);
    } else {
        tt->to_float(pack_emb_src_ + (size_t) token * ggml_row_size((ggml_type) pack_emb_type_, g.n_embd), F->emb_h,
                     g.n_embd);
        if (img) std::memcpy(F->emb_h, img, (size_t) g.n_embd * sizeof(float));   // an image
        cudaMemcpyAsync(F->emb, F->emb_h, (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice, F->cs);
    }
    abl_res(F->emb, 1, g.n_embd, F->cs);
    if (g.qwen35() && !g.q4x)   // one residual stream (qwen4exp: hc copies, as the others)
        cudaMemcpyAsync(state_, F->emb, (size_t) g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, F->cs);
    else
        gf::embed_streams(F->emb, state_, g.n_embd, F->cs);
    if (g.q4x && !q4x_ple_issue(token, p, err)) return false;   // (the PLE rows' SSD reads, collected before layer 1)
    if (g.v41) {
        static const float onehot[4] = {1.0f, 0.0f, 0.0f, 0.0f};   // layer 0's attention collapses stream 0
        cudaMemcpyAsync(F->v41_pre[1], onehot, sizeof(onehot), cudaMemcpyHostToDevice, F->cs);
        if (!dsv41_engram_token(token, err)) return false;
    }
    if (gp) {
        cudaEventRecord(F->ev_gs, F->cs);
        F->gap_emb += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - te0).count();
    }
    if (q35_mtp_on() && !q35_mtp_pend_commit(token, err)) return false;   // (qwen35moe's drafter: the last row)
    if (!fast_layers(p, false, err)) return false;
    if (dsp_ != nullptr && !dspark_commit(p, 1, false, F->cs)) {
        err = "dspark: the context commit";
        return false;
    }
    // each later part of a split: the residual crosses over through the previous part's pinned hop buffer, the
    // copy ordered on the devices by an event (the host never waits between the parts)
    Glm5Model* tail = this;
    for (Glm5Model* B = split_next_.get(); B != nullptr; B = B->split_next_.get()) {
        FastState* FA = tail->fast_;
        FastState* FB = B->fast_;
        const size_t hop = (size_t) g.hc * g.n_embd * sizeof(float);
        cudaSetDevice(tail->dev_);
        cudaMemcpyAsync(FA->hop_h, tail->state_, hop, cudaMemcpyDeviceToHost, FA->cs);
        cudaEventRecord(FA->ev_hop, FA->cs);
        cudaSetDevice(B->dev_);
        B->pos_ = pos_;
        cudaStreamWaitEvent(FB->cs, FA->ev_hop, 0);
        cudaMemcpyAsync(B->state_, FA->hop_h, hop, cudaMemcpyHostToDevice, FB->cs);
        if (!B->fast_layers(p, true, err)) return false;
        tail = B;
    }
    FastState* FT = tail->fast_;
    cudaSetDevice(tail->dev_);
    if (g.qwen35()) {
        // (fast_qwen35_layers left head_x / head_xq); the drafter's row of p waits for the next token
        if (q35_mtp_on()) q35_mtp_pend_set(p, g.q4x ? tail->state_ : FT->head_x);   // (qwen4exp: the final residual)
    } else if (g.v41) {
        if (!tail->dsv41_head(err)) return false;
    } else if (g.dsv4()) {
        if (!tail->dsv4_head(err)) return false;
    } else {
        gf::head_prep(tail->state_, tail->w_.at("output_norm.weight"), g.norm_eps, g.n_embd, FT->head_x, FT->head_xq,
                      FT->cs);
    }
    // GLM_DUMP_H=<file> (debug): append every position's final hidden state (output_norm of the stream mean - what the
    // NextN/MTP block reads) as n_embd floats
    static const char* dh = getenv("GLM_DUMP_H");
    if (dh != nullptr && dh[0]) {
        cudaStreamSynchronize(FT->cs);
        std::vector<float> hb((size_t) g.n_embd);
        cudaMemcpy(hb.data(), FT->head_x, hb.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (FILE* f = std::fopen(dh, "ab")) {
            std::fwrite(hb.data(), sizeof(float), hb.size(), f);
            std::fclose(f);
        }
    }
    const WSlot& ow = tail->ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, FT->head_xq, FT->head_x, tail->sc_ + tail->sc_logits, nullptr, 1.0f, ow.type, g.n_embd,
                   g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!gf::mv(&o, 1, FT->cs)) {
        err = "glm fast: output head (type " + std::to_string(ow.type) + ")";
        return false;
    }
    if (FT->zero_copy) {
        gf::argmax(tail->sc_ + tail->sc_logits, g.n_vocab, FT->tok_hd, FT->cs);   // straight into the host-mapped word
    } else {
        gf::argmax(tail->sc_ + tail->sc_logits, g.n_vocab, FT->tok, FT->cs);
        cudaMemcpyAsync(FT->tok_h, FT->tok, sizeof(int), cudaMemcpyDeviceToHost, FT->cs);
    }
    if (gp) cudaEventRecord(F->ev_ge[F->gap_par], F->cs);
    cudaEventRecord(FT->ev_done, FT->cs);
    // wait for the token (the service threads answer misses meanwhile)
    const auto tw = std::chrono::steady_clock::now();
    bool warned = false;
    cudaError_t done;
    while ((done = cudaEventQuery(FT->ev_done)) == cudaErrorNotReady) {
        if (F->token_spin) cpu_relax(); else std::this_thread::yield();
        if (!warned && std::chrono::steady_clock::now() - tw > std::chrono::seconds(10)) {
            warned = true;
            for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
                const FastState* Fm = m->fast_;
                unsigned int last = 0;
                for (int i = 0; i < gf::kRingSize; ++i) last = std::max(last, (unsigned int) Fm->ring_h[i].seq);
                std::fprintf(stderr, "glm fast: token %lld waiting > 10 s - CUDA%d last route seq %u, last answer %u\n",
                             (long long) p, m->dev_, last, (unsigned int) Fm->resp_h->seq);
            }
        }
    }
    if (!glmfast::cuda_ok(done, "glm token completion", err)) return false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (!m->fast_->route_ok(err)) return false;
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm fast: ") + cudaGetErrorString(e);
        return false;
    }
    if (gf::launch_errors() > 0) {
        err = "glm fast: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return false;
    }
    last_tok_ = FT->tok_h[0];
    if (gp) fast_gap_mark(tw);
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ && m->fast_->prof_on) {
            cudaSetDevice(m->dev_);
            m->fast_->collect();
        }
    cudaSetDevice(dev_);
    return true;
}

// ---------------------------------------------------------------- the NextN draft block
// At position p (whose final hidden state the head left in head_x): eh_proj([enorm(emb(next)), hnorm(h_p)]) -> a DSA
// mixer and a MoE FFN with plain pre-norm residuals (the trunk's own kernels: fast_dsa / fast_moe on the block's layer
// index; its caches at p) -> shared_head_norm -> the output head -> argmax into mtp_tok (mtp_tok_h after ev_mtp).
bool Glm5Model::fast_mtp(int64_t p, int32_t next_tok, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int il = mtp_il_;
    if (F == nullptr || il < 0) {
        err = "glm mtp: no draft block on this half";
        return false;
    }
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd;
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm mtp: no embedding dequantizer";
        return false;
    }
    if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "glm mtp embedding sync", err) || !F->route_ok(err))
        return false;   // emb_h may still feed an earlier copy
    tt->to_float(pack_emb_src_ + (size_t) next_tok * ggml_row_size((ggml_type) pack_emb_type_, E), F->emb_h, E);
    cudaMemcpyAsync(F->emb, F->emb_h, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s);
    gf::mtp_in(F->emb, F->head_x, Ly.enorm, Ly.hnorm, g.norm_eps, E, F->mtp_catq, s);
    gf::MvJob eh = {Ly.eh.q, F->mtp_catq, nullptr, F->mtp_h, nullptr, 1.0f, Ly.eh.type, 2 * E, E};
    if (!gf::mv(&eh, 1, s)) {
        err = "glm mtp: eh_proj";
        return false;
    }
    gf::rms_q8(F->mtp_h, nullptr, Ly.attn_norm, g.norm_eps, E, F->x, F->xq, s);
    if (!fast_dsa(il, p, err)) return false;
    gf::rms_q8(F->mtp_h, F->mixer, Ly.ffn_norm, g.norm_eps, E, F->x, F->xq, s);
    bool pf_pending = false;
    if (!fast_moe(il, pf_pending, err)) return false;
    gf::rms_q8(F->mtp_h, F->ffn, Ly.shnorm, g.norm_eps, E, F->head_x, F->head_xq, s);
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, F->head_xq, F->head_x, F->mtp_logits, nullptr, 1.0f, ow.type, E, g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!gf::mv(&o, 1, s)) {
        err = "glm mtp: the head";
        return false;
    }
    gf::argmax(F->mtp_logits, g.n_vocab, F->mtp_tok, s);
    cudaMemcpyAsync(F->mtp_tok_h, F->mtp_tok, sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaEventRecord(F->ev_mtp, s);
    return true;
}

// A sampled (not greedy) token from this half's logits, drawn on its own stream and read back through the pinned
// token word.  On the default stream the draw ended in a device-wide sync, which also waited for the tier copies in
// flight on the copy stream every sampled token - and on Windows / HIP lost the prompt after the first token (#6,
// found by jerem91150).  The stream holds this position's forward and nothing after it (the split's tail drains
// before it takes its next position), so the logits read are the ones just computed.
int Glm5Model::fast_sample(strata::kernels::SamplerParams& sp, std::string& err) {
    FastState* F = fast_;
    cudaSetDevice(dev_);
    const auto ts0 = std::chrono::steady_clock::now();
    if (F->zero_copy) {
        strata::kernels::sample_tokens(sc_ + sc_logits, 1, (int) g_.n_vocab, nullptr, 0, sp, F->tok_hd, F->cs);
    } else {
        strata::kernels::sample_tokens(sc_ + sc_logits, 1, (int) g_.n_vocab, nullptr, 0, sp, d_tok_, F->cs);
        cudaMemcpyAsync(F->tok_h, d_tok_, sizeof(int), cudaMemcpyDeviceToHost, F->cs);
    }
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm sampler", err) || !F->route_ok(err)) return -1;
    if (g_.v41 && !dsv41_engram_ok(err)) return -1;   // (an Engram row this token's layers waited for never arrived)
    if (F->gap_on) {
        F->gap_sample += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts0).count();
        ++F->gap_sample_n;
    }
    return F->tok_h[0];
}

// STRATA_GLM_GAP_PROF: a token (or window) is ready at the host - the spin since `tw` ends, the GPU's idle gap before
// this token (its previous end event -> its first launch) is read, and the clock for the host's share starts
void Glm5Model::fast_gap_mark(std::chrono::steady_clock::time_point tw) {
    FastState* F = fast_;
    const auto now = std::chrono::steady_clock::now();
    F->gap_spin += std::chrono::duration<double, std::milli>(now - tw).count();
    if (F->gap_live) {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, F->ev_ge[F->gap_par ^ 1], F->ev_gs) == cudaSuccess) {
            F->gap_gpu += ms;
            ++F->gap_gpu_n;
        }
        if (cudaEventElapsedTime(&ms, F->ev_ge[F->gap_par ^ 1], F->ev_g0) == cudaSuccess) F->gap_notice += ms;
    }
    {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, F->ev_gs, F->ev_ge[F->gap_par]) == cudaSuccess) F->gap_busy += ms;
    }
    F->gap_par ^= 1;
    F->gap_live = true;
    F->t_ready = now;
    ++F->gap_n;
}

void Glm5Model::gap_report() {
    FastState* F = fast_;
    if (F == nullptr || !F->gap_on || F->gap_n == 0) return;
    const double n = (double) F->gap_n;
    std::fprintf(stderr,
                 "glm gap prof (%llu calls, ms/call): host after token %.3f (sampler %.3f x %.2f) | boundary sync %.3f "
                 "wait %.3f scan %.3f | embed %.3f | spin %.3f | GPU busy %.3f, idle between tokens %.3f (%llu), "
                 "of it before the next token's first command %.3f\n",
                 (unsigned long long) F->gap_n, F->gap_host / n,
                 F->gap_sample / (double) std::max<uint64_t>(1, F->gap_sample_n), (double) F->gap_sample_n / n,
                 F->gap_bsync / n, F->gap_bwait / n, F->gap_bscan / n, F->gap_emb / n, F->gap_spin / n, F->gap_busy / n,
                 F->gap_gpu / (double) std::max<uint64_t>(1, F->gap_gpu_n), (unsigned long long) F->gap_gpu_n,
                 F->gap_notice / (double) std::max<uint64_t>(1, F->gap_gpu_n));
    F->gap_host = F->gap_bsync = F->gap_bwait = F->gap_bscan = F->gap_emb = F->gap_spin = F->gap_gpu = F->gap_sample = 0;
    F->gap_busy = F->gap_notice = 0;
    F->gap_n = F->gap_gpu_n = F->gap_sample_n = 0;
    F->gap_live = false;
}

// the bytes the expert pool may take from what is free: headroom kept (cuBLAS-free path; the context's own growth is
// already allocated in the state arena) - glmfast::default_reserve_mb, STRATA_GLM_RESERVE_MB; STRATA_GLM_POOL_GB caps it;
// STRATA_GLM_VRAM_GB=<n> behaves like a card with n GB (the cap
// counts everything this process already holds on the device, so the pool gets what such a card would have).  The
// prompt path and the layer split search size from the same number, before the pool is carved (a discrete card's:
// fast_setup prices a unified-memory APU's pool itself).
size_t Glm5Model::pool_avail(size_t free_b, size_t total_b) {
    const size_t reserve = glmfast::vram_reserve_bytes(glmfast::default_reserve_mb(total_b));
    size_t avail = glmfast::expert_pool_budget(false, free_b, 0, reserve, 0, 0);
    if (getenv("STRATA_GLM_POOL_GB")) avail = std::min(avail, glmfast::env_gib("STRATA_GLM_POOL_GB"));
    if (getenv("STRATA_GLM_VRAM_GB")) {
        const size_t cap = glmfast::env_gib("STRATA_GLM_VRAM_GB");
        const size_t used = total_b - free_b;
        avail = std::min(avail, cap > used + reserve ? cap - used - reserve : (size_t) 0);
    }
    return avail;
}

// ---- the run's facts for the server's INFO line (Abyss Terminal): read once at the start, nothing on the token path
const char* Glm5Model::lane_kind() const {
    const FastState* F = fast_;
    if (F == nullptr || F->cpu_pool == nullptr) return "off";
    if (F->igpu_experts != nullptr) return "igpu";
    for (const auto& Ly : F->L)
        if (Ly.exl3) return "exl3";
    return "cpu";
}

const char* Glm5Model::drafter_kind() const {
    if (dsp_ != nullptr) return "dspark";
    if (q35_mtp_on()) return g_.q4x ? "mtp" : "nextn";
    return has_mtp() ? "mtp" : "none";
}

int Glm5Model::tail_skip_rank() const {
    return fast_ != nullptr && fast_->tail_skip < gf::kMaxK ? fast_->tail_skip : 0;
}

bool Glm5Model::defer_on() const { return fast_ != nullptr && fast_->defer; }

bool Glm5Model::mirror_configured() const {
    for (const auto& s : pack_shards_)
        if (s.fd_mirror >= 0) return true;
    return false;
}

bool Glm5Model::helper_configured() const {
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ != nullptr && m->fast_->expert_helper != nullptr) return true;
    return false;
}

bool Glm5Model::has_mtp() const {
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ != nullptr && m->mtp_il_ >= 0) return true;
    return false;
}

int Glm5Model::mtp_draft(int32_t next_tok, std::string& err) {
    Glm5Model* t = nullptr;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ != nullptr && m->mtp_il_ >= 0) t = m;
    if (t == nullptr) return -1;
    cudaSetDevice(t->dev_);
    // between tokens: the draft block's routes go through the same tiers, so the boundary runs first
    if (!t->fast_boundary(err)) { cudaSetDevice(dev_); return -1; }
    if (!t->fast_mtp(t->pos_ - 1, next_tok, err)) {
        cudaSetDevice(dev_);
        return -1;
    }
    if (!glmfast::wait_event(t->fast_->ev_mtp, "glm mtp completion", err) || !t->fast_->route_ok(err)) {
        cudaSetDevice(dev_);
        return -1;
    }
    const int d = t->fast_->mtp_tok_h[0];
    cudaSetDevice(dev_);
    if (gf::launch_errors() > 0) {
        err = "glm mtp: a kernel launch failed (see stderr)";
        return -1;
    }
    return d;
}

// ---------------------------------------------------------------- pipelined speculative decode (two halves + NextN)
// The two halves of a split take turns on one token, so each GPU idles half the time.  With the NextN block's draft
// for the token after next, the HEAD half runs that draft while the TAIL half finishes the current token: when the
// tail's token equals the draft, the head's work stands and both GPUs stay busy (a token costs the slower half);
// when it differs, the head restores its recurrent states (saved before the speculative token) and reruns the
// position with the real token.  Only the head speculates - the tail, the NextN block and the sampler see confirmed
// tokens only - so the output is exactly what the token-at-a-time decode would produce for the same samples.
bool Glm5Model::spec_ready() const {
    if (fast_ == nullptr || split_next_ == nullptr || getenv("STRATA_GLM_NO_SPEC") != nullptr) return false;
    const Glm5Model* last = this;
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->fast_ == nullptr) return false;
        last = m;
    }
    return last->mtp_il_ >= 0;
}

Glm5Model* Glm5Model::spec_head_last() {
    int n = 0;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) ++n;
    int h = n / 2;
    if (const char* e = getenv("STRATA_GLM_SPEC_HEAD")) h = std::atoi(e);
    h = std::max(1, std::min(n - 1, h));
    Glm5Model* m = this;
    for (int i = 1; i < h; ++i) m = m->split_next_.get();
    return m;
}

Glm5Model* Glm5Model::spec_tail_last() {
    Glm5Model* m = this;
    while (m->split_next_) m = m->split_next_.get();
    return m;
}

bool Glm5Model::hop_to_next(Glm5Model* B, int64_t p, std::string& err) {
    const size_t hop = (size_t) g_.hc * g_.n_embd * sizeof(float);
    FastState* FA = fast_;
    FastState* FB = B->fast_;
    cudaSetDevice(dev_);
    cudaMemcpyAsync(FA->hop_h, state_, hop, cudaMemcpyDeviceToHost, FA->cs);
    cudaEventRecord(FA->ev_hop, FA->cs);
    cudaSetDevice(B->dev_);
    B->pos_ = p + 1;
    cudaStreamWaitEvent(FB->cs, FA->ev_hop, 0);
    cudaMemcpyAsync(B->state_, FA->hop_h, hop, cudaMemcpyHostToDevice, FB->cs);
    return B->fast_layers(p, true, err);
}

// the head's KDA states (S + conv history of every recurrent layer here: one contiguous run each) <-> kda_bak_
bool Glm5Model::spec_kda_copy(bool restore) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const size_t run = (size_t) g.d_inner() * g.kda_head_dim + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
    int n_rec = 0;
    for (int il = l0_; il < l1_; ++il) n_rec += g.is_recr(il);
    if (n_rec == 0) return true;
    if (kda_bak_ == nullptr && cudaMalloc(&kda_bak_, (size_t) n_rec * run * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        kda_bak_ = nullptr;
        return false;
    }
    int k = 0;
    for (int il = l0_; il < l1_; ++il)
        if (g.is_recr(il)) {
            float* live = state_ + kda_S_[(size_t) il];
            float* bak = kda_bak_ + (size_t) k++ * run;
            cudaMemcpyAsync(restore ? live : bak, restore ? bak : live, run * sizeof(float), cudaMemcpyDeviceToDevice,
                            F->cs);
        }
    return true;
}

// the head half's layers for (p, token); the residual rows go to hop slot p & 1 (spec_ev_hop_ marks them)
bool Glm5Model::spec_head(int64_t p, int32_t token, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaSetDevice(dev_);
    static const bool sprof = getenv("STRATA_GLM_SPEC_PROF") != nullptr;
    auto t0 = std::chrono::steady_clock::now();
    const auto lap = [&](int i) {
        if (!sprof) return;
        const auto t = std::chrono::steady_clock::now();
        spec_t_[i] += std::chrono::duration<double, std::milli>(t - t0).count();
        t0 = t;
    };
    Glm5Model* hl = spec_head_last();
    // every part of the head group idle first: a part's hop slot and boundary must not race its previous position
    for (Glm5Model* m = split_next_.get(); m != nullptr && m != hl->split_next_.get(); m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm spec head part sync", err) ||
            !m->fast_->route_ok(err) || !m->fast_boundary(err)) return false;
    }
    cudaSetDevice(dev_);
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm spec head sync", err) || !F->route_ok(err)) return false;
    lap(0);
    if (sprof) {
        if (spec_ev_[0] == nullptr)
            for (auto& e : spec_ev_) cudaEventCreate(&e);
        else if (spec_ev_live_) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, spec_ev_[0], spec_ev_[1]);
            spec_dev_ms_ += ms;
            ++spec_dev_n_;
        }
        if (spec_ev_live_) {
            float gap = 0.0f;
            cudaEventRecord(spec_ev_[2], F->cs);   // now: the gap since the last token ended
            cudaEventSynchronize(spec_ev_[2]);
            cudaEventElapsedTime(&gap, spec_ev_[1], spec_ev_[2]);
            spec_gap_ms_ += gap;
        }
        cudaEventRecord(spec_ev_[0], F->cs);
    }
    if (F->prof_on) F->collect();
    if (!fast_boundary(err)) return false;
    lap(1);
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    tt->to_float(pack_emb_src_ + (size_t) token * ggml_row_size((ggml_type) pack_emb_type_, g.n_embd), F->emb_h,
                 g.n_embd);
    if (const float* img = image_row(p)) std::memcpy(F->emb_h, img, (size_t) g.n_embd * sizeof(float));   // an image
    cudaMemcpyAsync(F->emb, F->emb_h, (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice, F->cs);
    abl_res(F->emb, 1, g.n_embd, F->cs);
    gf::embed_streams(F->emb, state_, g.n_embd, F->cs);
    lap(2);
    if (!fast_layers(p, false, err)) return false;
    for (Glm5Model* m = this; m != hl; m = m->split_next_.get())
        if (!m->hop_to_next(m->split_next_.get(), p, err)) return false;
    lap(3);
    ++spec_n_;
    const int sl = (int) (p & 1);
    // the head group's last part hands the residual to the tail group through hl's slots (events on hl's device)
    cudaSetDevice(hl->dev_);
    cudaMemcpyAsync(hl->spec_hop_h_[sl], hl->state_, (size_t) g.hc * g.n_embd * sizeof(float), cudaMemcpyDeviceToHost,
                    hl->fast_->cs);
    cudaEventRecord(hl->spec_ev_hop_[sl], hl->fast_->cs);
    cudaSetDevice(dev_);
    if (sprof) {
        cudaEventRecord(spec_ev_[1], F->cs);
        spec_ev_live_ = true;
    }
    return true;
}

// the tail half (this) for position p: waits on the head's hop slot ON THE DEVICE, then its layers, the head and the
// argmax (tok_h after ev_done)
bool Glm5Model::spec_tail(Glm5Model* head, int64_t p, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaSetDevice(dev_);
    static const bool sprof = getenv("STRATA_GLM_SPEC_PROF") != nullptr;
    auto t0 = std::chrono::steady_clock::now();
    const auto lap = [&](int i) {
        if (!sprof) return;
        const auto t = std::chrono::steady_clock::now();
        spec_t_[i] += std::chrono::duration<double, std::milli>(t - t0).count();
        t0 = t;
    };
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm spec tail sync", err)) return false;
    lap(0);
    if (F->prof_on) F->collect();
    if (!fast_boundary(err)) return false;
    lap(1);
    ++spec_n_;
    pos_ = p + 1;
    const int sl = (int) (p & 1);
    if (sprof) {
        if (spec_ev_[0] == nullptr)
            for (auto& e : spec_ev_) cudaEventCreate(&e);
        else if (spec_ev_live_) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, spec_ev_[0], spec_ev_[1]);
            spec_dev_ms_ += ms;
            ++spec_dev_n_;
        }
    }
    for (Glm5Model* m = split_next_.get(); m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm spec tail part sync", err) ||
            !m->fast_->route_ok(err) || !m->fast_boundary(err)) return false;
        m->pos_ = p + 1;
    }
    cudaSetDevice(dev_);
    cudaStreamWaitEvent(F->cs, head->spec_ev_hop_[sl], 0);
    if (sprof) cudaEventRecord(spec_ev_[0], F->cs);
    cudaMemcpyAsync(state_, head->spec_hop_h_[sl], (size_t) g.hc * g.n_embd * sizeof(float), cudaMemcpyHostToDevice,
                    F->cs);
    if (!fast_layers(p, true, err)) return false;
    if (split_next_) {
        // a tail GROUP: down its parts; the last one computes the head and the argmax
        Glm5Model* m = this;
        for (; m->split_next_; m = m->split_next_.get())
            if (!m->hop_to_next(m->split_next_.get(), p, err)) return false;
        cudaSetDevice(m->dev_);
        FastState* FL = m->fast_;
        gf::head_prep(m->state_, m->w_.at("output_norm.weight"), g.norm_eps, g.n_embd, FL->head_x, FL->head_xq, FL->cs);
        const WSlot& owl = m->ws_map_.at("output.weight");
        gf::MvJob ol = {owl.q, FL->head_xq, FL->head_x, m->sc_ + m->sc_logits, nullptr, 1.0f, owl.type, g.n_embd,
                        g.n_vocab};
        if (owl.type == 0) ol.w = owl.f32;
        if (!gf::mv(&ol, 1, FL->cs)) {
            err = "glm spec: output head";
            return false;
        }
        gf::argmax(m->sc_ + m->sc_logits, g.n_vocab, FL->tok, FL->cs);
        cudaMemcpyAsync(FL->tok_h, FL->tok, sizeof(int), cudaMemcpyDeviceToHost, FL->cs);
        cudaEventRecord(FL->ev_done, FL->cs);
        cudaSetDevice(dev_);
        return true;
    }
    gf::head_prep(state_, w_.at("output_norm.weight"), g.norm_eps, g.n_embd, F->head_x, F->head_xq, F->cs);
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, F->head_xq, F->head_x, sc_ + sc_logits, nullptr, 1.0f, ow.type, g.n_embd, g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!gf::mv(&o, 1, F->cs)) {
        err = "glm spec: output head";
        return false;
    }
    gf::argmax(sc_ + sc_logits, g.n_vocab, F->tok, F->cs);
    cudaMemcpyAsync(F->tok_h, F->tok, sizeof(int), cudaMemcpyDeviceToHost, F->cs);
    cudaEventRecord(F->ev_done, F->cs);
    if (sprof) {
        cudaEventRecord(spec_ev_[1], F->cs);
        spec_ev_live_ = true;
    }
    return true;
}

bool Glm5Model::decode_spec(strata::kernels::SamplerParams& sp, int64_t max_new, const std::function<bool(int)>& emit,
                            int64_t& produced, std::string& err) {
    produced = 0;
    if (!spec_ready()) {
        err = "glm spec: needs a split with the NextN block";
        return false;
    }
    Glm5Model* hl = spec_head_last();
    Glm5Model* B = hl->split_next_.get();     // the tail group's first part (spec_tail runs from it)
    Glm5Model* TL = spec_tail_last();         // its last: the token, the logits and the draft block live there
    const Glm5Geometry& g = g_;
    if (hl->spec_hop_h_[0] == nullptr) {
        cudaSetDevice(hl->dev_);
        for (int i = 0; i < 2; ++i)
            if (cudaHostAlloc((void**) &hl->spec_hop_h_[i], (size_t) g.hc * g.n_embd * sizeof(float),
                              cudaHostAllocPortable) != cudaSuccess ||
                cudaEventCreateWithFlags(&hl->spec_ev_hop_[i], cudaEventDisableTiming) != cudaSuccess) {
                err = "glm spec: the hop slots did not allocate";
                return false;
            }
        cudaSetDevice(dev_);
    }
    // the head group's recurrent-state backups (each part on its own device and stream)
    const auto kda_all = [&](bool restore) -> bool {
        for (Glm5Model* m = this; m != hl->split_next_.get(); m = m->split_next_.get()) {
            cudaSetDevice(m->dev_);
            if (!m->spec_kda_copy(restore)) return false;
        }
        cudaSetDevice(dev_);
        return true;
    };
    // the tail's token for its last position: the argmax it left, or a sample from its logits
    const auto take = [&](int& y) -> bool {
        if (sp.greedy) {
            cudaSetDevice(TL->dev_);
            if (!glmfast::wait_event(TL->fast_->ev_done, "glm spec token", err)) return false;
            y = TL->fast_->tok_h[0];
        } else {
            y = TL->fast_sample(sp, err);   // the tail group's last part holds the logits
            if (y < 0) return false;
        }
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
            if (!m->fast_->route_ok(err)) return false;
        y = forced(y);
        sp.counter += 1;
        last_tok_ = y;
        return true;
    };
    const auto draft = [&](int64_t p, int32_t next, int& d) -> bool {
        cudaSetDevice(TL->dev_);
        if (!TL->fast_mtp(p, next, err)) return false;
        if (!glmfast::wait_event(TL->fast_->ev_mtp, "glm spec draft", err) || !TL->fast_->route_ok(err)) return false;
        d = TL->fast_->mtp_tok_h[0];
        return true;
    };
    const auto check = [&]() -> bool {
        if (gf::launch_errors() > 0) {
            err = "glm spec: a kernel launch failed (see stderr)";
            return false;
        }
        return true;
    };
    // the prompt's last forward left position q = pos_ - 1 processed by both halves and its logits on the tail
    int64_t q = pos_ - 1;
    int y = -1;
    if (sp.greedy && last_tok_ >= 0) {
        y = forced(last_tok_);
        sp.counter += 1;
    } else {
        cudaSetDevice(TL->dev_);
        cudaEventRecord(TL->fast_->ev_done, TL->fast_->cs);
        if (!take(y)) return false;
    }
    ++produced;
    if (!emit(y) || produced >= max_new || q + 2 >= max_ctx_) return true;
    int d = -1;
    if (!draft(q, y, d)) return false;
    // the head: position q+1 with the confirmed token, then q+2 with the draft (states saved first)
    if (!spec_head(q + 1, y, err)) return false;
    if (!B->spec_tail(hl, q + 1, err)) return false;
    cudaSetDevice(dev_);
    if (!kda_all(false)) {
        err = "glm spec: the recurrent-state backup did not allocate";
        return false;
    }
    if (!spec_head(q + 2, d, err)) return false;
    bool ok = true;
    // STRATA_GLM_SPEC_PROF=1: host time per phase of a step (debug)
    static const bool sprof = getenv("STRATA_GLM_SPEC_PROF") != nullptr;
    double tp[6] = {0, 0, 0, 0, 0, 0};
    int64_t np = 0;
    auto tnow = std::chrono::steady_clock::now();
    static const char* const lap_names[6] = {"wait tail", "draft", "redo", "enqueue tail", "enqueue head", "emit"};
    const auto lap = [&](int i) {
        if (!sprof) return;
        const auto t = std::chrono::steady_clock::now();
        tp[i] += std::chrono::duration<double, std::milli>(t - tnow).count();
        tnow = t;
#if !defined(STRATA_USE_HIP)
        nvtxRangePop();
        nvtxRangePushA(lap_names[(i + 1) % 6]);
#endif
    };
    // STRATA_GLM_NSYS=<from>,<to>: a profiler capture window over those steps (nsys --capture-range=cudaProfilerApi)
    int64_t nsys_from = -1, nsys_to = -1;
    if (const char* ns = getenv("STRATA_GLM_NSYS")) std::sscanf(ns, "%lld,%lld", (long long*) &nsys_from, (long long*) &nsys_to);
#if !defined(STRATA_USE_HIP)
    if (sprof) nvtxRangePushA("emit");
#endif
    for (;;) {
        if (np == nsys_from) cudaProfilerStart();
        if (np == nsys_to) cudaProfilerStop();
        // (1) the tail's token for position q+1 -> y (the truth for position q+2)
        int y2 = -1;
        lap(5);
        if (!take(y2)) { ok = false; break; }
        lap(0);
        ++produced;
        const bool more = emit(y2) && produced < max_new && q + 3 < max_ctx_;
        if (!more) break;
        // (2) the draft for q+3 from (h_{q+1}, y2)
        int d2 = -1;
        if (!draft(q + 1, y2, d2)) { ok = false; break; }
        lap(1);
        // (3) the head's speculative position q+2 stands iff its draft was y2; else restore and redo it
        ++spec_steps_;
        if (d == y2) {
            ++spec_hits_;
        } else {
            kda_all(true);
            if (!spec_head(q + 2, y2, err)) { ok = false; break; }
        }
        lap(2);
        // (4) the tail's next position goes in first (it waits for the head's hop on the device), then the head's
        //     next speculative position (its boundary waits for the head's current one)
        if (!B->spec_tail(hl, q + 2, err)) { ok = false; break; }
        lap(3);
        kda_all(false);
        if (!spec_head(q + 3, d2, err)) { ok = false; break; }
        lap(4);
        ++np;
        if (!check()) { ok = false; break; }
        ++q;
        d = d2;
    }
    // drain: both devices idle; the head may stand one or two positions past the tail - a request after this one
    // restores its snapshot or resets, so nothing else reads that state
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        std::string drain_err;
        if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm spec drain", drain_err) ||
            !m->fast_->route_ok(drain_err)) {
            if (ok) err = drain_err;
            ok = false;
        }
    }
    cudaSetDevice(dev_);
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->pos_ = TL->pos_;
    if (sprof && np > 0) {
        std::fprintf(stderr, "glm spec prof (%lld steps, ms/step): wait tail token %.2f | draft %.2f | redo %.2f | "
                             "enqueue tail %.2f | enqueue head %.2f | emit %.2f\n", (long long) np, tp[0] / np, tp[1] / np,
                     tp[2] / np, tp[3] / np, tp[4] / np, tp[5] / np);
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
            if (m->spec_n_ > 0)
                std::fprintf(stderr, "glm spec prof CUDA%d (%lld calls, ms/call): sync %.3f | boundary %.3f | embed %.3f | "
                                     "enqueue layers %.3f | DEVICE token %.2f ms, gap before %.2f ms\n", m->dev_,
                             (long long) m->spec_n_, m->spec_t_[0] / m->spec_n_, m->spec_t_[1] / m->spec_n_,
                             m->spec_t_[2] / m->spec_n_, m->spec_t_[3] / m->spec_n_,
                             m->spec_dev_ms_ / (double) std::max<int64_t>(1, m->spec_dev_n_),
                             m->spec_gap_ms_ / (double) std::max<int64_t>(1, m->spec_dev_n_));
    }
    if (ok && !check()) ok = false;
    return ok;
}

// ---------------------------------------------------------------- conversation reuse
// The sequence state that cannot be rebuilt from the position alone is the KDA layers' recurrent state and conv
// history (one contiguous run per KDA layer in state_); the DSA caches are append-only, so restoring the position
// is enough for them (entries past it are rewritten as the new tokens arrive).
std::vector<int> Glm5Model::dsa_layers() const {
    std::vector<int> r;
    if (mtp_il_ >= 0) r.push_back(mtp_il_);
    for (int il = l0_; il < l1_; ++il)
        if (!g_.is_recr(il)) r.push_back(il);
    return r;
}

// the open pool's ik / ig rows at snap_pos_ (pool start .. + kpool - 2: contiguous in the ring, which is a multiple
// of kpool) <-> snap_pool_; on fast_->cs
void Glm5Model::snap_pool_copy(bool restore) {
    const Glm5Geometry& g = g_;
    const std::vector<int> ls = dsa_layers();
    const int64_t rows = (int64_t) (g.idx_kpool - 1) * g.idx_key;   // floats per cache
    if (ls.empty() || rows <= 0 || snap_pos_ < 0) return;
    if (snap_pool_ == nullptr &&
        cudaMalloc(&snap_pool_, (size_t) ls.size() * 2 * rows * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        snap_pool_ = nullptr;
        return;
    }
    const int64_t at = (int64_t) g.idx_key * ((snap_pos_ - snap_pos_ % g.idx_kpool) % ik_ring_);
    for (size_t i = 0; i < ls.size(); ++i) {
        float* ring[2] = {state_ + dsa_ik_[(size_t) ls[i]] + at, state_ + dsa_ig_[(size_t) ls[i]] + at};
        for (int c = 0; c < 2; ++c) {
            float* keep = snap_pool_ + (i * 2 + (size_t) c) * rows;
            cudaMemcpyAsync(restore ? ring[c] : keep, restore ? keep : ring[c], (size_t) rows * sizeof(float),
                            cudaMemcpyDeviceToDevice, fast_->cs);
        }
    }
}

bool Glm5Model::snapshot_save() {
    if (g_.qwen35()) {   // (no layer splits for qwen35moe)
        if (split_next_ != nullptr || !q35_snap_copy(false)) {
            snap_pos_ = -1;
            return false;
        }
        snap_pos_ = pos_;
        return true;
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->fast_ == nullptr) return false;
        const Glm5Geometry& g = m->g_;
        const size_t run = (size_t) g.d_inner() * g.kda_head_dim + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
        int n_rec = 0;
        for (int il = m->l0_; il < m->l1_; ++il) n_rec += g.is_recr(il);
        cudaSetDevice(m->dev_);
        if (m->snap_ == nullptr && n_rec > 0 &&
            cudaMalloc(&m->snap_, (size_t) n_rec * run * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            m->snap_ = nullptr;
            cudaSetDevice(dev_);
            return false;
        }
        int k = 0;
        for (int il = m->l0_; il < m->l1_; ++il)
            if (g.is_recr(il))
                cudaMemcpyAsync(m->snap_ + (size_t) k++ * run, m->state_ + m->kda_S_[(size_t) il], run * sizeof(float),
                                cudaMemcpyDeviceToDevice, m->fast_->cs);
        m->snap_pos_ = pos_;
        if (g.dsv4()) {
            const bool ok = m->dsv4_snap_copy(false);
            cudaStreamSynchronize(m->fast_->cs);
            if (!ok) {
                m->snap_pos_ = -1;
                cudaSetDevice(dev_);
                return false;
            }
            continue;
        }
        m->snap_pool_copy(false);
        cudaStreamSynchronize(m->fast_->cs);
        if (m->snap_pool_ == nullptr && !m->dsa_layers().empty()) {
            cudaSetDevice(dev_);
            return false;
        }
    }
    cudaSetDevice(dev_);
    return true;
}

bool Glm5Model::snapshot_restore() {
    if (g_.qwen35()) {
        if (split_next_ != nullptr || snap_pos_ < 0 || !q35_snap_copy(true)) return false;
        pos_ = snap_pos_;
        dspark_reset();
        return true;
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ == nullptr || m->snap_pos_ < 0) return false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        const Glm5Geometry& g = m->g_;
        const size_t run = (size_t) g.d_inner() * g.kda_head_dim + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
        cudaSetDevice(m->dev_);
        int k = 0;
        for (int il = m->l0_; il < m->l1_; ++il)
            if (g.is_recr(il))
                cudaMemcpyAsync(m->state_ + m->kda_S_[(size_t) il], m->snap_ + (size_t) k++ * run, run * sizeof(float),
                                cudaMemcpyDeviceToDevice, m->fast_->cs);
        if (g.dsv4()) m->dsv4_snap_copy(true);
        else m->snap_pool_copy(true);
        cudaStreamSynchronize(m->fast_->cs);
        m->pos_ = m->snap_pos_;
    }
    cudaSetDevice(dev_);
    dspark_reset();   // (its rings were written past the snapshot: the next prompt fills them again)
    return true;
}

// ---------------------------------------------------------------- conversation slots
// A conversation set aside while others run: per half, the snapshot's KDA states (snap_, taken just before its prompt's
// last token) and every DSA cache's rows up to that position - the latents, the indexer's two key rows and the pools,
// the NextN block's too.  The file: a header (magic, halves, positions, per half its layer range and run sizes), then
// per half the KDA snapshot and the DSA runs in that order.
namespace {
constexpr uint64_t kSlotMagic = 0x3254534C4159414Dull;   // "MAYALST2" (v2: the ik / ig ring's open pool only)

// (state_ offset, floats) of each DSA cache run a slot of n positions holds on one half, in file order
std::vector<std::pair<int64_t, int64_t>> slot_dsa_runs(const Glm5Geometry& g, bool fast_mode, bool lat_q8, int l0, int l1,
                                                       int mtp_il,
                                                       const std::vector<int64_t>& lat, const std::vector<int64_t>& ik,
                                                       const std::vector<int64_t>& ig, const std::vector<int64_t>& pool,
                                                       int64_t n) {
    std::vector<std::pair<int64_t, int64_t>> r;
    // FP16 latents take half a float each, INT8 records lat8_rec_bytes / 4
    const int64_t lat_pp = lat_q8      ? (int64_t) gf::lat8_rec_bytes(g.kv_lora) / 4
                           : fast_mode ? g.kv_lora / 2
                                       : g.kv_lora;
    const int64_t pools = (n + g.idx_kpool - 1) / g.idx_kpool;
    const auto dsa = [&](size_t il) {
        r.push_back({lat[il], n * lat_pp});
        r.push_back({pool[il], pools * g.idx_key});   // (ik / ig: the open pool's rows, in snap_pool_)
        (void) ik;
        (void) ig;
    };
    if (mtp_il >= 0) dsa((size_t) mtp_il);
    for (int il = l0; il < l1; ++il)
        if (!g.is_recr(il)) dsa((size_t) il);
    return r;
}
}  // namespace

uint64_t Glm5Model::slot_save(const std::string& path, std::string& err) {
    if (g_.dsv4()) {
        err = "deepseek4: conversation slots are not implemented (the live snapshot is)";
        return 0;
    }
    std::vector<Glm5Model*> halves;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->fast_ == nullptr || m->snap_pos_ < 0 || m->snap_pos_ != snap_pos_) {
            err = "no snapshot to save";
            return 0;
        }
        halves.push_back(m);
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        err = "cannot write " + path;
        return 0;
    }
    const int64_t n = snap_pos_;
    const auto put = [&](const void* p, size_t bytes) { f.write((const char*) p, (std::streamsize) bytes); };
    const uint32_t nh = (uint32_t) halves.size();
    put(&kSlotMagic, 8);
    put(&nh, 4);
    put(&n, 8);
    std::vector<float> buf;
    double t_sync = 0, t_copy = 0, t_write = 0;
    for (Glm5Model* m : halves) {
        const Glm5Geometry& g = m->g_;
        const int64_t run = (int64_t) g.d_inner() * g.kda_head_dim + (int64_t) 3 * g.d_inner() * (g.d_conv - 1);
        int32_t n_rec = 0;
        for (int il = m->l0_; il < m->l1_; ++il) n_rec += g.is_recr(il);
        const auto runs = slot_dsa_runs(g, m->fast_mode_, m->lat_q8_, m->l0_, m->l1_, m->mtp_il_, m->dsa_lat_, m->dsa_ik_,
                                        m->dsa_ig_, m->dsa_pool_, n);
        const int32_t hdr[4] = {m->l0_, m->l1_, n_rec, (int32_t) runs.size()};
        put(hdr, sizeof(hdr));
        put(&run, 8);
        for (const auto& r : runs) put(&r.second, 8);
        cudaSetDevice(m->dev_);
        auto tq = std::chrono::steady_clock::now();
        cudaStreamSynchronize(m->fast_->cs);
        t_sync += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tq).count();
        std::vector<std::pair<const float*, int64_t>> src;
        if (n_rec > 0) src.push_back({m->snap_, (int64_t) n_rec * run});
        const int64_t pool_rows = (int64_t) m->dsa_layers().size() * 2 * (g.idx_kpool - 1) * g.idx_key;
        if (pool_rows > 0) src.push_back({m->snap_pool_, pool_rows});
        for (const auto& r : runs) src.push_back({m->state_ + r.first, r.second});
        for (const auto& s : src) {
            constexpr int64_t kChunk = (int64_t) 16 << 20;   // floats
            for (int64_t at = 0; at < s.second; at += kChunk) {
                const int64_t c = std::min(kChunk, s.second - at);
                buf.resize((size_t) c);
                tq = std::chrono::steady_clock::now();
                if (cudaMemcpy(buf.data(), s.first + at, (size_t) c * sizeof(float), cudaMemcpyDeviceToHost) !=
                    cudaSuccess) {
                    cudaGetLastError();
                    cudaSetDevice(dev_);
                    err = "the state did not copy back";
                    return 0;
                }
                const auto tw = std::chrono::steady_clock::now();
                t_copy += std::chrono::duration<double, std::milli>(tw - tq).count();
                put(buf.data(), (size_t) c * sizeof(float));
                t_write += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count();
            }
        }
    }
    cudaSetDevice(dev_);
    if (getenv("STRATA_GLM_SLOT_TIMING"))
        std::fprintf(stderr, "glm slots: save %lld positions - sync %.0f ms, copy %.0f ms, write %.0f ms\n",
                     (long long) n, t_sync, t_copy, t_write);
    f.flush();
    if (!f) {
        err = "writing " + path + " failed (disk full?)";
        return 0;
    }
    return (uint64_t) f.tellp();
}

bool Glm5Model::slot_load(const std::string& path, int64_t n_pos, std::string& err) {
    if (g_.dsv4()) {
        err = "deepseek4: conversation slots are not implemented";
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot read " + path;
        return false;
    }
    const auto get = [&](void* p, size_t bytes) { return (bool) f.read((char*) p, (std::streamsize) bytes); };
    uint64_t magic = 0;
    uint32_t nh = 0;
    int64_t n = 0;
    std::vector<Glm5Model*> halves;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) halves.push_back(m);
    if (!get(&magic, 8) || !get(&nh, 4) || !get(&n, 8) || magic != kSlotMagic || nh != halves.size() || n != n_pos ||
        n <= 0 || n > max_ctx_) {
        err = "not a slot of this model";
        return false;
    }
    std::vector<float> buf;
    for (Glm5Model* m : halves) {
        if (m->fast_ == nullptr) {
            err = "no fast path";
            return false;
        }
        const Glm5Geometry& g = m->g_;
        const int64_t run = (int64_t) g.d_inner() * g.kda_head_dim + (int64_t) 3 * g.d_inner() * (g.d_conv - 1);
        int32_t n_rec = 0;
        for (int il = m->l0_; il < m->l1_; ++il) n_rec += g.is_recr(il);
        const auto runs = slot_dsa_runs(g, m->fast_mode_, m->lat_q8_, m->l0_, m->l1_, m->mtp_il_, m->dsa_lat_, m->dsa_ik_,
                                        m->dsa_ig_, m->dsa_pool_, n);
        int32_t hdr[4] = {};
        int64_t frun = 0;
        if (!get(hdr, sizeof(hdr)) || !get(&frun, 8) || hdr[0] != m->l0_ || hdr[1] != m->l1_ || hdr[2] != n_rec ||
            hdr[3] != (int32_t) runs.size() || frun != run) {
            err = "the slot's layout differs from this model's";
            return false;
        }
        for (const auto& r : runs) {
            int64_t fl = 0;
            if (!get(&fl, 8) || fl != r.second) {
                err = "the slot's layout differs from this model's";
                return false;
            }
        }
        cudaSetDevice(m->dev_);
        cudaStreamSynchronize(m->fast_->cs);
        if (m->snap_ == nullptr && n_rec > 0 &&
            cudaMalloc(&m->snap_, (size_t) n_rec * run * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            m->snap_ = nullptr;
            cudaSetDevice(dev_);
            err = "the snapshot did not allocate";
            return false;
        }
        std::vector<std::pair<float*, int64_t>> dst;
        if (n_rec > 0) dst.push_back({m->snap_, (int64_t) n_rec * run});
        const int64_t pool_rows = (int64_t) m->dsa_layers().size() * 2 * (g.idx_kpool - 1) * g.idx_key;
        if (pool_rows > 0) {
            if (m->snap_pool_ == nullptr && cudaMalloc(&m->snap_pool_, (size_t) pool_rows * sizeof(float)) != cudaSuccess) {
                cudaGetLastError();
                m->snap_pool_ = nullptr;
                cudaSetDevice(dev_);
                err = "the snapshot did not allocate";
                return false;
            }
            dst.push_back({m->snap_pool_, pool_rows});
        }
        for (const auto& r : runs) dst.push_back({m->state_ + r.first, r.second});
        for (const auto& d : dst) {
            constexpr int64_t kChunk = (int64_t) 16 << 20;
            for (int64_t at = 0; at < d.second; at += kChunk) {
                const int64_t c = std::min(kChunk, d.second - at);
                buf.resize((size_t) c);
                if (!get(buf.data(), (size_t) c * sizeof(float)) ||
                    cudaMemcpy(d.first + at, buf.data(), (size_t) c * sizeof(float), cudaMemcpyHostToDevice) !=
                        cudaSuccess) {
                    cudaGetLastError();
                    cudaSetDevice(dev_);
                    err = "the slot did not read back";
                    return false;
                }
            }
        }
        m->snap_pos_ = n;
    }
    cudaSetDevice(dev_);
    // the KDA states go from the snapshot into the live state, as snapshot_restore does (it also sets the position)
    if (!snapshot_restore()) {
        err = "the snapshot did not restore";
        return false;
    }
    return true;
}

bool Glm5Model::forward_fast(const std::vector<int32_t>& tokens, std::vector<float>& logits_out, std::string& err) {
    FastState* F = fast_;
    for (int32_t t : tokens) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!fast_token(t, err)) return false;
        F->ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++F->tokens;
    }
    if (host_logits_) {
        Glm5Model* tail = this;
        while (tail->split_next_) tail = tail->split_next_.get();
        cudaSetDevice(tail->dev_);
        logits_out.resize((size_t) g_.n_vocab);
        if (cudaMemcpy(logits_out.data(), tail->sc_ + tail->sc_logits, (size_t) g_.n_vocab * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "glm fast: logits copy";
            return false;
        }
        cudaSetDevice(dev_);
    }
    return true;
}

}  // namespace strata::core
