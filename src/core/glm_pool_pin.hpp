// src/core/glm_pool_pin.hpp - the expert pool's budget kept per PC + pack + config (STRATA_GLM_POOL_PIN), shared by
// src/core/glm_fast_path.cu (fast_setup carves the pool) and src/core/glm_prefill.cu (the prompt path's estimate of it).
//
// Why: the pool's slots a layer came from the free VRAM at each start, and the free VRAM moves with what other
// processes hold.  On an RTX 5090, ~0.12 GB more held elsewhere moved DeepSeek V4.1 Compact JA256 from 68 to 67 slots a
// layer, and V4.1 (no placement-independent expert products) then answered differently from the 5th token on
// (worklog 2026-11-23_v41ja-md5-change: md5 55e1d3f2d4 at 68, 3906291bef at 67).  So the first start keeps its budget,
// rounded down to 1 MiB, in <pack>/pool_pin.txt under a key of the card (UUID, VRAM), the layers, the bytes this
// process holds before the pool (dense weights, KV and state: the context and the KV type), the build and the settings
// that move the VRAM (cap, reserve, drafters); every later start with the same key carves the same pool:
//   - the saved budget is used while it is at most 256 MiB above what is free now (the excess comes out of the
//     reserve, 1.5-2 GB on a big card), so a few hundred MB of jitter keeps the count;
//   - more short than that (another process holds VRAM): this start is sized from what is free, says so in one
//     line, and the kept budget stays for the next start (a squeeze that passes must not shrink every later pool);
//   - 1 GiB or more free beyond the saved budget (the user closed what held it): learned again, said in one line.
// STRATA_GLM_POOL_PIN=0 turns it off (every start sizes the pool from its free VRAM, as before v1.3.198);
// STRATA_GLM_POOL_PIN=reset learns this key's budget again (or delete pool_pin.txt beside the pack).
// STRATA_GLM_POOL_GB (an explicit budget) and unified memory (the pool is RAM there) skip it.
// (The idea - a keyed budget file beside the pack - follows Maya's VRAM reserve file; the code is Abyss's.)
#pragma once

#include "strata/program/gpu_ident.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef STRATA_VERSION
#define STRATA_VERSION "dev"
#endif

namespace strata::core::poolpin {

constexpr size_t kStep = (size_t) 1 << 20;     // the budget's unit (32 MiB steps cost Ornith-1.5-35B one slot
                                               // a layer on a 16 GB cap, 155 -> 154: 84.0 against 85.2 tok/s)
constexpr size_t kSlack = (size_t) 256 << 20;  // how far above the free VRAM a saved budget is still used
constexpr size_t kGrow = (size_t) 1 << 30;     // free VRAM this far above the saved budget learns it again

enum class Mode { Off, On, Reset };

inline Mode mode() {
    const char* v = std::getenv("STRATA_GLM_POOL_PIN");
    if (v == nullptr || v[0] == 0) return Mode::On;
    if (std::strcmp(v, "reset") == 0) return Mode::Reset;
    return std::atoi(v) != 0 ? Mode::On : Mode::Off;
}

inline std::string file_for(const std::string& pack_dir) { return pack_dir + "/pool_pin.txt"; }

// the key: what decides the pool's budget apart from other processes.  `own_b`: the bytes this part holds before the
// pool (its dense weights, KV and state) - the context length and KV type show up here.  `extra`: what decides the
// budget that own_b does not see (Inkling: its KV slots and --parallel lanes sit outside state_bytes_, so its context
// length and lanes are named - "|ctx=<n>|par=<n>"; an entry saved under the key without them is not found, the
// budget is learnt once again)
inline std::string key(int dev, size_t total_b, int l0, int lt, size_t own_b, const std::string& extra = std::string()) {
    std::string u = strata::program::gpu_uuid(dev);
    if (u.empty()) u = "dev" + std::to_string(dev) + "@" + strata::program::gpu_pci(dev);
    std::ostringstream k;
    k << u << "|vram=" << (total_b >> 20) << "|layers=" << l0 << "-" << lt << "|own=" << (own_b >> 20)
      << extra << "|v=" << STRATA_VERSION;
    static const char* const kEnv[] = {"STRATA_GLM_VRAM_GB",   "STRATA_GLM_VRAM_GB_PARTS", "STRATA_GLM_RESERVE_MB",
                                       "STRATA_INK_DSPARK",    "STRATA_DS4_DSPARK",        "STRATA_GLM_VISION_LEND_MB",
                                       "STRATA_GLM_TAIL_ON_TOP"};
    for (const char* e : kEnv)
        if (const char* v = std::getenv(e); v != nullptr && v[0] != 0) {
            std::string s = v;
            for (char& c : s)
                if (c == '\t' || c == '\n' || c == '|') c = '_';
            k << "|" << e + 7 << "=" << s;   // (without the STRATA_ prefix)
        }
    return k.str();
}

// the saved budget (MiB) for `k`, or 0 when there is none
inline size_t load_mib(const std::string& file, const std::string& k) {
    std::ifstream f(file);
    std::string line;
    size_t mib = 0;
    while (std::getline(f, line)) {
        const size_t t = line.find('\t');
        if (t == std::string::npos || line.compare(0, t, k) != 0 || t != k.size()) continue;
        mib = (size_t) std::strtoull(line.c_str() + t + 1, nullptr, 10);
    }
    return mib;
}

// keep `k`'s budget (MiB) and a note; the other keys' lines stay.  Written to a temporary file and renamed over, so
// two engines on one pack never read half a file.  False when the folder cannot be written (a read-only pack)
// put `tmp` in place of `dst` (std::filesystem::rename replaces an existing file on Windows too; std::rename fails there
// when it exists).  Windows: a reader that holds `dst` open for a moment (an antivirus scan, the search indexer)
// refuses the replace, so it is tried a few times, then a read-only `dst` is made writable and removed first (#62:
// v1.3.219 on Windows still said "could not be kept in (read-only?)").  `why` gets the system's reason on a failure.
inline bool replace_file(const std::string& tmp, const std::string& dst, std::string* why = nullptr) {
    std::error_code ec;
    std::filesystem::rename(tmp, dst, ec);
#ifdef _WIN32
    for (int i = 1; ec && i <= 4; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50 * i));
        ec.clear();
        std::filesystem::rename(tmp, dst, ec);
    }
    if (ec) {
        std::error_code e2;
        std::filesystem::permissions(dst, std::filesystem::perms::owner_write, std::filesystem::perm_options::add, e2);
        std::filesystem::remove(dst, e2);
        e2.clear();
        std::filesystem::rename(tmp, dst, e2);
        if (!e2) ec.clear();
    }
#endif
    if (!ec) return true;
    if (why != nullptr) *why = ec.message();
    std::remove(tmp.c_str());
    return false;
}

inline bool save_mib(const std::string& file, const std::string& k, size_t mib, const std::string& note,
                     std::string* why = nullptr) {
    std::vector<std::string> keep;
    {
        std::ifstream f(file);
        std::string line;
        while (std::getline(f, line)) {
            const size_t t = line.find('\t');
            if (line.empty() || line[0] == '#' || (t == k.size() && line.compare(0, t, k) == 0)) continue;
            keep.push_back(line);
        }
    }
    const std::string tmp = file + ".tmp" + std::to_string((long long) std::rand());
    FILE* o = std::fopen(tmp.c_str(), "w");
    if (o == nullptr) return false;
    std::fprintf(o, "# the expert pool's budget per card + layers + config (MiB), kept so every start carves the same "
                    "pool; STRATA_GLM_POOL_PIN=reset learns one again, deleting the file learns all again\n");
    for (const std::string& l : keep) std::fprintf(o, "%s\n", l.c_str());
    std::fprintf(o, "%s\t%zu\t%s\n", k.c_str(), mib, note.c_str());
    if (std::fclose(o) != 0) {
        if (why != nullptr) *why = "write failed";
        std::remove(tmp.c_str());
        return false;
    }
    return replace_file(tmp, file, why);
}

enum Why { kSaved = 0, kFirst, kReset, kShort, kGrew };

struct Choice {
    size_t budget = 0;   // bytes the pool is sized from
    Why why = kFirst;
    size_t saved = 0;    // the saved budget, bytes (0: none)
};

// the budget for this start from what is free (`avail`, after the reserve, caps and lending) and the saved one
inline Choice decide(size_t avail, size_t saved_b, bool reset) {
    Choice c;
    c.saved = saved_b;
    const size_t learned = avail / kStep * kStep;
    if (reset) c = {learned, kReset, saved_b};
    else if (saved_b == 0) c = {learned, kFirst, 0};
    else if (saved_b > avail + kSlack) c = {learned, kShort, saved_b};
    else if (avail >= saved_b + kGrow) c = {learned, kGrew, saved_b};
    else c = {saved_b, kSaved, saved_b};
    return c;
}

}  // namespace strata::core::poolpin
