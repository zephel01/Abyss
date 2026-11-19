// src/platform/memory.cpp - see include/strata/platform/memory.hpp.
#include "strata/platform/memory.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <mutex>
#include <string>
#include <vector>
#else
#include <sys/mman.h>
#endif

namespace strata::platform {

#if defined(_WIN32)
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        r.note = "GetProcessWorkingSetSizeEx failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    // Locked pages count against the minimum working set, so it must grow by the region plus headroom for the
    // rest of the process. Soft limits: the maximum is not enforced, only the minimum is raised.
    const SIZE_T margin = (SIZE_T) 512 << 20;
    const SIZE_T new_min = min_ws + (SIZE_T) bytes + margin;
    const SIZE_T new_max = max_ws > new_min + margin ? max_ws : new_min + margin;
    if (!SetProcessWorkingSetSizeEx(self, new_min, new_max,
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        r.note = "SetProcessWorkingSetSizeEx(" + std::to_string((unsigned long long) (new_min >> 20)) +
                 " MiB) failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    const uint64_t chunk = 1ull << 30;
    uint8_t* base = (uint8_t*) p;
    for (uint64_t off = 0; off < bytes; off += chunk) {
        const uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (!VirtualLock(base + off, (SIZE_T) n)) {
            r.note = "VirtualLock stopped at " + std::to_string((unsigned long long) (off >> 20)) + " of " +
                     std::to_string((unsigned long long) (bytes >> 20)) + " MiB (error " +
                     std::to_string(GetLastError()) + ")";
            r.ok = off > 0;
            return r;
        }
        r.locked_bytes = off + n;
    }
    r.ok = true;
    r.note = "locked " + std::to_string((unsigned long long) (bytes >> 20)) + " MiB via working-set minimum + VirtualLock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return;
    const uint64_t chunk = 1ull << 30;
    for (uint64_t off = 0; off < bytes; off += chunk)
        VirtualUnlock((uint8_t*) p + off, (SIZE_T) (bytes - off < chunk ? bytes - off : chunk));
}

namespace {

// The page files and their sizes, from documented calls that need no administrator:
//  - the configuration: HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management, value
//    PagingFiles (REG_MULTI_SZ, what System > Advanced > Virtual memory writes): one "path initial maximum" entry per
//    file, sizes in MB; "path 0 0" or a bare path = system managed; "?:\pagefile.sys" = "automatically manage for all
//    drives" (the file is on the system drive).  An empty value = no page file (nothing can grow).
//  - the current size of each file: GetFileAttributesExW (a page file is open by the system, but its attributes read).
//  - the room on its volume: GetDiskFreeSpaceExW; growth on one volume never exceeds its free space less 1 GiB.
// NtQuerySystemInformation(SystemPageFileInformation) gives the current sizes too, but not the maximum and is not a
// documented class; WMI (Win32_PageFileSetting) needs COM in the engine.  Any failure: no growth counted (the caller
// then has exactly ullAvailPageFile, the behaviour before).
bool read_page_file_growth(uint64_t ram_total, uint64_t& growth) {
    const wchar_t* key = L"SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management";
    DWORD size = 0;
    if (RegGetValueW(HKEY_LOCAL_MACHINE, key, L"PagingFiles", RRF_RT_REG_MULTI_SZ, nullptr, nullptr, &size) !=
        ERROR_SUCCESS)
        return false;
    std::vector<wchar_t> buf(size / sizeof(wchar_t) + 2, L'\0');
    size = (DWORD) ((buf.size() - 1) * sizeof(wchar_t));
    if (RegGetValueW(HKEY_LOCAL_MACHINE, key, L"PagingFiles", RRF_RT_REG_MULTI_SZ, nullptr, buf.data(), &size) !=
        ERROR_SUCCESS)
        return false;
    wchar_t windir[MAX_PATH] = L"";
    const wchar_t sys_drive = GetSystemWindowsDirectoryW(windir, MAX_PATH) > 0 ? windir[0] : L'C';
    uint64_t room_left[26];   // per drive letter: free space not yet given to a page file; UINT64_MAX = not read
    for (uint64_t& r : room_left) r = UINT64_MAX;
    growth = 0;
    for (const wchar_t* e = buf.data(); *e; e += std::wcslen(e) + 1) {
        // split off up to two trailing numbers; what is left is the path (it may hold spaces)
        std::wstring s(e);
        while (!s.empty() && (s.back() == L' ' || s.back() == L'\t')) s.pop_back();
        uint64_t nums[2] = {0, 0};
        int n_nums = 0;
        while (n_nums < 2) {
            size_t sp = s.find_last_of(L" \t");
            if (sp == std::wstring::npos) break;
            const std::wstring tok = s.substr(sp + 1);
            if (tok.empty() || tok.find_first_not_of(L"0123456789") != std::wstring::npos) break;
            nums[n_nums++] = std::wcstoull(tok.c_str(), nullptr, 10);
            s.erase(sp);
            while (!s.empty() && (s.back() == L' ' || s.back() == L'\t')) s.pop_back();
        }
        if (s.empty()) continue;
        if (n_nums == 1) return false;   // "path N": not a form Windows writes - do not guess
        const uint64_t max_mb = nums[0], init_mb = nums[1];   // read from the end: maximum first
        const bool managed = n_nums == 0 || (init_mb == 0 && max_mb == 0);
        if (s.size() >= 2 && s[0] == L'?' && s[1] == L':') s[0] = sys_drive;
        const wchar_t letter = s[0] >= L'a' && s[0] <= L'z' ? (wchar_t) (s[0] - L'a' + L'A') : s[0];
        if (s.size() < 3 || s[1] != L':' || letter < L'A' || letter > L'Z') return false;
        WIN32_FILE_ATTRIBUTE_DATA fa{};
        if (!GetFileAttributesExW(s.c_str(), GetFileExInfoStandard, &fa)) return false;
        const uint64_t current = ((uint64_t) fa.nFileSizeHigh << 32) | fa.nFileSizeLow;
        const wchar_t root[4] = {letter, L':', L'\\', L'\0'};
        ULARGE_INTEGER avail{}, total{}, free_all{};
        if (!GetDiskFreeSpaceExW(root, &avail, &total, &free_all)) return false;
        uint64_t& room = room_left[letter - L'A'];
        if (room == UINT64_MAX) room = free_all.QuadPart > (1ull << 30) ? free_all.QuadPart - (1ull << 30) : 0;
        const uint64_t maximum = managed ? managed_page_file_max(ram_total, total.QuadPart)
                                         : (max_mb > (UINT64_MAX >> 20) ? UINT64_MAX : max_mb << 20);
        const uint64_t g = page_file_growth(current, maximum, room);
        room -= g;
        growth = add_saturating(growth, g);
    }
    return true;
}

}  // namespace

uint64_t page_file_growth_bytes(uint64_t ram_total) {
    const char* env = std::getenv("STRATA_PAGEFILE_GROWTH");
    if (env != nullptr && std::strcmp(env, "0") == 0) return 0;
    static std::mutex mu;
    static ULONGLONG last_tick = 0;
    static uint64_t last_ram = 0, last_growth = 0;
    static bool have = false;
    std::lock_guard<std::mutex> lock(mu);
    const ULONGLONG now = GetTickCount64();
    if (!have || ram_total != last_ram || now - last_tick >= 2000) {
        uint64_t g = 0;
        last_growth = read_page_file_growth(ram_total, g) ? g : 0;
        last_tick = now;
        last_ram = ram_total;
        have = true;
    }
    return last_growth;
}
#else
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    if (mlock(p, bytes) != 0) { r.note = "mlock failed (raise ulimit -l)"; return r; }
    r.ok = true;
    r.locked_bytes = bytes;
    r.note = "mlock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p != nullptr && bytes != 0) munlock(p, bytes);
}
#endif

}  // namespace strata::platform
