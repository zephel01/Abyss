// tests/page_file_growth_test.cpp - Windows' commit headroom: what a page file can still grow (fixed, initial < maximum,
// system managed), the volume's room, and saturation.  The arithmetic only: host only, any OS.
#include "strata/platform/memory.hpp"

#include <cstdint>
#include <cstdio>

int main() {
    using namespace strata::platform;
    const uint64_t G = 1ull << 30;
    int bad = 0;
    auto check = [&](const char* what, uint64_t got, uint64_t want) {
        if (got != want) { std::printf("%s: %llu, want %llu\n", what, (unsigned long long) got, (unsigned long long) want); ++bad; }
    };
    // initial 8 GB / maximum 32 GB, at 8 GB now (Strata's issue #5742): 24 GB can still come
    check("min<max", page_file_growth(8 * G, 32 * G, UINT64_MAX), 24 * G);
    check("fixed size", page_file_growth(16 * G, 16 * G, UINT64_MAX), 0);
    check("past its maximum", page_file_growth(20 * G, 16 * G, UINT64_MAX), 0);
    check("volume room caps it", page_file_growth(8 * G, 32 * G, 5 * G), 5 * G);
    check("no room", page_file_growth(8 * G, 32 * G, 0), 0);
    // system managed: 3 x RAM or 4 GB, whichever is larger, then at most 1/8 of the volume
    check("managed 40 GB RAM", managed_page_file_max(40 * G, 0), 120 * G);
    check("managed 1 GB RAM", managed_page_file_max(1 * G, 0), 4 * G);
    check("managed, 1 TB volume", managed_page_file_max(128 * G, 1024 * G), 128 * G);
    check("managed, huge RAM", managed_page_file_max(UINT64_MAX / 2, 0), UINT64_MAX);
    check("saturating add", add_saturating(UINT64_MAX - 1, 5), UINT64_MAX);
    check("plain add", add_saturating(3 * G, 24 * G), 27 * G);
    std::puts(bad ? "page_file_growth_test: FAILED" : "page_file_growth_test: ok");
    return bad ? 1 : 0;
}
