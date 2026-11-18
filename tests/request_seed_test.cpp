// tests/request_seed_test.cpp - a served request without a seed gets a fresh one (two such requests, even in the same
// clock tick, sample differently); a request's seed= and an engine --seed are kept as given.  Host only.
#include "strata/program/request_seed.hpp"

#include <cstdio>
#include <set>

int main() {
    using strata::program::request_seed;
    int bad = 0;
    uint64_t count = 0;
    if (request_seed(true, 42, false, 0, count, 123) != 42 || count != 0) { std::puts("seed= not kept"); ++bad; }
    if (request_seed(true, 42, true, 7, count, 123) != 42) { std::puts("seed= not kept over --seed"); ++bad; }
    if (request_seed(false, 0, true, 7, count, 123) != 7 || count != 0) { std::puts("--seed not kept"); ++bad; }
    std::set<uint64_t> seen;
    for (int i = 0; i < 1000; ++i) seen.insert(request_seed(false, 0, false, 0, count, 123));   // one clock tick
    for (int i = 0; i < 1000; ++i)
        seen.insert(request_seed(false, 0, false, 0, count, strata::program::request_seed_now()));
    if (seen.size() != 2000 || seen.count(0)) { std::printf("fresh seeds repeat: %zu distinct of 2000\n", seen.size()); ++bad; }
    std::puts(bad ? "request_seed_test: FAILED" : "request_seed_test: ok");
    return bad ? 1 : 0;
}
