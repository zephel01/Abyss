// conv_slots_test.cpp - the conversation slots' store (include/strata/program/conv_slots.hpp) with a fake model:
// the longest-prefix match, namespaces, the session hint, LRU eviction, the RAM tier and its spill to disk, the disk
// tier's files and a kept folder read back.  No GPU: the "model" is a byte string whose state the hooks save/load.
#include "strata/program/conv_slots.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace sl = strata::program::slots;

static int fails = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                                          \
        }                                                                     \
    } while (0)

// the fake model: its "state" is the conversation's tokens as bytes (save: the live state out; load: it back)
struct Fake {
    std::vector<int32_t> live;
    std::map<void*, uint64_t> pinned;
    int64_t avail = (int64_t) 64 << 30;
    std::string as_bytes(const std::vector<int32_t>& t, uint64_t pad) const {
        std::string s((const char*) t.data(), t.size() * 4);
        s.resize(s.size() + pad, 'x');
        return s;
    }
};

static std::vector<int32_t> seq(int a, int n) {
    std::vector<int32_t> v(n);
    for (int i = 0; i < n; ++i) v[i] = a + i;
    return v;
}
static std::vector<int32_t> cat(std::vector<int32_t> a, const std::vector<int32_t>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

static sl::Store::Hooks hooks(Fake& m, sl::Store& st, bool ram, uint64_t pad) {
    sl::Store::Hooks h;
    h.save = [&m, &st, pad](std::string&) -> sl::BufP {
        const std::string b = m.as_bytes(m.live, pad);
        sl::BufP p = st.alloc(b.size());
        std::memcpy(p->wdata(), b.data(), b.size());
        return p;
    };
    if (ram) {
        h.load_mem = [&m](const sl::Buf& b, int64_t n, std::string& e) {
            if (b.size() < (uint64_t) n * 4) {
                e = "short";
                return false;
            }
            m.live.assign((const int32_t*) b.data(), (const int32_t*) b.data() + n);
            return true;
        };
        h.mem_available = [&m] { return m.avail; };
        h.pinned_alloc = [&m](uint64_t n) -> void* {
            void* p = std::malloc((size_t) n);
            m.pinned[p] = n;
            return p;
        };
        h.pinned_free = [&m](void* p) {
            m.pinned.erase(p);
            std::free(p);
        };
    } else {
        h.load_file = [&m](const std::string& path, int64_t n, std::string& e) {
            std::FILE* f = std::fopen(path.c_str(), "rb");
            if (!f) {
                e = "open";
                return false;
            }
            m.live.resize((size_t) n);
            const size_t got = std::fread(m.live.data(), 4, (size_t) n, f);
            std::fclose(f);
            return got == (size_t) n;
        };
    }
    return h;
}

static sl::Store::Config config(const std::filesystem::path& dir, int max_slots, double ram_gb) {
    sl::Store::Config c;
    c.max_slots = max_slots;
    c.min_tokens = 1;
    c.dir = dir;
    c.ram_cap_gb = ram_gb;
    c.ram_reserve = (uint64_t) 1 << 30;
    c.identity = "conv-slots-test-1";
    c.max_context = 1 << 20;
    c.tag = "test slots";
    return c;
}

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("conv_slots_test_" + std::to_string((long long) std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(root);

    {   // the pure helpers
        const auto a = seq(0, 10), b = seq(0, 20), c = seq(100, 5);
        sl::Key k[3] = {{a.data(), a.size(), 0, 0, 1}, {b.data(), b.size(), 0, 0, 2}, {c.data(), c.size(), 0, 0, 3}};
        const auto p = cat(seq(0, 20), seq(500, 3));
        CHECK(sl::best_match(k, 3, p.data(), p.size(), 0, 0) == 1);       // the longest prefix
        CHECK(sl::best_match(k, 3, p.data(), 20, 0, 0) == 0);             // a slot as long as the prompt: not it
        CHECK(sl::best_match(k, 3, p.data(), p.size(), 7, 0) == SIZE_MAX);   // another namespace's: never
        k[1].ns = 7;
        CHECK(sl::best_match(k, 3, p.data(), p.size(), 0, 0) == 0);
        CHECK(sl::best_match(k, 3, p.data(), p.size(), 7, 0) == 1);
        sl::Key e[2] = {{a.data(), a.size(), 0, 5, 9}, {a.data(), a.size(), 0, 6, 1}};   // equal lengths: the session
        CHECK(sl::best_match(e, 2, p.data(), p.size(), 0, 6) == 1);
        CHECK(sl::best_match(e, 2, p.data(), p.size(), 0, 0) == 0);       // ... else the most recent
        const uint64_t used[4] = {5, 2, 9, 1};
        CHECK(sl::lru(used, 4, SIZE_MAX) == 3);
        CHECK(sl::lru(used, 4, 3) == 1);
        CHECK(sl::ram_cap_bytes(2.0, 0) == (uint64_t) 2 << 30);
        CHECK(sl::ram_cap_bytes(-1, (int64_t) 12 << 30) == (uint64_t) 3 << 30);    // auto: a quarter ...
        CHECK(sl::ram_cap_bytes(-1, (int64_t) 100 << 30) == (uint64_t) 8 << 30);   // ... at most 8 GB
        CHECK(sl::ram_fits(0, 100, 200, 0, 1000));                        // unknown memory: the cap decides
        CHECK(!sl::ram_fits(150, 100, 200, 0, 0));
        CHECK(!sl::ram_fits(0, 100, 200, 500, 1000));                     // the machine keeps its reserve
        CHECK(sl::ns_of_name("slot-1-2-k00000000000000ab.bin") == 0xab);
        CHECK(sl::ns_of_name("slot-1-2.bin") == 0);
    }

    {   // the disk tier (GLM's configuration): put, take back, superseded, LRU over the count
        Fake m;
        sl::Store st;
        CHECK(st.start(config(root / "disk", 2, 0), hooks(m, st, false, 0)));
        const auto A = seq(0, 100), B = seq(1000, 80), C = seq(5000, 60);
        m.live = A;
        CHECK(st.put(A, 0, 0, ""));
        m.live = B;
        CHECK(st.put(B, 0, 0, ""));
        std::vector<int32_t> t;
        bool failed = false;
        m.live.clear();
        CHECK(st.take(cat(A, seq(9, 4)), 0, 0, t, failed) == A.size() && t == A && m.live == A && !failed);
        CHECK(st.take(cat(B, seq(9, 4)), 3, 0, t, failed) == 0);         // another namespace
        // A continued: its longer state supersedes the slot
        const auto A2 = cat(A, seq(9, 50));
        m.live = A2;
        CHECK(st.put(A2, 0, 0, ""));
        CHECK(st.slots().size() == 2);
        m.live = C;   // the count is 2: the least recently used (B) leaves
        CHECK(st.put(C, 0, 0, ""));
        CHECK(st.take(cat(B, seq(9, 4)), 0, 0, t, failed) == 0);
        CHECK(st.take(cat(A2, seq(9, 4)), 0, 0, t, failed) == A2.size() && m.live == A2);
        st.finish();
        size_t files = 0;
        for (const auto& de : fs::directory_iterator(root / "disk"))
            if (de.path().extension() == ".bin") ++files;
        CHECK(files == 2);
    }

    {   // the RAM tier: slots stay in RAM; over the cap the least recently used goes to disk and comes back
        Fake m;
        sl::Store st;
        // 3 x 1.5 MB slots (bytes padded) under a cap of 3 MB... rounded to 64 MB buffers: cap 0.13 GB = 2 buffers
        CHECK(st.start(config(root / "ram", 8, 0.13), hooks(m, st, true, 1 << 20)));
        const auto A = seq(0, 100), B = seq(1000, 80), C = seq(5000, 60);
        std::vector<int32_t> t;
        bool failed = false;
        for (const auto* x : {&A, &B}) {
            m.live = *x;
            CHECK(st.put(*x, 0, 0, "", (1 << 20) + x->size() * 4));
        }
        CHECK(st.slots().size() == 2 && st.slots()[0].ram != nullptr && st.slots()[1].ram != nullptr);
        CHECK(m.pinned.size() == 2);
        m.live = C;   // a third: A (least recently used) goes to disk
        CHECK(st.put(C, 0, 0, "", (1 << 20) + C.size() * 4));
        size_t in_ram = 0;
        for (const auto& s : st.slots()) in_ram += s.ram != nullptr;
        CHECK(st.slots().size() == 3);
        CHECK(in_ram <= 2);
        m.live.clear();
        CHECK(st.take(cat(A, seq(9, 4)), 0, 0, t, failed) == A.size() && m.live == A);   // from disk
        CHECK(st.take(cat(C, seq(9, 4)), 0, 0, t, failed) == C.size() && m.live == C);   // from RAM
        m.avail = (int64_t) 512 << 20;   // the machine short of memory: no new pinned buffer, the slot goes to disk
        const auto D = seq(9000, 30);
        m.live = D;
        CHECK(st.put(D, 0, 0, "", 0));
        CHECK(st.take(cat(D, seq(9, 4)), 0, 0, t, failed) == D.size() && m.live == D);
        st.finish();
    }

    {   // keep: the conversations on disk when the engine ends, read back by the next start (namespaces kept)
        Fake m;
        {
            sl::Store st;
            auto c = config(root / "keep", 4, 0.13);
            c.keep = true;
            CHECK(st.start(c, hooks(m, st, true, 0)));
            m.live = seq(0, 40);
            CHECK(st.put(m.live, 0x1234, 0, ""));
            st.finish();
        }
        sl::Store st2;
        auto c = config(root / "keep", 4, 0.13);
        c.keep = true;
        CHECK(st2.start(c, hooks(m, st2, true, 0)));
        CHECK(st2.slots().size() == 1 && st2.slots()[0].ns == 0x1234);
        std::vector<int32_t> t;
        bool failed = false;
        m.live.clear();
        CHECK(st2.take(cat(seq(0, 40), seq(7, 2)), 0, 0, t, failed) == 0);   // not this key's
        CHECK(st2.take(cat(seq(0, 40), seq(7, 2)), 0x1234, 0, t, failed) == 40 && m.live == seq(0, 40));
        auto c2 = config(root / "keep", 4, 0.13);
        c2.identity = "another model";   // a foreign folder's files are removed
        sl::Store st3;
        st2.finish();
        CHECK(st3.start(c2, hooks(m, st3, true, 0)));
        CHECK(st3.slots().empty());
    }

    fs::remove_all(root);
    if (fails) {
        std::fprintf(stderr, "conv_slots_test: %d failure(s)\n", fails);
        return 1;
    }
    std::printf("conv_slots_test: ok\n");
    return 0;
}
