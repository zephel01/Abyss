// include/strata/program/conv_slots.hpp - conversation slots: conversations set aside while others run.
//
// The fast path keeps ONE live conversation in the model (its snapshot, taken just before its prompt's last token,
// and the caches' rows up to there).  A request that does not continue it would read its whole prompt again; with
// slots, the conversation in the model is first set aside (its bytes copied out of the model), and the longest
// slot the new prompt continues is put back - chats, and an agent's main thread and its subagents, that interleave
// no longer re-read each other's prompts.
//
// The store is the model-independent part: which slot a prompt continues (the longest whose tokens are an exact
// prefix of it, in the same namespace - an API key's slots are never matched by another key's requests), which
// slot leaves when the store is full (least recently used), where a slot's bytes live and the files of a slot on
// disk.  The model supplies the bytes (Hooks::save) and takes them back (Hooks::load_file / load_mem).
//
//   * the RAM tier (load_mem set, Config::ram_cap > 0): a slot's bytes stay in host memory (pinned when the model
//     can allocate it) - a switch back costs one host-to-device copy.  Over the cap, or with less than
//     Config::ram_reserve of the machine's memory left, the least recently used slot goes to disk.
//   * the disk tier: the bytes are written to a file by one thread (the next prompt never waits for the disk) and
//     read back when the slot is taken.  Config::disk_cap bounds the files, and nothing is written with less than
//     8 GB free on the drive.
//
// GLM-5.3's slots (disk only, the files written since v1.3.x) and the qwen35 path's (RAM first, then disk) are two
// configurations of this store; DeepSeek V4 / V4.1 can use it once their models write and read slot bytes.
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace strata::program::slots {

/// A slot's place in a match: the index of the longest slot whose tokens are a proper prefix of `prompt`, in
/// namespace `ns` (SIZE_MAX: none).  Equal lengths (not normally kept: a superseded slot is dropped) go to the one of
/// session `sess`, then to the most recently used.  Pure, so conv_slots_test.cpp can walk it.
struct Key {
    const int32_t* tokens = nullptr;
    size_t n = 0;
    uint64_t ns = 0, sess = 0, used = 0;
};
inline size_t best_match(const Key* keys, size_t n_keys, const int32_t* prompt, size_t n_prompt, uint64_t ns,
                         uint64_t sess) {
    size_t best = SIZE_MAX;
    for (size_t i = 0; i < n_keys; ++i) {
        const Key& k = keys[i];
        if (k.ns != ns || k.n == 0 || k.n >= n_prompt || !std::equal(k.tokens, k.tokens + k.n, prompt)) continue;
        if (best == SIZE_MAX || k.n > keys[best].n) {
            best = i;
            continue;
        }
        if (k.n < keys[best].n) continue;
        const bool ks = sess != 0 && k.sess == sess, bs = sess != 0 && keys[best].sess == sess;
        if (ks != bs ? ks : k.used > keys[best].used) best = i;
    }
    return best;
}
/// The least recently used of `used[0..n)` other than `keep` (SIZE_MAX: none).
inline size_t lru(const uint64_t* used, size_t n, size_t keep) {
    size_t v = SIZE_MAX;
    for (size_t i = 0; i < n; ++i)
        if (i != keep && (v == SIZE_MAX || used[i] < used[v])) v = i;
    return v;
}
/// The RAM tier's cap: `cap_gb` when set (>= 0), else auto - a quarter of the memory available when the first slot
/// is set aside, at most 8 GB.  Bytes.
inline uint64_t ram_cap_bytes(double cap_gb, int64_t mem_available) {
    if (cap_gb >= 0) return (uint64_t) (cap_gb * 1073741824.0);
    const uint64_t q = mem_available > 0 ? (uint64_t) mem_available / 4 : 0;
    return std::min<uint64_t>(q, (uint64_t) 8 << 30);
}
/// May a slot of `bytes` more stay in RAM: under the cap with what is there, and the machine keeps `reserve` free
/// after it (mem_available <= 0: unknown, the cap alone decides).
inline bool ram_fits(uint64_t in_ram, uint64_t bytes, uint64_t cap, int64_t mem_available, uint64_t reserve) {
    if (in_ram + bytes > cap) return false;
    return mem_available <= 0 || (uint64_t) mem_available >= bytes + reserve;
}
/// "-k<16 hex>" in a slot file's name: the namespace it belongs to (none: 0, the default namespace).
inline uint64_t ns_of_name(const std::string& fn) {
    const size_t k = fn.rfind("-k");
    if (k == std::string::npos || k + 2 + 16 > fn.size()) return 0;
    uint64_t v = 0;
    for (size_t i = k + 2; i < k + 18; ++i) {
        const char c = fn[i];
        const int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (d < 0) return 0;
        v = v << 4 | (uint64_t) d;
    }
    return v;
}

/// A slot's bytes in host memory: pinned (the model allocated it: free_fn gives it back) or an ordinary string.
struct Buf {
    uint8_t* p = nullptr;
    uint64_t n = 0, cap = 0;
    std::string s;                             // (when p is null)
    std::function<void(Buf&)> release;         // a pinned buffer back to its pool
    const uint8_t* data() const { return p != nullptr ? p : (const uint8_t*) s.data(); }
    uint8_t* wdata() { return p != nullptr ? p : (uint8_t*) s.data(); }
    uint64_t size() const { return p != nullptr ? n : (uint64_t) s.size(); }
    bool pinned() const { return p != nullptr; }
    ~Buf() {
        if (p != nullptr && release) release(*this);
    }
};
using BufP = std::shared_ptr<Buf>;

class Store {
public:
    struct Config {
        int max_slots = 0;                     // 0: off
        size_t min_tokens = 1024;
        uint64_t disk_cap = (uint64_t) 16 << 30;
        double ram_cap_gb = 0;                 // < 0: auto (ram_cap_bytes); 0: the RAM tier is off
        uint64_t ram_reserve = (uint64_t) 8 << 30;
        std::filesystem::path dir;
        bool keep = false;
        double days = 7;
        std::string identity;                  // what a kept slot's .meta must hold
        int64_t max_context = 0;
        const char* tag = "glm slots";         // stderr lines
    };
    struct Hooks {
        std::function<BufP(std::string& err)> save;                                     // the model's state, out
        std::function<bool(const std::string& path, int64_t n, std::string& err)> load_file;   // null: via load_mem
        std::function<bool(const Buf& b, int64_t n, std::string& err)> load_mem;        // null: no RAM tier
        std::function<void(const std::string&)> note;                                   // a line for the server
        std::function<int64_t()> mem_available;                                         // bytes (0: unknown)
        std::function<void*(uint64_t)> pinned_alloc;                                    // null / nullptr: pageable
        std::function<void(void*)> pinned_free;
    };
    struct Slot {
        std::vector<int32_t> tokens;
        std::string path;                      // its file (disk tier, or written on the way there)
        uint64_t bytes = 0, used = 0;
        uint64_t ns = 0, sess = 0;
        BufP ram;                              // its bytes in RAM (the RAM tier), else null
    };

    Store() = default;
    Store(const Store&) = delete;
    Store& operator=(const Store&) = delete;
    ~Store() {
        w_.finish();
        slots_.clear();   // (their pinned buffers back to the pool, then freed)
        trim_pool(0);
    }

    bool on() const { return cfg_.max_slots > 0; }
    bool ram_tier() const { return on() && hooks_.load_mem && cfg_.ram_cap_gb != 0; }
    const std::vector<Slot>& slots() const { return slots_; }

    /// Sets the store up: the folder (a previous run's files are removed, or with keep indexed and kept when they
    /// belong to this model).  false when the store cannot run (it is then off).
    bool start(const Config& cfg, Hooks hooks) {
        cfg_ = cfg;
        hooks_ = std::move(hooks);
        w_.write = [this](const Job& j) { return write_job(j); };
        if (cfg_.max_slots <= 0) return false;
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(cfg_.dir, ec);
        struct Found { std::string bin; std::vector<int32_t> toks; uint64_t bytes; fs::file_time_type t; };
        std::vector<Found> found;
        size_t refused = 0;
        for (const auto& de : fs::directory_iterator(cfg_.dir, ec)) {   // a previous run's slots
            const std::string fn = de.path().filename().string(), p = de.path().string();
            if (fn.rfind("slot-", 0) != 0) continue;
            const bool is_bin = fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0;
            const bool is_meta = fn.size() > 10 && fn.compare(fn.size() - 5, 5, ".meta") == 0;
            if (!cfg_.keep || (!is_bin && !is_meta)) {   // not kept, or a .tmp an interrupted write left
                fs::remove(de.path(), ec);
                continue;
            }
            if (!is_meta) continue;                      // a .bin is judged by its .meta
            const std::string bin = p.substr(0, p.size() - 5) + ".bin";
            Found fd{bin, {}, 0, fs::last_write_time(de.path(), ec)};
            const uint64_t have = fs::exists(bin, ec) ? (uint64_t) fs::file_size(bin, ec) : 0;
            if (meta_read(p, fd.toks, fd.bytes) && have == fd.bytes && fd.bytes > 0 && age_days(p) <= cfg_.days) {
                found.push_back(std::move(fd));
            } else {
                ++refused;
                fs::remove(p, ec);
                fs::remove(bin, ec);
            }
        }
        if (cfg_.keep)                                   // a .bin without its .meta was never completed
            for (const auto& de : fs::directory_iterator(cfg_.dir, ec)) {
                const std::string p = de.path().string(), fn = de.path().filename().string();
                if (fn.rfind("slot-", 0) == 0 && fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0 &&
                    !fs::exists(meta_of(p), ec))
                    fs::remove(de.path(), ec);
            }
        std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.t < b.t; });
        uint64_t total = 0;
        for (auto& fd : found) {
            total += fd.bytes;
            Slot s;
            s.tokens = std::move(fd.toks);
            s.ns = ns_of_name(fs::path(fd.bin).filename().string());
            s.path = fd.bin;
            s.bytes = fd.bytes;
            s.used = ++clock_;
            slots_.push_back(std::move(s));
        }
        if (cfg_.keep) {
            std::ostringstream m;
            m << "folder " << cfg_.dir.string() << ": " << slots_.size() << " conversation(s) kept from earlier runs ("
              << gb(total) << ")";
            if (refused) m << ", " << refused << " not this model's or incomplete - removed";
            note(m.str());
        }
        if (!fs::is_directory(cfg_.dir, ec)) {
            std::fprintf(stderr, "%s: %s is not writable - conversations are not set aside\n", cfg_.tag,
                         cfg_.dir.string().c_str());
            cfg_.max_slots = 0;
            return false;
        }
        return true;
    }

    /// The conversation in the model (its snapshot's `tokens`) into a slot of namespace `ns` (session hint `sess`).
    /// The caller has checked that it may be set aside (long enough, text only, the default state).  `size_hint`: its
    /// bytes, when known (room is made in RAM first).  false: not set aside.
    bool put(const std::vector<int32_t>& tokens, uint64_t ns, uint64_t sess, const char* why, uint64_t size_hint = 0) {
        if (!on()) return false;
        for (size_t i = slots_.size(); i-- > 0;)   // an earlier state of this conversation is superseded
            if (slots_[i].ns == ns && slots_[i].tokens.size() <= tokens.size() &&
                std::equal(slots_[i].tokens.begin(), slots_[i].tokens.end(), tokens.begin()))
                drop(i);
        while (!slots_.empty() && slots_.size() >= (size_t) cfg_.max_slots) drop(lru_of(SIZE_MAX));
        const bool ram = ram_tier();
        if (!ram && !disk_room()) {
            std::fprintf(stderr, "%s: under 8 GB free in %s - the conversation is not set aside\n", cfg_.tag,
                         cfg_.dir.string().c_str());
            return false;
        }
        const auto t0 = std::chrono::steady_clock::now();
        take_failed_writes();
        size_t spilled = 0;
        if (ram && size_hint > 0)   // room in RAM for this one: the least recently used RAM slots go to disk
            while (resident() + round_buf(size_hint) > ram_cap()) {
                const size_t i = lru_ram();
                if (i == SIZE_MAX) break;
                spilled += spill(i) ? 1 : 0;
            }
        std::string e;
        BufP buf = hooks_.save ? hooks_.save(e) : nullptr;
        if (buf == nullptr || buf->size() == 0) {
            std::fprintf(stderr, "%s: setting the conversation aside failed (%s)\n", cfg_.tag,
                         e.empty() ? "no snapshot" : e.c_str());
            return false;
        }
        Slot s;
        s.tokens = tokens;
        s.ns = ns;
        s.sess = sess;
        s.bytes = buf->size();
        s.used = ++clock_;
        const uint64_t bytes = s.bytes;
        const char* where = "";
        if (ram && buf->pinned()) {
            s.ram = std::move(buf);
            where = " in RAM";
        } else {   // (the RAM tier full - or writes to disk still holding its memory: this one goes to disk)
            if (ram && !disk_room()) {
                std::fprintf(stderr, "%s: under 8 GB free in %s - the conversation is not set aside\n", cfg_.tag,
                             cfg_.dir.string().c_str());
                return false;
            }
            if (ram) where = " on disk";
            s.path = new_path(ns);
            w_.push({s.path, std::move(buf), s.tokens, bytes});
        }
        slots_.push_back(std::move(s));
        const size_t evicted = cap_disk();
        char m[320];
        std::snprintf(m, sizeof m, "saved a %zu-token conversation%s%s (%s in %.1f s); %zu kept, %s%s%s", tokens.size(),
                      why, where, gb(bytes).c_str(),
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), slots_.size(),
                      gb(total_bytes()).c_str(),
                      evicted ? ", least recently used removed to stay under the size cap" : "",
                      spilled ? ", least recently used moved to disk" : "");
        note(m);
        return true;
    }

    /// The longest slot of namespace `ns` that `prompt` continues, back in the model: the tokens it covers (0: none;
    /// the model's state is then undefined only when a load failed - the caller resets it).  `tokens_out`: its tokens.
    size_t take(const std::vector<int32_t>& prompt, uint64_t ns, uint64_t sess, std::vector<int32_t>& tokens_out,
                bool& failed) {
        failed = false;
        if (!on()) return 0;
        std::vector<Key> keys(slots_.size());
        for (size_t i = 0; i < slots_.size(); ++i)
            keys[i] = {slots_[i].tokens.data(), slots_[i].tokens.size(), slots_[i].ns, slots_[i].sess, slots_[i].used};
        const size_t best = best_match(keys.data(), keys.size(), prompt.data(), prompt.size(), ns, sess);
        if (best == SIZE_MAX) return 0;
        Slot& s = slots_[best];
        const size_t len = s.tokens.size();
        const bool from_ram = s.ram != nullptr;
        if (!from_ram) {   // its file may still be being written (put / spill)
            w_.wait_for(s.path);
            if (w_.take_failed(s.path)) {
                note("setting a conversation aside failed (writing its files) - reading its prompt again");
                slots_.erase(slots_.begin() + (long) best);
                return 0;
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::string e;
        bool ok;
        if (from_ram) {
            ok = hooks_.load_mem(*s.ram, (int64_t) len, e);
        } else if (hooks_.load_file) {
            ok = hooks_.load_file(s.path, (int64_t) len, e);
        } else {
            BufP b = read_file(s.path, s.bytes, e);
            ok = b != nullptr && hooks_.load_mem(*b, (int64_t) len, e);
            if (ok && ram_tier()) promote(best, std::move(b));   // back in RAM (it is the most recent now)
        }
        if (!ok) {
            note("taking a conversation back failed (" + e + ") - reading its prompt again");
            drop(best);
            failed = true;
            return 0;
        }
        Slot& t = slots_[best];
        tokens_out = t.tokens;
        t.used = ++clock_;
        if (sess != 0) t.sess = sess;
        if (cfg_.keep && !t.path.empty() && t.ram == nullptr) {   // its age counts from its last use
            std::error_code ec;
            std::filesystem::last_write_time(meta_of(t.path), std::filesystem::file_time_type::clock::now(), ec);
        }
        char m[200];
        std::snprintf(m, sizeof m, "took back a %zu-token conversation from %s (%.1f s); %zu new tokens to read", len,
                      from_ram ? "RAM" : "disk",
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), prompt.size() - len);
        note(m);
        return len;
    }

    /// Every RAM slot to disk (keep: before the engine ends), then the writes finish.
    void finish() {
        if (on() && cfg_.keep)
            for (size_t i = 0; i < slots_.size(); ++i)
                if (slots_[i].ram != nullptr) spill(i);
        w_.finish();
    }

private:
    struct Job {
        std::string path;
        BufP bytes;
        std::vector<int32_t> toks;
        uint64_t n;
    };
    // slot files are written by one thread from a queue: the next prompt never waits for the disk - a 0.28 GB write
    // took 5-50 s on a 94 GB machine whose RAM tier pins most of the RAM.  A slot is read back only after its own
    // write ended (take waits for that one), and a failed write drops the slot
    struct Writer {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<Job> q;
        std::set<std::string> pending, failed;
        bool stop = false;
        std::thread t;
        std::function<bool(const Job&)> write;
        void push(Job j) {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return q.size() < 3; });   // (at most 3 waiting: their bytes are in RAM)
            pending.insert(j.path);
            q.push_back(std::move(j));
            if (!t.joinable()) t = std::thread([this] { run(); });
            cv.notify_all();
        }
        void run() {
            std::unique_lock<std::mutex> lk(mu);
            for (;;) {
                cv.wait(lk, [&] { return stop || !q.empty(); });
                if (q.empty()) return;
                Job j = std::move(q.front());
                q.pop_front();
                cv.notify_all();
                lk.unlock();
                const bool ok = write(j);
                j.bytes.reset();   // (a pinned buffer back to the pool)
                lk.lock();
                pending.erase(j.path);
                if (!ok) failed.insert(j.path);
                cv.notify_all();
            }
        }
        void wait_for(const std::string& path) {   // until that file is written (or failed)
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return pending.count(path) == 0; });
        }
        bool take_failed(const std::string& path) {
            std::lock_guard<std::mutex> lk(mu);
            return failed.erase(path) > 0;
        }
        std::vector<std::string> take_all_failed() {
            std::lock_guard<std::mutex> lk(mu);
            std::vector<std::string> v(failed.begin(), failed.end());
            failed.clear();
            return v;
        }
        void finish() {
            {
                std::lock_guard<std::mutex> lk(mu);
                stop = true;
            }
            cv.notify_all();
            if (t.joinable()) t.join();
            std::lock_guard<std::mutex> lk(mu);
            stop = false;   // (a later push starts the thread again)
        }
        ~Writer() { finish(); }
    };

    static std::string meta_of(const std::string& bin) { return bin.substr(0, bin.size() - 4) + ".meta"; }
    static std::string gb(uint64_t b) {
        char s[32];
        std::snprintf(s, sizeof s, "%.2f GB", (double) b / 1e9);
        return std::string(s);
    }
    static double age_days(const std::string& p) {
        std::error_code ec;
        const auto t = std::filesystem::last_write_time(p, ec);
        if (ec) return 1e9;
        return std::chrono::duration<double>(std::filesystem::file_time_type::clock::now() - t).count() / 86400.0;
    }
    static constexpr uint64_t kMetaMagic = 0x3153534559414d47ull;   // "GMAYESS1"

    void note(const std::string& m) const {
        std::fprintf(stderr, "%s: %s\n", cfg_.tag, m.c_str());
        if (hooks_.note) hooks_.note(m);
    }
    int64_t mem_available() const { return hooks_.mem_available ? hooks_.mem_available() : 0; }
    uint64_t ram_cap() {
        if (ram_cap_ == UINT64_MAX) ram_cap_ = ram_cap_bytes(cfg_.ram_cap_gb, mem_available() + (int64_t) ram_bytes());
        return ram_cap_;
    }
    uint64_t ram_bytes() const {
        uint64_t b = 0;
        for (const auto& s : slots_)
            if (s.ram != nullptr) b += s.bytes;
        return b;
    }
    uint64_t total_bytes() const {
        uint64_t b = 0;
        for (const auto& s : slots_) b += s.bytes;
        return b;
    }
    size_t lru_of(size_t keep) const {
        std::vector<uint64_t> u(slots_.size());
        for (size_t i = 0; i < slots_.size(); ++i) u[i] = slots_[i].used;
        return lru(u.data(), u.size(), keep);
    }
    size_t lru_ram() const {
        size_t v = SIZE_MAX;
        for (size_t i = 0; i < slots_.size(); ++i)
            if (slots_[i].ram != nullptr && (v == SIZE_MAX || slots_[i].used < slots_[v].used)) v = i;
        return v;
    }
    bool disk_room() const {
        std::error_code ec;
        const auto space = std::filesystem::space(cfg_.dir, ec);
        return ec || space.available >= (uint64_t) 8 << 30;
    }
    std::string new_path(uint64_t ns) {
        const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch()).count();
        std::string name = "slot-" + std::to_string(stamp) + "-" + std::to_string(++seq_);
        if (ns != 0) {
            char k[24];
            std::snprintf(k, sizeof k, "-k%016llx", (unsigned long long) ns);
            name += k;
        }
        return (cfg_.dir / (name + ".bin")).string();
    }
    void take_failed_writes() {
        for (const auto& fp : w_.take_all_failed()) {   // an earlier write that failed: that slot is gone
            for (size_t i = 0; i < slots_.size(); ++i)
                if (slots_[i].path == fp && slots_[i].ram == nullptr) {
                    slots_.erase(slots_.begin() + (long) i);
                    break;
                }
            note("setting a conversation aside failed (writing its files) - it is not kept");
        }
    }
    void drop(size_t i) {
        Slot& s = slots_[i];
        if (!s.path.empty()) {
            w_.wait_for(s.path);   // (its file may still be being written)
            w_.take_failed(s.path);
            std::error_code ec;
            std::filesystem::remove(s.path, ec);
            std::filesystem::remove(meta_of(s.path), ec);
        }
        slots_.erase(slots_.begin() + (long) i);
    }
    // a RAM slot to disk (false: no room on the drive - it is dropped)
    bool spill(size_t i) {
        Slot& s = slots_[i];
        if (!disk_room() || cfg_.disk_cap == 0) {
            drop(i);
            return false;
        }
        if (s.path.empty()) s.path = new_path(s.ns);
        w_.push({s.path, std::move(s.ram), s.tokens, s.bytes});
        s.ram.reset();
        return true;
    }
    // a slot read back from disk stays in RAM when it fits (its file goes)
    void promote(size_t i, BufP b) {
        if (b == nullptr || !b->pinned()) return;   // (alloc gave pinned memory only within the cap)
        std::error_code ec;
        std::filesystem::remove(slots_[i].path, ec);
        std::filesystem::remove(meta_of(slots_[i].path), ec);
        slots_[i].path.clear();
        slots_[i].ram = std::move(b);
    }
    // the files over the disk cap: least recently used disk slots removed (the newest is kept); how many
    size_t cap_disk() {
        size_t evicted = 0;
        for (;;) {
            uint64_t on_disk = 0;
            size_t n_disk = 0, v = SIZE_MAX;
            for (size_t i = 0; i < slots_.size(); ++i)
                if (slots_[i].ram == nullptr) {
                    on_disk += slots_[i].bytes;
                    ++n_disk;
                    if (i != slots_.size() - 1 && (v == SIZE_MAX || slots_[i].used < slots_[v].used)) v = i;
                }
            if (n_disk <= 1 || on_disk <= cfg_.disk_cap || v == SIZE_MAX) break;
            drop(v);
            ++evicted;
        }
        return evicted;
    }
    bool write_job(const Job& j) {
        std::error_code wec;
        bool ok = false;
        {
            std::ofstream f(j.path + ".tmp", std::ios::binary | std::ios::trunc);
            f.write((const char*) j.bytes->data(), (std::streamsize) j.bytes->size());
            f.flush();
            ok = (bool) f;
        }
        ok = ok && !(std::filesystem::rename(j.path + ".tmp", j.path, wec), wec) && meta_write(meta_of(j.path), j.toks, j.n);
        if (!ok) {
            std::filesystem::remove(j.path + ".tmp", wec);
            std::filesystem::remove(j.path, wec);
        }
        return ok;
    }
    bool meta_write(const std::string& path, const std::vector<int32_t>& toks, uint64_t bytes) const {
        const std::string tmp = path + ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            const uint64_t il = cfg_.identity.size(), nt = toks.size();
            f.write((const char*) &kMetaMagic, 8);
            f.write((const char*) &il, 8);
            f.write(cfg_.identity.data(), (std::streamsize) il);
            f.write((const char*) &bytes, 8);
            f.write((const char*) &nt, 8);
            f.write((const char*) toks.data(), (std::streamsize) (nt * sizeof(int32_t)));
            f.write((const char*) &kMetaMagic, 8);
            f.flush();
            if (!f) return false;
        }
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        return !ec;
    }
    // a .meta of this model -> its tokens and the slot's size (false: foreign, damaged or not this model's)
    bool meta_read(const std::string& path, std::vector<int32_t>& toks, uint64_t& bytes) const {
        std::ifstream f(path, std::ios::binary);
        uint64_t m0 = 0, il = 0, nt = 0, m1 = 0;
        if (!f.read((char*) &m0, 8) || m0 != kMetaMagic || !f.read((char*) &il, 8) || il != cfg_.identity.size())
            return false;
        std::string id(il, '\0');
        if (!f.read(id.data(), (std::streamsize) il) || id != cfg_.identity) return false;
        if (!f.read((char*) &bytes, 8) || !f.read((char*) &nt, 8) || nt == 0 || nt > (uint64_t) cfg_.max_context)
            return false;
        toks.resize(nt);
        return (bool) f.read((char*) toks.data(), (std::streamsize) (nt * sizeof(int32_t))) && f.read((char*) &m1, 8) &&
               m1 == kMetaMagic;
    }

public:
    /// A host buffer of `n` bytes for a slot: pinned from the pool when the model can allocate it and the RAM tier
    /// has room, else an ordinary string.
    BufP alloc(uint64_t n) {
        auto b = std::make_shared<Buf>();
        if (hooks_.pinned_alloc && ram_tier()) {
            void* p = nullptr;
            uint64_t cap = 0;
            {
                std::lock_guard<std::mutex> lk(pool_mu_);
                size_t pick = SIZE_MAX;
                for (size_t i = 0; i < pool_.size(); ++i)
                    if (pool_[i].second >= n && (pick == SIZE_MAX || pool_[i].second < pool_[pick].second)) pick = i;
                if (pick != SIZE_MAX) {
                    p = pool_[pick].first;
                    cap = pool_[pick].second;
                    pool_.erase(pool_.begin() + (long) pick);
                }
            }
            if (p == nullptr) {
                trim_pool(0);   // (idle buffers too small for this one)
                cap = round_buf(n);
                if (ram_fits(resident(), cap, ram_cap(), mem_available(), cfg_.ram_reserve)) {
                    p = hooks_.pinned_alloc(cap);
                    if (p != nullptr) {
                        std::lock_guard<std::mutex> lk(pool_mu_);
                        pinned_total_ += cap;
                    }
                }
            }
            if (p != nullptr) {
                b->p = (uint8_t*) p;
                b->n = n;
                b->cap = cap;
                b->release = [this](Buf& x) {
                    std::lock_guard<std::mutex> lk(pool_mu_);
                    pool_.push_back({x.p, x.cap});
                    x.p = nullptr;
                };
                return b;
            }
        }
        b->s.resize((size_t) n);
        return b;
    }
    /// The pinned bytes the RAM slots hold (what the cap bounds; a slot on its way to disk no longer counts - the
    /// machine's free memory, Config::ram_reserve, guards that moment).
    uint64_t resident() const {
        uint64_t b = 0;
        for (const auto& s : slots_)
            if (s.ram != nullptr) b += s.ram->pinned() ? s.ram->cap : s.ram->size();
        return b;
    }
    static uint64_t round_buf(uint64_t n) { return (n + ((uint64_t) 64 << 20) - 1) & ~(((uint64_t) 64 << 20) - 1); }
    uint64_t pinned_total() {
        std::lock_guard<std::mutex> lk(pool_mu_);
        return pinned_total_;
    }

private:
    // idle pinned buffers given back (main thread only: freeing pinned memory synchronizes the device) until they
    // hold at most `keep` bytes
    void trim_pool(uint64_t keep) {
        std::vector<void*> f;
        {
            std::lock_guard<std::mutex> lk(pool_mu_);
            uint64_t idle = 0;
            for (const auto& x : pool_) idle += x.second;
            std::sort(pool_.begin(), pool_.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
            while (!pool_.empty() && idle > keep) {
                idle -= pool_.back().second;
                pinned_total_ -= pool_.back().second;
                f.push_back(pool_.back().first);
                pool_.pop_back();
            }
        }
        for (void* p : f)
            if (hooks_.pinned_free) hooks_.pinned_free(p);
    }
    BufP read_file(const std::string& path, uint64_t bytes, std::string& err) {
        BufP b = alloc(bytes);
        std::ifstream f(path, std::ios::binary);
        if (!f || !f.read((char*) b->wdata(), (std::streamsize) bytes)) {
            err = "cannot read " + path;
            return nullptr;
        }
        return b;
    }

    Config cfg_;
    Hooks hooks_;
    std::vector<Slot> slots_;
    uint64_t clock_ = 0, seq_ = 0;
    uint64_t ram_cap_ = UINT64_MAX;   // (auto: priced at the first use)
    std::mutex pool_mu_;
    std::vector<std::pair<void*, uint64_t>> pool_;
    uint64_t pinned_total_ = 0;
    Writer w_;
};

}  // namespace strata::program::slots
