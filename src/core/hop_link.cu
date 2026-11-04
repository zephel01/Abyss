// SPDX-License-Identifier: MIT
//
// HopLink's transports (include/strata/core/hop_link.hpp): host, peer, sock.

#include "strata/core/hop_link.hpp"

#include <sys/socket.h>
#include <unistd.h>

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::core {

namespace {

// the caller's device back on every return
struct DevGuard {
    int prev = 0;
    DevGuard() { cudaGetDevice(&prev); }
    ~DevGuard() { cudaSetDevice(prev); }
};

bool ok(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("hop link: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

bool bad_slot(int slot, std::string& err) {
    if (slot >= 0 && slot < HopLink::kSlots) return false;
    err = "hop link: slot " + std::to_string(slot);
    return true;
}

// the part common to all three: per slot, the receiver's "copies done" event the next round's sender waits on
struct Base : HopLink {
    int a = 0, b = 0;
    struct Slot {
        cudaEvent_t used = nullptr;   // (device b) the receiver's last copy out of the slot
        bool used_rec = false;
        bool fresh = true;            // the next put starts a round
    } s[kSlots];
    Base(int da, int db) : a(da), b(db) {}
    ~Base() override {
        for (auto& x : s)
            if (x.used) {
                cudaSetDevice(b);
                cudaEventDestroy(x.used);
            }
    }
    // (device a current) a round's first put waits for the last round's copies out
    bool start(int slot, cudaStream_t sa, std::string& err) {
        Slot& x = s[slot];
        if (!x.fresh) return true;
        x.fresh = false;
        return !x.used_rec || ok(cudaStreamWaitEvent(sa, x.used, 0), "wait for the slot", err);
    }
    // (device b current) after a copy out of the slot
    bool mark_used(int slot, cudaStream_t sb, std::string& err) {
        Slot& x = s[slot];
        if (x.used == nullptr && !ok(cudaEventCreateWithFlags(&x.used, cudaEventDisableTiming), "event", err))
            return false;
        x.used_rec = true;
        return ok(cudaEventRecord(x.used, sb), "record", err);
    }
    // (any device) everything the slot's buffer may still be read or written by is done
    bool drain(int slot, std::string& err) {
        Slot& x = s[slot];
        return !x.used_rec || ok(cudaEventSynchronize(x.used), "drain", err);
    }
};

// ---------------------------------------------------------------- host: the pinned host buffer
struct HostLink : Base {
    struct Buf {
        uint8_t* h = nullptr;
        size_t cap = 0;
        cudaEvent_t ready = nullptr;   // (device a) the round's copies in
    } q[kSlots];
    using Base::Base;
    ~HostLink() override {
        for (auto& x : q) {
            if (x.h) cudaFreeHost(x.h);
            if (x.ready) {
                cudaSetDevice(a);
                cudaEventDestroy(x.ready);
            }
        }
    }
    const char* name() const override { return "host"; }
    bool reserve(int slot, size_t bytes, std::string& err) override {
        if (bad_slot(slot, err)) return false;
        Buf& x = q[slot];
        if (bytes <= x.cap) return true;
        if (!drain(slot, err)) return false;
        if (x.ready && !ok(cudaEventSynchronize(x.ready), "drain", err)) return false;
        if (x.h) cudaFreeHost(x.h);
        x.h = nullptr;
        x.cap = 0;
        // (portable: both parts' devices copy through it)
        if (!ok(cudaHostAlloc((void**) &x.h, bytes, cudaHostAllocPortable), "pinned slot", err)) return false;
        x.cap = bytes;
        return true;
    }
    bool put(int slot, size_t off, const void* src, size_t bytes, cudaStream_t sa, std::string& err) override {
        if (bad_slot(slot, err)) return false;
        DevGuard g;
        cudaSetDevice(a);
        if (off + bytes > q[slot].cap && (off > 0 || !reserve(slot, bytes, err))) {
            if (err.empty()) err = "hop link: a piece past its slot's reserve";
            return false;
        }
        return start(slot, sa, err) &&
               ok(cudaMemcpyAsync(q[slot].h + off, src, bytes, cudaMemcpyDeviceToHost, sa), "copy out", err);
    }
    bool ship(int slot, size_t bytes, cudaStream_t sa, std::string& err) override {
        if (bad_slot(slot, err)) return false;
        DevGuard g;
        cudaSetDevice(a);
        Buf& x = q[slot];
        if (x.ready == nullptr && !ok(cudaEventCreateWithFlags(&x.ready, cudaEventDisableTiming), "event", err))
            return false;
        s[slot].fresh = true;
        ++rounds;
        bytes_moved += bytes;
        return ok(cudaEventRecord(x.ready, sa), "record", err);
    }
    bool arrive(int slot, cudaStream_t sb, std::string& err) override {
        if (bad_slot(slot, err)) return false;
        DevGuard g;
        cudaSetDevice(b);
        return q[slot].ready == nullptr || ok(cudaStreamWaitEvent(sb, q[slot].ready, 0), "wait for the rows", err);
    }
    bool get(int slot, size_t off, void* dst, size_t bytes, cudaStream_t sb, std::string& err) override {
        if (bad_slot(slot, err)) return false;
        if (off + bytes > q[slot].cap) {
            err = "hop link: a read past the slot";
            return false;
        }
        DevGuard g;
        cudaSetDevice(b);
        return ok(cudaMemcpyAsync(dst, q[slot].h + off, bytes, cudaMemcpyHostToDevice, sb), "copy in", err) &&
               mark_used(slot, sb, err);
    }
};

// ---------------------------------------------------------------- peer: a buffer on the receiver
struct PeerLink : Base {
    struct Buf {
        uint8_t* d = nullptr;   // (device b)
        size_t cap = 0;
        cudaEvent_t ready = nullptr;
    } q[kSlots];
    using Base::Base;
    ~PeerLink() override {
        for (auto& x : q) {
            cudaSetDevice(b);
            if (x.d) cudaFree(x.d);
            if (x.ready) {
                cudaSetDevice(a);
                cudaEventDestroy(x.ready);
            }
        }
    }
    const char* name() const override { return a == b ? "peer (one device)" : "peer"; }
    bool reserve(int slot, size_t bytes, std::string& err) override {
        if (bad_slot(slot, err)) return false;
        Buf& x = q[slot];
        if (bytes <= x.cap) return true;
        if (!drain(slot, err)) return false;
        if (x.ready && !ok(cudaEventSynchronize(x.ready), "drain", err)) return false;
        DevGuard g;
        cudaSetDevice(b);
        if (x.d) cudaFree(x.d);
        x.d = nullptr;
        x.cap = 0;
        if (!ok(cudaMalloc((void**) &x.d, bytes), "device slot", err)) return false;
        x.cap = bytes;
        return true;
    }
    bool put(int slot, size_t off, const void* src, size_t bytes, cudaStream_t sa, std::string& err) override {
        if (bad_slot(slot, err)) return false;
        if (off + bytes > q[slot].cap && (off > 0 || !reserve(slot, bytes, err))) {
            if (err.empty()) err = "hop link: a piece past its slot's reserve";
            return false;
        }
        DevGuard g;
        cudaSetDevice(a);
        if (!start(slot, sa, err)) return false;
        const cudaError_t e = a == b ? cudaMemcpyAsync(q[slot].d + off, src, bytes, cudaMemcpyDeviceToDevice, sa)
                                     : cudaMemcpyPeerAsync(q[slot].d + off, b, src, a, bytes, sa);
        return ok(e, "peer copy", err);
    }
    bool ship(int slot, size_t bytes, cudaStream_t sa, std::string& err) override {
        if (bad_slot(slot, err)) return false;
        DevGuard g;
        cudaSetDevice(a);
        Buf& x = q[slot];
        if (x.ready == nullptr && !ok(cudaEventCreateWithFlags(&x.ready, cudaEventDisableTiming), "event", err))
            return false;
        s[slot].fresh = true;
        ++rounds;
        bytes_moved += bytes;
        return ok(cudaEventRecord(x.ready, sa), "record", err);
    }
    bool arrive(int slot, cudaStream_t sb, std::string& err) override {
        if (bad_slot(slot, err)) return false;
        DevGuard g;
        cudaSetDevice(b);
        return q[slot].ready == nullptr || ok(cudaStreamWaitEvent(sb, q[slot].ready, 0), "wait for the rows", err);
    }
    bool get(int slot, size_t off, void* dst, size_t bytes, cudaStream_t sb, std::string& err) override {
        if (bad_slot(slot, err)) return false;
        if (off + bytes > q[slot].cap) {
            err = "hop link: a read past the slot";
            return false;
        }
        DevGuard g;
        cudaSetDevice(b);
        return ok(cudaMemcpyAsync(dst, q[slot].d + off, bytes, cudaMemcpyDeviceToDevice, sb), "copy in", err) &&
               mark_used(slot, sb, err);
    }
};

// ---------------------------------------------------------------- sock: a byte stream between the two hosts' halves
// The sender's half: put copies into its pinned out-slot, ship hands the round to the writer thread, which waits for
// the copies (the event) and writes {slot, bytes} + the bytes.  The receiver's half: arrive reads messages (in order;
// one for another slot lands in that slot's in-buffer) until the slot's round is in, then get copies it in.
struct SockLink : Base {
    struct Out {
        uint8_t* h = nullptr;
        size_t cap = 0;
        cudaEvent_t ready = nullptr;
        uint64_t shipped = 0, sent = 0;   // rounds (sent: written by the writer)
    } o[kSlots];
    struct In {
        uint8_t* h = nullptr;
        size_t cap = 0;
        uint64_t got = 0, taken = 0;   // rounds read off the stream / handed to arrive
    } in[kSlots];
    int fd[2] = {-1, -1};   // [0] the sender's end, [1] the receiver's
    struct Job {
        int slot;
        size_t bytes;
    };
    std::deque<Job> jobs;
    std::mutex mu;
    std::condition_variable cv;
    bool stop = false, wfail = false;
    std::string werr;
    std::thread writer;

    SockLink(int da, int db) : Base(da, db) {}
    bool open(std::string& err) {
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, fd) != 0) {
            err = std::string("hop link: socketpair: ") + std::strerror(errno);
            return false;
        }
        writer = std::thread([this] { run(); });
        return true;
    }
    ~SockLink() override {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        if (fd[0] >= 0) shutdown(fd[0], SHUT_RDWR);
        if (writer.joinable()) writer.join();
        for (int k = 0; k < 2; ++k)
            if (fd[k] >= 0) close(fd[k]);
        for (auto& x : o) {
            if (x.h) cudaFreeHost(x.h);
            if (x.ready) {
                cudaSetDevice(a);
                cudaEventDestroy(x.ready);
            }
        }
        for (auto& x : in)
            if (x.h) cudaFreeHost(x.h);
    }
    static bool write_all(int f, const void* p, size_t n) {
        const uint8_t* c = (const uint8_t*) p;
        while (n > 0) {
            const ssize_t w = ::send(f, c, n, MSG_NOSIGNAL);
            if (w <= 0) {
                if (w < 0 && errno == EINTR) continue;
                return false;
            }
            c += w;
            n -= (size_t) w;
        }
        return true;
    }
    static bool read_all(int f, void* p, size_t n) {
        uint8_t* c = (uint8_t*) p;
        while (n > 0) {
            const ssize_t r = ::recv(f, c, n, 0);
            if (r <= 0) {
                if (r < 0 && errno == EINTR) continue;
                return false;
            }
            c += r;
            n -= (size_t) r;
        }
        return true;
    }
    void run() {
        cudaSetDevice(a);
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || !jobs.empty(); });
                if (stop) return;
                j = jobs.front();
                jobs.pop_front();
            }
            Out& x = o[j.slot];
            const uint64_t hdr[2] = {(uint64_t) j.slot, (uint64_t) j.bytes};
            const cudaError_t e = cudaEventSynchronize(x.ready);
            const bool w = e == cudaSuccess && write_all(fd[0], hdr, sizeof hdr) && write_all(fd[0], x.h, j.bytes);
            {
                std::lock_guard<std::mutex> lk(mu);
                if (!w) {
                    wfail = true;
                    werr = e != cudaSuccess ? cudaGetErrorString(e) : "the stream closed";
                }
                ++x.sent;
            }
            cv.notify_all();
        }
    }
    const char* name() const override { return "sock (in-process stream)"; }
    bool grow(uint8_t*& h, size_t& cap, size_t bytes, std::string& err) {
        if (bytes <= cap) return true;
        if (h) cudaFreeHost(h);
        h = nullptr;
        cap = 0;
        if (!ok(cudaHostAlloc((void**) &h, bytes, cudaHostAllocPortable), "pinned slot", err)) return false;
        cap = bytes;
        return true;
    }
    // (sender) the slot's last round is written: its out-buffer is free
    bool out_free(int slot, std::string& err) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return o[slot].sent == o[slot].shipped || wfail; });
        if (wfail) {
            err = "hop link: sock: " + werr;
            return false;
        }
        return true;
    }
    bool reserve(int slot, size_t bytes, std::string& err) override {
        if (bad_slot(slot, err)) return false;
        if (bytes <= o[slot].cap) return true;
        return out_free(slot, err) && grow(o[slot].h, o[slot].cap, bytes, err);
    }
    bool put(int slot, size_t off, const void* src, size_t bytes, cudaStream_t sa, std::string& err) override {
        if (bad_slot(slot, err)) return false;
        if (s[slot].fresh && !out_free(slot, err)) return false;
        if (off + bytes > o[slot].cap && (off > 0 || !reserve(slot, bytes, err))) {
            if (err.empty()) err = "hop link: a piece past its slot's reserve";
            return false;
        }
        DevGuard g;
        cudaSetDevice(a);
        s[slot].fresh = false;   // (the receiver's copies out of ITS buffer are its own concern: arrive drains them)
        return ok(cudaMemcpyAsync(o[slot].h + off, src, bytes, cudaMemcpyDeviceToHost, sa), "copy out", err);
    }
    bool ship(int slot, size_t bytes, cudaStream_t sa, std::string& err) override {
        if (bad_slot(slot, err)) return false;
        DevGuard g;
        cudaSetDevice(a);
        Out& x = o[slot];
        if (x.ready == nullptr && !ok(cudaEventCreateWithFlags(&x.ready, cudaEventDisableTiming), "event", err))
            return false;
        if (!ok(cudaEventRecord(x.ready, sa), "record", err)) return false;
        s[slot].fresh = true;
        ++rounds;
        bytes_moved += bytes;
        {
            std::lock_guard<std::mutex> lk(mu);
            ++x.shipped;
            jobs.push_back({slot, bytes});
        }
        cv.notify_all();
        return true;
    }
    bool arrive(int slot, cudaStream_t, std::string& err) override {
        if (bad_slot(slot, err)) return false;
        while (in[slot].got == in[slot].taken) {   // read the stream until this slot's next round is in
            uint64_t hdr[2];
            if (!read_all(fd[1], hdr, sizeof hdr) || hdr[0] >= (uint64_t) kSlots) {
                err = "hop link: sock: the stream broke";
                return false;
            }
            const int k = (int) hdr[0];
            const size_t n = (size_t) hdr[1];
            In& x = in[k];
            if (x.got != x.taken) {
                err = "hop link: sock: two rounds of slot " + std::to_string(k) + " before the first was read";
                return false;
            }
            // (its last round's copies in are done before the buffer is written again)
            if (!drain(k, err) || !grow(x.h, x.cap, n, err) || !read_all(fd[1], x.h, n)) {
                if (err.empty()) err = "hop link: sock: the stream broke";
                return false;
            }
            ++x.got;
        }
        ++in[slot].taken;
        return true;
    }
    bool get(int slot, size_t off, void* dst, size_t bytes, cudaStream_t sb, std::string& err) override {
        if (bad_slot(slot, err)) return false;
        if (off + bytes > in[slot].cap) {
            err = "hop link: a read past the slot";
            return false;
        }
        DevGuard g;
        cudaSetDevice(b);
        return ok(cudaMemcpyAsync(dst, in[slot].h + off, bytes, cudaMemcpyHostToDevice, sb), "copy in", err) &&
               mark_used(slot, sb, err);
    }
};

}  // namespace

std::unique_ptr<HopLink> HopLink::make(int dev_a, int dev_b) {
    const char* v = getenv("STRATA_GLM_HOP");
    const std::string want = v != nullptr ? v : "host";
    if (want == "peer") {
        int can = 0;
        if (dev_a == dev_b || (cudaDeviceCanAccessPeer(&can, dev_a, dev_b) == cudaSuccess && can)) {
            if (dev_a != dev_b) {
                DevGuard g;
                cudaSetDevice(dev_a);
                const cudaError_t e = cudaDeviceEnablePeerAccess(dev_b, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) can = 0;
                cudaGetLastError();
            }
            if (dev_a == dev_b || can) return std::unique_ptr<HopLink>(new PeerLink(dev_a, dev_b));
        }
        cudaGetLastError();
        std::fprintf(stderr, "glm split: CUDA%d -> CUDA%d has no peer access - the hop goes through host memory\n",
                     dev_a, dev_b);
    } else if (want == "sock") {
        auto* l = new SockLink(dev_a, dev_b);
        std::string err;
        if (l->open(err)) return std::unique_ptr<HopLink>(l);
        delete l;
        std::fprintf(stderr, "glm split: %s - the hop goes through host memory\n", err.c_str());
    } else if (want != "host") {
        std::fprintf(stderr, "glm split: STRATA_GLM_HOP=%s is not host / peer / sock - host\n", want.c_str());
    }
    return std::unique_ptr<HopLink>(new HostLink(dev_a, dev_b));
}

}  // namespace strata::core
