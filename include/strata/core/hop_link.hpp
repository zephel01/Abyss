// SPDX-License-Identifier: MIT
#pragma once
//
// The hand-over between two parts of a layer split: the rows a part ends with (a token's residual streams, a prompt
// chunk's, a verify window's columns) go from its device to the next part's.  Every boundary goes through one HopLink,
// so the way the bytes move is one swappable piece:
//   host  the pinned host buffer, ordered on the devices by events - what every split did (the default);
//   peer  device to device (cudaMemcpyPeerAsync into a buffer on the receiver; on a shared device a plain copy) - no
//         host memory and no host round trip (cards without peer access: host);
//   sock  a byte stream: the sender's host waits for each round's rows and writes them, the receiver reads them before
//         its copy in.  In one process the stream is a socketpair; it is the shape a link between two PCs takes (a
//         part in another process or on another machine), so its cost on this machine is the floor of such a link.
// STRATA_GLM_HOP=host|peer|sock picks it (each boundary falls back to host when its choice cannot be set up).
//
// A round: the sender put()s its rows (one or more pieces at offsets of the slot) and ship()s them on its stream; the
// receiver arrive()s and get()s them (pieces at offsets) on its stream.  Each call site has its own slot, so rounds of
// different sites never share a buffer; a slot's next round waits until the receiver's copies of the last one are done
// (on the device: an event the sender's stream waits on).  The calls set the device they need and put the caller's
// back.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace strata::core {

class HopLink {
public:
    enum Slot : int {
        kToken = 0,   // a token's residual (fast_token's split loop)
        kStep,        // a token's residual (hop_to_next: the pipelined decode's group hops)
        kSpec0,       // the pipelined decode's head -> tail hand-over, by position parity
        kSpec1,
        kPrompt,      // a prompt chunk's rows (the serial prompt path)
        kPipe0,       // a prompt chunk's rows (the pipelined prompt path), by chunk parity
        kPipe1,
        kWindow,      // a verify window's columns
        kSlots
    };
    virtual ~HopLink() = default;
    virtual const char* name() const = 0;
    // room for `bytes` in a slot (a round of several pieces reserves its whole size first; a single piece grows it)
    virtual bool reserve(int slot, size_t bytes, std::string& err) = 0;
    virtual bool put(int slot, size_t off, const void* src, size_t bytes, cudaStream_t sa, std::string& err) = 0;
    virtual bool ship(int slot, size_t bytes, cudaStream_t sa, std::string& err) = 0;
    virtual bool arrive(int slot, cudaStream_t sb, std::string& err) = 0;
    virtual bool get(int slot, size_t off, void* dst, size_t bytes, cudaStream_t sb, std::string& err) = 0;

    // one piece each way: put + ship, arrive + get
    bool send(int slot, const void* src, size_t bytes, cudaStream_t sa, std::string& err) {
        return put(slot, 0, src, bytes, sa, err) && ship(slot, bytes, sa, err);
    }
    bool recv(int slot, void* dst, size_t bytes, cudaStream_t sb, std::string& err) {
        return arrive(slot, sb, err) && get(slot, 0, dst, bytes, sb, err);
    }

    uint64_t rounds = 0, bytes_moved = 0;   // (shipped)

    // the link from dev_a's part to dev_b's (STRATA_GLM_HOP), never null: a choice that cannot be set up is host
    static std::unique_ptr<HopLink> make(int dev_a, int dev_b);
};

}  // namespace strata::core
