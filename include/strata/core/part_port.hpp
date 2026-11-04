// SPDX-License-Identifier: MIT
#pragma once
//
// The next part of a layer split as the part before it sees it.  A part never reaches into a later part's state: it
// sends its rows through its HopLink and asks the next part's PartPort for the rest - run a token, a prompt chunk or a
// verify window on its layers (and on every later part's), wait for the last part's answer, sample, rewind, draft.
// LocalPort runs them on a Glm5Model in this process (Glm5Model::part_*); a port to a part in another process sends
// the same requests over a stream and its server calls the same part_* functions.  Qwen3.8 (qwen4exp) splits only,
// for now: the other models' splits still walk split_next_.

#include "strata/kernels/sampler.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

class Glm5Model;
class HopLink;

class PartPort {
public:
    virtual ~PartPort() = default;
    virtual const char* name() const = 0;
    // between tokens (every device idle): the tiers' finished moves go live (fast_boundary)
    virtual bool boundary(std::string& err) = 0;
    // the token at the next position (hash routing, image rows)
    virtual void token_meta(int32_t token, bool image) = 0;
    // position p (pos: the position after it): the rows arrive through the link's kToken slot; the layers, the later
    // parts', the head and its argmax on the last part - queued, token_wait() collects it
    virtual bool token(int64_t p, int64_t pos, std::string& err) = 0;
    virtual int token_wait(std::string& err) = 0;
    // the last token's logits: sampled on the last part / copied out
    virtual int sample(strata::kernels::SamplerParams& sp, std::string& err) = 0;
    virtual bool logits(std::vector<float>& out, std::string& err) = 0;
    // a prompt chunk at p0: its rows from the link's `slot` (< 0: already in), the chunk's tokens and each row's next
    // token (the drafter's rows); chain: the later parts too
    virtual bool prompt(int slot, int64_t p0, int T, const int32_t* toks, const int32_t* next, bool chain,
                        std::string& err) = 0;
    // a verify window of toks.size() columns at p0 (rows through kWindow), then its argmax per column
    virtual bool window(const std::vector<int32_t>& toks, int64_t p0, int64_t pos, std::string& err) = 0;
    virtual bool window_wait(std::vector<int32_t>& argmax, std::string& err) = 0;
    virtual int window_sample(int i, strata::kernels::SamplerParams& sp, std::string& err) = 0;
    virtual bool window_logits(int i, std::vector<float>& out, std::string& err) = 0;
    // a window's tail dropped: the states back to pos
    virtual void rewind(int64_t pos) = 0;
    // the drafter (qwen4exp's MTP layer, on the last part)
    virtual bool mtp_on() const = 0;
    virtual bool mtp_commit(int32_t token, std::string& err) = 0;
    virtual bool mtp_draft(int32_t y, int32_t* drafts, float* conf, std::string& err) = 0;
    // after a token or window: its routes answered; the profiler's marks collected; a stalled token's state printed
    virtual bool route_ok(std::string& err) = 0;
    virtual void collect() = 0;
    virtual void stall_report(int64_t p) = 0;

    // the next part in this process; `in` is the link the rows arrive through (the previous part's)
    static std::unique_ptr<PartPort> local(Glm5Model* next, HopLink* in);
};

}  // namespace strata::core
