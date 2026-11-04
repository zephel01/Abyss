// SPDX-License-Identifier: MIT
//
// PartPort's in-process port (include/strata/core/part_port.hpp): every request is the next part's part_* call.

#include "strata/core/part_port.hpp"

#include "strata/core/glm_model.hpp"

namespace strata::core {

namespace {

struct LocalPort final : PartPort {
    Glm5Model* m;
    HopLink* in;
    LocalPort(Glm5Model* next, HopLink* link) : m(next), in(link) {}
    const char* name() const override { return "local"; }
    bool boundary(std::string& err) override { return m->part_boundary(err); }
    void token_meta(int32_t token, bool image) override { m->part_token_meta(token, image); }
    bool token(int64_t p, int64_t pos, std::string& err) override { return m->part_token(in, p, pos, err); }
    int token_wait(std::string& err) override { return m->part_token_wait(err); }
    int sample(strata::kernels::SamplerParams& sp, std::string& err) override { return m->part_sample(sp, err); }
    bool logits(std::vector<float>& out, std::string& err) override { return m->part_logits(out, err); }
    bool prompt(int slot, int64_t p0, int T, const int32_t* toks, const int32_t* next, bool chain,
                std::string& err) override {
        return m->part_prompt(in, slot, p0, T, toks, next, chain, err);
    }
    bool window(const std::vector<int32_t>& toks, int64_t p0, int64_t pos, std::string& err) override {
        return m->part_window(in, toks, p0, pos, err);
    }
    bool window_wait(std::vector<int32_t>& argmax, std::string& err) override {
        return m->part_window_wait(argmax, err);
    }
    int window_sample(int i, strata::kernels::SamplerParams& sp, std::string& err) override {
        return m->part_window_sample(i, sp, err);
    }
    bool window_logits(int i, std::vector<float>& out, std::string& err) override {
        return m->part_window_logits(i, out, err);
    }
    void rewind(int64_t pos) override { m->part_rewind(pos); }
    bool mtp_on() const override { return m->part_mtp_on(); }
    bool mtp_commit(int32_t token, std::string& err) override { return m->part_mtp_commit(token, err); }
    bool mtp_draft(int32_t y, int32_t* drafts, float* conf, std::string& err) override {
        return m->part_mtp_draft(y, drafts, conf, err);
    }
    bool route_ok(std::string& err) override { return m->part_route_ok(err); }
    void collect() override { m->part_collect(); }
    void stall_report(int64_t p) override { m->part_stall_report(p); }
};

}  // namespace

std::unique_ptr<PartPort> PartPort::local(Glm5Model* next, HopLink* in) {
    return std::unique_ptr<PartPort>(new LocalPort(next, in));
}

}  // namespace strata::core
