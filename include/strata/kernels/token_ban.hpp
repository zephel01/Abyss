// include/strata/kernels/token_ban.hpp - per-request logit biases: banned tokens (-inf) and OpenAI logit_bias.
//
// WHERE IT APPLIES.  Every token pick reads the logits through one of three primitives - glm_fast::argmax,
// glm_fast::argmax_mb and sample_tokens - so those apply the active set to the rows they are given, in place, right
// before selecting: greedy and sampled picks, the verify windows' columns (n-gram, MTP / NextN, DSpark), the
// drafters' own heads (a draft never proposes a banned token).  A banned token's logit is -inf, so it is never the
// argmax and has probability 0 in the sampled chain; a draft is accepted only when it equals the masked pick, so a
// banned draft is always rejected.  Rows of another width (a reduced-vocabulary drafter) are left as they are.
//
// NO SET ACTIVE: nothing runs - not a kernel, not a branch on the device - so outputs are bit-identical.
//
// SWITCHING MID-DECODE (the answer-only profile turns on at </think>): a verify window computed before the switch
// holds unmasked picks for the columns after it; Glm5Model re-picks such a column when its pick is banned
// (bias_banned), and drops the window's batched draws (bias_generation) - see Glm5Model::bias_repick.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::kernels {

/// A set of (token id, added value) pairs over a vocabulary of n_vocab; add = -inf bans the token.
struct BiasSet {
    int n_vocab = 0;
    std::vector<int32_t> ids;
    std::vector<float> add;
    std::vector<uint8_t> banned;   // host: 1 for each id whose add is -inf (bias_banned)
    // device copies (lazily, per device: the logits of a split live on the tail's GPU)
    struct Dev { int32_t* ids = nullptr; float* add = nullptr; };
    std::vector<Dev> dev;
    ~BiasSet();
};

/// Parses "<id>:<value>,<id>:<value>,..." (value <= -100 or "-inf": banned; OpenAI's -100 means "ban").  Ids outside
/// [0, n_vocab) and duplicate ids (the last one wins) are an error / folded.  nullptr + err on a malformed list.
std::shared_ptr<BiasSet> bias_parse(const std::string& spec, int n_vocab, std::string& err);

/// The set the token picks apply from now on (nullptr: none).  The caller keeps the set alive while active.
void bias_activate(const BiasSet* set);
const BiasSet* bias_active();
/// Counts bias_activate calls that changed the set: a batched draw made under another generation is stale.
uint64_t bias_generation();
/// Host check: the active set bans `id`.
bool bias_banned(int id);
/// Applies the active set to `rows` rows of n_vocab logits (row stride ld floats) on `stream`; nothing when no set
/// is active or the width is not the set's.  First use on a device copies the set there (once; skipped while the
/// stream is being captured into a graph).
void bias_apply(float* logits, int rows, int n_vocab, long long ld, void* stream);

}  // namespace strata::kernels
