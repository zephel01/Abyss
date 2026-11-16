// include/strata/prefill/moe_own.hpp - the prompt path's experts on Strata's own int8 tensor-core grouped GEMM (no
// llama.cpp sources): the weights stay quantized, the activations are rounded to int8 with one scale a 32-value
// block, and the products run on int8 tensor cores, one launch for a set of experts.
//
//   gate/up:  h[row, j] = silu(min(gate . x, limit)) * clamp(up . x, -limit, limit)   (the SwiGLU in the write-back)
//   down:     out[row, i] = down[i] . h
//
// Every output is one sum over K in one fixed order, whatever else the launch carries (the expert's other rows, the
// set's other experts, the chunking): a given binary gives the same bits for the same row.
// The sums are not the MMQ kernels' sums (another order, another rounding of the activations): the prompt path keeps
// MMQ selectable (STRATA_PROMPT_MMQ=1).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::own {

/// This build and the current device have the kernels (CUDA sm_80+; a HIP build has none yet).
bool built();
/// The kernels read weights of this ggml type.
bool supported(int ggml_type);
/// supported(type) and a [rows, cols] matrix of it fits the kernels' tiles (cols a multiple of 256 - of 128 for the
/// types of 32- and 64-value blocks -, rows of 16).
bool usable(int ggml_type, int64_t rows, int64_t cols);

/// Bytes of `rows` activation rows of `cols` values in the kernels' form: an int8 plane [rows][cols], then a float
/// plane [rows][cols / 32] (the blocks' scales).  cols a multiple of 128.  (The int8 plane is in the kernels' own
/// word order, see moe_own.cu.)
size_t act_bytes(int64_t rows, int64_t cols);
/// Row i of the output is row ids[i] of x (row i when ids is null); x has `ld` floats a row.  xq: 16-byte aligned.
void quantize(const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld, int64_t rows, void* stream);

/// A set of n experts whose blobs lie `stride` bytes apart from `w`.  Expert e owns the activation rows
/// [bounds[e], bounds[e + 1]) (device table, n + 1 entries; `host_bounds` the same values on the host); row b is row
/// b - row0 of `xq` (as quantize wrote `rows` rows of it) and of `dst` (`ld` floats a row).
struct Set {
    const void* w = nullptr;
    size_t stride = 0;
    int n = 0;
    const int32_t* bounds = nullptr;
    const int32_t* host_bounds = nullptr;
    int row0 = 0;
    const void* xq = nullptr;
    int64_t rows = 0;
    float* dst = nullptr;
    int64_t ld = 0;
};

/// The launch context: a small device table per launch.  One per prompt path; every launch of a context goes to the
/// same stream.
class Context {
public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    /// Gate [n_ff, n_embd] at the blob's start and up [n_ff, n_embd] `up_off` bytes on, both of `type`: dst row r gets
    /// h in its first n_ff floats.
    void gate_up(const Set& s, int type, int64_t n_ff, int64_t n_embd, size_t up_off, float limit, void* stream);
    /// Down [n_embd, n_ff] of `type`, `down_off` bytes into the blob: dst row r gets n_embd floats.
    void down(const Set& s, int type, int64_t n_embd, int64_t n_ff, size_t down_off, void* stream);

private:
    int32_t* tiles_ = nullptr;   // the set's tile prefix table
};

}  // namespace strata::prefill::own
