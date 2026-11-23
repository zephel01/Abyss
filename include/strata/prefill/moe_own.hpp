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
/// The current device is one the kernels were measured against MMQ on (sm_120: RTX 5090 class): the prompt path's
/// default types apply there.
bool tuned_device();
/// The prompt path gives products of this type to the own GEMM instead of MMQ: STRATA_PROMPT_MMQ=1 - none;
/// STRATA_PROMPT_OWN_TYPES=all | <type id>,.. - those; otherwise the default list on a tuned device (none elsewhere), and
/// every covered type in a build without MMQ (`mmq_built` false).
/// `dense`: a dense layer's product (not an expert's); the same list applies.
bool type_on(int ggml_type, bool mmq_built, bool dense = false);
/// The kernels read weights of this ggml type.
bool supported(int ggml_type);
/// supported(type) and a [rows, cols] matrix of it fits the kernels' tiles (cols a multiple of 256 - of 128 for the
/// types of 32- and 64-value blocks -, rows of 16).
bool usable(int ggml_type, int64_t rows, int64_t cols);

/// Bytes of `rows` activation rows of `cols` values in the kernels' form: an int8 plane [rows][cols], then a float
/// plane [rows][cols / 32] (the blocks' scales) and - `sums` - another of that shape (the blocks' sums).  cols a multiple of 128.  (The int8 plane is in the kernels' own
/// word order, see moe_own.cu.)
size_t act_bytes(int64_t rows, int64_t cols, bool sums = false);
/// The int8 plane holds the values in their own order (the WMMA backend); false: in the mma.sync threads' order.
bool act_natural_order();
/// The type's products read the activations' block sums: quantize its activations with `sums`.
bool needs_sums(int ggml_type);
/// (tests: false = the kernels that read a matrix on a multiple of 2 only, also for one on a multiple of 4)
void test_align4(bool on);
/// Row i of the output is row ids[i] of x (row i when ids is null); x has `ld` floats a row.  xq: 16-byte aligned.
void quantize(const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld, int64_t rows, void* stream,
              bool sums = false);

/// STRATA_INK_FP4ACT (opt-in, lossy): NVFP4 weights times FP4 activations on the block-scaled FP4 tensor cores (CUDA
/// sm_120a, moe_fp4.cuh).  The current device runs them (an sm_120 GPU, a binary with sm_120a code); false on HIP.
bool fp4_built();
/// Bytes of `rows` activation rows of `cols` values in fp4_quantize's form (1 or 2 terms; the row scales apart).
size_t fp4_act_bytes(int64_t rows, int64_t cols, int terms);
/// A [rows_out, K] NVFP4 product fits the FP4 kernels' tiles.
bool fp4_usable(int64_t rows_out, int64_t K);
/// The NVFP4 experts' gate/up scale folded into the row scales: row i (of a gather by token ids) is sorted row
/// row_base + i, the route of its token whose pos is that row has expert ids[route], scale nv[2 x expert].
struct Fp4Fold {
    const int32_t* ids = nullptr;
    const int32_t* pos = nullptr;
    int k = 0;
    const float* nv = nullptr;
    int row_base = 0;
};
/// Row i of the output is row ids[i] of x (i when ids is null): one float scale a row into `row_scale`, then E2M1 codes
/// with a UE4M3 scale a 16-value block (terms 2: the first term's residual the same way).  xq: 16-byte aligned.
void fp4_quantize(const float* x, const int32_t* ids, void* xq, float* row_scale, int64_t cols, int64_t ld, int64_t rows,
                  int terms, void* stream, const Fp4Fold* fold = nullptr);

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
    /// (Any plain product: a [rows, K] matrix `off` bytes into the blob as (type, rows, K, off) - a gate or an up
    /// matrix alone, a dense layer's projection with one "expert" whose rows are the tokens.)
    void down(const Set& s, int type, int64_t n_embd, int64_t n_ff, size_t down_off, void* stream);

    /// An NVFP4 [rows_out, K] matrix `off` bytes into each blob times fp4_quantize's activations (s.xq, `row_scale`,
    /// `terms`): dst row r gets rows_out floats - the sums without the experts' second-level scale, as down() writes
    /// them.  false: the shape or the alignment does not fit (nothing ran: take the exact path).
    /// `glu`: gate [rows_out, K] at `off` and up `up_off` bytes on: dst row r gets h = act(gate, up) in its first
    /// rows_out floats (the SwiGLU of swiglu_rows, `limit`; the row scales carry the experts' gate/up scale).
    bool fp4(const Set& s, const float* row_scale, int terms, int64_t rows_out, int64_t K, size_t off, void* stream,
             bool glu = false, size_t up_off = 0, float limit = 0.0f);

private:
    int32_t* tiles_ = nullptr;   // the set's tile prefix table
};

}  // namespace strata::prefill::own
