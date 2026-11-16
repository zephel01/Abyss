// include/strata/kernels/qsa27.hpp - QSA on the dense qwen35 (first: higa2515/Qwen3.8-27B-QSA-GGUF; the layer and
// head counts come from the GGUF, so any qwen35 dense GGUF with these indexers runs): the indexer pieces that differ
// from Qwen3.8-Flash-Next's (qsa.hpp / qsa_select.hpp, which the 27B path reuses for the block scores, the top-k and
// the attention over the selected cells).
//
// The model card's reference (Qwen3.8-27B-QSA-BF16, modeling_qsa.py / indexer.py) on each full-attention layer (16 on
// the 27B), from the layer's normed input x (FP32 weights, the norms stored as 1 + w):
//
//   q[h]  = rope(rms(x . q_proj[h]) * q_norm, p)               4 heads x 128, NEOX rope on the first 64 dims
//   k[c]  = rms(x_c . k_proj) * k_norm.pre_weight               a cell's key, normed BEFORE pooling
//   K[b]  = rope(rms(mean(k[4b .. 4b+3])) * k_norm.weight, 4b)  a block's key once its 4 cells exist, rotated at its
//                                                               first cell
//   s[b]  = sum_h relu(q[h] . K[b])                             (the reference's * 128^-0.5 does not change the order)
//
// A query at position p (n_kv = p + 1 cells) attends to the 512 (= top_k 2048 / 4) best complete blocks plus the cells
// of its own block up to p (the "current block": always kept, also when it is complete and not among the 512).  Up to
// 2,051 cells that is every cell (dense).  The selection is qsa_block_topk's over cells weighted by their block, the
// incomplete tail block given +1e9 (qsa_block_scores): width = top_k + the tail's cells keeps the tail plus exactly 512
// whole blocks.  A query whose own block is complete (n_kv % 4 == 0) selects top_k cells (512 blocks) and
// qsa27_own_block adds its block's 4 cells when the top-k left them out (the reference's union).
//
// What Flash-Next does differently, so the two cannot share the key kernel: there the raw keys are pooled first and
// normed once (no per-cell norm, one norm weight), and its selection width is top_k + 3 cells at every position.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::qsa27 {

constexpr int kDim = 128;     ///< indexer key width
constexpr int kHeads = 4;     ///< indexer query heads
constexpr int kBlock = 4;     ///< cells a block
constexpr int kRot = 64;      ///< rotated dims (NEOX pairs i, i + 32)

/// The QsaShapes the shared kernels (qsa_block_scores / qsa_block_topk) are called with: the indexer geometry these
/// kernels take (4 heads x 128, blocks of 4, 64 rotated dims - checked against the GGUF at load) and the model's own
/// attention shape and top_k from the GGUF (any qwen35 dense with indexers: Qwen3.8-27B, a Qwen3.5-9B with them).
inline QsaShapes shapes(int top_k, int n_head = 0, int n_head_kv = 0, int head_dim = 256) {
    QsaShapes s;
    s.n_head = n_head; s.n_head_kv = n_head_kv; s.head_dim = head_dim; s.n_rot = kRot;
    s.idx_n_head = kHeads; s.idx_dim = kDim; s.idx_block = kBlock; s.idx_top_k = top_k;
    s.page_size = 4;
    return s;
}

/// The selection width before the own-block step: top_k + the incomplete block's cells, at most n_kv (n_kv <= the
/// width: every cell).
inline int64_t width(int64_t n_kv, int top_k) {
    const int64_t w = (int64_t) top_k + n_kv % kBlock;
    return n_kv < w ? n_kv : w;
}
/// The most cells a selection holds (the width plus the own block's 4): the ids buffers' row length.
inline int64_t cap(int top_k) { return (int64_t) top_k + kBlock; }
/// Does a query at position p attend to every cell (its selection is all of them)?
inline bool dense_at(int64_t p, int top_k) { return p + 1 <= width(p + 1, top_k); }

/// A query's step record (qsa.hpp's layout: pos, n_kv, n_bid, width) with this model's width.
void step_fill(int32_t* host_step, int64_t pos, int top_k);

/// The raw keys of cells pos0 .. pos0 + n - 1 (row i of raw, ld floats apart, is cell pos0 + i): every block they
/// complete pooled into pooled[b] (its earlier cells' raw keys read from tail, slot = cell % 4), then the cells of a
/// block they leave incomplete into tail.  w_pre / w_blk: k_norm.pre_weight / k_norm.weight (1 + w); cos / sin: the
/// (positions, 32) rope table.  Two launches; n == 1 is the token path's.
void key_append_rows(const float* raw, int64_t ld, int64_t pos0, int n, const float* w_pre, const float* w_blk,
                     float eps, float* tail, float* pooled, const float* cos_tab, const float* sin_tab,
                     cudaStream_t s);

/// The reference's union with the current block, for nq queries: steps_in[q] the record the top-k ran with, ids[q]
/// (ld ids a row) its ascending cells.  A query with n_kv % 4 == 0 past its width whose last id is not n_kv - 1 gets
/// the block's 4 cells appended.  steps_out[q] = steps_in[q] with the width the attention reads (steps_in is shared by
/// the layers: not written).  ld >= cap(top_k).
void own_block(int32_t* ids, int64_t ld, const int32_t* steps_in, int32_t* steps_out, int nq, cudaStream_t s);

/// A test control (STRATA_Q35_QSA_PICK): overwrite the nq rows' block scores (max_blocks a row) with a rule that ignores
/// the indexer - mode 1 the most recent blocks (score = block index: a 2,048-cell sliding window), mode 2 the first
/// ones (score = -index) - the tail block keeps its +1e9.
void pick_scores(float* scores, int64_t max_blocks, const int32_t* steps, int nq, int mode, cudaStream_t s);

}  // namespace strata::kernels::qsa27
