// include/strata/kernels/qwen35_fast.hpp - the qwen35moe (Qwen3.5-MoE: Ornith-1.5-122B) decode kernels that have no
// equivalent in glm_fast / dsv4_fast: the gated delta-net's per-head gates (gdn.hpp's kernels are the recurrence
// itself), the attention layers' q / k norms + NEOX RoPE + KV cache write, the GQA attention of one token over the
// FP16 cache (the positions in chunks, then merged), the q projection's sigmoid output gate and the shared expert's
// sigmoid gate.  The math is llama.cpp's src/models/qwen35moe.cpp (build_layer_attn, build_layer_attn_linear,
// build_layer_ffn) and src/models/delta-net-base.cpp (build_delta_net_autoregressive).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::q35 {

/// x[i] = silu(x[i]), i < n.
void silu(float* x, int n, cudaStream_t s);
/// x[i] *= a, i < n.
void scale(float* x, int n, float a, cudaStream_t s);
/// The delta-net's per-head gates: gate[h] = softplus(alpha[h] + dt_bias[h]) * ssm_a[h] (ssm_a = -exp(A_log): the
/// decay exp(gate) lies in (0, 1]); beta[h] = sigmoid(beta[h]) in place.
void gdn_gates(const float* alpha, const float* dt_bias, const float* ssm_a, float* beta, float* gate, int h_v,
               cudaStream_t s);
/// `kv` (--kv on the qwen35moe path): the caches' format - 0 FP16, 1 q4_0 (K and V Hadamard-rotated q4_0), 2 int8 (K
/// and V, 8-bit codes, a scale per 32 dims), 3 k8v4 (int8 K, rotated q4_0 V); the layouts in qwen35_fast.cu.  A rotated
/// K: q is rotated the same way by the prologue; a rotated V: attn_gate / attn_gate_rows rotate the output back.  The
/// attention reads the rows in place.  Every function below that takes a cache takes kv.
/// The bytes of one cache row (a position and a KV head, head_dim 256) of K (v false) or V in format kv.
inline size_t kv_row_bytes(int kv, bool v) {
    const int f = kv == 1 ? 1 : kv == 2 ? 2 : kv == 3 ? (v ? 1 : 2) : 0;
    return f == 1 ? 144 : f == 2 ? 272 : 512;
}
/// The attention layer's prologue at position p: q_full ([n_head][2 * head_dim], a head's q then its gate) -> q
/// ([n_head][head_dim]: rms-normed by q_norm, NEOX RoPE on the first n_rot dims); k ([n_head_kv][head_dim]: k_norm,
/// RoPE) and v into row p of the FP16 caches kc / vc ([ctx][n_head_kv][head_dim]).  head_dim 256.
void attn_prep(const float* q_full, const float* k, const float* v, const float* q_norm, const float* k_norm, float eps,
               int n_head, int n_head_kv, int head_dim, int n_rot, float rope_base, int p, uint16_t* kc, uint16_t* vc,
               float* q, cudaStream_t s, int kv = 0);
/// The scratch (floats) attn_decode needs for n_head heads over a cache of max_ctx positions.
size_t attn_part_floats(int n_head, int head_dim, int max_ctx);
/// One token's attention: out[h] = softmax(scale * q[h] . K[g][t]) V[g][t] over t < n_pos, g = h / (n_head /
/// n_head_kv), from the FP16 caches; the positions in chunks (a block each: a partial max / sum / accumulator into
/// part), then merged.  head_dim 256.
void attn_decode(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_head_kv, int head_dim,
                 int n_pos, float scale, float* part, float* out, cudaStream_t s, int kv = 0);
/// attn_decode for `rows` rows at once, each over its own cells: row r's query q[r] ([n_head][head_dim]) attends to the
/// cache rows ids[r * ld_ids + j], j < steps[r * 4 + 3] (<= max_ids; a QSA step record a row, device), read in place,
/// into out[r]; bit-identical to gather_rows + attn_decode per row.  ids null: row r attends to cells 0 .. its width - 1
/// (the dense rows, bit-identical to attn_decode on the cache).  hpw: query heads a block (0: by the shape).  part: attn_rows_part_floats(rows, ..) floats.
/// false (nothing launched): a shape attn_decode runs another kernel for; the caller keeps its per-row path.
size_t attn_rows_part_floats(int rows, int n_head, int head_dim, int max_ids);
bool attn_decode_rows(const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* ids, int ld_ids,
                      const int32_t* steps, int rows, int max_ids, int n_head, int n_head_kv, int head_dim, float scale,
                      float* part, float* out, cudaStream_t s, int hpw = 0);
/// dst[i] = src[i] by a kernel (src may be mapped host memory: no copy engine).
void copy_f32(float* dst, const float* src, int n, cudaStream_t s);
/// The token's QSA step record (4 ints) and the 64 rope positions written by a kernel (the values by argument), not
/// two H2D copies that queue behind the background expert copies on the copy engine; the same values.
void q4_step_set(int32_t* step, int32_t* pos, int s0, int s1, int s2, int s3, int p, cudaStream_t s);
/// o[h][d] *= sigmoid(q_full[h][head_dim + d]) (the q projection's gate half), in place.
void attn_gate(float* o, const float* q_full, int n_head, int head_dim, cudaStream_t s, int kv = 0);
/// x[i] *= sigmoid(g[0]), i < n (the shared expert's gate).
void scale_sigmoid(float* x, int n, const float* g, cudaStream_t s);
/// x[t][i] *= sigmoid(g[t]) over T rows of n (the window's columns).
void scale_sigmoid_rows(float* x, int T, int n, const float* g, cudaStream_t s);

// ---- the prompt path: the same layers over T rows (a chunk's sub-batch of tokens, row t = position p0 + t)
/// R[t] (row stride ldr floats; its first n) += add[t] (add: null adds nothing), then x[t] = rms(R[t]) * w and x16 its
/// FP16 copy (the plain residual's read, as rms_q8 is the token path's).
void add_rms_rows(float* R, int ldr, const float* add, const float* w, float eps, int T, int n, float* x, void* x16,
                  cudaStream_t s);
/// The causal conv over T rows of x ([T][C]) with the carried state ([C][d_conv - 1], the oldest first):
/// out[t][c] = silu(sum_i kW[c][i] * in(t - (d_conv - 1) + i, c)), then the state holds the last d_conv - 1 rows.
void conv_rows(const float* x, int T, int C, const float* kW, int d_conv, float* conv_state, float* out,
               cudaStream_t s);
/// In place on the conv rows h ([T][C]: a row's q heads, then its k heads, then v): every q and k head L2-normed over
/// S, q scaled by 1 / sqrt(S).  S <= 128.
void gdn_prep_rows(float* h, int T, int C, int h_k, int S, float eps, cudaStream_t s);
/// gdn_gates over T rows: alpha[t][h] -> the gate softplus(alpha + dt_bias[h]) * ssm_a[h], beta[t][h] -> sigmoid.
void gdn_gates_rows(float* alpha, float* beta, const float* dt_bias, const float* ssm_a, int T, int h_v, cudaStream_t s);
/// gdn_prep_rows and gdn_gates_rows in one launch (the gates' blocks beside the heads').
void gdn_prep_gates(float* h, int T, int C, int h_k, int S, float eps, float* alpha, float* beta, const float* dt_bias,
                    const float* ssm_a, int h_v, cudaStream_t s);
/// conv_rows' state update alone: the state after T rows of x (the rewind's replay over the kept rows).
void conv_state_rows(const float* x, int T, int C, int d_conv, float* conv_state, cudaStream_t s);
/// The token path's conv step (gdn_conv_step: one input row x, the state slid) with the SiLU in the same launch.
void conv_silu_step(float* conv_state, const float* x, const float* kW, int C, int d_conv, float* out, cudaStream_t s);
/// The recurrence over T rows in order (gdn_step's math, the state (S, h_v, S) updated in place): q / k / v from the
/// conv rows (stride C), gate / beta [T][h_v]; o [T][h_v * S].  S == 128.
void gdn_scan(float* state, const float* h, int T, int C, const float* gate, const float* beta, int S, int h_k, int h_v,
              float* o, cudaStream_t s);
/// gdn_scan with the launches around it in one, the same bits (gdn_core_parity): prep - h holds the conv's raw rows
/// and gate / beta the raw alpha / beta, the heads' L2 norm and the gates computed in the kernel (gdn_prep_gates', not
/// written back); without prep they are gdn_prep_gates' outputs.  Then the output norm with the sigmoid gate
/// (gdn_out_norm's, z / ssm_norm / out_eps) into y [T][h_v][S] and y's q8_1 (yq, may be null).  S 128.
void gdn_core(float* state, const float* h, int T, int C, const float* gate, const float* beta, int S, int h_k, int h_v,
              bool prep, float eps, const float* dt_bias, const float* ssm_a, const float* z, const float* ssm_norm,
              float out_eps, float* y, void* yq, cudaStream_t s);
/// The token path's delta-net middle after the qkv / z projection in one launch (CUDA sm_90+, a cluster of h_v / h_k
/// blocks a k head): alpha / beta (F32 or BF16 (ab_bf16) rows w_alpha / w_beta [h_v][n_in] against x: mv's sums), conv_silu_step, then
/// gdn_core with prep (T 1) - the same bits as those launches (gdn_core_parity).  h_out (the conv rows, may be null)
/// and ab_out (raw alpha then beta, may be null) as the old launches leave them.  pf / pf_bytes (may be null): the next
/// GEMV's weights, pulled toward L2 while the recurrence runs.  False (nothing launched): not supported here.
bool gdn_mid_ok(int S, int h_k, int h_v, int d_conv, int n_in);
bool gdn_mid(float* state, float* conv_state, const float* qkv, const float* conv_w, int d_conv, const float* x,
             const void* w_alpha, const void* w_beta, bool ab_bf16, int n_in, int S, int h_k, int h_v, float eps,
             const float* dt_bias, const float* ssm_a, const float* z, const float* ssm_norm, float out_eps, float* h_out,
             float* ab_out, float* y, void* yq, const void* pf, size_t pf_bytes, cudaStream_t s);
/// attn_prep over T rows at positions p0 + t: q_full [T][n_head][2 * head_dim], k / v [T][n_head_kv][head_dim] ->
/// q [T][n_head][head_dim] and the caches' rows p0 .. p0 + T - 1.
void attn_prep_rows(const float* q_full, const float* k, const float* v, const float* q_norm, const float* k_norm,
                    float eps, int n_head, int n_head_kv, int head_dim, int n_rot, float rope_base, int64_t p0, int T,
                    uint16_t* kc, uint16_t* vc, float* q, cudaStream_t s, int kv = 0);
/// The causal attention of T queries (row t at position p0 + t, over the cache's positions <= p0 + t):
/// out [T][n_head][head_dim].  A block takes 4 queries and one KV head's group of n_head / n_head_kv heads (8, 16 or
/// 32 of them), a warp 1, 2 or 4 heads, a lane 8 dims; the keys stream past once a block.  head_dim 256.
void attn_prompt(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_head_kv, int head_dim,
                 int64_t p0, int T, float scale, float* out, cudaStream_t s, int kv = 0);
/// The scratch (floats) attn_window needs for max_rows rows of n_head heads over a cache of max_ctx positions.
size_t attn_window_floats(int n_head, int head_dim, int max_ctx, int max_rows);
/// attn_prompt for a verify window's few rows (row t at position p0 + t): the keys split into chunks of 128 positions,
/// a block each per 4 rows and KV group, the partials into part, then merged - the same sums as attn_prompt up to
/// their order.  head_dim 256.
void attn_window(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_head_kv, int head_dim,
                 int64_t p0, int T, float scale, float* part, float* out, cudaStream_t s, int kv = 0);
/// attn_gate over T rows.
void attn_gate_rows(float* o, const float* q_full, int T, int n_head, int head_dim, cudaStream_t s, int kv = 0);
/// attn_prep_rows' K / V half alone: k / v rows [T][n_head_kv][head_dim] (k_norm, RoPE) into the caches' rows p0 ..
/// p0 + T - 1 (the drafter's cache rows, whose queries are never needed).
void attn_kv_rows(const float* k, const float* v, const float* k_norm, float eps, int n_head_kv, int head_dim, int n_rot,
                  float rope_base, int64_t p0, int T, uint16_t* kc, uint16_t* vc, cudaStream_t s, int kv = 0);
/// out[0] = softmax(logits)[tok[0]] over n logits (a draft's confidence).
void top_prob(const float* logits, int n, const int* tok, float* out, cudaStream_t s);
/// ffn[t] *= sigmoid(x[t] . w) over T rows of n (the shared expert's gate from its input rows).
void shexp_gate_rows(const float* x, const float* w, float* ffn, int T, int n, cudaStream_t s);

/// qwen4exp's hyper-connections (llama.cpp src/models/qwen4exp.cpp), T rows, the residual R [T][hc][n_embd]:
/// R[t][c] += out[t] * 2 sigmoid(inject[t * inj_stride + c] / hc) when out != nullptr (build_hc_combine, comb the
/// identity), then xn[t][c] = rms(R[t][c]) * gamma[c] (build_hc_mix's grouped norm; gamma [hc][n_embd]).
void hc4_combine_norm(float* R, const float* out, const float* inject, int inj_stride, const float* gamma, float eps,
                      int n_embd, int hc, int T, float* xn, cudaStream_t s, void* xq = nullptr);   // xq: xn's q8_1 too
/// qwen4exp's QSA: the KV cache rows ids[0..n) (FP16 [ctx][row] of `row` halves) into the contiguous ks / vs.
void gather_rows(const uint16_t* kc, const uint16_t* vc, const int32_t* ids, int n, int row, uint16_t* ks, uint16_t* vs,
                 cudaStream_t s);
/// x[i] = silu(a * x[i]), i < n (build_hc_mix's low-rank activation, a = 1 / hc).
void silu_scale(float* x, int n, float a, cudaStream_t s, void* xq = nullptr);   // xq: x's q8_1 too
/// mixed[t][e] = mean_c xn[t][c][e] * sigmoid(gate[t][c][e]) (build_hc_mix's collapse of the streams).
void hc4_mix(const float* xn, const float* gate, int n_embd, int hc, int T, float* mixed, cudaStream_t s,
             void* xq = nullptr);   // xq: mixed's q8_1 too

/// qwen4exp's whole hyper-connection mix of one row in two launches, bit-identical to hc4_combine_norm + mv(down, inject)
/// + silu_scale + mv(up) + hc4_mix with BF16 down / inject / up (the same sums in the same order): a - every block the
/// streams' combine and norm into its own shared copy of xn (R is read, not written), then 8 rows of down / inject;
/// b - silu(lo / hc), the up rows of 32 embedding columns (all hc streams), R's combine, xn, gate, the mix and its q8_1.
/// lo: down's output before the silu; sc: hc floats of scratch (a's stream scales for b).  hc4_fused_ok: the shapes
/// and types the kernels take (hc 4, hc * n_embd <= 10240, lo <= 512 and a multiple of 8, BF16 down / up with BF16 or
/// F32 inject, or Q8_0 down / up - lo a multiple of 32 - with F32 inject: mv()'s sums, xn and lo as q8_1).
bool hc4_fused_ok(int n_embd, int hc, int lr, int down_type, int up_type, int inject_type);
void hc4_fused(float* R, const float* out, const float* inj_prev, const float* gamma, float eps, int n_embd, int hc,
               int lr, const void* down, const void* inject, const void* up, float* lo, float* inj_out, float* sc,
               float* xn, float* gate, float* x, void* xq, cudaStream_t s, bool inject_f32 = false,
               int wtype = 30, void* scratch = nullptr);
/// the same over a window's T columns (1..8; R [T][hc][n_embd], out [T][n_embd], inj_prev / inj_out [T][inj_stride /
/// hc], lo [T][lr], sc 4 T floats, xn / gate [T][hc][n_embd], x [T][n_embd]): bit-identical to q4x_mix_cols' five
/// launches (hc4_combine_norm over the columns, mvn down + inject, silu_scale, mvn up, hc4_mix) - a: hc4_fused's a with
/// a grid row a column, b: every column's up rows from one load of the weights (hc4_fused is this with T = 1)
bool hc4_fused_cols_ok(int n_embd, int hc, int lr, int T, int down_type, int up_type, int inject_type);
void hc4_fused_cols(float* R, const float* out, const float* inj_prev, int inj_stride, const float* gamma, float eps,
                    int n_embd, int hc, int lr, int T, const void* down, const void* inject, const void* up, float* lo,
                    float* inj_out, float* sc, float* xn, float* gate, float* x, void* xq, cudaStream_t s,
                    bool inject_f32 = false, int wtype = 30, void* scratch = nullptr);
/// (wtype: down's and up's type, BF16 30 or Q8_0 8; Q8_0 needs scratch of 4 T lr floats - a's down parts, the five's
/// xn q8_1 buffer is big enough)

}  // namespace strata::kernels::q35
