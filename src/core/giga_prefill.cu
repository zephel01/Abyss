// src/core/giga_prefill.cu - ai-sage GigaChat4 (GGUF arch gigachat4): the batched prompt path's half that is its own
// (Glm5Model members; prefill_half in glm_prefill.cu calls them, the FFNs - layer 0's dense one, the MoE - are the
// shared qwen35moe prompt code).  giga_fast_path.cu's layers over a chunk's rows (row t = position p0 + t):
//   the norms    every GatedNorm a row a block (the token path's kernel with row strides): the last sublayer's post
//                norm in place, R += it, x = the next pre norm - x16 its FP16 (the projections' input)
//   GDN layer    the projections as FP16 GEMMs; the conv, the gates and the recurrence qwen35's prompt kernels (state
//                and conv rows carried across the sub-batches), the value heads' key pairing and the closing norm the
//                token path's kernels over rows; ssm_out
//   MLA layer    q_a / kv_a / the gate / the indexer's key as GEMMs, the two latent norms (x alpha), q_b; every row's
//                cache row and indexer key written first, then the causal attention on tensor cores (raon's prompt
//                kernel with the indexer's choice), q_abs / v_b as batched products, the per-channel gate, attn_output
//   DSA          a row at a position past idx_top_k: its query (indexer.attn_q_b, roped), the head weights, the scores
//                of every position up to its own on tensor cores, the top idx_top_k (the token path's select, a block
//                a row) - the attention kernel reads those cache rows.  Rows up to idx_top_k attend every position
// What differs from the token path is the rounding: FP16 activations into the projections (q8_1 there), FP16 queries
// in the attention and the indexer's scores, the sums' order.  The state left behind is the token path's: the
// delta-net state / conv rows, the MLA cache rows, the indexer keys of every position.
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/giga_fast.hpp"
#include "strata/kernels/glm_batch.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/ling_fast.hpp"
#include "strata/kernels/qwen35_fast.hpp"
#include "strata/kernels/raon_fast.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::core {

namespace gb = strata::kernels::glmb;
namespace gf = strata::kernels::glmf;
namespace gk = strata::kernels::giga;
namespace lk = strata::kernels::ling;
namespace rk = strata::kernels::raon;
namespace q35 = strata::kernels::q35;

namespace {
constexpr int kNormForm = 6;   // (giga_fast_path.cu's: the gate reads rms(x) * w, SiLU, gate_w * sigmoid)
// the indexer's score rows (rows x positions floats): 64 MB, or one row of the longest context
constexpr size_t kScoreFloats = (size_t) 16 << 20;

struct Carve {
    uint8_t* base = nullptr;
    size_t off = 0;
    template <class Tp>
    Tp* take(size_t n) {
        Tp* p = base ? (Tp*) (base + off) : nullptr;
        off += (n * sizeof(Tp) + 255u) & ~(size_t) 255u;
        return p;
    }
};
// a delta-net layer over T rows: the projections, the conv rows, the value heads' q / k / v, the recurrence's output,
// the closing norm (and its FP16)
struct GdnBufs {
    float *qkv, *conv, *exp, *z, *alpha, *beta, *o, *y;
    uint16_t* y16;
};
GdnBufs carve_gdn(Carve& c, size_t T, const Glm5Geometry& g) {
    GdnBufs b{};
    const size_t C = (size_t) g.ssm_conv_dim(), V = (size_t) g.ssm_inner, HV = (size_t) g.ssm_v_heads;
    b.qkv = c.take<float>(T * C);
    b.conv = c.take<float>(T * C);
    b.exp = c.take<float>(T * 3 * V);
    b.z = c.take<float>(T * V);
    b.alpha = c.take<float>(T * HV);
    b.beta = c.take<float>(T * HV);
    b.o = c.take<float>(T * V);
    b.y = c.take<float>(T * V);
    b.y16 = c.take<uint16_t>(T * V);
    return b;
}
// an MLA layer over T rows: the q latent (and FP16), the kv rows, the normed latent, the gate's logits, q, q_abs (then
// the gated heads: o), the contexts, the heads' FP16; the indexer's raw keys, head weights, queries, chosen positions
struct MlaBufs {
    float *qa, *kv, *lat, *gate, *q, *qabs, *ctx, *o, *kraw, *iw, *iq;
    uint16_t *qa16, *o16;
    int* sel;
};
MlaBufs carve_mla(Carve& c, size_t T, const Glm5Geometry& g) {
    MlaBufs b{};
    const size_t H = (size_t) g.n_head, QL = (size_t) g.q_lora;
    b.qa = c.take<float>(T * QL);
    b.qa16 = c.take<uint16_t>(T * QL);
    b.kv = c.take<float>(T * gk::kRow);
    b.lat = c.take<float>(T * gk::kLat);
    b.gate = c.take<float>(T * H * lk::kV);
    b.q = c.take<float>(T * H * gk::kQHead);
    b.qabs = c.take<float>(T * H * gk::kLat);
    b.o = b.qabs;   // (q_abs is read by the attention alone; the v_b product writes the heads there)
    b.ctx = c.take<float>(T * H * gk::kLat);
    b.o16 = c.take<uint16_t>(T * H * lk::kV);
    b.kraw = c.take<float>(T * gk::kIdxKey);
    b.iw = c.take<float>(T * (size_t) g.idx_heads);
    b.iq = c.take<float>(T * (size_t) g.idx_heads * gk::kIdxKey);
    b.sel = c.take<int>(T * (size_t) g.idx_top_k);
    return b;
}
float yarn_mscale(const Glm5Geometry& g) {
    return g.yarn_factor > 1.0f ? g.yarn_log_mul * std::log(g.yarn_factor) + 1.0f : 1.0f;
}
bool f16_ok(const WSlot& w) {
    return w.q != nullptr &&
           (w.type == gf::kTypeBF16 || w.type == gf::kTypeF32 || strata::kernels::dequant_bf16_supported(w.type));
}
}  // namespace

size_t giga_prompt_bytes(const Glm5Geometry& g, size_t T) {
    Carve a, b;
    carve_gdn(a, T, g);
    carve_mla(b, T, g);
    return std::max(a.off, b.off);
}

bool Glm5Model::giga_prompt_ok(std::string& why) {
    const FastState* F = fast_;
    const Glm5Geometry& g = g_;
    if (!gk::prompt_attn_ok() || g.n_head < 32) {
        why = "the tensor-core attention does not run on this device";
        return false;
    }
    if (giga_buf_ == nullptr) {
        why = "the token path's scratch is not allocated";
        return false;
    }
    const auto ck = [&](const WSlot& w, const char* nm, int il) {
        if (!f16_ok(w)) why += "blk." + std::to_string(il) + "." + nm + " (type " + std::to_string(w.type) + ") ";
    };
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (Ly.recr) {
            ck(Ly.qkv, "attn_qkv", il);
            ck(Ly.zg, "attn_gate", il);
            ck(Ly.alpha_w, "ssm_alpha", il);
            ck(Ly.beta_w, "ssm_beta", il);
            ck(Ly.ssm_o, "ssm_out", il);
        } else {
            ck(Ly.rn_qa, "attn_q_a", il);
            ck(Ly.rn_qb, "attn_q_b", il);
            ck(Ly.lg_kva, "attn_kv_a_mqa", il);
            ck(Ly.lg_gate, "attn_gate", il);
            ck(Ly.out, "attn_output", il);
            ck(Ly.gi_iqb, "indexer.attn_q_b", il);
            ck(Ly.gi_ik, "indexer.attn_k", il);
            ck(Ly.gi_iproj, "indexer.proj", il);
        }
    }
    return why.empty();
}

void Glm5Model::giga_prompt_norm(int il, int site, int T, float* R, int ldr, float* add, float* x, uint16_t* x16,
                                 cudaStream_t s) {
    const FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const int E = g.n_embd;
    const auto& Ly = F->L[(size_t) il];
    // y = GN(h += a) over the chunk's rows (a: null = none; in place where y is h)
    const auto gn = [&](const FastState::Layer::GatedNorm& n, float* h, int ldh, const float* a, float* y) {
        gk::gated_norm_ld(h, ldh, a, E, n.w, n.up, n.down, g.giga_gate_w, g.norm_eps, E, g.giga_rank, kNormForm, y, E, s, T);
    };
    if (site == 0) {
        if (add != nullptr) gn(F->L[(size_t) il - 1].gn[3], add, E, nullptr, add);   // the last FFN's post norm
        gn(Ly.gn[0], R, ldr, add, x);
    } else if (site == 1) {
        gn(Ly.gn[1], add, E, nullptr, add);   // the mixer's post norm
        gn(Ly.gn[2], R, ldr, add, x);
    } else {
        gn(Ly.gn[3], add, E, nullptr, add);   // (the last layer's FFN: the head rows add it)
        return;
    }
    gb::f32_to_f16(x, x16, (int64_t) T * E, s);
}

void Glm5Model::giga_prompt_head(float* R_row, const float* ffn_row, cudaStream_t s) {
    const Glm5Geometry& g = g_;
    gk::gated_norm(R_row, ffn_row, giga_out_w_, giga_out_up_, giga_out_down_, g.giga_gate_w, g.norm_eps, g.n_embd,
                   g.giga_rank, kNormForm, fast_->head_x, s);
    gf::quantize_q8_1(fast_->head_x, fast_->head_xq, g.n_embd, s);
}

bool Glm5Model::giga_prompt_mixer(int il, int64_t p0, int tn, const uint16_t* x16, float* mixer, bool skip_output,
                                  const Hy4PromptIo& io, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd;
    const cublasHandle_t blas = (cublasHandle_t) io.blas;
    const float one = 1.0f, zero = 0.0f;
    bool blas_ok = true;
    // Y[t][N] = X16[t][Kd] . deq(W)[N][Kd]^T (tensor cores, F32 sums), the weight decoded to FP16 in slices
    const auto hgemm = [&](const WSlot& w, int N, int Kd, const uint16_t* X16, int ldx, float* Y, int ldy, int Tn) {
        const int rows = (int) std::max<int64_t>(1, std::min<int64_t>(N, io.w16_elems / Kd));
        for (int r0 = 0; r0 < N; r0 += rows) {
            const int n = std::min(rows, N - r0);
            if (w.type == gf::kTypeBF16)
                gb::bf16_to_f16((const uint16_t*) w.q + (size_t) r0 * Kd, io.w16, (int64_t) n * Kd, s);
            else if (w.type == gf::kTypeF32)
                gb::f32_to_f16((const float*) w.q + (size_t) r0 * Kd, io.w16, (int64_t) n * Kd, s);
            else
                strata::kernels::dequant_f16(w.type, w.q, r0, n, Kd, io.w16, s);
            blas_ok = cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, Tn, Kd, &one, io.w16, CUDA_R_16F, Kd, X16, CUDA_R_16F,
                                   ldx, &zero, Y + r0, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS && blas_ok;
        }
    };
    const auto mark = [&](const char* nm) {
        if (io.mark != nullptr) io.mark(io.mark_ctx, nm, (void*) s);
    };
    if (giga_prompt_bytes(g, (size_t) tn) > io.uni_bytes) {
        err = "gigachat4: the prompt's sub-batch does not fit its scratch";
        return false;
    }
    Carve c{io.uni};
    if (Ly.recr) {
        const int C = g.ssm_conv_dim(), V = g.ssm_inner, Sd = g.ssm_state, HK = g.ssm_k_heads, HV = g.ssm_v_heads;
        GdnBufs B = carve_gdn(c, (size_t) tn, g);
        hgemm(Ly.qkv, C, E, x16, E, B.qkv, C, tn);
        hgemm(Ly.zg, V, E, x16, E, B.z, V, tn);
        hgemm(Ly.alpha_w, HV, E, x16, E, B.alpha, HV, tn);
        hgemm(Ly.beta_w, HV, E, x16, E, B.beta, HV, tn);
        mark("giga_gdn_proj");
        q35::conv_rows(B.qkv, tn, C, Ly.conv1d, g.d_conv, Ly.gdn_conv, B.conv, s);
        q35::gdn_prep_gates(B.conv, tn, C, HK, Sd, g.norm_eps, B.alpha, B.beta, Ly.dt_bias, Ly.ssm_a, HV, s);
        if (HK != HV) {   // (value head v on key head v / (HV / HK))
            gk::gdn_expand(B.conv, B.exp, HK, HV, Sd, s, tn);
            q35::gdn_scan(Ly.gdn_state, B.exp, tn, 3 * V, B.alpha, B.beta, Sd, HV, HV, B.o, s);
        } else {
            q35::gdn_scan(Ly.gdn_state, B.conv, tn, C, B.alpha, B.beta, Sd, HK, HV, B.o, s);
        }
        gk::gdn_out(B.o, B.z, Ly.ssm_norm, B.y, HV, Sd, g.norm_eps, g.giga_gdn_gate, false, s, tn);
        mark("giga_gdn");
        if (!skip_output) {
            gb::f32_to_f16(B.y, B.y16, (int64_t) tn * V, s);
            hgemm(Ly.ssm_o, E, V, B.y16, V, mixer, E, tn);
            mark("giga_gdn_out");
        }
    } else {
        const int H = g.n_head, QL = g.q_lora, QD = H * gk::kQHead, HV = H * lk::kV, IH = g.idx_heads, K = g.idx_top_k;
        const int IQ = IH * gk::kIdxKey;
        MlaBufs B = carve_mla(c, (size_t) tn, g);
        int mla = 0;   // (the layer's place among the MLA layers: its indexer keys)
        for (int j = l0_; j < il; ++j) mla += !F->L[(size_t) j].recr;
        uint16_t* keys = giga_keys(mla);
        const double* freq = giga_freq();
        // rows [r_sel, tn) sit past idx_top_k positions: the indexer picks what they attend
        const int r_sel = (int) std::clamp<int64_t>((int64_t) K - p0, 0, (int64_t) tn), n_sel = tn - r_sel;
        hgemm(Ly.rn_qa, QL, E, x16, E, B.qa, QL, tn);
        hgemm(Ly.lg_kva, gk::kRow, E, x16, E, B.kv, gk::kRow, tn);
        hgemm(Ly.lg_gate, HV, E, x16, E, B.gate, HV, tn);
        hgemm(Ly.gi_ik, gk::kIdxKey, E, x16, E, B.kraw, gk::kIdxKey, tn);
        if (n_sel > 0) hgemm(Ly.gi_iproj, IH, E, x16 + (size_t) r_sel * E, E, B.iw, IH, n_sel);
        // the q latent: GN x alpha_q, then q_b (and the indexer's query)
        gk::gated_norm_ld(B.qa, QL, nullptr, 0, Ly.gn_qa.w, Ly.gn_qa.up, Ly.gn_qa.down, g.giga_gate_w, g.norm_eps, QL,
                          g.giga_rank, kNormForm, B.qa, QL, s, tn);
        gk::scale(B.qa, std::sqrt((float) E / (float) QL), tn * QL, s);
        gb::f32_to_f16(B.qa, B.qa16, (int64_t) tn * QL, s);
        hgemm(Ly.rn_qb, QD, QL, B.qa16, QL, B.q, QD, tn);
        if (n_sel > 0) hgemm(Ly.gi_iqb, IQ, QL, B.qa16 + (size_t) r_sel * QL, QL, B.iq, IQ, n_sel);
        // the kv latent: GN over the rows' first 512 x alpha_kv; the cache rows and q's rope; the indexer's keys
        gk::gated_norm_ld(B.kv, gk::kRow, nullptr, 0, Ly.gn_kva.w, Ly.gn_kva.up, Ly.gn_kva.down, g.giga_gate_w, g.norm_eps,
                          gk::kLat, g.giga_rank, kNormForm, B.lat, gk::kLat, s, tn);
        gk::scale(B.lat, std::sqrt((float) E / (float) gk::kLat), tn * gk::kLat, s);
        gk::mla_prep_rows(B.q, QD, H, B.lat, B.kv, freq, p0, tn, Ly.kc + (size_t) p0 * gk::kRow, s);
        gk::idx_keys_rows(B.kraw, Ly.gi_ikn, g.norm_eps, freq, p0, tn, keys + (size_t) p0 * gk::kIdxKey, s);
        mark("giga_mla_proj");
        if (n_sel > 0) {
            if (giga_psc_ == nullptr) {
                const size_t want = std::max<size_t>((size_t) std::max<int64_t>(max_ctx_, 1), kScoreFloats);
                void* d = nullptr;
                if (cudaMalloc(&d, want * sizeof(float)) != cudaSuccess) {
                    cudaGetLastError();
                    err = "gigachat4: the prompt's indexer scores (" + std::to_string(want * 4 >> 20) + " MB) did not allocate";
                    return false;
                }
                giga_owned_.push_back(d);
                giga_psc_ = (float*) d;
                giga_psc_floats_ = want;
            }
            gk::idx_q_rope_rows(B.iq, IH, freq, p0 + r_sel, n_sel, s);
            const int64_t ld = p0 + tn;   // (the last row's positions)
            const int grp = (int) std::max<int64_t>(1, std::min<int64_t>(n_sel, (int64_t) giga_psc_floats_ / ld));
            for (int a = 0; a < n_sel; a += grp) {
                const int gn = std::min(grp, n_sel - a);
                const int64_t pa = p0 + r_sel + a;
                gk::idx_score_rows(B.iq + (size_t) a * IQ, B.iw + (size_t) a * IH, keys, pa, gn, giga_psc_, ld, s);
                gk::idx_topk_rows(giga_psc_, ld, pa, gn, K, B.sel + (size_t) a * K, s);
            }
            mark("giga_dsa");
        }
        // q_abs = k_b . q_nope per head (one batched product), the attention, o = v_b . ctx per head, the gate
        blas_ok = cublasSgemmStridedBatched(blas, CUBLAS_OP_T, CUBLAS_OP_N, gk::kLat, tn, gk::kNope, &one, Ly.lg_kb,
                                            gk::kNope, (long long) gk::kLat * gk::kNope, B.q, QD, gk::kQHead, &zero,
                                            B.qabs, H * gk::kLat, gk::kLat, H) == CUBLAS_STATUS_SUCCESS && blas_ok;
        const float ms = yarn_mscale(g);
        gk::mla_prompt_attn(B.qabs, B.ctx, B.q, QD, Ly.kc, p0, tn, H, ms * ms / std::sqrt((float) gk::kQHead),
                            n_sel > 0 ? B.sel : nullptr, K, r_sel, s);
        blas_ok = cublasSgemmStridedBatched(blas, CUBLAS_OP_T, CUBLAS_OP_N, lk::kV, tn, gk::kLat, &one, Ly.lg_vb, gk::kLat,
                                            (long long) lk::kV * gk::kLat, B.ctx, H * gk::kLat, gk::kLat, &zero, B.o, HV,
                                            lk::kV, H) == CUBLAS_STATUS_SUCCESS && blas_ok;
        rk::gate_rows(B.o, B.gate, (int64_t) tn * HV, s);
        mark("giga_mla_attend");
        if (!skip_output) {
            gb::f32_to_f16(B.o, B.o16, (int64_t) tn * HV, s);
            hgemm(Ly.out, E, HV, B.o16, HV, mixer, E, tn);
            mark("giga_mla_out");
        }
    }
    if (!blas_ok) {
        err = "gigachat4: a prompt GEMM failed";
        return false;
    }
    if (gk::launch_errors() != 0 || rk::launch_errors() != 0) {
        err = "gigachat4: a prompt kernel launch failed";
        return false;
    }
    return true;
}

}  // namespace strata::core
