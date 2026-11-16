// src/prefill/moe_mmq.cu - see include/strata/prefill/moe_mmq.hpp.  llama.cpp's MMQ (ggml-cuda, MIT) is compiled
// from the pinned llama.cpp checkout the build already takes ggml from; src/prefill/ggml_cuda_host.cu supplies the
// few host symbols of ggml-cuda.cu it references.  The helpers that need none of it (the gathers, the SwiGLU, iota) are
// in moe_mmq_util.cu; a build without these sources (STRATA_MMQ=OFF) takes moe_mmq_none.cu instead of this file.
#include "strata/prefill/moe_mmq.hpp"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::mmq {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill mmq: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

}  // namespace

bool built() { return true; }

bool supported(int t) {
    switch ((ggml_type) t) {
        case GGML_TYPE_Q2_0: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q8_0:   // the draft layer's dense matrices (E-9)
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1:   // unsloth's Qwen3.8-27B Q4_0 / Q4_1 (every dense matrix)
        case GGML_TYPE_Q5_0:   // unsloth UD 3.1 Qwen3.8 down experts (UD-IQ4_XS: 33 layers, UD-Q3_K_XL: 20)
        case GGML_TYPE_IQ1_S:  // GLM-5.3-Flash's IQ1_S experts
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_MXFP4:  // deepseek4's FP4 expert rows
        case GGML_TYPE_NVFP4:  // NVIDIA's FP4 checkpoints (the per-expert scale_2 is applied by the caller)
            return true;   // (the Maya quants' candidates, docs/maya/QUANT-PLAN.md; IQ1_M has no MMQ)
        default:
            return false;
    }
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / ggml_blck_size((ggml_type) t)) * ggml_type_size((ggml_type) t);
}

size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) pad512(cols) * sizeof(block_q8_1_mmq) / (4 * QK8_1) + 128 * sizeof(block_q8_1_mmq);
}

size_t q8_block_bytes() { return sizeof(block_q8_1_mmq); }

void quantize(const float* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows, void* stream,
              float* scale) {
    if (rows <= 0) return;
    // MXFP4 on Blackwell: the MMQ kernel multiplies natively in FP4 and reads its activations in llama.cpp's
    // block_fp4_mmq form (ggml_cuda_mul_mat_q's use_native_fp4) - smaller than the q8_1 form q8_bytes sizes for
    // (NVFP4 the same, with its activations' per-row scale in `scale`)
    if ((ggml_type) t == GGML_TYPE_MXFP4 || (ggml_type) t == GGML_TYPE_NVFP4) {
        int dev = 0;
        cudaGetDevice(&dev);
        if (blackwell_mma_available(ggml_cuda_info().devices[dev].cc)) {
            if ((ggml_type) t == GGML_TYPE_NVFP4 && scale == nullptr) {
                std::fprintf(stderr, "prefill mmq: NVFP4 activations need a scale buffer\n");
                std::exit(1);
            }
            quantize_mmq_fp4_cuda(x, ids, xq, (ggml_type) t == GGML_TYPE_NVFP4 ? scale : nullptr, (ggml_type) t, false,
                                  cols, ld, rows * ld, rows * ld,
                                  pad512(cols), rows, 1, 1, (cudaStream_t) stream);
            ck(cudaGetLastError(), "quantize fp4");
            return;
        }
    }
    quantize_mmq_q8_1_cuda(x, ids, xq, (ggml_type) t, cols, ld, rows * ld, rows * ld, pad512(cols), rows, 1, 1,
                           (cudaStream_t) stream);
    ck(cudaGetLastError(), "quantize");
}

Context::Context() {
    int dev = 0;
    cudaGetDevice(&dev);
    ctx_ = new ggml_backend_cuda_context(dev);
}
Context::~Context() { delete (ggml_backend_cuda_context*) ctx_; }

namespace {

void dispatch(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s) {
    switch (a.type_x) {
        case GGML_TYPE_Q2_0: mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XS: mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_S: mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, a, s); break;
        case GGML_TYPE_IQ3_XXS: mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ3_S: mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, a, s); break;
        case GGML_TYPE_IQ4_NL: mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, a, s); break;
        case GGML_TYPE_IQ4_XS: mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, a, s); break;
        case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, a, s); break;
        case GGML_TYPE_Q4_0: mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, a, s); break;
        case GGML_TYPE_Q4_1: mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, a, s); break;
        case GGML_TYPE_Q5_0: mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, a, s); break;
        case GGML_TYPE_IQ1_S: mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, a, s); break;
        case GGML_TYPE_Q2_K: mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, a, s); break;
        case GGML_TYPE_Q3_K: mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, a, s); break;
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_K: mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, a, s); break;
        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, a, s); break;
        case GGML_TYPE_MXFP4: mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, a, s); break;
        case GGML_TYPE_NVFP4: mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, a, s); break;
        default:
            std::fprintf(stderr, "prefill mmq: type %d is not covered\n", (int) a.type_x);
            std::exit(1);
    }
    ck(cudaGetLastError(), "mul_mat_q");
}

}  // namespace

void Context::run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    const ggml_type t = (ggml_type) p.type;
    const int64_t qk = ggml_blck_size(t), bpr = p.w_cols / qk;
    const mmq_args a = {(const char*) p.w, t, (const int*) p.xq, p.ids, p.bounds, p.dst, p.y_scale,
                        p.w_cols, p.w_rows, p.total_rows, bpr, p.total_rows, p.ld_dst,
                        p.n, p.n, (int64_t) (p.expert_bytes / ggml_type_size(t)), 0, 0,
                        1, 1, 0, 0, 0,
                        p.max_rows, p.max_rows};
    dispatch(*(ggml_backend_cuda_context*) ctx_, a, (cudaStream_t) stream);
}

// STRATA_MMQ_GLU: the build compiles MMQ from a copy of ggml-cuda with third_party/ggml/mmq-fused-swiglu.patch
bool glu_supported(int t, int64_t n_ff) {
#if defined(STRATA_MMQ_GLU)
    if (!ggml_cuda_mmq_fusion_supported((ggml_type) t) || n_ff <= 0 || n_ff % 128 != 0) return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    return turing_mma_available(ggml_cuda_info().devices[dev].cc);
#else
    (void) t;
    (void) n_ff;
    return false;
#endif
}

void Context::run_glu(const Product& p, float limit, void* stream) {
#if defined(STRATA_MMQ_GLU)
    if (p.n <= 0 || p.max_rows <= 0) return;
    const ggml_type t = (ggml_type) p.type;
    const int64_t qk = ggml_blck_size(t), bpr = p.w_cols / qk, n_ff = p.w_rows / 2;
    // x the up rows, x_gate the gate rows of each expert's matrix; the output has n_ff rows
    mmq_args a = {(const char*) p.w + (size_t) n_ff * (size_t) bpr * ggml_type_size(t), t, (const int*) p.xq, p.ids,
                  p.bounds, p.dst, nullptr,
                  p.w_cols, n_ff, p.total_rows, bpr, p.total_rows, p.ld_dst,
                  p.n, p.n, (int64_t) (p.expert_bytes / ggml_type_size(t)), 0, 0,
                  1, 1, 0, 0, 0,
                  p.max_rows, p.max_rows};
    a.x_gate = (const char*) p.w;
    a.glu_limit = limit;
    dispatch(*(ggml_backend_cuda_context*) ctx_, a, (cudaStream_t) stream);
#else
    (void) p;
    (void) limit;
    (void) stream;
    std::fprintf(stderr, "prefill mmq: this build has no fused gate/up product\n");
    std::exit(1);
#endif
}

}  // namespace strata::prefill::mmq
