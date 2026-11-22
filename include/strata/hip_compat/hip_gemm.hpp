// include/strata/hip_compat/hip_gemm.hpp - the HIP build's own matrix products (src/kernels/hip/own_gemm.cu).
//
// On HIP the cuBLAS spellings the engine uses (cublasGemmEx, cublasSgemm and their strided batched forms, the handle
// calls) land here instead of in hipBLAS (hip_compat/cublas_v2.h maps them).  Every product the prompt path asks for
// is C = alpha op(A) op(B) + beta C with op(A) = A^T and op(B) = B (both operands read along k: a GGUF weight's rows
// and the tokens' rows), FP32 accumulation, FP16 / BF16 / F32 inputs, F32 or FP16 outputs: those run on the engine's
// own kernels - WMMA on RDNA3 / RDNA3.5 (gfx11) and RDNA4 (gfx12), a plain FMA tile kernel everywhere else.  Anything
// else - and everything with STRATA_HIP_OWN_GEMM=0 - goes to hipBLAS through a library handle created only when first
// needed, so a run on the own kernels never initializes rocBLAS or hipBLASLt (issue #49: an access violation inside hipblasGemmEx -> rocblas_gemm_ex -> hipblasLtMatmul).
#pragma once

#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>

#include <cstddef>

namespace strata::hipgemm {

struct Blas;
typedef Blas* handle_t;

hipblasStatus_t create(handle_t* h);
hipblasStatus_t destroy(handle_t h);
hipblasStatus_t set_stream(handle_t h, hipStream_t s);
hipblasStatus_t set_workspace(handle_t h, void* ws, size_t bytes);
hipblasStatus_t set_math_mode(handle_t h, hipblasMath_t mode);

hipblasStatus_t gemm_ex(handle_t h, hipblasOperation_t ta, hipblasOperation_t tb, int m, int n, int k,
                        const void* alpha, const void* A, hipDataType at, int lda, const void* B, hipDataType bt,
                        int ldb, const void* beta, void* C, hipDataType ct, int ldc, hipblasComputeType_t compute,
                        hipblasGemmAlgo_t algo);
hipblasStatus_t gemm_strided_batched_ex(handle_t h, hipblasOperation_t ta, hipblasOperation_t tb, int m, int n, int k,
                                        const void* alpha, const void* A, hipDataType at, int lda, long long sA,
                                        const void* B, hipDataType bt, int ldb, long long sB, const void* beta,
                                        void* C, hipDataType ct, int ldc, long long sC, int batch,
                                        hipblasComputeType_t compute, hipblasGemmAlgo_t algo);
hipblasStatus_t sgemm(handle_t h, hipblasOperation_t ta, hipblasOperation_t tb, int m, int n, int k,
                      const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta,
                      float* C, int ldc);
hipblasStatus_t sgemm_strided_batched(handle_t h, hipblasOperation_t ta, hipblasOperation_t tb, int m, int n, int k,
                                      const float* alpha, const float* A, int lda, long long sA, const float* B,
                                      int ldb, long long sB, const float* beta, float* C, int ldc, long long sC,
                                      int batch);

/// STRATA_HIP_OWN_GEMM is not 0: the products above run on the own kernels where they cover them.
bool own_on();

/// The own kernels alone, no handle: C (ldc, batch stride sC) = alpha A^T B + beta C for `batch` products.  false
/// (nothing launched) when the types or the layout are not covered.  at == bt: HIP_R_16F, HIP_R_16BF or HIP_R_32F;
/// ct: HIP_R_32F, or HIP_R_16F for 16-bit inputs.
bool own_gemm(hipStream_t s, int m, int n, int k, float alpha, const void* A, hipDataType at, int lda, long long sA,
              const void* B, hipDataType bt, int ldb, long long sB, float beta, void* C, hipDataType ct, int ldc,
              long long sC, int batch);

/// Which own kernel `own_gemm` takes for 16-bit inputs on device `dev`: "wmma11", "wmma12" or "plain".
const char* own_kernel(int dev);

}  // namespace strata::hipgemm
