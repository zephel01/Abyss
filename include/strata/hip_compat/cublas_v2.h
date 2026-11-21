#pragma once
#include <hipblas/hipblas.h>
// The handle and the products go to the engine's own GEMM (hip_gemm.hpp); hipBLAS stays behind it for what the own
// kernels do not cover and for STRATA_HIP_OWN_GEMM=0.
#include "hip_gemm.hpp"
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_DEFAULT_MATH HIPBLAS_DEFAULT_MATH
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
// hipBLAS selects the available matrix instructions with its default algorithm.
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUDA_R_16BF HIP_R_16BF
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_32F HIP_R_32F
#define cublasCreate ::strata::hipgemm::create
#define cublasDestroy ::strata::hipgemm::destroy
#define cublasGemmEx ::strata::hipgemm::gemm_ex
#define cublasGemmStridedBatchedEx ::strata::hipgemm::gemm_strided_batched_ex
#define cublasSgemm ::strata::hipgemm::sgemm
#define cublasSgemmStridedBatched ::strata::hipgemm::sgemm_strided_batched
#define cublasHandle_t ::strata::hipgemm::handle_t
#define cublasSetMathMode ::strata::hipgemm::set_math_mode
#define cublasSetStream ::strata::hipgemm::set_stream
#define cublasSetWorkspace ::strata::hipgemm::set_workspace
#define cublasStatus_t hipblasStatus_t
