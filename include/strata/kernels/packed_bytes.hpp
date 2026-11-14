// include/strata/kernels/packed_bytes.hpp - byte / half-word k of a packed register value, by shift and mask.
//
// The kernels used to read the bytes of a packed local (`int q2 = ...; const uint8_t* aux8 = (const uint8_t*) &q2;
// ... aux8[l]`).  NVCC has miscompiled that form: Windows CUDA 13.2 read wrong codebook rows in the MMQ IQ1_S
// loader.  These helpers give the same bytes on a little-endian GPU (byte k = bits 8k..8k+7, word 0 first) without
// taking the address of a register; with a constant k (unrolled loops) they fold to one shift + mask.
#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)

__device__ __forceinline__ int pk_u8(unsigned v, int k) { return (int) ((v >> (8 * k)) & 0xFFu); }
__device__ __forceinline__ int pk_u8(int v, int k) { return pk_u8((unsigned) v, k); }
__device__ __forceinline__ int pk_u8(uint2 v, int k) { return pk_u8(k < 4 ? v.x : v.y, k & 3); }
__device__ __forceinline__ int pk_u8(int2 v, int k) { return pk_u8((unsigned) (k < 4 ? v.x : v.y), k & 3); }

__device__ __forceinline__ int pk_u16(unsigned v, int k) { return (int) ((v >> (16 * k)) & 0xFFFFu); }
__device__ __forceinline__ int pk_u16(int2 v, int k) { return pk_u16((unsigned) (k < 2 ? v.x : v.y), k & 1); }
__device__ __forceinline__ int pk_u16(uint4 v, int k) {
    const unsigned w = k < 2 ? v.x : k < 4 ? v.y : k < 6 ? v.z : v.w;
    return pk_u16(w, k & 1);
}

#endif
