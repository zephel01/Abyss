// src/prefill/moe_own_decode.cuh - the weight types of Strata's own prompt GEMM (moe_own.cu): one quantized row's
// 32 values as int8, in the pieces one thread of a 16x32 int8 tensor-core tile holds.
//
// A thread with index t (0-3) inside its group of four holds values [4t, 4t + 4) and [16 + 4t, 16 + 4t + 4) of the 32
// (the `lo` and `hi` words, value k in byte k % 4): the layout of mma.sync m16n8k32's A operand, and of two
// m16n8k16 ones.  Nothing here is tensor-core code: a WMMA backend takes the same words (a 16x16x16 tile is the lo
// words of four threads, then the hi words).
//
// Dec<TYPE>:
//   WCH     values a weight chunk holds (256: a super-block; 128 for the 32-value block types)
//   CB      bytes of a chunk in the file's row
//   AL      what a chunk's offset in the row is a multiple of
//   SUB     values one scale covers: 32 (one float scale a step) or 16 (two integer multipliers a step and one float a
//           chunk)
//   head(p)                 the chunk's own scale (p: the chunk's first byte)
//   step(p, ib, t, hd)      the 32 values ib of the chunk -> Step
// The formats are the published GGUF block layouts (third_party/ggml/ggml-common.h holds the structs and the
// codebooks); the code is Strata's.
#pragma once

#include <cstdint>

namespace strata::prefill::own {

struct Step {
    uint32_t lo, hi;   // int8 x 4 each
    float s;           // SUB 32: the values' scale
    int m0, m1;        // SUB 16: lo's and hi's integer multipliers (the float scale is the chunk's head)
};

// bits 0-3 of n -> 0xFF in bytes 0-3
__device__ __forceinline__ uint32_t nib_mask(uint32_t n) { return (((n & 15u) * 0x00204081u) & 0x01010101u) * 0xFFu; }
// the bytes of v under m negated (v's bytes 1-127: no byte's carry leaves it)
__device__ __forceinline__ uint32_t neg_nz(uint32_t v, uint32_t m) { return (v ^ m) + (m & 0x01010101u); }
// the bytes of v under m negated, v's bytes 0-127
__device__ __forceinline__ uint32_t neg_any(uint32_t v, uint32_t m) {
    const uint32_t a = v ^ m, b = m & 0x01010101u;
    return ((a & 0x7F7F7F7Fu) + b) ^ (a & 0x80808080u);
}
// the i-quants' 7 stored sign bits -> 8 (the eighth makes the count even)
__device__ __forceinline__ uint32_t signs8(uint32_t s7) { return s7 | ((uint32_t) (__popc(s7) & 1) << 7); }

__device__ __forceinline__ uint32_t rd16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ float rd_f16(const uint8_t* p) { return __half2float(*(const __half*) p); }

template<int TY> struct Dec;

// IQ2_XXS: {f16 d; 8 x [4 codebook bytes (8 values each) | u32: 4 x 7 sign bits, 4 scale bits]}
template<> struct Dec<16> {
    static constexpr int WCH = 256, CB = 66, AL = 2, SUB = 32;
    __device__ static __forceinline__ float head(const uint8_t* p) { return rd_f16(p) * 0.125f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float hd) {
        const uint8_t* q = p + 2 + 8 * ib;
        const uint32_t idx = rd16(q) | rd16(q + 2) << 16, aux = rd16(q + 4) | rd16(q + 6) << 16;
        const uint32_t* grid = (const uint32_t*) iq2xxs_grid;
        const int e = t >> 1, half = t & 1;
        Step r;
        r.lo = neg_nz(grid[2 * ((idx >> (8 * e)) & 255u) + half], nib_mask(signs8((aux >> (7 * e)) & 127u) >> (4 * half)));
        r.hi = neg_nz(grid[2 * ((idx >> (8 * (e + 2))) & 255u) + half],
                      nib_mask(signs8((aux >> (7 * (e + 2))) & 127u) >> (4 * half)));
        r.s = hd * (float) (2 * (int) (aux >> 28) + 1);
        r.m0 = r.m1 = 0;
        return r;
    }
};

// IQ2_XS: {f16 d; 32 x u16: 9 codebook bits (8 values), 7 sign bits; 8 x u8: two 4-bit scales, one a 16 values}
template<> struct Dec<17> {
    static constexpr int WCH = 256, CB = 74, AL = 2, SUB = 16;
    __device__ static __forceinline__ float head(const uint8_t* p) { return rd_f16(p) * 0.125f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float) {
        const uint8_t* q = p + 2 + 8 * ib;
        const int e = t >> 1, half = t & 1;
        const uint32_t q0 = rd16(q + 2 * e), q1 = rd16(q + 2 * (e + 2)), sc = p[2 + 64 + ib];
        const uint32_t* grid = (const uint32_t*) iq2xs_grid;
        Step r;
        r.lo = neg_nz(grid[2 * (q0 & 511u) + half], nib_mask(signs8(q0 >> 9) >> (4 * half)));
        r.hi = neg_nz(grid[2 * (q1 & 511u) + half], nib_mask(signs8(q1 >> 9) >> (4 * half)));
        r.s = 0.0f;
        r.m0 = 2 * (int) (sc & 15u) + 1;
        r.m1 = 2 * (int) (sc >> 4) + 1;
        return r;
    }
};

// IQ3_XXS: {f16 d; 64 codebook bytes (4 values each); 8 x u32: 4 x 7 sign bits, 4 scale bits}
template<> struct Dec<18> {
    static constexpr int WCH = 256, CB = 98, AL = 2, SUB = 32;
    __device__ static __forceinline__ float head(const uint8_t* p) { return rd_f16(p) * 0.25f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float hd) {
        const uint8_t* q = p + 2 + 8 * ib;
        const uint8_t* a = p + 2 + 64 + 4 * ib;
        const uint32_t aux = rd16(a) | rd16(a + 2) << 16;
        const int e = t >> 1, half = t & 1;
        Step r;
        r.lo = neg_nz(iq3xxs_grid[q[t]], nib_mask(signs8((aux >> (7 * e)) & 127u) >> (4 * half)));
        r.hi = neg_nz(iq3xxs_grid[q[4 + t]], nib_mask(signs8((aux >> (7 * (e + 2))) & 127u) >> (4 * half)));
        r.s = hd * (float) (2 * (int) (aux >> 28) + 1);
        r.m0 = r.m1 = 0;
        return r;
    }
};

// MXFP4: 32 values a block: {u8 e (E8M0 scale); 16 bytes: value i in byte i's low nibble, value 16 + i in its high
// nibble}; a nibble is sign | 3 bits into {0, 1, 2, 3, 4, 6, 8, 12} / 2.  A chunk is four blocks.
__device__ __forceinline__ uint32_t fp4_bytes(uint32_t n) {   // 4 nibbles in the low nibbles of n's bytes -> 4 x int8
    uint32_t x = n & 0x07070707u;
    x = (x | x >> 4) & 0x00FF00FFu;
    x = (x | x >> 8) & 0x0000FFFFu;   // the 3-bit codes at bits 0, 4, 8, 12
    const uint32_t mag = __byte_perm(0x03020100u, 0x0C080604u, x);
    return neg_any(mag, ((n >> 3) & 0x01010101u) * 0xFFu);
}
template<> struct Dec<39> {
    static constexpr int WCH = 128, CB = 68, AL = 1, SUB = 32;
    __device__ static __forceinline__ float head(const uint8_t*) { return 0.0f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float) {
        const uint8_t* b = p + 17 * ib;
        const uint8_t* q = b + 1 + 4 * t;
        const uint32_t v = (uint32_t) q[0] | (uint32_t) q[1] << 8 | (uint32_t) q[2] << 16 | (uint32_t) q[3] << 24;
        const uint32_t e = b[0];
        Step r;
        r.lo = fp4_bytes(v & 0x0F0F0F0Fu);
        r.hi = fp4_bytes((v >> 4) & 0x0F0F0F0Fu);
        // 2^(e - 127) / 2 (the table's values are doubled); e = 0 is 2^-127, a subnormal
        r.s = e <= 1 ? __uint_as_float(e == 0 ? 0x00200000u : 0x00400000u) : __uint_as_float((e - 1u) << 23);
        r.m0 = r.m1 = 0;
        return r;
    }
};

}  // namespace strata::prefill::own
