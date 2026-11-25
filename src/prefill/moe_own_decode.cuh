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
//   GW, table()             the codebook (the kernel reads it from a copy in shared memory)
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
    float o;           // a Dec with SUMS: the values are s q - o (the kernel subtracts o x the activations' sum)
    float s2, o2;      // a Dec with FS16: hi's scale (s is lo's); with OFS16: hi's offset (o is lo's)
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
// The i-quants' signs: 7 stored bits s7 for 8 values (the eighth sign makes the count of minus signs even) -> 0xFF
// in the bytes of the negative ones among values 4 half .. 4 half + 3.  (A 256-entry table measured the same from
// global memory and 3% slower from shared memory: the kernel's time goes with its loads.)
__device__ __forceinline__ uint32_t sign_mask(uint32_t s7, int half) {
    const uint32_t s8 = (s7 & 127u) | ((uint32_t) (__popc(s7 & 127u) & 1) << 7);
    return nib_mask(s8 >> (4 * half));
}

__device__ __forceinline__ uint32_t rd16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ float rd_f16(const uint8_t* p) { return __half2float(*(const __half*) p); }
__device__ __forceinline__ uint32_t rd32(const uint8_t* p) { return rd16(p) | rd16(p + 2) << 16; }   // (p even)
// (A4: p is a multiple of 4 - one load.  The kernels of the 4-aligned matrices: Dec<112>, Dec<113>.)
template<bool A4> __device__ __forceinline__ uint32_t rd32x(const uint8_t* p) {
    if constexpr (A4) return *(const uint32_t*) p;
    else return rd32(p);
}

template<int TY> struct Dec;

// IQ2_XXS: {f16 d; 8 x [4 codebook bytes (8 values each) | u32: 4 x 7 sign bits, 4 scale bits]}
template<> struct Dec<16> {
    static constexpr int GW = 512;   // the codebook's 32-bit words
    __device__ static __forceinline__ const uint32_t* table() { return (const uint32_t*) iq2xxs_grid; }
    static constexpr int WCH = 256, CB = 66, AL = 2, SUB = 32;
    __device__ static __forceinline__ float head(const uint8_t* p) { return rd_f16(p) * 0.125f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float hd, const uint32_t* grid) {
        const uint8_t* q = p + 2 + 8 * ib;
        const uint32_t idx = rd16(q) | rd16(q + 2) << 16, aux = rd16(q + 4) | rd16(q + 6) << 16;
        const int e = t >> 1, half = t & 1;
        Step r;
        r.lo = neg_nz(grid[2 * ((idx >> (8 * e)) & 255u) + half], sign_mask((aux >> (7 * e)) & 127u, half));
        r.hi = neg_nz(grid[2 * ((idx >> (8 * (e + 2))) & 255u) + half],
                      sign_mask((aux >> (7 * (e + 2))) & 127u, half));
        r.s = hd * (float) (2 * (int) (aux >> 28) + 1);
        r.m0 = r.m1 = 0;
        return r;
    }
};

// IQ2_XS: {f16 d; 32 x u16: 9 codebook bits (8 values), 7 sign bits; 8 x u8: two 4-bit scales, one a 16 values}
template<> struct Dec<17> {
    static constexpr int GW = 1024;   // the codebook's 32-bit words
    __device__ static __forceinline__ const uint32_t* table() { return (const uint32_t*) iq2xs_grid; }
    static constexpr int WCH = 256, CB = 74, AL = 2, SUB = 16;
    __device__ static __forceinline__ float head(const uint8_t* p) { return rd_f16(p) * 0.125f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float, const uint32_t* grid) {
        const uint8_t* q = p + 2 + 8 * ib;
        const int e = t >> 1, half = t & 1;
        const uint32_t q0 = rd16(q + 2 * e), q1 = rd16(q + 2 * (e + 2)), sc = p[2 + 64 + ib];
        Step r;
        r.lo = neg_nz(grid[2 * (q0 & 511u) + half], sign_mask(q0 >> 9, half));
        r.hi = neg_nz(grid[2 * (q1 & 511u) + half], sign_mask(q1 >> 9, half));
        r.s = 0.0f;
        r.m0 = 2 * (int) (sc & 15u) + 1;
        r.m1 = 2 * (int) (sc >> 4) + 1;
        return r;
    }
};

// IQ2_S: {f16 d; 32 codebook bytes (8 values each, 2 more index bits in qh); 32 sign bytes; 8 x u8 qh; 8 x u8: two
// 4-bit scales, one a 16 values}
template<> struct Dec<22> {
    static constexpr int GW = 2048;   // the codebook's 32-bit words
    __device__ static __forceinline__ const uint32_t* table() { return (const uint32_t*) iq2s_grid; }
    static constexpr int WCH = 256, CB = 82, AL = 2, SUB = 16;
    __device__ static __forceinline__ float head(const uint8_t* p) { return rd_f16(p) * 0.125f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float, const uint32_t* grid) {
        const uint8_t* q = p + 2 + 4 * ib;
        const int e = t >> 1, half = t & 1;
        const uint32_t qh = p[2 + 64 + ib], sc = p[2 + 72 + ib];
        const uint32_t i0 = q[e] | ((qh << (8 - 2 * e)) & 0x300u), i1 = q[e + 2] | ((qh << (8 - 2 * (e + 2))) & 0x300u);
        Step r;
        r.lo = neg_nz(grid[2 * i0 + half], nib_mask((uint32_t) q[32 + e] >> (4 * half)));
        r.hi = neg_nz(grid[2 * i1 + half], nib_mask((uint32_t) q[32 + e + 2] >> (4 * half)));
        r.s = 0.0f;
        r.m0 = 2 * (int) (sc & 15u) + 1;
        r.m1 = 2 * (int) (sc >> 4) + 1;
        return r;
    }
};

// IQ3_XXS: {f16 d; 64 codebook bytes (4 values each); 8 x u32: 4 x 7 sign bits, 4 scale bits}
template<> struct Dec<18> {
    static constexpr int GW = 256;   // the codebook's 32-bit words
    __device__ static __forceinline__ const uint32_t* table() { return iq3xxs_grid; }
    static constexpr int WCH = 256, CB = 98, AL = 2, SUB = 32;
    __device__ static __forceinline__ float head(const uint8_t* p) { return rd_f16(p) * 0.25f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float hd, const uint32_t* grid) {
        const uint8_t* q = p + 2 + 8 * ib;
        const uint8_t* a = p + 2 + 64 + 4 * ib;
        const uint32_t aux = rd16(a) | rd16(a + 2) << 16;
        const int e = t >> 1, half = t & 1;
        Step r;
        r.lo = neg_nz(grid[q[t]], sign_mask((aux >> (7 * e)) & 127u, half));
        r.hi = neg_nz(grid[q[4 + t]], sign_mask((aux >> (7 * (e + 2))) & 127u, half));
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
    static constexpr int GW = 0;   // the codebook's 32-bit words
    __device__ static __forceinline__ const uint32_t* table() { return nullptr; }
    static constexpr int WCH = 128, CB = 68, AL = 1, SUB = 32;
    __device__ static __forceinline__ float head(const uint8_t*) { return 0.0f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float, const uint32_t* grid) {
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

// bytes of a + b, a byte at a time (no carry between bytes)
__device__ __forceinline__ uint32_t add8(uint32_t a, uint32_t b) {
    return ((a & 0x7F7F7F7Fu) + (b & 0x7F7F7F7Fu)) ^ ((a ^ b) & 0x80808080u);
}

// IQ1_S: {f16 d; 32 codebook bytes (8 values each); 8 x u16: 4 x 3 more index bits, 3 scale bits, the shift's sign}.
// A value is d (2 s + 1) (g +- 1/8), g in {-1, 0, 1}: as int8 8 g +- 1 under d (2 s + 1) / 8.  The codebook
// (iq1s_grid_gpu) holds g + 1 a nibble: values 0-3 in the bytes' low nibbles, 4-7 in the high ones.
__device__ __forceinline__ uint32_t iq1_word(uint32_t entry, int half, bool minus) {
    const uint32_t q = (half ? entry >> 4 : entry) & 0x0F0F0F0Fu;      // g + 1: 0, 1, 2
    return add8(q << 3, minus ? 0xF7F7F7F7u : 0xF9F9F9F9u);           // 8 (g + 1) - 8 -+ 1
}
template<> struct Dec<19> {
    static constexpr int GW = 2048;
    __device__ static __forceinline__ const uint32_t* table() { return iq1s_grid_gpu; }
    static constexpr int WCH = 256, CB = 50, AL = 2, SUB = 32;
    __device__ static __forceinline__ float head(const uint8_t* p) { return rd_f16(p) * 0.125f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float hd, const uint32_t* grid) {
        const uint8_t* q = p + 2 + 4 * ib;
        const uint32_t qh = rd16(p + 2 + 32 + 2 * ib);
        const int e = t >> 1, half = t & 1;
        const bool minus = (qh & 0x8000u) != 0;
        Step r;
        r.lo = iq1_word(grid[q[e] | (((qh >> (3 * e)) & 7u) << 8)], half, minus);
        r.hi = iq1_word(grid[q[e + 2] | (((qh >> (3 * (e + 2))) & 7u) << 8)], half, minus);
        r.s = hd * (float) (2 * (int) ((qh >> 12) & 7u) + 1);
        r.m0 = r.m1 = 0;
        return r;
    }
};

// IQ1_M: {32 codebook bytes; 16 x u8: two nibbles (3 more index bits, the shift's sign), one a 8 values; 4 x u16: four
// 3-bit scales (one a 16 values) and in the top nibbles the block's f16 scale}.  Values as IQ1_S.
template<> struct Dec<29> {
    static constexpr int GW = 2048;
    __device__ static __forceinline__ const uint32_t* table() { return iq1s_grid_gpu; }
    static constexpr int WCH = 256, CB = 56, AL = 2, SUB = 16;
    __device__ static __forceinline__ float head(const uint8_t* p) {
        const uint32_t s0 = rd16(p + 48), s1 = rd16(p + 50), s2 = rd16(p + 52), s3 = rd16(p + 54);
        const uint16_t u = (uint16_t) ((s0 >> 12) | ((s1 >> 8) & 0x00F0u) | ((s2 >> 4) & 0x0F00u) | (s3 & 0xF000u));
        return __half2float(__ushort_as_half(u)) * 0.125f;
    }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float, const uint32_t* grid) {
        const uint8_t* q = p + 4 * ib;
        const int e = t >> 1, half = t & 1;
        // (group l of 8 values: qh[2 ib + l / 2]'s nibble l % 2; lo is group e, hi group e + 2)
        const uint32_t h0 = (uint32_t) p[32 + 2 * ib] >> (4 * e), h1 = (uint32_t) p[32 + 2 * ib + 1] >> (4 * e);
        const uint32_t sc = rd16(p + 48 + 2 * (ib >> 1)) >> (6 * (ib & 1));
        Step r;
        r.lo = iq1_word(grid[q[e] | ((h0 & 7u) << 8)], half, (h0 & 8u) != 0);
        r.hi = iq1_word(grid[q[e + 2] | ((h1 & 7u) << 8)], half, (h1 & 8u) != 0);
        r.s = 0.0f;
        r.m0 = 2 * (int) (sc & 7u) + 1;
        r.m1 = 2 * (int) ((sc >> 3) & 7u) + 1;
        return r;
    }
};

// 4 nibbles in the low nibbles of n's bytes -> the 4 bytes table[nibble] (the table as four 32-bit words)
__device__ __forceinline__ uint32_t lut16(uint32_t n, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
    uint32_t x = n & 0x0F0F0F0Fu;
    x = (x | x >> 4) & 0x00FF00FFu;
    x = (x | x >> 8) & 0x0000FFFFu;                       // the nibbles at bits 0, 4, 8, 12
    const uint32_t a = __byte_perm(t0, t1, x), b = __byte_perm(t2, t3, x);   // (a selector's low 3 bits)
    return __byte_perm(a, b, 0x3210u | ((x & 0x8888u) >> 1));              // nibble bit 3: byte i of a, or of b
}
// the i-quants' 4-bit values (kvalues_iq4nl)
__device__ __forceinline__ uint32_t iq4_bytes(uint32_t n) { return lut16(n, 0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u); }

// IQ4_NL: 32 values a block: {f16 d; 16 bytes: value i in byte i's low nibble, value 16 + i in its high nibble}.  A
// chunk is four blocks.
template<> struct Dec<20> {
    static constexpr int GW = 0;
    __device__ static __forceinline__ const uint32_t* table() { return nullptr; }
    static constexpr int WCH = 128, CB = 72, AL = 2, SUB = 32;
    __device__ static __forceinline__ float head(const uint8_t*) { return 1.0f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float, const uint32_t*) {
        const uint8_t* b = p + 18 * ib;
        const uint32_t v = rd16(b + 2 + 4 * t) | rd16(b + 4 + 4 * t) << 16;
        Step r;
        r.lo = iq4_bytes(v);
        r.hi = iq4_bytes(v >> 4);
        r.s = rd_f16(b);
        r.m0 = r.m1 = 0;
        return r;
    }
};

// Q2_0: 64 values a block: {f16 d; 16 bytes: 4 values a byte, 2 bits each, value = code - 1}.  A chunk is two blocks.
__device__ __forceinline__ uint32_t q2_bytes(uint32_t byte) {
    const uint32_t v = (byte | byte << 6 | byte << 12 | byte << 18) & 0x03030303u;   // the four codes, a byte each
    return ((v | 0x80808080u) - 0x01010101u) ^ 0x80808080u;                           // - 1
}
template<> struct Dec<42> {
    static constexpr int GW = 0;
    __device__ static __forceinline__ const uint32_t* table() { return nullptr; }
    static constexpr int WCH = 128, CB = 36, AL = 2, SUB = 32;
    __device__ static __forceinline__ float head(const uint8_t*) { return 1.0f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float, const uint32_t*) {
        const uint8_t* b = p + 18 * (ib >> 1);
        const uint8_t* q = b + 2 + 8 * (ib & 1);
        Step r;
        r.lo = q2_bytes(q[t]);
        r.hi = q2_bytes(q[4 + t]);
        r.s = rd_f16(b);
        r.m0 = r.m1 = 0;
        return r;
    }
};

// PTQ1_0 (PrismML's Ternary Bonsai, type 143): 128 trits a block in 28 bytes - qs[24] five a byte, qh[2] four, then
// f16 d.  Trit n of byte q: ((uint8) (q 3^n) 3) >> 8 (0..2, value - 1).  qs[0..16) hold values 16 n + m, qs[16..24)
// 80 + 8 n + m, qh[h] 120 + 2 n + h.  A chunk is one block; group g of 4 values (0..31) is one 4-byte word's trit
// (g < 20: word g % 4, trit g / 4; g < 30: word 4 + (g - 20) % 2, trit (g - 20) / 2) or the qh bytes' (g 30 / 31)
__device__ __forceinline__ uint32_t ptq1_tri(uint32_t w, uint32_t p) {   // 4 bytes' trit (p = 3^n), 0..2 a byte
    uint32_t e = ((w & 0x00FF00FFu) * p) & 0x00FF00FFu, o = (((w >> 8) & 0x00FF00FFu) * p) & 0x00FF00FFu;
    e = ((e * 3u) >> 8) & 0x00030003u;
    o = ((o * 3u) >> 8) & 0x00030003u;
    return e | o << 8;
}
__device__ __forceinline__ uint32_t ptq1_grp(const uint8_t* b, int g) {
    uint32_t v;
    if (g < 20) {
        const int n = g >> 2;
        v = ptq1_tri(rd32(b + 4 * (g & 3)), n == 0 ? 1u : n == 1 ? 3u : n == 2 ? 9u : n == 3 ? 27u : 81u);
    } else if (g < 30) {
        const int n = (g - 20) >> 1;
        v = ptq1_tri(rd32(b + 16 + 4 * ((g - 20) & 1)), n == 0 ? 1u : n == 1 ? 3u : n == 2 ? 9u : n == 3 ? 27u : 81u);
    } else {
        const uint32_t q = rd16(b + 24), p = g == 30 ? 1u : 9u;
        v = (ptq1_tri(q, p) & 0xFFFFu) | (ptq1_tri(q, 3u * p) << 16);
    }
    return ((v | 0x80808080u) - 0x01010101u) ^ 0x80808080u;   // - 1
}
template<> struct Dec<143> {
    static constexpr int GW = 0;
    __device__ static __forceinline__ const uint32_t* table() { return nullptr; }
    static constexpr int WCH = 128, CB = 28, AL = 2, SUB = 32;
    __device__ static __forceinline__ float head(const uint8_t*) { return 1.0f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float, const uint32_t*) {
        Step r;
        r.lo = ptq1_grp(p, 8 * ib + t);
        r.hi = ptq1_grp(p, 8 * ib + 4 + t);
        r.s = rd_f16(p + 26);
        r.m0 = r.m1 = 0;
        return r;
    }
};

// IQ4_XS: 256 values: {f16 d; u16 scales_h; 4 x u8 scales_l; 128 bytes as IQ4_NL's, 16 a 32-value block}; block ib's
// scale is d (ls - 32), ls = scales_l's nibble ib | scales_h's 2 bits ib << 4.
// A chunk is half of it (the whole 136 bytes of 256 rows twice over do not fit a CTA's shared memory): the 8-byte
// header (HDR; both halves' chunks bring it) and the half's 64 bytes.
template<> struct Dec<23> {
    static constexpr int GW = 0;
    __device__ static __forceinline__ const uint32_t* table() { return nullptr; }
    static constexpr int WCH = 128, CB = 64, AL = 2, SUB = 32;
    static constexpr int HDR = 8, RB = 68;   // (header bytes; row bytes a chunk)
    static constexpr bool FOUR = true;
    __device__ static __forceinline__ size_t hdr(int c) { return (size_t) (c >> 1) * 136; }
    __device__ static __forceinline__ size_t dat(int c) { return (size_t) (c >> 1) * 136 + 8 + (size_t) (c & 1) * 64; }
    __device__ static __forceinline__ float head(const uint8_t* ph) { return rd_f16(ph); }
    // pd: the chunk's data; ph: its header; c: the chunk (its parity: which half)
    __device__ static __forceinline__ Step step_h(const uint8_t* pd, const uint8_t* ph, int c, int ib, int t, float hd) {
        const int g = (c & 1) * 4 + ib;   // the block in its 256
        const uint32_t sh = rd16(ph + 2), sl = ph[4 + (g >> 1)];
        const int ls = (int) (((sl >> (4 * (g & 1))) & 15u) | (((sh >> (2 * g)) & 3u) << 4));
        const uint8_t* q = pd + 16 * ib + 4 * t;
        const uint32_t v = rd16(q) | rd16(q + 2) << 16;
        Step r;
        r.lo = iq4_bytes(v);
        r.hi = iq4_bytes(v >> 4);
        r.s = hd * (float) (ls - 32);
        r.m0 = r.m1 = 0;
        return r;
    }
    // the half's four blocks at once: the header's scale bits read once
    __device__ static __forceinline__ void steps4(const uint8_t* pd, const uint8_t* ph, int c, int t, float hd, Step* out) {
        const int n = c & 1;
        const uint32_t sh = rd16(ph + 2) >> (8 * n), sl = rd16(ph + 4 + 2 * n);   // this half's 4 x 2 high bits, 4 x 4 low bits
#pragma unroll
        for (int ib = 0; ib < 4; ++ib) {
            const int ls = (int) (((sl >> (4 * ib)) & 15u) | (((sh >> (2 * ib)) & 3u) << 4));
            const uint32_t v = rd32(pd + 16 * ib + 4 * t);
            out[ib].lo = iq4_bytes(v);
            out[ib].hi = iq4_bytes(v >> 4);
            out[ib].s = hd * (float) (ls - 32);
            out[ib].m0 = out[ib].m1 = 0;
        }
    }
};

// Q4_K: 256 values: {f16 d; f16 dmin; 12 bytes: eight 6-bit scales and eight 6-bit mins; 128 bytes: a pair of 32-value
// blocks shares 32 bytes, the first block in the low nibbles, the second in the high ones}.  A value is
// d sc q - dmin m, q in 0..15: the int8 is q, and the kernel takes dmin m times the block's sum of activations off
// (SUMS).  A chunk is half of it, as IQ4_XS: the 16-byte header and the half's 64 bytes.
template<bool A4> struct DecQ4K {
    static constexpr int GW = 0;
    __device__ static __forceinline__ const uint32_t* table() { return nullptr; }
    static constexpr int WCH = 128, CB = 64, AL = 2, SUB = 32;
    static constexpr int HDR = 16, RB = 72;
    static constexpr bool FOUR = true;
    static constexpr bool SUMS = true;
    __device__ static __forceinline__ size_t hdr(int c) { return (size_t) (c >> 1) * 144; }
    __device__ static __forceinline__ size_t dat(int c) { return (size_t) (c >> 1) * 144 + 16 + (size_t) (c & 1) * 64; }
    __device__ static __forceinline__ float head(const uint8_t* ph) { return rd_f16(ph); }
    __device__ static __forceinline__ Step step_h(const uint8_t* pd, const uint8_t* ph, int c, int ib, int t, float hd) {
        const int j = (c & 1) * 4 + ib;   // the block in its 256
        const uint8_t* sc = ph + 4;
        uint32_t scale, mn;
        if (j < 4) {
            scale = sc[j] & 63u;
            mn = sc[j + 4] & 63u;
        } else {
            scale = (sc[j + 4] & 15u) | ((uint32_t) (sc[j - 4] >> 6) << 4);
            mn = (uint32_t) (sc[j + 4] >> 4) | ((uint32_t) (sc[j] >> 6) << 4);
        }
        const uint8_t* q = pd + 32 * (ib >> 1) + 4 * t;
        const int sh = 4 * (ib & 1);
        Step r;
        r.lo = ((rd16(q) | rd16(q + 2) << 16) >> sh) & 0x0F0F0F0Fu;
        r.hi = ((rd16(q + 16) | rd16(q + 18) << 16) >> sh) & 0x0F0F0F0Fu;
        r.s = hd * (float) scale;
        r.m0 = r.m1 = 0;
        r.o = rd_f16(ph + 2) * (float) mn;
        return r;
    }
    // the half's four blocks at once: a pair of blocks shares its bytes (low / high nibbles) - read once; the 12
    // scale bytes as three words
    __device__ static __forceinline__ void steps4(const uint8_t* pd, const uint8_t* ph, int c, int t, float hd, Step* out) {
        const uint32_t s0 = rd32x<A4>(ph + 4), s1 = rd32x<A4>(ph + 8), s2 = rd32x<A4>(ph + 12);   // scale bytes 0-3, 4-7, 8-11
        const float dmin = rd_f16(ph + 2);
        const bool second = (c & 1) != 0;
#pragma unroll
        for (int pr = 0; pr < 2; ++pr) {
            const uint32_t lo = rd32x<A4>(pd + 32 * pr + 4 * t), hi = rd32x<A4>(pd + 32 * pr + 16 + 4 * t);
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int ib = 2 * pr + k;   // (block j = 4 (c & 1) + ib of the 256)
                const uint32_t b0 = (s0 >> (8 * ib)) & 255u, b1 = (s1 >> (8 * ib)) & 255u, b2 = (s2 >> (8 * ib)) & 255u;
                // j < 4: scale = sc[j] & 63, min = sc[j + 4] & 63;  else: scale = sc[j + 4] & 15 | sc[j - 4] >> 6 << 4,
                // min = sc[j + 4] >> 4 | sc[j] >> 6 << 4   (sc[j - 4] = b0, sc[j] = b1, sc[j + 4] = b2 there)
                const uint32_t scale = second ? (b2 & 15u) | ((b0 >> 6) << 4) : b0 & 63u;
                const uint32_t mn = second ? (b2 >> 4) | ((b1 >> 6) << 4) : b1 & 63u;
                out[ib].lo = (lo >> (4 * k)) & 0x0F0F0F0Fu;
                out[ib].hi = (hi >> (4 * k)) & 0x0F0F0F0Fu;
                out[ib].s = hd * (float) scale;
                out[ib].m0 = out[ib].m1 = 0;
                out[ib].o = dmin * (float) mn;
            }
        }
    }
};
template<> struct Dec<12> : DecQ4K<false> {};
template<> struct Dec<112> : DecQ4K<true> {};   // (every row and chunk on a multiple of 4: the block is 144 bytes)

// Q6_K: 256 values: {128 bytes ql; 64 bytes qh; 16 x int8 scales (one a 16 values); f16 d}.  A value is
// d sc (q - 32), q = 4 bits of ql | 2 bits of qh << 4.  In each half (128 values) block ib takes the low (ib < 2) or
// high nibbles of ql[32 (ib & 1) ..] and bits 2 ib of qh[0 .. 31].
// A chunk is a half: the header (the 16 scales and d: 18 bytes), the half's 64 bytes of ql and - a second data range
// - its 32 bytes of qh.
template<> struct Dec<14> {
    static constexpr int GW = 0;
    __device__ static __forceinline__ const uint32_t* table() { return nullptr; }
    static constexpr int WCH = 128, CB = 64, AL = 2, SUB = 16;
    static constexpr int HDR = 18, RB = 105, CB2 = 32;
    static constexpr bool FOUR = true;
    __device__ static __forceinline__ size_t hdr(int c) { return (size_t) (c >> 1) * 210 + 192; }
    __device__ static __forceinline__ size_t dat(int c) { return (size_t) (c >> 1) * 210 + (size_t) (c & 1) * 64; }
    __device__ static __forceinline__ size_t dat2(int c) { return (size_t) (c >> 1) * 210 + 128 + (size_t) (c & 1) * 32; }
    __device__ static __forceinline__ float head(const uint8_t* ph) { return rd_f16(ph + 16); }
    __device__ static __forceinline__ uint32_t word(const uint8_t* ql, const uint8_t* qh, int ib) {
        const uint32_t l = rd16(ql) | rd16(ql + 2) << 16, h = rd16(qh) | rd16(qh + 2) << 16;
        const uint32_t v = ((l >> (4 * (ib >> 1))) & 0x0F0F0F0Fu) | (((h >> (2 * ib)) & 0x03030303u) << 4);   // 0..63
        return ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u;                                              // - 32
    }
    __device__ static __forceinline__ Step step_h(const uint8_t* pd, const uint8_t* ph, int c, int ib, int t, float,
                                                  const uint8_t* pd2) {
        const uint8_t* sc = ph + 8 * (c & 1) + 2 * ib;
        Step r;
        r.lo = word(pd + 32 * (ib & 1) + 4 * t, pd2 + 4 * t, ib);
        r.hi = word(pd + 32 * (ib & 1) + 16 + 4 * t, pd2 + 16 + 4 * t, ib);
        r.s = 0.0f;
        r.m0 = (int) (int8_t) sc[0];
        r.m1 = (int) (int8_t) sc[1];
        r.o = 0.0f;
        return r;
    }
    // the half's four blocks at once: ql's two byte sets (blocks 0 / 2 and 1 / 3) and qh's one read once
    __device__ static __forceinline__ void steps4(const uint8_t* pd, const uint8_t* ph, int c, int t, float, const uint8_t* pd2,
                                                  Step* out) {
        const uint32_t h_lo = rd32(pd2 + 4 * t), h_hi = rd32(pd2 + 16 + 4 * t);
        const uint8_t* sc = ph + 8 * (c & 1);
        const uint32_t sa = rd32(sc), sb = rd32(sc + 4);   // the half's eight int8 scales
#pragma unroll
        for (int pr = 0; pr < 2; ++pr) {   // (blocks pr and pr + 2: the low and the high nibbles of ql[32 pr ..])
            const uint32_t l_lo = rd32(pd + 32 * pr + 4 * t), l_hi = rd32(pd + 32 * pr + 16 + 4 * t);
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int ib = pr + 2 * k;
                const uint32_t ss = ib < 2 ? sa >> (16 * ib) : sb >> (16 * (ib - 2));
                const uint32_t v0 = ((l_lo >> (4 * k)) & 0x0F0F0F0Fu) | (((h_lo >> (2 * ib)) & 0x03030303u) << 4);
                const uint32_t v1 = ((l_hi >> (4 * k)) & 0x0F0F0F0Fu) | (((h_hi >> (2 * ib)) & 0x03030303u) << 4);
                out[ib].lo = ((v0 | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
                out[ib].hi = ((v1 | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
                out[ib].s = 0.0f;
                out[ib].m0 = (int) (int8_t) (ss & 255u);
                out[ib].m1 = (int) (int8_t) ((ss >> 8) & 255u);
                out[ib].o = 0.0f;
            }
        }
    }
};

// Q8_0: 32 values a block: {f16 d; 32 x int8}.  A chunk is four blocks.
template<> struct Dec<8> {
    static constexpr int GW = 0;
    __device__ static __forceinline__ const uint32_t* table() { return nullptr; }
    static constexpr int WCH = 128, CB = 136, AL = 2, SUB = 32;
    __device__ static __forceinline__ float head(const uint8_t*) { return 1.0f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float, const uint32_t*) {
        const uint8_t* b = p + 34 * ib;
        const uint8_t* q = b + 2 + 4 * t;
        Step r;
        r.lo = rd16(q) | rd16(q + 2) << 16;
        r.hi = rd16(q + 16) | rd16(q + 18) << 16;
        r.s = rd_f16(b);
        r.m0 = r.m1 = 0;
        r.o = 0.0f;
        return r;
    }
};

// NVFP4: 64 values a block: {4 x u8 UE4M3 scales, one a 16 values; 32 bytes: sub-block s's values 0-7 in the low
// nibbles of bytes 8 s .. 8 s + 7, 8-15 in their high nibbles}; the nibbles as MXFP4's (sign | 3 bits, the table
// doubled).  A step is two sub-blocks: lo is the first, hi the second, each under its own float scale (FS16).  The
// tensor's second-level scale (an expert's) is the caller's.  A chunk is two blocks.
__device__ __forceinline__ float ue4m3_half(uint32_t x) {   // the scale / 2 (the table's values are doubled)
    if (x == 0 || x == 0x7F) return 0.0f;
    const uint32_t e = (x >> 3) & 15u, m = x & 7u;
    return 0.5f * (e == 0 ? (float) m * (1.0f / 512.0f) : __uint_as_float((e + 120u) << 23 | m << 20));
}
template<> struct Dec<40> {
    static constexpr int GW = 0;
    __device__ static __forceinline__ const uint32_t* table() { return nullptr; }
    static constexpr int WCH = 128, CB = 72, AL = 1, SUB = 16;
    static constexpr bool FS16 = true;
    __device__ static __forceinline__ float head(const uint8_t*) { return 1.0f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float, const uint32_t*) {
        const uint8_t* b = p + 36 * (ib >> 1);
        const int s0 = 2 * (ib & 1);   // the step's first sub-block
        // thread t's four values of a sub-block: t < 2 the low nibbles of its bytes 4 t .., else the high ones of 4 (t - 2) ..
        const int o = 4 * (t & 1), sh = 4 * (t >> 1);
        const uint8_t* q0 = b + 4 + 8 * s0 + o;
        const uint8_t* q1 = q0 + 8;
        const uint32_t v0 = (uint32_t) q0[0] | (uint32_t) q0[1] << 8 | (uint32_t) q0[2] << 16 | (uint32_t) q0[3] << 24;
        const uint32_t v1 = (uint32_t) q1[0] | (uint32_t) q1[1] << 8 | (uint32_t) q1[2] << 16 | (uint32_t) q1[3] << 24;
        Step r;
        r.lo = fp4_bytes((v0 >> sh) & 0x0F0F0F0Fu);
        r.hi = fp4_bytes((v1 >> sh) & 0x0F0F0F0Fu);
        r.s = ue4m3_half(b[s0]);        // lo's scale
        r.s2 = ue4m3_half(b[s0 + 1]);   // hi's
        r.m0 = r.m1 = 0;
        return r;
    }
};
// (a Dec with FS16: two float scales a step, Step::s for lo and Step::s2 for hi)
template<int TY, typename = void> struct Fs16 { static constexpr bool on = false; };
template<int TY> struct Fs16<TY, decltype((void) Dec<TY>::FS16)> { static constexpr bool on = Dec<TY>::FS16; };

// IQ3_S: {f16 d; 64 codebook bytes (4 values each; a ninth index bit in qh); 8 x u8 qh; 32 sign bytes (one a 8
// values); 4 x u8: two 4-bit scales, one a 32 values}: a value is d (2 s + 1) x the codebook value x its sign.
template<> struct Dec<21> {
    static constexpr int GW = 512;
    __device__ static __forceinline__ const uint32_t* table() { return iq3s_grid; }
    static constexpr int WCH = 256, CB = 110, AL = 2, SUB = 32;
    __device__ static __forceinline__ float head(const uint8_t* p) { return rd_f16(p); }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float hd, const uint32_t* grid) {
        const uint8_t* q = p + 2 + 8 * ib;
        const uint32_t qh = p[2 + 64 + ib], sc = p[2 + 104 + (ib >> 1)];
        const uint32_t sg = (uint32_t) p[2 + 72 + 4 * ib + (t >> 1)] >> (4 * (t & 1));         // values 4 t .. 's signs
        const uint32_t sg2 = (uint32_t) p[2 + 72 + 4 * ib + 2 + (t >> 1)] >> (4 * (t & 1));    // values 16 + 4 t .. 's
        Step r;
        r.lo = neg_nz(grid[q[t] | (((qh >> t) & 1u) << 8)], nib_mask(sg));
        r.hi = neg_nz(grid[q[4 + t] | (((qh >> (4 + t)) & 1u) << 8)], nib_mask(sg2));
        r.s = hd * (float) (2 * (int) ((sc >> (4 * (ib & 1))) & 15u) + 1);
        r.m0 = r.m1 = 0;
        return r;
    }
};

// bytes of v (0-127) minus k (a byte value repeated), a byte at a time
__device__ __forceinline__ uint32_t sub8(uint32_t v, uint32_t k) { return ((v | 0x80808080u) - k) ^ 0x80808080u; }

// Q4_0: 32 values a block: {f16 d; 16 bytes: value i in byte i's low nibble, value 16 + i in its high nibble}, a value
// is d (q - 8).  A chunk is four blocks.
template<> struct Dec<2> {
    static constexpr int GW = 0;
    __device__ static __forceinline__ const uint32_t* table() { return nullptr; }
    static constexpr int WCH = 128, CB = 72, AL = 2, SUB = 32;
    __device__ static __forceinline__ float head(const uint8_t*) { return 1.0f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float, const uint32_t*) {
        const uint8_t* b = p + 18 * ib;
        const uint32_t v = rd32(b + 2 + 4 * t);
        Step r;
        r.lo = sub8(v & 0x0F0F0F0Fu, 0x08080808u);
        r.hi = sub8((v >> 4) & 0x0F0F0F0Fu, 0x08080808u);
        r.s = rd_f16(b);
        r.m0 = r.m1 = 0;
        return r;
    }
};

// Q4_1: {f16 d; f16 m; 16 bytes as Q4_0's}, a value is d q + m: the int8 is q, the offset - m (SUMS).
template<> struct Dec<3> {
    static constexpr int GW = 0;
    __device__ static __forceinline__ const uint32_t* table() { return nullptr; }
    static constexpr int WCH = 128, CB = 80, AL = 2, SUB = 32;
    static constexpr bool SUMS = true;
    __device__ static __forceinline__ float head(const uint8_t*) { return 1.0f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float, const uint32_t*) {
        const uint8_t* b = p + 20 * ib;
        const uint32_t v = rd32(b + 4 + 4 * t);
        Step r;
        r.lo = v & 0x0F0F0F0Fu;
        r.hi = (v >> 4) & 0x0F0F0F0Fu;
        r.s = rd_f16(b);
        r.m0 = r.m1 = 0;
        r.o = -rd_f16(b + 2);
        return r;
    }
};

// Q5_0: {f16 d; u32 qh; 16 bytes}: value i is byte i's low nibble | qh's bit i << 4, value 16 + i its high nibble |
// qh's bit 16 + i << 4; a value is d (q - 16).
template<> struct Dec<6> {
    static constexpr int GW = 0;
    __device__ static __forceinline__ const uint32_t* table() { return nullptr; }
    static constexpr int WCH = 128, CB = 88, AL = 2, SUB = 32;
    __device__ static __forceinline__ float head(const uint8_t*) { return 1.0f; }
    __device__ static __forceinline__ Step step(const uint8_t* p, int ib, int t, float, const uint32_t*) {
        const uint8_t* b = p + 22 * ib;
        const uint32_t qh = rd32(b + 2), v = rd32(b + 6 + 4 * t);
        const uint32_t h0 = (nib_mask(qh >> (4 * t)) & 0x01010101u) << 4, h1 = (nib_mask(qh >> (16 + 4 * t)) & 0x01010101u) << 4;
        Step r;
        r.lo = sub8((v & 0x0F0F0F0Fu) | h0, 0x10101010u);
        r.hi = sub8(((v >> 4) & 0x0F0F0F0Fu) | h1, 0x10101010u);
        r.s = rd_f16(b);
        r.m0 = r.m1 = 0;
        return r;
    }
};

// Q5_K: 256 values: {f16 d; f16 dmin; 12 bytes of scales and mins as Q4_K's; 32 bytes qh; 128 bytes as Q4_K's}: a
// value is d sc q - dmin m, q = the nibble | qh's bit (the block's number) of the value's byte << 4.  A chunk is half
// of it: the 48-byte header (d, dmin, scales, qh - both halves bring it) and the half's 64 bytes.
template<bool A4> struct DecQ5K {
    static constexpr int GW = 0;
    __device__ static __forceinline__ const uint32_t* table() { return nullptr; }
    static constexpr int WCH = 128, CB = 64, AL = 2, SUB = 32;
    static constexpr int HDR = 48, RB = 88;
    static constexpr bool FOUR = true;
    static constexpr bool SUMS = true;
    __device__ static __forceinline__ size_t hdr(int c) { return (size_t) (c >> 1) * 176; }
    __device__ static __forceinline__ size_t dat(int c) { return (size_t) (c >> 1) * 176 + 48 + (size_t) (c & 1) * 64; }
    __device__ static __forceinline__ float head(const uint8_t* ph) { return rd_f16(ph); }
    __device__ static __forceinline__ Step step_h(const uint8_t* pd, const uint8_t* ph, int c, int ib, int t, float hd) {
        const int j = (c & 1) * 4 + ib;   // the block in its 256
        const uint8_t* sc = ph + 4;
        uint32_t scale, mn;
        if (j < 4) {
            scale = sc[j] & 63u;
            mn = sc[j + 4] & 63u;
        } else {
            scale = (sc[j + 4] & 15u) | ((uint32_t) (sc[j - 4] >> 6) << 4);
            mn = (uint32_t) (sc[j + 4] >> 4) | ((uint32_t) (sc[j] >> 6) << 4);
        }
        const uint8_t* q = pd + 32 * (ib >> 1) + 4 * t;
        const uint8_t* qh = ph + 16 + 4 * t;
        const int sh = 4 * (ib & 1);
        Step r;
        r.lo = ((rd32(q) >> sh) & 0x0F0F0F0Fu) | (((rd32(qh) >> j) & 0x01010101u) << 4);
        r.hi = ((rd32(q + 16) >> sh) & 0x0F0F0F0Fu) | (((rd32(qh + 16) >> j) & 0x01010101u) << 4);
        r.s = hd * (float) scale;
        r.m0 = r.m1 = 0;
        r.o = rd_f16(ph + 2) * (float) mn;
        return r;
    }
    // the half's four blocks at once: a pair of blocks shares its bytes (low / high nibbles) - read once; the 12
    // scale bytes as three words; qh's bytes read once for the four
    __device__ static __forceinline__ void steps4(const uint8_t* pd, const uint8_t* ph, int c, int t, float hd, Step* out) {
        const uint32_t s0 = rd32x<A4>(ph + 4), s1 = rd32x<A4>(ph + 8), s2 = rd32x<A4>(ph + 12);   // scale bytes 0-3, 4-7, 8-11
        const float dmin = rd_f16(ph + 2);
        const bool second = (c & 1) != 0;
        const uint32_t qh_lo = rd32x<A4>(ph + 16 + 4 * t), qh_hi = rd32x<A4>(ph + 32 + 4 * t);
        const int j0 = second ? 4 : 0;
#pragma unroll
        for (int pr = 0; pr < 2; ++pr) {
            const uint32_t lo = rd32x<A4>(pd + 32 * pr + 4 * t), hi = rd32x<A4>(pd + 32 * pr + 16 + 4 * t);
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int ib = 2 * pr + k;   // (block j = 4 (c & 1) + ib of the 256)
                const uint32_t b0 = (s0 >> (8 * ib)) & 255u, b1 = (s1 >> (8 * ib)) & 255u, b2 = (s2 >> (8 * ib)) & 255u;
                // j < 4: scale = sc[j] & 63, min = sc[j + 4] & 63;  else: scale = sc[j + 4] & 15 | sc[j - 4] >> 6 << 4,
                // min = sc[j + 4] >> 4 | sc[j] >> 6 << 4   (sc[j - 4] = b0, sc[j] = b1, sc[j + 4] = b2 there)
                const uint32_t scale = second ? (b2 & 15u) | ((b0 >> 6) << 4) : b0 & 63u;
                const uint32_t mn = second ? (b2 >> 4) | ((b1 >> 6) << 4) : b1 & 63u;
                out[ib].lo = ((lo >> (4 * k)) & 0x0F0F0F0Fu) | (((qh_lo >> (j0 + ib)) & 0x01010101u) << 4);
                out[ib].hi = ((hi >> (4 * k)) & 0x0F0F0F0Fu) | (((qh_hi >> (j0 + ib)) & 0x01010101u) << 4);
                out[ib].s = hd * (float) scale;
                out[ib].m0 = out[ib].m1 = 0;
                out[ib].o = dmin * (float) mn;
            }
        }
    }
};
template<> struct Dec<13> : DecQ5K<false> {};
template<> struct Dec<113> : DecQ5K<true> {};   // (every row and chunk on a multiple of 4: the block is 176 bytes)

// Q3_K: 256 values: {32 bytes hmask; 64 bytes qs; 12 bytes: sixteen 6-bit scales (one a 16 values); f16 d}.  In half
// n block j takes bits 2 j of qs[32 n ..] and bit 4 n + j of hmask[0 .. 31]: a value is d (sc - 32) (q - (hmask ?
// 0 : 4)).  A chunk is a half: the header (scales and d: 14 bytes), the half's 32 bytes of qs, hmask (a second range).
template<> struct Dec<11> {
    static constexpr int GW = 0;
    __device__ static __forceinline__ const uint32_t* table() { return nullptr; }
    // The value is d x m x q: q 3 bits (-4..3), m the 6-bit scale of its 16 values (-32..31).  -m x q fits an int8
    // (-128..124), so the weights go to the multiply with their scale inside - one multiply a 32 values with the
    // step's scale -d, the same integer sums as two multiplies a 16 values times m (this ran 4-7% slower than MMQ
    // on a dense layer that way).
    static constexpr int WCH = 128, CB = 32, AL = 2, SUB = 32;
    static constexpr int HDR = 14, RB = 55, CB2 = 32;
    static constexpr bool FOUR = true;
    __device__ static __forceinline__ size_t hdr(int c) { return (size_t) (c >> 1) * 110 + 96; }
    __device__ static __forceinline__ size_t dat(int c) { return (size_t) (c >> 1) * 110 + 32 + (size_t) (c & 1) * 32; }
    __device__ static __forceinline__ size_t dat2(int c) { return (size_t) (c >> 1) * 110; }
    __device__ static __forceinline__ float head(const uint8_t* ph) { return rd_f16(ph + 12); }
    __device__ static __forceinline__ int scale6(const uint8_t* sc, int is) {   // scale is of the 16
        const uint32_t us = is < 4  ? (sc[is] & 15u) | ((sc[is + 8] & 3u) << 4)
                          : is < 8  ? (sc[is] & 15u) | (((uint32_t) (sc[is + 4] >> 2) & 3u) << 4)
                          : is < 12 ? (uint32_t) (sc[is - 8] >> 4) | (((uint32_t) (sc[is] >> 4) & 3u) << 4)
                                    : (uint32_t) (sc[is - 8] >> 4) | (((uint32_t) (sc[is - 4] >> 6) & 3u) << 4);
        return (int) us - 32;
    }
    // four values u (0..7, a byte each) as -m x (u - 4), bytes mod 256
    __device__ static __forceinline__ uint32_t scaled(uint32_t u, int m) {
        const uint32_t k = (uint32_t) (-m) & 255u;
        const uint32_t pe = ((u & 0x00FF00FFu) * k) & 0x00FF00FFu, po = (((u >> 8) & 0x00FF00FFu) * k) & 0x00FF00FFu;   // (u k <= 1785: no carry between the 16-bit lanes)
        return add8(pe | (po << 8), ((uint32_t) (4 * m) & 255u) * 0x01010101u);
    }
    __device__ static __forceinline__ uint32_t bits3(uint32_t q, uint32_t hm, int j, int bit) {
        return ((q >> (2 * j)) & 0x03030303u) | (((hm >> bit) & 0x01010101u) << 2);
    }
    __device__ static __forceinline__ Step step_h(const uint8_t* pd, const uint8_t* ph, int c, int ib, int t, float,
                                                  const uint8_t* pd2) {
        const int n = c & 1;
        Step r;
        r.lo = scaled(bits3(rd32(pd + 4 * t), rd32(pd2 + 4 * t), ib, 4 * n + ib), scale6(ph, 8 * n + 2 * ib));
        r.hi = scaled(bits3(rd32(pd + 16 + 4 * t), rd32(pd2 + 16 + 4 * t), ib, 4 * n + ib), scale6(ph, 8 * n + 2 * ib + 1));
        r.s = -rd_f16(ph + 12);
        r.m0 = r.m1 = 0;
        return r;
    }
    // the half's four blocks at once: its 2-bit bytes and the high-bit bytes are the same four words for the four
    // blocks; the 12 scale bytes as three words
    __device__ static __forceinline__ void steps4(const uint8_t* pd, const uint8_t* ph, int c, int t, float hd, const uint8_t* pd2,
                                                  Step* out) {
        const int n = c & 1;
        const uint32_t ql = rd32(pd + 4 * t), qh = rd32(pd + 16 + 4 * t), hl = rd32(pd2 + 4 * t), hh = rd32(pd2 + 16 + 4 * t);
        const uint32_t w[3] = {rd32(ph), rd32(ph + 4), rd32(ph + 8)};
        // scale is = 8 n + is0 (is0 0..7): its low 4 bits in byte is0 (the low nibble, n = 1: the high), its high 2
        // in byte 8 + is0 % 4 at bit 2 (is0 / 4) + 4 n   (scale6, by cases)
#pragma unroll
        for (int ib = 0; ib < 4; ++ib) {
            int m[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int is0 = 2 * ib + e;
                const uint32_t lo4 = ((w[is0 >> 2] >> (8 * (is0 & 3))) >> (4 * n)) & 15u;
                const uint32_t hi2 = ((w[2] >> (8 * (is0 & 3))) >> (2 * (is0 >> 2) + 4 * n)) & 3u;
                m[e] = (int) (lo4 | (hi2 << 4)) - 32;
            }
            out[ib].lo = scaled(bits3(ql, hl, ib, 4 * n + ib), m[0]);
            out[ib].hi = scaled(bits3(qh, hh, ib, 4 * n + ib), m[1]);
            out[ib].s = -hd;
            out[ib].m0 = out[ib].m1 = 0;
        }
    }
};

// Q2_K: 256 values: {16 bytes: a 4-bit scale and a 4-bit min a 16 values; 64 bytes qs; f16 d; f16 dmin}.  In half n
// block j takes bits 2 j of qs[32 n ..]: a value is d sc q - dmin m, q in 0..3.  Two float scales and two offsets a
// step (FS16, OFS16: the kernel gets each 16 values' sum of activations from a product with ones).
// A chunk is a half: the header (the 16 scale bytes), the half's 32 bytes of qs, d and dmin (a second range).
template<> struct Dec<10> {
    static constexpr int GW = 0;
    __device__ static __forceinline__ const uint32_t* table() { return nullptr; }
    // The 4-bit scale of a 16 values goes into the int8 weights (q x scale <= 45): one multiply a 32 values with the
    // step's scale d.  The offsets stay a 16 values (OFS16).
    static constexpr int WCH = 128, CB = 32, AL = 2, SUB = 32;
    static constexpr int HDR = 16, RB = 42, CB2 = 4;
    static constexpr bool OFS16 = true;
    static constexpr bool FOUR = true;
    __device__ static __forceinline__ size_t hdr(int c) { return (size_t) (c >> 1) * 84; }
    __device__ static __forceinline__ size_t dat(int c) { return (size_t) (c >> 1) * 84 + 16 + (size_t) (c & 1) * 32; }
    __device__ static __forceinline__ size_t dat2(int c) { return (size_t) (c >> 1) * 84 + 80; }
    __device__ static __forceinline__ float head(const uint8_t*) { return 1.0f; }
    __device__ static __forceinline__ Step step_h(const uint8_t* pd, const uint8_t* ph, int c, int ib, int t, float,
                                                  const uint8_t* pd2) {
        const uint32_t s0 = ph[8 * (c & 1) + 2 * ib], s1 = ph[8 * (c & 1) + 2 * ib + 1];
        const float d = rd_f16(pd2), dmin = rd_f16(pd2 + 2);
        Step r;
        r.lo = ((rd32(pd + 4 * t) >> (2 * ib)) & 0x03030303u) * (s0 & 15u);
        r.hi = ((rd32(pd + 16 + 4 * t) >> (2 * ib)) & 0x03030303u) * (s1 & 15u);
        r.s = d;
        r.s2 = 0.0f;
        r.o = dmin * (float) (s0 >> 4);
        r.o2 = dmin * (float) (s1 >> 4);
        r.m0 = r.m1 = 0;
        return r;
    }
    // the half's four blocks at once: its two data words, its eight scale bytes and d, dmin read once
    __device__ static __forceinline__ void steps4(const uint8_t* pd, const uint8_t* ph, int c, int t, float, const uint8_t* pd2,
                                                  Step* out) {
        const int n = c & 1;
        const uint32_t sa = rd32(ph + 8 * n), sb = rd32(ph + 8 * n + 4);
        const float d = rd_f16(pd2), dmin = rd_f16(pd2 + 2);
        const uint32_t ql = rd32(pd + 4 * t), qh = rd32(pd + 16 + 4 * t);
#pragma unroll
        for (int ib = 0; ib < 4; ++ib) {
            const uint32_t ss = (ib < 2 ? sa : sb) >> (16 * (ib & 1));
            const uint32_t s0 = ss & 255u, s1 = (ss >> 8) & 255u;
            out[ib].lo = ((ql >> (2 * ib)) & 0x03030303u) * (s0 & 15u);
            out[ib].hi = ((qh >> (2 * ib)) & 0x03030303u) * (s1 & 15u);
            out[ib].s = d;
            out[ib].s2 = 0.0f;
            out[ib].o = dmin * (float) (s0 >> 4);
            out[ib].o2 = dmin * (float) (s1 >> 4);
            out[ib].m0 = out[ib].m1 = 0;
        }
    }
};
// (a Dec with FOUR: steps4 decodes the half chunk's four blocks at once, sharing their loads)
template<int TY, typename = void> struct Four { static constexpr bool on = false; };
template<int TY> struct Four<TY, decltype((void) Dec<TY>::FOUR)> { static constexpr bool on = Dec<TY>::FOUR; };
// (a Dec with OFS16: an offset a 16 values, Step::o for lo and Step::o2 for hi)
template<int TY, typename = void> struct Ofs16 { static constexpr bool on = false; };
template<int TY> struct Ofs16<TY, decltype((void) Dec<TY>::OFS16)> { static constexpr bool on = Dec<TY>::OFS16; };

// A chunk's bytes in its row: one range (chunk c at c * CB), or - a Dec with HDR - a header range and a data range
template<int TY, typename = void> struct Rng {
    static constexpr int HDR = 0, RB = Dec<TY>::CB;
    __device__ static __forceinline__ size_t hdr(int) { return 0; }
    __device__ static __forceinline__ size_t dat(int c) { return (size_t) c * Dec<TY>::CB; }
};
// (a Dec with SUMS needs the activations' block sums)
template<int TY, typename = void> struct Sums { static constexpr bool on = false; };
template<int TY> struct Sums<TY, decltype((void) Dec<TY>::SUMS)> { static constexpr bool on = Dec<TY>::SUMS; };
template<int TY> struct Rng<TY, decltype((void) Dec<TY>::HDR)> {
    static constexpr int HDR = Dec<TY>::HDR, RB = Dec<TY>::RB;
    __device__ static __forceinline__ size_t hdr(int c) { return Dec<TY>::hdr(c); }
    __device__ static __forceinline__ size_t dat(int c) { return Dec<TY>::dat(c); }
};
// (a Dec with CB2: a second data range a chunk, dat2(c))
template<int TY, typename = void> struct Rng2 {
    static constexpr int CB2 = 0;
    __device__ static __forceinline__ size_t dat2(int) { return 0; }
};
template<int TY> struct Rng2<TY, decltype((void) Dec<TY>::CB2)> {
    static constexpr int CB2 = Dec<TY>::CB2;
    __device__ static __forceinline__ size_t dat2(int c) { return Dec<TY>::dat2(c); }
};

}  // namespace strata::prefill::own
