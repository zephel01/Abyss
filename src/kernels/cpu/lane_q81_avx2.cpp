// src/kernels/cpu/lane_q81_avx2.cpp - lane_q81's AVX2 rows: every type the device's expert kernels take, against a Q81
// activation (lane_q81.hpp), in the device's integer and float steps.
//
// A row is read in groups of 256 values (8 sub-blocks of 32: a q8_1 block each).  A group is decoded once for up to 4
// tokens: each sub-block's 32 weights as bytes (unsigned, signed, or a grid's magnitudes with a sign mask), then per
// token maddubs + madd give 8 int32 lanes of 4 products each, and two levels of hadd the 16 sums of 16 values (A: the
// first 16 of each sub-block, B: the second).  Those are the device's integer sums (its dp4a chains summed in
// another order - integer, so exact); the type's finish then takes the device's integer steps on them (IQ2_XXS's
// sumi * ls / 8 truncated toward zero, IQ2_XS's (s0 ls0 + s1 ls1 + (s0 + s1) / 2) / 4, ...) and its float steps for
// each 32 values (d_w * d_x, then times the sum) - only the float additions across the row come in another order.
//
// Built with AVX2 / FMA / F16C and no FP contraction (each product rounds as the device's does); built again with
// AVX-VNNI (lane_q81_vnni.cpp: one dpbusd for each maddubs + madd pair, the same integers) for the CPUs that have it.
#include "strata/kernels/cpu/lane_q81.hpp"
#include "lane_q81_impl.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace strata::kernels::cpu {
namespace {

using q81::h2f;

inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline float hf(const uint8_t* p) { return h2f(u16(p)); }
inline __m256i set4(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {   // a: bytes 0-7
    return _mm256_set_epi64x((long long) d, (long long) c, (long long) b, (long long) a);
}

// bit i of a byte -> byte i of a qword 0xff (IQ2_S / IQ3_S signs); ksigns64: the same for ksigns_iq2xs[i]
struct Tabs {
    alignas(64) uint64_t bit64[256];
    alignas(64) uint32_t q20[256];       // a Q2_0 byte -> its 4 codes - 1 as int8
    alignas(64) uint64_t iq1p[2048];     // iq1s_grid + 1 (the device's iq1s_grid_gpu values as bytes)
    Tabs() {
        for (int b = 0; b < 256; ++b) {
            uint64_t m = 0;
            for (int i = 0; i < 8; ++i)
                if (b >> i & 1) m |= 0xffull << (8 * i);
            bit64[b] = m;
            uint32_t q = 0;
            for (int i = 0; i < 4; ++i) q |= (uint32_t) (uint8_t) (int8_t) (((b >> (2 * i)) & 3) - 1) << (8 * i);
            q20[b] = q;
        }
        for (int e = 0; e < 2048; ++e) {
            uint64_t v = 0;
            for (int i = 0; i < 8; ++i) v |= (uint64_t) (uint8_t) ((int8_t) (iq1s_grid[e] >> (8 * i)) + 1) << (8 * i);
            iq1p[e] = v;
        }
    }
};
const Tabs& tabs() {
    static const Tabs T;
    return T;
}

inline __m256 i2f(__m256i v) { return _mm256_cvtepi32_ps(v); }
// x / 2^k truncated toward zero (C's integer division)
template <int K> inline __m256i tdiv(__m256i x) {
    return _mm256_srai_epi32(_mm256_add_epi32(x, _mm256_and_si256(_mm256_srai_epi32(x, 31), _mm256_set1_epi32((1 << K) - 1))), K);
}
inline __m256i mul(__m256i a, __m256i b) { return _mm256_mullo_epi32(a, b); }
inline __m256i add(__m256i a, __m256i b) { return _mm256_add_epi32(a, b); }
inline __m256 fmul(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
inline __m256 fadd(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
inline __m256 fsub(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
inline __m256 bc(float v) { return _mm256_set1_ps(v); }

// a nibble vector's 32 values (16 bytes: low nibbles = values 0-15, high = 16-31) through a 16-entry int8 table
inline __m256i nib_lookup(const uint8_t* q16, __m256i table) {
    const __m128i v = _mm_loadu_si128((const __m128i*) q16);
    const __m128i m4 = _mm_set1_epi8(15);
    const __m256i idx = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(v, 4), m4), _mm_and_si128(v, m4));
    return _mm256_shuffle_epi8(table, idx);
}
inline __m256i nib_plain(const uint8_t* q16) {
    const __m128i v = _mm_loadu_si128((const __m128i*) q16);
    const __m128i m4 = _mm_set1_epi8(15);
    return _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(v, 4), m4), _mm_and_si128(v, m4));
}
// 32 bits -> 32 bytes, 0xff where set
inline __m256i bits_to_bytes(uint32_t x) {
    const __m256i shuf = _mm256_setr_epi64x(0x0000000000000000, 0x0101010101010101, 0x0202020202020202, 0x0303030303030303);
    __m256i b = _mm256_shuffle_epi8(_mm256_set1_epi32((int) x), shuf);
    const __m256i bm = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    b = _mm256_or_si256(b, bm);
    return _mm256_cmpeq_epi8(b, _mm256_set1_epi64x(-1));
}
inline __m256i table16(const int8_t* t) {
    const __m128i v = _mm_loadu_si128((const __m128i*) t);
    return _mm256_set_m128i(v, v);
}
inline float e8m0(uint8_t x) { return q81::bits_f(x == 0 ? 0x00400000u : (uint32_t) x << 23); }
inline float ue4m3(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (x >> 3) & 0xF, m = x & 7;
    const float v = e == 0 ? (float) m * (1.0f / 512.0f) : q81::bits_f((uint32_t) (e + 120) << 23 | (uint32_t) m << 20);
    return 0.5f * v;
}

// a token's data for one group of 256 values
struct TokG {
    __m256 d8, s8;     // the 8 sub-blocks' scales and sums (0 past the row's end)
    __m256i ua, ub;    // the sums of each sub-block's first and second 16 codes
    int16_t b8[32];    // the sums of each 8 codes
};

enum Form { PLAIN = 0, SIGNED = 1, GRID = 2 };

// ---------------------------------------------------------------- the types
// dec(b, ns, w, m, P): the group at b (its first block), ns sub-blocks; fin: the group's float contribution
struct T16 {   // IQ2_XXS
    static constexpr int qk = 256, bytes = 66, form = GRID, needs_u = 0;
    struct P { __m256i ls; float d; };
    static void dec(const uint8_t* b, int, __m256i* w, __m256i* m, P& p) {
        alignas(32) int ls[8];
        p.d = hf(b);
        for (int j = 0; j < 8; ++j) {
            const uint8_t* q = b + 2 + 8 * j;
            const uint32_t a = u32(q + 4);
            w[j] = set4(iq2xxs_grid[q[0]], iq2xxs_grid[q[1]], iq2xxs_grid[q[2]], iq2xxs_grid[q[3]]);
            m[j] = set4(ksigns64[a & 127], ksigns64[(a >> 7) & 127], ksigns64[(a >> 14) & 127], ksigns64[(a >> 21) & 127]);
            ls[j] = (int) (a >> 27 | 1);
        }
        p.ls = _mm256_load_si256((const __m256i*) ls);
    }
    static __m256 fin(const P& p, const __m256i*, __m256i A, __m256i B, const TokG& g) {
        const __m256i t = tdiv<3>(mul(add(A, B), p.ls));
        return fmul(fmul(bc(p.d), g.d8), i2f(t));
    }
};
struct T17 {   // IQ2_XS
    static constexpr int qk = 256, bytes = 74, form = GRID, needs_u = 0;
    struct P { __m256i l0, l1; float d; };
    static void dec(const uint8_t* b, int, __m256i* w, __m256i* m, P& p) {
        alignas(32) int l0[8], l1[8];
        p.d = hf(b);
        for (int j = 0; j < 8; ++j) {
            uint16_t q[4];
            std::memcpy(q, b + 2 + 8 * j, 8);
            w[j] = set4(iq2xs_grid[q[0] & 511], iq2xs_grid[q[1] & 511], iq2xs_grid[q[2] & 511], iq2xs_grid[q[3] & 511]);
            m[j] = set4(ksigns64[q[0] >> 9], ksigns64[q[1] >> 9], ksigns64[q[2] >> 9], ksigns64[q[3] >> 9]);
            l0[j] = b[66 + j] & 15;
            l1[j] = b[66 + j] >> 4;
        }
        p.l0 = _mm256_load_si256((const __m256i*) l0);
        p.l1 = _mm256_load_si256((const __m256i*) l1);
    }
    static __m256 fin(const P& p, const __m256i*, __m256i A, __m256i B, const TokG& g) {
        const __m256i t = tdiv<2>(add(add(mul(A, p.l0), mul(B, p.l1)), tdiv<1>(add(A, B))));
        return fmul(fmul(bc(p.d), g.d8), i2f(t));
    }
};
struct T22 {   // IQ2_S
    static constexpr int qk = 256, bytes = 82, form = GRID, needs_u = 0;
    using P = T17::P;
    static void dec(const uint8_t* b, int, __m256i* w, __m256i* m, P& p) {
        const Tabs& T = tabs();
        alignas(32) int l0[8], l1[8];
        p.d = hf(b);
        const uint8_t* qs = b + 2;
        for (int j = 0; j < 8; ++j) {
            const int qh = b[66 + j];
            uint64_t g[4], s[4];
            for (int e = 0; e < 4; ++e) {
                g[e] = iq2s_grid[qs[4 * j + e] | ((qh >> (2 * e)) & 3) << 8];
                s[e] = T.bit64[qs[32 + 4 * j + e]];
            }
            w[j] = set4(g[0], g[1], g[2], g[3]);
            m[j] = set4(s[0], s[1], s[2], s[3]);
            l0[j] = b[74 + j] & 15;
            l1[j] = b[74 + j] >> 4;
        }
        p.l0 = _mm256_load_si256((const __m256i*) l0);
        p.l1 = _mm256_load_si256((const __m256i*) l1);
    }
    static __m256 fin(const P& p, const __m256i* v, __m256i A, __m256i B, const TokG& g) { return T17::fin(p, v, A, B, g); }
};
struct T18 {   // IQ3_XXS
    static constexpr int qk = 256, bytes = 98, form = GRID, needs_u = 0;
    struct P { __m256i ls; float d; };
    static void dec(const uint8_t* b, int, __m256i* w, __m256i* m, P& p) {
        alignas(32) int ls[8];
        p.d = hf(b);
        for (int j = 0; j < 8; ++j) {
            const uint8_t* q = b + 2 + 8 * j;
            const uint32_t a = u32(b + 66 + 4 * j);
            w[j] = _mm256_setr_epi32((int) iq3xxs_grid[q[0]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[2]],
                                     (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[4]], (int) iq3xxs_grid[q[5]],
                                     (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[7]]);
            m[j] = set4(ksigns64[a & 127], ksigns64[(a >> 7) & 127], ksigns64[(a >> 14) & 127], ksigns64[(a >> 21) & 127]);
            ls[j] = (int) (a >> 28);
        }
        p.ls = _mm256_load_si256((const __m256i*) ls);
    }
    static __m256 fin(const P& p, const __m256i*, __m256i A, __m256i B, const TokG& g) {
        const __m256i s = add(A, B);
        const __m256i t = tdiv<1>(add(mul(p.ls, s), tdiv<1>(s)));
        return fmul(fmul(bc(p.d), g.d8), i2f(t));
    }
};
struct T21 {   // IQ3_S
    static constexpr int qk = 256, bytes = 110, form = GRID, needs_u = 0;
    struct P { __m256i ls; float d; };
    static void dec(const uint8_t* b, int, __m256i* w, __m256i* m, P& p) {
        const Tabs& T = tabs();
        alignas(32) int ls[8];
        p.d = hf(b);
        const uint8_t* qs = b + 2;
        for (int j = 0; j < 8; ++j) {
            const int qh = b[66 + j];
            int g[8];
            for (int l = 0; l < 8; ++l) g[l] = (int) iq3s_grid[qs[8 * j + l] | ((qh >> l) & 1) << 8];
            w[j] = _mm256_setr_epi32(g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7]);
            const uint8_t* s = b + 74 + 4 * j;
            m[j] = set4(T.bit64[s[0]], T.bit64[s[1]], T.bit64[s[2]], T.bit64[s[3]]);
            ls[j] = 1 + 2 * ((b[106 + j / 2] >> (4 * (j & 1))) & 15);
        }
        p.ls = _mm256_load_si256((const __m256i*) ls);
    }
    static __m256 fin(const P& p, const __m256i*, __m256i A, __m256i B, const TokG& g) {
        return fmul(fmul(bc(p.d), g.d8), i2f(mul(add(A, B), p.ls)));
    }
};
struct T19 {   // IQ1_S: the grid's values + 1 (0..2), as the device's iq1s_grid_gpu
    static constexpr int qk = 256, bytes = 50, form = PLAIN, needs_u = 0;
    struct P { __m256 d1q, delta; };
    static void dec(const uint8_t* b, int, __m256i* w, __m256i*, P& p) {
        const Tabs& T = tabs();
        alignas(32) float d1q[8], dl[8];
        const float d = hf(b);
        for (int j = 0; j < 8; ++j) {
            const int qh = u16(b + 34 + 2 * j);
            const uint8_t* q = b + 2 + 4 * j;
            w[j] = set4(T.iq1p[q[0] | ((qh >> 0) & 7) << 8], T.iq1p[q[1] | ((qh >> 3) & 7) << 8],
                        T.iq1p[q[2] | ((qh >> 6) & 7) << 8], T.iq1p[q[3] | ((qh >> 9) & 7) << 8]);
            d1q[j] = d * (((qh >> 11) & 0x0E) + 1);
            dl[j] = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
        }
        p.d1q = _mm256_load_ps(d1q);
        p.delta = _mm256_load_ps(dl);
    }
    static __m256 fin(const P& p, const __m256i*, __m256i A, __m256i B, const TokG& g) {
        return fmul(p.d1q, fadd(fmul(g.d8, i2f(add(A, B))), fmul(g.s8, p.delta)));
    }
};
struct T29 {   // IQ1_M: sums of 8 (its deltas a 8 values), the finish one sub-block at a time
    static constexpr int qk = 256, bytes = 56, form = PLAIN, needs_u = 0;
    struct P { float dl[8][4]; int sc0[8], sc1[8]; float d; };
    static void dec(const uint8_t* b, int, __m256i* w, __m256i*, P& p) {
        const Tabs& T = tabs();
        const uint8_t* qh = b + 32;
        uint16_t sc[4];
        std::memcpy(sc, b + 48, 8);
        p.d = h2f((uint16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000)));
        for (int j = 0; j < 8; ++j) {
            uint64_t g[4];
            for (int e = 0; e < 4; ++e) {
                const int qhl = qh[2 * j + e / 2] >> (4 * (e % 2));
                g[e] = T.iq1p[b[4 * j + e] | ((qhl & 7) << 8)];
                p.dl[j][e] = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
            }
            w[j] = set4(g[0], g[1], g[2], g[3]);
            const int tmp = sc[j / 2] >> (6 * (j % 2));
            p.sc0[j] = 2 * ((tmp >> 0) & 7) + 1;
            p.sc1[j] = 2 * ((tmp >> 3) & 7) + 1;
        }
    }
    static __m256 fin(const P& p, const __m256i* v, __m256i, __m256i, const TokG& g) {
        alignas(32) int s8[4][8];
        for (int k = 0; k < 4; ++k)   // per-8 sums: [s(2k) 0, s(2k) 1, s(2k+1) 0, s(2k+1) 1 | s(2k) 2, s(2k) 3, ...]
            _mm256_store_si256((__m256i*) s8[k], _mm256_hadd_epi32(v[2 * k], v[2 * k + 1]));
        alignas(32) float d8[8];
        _mm256_store_ps(d8, g.d8);
        float acc = 0.0f;
        for (int j = 0; j < 8; ++j) {
            const int* s = s8[j / 2];
            const int o = (j & 1) * 2;
            const int a0 = s[o], a1 = s[o + 1], a2 = s[4 + o], a3 = s[4 + o + 1];
            float sumf0 = 0.0f, sumf1 = 0.0f;
            sumf0 += p.dl[j][0] * (float) g.b8[4 * j + 0];
            sumf0 += p.dl[j][1] * (float) g.b8[4 * j + 1];
            sumf1 += p.dl[j][2] * (float) g.b8[4 * j + 2];
            sumf1 += p.dl[j][3] * (float) g.b8[4 * j + 3];
            const float d = p.d * d8[j];
            acc += d * (((float) (a0 + a1) + sumf0) * (float) p.sc0[j] + ((float) (a2 + a3) + sumf1) * (float) p.sc1[j]);
        }
        return _mm256_setr_ps(acc, 0, 0, 0, 0, 0, 0, 0);
    }
};
struct T23 {   // IQ4_XS
    static constexpr int qk = 256, bytes = 136, form = SIGNED, needs_u = 0;
    struct P { __m256i ls; float d; };
    static void dec(const uint8_t* b, int, __m256i* w, __m256i*, P& p) {
        const __m256i t = table16(kvalues_iq4nl);
        alignas(32) int ls[8];
        p.d = hf(b);
        const int sh = u16(b + 2);
        for (int j = 0; j < 8; ++j) {
            w[j] = nib_lookup(b + 8 + 16 * j, t);
            ls[j] = (((b[4 + j / 2] >> (4 * (j & 1))) & 15) | (((sh >> (2 * j)) & 3) << 4)) - 32;
        }
        p.ls = _mm256_load_si256((const __m256i*) ls);
    }
    static __m256 fin(const P& p, const __m256i*, __m256i A, __m256i B, const TokG& g) {
        return fmul(fmul(bc(p.d), g.d8), i2f(mul(add(A, B), p.ls)));
    }
};
// the 32-value formats: a group is 8 blocks; P.dw the blocks' scale factor (0 past the row's end)
struct T20 {   // IQ4_NL
    static constexpr int qk = 32, bytes = 18, form = SIGNED, needs_u = 0;
    struct P { __m256 dw; };
    static void dec(const uint8_t* b, int ns, __m256i* w, __m256i*, P& p) {
        const __m256i t = table16(kvalues_iq4nl);
        alignas(32) float dw[8] = {};
        for (int j = 0; j < ns; ++j) {
            w[j] = nib_lookup(b + bytes * j + 2, t);
            dw[j] = hf(b + bytes * j);
        }
        p.dw = _mm256_load_ps(dw);
    }
    static __m256 fin(const P& p, const __m256i*, __m256i A, __m256i B, const TokG& g) {
        return fmul(fmul(p.dw, g.d8), i2f(add(A, B)));
    }
};
struct T39 {   // MXFP4
    static constexpr int qk = 32, bytes = 17, form = SIGNED, needs_u = 0;
    using P = T20::P;
    static void dec(const uint8_t* b, int ns, __m256i* w, __m256i*, P& p) {
        const __m256i t = table16(kvalues_mxfp4);
        alignas(32) float dw[8] = {};
        for (int j = 0; j < ns; ++j) {
            w[j] = nib_lookup(b + bytes * j + 1, t);
            dw[j] = e8m0(b[bytes * j]) * 0.5f;
        }
        p.dw = _mm256_load_ps(dw);
    }
    static __m256 fin(const P& p, const __m256i* v, __m256i A, __m256i B, const TokG& g) { return T20::fin(p, v, A, B, g); }
};
struct T8 {   // Q8_0
    static constexpr int qk = 32, bytes = 34, form = SIGNED, needs_u = 0;
    using P = T20::P;
    static void dec(const uint8_t* b, int ns, __m256i* w, __m256i*, P& p) {
        alignas(32) float dw[8] = {};
        for (int j = 0; j < ns; ++j) {
            w[j] = _mm256_loadu_si256((const __m256i*) (b + bytes * j + 2));
            dw[j] = hf(b + bytes * j);
        }
        p.dw = _mm256_load_ps(dw);
    }
    static __m256 fin(const P& p, const __m256i* v, __m256i A, __m256i B, const TokG& g) { return T20::fin(p, v, A, B, g); }
};
struct T6 {   // Q5_0: the codes - 16 (the device's signed form)
    static constexpr int qk = 32, bytes = 22, form = SIGNED, needs_u = 0;
    using P = T20::P;
    static void dec(const uint8_t* b, int ns, __m256i* w, __m256i*, P& p) {
        alignas(32) float dw[8] = {};
        for (int j = 0; j < ns; ++j) {
            const uint8_t* q = b + bytes * j;
            const __m256i hb = _mm256_and_si256(bits_to_bytes(u32(q + 2)), _mm256_set1_epi8(0x10));
            w[j] = _mm256_sub_epi8(_mm256_or_si256(nib_plain(q + 6), hb), _mm256_set1_epi8(16));
            dw[j] = hf(q);
        }
        p.dw = _mm256_load_ps(dw);
    }
    static __m256 fin(const P& p, const __m256i* v, __m256i A, __m256i B, const TokG& g) { return T20::fin(p, v, A, B, g); }
};
struct T7 {   // Q5_1
    static constexpr int qk = 32, bytes = 24, form = PLAIN, needs_u = 0;
    struct P { __m256 dx, dy; };
    static void dec(const uint8_t* b, int ns, __m256i* w, __m256i*, P& p) {
        alignas(32) float dx[8] = {}, dy[8] = {};
        for (int j = 0; j < ns; ++j) {
            const uint8_t* q = b + bytes * j;
            const __m256i hb = _mm256_and_si256(bits_to_bytes(u32(q + 4)), _mm256_set1_epi8(0x10));
            w[j] = _mm256_or_si256(nib_plain(q + 8), hb);
            dx[j] = hf(q);
            dy[j] = hf(q + 2);
        }
        p.dx = _mm256_load_ps(dx);
        p.dy = _mm256_load_ps(dy);
    }
    static __m256 fin(const P& p, const __m256i*, __m256i A, __m256i B, const TokG& g) {
        return fadd(fmul(fmul(p.dx, g.d8), i2f(add(A, B))), fmul(p.dy, g.s8));
    }
};
struct T42 {   // Q2_0: 64 values a block (two sub-blocks), codes - 1
    static constexpr int qk = 64, bytes = 18, form = SIGNED, needs_u = 0;
    using P = T20::P;
    static void dec(const uint8_t* b, int ns, __m256i* w, __m256i*, P& p) {
        const Tabs& T = tabs();
        alignas(32) float dw[8] = {};
        for (int j = 0; j < ns; ++j) {
            const uint8_t* blk = b + bytes * (j / 2);
            const uint8_t* q = blk + 2 + 8 * (j & 1);
            w[j] = _mm256_setr_epi32((int) T.q20[q[0]], (int) T.q20[q[1]], (int) T.q20[q[2]], (int) T.q20[q[3]],
                                     (int) T.q20[q[4]], (int) T.q20[q[5]], (int) T.q20[q[6]], (int) T.q20[q[7]]);
            dw[j] = hf(blk);
        }
        p.dw = _mm256_load_ps(dw);
    }
    static __m256 fin(const P& p, const __m256i* v, __m256i A, __m256i B, const TokG& g) { return T20::fin(p, v, A, B, g); }
};
struct T40 {   // NVFP4: 64 values a block, a UE4M3 scale a 16
    static constexpr int qk = 64, bytes = 36, form = SIGNED, needs_u = 0;
    struct P { __m256 u0, u1; };
    static void dec(const uint8_t* b, int ns, __m256i* w, __m256i*, P& p) {
        const __m256i t = table16(kvalues_mxfp4);
        const __m128i m4 = _mm_set1_epi8(15);
        alignas(32) float u0[8] = {}, u1[8] = {};
        for (int j = 0; j < ns; ++j) {
            const uint8_t* blk = b + bytes * (j / 2);
            const int iqs = j & 1;
            const __m128i v = _mm_loadu_si128((const __m128i*) (blk + 4 + 16 * iqs));
            const __m128i lo = _mm_and_si128(v, m4), hi = _mm_and_si128(_mm_srli_epi16(v, 4), m4);
            const __m256i idx = _mm256_set_m128i(_mm_unpackhi_epi64(lo, hi), _mm_unpacklo_epi64(lo, hi));
            w[j] = _mm256_shuffle_epi8(t, idx);
            u0[j] = ue4m3(blk[2 * iqs]);
            u1[j] = ue4m3(blk[2 * iqs + 1]);
        }
        p.u0 = _mm256_load_ps(u0);
        p.u1 = _mm256_load_ps(u1);
    }
    static __m256 fin(const P& p, const __m256i*, __m256i A, __m256i B, const TokG& g) {
        return fmul(g.d8, fadd(fmul(p.u0, i2f(A)), fmul(p.u1, i2f(B))));
    }
};
// ---- K quants: sub-block j of the 256 is q8_1 block j; Q2_K / Q3_K / Q6_K have a scale a 16 values
struct T10 {   // Q2_K
    static constexpr int qk = 256, bytes = 84, form = PLAIN, needs_u = 1;
    struct P { __m256i sa, sb, ma, mb; float dx, dy; };
    static void dec(const uint8_t* b, int, __m256i* w, __m256i*, P& p) {
        const __m256i m3 = _mm256_set1_epi8(3);
        for (int hh = 0; hh < 2; ++hh) {
            const __m256i q = _mm256_loadu_si256((const __m256i*) (b + 16 + 32 * hh));
            w[4 * hh + 0] = _mm256_and_si256(q, m3);
            w[4 * hh + 1] = _mm256_and_si256(_mm256_srli_epi16(q, 2), m3);
            w[4 * hh + 2] = _mm256_and_si256(_mm256_srli_epi16(q, 4), m3);
            w[4 * hh + 3] = _mm256_and_si256(_mm256_srli_epi16(q, 6), m3);
        }
        const __m128i s = _mm_loadu_si128((const __m128i*) b);
        const __m128i m4 = _mm_set1_epi8(15);
        const __m128i lo = _mm_and_si128(s, m4), hi = _mm_and_si128(_mm_srli_epi16(s, 4), m4);
        const __m128i ev = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i od = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1);
        p.sa = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(lo, ev));
        p.sb = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(lo, od));
        p.ma = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(hi, ev));
        p.mb = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(hi, od));
        p.dx = hf(b + 80);
        p.dy = hf(b + 82);
    }
    static __m256 fin(const P& p, const __m256i*, __m256i A, __m256i B, const TokG& g) {
        const __m256i D = add(mul(A, p.sa), mul(B, p.sb));
        const __m256i M = add(mul(g.ua, p.ma), mul(g.ub, p.mb));
        return fsub(fmul(bc(p.dx), fmul(g.d8, i2f(D))), fmul(bc(p.dy), fmul(g.d8, i2f(M))));
    }
};
struct T11 {   // Q3_K
    static constexpr int qk = 256, bytes = 110, form = SIGNED, needs_u = 0;
    struct P { __m256i sa, sb; float d; };
    static void dec(const uint8_t* b, int, __m256i* w, __m256i*, P& p) {
        const __m256i m3 = _mm256_set1_epi8(3), four = _mm256_set1_epi8(4), zero = _mm256_setzero_si256();
        const __m256i hm = _mm256_loadu_si256((const __m256i*) b);
        for (int hh = 0; hh < 2; ++hh) {
            const __m256i q = _mm256_loadu_si256((const __m256i*) (b + 32 + 32 * hh));
            for (int i = 0; i < 4; ++i) {
                const int j = 4 * hh + i;
                const __m256i lo = _mm256_and_si256(_mm256_srli_epi16(q, 2 * i), m3);
                const __m256i clear = _mm256_cmpeq_epi8(_mm256_and_si256(hm, _mm256_set1_epi8((char) (1 << j))), zero);
                w[j] = _mm256_sub_epi8(lo, _mm256_and_si256(clear, four));
            }
        }
        alignas(32) int sa[8], sb[8];
        const uint8_t* sc = b + 96;
        for (int isc = 0; isc < 16; ++isc) {
            const int low = (sc[isc % 8] >> (4 * (isc / 8))) & 0xF;
            const int high = ((sc[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
            ((isc & 1) ? sb : sa)[isc / 2] = (low | high) - 32;
        }
        p.sa = _mm256_load_si256((const __m256i*) sa);
        p.sb = _mm256_load_si256((const __m256i*) sb);
        p.d = hf(b + 108);
    }
    static __m256 fin(const P& p, const __m256i*, __m256i A, __m256i B, const TokG& g) {
        const __m256i D = add(mul(A, p.sa), mul(B, p.sb));
        return fmul(bc(p.d), fmul(g.d8, i2f(D)));
    }
};
inline void k4_scales(const uint8_t* q, int* sc, int* mn) {   // ggml's get_scale_min_k4, the 8 sub-blocks'
    for (int j = 0; j < 8; ++j) {
        if (j < 4) {
            sc[j] = q[j] & 63;
            mn[j] = q[j + 4] & 63;
        } else {
            sc[j] = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
            mn[j] = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        }
    }
}
struct T12 {   // Q4_K
    static constexpr int qk = 256, bytes = 144, form = PLAIN, needs_u = 1;
    struct P { __m256i sc, mn; float dx, dy; };
    static void dec(const uint8_t* b, int, __m256i* w, __m256i*, P& p) {
        const __m256i m4 = _mm256_set1_epi8(15);
        for (int jj = 0; jj < 4; ++jj) {
            const __m256i q = _mm256_loadu_si256((const __m256i*) (b + 16 + 32 * jj));
            w[2 * jj] = _mm256_and_si256(q, m4);
            w[2 * jj + 1] = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
        }
        alignas(32) int sc[8], mn[8];
        k4_scales(b + 4, sc, mn);
        p.sc = _mm256_load_si256((const __m256i*) sc);
        p.mn = _mm256_load_si256((const __m256i*) mn);
        p.dx = hf(b);
        p.dy = hf(b + 2);
    }
    static __m256 fin(const P& p, const __m256i*, __m256i A, __m256i B, const TokG& g) {
        const __m256i D = mul(add(A, B), p.sc);
        const __m256i M = mul(add(g.ua, g.ub), p.mn);
        return fsub(fmul(bc(p.dx), fmul(g.d8, i2f(D))), fmul(bc(p.dy), fmul(g.d8, i2f(M))));
    }
};
struct T13 {   // Q5_K
    static constexpr int qk = 256, bytes = 176, form = PLAIN, needs_u = 1;
    using P = T12::P;
    static void dec(const uint8_t* b, int, __m256i* w, __m256i*, P& p) {
        const __m256i m4 = _mm256_set1_epi8(15), one = _mm256_set1_epi8(1);
        const __m256i qh = _mm256_loadu_si256((const __m256i*) (b + 16));
        for (int jj = 0; jj < 4; ++jj) {
            const __m256i q = _mm256_loadu_si256((const __m256i*) (b + 48 + 32 * jj));
            for (int i = 0; i < 2; ++i) {
                const int j = 2 * jj + i;
                const __m256i lo = i == 0 ? _mm256_and_si256(q, m4) : _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                const __m256i hb = _mm256_and_si256(_mm256_srli_epi16(qh, j), one);
                w[j] = _mm256_or_si256(lo, _mm256_slli_epi16(hb, 4));
            }
        }
        alignas(32) int sc[8], mn[8];
        k4_scales(b + 4, sc, mn);
        p.sc = _mm256_load_si256((const __m256i*) sc);
        p.mn = _mm256_load_si256((const __m256i*) mn);
        p.dx = hf(b);
        p.dy = hf(b + 2);
    }
    static __m256 fin(const P& p, const __m256i* v, __m256i A, __m256i B, const TokG& g) { return T12::fin(p, v, A, B, g); }
};
struct T14 {   // Q6_K
    static constexpr int qk = 256, bytes = 210, form = SIGNED, needs_u = 0;
    struct P { __m256i sa, sb; float d; };
    static void dec(const uint8_t* b, int, __m256i* w, __m256i*, P& p) {
        const __m256i m4 = _mm256_set1_epi8(15), m3 = _mm256_set1_epi8(3), o32 = _mm256_set1_epi8(32);
        for (int hh = 0; hh < 2; ++hh) {
            const __m256i qh = _mm256_loadu_si256((const __m256i*) (b + 128 + 32 * hh));
            for (int k = 0; k < 4; ++k) {
                const __m256i ql = _mm256_loadu_si256((const __m256i*) (b + 64 * hh + 32 * (k & 1)));
                const __m256i lo = (k >> 1) ? _mm256_and_si256(_mm256_srli_epi16(ql, 4), m4) : _mm256_and_si256(ql, m4);
                const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qh, 2 * k), m3);
                w[4 * hh + k] = _mm256_sub_epi8(_mm256_or_si256(lo, _mm256_slli_epi16(hi, 4)), o32);
            }
        }
        const __m128i s = _mm_loadu_si128((const __m128i*) (b + 192));
        const __m128i ev = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i od = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1);
        p.sa = _mm256_cvtepi8_epi32(_mm_shuffle_epi8(s, ev));
        p.sb = _mm256_cvtepi8_epi32(_mm_shuffle_epi8(s, od));
        p.d = hf(b + 208);
    }
    static __m256 fin(const P& p, const __m256i*, __m256i A, __m256i B, const TokG& g) {
        const __m256i D = add(mul(A, p.sa), mul(B, p.sb));
        return fmul(bc(p.d), fmul(g.d8, i2f(D)));
    }
};

// ---------------------------------------------------------------- the rows
inline float hsum(__m256 x) {
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// the token data of every group of a Q81 activation
void tok_groups(const void* act, int n, TokG* g) {
    const q81::View v = q81::view(act, n);
    const int ng = (n + 255) / 256;
    const __m256i ones = _mm256_set1_epi16(1), idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    for (int k = 0; k < ng; ++k) {
        const int ns = n / 32 - 8 * k < 8 ? n / 32 - 8 * k : 8;
        alignas(32) float d[8] = {}, s[8] = {};
        std::memset(g[k].b8, 0, sizeof g[k].b8);
        for (int j = 0; j < ns; ++j) {
            d[j] = v.d[8 * k + j];
            s[j] = v.s[8 * k + j];
        }
        std::memcpy(g[k].b8, v.b8 + 32 * k, (size_t) ns * 4 * sizeof(int16_t));
        g[k].d8 = _mm256_load_ps(d);
        g[k].s8 = _mm256_load_ps(s);
        // b8 pairs: [s0a, s0b, s1a, s1b, s2a, s2b, s3a, s3b] -> [s0a..s3a | s0b..s3b]
        const __m256i x = _mm256_permutevar8x32_epi32(_mm256_madd_epi16(_mm256_loadu_si256((const __m256i*) g[k].b8), ones), idx);
        const __m256i y =
            _mm256_permutevar8x32_epi32(_mm256_madd_epi16(_mm256_loadu_si256((const __m256i*) (g[k].b8 + 16)), ones), idx);
        g[k].ua = _mm256_permute2x128_si256(x, y, 0x20);
        g[k].ub = _mm256_permute2x128_si256(x, y, 0x31);
    }
}

template <class Ty, int NT>
void rows_t(const uint8_t* w, size_t row_bytes, int n, const TokG* const* tg, const int8_t* const* yq, float* const* out,
            int r0, int r1) {
    const int nsub = n / 32, ng = (nsub + 7) / 8;
    const __m256i ones = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m256 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
        for (int k = 0; k < ng; ++k) {
            const int ns = nsub - 8 * k < 8 ? nsub - 8 * k : 8;
            __m256i wv[8], mv[8];
            typename Ty::P p;
            Ty::dec(row + (size_t) (8 * k) * 32 / Ty::qk * Ty::bytes, ns, wv, mv, p);
            for (int t = 0; t < NT; ++t) {
                const int8_t* y = yq[t] + 256 * k;
                __m256i v[8];
                for (int j = 0; j < 8; ++j) {
                    if (j >= ns) {
                        v[j] = _mm256_setzero_si256();
                        continue;
                    }
                    __m256i yy = _mm256_loadu_si256((const __m256i*) (y + 32 * j));
                    __m256i mag = wv[j];
                    if constexpr (Ty::form == GRID) {
                        yy = _mm256_sub_epi8(_mm256_xor_si256(yy, mv[j]), mv[j]);
                    } else if constexpr (Ty::form == SIGNED) {
                        yy = _mm256_sign_epi8(yy, mag);
                        mag = _mm256_abs_epi8(mag);
                    }
#if defined(LANE_Q81_VNNI)
                    v[j] = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), mag, yy);   // (one step; no pair saturates)
#else
                    v[j] = _mm256_madd_epi16(_mm256_maddubs_epi16(mag, yy), ones);
#endif
                }
                const __m256i e = _mm256_hadd_epi32(_mm256_hadd_epi32(v[0], v[1]), _mm256_hadd_epi32(v[2], v[3]));
                const __m256i f = _mm256_hadd_epi32(_mm256_hadd_epi32(v[4], v[5]), _mm256_hadd_epi32(v[6], v[7]));
                const __m256i A = _mm256_permute2x128_si256(e, f, 0x20), B = _mm256_permute2x128_si256(e, f, 0x31);
                acc[t] = fadd(acc[t], Ty::fin(p, v, A, B, tg[t][k]));
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = hsum(acc[t]);
    }
}

template <class Ty>
void rows_any(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
              int r1) {
    const int ng = (n + 255) / 256;
    thread_local std::vector<TokG> tgb;
    if (tgb.size() < (size_t) nt * ng) tgb.resize((size_t) nt * ng);
    const TokG* tg[64];
    const int8_t* yq[64];
    for (int t = 0; t < nt; ++t) {
        tok_groups(act[t], n, tgb.data() + (size_t) t * ng);
        tg[t] = tgb.data() + (size_t) t * ng;
        yq[t] = q81::view(act[t], n).q;
    }
    for (int t0 = 0; t0 < nt; t0 += 4) {
        switch (nt - t0 < 4 ? nt - t0 : 4) {
            case 1: rows_t<Ty, 1>(w, row_bytes, n, tg + t0, yq + t0, out + t0, r0, r1); break;
            case 2: rows_t<Ty, 2>(w, row_bytes, n, tg + t0, yq + t0, out + t0, r0, r1); break;
            case 3: rows_t<Ty, 3>(w, row_bytes, n, tg + t0, yq + t0, out + t0, r0, r1); break;
            default: rows_t<Ty, 4>(w, row_bytes, n, tg + t0, yq + t0, out + t0, r0, r1); break;
        }
    }
}

// x -> Q81, glm_fast's q8_1_store_warp's steps (see lane_q81.cpp's scalar form)
inline __m256 round_away(__m256 r) {   // roundf
    const __m256 t = _mm256_round_ps(r, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __m256 sgn = _mm256_set1_ps(-0.0f);
    const __m256 f = _mm256_andnot_ps(sgn, _mm256_sub_ps(r, t));
    const __m256 big = _mm256_cmp_ps(f, _mm256_set1_ps(0.5f), _CMP_GE_OQ);
    const __m256 one = _mm256_or_ps(_mm256_and_ps(r, sgn), _mm256_set1_ps(1.0f));
    return _mm256_add_ps(t, _mm256_and_ps(big, one));
}

}  // namespace

#if !defined(LANE_Q81_VNNI)
bool lane_q81_avx2_type(int t) noexcept {
    switch (t) {
        case 16: case 17: case 22: case 18: case 21: case 19: case 29: case 23: case 20: case 39: case 8: case 6:
        case 7: case 42: case 40: case 10: case 11: case 12: case 13: case 14: return true;
        default: return false;
    }
}

#endif

#if defined(LANE_Q81_VNNI)
void lane_q81_vnni_rows(int t, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                        float* const* out, int r0, int r1) 
#else
void lane_q81_avx2_rows(int t, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                        float* const* out, int r0, int r1) 
#endif
{
    switch (t) {
        case 16: rows_any<T16>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 17: rows_any<T17>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 22: rows_any<T22>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 18: rows_any<T18>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 21: rows_any<T21>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 19: rows_any<T19>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 29: rows_any<T29>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 23: rows_any<T23>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 20: rows_any<T20>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 39: rows_any<T39>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 8: rows_any<T8>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 6: rows_any<T6>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 7: rows_any<T7>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 42: rows_any<T42>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 40: rows_any<T40>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 10: rows_any<T10>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 11: rows_any<T11>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 12: rows_any<T12>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 13: rows_any<T13>(w, row_bytes, n, act, nt, out, r0, r1); break;
        case 14: rows_any<T14>(w, row_bytes, n, act, nt, out, r0, r1); break;
        default: break;
    }
}

#if !defined(LANE_Q81_VNNI)
void lane_q81_quant_avx2(const float* x, int64_t n, void* dst) {
    const q81::MView m = q81::mview(dst, n);
    const __m256 sgn = _mm256_set1_ps(-0.0f);
    const __m256 lo = _mm256_set1_ps(-127.0f), hi = _mm256_set1_ps(127.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int64_t b = 0; b < n / 32; ++b) {
        const float* xb = x + 32 * b;
        const __m256 x0 = _mm256_loadu_ps(xb), x1 = _mm256_loadu_ps(xb + 8), x2 = _mm256_loadu_ps(xb + 16),
                     x3 = _mm256_loadu_ps(xb + 24);
        __m256 mx = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sgn, x0), _mm256_andnot_ps(sgn, x1)),
                                  _mm256_max_ps(_mm256_andnot_ps(sgn, x2), _mm256_andnot_ps(sgn, x3)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const float amax = _mm_cvtss_f32(m4);
        // the warp's butterfly: lanes i and i ^ 16, then ^ 8, ^ 4, ^ 2, ^ 1
        const __m256 c = _mm256_add_ps(_mm256_add_ps(x0, x2), _mm256_add_ps(x1, x3));
        __m128 e = _mm_add_ps(_mm256_castps256_ps128(c), _mm256_extractf128_ps(c, 1));
        e = _mm_add_ps(e, _mm_shuffle_ps(e, e, _MM_SHUFFLE(1, 0, 3, 2)));
        e = _mm_add_ss(e, _mm_shuffle_ps(e, e, _MM_SHUFFLE(2, 3, 0, 1)));
        const float sum = _mm_cvtss_f32(e);
        const float d = std::fmin(amax / 127.0f, 65504.0f);
        m.d[b] = h2f(q81::f2h(d));
        m.s[b] = h2f(q81::f2h(std::fmax(-65504.0f, std::fmin(65504.0f, sum))));
        int8_t* q = m.q + 32 * b;
        if (amax == 0.0f) {
            std::memset(q, 0, 32);
            for (int g = 0; g < 4; ++g) m.b8[4 * b + g] = 0;
            continue;
        }
        const __m256 dv = _mm256_set1_ps(d);
        __m256i iv[4];
        const __m256 xs[4] = {x0, x1, x2, x3};
        for (int k = 0; k < 4; ++k)
            iv[k] = _mm256_cvtps_epi32(_mm256_max_ps(lo, _mm256_min_ps(hi, round_away(_mm256_div_ps(xs[k], dv)))));
        const __m256i p = _mm256_permutevar8x32_epi32(
            _mm256_packs_epi16(_mm256_packs_epi32(iv[0], iv[1]), _mm256_packs_epi32(iv[2], iv[3])), perm);
        _mm256_storeu_si256((__m256i*) q, p);
        const __m256i s4 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), p), _mm256_set1_epi16(1));
        alignas(32) int s[8];
        _mm256_store_si256((__m256i*) s, s4);
        for (int g = 0; g < 4; ++g) m.b8[4 * b + g] = (int16_t) (s[2 * g] + s[2 * g + 1]);
    }
}

#endif

}  // namespace strata::kernels::cpu
