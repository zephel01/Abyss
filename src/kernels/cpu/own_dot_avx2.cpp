// src/kernels/cpu/own_dot_avx2.cpp - own_dot_body.hpp built with AVX2 + FMA (x86 only; picked at run time).
#define OD_NS avx2
#include "own_dot_body.hpp"

// ---- the same dot products in AVX2 intrinsics for the types without a faster kernel elsewhere (the i-quants and
// Q4_K / Q5_K have iq_avx2.cpp / iq_avx512*.cpp).  Every kernel keeps own_dot_body.hpp's lanes and float steps
// (own_dot_test checks them against it and against ggml): integer lanes by madd pairs, one fmadd per lane a block.
#include <immintrin.h>

namespace strata::kernels::cpu::avx2 {
namespace {

inline float hsum(__m256 x) {
    __m128 r = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}
inline float f16(ggml_half h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) (unsigned short) h))); }   // (_cvtsh_ss: not in MSVC)
inline uint16_t f2h(float f) { return (uint16_t) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), 0), 0); }   // (_cvtss_sh: not in MSVC)
// signed bytes x . signed bytes y -> 8 int32 lanes (4 bytes each); |x| <= 127
inline __m256i sdot(__m256i x, __m256i y) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x)), _mm256_set1_epi16(1));
}
// unsigned bytes x (<= 127) . signed y
inline __m256i udot(__m256i x, __m256i y) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(x, y), _mm256_set1_epi16(1));
}
// 16 packed nibbles -> 32 bytes: values 0-15 the low nibbles, 16-31 the high ones
inline __m256i nibbles(const uint8_t* q) {
    const __m128i t = _mm_loadu_si128((const __m128i*) q);
    return _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(t, 4), t), _mm256_set1_epi8(0xF));
}
// 32 bits -> 32 bytes 0xFF (bit set) / 0x00
inline __m256i bits32(uint32_t b) {
    const __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32((int) b),
                                          _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                                           2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i sel = _mm256_set1_epi64x((long long) 0x8040201008040201ull);
    return _mm256_cmpeq_epi8(_mm256_and_si256(v, sel), sel);
}
inline __m256i lookup(const int8_t* kv, const uint8_t* q) {
    return _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) kv)), nibbles(q));
}
inline __m256i ld(const void* p) { return _mm256_loadu_si256((const __m256i*) p); }
inline __m256 cvt(__m256i v) { return _mm256_cvtepi32_ps(v); }

float s_q4_0(int n, const void* vx, const void* vy) {
    const block_q4_0* x = (const block_q4_0*) vx;
    const block_q8_0* y = (const block_q8_0*) vy;
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / 32; ++i) {
        const __m256i q = _mm256_sub_epi8(nibbles(x[i].qs), _mm256_set1_epi8(8));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(f16(x[i].d) * f16(y[i].d)), cvt(sdot(q, ld(y[i].qs))), acc);
    }
    return hsum(acc);
}
float s_q5_0(int n, const void* vx, const void* vy) {
    const block_q5_0* x = (const block_q5_0*) vx;
    const block_q8_0* y = (const block_q8_0*) vy;
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / 32; ++i) {
        // nibble | 0x10 where the bit is set, minus 16: nibble - 16 (bit clear) as nibble | 0xF0
        const __m256i hi = _mm256_andnot_si256(bits32(rd32(x[i].qh)), _mm256_set1_epi8((char) 0xF0));
        const __m256i q = _mm256_or_si256(nibbles(x[i].qs), hi);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(f16(x[i].d) * f16(y[i].d)), cvt(sdot(q, ld(y[i].qs))), acc);
    }
    return hsum(acc);
}
float s_q8_0(int n, const void* vx, const void* vy) {
    const block_q8_0* x = (const block_q8_0*) vx;
    const block_q8_0* y = (const block_q8_0*) vy;
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / 32; ++i)
        acc = _mm256_fmadd_ps(_mm256_set1_ps(f16(x[i].d) * f16(y[i].d)), cvt(sdot(ld(x[i].qs), ld(y[i].qs))), acc);
    return hsum(acc);
}
template <bool Q5>
float s_q41(int n, const void* vx, const void* vy) {
    using B = typename std::conditional<Q5, block_q5_1, block_q4_1>::type;
    const B* x = (const B*) vx;
    const block_q8_1* y = (const block_q8_1*) vy;
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.0f;
    for (int i = 0; i < n / 32; ++i) {
        summs = std::fma(f16(x[i].data.data.m), f16(y[i].data.data.s), summs);
        __m256i q = nibbles(x[i].qs);
        if constexpr (Q5) q = _mm256_or_si256(q, _mm256_and_si256(bits32(rd32(x[i].qh)), _mm256_set1_epi8(0x10)));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(f16(x[i].data.data.d) * f16(y[i].data.data.d)), cvt(udot(q, ld(y[i].qs))), acc);
    }
    return hsum(acc) + summs;
}
template <bool MX>
float s_kv(int n, const void* vx, const void* vy) {
    using B = typename std::conditional<MX, block_mxfp4, block_iq4_nl>::type;
    const B* x = (const B*) vx;
    const block_q8_0* y = (const block_q8_0*) vy;
    const int8_t* kv = MX ? kvalues_mxfp4 : kvalues_iq4nl;
    const auto scale = [&](int i) {
        if constexpr (MX) return f16(y[i].d) * e8m0_half(x[i].e);
        else return f16(y[i].d) * f16(x[i].d);
    };
    const int nb = n / 32;
    __m256 a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 1 < nb; i += 2) {
        const __m256i p1 = sdot(lookup(kv, x[i].qs), ld(y[i].qs));
        const __m256i p2 = sdot(lookup(kv, x[i + 1].qs), ld(y[i + 1].qs));
        a1 = _mm256_fmadd_ps(_mm256_set1_ps(scale(i)), cvt(p1), a1);
        a2 = _mm256_fmadd_ps(_mm256_set1_ps(scale(i + 1)), cvt(p2), a2);
    }
    float sumf = hsum(_mm256_add_ps(a1, a2));
    for (; i < nb; ++i) {
        const __m256i p = sdot(lookup(kv, x[i].qs), ld(y[i].qs));
        const __m128i h = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
        const __m128i h2 = _mm_add_epi32(h, _mm_unpackhi_epi64(h, h));
        const int s = _mm_cvtsi128_si32(_mm_add_epi32(h2, _mm_shuffle_epi32(h2, 1)));
        sumf = MX ? std::fma(scale(i), (float) s, sumf) : sumf + scale(i) * (float) s;
    }
    return sumf;
}
// ue4m3_half for every byte, once (the same bits: a table of its results)
const float* ue4m3_tab() {
    static const struct T {
        float v[256];
        T() { for (int i = 0; i < 256; ++i) v[i] = ue4m3_half((uint8_t) i); }
    } t;
    return t.v;
}
float s_nvfp4(int n, const void* vx, const void* vy) {
    const float* ut = ue4m3_tab();
    const block_nvfp4* x = (const block_nvfp4*) vx;
    const block_q8_0* y = (const block_q8_0*) vy;
    const __m128i tab = _mm_loadu_si128((const __m128i*) kvalues_mxfp4);
    const __m128i m = _mm_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / 64; ++i) {
        __m256i w[2];
        for (int h = 0; h < 2; ++h) {   // sub-blocks 2h, 2h+1: each 8 bytes, low nibbles then high nibbles
            const __m128i t = _mm_loadu_si128((const __m128i*) (x[i].qs + 16 * h));
            const __m128i lo = _mm_shuffle_epi8(tab, _mm_and_si128(t, m));
            const __m128i hi = _mm_shuffle_epi8(tab, _mm_and_si128(_mm_srli_epi16(t, 4), m));
            w[h] = _mm256_set_m128i(_mm_unpackhi_epi64(lo, hi), _mm_unpacklo_epi64(lo, hi));
        }
        const float dy0 = f16(y[2 * i].d), dy1 = f16(y[2 * i + 1].d);
        const float s0 = ut[x[i].d[0]] * dy0, s1 = ut[x[i].d[1]] * dy0;
        const float s2 = ut[x[i].d[2]] * dy1, s3 = ut[x[i].d[3]] * dy1;
        acc = _mm256_fmadd_ps(_mm256_set_m128(_mm_set1_ps(s1), _mm_set1_ps(s0)), cvt(sdot(w[0], ld(y[2 * i].qs))), acc);
        acc = _mm256_fmadd_ps(_mm256_set_m128(_mm_set1_ps(s3), _mm_set1_ps(s2)), cvt(sdot(w[1], ld(y[2 * i + 1].qs))), acc);
    }
    return hsum(acc);
}
inline int hsum_i(__m256i p) {
    const __m128i h = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
    const __m128i h2 = _mm_add_epi32(h, _mm_unpackhi_epi64(h, h));
    return _mm_cvtsi128_si32(_mm_add_epi32(h2, _mm_shuffle_epi32(h2, 1)));
}
float s_q2_0(int n, const void* vx, const void* vy) {
    const block_q2_0* x = (const block_q2_0*) vx;
    const block_q8_0* y = (const block_q8_0*) vy;
    const __m128i m3 = _mm_set1_epi8(3);
    float sumf = 0.0f;
    for (int i = 0; i < n / 64; ++i) {
        float sumi = 0.0f;
        for (int k = 0; k < 2; ++k) {
            // 8 bytes, 4 codes each (value 4b + j at bits 2j of byte b) -> 32 codes in value order, minus 1
            const __m128i b = _mm_loadl_epi64((const __m128i*) (x[i].qs + 8 * k));
            const __m128i t0 = _mm_and_si128(b, m3), t1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
            const __m128i t2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3), t3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
            const __m128i a = _mm_unpacklo_epi8(t0, t1), c = _mm_unpacklo_epi8(t2, t3);
            const __m256i q = _mm256_sub_epi8(_mm256_set_m128i(_mm_unpackhi_epi16(a, c), _mm_unpacklo_epi16(a, c)),
                                              _mm256_set1_epi8(1));
            const int s = hsum_i(sdot(q, ld(y[2 * i + k].qs)));
            sumi = std::fma(f16(y[2 * i + k].d), (float) s, sumi);
        }
        sumf = sumf + f16(x[i].d) * sumi;
    }
    return sumf;
}
// two 16-value scales of a 32-value chunk as int16 lanes (low half: values 0-15, high half: 16-31)
inline __m256i sc2(int a, int b) { return _mm256_set_m128i(_mm_set1_epi16((short) b), _mm_set1_epi16((short) a)); }
// the same from a vector of 16 int16 scales (scale 2c and 2c + 1 the 32-bit element c): chunk c's scale vector
inline __m256i pair_of(__m256i s16, int c) {
    const __m256i p = _mm256_permutevar8x32_epi32(s16, _mm256_set1_epi32(c));
    return _mm256_shuffle_epi8(p, _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                                                   2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3));
}

float s_q2_K(int n, const void* vx, const void* vy) {
    const block_q2_K* x = (const block_q2_K*) vx;
    const block_q8_K* y = (const block_q8_K*) vy;
    const __m256i m3 = _mm256_set1_epi8(3);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / 256; ++i) {
        const float d = y[i].d * f16(x[i].data.data.d);
        const float dmin = -y[i].d * f16(x[i].data.data.dmin);
        const __m128i sm = _mm_loadu_si128((const __m128i*) x[i].scales);
        const __m256i mins = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(sm, 4), _mm_set1_epi8(0xF)));
        const __m256i scl = _mm256_cvtepu8_epi16(_mm_and_si128(sm, _mm_set1_epi8(0xF)));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(dmin), cvt(_mm256_madd_epi16(mins, ld(y[i].bsums))), acc);
        __m256i sumi = _mm256_setzero_si256();
        for (int h = 0; h < 2; ++h) {
            const __m256i bits = ld(x[i].qs + 32 * h);
            for (int k = 0; k < 4; ++k) {
                const __m256i q = _mm256_and_si256(_mm256_srli_epi16(bits, 2 * k), m3);
                const __m256i p = _mm256_maddubs_epi16(q, ld(y[i].qs + 128 * h + 32 * k));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p, pair_of(scl, 4 * h + k)));
            }
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), cvt(sumi), acc);
    }
    return hsum(acc);
}
float s_q3_K(int n, const void* vx, const void* vy) {
    const block_q3_K* x = (const block_q3_K*) vx;
    const block_q8_K* y = (const block_q8_K*) vy;
    const __m256i m3 = _mm256_set1_epi8(3);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / 256; ++i) {
        const float d = y[i].d * f16(x[i].d);
        // the 16 6-bit scales: low nibbles from bytes 0-7 (scales 0-7) / their high nibbles (8-15), the top 2 bits
        // from bytes 8-11 (scale j: byte 8 + j % 4, bits 2 (j / 4))
        const __m128i b8 = _mm_loadl_epi64((const __m128i*) x[i].scales);
        const __m128i lo = _mm_and_si128(_mm_unpacklo_epi64(b8, _mm_srli_epi16(b8, 4)), _mm_set1_epi8(0xF));
        const uint32_t hb = rd32(x[i].scales + 8);
        const __m128i hw = _mm_setr_epi32((int) hb, (int) (hb >> 2), (int) (hb >> 4), (int) (hb >> 6));
        const __m128i hi = _mm_slli_epi16(_mm_and_si128(hw, _mm_set1_epi8(3)), 4);
        const __m256i sc16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_or_si128(lo, hi)), _mm256_set1_epi16(32));
        const __m256i hm = ld(x[i].hmask);
        __m256i sumi = _mm256_setzero_si256();
        for (int h = 0; h < 2; ++h) {
            const __m256i bits = ld(x[i].qs + 32 * h);
            for (int k = 0; k < 4; ++k) {
                const __m256i bit = _mm256_set1_epi8((char) (1 << (4 * h + k)));
                const __m256i clear = _mm256_cmpeq_epi8(_mm256_and_si256(hm, bit), _mm256_setzero_si256());
                const __m256i q = _mm256_sub_epi8(_mm256_and_si256(_mm256_srli_epi16(bits, 2 * k), m3),
                                                  _mm256_and_si256(clear, _mm256_set1_epi8(4)));
                const __m256i y8 = ld(y[i].qs + 128 * h + 32 * k);
                const __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(q, q), _mm256_sign_epi8(y8, q));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p, pair_of(sc16, 4 * h + k)));
            }
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), cvt(sumi), acc);
    }
    return hsum(acc);
}
float s_q6_K(int n, const void* vx, const void* vy) {
    const block_q6_K* x = (const block_q6_K*) vx;
    const block_q8_K* y = (const block_q8_K*) vy;
    const __m256i m4 = _mm256_set1_epi8(0xF), m3 = _mm256_set1_epi8(3);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / 256; ++i) {
        const float d = y[i].d * f16(x[i].d);
        const __m256i sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) x[i].scales));
        const __m256i off = _mm256_slli_epi32(_mm256_madd_epi16(ld(y[i].bsums), sc16), 5);
        __m256i sumi = _mm256_setzero_si256();
        for (int h = 0; h < 2; ++h) {
            const __m256i l0 = ld(x[i].ql + 64 * h), l1 = ld(x[i].ql + 64 * h + 32), qh = ld(x[i].qh + 32 * h);
            for (int k = 0; k < 4; ++k) {
                const __m256i l = k & 1 ? l1 : l0;
                const __m256i lo = k < 2 ? _mm256_and_si256(l, m4) : _mm256_and_si256(_mm256_srli_epi16(l, 4), m4);
                const __m256i hb = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * k), m3), 4);
                const __m256i q = _mm256_or_si256(lo, hb);
                const __m256i p = _mm256_maddubs_epi16(q, ld(y[i].qs + 128 * h + 32 * k));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p, pair_of(sc16, 4 * h + k)));
            }
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), cvt(_mm256_sub_epi32(sumi, off)), acc);
    }
    return hsum(acc);
}

// ---- 256-value K-quants and i-quants against Q8_K
// 32 sign bits (bit v: value v negative) -> 32 bytes of -1 / +1
inline __m256i sign_bytes(uint32_t b) { return _mm256_or_si256(bits32(b), _mm256_set1_epi8(1)); }
// grid bytes (unsigned) x the activation with the weight's signs, x a 16-bit scale per half -> 8 int32 lanes
inline __m256i gdot(__m256i grid, __m256i sgn, __m256i y8, __m256i sc) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(grid, _mm256_sign_epi8(y8, sgn)), sc);
}
inline __m256i sc1(int a) { return _mm256_set1_epi16((short) a); }
// a sign byte (bit j: value j negative) as 8 bytes of -1 / +1, for every byte value; ks: for the 7-bit ksigns index
struct SignTab {
    uint64_t raw[256], ks[128];
    SignTab() {
        for (int b = 0; b < 256; ++b) {
            uint64_t v = 0;
            for (int j = 0; j < 8; ++j) v |= (uint64_t) ((b >> j) & 1 ? 0xFF : 0x01) << (8 * j);
            raw[b] = v;
        }
        for (int k = 0; k < 128; ++k) ks[k] = raw[ksigns_iq2xs[k]];
    }
};
const SignTab& signtab() {
    static const SignTab t;
    return t;
}
inline __m256i sgn4(const uint64_t* t, int a, int b, int c, int d) {
    return _mm256_set_epi64x((long long) t[d], (long long) t[c], (long long) t[b], (long long) t[a]);
}
// IQ2_XS / IQ2_S: the 8 scale bytes (two 4-bit scales each, 2s + 1) as 32-bit pairs (lo, hi) of 16-bit values
inline __m256i iq2_scale_pairs(const uint8_t* scales) {
    const __m128i b = _mm_loadl_epi64((const __m128i*) scales);
    const __m128i m = _mm_set1_epi8(0xF);
    const __m128i lh = _mm_unpacklo_epi8(_mm_and_si128(b, m), _mm_and_si128(_mm_srli_epi16(b, 4), m));
    const __m256i s16 = _mm256_cvtepu8_epi16(lh);
    return _mm256_add_epi16(_mm256_add_epi16(s16, s16), _mm256_set1_epi16(1));
}
// the pair of sub-block ib as a scale vector: low 128 bits its first scale, high its second
inline __m256i iq2_scale_of(__m256i pairs, int ib) {
    const __m256i p = _mm256_permutevar8x32_epi32(pairs, _mm256_set1_epi32(ib));
    return _mm256_shuffle_epi8(p, _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                                                   2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3));
}

template <bool Q5>
float s_q45_K(int n, const void* vx, const void* vy) {
    using B = typename std::conditional<Q5, block_q5_K, block_q4_K>::type;
    const B* x = (const B*) vx;
    const block_q8_K* y = (const block_q8_K*) vy;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    __m128 accm4 = _mm_setzero_ps();
    float summs = 0.0f;
    for (int i = 0; i < n / 256; ++i) {
        const float d = y[i].d * f16(x[i].data.data.d);
        const float dmin = -y[i].d * f16(x[i].data.data.dmin);
        // the 6-bit scales and mins of the 12 scale bytes, four at a time in 32-bit words
        const uint32_t a = rd32(x[i].scales), b = rd32(x[i].scales + 4), c = rd32(x[i].scales + 8);
        const uint32_t sw[2] = {a & 0x3F3F3F3Fu, (c & 0x0F0F0F0Fu) | (((a >> 6) & 0x03030303u) << 4)};
        const uint32_t mw[2] = {b & 0x3F3F3F3Fu, ((c >> 4) & 0x0F0F0F0Fu) | (((b >> 6) & 0x03030303u) << 4)};
        uint8_t sc[8];
        std::memcpy(sc, sw, 8);
        // the mins x the activation's 32-value sums: 4 int32 lanes (pairs of 32-value chunks)
        const __m128i bs = _mm_hadd_epi16(_mm_loadu_si128((const __m128i*) y[i].bsums),
                                          _mm_loadu_si128((const __m128i*) (y[i].bsums + 8)));
        const __m128i prod = _mm_madd_epi16(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) mw)), bs);
        if constexpr (Q5) {
            const __m128i t2 = _mm_add_epi32(prod, _mm_unpackhi_epi64(prod, prod));
            const int32_t t = _mm_cvtsi128_si32(_mm_add_epi32(t2, _mm_shuffle_epi32(t2, 1)));
            summs = std::fma(dmin, (float) t, summs);
        } else {
            accm4 = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), accm4);
        }
        __m256i qh;
        if constexpr (Q5) qh = ld(x[i].qh);
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            const __m256i b = ld(x[i].qs + 32 * j);
            __m256i lo = _mm256_and_si256(b, m4), hi = _mm256_and_si256(_mm256_srli_epi16(b, 4), m4);
            if constexpr (Q5) {
                const __m256i b0 = _mm256_set1_epi8((char) (1 << (2 * j))), b1 = _mm256_set1_epi8((char) (2 << (2 * j)));
                const __m256i x10 = _mm256_set1_epi8(0x10);
                lo = _mm256_or_si256(lo, _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, b0), b0), x10));
                hi = _mm256_or_si256(hi, _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, b1), b1), x10));
            }
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(lo, ld(y[i].qs + 64 * j)), sc1(sc[2 * j])));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(hi, ld(y[i].qs + 64 * j + 32)), sc1(sc[2 * j + 1])));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), cvt(sumi), acc);
    }
    if constexpr (Q5) return hsum(acc) + summs;
    float accm[4];
    _mm_storeu_ps(accm, accm4);
    return hsum(acc) + ((accm[0] + accm[2]) + (accm[1] + accm[3]));
}

template <int T>
float s_iq(int n, const void* vx, const void* vy) {
    const block_q8_K* y = (const block_q8_K*) vy;
    const SignTab& st = signtab();
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / 256; ++i) {
        float dx;
        __m256i sumi = _mm256_setzero_si256();
        if constexpr (T == 16) {   // IQ2_XXS
            const block_iq2_xxs& x = ((const block_iq2_xxs*) vx)[i];
            dx = f16(x.d);
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t* q = (const uint8_t*) (x.qs + 4 * ib);
                const uint32_t s = rd32(q + 4);
                const __m256i g = _mm256_set_epi64x((long long) iq2xxs_grid[q[3]], (long long) iq2xxs_grid[q[2]],
                                                    (long long) iq2xxs_grid[q[1]], (long long) iq2xxs_grid[q[0]]);
                const __m256i sv = sgn4(st.ks, s & 127, (s >> 7) & 127, (s >> 14) & 127, (s >> 21) & 127);
                sumi = _mm256_add_epi32(sumi, gdot(g, sv, ld(y[i].qs + 32 * ib), sc1(2 * (int) (s >> 28) + 1)));
            }
        } else if constexpr (T == 17) {   // IQ2_XS
            const block_iq2_xs& x = ((const block_iq2_xs*) vx)[i];
            dx = f16(x.d);
            const __m256i scp = iq2_scale_pairs(x.scales);
            for (int ib = 0; ib < 8; ++ib) {
                const uint16_t* q = x.qs + 4 * ib;
                const __m256i g = _mm256_set_epi64x((long long) iq2xs_grid[q[3] & 511], (long long) iq2xs_grid[q[2] & 511],
                                                    (long long) iq2xs_grid[q[1] & 511], (long long) iq2xs_grid[q[0] & 511]);
                const __m256i sv = sgn4(st.ks, q[0] >> 9, q[1] >> 9, q[2] >> 9, q[3] >> 9);
                sumi = _mm256_add_epi32(sumi, gdot(g, sv, ld(y[i].qs + 32 * ib), iq2_scale_of(scp, ib)));
            }
        } else if constexpr (T == 22) {   // IQ2_S
            const block_iq2_s& x = ((const block_iq2_s*) vx)[i];
            dx = f16(x.d);
            const uint8_t* signs = x.qs + 32;
            const __m256i scp = iq2_scale_pairs(x.scales);
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t* q = x.qs + 4 * ib;
                const int h = x.qh[ib];
                const __m256i g = _mm256_set_epi64x((long long) iq2s_grid[q[3] | ((h << 2) & 0x300)],
                                                    (long long) iq2s_grid[q[2] | ((h << 4) & 0x300)],
                                                    (long long) iq2s_grid[q[1] | ((h << 6) & 0x300)],
                                                    (long long) iq2s_grid[q[0] | ((h << 8) & 0x300)]);
                sumi = _mm256_add_epi32(sumi, gdot(g, sign_bytes(rd32(signs + 4 * ib)), ld(y[i].qs + 32 * ib),
                                                   iq2_scale_of(scp, ib)));
            }
        } else if constexpr (T == 18) {   // IQ3_XXS
            const block_iq3_xxs& x = ((const block_iq3_xxs*) vx)[i];
            dx = f16(x.d);
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t* q = x.qs + 8 * ib;
                const uint32_t s = rd32(x.qs + 64 + 4 * ib);
                const __m256i g = _mm256_set_epi32((int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]],
                                                   (int) iq3xxs_grid[q[4]], (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]],
                                                   (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
                const __m256i sv = sgn4(st.ks, s & 127, (s >> 7) & 127, (s >> 14) & 127, (s >> 21) & 127);
                sumi = _mm256_add_epi32(sumi, gdot(g, sv, ld(y[i].qs + 32 * ib), sc1(2 * (int) (s >> 28) + 1)));
            }
        } else if constexpr (T == 21) {   // IQ3_S
            const block_iq3_s& x = ((const block_iq3_s*) vx)[i];
            dx = f16(x.d);
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t* q = x.qs + 8 * ib;
                const int h = x.qh[ib];
                int gi[8];
                for (int k = 0; k < 8; ++k) gi[k] = (int) iq3s_grid[q[k] | ((h << (8 - k)) & 256)];
                const __m256i g = _mm256_set_epi32(gi[7], gi[6], gi[5], gi[4], gi[3], gi[2], gi[1], gi[0]);
                const int ls = 2 * (ib % 2 == 0 ? x.scales[ib / 2] & 0xF : x.scales[ib / 2] >> 4) + 1;
                sumi = _mm256_add_epi32(sumi, gdot(g, sign_bytes(rd32(x.signs + 4 * ib)), ld(y[i].qs + 32 * ib), sc1(ls)));
            }
        } else {   // IQ4_XS
            const block_iq4_xs& x = ((const block_iq4_xs*) vx)[i];
            dx = f16(x.d);
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = (((x.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((x.scales_h >> (2 * ib)) & 3) << 4)) - 32;
                const __m256i w = lookup(kvalues_iq4nl, x.qs + 16 * ib);
                const __m256i y8 = ld(y[i].qs + 32 * ib);
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(y8, w)), sc1(ls)));
            }
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[i].d), cvt(sumi), acc);
    }
    const float h = hsum(acc);
    if constexpr (T == 16 || T == 17 || T == 22) return 0.125f * h;
    else if constexpr (T == 18) return 0.25f * h;
    else return h;
}
float s_iq1_s(int n, const void* vx, const void* vy) {
    const block_iq1_s* x = (const block_iq1_s*) vx;
    const block_q8_K* y = (const block_q8_K*) vy;
    __m256 acc = _mm256_setzero_ps();
    float accum1 = 0.0f;
    for (int i = 0; i < n / 256; ++i) {
        int32_t sumi1 = 0;
        __m256i sumi = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ++ib) {
            const uint16_t h = x[i].qh[ib];
            const int ls = 2 * ((h >> 12) & 7) + 1;
            const uint8_t* q = x[i].qs + 4 * ib;
            const __m256i g = _mm256_set_epi64x((long long) iq1s_grid[q[3] | (((h >> 9) & 7) << 8)],
                                                (long long) iq1s_grid[q[2] | (((h >> 6) & 7) << 8)],
                                                (long long) iq1s_grid[q[1] | (((h >> 3) & 7) << 8)],
                                                (long long) iq1s_grid[q[0] | ((h & 7) << 8)]);
            const __m256i y8 = ld(y[i].qs + 32 * ib);
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), _mm256_sign_epi8(y8, g)), sc1(ls)));
            sumi1 += (y[i].bsums[2 * ib] + y[i].bsums[2 * ib + 1]) * (h & 0x8000 ? -1 : 1) * ls;
        }
        const float d = y[i].d * f16(x[i].d);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), cvt(sumi), acc);
        accum1 = std::fma(d, (float) sumi1, accum1);
    }
    return hsum(acc) + IQ1S_DELTA * accum1;
}
float s_iq1_m(int n, const void* vx, const void* vy) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const block_q8_K* y = (const block_q8_K*) vy;
    __m256 a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps();
    const __m256i one = _mm256_set1_epi8(1);
    for (int i = 0; i < n / 256; ++i) {
        uint16_t sc[4];
        std::memcpy(sc, x[i].scales, 8);
        const uint16_t sf = (uint16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
        __m256i s1 = _mm256_setzero_si256(), s2 = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t* qs = x[i].qs + 4 * ib;
            const uint8_t* qh = x[i].qh + 2 * ib;
            const __m256i g = _mm256_set_epi64x((long long) iq1s_grid[qs[3] | ((qh[1] << 4) & 0x700)],
                                                (long long) iq1s_grid[qs[2] | ((qh[1] << 8) & 0x700)],
                                                (long long) iq1s_grid[qs[1] | ((qh[0] << 4) & 0x700)],
                                                (long long) iq1s_grid[qs[0] | ((qh[0] << 8) & 0x700)]);
            const long long neg = -1, pos = 0x0101010101010101ll;
            const __m256i dl = _mm256_set_epi64x(qh[1] & 0x80 ? neg : pos, qh[1] & 0x08 ? neg : pos,
                                                 qh[0] & 0x80 ? neg : pos, qh[0] & 0x08 ? neg : pos);
            const __m256i y8 = ld(y[i].qs + 32 * ib);
            const __m256i scv = sc2(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7) + 1, 2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1);
            s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_maddubs_epi16(one, _mm256_sign_epi8(y8, g)), scv));
            s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(_mm256_maddubs_epi16(one, _mm256_sign_epi8(y8, dl)), scv));
        }
        const __m256 d = _mm256_set1_ps(y[i].d * f16((ggml_half) sf));
        a1 = _mm256_fmadd_ps(d, cvt(s1), a1);
        a2 = _mm256_fmadd_ps(d, cvt(s2), a2);
    }
    return hsum(a1) + IQ1M_DELTA * hsum(a2);
}

}  // namespace

// NaN: no intrinsics kernel here for the type (own_dot_fast then takes dot())
float dot_simd(int type, int n, const void* w, const void* a) {
    switch (type) {
        case 2: return s_q4_0(n, w, a);
        case 3: return s_q41<false>(n, w, a);
        case 6: return s_q5_0(n, w, a);
        case 7: return s_q41<true>(n, w, a);
        case 8: return s_q8_0(n, w, a);
        case 10: return s_q2_K(n, w, a);
        case 11: return s_q3_K(n, w, a);
        case 12: return s_q45_K<false>(n, w, a);
        case 13: return s_q45_K<true>(n, w, a);
        case 14: return s_q6_K(n, w, a);
        case 16: return s_iq<16>(n, w, a);
        case 17: return s_iq<17>(n, w, a);
        case 18: return s_iq<18>(n, w, a);
        case 19: return s_iq1_s(n, w, a);
        case 21: return s_iq<21>(n, w, a);
        case 22: return s_iq<22>(n, w, a);
        case 23: return s_iq<23>(n, w, a);
        case 20: return s_kv<false>(n, w, a);
        case 29: return s_iq1_m(n, w, a);
        case 39: return s_kv<true>(n, w, a);
        case 40: return s_nvfp4(n, w, a);
        case 42: return s_q2_0(n, w, a);
        default: return NAN;
    }
}
bool has_simd(int type) {
    switch (type) {
        case 2: case 3: case 6: case 7: case 8: case 10: case 11: case 12: case 13: case 14: case 16: case 17: case 18:
        case 19: case 20: case 21: case 22: case 23: case 29: case 39: case 40: case 42:
            return true;
        default: return false;
    }
}

// ---- the activation quantizers (own_dot.cpp's quant_q8_0 / quant_q8_K, the same bits): the rounding to nearest
// even of a value below 2^22 is _mm256_round_ps's, the scale steps the same single multiplies and divides
namespace {
inline float hmax(__m256 v) {
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_movehdup_ps(m));
    return _mm_cvtss_f32(m);
}
// 32 int32 (four vectors) -> 32 int8 in order
inline __m256i pack32(__m256i a, __m256i b, __m256i c, __m256i d) {
    const __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(a, b), _mm256_packs_epi32(c, d));
    return _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}
inline __m256i rnd(__m256 v) { return _mm256_cvtps_epi32(_mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); }

template <bool Q81>
void q8_0(const float* x, void* dst, int64_t n) {
    uint8_t* o = (uint8_t*) dst;
    const size_t bb = Q81 ? sizeof(block_q8_1) : sizeof(block_q8_0);
    const __m256 sm = _mm256_set1_ps(-0.0f);
    for (int64_t i = 0; i < n / 32; ++i, x += 32, o += bb) {
        const __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8), v2 = _mm256_loadu_ps(x + 16), v3 = _mm256_loadu_ps(x + 24);
        const __m256 a = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sm, v0), _mm256_andnot_ps(sm, v1)),
                                       _mm256_max_ps(_mm256_andnot_ps(sm, v2), _mm256_andnot_ps(sm, v3)));
        const float mx = hmax(a);
        const float d = mx / 127.0f;
        const float id = mx != 0.0f ? 127.0f / mx : 0.0f;
        const __m256 m = _mm256_set1_ps(id);
        const __m256i i0 = rnd(_mm256_mul_ps(v0, m)), i1 = rnd(_mm256_mul_ps(v1, m)), i2 = rnd(_mm256_mul_ps(v2, m)),
                      i3 = rnd(_mm256_mul_ps(v3, m));
        _mm256_storeu_si256((__m256i*) (o + (Q81 ? 4 : 2)), pack32(i0, i1, i2, i3));
        const uint16_t dh = f2h(d);
        std::memcpy(o, &dh, 2);
        if (Q81) {
            const __m256i s4 = _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3));
            const int sum = hsum_i(s4);
            const uint16_t sh = f2h(d * (float) sum);
            std::memcpy(o + 2, &sh, 2);
        }
    }
}
void q8_K(const float* x, void* dst, int64_t n) {
    block_q8_K* y = (block_q8_K*) dst;
    const __m256 sm = _mm256_set1_ps(-0.0f);
    for (int64_t i = 0; i < n / 256; ++i, x += 256) {
        __m256 a = _mm256_setzero_ps();
        for (int j = 0; j < 256; j += 8) a = _mm256_max_ps(a, _mm256_andnot_ps(sm, _mm256_loadu_ps(x + j)));
        const float amax = hmax(a);
        if (!(amax > 0.0f)) {   // all zero (the reference's first-larger scan keeps max 0)
            std::memset(&y[i], 0, sizeof(block_q8_K));
            continue;
        }
        float mx = 0.0f;
        for (int j = 0; j < 256; ++j)
            if (std::fabs(x[j]) == amax) { mx = x[j]; break; }
        const float iscale = -127.0f / mx;
        const __m256 m = _mm256_set1_ps(iscale);
        const __m256i cap = _mm256_set1_epi32(127);
        for (int j = 0; j < 256; j += 32) {
            __m256i q[4];
            for (int k = 0; k < 4; ++k) q[k] = _mm256_min_epi32(rnd(_mm256_mul_ps(m, _mm256_loadu_ps(x + j + 8 * k))), cap);
            _mm256_storeu_si256((__m256i*) (y[i].qs + j), pack32(q[0], q[1], q[2], q[3]));
        }
        const __m128i bias = _mm_set1_epi8((char) 0x80);
        for (int j = 0; j < 16; ++j) {
            const __m128i s = _mm_sad_epu8(_mm_xor_si128(_mm_loadu_si128((const __m128i*) (y[i].qs + 16 * j)), bias), _mm_setzero_si128());
            y[i].bsums[j] = (int16_t) (_mm_cvtsi128_si32(s) + _mm_extract_epi16(s, 4) - 16 * 128);
        }
        y[i].d = 1 / iscale;
    }
}
}  // namespace

bool quant_simd(int act_type, const float* x, void* dst, int64_t n) {
    switch (act_type) {
        case 8: q8_0<false>(x, dst, n); return true;
        case 9: q8_0<true>(x, dst, n); return true;
        case 15: q8_K(x, dst, n); return true;
        default: return false;
    }
}

}  // namespace strata::kernels::cpu::avx2
