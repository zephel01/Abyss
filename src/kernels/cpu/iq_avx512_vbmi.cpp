// src/kernels/cpu/iq_avx512_vbmi.cpp - E-3: IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS rows (the CPU lane's experts: a decode
// step's one token, a verify window's or a prompt chunk's several) in 512-bit lanes, the grids looked up in registers;
// IQ1_S's 2048-entry grid gathered (vpgatherqq, 8 entries a 64-value chunk).
//
// ggml-cpu's AVX2 dots rebuild every weight from memory lookups: 16 grid words assembled into vectors for 128 IQ2_XS
// values, most of a row's time on a Zen 5, which runs these rows at ~5 GB/s a core (16 cores ~75 of the ~113 GB/s it
// reads).  The IQ2 grids' bytes are only 8 / 25 / 43 and iq3xxs_grid's 4, 12, ..., 52, 62, so an entry is a 16-bit
// code (2 bits a value; IQ3_XXS 3 bits, 12 used): the 256 IQ2_XXS / IQ3_XXS codes fill 8 registers, the 512 IQ2_XS
// codes 16, the 1024 IQ2_S codes 32 (read from L1), and vpermt2w looks 32 entries up at once.  vpmultishiftqb + vpshufb turn the codes into the grid bytes, vpermi2b turns
// the 7-bit sign indices into ksigns_iq2xs's bytes (a 64-bit mask a 64-value chunk), and maddubs + vpdpwssd take
// ggml's integer steps; a block is decoded once for up to 4 tokens.  Each token's result is BIT-IDENTICAL to ggml's dot: the int32 lanes hold the same products (16 lanes
// folded to ggml's 8, as in iq_avx512.cpp), the floats take ggml's steps (one fmadd a block with d_x * d_y, its
// hsum_float_8, the format's constant); native_expert.cpp checks it against ggml-cpu at every start.
//
// IQ1_S (DeepSeek V4.1's unsloth quants, GLM's IQ1_S gate/up): 4 gathers a block from iq1s_grid's 2048 qwords, |v|
// and the negative values' mask from the gathered bytes, ggml's integer steps as above, and its delta term (the bsums
// against each 32-value group's signed scale, one scalar a block) with ggml's float steps.  A block decoded once
// pays off for several tokens only (L2-resident rows, a Zen 5 core, ns a block a token: 1 token 12.2 against ggml's
// 9.4, 2 tokens 7.3 / 9.5, 4 tokens 4.8 / 9.5); native_expert.cpp keeps ggml's dot for one token.
//
// AVX-512 F / BW / VL / VNNI / VBMI (cpu_avx512_ok(): Ice Lake, Zen 4 and later).  This file is built with those
// flags: nothing in it runs before that check (no initializers at namespace scope; the tables are built on the
// first call).
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/lane_q81.hpp"
#include "lane_q81_impl.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::kernels::cpu {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

// ggml's hsum_float_8: the same additions in the same order
inline float hsum_ggml(__m256 x) {
    __m128 r = _mm256_extractf128_ps(x, 1);
    r = _mm_add_ps(r, _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// a block's 16 int32 lanes (values 4m..4m+3 of every 64-value chunk) -> ggml's 8 (values 4k..4k+3 of every 32-value
// chunk): lane k + lane k + 8, exact
inline __m256i fold(__m512i a) { return _mm256_add_epi32(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1)); }

// 8 int32 lanes' sum, exact
inline int hsum_i32(__m256i v) {
    __m128i r = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 0x4e));
    r = _mm_add_epi32(r, _mm_shuffle_epi32(r, 0xb1));
    return _mm_cvtsi128_si32(r);
}

// IQ1_S's delta term: ggml's `accum1 += d * sumi1` is one scalar line that GCC and Clang contract into an fmadd and
// MSVC does not (as Q5_K's min term, iq_avx512.cpp); native_expert.cpp's check picks the form this build's ggml-cpu
// took (this file is built without contraction)
std::atomic<bool> iq1s_fused{true};
// the lane's newer decodes (iq512v_set_lane_kernels; each env, when set, wins): IQ2_S by gathering grid qwords,
// IQ3_XXS by byte planes, IQ4_NL in 512-bit lanes (else AVX2 / VNNI) - all the same bits as the older forms
std::atomic<bool> iq2s_gather{false}, iq3xxs_planes{false}, iq4nl_512{false};

alignas(64) uint16_t code2xxs[256];  // iq2xxs_grid as 2-bit codes, value v at bits 2v: 0 = 8, 1 = 25, 2 = 43
alignas(64) uint16_t code2xs[512];   // iq2xs_grid, the same
alignas(64) uint16_t code2s[1024];   // iq2s_grid, the same
alignas(64) uint16_t code3xxs[256];  // iq3xxs_grid as 3-bit codes, value v at bits 3v: 4, 12, ..., 52, 62
alignas(64) uint8_t plane3xxs[4][256];   // iq3xxs_grid's byte v of each entry (rows_iq3xxs_q81)
alignas(64) uint8_t ksigns[128];     // ksigns_iq2xs: a 7-bit sign index with its parity as bit 7

// an IQ2 grid's entries as codes (false: a byte that is not 8 / 25 / 43 - never in ggml's grids)
bool codes2(const uint64_t* grid, int n, uint16_t* code) {
    for (int e = 0; e < n; ++e) {
        uint16_t c = 0;
        for (int v = 0; v < 8; ++v) {
            const uint8_t b = (uint8_t) (grid[e] >> (8 * v));
            const int k = b == 8 ? 0 : b == 25 ? 1 : b == 43 ? 2 : -1;
            if (k < 0) return false;
            c = (uint16_t) (c | (k << (2 * v)));
        }
        code[e] = c;
    }
    return true;
}

bool build_codes() {
    if (!codes2(iq2xxs_grid, 256, code2xxs) || !codes2(iq2xs_grid, 512, code2xs) || !codes2(iq2s_grid, 1024, code2s))
        return false;
    static const uint8_t lv3[8] = {4, 12, 20, 28, 36, 44, 52, 62};
    for (int e = 0; e < 256; ++e) {
        uint16_t c = 0;
        for (int v = 0; v < 4; ++v) {
            const uint8_t b = (uint8_t) (iq3xxs_grid[e] >> (8 * v));
            int k = -1;
            for (int l = 0; l < 8; ++l)
                if (lv3[l] == b) k = l;
            if (k < 0) return false;
            c = (uint16_t) (c | (k << (3 * v)));
        }
        code3xxs[e] = c;
        for (int v = 0; v < 4; ++v) plane3xxs[v][e] = (uint8_t) (iq3xxs_grid[e] >> (8 * v));
    }
    std::memcpy(ksigns, ksigns_iq2xs, sizeof ksigns);
    return true;
}

// the codes of 32 entries (idx: 9-bit word indices) from code2xs's 16 registers: pair k holds entries 64k..64k+63
// (vpermt2w reads index bits 0-5), bits 6-8 pick the pair
inline __m512i look512(__m512i idx, const __m512i* T) {
    __m512i r0 = _mm512_permutex2var_epi16(T[0], idx, T[1]);
    __m512i r1 = _mm512_permutex2var_epi16(T[2], idx, T[3]);
    __m512i r2 = _mm512_permutex2var_epi16(T[4], idx, T[5]);
    __m512i r3 = _mm512_permutex2var_epi16(T[6], idx, T[7]);
    __m512i r4 = _mm512_permutex2var_epi16(T[8], idx, T[9]);
    __m512i r5 = _mm512_permutex2var_epi16(T[10], idx, T[11]);
    __m512i r6 = _mm512_permutex2var_epi16(T[12], idx, T[13]);
    __m512i r7 = _mm512_permutex2var_epi16(T[14], idx, T[15]);
    const __mmask32 b6 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(64));
    const __mmask32 b7 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(128));
    const __mmask32 b8 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(256));
    r0 = _mm512_mask_mov_epi16(r0, b6, r1);
    r2 = _mm512_mask_mov_epi16(r2, b6, r3);
    r4 = _mm512_mask_mov_epi16(r4, b6, r5);
    r6 = _mm512_mask_mov_epi16(r6, b6, r7);
    r0 = _mm512_mask_mov_epi16(r0, b7, r2);
    r4 = _mm512_mask_mov_epi16(r4, b7, r6);
    return _mm512_mask_mov_epi16(r0, b8, r4);
}

// ... of 32 IQ2_S entries (10-bit indices) from code2s's 32 registers' worth (bits 6-9 pick the pair)
inline __m512i look1024(__m512i idx, const __m512i* T) {
    __m512i r[16];
    for (int k = 0; k < 16; ++k) r[k] = _mm512_permutex2var_epi16(T[2 * k], idx, T[2 * k + 1]);
    for (int bit = 6, step = 1; bit <= 9; ++bit, step *= 2) {
        const __mmask32 m = _mm512_test_epi16_mask(idx, _mm512_set1_epi16((short) (1 << bit)));
        for (int k = 0; k + step < 16; k += 2 * step) r[k] = _mm512_mask_mov_epi16(r[k], m, r[k + step]);
    }
    return r[0];
}

// ... and of 32 IQ2_XXS / IQ3_XXS entries (8-bit indices) from code2xxs's / code3xxs's 8 registers
inline __m512i look256(__m512i idx, const __m512i* T) {
    __m512i r0 = _mm512_permutex2var_epi16(T[0], idx, T[1]);
    __m512i r1 = _mm512_permutex2var_epi16(T[2], idx, T[3]);
    __m512i r2 = _mm512_permutex2var_epi16(T[4], idx, T[5]);
    __m512i r3 = _mm512_permutex2var_epi16(T[6], idx, T[7]);
    const __mmask32 b6 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(64));
    const __mmask32 b7 = _mm512_test_epi16_mask(idx, _mm512_set1_epi16(128));
    r0 = _mm512_mask_mov_epi16(r0, b6, r1);
    r2 = _mm512_mask_mov_epi16(r2, b6, r3);
    return _mm512_mask_mov_epi16(r0, b7, r2);
}

// int16 lane l of a chunk's scale vector <- scale (4 / 2) j + l / 8 (IQ2_XS: a scale a 16 values) or l / 16 (IQ3_XXS:
// a scale a 32 values)
inline __m512i lane_groups(int per) {
    alignas(64) int16_t v[32];
    for (int l = 0; l < 32; ++l) v[l] = (int16_t) (l / per);
    return _mm512_load_si512(v);
}

// ---- per format: a 256-value block -> its four 64-value chunks' grid bytes g[j], sign masks m[j], int16 scales sc[j]
template <int TY> struct Dec;

template <> struct Dec<16> {   // IQ2_XXS: d, 8 x (4 grid indices, a word of 4 sign indices | scale << 28)
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    const __m512i* T = (const __m512i*) code2xxs;
    __m512i ks0, ks1, sh2, three, lut, pick, sp[4];
    __m256i shs, m7f, one32;
    Dec() {
        ks0 = _mm512_load_si512(ksigns);
        ks1 = _mm512_load_si512(ksigns + 64);
        sh2 = _mm512_set1_epi64(0x0e0c0a0806040200ll);   // byte v of a qword <- its bits 2v..2v+7
        three = _mm512_set1_epi8(3);
        lut = _mm512_set4_epi32(0, 0, 0, 0x002b1908);     // codes 0 / 1 / 2 -> 8 / 25 / 43
        alignas(64) uint8_t gi[64];
        for (int k = 0; k < 64; ++k) gi[k] = (uint8_t) ((k / 4) * 8 + k % 4);
        pick = _mm512_load_si512(gi);
        shs = _mm256_set1_epi64x(0x352e2720150e0700ll);   // a qword's two sign words -> bits 0, 7, 14, 21 / 32, ..
        m7f = _mm256_set1_epi8(0x7f);
        one32 = _mm256_set1_epi32(1);
        const __m512i g16 = lane_groups(16);
        for (int j = 0; j < 4; ++j) sp[j] = _mm512_add_epi16(g16, _mm512_set1_epi16((short) (2 * j)));
    }
    inline void block(const uint8_t* b, __m512i* g, uint64_t* m, __m512i* sc) const {
        const __m512i qs = _mm512_loadu_si512(b + 2);
        const __m512i codes =
            look256(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutexvar_epi8(pick, qs))), T);
        const __m256i aux = _mm512_cvtepi64_epi32(_mm512_srli_epi64(qs, 32));   // the 8 sign / scale words
        const __m512i s7 = _mm512_zextsi256_si512(_mm256_and_si256(_mm256_multishift_epi64_epi8(shs, aux), m7f));
        _mm256_storeu_si256((__m256i*) m, _mm512_castsi512_si256(_mm512_permutex2var_epi8(ks0, s7, ks1)));
        const __m256i ls = _mm256_add_epi32(_mm256_slli_epi32(_mm256_srli_epi32(aux, 28), 1), one32);   // 2s + 1
        const __m512i s16 = _mm512_zextsi128_si512(_mm256_cvtepi32_epi16(ls));
#define CH(j) g[j] = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_multishift_epi64_epi8(sh2, \
                     _mm512_cvtepu16_epi64(_mm512_extracti32x4_epi32(codes, j))), three)); \
              sc[j] = _mm512_permutexvar_epi16(sp[j], s16);
        CH(0) CH(1) CH(2) CH(3)
#undef CH
    }
};

template <> struct Dec<17> {   // IQ2_XS: d, 32 x (9-bit grid index | 7-bit sign index << 9), 8 scale bytes
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    const __m512i* T = (const __m512i*) code2xs;
    __m512i ks0, ks1, m511, sh2, three, lut, sp[4];
    __m128i m4, one8;
    Dec() {
        ks0 = _mm512_load_si512(ksigns);
        ks1 = _mm512_load_si512(ksigns + 64);
        m511 = _mm512_set1_epi16(511);
        sh2 = _mm512_set1_epi64(0x0e0c0a0806040200ll);
        three = _mm512_set1_epi8(3);
        lut = _mm512_set4_epi32(0, 0, 0, 0x002b1908);
        const __m512i g8 = lane_groups(8);
        for (int j = 0; j < 4; ++j) sp[j] = _mm512_add_epi16(g8, _mm512_set1_epi16((short) (4 * j)));
        m4 = _mm_set1_epi8(15);
        one8 = _mm_set1_epi8(1);
    }
    inline void block(const uint8_t* b, __m512i* g, uint64_t* m, __m512i* sc) const {
        const __m512i q = _mm512_loadu_si512(b + 2);
        const __m512i codes = look512(_mm512_and_si512(q, m511), T);
        const __m512i s7 = _mm512_zextsi256_si512(_mm512_cvtepi16_epi8(_mm512_srli_epi16(q, 9)));
        _mm256_storeu_si256((__m256i*) m, _mm512_castsi512_si256(_mm512_permutex2var_epi8(ks0, s7, ks1)));
        // ggml's scales: the 16 nibbles in order, 2s + 1, as int16
        const __m128i s8 = _mm_loadl_epi64((const __m128i*) (b + 66));
        const __m128i nib = _mm_unpacklo_epi8(_mm_and_si128(s8, m4), _mm_and_si128(_mm_srli_epi16(s8, 4), m4));
        const __m512i s16 = _mm512_zextsi256_si512(_mm256_cvtepu8_epi16(_mm_add_epi8(_mm_add_epi8(nib, nib), one8)));
#define CH(j) g[j] = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_multishift_epi64_epi8(sh2, \
                     _mm512_cvtepu16_epi64(_mm512_extracti32x4_epi32(codes, j))), three)); \
              sc[j] = _mm512_permutexvar_epi16(sp[j], s16);
        CH(0) CH(1) CH(2) CH(3)
#undef CH
    }
};

template <> struct Dec<22> {   // IQ2_S: d, 32 grid index bytes, 32 sign bytes, qh[8] (2 high index bits each), 8 scales
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    const __m512i* T = (const __m512i*) code2s;
    __m512i sh2, three, lut, hsh, h3, sp[4];
    __m128i m4, one8;
    Dec() {
        sh2 = _mm512_set1_epi64(0x0e0c0a0806040200ll);
        three = _mm512_set1_epi8(3);
        lut = _mm512_set4_epi32(0, 0, 0, 0x002b1908);
        // entry e's high index bits are bits 2e, 2e + 1 of qh (one qword, in every lane): to the low byte of word e
        alignas(64) uint8_t hc[64];
        for (int k = 0; k < 64; ++k) hc[k] = (uint8_t) (k % 2 == 0 ? (k / 2) * 2 : 0);
        hsh = _mm512_load_si512(hc);
        h3 = _mm512_set1_epi16(3);
        const __m512i g8 = lane_groups(8);
        for (int j = 0; j < 4; ++j) sp[j] = _mm512_add_epi16(g8, _mm512_set1_epi16((short) (4 * j)));
        m4 = _mm_set1_epi8(15);
        one8 = _mm_set1_epi8(1);
    }
    inline void block(const uint8_t* b, __m512i* g, uint64_t* m, __m512i* sc) const {
        uint64_t qh;
        std::memcpy(&qh, b + 66, 8);
        const __m512i hi = _mm512_and_si512(_mm512_multishift_epi64_epi8(hsh, _mm512_set1_epi64((long long) qh)), h3);
        const __m512i idx = _mm512_or_si512(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*) (b + 2))),
                                            _mm512_slli_epi16(hi, 8));
        std::memcpy(m, b + 34, 32);   // the sign bytes: chunk j's 64-bit mask at 8j
        if (iq2s_gather.load(std::memory_order_relaxed)) {
            // the grid's qwords themselves, 8 entries (a 64-value chunk) a gather from iq2s_grid (8 KB, L1): the
            // same bytes look1024's codes turn into, without its 16 two-table permutes and 15 blends a block
            // (STRATA_IQ2S_GATHER=0: the permutes).  The scales below are not read by the q8_1 rows
            const __m512i i0 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(idx));
            const __m512i i1 = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(idx, 1));
            const long long* G = (const long long*) iq2s_grid;
            g[0] = _mm512_i32gather_epi64(_mm512_castsi512_si256(i0), G, 8);
            g[1] = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(i0, 1), G, 8);
            g[2] = _mm512_i32gather_epi64(_mm512_castsi512_si256(i1), G, 8);
            g[3] = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(i1, 1), G, 8);
            const __m128i s8 = _mm_loadl_epi64((const __m128i*) (b + 74));
            const __m128i nib = _mm_unpacklo_epi8(_mm_and_si128(s8, m4), _mm_and_si128(_mm_srli_epi16(s8, 4), m4));
            const __m512i s16 = _mm512_zextsi256_si512(_mm256_cvtepu8_epi16(_mm_add_epi8(_mm_add_epi8(nib, nib), one8)));
            for (int j = 0; j < 4; ++j) sc[j] = _mm512_permutexvar_epi16(sp[j], s16);
            return;
        }
        const __m512i codes = look1024(idx, T);
        const __m128i s8 = _mm_loadl_epi64((const __m128i*) (b + 74));
        const __m128i nib = _mm_unpacklo_epi8(_mm_and_si128(s8, m4), _mm_and_si128(_mm_srli_epi16(s8, 4), m4));
        const __m512i s16 = _mm512_zextsi256_si512(_mm256_cvtepu8_epi16(_mm_add_epi8(_mm_add_epi8(nib, nib), one8)));
#define CH(j) g[j] = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_multishift_epi64_epi8(sh2, \
                     _mm512_cvtepu16_epi64(_mm512_extracti32x4_epi32(codes, j))), three)); \
              sc[j] = _mm512_permutexvar_epi16(sp[j], s16);
        CH(0) CH(1) CH(2) CH(3)
#undef CH
    }
};

template <> struct Dec<18> {   // IQ3_XXS: d, 64 grid index bytes, 8 x (4 sign indices | scale << 28)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    const __m512i* T = (const __m512i*) code3xxs;
    __m512i ks0, ks1, sh3, seven, lut, sp[4];
    __m256i shs, m7f, one32;
    Dec() {
        ks0 = _mm512_load_si512(ksigns);
        ks1 = _mm512_load_si512(ksigns + 64);
        sh3 = _mm512_set1_epi64(0x2926232009060300ll);   // a qword's two codes -> bits 0, 3, 6, 9 / 32, 35, 38, 41
        seven = _mm512_set1_epi8(7);
        lut = _mm512_set4_epi32(0, 0, 0x3e342c24, 0x1c140c04);   // codes 0..7 -> 4, 12, ..., 52, 62
        shs = _mm256_set1_epi64x(0x352e2720150e0700ll);
        m7f = _mm256_set1_epi8(0x7f);
        one32 = _mm256_set1_epi32(1);
        const __m512i g16 = lane_groups(16);
        for (int j = 0; j < 4; ++j) sp[j] = _mm512_add_epi16(g16, _mm512_set1_epi16((short) (2 * j)));
    }
    inline void block(const uint8_t* b, __m512i* g, uint64_t* m, __m512i* sc) const {
        const __m512i clo = look256(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*) (b + 2))), T);
        const __m512i chi = look256(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*) (b + 34))), T);
        const __m256i aux = _mm256_loadu_si256((const __m256i*) (b + 66));
        const __m512i s7 = _mm512_zextsi256_si512(_mm256_and_si256(_mm256_multishift_epi64_epi8(shs, aux), m7f));
        _mm256_storeu_si256((__m256i*) m, _mm512_castsi512_si256(_mm512_permutex2var_epi8(ks0, s7, ks1)));
        const __m256i ls = _mm256_add_epi32(_mm256_slli_epi32(_mm256_srli_epi32(aux, 28), 1), one32);
        const __m512i s16 = _mm512_zextsi128_si512(_mm256_cvtepi32_epi16(ls));
#define CH(j, cw, half) g[j] = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_multishift_epi64_epi8(sh3, \
                            _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(cw, half))), seven)); \
                        sc[j] = _mm512_permutexvar_epi16(sp[j], s16);
        CH(0, clo, 0) CH(1, clo, 1) CH(2, chi, 0) CH(3, chi, 1)
#undef CH
    }
};

template <> struct Dec<19> {   // IQ1_S: d, 32 grid index bytes (the low 8 bits), qh[8] (a 32-value group's 4 x 3 high
                               // index bits | its scale << 12 | its delta's sign << 15)
    static constexpr int bytes = 50;
    static constexpr float K = 1.0f;
    __m512i hpick, hshift, m7, sp[4];
    __m256i wpick;
    __m128i m7w, one16;
    Dec() {
        alignas(64) int16_t hp[32], hs[32];
        for (int e = 0; e < 32; ++e) {
            hp[e] = (int16_t) (e / 4);         // entry e's qh word
            hs[e] = (int16_t) (3 * (e % 4));   // and its 3 high bits' place there
        }
        hpick = _mm512_load_si512(hp);
        hshift = _mm512_load_si512(hs);
        m7 = _mm512_set1_epi16(7);
        const __m512i g16 = lane_groups(16);
        for (int j = 0; j < 4; ++j) sp[j] = _mm512_add_epi16(g16, _mm512_set1_epi16((short) (2 * j)));
        alignas(32) int16_t wp[16];
        for (int l = 0; l < 16; ++l) wp[l] = (int16_t) (l / 2);   // a 16-value group's 32-value group
        wpick = _mm256_load_si256((const __m256i*) wp);
        m7w = _mm_set1_epi16(7);
        one16 = _mm_set1_epi16(1);
    }
    // chunk j's 8 grid entries gathered from iq1s_grid (bytes -1 / 0 / 1): |v| for maddubs and the negative values'
    // mask; the 8 scales 2 s + 1 as int16 lanes; w: each 16-value group's signed scale, for the delta term (bsums . w)
    inline void block(const uint8_t* b, __m512i* g, uint64_t* m, __m512i* sc, __m256i& w) const {
        const __m128i qh = _mm_loadu_si128((const __m128i*) (b + 34));
        const __m512i hi =
            _mm512_and_si512(_mm512_srlv_epi16(_mm512_permutexvar_epi16(hpick, _mm512_castsi128_si512(qh)), hshift), m7);
        const __m512i idx = _mm512_or_si512(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*) (b + 2))),
                                            _mm512_slli_epi16(hi, 8));
        const __m512i i0 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(idx));
        const __m512i i1 = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(idx, 1));
        const __m256i ix[4] = {_mm512_castsi512_si256(i0), _mm512_extracti64x4_epi64(i0, 1),
                               _mm512_castsi512_si256(i1), _mm512_extracti64x4_epi64(i1, 1)};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __m512i v = _mm512_i32gather_epi64(ix[j], (const void*) iq1s_grid, 8);
            g[j] = _mm512_abs_epi8(v);
            m[j] = _cvtmask64_u64(_mm512_movepi8_mask(v));
        }
        const __m128i ls = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh, 12), m7w), 1), one16);
        const __m512i s16 = _mm512_zextsi128_si512(ls);
#pragma unroll
        for (int j = 0; j < 4; ++j) sc[j] = _mm512_permutexvar_epi16(sp[j], s16);
        // (qh | 1: never 0, negative exactly when the sign bit is set)
        w = _mm256_permutexvar_epi16(wpick, _mm256_zextsi128_si256(_mm_sign_epi16(ls, _mm_or_si128(qh, one16))));
    }
};

// rows [r0, r1) for NT tokens: a block decoded once (Dec), then each token's maddubs / vpdpwssd (ggml's integer
// steps) and fmadd with d_x * d_y (its float steps); IQ1_S also its delta term, ggml's scalar accum1
template <int TY, int NT>
void rows(const Dec<TY>& D, const uint8_t* w, size_t row_bytes, int nb, const block_q8_K* const* y, float* const* out,
          int r0, int r1, int ahead) {
    const __m512i zero = _mm512_setzero_si512();
    const bool fused = TY == 19 && iq1s_fused.load(std::memory_order_relaxed);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m256 accf[NT];
        float acc1[NT];
        for (int t = 0; t < NT; ++t) {
            accf[t] = _mm256_setzero_ps();
            acc1[t] = 0.0f;
        }
        for (int i = 0; i < nb; ++i) {
            const uint8_t* b = row + (size_t) i * Dec<TY>::bytes;
            if (ahead > 0) {
                _mm_prefetch((const char*) b + ahead, _MM_HINT_T0);
                _mm_prefetch((const char*) b + ahead + 64, _MM_HINT_T0);
            }
            __m512i g[4], sc[4];
            __m256i dw;
            alignas(32) uint64_t m[4];
            if constexpr (TY == 19) D.block(b, g, m, sc, dw);
            else D.block(b, g, m, sc);
            const float dx = h2f(u16(b));
            for (int t = 0; t < NT; ++t) {
                const int8_t* yq = y[t][i].qs;
                __m512i acc = _mm512_setzero_si512();
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const __m512i yv = _mm512_loadu_si512(yq + 64 * j);
                    const __m512i ys = _mm512_mask_sub_epi8(yv, _cvtu64_mask64(m[j]), zero, yv);
                    acc = _mm512_dpwssd_epi32(acc, _mm512_maddubs_epi16(g[j], ys), sc[j]);
                }
                const float d = dx * y[t][i].d;
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(fold(acc)), accf[t]);
                if constexpr (TY == 19) {
                    const float s1 =
                        (float) hsum_i32(_mm256_madd_epi16(_mm256_loadu_si256((const __m256i*) y[t][i].bsums), dw));
                    acc1[t] = fused ? std::fma(d, s1, acc1[t]) : acc1[t] + d * s1;
                }
            }
        }
        for (int t = 0; t < NT; ++t)
            out[t][r] = TY == 19 ? hsum_ggml(accf[t]) + IQ1S_DELTA * acc1[t] : Dec<TY>::K * hsum_ggml(accf[t]);
    }
}

template <int TY>
void rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
             int r1, int ahead) {
    static const Dec<TY> D;   // (built on the first call: after the CPU check)
    const int nb = n / QK_K;
    const block_q8_K* const* y = (const block_q8_K* const*) act;
    for (int t0 = 0; t0 < nt; t0 += 4) {
        switch (nt - t0 < 4 ? nt - t0 : 4) {
            case 1: rows<TY, 1>(D, w, row_bytes, nb, y + t0, out + t0, r0, r1, ahead); break;
            case 2: rows<TY, 2>(D, w, row_bytes, nb, y + t0, out + t0, r0, r1, ahead); break;
            case 3: rows<TY, 3>(D, w, row_bytes, nb, y + t0, out + t0, r0, r1, ahead); break;
            default: rows<TY, 4>(D, w, row_bytes, nb, y + t0, out + t0, r0, r1, ahead); break;
        }
    }
}


// ---- lane_q81 (lane_q81.hpp): the same decoded blocks against a Q81 activation (q8_1: a scale a 32 values), in the
// device's integer and float steps.  Per token and 64-value chunk: dpbusd (VNNI) - 16 int32 lanes, 4
// products each; three unpack+add steps give every 16 values' sum (U: lane 4q + j = chunk j's quarter q) and one
// permute the sub-blocks' first and second 16 (A, B: as lane_q81_avx2.cpp's).  The finish takes the device's integer
// steps on them (IQ2_XXS's sumi * ls / 8, ... truncated toward zero) and its d_w * d_x per 32 values; IQ1_S's sums are
// of the grid's values -1/0/1, the device's of the values + 1: plus the sub-block's code sum (an identity of integers).
template <int K> inline __m256i tdiv(__m256i x) {
    return _mm256_srai_epi32(_mm256_add_epi32(x, _mm256_and_si256(_mm256_srai_epi32(x, 31), _mm256_set1_epi32((1 << K) - 1))), K);
}

struct Q81Tok {
    const int8_t* q;
    const float* d;
    const float* s;
    const int16_t* b8;
};

template <int TY, int NT>
void rows_q81(const Dec<TY>& D, const uint8_t* w, size_t row_bytes, int nb, const Q81Tok* tk, float* const* out, int r0,
              int r1, int ahead) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i pab = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
    const __m256i one32 = _mm256_set1_epi32(1);
    // (IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS: each 32 values' d_w * d_x times the integer sum, in the device's float
    // order - q81::dev_sum - so the CPU lane's expert has the GPU's bits; IQ1_S keeps the sum below)
    constexpr bool dev = TY != 19;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m256 dv[dev ? NT : 1][4];   // (block i's 8 sub-blocks: the device's lanes 8 (i % 4) .. +7)
        if constexpr (dev)
            for (int t = 0; t < NT; ++t) dv[t][0] = dv[t][1] = dv[t][2] = dv[t][3] = _mm256_setzero_ps();
        __m256 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* b = row + (size_t) i * Dec<TY>::bytes;
            if (ahead > 0) {
                _mm_prefetch((const char*) b + ahead, _MM_HINT_T0);
                _mm_prefetch((const char*) b + ahead + 64, _MM_HINT_T0);
            }
            __m512i g[4], sc[4];
            __m256i dw;
            alignas(32) uint64_t m[4];
            if constexpr (TY == 19) D.block(b, g, m, sc, dw);
            else D.block(b, g, m, sc);
            const float dx = h2f(u16(b));
            // the block's integer scales (and IQ1_S's float factors), one lane a sub-block
            __m256i l0, l1;
            __m256 d1q, dl;
            if constexpr (TY == 16) {
                const __m256i aux = _mm512_cvtepi64_epi32(_mm512_srli_epi64(_mm512_loadu_si512(b + 2), 32));
                l0 = _mm256_or_si256(_mm256_srli_epi32(aux, 27), one32);
            } else if constexpr (TY == 18) {
                l0 = _mm256_srli_epi32(_mm256_loadu_si256((const __m256i*) (b + 66)), 28);
            } else if constexpr (TY == 17 || TY == 22) {
                const __m256i s8 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (b + (TY == 17 ? 66 : 74))));
                l0 = _mm256_and_si256(s8, _mm256_set1_epi32(15));
                l1 = _mm256_srli_epi32(s8, 4);
            } else {   // 19
                const __m256i qh = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) (b + 34)));
                d1q = _mm256_mul_ps(_mm256_set1_ps(dx), _mm256_cvtepi32_ps(_mm256_add_epi32(
                    _mm256_and_si256(_mm256_srli_epi32(qh, 11), _mm256_set1_epi32(0x0E)), one32)));
                const __m256 neg = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(qh, _mm256_set1_epi32(0x8000)), _mm256_set1_epi32(0x8000)));
                dl = _mm256_blendv_ps(_mm256_set1_ps(-1.0f + IQ1S_DELTA), _mm256_set1_ps(-1.0f + IQ1S_DELTA - 2.0f * IQ1S_DELTA), neg);
            }
            for (int t = 0; t < NT; ++t) {
                const int8_t* yq = tk[t].q + 256 * i;
                __m512i Q[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const __m512i yv = _mm512_loadu_si512(yq + 64 * j);
                    const __m512i ys = _mm512_mask_sub_epi8(yv, _cvtu64_mask64(m[j]), zero, yv);
                    Q[j] = _mm512_dpbusd_epi32(zero, g[j], ys);   // (= madd(maddubs(g, ys), 1): no pair saturates)
                }
                const __m512i T0 = _mm512_add_epi32(_mm512_unpacklo_epi32(Q[0], Q[1]), _mm512_unpackhi_epi32(Q[0], Q[1]));
                const __m512i T1 = _mm512_add_epi32(_mm512_unpacklo_epi32(Q[2], Q[3]), _mm512_unpackhi_epi32(Q[2], Q[3]));
                const __m512i U = _mm512_add_epi32(_mm512_unpacklo_epi64(T0, T1), _mm512_unpackhi_epi64(T0, T1));
                const __m512i AB = _mm512_permutexvar_epi32(pab, U);
                const __m256i A = _mm512_castsi512_si256(AB), B = _mm512_extracti64x4_epi64(AB, 1);
                const __m256 d8 = _mm256_loadu_ps(tk[t].d + 8 * i);
                __m256 c;
                if constexpr (TY == 16) {
                    const __m256i s = tdiv<3>(_mm256_mullo_epi32(_mm256_add_epi32(A, B), l0));
                    if constexpr (dev) {
                        dv[t][i & 3] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(dx), d8), _mm256_cvtepi32_ps(s), dv[t][i & 3]);
                        continue;
                    }
                    c = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(dx), d8), _mm256_cvtepi32_ps(s));
                } else if constexpr (TY == 18) {
                    const __m256i S = _mm256_add_epi32(A, B);
                    const __m256i s = tdiv<1>(_mm256_add_epi32(_mm256_mullo_epi32(l0, S), tdiv<1>(S)));
                    if constexpr (dev) {
                        dv[t][i & 3] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(dx), d8), _mm256_cvtepi32_ps(s), dv[t][i & 3]);
                        continue;
                    }
                    c = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(dx), d8), _mm256_cvtepi32_ps(s));
                } else if constexpr (TY == 17 || TY == 22) {
                    const __m256i s = tdiv<2>(_mm256_add_epi32(
                        _mm256_add_epi32(_mm256_mullo_epi32(A, l0), _mm256_mullo_epi32(B, l1)), tdiv<1>(_mm256_add_epi32(A, B))));
                    if constexpr (dev) {
                        dv[t][i & 3] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(dx), d8), _mm256_cvtepi32_ps(s), dv[t][i & 3]);
                        continue;
                    }
                    c = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(dx), d8), _mm256_cvtepi32_ps(s));
                } else {   // 19: + each sub-block's code sum (its 4 sums of 8)
                    const __m256i b4 = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*) (tk[t].b8 + 32 * i)));
                    const __m256i b4h = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*) (tk[t].b8 + 32 * i + 8)));
                    const __m256i b4c = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*) (tk[t].b8 + 32 * i + 16)));
                    const __m256i b4d = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*) (tk[t].b8 + 32 * i + 24)));
                    // [sub0's 4 | sub1's 4], ... -> hadd twice -> the 8 sub-blocks' sums in order (after a lane fix)
                    const __m256i h = _mm256_hadd_epi32(_mm256_hadd_epi32(b4, b4h), _mm256_hadd_epi32(b4c, b4d));
                    const __m256i us = _mm256_permutevar8x32_epi32(h, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
                    const __m256i S = _mm256_add_epi32(_mm256_add_epi32(A, B), us);
                    const __m256 s8 = _mm256_loadu_ps(tk[t].s + 8 * i);
                    c = _mm256_mul_ps(d1q, _mm256_add_ps(_mm256_mul_ps(d8, _mm256_cvtepi32_ps(S)), _mm256_mul_ps(s8, dl)));
                }
                acc[t] = _mm256_add_ps(acc[t], c);
            }
        }
        for (int t = 0; t < NT; ++t)
            if constexpr (dev) out[t][r] = q81::dev_sum(dv[t]);
            else out[t][r] = hsum_ggml(acc[t]);
    }
}

template <int TY>
void rows_q81_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1, int ahead) {
    static const Dec<TY> D;
    const int nb = n / QK_K;
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        Q81Tok tk[4];
        for (int t = 0; t < k; ++t) {
            const q81::View v = q81::view(act[t0 + t], n);
            tk[t] = Q81Tok{v.q, v.d, v.s, v.b8};
        }
        switch (k) {
            case 1: rows_q81<TY, 1>(D, w, row_bytes, nb, tk, out + t0, r0, r1, ahead); break;
            case 2: rows_q81<TY, 2>(D, w, row_bytes, nb, tk, out + t0, r0, r1, ahead); break;
            case 3: rows_q81<TY, 3>(D, w, row_bytes, nb, tk, out + t0, r0, r1, ahead); break;
            default: rows_q81<TY, 4>(D, w, row_bytes, nb, tk, out + t0, r0, r1, ahead); break;
        }
    }
}

// IQ3_XXS against a Q81 activation, the grid by byte planes (STRATA_IQ3XXS_PLANES=0: rows_q81's codes): iq3xxs_grid's
// byte v of all 256 entries is a 256-byte table, and vpermt2b looks 64 entries up at once (two tables of 128 and a
// blend on the index's bit 7) - 13 instructions for a block's 64 entries, where look256 and the codes' unpacking took
// ~38.  The planes interleave to dwords within 128-bit lanes: vector k's lane L holds entries 16 L + 4 k .. + 3, values
// 64 L + 16 k .. + 15, so the activation is read in that order (a copy of each token's codes, permuted once a call)
// and the signs likewise (one vpermb of the sign bytes).  Vector k's lane L lies in block 2 L + k / 2, so each 32
// values' integer sum is exact as before, and the float steps are rows_q81's for IQ3_XXS: the same bits.

template <int NT>
void rows_iq3xxs_q81(const uint8_t* w, size_t row_bytes, int nb, const Q81Tok* tk, float* const* out, int r0, int r1,
                     int ahead) {
    constexpr int kB = 98;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i* P = (const __m512i*) plane3xxs;   // plane v: P[4 v .. 4 v + 3]
    const __m512i ks0 = _mm512_load_si512(ksigns), ks1 = _mm512_load_si512(ksigns + 64);
    const __m256i shs = _mm256_set1_epi64x(0x352e2720150e0700ll), m7f = _mm256_set1_epi8(0x7f);
    const __m256i one32 = _mm256_set1_epi32(1);
    // the sign bytes (byte q: values 8 q .. 8 q + 7) in the vectors' order: vector k's lanes L take bytes 8 L + 2 k, + 1
    alignas(64) uint8_t sp[64] = {};
    for (int k = 0; k < 4; ++k)
        for (int L = 0; L < 4; ++L) {
            sp[8 * k + 2 * L] = (uint8_t) (8 * L + 2 * k);
            sp[8 * k + 2 * L + 1] = (uint8_t) (8 * L + 2 * k + 1);
        }
    const __m512i sperm = _mm512_load_si512(sp);
    const __m512i pick = _mm512_setr_epi32(0, 1, 4, 5, 8, 9, 12, 13, 0, 0, 0, 0, 0, 0, 0, 0);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m256 dv[NT][4];
        for (int t = 0; t < NT; ++t) dv[t][0] = dv[t][1] = dv[t][2] = dv[t][3] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* b = row + (size_t) i * kB;
            if (ahead > 0) {
                _mm_prefetch((const char*) b + ahead, _MM_HINT_T0);
                _mm_prefetch((const char*) b + ahead + 64, _MM_HINT_T0);
            }
            const __m512i X = _mm512_loadu_si512(b + 2);   // the 64 grid indices
            const __mmask64 hi = _mm512_movepi8_mask(X);
            __m512i pl[4];
#pragma unroll
            for (int v = 0; v < 4; ++v)
                pl[v] = _mm512_mask_blend_epi8(hi, _mm512_permutex2var_epi8(P[4 * v], X, P[4 * v + 1]),
                                               _mm512_permutex2var_epi8(P[4 * v + 2], X, P[4 * v + 3]));
            const __m512i a01l = _mm512_unpacklo_epi8(pl[0], pl[1]), a01h = _mm512_unpackhi_epi8(pl[0], pl[1]);
            const __m512i a23l = _mm512_unpacklo_epi8(pl[2], pl[3]), a23h = _mm512_unpackhi_epi8(pl[2], pl[3]);
            __m512i g[4];
            g[0] = _mm512_unpacklo_epi16(a01l, a23l);
            g[1] = _mm512_unpackhi_epi16(a01l, a23l);
            g[2] = _mm512_unpacklo_epi16(a01h, a23h);
            g[3] = _mm512_unpackhi_epi16(a01h, a23h);
            const __m256i aux = _mm256_loadu_si256((const __m256i*) (b + 66));
            const __m512i s7 = _mm512_zextsi256_si512(_mm256_and_si256(_mm256_multishift_epi64_epi8(shs, aux), m7f));
            alignas(64) uint64_t m[8];
            _mm512_store_si512(m, _mm512_permutexvar_epi8(sperm, _mm512_permutex2var_epi8(ks0, s7, ks1)));
            const float dx = h2f(u16(b));
            const __m256i l0 = _mm256_srli_epi32(aux, 28);
            for (int t = 0; t < NT; ++t) {
                const int8_t* yq = tk[t].q + 256 * i;   // (the permuted copy)
                __m512i Q[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const __m512i yv = _mm512_loadu_si512(yq + 64 * k);
                    const __m512i ys = _mm512_mask_sub_epi8(yv, _cvtu64_mask64(m[k]), zero, yv);
                    Q[k] = _mm512_dpbusd_epi32(zero, g[k], ys);
                }
                // lane L of Q0 + Q1: block 2 L's values, of Q2 + Q3: block 2 L + 1's
                const __m512i P0 = _mm512_add_epi32(Q[0], Q[1]), P1 = _mm512_add_epi32(Q[2], Q[3]);
                const __m512i c = _mm512_add_epi32(_mm512_unpacklo_epi32(P0, P1), _mm512_unpackhi_epi32(P0, P1));
                const __m512i d = _mm512_add_epi32(c, _mm512_shuffle_epi32(c, (_MM_PERM_ENUM) 0x4e));
                const __m256i S = _mm512_castsi512_si256(_mm512_permutexvar_epi32(pick, d));
                const __m256i sv = tdiv<1>(_mm256_add_epi32(_mm256_mullo_epi32(l0, S), tdiv<1>(S)));
                const __m256 d8 = _mm256_loadu_ps(tk[t].d + 8 * i);
                dv[t][i & 3] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(dx), d8), _mm256_cvtepi32_ps(sv), dv[t][i & 3]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = q81::dev_sum(dv[t]);
    }
}

void rows_iq3xxs_q81_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out,
                        int r0, int r1, int ahead) {
    const int nb = n / QK_K;
    thread_local std::vector<int8_t> perm;
    if (perm.size() < (size_t) 4 * n) perm.resize((size_t) 4 * n);
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        Q81Tok tk[4];
        for (int t = 0; t < k; ++t) {
            const q81::View v = q81::view(act[t0 + t], n);
            int8_t* pq = perm.data() + (size_t) t * n;
            for (int i = 0; i < nb; ++i) {   // vector k' of block i: [Y0, Y1, Y2, Y3]'s 128-bit lanes k'
                const int8_t* y = v.q + 256 * i;
                const __m512i Y0 = _mm512_loadu_si512(y), Y1 = _mm512_loadu_si512(y + 64);
                const __m512i Y2 = _mm512_loadu_si512(y + 128), Y3 = _mm512_loadu_si512(y + 192);
                const __m512i t0v = _mm512_shuffle_i64x2(Y0, Y1, 0x44), t1v = _mm512_shuffle_i64x2(Y0, Y1, 0xEE);
                const __m512i t2v = _mm512_shuffle_i64x2(Y2, Y3, 0x44), t3v = _mm512_shuffle_i64x2(Y2, Y3, 0xEE);
                int8_t* o = pq + 256 * i;
                _mm512_storeu_si512(o, _mm512_shuffle_i64x2(t0v, t2v, 0x88));
                _mm512_storeu_si512(o + 64, _mm512_shuffle_i64x2(t0v, t2v, 0xDD));
                _mm512_storeu_si512(o + 128, _mm512_shuffle_i64x2(t1v, t3v, 0x88));
                _mm512_storeu_si512(o + 192, _mm512_shuffle_i64x2(t1v, t3v, 0xDD));
            }
            tk[t] = Q81Tok{pq, v.d, v.s, v.b8};
        }
        switch (k) {
            case 1: rows_iq3xxs_q81<1>(w, row_bytes, nb, tk, out + t0, r0, r1, ahead); break;
            case 2: rows_iq3xxs_q81<2>(w, row_bytes, nb, tk, out + t0, r0, r1, ahead); break;
            case 3: rows_iq3xxs_q81<3>(w, row_bytes, nb, tk, out + t0, r0, r1, ahead); break;
            default: rows_iq3xxs_q81<4>(w, row_bytes, nb, tk, out + t0, r0, r1, ahead); break;
        }
    }
}

bool iq3xxs_planes_on() { return iq3xxs_planes.load(std::memory_order_relaxed); }

// IQ4_NL (the CPU lane's down rows on Qwen3.8's packs) against a Q81 activation in 512-bit lanes: two 32-value blocks a
// vector (the nibbles through kvalues_iq4nl by vpshufb, |w| and w's sign mask once for every token), dpbusd for the
// products, and each block's integer sum (exact in any order).  The float steps are lane_q81_avx2.cpp's T20, group by
// group (8 blocks, 256 values): d_w * d_x (fmul) times the sum, FMA-accumulated in dv[k % 4] - the device's order, so
// the rows give that file's bits.  STRATA_IQ4NL_512=0: the AVX2 rows
template <int NT>
void rows_iq4nl_q81(const uint8_t* w, size_t row_bytes, int n, const Q81Tok* tk, float* const* out, int r0, int r1,
                    int ahead) {
    constexpr int kB = 18;
    const int nsub = n / 32, ng = (nsub + 7) / 8;
    const __m512i tab = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) kvalues_iq4nl));
    const __m512i m4 = _mm512_set1_epi8(15), zero = _mm512_setzero_si512();
    const __m512i p01 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11), p23 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
    alignas(64) uint8_t xa[64], xb[64];
    alignas(64) uint16_t xw[32] = {};
    for (int b = 0; b < 4; ++b)
        for (int i = 0; i < 16; ++i) {
            xa[16 * b + i] = (uint8_t) (kB * b + 2 + i);              // from z0 | z1 (offsets 0..127)
            xb[16 * b + i] = (uint8_t) (kB * (b + 4) + 2 + i - 64);   // from z1 | z2 (offsets 64..191)
        }
    for (int b = 0; b < 8; ++b) xw[b] = (uint16_t) (kB * b / 2);
    const __m512i ia = _mm512_load_si512(xa), ib = _mm512_load_si512(xb), iw = _mm512_load_si512(xw);
    const __m256i pick = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);  // [s0 s2 s4 s6 | s1 s3 s5 s7] -> s0..s7
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        __m256 dv[NT][4];
        for (int t = 0; t < NT; ++t) dv[t][0] = dv[t][1] = dv[t][2] = dv[t][3] = _mm256_setzero_ps();
        for (int k = 0; k < ng; ++k) {
            const int ns = nsub - 8 * k < 8 ? nsub - 8 * k : 8;
            const uint8_t* g = row + (size_t) (8 * k) * kB;
            if (ahead > 0) {
                _mm_prefetch((const char*) g + ahead, _MM_HINT_T0);
                _mm_prefetch((const char*) g + ahead + 64, _MM_HINT_T0);
                _mm_prefetch((const char*) g + ahead + 128, _MM_HINT_T0);
            }
            // the group's 8 blocks (144 bytes, 18 * ns of them in the row): their nibbles gathered by vpermb into
            // [q0 q1 q2 q3] and [q4 q5 q6 q7], their scales by vpermw (offsets 18 j: whole words)
            const int len = kB * ns;
            const __m512i z0 = _mm512_maskz_loadu_epi8(len >= 64 ? ~0ull : (1ull << len) - 1ull, g);
            const __m512i z1 = len <= 64 ? zero : _mm512_maskz_loadu_epi8(len >= 128 ? ~0ull : (1ull << (len - 64)) - 1ull, g + 64);
            const __m512i z2 = len <= 128 ? zero : _mm512_maskz_loadu_epi8((1ull << (len - 128)) - 1ull, g + 128);
            const __m512i qa = _mm512_permutex2var_epi8(z0, ia, z1), qb = _mm512_permutex2var_epi8(z1, ib, z2);
            const __m512i la = _mm512_and_si512(qa, m4), ha = _mm512_and_si512(_mm512_srli_epi16(qa, 4), m4);
            const __m512i lb = _mm512_and_si512(qb, m4), hb = _mm512_and_si512(_mm512_srli_epi16(qb, 4), m4);
            // block 2j's values in 128-bit lanes 0, 1 of vector j, block 2j + 1's in lanes 2, 3
            __m512i v[4];
            v[0] = _mm512_shuffle_epi8(tab, _mm512_permutex2var_epi64(la, p01, ha));
            v[1] = _mm512_shuffle_epi8(tab, _mm512_permutex2var_epi64(la, p23, ha));
            v[2] = _mm512_shuffle_epi8(tab, _mm512_permutex2var_epi64(lb, p01, hb));
            v[3] = _mm512_shuffle_epi8(tab, _mm512_permutex2var_epi64(lb, p23, hb));
            __m512i mag[4];
            __mmask64 neg[4], ld[4];
            for (int j = 0; j < 4; ++j) {
                // (blocks past the row's end: out - kvalues_iq4nl[0] is not 0)
                ld[j] = 2 * j >= ns ? 0ull : 2 * j + 1 < ns ? ~0ull : 0xFFFFFFFFull;
                neg[j] = _mm512_movepi8_mask(v[j]) & ld[j];
                mag[j] = _mm512_maskz_abs_epi8(ld[j], v[j]);
            }
            const __m256 dw = _mm256_maskz_mov_ps((__mmask8) ((1u << ns) - 1u),
                                                  _mm256_cvtph_ps(_mm512_castsi512_si128(_mm512_permutex2var_epi16(z0, iw, z1))));
            const __mmask8 lm = (__mmask8) ((1u << ns) - 1u);
            for (int t = 0; t < NT; ++t) {
                const int8_t* y = tk[t].q + 256 * k;
                __m512i Q[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const __m512i yv = _mm512_maskz_loadu_epi8(ld[j], y + 64 * j);   // (nothing past the row)
                    Q[j] = _mm512_dpbusd_epi32(zero, mag[j], _mm512_mask_sub_epi8(yv, neg[j], zero, yv));
                }
                // each 128-bit lane's 4 sums: u's lane L holds [Q0, Q1, Q2, Q3]'s lane-L totals; block 2j = lanes 0 + 1
                // of Q[j], block 2j + 1 = lanes 2 + 3
                const __m512i T0 = _mm512_add_epi32(_mm512_unpacklo_epi32(Q[0], Q[1]), _mm512_unpackhi_epi32(Q[0], Q[1]));
                const __m512i T1 = _mm512_add_epi32(_mm512_unpacklo_epi32(Q[2], Q[3]), _mm512_unpackhi_epi32(Q[2], Q[3]));
                const __m512i U = _mm512_add_epi32(_mm512_unpacklo_epi64(T0, T1), _mm512_unpackhi_epi64(T0, T1));
                const __m128i e = _mm_add_epi32(_mm512_extracti32x4_epi32(U, 0), _mm512_extracti32x4_epi32(U, 1));
                const __m128i o = _mm_add_epi32(_mm512_extracti32x4_epi32(U, 2), _mm512_extracti32x4_epi32(U, 3));
                const __m256i S = _mm256_permutevar8x32_epi32(_mm256_set_m128i(o, e), pick);
                const __m256 d8 = _mm256_maskz_loadu_ps(lm, tk[t].d + 8 * k);
                const __m256 fa = _mm256_mul_ps(dw, d8), fb = _mm256_cvtepi32_ps(S);
                dv[t][k & 3] = _mm256_fmadd_ps(fa, fb, dv[t][k & 3]);
            }
        }
        for (int t = 0; t < NT; ++t) out[t][r] = q81::dev_sum(dv[t]);
    }
}

void rows_iq4nl_q81_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out,
                       int r0, int r1, int ahead) {
    for (int t0 = 0; t0 < nt; t0 += 4) {
        const int k = nt - t0 < 4 ? nt - t0 : 4;
        Q81Tok tk[4];
        for (int t = 0; t < k; ++t) {
            const q81::View v = q81::view(act[t0 + t], n);
            tk[t] = Q81Tok{v.q, v.d, v.s, v.b8};
        }
        switch (k) {
            case 1: rows_iq4nl_q81<1>(w, row_bytes, n, tk, out + t0, r0, r1, ahead); break;
            case 2: rows_iq4nl_q81<2>(w, row_bytes, n, tk, out + t0, r0, r1, ahead); break;
            case 3: rows_iq4nl_q81<3>(w, row_bytes, n, tk, out + t0, r0, r1, ahead); break;
            default: rows_iq4nl_q81<4>(w, row_bytes, n, tk, out + t0, r0, r1, ahead); break;
        }
    }
}

bool iq4nl_512_on() { return iq4nl_512.load(std::memory_order_relaxed); }

}  // namespace

bool lane_q81_vbmi_type(int type) noexcept {
    return (type == 16 || type == 17 || type == 18 || type == 19 || type == 22 || (type == 20 && iq4nl_512_on())) &&
           iq512v_ready();
}

void lane_q81_vbmi_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                        float* const* out, int r0, int r1) {
    static const int ahead = [] {
        const char* v = std::getenv("STRATA_IQ_PREFETCH");
        return v ? std::atoi(v) : 2048;
    }();
    switch (type) {
        case 16: rows_q81_nt<16>(nt, w, row_bytes, n, act, out, r0, r1, ahead); break;
        case 17: rows_q81_nt<17>(nt, w, row_bytes, n, act, out, r0, r1, ahead); break;
        case 18:
            if (iq3xxs_planes_on()) rows_iq3xxs_q81_nt(nt, w, row_bytes, n, act, out, r0, r1, ahead);
            else rows_q81_nt<18>(nt, w, row_bytes, n, act, out, r0, r1, ahead);
            break;
        case 19: rows_q81_nt<19>(nt, w, row_bytes, n, act, out, r0, r1, ahead); break;
        case 22: rows_q81_nt<22>(nt, w, row_bytes, n, act, out, r0, r1, ahead); break;
        case 20: rows_iq4nl_q81_nt(nt, w, row_bytes, n, act, out, r0, r1, ahead); break;
        default: break;
    }
}

bool iq512v_supported(int type) noexcept { return type == 16 || type == 17 || type == 18 || type == 19 || type == 22; }

void iq512v_set_iq1s_fused(bool fused) noexcept { iq1s_fused.store(fused); }
void iq512v_set_lane_kernels(bool on) noexcept {
    const auto pick = [on](const char* name) {
        const char* v = std::getenv(name);
        return v != nullptr ? std::atoi(v) != 0 : on;
    };
    iq2s_gather.store(pick("STRATA_IQ2S_GATHER"));
    iq3xxs_planes.store(pick("STRATA_IQ3XXS_PLANES"));
    iq4nl_512.store(pick("STRATA_IQ4NL_512"));
}

bool iq512v_ready() noexcept {
    static const bool ok = build_codes();
    return ok;
}

void iq512v_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                 int r0, int r1) {
    // E-2's software prefetch distance (STRATA_IQ_PREFETCH, bytes; 0 = off)
    static const int ahead = [] {
        const char* v = std::getenv("STRATA_IQ_PREFETCH");
        return v ? std::atoi(v) : 2048;
    }();
    switch (type) {
        case 16: rows_nt<16>(nt, w, row_bytes, n, act, out, r0, r1, ahead); break;
        case 17: rows_nt<17>(nt, w, row_bytes, n, act, out, r0, r1, ahead); break;
        case 18: rows_nt<18>(nt, w, row_bytes, n, act, out, r0, r1, ahead); break;
        case 19: rows_nt<19>(nt, w, row_bytes, n, act, out, r0, r1, ahead); break;
        case 22: rows_nt<22>(nt, w, row_bytes, n, act, out, r0, r1, ahead); break;
        default: break;
    }
}

}  // namespace strata::kernels::cpu
