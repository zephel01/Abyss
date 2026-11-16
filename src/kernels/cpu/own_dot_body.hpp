// src/kernels/cpu/own_dot_body.hpp - Strata's own CPU dot products of a GGUF weight row with a quantized activation,
// in the "lane model": the value a 256-bit integer dot product leaves in each of its 8 int32 lanes is computed per
// block (lane l holds the products of the values at positions 4l .. 4l+3 of every 32-value chunk - exact integers),
// each block adds its float scale times the lanes into 8 float accumulators with one fused multiply-add per lane, and
// the 8 accumulators are added in the order a 256 -> 128 -> 64 -> 32-bit horizontal sum takes:
// ((a4 + a0) + (a6 + a2)) + ((a5 + a1) + (a7 + a3)).  Only these float steps decide the result's bits; the integer
// parts are exact whatever instructions compute them.  This is the arithmetic of the x86 AVX2 dot products the
// GGUF ecosystem uses (ggml-cpu built for x86), so the CPU lane's experts keep giving the numbers they gave when
// Strata called ggml-cpu: own_dot_test checks every type against a ggml build bit for bit when one is configured.
//
// The file is included twice: own_dot.cpp (portable: any CPU, std::fma) and own_dot_avx2.cpp (the same code built
// with AVX2 + FMA, where the compiler vectorises the integer lanes and std::fma is one instruction).  Both are
// built without FP contraction, so every float step is the one written here.
//
// OD_NS names the namespace of this copy.
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace strata::kernels::cpu::OD_NS {
namespace {

inline float h2f(ggml_half h) {
    const uint16_t x = (uint16_t) h;
    const uint32_t s = (uint32_t) (x & 0x8000u) << 16, e = (x >> 10) & 0x1Fu, m = x & 0x3FFu;
    uint32_t b;
    if (e == 0x1F) b = s | 0x7F800000u | (m << 13) | (m != 0 ? 0x00400000u : 0u);   // a NaN comes out quiet
    else if (e != 0) b = s | ((e + 112u) << 23) | (m << 13);
    else if (m == 0) b = s;
    else {
        int k = 0;
        uint32_t mm = m;
        while ((mm & 0x400u) == 0) { mm <<= 1; ++k; }
        b = s | ((uint32_t) (113 - k) << 23) | ((mm & 0x3FFu) << 13);
    }
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
inline uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline float e8m0_half(uint8_t e) {
    const uint32_t b = e < 2 ? 0x00200000u << e : (uint32_t) (e - 1) << 23;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
inline float ue4m3_half(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (x >> 3) & 0xF, m = x & 7;
    const float v = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + (float) m / 8.0f, e - 7);
    return v * 0.5f;
}

struct Acc8 {
    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    inline void fma(float d, const int32_t* lane) {
        for (int l = 0; l < 8; ++l) a[l] = std::fma(d, (float) lane[l], a[l]);
    }
    inline float hsum() const { return ((a[4] + a[0]) + (a[6] + a[2])) + ((a[5] + a[1]) + (a[7] + a[3])); }
};

// lanes of n values (n a multiple of 32): lane l += w[v] * y[v] for every v with (v % 32) / 4 == l
inline void lanes(const int32_t* w, const int8_t* y, int n, int32_t* ln) {
    for (int c = 0; c < n; c += 32)
        for (int l = 0; l < 8; ++l) {
            int32_t s = 0;
            for (int k = 0; k < 4; ++k) s += w[c + 4 * l + k] * (int32_t) y[c + 4 * l + k];
            ln[l] += s;
        }
}

// ---------------- 32-value formats against Q8_0 / Q8_1
// the weights of one 32-value block as integers in value order
inline void dec_q4_0(const block_q4_0& x, int32_t* w) {
    for (int j = 0; j < 16; ++j) { w[j] = (x.qs[j] & 0xF) - 8; w[j + 16] = (x.qs[j] >> 4) - 8; }
}
inline void dec_q4_1(const block_q4_1& x, int32_t* w) {
    for (int j = 0; j < 16; ++j) { w[j] = x.qs[j] & 0xF; w[j + 16] = x.qs[j] >> 4; }
}
inline void dec_q5_0(const block_q5_0& x, int32_t* w) {
    const uint32_t qh = rd32(x.qh);
    for (int j = 0; j < 16; ++j) {
        w[j] = (int32_t) ((x.qs[j] & 0xF) | (((qh >> j) & 1u) << 4)) - 16;
        w[j + 16] = (int32_t) ((x.qs[j] >> 4) | (((qh >> (j + 16)) & 1u) << 4)) - 16;
    }
}
inline void dec_q5_1(const block_q5_1& x, int32_t* w) {
    const uint32_t qh = rd32(x.qh);
    for (int j = 0; j < 16; ++j) {
        w[j] = (int32_t) ((x.qs[j] & 0xF) | (((qh >> j) & 1u) << 4));
        w[j + 16] = (int32_t) ((x.qs[j] >> 4) | (((qh >> (j + 16)) & 1u) << 4));
    }
}
inline void dec_kv(const uint8_t* qs, const int8_t* kv, int32_t* w) {
    for (int j = 0; j < 16; ++j) { w[j] = kv[qs[j] & 0xF]; w[j + 16] = kv[qs[j] >> 4]; }
}

float dot_q4_0(int n, const void* vx, const void* vy) {
    const block_q4_0* x = (const block_q4_0*) vx;
    const block_q8_0* y = (const block_q8_0*) vy;
    Acc8 acc;
    int32_t w[32];
    for (int i = 0; i < n / 32; ++i) {
        dec_q4_0(x[i], w);
        int32_t ln[8] = {};
        lanes(w, y[i].qs, 32, ln);
        acc.fma(h2f(x[i].d) * h2f(y[i].d), ln);
    }
    return acc.hsum();
}
float dot_q5_0(int n, const void* vx, const void* vy) {
    const block_q5_0* x = (const block_q5_0*) vx;
    const block_q8_0* y = (const block_q8_0*) vy;
    Acc8 acc;
    int32_t w[32];
    for (int i = 0; i < n / 32; ++i) {
        dec_q5_0(x[i], w);
        int32_t ln[8] = {};
        lanes(w, y[i].qs, 32, ln);
        acc.fma(h2f(x[i].d) * h2f(y[i].d), ln);
    }
    return acc.hsum();
}
float dot_q8_0(int n, const void* vx, const void* vy) {
    const block_q8_0* x = (const block_q8_0*) vx;
    const block_q8_0* y = (const block_q8_0*) vy;
    Acc8 acc;
    int32_t w[32];
    for (int i = 0; i < n / 32; ++i) {
        for (int j = 0; j < 32; ++j) w[j] = x[i].qs[j];
        int32_t ln[8] = {};
        lanes(w, y[i].qs, 32, ln);
        acc.fma(h2f(x[i].d) * h2f(y[i].d), ln);
    }
    return acc.hsum();
}
// Q4_1 / Q5_1: unsigned codes; the min term a scalar sum (m x the activation block's s, one fused step a block)
float dot_q4_1(int n, const void* vx, const void* vy) {
    const block_q4_1* x = (const block_q4_1*) vx;
    const block_q8_1* y = (const block_q8_1*) vy;
    Acc8 acc;
    float summs = 0.0f;
    int32_t w[32];
    for (int i = 0; i < n / 32; ++i) {
        summs = std::fma(h2f(x[i].data.data.m), h2f(y[i].data.data.s), summs);
        dec_q4_1(x[i], w);
        int32_t ln[8] = {};
        lanes(w, y[i].qs, 32, ln);
        acc.fma(h2f(x[i].data.data.d) * h2f(y[i].data.data.d), ln);
    }
    return acc.hsum() + summs;
}
float dot_q5_1(int n, const void* vx, const void* vy) {
    const block_q5_1* x = (const block_q5_1*) vx;
    const block_q8_1* y = (const block_q8_1*) vy;
    Acc8 acc;
    float summs = 0.0f;
    int32_t w[32];
    for (int i = 0; i < n / 32; ++i) {
        summs = std::fma(h2f(x[i].data.data.m), h2f(y[i].data.data.s), summs);
        dec_q5_1(x[i], w);
        int32_t ln[8] = {};
        lanes(w, y[i].qs, 32, ln);
        acc.fma(h2f(x[i].data.data.d) * h2f(y[i].data.data.d), ln);
    }
    return acc.hsum() + summs;
}
// IQ4_NL / MXFP4: blocks in pairs, even ones into one accumulator set and odd ones into another (added lane by lane
// before the horizontal sum); an odd last block is a scalar step (MXFP4 fused, IQ4_NL a multiply then an add)
template <bool MX>
float dot_kv_pairs(int n, const void* vx, const void* vy) {
    using B = typename std::conditional<MX, block_mxfp4, block_iq4_nl>::type;
    const B* x = (const B*) vx;
    const block_q8_0* y = (const block_q8_0*) vy;
    const int nb = n / 32;
    const int8_t* kv = MX ? kvalues_mxfp4 : kvalues_iq4nl;
    const auto scale = [&](int i) {
        if constexpr (MX) return h2f(y[i].d) * e8m0_half(x[i].e);
        else return h2f(y[i].d) * h2f(x[i].d);
    };
    Acc8 a1, a2;
    int32_t w[32];
    int i = 0;
    for (; i + 1 < nb; i += 2) {
        int32_t l1[8] = {}, l2[8] = {};
        dec_kv(x[i].qs, kv, w);
        lanes(w, y[i].qs, 32, l1);
        dec_kv(x[i + 1].qs, kv, w);
        lanes(w, y[i + 1].qs, 32, l2);
        a1.fma(scale(i), l1);
        a2.fma(scale(i + 1), l2);
    }
    for (int l = 0; l < 8; ++l) a1.a[l] = a1.a[l] + a2.a[l];
    float sumf = a1.hsum();
    for (; i < nb; ++i) {
        dec_kv(x[i].qs, kv, w);
        int32_t s = 0;
        for (int j = 0; j < 32; ++j) s += w[j] * (int32_t) y[i].qs[j];
        sumf = MX ? std::fma(scale(i), (float) s, sumf) : sumf + scale(i) * (float) s;
    }
    return sumf;
}
// NVFP4: 64 values (four 16-value sub-blocks, each its own UE4M3 scale) against two Q8_0 blocks
float dot_nvfp4(int n, const void* vx, const void* vy) {
    const block_nvfp4* x = (const block_nvfp4*) vx;
    const block_q8_0* y = (const block_q8_0*) vy;
    Acc8 acc;
    int32_t w[64];
    for (int i = 0; i < n / 64; ++i) {
        for (int sb = 0; sb < 4; ++sb)
            for (int j = 0; j < 8; ++j) {
                w[16 * sb + j] = kvalues_mxfp4[x[i].qs[8 * sb + j] & 0xF];
                w[16 * sb + j + 8] = kvalues_mxfp4[x[i].qs[8 * sb + j] >> 4];
            }
        int32_t l1[8] = {}, l2[8] = {};
        lanes(w, y[2 * i].qs, 32, l1);
        lanes(w + 32, y[2 * i + 1].qs, 32, l2);
        const float dy0 = h2f(y[2 * i].d), dy1 = h2f(y[2 * i + 1].d);
        const float s0 = ue4m3_half(x[i].d[0]) * dy0, s1 = ue4m3_half(x[i].d[1]) * dy0;
        const float s2 = ue4m3_half(x[i].d[2]) * dy1, s3 = ue4m3_half(x[i].d[3]) * dy1;
        for (int l = 0; l < 8; ++l) acc.a[l] = std::fma(l < 4 ? s0 : s1, (float) l1[l], acc.a[l]);
        for (int l = 0; l < 8; ++l) acc.a[l] = std::fma(l < 4 ? s2 : s3, (float) l2[l], acc.a[l]);
    }
    return acc.hsum();
}
// Q2_0 (64 values, codes 0..3 meaning -1..2) against two Q8_0 blocks: a scalar sum, one fused step per Q8_0 block,
// then the weight block's scale a multiply and an add
float dot_q2_0(int n, const void* vx, const void* vy) {
    const block_q2_0* x = (const block_q2_0*) vx;
    const block_q8_0* y = (const block_q8_0*) vy;
    float sumf = 0.0f;
    for (int i = 0; i < n / 64; ++i) {
        float sumi = 0.0f;
        for (int k = 0; k < 2; ++k) {
            const block_q8_0& b = y[2 * i + k];
            int32_t s = 0;
            for (int j = 0; j < 32; ++j) s += ((int32_t) ((x[i].qs[8 * k + j / 4] >> (2 * (j % 4))) & 3) - 1) * b.qs[j];
            sumi = std::fma(h2f(b.d), (float) s, sumi);
        }
        sumf = sumf + h2f(x[i].d) * sumi;
    }
    return sumf;
}

// ---------------- 256-value formats against Q8_K
inline void k4_scale_min(int j, const uint8_t* q, int& sc, int& m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

float dot_q2_K(int n, const void* vx, const void* vy) {
    const block_q2_K* x = (const block_q2_K*) vx;
    const block_q8_K* y = (const block_q8_K*) vy;
    Acc8 acc;
    int32_t w[256];
    for (int i = 0; i < n / 256; ++i) {
        const float d = y[i].d * h2f(x[i].data.data.d);
        const float dmin = -y[i].d * h2f(x[i].data.data.dmin);
        int32_t prod[8];
        for (int l = 0; l < 8; ++l)
            prod[l] = (x[i].scales[2 * l] >> 4) * y[i].bsums[2 * l] + (x[i].scales[2 * l + 1] >> 4) * y[i].bsums[2 * l + 1];
        acc.fma(dmin, prod);
        for (int v = 0; v < 256; ++v) {
            const int half = v / 128, k = (v / 32) % 4, p = v % 32;
            w[v] = ((x[i].qs[32 * half + p] >> (2 * k)) & 3) * (x[i].scales[8 * half + 2 * k + (p >= 16)] & 0xF);
        }
        int32_t ln[8] = {};
        lanes(w, y[i].qs, 256, ln);
        acc.fma(d, ln);
    }
    return acc.hsum();
}
float dot_q3_K(int n, const void* vx, const void* vy) {
    const block_q3_K* x = (const block_q3_K*) vx;
    const block_q8_K* y = (const block_q8_K*) vy;
    Acc8 acc;
    int32_t w[256];
    for (int i = 0; i < n / 256; ++i) {
        const float d = y[i].d * h2f(x[i].d);
        int sc[16];
        for (int j = 0; j < 16; ++j) {
            const int lo = j < 8 ? x[i].scales[j] & 0xF : x[i].scales[j - 8] >> 4;
            const int hi = (x[i].scales[8 + (j % 4)] >> (2 * (j / 4))) & 3;
            sc[j] = (lo | (hi << 4)) - 32;
        }
        for (int v = 0; v < 256; ++v) {
            const int half = v / 128, k = (v / 32) % 4, p = v % 32;
            const int q = ((x[i].qs[32 * half + p] >> (2 * k)) & 3) - ((x[i].hmask[p] >> (4 * half + k)) & 1 ? 0 : 4);
            w[v] = q * sc[8 * half + 2 * k + (p >= 16)];
        }
        int32_t ln[8] = {};
        lanes(w, y[i].qs, 256, ln);
        acc.fma(d, ln);
    }
    return acc.hsum();
}
// Q4_K / Q5_K: the mins against the activation's 32-value sums - Q4_K four float lanes (a 128-bit accumulator, its
// own 4 -> 1 sum), Q5_K a scalar per block - and the codes x scales as 8 lanes
template <bool Q5>
float dot_q45_K(int n, const void* vx, const void* vy) {
    using B = typename std::conditional<Q5, block_q5_K, block_q4_K>::type;
    const B* x = (const B*) vx;
    const block_q8_K* y = (const block_q8_K*) vy;
    Acc8 acc;
    float accm[4] = {0, 0, 0, 0};
    float summs = 0.0f;
    int32_t w[256];
    for (int i = 0; i < n / 256; ++i) {
        const float d = y[i].d * h2f(x[i].data.data.d);
        const float dmin = -y[i].d * h2f(x[i].data.data.dmin);
        int sc[8], mn[8];
        for (int j = 0; j < 8; ++j) k4_scale_min(j, x[i].scales, sc[j], mn[j]);
        int32_t q8s[8];
        for (int k = 0; k < 8; ++k) q8s[k] = y[i].bsums[2 * k] + y[i].bsums[2 * k + 1];
        if constexpr (Q5) {
            int32_t t = 0;
            for (int k = 0; k < 8; ++k) t += mn[k] * q8s[k];
            summs = std::fma(dmin, (float) t, summs);
        } else {
            for (int m = 0; m < 4; ++m)
                accm[m] = std::fma(dmin, (float) (mn[2 * m] * q8s[2 * m] + mn[2 * m + 1] * q8s[2 * m + 1]), accm[m]);
        }
        for (int v = 0; v < 256; ++v) {
            const int j = v / 64, hi = (v / 32) % 2, p = v % 32;
            int q = hi ? x[i].qs[32 * j + p] >> 4 : x[i].qs[32 * j + p] & 0xF;
            if constexpr (Q5) q += ((x[i].qh[p] >> (2 * j + hi)) & 1) << 4;
            w[v] = q * sc[2 * j + hi];
        }
        int32_t ln[8] = {};
        lanes(w, y[i].qs, 256, ln);
        acc.fma(d, ln);
    }
    if constexpr (Q5) return acc.hsum() + summs;
    else return acc.hsum() + ((accm[0] + accm[2]) + (accm[1] + accm[3]));
}
// Q6_K: unsigned 6-bit codes x scales as lanes, the -32 offset taken off each lane as 32 x (the two 16-value
// activation sums of the lane's 32 values x their scales)
float dot_q6_K(int n, const void* vx, const void* vy) {
    const block_q6_K* x = (const block_q6_K*) vx;
    const block_q8_K* y = (const block_q8_K*) vy;
    Acc8 acc;
    int32_t w[256];
    for (int i = 0; i < n / 256; ++i) {
        const float d = y[i].d * h2f(x[i].d);
        for (int v = 0; v < 256; ++v) {
            const int half = v / 128, k = (v / 32) % 4, p = v % 32;
            const uint8_t l = x[i].ql[64 * half + 32 * (k & 1) + p];
            const int lo = k < 2 ? l & 0xF : l >> 4;
            const int q = lo | (((x[i].qh[32 * half + p] >> (2 * k)) & 3) << 4);
            w[v] = q * x[i].scales[v / 16];
        }
        int32_t ln[8] = {};
        lanes(w, y[i].qs, 256, ln);
        for (int l = 0; l < 8; ++l)
            ln[l] -= 32 * (y[i].bsums[2 * l] * x[i].scales[2 * l] + y[i].bsums[2 * l + 1] * x[i].scales[2 * l + 1]);
        acc.fma(d, ln);
    }
    return acc.hsum();
}

inline int sgn(uint8_t signs, int j) { return (signs >> j) & 1 ? -1 : 1; }

// the i-quants: grid values x signs x the integer sub-block scale as lanes, one fused step a block, then the
// format's constant
template <int T>
float dot_iq(int n, const void* vx, const void* vy) {
    const block_q8_K* y = (const block_q8_K*) vy;
    Acc8 acc;
    int32_t w[256];
    for (int i = 0; i < n / 256; ++i) {
        float dx = 0.0f;
        if constexpr (T == 16) {   // IQ2_XXS
            const block_iq2_xxs& x = ((const block_iq2_xxs*) vx)[i];
            dx = h2f(x.d);
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t* q = (const uint8_t*) (x.qs + 4 * ib);
                const uint32_t g = rd32(q), s = rd32(q + 4);
                const int ls = 2 * (int) (s >> 28) + 1;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t* gr = (const uint8_t*) (iq2xxs_grid + ((g >> (8 * l)) & 0xFF));
                    const uint8_t sg = ksigns_iq2xs[(s >> (7 * l)) & 127];
                    for (int j = 0; j < 8; ++j) w[32 * ib + 8 * l + j] = gr[j] * sgn(sg, j) * ls;
                }
            }
        } else if constexpr (T == 17) {   // IQ2_XS
            const block_iq2_xs& x = ((const block_iq2_xs*) vx)[i];
            dx = h2f(x.d);
            for (int ib = 0; ib < 8; ++ib)
                for (int l = 0; l < 4; ++l) {
                    const uint16_t q = x.qs[4 * ib + l];
                    const int ls = 2 * (l < 2 ? x.scales[ib] & 0xF : x.scales[ib] >> 4) + 1;
                    const uint8_t* gr = (const uint8_t*) (iq2xs_grid + (q & 511));
                    const uint8_t sg = ksigns_iq2xs[q >> 9];
                    for (int j = 0; j < 8; ++j) w[32 * ib + 8 * l + j] = gr[j] * sgn(sg, j) * ls;
                }
        } else if constexpr (T == 22) {   // IQ2_S
            const block_iq2_s& x = ((const block_iq2_s*) vx)[i];
            dx = h2f(x.d);
            const uint8_t* signs = x.qs + 32;
            for (int ib = 0; ib < 8; ++ib)
                for (int l = 0; l < 4; ++l) {
                    const int ls = 2 * (l < 2 ? x.scales[ib] & 0xF : x.scales[ib] >> 4) + 1;
                    const uint8_t* gr = (const uint8_t*) (iq2s_grid + (x.qs[4 * ib + l] | ((x.qh[ib] << (8 - 2 * l)) & 0x300)));
                    for (int j = 0; j < 8; ++j) w[32 * ib + 8 * l + j] = gr[j] * sgn(signs[4 * ib + l], j) * ls;
                }
        } else if constexpr (T == 18) {   // IQ3_XXS
            const block_iq3_xxs& x = ((const block_iq3_xxs*) vx)[i];
            dx = h2f(x.d);
            for (int ib = 0; ib < 8; ++ib) {
                const uint32_t s = rd32(x.qs + 64 + 4 * ib);
                const int ls = 2 * (int) (s >> 28) + 1;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t sg = ksigns_iq2xs[(s >> (7 * l)) & 127];
                    const uint8_t* g1 = (const uint8_t*) (iq3xxs_grid + x.qs[8 * ib + 2 * l]);
                    const uint8_t* g2 = (const uint8_t*) (iq3xxs_grid + x.qs[8 * ib + 2 * l + 1]);
                    for (int j = 0; j < 4; ++j) {
                        w[32 * ib + 8 * l + j] = g1[j] * sgn(sg, j) * ls;
                        w[32 * ib + 8 * l + j + 4] = g2[j] * sgn(sg, j + 4) * ls;
                    }
                }
            }
        } else if constexpr (T == 21) {   // IQ3_S
            const block_iq3_s& x = ((const block_iq3_s*) vx)[i];
            dx = h2f(x.d);
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = 2 * (ib % 2 == 0 ? x.scales[ib / 2] & 0xF : x.scales[ib / 2] >> 4) + 1;
                const uint8_t qh = x.qh[ib];
                for (int l = 0; l < 4; ++l) {
                    const uint8_t sg = x.signs[4 * ib + l];
                    const uint8_t* g1 = (const uint8_t*) (iq3s_grid + (x.qs[8 * ib + 2 * l] | ((qh << (8 - 2 * l)) & 256)));
                    const uint8_t* g2 = (const uint8_t*) (iq3s_grid + (x.qs[8 * ib + 2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
                    for (int j = 0; j < 4; ++j) {
                        w[32 * ib + 8 * l + j] = g1[j] * sgn(sg, j) * ls;
                        w[32 * ib + 8 * l + j + 4] = g2[j] * sgn(sg, j + 4) * ls;
                    }
                }
            }
        } else if constexpr (T == 23) {   // IQ4_XS
            const block_iq4_xs& x = ((const block_iq4_xs*) vx)[i];
            dx = h2f(x.d);
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = (((x.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((x.scales_h >> (2 * ib)) & 3) << 4)) - 32;
                for (int j = 0; j < 16; ++j) {
                    w[32 * ib + j] = kvalues_iq4nl[x.qs[16 * ib + j] & 0xF] * ls;
                    w[32 * ib + j + 16] = kvalues_iq4nl[x.qs[16 * ib + j] >> 4] * ls;
                }
            }
        }
        int32_t ln[8] = {};
        lanes(w, y[i].qs, 256, ln);
        acc.fma(dx * y[i].d, ln);
    }
    const float h = acc.hsum();
    if constexpr (T == 16 || T == 17 || T == 22) return 0.125f * h;
    else if constexpr (T == 18) return 0.25f * h;
    else return h;
}

// IQ1_S: grid x scale as lanes; the delta term (+-1/8 x the activation's 32-value sums) a scalar integer per block,
// one fused step a block, added after the horizontal sum
float dot_iq1_s(int n, const void* vx, const void* vy) {
    const block_iq1_s* x = (const block_iq1_s*) vx;
    const block_q8_K* y = (const block_q8_K*) vy;
    Acc8 acc;
    float accum1 = 0.0f;
    int32_t w[256];
    for (int i = 0; i < n / 256; ++i) {
        int32_t sumi1 = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const uint16_t h = x[i].qh[ib];
            const int ls = 2 * ((h >> 12) & 7) + 1;
            for (int l = 0; l < 4; ++l) {
                const int8_t* g = (const int8_t*) (iq1s_grid + (x[i].qs[4 * ib + l] | (((h >> (3 * l)) & 7) << 8)));
                for (int j = 0; j < 8; ++j) w[32 * ib + 8 * l + j] = g[j] * ls;
            }
            sumi1 += (y[i].bsums[2 * ib] + y[i].bsums[2 * ib + 1]) * (h & 0x8000 ? -1 : 1) * ls;
        }
        int32_t ln[8] = {};
        lanes(w, y[i].qs, 256, ln);
        const float d = y[i].d * h2f(x[i].d);
        acc.fma(d, ln);
        accum1 = std::fma(d, (float) sumi1, accum1);
    }
    return acc.hsum() + IQ1S_DELTA * accum1;
}
// IQ1_M: the grid lanes and the delta lanes (+-1 per 8 values x the activation x the 16-value scale), each its own
// accumulator set and horizontal sum
float dot_iq1_m(int n, const void* vx, const void* vy) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const block_q8_K* y = (const block_q8_K*) vy;
    Acc8 a1, a2;
    int32_t w[256], dl[256];
    for (int i = 0; i < n / 256; ++i) {
        uint16_t sc[4];
        std::memcpy(sc, x[i].scales, 8);
        const uint16_t sf = (uint16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t* qs = x[i].qs + 4 * ib;
            const uint8_t* qh = x[i].qh + 2 * ib;
            const int s1 = 2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7) + 1;
            const int s2 = 2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1;
            const int idx[4] = {qs[0] | ((qh[0] << 8) & 0x700), qs[1] | ((qh[0] << 4) & 0x700),
                                qs[2] | ((qh[1] << 8) & 0x700), qs[3] | ((qh[1] << 4) & 0x700)};
            const int dsg[4] = {qh[0] & 0x08 ? -1 : 1, qh[0] & 0x80 ? -1 : 1, qh[1] & 0x08 ? -1 : 1, qh[1] & 0x80 ? -1 : 1};
            for (int l = 0; l < 4; ++l) {
                const int8_t* g = (const int8_t*) (iq1s_grid + idx[l]);
                const int s = l < 2 ? s1 : s2;
                for (int j = 0; j < 8; ++j) {
                    w[32 * ib + 8 * l + j] = g[j] * s;
                    dl[32 * ib + 8 * l + j] = dsg[l] * s;
                }
            }
        }
        int32_t l1[8] = {}, l2[8] = {};
        lanes(w, y[i].qs, 256, l1);
        lanes(dl, y[i].qs, 256, l2);
        const float d = y[i].d * h2f((ggml_half) sf);
        a1.fma(d, l1);
        a2.fma(d, l2);
    }
    return a1.hsum() + IQ1M_DELTA * a2.hsum();
}

}  // namespace

// one row's dot product with one activation (the reference: every type)
float dot(int type, int n, const void* w, const void* a) {
    switch (type) {
        case 2: return dot_q4_0(n, w, a);
        case 3: return dot_q4_1(n, w, a);
        case 6: return dot_q5_0(n, w, a);
        case 7: return dot_q5_1(n, w, a);
        case 8: return dot_q8_0(n, w, a);
        case 10: return dot_q2_K(n, w, a);
        case 11: return dot_q3_K(n, w, a);
        case 12: return dot_q45_K<false>(n, w, a);
        case 13: return dot_q45_K<true>(n, w, a);
        case 14: return dot_q6_K(n, w, a);
        case 16: return dot_iq<16>(n, w, a);
        case 17: return dot_iq<17>(n, w, a);
        case 18: return dot_iq<18>(n, w, a);
        case 19: return dot_iq1_s(n, w, a);
        case 20: return dot_kv_pairs<false>(n, w, a);
        case 21: return dot_iq<21>(n, w, a);
        case 22: return dot_iq<22>(n, w, a);
        case 23: return dot_iq<23>(n, w, a);
        case 29: return dot_iq1_m(n, w, a);
        case 39: return dot_kv_pairs<true>(n, w, a);
        case 40: return dot_nvfp4(n, w, a);
        case 42: return dot_q2_0(n, w, a);
        default: return NAN;
    }
}

}  // namespace strata::kernels::cpu::OD_NS
