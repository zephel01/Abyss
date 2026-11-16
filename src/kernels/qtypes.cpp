// src/kernels/qtypes.cpp - see include/strata/qtypes.hpp.  Strata's own dequantizers, written from the block layouts
// in third_party/ggml/ggml-common.h.  Every float step is a single rounded multiply or add in the order the format
// defines a value (scale x integer, then the offset); this file is built without FMA contraction (CMakeLists).
#include "strata/qtypes.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace strata::qt {

// The ids are the GGUF format's: they must stay these numbers.
static_assert(Q4_0 == 2 && Q8_0 == 8 && Q2_K == 10 && Q8_K == 15 && IQ2_XXS == 16 && IQ4_XS == 23 && IQ1_M == 29 &&
              BF16 == 30 && TQ2_0 == 35 && MXFP4 == 39 && NVFP4 == 40 && Q2_0 == 42 && COUNT == 43, "GGUF type ids");

// ---- fp16 / bf16
float fp16_to_fp32(uint16_t h) noexcept {
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t b;
    if (e == 0x1F) {
        b = s | 0x7F800000u | (m << 13) | (m != 0 ? 0x00400000u : 0u);   // a NaN comes out quiet
    } else if (e != 0) {
        b = s | ((e + 112u) << 23) | (m << 13);
    } else if (m == 0) {
        b = s;
    } else {   // a subnormal: m x 2^-24, made normal
        int k = 0;
        uint32_t mm = m;
        while ((mm & 0x400u) == 0) {
            mm <<= 1;
            ++k;
        }
        b = s | ((uint32_t) (113 - k) << 23) | ((mm & 0x3FFu) << 13);
    }
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

uint16_t fp32_to_fp16(float f) noexcept {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000u, a = x & 0x7FFFFFFFu;
    if (a > 0x7F800000u) return (uint16_t) (s | 0x7E00u);     // NaN
    if (a >= 0x477FF000u) return (uint16_t) (s | 0x7C00u);    // >= 65520: infinity
    if (a >= 0x38800000u) {                                   // a normal half
        const uint32_t r = a - 0x38000000u;
        uint32_t h = r >> 13;
        const uint32_t rem = r & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
        return (uint16_t) (s | h);
    }
    if (a < 0x33000000u) return (uint16_t) s;                 // below 2^-25: zero
    const int e = (int) (a >> 23);
    const uint32_t m = (a & 0x7FFFFFu) | 0x800000u;
    const int sh = 126 - e;                                   // 14 .. 24
    uint32_t h = m >> sh;
    const uint32_t rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (h & 1u))) ++h;
    return (uint16_t) (s | h);
}

void fp32_to_fp16_row(const float* src, uint16_t* dst, int64_t n) noexcept {
    for (int64_t i = 0; i < n; ++i) dst[i] = fp32_to_fp16(src[i]);
}

float bf16_to_fp32(uint16_t h) noexcept {
    const uint32_t b = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

uint16_t fp32_to_bf16(float f) noexcept {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t) ((u >> 16) | 64u);
    return (uint16_t) ((u + (0x7FFFu + ((u >> 16) & 1u))) >> 16);
}

namespace {

inline float h2f(ggml_half h) { return fp16_to_fp32((uint16_t) h); }
inline uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }

// MXFP4's shared exponent: 2^(e-127), halved (the codebook holds twice the E2M1 values)
inline float e8m0_half(uint8_t e) {
    const uint32_t b = e < 2 ? 0x00200000u << e : (uint32_t) (e - 1) << 23;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
// NVFP4's sub-block scale: unsigned E4M3 (bias 7), halved likewise; 0x00 and 0x7F read as zero
inline float ue4m3_half(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (x >> 3) & 0xF, m = x & 7;
    const float v = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + (float) m / 8.0f, e - 7);
    return v * 0.5f;
}

// K-quants' 6-bit (scale, min) pair j of 8 from the 12 packed bytes
inline void k4_scale_min(int j, const uint8_t* q, int& sc, int& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

inline float sgn(uint8_t signs, int j) { return (signs & kmask_iq2xs[j]) ? -1.f : 1.f; }

void dq_f32(const void* s, float* y, int64_t n) { std::memcpy(y, s, (size_t) n * 4); }
void dq_f16(const void* s, float* y, int64_t n) {
    const uint16_t* h = (const uint16_t*) s;
    for (int64_t i = 0; i < n; ++i) y[i] = fp16_to_fp32(h[i]);
}
void dq_bf16(const void* s, float* y, int64_t n) {
    const uint16_t* h = (const uint16_t*) s;
    for (int64_t i = 0; i < n; ++i) y[i] = bf16_to_fp32(h[i]);
}

void dq_q4_0(const void* s, float* y, int64_t n) {
    const block_q4_0* x = (const block_q4_0*) s;
    for (int64_t i = 0; i < n / QK4_0; ++i, y += QK4_0) {
        const float d = h2f(x[i].d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            y[j] = (float) ((x[i].qs[j] & 0xF) - 8) * d;
            y[j + QK4_0 / 2] = (float) ((x[i].qs[j] >> 4) - 8) * d;
        }
    }
}
void dq_q4_1(const void* s, float* y, int64_t n) {
    const block_q4_1* x = (const block_q4_1*) s;
    for (int64_t i = 0; i < n / QK4_1; ++i, y += QK4_1) {
        const float d = h2f(x[i].data.data.d), m = h2f(x[i].data.data.m);
        for (int j = 0; j < QK4_1 / 2; ++j) {
            y[j] = (float) (x[i].qs[j] & 0xF) * d + m;
            y[j + QK4_1 / 2] = (float) (x[i].qs[j] >> 4) * d + m;
        }
    }
}
void dq_q5_0(const void* s, float* y, int64_t n) {
    const block_q5_0* x = (const block_q5_0*) s;
    for (int64_t i = 0; i < n / QK5_0; ++i, y += QK5_0) {
        const float d = h2f(x[i].d);
        const uint32_t qh = rd32(x[i].qh);
        for (int j = 0; j < QK5_0 / 2; ++j) {
            const int lo = (int) ((x[i].qs[j] & 0xF) | (((qh >> j) & 1u) << 4));
            const int hi = (int) ((x[i].qs[j] >> 4) | (((qh >> (j + 16)) & 1u) << 4));
            y[j] = (float) (lo - 16) * d;
            y[j + QK5_0 / 2] = (float) (hi - 16) * d;
        }
    }
}
void dq_q5_1(const void* s, float* y, int64_t n) {
    const block_q5_1* x = (const block_q5_1*) s;
    for (int64_t i = 0; i < n / QK5_1; ++i, y += QK5_1) {
        const float d = h2f(x[i].data.data.d), m = h2f(x[i].data.data.m);
        const uint32_t qh = rd32(x[i].qh);
        for (int j = 0; j < QK5_1 / 2; ++j) {
            const int lo = (int) ((x[i].qs[j] & 0xF) | (((qh >> j) & 1u) << 4));
            const int hi = (int) ((x[i].qs[j] >> 4) | (((qh >> (j + 16)) & 1u) << 4));
            y[j] = (float) lo * d + m;
            y[j + QK5_1 / 2] = (float) hi * d + m;
        }
    }
}
void dq_q8_0(const void* s, float* y, int64_t n) {
    const block_q8_0* x = (const block_q8_0*) s;
    for (int64_t i = 0; i < n / QK8_0; ++i, y += QK8_0) {
        const float d = h2f(x[i].d);
        for (int j = 0; j < QK8_0; ++j) y[j] = (float) x[i].qs[j] * d;
    }
}
void dq_mxfp4(const void* s, float* y, int64_t n) {
    const block_mxfp4* x = (const block_mxfp4*) s;
    for (int64_t i = 0; i < n / QK_MXFP4; ++i, y += QK_MXFP4) {
        const float d = e8m0_half(x[i].e);
        for (int j = 0; j < QK_MXFP4 / 2; ++j) {
            y[j] = (float) kvalues_mxfp4[x[i].qs[j] & 0xF] * d;
            y[j + QK_MXFP4 / 2] = (float) kvalues_mxfp4[x[i].qs[j] >> 4] * d;
        }
    }
}
void dq_nvfp4(const void* s, float* y, int64_t n) {
    const block_nvfp4* x = (const block_nvfp4*) s;
    for (int64_t i = 0; i < n / QK_NVFP4; ++i) {
        for (int sb = 0; sb < QK_NVFP4 / QK_NVFP4_SUB; ++sb, y += QK_NVFP4_SUB) {
            const float d = ue4m3_half(x[i].d[sb]);
            const uint8_t* q = x[i].qs + sb * (QK_NVFP4_SUB / 2);
            for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
                y[j] = (float) kvalues_mxfp4[q[j] & 0xF] * d;
                y[j + QK_NVFP4_SUB / 2] = (float) kvalues_mxfp4[q[j] >> 4] * d;
            }
        }
    }
}
void dq_q1_0(const void* s, float* y, int64_t n) {
    const block_q1_0* x = (const block_q1_0*) s;
    for (int64_t i = 0; i < n / QK1_0; ++i, y += QK1_0) {
        const float d = h2f(x[i].d), nd = -d;
        for (int j = 0; j < QK1_0; ++j) y[j] = ((x[i].qs[j / 8] >> (j % 8)) & 1) ? d : nd;
    }
}
void dq_q2_0(const void* s, float* y, int64_t n) {
    const block_q2_0* x = (const block_q2_0*) s;
    for (int64_t i = 0; i < n / QK2_0; ++i, y += QK2_0) {
        const float d = h2f(x[i].d);
        for (int j = 0; j < QK2_0; ++j) y[j] = (float) ((int) ((x[i].qs[j / 4] >> ((j % 4) * 2)) & 3) - 1) * d;
    }
}

void dq_q2_K(const void* s, float* y, int64_t n) {
    const block_q2_K* x = (const block_q2_K*) s;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = h2f(x[i].data.data.d), mn = h2f(x[i].data.data.dmin);
        const uint8_t* q = x[i].qs;
        int is = 0;
        for (int half = 0; half < 2; ++half, q += 32) {
            for (int sh = 0; sh < 8; sh += 2) {
                for (int part = 0; part < 2; ++part) {
                    const uint8_t sc = x[i].scales[is++];
                    const float dl = d * (float) (sc & 0xF), ml = mn * (float) (sc >> 4);
                    for (int l = 0; l < 16; ++l) *y++ = dl * (float) ((q[l + 16 * part] >> sh) & 3) - ml;
                }
            }
        }
    }
}

void q3_scales(const uint8_t* packed, int8_t* sc) {   // 16 signed 6-bit scales, minus 32
    for (int j = 0; j < 16; ++j) {
        const int lo = j < 8 ? packed[j] & 0xF : packed[j - 8] >> 4;
        const int hi = (packed[8 + (j % 4)] >> (2 * (j / 4))) & 3;
        sc[j] = (int8_t) ((lo | (hi << 4)) - 32);
    }
}
void dq_q3_K(const void* s, float* y, int64_t n) {
    const block_q3_K* x = (const block_q3_K*) s;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = h2f(x[i].d);
        int8_t sc[16];
        q3_scales(x[i].scales, sc);
        const uint8_t* q = x[i].qs;
        uint8_t m = 1;
        int is = 0;
        for (int half = 0; half < 2; ++half, q += 32) {
            for (int sh = 0; sh < 8; sh += 2, m <<= 1) {
                for (int part = 0; part < 2; ++part) {
                    const float dl = d * (float) sc[is++];
                    for (int l = 0; l < 16; ++l) {
                        const int k = l + 16 * part;
                        *y++ = dl * (float) (((q[k] >> sh) & 3) - ((x[i].hmask[k] & m) ? 0 : 4));
                    }
                }
            }
        }
    }
}
void dq_q4_K(const void* s, float* y, int64_t n) {
    const block_q4_K* x = (const block_q4_K*) s;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = h2f(x[i].data.data.d), mn = h2f(x[i].data.data.dmin);
        const uint8_t* q = x[i].qs;
        for (int j = 0; j < 4; ++j, q += 32) {
            int sc, m;
            k4_scale_min(2 * j, x[i].scales, sc, m);
            const float d1 = d * (float) sc, m1 = mn * (float) m;
            k4_scale_min(2 * j + 1, x[i].scales, sc, m);
            const float d2 = d * (float) sc, m2 = mn * (float) m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (float) (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (float) (q[l] >> 4) - m2;
        }
    }
}
void dq_q5_K(const void* s, float* y, int64_t n) {
    const block_q5_K* x = (const block_q5_K*) s;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = h2f(x[i].data.data.d), mn = h2f(x[i].data.data.dmin);
        const uint8_t* q = x[i].qs;
        for (int j = 0; j < 4; ++j, q += 32) {
            int sc, m;
            k4_scale_min(2 * j, x[i].scales, sc, m);
            const float d1 = d * (float) sc, m1 = mn * (float) m;
            k4_scale_min(2 * j + 1, x[i].scales, sc, m);
            const float d2 = d * (float) sc, m2 = mn * (float) m;
            const uint8_t u1 = (uint8_t) (1u << (2 * j)), u2 = (uint8_t) (2u << (2 * j));
            for (int l = 0; l < 32; ++l) *y++ = d1 * (float) ((q[l] & 0xF) + (x[i].qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (float) ((q[l] >> 4) + (x[i].qh[l] & u2 ? 16 : 0)) - m2;
        }
    }
}
void dq_q6_K(const void* s, float* y, int64_t n) {
    const block_q6_K* x = (const block_q6_K*) s;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = h2f(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;
        for (int half = 0; half < 2; ++half, y += 128, ql += 64, qh += 32, sc += 8) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = (int) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l] = d * (float) sc[is] * (float) q1;
                y[l + 32] = d * (float) sc[is + 2] * (float) q2;
                y[l + 64] = d * (float) sc[is + 4] * (float) q3;
                y[l + 96] = d * (float) sc[is + 6] * (float) q4;
            }
        }
    }
}
void dq_q8_K(const void* s, float* y, int64_t n) {
    const block_q8_K* x = (const block_q8_K*) s;
    for (int64_t i = 0; i < n / QK_K; ++i)
        for (int j = 0; j < QK_K; ++j) *y++ = x[i].d * (float) x[i].qs[j];
}
void dq_tq2_0(const void* s, float* y, int64_t n) {
    const block_tq2_0* x = (const block_tq2_0*) s;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = h2f(x[i].d);
        for (size_t j = 0; j < sizeof(x->qs); j += 32)
            for (int l = 0; l < 4; ++l)
                for (int m = 0; m < 32; ++m) *y++ = (float) ((int) ((x[i].qs[j + m] >> (l * 2)) & 3) - 1) * d;
    }
}

void dq_iq2_xxs(const void* s, float* y, int64_t n) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) s;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = h2f(x[i].d);
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const uint8_t* q = (const uint8_t*) (x[i].qs + 4 * ib);
            const uint32_t g = rd32(q), w = rd32(q + 4);
            const float db = d * (0.5f + (float) (w >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l, y += 8) {
                const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + ((g >> (8 * l)) & 0xFF));
                const uint8_t signs = ksigns_iq2xs[(w >> (7 * l)) & 127];
                for (int j = 0; j < 8; ++j) y[j] = db * (float) grid[j] * sgn(signs, j);
            }
        }
    }
}
void dq_iq2_xs(const void* s, float* y, int64_t n) {
    const block_iq2_xs* x = (const block_iq2_xs*) s;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = h2f(x[i].d);
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const float db[2] = {d * (0.5f + (float) (x[i].scales[ib] & 0xF)) * 0.25f,
                                 d * (0.5f + (float) (x[i].scales[ib] >> 4)) * 0.25f};
            for (int l = 0; l < 4; ++l, y += 8) {
                const uint16_t q = x[i].qs[4 * ib + l];
                const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q & 511));
                const uint8_t signs = ksigns_iq2xs[q >> 9];
                for (int j = 0; j < 8; ++j) y[j] = db[l / 2] * (float) grid[j] * sgn(signs, j);
            }
        }
    }
}
void dq_iq2_s(const void* s, float* y, int64_t n) {
    const block_iq2_s* x = (const block_iq2_s*) s;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = h2f(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* signs = x[i].qs + QK_K / 8;
        for (int ib = 0; ib < QK_K / 32; ++ib, qs += 4, signs += 4) {
            const float db[2] = {d * (0.5f + (float) (x[i].scales[ib] & 0xF)) * 0.25f,
                                 d * (0.5f + (float) (x[i].scales[ib] >> 4)) * 0.25f};
            for (int l = 0; l < 4; ++l, y += 8) {
                const uint8_t* grid = (const uint8_t*) (iq2s_grid + (qs[l] | ((x[i].qh[ib] << (8 - 2 * l)) & 0x300)));
                for (int j = 0; j < 8; ++j) y[j] = db[l / 2] * (float) grid[j] * sgn(signs[l], j);
            }
        }
    }
}
void dq_iq3_xxs(const void* s, float* y, int64_t n) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) s;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = h2f(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* ss = x[i].qs + QK_K / 4;
        for (int ib = 0; ib < QK_K / 32; ++ib, qs += 8) {
            const uint32_t w = rd32(ss + 4 * ib);
            const float db = d * (0.5f + (float) (w >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l, y += 8) {
                const uint8_t signs = ksigns_iq2xs[(w >> (7 * l)) & 127];
                const uint8_t* g1 = (const uint8_t*) (iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* g2 = (const uint8_t*) (iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * (float) g1[j] * sgn(signs, j);
                    y[j + 4] = db * (float) g2[j] * sgn(signs, j + 4);
                }
            }
        }
    }
}
void dq_iq3_s(const void* s, float* y, int64_t n) {
    const block_iq3_s* x = (const block_iq3_s*) s;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = h2f(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint8_t* signs = x[i].signs;
        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            for (int part = 0; part < 2; ++part, qs += 8, signs += 4) {
                const int sc = part == 0 ? x[i].scales[ib / 2] & 0xF : x[i].scales[ib / 2] >> 4;
                const float db = d * (float) (1 + 2 * sc);
                for (int l = 0; l < 4; ++l, y += 8) {
                    const uint8_t* g1 = (const uint8_t*) (iq3s_grid + (qs[2 * l + 0] | ((qh[part] << (8 - 2 * l)) & 256)));
                    const uint8_t* g2 = (const uint8_t*) (iq3s_grid + (qs[2 * l + 1] | ((qh[part] << (7 - 2 * l)) & 256)));
                    for (int j = 0; j < 4; ++j) {
                        y[j + 0] = db * (float) g1[j] * sgn(signs[l], j);
                        y[j + 4] = db * (float) g2[j] * sgn(signs[l], j + 4);
                    }
                }
            }
            qh += 2;
        }
    }
}
void dq_iq1_s(const void* s, float* y, int64_t n) {
    const block_iq1_s* x = (const block_iq1_s*) s;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = h2f(x[i].d);
        const uint8_t* qs = x[i].qs;
        for (int ib = 0; ib < QK_K / 32; ++ib, qs += 4) {
            const uint16_t h = x[i].qh[ib];
            const float dl = d * (float) (2 * ((h >> 12) & 7) + 1);
            const float delta = (h & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 4; ++l, y += 8) {
                const int8_t* g = (const int8_t*) (iq1s_grid + (qs[l] | (((h >> (3 * l)) & 7) << 8)));
                for (int j = 0; j < 8; ++j) y[j] = dl * ((float) g[j] + delta);
            }
        }
    }
}
void dq_iq1_m(const void* s, float* y, int64_t n) {
    const block_iq1_m* x = (const block_iq1_m*) s;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        uint16_t sc[4];
        std::memcpy(sc, x[i].scales, 8);
        const uint16_t sf = (uint16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
        const float d = fp16_to_fp32(sf);
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        for (int ib = 0; ib < QK_K / 32; ++ib, qs += 4, qh += 2) {
            const float dl1 = d * (float) (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7) + 1);
            const float dl2 = d * (float) (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1);
            const int idx[4] = {qs[0] | ((qh[0] << 8) & 0x700), qs[1] | ((qh[0] << 4) & 0x700),
                                qs[2] | ((qh[1] << 8) & 0x700), qs[3] | ((qh[1] << 4) & 0x700)};
            const float delta[4] = {qh[0] & 0x08 ? -IQ1M_DELTA : IQ1M_DELTA, qh[0] & 0x80 ? -IQ1M_DELTA : IQ1M_DELTA,
                                    qh[1] & 0x08 ? -IQ1M_DELTA : IQ1M_DELTA, qh[1] & 0x80 ? -IQ1M_DELTA : IQ1M_DELTA};
            for (int l = 0; l < 4; ++l, y += 8) {
                const int8_t* g = (const int8_t*) (iq1s_grid + idx[l]);
                const float dl = l < 2 ? dl1 : dl2;
                for (int j = 0; j < 8; ++j) y[j] = dl * ((float) g[j] + delta[l]);
            }
        }
    }
}
void dq_iq4_nl(const void* s, float* y, int64_t n) {
    const block_iq4_nl* x = (const block_iq4_nl*) s;
    for (int64_t i = 0; i < n / QK4_NL; ++i, y += QK4_NL) {
        const float d = h2f(x[i].d);
        for (int j = 0; j < QK4_NL / 2; ++j) {
            y[j] = d * (float) kvalues_iq4nl[x[i].qs[j] & 0xF];
            y[j + QK4_NL / 2] = d * (float) kvalues_iq4nl[x[i].qs[j] >> 4];
        }
    }
}
void dq_iq4_xs(const void* s, float* y, int64_t n) {
    const block_iq4_xs* x = (const block_iq4_xs*) s;
    for (int64_t i = 0; i < n / QK_K; ++i) {
        const float d = h2f(x[i].d);
        const uint8_t* qs = x[i].qs;
        for (int ib = 0; ib < QK_K / 32; ++ib, qs += 16, y += 32) {
            const int ls = ((x[i].scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((x[i].scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * (float) (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j] = dl * (float) kvalues_iq4nl[qs[j] & 0xF];
                y[j + 16] = dl * (float) kvalues_iq4nl[qs[j] >> 4];
            }
        }
    }
}

// ---- quantizers
void q8_0_ref(const float* x, void* dst, int64_t n) {
    block_q8_0* y = (block_q8_0*) dst;
    for (int64_t i = 0; i < n / QK8_0; ++i, x += QK8_0) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) amax = std::max(amax, std::fabs(x[j]));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = (ggml_half) fp32_to_fp16(d);
        for (int j = 0; j < QK8_0; ++j) y[i].qs[j] = (int8_t) std::round(x[j] * id);
    }
}

// IQ4_NL (own): the block's largest magnitude maps to the codebook's end, each value to its nearest code
void iq4_nl_own(const float* x, void* dst, int64_t n) {
    block_iq4_nl* y = (block_iq4_nl*) dst;
    for (int64_t i = 0; i < n / QK4_NL; ++i, x += QK4_NL) {
        float mx = 0.0f, amax = 0.0f;
        for (int j = 0; j < QK4_NL; ++j)
            if (std::fabs(x[j]) > amax) {
                amax = std::fabs(x[j]);
                mx = x[j];
            }
        const float d = amax > 0.0f ? mx / (float) kvalues_iq4nl[0] : 0.0f;
        y[i].d = (ggml_half) fp32_to_fp16(d);
        const float dd = fp16_to_fp32((uint16_t) y[i].d);
        uint8_t code[QK4_NL];
        for (int j = 0; j < QK4_NL; ++j) {
            int best = 0;
            float be = INFINITY;
            for (int k = 0; k < 16; ++k) {
                const float e = std::fabs(x[j] - dd * (float) kvalues_iq4nl[k]);
                if (e < be) {
                    be = e;
                    best = k;
                }
            }
            code[j] = (uint8_t) best;
        }
        for (int j = 0; j < QK4_NL / 2; ++j) y[i].qs[j] = (uint8_t) (code[j] | (code[j + QK4_NL / 2] << 4));
    }
}

// Q4_K (own): each 32-value sub-block's range [min(0, lo), hi] in 15 steps, the 8 scales and mins in 6 bits
void q4_K_own(const float* x, void* dst, int64_t n) {
    block_q4_K* y = (block_q4_K*) dst;
    for (int64_t i = 0; i < n / QK_K; ++i, x += QK_K) {
        float sc[8], mn[8], msc = 0.0f, mmn = 0.0f;
        for (int b = 0; b < 8; ++b) {
            float lo = 0.0f, hi = 0.0f;
            for (int j = 0; j < 32; ++j) {
                lo = std::min(lo, x[32 * b + j]);
                hi = std::max(hi, x[32 * b + j]);
            }
            sc[b] = (hi - lo) / 15.0f;
            mn[b] = -lo;
            msc = std::max(msc, sc[b]);
            mmn = std::max(mmn, mn[b]);
        }
        const float d = msc / 63.0f, dm = mmn / 63.0f;
        y[i].data.data.d = (ggml_half) fp32_to_fp16(d);
        y[i].data.data.dmin = (ggml_half) fp32_to_fp16(dm);
        const float dd = fp16_to_fp32((uint16_t) y[i].data.data.d), ddm = fp16_to_fp32((uint16_t) y[i].data.data.dmin);
        int ls[8], lm[8];
        for (int b = 0; b < 8; ++b) {
            ls[b] = dd > 0.0f ? std::min(63, (int) std::lround(sc[b] / dd)) : 0;
            lm[b] = ddm > 0.0f ? std::min(63, (int) std::lround(mn[b] / ddm)) : 0;
        }
        std::memset(y[i].scales, 0, sizeof(y[i].scales));
        for (int b = 0; b < 8; ++b) {
            if (b < 4) {
                y[i].scales[b] |= (uint8_t) ls[b];
                y[i].scales[b + 4] |= (uint8_t) lm[b];
            } else {
                y[i].scales[b + 4] = (uint8_t) ((ls[b] & 0xF) | ((lm[b] & 0xF) << 4));
                y[i].scales[b - 4] |= (uint8_t) ((ls[b] >> 4) << 6);
                y[i].scales[b] |= (uint8_t) ((lm[b] >> 4) << 6);
            }
        }
        uint8_t q[QK_K];
        for (int b = 0; b < 8; ++b) {
            const float s = dd * (float) ls[b], m = ddm * (float) lm[b];
            for (int j = 0; j < 32; ++j) {
                const int v = s > 0.0f ? (int) std::lround((x[32 * b + j] + m) / s) : 0;
                q[32 * b + j] = (uint8_t) std::max(0, std::min(15, v));
            }
        }
        for (int j = 0; j < 4; ++j)
            for (int l = 0; l < 32; ++l) y[i].qs[32 * j + l] = (uint8_t) (q[64 * j + l] | (q[64 * j + 32 + l] << 4));
    }
}

struct Table {
    Traits t[COUNT];
    Table() {
        const auto set = [this](int id, const char* nm, int64_t b, size_t sz, bool q, ToFloat tf, FromFloat ff = nullptr) {
            t[id] = Traits{nm, b, sz, q, tf, ff};
        };
        set(F32, "f32", 1, 4, false, dq_f32);
        set(F16, "f16", 1, 2, false, dq_f16);
        set(BF16, "bf16", 1, 2, false, dq_bf16);
        set(F64, "f64", 1, 8, false, nullptr);
        set(I8, "i8", 1, 1, false, nullptr);
        set(I16, "i16", 1, 2, false, nullptr);
        set(I32, "i32", 1, 4, false, nullptr);
        set(I64, "i64", 1, 8, false, nullptr);
        set(Q4_0, "q4_0", QK4_0, sizeof(block_q4_0), true, dq_q4_0);
        set(Q4_1, "q4_1", QK4_1, sizeof(block_q4_1), true, dq_q4_1);
        set(Q5_0, "q5_0", QK5_0, sizeof(block_q5_0), true, dq_q5_0);
        set(Q5_1, "q5_1", QK5_1, sizeof(block_q5_1), true, dq_q5_1);
        set(Q8_0, "q8_0", QK8_0, sizeof(block_q8_0), true, dq_q8_0, q8_0_ref);
        set(Q8_1, "q8_1", QK8_1, sizeof(block_q8_1), true, nullptr);
        set(Q2_K, "q2_K", QK_K, sizeof(block_q2_K), true, dq_q2_K);
        set(Q3_K, "q3_K", QK_K, sizeof(block_q3_K), true, dq_q3_K);
        set(Q4_K, "q4_K", QK_K, sizeof(block_q4_K), true, dq_q4_K);
        set(Q5_K, "q5_K", QK_K, sizeof(block_q5_K), true, dq_q5_K);
        set(Q6_K, "q6_K", QK_K, sizeof(block_q6_K), true, dq_q6_K);
        set(Q8_K, "q8_K", QK_K, sizeof(block_q8_K), true, dq_q8_K);
        set(IQ2_XXS, "iq2_xxs", QK_K, sizeof(block_iq2_xxs), true, dq_iq2_xxs);
        set(IQ2_XS, "iq2_xs", QK_K, sizeof(block_iq2_xs), true, dq_iq2_xs);
        set(IQ3_XXS, "iq3_xxs", QK_K, sizeof(block_iq3_xxs), true, dq_iq3_xxs);
        set(IQ1_S, "iq1_s", QK_K, sizeof(block_iq1_s), true, dq_iq1_s);
        set(IQ4_NL, "iq4_nl", QK4_NL, sizeof(block_iq4_nl), true, dq_iq4_nl);
        set(IQ3_S, "iq3_s", QK_K, sizeof(block_iq3_s), true, dq_iq3_s);
        set(IQ2_S, "iq2_s", QK_K, sizeof(block_iq2_s), true, dq_iq2_s);
        set(IQ4_XS, "iq4_xs", QK_K, sizeof(block_iq4_xs), true, dq_iq4_xs);
        set(IQ1_M, "iq1_m", QK_K, sizeof(block_iq1_m), true, dq_iq1_m);
        set(TQ1_0, "tq1_0", QK_K, sizeof(block_tq1_0), true, nullptr);
        set(TQ2_0, "tq2_0", QK_K, sizeof(block_tq2_0), true, dq_tq2_0);
        set(MXFP4, "mxfp4", QK_MXFP4, sizeof(block_mxfp4), true, dq_mxfp4);
        set(NVFP4, "nvfp4", QK_NVFP4, sizeof(block_nvfp4), true, dq_nvfp4);
        set(Q1_0, "q1_0", QK1_0, sizeof(block_q1_0), true, dq_q1_0);
        set(Q2_0, "q2_0", QK2_0, sizeof(block_q2_0), true, dq_q2_0);
    }
};
const Table& table() {
    static const Table t;
    return t;
}

}  // namespace

const Traits* traits(int type) noexcept { return type >= 0 && type < COUNT ? &table().t[type] : nullptr; }
const char* type_name(int type) noexcept {
    const Traits* t = traits(type);
    return t != nullptr && t->name != nullptr ? t->name : "?";
}
int64_t blck_size(int type) noexcept {
    const Traits* t = traits(type);
    return t != nullptr ? t->blck_size : 0;
}
size_t type_size(int type) noexcept {
    const Traits* t = traits(type);
    return t != nullptr ? t->type_size : 0;
}
size_t row_size(int type, int64_t n) noexcept {
    const Traits* t = traits(type);
    return t != nullptr && t->blck_size > 0 ? t->type_size * (size_t) (n / t->blck_size) : 0;
}

size_t quantize_chunk(int type, const float* src, void* dst, int64_t start, int64_t nrows, int64_t n_per_row) noexcept {
    FromFloat f = nullptr;
    if (type == Q8_0) f = q8_0_ref;
    else if (type == Q4_K) f = q4_K_own;
    else if (type == IQ4_NL) f = iq4_nl_own;
    if (f == nullptr || n_per_row % blck_size(type) != 0) return 0;
    const size_t rb = row_size(type, n_per_row);
    const int64_t r0 = start / n_per_row;
    for (int64_t r = 0; r < nrows; ++r)
        f(src + start + r * n_per_row, (uint8_t*) dst + (size_t) (r0 + r) * rb, n_per_row);
    return (size_t) nrows * rb;
}

}  // namespace strata::qt
