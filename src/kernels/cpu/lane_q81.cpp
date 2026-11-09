// src/kernels/cpu/lane_q81.cpp - see include/strata/kernels/cpu/lane_q81.hpp.  The quantizer, the dispatch, and the
// reference: glm_fast.cu's expert dot functions (llama.cpp's vecdotq, MIT, third_party/ggml/LICENSE) transcribed with
// their CUDA intrinsics emulated byte for byte - the vector forms (lane_q81_avx2.cpp, iq_avx512_vbmi.cpp) are checked
// against it (lane_q81_test).
#include "strata/kernels/cpu/lane_q81.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "lane_q81_impl.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::kernels::cpu {
const uint32_t* lane_q81_iq1s_grid_gpu();   // (lane_q81_gpu_tables.cpp)
namespace {

using q81::h2f;

// ---- the CUDA intrinsics the dot functions use
inline int dp4a(int a, int b, int c) {
    for (int i = 0; i < 4; ++i) c += (int) (int8_t) (a >> (8 * i)) * (int) (int8_t) (b >> (8 * i));
    return c;
}
inline int vcmpne4(uint32_t a, uint32_t b) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i)
        if (((a >> (8 * i)) & 0xff) != ((b >> (8 * i)) & 0xff)) r |= 0xffu << (8 * i);
    return (int) r;
}
inline int vsub4(uint32_t a, uint32_t b) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) r |= ((((a >> (8 * i)) & 0xff) - ((b >> (8 * i)) & 0xff)) & 0xff) << (8 * i);
    return (int) r;
}
inline int vsubss4(uint32_t a, uint32_t b) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        int v = (int) (int8_t) (a >> (8 * i)) - (int) (int8_t) (b >> (8 * i));
        v = std::min(127, std::max(-128, v));
        r |= ((uint32_t) v & 0xff) << (8 * i);
    }
    return (int) r;
}
inline int popc(uint32_t v) {
    int c = 0;
    for (; v; v &= v - 1) ++c;
    return c;
}
inline uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}
inline int get_int(const void* x, int i32) {   // get_int_b1 / b2 / b4: the same little-endian int
    int v;
    std::memcpy(&v, (const uint8_t*) x + 4 * i32, 4);
    return v;
}
struct I2 {
    int x, y;
};
inline I2 get_int_from_table_16(int q4, const int8_t* table) {
    uint32_t x = 0, y = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t b = ((uint32_t) q4 >> (8 * i)) & 0xff;
        x |= (uint32_t) (uint8_t) table[b & 15] << (8 * i);
        y |= (uint32_t) (uint8_t) table[b >> 4] << (8 * i);
    }
    return I2{(int) x, (int) y};
}
inline float half_of(const void* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return h2f(h);
}
inline float bits_f(uint32_t u) { return q81::bits_f(u); }

// a q8_1 block as the dots read it: its 8 ints, d (__low2float(ds)) and s (the high half)
struct XB {
    int u[8];
    float d, s;
};

// ---- K quants
float dot_q4_K(const void* vbq, const XB* bq8_1, int kbx, int iqs) {
    const block_q4_K* bq4 = (const block_q4_K*) vbq + kbx;
    int v[2], u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const uint8_t* qlb = bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4);
    v[0] = get_int(qlb, 0);
    v[1] = get_int(qlb, 4);
    uint16_t scales[6];
    std::memcpy(scales, bq4->scales, 12);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm], s2 = scales[jm + 2], s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
    for (int i = 0; i < 2; ++i) {
        const XB* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = bq8i->d;
        const int* q8 = bq8i->u + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    float sumf_d = 0.0f, sumf_m = 0.0f;
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = dp4a(v1i, u[2 * i + 1], dp4a(v0i, u[2 * i], 0));
        const int dot2 = dp4a(0x01010101, u[2 * i + 1], dp4a(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float dmx = half_of((const uint8_t*) bq4), dmy = half_of((const uint8_t*) bq4 + 2);
    return dmx * sumf_d - dmy * sumf_m;
}

float dot_q5_K(const void* vbq, const XB* bq8_1, int kbx, int iqs) {
    const block_q5_K* bq5 = (const block_q5_K*) vbq + kbx;
    int vl[2], vh[2], u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const uint8_t* qlb = bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4);
    const uint8_t* qhb = bq5->qh + 4 * ((iqs / 2) % 4);
    vl[0] = get_int(qlb, 0);
    vl[1] = get_int(qlb, 4);
    vh[0] = get_int(qhb, 0) >> bq8_offset;
    vh[1] = get_int(qhb, 4) >> bq8_offset;
    uint16_t scales[6];
    std::memcpy(scales, bq5->scales, 12);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm], s2 = scales[jm + 2], s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
    for (int i = 0; i < 2; ++i) {
        const XB* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = bq8i->d;
        const int* q8 = bq8i->u + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    float sumf_d = 0.0f, sumf_m = 0.0f;
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = dp4a(v0i, u[2 * i], dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = dp4a(0x01010101, u[2 * i], dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float dmx = half_of((const uint8_t*) bq5), dmy = half_of((const uint8_t*) bq5 + 2);
    return dmx * sumf_d - dmy * sumf_m;
}

float dot_q6_K(const void* vbq, const XB* bq8_1, int kbx, int iqs) {
    const block_q6_K* w = (const block_q6_K*) vbq + kbx;
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = get_int(w->ql, iqs);
    const int vh = get_int(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    const int8_t* scales = w->scales + scale_offset;
    float sumf = 0.0f;
    for (int i = 0; i < 2; ++i) {
        const int u = bq8_1[bq8_offset + 2 * i].u[iqs % 8];
        const float d8 = bq8_1[bq8_offset + 2 * i].d;
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = vsubss4(vil | vih, 0x20202020);
        sumf += d8 * (dp4a(vi, u, 0) * sc);
    }
    return half_of(&w->d) * sumf;
}

float dot_q8_0(const void* vbq, const XB* bq8_1, int kbx, int iqs) {
    const block_q8_0* w = (const block_q8_0*) vbq + kbx;
    int sumi = 0;
    for (int i = 0; i < 2; ++i) {
        const int v = get_int(w->qs, iqs + i);
        const int u = bq8_1->u[iqs + i];
        sumi = dp4a(v, u, sumi);
    }
    return half_of(&w->d) * bq8_1->d * float(sumi);
}

float dot_q2_K(const void* vbq, const XB* bq8_1, int kbx, int iqs) {
    const block_q2_K* b = (const block_q2_K*) vbq + kbx;
    const int bq8_offset = 4 * (iqs / 8);
    const uint8_t* scales = b->scales + (iqs - iqs % 8 + (iqs % 8) / 4);
    const int v = get_int(b->qs, iqs);
    float sumf_d = 0.0f, sumf_m = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const XB* q8 = bq8_1 + bq8_offset + i;
        const int u = q8->u[iqs % 8];
        const float d8 = q8->d;
        const int sc = scales[2 * i];
        const int vi = (v >> (2 * i)) & 0x03030303;
        sumf_d += d8 * (dp4a(vi, u, 0) * (sc & 0xF));
        int m = sc >> 4;
        m |= m << 8;
        m |= m << 16;
        sumf_m += d8 * dp4a(m, u, 0);
    }
    const float dmx = half_of((const uint8_t*) b + 80), dmy = half_of((const uint8_t*) b + 82);
    return dmx * sumf_d - dmy * sumf_m;
}

float dot_q3_K(const void* vbq, const XB* bq8_1, int kbx, int iqs) {
    const block_q3_K* b = (const block_q3_K*) vbq + kbx;
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const float d = half_of(&b->d);
    const int vl = get_int(b->qs, iqs);
    const int vh = ~get_int(b->hmask, iqs % 8) >> bq8_offset;
    float sumf = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const XB* q8 = bq8_1 + bq8_offset + i;
        const int u = q8->u[iqs % 8];
        const float d8 = q8->d;
        const int isc = scale_offset + 2 * i;
        const int sc_low = (b->scales[isc % 8] >> (4 * (isc / 8))) & 0xF;
        const int sc_high = ((b->scales[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        sumf += d8 * (dp4a(vsubss4(vil, vih), u, 0) * sc);
    }
    return d * sumf;
}

// ---- i-quants (the dw<T> forms: x = the call's one q8_1 block)
float dw16(const uint8_t* row, int kbx, int iqs, const XB& x) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) row + kbx;
    const int q2 = get_int(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = (uint32_t) get_int(bq2->qs, iqs + 1);
    int sumi = 0;
    for (int k0 = 0; k0 < 8; k0 += 2) {
        uint32_t gp[2];
        std::memcpy(gp, &iq2xxs_grid[aux8[k0 / 2]], 8);
        const uint32_t signs = unpack_ksigns((uint8_t) (aux32 >> (7 * k0 / 2)));
        const int signs0 = vcmpne4(signs & 0x08040201, 0);
        sumi = dp4a(vsub4(gp[0] ^ signs0, signs0), x.u[k0 + 0], sumi);
        const int signs1 = vcmpne4(signs & 0x80402010, 0);
        sumi = dp4a(vsub4(gp[1] ^ signs1, signs1), x.u[k0 + 1], sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    return half_of(&bq2->d) * x.d * sumi;
}
float dw18(const uint8_t* row, int kbx, int iqs, const XB& x) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) row + kbx;
    int q3_packed[2] = {get_int(bq3->qs, iqs), get_int(bq3->qs, iqs + 1)};
    const uint8_t* q3 = (const uint8_t*) q3_packed;
    const uint32_t aux32 = (uint32_t) get_int(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t gx = iq3xxs_grid[q3[l0 + 0]], gy = iq3xxs_grid[q3[l0 + 1]];
        const uint32_t signs = unpack_ksigns((uint8_t) (aux32 >> (7 * l0 / 2)));
        const int signs0 = vcmpne4(signs & 0x08040201, 0);
        const int signs1 = vcmpne4(signs & 0x80402010, 0);
        sumi = dp4a(vsub4(gx ^ signs0, signs0), x.u[l0 + 0], sumi);
        sumi = dp4a(vsub4(gy ^ signs1, signs1), x.u[l0 + 1], sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    return half_of(&bq3->d) * x.d * sumi;
}
float dw19(const uint8_t* row, int kbx, int iqs, const XB& x) {
    const block_iq1_s* bq1 = (const block_iq1_s*) row + kbx;
    const int qs_packed = get_int(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq1->qh[iqs];
    int sumi = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = (int) lane_q81_iq1s_grid_gpu()[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        sumi = dp4a((grid >> 0) & 0x0F0F0F0F, x.u[l0 + 0], sumi);
        sumi = dp4a((grid >> 4) & 0x0F0F0F0F, x.u[l0 + 1], sumi);
    }
    const float d1q = half_of(&bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    return d1q * (x.d * sumi + x.s * delta);
}
float dw23(const uint8_t* row, int kbx, int iqs, const XB& x) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) row + kbx;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int(bq4->qs, iqs + j);
        const I2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = dp4a(v.x, x.u[j + 0], sumi);
        sumi = dp4a(v.y, x.u[j + 4], sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    return half_of(&bq4->d) * x.d * sumi;
}
float dw17(const uint8_t* row, int kbx, int iqs, const XB& x) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) row + kbx;
    uint16_t q2[4];
    std::memcpy(q2, (const uint8_t*) bq2->qs + 4 * iqs, 8);
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        uint32_t gp[2];
        std::memcpy(gp, &iq2xs_grid[q2[l0 / 2] & 0x1FF], 8);
        const uint32_t signs = unpack_ksigns((uint8_t) (q2[l0 / 2] >> 9));
        const int signs0 = vcmpne4(signs & 0x08040201, 0);
        const int signs1 = vcmpne4(signs & 0x80402010, 0);
        const int grid_l = vsub4(gp[0] ^ signs0, signs0);
        const int grid_h = vsub4(gp[1] ^ signs1, signs1);
        if (l0 < 4) {
            sumi0 = dp4a(grid_l, x.u[l0 + 0], sumi0);
            sumi0 = dp4a(grid_h, x.u[l0 + 1], sumi0);
        } else {
            sumi1 = dp4a(grid_l, x.u[l0 + 0], sumi1);
            sumi1 = dp4a(grid_h, x.u[l0 + 1], sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    return half_of(&bq2->d) * x.d * sumi;
}
float dw22(const uint8_t* row, int kbx, int iqs, const XB& x) {
    const block_iq2_s* bq2 = (const block_iq2_s*) row + kbx;
    const int qs_packed = get_int(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        uint32_t gp[2];
        std::memcpy(gp, &iq2s_grid[qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)], 8);
        const int signs0 =
            vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 =
            vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = vsub4(gp[0] ^ signs0, signs0);
        const int grid_h = vsub4(gp[1] ^ signs1, signs1);
        if (l0 < 4) {
            sumi0 = dp4a(grid_l, x.u[l0 + 0], sumi0);
            sumi0 = dp4a(grid_h, x.u[l0 + 1], sumi0);
        } else {
            sumi1 = dp4a(grid_l, x.u[l0 + 0], sumi1);
            sumi1 = dp4a(grid_h, x.u[l0 + 1], sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    return half_of(&bq2->d) * x.d * sumi;
}
float dw21(const uint8_t* row, int kbx, int iqs, const XB& x) {
    const block_iq3_s* bq3 = (const block_iq3_s*) row + kbx;
    int qs_packed[2] = {get_int(bq3->qs, iqs + 0), get_int(bq3->qs, iqs + 1)};
    const uint8_t* qs = (const uint8_t*) qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t gx = iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
        const uint32_t gy = iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
        const int signs0 =
            vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 =
            vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = vsub4(gx ^ signs0, signs0);
        const int grid_h = vsub4(gy ^ signs1, signs1);
        sumi = dp4a(grid_l, x.u[l0 + 0], sumi);
        sumi = dp4a(grid_h, x.u[l0 + 1], sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    return half_of(&bq3->d) * x.d * sumi;
}
float dw29(const uint8_t* row, int kbx, int iqs, const XB& x) {
    const block_iq1_m* bq1 = (const block_iq1_m*) row + kbx;
    const int qs_packed = get_int(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = (int) lane_q81_iq1s_grid_gpu()[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        sumi[l0 / 4] = dp4a((grid >> 0) & 0x0F0F0F0F, x.u[l0 + 0], sumi[l0 / 4]);
        sumi[l0 / 4] = dp4a((grid >> 4) & 0x0F0F0F0F, x.u[l0 + 1], sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = dp4a(x.u[l0 + 0], 0x01010101, sumy);
        sumy = dp4a(x.u[l0 + 1], 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    uint16_t sc[4];
    std::memcpy(sc, bq1->scales, 8);
    const uint16_t su = (uint16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
    const float d = h2f(su) * x.d;
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}
float e8m0_to_fp32(uint8_t x) {
    const uint32_t bits = x == 0 ? 0x00400000u : (uint32_t) x << 23;
    return bits_f(bits);
}
float dw39(const uint8_t* row, int kbx, const XB& x) {
    const block_mxfp4* b = (const block_mxfp4*) row + kbx;
    int sumi = 0;
    for (int l = 0; l < 4; ++l) {
        const I2 v = get_int_from_table_16(get_int(b->qs, l), kvalues_mxfp4);
        sumi = dp4a(v.x, x.u[l], sumi);
        sumi = dp4a(v.y, x.u[l + 4], sumi);
    }
    return e8m0_to_fp32(b->e) * 0.5f * x.d * (float) sumi;
}
float ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (x >> 3) & 0xF, m = x & 7;
    const float v = e == 0 ? (float) m * (1.0f / 512.0f) : bits_f((uint32_t) (e + 120) << 23 | (uint32_t) m << 20);
    return 0.5f * v;
}
float dw40(const uint8_t* row, int kbx, int iqs, const XB& x) {
    const uint8_t* b = row + (size_t) kbx * 36;
    float acc = 0.0f;
    for (int h = 0; h < 2; ++h) {
        const int sb = 2 * iqs + h;
        const uint8_t* q = b + 4 + 8 * sb;
        const I2 v0 = get_int_from_table_16(get_int(q, 0), kvalues_mxfp4);
        const I2 v1 = get_int_from_table_16(get_int(q, 1), kvalues_mxfp4);
        int sumi = dp4a(v0.x, x.u[4 * h + 0], 0);
        sumi = dp4a(v1.x, x.u[4 * h + 1], sumi);
        sumi = dp4a(v0.y, x.u[4 * h + 2], sumi);
        sumi = dp4a(v1.y, x.u[4 * h + 3], sumi);
        acc += ue4m3_to_fp32(b[sb]) * (float) sumi;
    }
    return x.d * acc;
}
float dw20(const uint8_t* row, int kbx, const XB& x) {
    const block_iq4_nl* b = (const block_iq4_nl*) row + kbx;
    int sumi = 0;
    for (int l = 0; l < 4; ++l) {
        const I2 v = get_int_from_table_16(get_int(b->qs, l), kvalues_iq4nl);
        sumi = dp4a(v.x, x.u[l], sumi);
        sumi = dp4a(v.y, x.u[l + 4], sumi);
    }
    return half_of(&b->d) * x.d * (float) sumi;
}
float dw7(const uint8_t* row, int kbx, const XB& x) {
    const block_q5_1* b = (const block_q5_1*) row + kbx;
    const int qh = get_int(b->qh, 0);
    int sumi = 0;
    for (int i = 0; i < 4; ++i) {
        const int vl = get_int(b->qs, i), vh = qh >> (4 * i);
        int vi0 = vl & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = dp4a(vi0, x.u[i], sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = dp4a(vi1, x.u[i + 4], sumi);
    }
    const float dmx = half_of((const uint8_t*) b), dmy = half_of((const uint8_t*) b + 2);
    return dmx * x.d * (float) sumi + dmy * x.s;
}
float dw6(const uint8_t* row, int kbx, const XB& x) {
    const block_q5_0* b = (const block_q5_0*) row + kbx;
    const int qh = get_int(b->qh, 0);
    int sumi = 0;
    for (int i = 0; i < 4; ++i) {
        const int vl = get_int(b->qs, i), vh = qh >> (4 * i);
        int vi0 = vl & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = dp4a(vsub4(vi0, 0x10101010), x.u[i], sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = dp4a(vsub4(vi1, 0x10101010), x.u[i + 4], sumi);
    }
    return half_of(&b->d) * x.d * (float) sumi;
}
inline int q2_0_bytes(int q) {
    const int b = (q & 0x03) | ((q & 0x0C) << 6) | ((q & 0x30) << 12) | ((q & 0xC0) << 18);
    return vsub4(b, 0x01010101);
}
float dw42(const uint8_t* row, int kbx, int iqs, const XB& x) {
    const block_q2_0* b = (const block_q2_0*) row + kbx;
    uint16_t qs[4];
    std::memcpy(qs, b->qs + 8 * iqs, 8);
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        sumi = dp4a(q2_0_bytes(q & 0xFF), x.u[2 * j], sumi);
        sumi = dp4a(q2_0_bytes(q >> 8), x.u[2 * j + 1], sumi);
    }
    return half_of(&b->d) * x.d * (float) sumi;
}

// qk values a block, ipb calls a block, iqs = step * call, block bytes
struct Fmt {
    int qk, ipb, step;
    size_t bytes;
};
bool fmt_of(int t, Fmt& f) {
    switch (t) {
        case 12: f = {256, 16, 2, sizeof(block_q4_K)}; return true;
        case 13: f = {256, 16, 2, sizeof(block_q5_K)}; return true;
        case 14: f = {256, 32, 1, sizeof(block_q6_K)}; return true;
        case 8: f = {32, 4, 2, sizeof(block_q8_0)}; return true;
        case 16: f = {256, 8, 2, sizeof(block_iq2_xxs)}; return true;
        case 18: f = {256, 8, 2, sizeof(block_iq3_xxs)}; return true;
        case 19: f = {256, 8, 1, sizeof(block_iq1_s)}; return true;
        case 10: f = {256, 16, 1, sizeof(block_q2_K)}; return true;
        case 11: f = {256, 16, 1, sizeof(block_q3_K)}; return true;
        case 23: f = {256, 8, 4, sizeof(block_iq4_xs)}; return true;
        case 17: f = {256, 8, 2, sizeof(block_iq2_xs)}; return true;
        case 22: f = {256, 8, 2, sizeof(block_iq2_s)}; return true;
        case 21: f = {256, 8, 2, sizeof(block_iq3_s)}; return true;
        case 29: f = {256, 8, 1, sizeof(block_iq1_m)}; return true;
        case 39: f = {32, 1, 1, sizeof(block_mxfp4)}; return true;
        case 40: f = {64, 2, 1, 36}; return true;
        case 20: f = {32, 1, 1, sizeof(block_iq4_nl)}; return true;
        case 7: f = {32, 1, 1, sizeof(block_q5_1)}; return true;
        case 6: f = {32, 1, 1, sizeof(block_q5_0)}; return true;
        case 42: f = {64, 2, 1, sizeof(block_q2_0)}; return true;
        default: return false;
    }
}

// one call's float, as the device's F<T>::dot / dw<T> / dwk<T> (y: the weight block's first q8_1 block)
float call(int t, const uint8_t* row, const XB* y, int kbx, int iqs) {
    switch (t) {
        case 12: return dot_q4_K(row, y, kbx, iqs);
        case 13: return dot_q5_K(row, y, kbx, iqs);
        case 14: return dot_q6_K(row, y, kbx, iqs);
        case 8: return dot_q8_0(row, y, kbx, iqs);
        case 10: return dot_q2_K(row, y, kbx, iqs);
        case 11: return dot_q3_K(row, y, kbx, iqs);
        case 16: return dw16(row, kbx, iqs, y[iqs / 2]);
        case 18: return dw18(row, kbx, iqs, y[iqs / 2]);
        case 19: return dw19(row, kbx, iqs, y[iqs]);
        case 23: return dw23(row, kbx, iqs, y[iqs / 4]);
        case 17: return dw17(row, kbx, iqs, y[iqs / 2]);
        case 22: return dw22(row, kbx, iqs, y[iqs / 2]);
        case 21: return dw21(row, kbx, iqs, y[iqs / 2]);
        case 29: return dw29(row, kbx, iqs, y[iqs]);
        case 39: return dw39(row, kbx, y[0]);
        case 40: return dw40(row, kbx, iqs, y[iqs]);
        case 20: return dw20(row, kbx, y[0]);
        case 7: return dw7(row, kbx, y[0]);
        case 6: return dw6(row, kbx, y[0]);
        case 42: return dw42(row, kbx, iqs, y[iqs]);
        default: return 0.0f;
    }
}

void blocks_of(const void* act, int n, std::vector<XB>& xb) {
    const q81::View v = q81::view(act, n);
    xb.resize((size_t) n / 32);
    for (int b = 0; b < n / 32; ++b) {
        std::memcpy(xb[(size_t) b].u, v.q + 32 * b, 32);
        xb[(size_t) b].d = v.d[b];
        xb[(size_t) b].s = v.s[b];
    }
}

// glm_fast's warp_max / warp_sum over a block's 32 values (the sum in the butterfly's order: every lane ends with the
// same value, lane 0's taken)
void quant_block(const float* x, int8_t* q, float& d_out, float& s_out, int16_t* b8) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[i]));
    float t[32];
    std::memcpy(t, x, sizeof t);
    for (int o = 16; o > 0; o >>= 1) {
        float u[32];
        for (int i = 0; i < 32; ++i) u[i] = t[i] + t[i ^ o];
        std::memcpy(t, u, sizeof t);
    }
    const float sum = t[0];
    const float d = std::fmin(amax / 127.0f, 65504.0f);
    for (int i = 0; i < 32; ++i)
        q[i] = amax == 0.0f ? 0 : (int8_t) std::fmax(-127.0f, std::fmin(127.0f, std::round(x[i] / d)));
    d_out = h2f(q81::f2h(d));
    s_out = h2f(q81::f2h(std::fmax(-65504.0f, std::fmin(65504.0f, sum))));
    for (int g = 0; g < 4; ++g) {
        int s = 0;
        for (int i = 0; i < 8; ++i) s += q[8 * g + i];
        b8[g] = (int16_t) s;
    }
}

}  // namespace

// (lane_q81_avx2.cpp, lane_q81_vnni.cpp; x86 builds)
void lane_q81_quant_avx2(const float* x, int64_t n, void* dst);

bool lane_q81_type(int t) noexcept {
    Fmt f;
    return fmt_of(t, f);
}

size_t lane_q81_act_bytes(int64_t n) noexcept { return q81::bytes(n); }

void lane_q81_quant(const float* x, int64_t n, void* dst) {
#if defined(__x86_64__) || defined(_M_X64)
    static const bool avx2 = cpu_avx2_ok();
    if (avx2) {
        lane_q81_quant_avx2(x, n, dst);
        return;
    }
#endif
    const q81::MView m = q81::mview(dst, n);
    for (int64_t b = 0; b < n / 32; ++b) quant_block(x + 32 * b, m.q + 32 * b, m.d[b], m.s[b], m.b8 + 4 * b);
}

void lane_q81_from_blocks(const void* blocks, int64_t n, void* dst) {
    const q81::MView m = q81::mview(dst, n);
    const uint8_t* p = (const uint8_t*) blocks;
    for (int64_t b = 0; b < n / 32; ++b, p += 36) {
        uint16_t d, s;
        std::memcpy(&d, p, 2);
        std::memcpy(&s, p + 2, 2);
        m.d[b] = h2f(d);
        m.s[b] = h2f(s);
        std::memcpy(m.q + 32 * b, p + 4, 32);
        for (int g = 0; g < 4; ++g) {
            int sum = 0;
            for (int i = 0; i < 8; ++i) sum += m.q[32 * b + 8 * g + i];
            m.b8[4 * b + g] = (int16_t) sum;
        }
    }
}

double lane_q81_ref_dot(int t, const uint8_t* row, int n, const void* act) {
    Fmt f;
    if (!fmt_of(t, f)) return 0.0;
    std::vector<XB> xb;
    blocks_of(act, n, xb);
    double s = 0.0;
    for (int kbx = 0; kbx < n / f.qk; ++kbx)
        for (int c = 0; c < f.ipb; ++c) s += call(t, row, xb.data() + (size_t) kbx * (f.qk / 32), kbx, f.step * c);
    return s;
}

void lane_q81_rows_scalar(int t, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                          float* const* out, int r0, int r1) {
    Fmt f;
    if (!fmt_of(t, f)) return;
    thread_local std::vector<XB> xb;
    xb.resize((size_t) nt * (n / 32));
    for (int k = 0; k < nt; ++k) {
        std::vector<XB> one;
        blocks_of(act[k], n, one);
        std::copy(one.begin(), one.end(), xb.begin() + (size_t) k * (n / 32));
    }
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        for (int k = 0; k < nt; ++k) {
            const XB* y0 = xb.data() + (size_t) k * (n / 32);
            float s = 0.0f;
            for (int kbx = 0; kbx < n / f.qk; ++kbx)
                for (int c = 0; c < f.ipb; ++c) s += call(t, row, y0 + (size_t) kbx * (f.qk / 32), kbx, f.step * c);
            out[k][r] = s;
        }
    }
}

#if !(defined(__x86_64__) || defined(_M_X64))
// (no vector forms off x86: the portable rows)
bool lane_q81_avx2_type(int) noexcept { return false; }
void lane_q81_avx2_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {}
bool lane_q81_vbmi_type(int) noexcept { return false; }
void lane_q81_vbmi_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {}
#endif

bool lane_q81_fast(int t) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    static const bool avx512 = cpu_avx512_ok() && std::getenv("STRATA_NO_IQ512") == nullptr &&
                               std::getenv("STRATA_NO_IQ512_VBMI") == nullptr;
    static const bool avx2 = cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr;
    return (avx512 && lane_q81_vbmi_type(t)) || (avx2 && lane_q81_avx2_type(t));
#else
    (void) t;
    return false;
#endif
}

void lane_q81_rows(int t, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                   int r0, int r1) {
#if defined(__x86_64__) || defined(_M_X64)
    static const bool avx512 = cpu_avx512_ok() && std::getenv("STRATA_NO_IQ512") == nullptr &&
                               std::getenv("STRATA_NO_IQ512_VBMI") == nullptr;
    static const bool avx2 = cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr;
    if (avx512 && lane_q81_vbmi_type(t)) {
        lane_q81_vbmi_rows(t, w, row_bytes, n, act, nt, out, r0, r1);
        return;
    }
    if (avx2 && lane_q81_avx2_type(t)) {
#if defined(STRATA_LANE_Q81_VNNI)
        static const bool vnni = __builtin_cpu_supports("avxvnni") && std::getenv("STRATA_NO_AVXVNNI") == nullptr;
        if (vnni) {
            lane_q81_vnni_rows(t, w, row_bytes, n, act, nt, out, r0, r1);
            return;
        }
#endif
        lane_q81_avx2_rows(t, w, row_bytes, n, act, nt, out, r0, r1);
        return;
    }
#endif
    lane_q81_rows_scalar(t, w, row_bytes, n, act, nt, out, r0, r1);
}

}  // namespace strata::kernels::cpu

namespace strata::kernels::cpu {
void lane_q81_gu_rows(int gate_type, int up_type, const uint8_t* gate_base, size_t gate_row, const uint8_t* up_base,
                      size_t up_row, int n, const void* const* act, int nt, float* const* ff, int r0, int r1,
                      float limit, float scale) {
    if (r1 <= r0 || nt <= 0) return;
    const int nr = r1 - r0;
    thread_local std::vector<float> buf;
    thread_local std::vector<float*> go, uo;
    buf.resize((size_t) 2 * nt * nr);
    go.resize((size_t) nt);
    uo.resize((size_t) nt);
    for (int t = 0; t < nt; ++t) {
        go[(size_t) t] = buf.data() + (size_t) t * nr;
        uo[(size_t) t] = buf.data() + (size_t) (nt + t) * nr;
    }
    lane_q81_rows(gate_type, gate_base + (size_t) r0 * gate_row, gate_row, n, act, nt, go.data(), 0, nr);
    lane_q81_rows(up_type, up_base + (size_t) r0 * up_row, up_row, n, act, nt, uo.data(), 0, nr);
    for (int t = 0; t < nt; ++t)
        for (int r = 0; r < nr; ++r) {
            const float g = std::fmin(scale * go[(size_t) t][r], limit);
            const float u = std::fmin(std::fmax(scale * uo[(size_t) t][r], -limit), limit);
            ff[t][r0 + r] = (g / (1.0f + std::exp(-g))) * u;
        }
}
}  // namespace strata::kernels::cpu
