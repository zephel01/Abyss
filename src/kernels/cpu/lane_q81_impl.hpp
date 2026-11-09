// src/kernels/cpu/lane_q81_impl.hpp - lane_q81's shared pieces: the Q81 activation's view and FP16 conversions.
#pragma once

#include <cstdint>
#include <cstring>

namespace strata::kernels::cpu::q81 {

// A Q81 activation of n values: q[n] int8 | d[n/32] float | s[n/32] float | b8[n/8] int16 (sums of 8 codes)
struct View {
    const int8_t* q;
    const float* d;
    const float* s;
    const int16_t* b8;
};
inline size_t bytes(int64_t n) { return (size_t) n + 8 * (size_t) (n / 32) + 2 * (size_t) (n / 8); }
inline View view(const void* a, int64_t n) {
    const uint8_t* p = (const uint8_t*) a;
    return View{(const int8_t*) p, (const float*) (p + n), (const float*) (p + n + 4 * (n / 32)),
                (const int16_t*) (p + n + 8 * (n / 32))};
}
struct MView {
    int8_t* q;
    float* d;
    float* s;
    int16_t* b8;
};
inline MView mview(void* a, int64_t n) {
    uint8_t* p = (uint8_t*) a;
    return MView{(int8_t*) p, (float*) (p + n), (float*) (p + n + 4 * (n / 32)), (int16_t*) (p + n + 8 * (n / 32))};
}

inline float bits_f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
inline uint32_t f_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// FP16 -> FP32, exact
inline float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) {
        const float v = (float) m * (1.0f / 16777216.0f);   // m * 2^-24
        return sign ? -v : v;
    }
    if (e == 31) return bits_f(sign | 0x7f800000u | (m << 13));
    return bits_f(sign | ((e + 112) << 23) | (m << 13));
}
// FP32 -> FP16, round to nearest even (CUDA's __float2half, F16C's cvtps_ph)
inline uint16_t f2h(float x) {
    uint32_t f = f_bits(x);
    const uint32_t sign = f & 0x80000000u;
    f ^= sign;
    uint16_t o;
    if (f >= (uint32_t) (127 + 16) << 23) {
        o = f > 0x7f800000u ? 0x7e00 : 0x7c00;
    } else if (f < (uint32_t) 113 << 23) {
        const uint32_t magic = (uint32_t) ((127 - 15) + (23 - 10) + 1) << 23;
        const float v = bits_f(f) + bits_f(magic);
        o = (uint16_t) (f_bits(v) - magic);
    } else {
        const uint32_t odd = (f >> 13) & 1;
        f += ((uint32_t) (15 - 127) << 23) + 0xfff;
        f += odd;
        o = (uint16_t) (f >> 13);
    }
    return (uint16_t) (o | (sign >> 16));
}

}  // namespace strata::kernels::cpu::q81
