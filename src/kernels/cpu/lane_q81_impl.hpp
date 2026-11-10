// src/kernels/cpu/lane_q81_impl.hpp - lane_q81's shared pieces: the Q81 activation's view and FP16 conversions.
#pragma once

#include <cmath>
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

// e^x in the steps glm_fast's lane_expf takes (the decode expert epilogue's SiLU; the CPU lane's too): Cody-Waite
// reduction by ln 2, a degree-7 Taylor polynomial in FMAs, the power of two built from bits.  ~1 ulp, and the same
// bits on the device and here (CUDA's expf goes through the SFU's ex2, which a CPU cannot reproduce)
inline float lane_expf(float x) {
    if (!(x > -87.0f)) return x != x ? x : 0.0f;
    if (x > 88.72f) return bits_f(0x7f800000u);
    const float kf = std::nearbyint(x * 1.44269504f);
    float r = std::fma(kf, -0.693359375f, x);
    r = std::fma(kf, 2.12194440e-4f, r);
    float p = 1.0f / 5040.0f;
    p = std::fma(p, r, 1.0f / 720.0f);
    p = std::fma(p, r, 1.0f / 120.0f);
    p = std::fma(p, r, 1.0f / 24.0f);
    p = std::fma(p, r, 1.0f / 6.0f);
    p = std::fma(p, r, 0.5f);
    p = std::fma(p, r, 1.0f);
    p = std::fma(p, r, 1.0f);
    const int k = (int) kf;
    if (k > 127) return p * bits_f(254u << 23) * 2.0f;
    return p * bits_f((uint32_t) (k + 127) << 23);
}

}  // namespace strata::kernels::cpu::q81

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
namespace strata::kernels::cpu::q81 {
// A row's dot in the DEVICE's float order (glm_fast rows_dot): call j (one 32-value q8_1 block: the weight's and the
// activation's scale product times the integer sum as a float) goes to lane j % 32, each lane FMA-accumulates its
// calls in order (nvcc contracts `acc += x * y`), then warp_sum's butterfly (lanes ^16, ^8, ^4, ^2, ^1).  The rows keep
// the 32 lanes as c[4] (c[q]: lanes 8q .. 8q + 7, i.e. 256-value group g's 8 blocks go to c[g % 4]), starting at +0;
// a lane without calls stays +0 as the device's does (and none is -0: x + y = 0 rounds to +0).  dev_sum: the butterfly.
inline float dev_sum(const __m256* c4) {
    const __m256 c = _mm256_add_ps(_mm256_add_ps(c4[0], c4[2]), _mm256_add_ps(c4[1], c4[3]));   // ^16, then ^8
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(c), _mm256_extractf128_ps(c, 1));             // ^4
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));                                                     // ^2
    r = _mm_add_ss(r, _mm_movehdup_ps(r));                                                      // ^1
    return _mm_cvtss_f32(r);
}
}  // namespace strata::kernels::cpu::q81
#endif
