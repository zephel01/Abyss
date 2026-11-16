// src/kernels/cpu/own_dot.cpp - see include/strata/kernels/cpu/own_dot.hpp.  The portable build of the dot products
// (own_dot_body.hpp), the run-time pick of its AVX2 build, and the activation quantizers.
#define OD_NS portable
#include "own_dot_body.hpp"

#include "strata/kernels/cpu/own_dot.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/qtypes.hpp"

#include <algorithm>
#include <cstdlib>

namespace strata::kernels::cpu {
#if defined(__x86_64__) || defined(_M_X64)
namespace avx2 {
float dot(int type, int n, const void* w, const void* a);
float dot_simd(int type, int n, const void* w, const void* a);
bool has_simd(int type);
bool quant_simd(int act_type, const float* x, void* dst, int64_t n);
}
#endif

int own_vec_dot_type(int type) noexcept {
    switch (type) {
        case 2: case 6: case 8: case 20: case 39: case 40: case 42: return 8;
        case 3: case 7: return 9;
        case 10: case 11: case 12: case 13: case 14: case 16: case 17: case 18: case 19: case 21: case 22: case 23:
        case 29: return 15;
        default: return -1;
    }
}

namespace {
// round to nearest, ties to even (|v| < 2^22)
inline int rne(float v) {
    const float r = v + 12582912.0f - 12582912.0f;   // 1.5 x 2^23: the float adds round the fraction away
    return (int) r;
}

// Q8_0 / Q8_1: the block's largest magnitude maps to 127; d = max / 127 (stored as fp16), the codes are x x (127 /
// max) rounded to nearest even; Q8_1 also stores d x (the sum of the codes)
template <bool Q81>
void quant_q8_0(const float* x, void* dst, int64_t n) {
    uint8_t* o = (uint8_t*) dst;
    const size_t bb = Q81 ? sizeof(block_q8_1) : sizeof(block_q8_0);
    for (int64_t i = 0; i < n / 32; ++i, x += 32, o += bb) {
        float mx = 0.0f;
        for (int j = 0; j < 32; ++j) mx = std::max(mx, std::fabs(x[j]));
        const float d = mx / 127.0f;
        const float id = mx != 0.0f ? 127.0f / mx : 0.0f;
        int8_t* qs = (int8_t*) (o + (Q81 ? 4 : 2));
        int32_t sum = 0;
        for (int j = 0; j < 32; ++j) {
            const int q = rne(x[j] * id);
            qs[j] = (int8_t) q;
            sum += q;
        }
        const uint16_t dh = qt::fp32_to_fp16(d);
        std::memcpy(o, &dh, 2);
        if (Q81) {
            const uint16_t sh = qt::fp32_to_fp16(d * (float) sum);
            std::memcpy(o + 2, &sh, 2);
        }
    }
}
// Q8_K: max = the value of largest magnitude (the first one), iscale = -127 / max, codes = x x iscale rounded to
// nearest even (capped at 127), d = 1 / iscale, and the 16-value sums of the codes (an all-zero block: all zero)
void quant_q8_K(const float* x, void* dst, int64_t n) {
    block_q8_K* y = (block_q8_K*) dst;
    for (int64_t i = 0; i < n / 256; ++i, x += 256) {
        float amax = 0.0f, mx = 0.0f;
        for (int j = 0; j < 256; ++j) {
            const float a = std::fabs(x[j]);
            if (a > amax) { amax = a; mx = x[j]; }
        }
        if (amax == 0.0f) {
            std::memset(&y[i], 0, sizeof(block_q8_K));
            continue;
        }
        const float iscale = -127.0f / mx;
        for (int j = 0; j < 256; ++j) y[i].qs[j] = (int8_t) std::min(127, rne(iscale * x[j]));
        for (int j = 0; j < 16; ++j) {
            int s = 0;
            for (int k = 0; k < 16; ++k) s += y[i].qs[16 * j + k];
            y[i].bsums[j] = (int16_t) s;
        }
        y[i].d = 1 / iscale;
    }
}

bool use_avx2() {
#if defined(__x86_64__) || defined(_M_X64)
    static const bool on = cpu_avx2_ok() && std::getenv("STRATA_NO_OWN_AVX2") == nullptr;
    return on;
#else
    return false;
#endif
}
}  // namespace

bool own_dot_simd() noexcept { return use_avx2(); }

bool own_quant_act(int act_type, const float* x, void* dst, int64_t n) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    if (use_avx2()) return avx2::quant_simd(act_type, x, dst, n);
#endif
    switch (act_type) {
        case 8: quant_q8_0<false>(x, dst, n); return true;
        case 9: quant_q8_0<true>(x, dst, n); return true;
        case 15: quant_q8_K(x, dst, n); return true;
        default: return false;
    }
}

float own_dot(int type, int n, const void* w, const void* act) noexcept { return portable::dot(type, n, w, act); }

float own_dot_fast(int type, int n, const void* w, const void* act) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    if (use_avx2()) return avx2::has_simd(type) ? avx2::dot_simd(type, n, w, act) : avx2::dot(type, n, w, act);
#endif
    return portable::dot(type, n, w, act);
}

void own_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
              int r0, int r1) {
#if defined(__x86_64__) || defined(_M_X64)
    if (use_avx2()) {
        const bool s = avx2::has_simd(type);
        for (int r = r0; r < r1; ++r)
            for (int t = 0; t < nt; ++t) {
                const uint8_t* p = w + (size_t) r * row_bytes;
                out[t][r] = s ? avx2::dot_simd(type, n, p, act[t]) : avx2::dot(type, n, p, act[t]);
            }
        return;
    }
#endif
    for (int r = r0; r < r1; ++r)
        for (int t = 0; t < nt; ++t) out[t][r] = portable::dot(type, n, w + (size_t) r * row_bytes, act[t]);
}

}  // namespace strata::kernels::cpu
