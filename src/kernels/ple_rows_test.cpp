// ple_rows_test.cpp - the PLE table's row decoders against hand-built rows: Q5_1 (unsloth's engram-Q5_1, 2026-11-04)
// with every 5-bit code once per block, so a wrong nibble or high-bit order shows; BF16 bit for bit.
#include "strata/kernels/ngram.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace k = strata::kernels;

static uint16_t f16_bits(float f) {   // exact for the small values used here
    if (f == 0.0f) return 0;
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const int e = (int) ((u >> 23) & 0xFF) - 127 + 15;
    return (uint16_t) (((u >> 16) & 0x8000) | (e << 10) | ((u >> 13) & 0x3FF));
}

int main() {
    int bad = 0;
    // Q5_1: block b has d = 0.5 * (b + 1), m = -b; element i of a block holds code (i * 7 + b) % 32
    std::vector<uint8_t> row(k::PLE_ROW_BYTES_Q5_1, 0);
    std::vector<float> want(k::PLE_HEAD_DIM), got(k::PLE_HEAD_DIM);
    for (int b = 0; b < k::PLE_HEAD_DIM / 32; ++b) {
        uint8_t* blk = row.data() + b * 24;
        const float d = 0.5f * (float) (b + 1), m = -(float) b;
        const uint16_t db = f16_bits(d), mb = f16_bits(m);
        std::memcpy(blk, &db, 2);
        std::memcpy(blk + 2, &mb, 2);
        uint32_t qh = 0;
        for (int i = 0; i < 32; ++i) {
            const int c = (i * 7 + b) % 32;
            want[b * 32 + i] = d * (float) c + m;
            if (c & 16) qh |= 1u << i;
            uint8_t& q = blk[8 + (i & 15)];
            q |= (uint8_t) ((c & 15) << (i < 16 ? 0 : 4));
        }
        std::memcpy(blk + 4, &qh, 4);
    }
    k::q5_1_dequant_row(row.data(), got.data());
    for (int i = 0; i < k::PLE_HEAD_DIM; ++i)
        if (got[i] != want[i]) {
            if (bad++ < 4) std::printf("q5_1 element %d: %g, want %g\n", i, got[i], want[i]);
        }
    // BF16: the top half of the float's bits
    std::vector<uint8_t> rb(k::PLE_ROW_BYTES_BF16);
    for (int i = 0; i < k::PLE_HEAD_DIM; ++i) {
        const float v = std::ldexp((float) (i - 80), -3);
        uint32_t u;
        std::memcpy(&u, &v, 4);
        const uint16_t h = (uint16_t) (u >> 16);
        std::memcpy(rb.data() + 2 * i, &h, 2);
        want[i] = v;
    }
    k::bf16_row(rb.data(), got.data());
    for (int i = 0; i < k::PLE_HEAD_DIM; ++i)
        if (got[i] != want[i]) {
            if (bad++ < 8) std::printf("bf16 element %d: %g, want %g\n", i, got[i], want[i]);
        }
    std::printf("ple_rows_test: %s (%d bad)\n", bad ? "FAIL" : "ok", bad);
    return bad ? 1 : 0;
}
