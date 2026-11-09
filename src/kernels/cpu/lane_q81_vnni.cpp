// src/kernels/cpu/lane_q81_vnni.cpp - lane_q81_avx2.cpp's rows built with AVX-VNNI (lane_q81_vnni_rows): one dpbusd
// for each maddubs + madd pair - the same integer sums.  lane_q81_rows picks it where the CPU has AVX-VNNI.
#define LANE_Q81_VNNI 1
#include "lane_q81_avx2.cpp"
