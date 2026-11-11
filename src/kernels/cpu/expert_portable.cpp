// src/kernels/cpu/expert_portable.cpp - the CPU expert kernels' entry points on a CPU that is not x86 (aarch64: the
// DGX Spark's Grace).  CMakeLists.txt builds this file INSTEAD of expert.cpp, q2_avx2.cpp, iq_avx2.cpp,
// iq_avx512.cpp, iq_avx512_vbmi.cpp and kq_avx512.cpp, which are AVX code throughout.
//
// What runs here:
//   * the probes answer "no" (cpu_features(), kq_avx512_ok(), the *_supported() tables; expert_layout.cpp has the
//     cpu_*_ok() ones), so every caller that picks a kernel by them takes ggml-cpu's own dots - the native experts'
//     path (native_expert.cpp), which is what the GLM / DeepSeek / Qwen fast path's CPU lane runs - or the GPU;
//   * act_quant_q8_1 is the legacy contract's scalar loop, the one the AVX-512 version is bitwise equal to;
//   * every other entry point is an x86 kernel with no portable form: it stops the engine with a message saying so.
//     Nothing reaches them when the probes are honoured (generate.cpp refuses the packs that need them up front).
#include "strata/arch.hpp"

#if STRATA_X86
#error "expert_portable.cpp is the non-x86 build's; x86 builds expert.cpp and the AVX kernels"
#endif

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/kq_avx512.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels::cpu {
namespace {

std::atomic<bool> oracle_q8_0{false};

[[noreturn]] void x86_only(const char* what) {
    std::fprintf(stderr,
                 "maya: %s is an x86 CPU kernel (AVX2 / AVX-512) and this CPU is not x86 - it was called although "
                 "the CPU probes answer no; run with STRATA_GLM_CPU_LANE=0 so the GPUs compute every expert\n",
                 what);
    std::abort();
}

}  // namespace

// ---- expert.hpp ----

const char* CpuFeatures::reason() const { return usable() ? "ok" : "not an x86 CPU (no AVX-512)"; }

CpuFeatures cpu_features() { return CpuFeatures{}; }

void cpu_require_expert_support() {
    std::fprintf(stderr, "maya: this CPU cannot run the Q2_0 expert kernel: it is not x86 (the kernel needs "
                         "AVX512-VNNI and AVX512-VBMI)\n");
    std::abort();
}

void expert_set_oracle_q8_0(bool enabled) { oracle_q8_0.store(enabled, std::memory_order_relaxed); }

bool expert_oracle_q8_0_enabled() { return oracle_q8_0.load(std::memory_order_relaxed); }

void act_quant_q8_1(const float* x, int n, ActQ& a) {
    // (the oracle contract's quantizer is ggml's x86 one, transcribed in AVX: no portable form here)
    if (oracle_q8_0.load(std::memory_order_relaxed)) x86_only("the --cpu-oracle-q8-0 activation quantizer");
    // expert.cpp's scalar loop: max |x|, one multiply, +-0.5 away from zero, truncation, clamp
    a.nchunks = n / QKA;
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        float amax = 0.f;
        for (int j = 0; j < QKA; ++j) amax = std::fmax(amax, std::fabs(xb[j]));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32_t sum = 0;
        int8_t* q = a.q + k * QKA;
        for (int j = 0; j < QKA; ++j) {
            const float t = xb[j] * inv;
            const float r = t + (t >= 0.f ? 0.5f : -0.5f);
            int v = (int) r;
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            q[j] = (int8_t) v;
            sum += v;
        }
        a.scale[k] = s;
        a.sum[k] = sum;
        a.hx[k] = s * (float) sum;
    }
}

void s2_expert_vnni(const uint8_t*, const float*, float*, ExpertScratch&) { x86_only("s2_expert_vnni"); }
void s2_expert_vnni_q(const uint8_t*, const ActQ&, float*, ExpertScratch&) { x86_only("s2_expert_vnni_q"); }
void s2_expert_gu_rows(const uint8_t*, const ActQ&, float*, int, int) { x86_only("s2_expert_gu_rows"); }
void s2_expert_down_rows(const uint8_t*, const ActQ&, float*, int, int) { x86_only("s2_expert_down_rows"); }
void s2_expert_gu_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    x86_only("s2_expert_gu_rows_multi");
}
void s2_expert_down_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    x86_only("s2_expert_down_rows_multi");
}
void s2_expert_vnni_multi(const uint8_t*, const ActQ* const*, int, float* const*, ExpertScratchMulti&) {
    x86_only("s2_expert_vnni_multi");
}
void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    x86_only("q2_0_gguf_rows_multi");
}
void q2_0_gguf_rows_multi_avx2(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    x86_only("q2_0_gguf_rows_multi_avx2");
}
void act_quant_q8_1_avx2(const float*, int, ActQ&) { x86_only("act_quant_q8_1_avx2"); }
// (the test oracle: expert_parity, which skips on a CPU whose cpu_features() is not usable)
void s2_expert_scalar(const uint8_t*, const float*, float*, bool) { x86_only("s2_expert_scalar"); }

#if defined(STRATA_NATIVE_EXPERTS)
// ---- iq_avx2.hpp ----

bool iq256_supported(int) noexcept { return false; }
bool iq256_rows1_supported(int) noexcept { return false; }
void iq256_set_q5k_fused(bool) noexcept {}
void iq256_set_iq1s_fused(bool) noexcept {}
void iq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq256_gu_rows");
}
void iq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq256_rows");
}
void iq256_rows1(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq256_rows1");
}
void iq4nl256_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq4nl256_down_rows");
}

// ---- iq_avx512.hpp ----

bool iq512_supported(int) noexcept { return false; }
void iq512_set_q5k_fused(bool) noexcept {}
void iq512_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq512_gu_rows");
}
void iq512_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq512_rows");
}
bool iq512v_supported(int) noexcept { return false; }
bool iq512v_ready() noexcept { return false; }
void iq512v_set_iq1s_fused(bool) noexcept {}
void iq512v_set_lane_kernels(bool) noexcept {}
void iq512v_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("iq512v_rows");
}

// ---- kq_avx512.hpp ----

bool kq_avx512_ok() noexcept { return false; }
bool kq_type_ok(int t) noexcept { return t == 10 || t == 11; }
size_t kq_act_bytes(int, int) { x86_only("kq_act_bytes"); }
void kq_pack_act(const void* const*, int, int, void*) { x86_only("kq_pack_act"); }
void kq_rows_packed(int, const uint8_t*, size_t, int, const void*, int, float* const*, int, int) {
    x86_only("kq_rows_packed");
}
void kq_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    x86_only("kq_rows");
}
#endif

}  // namespace strata::kernels::cpu
