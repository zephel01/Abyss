// src/kernels/cpu/expert_layout.cpp - plan v0.3 P6: the per-layer expert table.  See the header.
#include "strata/kernels/cpu/expert_layout.hpp"

#include "strata/arch.hpp"

#include <cstdio>
#include <cstdlib>
#if STRATA_X86
#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#else
#include <cpuid.h>
#endif
#endif
#include <fstream>
#include <sstream>

namespace strata::kernels::cpu {
namespace {
ExpertLayout g_layout;
}

const ExpertLayout& expert_layout() { return g_layout; }

#if STRATA_X86
bool cpu_avx512_ok() {
    static const bool ok = [] {
        if (const char* f = std::getenv("STRATA_FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        if (!((r[2] >> 27) & 1u)) return false;             // OSXSAVE
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0xE6) != 0xE6) return false;          // the OS saves the AVX-512 state
        cpuid(7, 0);
        const unsigned ebx = r[1], ecx = r[2];
        return ((ebx >> 16) & 1u) && ((ebx >> 30) & 1u) && ((ebx >> 31) & 1u) && ((ecx >> 11) & 1u) && ((ecx >> 1) & 1u);
    }();
    return ok;
}

bool cpu_avx512bw_ok() {
    static const bool ok = [] {
        if (const char* f = std::getenv("STRATA_FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        const unsigned ecx1 = r[2];
        if (!((ecx1 >> 27) & 1u) || !((ecx1 >> 12) & 1u) || !((ecx1 >> 29) & 1u)) return false;   // OSXSAVE, FMA, F16C
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0xE6) != 0xE6) return false;          // the OS saves the AVX-512 state
        cpuid(7, 0);
        const unsigned ebx = r[1];
        return ((ebx >> 16) & 1u) && ((ebx >> 17) & 1u) && ((ebx >> 30) & 1u) && ((ebx >> 31) & 1u);   // F DQ BW VL
    }();
    return ok;
}

// AVX2 + FMA with the OS saving the YMM state (cpuid as cpu_avx512_ok does, so MSVC builds too): the AVX2 kernels'
// gate - a CPU without it (Sandy/Ivy Bridge) takes ggml-cpu's own kernels
bool cpu_avx2_ok() {
    static const bool ok = [] {
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        const unsigned ecx1 = r[2];
        if (!((ecx1 >> 27) & 1u) || !((ecx1 >> 28) & 1u) || !((ecx1 >> 12) & 1u)) return false;   // OSXSAVE, AVX, FMA
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0x6) != 0x6) return false;             // the OS saves the XMM and YMM state
        cpuid(7, 0);
        return ((r[1] >> 5) & 1u) != 0;                    // AVX2
    }();
    return ok;
}

#if (defined(__GNUC__) && (__GNUC__ >= 11)) || (defined(__clang__) && (__clang_major__ >= 12)) || (defined(_MSC_VER) && _MSC_VER >= 1930)
#define STRATA_HAS_AVXVNNI 1
#else
#define STRATA_HAS_AVXVNNI 0
#endif

bool cpu_avxvnni_ok() {
#if !STRATA_HAS_AVXVNNI
    return false;
#else
    static const bool ok = [] {
        if (const char* f = std::getenv("STRATA_NO_AVXVNNI"); f != nullptr && f[0] == '1') return false;
        if (const char* f = std::getenv("STRATA_FORCE_NO_VNNI"); f != nullptr && f[0] == '1') return false;
        if (!cpu_avx2_ok()) return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(7, 0);
        if (r[0] < 1) return false;
        cpuid(7, 1);
        return ((r[0] >> 4) & 1u) != 0;                    // EAX bit 4: AVX-VNNI
    }();
    return ok;
#endif
}

// A CPU without AVX2 (and without AVX-512) has no kernel for these: say so instead of an illegal instruction.
// Run with the CPU expert lane off (STRATA_GLM_CPU_LANE=0) so the GPUs compute every expert.
[[noreturn]] static void no_simd_kernel(const char* what) {
    std::fprintf(stderr, "maya: %s needs AVX2 or AVX-512 and this CPU has neither; run with STRATA_GLM_CPU_LANE=0\n", what);
    std::abort();
}

void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1) {
    if (cpu_avx512_ok()) q2_0_gguf_rows_multi(w, row_bytes, nblocks, a, nt, out, r0, r1);
    else if (!cpu_avx2_ok()) no_simd_kernel("Q2_0 expert rows");
    else q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void act_quant_any(const float* x, int n, ActQ& a) {
    if (cpu_avx512_ok()) act_quant_q8_1(x, n, a);
    else if (!cpu_avx2_ok()) no_simd_kernel("the Q8_1 activation quantizer");
    else act_quant_q8_1_avx2(x, n, a);
}
#else
// Not an x86 CPU (aarch64: the DGX Spark's Grace): none of the AVX kernels is built (CMakeLists.txt compiles
// expert_portable.cpp in their place), so every probe answers no and the callers take ggml-cpu's own dots (native
// experts), the scalar activation quantizer, or the GPU.
bool cpu_avx512_ok() { return false; }
bool cpu_avx512bw_ok() { return false; }
bool cpu_avx2_ok() { return false; }
bool cpu_avxvnni_ok() { return false; }

void q2_rows_any(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    std::fprintf(stderr, "maya: the Q2_0 expert rows have x86 (AVX2 / AVX-512) kernels only and this CPU is not x86\n");
    std::abort();
}

// expert_portable.cpp's act_quant_q8_1 is the scalar loop, bit for bit what the AVX-512 one computes
void act_quant_any(const float* x, int n, ActQ& a) { act_quant_q8_1(x, n, a); }
#endif

bool parse_gu_field(const std::string& field, int& gate_type, int& up_type) {
    // "T" or "G,U" (decimal ggml types)
    const size_t c = field.find(',');
    const auto num = [](const std::string& t, int& v) {
        if (t.empty() || t.size() > 4) return false;
        v = 0;
        for (char ch : t) {
            if (ch < '0' || ch > '9') return false;
            v = 10 * v + (ch - '0');
        }
        return true;
    };
    if (c == std::string::npos) {
        if (!num(field, gate_type)) return false;
        up_type = gate_type;
        return true;
    }
    return num(field.substr(0, c), gate_type) && num(field.substr(c + 1), up_type);
}

#if !defined(STRATA_NATIVE_EXPERTS)
// Without ggml-cpu no native pack loads (expert_layout_load refuses), so these are never reached.
bool native_experts_available() noexcept { return false; }
bool native_fmt(int, int, int64_t, int64_t, NativeFmt&, std::string& err) { err = "built without native experts"; return false; }
bool native_fmt(int, int, int, int64_t, int64_t, NativeFmt&, std::string& err) { err = "built without native experts"; return false; }
void native_quant_act(const NativeFmt&, const float*, void*) { std::abort(); }
int64_t native_act_blocks(const NativeFmt&) { return 0; }
int64_t native_h_blocks(const NativeFmt&) { return 0; }
void native_quant_h_blocks(const NativeFmt&, const float*, void*, int64_t, int64_t) { std::abort(); }
void native_quant_act_blocks(const NativeFmt&, const float*, void*, int64_t, int64_t) { std::abort(); }
void native_quant_h(const NativeFmt&, const float*, void*) { std::abort(); }
void native_gu_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) { std::abort(); }
void native_down_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) { std::abort(); }
#endif

bool expert_layout_load(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    ExpertLayout L;
    L.n_layers = n_layers;
    L.n_expert = n_expert;
    std::ifstream in(pack_dir + "/native_experts.txt");
    if (!in) {
        L.total = (uint64_t) n_layers * (uint64_t) n_expert * (uint64_t) BLOB;
        g_layout = L;
        return true;
    }
#if !defined(STRATA_NATIVE_EXPERTS)
    err = "this pack has native (IQ) experts but the engine was built without STRATA_NATIVE_EXPERTS";
    return false;
#else
    L.native = true;
    L.fmt.resize((size_t) n_layers);
    L.offset.assign((size_t) n_layers, ~0ull);
    L.bytes.assign((size_t) n_layers, 0);
    L.max_blob = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            if (!line.empty() && line[0] == '#') {
                // v3 packs record their expert count in the header; a pruned model (the Qwen Coder) ships
                // fewer experts than the canonical geometry the caller passes, which is a compile-time
                // default, so the header wins.
                const size_t at = line.find("(n_expert ");
                if (at != std::string::npos) L.n_expert = std::atoll(line.c_str() + at + 10);
            }
            continue;
        }
        std::istringstream ss(line);
        long long l = -1, dt = -1;
        int gt = -1, ut = -1;
        std::string gu_field;
        unsigned long long off = 0, blob = 0, go = 0, uo = 0, dox = 0;
        if (!(ss >> l >> gu_field >> dt >> off >> blob) || l < 0 || l >= n_layers || !parse_gu_field(gu_field, gt, ut)) {
            err = "native_experts.txt: a malformed line: " + line;
            return false;
        }
        if (gt != ut) {   // (gate and up of different types: the --glm-pack fast path's kernels take them)
            err = "native_experts.txt: layer " + std::to_string(l) + "'s gate and up differ in type (" + gu_field +
                  "): this pack runs on the GLM-pack fast path only";
            return false;
        }
        NativeFmt f;
        if (!native_fmt((int) gt, (int) dt, H, FF, f, err)) return false;
        if (f.bytes != blob) {
            err = "native_experts.txt: layer " + std::to_string(l) + " blob is " + std::to_string(blob) +
                  " B but its formats make " + std::to_string(f.bytes);
            return false;
        }
        if (ss >> go >> uo >> dox) {   // v2 lines: the GGUF offsets
            if (L.gguf_off.empty()) L.gguf_off.assign((size_t) (3 * n_layers), 0);
            L.gguf_off[(size_t) (3 * l)] = go;
            L.gguf_off[(size_t) (3 * l + 1)] = uo;
            L.gguf_off[(size_t) (3 * l + 2)] = dox;
            std::string file;             // v3: the shard that holds this layer (a file name beside --native)
            if (ss >> file) {
                if (L.gguf_file.empty()) L.gguf_file.assign((size_t) n_layers, std::string());
                L.gguf_file[(size_t) l] = file;
            }
        }
        L.fmt[(size_t) l] = f;
        L.offset[(size_t) l] = off;
        L.bytes[(size_t) l] = blob;
        if (blob > L.max_blob) L.max_blob = blob;
    }
    uint64_t at = 0;
    for (int64_t l = 0; l < n_layers; ++l) {
        if (L.offset[(size_t) l] != at) {
            err = "native_experts.txt: layer " + std::to_string(l) + " is missing or not contiguous";
            return false;
        }
        at += L.bytes[(size_t) l] * (uint64_t) L.n_expert;
    }
    L.total = at;
    g_layout = L;
    return true;
#endif
}

}  // namespace strata::kernels::cpu
