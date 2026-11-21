// src/kernels/cpu/native_expert.cpp - plan v0.3 P6: native (GGUF-form) experts on the CPU.
// See the header.  The activation quantizers and the row dot products are Strata's own (own_dot.hpp): the arithmetic
// of the x86 AVX2 dot products, so an expert computes what it computed when this file called ggml-cpu.  A build with
// ggml (STRATA_GGML=ON) can still take ggml-cpu's own (STRATA_CPU_GGML=1: the old path, for comparisons).
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/own_dot.hpp"
#include "strata/qtypes.hpp"
#include "ggml_cpu_oracle.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <vector>

namespace strata::kernels::cpu {
namespace {

// STRATA_CPU_GGML=1 (a build with ggml only): ggml-cpu's quantizers and dots, as before own_dot.  Unset: the own
// dots, except where they have no SIMD build (aarch64: the DGX Spark) and ggml-cpu is linked in - its NEON dots are
// the faster ones there; =0 takes the own ones anyway.
bool use_ggml() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_CPU_GGML");
        if (v == nullptr) return ggml_cpu_have() && !own_dot_simd();
        if (std::atoi(v) == 0) return false;
        if (ggml_cpu_have()) return true;
        std::fprintf(stderr, "native experts: STRATA_CPU_GGML=1 needs a build with ggml (STRATA_GGML=ON) - own dots\n");
        return false;
    }();
    return on;
}
int act_type(int type) { return use_ggml() ? ggml_cpu_vec_dot_type(type) : own_vec_dot_type(type); }
void quant(int at, const float* x, void* dst, int64_t n) {
    if (use_ggml()) ggml_cpu_quant(at, x, dst, n);
    else own_quant_act(at, x, dst, n);
}
// one row's dot: the reference the exact kernels are checked against, and the rows without a faster kernel
inline float dot1(int type, int n, const void* w, const void* a) {
    return use_ggml() ? ggml_cpu_dot(type, n, w, a) : own_dot_fast(type, n, w, a);
}
void init_once() {}

// iq512_rows / iq256_rows
using RowsFn = void (*)(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int);

// fn's rows against the reference dot (dot1: own_dot, or ggml-cpu's with STRATA_CPU_GGML=1), bit for bit: synthetic rows of `type` (random codes, every block's f16 scale
// - and a K-quant's min scale - a small normal value) and Q8_K activations from ggml's quantizer, `rows` rows for each
// token count of `nts` (the multi-token default 2, 5 and 11 - 11: the kernels' groups of 8 / 4 and their remainders).
// 306 results: a form off in a few % of them (Q5_K's fused min term under MSVC: ~7%) cannot pass by chance - 39 did
// in ~6% of starts
bool rows_match_ggml(RowsFn fn, int type, std::initializer_list<int> nts = {2, 5, 11}, int rows = 17) {
    const int at = act_type(type);
    const int n = 2048;   // eight 256-value blocks
    const size_t rb = qt::row_size(type, n), bb = qt::type_size(type), ab = qt::row_size(at, n);
    uint64_t st = 0x9E3779B97F4A7C15ull;
    const auto next = [&st] {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        return st;
    };
    std::vector<uint8_t> w(rb * rows);
    for (auto& b : w) b = (uint8_t) next();
    for (size_t o = 0; o < w.size(); o += bb) {
        const uint16_t d = qt::fp32_to_fp16(0.002f + 0.001f * (float) (next() % 16));
        std::memcpy(&w[o], &d, sizeof d);
        if (type == qt::Q4_K || type == qt::Q5_K) {
            const uint16_t m = qt::fp32_to_fp16(0.001f + 0.0005f * (float) (next() % 16));
            std::memcpy(&w[o + sizeof d], &m, sizeof m);
        }
    }
    for (int nt : nts) {
        std::vector<uint8_t> a(ab * (size_t) nt);
        std::vector<float> x((size_t) n), got((size_t) nt * rows), ref((size_t) nt * rows);
        std::vector<const void*> ap((size_t) nt);
        std::vector<float*> op((size_t) nt);
        for (int k = 0; k < nt; ++k) {
            for (auto& v : x) v = ((float) (next() % 2001) - 1000.0f) * 1e-3f;
            quant(at, x.data(), a.data() + ab * (size_t) k, n);
            ap[(size_t) k] = a.data() + ab * (size_t) k;
            op[(size_t) k] = got.data() + (size_t) k * rows;
        }
        fn(type, w.data(), rb, n, ap.data(), nt, op.data(), 0, rows);
        for (int k = 0; k < nt; ++k)
            for (int r = 0; r < rows; ++r)
                ref[(size_t) k * rows + r] = dot1(type, n, w.data() + (size_t) r * rb, ap[(size_t) k]);
        if (std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) != 0) return false;
    }
    return true;
}

// the VBMI rows against ggml's dot; IQ1_S's delta term in the form this build's ggml-cpu took (fused: GCC / Clang)
bool iq512v_match(int type, std::initializer_list<int> nts, int rows) {
    if (rows_match_ggml(iq512v_rows, type, nts, rows)) return true;
    if (type != qt::IQ1_S) return false;
    iq512v_set_iq1s_fused(false);   // (MSVC: the multiply, then the add)
    if (rows_match_ggml(iq512v_rows, type, nts, rows)) return true;
    iq512v_set_iq1s_fused(true);
    return false;
}

RowsFn pick_rows(int type) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_GLM_CPU_MT");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (!on) return nullptr;
    init_once();
    // AVX-512 where cpu_avx512_ok() (VNNI and VBMI: Ice Lake / Zen 4 and later, which run 512-bit integer code at
    // full clock); on a Skylake-SP Xeon 6152 the AVX-2 rows were 7-13% faster (glm_cpu_bench, Maya-L, 2-16 tokens)
    const bool avx512 = cpu_avx512_ok() && std::getenv("STRATA_NO_IQ512") == nullptr;
    const bool avx2 = cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr;
    RowsFn fn = nullptr;
    void (*set_q5k)(bool) noexcept = nullptr;
    // (E-3, see pick_rows1: the grid looked up in registers; a mismatch here falls back to iq512_rows.  IQ2_S's
    // 1024-entry grid takes 32 registers' worth of lookups there: iq512_rows is faster for it - L2-resident rows, a
    // Zen 5 core, 2 / 4 tokens 8.0 / 4.8 ns a block a token against 8.8 / 5.7)
    if (avx512 && iq512v_supported(type) && type != qt::IQ2_S && std::getenv("STRATA_NO_IQ512_VBMI") == nullptr &&
        iq512v_ready()) {
        if (iq512v_match(type, {2, 5, 11}, 17)) return iq512v_rows;
        std::fprintf(stderr, "native experts: the AVX-512 VBMI %s rows differ from ggml-cpu's dot in this build - "
                             "the AVX-512 ones\n", qt::type_name(type));
    }
    if (avx512 && iq512_supported(type)) {
        fn = iq512_rows;
        set_q5k = iq512_set_q5k_fused;
    } else if (avx2 && iq256_supported(type)) {
        fn = iq256_rows;
        set_q5k = iq256_set_q5k_fused;
    }
    if (fn == nullptr) return nullptr;
    if (rows_match_ggml(fn, type)) return fn;
    if (type == qt::Q5_K) {   // a ggml-cpu built without contraction (MSVC): its min term unfused
        set_q5k(false);
        if (rows_match_ggml(fn, type)) return fn;
        set_q5k(true);
    }
    std::fprintf(stderr, "native experts: the multi-token %s rows differ from ggml-cpu's dot in this build - "
                         "ggml's dots for them\n", qt::type_name(type));
    return nullptr;
}

// the multi-token rows for a weight type when they are exact here, nullptr otherwise (picked and checked once)
RowsFn exact_rows(int type) {
    if (type < 0 || type >= qt::COUNT) return nullptr;
    static std::once_flag once[qt::COUNT];
    static RowsFn fn[qt::COUNT];
    std::call_once(once[type], [type] { fn[type] = pick_rows(type); });
    return fn[type];
}

// The one-token rows (a decode step's CPU experts) for the formats whose ggml-cpu AVX2 dot rebuilds every value from
// lookups: IQ3_XXS's four keven_signs loads per 32 values, IQ3_S's 9-bit indices built in vectors and read back from
// memory.  iq256_rows1 is ggml's dot with two changes - IQ3_XXS's signs built in registers, IQ3_S's
// indices in scalar registers, IQ1_S's grid indices and delta sums in vectors (Qwen3.8-Flash-Next 12G32G L448, a
// gate/up expert from DRAM on the Ryzen AI Max+ 395: 1 thread 1142 -> 944 us, 8 threads 174 -> 142) - so it is
// bit-identical to ggml's dot (checked here once, 306 results).  On every AVX-2
// CPU, like ggml-cpu's own dot (it has no AVX-512 form for these).  STRATA_GLM_CPU_1T=0 keeps ggml's dot.
RowsFn pick_rows1(int type) {
    static const bool on = [] {
        const char* mt = std::getenv("STRATA_GLM_CPU_MT");
        const char* v = std::getenv("STRATA_GLM_CPU_1T");
        return (mt == nullptr || std::atoi(mt) != 0) && (v == nullptr || std::atoi(v) != 0);
    }();
    if (!on) return nullptr;
    init_once();
    // E-3: IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS in 512-bit lanes with the grid looked up in registers where
    // cpu_avx512_ok() (VNNI and VBMI; the multi-token rows too, pick_rows): bit-identical to ggml's dot, checked the
    // same way.  Ryzen AI Max+ 395 (Zen 5),
    // DeepSeek V4's experts (glm_cpu_bench): one thread an IQ2_XS / IQ3_XXS expert 1175 -> 665 us, an IQ2_XXS gate/up
    // 848 -> 424; 16 threads 109 -> 90-102 us and 111 -> 80 (now ~89 GB/s of the ~113 the RAM gives: memory-bound).
    // Decode without the iGPU lane (AI PRO R9700): DeepSeek V4 UD-Q2_K_XL 42.3-42.9 -> 44.8-45.6 tok/s, huihui Q2
    // +3-5%, GLM-5.3-Flash Maya-S-v2 +6-9%.  STRATA_NO_IQ512_VBMI=1 keeps the rows below.  Not IQ2_S and IQ1_S: for
    // one token ggml's dot is faster (L2-resident rows, a Zen 5 core: IQ2_S 9.3 against 16.6 ns a block - its 1024
    // codes take 32 registers' worth of lookups -, IQ1_S 9.4 against 12.2 - its 2048 qwords gathered)
    if (iq512v_supported(type) && type != qt::IQ2_S && type != qt::IQ1_S && cpu_avx512_ok() &&
        std::getenv("STRATA_NO_IQ512_VBMI") == nullptr && iq512v_ready()) {
        if (iq512v_match(type, {1}, 306)) return iq512v_rows;
        std::fprintf(stderr, "native experts: the AVX-512 one-token %s rows differ from ggml-cpu's dot in this build - "
                             "the AVX-2 ones\n", qt::type_name(type));
    }
    if (!iq256_rows1_supported(type)) return nullptr;
    if (!cpu_avx2_ok() || std::getenv("STRATA_NO_IQ256") != nullptr) return nullptr;
    if (rows_match_ggml(iq256_rows1, type, {1}, 306)) return iq256_rows1;
    if (type == qt::IQ1_S) {   // the delta term in the other form (MSVC's ggml: the multiply, then the add)
        iq256_set_iq1s_fused(false);
        if (rows_match_ggml(iq256_rows1, type, {1}, 306)) return iq256_rows1;
        iq256_set_iq1s_fused(true);
    }
    std::fprintf(stderr, "native experts: the one-token %s rows differ from ggml-cpu's dot in this build - "
                         "ggml's dot for them\n", qt::type_name(type));
    return nullptr;
}

RowsFn exact_rows1(int type) {
    if (type < 0 || type >= qt::COUNT) return nullptr;
    static std::once_flag once[qt::COUNT];
    static RowsFn fn[qt::COUNT];
    std::call_once(once[type], [type] { fn[type] = pick_rows1(type); });
    return fn[type];
}

// the exact rows for nt tokens of a weight type (nt >= 1), nullptr: ggml-cpu's dot
RowsFn exact_rows_nt(int type, int nt) { return nt >= 2 ? exact_rows(type) : exact_rows1(type); }

}  // namespace

bool native_experts_available() noexcept { return true; }

bool native_multi_rows_exact(int type) { return exact_rows(type) != nullptr; }

bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err) {
    return native_fmt(gu_type, gu_type, d_type, n_embd, n_ff, f, err);
}

bool native_fmt(int gu_type, int up_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err) {
    if (gu_type < 0 || gu_type >= qt::COUNT || d_type < 0 || d_type >= qt::COUNT || up_type < 0 ||
        up_type >= qt::COUNT) {
        err = "native experts: type " + std::to_string(gu_type) + "/" + std::to_string(d_type) + " is not a GGUF type";
        if (up_type != gu_type) err += " (up " + std::to_string(up_type) + ")";
        return false;
    }
    const int ag = act_type(gu_type), au = act_type(up_type), ad = act_type(d_type);
    if (ag < 0 || ad < 0) {
        err = "native experts: no CPU dot product for type " + std::to_string(ag < 0 ? gu_type : d_type);
        return false;
    }
    if (au < 0) {
        err = "native experts: no CPU dot product for the up type " + std::to_string(up_type);
        return false;
    }
    if (au != ag) {
        err = "native experts: gate type " + std::to_string(gu_type) + " and up type " + std::to_string(up_type) +
              " take different activation formats (one quantized activation serves both)";
        return false;
    }
    if (n_embd % qt::blck_size(gu_type) || n_embd % qt::blck_size(up_type) || n_ff % qt::blck_size(d_type) ||
        n_embd % qt::blck_size(ag) || n_ff % qt::blck_size(ad)) {
        err = "native experts: expert geometry is not whole blocks";
        return false;
    }
    f.gu_type = gu_type;
    f.up_type = up_type;
    f.d_type = d_type;
    f.gu_act = ag;
    f.d_act = ad;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_row = qt::row_size(gu_type, n_embd);
    f.up_row = qt::row_size(up_type, n_embd);
    f.d_row = qt::row_size(d_type, n_ff);
    f.up_off = f.gu_row * (size_t) n_ff;
    f.down_off = f.up_off + f.up_row * (size_t) n_ff;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    f.act_bytes = qt::row_size(ag, n_embd);
    f.h_bytes = qt::row_size(ad, n_ff);
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes) {
        err = "native experts: activation larger than the pool's buffers";
        return false;
    }
    return true;
}

void native_quant_act(const NativeFmt& f, const float* x, void* dst) {
    quant(f.gu_act, x, dst, f.n_embd);
}

int64_t native_act_blocks(const NativeFmt& f) {
    const int64_t b = qt::blck_size(f.gu_act);
    return b > 0 && f.n_embd % b == 0 ? f.n_embd / b : 0;
}

void native_quant_act_blocks(const NativeFmt& f, const float* x, void* dst, int64_t b0, int64_t b1) {
    if (b1 <= b0) return;
    const int64_t b = qt::blck_size(f.gu_act);
    quant(f.gu_act, x + b0 * b, (uint8_t*) dst + (size_t) b0 * qt::type_size(f.gu_act), (b1 - b0) * b);
}

int64_t native_h_blocks(const NativeFmt& f) {
    const int64_t b = qt::blck_size(f.d_act);
    return b > 0 && f.n_ff % b == 0 ? f.n_ff / b : 0;
}

void native_quant_h_blocks(const NativeFmt& f, const float* h, void* dst, int64_t b0, int64_t b1) {
    if (b1 <= b0) return;
    const int64_t b = qt::blck_size(f.d_act);
    quant(f.d_act, h + b0 * b, (uint8_t*) dst + (size_t) b0 * qt::type_size(f.d_act), (b1 - b0) * b);
}

void native_quant_h(const NativeFmt& f, const float* h, void* dst) {
    quant(f.d_act, h, dst, f.n_ff);
}

void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1) {
    // the multi-token kernels decode the weights once for all tokens: 2.0-2.4x ggml-cpu at three tokens, no faster
    // at one (all are bound by the codebook lookups, ~5 GB/s per core), measured by native_expert_parity.  AVX-512
    // first, then the AVX-2 one (Zen 2/3, Intel 12th-14th gen).  STRATA_NO_IQ512 drops an AVX-512 CPU to the
    // AVX-2 kernel, STRATA_NO_IQ256 drops the AVX-2 kernel; ggml-cpu's single-token vec_dot is reached only with
    // both set (and on a CPU without AVX-512, STRATA_NO_IQ512 changes nothing).
    static const bool avx512 = cpu_avx512_ok() && std::getenv("STRATA_NO_IQ512") == nullptr;
    static const bool avx2 = cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr;   // pre-AVX2: ggml's vec_dot
    if (nt >= 2 && !f.mixed() && iq512_supported(f.gu_type)) {   // one token: ggml-cpu is as fast or faster
        if (avx512) {
            iq512_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
        if (avx2) {
            iq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
    }
    const int n = (int) f.n_embd;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * f.gu_row;
        const uint8_t* ur = blob + f.up_off + (size_t) r * f.up_row;
        for (int t = 0; t < nt; ++t) {
            const float g = dot1(f.gu_type, n, gr, act[t]), u = dot1(f.up_type, n, ur, act[t]);
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

void native_gu_rows_split(const NativeFmt& f, const uint8_t* gate_base, const uint8_t* up_base,
                          const void* const* act, int nt, float* const* ff, int r0, int r1,
                          float swiglu_limit, float scale) {
    // scale: NVFP4's second-level scale of the gate and up rows (inside the SiLU; 1 for every other type)
    const int n = (int) f.n_embd;
    const bool situ = swiglu_limit < 0.0f;   // (Inkling's SITU: glm_fast.cu's situ_act)
    const bool clamp = !situ && swiglu_limit != std::numeric_limits<float>::infinity();
    const auto situ_act = [](float g, float u) {
        return (4.0f * std::tanh(g * 0.25f)) * (1.0f / (1.0f + std::exp(-g))) * (25.0f * std::tanh(u * 0.04f));
    };
    if (r1 > r0) {
        const RowsFn fn = exact_rows_nt(f.gu_type, nt);
        // (a mixed layer: up's own rows kernel - both exact, or ggml's dots below for both)
        const RowsFn ufn = f.mixed() ? exact_rows_nt(f.up_type, nt) : fn;
        if (fn != nullptr && ufn != nullptr) {
            // every gate row of the range for all tokens (each row decoded once per group of tokens), then every up
            // row, then the same epilogue as below on the same values
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
            fn(f.gu_type, gate_base + (size_t) r0 * f.gu_row, f.gu_row, n, act, nt, go.data(), 0, nr);
            ufn(f.up_type, up_base + (size_t) r0 * f.up_row, f.up_row, n, act, nt, uo.data(), 0, nr);
            for (int t = 0; t < nt; ++t) {
                for (int r = 0; r < nr; ++r) {
                    float g = scale * go[(size_t) t][r], u = scale * uo[(size_t) t][r];
                    if (clamp) {
                        g = std::fmin(g, swiglu_limit);
                        u = std::fmin(std::fmax(u, -swiglu_limit), swiglu_limit);
                    }
                    ff[t][r0 + r] = situ ? situ_act(g, u) : (g / (1.f + std::exp(-g))) * u;
                }
            }
            return;
        }
    }
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = gate_base + (size_t) r * f.gu_row;
        const uint8_t* ur = up_base + (size_t) r * f.up_row;
        for (int t = 0; t < nt; ++t) {
            float g = dot1(f.gu_type, n, gr, act[t]), u = dot1(f.up_type, n, ur, act[t]);
            g *= scale;
            u *= scale;
            if (clamp) {   // GLM's clamped swiglu (glm_ffn's semantics), matching the device path
                g = std::fmin(g, swiglu_limit);
                u = std::fmin(std::fmax(u, -swiglu_limit), swiglu_limit);
            }
            ff[t][r] = situ ? situ_act(g, u) : (g / (1.f + std::exp(-g))) * u;
        }
    }
}

void native_down_rows_split(const NativeFmt& f, const uint8_t* down_base, const void* const* hq,
                            int nt, float* const* out, int r0, int r1) {
    if (r1 > r0) {
        if (const RowsFn fn = exact_rows_nt(f.d_type, nt)) {
            thread_local std::vector<float*> o;
            o.resize((size_t) nt);
            for (int t = 0; t < nt; ++t) o[(size_t) t] = out[t] + r0;
            fn(f.d_type, down_base + (size_t) r0 * f.d_row, f.d_row, (int) f.n_ff, hq, nt, o.data(), 0, r1 - r0);
            return;
        }
    }
    const int n = (int) f.n_ff;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = down_base + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) out[t][r] = dot1(f.d_type, n, dr, hq[t]);
    }
}

void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1) {
    // IQ4_NL down rows: the AVX-2 multi-token kernel decodes the nibbles and absolutises the weights once per
    // block instead of once per token; ggml-cpu's dot is single-token.  STRATA_NO_IQ4NL falls back to it, and so
    // does a CPU without AVX2 (aarch64: the kernel is x86 only)
    static const bool iq4nl_mt = std::getenv("STRATA_NO_IQ4NL") == nullptr && cpu_avx2_ok();
    if (nt >= 2 && f.d_type == 20 && iq4nl_mt) {
        iq4nl256_down_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    const int n = (int) f.n_ff;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) out[t][r] = dot1(f.d_type, n, dr, hq[t]);
    }
}

}  // namespace strata::kernels::cpu
