// include/strata/kernels/pdl.hpp - Programmatic Dependent Launch (PDL) for the decode path's small kernels.
//
// A decode token is ~1,360 launches (Qwen3.8 q4x), most of them a few microseconds long: the gap between two of them
// (the next grid's launch, its blocks' dispatch) is paid every time.  With PDL (sm_90+) a kernel launched with
// cudaLaunchAttributeProgrammaticStreamSerialization may be dispatched while the kernel before it in the stream is
// still running; it then waits in griddepcontrol.wait until that kernel has completed and its writes are visible.
//
// The rule that keeps this exact: a kernel launched with the attribute (pdl_launch) must start with STRATA_PDL_ENTRY()
// - the wait first, before ANY global read or write, then the trigger that lets the kernel after it be dispatched.
// So every kernel still reads only what the kernels before it finished writing: the same arithmetic, the same bits.
// Without the attribute (PDL off, a device below sm_90, a stream being captured, HIP) both instructions are no-ops and
// pdl_launch is a plain <<<>>> launch.
//
// Default: on for an integrated GPU of compute capability 9.0+ (DGX Spark's GB10: Qwen3.8 16G64G 25.0 -> 30.7 tok/s
// without the drafter, 43.4 -> 46.5 with it) and for Qwen3.8 on a discrete sm_120+ card (RTX 5090: +7.8% without the
// drafter); off for other discrete cards and models.  STRATA_GLM_PDL=0/1 sets it.
#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include <cuda_runtime.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(STRATA_USE_HIP)
#define STRATA_PDL_ENTRY()                                        \
    do {                                                          \
        asm volatile("griddepcontrol.wait;" ::: "memory");        \
        asm volatile("griddepcontrol.launch_dependents;" :::);    \
    } while (0)
#else
#define STRATA_PDL_ENTRY() \
    do {                   \
    } while (0)
#endif

namespace strata::kernels {

// The model's say on a discrete card: Qwen3.8 (qwen4exp) turns PDL on by default for a discrete sm_120+ card
// (pdl_model_default_on, set once at load before any decode launch); every other model keeps it off there.  RTX 5090,
// Qwen3.8 UD 3.1 IQ3_XXS 4K: 130.3 -> 140.4 tok/s without the drafter, 186.5 -> 191.6 with MTP, tokens unchanged
inline std::atomic<bool>& pdl_model_default() {
    static std::atomic<bool> on{false};
    return on;
}
inline void pdl_model_default_on() { pdl_model_default().store(true, std::memory_order_relaxed); }

// PDL for the current device: sm_90+ and (STRATA_GLM_PDL=1, an integrated GPU, or a discrete sm_120+ card when the
// model asks for it); the device is checked once, the decision said once in the log when on
inline bool pdl_enabled() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    // STRATA_GLM_PDL=0/1: off / on; unset: on for an integrated GPU (GB10: launch latency is the decode's cost there)
    // and for a discrete sm_120+ card under Qwen3.8 (above), off for other discrete cards and models (RTX 5090 under
    // GLM / V4 / V4.1: not measured as a gain; sm_89 / sm_86: no PDL)
    static const int env = [] {
        const char* v = std::getenv("STRATA_GLM_PDL");
        return v == nullptr ? -1 : std::atoi(v) != 0 ? 1 : 0;
    }();
    if (env == 0) return false;
    constexpr int kDevs = 64;
    static std::atomic<signed char> state[kDevs];   // 0 not checked; else 1 + (sm_90+) + 2 (integrated) + 4 (sm_120+)
    static std::atomic<signed char> said[kDevs];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kDevs) return false;
    signed char st = state[dev].load(std::memory_order_relaxed);
    if (st == 0) {
        int major = 0, integrated = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev);
        st = (signed char) (8 | (major >= 9 ? 1 : 0) | (integrated != 0 ? 2 : 0) | (major >= 12 ? 4 : 0));
        state[dev].store(st, std::memory_order_relaxed);
    }
    const bool cc9 = (st & 1) != 0, igpu = (st & 2) != 0, cc12 = (st & 4) != 0;
    const bool model = cc12 && pdl_model_default().load(std::memory_order_relaxed);
    const bool on = cc9 && (env > 0 || igpu || model);
    if ((on || env > 0) && said[dev].load(std::memory_order_relaxed) == 0) {
        signed char expect = 0;
        if (said[dev].compare_exchange_strong(expect, 1))
            std::fprintf(stderr, "glm fast: CUDA%d programmatic dependent launch %s\n", dev,
                         on ? (env > 0 ? "on (STRATA_GLM_PDL=1)"
                                       : igpu ? "on (integrated GPU; STRATA_GLM_PDL=0: off)"
                                              : "on (Qwen3.8 on sm_120+; STRATA_GLM_PDL=0: off)")
                            : "off (STRATA_GLM_PDL=1: needs sm_90 or newer)");
    }
    return on;
#endif
}

// Stream capture: pdl_launch keeps the attribute while a stream is being captured only inside a PdlCapture scope on
// this thread - the graph then carries programmatic edges (CUDA 12.3+), the same rule as the plain launches.  Outside
// a scope a captured launch is plain, as before.  The scope is opened by the q4x verify window's per-layer graphs
inline int& pdl_capture_depth() {
    static thread_local int d = 0;
    return d;
}
struct PdlCapture {
    bool on;
    explicit PdlCapture(bool want) : on(want) { if (on) ++pdl_capture_depth(); }
    ~PdlCapture() { if (on) --pdl_capture_depth(); }
    PdlCapture(const PdlCapture&) = delete;
    PdlCapture& operator=(const PdlCapture&) = delete;
};

#if !defined(STRATA_USE_HIP)
// a launch on s may carry the PDL attribute: PDL on and the stream not captured, or captured inside a PdlCapture
inline bool pdl_attr_ok(cudaStream_t s) {
    if (!pdl_enabled()) return false;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(s, &cs) != cudaSuccess) return false;
    return cs == cudaStreamCaptureStatusNone || (cs == cudaStreamCaptureStatusActive && pdl_capture_depth() > 0);
}

// a kernel launch with the PDL attribute when pdl_attr_ok(s), else <<<>>>.  The kernel MUST begin with
// STRATA_PDL_ENTRY()
template <typename... KArgs, typename... Args>
inline void pdl_launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, Args&&... args) {
    if (pdl_attr_ok(s)) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = smem;
        cfg.stream = s;
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = at;
        cfg.numAttrs = 1;
        if (cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...) == cudaSuccess) return;
        (void) cudaGetLastError();   // (refused: launched the plain way below)
    }
    kernel<<<grid, block, smem, s>>>(std::forward<Args>(args)...);
}
#else
inline bool pdl_attr_ok(cudaStream_t) { return false; }
template <typename... KArgs, typename... Args>
inline void pdl_launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, Args&&... args) {
    kernel<<<grid, block, smem, s>>>(std::forward<Args>(args)...);
}
#endif

}  // namespace strata::kernels
