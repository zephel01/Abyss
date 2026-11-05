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
// without the drafter, 43.4 -> 46.5 with it), off for a discrete card (RTX 5090: noise).  STRATA_GLM_PDL=0/1 sets it.
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

// PDL for the current device: sm_90+ and (STRATA_GLM_PDL=1 or an integrated GPU); checked per device, once, and said
// once in the log when on
inline bool pdl_enabled() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    // STRATA_GLM_PDL=0/1: off / on; unset: on for an integrated GPU (GB10: launch latency is the decode's cost there),
    // off for a discrete card (RTX 5090: +0.6%, noise)
    static const int env = [] {
        const char* v = std::getenv("STRATA_GLM_PDL");
        return v == nullptr ? -1 : std::atoi(v) != 0 ? 1 : 0;
    }();
    if (env == 0) return false;
    constexpr int kDevs = 64;
    static std::atomic<signed char> state[kDevs];   // 0 not checked, 1 on, -1 off
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kDevs) return false;
    signed char st = state[dev].load(std::memory_order_relaxed);
    if (st == 0) {
        int major = 0, integrated = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev);
        st = major >= 9 && (env > 0 || integrated != 0) ? 1 : -1;
        signed char expect = 0;
        if (state[dev].compare_exchange_strong(expect, st) && (st > 0 || env > 0))
            std::fprintf(stderr, "glm fast: CUDA%d programmatic dependent launch %s\n", dev,
                         st > 0 ? (env > 0 ? "on (STRATA_GLM_PDL=1)" : "on (integrated GPU; STRATA_GLM_PDL=0: off)")
                                : "off (STRATA_GLM_PDL=1: needs sm_90 or newer)");
    }
    return st > 0;
#endif
}

#if !defined(STRATA_USE_HIP)
// a kernel launch with the PDL attribute when pdl_enabled() (and the stream is not being captured), else <<<>>>.
// The kernel MUST begin with STRATA_PDL_ENTRY().
template <typename... KArgs, typename... Args>
inline void pdl_launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, Args&&... args) {
    if (pdl_enabled()) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(s, &cs) == cudaSuccess && cs == cudaStreamCaptureStatusNone) {
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
    }
    kernel<<<grid, block, smem, s>>>(std::forward<Args>(args)...);
}
#else
template <typename... KArgs, typename... Args>
inline void pdl_launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, Args&&... args) {
    kernel<<<grid, block, smem, s>>>(std::forward<Args>(args)...);
}
#endif

}  // namespace strata::kernels
