// include/strata/arch.hpp - the few CPU-architecture specifics the host code needs, in one place.
//
// STRATA_X86 is 1 on x86-64 / x86 (GCC, Clang, MSVC), 0 elsewhere (aarch64: the DGX Spark's Grace CPU).  The CPU
// expert kernels (src/kernels/cpu/*avx*.cpp, expert.cpp) are x86 only; CMakeLists.txt builds
// src/kernels/cpu/expert_portable.cpp in their place on other CPUs, and the probes in expert_layout.cpp answer "no".
//
// cpu_pause(): one spin-wait hint - x86's PAUSE (exactly the `_mm_pause()` the spin loops used before), aarch64's
// YIELD, nothing elsewhere.
// store_fence(): the fence the doorbells put between the CPU's result writes and the flag a GPU polls in pinned host
// memory - x86's SFENCE (exactly the `_mm_sfence()` used before: it also drains write-combining buffers), aarch64's
// `dmb oshst`, a sequentially consistent fence elsewhere.
//   What it orders, at every call site (session.cpp's layer doorbell, verify.cpp's window doorbell and publish_plan):
// this CPU's earlier STORES into pinned host memory (the CPU experts' output rows, the plan) before its later STORE of
// the flag, as seen by the GPU, which polls the flag and then reads the rows.  Store -> store only (nothing the CPU
// loads afterwards depends on it), and the reads are the GPU's - a device, i.e. an observer in the OUTER shareable
// domain, not the inner one `dmb ishst` covers.  That is exactly Linux's dma_wmb() (`dmb oshst`): descriptor writes
// before the ownership flag in coherent DMA memory (the GB10's NVLink-C2C, or a PCIe card's pinned pages, Normal
// memory either way - not MMIO, which would need wmb()'s `dsb st`).  DMB is cumulative, so the pool workers' rows,
// already observed by this thread through the pool's acquire join, are ordered too; the call sites' preceding
// std::atomic_thread_fence(seq_cst) stays for the threads' own ordering.  `dmb sy` (the first port) also ordered loads
// and the full system - more than the doorbell needs.
#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define STRATA_X86 1
#include <immintrin.h>
#else
#define STRATA_X86 0
#include <atomic>
#if defined(_M_ARM64)
#include <intrin.h>
#endif
#endif

namespace strata {

inline void cpu_pause() {
#if STRATA_X86
    _mm_pause();
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(_M_ARM64)
    __yield();
#endif
}

inline void store_fence() {
#if STRATA_X86
    _mm_sfence();
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    __asm__ __volatile__("dmb oshst" ::: "memory");
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

}  // namespace strata
