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
// `dmb sy` (full system barrier: orders the stores for every observer, the GPU across NVLink-C2C / PCIe included),
// a sequentially consistent fence elsewhere.
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
    __asm__ __volatile__("dmb sy" ::: "memory");
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

}  // namespace strata
