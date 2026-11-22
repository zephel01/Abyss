// src/kernels/hip/own_gemm.cu - see include/strata/hip_compat/hip_gemm.hpp.
//
// C[i + j ldc] = alpha sum_k A[k + i lda] B[k + j ldb] + beta C: i < m runs along a weight's rows, j < n along the
// tokens, both operands are read along k (the cuBLAS call shape OP_T / OP_N the engine uses everywhere).
//
// Two kernels:
//   wmma - 16-bit inputs on the matrix cores, F32 accumulation.  A block holds 128 i x 128 j, eight waves of 32 i x
//          64 j (2 x 4 fragments of 16 x 16), k in steps of 32 through LDS with the next step's loads in registers.
//          The fragment's "M" side is the tokens (j) and its "N" side the weight rows (i), so a lane's results are one
//          weight row each and 16 lanes write 16 consecutive floats of C.
//            gfx11 (RDNA3 / 3.5, v_wmma_f32_16x16x16_{f16,bf16}): an operand lane l holds the 16 k of row l % 16 (both
//              halves of the wave the same); D lane l holds D[2 e + l / 16][l % 16], e < 8.
//            gfx12 (RDNA4, the _gfx12 forms): an operand lane l holds k 8 (l / 16) .. + 7 of row l % 16; D lane l
//              holds D[8 (l / 16) + e][l % 16].  The gfx12 default since #49's RX 9070 XT run (own_gemm_test --lib
//              40 / 40, prompts 1.4 - 1.8x the plain kernel); STRATA_HIP_GEMM_WMMA12=0 takes the plain one.
//   plain - any input type, FMA in F32 on a tile of 16 R x 16 R outputs (16 x 16 threads, R x R each, strided by 16
//          so that a row of threads reads and writes consecutive addresses), k in steps of 16 through LDS.  Its sum
//          is the plain k order.  The default for F32 inputs and for every architecture without a WMMA kernel.
// Both are deterministic (one fixed order a product, no atomics, no split k).
#include "strata/hip_compat/hip_gemm.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace strata::hipgemm {

struct Blas {
    hipStream_t stream = nullptr;
    void* ws = nullptr;
    size_t ws_bytes = 0;
    bool ws_set = false;
    hipblasMath_t math = HIPBLAS_DEFAULT_MATH;
    bool math_set = false;
    int device = 0;
    hipblasHandle_t lib = nullptr;   // created on the first product the own kernels do not take
};

namespace {

struct Args {
    const void* A;
    const void* B;
    void* C;
    int m, n, k, lda, ldb, ldc;
    long long sA, sB, sC;
    float alpha, beta;
};

typedef _Float16 hv8 __attribute__((ext_vector_type(8)));
typedef _Float16 hv16 __attribute__((ext_vector_type(16)));
typedef short sv8 __attribute__((ext_vector_type(8)));
typedef short sv16 __attribute__((ext_vector_type(16)));
typedef float fv8 __attribute__((ext_vector_type(8)));

#if defined(__GFX11__)
#define ABYSS_HG_W11 1
#else
#define ABYSS_HG_W11 0
#endif
#if defined(__GFX12__)
#define ABYSS_HG_W12 1
#else
#define ABYSS_HG_W12 0
#endif

__device__ __forceinline__ float load_f(const float* p) { return *p; }
__device__ __forceinline__ float load_f(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float load_f(const uint16_t* p) {   // (bf16 bits)
    return __uint_as_float((uint32_t) *p << 16);
}
__device__ __forceinline__ void store_c(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_c(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ float load_c(const float* p) { return *p; }
__device__ __forceinline__ float load_c(const __half* p) { return __half2float(*p); }

// the block's tile (i, j): bands of 8 i tiles walk all j before the next band, so that the blocks resident at once
// share their weight and token tiles in the cache (in launch order, a whole i column would read every weight tile)
__device__ __forceinline__ void tile_of(int& bi, int& bj) {
    const int gx = (int) gridDim.x, gy = (int) gridDim.y, G = 8;
    const int lin = (int) blockIdx.y * gx + (int) blockIdx.x, per = G * gy, first = lin / per * G;
    const int gs = gx - first < G ? gx - first : G, in = lin % per;
    bi = first + in % gs;
    bj = in / gs;
}

// ================================================================ wmma
constexpr int WI = 128, WJ = 128, WK = 32, WSTR = WK + 8;   // LDS row stride in 16-bit values (80 bytes)
constexpr int WNT = 256;

// one 16 x 16 x 16 product, a lane's operands as the 16 (gfx11) or 8 (gfx12) values it holds
template <bool BF>
__device__ __forceinline__ fv8 mma(const uint4* a, const uint4* b, fv8 c) {
#if ABYSS_HG_W11
    sv16 av, bv;
    __builtin_memcpy(&av, a, 32);
    __builtin_memcpy(&bv, b, 32);
    if constexpr (BF) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(av, bv, c);
    } else {
        hv16 ah, bh;
        __builtin_memcpy(&ah, &av, 32);
        __builtin_memcpy(&bh, &bv, 32);
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, bh, c);
    }
#elif ABYSS_HG_W12
    sv8 av, bv;
    __builtin_memcpy(&av, a, 16);
    __builtin_memcpy(&bv, b, 16);
    if constexpr (BF) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av, bv, c);
    } else {
        hv8 ah, bh;
        __builtin_memcpy(&ah, &av, 16);
        __builtin_memcpy(&bh, &bv, 16);
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, bh, c);
    }
#else
    (void) a; (void) b;
    __builtin_trap();   // (no WMMA here: the host never picks this kernel)
    return c;
#endif
}

// 8 consecutive 16-bit values of row `r` (< rows) from k0 + kc, zeros past the ends
template <bool VEC>
__device__ __forceinline__ uint4 load8(const uint16_t* base, int ld, int r, int rows, int kk, int K) {
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r >= rows) return v;
    const uint16_t* p = base + (size_t) r * ld + kk;
    if (VEC && kk + 8 <= K) return *reinterpret_cast<const uint4*>(p);
    uint16_t t[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) t[e] = kk + e < K ? p[e] : (uint16_t) 0;
    __builtin_memcpy(&v, t, 16);
    return v;
}

template <bool BF, typename TO, bool VEC>
__global__ void __launch_bounds__(WNT) wmma_kernel(const Args a) {
    __shared__ __align__(16) uint16_t sa[WI * WSTR];   // the weight rows (i), k contiguous
    __shared__ __align__(16) uint16_t sb[WJ * WSTR];   // the token rows (j)
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, r16 = lane & 15, hi = lane >> 4;
    int bi, bj;
    tile_of(bi, bj);
    const int i0 = bi * WI, j0 = bj * WJ;
    const uint16_t* A = (const uint16_t*) a.A + (size_t) blockIdx.z * a.sA + (size_t) i0 * a.lda;
    const uint16_t* B = (const uint16_t*) a.B + (size_t) blockIdx.z * a.sB + (size_t) j0 * a.ldb;
    TO* C = (TO*) a.C + (size_t) blockIdx.z * a.sC;
    const int mi = a.m - i0, nj = a.n - j0, K = a.k;
    const int wi = (wave & 3) * 32, wj = (wave >> 2) * 64;
    // loads: thread t takes rows t / 4 and t / 4 + 64 of both tiles, 8 values from (t % 4) * 8
    const int lr = tid >> 2, lc = (tid & 3) * 8;
    uint4 ra[2], rb[2];
    const auto fetch = [&](int k0) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            ra[q] = load8<VEC>(A, a.lda, lr + 64 * q, mi, k0 + lc, K);
            rb[q] = load8<VEC>(B, a.ldb, lr + 64 * q, nj, k0 + lc, K);
        }
    };
    const auto put = [&]() {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            *reinterpret_cast<uint4*>(&sa[(lr + 64 * q) * WSTR + lc]) = ra[q];
            *reinterpret_cast<uint4*>(&sb[(lr + 64 * q) * WSTR + lc]) = rb[q];
        }
    };
    fv8 acc[4][2];
#pragma unroll
    for (int x = 0; x < 4; ++x)
#pragma unroll
        for (int y = 0; y < 2; ++y)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[x][y][e] = 0.0f;
    fetch(0);
    put();
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += WK) {
        const bool more = k0 + WK < K;
        if (more) fetch(k0 + WK);
#pragma unroll
        for (int s = 0; s < WK / 16; ++s) {
#if ABYSS_HG_W12
            const int off = s * 16 + 8 * hi;   // a lane's half of the 16 k
            uint4 fa[4], fb[2];
#pragma unroll
            for (int x = 0; x < 4; ++x) fa[x] = *reinterpret_cast<const uint4*>(&sb[(wj + 16 * x + r16) * WSTR + off]);
#pragma unroll
            for (int y = 0; y < 2; ++y) fb[y] = *reinterpret_cast<const uint4*>(&sa[(wi + 16 * y + r16) * WSTR + off]);
#pragma unroll
            for (int x = 0; x < 4; ++x)
#pragma unroll
                for (int y = 0; y < 2; ++y) acc[x][y] = mma<BF>(&fa[x], &fb[y], acc[x][y]);
#else
            const int off = s * 16;
            uint4 fa[4][2], fb[2][2];
#pragma unroll
            for (int x = 0; x < 4; ++x) {
                const uint4* p = reinterpret_cast<const uint4*>(&sb[(wj + 16 * x + r16) * WSTR + off]);
                fa[x][0] = p[0];
                fa[x][1] = p[1];
            }
#pragma unroll
            for (int y = 0; y < 2; ++y) {
                const uint4* p = reinterpret_cast<const uint4*>(&sa[(wi + 16 * y + r16) * WSTR + off]);
                fb[y][0] = p[0];
                fb[y][1] = p[1];
            }
#pragma unroll
            for (int x = 0; x < 4; ++x)
#pragma unroll
                for (int y = 0; y < 2; ++y) acc[x][y] = mma<BF>(fa[x], fb[y], acc[x][y]);
#endif
        }
        __syncthreads();
        if (more) {
            put();
            __syncthreads();
        }
    }
    (void) hi;
#pragma unroll
    for (int x = 0; x < 4; ++x)
#pragma unroll
        for (int y = 0; y < 2; ++y) {
            const int i = wi + 16 * y + r16;
            if (i >= mi) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
#if ABYSS_HG_W12
                const int j = wj + 16 * x + 8 * hi + e;
#else
                const int j = wj + 16 * x + 2 * e + hi;
#endif
                if (j >= nj) continue;
                TO* p = C + (size_t) (j0 + j) * a.ldc + (i0 + i);
                float v = a.alpha * acc[x][y][e];
                if (a.beta != 0.0f) v += a.beta * load_c(p);
                store_c(p, v);
            }
        }
}

// ================================================================ plain
constexpr int PK = 16;

template <typename TI, int R, bool VEC>
__device__ __forceinline__ void plain_fetch(const TI* base, int ld, int rows, int K, int k0, int tid, float (&v)[R / 4][4]) {
    // R * 16 rows x 16 k = 256 R values: thread t takes R / 4 chunks of 4 consecutive k
#pragma unroll
    for (int q = 0; q < R / 4; ++q) {
        const int c = tid + 256 * q, r = c >> 2, kk = k0 + (c & 3) * 4;
        if (r >= rows) {
            v[q][0] = v[q][1] = v[q][2] = v[q][3] = 0.0f;
            continue;
        }
        const TI* p = base + (size_t) r * ld + kk;
        if (VEC && kk + 4 <= K) {
            if constexpr (sizeof(TI) == 4) {
                const float4 f = *reinterpret_cast<const float4*>(p);
                v[q][0] = f.x; v[q][1] = f.y; v[q][2] = f.z; v[q][3] = f.w;
            } else {
                const uint2 u = *reinterpret_cast<const uint2*>(p);
                TI t[4];
                __builtin_memcpy(t, &u, 8);
#pragma unroll
                for (int e = 0; e < 4; ++e) v[q][e] = load_f(&t[e]);
            }
        } else {
#pragma unroll
            for (int e = 0; e < 4; ++e) v[q][e] = kk + e < K ? load_f(p + e) : 0.0f;
        }
    }
}

template <typename TI, typename TO, int R, bool VEC>
__global__ void __launch_bounds__(256) plain_kernel(const Args a) {
    static_assert(R % 4 == 0, "R: 4 or 8");
    constexpr int T = 16 * R;
    __shared__ float sa[PK][T + 4], sb[PK][T + 4];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    int bi, bj;
    tile_of(bi, bj);
    const int i0 = bi * T, j0 = bj * T;
    const TI* A = (const TI*) a.A + (size_t) blockIdx.z * a.sA + (size_t) i0 * a.lda;
    const TI* B = (const TI*) a.B + (size_t) blockIdx.z * a.sB + (size_t) j0 * a.ldb;
    TO* C = (TO*) a.C + (size_t) blockIdx.z * a.sC;
    const int mi = a.m - i0, nj = a.n - j0, K = a.k;
    float acc[R][R];
#pragma unroll
    for (int x = 0; x < R; ++x)
#pragma unroll
        for (int y = 0; y < R; ++y) acc[x][y] = 0.0f;
    float va[R / 4][4], vb[R / 4][4];
    const auto put = [&]() {
#pragma unroll
        for (int q = 0; q < R / 4; ++q) {
            const int c = tid + 256 * q, r = c >> 2, kc = (c & 3) * 4;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                sa[kc + e][r] = va[q][e];
                sb[kc + e][r] = vb[q][e];
            }
        }
    };
    plain_fetch<TI, R, VEC>(A, a.lda, mi, K, 0, tid, va);
    plain_fetch<TI, R, VEC>(B, a.ldb, nj, K, 0, tid, vb);
    put();
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += PK) {
        const bool more = k0 + PK < K;
        if (more) {
            plain_fetch<TI, R, VEC>(A, a.lda, mi, K, k0 + PK, tid, va);
            plain_fetch<TI, R, VEC>(B, a.ldb, nj, K, k0 + PK, tid, vb);
        }
        // (unrolled further the compiler hoists every step's LDS reads ahead and spills the accumulators)
#pragma unroll 2
        for (int kk = 0; kk < PK; ++kk) {
            float ai[R], bj[R];
#pragma unroll
            for (int x = 0; x < R; ++x) ai[x] = sa[kk][tx + 16 * x];
#pragma unroll
            for (int y = 0; y < R; ++y) bj[y] = sb[kk][ty + 16 * y];
#pragma unroll
            for (int x = 0; x < R; ++x)
#pragma unroll
                for (int y = 0; y < R; ++y) acc[x][y] = __builtin_fmaf(ai[x], bj[y], acc[x][y]);
        }
        __syncthreads();
        if (more) {
            put();
            __syncthreads();
        }
    }
#pragma unroll
    for (int y = 0; y < R; ++y) {
        const int j = ty + 16 * y;
        if (j >= nj) continue;
#pragma unroll
        for (int x = 0; x < R; ++x) {
            const int i = tx + 16 * x;
            if (i >= mi) continue;
            TO* p = C + (size_t) (j0 + j) * a.ldc + (i0 + i);
            float v = a.alpha * acc[x][y];
            if (a.beta != 0.0f) v += a.beta * load_c(p);
            store_c(p, v);
        }
    }
}

// ================================================================ host
enum class Kern { plain, wmma11, wmma12 };

bool env_off(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '0' && v[1] == 0;
}
bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != 0 && !(v[0] == '0' && v[1] == 0);
}

constexpr int kMaxDev = 64;
std::atomic<int> g_kern[kMaxDev];   // 0: not looked up yet, else 1 + Kern
std::atomic<bool> g_said[kMaxDev];

Kern kern_of(int dev) {
    if (dev < 0 || dev >= kMaxDev) return Kern::plain;
    const int c = g_kern[dev].load(std::memory_order_relaxed);
    if (c != 0) return (Kern) (c - 1);
    Kern k = Kern::plain;
    std::string arch = "?";
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, dev) == hipSuccess) {
        arch = p.gcnArchName;
        const size_t colon = arch.find(':');
        if (colon != std::string::npos) arch.resize(colon);
        if (env_on("STRATA_HIP_GEMM_PORTABLE")) k = Kern::plain;
        else if (arch.rfind("gfx11", 0) == 0) k = Kern::wmma11;
        else if (arch.rfind("gfx12", 0) == 0 && !env_off("STRATA_HIP_GEMM_WMMA12")) k = Kern::wmma12;
    }
    g_kern[dev].store(1 + (int) k, std::memory_order_relaxed);
    if (!g_said[dev].exchange(true))
        std::fprintf(stderr, "hip gemm: HIP%d (%s) on the own GEMM - 16-bit inputs: %s, F32: plain%s\n", dev,
                     arch.c_str(),
                     k == Kern::wmma11 ? "WMMA (gfx11)" : k == Kern::wmma12 ? "WMMA (gfx12)" : "plain",
                     arch.rfind("gfx12", 0) == 0 && k == Kern::wmma12 ? " (STRATA_HIP_GEMM_WMMA12=0: the plain kernel)" : "");
    return k;
}

bool aligned(const void* p, size_t bytes) { return ((uintptr_t) p % bytes) == 0; }

template <typename TI, typename TO>
void launch_plain(const Args& a, int batch, hipStream_t s) {
    const bool vec = aligned(a.A, 4 * sizeof(TI)) && aligned(a.B, 4 * sizeof(TI)) && a.lda % 4 == 0 &&
                     a.ldb % 4 == 0 && a.sA % 4 == 0 && a.sB % 4 == 0;
    // the 128-wide tile where it still gives the card a few blocks, else the 64-wide one
    const long long big = (long long) ((a.m + 127) / 128) * ((a.n + 127) / 128) * batch;
    if (big >= 64) {
        const dim3 g((unsigned) ((a.m + 127) / 128), (unsigned) ((a.n + 127) / 128), (unsigned) batch);
        if (vec) plain_kernel<TI, TO, 8, true><<<g, 256, 0, s>>>(a);
        else plain_kernel<TI, TO, 8, false><<<g, 256, 0, s>>>(a);
    } else {
        const dim3 g((unsigned) ((a.m + 63) / 64), (unsigned) ((a.n + 63) / 64), (unsigned) batch);
        if (vec) plain_kernel<TI, TO, 4, true><<<g, 256, 0, s>>>(a);
        else plain_kernel<TI, TO, 4, false><<<g, 256, 0, s>>>(a);
    }
}

template <bool BF, typename TO>
void launch_wmma(const Args& a, int batch, hipStream_t s) {
    const bool vec = aligned(a.A, 16) && aligned(a.B, 16) && a.lda % 8 == 0 && a.ldb % 8 == 0 && a.sA % 8 == 0 &&
                     a.sB % 8 == 0;
    const dim3 g((unsigned) ((a.m + WI - 1) / WI), (unsigned) ((a.n + WJ - 1) / WJ), (unsigned) batch);
    if (vec) wmma_kernel<BF, TO, true><<<g, WNT, 0, s>>>(a);
    else wmma_kernel<BF, TO, false><<<g, WNT, 0, s>>>(a);
}

int stream_device(hipStream_t s) {
    int dev = 0;
    if (s == nullptr || hipStreamGetDevice(s, &dev) != hipSuccess) {
        (void) hipGetLastError();
        if (hipGetDevice(&dev) != hipSuccess) dev = 0;
    }
    return dev;
}

std::mutex g_lib_mu;

hipblasStatus_t library(Blas* h, hipblasHandle_t* out, const char* what) {
    if (h->lib == nullptr) {
        std::lock_guard<std::mutex> lock(g_lib_mu);
        if (h->lib == nullptr) {
            static std::atomic<bool> said{false};
            if (!said.exchange(true))
                std::fprintf(stderr, "hip gemm: %s goes to hipBLAS (%s)\n", what,
                             own_on() ? "not covered by the own GEMM" : "STRATA_HIP_OWN_GEMM=0");
            int cur = 0;
            (void) hipGetDevice(&cur);
            if (cur != h->device) (void) hipSetDevice(h->device);
            hipblasHandle_t lib = nullptr;
            const hipblasStatus_t st = hipblasCreate(&lib);
            if (st == HIPBLAS_STATUS_SUCCESS) {
                hipblasSetStream(lib, h->stream);
                if (h->ws_set) hipblasSetWorkspace(lib, h->ws, h->ws_bytes);
                if (h->math_set) hipblasSetMathMode(lib, h->math);
            }
            if (cur != h->device) (void) hipSetDevice(cur);
            if (st != HIPBLAS_STATUS_SUCCESS) return st;
            h->lib = lib;
        }
    }
    *out = h->lib;
    return HIPBLAS_STATUS_SUCCESS;
}

bool covered(hipblasOperation_t ta, hipblasOperation_t tb, hipblasComputeType_t compute) {
    return ta == HIPBLAS_OP_T && tb == HIPBLAS_OP_N && compute == HIPBLAS_COMPUTE_32F;
}

}  // namespace

bool own_on() {
    static const bool on = !env_off("STRATA_HIP_OWN_GEMM");
    return on;
}

const char* own_kernel(int dev) {
    switch (kern_of(dev)) {
    case Kern::wmma11: return "wmma11";
    case Kern::wmma12: return "wmma12";
    default: return "plain";
    }
}

bool own_gemm(hipStream_t s, int m, int n, int k, float alpha, const void* A, hipDataType at, int lda, long long sA,
              const void* B, hipDataType bt, int ldb, long long sB, float beta, void* C, hipDataType ct, int ldc,
              long long sC, int batch) {
    if (at != bt || m < 0 || n < 0 || k < 0 || batch < 0 || lda < k || ldb < k || ldc < m) return false;
    if (!(at == HIP_R_16F || at == HIP_R_16BF || at == HIP_R_32F)) return false;
    if (!(ct == HIP_R_32F || (ct == HIP_R_16F && at != HIP_R_32F))) return false;
    if (batch > 65535 || (m + 63) / 64 > 0x7fffffff || (n + 63) / 64 > 65535) return false;
    if (m == 0 || n == 0 || batch == 0) return true;
    const Args a{A, B, C, m, n, k, lda, ldb, ldc, sA, sB, sC, alpha, beta};
    const bool oh = ct == HIP_R_16F;
    if (at == HIP_R_32F) {
        launch_plain<float, float>(a, batch, s);
    } else {
        const Kern kk = kern_of(stream_device(s));
        const bool bf = at == HIP_R_16BF;
        if (kk == Kern::plain) {
            if (bf) {
                if (oh) launch_plain<uint16_t, __half>(a, batch, s);
                else launch_plain<uint16_t, float>(a, batch, s);
            } else {
                if (oh) launch_plain<__half, __half>(a, batch, s);
                else launch_plain<__half, float>(a, batch, s);
            }
        } else {
            if (bf) {
                if (oh) launch_wmma<true, __half>(a, batch, s);
                else launch_wmma<true, float>(a, batch, s);
            } else {
                if (oh) launch_wmma<false, __half>(a, batch, s);
                else launch_wmma<false, float>(a, batch, s);
            }
        }
    }
    return hipGetLastError() == hipSuccess;
}

hipblasStatus_t create(handle_t* h) {
    if (h == nullptr) return HIPBLAS_STATUS_INVALID_VALUE;
    Blas* b = new Blas();
    if (hipGetDevice(&b->device) != hipSuccess) b->device = 0;
    *h = b;
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t destroy(handle_t h) {
    if (h == nullptr) return HIPBLAS_STATUS_SUCCESS;
    hipblasStatus_t st = HIPBLAS_STATUS_SUCCESS;
    if (h->lib) st = hipblasDestroy(h->lib);
    delete h;
    return st;
}

hipblasStatus_t set_stream(handle_t h, hipStream_t s) {
    if (h == nullptr) return HIPBLAS_STATUS_NOT_INITIALIZED;
    h->stream = s;
    return h->lib ? hipblasSetStream(h->lib, s) : HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t set_workspace(handle_t h, void* ws, size_t bytes) {
    if (h == nullptr) return HIPBLAS_STATUS_NOT_INITIALIZED;
    h->ws = ws;
    h->ws_bytes = bytes;
    h->ws_set = true;
    return h->lib ? hipblasSetWorkspace(h->lib, ws, bytes) : HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t set_math_mode(handle_t h, hipblasMath_t mode) {
    if (h == nullptr) return HIPBLAS_STATUS_NOT_INITIALIZED;
    h->math = mode;
    h->math_set = true;
    return h->lib ? hipblasSetMathMode(h->lib, mode) : HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t gemm_strided_batched_ex(handle_t h, hipblasOperation_t ta, hipblasOperation_t tb, int m, int n, int k,
                                        const void* alpha, const void* A, hipDataType at, int lda, long long sA,
                                        const void* B, hipDataType bt, int ldb, long long sB, const void* beta,
                                        void* C, hipDataType ct, int ldc, long long sC, int batch,
                                        hipblasComputeType_t compute, hipblasGemmAlgo_t algo) {
    if (h == nullptr) return HIPBLAS_STATUS_NOT_INITIALIZED;
    if (own_on() && covered(ta, tb, compute) && alpha != nullptr && beta != nullptr &&
        own_gemm(h->stream, m, n, k, *(const float*) alpha, A, at, lda, sA, B, bt, ldb, sB, *(const float*) beta, C,
                 ct, ldc, sC, batch))
        return HIPBLAS_STATUS_SUCCESS;
    hipblasHandle_t lib = nullptr;
    const hipblasStatus_t st = library(h, &lib, "a strided batched GEMM");
    if (st != HIPBLAS_STATUS_SUCCESS) return st;
    return hipblasGemmStridedBatchedEx(lib, ta, tb, m, n, k, alpha, A, at, lda, sA, B, bt, ldb, sB, beta, C, ct, ldc,
                                       sC, batch, compute, algo);
}

hipblasStatus_t gemm_ex(handle_t h, hipblasOperation_t ta, hipblasOperation_t tb, int m, int n, int k,
                        const void* alpha, const void* A, hipDataType at, int lda, const void* B, hipDataType bt,
                        int ldb, const void* beta, void* C, hipDataType ct, int ldc, hipblasComputeType_t compute,
                        hipblasGemmAlgo_t algo) {
    if (h == nullptr) return HIPBLAS_STATUS_NOT_INITIALIZED;
    if (own_on() && covered(ta, tb, compute) && alpha != nullptr && beta != nullptr &&
        own_gemm(h->stream, m, n, k, *(const float*) alpha, A, at, lda, 0, B, bt, ldb, 0, *(const float*) beta, C, ct,
                 ldc, 0, 1))
        return HIPBLAS_STATUS_SUCCESS;
    hipblasHandle_t lib = nullptr;
    const hipblasStatus_t st = library(h, &lib, "a GEMM");
    if (st != HIPBLAS_STATUS_SUCCESS) return st;
    return hipblasGemmEx(lib, ta, tb, m, n, k, alpha, A, at, lda, B, bt, ldb, beta, C, ct, ldc, compute, algo);
}

hipblasStatus_t sgemm(handle_t h, hipblasOperation_t ta, hipblasOperation_t tb, int m, int n, int k,
                      const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta,
                      float* C, int ldc) {
    if (h == nullptr) return HIPBLAS_STATUS_NOT_INITIALIZED;
    if (own_on() && ta == HIPBLAS_OP_T && tb == HIPBLAS_OP_N && alpha != nullptr && beta != nullptr &&
        own_gemm(h->stream, m, n, k, *alpha, A, HIP_R_32F, lda, 0, B, HIP_R_32F, ldb, 0, *beta, C, HIP_R_32F, ldc, 0,
                 1))
        return HIPBLAS_STATUS_SUCCESS;
    hipblasHandle_t lib = nullptr;
    const hipblasStatus_t st = library(h, &lib, "an F32 GEMM");
    if (st != HIPBLAS_STATUS_SUCCESS) return st;
    return hipblasSgemm(lib, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

hipblasStatus_t sgemm_strided_batched(handle_t h, hipblasOperation_t ta, hipblasOperation_t tb, int m, int n, int k,
                                      const float* alpha, const float* A, int lda, long long sA, const float* B,
                                      int ldb, long long sB, const float* beta, float* C, int ldc, long long sC,
                                      int batch) {
    if (h == nullptr) return HIPBLAS_STATUS_NOT_INITIALIZED;
    if (own_on() && ta == HIPBLAS_OP_T && tb == HIPBLAS_OP_N && alpha != nullptr && beta != nullptr &&
        own_gemm(h->stream, m, n, k, *alpha, A, HIP_R_32F, lda, sA, B, HIP_R_32F, ldb, sB, *beta, C, HIP_R_32F, ldc,
                 sC, batch))
        return HIPBLAS_STATUS_SUCCESS;
    hipblasHandle_t lib = nullptr;
    const hipblasStatus_t st = library(h, &lib, "a strided batched F32 GEMM");
    if (st != HIPBLAS_STATUS_SUCCESS) return st;
    return hipblasSgemmStridedBatched(lib, ta, tb, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, batch);
}

}  // namespace strata::hipgemm
