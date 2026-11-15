// src/kernels/cuda/prompt_flash_attn.cu - see include/strata/kernels/prompt_flash_attn.hpp.
#include "strata/kernels/prompt_flash_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels::fpa {
namespace {

// The MMA, ldmatrix and cp.async below need sm_80; an older CUDA target compiles them to a trap (the host refuses such a
// device), AMD compiles none of it (the host returns false: the callers keep their kernels).
#if defined(__HIPCC__) || defined(STRATA_USE_HIP)
#define STRATA_FPA_ON 0
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define STRATA_FPA_ON 1
#else
#define STRATA_FPA_ON 2   // (device code for an older sm: traps)
#endif

#if STRATA_FPA_ON != 0

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if STRATA_FPA_ON != 1
    __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t* r, const void* p) {
#if STRATA_FPA_ON != 1
    __trap();
#else
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
#endif
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool on) {
#if STRATA_FPA_ON != 1
    __trap();
#else
    const unsigned d = (unsigned) __cvta_generic_to_shared(dst);
    const int n = on ? 16 : 0;   // (0: the 16 bytes zero-filled)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if STRATA_FPA_ON == 1
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if STRATA_FPA_ON == 1
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

// grid (ceil(T * G / 64), n_head_kv), 128 threads.  Block b of KV head g: query rows r = 64 b .. 64 b + 63 of the
// group (row r: token r / G, head g G + r % G), warp w its rows 16 w .. 16 w + 15.  The keys stream through shared
// memory in tiles of BN (two buffers, cp.async), each warp S = Q K^T on tensor cores, the masked online softmax in
// registers (a row's max / sum over the 4 lanes that hold it), O += P V with P from the S fragments.
//   D 128: q's fragments in registers;  D 256: q in shared memory, its fragments loaded a tile at a time.
template <int D, int BN, int W>
struct Cfg {
    static constexpr int BM = 16 * W;                      // query rows a block: a warp 16 rows of one KV group
    static constexpr int KS = D + 8;                       // a row's pitch in halves (conflict-free fragments)
    static constexpr bool QREG = D <= 128;
    static constexpr size_t kv_bytes = (size_t) 2 * 2 * BN * KS * 2;   // K, V, two buffers
    static constexpr size_t q_bytes = (size_t) BM * KS * 2;
    static constexpr size_t smem = QREG ? (kv_bytes > q_bytes ? kv_bytes : q_bytes) : kv_bytes + q_bytes;
};

template <int D, int BN, int W>
__global__ void __launch_bounds__(W * 32) fpa_kernel(const float* __restrict__ q, const __half* __restrict__ kc,
                                                          const __half* __restrict__ vc, int64_t ring, int n_head,
                                                          int n_kv, int G, int64_t p0, int T, int window, float qscale,
                                                          float* __restrict__ out) {
    using C = Cfg<D, BN, W>;
    constexpr int KS = C::KS, kBM = C::BM, kWarps = W;
    extern __shared__ __align__(16) unsigned char fpa_sm[];
    __half* sKV = reinterpret_cast<__half*>(fpa_sm);   // [buf][K, V][BN][KS]
    __half* sQ = C::QREG ? sKV : sKV + 2 * 2 * BN * KS;  // [kBM][KS] (D 128: before the tiles, aliased)
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const int g = blockIdx.y;
    const int rows = T * G, rb = blockIdx.x * kBM;
    const int rlast = min(rb + kBM, rows) - 1;
    // the block's keys: from the first row's lo to the last row's position
    const int64_t bhi = p0 + rlast / G;
    const int64_t blo = window > 0 ? max((int64_t) 0, p0 + rb / G - window + 1) : 0;

    // q * qscale in FP16 into sQ
    for (int i = tid; i < kBM * (D / 4); i += kWarps * 32) {
        const int rr = i / (D / 4), c4 = i % (D / 4), r = rb + rr;
        float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
        if (r < rows) {
            const int t = r / G, h = g * G + r % G;
            x = *reinterpret_cast<const float4*>(q + ((size_t) t * n_head + h) * D + c4 * 4);
        }
        uint2 u;
        u.x = pack_h2(x.x * qscale, x.y * qscale);
        u.y = pack_h2(x.z * qscale, x.w * qscale);
        *reinterpret_cast<uint2*>(sQ + rr * KS + c4 * 4) = u;
    }
    __syncthreads();
    const int w0 = rb + warp * 16;                       // the warp's first row
    const bool wact = w0 < rows;
    const int wl = min(w0 + 15, rows - 1);
    const int rA = w0 + gid, rB = rA + 8;                // this lane's two rows
    const int64_t hiA = rA < rows ? p0 + rA / G : -1, hiB = rB < rows ? p0 + rB / G : -1;
    const int64_t loA = window > 0 ? max((int64_t) 0, hiA - window + 1) : 0;
    const int64_t loB = window > 0 ? max((int64_t) 0, hiB - window + 1) : 0;
    const int64_t whi = p0 + wl / G, whi_min = p0 + w0 / G;   // the warp's last key; every row's keys reach whi_min
    const int64_t wlo = window > 0 ? max((int64_t) 0, p0 + w0 / G - window + 1) : 0;
    const int64_t wlo_max = window > 0 ? max((int64_t) 0, whi - window + 1) : 0;   // every row's keys start by it

    uint32_t qa[C::QREG ? D / 16 : 1][4];
    if constexpr (C::QREG) {
#pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) {
            const __half* p = sQ + (warp * 16 + gid) * KS + kk * 16 + tig * 2;
            qa[kk][0] = *reinterpret_cast<const uint32_t*>(p);
            qa[kk][1] = *reinterpret_cast<const uint32_t*>(p + 8 * KS);
            qa[kk][2] = *reinterpret_cast<const uint32_t*>(p + 8);
            qa[kk][3] = *reinterpret_cast<const uint32_t*>(p + 8 * KS + 8);
        }
        __syncthreads();   // (sQ is the tiles' memory from here)
    }

    const int64_t kt0 = blo / BN * BN;
    const auto load = [&](int64_t kt, int buf) {
        __half* sk = sKV + (size_t) buf * 2 * BN * KS;
        __half* sv = sk + BN * KS;
#pragma unroll
        for (int i = tid; i < BN * (D / 8); i += kWarps * 32) {
            const int row = i / (D / 8), c = i % (D / 8);
            const int64_t kp = kt + row;
            const bool on = kp >= blo && kp <= bhi;
            const int64_t rr = on ? (ring > 0 ? kp % ring : kp) : 0;
            const size_t off = ((size_t) rr * n_kv + g) * D + c * 8;
            cp_async16(sk + row * KS + c * 8, kc + off, on);
            cp_async16(sv + row * KS + c * 8, vc + off, on);
        }
        cp_async_commit();
    };

    float o[D / 8][4];
#pragma unroll
    for (int n = 0; n < D / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float mA = -INFINITY, mB = -INFINITY, lA = 0.0f, lB = 0.0f;

    load(kt0, 0);
    int buf = 0;
    for (int64_t kt = kt0; kt <= bhi; kt += BN, buf ^= 1) {
        if (kt + BN <= bhi) {
            load(kt + BN, buf ^ 1);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        const __half* sk = sKV + (size_t) buf * 2 * BN * KS;
        const __half* sv = sk + BN * KS;
        if (wact && kt <= whi && kt + BN - 1 >= wlo) {
            float sc[BN / 8][4];
#pragma unroll
            for (int n = 0; n < BN / 8; ++n) sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < D / 16; ++kk) {
                uint32_t a[4];
                if constexpr (C::QREG) {
                    a[0] = qa[kk][0];
                    a[1] = qa[kk][1];
                    a[2] = qa[kk][2];
                    a[3] = qa[kk][3];
                } else {
                    const __half* p = sQ + (warp * 16 + gid) * KS + kk * 16 + tig * 2;
                    a[0] = *reinterpret_cast<const uint32_t*>(p);
                    a[1] = *reinterpret_cast<const uint32_t*>(p + 8 * KS);
                    a[2] = *reinterpret_cast<const uint32_t*>(p + 8);
                    a[3] = *reinterpret_cast<const uint32_t*>(p + 8 * KS + 8);
                }
#pragma unroll
                for (int n = 0; n < BN / 8; ++n) {
                    const __half* p = sk + (n * 8 + gid) * KS + kk * 16 + tig * 2;
                    mma16816(sc[n], a, *reinterpret_cast<const uint32_t*>(p), *reinterpret_cast<const uint32_t*>(p + 8));
                }
            }
            // the mask: only a tile that crosses a row's first or last key
            if (kt < wlo_max || kt + BN - 1 > whi_min) {
#pragma unroll
                for (int n = 0; n < BN / 8; ++n)
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const int64_t kp = kt + n * 8 + tig * 2 + j;
                        if (kp < loA || kp > hiA) sc[n][j] = -INFINITY;
                        if (kp < loB || kp > hiB) sc[n][2 + j] = -INFINITY;
                    }
            }
            float xA = -INFINITY, xB = -INFINITY;
#pragma unroll
            for (int n = 0; n < BN / 8; ++n) {
                xA = fmaxf(xA, fmaxf(sc[n][0], sc[n][1]));
                xB = fmaxf(xB, fmaxf(sc[n][2], sc[n][3]));
            }
            xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, 1));
            xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, 2));
            xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, 1));
            xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, 2));
            const float nA = fmaxf(mA, xA), nB = fmaxf(mB, xB);
            // (a row with no key yet: its base 0, every p 0, nothing to rescale)
            const float bA = nA == -INFINITY ? 0.0f : nA, bB = nB == -INFINITY ? 0.0f : nB;
            const float cA = exp2f(mA - bA), cB = exp2f(mB - bB);
            mA = nA;
            mB = nB;
            float sA = 0.0f, sB = 0.0f;
#pragma unroll
            for (int n = 0; n < BN / 8; ++n) {
                sc[n][0] = exp2f(sc[n][0] - bA);
                sc[n][1] = exp2f(sc[n][1] - bA);
                sc[n][2] = exp2f(sc[n][2] - bB);
                sc[n][3] = exp2f(sc[n][3] - bB);
                sA += sc[n][0] + sc[n][1];
                sB += sc[n][2] + sc[n][3];
            }
            lA = lA * cA + sA;
            lB = lB * cB + sB;
#pragma unroll
            for (int n = 0; n < D / 8; ++n) {
                o[n][0] *= cA;
                o[n][1] *= cA;
                o[n][2] *= cB;
                o[n][3] *= cB;
            }
            // O += P V: P's A fragments are the S fragments of two key tiles; V's B fragments by ldmatrix.trans
#pragma unroll
            for (int k2 = 0; k2 < BN / 16; ++k2) {
                uint32_t a[4];
                a[0] = pack_h2(sc[2 * k2][0], sc[2 * k2][1]);
                a[1] = pack_h2(sc[2 * k2][2], sc[2 * k2][3]);
                a[2] = pack_h2(sc[2 * k2 + 1][0], sc[2 * k2 + 1][1]);
                a[3] = pack_h2(sc[2 * k2 + 1][2], sc[2 * k2 + 1][3]);
                const int mi = lane >> 3;   // the matrix this lane addresses: keys + 8 (mi & 1), dims + 8 (mi >> 1)
                const __half* vb = sv + (k2 * 16 + (mi & 1) * 8 + (lane & 7)) * KS + (mi >> 1) * 8;
#pragma unroll
                for (int nd = 0; nd < D / 16; ++nd) {
                    uint32_t b[4];
                    ldsm_x4_trans(b, vb + nd * 16);
                    mma16816(o[2 * nd], a, b[0], b[1]);
                    mma16816(o[2 * nd + 1], a, b[2], b[3]);
                }
            }
        }
        __syncthreads();   // (the next load overwrites this buffer)
    }
    lA += __shfl_xor_sync(0xffffffffu, lA, 1);
    lA += __shfl_xor_sync(0xffffffffu, lA, 2);
    lB += __shfl_xor_sync(0xffffffffu, lB, 1);
    lB += __shfl_xor_sync(0xffffffffu, lB, 2);
    const float iA = lA > 0.0f ? 1.0f / lA : 0.0f, iB = lB > 0.0f ? 1.0f / lB : 0.0f;
    if (rA < rows) {
        float* op = out + ((size_t) (rA / G) * n_head + g * G + rA % G) * D + tig * 2;
#pragma unroll
        for (int n = 0; n < D / 8; ++n) *reinterpret_cast<float2*>(op + n * 8) = make_float2(o[n][0] * iA, o[n][1] * iA);
    }
    if (rB < rows) {
        float* op = out + ((size_t) (rB / G) * n_head + g * G + rB % G) * D + tig * 2;
#pragma unroll
        for (int n = 0; n < D / 8; ++n) *reinterpret_cast<float2*>(op + n * 8) = make_float2(o[n][2] * iB, o[n][3] * iB);
    }
}

template <int D, int BN, int W>
bool launch(int dev, const float* q, const uint16_t* kc, const uint16_t* vc, int64_t ring, int n_head, int n_kv,
            int64_t p0, int T, int window, float scale, float* out, cudaStream_t s) {
    using C = Cfg<D, BN, W>;
    static int oks[64] = {};   // a device's: 1 set, -1 refused
    int& ok = oks[dev];
    if (ok == 0) {
        ok = cudaFuncSetAttribute(fpa_kernel<D, BN, W>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) C::smem) ==
                     cudaSuccess
                 ? 1
                 : -1;
        if (ok < 0) cudaGetLastError();
    }
    if (ok < 0) return false;
    const int G = n_head / n_kv;
    const dim3 grid((unsigned) ((T * G + C::BM - 1) / C::BM), (unsigned) n_kv);
    const float log2e = 1.4426950408889634f;
    fpa_kernel<D, BN, W><<<grid, W * 32, C::smem, s>>>(q, reinterpret_cast<const __half*>(kc),
                                                          reinterpret_cast<const __half*>(vc), ring, n_head, n_kv, G, p0,
                                                          T, window, scale * log2e, out);
    return true;
}
#endif  // STRATA_FPA_ON

}  // namespace

bool attn(const float* q, const uint16_t* kc, const uint16_t* vc, int64_t ring, int n_head, int n_head_kv,
          int head_dim, int64_t p0, int T, int window, float scale, float* out, cudaStream_t s) {
#if STRATA_FPA_ON == 0
    (void) q, (void) kc, (void) vc, (void) ring, (void) n_head, (void) n_head_kv, (void) head_dim, (void) p0, (void) T;
    (void) window, (void) scale, (void) out, (void) s;
    return false;
#else
    static const bool old = [] {
        const char* v = getenv("STRATA_PROMPT_FA_OLD");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (old || T <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0) return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) {
        cudaGetLastError();
        return false;
    }
    static int cc_major[64] = {};
    if (cc_major[dev] == 0) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
            cudaGetLastError();
            major = -1;
        }
        cc_major[dev] = major;
    }
    if (cc_major[dev] < 8) return false;
    // (key tiles of 32: Laguna's 60K prompt attention 0.079 ms a token against 0.097 with 64; 8 warps a block 0.076-0.079)
    if (head_dim == 128) return launch<128, 32, 4>(dev, q, kc, vc, ring, n_head, n_head_kv, p0, T, window, scale, out, s);
    if (head_dim == 256) return launch<256, 32, 4>(dev, q, kc, vc, ring, n_head, n_head_kv, p0, T, window, scale, out, s);
    return false;
#endif
}

}  // namespace strata::kernels::fpa
