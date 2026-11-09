// src/igpu/igpu_lane.cpp - the iGPU lane (include/strata/igpu/igpu_lane.h): glm_fast's HIP expert kernels on the
// APU's integrated GPU, reading the engine's RAM-tier slots in place.  One call: x (host) -> q8_1 -> gate/up ->
// down, the k experts' weighted sum -> out (host).  Built by hipcc with the HIP compat header (cuda* -> hip*), as one
// translation unit with glm_fast.cu (its row-dot helpers are file-local).
#include "strata/igpu/igpu_lane.h"
#include "igpu_fused.h"
#include "../kernels/cuda/glm_fast.cu"
#include "strata/kernels/exl3.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

namespace gf = strata::kernels::glmf;

namespace strata::kernels::glmf {
namespace {
// The lane's down for a FEW experts (a decode's lane call holds 1-2): every warp of a block on NR rows of its own,
// looping over the experts and combining in plan order as moe_down does (the same sum).  moe_down gives each expert
// one warp of a block's 8 (the device's plans hold 8), so at k = 1 seven of them idle: ~55 us for one expert's 3 MB.
// MODE 0: rows_dot; 1: rows_dot_rdna with the signs expanded in registers (types 16 / 18 / 22); the grid in LDS either way
template<int TD, int NR, int MODE = 0>
__global__ void __launch_bounds__(256) lane_down_kernel(const unsigned long long* plan_ptr, const float* plan_w, int k,
                                                        int n_ff, size_t down_off, const block_q8_1* __restrict__ hq,
                                                        float* __restrict__ out, unsigned int* done) {
    __shared__ unsigned long long s_ptr[gf::kMaxK];
    __shared__ float s_w[gf::kMaxK];
    __shared__ uint32_t s_tab[TabWords<TD>::n > 0 ? TabWords<TD>::n : 1];
    load_tab<TD>(s_tab);
    if (threadIdx.x < gf::kMaxK) {
        s_ptr[threadIdx.x] = (int) threadIdx.x < k ? plan_ptr[threadIdx.x] : 0ull;
        s_w[threadIdx.x] = (int) threadIdx.x < k ? plan_w[threadIdx.x] : 0.0f;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r0 = (blockIdx.x * 8 + warp) * NR;
    const size_t drb = rbytes<TD>(n_ff);
    float m[NR];
#pragma unroll
    for (int j = 0; j < NR; ++j) m[j] = 0.0f;
    for (int i = 0; i < k; ++i) {
        float s[NR];
        if (s_ptr[i] != 0ull) {
            const uint8_t* rows[NR];
#pragma unroll
            for (int j = 0; j < NR; ++j) rows[j] = (const uint8_t*) s_ptr[i] + down_off + (size_t) (r0 + j) * drb;
            if constexpr (MODE == 0) {
                rows_dot<TD, NR>(rows, hq + (size_t) i * (n_ff / 32), n_ff, lane, s_tab, s);
            } else {
                rows_dot_rdna<TD, NR>(rows, hq + (size_t) i * (n_ff / 32), n_ff, lane, s_tab, nullptr, s, true, false);
            }
        } else {
#pragma unroll
            for (int j = 0; j < NR; ++j) s[j] = 0.0f;
        }
#pragma unroll
        for (int j = 0; j < NR; ++j) m[j] += s_w[i] * s[j];
    }
    if (lane == 0)
#pragma unroll
        for (int j = 0; j < NR; ++j) out[r0 + j] = m[j] + 0.0f;
    // the call's completion, for the host: every block counts itself in once its rows are visible system-wide
    if (done != nullptr) {
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(done, 1u);
    }
}
// The lane's Q2_K down (DeepSeek V4.1's experts): 8 lanes a row, lane j on q8_1 block j of each 256-value super-block,
// so a 2304-wide row is 9 steps with no lane idle (rows_dot's 16 calls a block over 32 lanes leave half of them idle in
// the last of 5).  A block's 32 values are 8 ints of one 32-byte qs run at shift 2 * (j % 4), with scales 2j / 2j + 1:
// the products and the min terms stay integer, one float multiply-add a block instead of 8.  Not bit-identical to
// rows_dot (another sum order).  NR rows a lane group: 4 * NR rows a warp
template<int NR>
__global__ void __launch_bounds__(256) lane_down_q2k_kernel(const unsigned long long* plan_ptr, const float* plan_w,
                                                            int k, int n_ff, size_t down_off,
                                                            const block_q8_1* __restrict__ hq, float* __restrict__ out,
                                                            unsigned int* done) {
    __shared__ unsigned long long s_ptr[gf::kMaxK];
    __shared__ float s_w[gf::kMaxK];
    if (threadIdx.x < gf::kMaxK) {
        s_ptr[threadIdx.x] = (int) threadIdx.x < k ? plan_ptr[threadIdx.x] : 0ull;
        s_w[threadIdx.x] = (int) threadIdx.x < k ? plan_w[threadIdx.x] : 0.0f;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 3, j = lane & 7;
    const int r0 = ((blockIdx.x * 8 + warp) * 4 + g) * NR;
    const int nsb = n_ff / 256;
    const size_t drb = (size_t) nsb * sizeof(block_q2_K);
    const int sh = 2 * (j & 3), qo = (j >> 2) * 32;
    float m[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) m[r] = 0.0f;
    for (int i = 0; i < k; ++i) {
        if (s_ptr[i] == 0ull) continue;
        const block_q2_K* rows[NR];
#pragma unroll
        for (int r = 0; r < NR; ++r)
            rows[r] = (const block_q2_K*) ((const uint8_t*) s_ptr[i] + down_off + (size_t) (r0 + r) * drb);
        const block_q8_1* x = hq + (size_t) i * (n_ff / 32) + j;
        float acc[NR];
#pragma unroll
        for (int r = 0; r < NR; ++r) acc[r] = 0.0f;
        for (int sb = 0; sb < nsb; ++sb) {
            const block_q8_1* xb = x + sb * 8;
            int u[8];
#pragma unroll
            for (int l = 0; l < 8; ++l) u[l] = get_int_b4(xb->qs, l);
            const float d8 = __low2float(xb->ds);
            int xs0 = 0, xs1 = 0;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                xs0 = __dp4a(0x01010101, u[l], xs0);
                xs1 = __dp4a(0x01010101, u[4 + l], xs1);
            }
#pragma unroll
            for (int r = 0; r < NR; ++r) {
                const block_q2_K* b = rows[r] + sb;
                const uint8_t* q = b->qs + qo;
                int s0 = 0, s1 = 0;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    s0 = __dp4a((get_int_b4(q, l) >> sh) & 0x03030303, u[l], s0);
                    s1 = __dp4a((get_int_b4(q, 4 + l) >> sh) & 0x03030303, u[4 + l], s1);
                }
                const uint32_t sc = ((const uint16_t*) b->scales)[j];
                const int sd = (int) (sc & 0xF) * s0 + (int) ((sc >> 8) & 0xF) * s1;
                const int sm = (int) ((sc >> 4) & 0xF) * xs0 + (int) (sc >> 12) * xs1;
                const float2 dm = __half22float2(b->dm);
                acc[r] += d8 * (dm.x * (float) sd - dm.y * (float) sm);
            }
        }
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            float v = acc[r];
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            m[r] += s_w[i] * v;
        }
    }
    if (j == 0)
#pragma unroll
        for (int r = 0; r < NR; ++r) out[r0 + r] = m[r] + 0.0f;
    if (done != nullptr) {
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(done, 1u);
    }
}

// THE UNION (abyss_igpu_experts_cols): a verify window's columns route ~20% of their lane experts in common.  The
// one-column kernels read an expert once per column that routes it; these read each expert of the union ONCE and
// run every column that routes it over the same rows (the later columns' reads hit L2), each (expert, column) dot
// the one-column kernel's call bit for bit (the same rows_dot / rows_dot_rdna, the same inputs), each column's
// sum in its own plan order (the one-column down's).  UN_MAX experts a call (8 columns x 8).
constexpr int UN_COLS = 8, UN_MAX = UN_COLS * 8;
struct UnionPlan {   // host-mapped
    unsigned long long ptr[UN_MAX];       // the union's experts (device addresses of their blobs)
    int ncol[UN_MAX];                     // ... how many columns route each
    unsigned char col[UN_MAX][UN_COLS];   // ... which, and at which slot of that column's plan
    unsigned char slot[UN_MAX][UN_COLS];
    float w[UN_COLS][gf::kMaxK];          // column c's weights by plan slot
    int ne[UN_COLS];                      // column c's expert count
    int nu, nc;
};
// gate/up: block (chunk, u) - the expert's 32 gate + 32 up rows for every column that routes it, h to that column's
// plan slot.  RDNA: rows_dot_rdna with the grid in LDS and the signs expanded in registers (what gate_up_impl's RDNA
// branch runs for the type and shape); else rows_dot (the generic moe_gate_up_kernel's)
template<int TG, bool RDNA>
__global__ void __launch_bounds__(GU_WARPS * 32) lane_gu_union_kernel(const UnionPlan* __restrict__ up, int n_embd, int n_ff,
                                                                      float limit, const uint8_t* __restrict__ xq, size_t xq_cs,
                                                                      uint8_t* __restrict__ hq, size_t hq_cs) {
    const int u = blockIdx.y, chunk = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ float sg[GU_ROWS], su[GU_ROWS];
    __shared__ uint32_t s_tab[TabWords<TG>::n > 0 ? TabWords<TG>::n : 1];
    load_tab<TG>(s_tab);
    __syncthreads();
    const unsigned long long ptr = up->ptr[u];
    if (ptr == 0ull) return;
    const size_t rb = rbytes<TG>(n_embd);
    const uint8_t* blob = (const uint8_t*) ptr;
    constexpr int RPW = GU_ROWS / GU_WARPS;
    const uint8_t* rows[2 * RPW];
#pragma unroll
    for (int j = 0; j < RPW; ++j) {
        const int r = chunk * GU_ROWS + warp * RPW + j;
        rows[j] = blob + (size_t) r * rb;
        rows[RPW + j] = blob + (size_t) n_ff * rb + (size_t) r * rb;
    }
    const int ncol = up->ncol[u];
    for (int q = 0; q < ncol; ++q) {
        const int c = up->col[u][q], sl = up->slot[u][q];
        const block_q8_1* x = (const block_q8_1*) (xq + (size_t) c * xq_cs);
        float s[2 * RPW];
        if constexpr (RDNA) rows_dot_rdna<TG, 2 * RPW>(rows, x, n_embd, lane, s_tab, nullptr, s, true, false);
        else rows_dot<TG, 2 * RPW>(rows, x, n_embd, lane, s_tab, s);
        if (lane == 0) {
#pragma unroll
            for (int j = 0; j < RPW; ++j) {
                sg[warp * RPW + j] = s[j];
                su[warp * RPW + j] = s[RPW + j];
            }
        }
        __syncthreads();
        if (warp == 0) {
            const float g = fminf(sg[lane], limit);
            const float uu = fminf(fmaxf(su[lane], -limit), limit);
            q8_1_store_warp((g / (1.0f + expf(-g))) * uu, (block_q8_1*) (hq + (size_t) c * hq_cs) + (size_t) sl * (n_ff / 32) + chunk, lane);
        }
        __syncthreads();
    }
}
// down: a warp on NR rows, every union expert's rows read once and dotted with each routing column's h; the partial
// dots parked in LDS by (column, slot), then each column's sum in plan order - lane_down_kernel's sum, bit for bit
template<int TD, int NR, bool RDNA>
__global__ void __launch_bounds__(256) lane_dn_union_kernel(const UnionPlan* __restrict__ up, int n_ff, size_t down_off,
                                                            const uint8_t* __restrict__ hq, size_t hq_cs, float* __restrict__ out,
                                                            size_t out_cs, unsigned int* done) {
    __shared__ uint32_t s_tab[TabWords<TD>::n > 0 ? TabWords<TD>::n : 1];
    __shared__ float s_part[UN_COLS][8][8 * NR];
    load_tab<TD>(s_tab);
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r0 = (blockIdx.x * 8 + warp) * NR;
    const size_t drb = rbytes<TD>(n_ff);
    const int nu = up->nu, nc = up->nc;
    for (int u = 0; u < nu; ++u) {
        const uint8_t* rows[NR];
#pragma unroll
        for (int j = 0; j < NR; ++j) rows[j] = (const uint8_t*) up->ptr[u] + down_off + (size_t) (r0 + j) * drb;
        const int ncol = up->ncol[u];
        for (int q = 0; q < ncol; ++q) {
            const int c = up->col[u][q], sl = up->slot[u][q];
            const block_q8_1* h = (const block_q8_1*) (hq + (size_t) c * hq_cs) + (size_t) sl * (n_ff / 32);
            float s[NR];
            if constexpr (RDNA) rows_dot_rdna<TD, NR>(rows, h, n_ff, lane, s_tab, nullptr, s, true, false);
            else rows_dot<TD, NR>(rows, h, n_ff, lane, s_tab, s);
            if (lane == 0)
#pragma unroll
                for (int j = 0; j < NR; ++j) s_part[c][sl][warp * NR + j] = s[j];
        }
    }
    __syncthreads();
    for (int c = 0; c < nc; ++c) {
        float m[NR];
#pragma unroll
        for (int j = 0; j < NR; ++j) m[j] = 0.0f;
        const int k = up->ne[c];
        for (int i = 0; i < k; ++i) {
            const float wi = up->w[c][i];
#pragma unroll
            for (int j = 0; j < NR; ++j) m[j] += wi * s_part[c][i][warp * NR + j];
        }
        if (lane == 0)
#pragma unroll
            for (int j = 0; j < NR; ++j) out[(size_t) c * out_cs + r0 + j] = m[j] + 0.0f;
    }
    if (done != nullptr) {
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(done, 1u);
    }
}
// the per-column gate/up kernel gate_up_impl would run for this type and shape on this device: RDNA (true) or the generic
bool gu_rdna_here(int gu_type, int n_embd, int n_ff) {
    const RdnaArch ra = rdna_expert_arch();
    const bool arch = ra == RdnaArch::gfx1100 || ra == RdnaArch::gfx1151 || ra == RdnaArch::gfx1201;
    static const bool gu17_legacy = std::getenv("ABYSS_IGPU_GU17_LEGACY") != nullptr;
    static const bool gu16_any = std::getenv("ABYSS_IGPU_GU16_4096") == nullptr;
    if (gu_type == 17 && n_embd == 4096 && n_ff == 2048 && arch && !gu17_legacy) return true;
    if ((gu_type == 17 || gu_type == 22) && n_embd == 5120 && n_ff == 2304 && arch) return true;
    if (gu_type == 16 && ((n_embd == 4096 && n_ff == 2048) || (gu16_any && n_embd % 256 == 0 && n_ff % GU_ROWS == 0)) &&
        rdna3_expert_device())
        return true;
    return false;
}
// the union gate/up launch (false: no kernel for the type - the caller runs the per-column grid)
bool lane_gu_union(int gu_type, const UnionPlan* up_d, int nu, int n_embd, int n_ff, float limit, const void* xq, size_t xq_cs,
                   void* hq, size_t hq_cs, hipStream_t s) {
    const dim3 grid((unsigned) (n_ff / GU_ROWS), (unsigned) nu);
    const auto* X = (const uint8_t*) xq;
    auto* H = (uint8_t*) hq;
    if (gu_rdna_here(gu_type, n_embd, n_ff)) {
        switch (gu_type) {
            case 16: lane_gu_union_kernel<16, true><<<grid, GU_WARPS * 32, 0, s>>>(up_d, n_embd, n_ff, limit, X, xq_cs, H, hq_cs); break;
            case 17: lane_gu_union_kernel<17, true><<<grid, GU_WARPS * 32, 0, s>>>(up_d, n_embd, n_ff, limit, X, xq_cs, H, hq_cs); break;
            case 22: lane_gu_union_kernel<22, true><<<grid, GU_WARPS * 32, 0, s>>>(up_d, n_embd, n_ff, limit, X, xq_cs, H, hq_cs); break;
            default: return false;
        }
        return hipGetLastError() == hipSuccess;
    }
    switch (gu_type) {
#define LANE_GU_U(T) \
        case T: lane_gu_union_kernel<T, false><<<grid, GU_WARPS * 32, 0, s>>>(up_d, n_embd, n_ff, limit, X, xq_cs, H, hq_cs); break;
        LANE_GU_U(16) LANE_GU_U(18) LANE_GU_U(19) LANE_GU_U(23) LANE_GU_U(10) LANE_GU_U(11) LANE_GU_U(17) LANE_GU_U(22) LANE_GU_U(21)
        LANE_GU_U(29) LANE_GU_U(12) LANE_GU_U(13) LANE_GU_U(14) LANE_GU_U(39)
#undef LANE_GU_U
        default: return false;
    }
    return hipGetLastError() == hipSuccess;
}
}  // namespace

// ABYSS_IGPU_Q2K_NR: the Q2_K down's rows a lane group (1 / 2 / 4; 0: rows_dot as before)
static int lane_q2k_nr() {
    static const int nr = [] { const char* v = getenv("ABYSS_IGPU_Q2K_NR"); return v != nullptr ? std::atoi(v) : 1; }();
    return nr == 1 || nr == 2 || nr == 4 ? nr : 0;
}

// The RDNA dot (rows_dot_rdna: the signs expanded in registers, the grid in LDS) at 2 rows a warp for the i-quants it
// has (16 / 18 / 22), else rows_dot at DOWN_ROWS.  lane_bench on the Radeon 8060S, 4 experts a call: IQ2_S (GLM-5.3
// -Flash) 60.9 -> 56.3 us, IQ3_XXS (DeepSeek V4) 66.1 -> 62.6, the sums bit-identical; 8 lanes a row (fewer shuffles)
// and the rows staged in LDS by 16-byte loads were slower.  ABYSS_IGPU_DN_LEGACY=1: rows_dot at 4 rows for every type
static bool lane_dn_rdna(int d_type) {
    static const bool legacy = [] { const char* v = getenv("ABYSS_IGPU_DN_LEGACY"); return v != nullptr && std::atoi(v) != 0; }();
    return !legacy && (d_type == 16 || d_type == 18 || d_type == 22);
}
int lane_down_blocks(int d_type, int n_embd) {
    if (d_type == 10 && lane_q2k_nr() > 0) {
        const int rows = 32 * lane_q2k_nr();
        return n_embd % rows == 0 ? n_embd / rows : 0;
    }
    const int nr = lane_dn_rdna(d_type) ? 2 : DOWN_ROWS;
    return n_embd % (8 * nr) == 0 ? n_embd / (8 * nr) : 0;
}
bool lane_down(int d_type, const unsigned long long* plan_ptr, const float* plan_w, int k, int n_embd, int n_ff,
               size_t down_off, const void* hq, float* out, unsigned int* done, hipStream_t s) {
    const auto* H = (const block_q8_1*) hq;
    if (d_type == 10 && lane_q2k_nr() > 0 && n_ff % 256 == 0) {
        const int nr = lane_q2k_nr();
        if (n_embd % (32 * nr) != 0) return false;
        const int blocks = n_embd / (32 * nr);
        switch (nr) {
            case 1: lane_down_q2k_kernel<1><<<blocks, 256, 0, s>>>(plan_ptr, plan_w, k, n_ff, down_off, H, out, done); break;
            case 2: lane_down_q2k_kernel<2><<<blocks, 256, 0, s>>>(plan_ptr, plan_w, k, n_ff, down_off, H, out, done); break;
            default: lane_down_q2k_kernel<4><<<blocks, 256, 0, s>>>(plan_ptr, plan_w, k, n_ff, down_off, H, out, done); break;
        }
        return hipGetLastError() == hipSuccess;
    }
    if (lane_dn_rdna(d_type)) {
        if (n_embd % 16 != 0) return false;
        const int blocks = n_embd / 16;
        switch (d_type) {
            case 16: lane_down_kernel<16, 2, 1><<<blocks, 256, 0, s>>>(plan_ptr, plan_w, k, n_ff, down_off, H, out, done); break;
            case 18: lane_down_kernel<18, 2, 1><<<blocks, 256, 0, s>>>(plan_ptr, plan_w, k, n_ff, down_off, H, out, done); break;
            default: lane_down_kernel<22, 2, 1><<<blocks, 256, 0, s>>>(plan_ptr, plan_w, k, n_ff, down_off, H, out, done); break;
        }
        return hipGetLastError() == hipSuccess;
    }
    constexpr int NR = DOWN_ROWS;
    if (n_embd % (8 * NR) != 0) return false;
    const int blocks = n_embd / (8 * NR);
    switch (d_type) {
#define LANE_DN(T) \
        case T: lane_down_kernel<T, NR><<<blocks, 256, 0, s>>>(plan_ptr, plan_w, k, n_ff, down_off, H, out, done); break;
        LANE_DN(16) LANE_DN(18) LANE_DN(19) LANE_DN(23) LANE_DN(10) LANE_DN(11) LANE_DN(17) LANE_DN(22) LANE_DN(21)
        LANE_DN(29) LANE_DN(12) LANE_DN(13) LANE_DN(14) LANE_DN(39)
#undef LANE_DN
        default: return false;
    }
    return hipGetLastError() == hipSuccess;
}
// the union down (lane_dn_union_kernel) where lane_down would run lane_down_kernel for the type: the RDNA dot at 2 rows
// for 16 / 18 / 22, rows_dot at DOWN_ROWS for the others it has; the Q2_K lane group kernel (another sum order) and the
// types lane_down leaves to moe_down are not here (false: the caller runs lane_down a column).  blocks: how many count in done
bool lane_dn_union(int d_type, const UnionPlan* up_d, int n_embd, int n_ff, size_t down_off, const void* hq, size_t hq_cs,
                   float* out, size_t out_cs, unsigned int* done, int* blocks, hipStream_t s) {
    if (d_type == 10 && lane_q2k_nr() > 0 && n_ff % 256 == 0) return false;
    const auto* H = (const uint8_t*) hq;
    if (lane_dn_rdna(d_type)) {
        if (n_embd % 16 != 0) return false;
        *blocks = n_embd / 16;
        switch (d_type) {
            case 16: lane_dn_union_kernel<16, 2, true><<<*blocks, 256, 0, s>>>(up_d, n_ff, down_off, H, hq_cs, out, out_cs, done); break;
            case 18: lane_dn_union_kernel<18, 2, true><<<*blocks, 256, 0, s>>>(up_d, n_ff, down_off, H, hq_cs, out, out_cs, done); break;
            default: lane_dn_union_kernel<22, 2, true><<<*blocks, 256, 0, s>>>(up_d, n_ff, down_off, H, hq_cs, out, out_cs, done); break;
        }
        return hipGetLastError() == hipSuccess;
    }
    constexpr int NR = DOWN_ROWS;
    if (n_embd % (8 * NR) != 0) return false;
    *blocks = n_embd / (8 * NR);
    switch (d_type) {
#define LANE_DN_U(T) \
        case T: lane_dn_union_kernel<T, NR, false><<<*blocks, 256, 0, s>>>(up_d, n_ff, down_off, H, hq_cs, out, out_cs, done); break;
        LANE_DN_U(16) LANE_DN_U(18) LANE_DN_U(19) LANE_DN_U(23) LANE_DN_U(10) LANE_DN_U(11) LANE_DN_U(17) LANE_DN_U(22) LANE_DN_U(21)
        LANE_DN_U(29) LANE_DN_U(12) LANE_DN_U(13) LANE_DN_U(14) LANE_DN_U(39)
#undef LANE_DN_U
        default: return false;
    }
    return hipGetLastError() == hipSuccess;
}
__global__ void lane_poke_kernel() {}
void lane_poke(hipStream_t s) { lane_poke_kernel<<<1, 32, 0, s>>>(); }
// ABYSS_IGPU_TEST_STALL (test only): spins `ticks` of the device's wall clock, standing in for a queue the kernel
// driver holds while it re-validates registered memory
__global__ void lane_stall_kernel(unsigned long long ticks) {
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) {}
}
}  // namespace strata::kernels::glmf

namespace {
constexpr size_t kQ8 = 36;
struct Range {
    const uint8_t* host;
    size_t bytes;
    uint8_t* dev;
    bool owned;   // registered here (not already pinned by the engine: a HIP engine's own RAM tier) - unregistered here
};
struct Plan {                         // host-mapped: the kernels read it in place
    unsigned long long ptr[gf::kMaxK];
    float w[gf::kMaxK];
    int cpu_flag;
    unsigned int done;                // lane_down's blocks count themselves in here (the host waits on it)
};
struct Lane {
    bool open = false;
    int dev = -1;   // the lane's HIP device: the first integrated GPU among the visible ones (a HIP engine sees its own too)
    int n_embd = 0, n_ff = 0;
    hipStream_t s = nullptr;
    float *x_h = nullptr, *out_h = nullptr, *x_d = nullptr, *out_d = nullptr;
    Plan *plan_h = nullptr, *plan_d = nullptr;
    void *xq = nullptr, *hq = nullptr;
    void *xqh = nullptr, *xqh_d = nullptr;   // pinned, mapped: a caller's q8_1 x (abyss_igpu_experts_q)
    // a call started by abyss_igpu_experts_start, for abyss_igpu_experts_finish
    // abyss_igpu_experts_cols (a verify window's columns in one call): per-column plans (host-mapped), the column
    // table the kernels read, one q8_1 x buffer and one h buffer a column, the sums a column (host-mapped)
    static constexpr int kCols = 8;
    struct PlanCols {
        unsigned long long ptr[kCols][gf::kMaxK];
        float w[kCols][gf::kMaxK];
        int cpu_flag;
        unsigned int done;
    };
    PlanCols *planc_h = nullptr, *planc_d = nullptr;
    gf::MoeColArgs *ca_h = nullptr, *ca_d = nullptr;
    void *xqc_h = nullptr, *xqc_d = nullptr;   // kCols x (n_embd / 32 x 36)
    float *xc_h = nullptr, *xc_d = nullptr;    // kCols x n_embd (a column given as floats: quantized here)
    void* hqc = nullptr;                       // kCols x kMaxK x (n_ff / 32 x 36)
    float *outc_h = nullptr, *outc_d = nullptr;
    gf::UnionPlan *un_h = nullptr, *un_d = nullptr;   // the union (host-mapped)
    unsigned long long un_calls = 0, un_gu = 0, un_dn = 0;
    bool cols_ok = false, cols_tried = false;
    bool g_counted = false;
    int g_want = 0;
    unsigned long long g_dout = 0ull;
    std::chrono::steady_clock::time_point g_t0;
    std::vector<Range> ranges;
    std::mutex mu;
    // abyss_igpu_rows: the experts' table (host-mapped) and the scratch, grown as needed
    unsigned long long* rptr_h = nullptr;
    int* rlight_h = nullptr;
    int r_cap = 0;
    void *rxq = nullptr, *rhq = nullptr;
    size_t rxq_b = 0, rhq_b = 0;
    // the fused pass (igpu_fused.cu): the experts' row bounds and 64-row tiles (host-mapped), a scratch for the check
    int* fb_h = nullptr;
    int fb_cap = 0;
    int2* ft_h = nullptr;
    int ft_cap = 0;
    void *rchk = nullptr, *rxsave = nullptr;
    size_t rchk_b = 0, rxsave_b = 0;
    unsigned long long n_fused = 0, n_rows_k = 0;
    // A call that has not finished by its deadline is left running and answered by the CPU lane (return 1); the calls
    // after it go to the CPU too until it has (the queue may be held for seconds - see abyss_igpu_experts)
    bool pending = false;
    std::chrono::steady_clock::time_point t_pend, t_busy;   // (t_busy: the last call that found the late one running)
    unsigned long long on_cpu = 0;
};
Lane L;

// the device address of p inside a registered range, or 0
unsigned long long dev_of(const void* p) {
    const uint8_t* b = (const uint8_t*) p;
    auto it = std::upper_bound(L.ranges.begin(), L.ranges.end(), b, [](const uint8_t* q, const Range& r) { return q < r.host; });
    if (it != L.ranges.begin() && b < (--it)->host + it->bytes) return (unsigned long long) (it->dev + (b - it->host));
    return 0ull;
}
// The lane's device current on the calling thread for one entry point, the caller's put back after: a HIP engine
// shares this HIP runtime, and its threads keep their own device current (a CUDA engine's HIP sees the iGPU alone)
struct DevScope {
    int prev = -1;
    DevScope() {
        if (L.dev < 0 || hipGetDevice(&prev) != hipSuccess || prev == L.dev) { prev = -1; return; }
        hipSetDevice(L.dev);
    }
    ~DevScope() { if (prev >= 0) hipSetDevice(prev); }
};
void say(char* err, int n, const char* what, hipError_t e) {
    if (err != nullptr && n > 0) std::snprintf(err, (size_t) n, "%s: %s", what, hipGetErrorString(e));
}
}  // namespace

extern "C" {

__attribute__((visibility("default"))) int abyss_igpu_version(void) { return ABYSS_IGPU_API; }

__attribute__((visibility("default"))) int abyss_igpu_open(int n_embd, int n_ff, char* err, int err_n) {
    std::lock_guard<std::mutex> lk(L.mu);
    if (L.open) return 0;
    int n = 0;
    hipError_t e = hipGetDeviceCount(&n);
    if (e != hipSuccess || n < 1) {
        say(err, err_n, "no HIP device", e == hipSuccess ? hipErrorNoDevice : e);
        return -1;
    }
    // the first integrated GPU among the visible devices, else device 0 as before
    int dev = -1;
    for (int i = 0; i < n && dev < 0; ++i) {
        hipDeviceProp_t p{};
        if (hipGetDeviceProperties(&p, i) == hipSuccess && p.integrated) dev = i;
    }
    L.dev = dev < 0 ? 0 : dev;
    int prev = -1;
    hipGetDevice(&prev);
    if ((e = hipSetDevice(L.dev)) != hipSuccess) { say(err, err_n, "hipSetDevice", e); return -1; }
    DevScope ds;
    ds.prev = prev != L.dev ? prev : -1;   // (the caller's device back on the way out)
    hipSetDeviceFlags(hipDeviceScheduleSpin);   // the service thread waits ~0.1 ms a call: spin, not sleep
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    const auto hm = [&](void** p, size_t b) { return hipHostMalloc(p, b, hipHostMallocMapped); };
    if ((e = hipStreamCreateWithFlags(&L.s, hipStreamNonBlocking)) != hipSuccess ||
        (e = hm((void**) &L.x_h, (size_t) n_embd * sizeof(float))) != hipSuccess ||
        (e = hm((void**) &L.out_h, (size_t) n_embd * sizeof(float))) != hipSuccess ||
        (e = hm((void**) &L.plan_h, sizeof(Plan))) != hipSuccess ||
        (e = hipMalloc(&L.xq, (size_t) n_embd / 32 * kQ8)) != hipSuccess ||
        (e = hm(&L.xqh, (size_t) n_embd / 32 * kQ8)) != hipSuccess ||
        (e = hipMalloc(&L.hq, (size_t) gf::kMaxK * n_ff / 32 * kQ8)) != hipSuccess) {
        say(err, err_n, "iGPU lane buffers", e);
        return -1;
    }
    hipHostGetDevicePointer((void**) &L.x_d, L.x_h, 0);
    hipHostGetDevicePointer(&L.xqh_d, L.xqh, 0);
    hipHostGetDevicePointer((void**) &L.out_d, L.out_h, 0);
    hipHostGetDevicePointer((void**) &L.plan_d, L.plan_h, 0);
    std::memset(L.plan_h, 0, sizeof(Plan));
    L.open = true;
    return 0;
}

__attribute__((visibility("default"))) int abyss_igpu_register(const void* host, size_t bytes, char* err, int err_n) {
    std::lock_guard<std::mutex> lk(L.mu);
    if (!L.open) { say(err, err_n, "iGPU lane not open", hipErrorNotInitialized); return -1; }
    DevScope ds;
    hipError_t e = hipHostRegister(const_cast<void*>(host), bytes, hipHostRegisterMapped);
    // a HIP engine pinned it already (portable: every device maps it) - the lane reads it as it is
    const bool owned = e == hipSuccess;
    void* d = nullptr;
    if (e == hipErrorHostMemoryAlreadyRegistered) (void) hipGetLastError();
    else if (e == hipErrorInvalidValue) {   // (ROCm 7.2: an already pinned range says this - the lane reads it if mapped)
        (void) hipGetLastError();
        if (hipHostGetDevicePointer(&d, const_cast<void*>(host), 0) != hipSuccess) {
            (void) hipGetLastError();   // (the engine's own launch checks read the thread's last error: issue #11)
            say(err, err_n, "hipHostRegister", e);
            return -1;
        }
    } else if (e != hipSuccess) {
        (void) hipGetLastError();
        say(err, err_n, "hipHostRegister", e);
        return -1;
    }
    if (d == nullptr && (e = hipHostGetDevicePointer(&d, const_cast<void*>(host), 0)) != hipSuccess) {
        (void) hipGetLastError();
        say(err, err_n, "hipHostGetDevicePointer", e);
        return -1;
    }
    // kept sorted by host address: a call looks its experts up by binary search (the RAM tier is registered in a few
    // hundred pieces)
    const Range nr{(const uint8_t*) host, bytes, (uint8_t*) d, owned};
    L.ranges.insert(std::upper_bound(L.ranges.begin(), L.ranges.end(), nr,
                                     [](const Range& a, const Range& b) { return a.host < b.host; }),
                    nr);
    return 0;
}

static int igpu_experts(int gu_type, int d_type, int n_embd, int n_ff, size_t down_off, float limit, int ne,
                        const void* const* blobs, const float* w, const float* x, const void* xq_in, float* out,
                        bool start_only = false);
static int igpu_wait(bool counted, int want, unsigned long long dout, float* out, int n_embd,
                     std::chrono::steady_clock::time_point t0, volatile unsigned int* done_h = nullptr);
// in two halves (the engine's service thread starts a route's lane experts before that route's disk reads, and
// collects the sum after them): _start queues the kernels and keeps the lane (0; 1: busy or not in registered memory -
// nothing queued, compute it elsewhere), _finish waits up to the deadline and copies the sum out (0), or leaves the
// call pending and returns 1 (compute those experts elsewhere); either way it releases the lane
__attribute__((visibility("default"))) int abyss_igpu_experts_start(int gu_type, int d_type, int n_embd, int n_ff,
                                                                    size_t down_off, float limit, int ne,
                                                                    const void* const* blobs, const float* w,
                                                                    const float* x, const void* xq) {
    DevScope ds;
    if (xq != nullptr && n_embd % 32 != 0) xq = nullptr;
    return igpu_experts(gu_type, d_type, n_embd, n_ff, down_off, limit, ne, blobs, w, xq != nullptr ? nullptr : x, xq,
                        nullptr, true);
}
__attribute__((visibility("default"))) int abyss_igpu_experts_finish(float* out, int n_embd) {
    DevScope ds;
    std::unique_lock<std::mutex> lk(L.mu, std::adopt_lock);
    return igpu_wait(L.g_counted, L.g_want, L.g_dout, out, n_embd, L.g_t0);
}
__attribute__((visibility("default"))) int abyss_igpu_experts(int gu_type, int d_type, int n_embd, int n_ff,
                                                              size_t down_off, float limit, int ne,
                                                              const void* const* blobs, const float* w,
                                                              const float* x, float* out) {
    DevScope ds;
    return igpu_experts(gu_type, d_type, n_embd, n_ff, down_off, limit, ne, blobs, w, x, nullptr, out);
}
// x as the caller's q8_1 (the device's own quantization of it, from the engine's request): no quantize here - the
// call's first kernel was a 5 us launch of its own - and a 4.6 KB copy instead of 16 KB
__attribute__((visibility("default"))) int abyss_igpu_experts_q(int gu_type, int d_type, int n_embd, int n_ff,
                                                                size_t down_off, float limit, int ne,
                                                                const void* const* blobs, const float* w,
                                                                const void* xq, float* out) {
    DevScope ds;
    if (xq == nullptr || n_embd % 32 != 0) return 1;
    return igpu_experts(gu_type, d_type, n_embd, n_ff, down_off, limit, ne, blobs, w, nullptr, xq, out);
}
// A verify window's columns in ONE call (abyss_igpu_experts_cols): column c's ne[c] experts (blobs / w at c * kMaxK),
// its x as q8_1 (xq[c]) or floats (x[c], quantized here), its sum into out[c].  The gate/up of every column in one
// grid (moe_gate_up_cols: the blocks nc single-column launches would run, the same sums), the down a column after
// another on the same stream (lane_down, the same kernel as the one-column call), one wait at the end.  What it saves
// is the per-call cost - the launches' gaps and the host's wait and copies a column - not bytes: each column's
// blocks read their own experts (a window's columns share ~20% of them: see the engine's verify-window stats).
// 0 ok; 1 not here (busy, a blob outside registered memory, too many columns / experts: the caller does it a column
// at a time); < 0 an error.  Every (expert, column) product and every column's sum is the one-column call's bit for bit
static bool cols_alloc() {
    if (L.cols_tried) return L.cols_ok;
    L.cols_tried = true;
    const auto hm = [&](void** p, size_t b) { return hipHostMalloc(p, b, hipHostMallocMapped); };
    const size_t xq_b = (size_t) L.n_embd / 32 * kQ8, hq_b = (size_t) gf::kMaxK * L.n_ff / 32 * kQ8;
    hipError_t e;
    if ((e = hm((void**) &L.planc_h, sizeof(Lane::PlanCols))) != hipSuccess ||
        (e = hm((void**) &L.ca_h, sizeof(gf::MoeColArgs) * Lane::kCols)) != hipSuccess ||
        (e = hm(&L.xqc_h, xq_b * Lane::kCols)) != hipSuccess ||
        (e = hm((void**) &L.xc_h, (size_t) L.n_embd * sizeof(float) * Lane::kCols)) != hipSuccess ||
        (e = hipMalloc(&L.hqc, hq_b * Lane::kCols)) != hipSuccess ||
        (e = hm((void**) &L.outc_h, (size_t) L.n_embd * sizeof(float) * Lane::kCols)) != hipSuccess ||
        (e = hm((void**) &L.un_h, sizeof(gf::UnionPlan))) != hipSuccess) {
        std::fprintf(stderr, "iGPU lane: the columns call's buffers did not allocate (%s) - a call a column\n", hipGetErrorString(e));
        return false;
    }
    hipHostGetDevicePointer((void**) &L.planc_d, L.planc_h, 0);
    hipHostGetDevicePointer((void**) &L.ca_d, L.ca_h, 0);
    hipHostGetDevicePointer(&L.xqc_d, L.xqc_h, 0);
    hipHostGetDevicePointer((void**) &L.xc_d, L.xc_h, 0);
    hipHostGetDevicePointer((void**) &L.outc_d, L.outc_h, 0);
    hipHostGetDevicePointer((void**) &L.un_d, L.un_h, 0);
    std::memset(L.planc_h, 0, sizeof(Lane::PlanCols));
    std::memset(L.un_h, 0, sizeof(gf::UnionPlan));
    for (int c = 0; c < Lane::kCols; ++c) {
        L.ca_h[c].plan_ptr = L.planc_d->ptr[c];
        L.ca_h[c].plan_w = L.planc_d->w[c];
        L.ca_h[c].cpu_flag = &L.planc_d->cpu_flag;
        L.ca_h[c].cpu_part = nullptr;
    }
    L.cols_ok = true;
    return true;
}
__attribute__((visibility("default"))) int abyss_igpu_experts_cols(int gu_type, int d_type, int n_embd, int n_ff,
                                                                   size_t down_off, float limit, int nc, const int* ne,
                                                                   const void* const* blobs, const float* w,
                                                                   const float* const* x, const void* const* xq,
                                                                   float* const* out) {
    DevScope ds;
    if (!L.open || nc < 1 || nc > Lane::kCols || n_embd > L.n_embd || n_ff > L.n_ff || n_embd % 32 != 0) return 1;
    int kmax = 0;
    for (int c = 0; c < nc; ++c) {
        if (ne[c] < 1 || ne[c] > gf::kMaxK) return 1;
        kmax = std::max(kmax, ne[c]);
    }
    std::unique_lock<std::mutex> lk(L.mu, std::try_to_lock);
    if (!lk.owns_lock()) return 1;
    if (L.pending) {   // (as abyss_igpu_experts: a late call still owns the queue)
        if (hipStreamQuery(L.s) != hipSuccess) {
            ++L.on_cpu;
            L.t_busy = std::chrono::steady_clock::now();
            return 1;
        }
        const auto now = std::chrono::steady_clock::now();
        const double lo = std::chrono::duration<double, std::milli>(L.t_busy - L.t_pend).count();
        const double hi = std::chrono::duration<double, std::milli>(now - L.t_pend).count();
        std::fprintf(stderr, "iGPU lane: the late call ran %.0f-%.0f ms (busy at the last of %llu calls the CPU took "
                             "meanwhile, done by this one)\n", lo, hi, L.on_cpu);
        L.pending = false;
        L.on_cpu = 0;
    }
    if (!cols_alloc()) return 1;
    const size_t xq_b = (size_t) L.n_embd / 32 * kQ8, hq_b = (size_t) gf::kMaxK * L.n_ff / 32 * kQ8;
    for (int c = 0; c < nc; ++c) {
        for (int i = 0; i < ne[c]; ++i) {
            const unsigned long long d = dev_of(blobs[(size_t) c * gf::kMaxK + i]);
            if (d == 0ull) return 1;
            L.planc_h->ptr[c][i] = d;
            L.planc_h->w[c][i] = w[(size_t) c * gf::kMaxK + i];
        }
        for (int i = ne[c]; i < gf::kMaxK; ++i) { L.planc_h->ptr[c][i] = 0ull; L.planc_h->w[c][i] = 0.0f; }
    }
    L.planc_h->cpu_flag = 0;
    volatile unsigned int* done_h = &L.planc_h->done;
    *done_h = 0u;
    for (int c = 0; c < nc; ++c) {
        if (xq != nullptr && xq[c] != nullptr) {
            std::memcpy((uint8_t*) L.xqc_h + (size_t) c * xq_b, xq[c], (size_t) n_embd / 32 * kQ8);
        } else {
            if (x == nullptr || x[c] == nullptr) return 1;
            std::memcpy(L.xc_h + (size_t) c * L.n_embd, x[c], (size_t) n_embd * sizeof(float));
            gf::quantize_q8_1(L.xc_d + (size_t) c * L.n_embd, (uint8_t*) L.xqc_d + (size_t) c * xq_b, n_embd, L.s);
        }
    }
    // THE UNION (ABYSS_IGPU_COLS_UNION=0: the per-column grids, every column reading its own experts): the columns'
    // distinct experts, each with the columns that route it
    static const bool union_on = [] { const char* v = getenv("ABYSS_IGPU_COLS_UNION"); return v == nullptr || std::atoi(v) != 0; }();
    bool gu_done = false, dn_done = false;
    int want = 0;
    bool counted = true;
    if (union_on) {
        gf::UnionPlan* U = L.un_h;
        int nu = 0;
        bool fits = true;
        for (int c = 0; c < nc && fits; ++c) {
            U->ne[c] = ne[c];
            for (int i = 0; i < ne[c]; ++i) {
                const unsigned long long d = L.planc_h->ptr[c][i];
                U->w[c][i] = L.planc_h->w[c][i];
                int u = 0;
                while (u < nu && U->ptr[u] != d) ++u;
                if (u == nu) {
                    if (nu == gf::UN_MAX) { fits = false; break; }
                    U->ptr[nu] = d;
                    U->ncol[nu] = 0;
                    ++nu;
                }
                if (U->ncol[u] >= gf::UN_COLS) { fits = false; break; }   // (a column routes an expert once: never)
                U->col[u][U->ncol[u]] = (unsigned char) c;
                U->slot[u][U->ncol[u]] = (unsigned char) i;
                ++U->ncol[u];
            }
        }
        if (fits && ne[0] <= 8) {
            bool ok8 = true;
            for (int c = 0; c < nc; ++c) ok8 = ok8 && ne[c] <= 8;   // (s_part holds 8 slots a column)
            if (ok8) {
                U->nu = nu;
                U->nc = nc;
                gu_done = gf::lane_gu_union(gu_type, L.un_d, nu, n_embd, n_ff, limit, L.xqc_d, xq_b, L.hqc, hq_b, L.s);
                if (gu_done) {
                    int blocks = 0;
                    dn_done = gf::lane_dn_union(d_type, L.un_d, n_embd, n_ff, down_off, L.hqc, hq_b, L.outc_d, (size_t) L.n_embd,
                                                &L.planc_d->done, &blocks, L.s);
                    if (dn_done) want = blocks;
                }
                ++L.un_calls;
                L.un_gu += gu_done;
                L.un_dn += dn_done;
            }
        }
    }
    if (!gu_done)
        gf::moe_gate_up_cols(gu_type, L.ca_d, nc, kmax, n_embd, n_ff, limit, L.xqc_d, xq_b, L.hqc, hq_b, nullptr, 0,
                             nullptr, 0, 0, nullptr, 0, L.s);
    for (int c = 0; c < nc && !dn_done; ++c) {
        const int wb = gf::lane_down_blocks(d_type, n_embd);
        unsigned int* dn = counted && wb > 0 ? &L.planc_d->done : nullptr;
        if (!gf::lane_down(d_type, L.planc_d->ptr[c], L.planc_d->w[c], ne[c], n_embd, n_ff, down_off,
                           (uint8_t*) L.hqc + (size_t) c * hq_b, L.outc_d + (size_t) c * L.n_embd, dn, L.s)) {
            gf::MoeDev md;
            md.plan_ptr = L.planc_d->ptr[c];
            md.plan_w = L.planc_d->w[c];
            md.cpu_flag = &L.planc_d->cpu_flag;
            gf::moe_down(d_type, md, ne[c], n_embd, n_ff, down_off, (uint8_t*) L.hqc + (size_t) c * hq_b, nullptr,
                         L.outc_d + (size_t) c * L.n_embd, L.s);
            counted = false;
        } else if (dn != nullptr) {
            want += wb;
        } else {
            counted = false;
        }
    }
    const int rc = igpu_wait(counted, counted ? want : 0, 1ull /* (no copy there) */, nullptr, n_embd,
                             std::chrono::steady_clock::now(), done_h);
    if (rc != 0) return rc;
    for (int c = 0; c < nc; ++c) std::memcpy(out[c], L.outc_h + (size_t) c * L.n_embd, (size_t) n_embd * sizeof(float));
    return 0;
}
static int igpu_experts(int gu_type, int d_type, int n_embd, int n_ff, size_t down_off, float limit, int ne,
                        const void* const* blobs, const float* w, const float* x, const void* xq_in, float* out,
                        bool start_only) {
    if (!L.open || ne < 1 || ne > gf::kMaxK || n_embd > L.n_embd || n_ff > L.n_ff) return 1;
    // one caller at a time (a layer split's parts share the lane): busy -> the CPU takes this one, a decode does not wait
    std::unique_lock<std::mutex> lk(L.mu, std::try_to_lock);
    if (!lk.owns_lock()) return 1;
    // A late call still owns the buffers and the queue: the CPU takes this one, until it is done.  The kernel driver
    // can hold the iGPU's queues for seconds while it faults registered memory in again (seen: 0.7-18 s, a few times
    // after the host had been idle for hours; kernel log "svm_range_restore_work / amdgpu_amdkfd_restore_userptr_worker
    // hogged CPU"), and a decode waited all of it for one layer's experts.
    if (L.pending) {
        if (hipStreamQuery(L.s) != hipSuccess) {
            ++L.on_cpu;
            L.t_busy = std::chrono::steady_clock::now();
            return 1;
        }
        // seen done only now, at the next call: the stall lasted at least until the last call that found it busy and
        // at most until now (an idle server between requests made the old "finished after" read minutes)
        const auto now = std::chrono::steady_clock::now();
        const double lo = std::chrono::duration<double, std::milli>(L.t_busy - L.t_pend).count();
        const double hi = std::chrono::duration<double, std::milli>(now - L.t_pend).count();
        std::fprintf(stderr, "iGPU lane: the late call ran %.0f-%.0f ms (busy at the last of %llu calls the CPU took "
                             "meanwhile, done by this one)\n", lo, hi, L.on_cpu);
        L.pending = false;
        L.on_cpu = 0;
    }
    for (int i = 0; i < ne; ++i) {
        const uint8_t* b = (const uint8_t*) blobs[i];
        unsigned long long d = 0ull;
        auto it = std::upper_bound(L.ranges.begin(), L.ranges.end(), b,
                                   [](const uint8_t* p, const Range& r) { return p < r.host; });
        if (it != L.ranges.begin() && b < (--it)->host + it->bytes) d = (unsigned long long) (it->dev + (b - it->host));
        if (d == 0ull) return 1;
        L.plan_h->ptr[i] = d;
        L.plan_h->w[i] = w[i];
    }
    for (int i = ne; i < gf::kMaxK; ++i) { L.plan_h->ptr[i] = 0ull; L.plan_h->w[i] = 0.0f; }
    L.plan_h->cpu_flag = 0;
    // DIRECT (ABYSS_IGPU_DIRECT=1; default 0): x read from, and the sum written to, the caller's buffers in place when
    // they lie in registered host memory (the engine registers its request ring and answer slots) - no 16 KB memcpy
    // either way; and the completion read from a host-mapped counter the down kernel's blocks bump, not from
    // hipStreamQuery (a runtime call of ~3 us each, spun on).  0: the copies and the query, as before
    // (measured the same as the copies on an RTX 5090 + Radeon 8060S: a call 109-117 us either way, 83.4 / 83.9 against
    // 84.6 / 84.5 tok/s - the call's time is the iGPU's, ~20 us fixed + 41 an expert)
    static const bool direct = [] { const char* v = getenv("ABYSS_IGPU_DIRECT"); return v != nullptr && std::atoi(v) != 0; }();
    const unsigned long long dx = direct && x != nullptr ? dev_of(x) : 0ull,
                             dout = direct && out != nullptr ? dev_of(out) : 0ull;
    if (xq_in != nullptr) std::memcpy(L.xqh, xq_in, (size_t) n_embd / 32 * kQ8);
    else if (dx == 0ull) std::memcpy(L.x_h, x, (size_t) n_embd * sizeof(float));
    const float* xd = dx != 0ull ? (const float*) dx : L.x_d;
    float* od = dout != 0ull ? (float*) dout : L.out_d;
    gf::MoeDev md;
    md.plan_ptr = L.plan_d->ptr;
    md.plan_w = L.plan_d->w;
    md.cpu_flag = &L.plan_d->cpu_flag;
    volatile unsigned int* done_h = &L.plan_h->done;
    *done_h = 0u;
    const int want = direct ? gf::lane_down_blocks(d_type, n_embd) : 0;
    if (xq_in == nullptr) gf::quantize_q8_1(xd, L.xq, n_embd, L.s);
    gf::moe_gate_up(gu_type, md, ne, n_embd, n_ff, limit, xq_in != nullptr ? L.xqh_d : L.xq, L.hq, nullptr, 0, nullptr,
                    0, nullptr, L.s);
    bool counted = want > 0;
    if (!gf::lane_down(d_type, L.plan_d->ptr, L.plan_d->w, ne, n_embd, n_ff, down_off, L.hq, od,
                       counted ? &L.plan_d->done : nullptr, L.s)) {
        counted = false;
        gf::moe_down(d_type, md, ne, n_embd, n_ff, down_off, L.hq, nullptr, od, L.s);
    }
    static const unsigned long long stall_ticks = [] {
        const char* v = getenv("ABYSS_IGPU_TEST_STALL");   // "<ms>:<every n calls>"
        int rate = 0;
        if (v == nullptr || hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0) != hipSuccess) return 0ull;
        return (unsigned long long) (std::atof(v) * (double) rate);   // rate: kHz
    }();
    static const unsigned long long stall_every = [] {
        const char* v = getenv("ABYSS_IGPU_TEST_STALL");
        const char* c = v ? std::strchr(v, ':') : nullptr;
        return c ? std::max(1ull, std::strtoull(c + 1, nullptr, 10)) : 0ull;
    }();
    static unsigned long long n_calls = 0;
    if (stall_ticks > 0 && stall_every > 0 && ++n_calls % stall_every == 0)
        gf::lane_stall_kernel<<<1, 32, 0, L.s>>>(stall_ticks);
    if (start_only) {
        L.g_counted = counted;
        L.g_want = want;
        L.g_dout = dout;
        L.g_t0 = std::chrono::steady_clock::now();
        lk.release();   // (the lane stays held until abyss_igpu_experts_finish)
        return 0;
    }
    return igpu_wait(counted, want, dout, out, n_embd, std::chrono::steady_clock::now());
}

static int igpu_wait(bool counted, int want, unsigned long long dout, float* out, int n_embd,
                     std::chrono::steady_clock::time_point t0, volatile unsigned int* done_h) {
    if (done_h == nullptr) done_h = &L.plan_h->done;   // (the columns call counts in its own plan's)
    // wait (spinning: a call is ~0.05-0.3 ms) up to the deadline, STRATA_GLM_IGPU_TIMEOUT_MS (20; 0: no deadline)
    static const double timeout_ms = [] {
        const char* v = getenv("STRATA_GLM_IGPU_TIMEOUT_MS");
        return v != nullptr ? std::max(0.0, std::atof(v)) : 20.0;
    }();
    hipError_t e = hipErrorNotReady;
    if (counted) {   // the blocks' count first (memory, no runtime call); the query below only confirms
        unsigned int polls = 0;
        while (*done_h < (unsigned int) want) {
            if ((++polls & 255u) == 0 && timeout_ms > 0.0 &&
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() > timeout_ms)
                break;
        }
    }
    while (e == hipErrorNotReady && (e = hipStreamQuery(L.s)) == hipErrorNotReady) {
        if (timeout_ms > 0.0 &&
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() > timeout_ms) {
            L.pending = true;
            L.t_pend = t0;
            L.t_busy = std::chrono::steady_clock::now();
            L.on_cpu = 0;
            std::fprintf(stderr, "iGPU lane: a call is still running after %.0f ms - the CPU lane takes the calls until "
                                 "it finishes\n", timeout_ms);
            return 1;
        }
    }
    if (e != hipSuccess) {
        std::fprintf(stderr, "iGPU lane: %s\n", hipGetErrorString(e));
        return -1;
    }
    if (dout == 0ull) std::memcpy(out, L.out_h, (size_t) n_embd * sizeof(float));
    return 0;
}

// the same for an EXL3 layer (exllamav3's trellis experts, src/kernels/cuda/exl3.cu): the blob layout in off9 (suh,
// svh, trellis byte offsets of gate, up, down) and K3 / cb; F32 input, no q8_1.  The same queue, deadline and late-call
// rule as abyss_igpu_experts.  In two halves so the caller's CPU threads compute another share of the route meanwhile:
// abyss_igpu_x3_start queues the kernels and keeps the lane (0; 1: busy, nothing queued), abyss_igpu_x3_finish waits
// up to the deadline and copies the sum out (0), or leaves the late call pending and returns 1 (the caller computes
// those experts itself); either way it releases the lane
__attribute__((visibility("default"))) int abyss_igpu_x3_start(const int* K3, int cb, const size_t* off9, int n_embd,
                                                               int n_ff, float limit, int ne, const void* const* blobs,
                                                               const float* w, const float* x) {
    DevScope ds;
    namespace x3 = strata::kernels::exl3;
    if (!L.open || ne < 1 || ne > gf::kMaxK || n_embd > L.n_embd || n_ff > L.n_ff) return 1;
    if (!L.mu.try_lock()) return 1;
    std::unique_lock<std::mutex> lk(L.mu, std::adopt_lock);
    if (L.pending) {
        if (hipStreamQuery(L.s) != hipSuccess) {
            ++L.on_cpu;
            L.t_busy = std::chrono::steady_clock::now();
            return 1;
        }
        L.pending = false;
        L.on_cpu = 0;
    }
    static const bool ready = x3::prepare(L.s);
    if (!ready) return 1;
    for (int i = 0; i < ne; ++i) {
        const unsigned long long d = dev_of(blobs[i]);
        if (d == 0ull) return 1;
        L.plan_h->ptr[i] = d;
        L.plan_h->w[i] = w[i];
    }
    for (int i = ne; i < gf::kMaxK; ++i) { L.plan_h->ptr[i] = 0ull; L.plan_h->w[i] = 0.0f; }
    std::memcpy(L.x_h, x, (size_t) n_embd * sizeof(float));
    x3::MoeArgs xa;
    xa.plan_ptr = L.plan_d->ptr;
    xa.plan_w = L.plan_d->w;
    for (int r = 0; r < 3; ++r) {
        xa.lay.suh[r] = off9[r];
        xa.lay.svh[r] = off9[3 + r];
        xa.lay.trellis[r] = off9[6 + r];
        xa.lay.K[r] = K3[r];
    }
    xa.lay.cb = cb;
    xa.k = ne;
    xa.n_embd = n_embd;
    xa.n_ff = n_ff;
    xa.limit = limit;
    x3::moe_gate_up(xa, L.x_d, nullptr, L.s);
    x3::moe_down(xa, nullptr, nullptr, L.out_d, L.s);
    L.t_pend = std::chrono::steady_clock::now();
    lk.release();   // (the lane stays held until abyss_igpu_x3_finish)
    return 0;
}

__attribute__((visibility("default"))) int abyss_igpu_x3_finish(float* out, int n_embd) {
    DevScope ds;
    namespace x3 = strata::kernels::exl3;
    std::unique_lock<std::mutex> lk(L.mu, std::adopt_lock);
    static const double timeout_ms = [] {
        const char* v = getenv("STRATA_GLM_IGPU_TIMEOUT_MS");
        return v != nullptr ? std::max(0.0, std::atof(v)) : 20.0;
    }();
    const auto t0 = L.t_pend;
    hipError_t e;
    while ((e = hipStreamQuery(L.s)) == hipErrorNotReady) {
        if (timeout_ms > 0.0 &&
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() > timeout_ms) {
            L.pending = true;
            L.t_busy = std::chrono::steady_clock::now();
            L.on_cpu = 0;
            std::fprintf(stderr, "iGPU lane: an EXL3 call is still running after %.0f ms - the CPU lane takes the calls "
                                 "until it finishes\n", timeout_ms);
            return 1;
        }
    }
    if (e != hipSuccess || x3::launch_errors() > 0) {
        std::fprintf(stderr, "iGPU lane: EXL3 %s\n", hipGetErrorString(e));
        return -1;
    }
    std::memcpy(out, L.out_h, (size_t) n_embd * sizeof(float));
    return 0;
}

// both halves at once (the whole route on the iGPU)
__attribute__((visibility("default"))) int abyss_igpu_experts_x3(const int* K3, int cb, const size_t* off9, int n_embd,
                                                                 int n_ff, float limit, int ne, const void* const* blobs,
                                                                 const float* w, const float* x, float* out) {
    const int rc = abyss_igpu_x3_start(K3, cb, off9, n_embd, n_ff, limit, ne, blobs, w, x);
    return rc != 0 ? rc : abyss_igpu_x3_finish(out, n_embd);
}

__attribute__((visibility("default"))) void abyss_igpu_unregister(const void* host) {
    std::lock_guard<std::mutex> lk(L.mu);
    if (!L.open) return;
    for (size_t i = 0; i < L.ranges.size(); ++i)
        if (L.ranges[i].host == (const uint8_t*) host) {
            DevScope ds;
            hipStreamSynchronize(L.s);
            if (L.ranges[i].owned) hipHostUnregister(const_cast<void*>(host));
            L.ranges.erase(L.ranges.begin() + (long) i);
            return;
        }
}

__attribute__((visibility("default"))) int abyss_igpu_rows(int gu_type, int d_type, int n_embd, int n_ff,
                                                           size_t down_off, float limit, int ne,
                                                           const void* const* blobs, const int* r0, const int* nr,
                                                           const int* row_tok, const float* x, int n_tok, int rlo,
                                                           int rhi, float* out, int ld) {
    DevScope ds;
    std::lock_guard<std::mutex> lk(L.mu);   // (a split's parts share the lane: a prompt's call waits its turn)
    if (!L.open || ne < 1 || rhi <= rlo || ld < n_embd || ld < 2 * n_ff) return 1;
    if (L.pending) {   // a late decode call still holds the queue (abyss_igpu_experts)
        if (hipStreamQuery(L.s) != hipSuccess) return 1;
        L.pending = false;
    }
    const unsigned long long xd = dev_of(x), td = dev_of(row_tok), od = dev_of(out);
    if (xd == 0ull || td == 0ull || od == 0ull || dev_of(x + (size_t) n_tok * n_embd - 1) != xd + ((size_t) n_tok * n_embd - 1) * sizeof(float) ||
        dev_of(out + (size_t) rhi * ld - 1) != od + ((size_t) rhi * ld - 1) * sizeof(float))
        return 1;
    hipError_t e = hipSuccess;
    if (ne > L.r_cap) {
        if (L.rptr_h) hipHostFree(L.rptr_h);
        if (L.rlight_h) hipHostFree(L.rlight_h);
        L.rptr_h = nullptr;
        L.rlight_h = nullptr;
        L.r_cap = 0;
        if ((e = hipHostMalloc((void**) &L.rptr_h, (size_t) ne * sizeof(unsigned long long), hipHostMallocMapped)) != hipSuccess ||
            (e = hipHostMalloc((void**) &L.rlight_h, (size_t) ne * 3 * sizeof(int), hipHostMallocMapped)) != hipSuccess)
            return -1;
        L.r_cap = ne;
    }
    const size_t xq_b = (size_t) n_tok * (n_embd / 32) * kQ8, hq_b = (size_t) (rhi - rlo) * (n_ff / 32) * kQ8;
    const auto grow = [&](void*& p, size_t& have, size_t need) {
        if (need <= have) return hipSuccess;
        if (p) hipFree(p);
        p = nullptr;
        have = 0;
        const hipError_t r = hipMalloc(&p, need + need / 4);
        if (r == hipSuccess) have = need + need / 4;
        return r;
    };
    if ((e = grow(L.rxq, L.rxq_b, xq_b)) != hipSuccess || (e = grow(L.rhq, L.rhq_b, hq_b)) != hipSuccess) return -1;
    for (int i = 0; i < ne; ++i) {
        const unsigned long long d = dev_of(blobs[i]);
        if (d == 0ull) return 1;
        L.rptr_h[i] = d;
        L.rlight_h[3 * i] = i;
        L.rlight_h[3 * i + 1] = r0[i];
        L.rlight_h[3 * i + 2] = nr[i];
    }
    unsigned long long pd = 0ull, ld_d = 0ull;
    hipHostGetDevicePointer((void**) &pd, L.rptr_h, 0);
    hipHostGetDevicePointer((void**) &ld_d, L.rlight_h, 0);
    // The fused int8-WMMA pass (igpu_fused.cu, from Maya's moe_fused_rdna4.cu) where it covers the formats and the
    // geometry (DS4's IQ2_XS / IQ3_XXS experts at 4096 / 2048) and the experts' rows are one ascending, contiguous
    // range; else glm_fast's row kernels.  STRATA_GLM_IGPU_FUSED=0: the row kernels always.
    // ABYSS_IGPU_FUSED_CHECK=<n>: the first n fused calls are also computed by the row kernels and compared (stderr).
    namespace fz = abyss::igpu_fused;
    static const int chk = [] { const char* v = getenv("ABYSS_IGPU_FUSED_CHECK"); return v != nullptr ? std::atoi(v) : 0; }();
    // (L.dev, not 0: beside a HIP engine on a discrete Radeon, HIP's device 0 is that card)
    bool fused = fz::available(L.dev) && fz::supported(gu_type, d_type, n_embd, n_ff) && ld == n_embd && n_tok > 0 &&
                 r0[0] == rlo && (((uintptr_t) od & 15) == 0);
    const size_t gu_row = fz::row_bytes(gu_type, n_embd);
    if (fused && down_off != 2 * (size_t) n_ff * gu_row) fused = false;   // (gate rows, up rows, then down: the lane's layout)
    for (int i = 0; fused && i < ne; ++i)
        fused = nr[i] >= 0 && r0[i] + nr[i] <= rhi && (i + 1 == ne ? r0[i] + nr[i] == rhi : r0[i] + nr[i] == r0[i + 1]);
    if (fused) {
        if (ne + 1 > L.fb_cap) {
            if (L.fb_h) hipHostFree(L.fb_h);
            L.fb_h = nullptr;
            L.fb_cap = 0;
            if ((e = hipHostMalloc((void**) &L.fb_h, (size_t) (ne + 1) * sizeof(int), hipHostMallocMapped)) != hipSuccess) return -1;
            L.fb_cap = ne + 1;
        }
        int ntiles = 0;
        for (int i = 0; i < ne; ++i) {
            L.fb_h[i] = r0[i] - rlo;
            ntiles += (nr[i] + 63) / 64;
        }
        L.fb_h[ne] = rhi - rlo;
        if (ntiles > L.ft_cap) {
            if (L.ft_h) hipHostFree(L.ft_h);
            L.ft_h = nullptr;
            L.ft_cap = 0;
            if ((e = hipHostMalloc((void**) &L.ft_h, (size_t) ntiles * sizeof(int2), hipHostMallocMapped)) != hipSuccess) return -1;
            L.ft_cap = ntiles;
        }
        ntiles = fz::make_tiles(L.fb_h, ne, L.ft_h, L.ft_cap);
        const size_t xa_b = fz::act_bytes((size_t) n_tok, n_embd), ha_b = fz::act_bytes((size_t) (rhi - rlo), n_ff);
        if (ntiles < 0 || (e = grow(L.rxq, L.rxq_b, std::max(xa_b, xq_b))) != hipSuccess ||
            (e = grow(L.rhq, L.rhq_b, std::max(ha_b, hq_b))) != hipSuccess)
            return -1;
        unsigned long long fb_d = 0ull, ft_d = 0ull;
        hipHostGetDevicePointer((void**) &fb_d, L.fb_h, 0);
        hipHostGetDevicePointer((void**) &ft_d, L.ft_h, 0);
        fz::Geometry geo;
        geo.gu_type = gu_type;
        geo.d_type = d_type;
        geo.gu_row = gu_row;
        geo.d_row = fz::row_bytes(d_type, n_ff);
        geo.up_off = (size_t) n_ff * gu_row;
        geo.down_off = down_off;
        geo.limit = limit;
        if (chk > 0 && (long long) L.n_fused < chk) {   // the check's x: saved first, as the gathered x IS out (overwritten below)
            if ((e = grow(L.rxsave, L.rxsave_b, (size_t) n_tok * n_embd * sizeof(float))) != hipSuccess) return -1;
            hipMemcpyAsync(L.rxsave, (const void*) xd, (size_t) n_tok * n_embd * sizeof(float), hipMemcpyDeviceToDevice, L.s);
        }
        fz::quantize((const float*) xd, (size_t) n_tok, n_embd, L.rxq, L.s);
        if (!fz::experts((const unsigned long long*) pd, ne, (const int*) fb_d, (const int2*) ft_d, ntiles, geo, L.rxq,
                         (const int*) td + rlo, L.rhq, (float*) od + (size_t) rlo * ld, L.s)) {
            hipStreamSynchronize(L.s);
            fused = false;
        }
    }
    if (fused && chk > 0 && (long long) L.n_fused < chk) {
        // the row kernels' answer beside the fused one: out rows [rlo, rhi) of each, relative error and largest gap
        if ((e = hipStreamSynchronize(L.s)) != hipSuccess) return -1;
        const size_t need = (size_t) rhi * ld * sizeof(float);
        if ((e = grow(L.rchk, L.rchk_b, need)) != hipSuccess) return -1;
        if (!gf::rows_experts(gu_type, d_type, nullptr, 0, down_off, (const int*) ld_d, ne, (const int*) td,
                              (const float*) L.rxsave, n_tok, n_embd, n_ff, limit, rlo, rhi, L.rxq, L.rhq, (float*) L.rchk, ld,
                              L.s, (const unsigned long long*) pd) ||
            (e = hipStreamSynchronize(L.s)) != hipSuccess)
            return -1;
        std::vector<float> ref((size_t) (rhi - rlo) * ld);
        if ((e = hipMemcpy(ref.data(), (const float*) L.rchk + (size_t) rlo * ld, ref.size() * sizeof(float),
                           hipMemcpyDeviceToHost)) != hipSuccess)
            return -1;
        double num = 0.0, den = 0.0, mx = 0.0;
        size_t nonfinite = 0;
        for (int r = rlo; r < rhi; ++r)
            for (int c = 0; c < n_embd; ++c) {
                const double a = out[(size_t) r * ld + c], b = ref[(size_t) (r - rlo) * ld + c];
                if (!std::isfinite(a)) ++nonfinite;
                num += (a - b) * (a - b);
                den += b * b;
                mx = std::max(mx, std::fabs(a - b));
            }
        std::fprintf(stderr, "iGPU fused check %llu: gu %d down %d, %d experts, %d rows: rel %.3e, max |d| %.3e, nonfinite %zu\n",
                     L.n_fused + 1, gu_type, d_type, ne, rhi - rlo, std::sqrt(num / std::max(den, 1e-30)), mx, nonfinite);
    }
    if (!fused) {
        if (!gf::rows_experts(gu_type, d_type, nullptr, 0, down_off, (const int*) ld_d, ne, (const int*) td,
                              (const float*) xd, n_tok, n_embd, n_ff, limit, rlo, rhi, L.rxq, L.rhq, (float*) od, ld, L.s,
                              (const unsigned long long*) pd))
            return 1;
    }
    if ((e = hipStreamSynchronize(L.s)) != hipSuccess) {
        std::fprintf(stderr, "iGPU lane (prompt rows): %s\n", hipGetErrorString(e));
        return -1;
    }
    if (fused) ++L.n_fused; else ++L.n_rows_k;
    return 0;
}

__attribute__((visibility("default"))) void abyss_igpu_poke(void) {
    DevScope ds;
    // ABYSS_IGPU_POKE_STAT=1 (investigation): how long the idle-stream query and the launch take, every 65,536 pokes
    static const bool stat = getenv("ABYSS_IGPU_POKE_STAT") != nullptr;
    if (!stat) {
        if (L.open && hipStreamQuery(L.s) == hipSuccess) gf::lane_poke(L.s);
        return;
    }
    static unsigned long long n = 0, q_us = 0, q_max = 0, l_us = 0, l_max = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const bool idle = L.open && hipStreamQuery(L.s) == hipSuccess;
    const auto t1 = std::chrono::steady_clock::now();
    if (idle) gf::lane_poke(L.s);
    const auto t2 = std::chrono::steady_clock::now();
    const unsigned long long q = (unsigned long long) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    const unsigned long long l = (unsigned long long) std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    q_us += q; q_max = std::max(q_max, q); l_us += l; l_max = std::max(l_max, l);
    if (++n % 65536 == 0) {
        std::fprintf(stderr, "iGPU poke stat: %llu pokes | query %.1f us (max %llu) | launch %.1f us (max %llu)\n", n,
                     (double) q_us / 65536.0, q_max, (double) l_us / 65536.0, l_max);
        q_us = l_us = 0; q_max = l_max = 0;
    }
}

__attribute__((visibility("default"))) void abyss_igpu_close(void) {
    std::lock_guard<std::mutex> lk(L.mu);
    if (!L.open) return;
    DevScope ds;
    hipStreamSynchronize(L.s);
    for (const Range& r : L.ranges)
        if (r.owned) hipHostUnregister(const_cast<uint8_t*>(r.host));
    L.ranges.clear();
    hipFree(L.xq);
    hipFree(L.hq);
    if (L.planc_h) hipHostFree(L.planc_h);
    if (L.ca_h) hipHostFree(L.ca_h);
    if (L.xqc_h) hipHostFree(L.xqc_h);
    if (L.xc_h) hipHostFree(L.xc_h);
    if (L.hqc) hipFree(L.hqc);
    if (L.outc_h) hipHostFree(L.outc_h);
    if (L.un_h) hipHostFree(L.un_h);
    if (L.rxq) hipFree(L.rxq);
    if (L.rhq) hipFree(L.rhq);
    if (L.rptr_h) hipHostFree(L.rptr_h);
    if (L.rlight_h) hipHostFree(L.rlight_h);
    if (L.fb_h) hipHostFree(L.fb_h);
    if (L.ft_h) hipHostFree(L.ft_h);
    if (L.rchk) hipFree(L.rchk);
    if (L.rxsave) hipFree(L.rxsave);
    L.fb_h = nullptr;
    L.ft_h = nullptr;
    L.rchk = L.rxsave = nullptr;
    L.fb_cap = L.ft_cap = 0;
    L.rchk_b = L.rxsave_b = 0;
    if (L.n_fused + L.n_rows_k > 0)
        std::fprintf(stderr, "iGPU lane: prompt calls - %llu fused, %llu on the row kernels\n", L.n_fused, L.n_rows_k);
    L.rxq = L.rhq = nullptr;
    L.rxq_b = L.rhq_b = 0;
    L.rptr_h = nullptr;
    L.rlight_h = nullptr;
    L.r_cap = 0;
    hipHostFree(L.x_h);
    hipHostFree(L.out_h);
    hipHostFree(L.plan_h);
    hipStreamDestroy(L.s);
    L.open = false;
}

}  // extern "C"
