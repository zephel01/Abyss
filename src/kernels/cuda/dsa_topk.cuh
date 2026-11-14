// src/kernels/cuda/dsa_topk.cuh - the DSA indexer's choice of the top pools in O(n_vis) (one block, 1024 threads).
//
// The order it writes is the one the O(n_vis^2) rank loop wrote (score descending, ties by the lower pool index), so
// the selected cells - and every result built on them - are unchanged; only the cost stops growing with the square of
// the context (at 30k tokens the old loop was ~56M comparisons per DSA layer per token, in one block).
//   1. a radix select over the scores' order-preserving uint32 keys (three passes of 12, 12 and 8 bits, the cut-off
//      bin found by a block scan) finds the cut-off key - the top-th largest - and how many keys equal to it are taken;
//   2. the winners above the cut are collected in any order with coalesced reads; the ones equal to it too when all of
//      them are taken, else the first `need` in pool order (the lower indices, as the rank loop chose);
//   3. each winner's rank is the count of the winners larger than it (the values are distinct).
// One block at 2k keys: 18 -> 12 us, at 31k: 35 -> 25 (RTX 5090, v1.3.3; the earlier version: four 8-bit passes, a
// serial bin walk, a bitonic sort). The warp scans assume 32 lanes: the HIP backend runs wave32 only (device.cu), and
// on a Radeon 8060S (gfx1151) the same code takes 34 -> 14 us at 2k keys, 61 -> 27 at 31k, with the same cells.
#pragma once

#include <cuda_runtime.h>

namespace strata::kernels::dsa {

// a float's order-preserving key: larger float, larger key (-0 is +0; the scores are never NaN)
__device__ __forceinline__ unsigned score_key(float f) {
    const unsigned u = __float_as_uint(f == 0.0f ? 0.0f : f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}


// the most pools a call may select: DeepSeek V4's indexer takes 512 (2048 cells / 4), V4.1's candidate blocks 2048
// (cand_blocks, 1 a pool).  Before 2026-10-19 the winners' arrays held 1024: V4.1 past 8K tokens (more blocks than
// that) selected from corrupt ranks, and past 16K (top = 2048 with blocks left over) wrote the selection out of
// range - an illegal address on the 16,385th token of every chat
constexpr int kMaxTop = 2048;

// cells[r * kpool + m] = p * kpool + m for the top `top` pools p of sc[0, n_vis) in rank order; 1 <= top <= kMaxTop,
// top <= n_vis, blockDim.x == 1024.  Every thread of the block must call it.  (static: each kernel file that includes
// this gets its own copy - an external symbol would be defined twice at link time)
static __device__ void select_top_pools(const float* __restrict__ sc, int n_vis, int top, int kpool, int* __restrict__ cells) {
    constexpr int NT = 1024;
    __shared__ unsigned s_hist[4096];
    __shared__ unsigned s_wsum[32];
    __shared__ unsigned s_prefix, s_mask, s_eq_all;
    __shared__ int s_need, s_ngt, s_neq;
    __shared__ int s_eq[NT];
    __shared__ unsigned long long s_win[kMaxTop];
    __shared__ unsigned short s_rank[2][kMaxTop];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    if (tid == 0) {
        s_prefix = 0u;
        s_mask = 0u;
        s_need = top;
        s_ngt = 0;
        s_neq = 0;
    }
    __syncthreads();
    // 1. the cut-off key in three passes of 12, 12 and 8 bits from the top
    for (int pass = 0; pass < 3; ++pass) {
        const int bits = pass < 2 ? 12 : 8, shift = pass == 0 ? 20 : pass == 1 ? 8 : 0;
        const unsigned nb = 1u << bits, bmask = nb - 1u;
        for (int i = tid; i < (int) nb; i += NT) s_hist[i] = 0u;
        __syncthreads();
        // (need too, here: the thread that finds the bin below rewrites s_need, and a thread of another warp that read
        // it after that write could take a second bin - a race seen on HIP as wrong cells, an illegal access later)
        const unsigned prefix = s_prefix, mask = s_mask, need = (unsigned) s_need;
        for (int i = tid; i < n_vis; i += NT) {
            const unsigned k = score_key(sc[i]);
            if ((k & mask) == prefix) atomicAdd(&s_hist[(k >> shift) & bmask], 1u);
        }
        __syncthreads();
        // the bin that holds the need-th largest: thread t sums the 4 bins nb-1-4t .. nb-4-4t (from the top), then a
        // block scan of those sums
        const int per = (int) nb / NT;   // 4, or 0 for the 8-bit pass (256 bins: one a thread for t < 256)
        unsigned c4[4] = {0u, 0u, 0u, 0u}, own = 0u;
        if (per == 4) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                c4[j] = s_hist[nb - 1u - (unsigned) (4 * tid + j)];
                own += c4[j];
            }
        } else if (tid < (int) nb) {
            c4[0] = s_hist[nb - 1u - (unsigned) tid];
            own = c4[0];
        }
        unsigned incl = own;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned v = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane >= o) incl += v;
        }
        if (lane == 31) s_wsum[wid] = incl;
        __syncthreads();
        if (wid == 0) {
            unsigned v = s_wsum[lane], x = v;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned y = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += y;
            }
            s_wsum[lane] = x - v;   // exclusive: the warps before
        }
        __syncthreads();
        incl += s_wsum[wid];
        unsigned before = incl - own;
        if (before < need && incl >= need) {
            const int m = per == 4 ? 4 : 1;
            for (int j = 0; j < m; ++j) {
                if (before + c4[j] >= need) {
                    const unsigned bin = nb - 1u - (unsigned) (per == 4 ? 4 * tid + j : tid);
                    s_prefix = prefix | (bin << shift);
                    s_mask = mask | (bmask << shift);
                    s_need = (int) (need - before);
                    s_eq_all = c4[j];   // after the last pass: how many keys equal the cut
                    break;
                }
                before += c4[j];
            }
        }
        __syncthreads();
    }
    const unsigned cut = s_prefix;
    const int need_eq = s_need;
    const int n_gt = top - need_eq;
    // 2. the winners. Every key above the cut, in any order (the sort below orders them); the keys equal to it in any
    // order too when all of them are taken, else the first need_eq in pool order (a contiguous run a thread, scans)
    const bool all_eq = (int) s_eq_all == need_eq;
    for (int i = tid; i < n_vis; i += NT) {
        const unsigned k = score_key(sc[i]);
        const unsigned long long w = ((unsigned long long) k << 32) | (unsigned long long) (~(unsigned) i);
        if (k > cut) s_win[atomicAdd(&s_ngt, 1)] = w;
        else if (k == cut && all_eq) s_win[n_gt + atomicAdd(&s_neq, 1)] = w;
    }
    if (!all_eq) {
        const int chunk = (n_vis + NT - 1) / NT;
        const int lo = min(n_vis, tid * chunk), hi = min(n_vis, lo + chunk);
        int eq = 0;
        for (int i = lo; i < hi; ++i) eq += score_key(sc[i]) == cut;
        s_eq[tid] = eq;
        __syncthreads();
        for (int off = 1; off < NT; off <<= 1) {
            int b = 0;
            if (tid >= off) b = s_eq[tid - off];
            __syncthreads();
            s_eq[tid] += b;
            __syncthreads();
        }
        int oe = s_eq[tid] - eq;
        for (int i = lo; i < hi && oe < need_eq; ++i) {
            const unsigned k = score_key(sc[i]);
            if (k == cut) {
                s_win[n_gt + oe] = ((unsigned long long) k << 32) | (unsigned long long) (~(unsigned) i);
                ++oe;
            }
        }
    }
    __syncthreads();
    // 3. the rank of each winner: how many are larger (the values are distinct: the index is in the low bits)
    {
        const int i = tid % top, half = tid / top;   // top <= 512: two threads a winner, each half of the others
        if (top <= NT / 2) {
            if (tid < 2 * top) {
                const unsigned long long v = s_win[i];
                const int j0 = half ? top / 2 : 0, j1 = half ? top : top / 2;
                int r = 0;
                for (int j = j0; j < j1; ++j) r += s_win[j] > v;
                s_rank[half][i] = (unsigned short) r;
            }
        } else {                                      // up to kMaxTop: a thread takes the winners tid, tid + NT, ...
            for (int w = tid; w < top; w += NT) {
                const unsigned long long v = s_win[w];
                int r = 0;
                for (int j = 0; j < top; ++j) r += s_win[j] > v;
                s_rank[0][w] = (unsigned short) r;
                s_rank[1][w] = 0;
            }
        }
    }
    __syncthreads();
    for (int w = tid; w < top; w += NT) {
        const int r = s_rank[0][w] + s_rank[1][w];
        const int p = (int) ~(unsigned) (s_win[w] & 0xffffffffull);
        for (int m = 0; m < kpool; ++m) cells[r * kpool + m] = p * kpool + m;
    }
}


// ---------------------------------------------------------------- the same choice over several blocks
// Past ~10k pools the one block above is a long serial walk while the rest of the GPU waits: its four passes over the
// scores (three histograms, the collection) are a block's loads one after another, and its histogram atomics pile up
// on the few bins the similar scores share.  select_top_pools_mb splits every pass over gridDim.x blocks (each a slice
// of the scores, kept in registers) and adds their histograms in global memory, with a grid barrier between passes:
//   1. the same three radix passes (12, 12, 8 bits): each block counts its slice into shared bins, adds the non-zero
//      ones to MbScratch::hist[pass], waits at the barrier, then every block reads the whole histogram and runs the
//      same bin search as the one-block kernel - so every block holds the same cut, need and count of equal keys;
//   2. the winners: each block stages its keys above the cut (and the equal ones when all of them are taken) and
//      appends them to MbScratch::win with one atomic a block; when only some equal keys are taken, block 0 picks the
//      first need_eq in pool order exactly as the one-block kernel does (contiguous runs a thread, a block scan);
//   3. after the last barrier each block loads the `top` winners and ranks its share of them (count of larger values;
//      the values are distinct), writing cells.
// Same counts, same cut, same winners, same ranks: the cells are the one-block kernel's, bit for bit.
// Rules: every block of the grid must be resident at once (the caller keeps gridDim.x well under the SM count),
// 1024 threads a block, top <= n_vis, the scratch zero before the first call; the kernel leaves it zero again (the
// last block out resets the barrier), so one scratch serves calls one after another on one stream, never two at once.
constexpr int kMbMaxBlocks = 64;
struct MbScratch {
    unsigned hist[3][4096];
    unsigned long long win[kMaxTop];
    unsigned bar, done, ngt, neq;
};

__device__ __forceinline__ void mb_grid_sync(unsigned* bar, unsigned target) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(bar, 1u);
        while (*(volatile unsigned*) bar < target) __nanosleep(32);
        __threadfence();
    }
    __syncthreads();
}

static __device__ void select_top_pools_mb(const float* __restrict__ sc, int n_vis, int top, int kpool,
                                           int* __restrict__ cells, MbScratch* __restrict__ ws) {
    constexpr int NT = 1024, KR = 8;   // KR keys a thread in registers: slices up to 8192 (the rest re-read)
    __shared__ unsigned s_hist[4096];
    __shared__ unsigned s_wsum[32];
    __shared__ unsigned s_prefix, s_mask, s_eq_all;
    __shared__ int s_need, s_cgt, s_ceq, s_bgt, s_beq;
    __shared__ int s_eq[NT];
    __shared__ int s_rk[NT];
    __shared__ unsigned long long s_win[kMaxTop];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int G = (int) gridDim.x, g = (int) blockIdx.x;
    const int chunk = (n_vis + G - 1) / G;
    const int lo = min(n_vis, g * chunk), hi = min(n_vis, lo + chunk);
    unsigned kv[KR];
#pragma unroll
    for (int j = 0; j < KR; ++j) {
        const int i = lo + tid + j * NT;
        kv[j] = i < hi ? score_key(sc[i]) : 0u;
    }
    if (tid == 0) {
        s_prefix = 0u;
        s_mask = 0u;
        s_need = top;
        s_cgt = 0;
        s_ceq = 0;
    }
    __syncthreads();
    // 1. the cut-off key: the one-block kernel's three passes, the histograms summed over the blocks
    for (int pass = 0; pass < 3; ++pass) {
        const int bits = pass < 2 ? 12 : 8, shift = pass == 0 ? 20 : pass == 1 ? 8 : 0;
        const unsigned nb = 1u << bits, bmask = nb - 1u;
        for (int i = tid; i < (int) nb; i += NT) s_hist[i] = 0u;
        __syncthreads();
        const unsigned prefix = s_prefix, mask = s_mask, need = (unsigned) s_need;
#pragma unroll
        for (int j = 0; j < KR; ++j) {
            const unsigned k = kv[j];
            if (lo + tid + j * NT < hi && (k & mask) == prefix) atomicAdd(&s_hist[(k >> shift) & bmask], 1u);
        }
        for (int i = lo + tid + KR * NT; i < hi; i += NT) {
            const unsigned k = score_key(sc[i]);
            if ((k & mask) == prefix) atomicAdd(&s_hist[(k >> shift) & bmask], 1u);
        }
        __syncthreads();
        unsigned* gh = ws->hist[pass];
        for (int i = tid; i < (int) nb; i += NT) {
            const unsigned v = s_hist[i];
            if (v != 0u) atomicAdd(&gh[i], v);
        }
        mb_grid_sync(&ws->bar, (unsigned) (G * (pass + 1)));
        for (int i = tid; i < (int) nb; i += NT) s_hist[i] = ((volatile unsigned*) gh)[i];
        __syncthreads();
        // (from here as select_top_pools: the bin that holds the need-th largest)
        const int per = (int) nb / NT;
        unsigned c4[4] = {0u, 0u, 0u, 0u}, own = 0u;
        if (per == 4) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                c4[j] = s_hist[nb - 1u - (unsigned) (4 * tid + j)];
                own += c4[j];
            }
        } else if (tid < (int) nb) {
            c4[0] = s_hist[nb - 1u - (unsigned) tid];
            own = c4[0];
        }
        unsigned incl = own;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned v = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane >= o) incl += v;
        }
        if (lane == 31) s_wsum[wid] = incl;
        __syncthreads();
        if (wid == 0) {
            unsigned v = s_wsum[lane], x = v;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned y = __shfl_up_sync(0xffffffffu, x, o);
                if (lane >= o) x += y;
            }
            s_wsum[lane] = x - v;
        }
        __syncthreads();
        incl += s_wsum[wid];
        unsigned before = incl - own;
        if (before < need && incl >= need) {
            const int m = per == 4 ? 4 : 1;
            for (int j = 0; j < m; ++j) {
                if (before + c4[j] >= need) {
                    const unsigned bin = nb - 1u - (unsigned) (per == 4 ? 4 * tid + j : tid);
                    s_prefix = prefix | (bin << shift);
                    s_mask = mask | (bmask << shift);
                    s_need = (int) (need - before);
                    s_eq_all = c4[j];
                    break;
                }
                before += c4[j];
            }
        }
        __syncthreads();
    }
    const unsigned cut = s_prefix;
    const int need_eq = s_need;
    const int n_gt = top - need_eq;
    const bool all_eq = (int) s_eq_all == need_eq;
    // (every block is past the third barrier: no one reads hist[0] or hist[1] again - clear them for the next call)
    for (int i = g * NT + tid; i < 2 * 4096; i += G * NT) (&ws->hist[0][0])[i] = 0u;
    // 2. the winners: staged in shared memory (the ones above the cut from the bottom, equal ones from the top), then
    // appended to ws->win with one atomic a block
#pragma unroll
    for (int j = 0; j < KR; ++j) {
        const int i = lo + tid + j * NT;
        const unsigned k = kv[j];
        if (i < hi) {
            const unsigned long long w = ((unsigned long long) k << 32) | (unsigned long long) (~(unsigned) i);
            if (k > cut) s_win[atomicAdd(&s_cgt, 1)] = w;
            else if (k == cut && all_eq) s_win[kMaxTop - 1 - atomicAdd(&s_ceq, 1)] = w;
        }
    }
    for (int i = lo + tid + KR * NT; i < hi; i += NT) {
        const unsigned k = score_key(sc[i]);
        const unsigned long long w = ((unsigned long long) k << 32) | (unsigned long long) (~(unsigned) i);
        if (k > cut) s_win[atomicAdd(&s_cgt, 1)] = w;
        else if (k == cut && all_eq) s_win[kMaxTop - 1 - atomicAdd(&s_ceq, 1)] = w;
    }
    __syncthreads();
    if (tid == 0) {
        s_bgt = s_cgt ? (int) atomicAdd(&ws->ngt, (unsigned) s_cgt) : 0;
        s_beq = s_ceq ? (int) atomicAdd(&ws->neq, (unsigned) s_ceq) : 0;
    }
    __syncthreads();
    for (int t = tid; t < s_cgt; t += NT) ws->win[s_bgt + t] = s_win[t];
    for (int t = tid; t < s_ceq; t += NT) ws->win[n_gt + s_beq + t] = s_win[kMaxTop - 1 - t];
    if (!all_eq && g == 0) {   // the first need_eq equal keys in pool order: select_top_pools' scan, over all n_vis
        const int ch = (n_vis + NT - 1) / NT;
        const int a = min(n_vis, tid * ch), b = min(n_vis, a + ch);
        int eq = 0;
        for (int i = a; i < b; ++i) eq += score_key(sc[i]) == cut;
        s_eq[tid] = eq;
        __syncthreads();
        for (int off = 1; off < NT; off <<= 1) {
            int x = 0;
            if (tid >= off) x = s_eq[tid - off];
            __syncthreads();
            s_eq[tid] += x;
            __syncthreads();
        }
        int oe = s_eq[tid] - eq;
        for (int i = a; i < b && oe < need_eq; ++i) {
            const unsigned k = score_key(sc[i]);
            if (k == cut) {
                ws->win[n_gt + oe] = ((unsigned long long) k << 32) | (unsigned long long) (~(unsigned) i);
                ++oe;
            }
        }
    }
    mb_grid_sync(&ws->bar, (unsigned) (G * 4));
    // 3. the ranks of this block's share of the winners
    for (int i = g * NT + tid; i < 256; i += G * NT) ws->hist[2][i] = 0u;
    if (g == 0 && tid == 0) {
        ws->ngt = 0u;
        ws->neq = 0u;
    }
    for (int t = tid; t < top; t += NT) s_win[t] = ((volatile unsigned long long*) ws->win)[t];
    const int per = (top + G - 1) / G, w0 = min(top, g * per), nw = min(top, w0 + per) - w0;
    if (tid < nw && nw <= NT) s_rk[tid] = 0;
    __syncthreads();
    if (nw > NT) {   // (a grid of one block with top > 1024: a thread a winner, all of them compared)
        for (int wl = tid; wl < nw; wl += NT) {
            const unsigned long long v = s_win[w0 + wl];
            int r = 0;
            for (int j = 0; j < top; ++j) r += s_win[j] > v;
            const int p = (int) ~(unsigned) (v & 0xffffffffull);
            for (int m = 0; m < kpool; ++m) cells[r * kpool + m] = p * kpool + m;
        }
    } else if (nw > 0) {
        const int parts = NT / nw, wl = tid % nw, part = tid / nw;
        if (part < parts) {
            const unsigned long long v = s_win[w0 + wl];
            int r = 0;
            for (int j = part; j < top; j += parts) r += s_win[j] > v;
            atomicAdd(&s_rk[wl], r);
        }
    }
    __syncthreads();
    for (int wl = tid; wl < nw && nw <= NT; wl += NT) {
        const int r = s_rk[wl];
        const int p = (int) ~(unsigned) (s_win[w0 + wl] & 0xffffffffull);
        for (int m = 0; m < kpool; ++m) cells[r * kpool + m] = p * kpool + m;
    }
    // the last block out resets the barrier (every block is past it by then)
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(&ws->done, 1u) == (unsigned) G - 1u) {
            atomicExch(&ws->bar, 0u);
            atomicExch(&ws->done, 0u);
        }
    }
}


}  // namespace strata::kernels::dsa
