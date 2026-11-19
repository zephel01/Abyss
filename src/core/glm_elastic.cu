// src/core/glm_elastic.cu - elastic KV (--kv-elastic, --kv-min): the qwen35 family's full-context attention caches on
// CUDA's virtual memory API, mapped as the conversation grows, and the VRAM they do not use yet lent to the weights.
//
// --max-context reserves the caches' ADDRESS SPACE whole (one reservation, every cache a contiguous range, so the
// attention kernels index rows exactly as before); physical VRAM is mapped behind the first --kv-min positions at load
// and more as a prompt or the decode reaches past it (in steps of kStep positions).  What the unmapped part would have
// taken goes to the weights:
//   - an MoE model (Laguna, Qwen3.8-Flash-Next, Ornith): every layer's expert partition gets E more slots on top of
//     what the fixed layout gives it (the "elastic ranks", the partition's last E slots).  The partitions are on the
//     same API: a layer's core slots are one mapping, its elastic ranks 2 MB pieces after them.  When the KV must grow
//     the top ranks leave (their experts move over colder ones or go to the RAM tier - lend_slots, the prompt path's
//     lending - and the slots become kElastic), their pieces are unmapped and the KV maps the memory they held.
//   - the dense Qwen3.8-27B: whole layers' FFN that the fixed layout leaves (partly) on the CPU get a full VRAM copy
//     (q35d_elastic_*, qwen35_dense_ram.cu); the KV takes those back the same way (the layer runs on its RAM part again).
// A new conversation (reset) lets the next prompt unmap the KV past what it needs (and --kv-min) and map the ranks
// again (their slots refill as misses).  The math is the same either way: only where the weights are read from moves.
// Invariant: mapped KV + mapped elastic weight pieces <= the caches' full size, so the KV can always reach the context.
// Where the API is missing (HIP, a device without it), a split across GPUs, unified memory or --kv-elastic 0: the
// caches are one cudaMalloc as before (fixed).
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"

#include <chrono>
#include <cuda_runtime.h>
#if !defined(STRATA_USE_HIP)
#include <cuda.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;

namespace {
constexpr int64_t kStep = 8192;     // positions mapped at a time past --kv-min
constexpr int64_t kMargin = 512;    // positions past a request's end mapped as well (verify windows, drafts, tiles)
constexpr int64_t kSlack = 256;     // ... and at least this many must be mapped past it already, else the KV grows
constexpr int64_t kMinTokens = 32768;
size_t up(size_t b, size_t g) { return (b + g - 1) / g * g; }
}  // namespace

#if !defined(STRATA_USE_HIP)
namespace {
// the driver's virtual memory entry points through the runtime (no link against libcuda)
struct Vmm {
    CUresult (*reserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long) = nullptr;
    CUresult (*addr_free)(CUdeviceptr, size_t) = nullptr;
    CUresult (*create)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) = nullptr;
    CUresult (*release)(CUmemGenericAllocationHandle) = nullptr;
    CUresult (*map)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) = nullptr;
    CUresult (*unmap)(CUdeviceptr, size_t) = nullptr;
    CUresult (*access)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t) = nullptr;
    CUresult (*gran)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags) = nullptr;
    CUresult (*dev_get)(CUdevice*, int) = nullptr;
    CUresult (*dev_attr)(int*, CUdevice_attribute, CUdevice) = nullptr;
    bool ok = false;
};

template <class F>
bool entry(const char* name, F& fn) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
#if CUDART_VERSION >= 12050
    if (cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q) != cudaSuccess) {
#else
    if (cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q) != cudaSuccess) {
#endif
        cudaGetLastError();
        return false;
    }
    if (q != cudaDriverEntryPointSuccess || p == nullptr) return false;
    fn = reinterpret_cast<F>(p);
    return true;
}

const Vmm& vmm() {
    static const Vmm v = [] {
        Vmm a;
        a.ok = entry("cuMemAddressReserve", a.reserve) && entry("cuMemAddressFree", a.addr_free) &&
               entry("cuMemCreate", a.create) && entry("cuMemRelease", a.release) && entry("cuMemMap", a.map) &&
               entry("cuMemUnmap", a.unmap) && entry("cuMemSetAccess", a.access) &&
               entry("cuMemGetAllocationGranularity", a.gran) && entry("cuDeviceGet", a.dev_get) &&
               entry("cuDeviceGetAttribute", a.dev_attr);
        return a;
    }();
    return v;
}

CUmemAllocationProp prop_of(int dev) {
    CUmemAllocationProp p;
    std::memset(&p, 0, sizeof p);
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = dev;
    return p;
}
}  // namespace
#endif

struct Glm5Model::ElasticKv {
    struct Chunk {
        size_t off = 0, size = 0;
        unsigned long long h = 0;   // (CUmemGenericAllocationHandle)
    };
    struct Region {   // one cache: [off, off + cap) of the reservation, row bytes a position
        size_t off = 0, cap = 0, row = 0, mapped = 0;
        std::vector<Chunk> ch;
    };
    struct PLayer {   // one layer's expert partition: core slots, then up to e elastic ones
        int il = -1, core = 0, e = 0;
        size_t off = 0, stride = 0, guard = 0, mapped = 0;
        std::vector<Chunk> ch;
        size_t extent(size_t gran, int r) const {   // the bytes mapped with r elastic slots live
            return up((size_t) (core + std::min(r, e)) * stride + guard, gran);
        }
    };
    int dev = 0;
    size_t gran = 0;
    unsigned long long kv_va = 0, p_va = 0;
    size_t kv_va_size = 0, p_va_size = 0;
    std::vector<Region> kv;
    int64_t ctx = 0, min_tok = 0, tok = 0;   // every cache holds positions [0, tok)
    size_t kv_cap = 0, kv_mapped = 0;        // (the budget: kv_mapped + lent <= kv_cap)
    std::vector<PLayer> pl;
    int E = 0, R = 0;                         // elastic ranks laid out / live
    size_t ep_mapped = 0;                     // the elastic pieces mapped (pool or dense layers)
    bool pool_on = false, zero_pool = false, shrink_pending = false;
    uint64_t grows = 0, shrinks = 0;

    size_t lent() const { return ep_mapped; }
    size_t pool_cost(int r) const {   // the elastic bytes mapped with r ranks live
        size_t b = 0;
        for (const auto& L : pl) b += L.extent(gran, r) - L.extent(gran, 0);
        return b;
    }
    int64_t tokens_held() const {
        int64_t t = ctx;
        for (const auto& r : kv) t = std::min<int64_t>(t, r.row > 0 ? (int64_t) (r.mapped / r.row) : ctx);
        return t;
    }
#if !defined(STRATA_USE_HIP)
    // maps [off, off + size) of the reservation at va: one physical piece; false (nothing held) when it did not
    bool map_piece(unsigned long long va, size_t off, size_t size, std::vector<Chunk>& out) {
        const Vmm& v = vmm();
        const CUmemAllocationProp p = prop_of(dev);
        CUmemGenericAllocationHandle h = 0;
        if (v.create(&h, size, &p, 0) != CUDA_SUCCESS) return false;
        if (v.map((CUdeviceptr) (va + off), size, 0, h, 0) != CUDA_SUCCESS) {
            v.release(h);
            return false;
        }
        CUmemAccessDesc a;
        std::memset(&a, 0, sizeof a);
        a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        a.location.id = dev;
        a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (v.access((CUdeviceptr) (va + off), size, &a, 1) != CUDA_SUCCESS) {
            v.unmap((CUdeviceptr) (va + off), size);
            v.release(h);
            return false;
        }
        out.push_back({off, size, (unsigned long long) h});
        return true;
    }
    // unmaps every piece of `ch` that starts at or past `from` (the device must be idle on them); bytes freed
    size_t unmap_from(unsigned long long va, std::vector<Chunk>& ch, size_t from) {
        const Vmm& v = vmm();
        size_t freed = 0;
        while (!ch.empty() && ch.back().off >= from) {
            const Chunk c = ch.back();
            ch.pop_back();
            v.unmap((CUdeviceptr) (va + c.off), c.size);
            v.release((CUmemGenericAllocationHandle) c.h);
            freed += c.size;
        }
        return freed;
    }
#endif
};

bool Glm5Model::ekv_on() const { return ekv_ != nullptr; }

size_t Glm5Model::ekv_lendable() const {
    return ekv_ != nullptr && ekv_->kv_cap > ekv_->kv_mapped ? ekv_->kv_cap - ekv_->kv_mapped : 0;
}

size_t Glm5Model::ekv_lent() const { return ekv_ != nullptr ? ekv_->lent() : 0; }

int64_t Glm5Model::ekv_min_tokens() {
    const char* v = getenv("STRATA_GLM_KV_MIN");
    const long long n = v != nullptr && v[0] != '\0' ? std::atoll(v) : (long long) kMinTokens;
    return std::max<int64_t>(4096, n);
}

bool Glm5Model::ekv_kv_setup(const std::vector<size_t>& rows, std::vector<void*>& ptrs) {
    ptrs.clear();
    const char* ev = getenv("STRATA_GLM_KV_ELASTIC");
    if (ev != nullptr && ev[0] != '\0' && std::atoi(ev) == 0) return false;   // --kv-elastic 0: the fixed layout
    if (rows.empty()) return false;
    const int64_t ctx = std::max<int64_t>(max_ctx_, 1), kmin = ekv_min_tokens();
    const auto fixed = [&](const char* why) {
        std::fprintf(stderr, "glm elastic: CUDA%d %s - the KV cache is allocated whole for %lld tokens (fixed)\n", dev_,
                     why, (long long) ctx);
        return false;
    };
    // (STRATA_GLM_KV_ELASTIC=2, a test: the caches on the API but mapped whole - nothing lent)
    const bool whole = ev != nullptr && std::atoi(ev) == 2;
    if (ctx <= kmin && !whole) return false;   // (the whole context is within --kv-min: nothing to lend)
#if defined(STRATA_USE_HIP)
    return fixed("HIP has no elastic KV yet (CUDA's virtual memory API)");
#else
    if (n_parts_ > 1 || split_next_ != nullptr || l0_ != 0 || l1_ != g_.n_layers)
        return fixed("a split across GPUs");
    if (fast_ != nullptr && fast_->unified_memory) return fixed("unified memory");
    const Vmm& v = vmm();
    if (!v.ok) return fixed("the driver's virtual memory entry points are missing");
    int vmm_ok = 0;
    CUdevice cd = 0;
    if (v.dev_get(&cd, dev_) != CUDA_SUCCESS ||
        v.dev_attr(&vmm_ok, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cd) != CUDA_SUCCESS || !vmm_ok)
        return fixed("this device has no virtual memory management");
    const CUmemAllocationProp p = prop_of(dev_);
    size_t gran = 0;
    if (v.gran(&gran, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || gran == 0)
        return fixed("no allocation granularity");
    auto* E = new ElasticKv();
    E->dev = dev_;
    E->gran = gran;
    E->ctx = ctx;
    E->min_tok = whole ? ctx : std::min<int64_t>(ctx, kmin);
    size_t total = 0;
    for (size_t r : rows) {
        ElasticKv::Region R;
        R.off = total;
        R.row = r;
        R.cap = up((size_t) ctx * r, gran);
        total += R.cap;
        E->kv.push_back(R);
    }
    CUdeviceptr va = 0;
    if (v.reserve(&va, total, gran, 0, 0) != CUDA_SUCCESS) {
        delete E;
        return fixed("the address space did not reserve");
    }
    E->kv_va = (unsigned long long) va;
    E->kv_va_size = total;
    E->kv_cap = total;
    ekv_ = E;
    std::string err;
    if (!ekv_kv_map(E->min_tok, err)) {
        ekv_destroy();
        return fixed("the first --kv-min positions did not map");
    }
    for (const auto& R : E->kv) ptrs.push_back((void*) (E->kv_va + R.off));
    const double G = 1073741824.0;
    std::fprintf(stderr, "glm elastic: CUDA%d KV on demand: %zu caches, %.2f GB reserved for %lld tokens, %.2f GB mapped "
                         "for the first %lld (--kv-min %lld); %.2f GB free for the weights until the KV needs it\n",
                 dev_, E->kv.size(), (double) E->kv_cap / G, (long long) ctx, (double) E->kv_mapped / G,
                 (long long) E->tok, (long long) kmin, (double) ekv_lendable() / G);
    return true;
#endif
}

// maps every cache up to `tokens` positions (the new memory zeroed, as the fixed layout's is at load)
bool Glm5Model::ekv_kv_map(int64_t tokens, std::string& err) {
#if defined(STRATA_USE_HIP)
    (void) tokens;
    err = "elastic KV on HIP";
    return false;
#else
    ElasticKv* E = ekv_;
    tokens = std::min<int64_t>(tokens, E->ctx);
    for (auto& R : E->kv) {
        const size_t want = std::min(R.cap, up((size_t) tokens * R.row, E->gran));
        if (want <= R.mapped) continue;
        const size_t at = R.mapped, n = want - R.mapped;
        if (!E->map_piece(E->kv_va + R.off, at, n, R.ch)) {
            err = "the KV cache's next " + std::to_string(n >> 20) + " MB did not map";
            return false;
        }
        R.mapped = want;
        E->kv_mapped += n;
        if (cudaMemset((void*) (E->kv_va + R.off + at), 0, n) != cudaSuccess) {
            cudaGetLastError();
            err = "the KV cache's new rows did not clear";
            return false;
        }
    }
    E->tok = E->tokens_held();
    return true;
#endif
}

void Glm5Model::ekv_destroy() {
    ElasticKv* E = ekv_;
    if (E == nullptr) return;
#if !defined(STRATA_USE_HIP)
    cudaSetDevice(dev_);
    cudaDeviceSynchronize();
    const Vmm& v = vmm();
    for (auto& R : E->kv) E->unmap_from(E->kv_va + R.off, R.ch, 0);
    for (auto& L : E->pl) E->unmap_from(E->p_va + L.off, L.ch, 0);
    if (E->kv_va != 0) v.addr_free((CUdeviceptr) E->kv_va, E->kv_va_size);
    if (E->p_va != 0) v.addr_free((CUdeviceptr) E->p_va, E->p_va_size);
#endif
    delete E;
    ekv_ = nullptr;
}

// the expert pool's main slots on the API: per MoE layer `core[il]` slots (the fixed layout's), then the elastic ranks
// up to cap_slots a layer as far as the lendable bytes go.  core[il] < 0: not an MoE layer.  false: the caller
// allocates the pool as before and the KV maps whole (ekv_kv_all)
bool Glm5Model::ekv_pool_layout(std::vector<int>& slots, const std::vector<size_t>& stride, size_t guard, bool zero,
                                int cap_slots, std::vector<uint8_t*>& base, std::string& err) {
#if defined(STRATA_USE_HIP)
    (void) slots; (void) stride; (void) guard; (void) zero; (void) cap_slots; (void) base; (void) err;
    return false;
#else
    ElasticKv* E = ekv_;
    if (E == nullptr) return false;
    const size_t lend = ekv_lendable();
    E->pl.clear();
    for (size_t il = 0; il < slots.size(); ++il) {
        if (slots[il] < 0) continue;
        ElasticKv::PLayer L;
        L.il = (int) il;
        L.core = slots[il];
        L.e = std::max(0, cap_slots - L.core);
        L.stride = stride[il];
        L.guard = guard;
        E->pl.push_back(L);
    }
    if (E->pl.empty()) return false;
    // the most ranks the lendable bytes map (each layer up to its cap)
    int lo = 0, hi = 0;
    for (const auto& L : E->pl) hi = std::max(hi, L.e);
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (E->pool_cost(mid) <= lend) lo = mid;
        else hi = mid - 1;
    }
    if (lo == 0) {
        E->pl.clear();
        return false;
    }
    for (auto& L : E->pl) L.e = std::min(L.e, lo);
    size_t total = 0;
    for (auto& L : E->pl) {
        L.off = total;
        total += L.extent(E->gran, L.e);
    }
    const Vmm& v = vmm();
    CUdeviceptr va = 0;
    if (v.reserve(&va, total, E->gran, 0, 0) != CUDA_SUCCESS) {
        E->pl.clear();
        return false;
    }
    E->p_va = (unsigned long long) va;
    E->p_va_size = total;
    bool ok = true;
    for (auto& L : E->pl) {
        const size_t c = L.extent(E->gran, 0);
        if (ok && c > 0) ok = E->map_piece(E->p_va + L.off, 0, c, L.ch);
        L.mapped = c;
        for (size_t at = c; ok && at < L.extent(E->gran, L.e); at += E->gran) {
            ok = E->map_piece(E->p_va + L.off, at, E->gran, L.ch);
            L.mapped = at + E->gran;
        }
        if (ok && zero) ok = cudaMemset((void*) (E->p_va + L.off), 0, L.mapped) == cudaSuccess;
    }
    if (!ok) {
        cudaGetLastError();
        for (auto& L : E->pl) E->unmap_from(E->p_va + L.off, L.ch, 0);
        v.addr_free((CUdeviceptr) E->p_va, E->p_va_size);
        E->p_va = 0;
        E->pl.clear();
        err.clear();
        std::fprintf(stderr, "glm elastic: CUDA%d the expert pool's pieces did not map - the pool is allocated as before\n",
                     dev_);
        return false;
    }
    E->E = E->R = lo;
    E->ep_mapped = E->pool_cost(lo);
    E->pool_on = true;
    E->zero_pool = zero;
    base.assign(slots.size(), nullptr);
    for (const auto& L : E->pl) {
        base[(size_t) L.il] = (uint8_t*) (E->p_va + L.off);
        slots[(size_t) L.il] = L.core + L.e;
    }
    return true;
#endif
}

// the pool could not take the lendable bytes (no MoE pool, its pieces did not map, nothing to give): the KV maps
// whole now and stays so - the fixed layout's VRAM, on the API
bool Glm5Model::ekv_kv_all(std::string& err) {
    ElasticKv* E = ekv_;
    if (E == nullptr || E->pool_on || E->lent() > 0) return true;
    if (!ekv_kv_map(E->ctx, err)) return false;
    std::fprintf(stderr, "glm elastic: CUDA%d nothing borrows the KV's room: the cache is mapped whole (%.2f GB)\n", dev_,
                 (double) E->kv_mapped / 1073741824.0);
    return true;
}

// live elastic ranks -> r: down, the ranks' experts leave (lend_slots) and their pieces unmap; up, the pieces map again
// and the slots are free (misses refill them)
bool Glm5Model::ekv_pool_ranks(int r, std::string& err) {
#if defined(STRATA_USE_HIP)
    (void) r; (void) err;
    return true;
#else
    ElasticKv* E = ekv_;
    FastState* F = fast_;
    if (E == nullptr || !E->pool_on || F == nullptr) return true;
    r = std::max(0, std::min(r, E->E));
    if (r == E->R) return true;
    const int r0 = E->R;
    if (r < r0) {
        std::vector<int> core((size_t) F->lp.size(), -1), ecap((size_t) F->lp.size(), 0);
        for (const auto& L : E->pl) {
            core[(size_t) L.il] = L.core;
            ecap[(size_t) L.il] = L.e;
        }
        uint64_t moved = 0, dropped = 0;
        const auto take = [&](int il, int s) {
            const int c = core[(size_t) il];
            return c >= 0 && s >= c + std::min(r, ecap[(size_t) il]) && s < c + std::min(r0, ecap[(size_t) il]);
        };
        if (!lend_slots(take, FastState::kElastic, moved, dropped, err)) return false;
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaGetLastError();
            err = "glm elastic: the device did not settle before the pool's pieces unmapped";
            return false;
        }
        int64_t gone = 0;
        for (auto& L : E->pl) {
            const size_t keep = L.extent(E->gran, r);
            E->unmap_from(E->p_va + L.off, L.ch, keep);
            L.mapped = std::min(L.mapped, keep);
            gone += std::min(r0, L.e) - std::min(r, L.e);
        }
        F->pool_slots -= gone;
        E->R = r;
        E->ep_mapped = E->pool_cost(r);
        std::fprintf(stderr, "glm elastic: CUDA%d expert pool -%lld slots (%d -> %d elastic a layer: %llu experts moved "
                             "over colder ones, %llu left VRAM)\n", dev_, (long long) gone, r0, r,
                     (unsigned long long) moved, (unsigned long long) (dropped - std::min(dropped, moved)));
        return true;
    }
    bool ok = true;
    int64_t back = 0;
    for (auto& L : E->pl) {
        const size_t want = L.extent(E->gran, r);
        for (size_t at = L.mapped; ok && at < want; at += E->gran) {
            ok = E->map_piece(E->p_va + L.off, at, E->gran, L.ch);
            if (ok && E->zero_pool) ok = cudaMemset((void*) (E->p_va + L.off + at), 0, E->gran) == cudaSuccess;
            if (ok) L.mapped = at + E->gran;
        }
        if (!ok) break;
    }
    // the ranks every layer mapped whole come back (a partial map leaves them out: they are mapped again next time)
    int got = r0;
    for (int rr = r0 + 1; rr <= r; ++rr) {
        bool all = true;
        for (const auto& L : E->pl) all = all && L.mapped >= L.extent(E->gran, rr);
        if (!all) break;
        got = rr;
    }
    if (!ok) cudaGetLastError();
    {
        std::lock_guard<std::mutex> lk(F->mu);
        for (const auto& L : E->pl) {
            auto& P = F->lp[(size_t) L.il];
            for (int s = L.core + std::min(r0, L.e); s < L.core + std::min(got, L.e); ++s)
                if (P.st[(size_t) s] == FastState::kElastic) {
                    P.st[(size_t) s] = FastState::kFree;
                    P.key[(size_t) s] = -1;
                    ++back;
                }
        }
    }
    F->pool_slots += back;
    E->R = got;
    E->ep_mapped = E->pool_cost(got);
    std::fprintf(stderr, "glm elastic: CUDA%d expert pool +%lld slots (%d -> %d elastic a layer)\n", dev_,
                 (long long) back, r0, got);
    return true;
#endif
}

int Glm5Model::ekv_pool_core(int il) const {
    if (ekv_ == nullptr || !ekv_->pool_on) return -1;
    for (const auto& L : ekv_->pl)
        if (L.il == il) return L.core;
    return -1;
}

void Glm5Model::ekv_reset_mark() {
    if (ekv_ != nullptr) ekv_->shrink_pending = true;
}

// before work that writes positions below `end`: the caches hold them (the weights give back what the KV needs)
bool Glm5Model::kv_ensure(int64_t end, std::string& err) {
    ElasticKv* E = ekv_;
    if (E == nullptr) return true;
#if defined(STRATA_USE_HIP)
    (void) end; (void) err;
    return true;
#else
    const double G = 1073741824.0;
    const int64_t want = std::min<int64_t>(E->ctx, std::max<int64_t>(E->min_tok, (end + kMargin + kStep - 1) / kStep * kStep));
    if (E->shrink_pending) {   // a new conversation: the KV past what this one needs goes, the weights come back
        E->shrink_pending = false;
        if (want + kStep <= E->tok) {
            cudaSetDevice(dev_);
            if (cudaDeviceSynchronize() != cudaSuccess) {
                cudaGetLastError();
                err = "glm elastic: the device did not settle before the KV shrank";
                return false;
            }
            const int64_t t0 = E->tok;
            size_t freed = 0;
            for (auto& R : E->kv) {
                const size_t keep = std::min(R.cap, up((size_t) want * R.row, E->gran));
                freed += E->unmap_from(E->kv_va + R.off, R.ch, keep);
                R.mapped = R.ch.empty() ? 0 : R.ch.back().off + R.ch.back().size;
            }
            E->kv_mapped -= freed;
            E->tok = E->tokens_held();
            ++E->shrinks;
            if (E->pool_on) {
                int r = E->R;
                while (r < E->E && E->kv_mapped + E->pool_cost(r + 1) <= E->kv_cap) ++r;
                if (r > E->R && !ekv_pool_ranks(r, err)) return false;
            } else if (!q35d_elastic_grow(err)) {
                return false;
            }
            std::fprintf(stderr, "glm elastic: CUDA%d KV %lld -> %lld tokens (%.2f of %.2f GB mapped; the weights hold "
                                 "%.2f GB of the rest - a new conversation, %.2f GB back to them)\n", dev_, (long long) t0,
                         (long long) E->tok, (double) E->kv_mapped / G, (double) E->kv_cap / G, (double) E->lent() / G,
                         (double) freed / G);
        }
    }
    if (end + kSlack <= E->tok || E->tok >= E->ctx) return true;
    cudaSetDevice(dev_);
    size_t need = 0;
    for (const auto& R : E->kv) {
        const size_t w = std::min(R.cap, up((size_t) want * R.row, E->gran));
        if (w > R.mapped) need += w - R.mapped;
    }
    // the weights give back what the KV needs first
    const auto t_ev = std::chrono::steady_clock::now();
    if (E->kv_mapped + need + E->lent() > E->kv_cap) {
        if (E->pool_on) {
            int r = E->R;
            while (r > 0 && E->kv_mapped + need + E->pool_cost(r) > E->kv_cap) --r;
            if (!ekv_pool_ranks(r, err)) return false;
        } else if (!q35d_elastic_shrink(E->kv_cap - std::min(E->kv_cap, E->kv_mapped + need), err)) {
            return false;
        }
    }
    const auto t_map = std::chrono::steady_clock::now();
    const int64_t t0 = E->tok;
    std::string merr;
    if (!ekv_kv_map(want, merr)) {
        // the VRAM the budget counted is held by something else (a buffer allocated later): more of the weights go
        cudaGetLastError();
        bool more = false;
        if (E->pool_on && E->R > 0) more = ekv_pool_ranks(0, err);
        else if (!E->pool_on && E->lent() > 0) more = q35d_elastic_shrink(0, err);
        if (!more || !ekv_kv_map(want, merr)) {
            err = "glm elastic: " + merr + " (out of VRAM: a smaller --max-context or --kv int8)";
            return false;
        }
    }
    ++E->grows;
    const auto t_end = std::chrono::steady_clock::now();
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::fprintf(stderr, "glm elastic: CUDA%d KV %lld -> %lld tokens (%.2f of %.2f GB mapped; the weights hold %.2f GB "
                         "of the rest; %.0f ms giving back, %.0f ms mapping)\n", dev_, (long long) t0, (long long) E->tok,
                 (double) E->kv_mapped / G, (double) E->kv_cap / G, (double) E->lent() / G, ms(t_ev, t_map),
                 ms(t_map, t_end));
    return true;
#endif
}

// the dense path's elastic layers count as lent bytes (q35d_elastic_*): their pieces are this reservation's too
void Glm5Model::ekv_lent_set(size_t bytes) {
    if (ekv_ != nullptr) ekv_->ep_mapped = bytes;
}

std::string Glm5Model::ekv_status() const {
    const ElasticKv* E = ekv_;
    if (E == nullptr) return "";
    char b[256];
    std::snprintf(b, sizeof b, "kv_tokens=%lld kv_gb=%.2f kv_cap_gb=%.2f lent_gb=%.2f grows=%llu shrinks=%llu",
                  (long long) E->tok, (double) E->kv_mapped / 1073741824.0, (double) E->kv_cap / 1073741824.0,
                  (double) E->lent() / 1073741824.0, (unsigned long long) E->grows, (unsigned long long) E->shrinks);
    return b;
}

}  // namespace strata::core
