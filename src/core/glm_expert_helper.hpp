#pragma once
// Private Decode-only expert tier. Host expert blobs stay in the existing RAM tier for CPU fallback.
#include "strata/kernels/glm_fast.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace strata::core {
class GlmExpertHelper {
    using Clock = std::chrono::steady_clock;
    static double ms(Clock::time_point t) {
        return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
    }
    // CUDA's current device belongs to the calling thread, including the CPU lane's service thread.
    struct DeviceScope {
        int previous = -1;
        cudaError_t status;
        explicit DeviceScope(int target) : status(cudaGetDevice(&previous)) {
            if (status == cudaSuccess) status = cudaSetDevice(target);
        }
        ~DeviceScope() {
            if (previous >= 0) {
                const cudaError_t restored = cudaSetDevice(previous);
                if (restored != cudaSuccess)
                    std::fprintf(stderr, "expert helper: restoring GPU%d failed: %s\n", previous, cudaGetErrorString(restored));
            }
        }
    };
    bool check(cudaError_t status, const char* operation, std::string& err) {
        if (status == cudaSuccess) return true;
        err = std::string("expert helper: ") + operation + ": " + cudaGetErrorString(status);
        failed = true;
        return false;
    }
public:
    int device, primary = -1, emb = 0, ff = 0, experts = 0;
    size_t budget, used = 0;
    bool opened = false;
    std::atomic<bool> active{false}, failed{false};
    cudaStream_t stream = nullptr;
    uint8_t *pool = nullptr, *owned_d = nullptr;
    float *x_d = nullptr, *out_d = nullptr, *out_h = nullptr;
    void *xq = nullptr, *hq = nullptr;
    unsigned long long* ptr_h = nullptr;
    float* w_h = nullptr;
    strata::kernels::glmf::MoeDev md;
    struct Slot { int key; uint8_t* ptr; };
    std::vector<std::vector<Slot>> slots;
    std::vector<uint8_t*> resident;
    std::vector<uint8_t> owned;
    uint64_t calls = 0, computed = 0, uploads = 0, upload_bytes = 0, fallbacks = 0;
    double launch_ms = 0, wait_ms = 0, prepare_ms = 0;
    // A verify window's columns (DSpark / NextN): up to kCols requests of one layer in one call, each column with
    // its own input, plan and output; begin() is column 0.
    static constexpr int kCols = 8, kPer = strata::kernels::glmf::kMaxK;   // (Ornith-397B and Qwen3.8 route 10)
    uint64_t col_calls = 0, cols = 0;

    GlmExpertHelper(int d, size_t cap) : device(d), budget(cap) {}
    GlmExpertHelper(const GlmExpertHelper&) = delete;
    GlmExpertHelper& operator=(const GlmExpertHelper&) = delete;

    bool open(int p, int E, int FF, int L, int NE, std::string& err) {
        primary = p; emb = E; ff = FF; experts = NE;
        int count = 0;
        if (!check(cudaGetDeviceCount(&count), "device count", err)) return false;
        // STRATA_GLM_EXPERT_HELPER_SAME_GPU=1 (debug): the helper on the primary GPU itself - no speed-up, it
        // checks the helper's answers on a one-GPU box
        const char* same = getenv("STRATA_GLM_EXPERT_HELPER_SAME_GPU");
        const bool same_ok = same && std::strcmp(same, "0") != 0;
        if (device < 0 || device >= count || (device == primary && !same_ok)) {
            err = "expert helper: a separate visible GPU is required";
            return false;
        }
        slots.resize((size_t) L);
        resident.assign((size_t) L * NE, nullptr);
        owned.assign((size_t) L * NE, 0);
        DeviceScope selected(device);
        if (!check(selected.status, "select device", err)) return false;
        size_t free = 0, total = 0;
        if (!check(cudaMemGetInfo(&free, &total), "memory", err)) return false;
        constexpr size_t reserve = (size_t) 512 << 20;
        if (budget > free || reserve > free - budget) {
            err = "expert helper: requested pool exceeds free VRAM after 512 MiB reserve";
            return false;
        }
        const auto alloc = [&](auto** ptr, size_t bytes) {
            return check(cudaMalloc((void**) ptr, bytes), "allocate", err);
        };
        if (!check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream", err) ||
            !alloc(&pool, budget) || !alloc(&x_d, (size_t) kCols * E * sizeof(float)) ||
            !alloc(&out_d, (size_t) kCols * E * sizeof(float)) || !alloc(&xq, (size_t) E / 32 * 36) ||
            !alloc(&hq, (size_t) kPer * FF / 32 * 36) ||
            !alloc(&md.plan_ptr, (size_t) kCols * kPer * sizeof(unsigned long long)) ||
            !alloc(&md.plan_w, (size_t) kCols * kPer * sizeof(float)) || !alloc(&md.cpu_flag, sizeof(int)) ||
            !alloc(&md.cpu_part, (size_t) E * sizeof(float)) ||
            !check(cudaHostAlloc((void**) &out_h, (size_t) kCols * E * sizeof(float), cudaHostAllocPortable), "output staging", err) ||
            !check(cudaHostAlloc((void**) &ptr_h, (size_t) kCols * kPer * sizeof(unsigned long long), cudaHostAllocPortable), "pointer staging", err) ||
            !check(cudaHostAlloc((void**) &w_h, (size_t) kCols * kPer * sizeof(float), cudaHostAllocPortable), "weight staging", err) ||
            !check(cudaMemsetAsync(md.cpu_flag, 0, sizeof(int), stream), "clear flag", err) ||
            !check(cudaStreamSynchronize(stream), "init sync", err)) return false;
        DeviceScope main_device(primary);
        if (!check(main_device.status, "select primary", err) || !alloc(&owned_d, owned.size()) ||
            !check(cudaMemset(owned_d, 0, owned.size()), "clear ownership", err)) return false;
        opened = true;
        return true;
    }

    // Equal byte budgets per layer; keep the selected resident experts between requests.
    bool prepare(const std::vector<std::vector<int>>& ranked, const std::vector<size_t>& blobs,
                 const std::function<const uint8_t*(int)>& source, std::string& err) {
        const auto t = Clock::now();
        active = false;
        DeviceScope selected(device);
        if (!check(selected.status, "select device", err) ||
            !check(cudaStreamSynchronize(stream), "prepare settle", err)) return false;
        int layers = 0;
        for (size_t bytes : blobs) layers += bytes != 0;
        for (size_t il = 0; il < slots.size(); ++il) {
            const size_t bytes = blobs[il];
            if (bytes == 0) continue;
            if (slots[il].empty()) {
                const size_t n = std::min((size_t) experts, budget / std::max(1, layers) / bytes);
                for (size_t j = 0; j < n; ++j) {
                    if (bytes > budget - used) { err = "expert helper: pool accounting"; return false; }
                    slots[il].push_back({-1, pool + used});
                    used += bytes;
                }
            }
            const auto& rank = ranked[il];
            const size_t n = std::min(rank.size(), slots[il].size());
            for (auto& slot : slots[il]) {
                if (slot.key >= 0 && std::find(rank.begin(), rank.begin() + n, slot.key) == rank.begin() + n) {
                    resident[(size_t) slot.key] = nullptr;
                    owned[(size_t) slot.key] = 0;
                    slot.key = -1;
                }
            }
            size_t cursor = 0;
            for (size_t j = 0; j < n; ++j) {
                const int key = rank[j];
                if (resident[(size_t) key]) continue;
                while (cursor < slots[il].size() && slots[il][cursor].key >= 0) ++cursor;
                if (cursor == slots[il].size()) { err = "expert helper: slot accounting"; return false; }
                const uint8_t* src = source(key);
                if (!src) continue;
                auto& slot = slots[il][cursor++];
                if (!check(cudaMemcpyAsync(slot.ptr, src, bytes, cudaMemcpyHostToDevice, stream), "expert upload", err))
                    return false;
                slot.key = key;
                resident[(size_t) key] = slot.ptr;
                owned[(size_t) key] = 1;
                ++uploads; upload_bytes += bytes;
            }
        }
        if (!check(cudaStreamSynchronize(stream), "uploads", err)) return false;
        DeviceScope main_device(primary);
        if (!check(main_device.status, "select primary", err) ||
            !check(cudaMemcpy(owned_d, owned.data(), owned.size(), cudaMemcpyHostToDevice), "ownership", err)) return false;
        prepare_ms += ms(t);
        active = !failed;
        return active;
    }
    bool owns(int key) const {
        return active && !failed && key >= 0 && (size_t) key < resident.size() && resident[(size_t) key];
    }
    bool begin(int il, int ne, const int* ids, const float* weights, const float* x,
               int gu, int down, size_t down_off, float limit, std::string& err) {
        const auto t = Clock::now();
        DeviceScope selected(device);
        if (!check(selected.status, "select device", err)) return false;
        for (int i = 0; i < ne; ++i) {
            ptr_h[i] = (unsigned long long) resident[(size_t) il * experts + ids[i]];
            w_h[i] = weights[i];
        }
        bool ok = check(cudaMemcpyAsync(x_d, x, (size_t) emb * sizeof(float), cudaMemcpyHostToDevice, stream), "input", err) &&
            check(cudaMemcpyAsync(md.plan_ptr, ptr_h, (size_t) ne * sizeof(unsigned long long), cudaMemcpyHostToDevice, stream), "plan", err) &&
            check(cudaMemcpyAsync(md.plan_w, w_h, (size_t) ne * sizeof(float), cudaMemcpyHostToDevice, stream), "weights", err);
        if (ok) {
            namespace gf = strata::kernels::glmf;
            gf::helper_launch_begin();
            gf::quantize_q8_1(x_d, xq, emb, stream);
            gf::moe_gate_up(gu, md, ne, emb, ff, limit, xq, hq, nullptr, 0, nullptr, 0, nullptr, stream);
            gf::moe_down(down, md, ne, emb, ff, down_off, hq, nullptr, out_d, stream);
            ok = check(gf::helper_launch_end(), "kernels", err) &&
                check(cudaGetLastError(), "launch status", err) &&
                check(cudaMemcpyAsync(out_h, out_d, (size_t) emb * sizeof(float), cudaMemcpyDeviceToHost, stream), "result", err);
        }
        if (!ok) cudaStreamSynchronize(stream); // Retire reads of the request before CPU fallback publishes its answer.
        if (ok) { ++calls; computed += ne; }
        launch_ms += ms(t);
        return ok;
    }
    // A window's columns: column t has ne[t] (<= kPer) helper experts ids[kMaxK * t + i] with weights w[kMaxK * t + i]
    // and input x[t]; a column with ne[t] == 0 is skipped.  The kernels queue on the helper's stream in column order
    // (xq / hq are reused: the stream runs them one after another); finish_cols adds each column's answer.
    bool begin_cols(int il, int n, const int* ne, const int* ids, const float* w, const float* const* x,
                    int gu, int down, size_t down_off, float limit, std::string& err) {
        namespace gf = strata::kernels::glmf;
        const auto t = Clock::now();
        DeviceScope selected(device);
        if (!check(selected.status, "select device", err)) return false;
        if (n < 1 || n > kCols) { err = "expert helper: window too wide"; failed = true; return false; }
        bool ok = true;
        int launched = 0;
        uint64_t nexp = 0;
        for (int c = 0; c < n; ++c) {
            if (ne[c] <= 0) continue;
            if (ne[c] > kPer) { err = "expert helper: too many experts in a column"; failed = true; return false; }
            for (int i = 0; i < ne[c]; ++i) {
                ptr_h[kPer * c + i] = (unsigned long long) resident[(size_t) il * experts + ids[gf::kMaxK * c + i]];
                w_h[kPer * c + i] = w[gf::kMaxK * c + i];
            }
            ok = ok && check(cudaMemcpyAsync(x_d + (size_t) c * emb, x[c], (size_t) emb * sizeof(float),
                                             cudaMemcpyHostToDevice, stream), "input", err);
        }
        ok = ok && check(cudaMemcpyAsync(md.plan_ptr, ptr_h, (size_t) n * kPer * sizeof(unsigned long long),
                                         cudaMemcpyHostToDevice, stream), "plan", err) &&
             check(cudaMemcpyAsync(md.plan_w, w_h, (size_t) n * kPer * sizeof(float), cudaMemcpyHostToDevice, stream),
                   "weights", err);
        if (ok) {
            gf::helper_launch_begin();
            for (int c = 0; c < n; ++c) {
                if (ne[c] <= 0) continue;
                gf::MoeDev mc = md;
                mc.plan_ptr = md.plan_ptr + (size_t) kPer * c;
                mc.plan_w = md.plan_w + (size_t) kPer * c;
                gf::quantize_q8_1(x_d + (size_t) c * emb, xq, emb, stream);
                gf::moe_gate_up(gu, mc, ne[c], emb, ff, limit, xq, hq, nullptr, 0, nullptr, 0, nullptr, stream);
                gf::moe_down(down, mc, ne[c], emb, ff, down_off, hq, nullptr, out_d + (size_t) c * emb, stream);
                ++launched;
                nexp += (uint64_t) ne[c];
            }
            ok = check(gf::helper_launch_end(), "kernels", err) &&
                check(cudaGetLastError(), "launch status", err) &&
                check(cudaMemcpyAsync(out_h, out_d, (size_t) n * emb * sizeof(float), cudaMemcpyDeviceToHost, stream),
                      "result", err);
        }
        if (!ok) cudaStreamSynchronize(stream);
        if (ok) { ++col_calls; cols += (uint64_t) launched; calls += (uint64_t) launched; computed += nexp; }
        launch_ms += ms(t);
        return ok;
    }
    bool finish_cols(int n, const int* ne, float* const* out, std::string& err) {
        const auto t = Clock::now();
        DeviceScope selected(device);
        const bool ok = check(selected.status, "select device", err) &&
            check(cudaStreamSynchronize(stream), "completion", err);
        if (ok)
            for (int c = 0; c < n; ++c) {
                if (ne[c] <= 0) continue;
                const float* h = out_h + (size_t) c * emb;
                for (int i = 0; i < emb; ++i) out[c][i] += h[i];
            }
        wait_ms += ms(t);
        return ok;
    }
    bool finish(float* sum, std::string& err) {
        const auto t = Clock::now();
        DeviceScope selected(device);
        const bool ok = check(selected.status, "select device", err) &&
            check(cudaStreamSynchronize(stream), "completion", err);
        if (ok) for (int i = 0; i < emb; ++i) sum[i] += out_h[i];
        wait_ms += ms(t);
        return ok;
    }
    void disable() {
        active = false;
        failed = true;
        DeviceScope selected(device);
        if (selected.status == cudaSuccess && stream) cudaStreamSynchronize(stream);
        cudaGetLastError(); // Do not leave an optional helper's launch/allocation error on its caller's thread.
    }
    void report(const char* phase) const {
        std::fprintf(stderr, "expert helper %s: GPU%d pool %.3f GiB, resident %zu, calls %llu experts %llu "
                             "upload_bytes %llu prepare_ms %.3f launch_ms %.3f wait_ms %.3f fallbacks %llu window_calls %llu columns %llu\n",
                     phase, device, (double) used / 1073741824.0,
                     (size_t) std::count(owned.begin(), owned.end(), 1), (unsigned long long) calls,
                     (unsigned long long) computed, (unsigned long long) upload_bytes,
                     prepare_ms, launch_ms, wait_ms, (unsigned long long) fallbacks,
                     (unsigned long long) col_calls, (unsigned long long) cols);
    }
    ~GlmExpertHelper() {
        DeviceScope selected(device);
        if (selected.status == cudaSuccess) {
            if (stream) cudaStreamSynchronize(stream);
            for (void* ptr : {(void*) pool, (void*) x_d, (void*) out_d, xq, hq,
                             (void*) md.plan_ptr, (void*) md.plan_w, (void*) md.cpu_flag, (void*) md.cpu_part})
                if (ptr) cudaFree(ptr);
            if (stream) cudaStreamDestroy(stream);
        }
        if (out_h) cudaFreeHost(out_h);
        if (ptr_h) cudaFreeHost(ptr_h);
        if (w_h) cudaFreeHost(w_h);
        if (owned_d) {
            DeviceScope main_device(primary);
            if (main_device.status == cudaSuccess) cudaFree(owned_d);
        }
        cudaGetLastError();
    }
};
}
