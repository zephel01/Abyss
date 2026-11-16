// src/kernels/cuda/token_ban.cu - see include/strata/kernels/token_ban.hpp.
#include "strata/kernels/token_ban.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

namespace strata::kernels {
namespace {

std::atomic<const BiasSet*> g_active{nullptr};
std::atomic<uint64_t> g_gen{0};
std::mutex g_dev_mu;

__global__ void bias_kernel(float* __restrict__ x, int rows, long long ld, const int32_t* __restrict__ ids,
                            const float* __restrict__ add, int n) {
    const long long k = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (long long) rows * n) return;
    const int r = (int) (k / n), j = (int) (k - (long long) r * n);
    float* p = x + (long long) r * ld + ids[j];
    const float a = add[j];
    *p = (a == -INFINITY) ? -INFINITY : *p + a;
}

}  // namespace

BiasSet::~BiasSet() {
    for (const Dev& d : dev) {
        if (d.ids) cudaFree(d.ids);
        if (d.add) cudaFree(d.add);
    }
}

std::shared_ptr<BiasSet> bias_parse(const std::string& spec, int n_vocab, std::string& err) {
    auto s = std::make_shared<BiasSet>();
    s->n_vocab = n_vocab;
    std::map<int32_t, float> m;
    size_t i = 0;
    while (i < spec.size()) {
        size_t j = spec.find(',', i);
        if (j == std::string::npos) j = spec.size();
        const std::string item = spec.substr(i, j - i);
        i = j + 1;
        if (item.empty()) continue;
        const size_t c = item.find(':');
        char* end = nullptr;
        const long long id = std::strtoll(item.c_str(), &end, 10);
        if (c == std::string::npos || end != item.c_str() + c || id < 0 || id >= n_vocab) {
            err = "bad bias entry '" + item + "' (expected <id>:<value>, id in [0, " + std::to_string(n_vocab) + "))";
            return nullptr;
        }
        const std::string vs = item.substr(c + 1);
        float v = 0.0f;
        if (vs == "-inf" || vs == "ban") {
            v = -INFINITY;
        } else {
            char* e2 = nullptr;
            v = std::strtof(vs.c_str(), &e2);
            if (e2 == vs.c_str() || *e2 != '\0' || !std::isfinite(v)) {
                err = "bad bias value '" + vs + "'";
                return nullptr;
            }
            if (v <= -100.0f) v = -INFINITY;   // OpenAI: -100 bans the token
        }
        m[(int32_t) id] = v;
    }
    s->banned.assign((size_t) n_vocab, 0);
    for (const auto& [id, v] : m) {
        if (v == 0.0f) continue;
        s->ids.push_back(id);
        s->add.push_back(v);
        if (v == -INFINITY) s->banned[(size_t) id] = 1;
    }
    return s;
}

void bias_activate(const BiasSet* set) {
    if (set != nullptr && set->ids.empty()) set = nullptr;
    if (g_active.exchange(set) != set) g_gen.fetch_add(1);
}

const BiasSet* bias_active() { return g_active.load(); }
uint64_t bias_generation() { return g_gen.load(); }

bool bias_banned(int id) {
    const BiasSet* s = g_active.load();
    return s != nullptr && id >= 0 && id < s->n_vocab && s->banned[(size_t) id] != 0;
}

void bias_apply(float* logits, int rows, int n_vocab, long long ld, void* stream) {
    const BiasSet* cs = g_active.load();
    if (cs == nullptr || rows <= 0 || n_vocab != cs->n_vocab) return;
    auto* s = const_cast<BiasSet*>(cs);
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0) return;
    const cudaStream_t st = (cudaStream_t) stream;
    BiasSet::Dev d;
    {
        std::lock_guard<std::mutex> lk(g_dev_mu);
        if ((int) s->dev.size() <= dev) s->dev.resize((size_t) dev + 1);
        d = s->dev[(size_t) dev];
        if (d.ids == nullptr) {
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            if (st != nullptr && cudaStreamIsCapturing(st, &cap) == cudaSuccess && cap != cudaStreamCaptureStatusNone)
                return;   // (no copy inside a capture: the picks outside the graph still apply the set)
            const size_t n = s->ids.size();
            if (cudaMalloc(&d.ids, n * sizeof(int32_t)) != cudaSuccess ||
                cudaMalloc(&d.add, n * sizeof(float)) != cudaSuccess ||
                cudaMemcpy(d.ids, s->ids.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(d.add, s->add.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
                std::fprintf(stderr, "token ban: device copy failed on device %d: %s\n", dev,
                             cudaGetErrorString(cudaGetLastError()));
                std::exit(1);   // a ban that silently does not apply is worse than an engine error
            }
            s->dev[(size_t) dev] = d;
        }
    }
    const int n = (int) s->ids.size();
    const long long total = (long long) rows * n;
    bias_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, st>>>(logits, rows, ld, d.ids, d.add, n);
}

}  // namespace strata::kernels
