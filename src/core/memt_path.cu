// src/core/memt_path.cu - MemT-1.0-Nano's native memory on the qwen35moe fast path (Glm5Model members).
//
// Every full-attention layer has a memory block (worklog 2026-11-25_memt-nano.md; include/strata/kernels/memt.hpp):
// from the layer's normed input x, q = memt_in[q] x (L2-normed a head) reads the layer's state S (heads x 256 x 256,
// FP32), the read is RMS-normed, scaled by memt_o_norm and gated by softcap-SiLU(g_down . memt_in[g] x), and added to
// the attention output after its sigmoid gate (before attn_output).  A turn's end replays the turn's tokens from
// position 0 on an empty context: the reads come from the state as it was, and k / v / b / w / f from the same x
// write a second state token by token (a row-decayed delta rule); that state becomes the memory.
//
// The projections run in FP32 on the BF16 weights (the GEMV's BF16 rows with the float input; the prompt path's
// FP32 GEMM on rows widened to FP32).  With the memory off (memt_use(false), the default) nothing runs: the backbone
// alone, the same bits as without these weights.
#include <cstring>
#include <vector>
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/memt.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <string>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace mk = strata::kernels::memt;

struct Memt {
    int n = 0;                    // memory layers
    size_t per = 0;               // floats a layer's state
    float* S = nullptr;           // the memory (read) [n][per]
    float* Sw = nullptr;          // the replay's write state [n][per]
    bool on = false;              // reads run
    bool commit = false;          // a replay: writes run too
    // the token path's scratch: q, g_in, g (read); k v b w f(256) rows, f_down's output (write)
    float *q = nullptr, *gin = nullptr, *g = nullptr, *kvbw = nullptr, *fl = nullptr, *part = nullptr;
    // the prompt path's rows (grown to the largest sub-batch): q | g | gin, and k | v | b | w | fl | f
    float* rows = nullptr;
    size_t rows_cap = 0;
};

namespace {
constexpr int kD = mk::kD;
bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("memt: ") + what + ": " + cudaGetErrorString(e);
    return false;
}
}  // namespace

void Glm5Model::memt_bind(int il, std::string& missing) {
    auto& Ly = fast_->L[(size_t) il];
    const std::string P = "blk." + std::to_string(il) + ".";
    const auto b16 = [&](const char* n) -> const uint16_t* {
        const auto it = w16_.find(P + n);
        if (it == w16_.end()) { missing += P + n + " (bf16) "; return nullptr; }
        return it->second;
    };
    const auto f32 = [&](const char* n) -> const float* {
        const auto it = w_.find(P + n);
        if (it == w_.end()) { missing += P + n + " (f32) "; return nullptr; }
        return it->second;
    };
    Ly.mt_in = b16("memt_in.weight");
    Ly.mt_fd = b16("memt_f_down.weight");
    Ly.mt_gd = b16("memt_g_down.weight");
    Ly.mt_bias = f32("memt_decay_bias");
    Ly.mt_dsl = f32("memt_decay_scale_log");
    Ly.mt_onorm = f32("memt_o_norm.weight");
    int n = 0;
    for (int j = 0; j < il; ++j) n += g_.n_head_kv[(size_t) j] != 0 ? 1 : 0;
    Ly.mt = n;
}

bool Glm5Model::memt_alloc(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    if (F->memt != nullptr) return true;
    Memt* M = new Memt();
    F->memt = M;
    for (int il = 0; il < g.n_layers; ++il) M->n += g.n_head_kv[(size_t) il] != 0 ? 1 : 0;
    const size_t W = (size_t) g.memt_heads * kD;
    M->per = W * kD;
    const size_t sb = (size_t) M->n * M->per * sizeof(float);
    if (!ck(cudaMalloc(&M->S, sb), "the state", err) || !ck(cudaMalloc(&M->Sw, sb), "the write state", err)) return false;
    cudaMemset(M->S, 0, sb);
    cudaMemset(M->Sw, 0, sb);
    const size_t scratch = (W + kD + W + 4 * W + kD + W + (size_t) g.memt_heads * mk::kSplit * kD) * sizeof(float);
    float* a = nullptr;
    if (!ck(cudaMalloc(&a, scratch), "the token scratch", err)) return false;
    M->q = a;
    M->gin = M->q + W;
    M->g = M->gin + kD;
    M->kvbw = M->g + W;   // k | v | b | w | f (4 W + 256: memt_in's rows W .. 5 W + 256)
    M->fl = M->kvbw + 4 * W + kD;
    M->part = M->fl + W;
    std::fprintf(stderr, "memt: %d memory layers, state %.1f MiB (x2 with the replay's)\n", M->n, sb / 1048576.0);
    return true;
}

void Glm5Model::memt_free() {
    if (fast_ == nullptr || fast_->memt == nullptr) return;
    Memt* M = fast_->memt;
    cudaFree(M->S);
    cudaFree(M->Sw);
    cudaFree(M->q);
    cudaFree(M->rows);
    delete M;
    fast_->memt = nullptr;
}

size_t Glm5Model::memt_state_floats() const {
    return fast_ != nullptr && fast_->memt != nullptr ? (size_t) fast_->memt->n * fast_->memt->per : 0;
}

bool Glm5Model::memt_get(float* host, std::string& err) {
    Memt* M = fast_ != nullptr ? fast_->memt : nullptr;
    if (M == nullptr) { err = "memt: not a MemT model"; return false; }
    cudaSetDevice(dev_);
    return ck(cudaStreamSynchronize(fast_->cs), "sync", err) &&
           ck(cudaMemcpy(host, M->S, memt_state_floats() * sizeof(float), cudaMemcpyDeviceToHost), "read back", err);
}

bool Glm5Model::memt_set(const float* host, std::string& err) {
    Memt* M = fast_ != nullptr ? fast_->memt : nullptr;
    if (M == nullptr) { err = "memt: not a MemT model"; return false; }
    cudaSetDevice(dev_);
    if (!ck(cudaStreamSynchronize(fast_->cs), "sync", err)) return false;
    if (host == nullptr) return ck(cudaMemset(M->S, 0, memt_state_floats() * sizeof(float)), "zero", err);
    return ck(cudaMemcpy(M->S, host, memt_state_floats() * sizeof(float), cudaMemcpyHostToDevice), "upload", err);
}

void Glm5Model::memt_use(bool on) {
    if (fast_ != nullptr && fast_->memt != nullptr) fast_->memt->on = on;
}

bool Glm5Model::memt_using() const { return fast_ != nullptr && fast_->memt != nullptr && fast_->memt->on; }

// the token path: x (F->x, the layer's normed input) -> += q35_ao; in a replay also the write of this token
bool Glm5Model::memt_token(int il, std::string& err) {
    FastState* F = fast_;
    Memt* M = F->memt;
    if (M == nullptr || !(M->on || M->commit)) return true;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    cudaStream_t s = F->cs;
    const int E = g.n_embd, H = g.memt_heads, W = H * kD;
    const uint16_t* in = Ly.mt_in;
    gf::MvJob j[3];
    j[0] = {in, nullptr, F->x, M->q, nullptr, 1.0f, gf::kTypeBF16, E, W};
    j[1] = {in + (size_t) (5 * W + kD) * E, nullptr, F->x, M->gin, nullptr, 1.0f, gf::kTypeBF16, E, kD};
    int nj = 2;
    if (M->commit) j[nj++] = {in + (size_t) W * E, nullptr, F->x, M->kvbw, nullptr, 1.0f, gf::kTypeBF16, E, 4 * W + kD};
    if (!gf::mv(j, nj, s)) { err = "memt: the projections"; return false; }
    gf::MvJob d[2];
    d[0] = {Ly.mt_gd, nullptr, M->gin, M->g, nullptr, 1.0f, gf::kTypeBF16, kD, W};
    int nd = 1;
    if (M->commit) d[nd++] = {Ly.mt_fd, nullptr, M->kvbw + 4 * W, M->fl, nullptr, 1.0f, gf::kTypeBF16, kD, W};
    if (!gf::mv(d, nd, s)) { err = "memt: g_down / f_down"; return false; }
    float* S = M->S + (size_t) Ly.mt * M->per;
    mk::read_tok(M->q, M->g, S, Ly.mt_onorm, F->q35_ao, M->part, H, 1.0f / 16.0f, g.memt_eps, g.memt_softcap, s);
    if (M->commit) {
        float* k = M->kvbw;
        mk::write_prep(k, k + W, k + 2 * W, k + 3 * W, M->fl, Ly.mt_bias, Ly.mt_dsl, 1, H, s);
        mk::write_scan(M->Sw + (size_t) Ly.mt * M->per, k, k + 3 * W, M->fl, 1, H, s);
    }
    return ck(cudaGetLastError(), "the kernels", err);
}

bool Glm5Model::memt_rows(int il, const float* xs, int tn, float* ao, cudaStream_t s,
                          const std::function<void(const uint16_t*, int, int, const float*, float*, int)>& gemm,
                          std::string& err) {
    FastState* F = fast_;
    Memt* M = F->memt;
    if (M == nullptr || !(M->on || M->commit) || tn <= 0) return true;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, H = g.memt_heads, W = H * kD;
    const size_t per_row = (size_t) (2 * W + kD) + (M->commit ? (size_t) (5 * W + kD) : 0);
    if ((size_t) tn * per_row > M->rows_cap) {
        cudaStreamSynchronize(s);
        cudaFree(M->rows);
        M->rows = nullptr;
        const size_t want = (size_t) tn * (size_t) (7 * W + 2 * kD);
        if (!ck(cudaMalloc(&M->rows, want * sizeof(float)), "the rows", err)) { M->rows_cap = 0; return false; }
        M->rows_cap = want;
    }
    float* q = M->rows;
    float* gr = q + (size_t) tn * W;
    float* gin = gr + (size_t) tn * W;
    const uint16_t* in = Ly.mt_in;
    gemm(in, W, E, xs, q, tn);
    gemm(in + (size_t) (5 * W + kD) * E, kD, E, xs, gin, tn);
    gemm(Ly.mt_gd, W, kD, gin, gr, tn);
    float* S = M->S + (size_t) Ly.mt * M->per;
    mk::read_rows(q, gr, S, Ly.mt_onorm, ao, tn, H, 1.0f / 16.0f, g.memt_eps, g.memt_softcap, s);
    if (M->commit) {
        // k | v | b | w | f as separate [tn][..] blocks (each its own GEMM: rows W apiece, f 256)
        float* k = gin + (size_t) tn * kD;
        float* v = k + (size_t) tn * W;
        float* b = v + (size_t) tn * W;
        float* w = b + (size_t) tn * W;
        float* fl = w + (size_t) tn * W;
        float* f = fl + (size_t) tn * W;
        gemm(in + (size_t) W * E, W, E, xs, k, tn);
        gemm(in + (size_t) 2 * W * E, W, E, xs, v, tn);
        gemm(in + (size_t) 3 * W * E, W, E, xs, b, tn);
        gemm(in + (size_t) 4 * W * E, W, E, xs, w, tn);
        gemm(in + (size_t) 5 * W * E, kD, E, xs, f, tn);
        gemm(Ly.mt_fd, W, kD, f, fl, tn);
        mk::write_prep(k, v, b, w, fl, Ly.mt_bias, Ly.mt_dsl, tn, H, s);
        mk::write_scan(M->Sw + (size_t) Ly.mt * M->per, k, w, fl, tn, H, s);
    }
    return ck(cudaGetLastError(), "the rows' kernels", err);
}

bool Glm5Model::memt_commit(const std::vector<int32_t>& ids, bool document, std::string& err) {
    Memt* M = fast_ != nullptr ? fast_->memt : nullptr;
    if (M == nullptr) { err = "memt: not a MemT model"; return false; }
    if (ids.empty()) return true;
    cudaSetDevice(dev_);
    const size_t sb = memt_state_floats() * sizeof(float);
    const bool was_on = M->on;
    // a turn (QA): one replay, every read from the state as it was; a document: chunks of chunk_tokens, each from an
    // empty context reading the state the chunk before wrote
    const size_t chunk = document ? (size_t) std::max(1, g_.memt_chunk) : ids.size();
    bool ok = ck(cudaMemcpyAsync(M->Sw, M->S, sb, cudaMemcpyDeviceToDevice, fast_->cs), "copy", err);
    for (size_t c0 = 0; ok && c0 < ids.size(); c0 += chunk) {
        const std::vector<int32_t> part(ids.begin() + (long) c0, ids.begin() + (long) std::min(ids.size(), c0 + chunk));
        if (document && c0 > 0) ok = ck(cudaMemcpyAsync(M->S, M->Sw, sb, cudaMemcpyDeviceToDevice, fast_->cs), "copy", err);
        reset();
        M->on = true;
        M->commit = true;
        std::string e;
        bool done = ok && prefill(part, e);
        if (ok && !done && e.empty()) {   // (no prompt path for this many tokens: the token path)
            std::vector<float> lg;
            done = true;
            for (size_t i = 0; done && i < part.size(); i += 8) {
                const std::vector<int32_t> p8(part.begin() + (long) i, part.begin() + (long) std::min(part.size(), i + 8));
                done = forward(p8, lg, e);
            }
        }
        M->commit = false;
        if (!done) { err = "memt replay: " + e; ok = false; }
    }
    if (ok) ok = ck(cudaMemcpyAsync(M->S, M->Sw, sb, cudaMemcpyDeviceToDevice, fast_->cs), "adopt", err) &&
                 ck(cudaStreamSynchronize(fast_->cs), "sync", err);
    M->on = was_on;
    reset();
    return ok;
}

}  // namespace strata::core
