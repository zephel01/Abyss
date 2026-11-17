// src/core/glm_abl.cu - the refusal-direction edit at run time (uncensoring without a new file): STRATA_GLM_ABL=<file>
// or --uncensor <file>, built by --abl-build (src/program/generate.cpp) or exported from an abliteration's edit
// (tools/abl_export.py).
//
// An edit is an orthonormal Q [n_embd][r] (r <= 4) and a strength a layer; a write w into the residual becomes
// w - s Q Q^T w - the same as the abliterated weights W' = W - s Q Q^T W, without re-quantizing them (the routed
// experts stay on disk untouched).  Two kinds:
//   ABL1 - an abliteration's own edit (mrfakename's): the attention and the shared expert's outputs (strengths
//          [layer][attention, shared expert]); the call sites are the abl_apply()s in the paths.
//   ABL2 - every write into the residual: the embedding (one strength), each layer's attention output and its whole
//          FFN output (routed + shared + the CPU lane's: strengths [layer][attention, FFN]), and V4.1's Engram adds -
//          a residual that never holds the direction (Arditi et al.'s directional ablation).  The sites are
//          abl_out() / abl_res(); the CPU lane's answer is then added before the next read (no hc3 fusion, no
//          deferral), so the FFN output is complete when it is edited.
// Files: "ABL1" | "ABL2" | "ABL3", int32 n_embd, rank, layers, float32 Q [n_embd][rank] row-major, (ABL3) float32
// scale[rank] - each direction's own factor on top of the layer's strength -, strengths [layers][2], and (ABL2 / ABL3)
// a float32 embedding strength.  ABL3 is an ABL2 edit whose directions are taken out by different amounts.
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core {
namespace gf = strata::kernels::glmf;

bool Glm5Model::abl_set(const std::vector<float>& q, int rank, const std::vector<float>& strengths, float emb,
                        std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    if (F == nullptr) { err = "abl: the fast path is not set up"; return false; }
    if (rank < 1 || rank > 4 || q.size() != (size_t) g.n_embd * rank || strengths.empty() || strengths.size() % 2 != 0) {
        err = "abl: the edit's shapes (rank 1..4, Q n_embd x rank, strengths [layers][2])";
        return false;
    }
    cudaSetDevice(dev_);
    if (F->abl_q == nullptr && cudaMalloc((void**) &F->abl_q, (size_t) g.n_embd * 4 * sizeof(float)) != cudaSuccess) {
        err = "abl: cudaMalloc";
        return false;
    }
    if (F->abl_on == nullptr) {
        if (cudaMalloc((void**) &F->abl_on, sizeof(int)) != cudaSuccess) { err = "abl: cudaMalloc"; return false; }
        const int one = 1;
        cudaMemcpy(F->abl_on, &one, sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaStreamSynchronize(F->cs);   // (the kernels in flight read the old one)
    if (cudaMemcpy(F->abl_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "abl: the edit's upload";
        return false;
    }
    F->abl_r = rank;
    F->abl_s = strengths;
    F->abl_emb = emb;
    F->abl_full = true;   // (abl_load sets an ABL1 file's kind after)
    if (F->defer) {   // (a deferred CPU-lane part would join the residual unedited, a layer late)
        F->defer = false;
        std::fprintf(stderr, "glm fast: STRATA_GLM_DEFER is off while the refusal-direction edit is loaded\n");
    }
    // (a layer split: the parts set up after this one load the file themselves in their fast_setup)
    if (split_next_ && split_next_->fast_ != nullptr && !split_next_->abl_set(q, rank, strengths, emb, err)) return false;
    return true;
}

bool Glm5Model::abl_load(const std::string& path, std::string& err) {
    const Glm5Geometry& g = g_;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) { err = "abl: cannot open " + path; return false; }
    char magic[4];
    int32_t hd[3];
    std::vector<float> q, s;
    float emb = 0.0f;
    bool ok = std::fread(magic, 1, 4, f) == 4 &&
              (std::memcmp(magic, "ABL1", 4) == 0 || std::memcmp(magic, "ABL2", 4) == 0 || std::memcmp(magic, "ABL3", 4) == 0) &&
              std::fread(hd, 4, 3, f) == 3 && hd[0] == (int32_t) g.n_embd && hd[1] >= 1 && hd[1] <= 4 && hd[2] >= 1 &&
              hd[2] <= 1024;
    const bool full = ok && magic[3] != '1', per_dir = ok && magic[3] == '3';
    std::vector<float> ds;   // ABL3: each direction's own scale
    if (ok) {
        q.resize((size_t) hd[0] * hd[1]);
        s.resize((size_t) hd[2] * 2);
        ds.assign((size_t) hd[1], 1.0f);
        ok = std::fread(q.data(), 4, q.size(), f) == q.size() &&
             (!per_dir || std::fread(ds.data(), 4, ds.size(), f) == ds.size()) &&
             std::fread(s.data(), 4, s.size(), f) == s.size() && (!full || std::fread(&emb, 4, 1, f) == 1);
        for (float c : ds) ok = ok && c >= 0.0f && c <= 16.0f;
    }
    std::fclose(f);
    if (!ok) {
        err = "abl: " + path + " is not an ABL1 / ABL2 / ABL3 file for n_embd " + std::to_string(g.n_embd);
        return false;
    }
    // ABL3: w - s * sum_j c_j q_j (q_j . w) is the same projection with column j of Q times sqrt(c_j) (the kernel takes
    // any Q: its two products with Q give c_j); ABL1 / ABL2 files are uploaded as they are
    if (per_dir)
        for (size_t i = 0; i < (size_t) hd[0]; ++i)
            for (int j = 0; j < hd[1]; ++j) q[i * (size_t) hd[1] + (size_t) j] *= std::sqrt(ds[(size_t) j]);
    if (!abl_set(q, hd[1], s, emb, err)) return false;
    for (Glm5Model* m = this; m != nullptr && m->fast_ != nullptr; m = m->split_next_.get()) m->fast_->abl_full = full;
    int na = 0, nf = 0;
    for (int l = 0; l < hd[2]; ++l) {
        na += s[(size_t) l * 2] != 0.0f;
        nf += s[(size_t) l * 2 + 1] != 0.0f;
    }
    if (full)
        std::fprintf(stderr, "glm fast: uncensored (%s): the refusal direction (rank %d%s) out of every residual write - "
                             "the embedding%s, the attention outputs of %d layers, the FFN outputs of %d\n",
                     path.c_str(), hd[1], per_dir ? ", a scale a direction" : "", emb != 0.0f ? "" : " (not)", na, nf);
    else
        std::fprintf(stderr, "glm fast: refusal-direction edit at run time (%s): rank %d, attention outputs of %d layers, "
                             "shared-expert outputs of %d\n", path.c_str(), hd[1], na, nf);
    return true;
}

void Glm5Model::abl_clear() {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (FastState* F = m->fast_) {
            cudaSetDevice(m->dev_);
            cudaStreamSynchronize(F->cs);
            F->abl_s.clear();
            F->abl_r = 0;
            F->abl_full = false;
            F->abl_emb = 0.0f;
        }
}

bool Glm5Model::abl_loaded() const { return fast_ != nullptr && fast_->abl_r > 0; }
bool Glm5Model::abl_full() const { return fast_ != nullptr && fast_->abl_r > 0 && fast_->abl_full; }

void Glm5Model::abl_enable(bool on) {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        FastState* F = m->fast_;
        if (F == nullptr || F->abl_on == nullptr) continue;
        cudaSetDevice(m->dev_);
        const int v = on ? 1 : 0;
        cudaMemcpyAsync(F->abl_on, &v, sizeof(int), cudaMemcpyHostToDevice, F->cs);
        cudaStreamSynchronize(F->cs);
    }
}

void Glm5Model::abl_apply(int il, int which, float* x, int rows, int ld, cudaStream_t s) {
    const FastState* F = fast_;
    if (F == nullptr || F->abl_r == 0 || F->abl_full || il < 0 || (size_t) il * 2 + 1 >= F->abl_s.size()) return;
    const float sc = F->abl_s[(size_t) il * 2 + which];
    if (sc != 0.0f) gf::abl_project(x, rows, ld, g_.n_embd, F->abl_q, F->abl_r, sc, s, F->abl_on);
}

void Glm5Model::abl_out(int il, int which, float* x, int rows, int ld, cudaStream_t s) {
    const FastState* F = fast_;
    if (F == nullptr || F->abl_r == 0 || !F->abl_full || il < 0 || (size_t) il * 2 + 1 >= F->abl_s.size()) return;
    const float sc = F->abl_s[(size_t) il * 2 + which];
    if (sc != 0.0f) gf::abl_project(x, rows, ld, g_.n_embd, F->abl_q, F->abl_r, sc, s, F->abl_on);
}

void Glm5Model::abl_res(float* x, int rows, int ld, cudaStream_t s) {
    const FastState* F = fast_;
    if (F == nullptr || F->abl_r == 0 || !F->abl_full || F->abl_emb == 0.0f) return;
    gf::abl_project(x, rows, ld, g_.n_embd, F->abl_q, F->abl_r, F->abl_emb, s, F->abl_on);
}

void Glm5Model::abl_cap(int slot, const float* R, int streams, cudaStream_t s) {
    const FastState* F = fast_;
    if (F == nullptr || !F->abl_cap_on || F->abl_cap == nullptr || slot < 0 || slot > g_.n_layers) return;
    gf::abl_capture(R, 1, streams * (int) g_.n_embd, (int) g_.n_embd, streams, F->abl_cap + (size_t) slot * g_.n_embd, s);
}

bool Glm5Model::abl_capture(bool on, std::string& err) {
    FastState* F = fast_;
    if (F == nullptr) { err = "abl capture: the fast path is not set up"; return false; }
    if (on && (split_next_ || l0_ != 0 || l1_ != g_.n_layers)) {
        err = "abl capture: one device holding every layer only";
        return false;
    }
    cudaSetDevice(dev_);   // (the iGPU lane's device may be current)
    if (on && F->abl_cap == nullptr) {
        const size_t n = (size_t) (g_.n_layers + 1) * g_.n_embd;
        if (cudaMalloc((void**) &F->abl_cap, n * sizeof(float)) != cudaSuccess) { err = "abl capture: cudaMalloc"; return false; }
        cudaMemset(F->abl_cap, 0, n * sizeof(float));
    }
    F->abl_cap_on = on;
    return true;
}

bool Glm5Model::abl_captured(std::vector<float>& out) {
    FastState* F = fast_;
    if (F == nullptr || F->abl_cap == nullptr) return false;
    out.resize((size_t) (g_.n_layers + 1) * g_.n_embd);
    cudaSetDevice(dev_);
    cudaError_t e = cudaStreamSynchronize(F->cs);
    if (e == cudaSuccess) e = cudaMemcpy(out.data(), F->abl_cap, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) std::fprintf(stderr, "abl capture: %s\n", cudaGetErrorString(e));
    return e == cudaSuccess;
}

}  // namespace strata::core
