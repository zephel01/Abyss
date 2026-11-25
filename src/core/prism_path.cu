// src/core/prism_path.cu - PrismML's Ternary Bonsai GGUFs on the dense qwen35 path (Glm5Model members).
//
// The file's prism.hadamard.* keys (Glm5Geometry::prism_*) say which weights are stored in a blockwise Hadamard basis;
// their input goes through H_B S (P) first (include/strata/kernels/prism_had.hpp).  Every rotated weight of a dense
// qwen35 layer shares its input with the others of its site - x after attn_norm (qkv, gate / q, k, v), x after
// post_attention_norm (gate, up), the gated GDN output (ssm_out, with P), the gated attention output (attn_output), the
// SwiGLU product (down), the head's x (output) - so each site rotates its float rows once in place and requantizes.
// The F32 ssm_alpha / ssm_beta rows read that same x: they are rotated once at load (row' = H S row, exact up to FP32
// rounding), so W x = W' x' holds for them too.  The embedding rows (token_embd, stored rotated) are restored after the
// lookup: h = S H z.
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/prism_had.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::core {

namespace gf = strata::kernels::glmf;
namespace pk = strata::kernels::prism;

struct Prism {
    std::map<int, float*> signs;   // by width, on the device
    float* tmp = nullptr;          // the GDN rows' rotated copy
    size_t tmp_floats = 0;
    std::vector<uint8_t> done;     // prism_weights: a layer's alpha / beta rotated
};

bool Glm5Model::prism_alloc(std::string& err) {
    if (!prism_on() || fast_ == nullptr) return true;
    FastState* F = fast_;
    if (F->prism != nullptr) return true;
    Prism* P = new Prism();
    F->prism = P;
    for (const auto& kv : g_.prism_signs) {
        float* d = nullptr;
        if (cudaMalloc(&d, kv.second.size() * sizeof(float)) != cudaSuccess ||
            cudaMemcpy(d, kv.second.data(), kv.second.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaGetLastError();
            err = "prism: the sign vectors";
            return false;
        }
        P->signs[kv.first] = d;
    }
    // the GDN output's rotated copy: the prompt path's largest sub-batch, or the verify window's rows
    P->tmp_floats = (size_t) g_.ssm_inner * (size_t) std::max(kVerifyMax, 1024);
    if (cudaMalloc(&P->tmp, P->tmp_floats * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        err = "prism: the scratch rows";
        return false;
    }
    P->done.assign((size_t) g_.n_layers + 8, 0);
    std::fprintf(stderr, "prism: Hadamard blocks of %d, %zu rotated weights, sign vectors for %zu widths%s%s\n",
                 g_.prism_block, g_.prism_rot.size(), g_.prism_signs.size(), g_.prism_inv_embd ? ", embedding restored" : "",
                 g_.prism_gdn_grouped ? ", GDN grouped" : "");
    return true;
}

void Glm5Model::prism_free() {
    if (fast_ == nullptr || fast_->prism == nullptr) return;
    Prism* P = fast_->prism;
    for (auto& kv : P->signs) cudaFree(kv.second);
    cudaFree(P->tmp);
    delete P;
    fast_->prism = nullptr;
}

static pk::Rot prism_rot_of(const Glm5Geometry& g, const Prism* P, int n, bool gdn) {
    pk::Rot r;
    r.n = n;
    r.B = g.prism_block;
    const auto it = P->signs.find(n);
    r.signs = it != P->signs.end() ? it->second : nullptr;
    if (gdn && g.prism_gdn_grouped) {
        r.hd = g.ssm_state;
        r.nk = g.ssm_k_heads;
        r.rep = g.ssm_v_heads / g.ssm_k_heads;
    }
    return r;
}

bool Glm5Model::prism_in(float* x, int n, int rows, int ld, void* xq, uint16_t* x16, int ld16, cudaStream_t s) {
    if (!prism_on()) return true;
    const Prism* P = fast_->prism;
    if (P == nullptr) return false;
    const pk::Rot r = prism_rot_of(g_, P, n, false);
    if (!g_.prism_signs.empty() && r.signs == nullptr) return false;   // (explicit signs without this width)
    return pk::rotate(r, x, ld, x, ld, x16, ld16, rows, s, xq);   // (the q8_1 rows in the same launch)
}

bool Glm5Model::prism_in_ff(float* x, int n, int rows, int ld, void* xq, cudaStream_t s) {
    if (!prism_on()) return true;
    const Prism* P = fast_->prism;
    if (P == nullptr || n <= 0 || n > g_.n_ff_dense || n % g_.prism_block != 0) return false;
    pk::Rot r = prism_rot_of(g_, P, g_.n_ff_dense, false);
    if (!g_.prism_signs.empty() && r.signs == nullptr) return false;
    r.n = n;   // (the first n values: whole blocks, their signs the first n of n_ff's)
    return pk::rotate(r, x, ld, x, ld, nullptr, 0, rows, s, xq);
}

bool Glm5Model::prism_gdn(float* y, int rows, int ld, void* yq, uint16_t* y16, int ld16, cudaStream_t s) {
    if (!prism_on()) return true;
    Prism* P = fast_->prism;
    const int n = g_.ssm_inner;
    if (P == nullptr) return false;
    const pk::Rot r = prism_rot_of(g_, P, n, true);
    if (!g_.prism_signs.empty() && r.signs == nullptr) return false;
    // (out of place - the reorder reads other blocks - then back over y, a scratch's worth of rows at a time)
    const int step = (int) std::max<size_t>(1, P->tmp_floats / (size_t) n);
    for (int r0 = 0; r0 < rows; r0 += step) {
        const int rn = std::min(step, rows - r0);
        float* yr = y + (size_t) r0 * ld;
        if (!pk::rotate(r, yr, ld, P->tmp, n, y16 != nullptr ? y16 + (size_t) r0 * ld16 : nullptr, ld16, rn, s,
                        yq != nullptr ? (uint8_t*) yq + (size_t) r0 * (n / 32) * 36 : nullptr))
            return false;
        // (with yq the products read only the q8_1 rows - every rotated weight is a quantized type, from_gguf_qwen35 -
        // so y keeps its unrotated floats and the copy back is skipped: the token path's and the window's case)
        if (yq == nullptr &&
            cudaMemcpy2DAsync(yr, (size_t) ld * sizeof(float), P->tmp, (size_t) n * sizeof(float), (size_t) n * sizeof(float),
                              (size_t) rn, cudaMemcpyDeviceToDevice, s) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
    }
    return true;
}

bool Glm5Model::prism_q8(const float* x, int n, int rows, void* xq, cudaStream_t s) {
    if (!prism_on()) return true;
    Prism* P = fast_->prism;
    if (P == nullptr || (size_t) rows * n > P->tmp_floats) return false;
    const pk::Rot r = prism_rot_of(g_, P, n, false);
    if (!g_.prism_signs.empty() && r.signs == nullptr) return false;
    return pk::rotate(r, x, n, P->tmp, n, nullptr, 0, rows, s, xq);
}

bool Glm5Model::prism_emb(float* e, int rows, int ld, cudaStream_t s) {
    if (!prism_on() || !g_.prism_inv_embd) return true;
    const Prism* P = fast_->prism;
    if (P == nullptr) return false;
    const pk::Rot r = prism_rot_of(g_, P, g_.n_embd, false);
    if (!g_.prism_signs.empty() && r.signs == nullptr) return false;
    return pk::unrotate(r, e, ld, e, ld, nullptr, 0, rows, s);
}

// the kernel's steps in the same order (each butterfly one FP32 add or subtract, then the exact scale and sign)
void Glm5Model::prism_emb_host(float* row) const {
    if (!prism_on() || !g_.prism_inv_embd) return;
    const int B = g_.prism_block, n = g_.n_embd;
    const float scale = (float) (1.0 / std::sqrt((double) B));
    const auto it = g_.prism_signs.find(n);
    const float* sg = it != g_.prism_signs.end() ? it->second.data() : nullptr;
    for (int b0 = 0; b0 < n; b0 += B) {
        float* x = row + b0;
        for (int h = 1; h < B; h <<= 1)
            for (int p = 0; p < B / 2; ++p) {
                const int i = (p / h) * 2 * h + (p % h), j = i + h;
                const float a = x[i], c = x[j];
                x[i] = a + c;
                x[j] = a - c;
            }
        for (int i = 0; i < B; ++i) {
            float v = x[i] * scale;
            if (sg != nullptr) v *= sg[b0 + i];
            x[i] = v;
        }
    }
}

bool Glm5Model::prism_weights(int il, std::string& err) {
    if (!prism_on()) return true;
    FastState* F = fast_;
    Prism* P = F->prism;
    if (P == nullptr || il < 0 || (size_t) il >= P->done.size()) { err = "prism: not set up"; return false; }
    if (P->done[(size_t) il]) return true;
    auto& Ly = F->L[(size_t) il];
    {   // every rotated weight a quantized type: their products read only the q8_1 rows (prism_gdn leaves y unrotated)
        const WSlot* ws[] = {&Ly.qkv, &Ly.zg, &Ly.ssm_o, &Ly.aq, &Ly.ak, &Ly.av, &Ly.out, &Ly.ffn_gate, &Ly.ffn_up, &Ly.ffn_down};
        for (const WSlot* w : ws)
            if (w->q != nullptr && (w->type == gf::kTypeF32 || w->type == gf::kTypeBF16 || w->type == 1)) {
                err = "prism: layer " + std::to_string(il) + " has an unquantized rotated weight (type " +
                      std::to_string(w->type) + ")";
                return false;
            }
    }
    if (!Ly.recr) { P->done[(size_t) il] = 1; return true; }
    // the GDN layer's qkv / gate inputs are rotated: so the x alpha / beta read; their rows get the same transform
    const std::string pre = "blk." + std::to_string(il) + ".";
    if (!g_.prism_has(pre + "attn_qkv.weight") || !g_.prism_has(pre + "attn_gate.weight")) {
        err = "prism: layer " + std::to_string(il) + " rotates only some of attn_norm's consumers";
        return false;
    }
    const WSlot* ab[2] = {&Ly.alpha_w, &Ly.beta_w};
    for (const WSlot* w : ab) {
        if (w->type != gf::kTypeF32 || w->q == nullptr) {
            err = "prism: ssm_alpha / ssm_beta must be F32 (they are rotated at load)";
            return false;
        }
        const pk::Rot r = prism_rot_of(g_, P, g_.n_embd, false);
        float* rows = (float*) w->q;
        if (!pk::rotate(r, rows, g_.n_embd, rows, g_.n_embd, nullptr, 0, g_.ssm_v_heads, F->cs)) {
            err = "prism: rotating ssm_alpha / ssm_beta";
            return false;
        }
    }
    if (cudaStreamSynchronize(F->cs) != cudaSuccess) { cudaGetLastError(); err = "prism: rotating ssm_alpha / beta"; return false; }
    P->done[(size_t) il] = 1;
    return true;
}

}  // namespace strata::core
