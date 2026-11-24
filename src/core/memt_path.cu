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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/memt.hpp"
#include "strata/kernels/dsv4_fast.hpp"

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
    // Flash (memt4_*): the token path's latents and their q8_1 (ql q 1024 | kvbwf 9728 | gl 512), the prompt path's
    // FP16 latents
    float* lat = nullptr;
    void* latq = nullptr;
    uint16_t* lat16 = nullptr;
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
    M->n = (int) memt_layers().size();
    const int Dm = g.memt_dim;
    const size_t W = (size_t) g.memt_heads * Dm;
    M->per = W * Dm;
    const size_t sb = (size_t) M->n * M->per * sizeof(float);
    if (!ck(cudaMalloc(&M->S, sb), "the state", err) || !ck(cudaMalloc(&M->Sw, sb), "the write state", err)) return false;
    cudaMemset(M->S, 0, sb);
    cudaMemset(M->Sw, 0, sb);
    if (g.memt_qrank > 0) {   // Flash: q g k v b w f [W] each, the split read's parts; the latents and their q8_1
        const size_t L = (size_t) g.memt_qrank + 2 * (size_t) g.memt_groups * g.memt_kvrank + 4 * (size_t) g.memt_grank;
        float* a = nullptr;
        if (!ck(cudaMalloc(&a, (7 * W + (size_t) g.memt_heads * mk::kSplit * Dm + L) * sizeof(float)), "the token scratch", err) ||
            !ck(cudaMalloc(&M->latq, L / 32 * 36 + 256), "the token scratch", err))
            return false;
        M->q = a;
        M->g = a + W;
        M->kvbw = a + 2 * W;    // k | v | b | w | f, W apiece
        M->part = a + 7 * W;
        M->lat = M->part + (size_t) g.memt_heads * mk::kSplit * Dm;
        std::fprintf(stderr, "memt: %d memory layers, state %.1f MiB (x2 with the replay's)\n", M->n, sb / 1048576.0);
        return true;
    }
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
    cudaFree(M->latq);
    cudaFree(M->lat16);
    delete M;
    fast_->memt = nullptr;
}

std::vector<int> Glm5Model::memt_layers() const {
    std::vector<int> out;
    for (int il = 0; il < g_.n_layers; ++il)
        if (g_.memt_layer.empty() ? g_.n_head_kv[(size_t) il] != 0 : g_.memt_layer[(size_t) il] >= 0) out.push_back(il);
    return out;
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
    mk::read_tok(M->q, M->g, nullptr, S, Ly.mt_onorm, F->q35_ao, M->part, H, kD, false, 1.0f / 16.0f, g.memt_eps,
                 g.memt_softcap, s);
    if (M->commit) {
        float* k = M->kvbw;
        mk::write_prep(k, k + W, k + 2 * W, k + 3 * W, M->fl, Ly.mt_bias, Ly.mt_dsl, 1, H, kD, s);
        mk::write_scan(M->Sw + (size_t) Ly.mt * M->per, k, k + 3 * W, M->fl, 1, H, kD, s);
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
    mk::read_rows(q, gr, nullptr, S, Ly.mt_onorm, ao, tn, H, kD, false, 1.0f / 16.0f, g.memt_eps, g.memt_softcap, s);
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
        mk::write_prep(k, v, b, w, fl, Ly.mt_bias, Ly.mt_dsl, tn, H, kD, s);
        mk::write_scan(M->Sw + (size_t) Ly.mt * M->per, k, w, fl, tn, H, kD, s);
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


// ---------------------------------------------------------------- MemT-1.0-Flash (the deepseek4 path)
// The memory block of a CSA layer (worklog 2026-11-25_memt-flash.md section 2): from the layer's normed input x,
// memt_in gives the latents q (q_rank) | k, v (groups x kv_rank) | b, w, f, g (gate_rank each).  q = q_up(RMSNorm(q));
// the read (against the state rounded to BF16) is RMS-normed, scaled by o_norm, gated by softcap-SiLU(g_down(g) +
// g_bias) and added to the de-roped attention output before wo_a (wo_a is linear: the reference's wo_a(attn) +
// wo_a(read)).  A replay's write: k / v group-RMS-normed and expanded to the group's heads, b / w / f expanded, then the
// same delta rule as Nano's.  The projections are Q8_0 (tools/memt_flash.py) on the quantized GEMV (token path) or the
// FP16 GEMM on dequantized rows (prompt path).
namespace {
WSlot rows_of(const WSlot& w, int64_t r0, int64_t n) {
    WSlot o = w;
    o.q = (const uint8_t*) w.q + (size_t) r0 * gf::row_bytes(w.type, w.n_in);
    o.n_out = n;
    return o;
}
size_t qoff(int64_t elem) { return (size_t) (elem / 32) * 36; }   // a q8_1 vector's byte offset of element elem
// STRATA_MEMT_DUMP=<dir> (checks against recipes/memt-flash/ref_block.py): the rows of a memory layer's input appended
// to <dir>/x-<il>.f32 (a replay: every row), and a read's last row: x, the output before and after the read
void mdump(cudaStream_t s, const std::string& name, const float* dev, size_t n, bool append) {
    static const char* dir = getenv("STRATA_MEMT_DUMP");
    if (dir == nullptr || dir[0] == 0) return;
    cudaStreamSynchronize(s);
    std::vector<float> buf(n);
    cudaMemcpy(buf.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (FILE* f = std::fopen((std::string(dir) + "/" + name + ".f32").c_str(), append ? "ab" : "wb")) {
        std::fwrite(buf.data(), 4, n, f);
        std::fclose(f);
    }
}
bool mdump_on() {
    static const bool on = getenv("STRATA_MEMT_DUMP") != nullptr && getenv("STRATA_MEMT_DUMP")[0] != 0;
    return on;
}
}  // namespace

void Glm5Model::memt4_bind(int il, std::string& missing) {
    auto& Ly = fast_->L[(size_t) il];
    const std::string P = "blk." + std::to_string(il) + ".";
    const auto q = [&](const char* n) -> WSlot {
        const auto it = ws_map_.find(P + n);
        if (it == ws_map_.end() || it->second.type == 0 || !gf::mv_supported(it->second.type)) {
            missing += P + n + " (quant) ";
            return WSlot{};
        }
        return it->second;
    };
    const auto f32 = [&](const char* n) -> const float* {
        const auto it = w_.find(P + n);
        if (it == w_.end()) { missing += P + n + " (f32) "; return nullptr; }
        return it->second;
    };
    Ly.mt4_in = q("memt_in.weight");
    Ly.mt4_qup = q("memt_q_up.weight");
    Ly.mt4_kup = q("memt_k_up.weight");
    Ly.mt4_vup = q("memt_v_up.weight");
    Ly.mt4_bup = q("memt_b_up.weight");
    Ly.mt4_wup = q("memt_w_up.weight");
    Ly.mt4_fd = q("memt_f_down.weight");
    Ly.mt4_gd = q("memt_g_down.weight");
    Ly.mt4_qn = f32("memt_q_norm.weight");
    Ly.mt4_kn = f32("memt_k_norm.weight");
    Ly.mt4_vn = f32("memt_v_norm.weight");
    Ly.mt4_gb = f32("memt_g_bias");
    Ly.mt_bias = f32("memt_decay_bias");
    Ly.mt_dsl = f32("memt_decay_scale_log");
    Ly.mt_onorm = f32("memt_o_norm.weight");
    Ly.mt = g_.memt_layer[(size_t) il];
}

// the token path: x (F->x / F->xq, the layer's normed input) -> += F->ds_o (and its q8_1 again); a replay's write
bool Glm5Model::memt4_token(int il, std::string& err) {
    FastState* F = fast_;
    Memt* M = F->memt;
    if (M == nullptr || !(M->on || M->commit)) return true;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    cudaStream_t s = F->cs;
    const int E = g.n_embd, H = g.memt_heads, Dm = g.memt_dim, W = H * Dm;
    const int QR = g.memt_qrank, G = g.memt_groups, KR = g.memt_kvrank, GR = g.memt_grank, KV = G * KR;
    const int L = QR + 2 * KV + 4 * GR, WG = W / G;
    const WSlot& in = Ly.mt4_in;
    float* lat = M->lat;
    uint8_t* lq = (uint8_t*) M->latq;
    gf::MvJob j[gf::kMaxMvJobs];
    int nj = 0;
    const auto job = [&](const WSlot& w, int64_t r0, int nout, int64_t in_off, bool from_x, float* y) {
        j[nj++] = {(const uint8_t*) w.q + (size_t) r0 * gf::row_bytes(w.type, w.n_in), from_x ? F->xq : lq + qoff(in_off),
                   from_x ? F->x : lat + in_off, y, nullptr, 1.0f, w.type, (int) w.n_in, nout};
    };
    const auto run = [&](const char* what) {
        const bool ok = gf::mv(j, nj, s);
        nj = 0;
        if (!ok) err = std::string("memt: ") + what;
        return ok;
    };
    const std::string Ls = std::to_string(il);
    if (mdump_on()) {
        if (M->commit) mdump(s, "wx-" + Ls, F->x, (size_t) E, true);
        mdump(s, "rx-" + Ls, F->x, (size_t) E, false);
        mdump(s, "rpre-" + Ls, F->ds_o, (size_t) W, false);
    }
    if (M->commit) job(in, 0, L, 0, true, lat);
    else {
        job(in, 0, QR, 0, true, lat);
        job(in, L - GR, GR, 0, true, lat + L - GR);
    }
    if (!run("memt_in")) return false;
    mk::group_rms(lat, Ly.mt4_qn, 1, 1, QR, L, g.memt_eps, s);
    if (M->commit) {
        mk::group_rms(lat + QR, Ly.mt4_kn, 1, G, KR, L, g.memt_eps, s);
        mk::group_rms(lat + QR + KV, Ly.mt4_vn, 1, G, KR, L, g.memt_eps, s);
    }
    gf::quantize_q8_1(lat, lq, L, s);
    job(Ly.mt4_qup, 0, W, 0, false, M->q);
    job(Ly.mt4_gd, 0, W, L - GR, false, M->g);
    if (!run("q_up / g_down")) return false;
    const float* S = M->S + (size_t) Ly.mt * M->per;
    mk::read_tok(M->q, M->g, Ly.mt4_gb, S, Ly.mt_onorm, F->ds_o, M->part, H, Dm, true, 1.0f / std::sqrt((float) Dm),
                 g.memt_eps, g.memt_softcap, s);
    gf::quantize_q8_1(F->ds_o, F->ds_oq, W, s);
    if (mdump_on()) mdump(s, "rpost-" + Ls, F->ds_o, (size_t) W, false);
    if (M->commit) {
        float *K = M->kvbw, *V = K + W, *B = V + W, *Wv = B + W, *Fl = Wv + W;
        for (int gi = 0; gi < G; ++gi) job(Ly.mt4_kup, (int64_t) gi * WG, WG, QR + gi * KR, false, K + (size_t) gi * WG);
        if (!run("k_up")) return false;
        for (int gi = 0; gi < G; ++gi) job(Ly.mt4_vup, (int64_t) gi * WG, WG, QR + KV + gi * KR, false, V + (size_t) gi * WG);
        if (!run("v_up")) return false;
        job(Ly.mt4_bup, 0, W, QR + 2 * KV, false, B);
        job(Ly.mt4_wup, 0, W, QR + 2 * KV + GR, false, Wv);
        job(Ly.mt4_fd, 0, W, QR + 2 * KV + 2 * GR, false, Fl);
        if (!run("b_up / w_up / f_down")) return false;
        mk::write_prep(K, V, B, Wv, Fl, Ly.mt_bias, Ly.mt_dsl, 1, H, Dm, s);
        mk::write_scan(M->Sw + (size_t) Ly.mt * M->per, K, Wv, Fl, 1, H, Dm, s);
    }
    return ck(cudaGetLastError(), "the kernels", err);
}

// the prompt path: rows x16 / xs [tn][n_embd] -> += o [tn][heads x 512] (before wo_a), in pieces of kRc rows
bool Glm5Model::memt4_rows(int il, const uint16_t* x16, const float* xs, int tn, float* o, cudaStream_t s,
                           const std::function<void(const WSlot&, int, int, const uint16_t*, int, float*, int, int)>& gemm,
                           std::string& err) {
    FastState* F = fast_;
    Memt* M = F->memt;
    if (M == nullptr || !(M->on || M->commit) || tn <= 0) return true;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd, H = g.memt_heads, Dm = g.memt_dim, W = H * Dm;
    const int QR = g.memt_qrank, G = g.memt_groups, KR = g.memt_kvrank, GR = g.memt_grank, KV = G * KR;
    const int L = QR + 2 * KV + 4 * GR, WG = W / G;
    constexpr int kRc = 64;
    const size_t want = (size_t) kRc * ((size_t) L + 7 * (size_t) W);
    if (M->rows_cap < want) {
        cudaStreamSynchronize(s);
        cudaFree(M->rows);
        cudaFree(M->lat16);
        M->rows = nullptr;
        M->lat16 = nullptr;
        M->rows_cap = 0;
        if (!ck(cudaMalloc(&M->rows, want * sizeof(float)), "the rows", err) ||
            !ck(cudaMalloc(&M->lat16, (size_t) kRc * L * 2), "the rows", err))
            return false;
        M->rows_cap = want;
    }
    float* lat = M->rows;
    float* q = lat + (size_t) kRc * L;
    float* gr = q + (size_t) kRc * W;
    float *K = gr + (size_t) kRc * W, *V = K + (size_t) kRc * W, *B = V + (size_t) kRc * W;
    float *Wv = B + (size_t) kRc * W, *Fl = Wv + (size_t) kRc * W;
    uint16_t* l16 = M->lat16;
    const float* S = M->S + (size_t) Ly.mt * M->per;
    const WSlot& in = Ly.mt4_in;
    for (int r0 = 0; r0 < tn; r0 += kRc) {
        const int rc = std::min(kRc, tn - r0);
        const auto hgemm = [&](const WSlot& w, int N, int Kd, const uint16_t* X, int ldx, float* Y, int ldy) {
            gemm(w, N, Kd, X, ldx, Y, ldy, rc);
        };
        const uint16_t* x = x16 + (size_t) r0 * E;
        const std::string Ls = std::to_string(il);
        const bool last = r0 + rc == tn;
        if (mdump_on()) {
            if (M->commit) mdump(s, "wx-" + Ls, xs + (size_t) r0 * E, (size_t) rc * E, true);
            if (last) mdump(s, "rx-" + Ls, xs + (size_t) (tn - 1) * E, (size_t) E, false);
            if (last) mdump(s, "rpre-" + Ls, o + (size_t) (tn - 1) * W, (size_t) W, false);
        }
        if (M->commit) hgemm(rows_of(in, 0, L), L, E, x, E, lat, L);
        else {
            hgemm(rows_of(in, 0, QR), QR, E, x, E, lat, L);
            hgemm(rows_of(in, L - GR, GR), GR, E, x, E, lat + L - GR, L);
        }
        mk::group_rms(lat, Ly.mt4_qn, rc, 1, QR, L, g.memt_eps, s);
        if (M->commit) {
            mk::group_rms(lat + QR, Ly.mt4_kn, rc, G, KR, L, g.memt_eps, s);
            mk::group_rms(lat + QR + KV, Ly.mt4_vn, rc, G, KR, L, g.memt_eps, s);
        }
        strata::kernels::dsv4::to_f16(lat, l16, (int64_t) rc * L, s);
        hgemm(Ly.mt4_qup, W, QR, l16, L, q, W);
        hgemm(Ly.mt4_gd, W, GR, l16 + (L - GR), L, gr, W);
        mk::read_rows(q, gr, Ly.mt4_gb, S, Ly.mt_onorm, o + (size_t) r0 * W, rc, H, Dm, true,
                      1.0f / std::sqrt((float) Dm), g.memt_eps, g.memt_softcap, s);
        if (mdump_on() && last) mdump(s, "rpost-" + Ls, o + (size_t) (tn - 1) * W, (size_t) W, false);
        if (M->commit) {
            for (int gi = 0; gi < G; ++gi) {
                hgemm(rows_of(Ly.mt4_kup, (int64_t) gi * WG, WG), WG, KR, l16 + QR + gi * KR, L, K + (size_t) gi * WG, W);
                hgemm(rows_of(Ly.mt4_vup, (int64_t) gi * WG, WG), WG, KR, l16 + QR + KV + gi * KR, L, V + (size_t) gi * WG, W);
            }
            hgemm(Ly.mt4_bup, W, GR, l16 + QR + 2 * KV, L, B, W);
            hgemm(Ly.mt4_wup, W, GR, l16 + QR + 2 * KV + GR, L, Wv, W);
            hgemm(Ly.mt4_fd, W, GR, l16 + QR + 2 * KV + 2 * GR, L, Fl, W);
            mk::write_prep(K, V, B, Wv, Fl, Ly.mt_bias, Ly.mt_dsl, rc, H, Dm, s);
            mk::write_scan(M->Sw + (size_t) Ly.mt * M->per, K, Wv, Fl, rc, H, Dm, s);
        }
    }
    return ck(cudaGetLastError(), "the rows' kernels", err);
}

}  // namespace strata::core
