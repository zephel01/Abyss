// include/strata/core/glm_model.hpp - a glm5-next model runner: GGUF in, streaming logits out.
//
// Phase 3 of docs/maya/GLM5-FLASH.md: the kernel family (glm_hc, glm_kda, glm_dsa, glm_ffn,
// router_sigmoid) composed into a MODEL, with the state a real decode needs living between calls:
//
//   * the mHC residual stack R;
//   * per KDA layer: the recurrent state S and the three conv histories;
//   * per DSA layer: the LATENT cache (kv_lora floats per token - no per-head K/V exists), the
//     indexer key/gate buffers, and the completed pools' pooled keys.
//
// `forward` APPENDS tokens: a batched call and a sequence of single-token calls on the same state
// produce identical logits (glm_model_test holds both to the oracle's dumps - the streaming path
// IS the decode path, so that equivalence is the gate that matters for the engine).
//
// CORRECTNESS-FIRST SCOPE, and the two things that follow from it:
//   * weights are consumed as F32 exactly as the GGUF stores them, in the ggml layouts the kernels
//     document (features fastest everywhere, no transposes anywhere between file and kernel);
//   * a QUANTIZED glm5-next GGUF is refused with a pointer to the packer - riding the engine's
//     quantized GEMV/expert machinery is the pack-integration step, not this runner's.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/core/hop_link.hpp"
#include "strata/core/part_port.hpp"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <condition_variable>
#include <functional>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <chrono>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata::core {

/// One served weight: F32 (dequantized at load) or a ggml-quantized row tensor uploaded RAW to
/// VRAM and dotted by the device MMVQ.  type 0 = F32.
struct WSlot {
    const float* f32 = nullptr;
    int type = 0;
    const void* q = nullptr;
    int64_t n_in = 0, n_out = 0;
};

/// Every number the runner's kernels depend on, parsed from the GGUF's glm5-next.* metadata with
/// the released GLM-5.3-Flash values as defaults (mirrors ref/glm.py::Glm5Config - the Python and
/// C++ readers must agree key for key, which glm_model_test checks against the dumps).
struct Glm5Geometry {
    int n_embd = 4096, n_layers = 45, n_head = 64, n_vocab = 154880;
    int nextn = 0;                     // NextN/MTP layers counted INSIDE block_count (unsloth
                                       // ships 46 with nextn 1); tensors of blk.{n_layers}+ are
                                       // skipped at load and MTP stays deferred
    std::vector<int> n_head_kv;        // 0 marks a KDA layer
    int q_lora = 1536, kv_lora = 512, qk_nope = 256, v_head = 256;
    int kda_head_dim = 128, d_conv = 4;
    float kda_lb = -5.0f;
    int idx_heads = 32, idx_key = 128, idx_top_k = 2048, idx_kpool = 4;
    int idx_select_tail = 1;
    std::vector<int> indexer_full;     // per layer; the arch allows a non-full DSA layer to reuse
                                       // the previous full layer's selection (prev_sel) - this
                                       // runner implements full indexers only and REFUSES the rest
    int hc = 4, sinkhorn_iters = 20;
    float hc_eps = 1e-6f, norm_eps = 1e-5f;
    int n_expert = 288, n_exp_used = 8, n_ff_exp = 2048, n_shared = 1, n_ff_dense = 12288, dense_lead = 3;
    float w_scale = 1.0f;
    int norm_w = 1;
    float swiglu_exp = 10.0f, swiglu_shexp = 10.0f;

    // ---- deepseek4 (DeepSeek V4 Flash, docs/deepseek4/SPEC.md): the same mHC residual, MoE tiers and shared expert,
    // a different attention mixer (a 512-wide K=V row with sliding window + compressed rows, sinks, partial RoPE),
    // sqrt(softplus) routing with hash-routed first layers, a gated hyper-connection head
    int arch = 0;                      // 0 glm5-next, 1 deepseek4
    int head_dim = 512, n_rot = 64, n_swa = 128;
    std::vector<int> comp_ratio;       // per layer: 0 (window only), 4 (CSA: + indexer top-k), 128 (HCA)
    float rope_base = 10000.0f, comp_rope_base = 160000.0f;
    float yarn_factor = 16.0f, yarn_beta_fast = 32.0f, yarn_beta_slow = 1.0f;
    int yarn_orig_ctx = 65536;
    int o_groups = 8, o_lora = 1024;
    int hash_layers = 3;
    int gating = 0;                    // the router's probability: 0 sigmoid, 1 sqrt(softplus)
    // ---- deepseek41 (DeepSeek V4.1 Flash, worklog 2026-10-18_deepseek41-spec.md): arch stays 1 (the deepseek4 path)
    // with these differences: ratios 1/2 (rows of 1 or 2 tokens, no overlap / APE), four KV-source layers owning the
    // compressed caches that the layers after them read (kv_slot), index sources computing the top-k that the layers
    // after them reuse, the candidate blocks, Engram on two layers, the hyper-connections' pre lagging one sublayer,
    // no hash layers, no per-head q norm, no output_hc head
    bool v41 = false;
    std::vector<int> kv_slot;          // per layer: the compressed cache it reads (0..3), -1 = window only
    std::vector<char> kv_src;          // per layer: it computes (owns) slot kv_slot's rows and index keys
    std::vector<char> idx_src;         // per layer: it computes the top-k (readers reuse the last one's)
    std::vector<int> eng_slot;         // per layer: Engram table 0 / 1, -1 = none
    int cand_layer = 20, cand_blocks = 2048, cand_bs = 8;
    bool dsv4() const { return arch == 1; }
    int ratio(int il) const { return il < (int) comp_ratio.size() ? comp_ratio[(size_t) il] : 0; }
    static bool from_gguf_dsv4(const strata::GgufFile& g, Glm5Geometry& out, std::string& err);
    static bool from_gguf_dsv41(const strata::GgufFile& g, Glm5Geometry& out, std::string& err);
    // ---- qwen35moe (Qwen3.5-MoE: Ornith-1.5-122B; src/core/qwen35_fast_path.cu): arch 2 - plain residuals (no
    // hyper-connections, hc 1), gated delta-net layers (ssm_*: S x h_v x S states, a d_conv conv) with a full-attention
    // layer (GQA n_head / attn_head_kv of head_dim, NEOX RoPE on the first n_rot dims, q / k norms, a sigmoid output
    // gate from the q projection) where attention.recurrent_layers says so, a softmax top-k router with renormalised
    // weights (gating 2), a shared expert behind a sigmoid gate, a NextN block the pack drops
    bool qwen35() const { return arch == 2; }
    int ssm_state = 128, ssm_k_heads = 16, ssm_v_heads = 64, ssm_inner = 8192;   // S, h_k, h_v, h_v * S
    int attn_head_kv = 2, n_ff_shexp = 1024;
    int ssm_conv_dim() const { return 2 * ssm_k_heads * ssm_state + ssm_inner; }
    static bool from_gguf_qwen35(const strata::GgufFile& g, Glm5Geometry& out, std::string& err);
    // ---- qwen35 (dense Qwen3.5 / Qwen3.8-27B): arch 2 as qwen35moe with no experts - every layer (the NextN block
    // too) a dense SwiGLU FFN of n_ff_dense (dense_lead past the last block, n_expert 0, n_exp_used 0)
    bool q35_dense() const { return arch == 2 && n_expert == 0; }
    // ---- qwen4exp (Qwen3.8-Flash-Next, llama.cpp src/models/qwen4exp.cpp): arch stays 2 (the qwen35moe path) with
    // hc streams mixed by low-rank hyper-connections in place of the norms (hc_lr wide), the delta-net's output gate a
    // sigmoid, top-k up to glmf::kMaxK, the attention layers' QSA indexer and layer 1's per-layer embedding (PLE)
    bool q4x = false;
    int hc_lr = 320;
    // ---- Qwen3.8-27B with QSA (higa2515/Qwen3.8-27B-QSA-GGUF: the dense qwen35 plus FP32 indexers on the attention
    // layers, attention.indexer.* and compress_ratios in the GGUF; src/core/qwen35_fast_path.cu, kernels/qsa27.hpp):
    // past 2,051 cells a query attends to the indexer's 512 best 4-cell blocks and its own block.  STRATA_Q35_QSA=0:
    // the indexers are not used (the dense model, the same tokens as the file without them)
    bool qsa27 = false;
    // ---- sensenova_loop (SenseNova-Loop-9B): the dense qwen35 with a looped block - n_layers counts the layer VISITS
    // (56); the second visit of the loop [loop_lo, loop_lo + loop_shift) owns no tensors and reads layer il - loop_shift's
    // weights (phys), with its own delta-net states and KV cache rows.  Inside the loop the residual is two streams
    // with an mHC router on every sublayer (blk.N.mhc_{attn,ffn}.{weight,bias}: 8 logits from the flat streams' rms
    // -> 2 pre gates (sigmoid), 2 post gates (2 sigmoid), a 2x2 mix made doubly stochastic by Sinkhorn); mhc_kind 1
    // enters (both streams = the residual), 4 stays, 2 leaves (the mean of the two streams), 0 a plain layer.
    // q35_yarn: YaRN on the attention's RoPE (rope.scaling.type yarn: yarn_factor over yarn_orig_ctx)
    bool loop = false;
    int loop_lo = 0, loop_shift = 0;
    std::vector<int> mhc_kind;
    bool q35_yarn = false;
    int phys(int il) const { return loop && il >= loop_lo && il < loop_lo + loop_shift ? il - loop_shift : il; }
    int kind(int il) const { return il >= 0 && il < (int) mhc_kind.size() ? mhc_kind[(size_t) il] : 0; }
    static bool from_gguf_qwen4exp(const strata::GgufFile& g, Glm5Geometry& out, std::string& err);
    // ---- laguna (poolside Laguna-S-2.1, llama.cpp src/models/laguna.cpp; src/core/qwen35_fast_path.cu's laguna
    // branches, kernels/laguna_fast.hpp): arch stays 2 with laguna set - every layer attention (no delta-net), head_dim
    // 128, q heads per layer (lag_heads; n_head their maximum), a per-head softplus output gate (attn_gate), q / k
    // norms, NEOX RoPE: the full layers n_rot dims of rope_base with YaRN (yarn_*), the sliding-window layers (lag_swa,
    // window n_swa) n_rot_swa dims of rope_base_swa plain; pre-norms attn_norm / ffn_norm; dense_lead dense FFN layers;
    // a sigmoid router with exp_probs_b (gating 0, normalised, w_scale); an ungated shared expert; no NextN
    bool laguna = false;
    std::vector<int> lag_heads;
    std::vector<char> lag_swa;
    float rope_base_swa = 10000.0f;
    int n_rot_swa = 128;
    static constexpr int kLagSlice = 2048;   // the prompt path's attention rows a pass (a window layer's ring slack)
    /// a layer's KV cache rows for a context: a window layer a ring of n_swa + kLagSlice rows, else the context
    int64_t lag_rows(int il, int64_t ctx) const {
        return lag_swa[(size_t) il] ? std::min<int64_t>(ctx, (int64_t) n_swa + kLagSlice) : ctx;
    }
    static bool from_gguf_laguna(const strata::GgufFile& g, Glm5Geometry& out, std::string& err);

    // ---- hy4 (Tencent Hy4-flash, GGUF arch hy_v4_flash; src/core/hy4_fast_path.cu, kernels/hy4_fast.hpp): arch 3.
    // n_embd is the EXPERTS' width here - the latent MoE runs its 512 experts in 1536 (ffn_routed_down / _up around
    // them), so the expert machinery (pool, RAM tier, CPU lane) is sized by n_embd as for every other model; the
    // residual is hy_dim wide (4 hyper-connection streams, no stream-mixing matrix).  Gated MLA attention (one shared
    // 512 latent + a 64-dim rope key, a learned sink per head, a sigmoid output gate per head), q heads per layer
    // (lag_heads), sliding-window layers (lag_swa: window n_swa, rope_base_swa) and the others' compressed indexer
    // (top idx_top_k cells past that many), Engram on hy_eng layers (FP8 rows in the GGUF's second shard)
    bool hy4 = false;
    int hy_dim = 3072;
    float hc_mag = 2.0f;
    std::vector<int> hy_eng;            // the Engram layers (0-based, as the GGUF's ple.layers)
    bool is_hy4() const { return arch == 3; }
    static bool from_gguf_hy4(const strata::GgufFile& g, Glm5Geometry& out, std::string& err);

    // ---- inkling (Thinking Machines Inkling 2 Mini, GGUF arch inkling2_loop / inkling2; src/core/inkling_fast_path.cu,
    // kernels/inkling_fast.hpp): arch stays 2 (the qwen35moe path's MoE, pool, tiers and CPU lane) with ink set.  n_layers
    // counts the PHYSICAL layers (24); layers [ink_lo, ink_hi) run ink_passes times with shared weights (48 layer
    // applications a token), each pass with its own KV and convolution state (a "slot").  No RoPE: a relative-position
    // bias (ink_rel dims a head, a table of ink_extent distances), log-scaled q / r on the full layers, qk / r L2 clips,
    // a per-query sink logit, a sigmoid output gate, short causal convolutions (ink_conv taps) on K, V and the two
    // sublayer outputs, attention residual blocks of ink_block layers (learned depth-attention over the earlier blocks),
    // the loop's controllers mixing the previous pass's messages in.  128 experts, top 8, no shared expert, SITU
    // activation (gate beta ink_situ_b, up ink_situ_lb), a final logit softcap
    bool ink = false;
    std::vector<char> ink_swa;          // per physical layer: a sliding-window layer (window n_swa)
    int ink_rel = 16, ink_extent = 1024, ink_conv = 4, ink_block = 6;
    int ink_lo = 6, ink_hi = 18, ink_passes = 3, ink_rank = 64;
    float ink_qk_clip = 22.627417f, ink_rel_clip = 8.0f, ink_log_floor = 8192.0f, ink_log_alpha = 0.1f;
    float ink_situ_b = 4.0f, ink_situ_lb = 25.0f, ink_softcap = 30.0f;
    int ink_vocab_pad = 0;              // the embedding / output rows (n_vocab is the tokenizer's, unpadded)
    /// the layer applications a token (prelude + passes x core + coda) - every one a slot of its own state
    int ink_slots() const { return n_layers + (ink_passes - 1) * (ink_hi - ink_lo); }
    /// a slot's physical layer
    int ink_phys(int sl) const { return sl < n_layers ? sl : ink_lo + (sl - n_layers) % (ink_hi - ink_lo); }
    /// application ia (in a token's order: the prelude, pass 0's core, pass 1's, ..., the coda): its pass (0 outside
    /// the core), physical layer and slot
    int ink_app_pass(int ia) const {
        const int core = ink_hi - ink_lo;
        return ink_passes > 1 && ia >= ink_lo && ia < ink_lo + ink_passes * core ? (ia - ink_lo) / core : 0;
    }
    int ink_app_phys(int ia) const {
        const int core = ink_hi - ink_lo;
        if (ink_passes <= 1 || ia < ink_lo) return ia;
        if (ia < ink_lo + ink_passes * core) return ink_lo + (ia - ink_lo) % core;
        return ia - (ink_passes - 1) * core;
    }
    int ink_app_slot(int ia) const {
        const int pass = ink_app_pass(ia), il = ink_app_phys(ia);
        return pass == 0 ? il : n_layers + (pass - 1) * (ink_hi - ink_lo) + (il - ink_lo);
    }
    static bool from_gguf_inkling(const strata::GgufFile& g, Glm5Geometry& out, std::string& err);

    int d_inner() const { return kda_head_dim * n_head; }
    int top_pools_max() const { return idx_top_k / idx_kpool; }
    int n_sel_max() const { return idx_top_k + (idx_select_tail ? idx_kpool - 1 : 0); }
    bool is_recr(int il) const { return n_head_kv[(size_t) il] == 0; }

    // arch 2 (qwen35 / qwen35moe / qwen4exp / laguna): the GGUF's end tokens - tokenizer.ggml.eot_token_id, then
    // eos_token_id, eom_token_id - the turn's end list when neither --eos-ids nor the server's stop_ids give one
    // (GLM's ids are ordinary pieces of Qwen's 248K vocab and out of Laguna's)
    std::vector<int64_t> eog;

    static bool from_gguf(const strata::GgufFile& g, Glm5Geometry& out, std::string& err);
};

/// The routing profile's counts ("layer c0 c1 ..." by expert id, tools/glm_expert_prior.py): the pack's
/// expert_counts.txt, else STRATA_GLM_PRIOR=<file> (the server points it at Abyss's shipped profile for a model it has
/// one for - data/expert_counts/ - when the pack has none)
std::string expert_counts_path(const std::string& pack_dir);

class Glm5Model {
public:
    Glm5Model() = default;
    ~Glm5Model();
    Glm5Model(const Glm5Model&) = delete;
    Glm5Model& operator=(const Glm5Model&) = delete;

    /// Parses the metadata, uploads every tensor (F32 only - see the header note) and sizes the
    /// state for `max_ctx` tokens.  `max_ctx` caps the latent/indexer caches, nothing else.
    bool load(const std::string& gguf_path, int64_t max_ctx, std::string& err);
    /// Loads a PACK (tools/iq_pack.py output: index.txt + dense.bin + native_experts.txt beside
    /// the GGUF shards).  Dense tensors are dequantized into the same F32 device arena at load;
    /// the routed experts stay quantized on the CPU path (native_expert), read from the shards'
    /// mmaps per token.  docs/maya/GLM5-FLASH.md §9.  `dev` picks the CUDA device; `l0`/`l1` restrict
    /// this instance to layers [l0, l1) (l1 == 0 means "to the end") - the layer split's halves.
    bool load_pack(const std::string& pack_dir, int64_t max_ctx, std::string& err, int dev = 0, int l0 = 0,
                   int l1 = 0);
    /// The split entry the driver/test use: STRATA_GLM_SPLIT (else `layer_split`, the driver's --layer-split) =
    /// "0" one device; "<layer>" two parts - layers [0,layer) on device 0, the rest on STRATA_GLM_DEV1 (default
    /// 1); "K1,K2,.." one part per visible device, each later one starting at its K; "auto"/absent: every visible
    /// device (STRATA_GLM_DEVS="0,2,.." picks and orders them).  The parts are joined by a per-token 64 KB host
    /// hop of the residual at each boundary (no peer access needed).
    bool load_pack_env(const std::string& pack_dir, int64_t max_ctx, std::string& err,
                       const std::string& layer_split = "");
    /// The explicit split form behind load_pack_env: devs[i] runs layers [bounds[i-1], bounds[i]) (bounds holds
    /// the first layer of each later part, devs.size() - 1 of them).
    bool load_pack_split(const std::string& pack_dir, int64_t max_ctx, const std::vector<int>& bounds,
                         const std::vector<int>& devs, std::string& err);
    static constexpr int kMaxParts = 16;
    /// The fast decode path (src/core/glm_fast_path.cu) is the default for packs; STRATA_GLM_SLOW=1
    /// keeps the correctness-first per-op path (the reference the fast path is checked against).
    bool fast() const { return fast_ != nullptr; }
    /// Skip the per-token logits download (the driver samples on the device).
    void set_host_logits(bool on) { host_logits_ = on; }
    /// Live counters of the fast path (both halves summed), for the driver's STAT lines.
    struct FastStats {
        uint64_t tokens = 0, hits = 0, misses = 0, miss_layers = 0;
        uint64_t ram_hits = 0, disk_reads = 0, promotions = 0, cpu_experts = 0, tail_skips = 0;
        double ms = 0, miss_ms = 0, disk_ms = 0, cpu_ms = 0;
        int64_t pool_slots = 0, pool_used = 0, ram_slots = 0, ram_used = 0;
        double pool_gb = 0, ram_gb = 0;
        uint64_t mirror_b0 = 0, mirror_b1 = 0;   // expert bytes read from the shards' drive / the mirror (STRATA_GLM_MIRROR)
    };
    FastStats fast_stats() const;
    /// The run's facts for the server's INFO line (Abyss Terminal): the lane that computes the RAM tier's experts
    /// ("cpu" / "igpu" / "exl3" / "off"), the drafter ("mtp" GLM's NextN block, "nextn" qwen35moe's, "dspark",
    /// "none"), the disk tail skip's rank (0 = off), expert deferral, a shard mirror, an expert helper card.
    const char* lane_kind() const;
    const char* drafter_kind() const;
    int tail_skip_rank() const;
    bool defer_on() const;
    bool mirror_configured() const;
    bool helper_configured() const;
    void gap_report();   // STRATA_GLM_GAP_PROF=1: prints and resets the between-tokens profile (nothing otherwise)
    /// The CPU LANE (a decode's RAM-tier experts): the share of them copied over PCIe to the GPU instead of computed
    /// on the CPU - 0: the CPU takes every one, 1: none (of f such experts in a route, f - round(share f) go to the
    /// CPU).  pcie_share(): the split in use, as a share; set_pcie_share(s): this split from the next token on, s < 0
    /// the one the engine started with (STRATA_GLM_PCIE_SHARE, else measured at start).  cpu_lane_threads(): the
    /// threads taking its jobs; set_cpu_lane_threads(n): at most n of the pool's, n <= 0 all of them.  Both are for
    /// setup's calibration (per request, no restart); false when this engine has no CPU lane.
    double pcie_share() const;
    bool set_pcie_share(double share);
    // Decode-only optional expert tier; unsupported or failed helpers leave the CPU lane in charge.
    void expert_helper_decode_begin(bool speculative);
    void expert_helper_decode_end();
    // STRATA_GLM_EXPERT_HELPER_RESYNC (#24): the promotions a helper decode held back, run after DONE (not on the clock);
    // `pending` (when given) is asked between rounds: true = the next request is here, stop
    void expert_helper_resync(const std::function<bool()>& pending = {});
    int cpu_lane_threads() const;
    bool set_cpu_lane_threads(int n);
    /// Conversation reuse (the fast path): save the sequence state at the current position - the recurrent KDA
    /// states and conv histories (the DSA caches are append-only, so the position alone restores them) - and
    /// restore it later to continue a prompt that extends the saved one.  false when unsupported.
    bool snapshot_save();
    // the expert usage profile kept between sessions (the warm-up follows it): written after each request
    bool save_usage();
    std::string usage_path() const;
    bool snapshot_restore();
    int64_t snapshot_pos() const { return snap_pos_; }
    /// Conversation slots: the snapshot (its KDA states) and every DSA cache's rows up to its position, every half,
    /// written to a file; slot_load puts them back as the live state and the snapshot, so a conversation set aside
    /// while others ran continues where it was.  The bytes written, or 0 + err.
    // mem != nullptr: the file's bytes go into *mem instead (the server writes them on a thread: a 0.25 GB write took
    // 4-23 s on a 94 GB machine whose RAM tier pins 60 GB, and the next prompt waited for it)
    uint64_t slot_save(const std::string& path, std::string& err, std::string* mem = nullptr);
    bool slot_load(const std::string& path, int64_t n_pos, std::string& err);
    /// Conversation slots on the qwen35 path (Qwen3.8, Ornith, Qwen3.8-27B; qwen35_fast_path.cu): the snapshot
    /// (q35_snap_h_: the delta-net states, taken just before the prompt's last token) and every attention cache's
    /// rows 0 .. n-1 - the NextN block's and the qwen4exp draft layer's caches, qwen4exp's QSA block keys - in one
    /// host buffer of q35_slot_bytes(n).  q35_slot_load puts them back as the live state and the snapshot: the
    /// model is then exactly as it was when that snapshot was taken.  q35_slot_supported: false (and why) on a
    /// layout the slots do not cover (a model split across GPUs, laguna).
    bool q35_slot_supported(std::string& why) const;
    uint64_t q35_slot_bytes(int64_t n) const;
    bool q35_slot_save(uint8_t* dst, uint64_t bytes, std::string& err);
    bool q35_slot_load(const uint8_t* src, uint64_t bytes, int64_t n, std::string& err);
    /// The machine's available memory (MemAvailable; 0 unknown) and pinned host buffers (null: none) for the slots.
    static int64_t host_mem_available();
    static void* host_pinned_alloc(size_t bytes);
    static void host_pinned_free(void* p);
    /// Runs the engine's sampler on the last forward's logits and returns the picked token id
    /// (-1 + err on failure).  Handles the device bookkeeping: with a split the logits (and the
    /// sampler buffer) live on the tail half's device, where peer reads are not available.
    int sample_token(strata::kernels::SamplerParams& sp, std::string& err);
    /// A pick `tok` made from logits the request's bias set (token_ban.hpp) did not yet apply to - a verify window or
    /// a pipelined token computed before the set switched on: when the set bans it, the same row (`row`: device
    /// logits on the tail's GPU; nullptr: the tail's sampler row) is picked again with the set.  Otherwise `tok`.
    int bias_repick(int tok, const float* row, strata::kernels::SamplerParams& sp, std::string& err);
    /// Zeroes R, every KDA state and every DSA cache: the next `forward` starts a fresh sequence.
    void reset();

    /// Appends `tokens` to the current sequence and writes the LAST token's logits (n_vocab f32)
    /// to `logits_out`.  State persists across calls; call `reset` between sequences.
    bool forward(const std::vector<int32_t>& tokens, std::vector<float>& logits_out, std::string& err);
    /// The DEVICE pointer of the last forward()'s logits (n_vocab floats) - for
    /// strata::kernels::sample_tokens, which is a kernel and takes device memory (see sampler.hpp).
    const float* device_logits() const { return sc_ + sc_logits; }

    /// The position the next token will occupy (tokens consumed since the last reset).
    int64_t position() const { return pos_; }
    const Glm5Geometry& geometry() const { return g_; }
    /// The next decoded token is `tok` instead of the sampled one (once; -1 clears): the serve loop's thinking budget
    /// closes the reasoning block this way inside the running decode - the speculative loop takes it as the truth
    /// for its position, like a rejected draft, so nothing is read again.
    void force_next(int32_t tok) { force_q_.clear(); if (tok >= 0) force_q_.push_back(tok); }
    /// force_next for a run of tokens: the next decoded tokens are `toks`, in order (the thinking budget's bridge text
    /// and its </think>); one per position, each taken by the speculative loops as the truth for its position.
    void force_seq(const std::vector<int32_t>& toks) { force_q_.assign(toks.begin(), toks.end()); }
    bool forcing() const { return !force_q_.empty(); }
    /// The on-demand vision encoder (serve VLEND / VRECLAIM; STRATA_GLM_VISION_LEND_MB at load): the first GPU's
    /// lendable pool tail is emptied and freed so the encoder's process can use that VRAM, and allocated again
    /// once it has ended (false while the memory is still taken).  vision_lend_bytes(): 0 when unavailable.
    bool vision_lend(size_t& bytes, std::string& err);
    bool vision_reclaim(std::string& err);
    size_t vision_lend_bytes() const;
    int32_t forced(int32_t sampled) {
        if (!force_q_.empty()) {
            const int32_t t = force_q_.front();
            force_q_.pop_front();
            return t;
        }
        return guard_ids_ != nullptr ? end_guard_pick(sampled) : sampled;
    }
    /// The end-of-turn guard (serve): while set, a SAMPLED token that is one of `*ids` (the turn's end tokens) is
    /// replaced, once - by `replace` (>= 0: the reasoning's </think>, for a turn ended inside the reasoning), or
    /// (-1) by the most likely other token of that position, read from the sampler's logits row.  A quantized
    /// model can close its reasoning and end the turn at once: an empty answer.  nullptr: off.
    void end_guard(const std::vector<int64_t>* ids, int32_t replace) { guard_ids_ = ids; guard_tok_ = replace; }
    bool end_guard_fired() { const bool f = guard_fired_; guard_fired_ = false; return f; }

private:
    bool step(int32_t token, std::string& err);   // one token through all layers

    Glm5Geometry g_;
    std::map<std::string, const float*> w_;       // name -> device pointer (the weights arena)
    float* w_arena_ = nullptr;
    uint64_t w_arena_bytes_ = 0;                           // (the VRAM report at load)
    // state and scratch arenas (device), carved in load(); see glm_model.cpp for the layout
    float* state_ = nullptr;
    uint64_t state_bytes_ = 0;
    float* sc_ = nullptr;
    uint64_t sc_bytes_ = 0;
    int* d_pos_ = nullptr;                        // the select kernel reads the position on device
    int64_t pos_ = 0;
    int64_t max_ctx_ = 0;
    bool loaded_ = false;

    // per-layer state offsets, in floats into state_ (-1: the layer has no such state)
    std::vector<int64_t> kda_S_, kda_conv_, dsa_lat_, dsa_ik_, dsa_ig_, dsa_pool_;

public:
    // ---- per-weight serving slots (strata::core::WSlot below): F32 pointers for the
    // dequantized tensors, typed raw pointers for the quantized ones (device MMVQ)
    std::map<std::string, WSlot> ws_map_;
    std::map<std::string, std::string> row_kind_;

    // ---- the layer split (speed): this instance owns layers [l0_, l1_) on device dev_.  When
    // split_next_ is set, step() runs this half, hands the residual over with one 64 KB host hop
    // (peer access is closed on this driver and not needed for a single boundary copy) and lets the
    // next half continue through its own layers (and, if it is the last, the head).
    int l0_ = 0, l1_ = 0;
    int dev_ = 0;
    int n_parts_ = 1;                                      // the devices the layers are split across (the CPU lane's share)
    int part_ = 0;                                         // ... and which of them this one is (0 = the first layers)
    int cap_index_ = -1;                                   // a split that fell back to one card: its place in the device list
    std::vector<int> split_devs_;                          // ... and the CUDA devices of all of them, part by part
    size_t vram_base_ = 0;                                 // bytes earlier parts held on this part's device when it
                                                           // started loading (STRATA_GLM_SPLIT_SAME: not counted
                                                           // against this part's VRAM cap)
    size_t vram_cap() const;                               // STRATA_GLM_VRAM_GB_PARTS[part_] or STRATA_GLM_VRAM_GB, bytes
    std::unique_ptr<Glm5Model> split_next_;
    // the hand-over to split_next_ (hop_link.hpp: STRATA_GLM_HOP picks how the rows move)
    std::unique_ptr<HopLink> hop_link_;
    // qwen4exp's splits: the next part as a PartPort (part_port.hpp) - this part never reaches into its state
    std::unique_ptr<PartPort> next_port_;
    std::vector<int32_t> win_tok_;   // (the last window's argmax per column, from the last part)

public:
    // ---- a later part's side of the PartPort requests (LocalPort calls them; a part server would too)
    bool part_boundary(std::string& err);
    void part_token_meta(int32_t token, bool image);
    bool part_token(HopLink* in, int64_t p, int64_t pos, std::string& err);
    int part_token_wait(std::string& err);
    int part_sample(strata::kernels::SamplerParams& sp, std::string& err);
    bool part_logits(std::vector<float>& out, std::string& err);
    bool part_prompt(HopLink* in, int slot, int64_t p0, int T, const int32_t* toks, const int32_t* next, bool chain,
                     std::string& err);
    bool part_window(HopLink* in, const std::vector<int32_t>& toks, int64_t p0, int64_t pos, std::string& err);
    bool part_window_wait(std::vector<int32_t>& argmax, std::string& err);
    int part_window_sample(int i, strata::kernels::SamplerParams& sp, std::string& err);
    bool part_window_logits(int i, std::vector<float>& out, std::string& err);
    void part_rewind(int64_t pos);
    bool part_mtp_on() const;
    bool part_mtp_commit(int32_t token, std::string& err);
    bool part_mtp_draft(int32_t y, int32_t* drafts, float* conf, std::string& err);
    bool part_route_ok(std::string& err);
    void part_collect();
    void part_stall_report(int64_t p);
    bool part_token_head(int64_t p, std::string& err);   // (the last part: the head and its argmax)
    std::vector<float> hop_;                               // the boundary residual staging buffer
    int* d_tok_ = nullptr;                                 // the sampler's one-int output, on dev_
    bool step_layers(int64_t p, std::string& err);
    // ---- the cross-token pipeline: with a split and more than one token to feed (a teacher-forced
    // prompt), the next token's first half is queued on device 0 BEFORE the host waits for this
    // token's residual, so half-A(t+1) runs while half-B(t) runs on device 1.
    void* stream_hop_ = nullptr;                           // the residual D2H stream on dev_
    void* ev_a_pipe_ = nullptr, *ev_h_pipe_ = nullptr;
    void* hop_side_ = nullptr;                             // 2x hop floats on dev_ (the end-copy slots)
    void* ev_h2d_[2] = {nullptr, nullptr};                 // dev1-side H2D completion, per host slot
    bool step_issue(int32_t token, int64_t& p, std::string& err);   // embed + this half's layers
    bool forward_pipeline(const std::vector<int32_t>& tokens, std::vector<float>& logits_out,
                          std::string& err);

    // ---- the pack (set by load_pack; empty when load()ing a plain F32 GGUF)
    bool pack_ = false;
    struct Shard {
        std::string path;
        uint8_t* base = nullptr;
        uint64_t size = 0, data_start = 0;
        int fd = -1;                                   // kept open for posix_fadvise reads (prefetch)
        int fd_direct = -1;                            // O_DIRECT: expert reads that bypass the page cache
        int fd_mirror = -1;                            // O_DIRECT: the same file on a 2nd SSD (STRATA_GLM_MIRROR)
        void* h_direct = nullptr;                      // Windows: the same, a FILE_FLAG_NO_BUFFERING HANDLE
    };
    std::vector<Shard> pack_shards_;
    struct NativeLayer {
        int layer = -1;
        strata::kernels::cpu::NativeFmt fmt;
        int gate_shard = 0, up_shard = 0, down_shard = 0;
        uint64_t gate_off = 0, up_off = 0, down_off = 0;   // absolute file offsets of the tensors
        // a fused ffn_gate_up_exps tensor (Inkling 2): expert e's gate rows at gate_off + e gu_stride, its up rows right
        // after them (up_off = gate_off + the gate rows' bytes, the same stride); 0 = separate gate / up tensors
        uint64_t gu_stride = 0;
        // EXL3 experts (exl3.txt, x_cb >= 0): expert e's part r (gate, up, down) piece j (trellis, suh, svh) lies at
        // x_off[9 e + 3 r + j] of shard x_shard[...]; the blob is [gate | up | down], a part [trellis | suh | svh]
        int x_cb = -1;
        int x_K[3] = {0, 0, 0}, x_k[3] = {0, 0, 0}, x_n[3] = {0, 0, 0};
        std::vector<int> x_shard;
        std::vector<uint64_t> x_off;
        size_t x_piece(int r, int j) const {
            return j == 0 ? (size_t) x_k[r] * (size_t) x_n[r] * (size_t) x_K[r] / 8
                          : 2 * (size_t) (j == 1 ? x_k[r] : x_n[r]);
        }
        size_t x_part(int r) const { return x_piece(r, 0) + x_piece(r, 1) + x_piece(r, 2); }
    };
    std::vector<NativeLayer> pack_layers_;                 // indexed by layer id (dense lead = null)
    // ---- an EXL3 pack (tools/exl3_pack.py): the trellis-coded matrices, read from the model's safetensors (shards
    // exl3_shard0_ on, after the GGUF), uploaded per piece; the WSlots of type exl3::kTypeEXL3 point at exl3_mats_
    bool exl3_ = false;
    int exl3_shard0_ = -1;
    std::map<std::string, strata::kernels::exl3::Mat> exl3_mats_;
    std::vector<float> host_x_, host_moe_, host_h_;        // per-token CPU MoE scratch
    std::vector<float> host_probs_b_;                      // the per-layer router bias (prefetch ranking)
    std::vector<float> host_ff_, host_dn_;                 // per-expert gate/up and down outputs
    std::vector<uint8_t> host_act_, host_hq_;              // quantized activations
    cudaError_t pack_moe_tail(std::string& err, const std::vector<int>& ids,
                              const std::vector<float>& rw, float* A, int layer);
    // token_embd stays QUANTIZED in the pack arena; step() dequantizes the one row it needs per
    // token from the host mapping (2.2 GB of F32 embedding the expert pool now gets instead).  A row's
    // bytes come from ggml_row_size, which knows every GGUF type: iq_row_bytes knows only the device
    // kernels' and gave 0 for F16 / BF16 / Q2_K / Q4_1 / Q5_1, so every token read row 0
    const uint8_t* pack_emb_src_ = nullptr;
    int pack_emb_type_ = -1;
    std::vector<uint8_t> pack_emb_copy_;                   // Windows: token_embd once the shard views are gone
    void pack_release_views();                             // Windows, fast path: unmap the shards after the load

    // ---- M2b: the device expert pool.  Slots of [gate|up|down] rows (the native blob layout),
    // filled on miss by streaming the rows straight out of the mmap'd shards; LFU-with-aging
    // residency when the working set exceeds VRAM.  SIZE CLASSES: one class per distinct
    // (gu_row, d_row) pair - uniform slots at the largest blob would pad ~20% of the arena
    // (8.78 MB max vs 7.03 MB average).  Class c's slots live at pool_ + class_off_[c] with
    // GLOBAL ids [class_base_[c], class_base_[c] + class_slots_[c]); eviction scans only its own
    // range.  Empty unless load_pack found headroom.
    bool dev_experts_ = false;
    uint8_t* pool_ = nullptr;                              // the compute device's arena
    size_t pool_bytes_ = 0, slot_bytes_ = 0;               // slot_bytes_ = the LARGEST class (staging sizes)
    int pool_nclasses_ = 0;
    size_t class_off_[8] = {};                             // byte offset of each class's sub-arena
    int64_t class_base_[8] = {}, class_slots_[8] = {};     // global slot-id range per class
    size_t class_stride_[8] = {};                          // bytes per slot of that class
    std::vector<int> layer_class_;                         // absolute layer -> class
    std::vector<int32_t> pool_slot_of_;                    // layer*n_expert+expert -> GLOBAL slot, -1 absent
    std::vector<int64_t> pool_key_of_;                     // global slot -> key, -1 free
    std::vector<uint64_t> pool_tick_;                      // recency per global slot
    std::vector<uint32_t> pool_count_;                     // per-slot use count (LFU with aging, see pool_evict)
    uint64_t pool_evictions_ = 0;
    uint64_t pool_clock_ = 0, pool_hits_ = 0, pool_misses_ = 0;
    // ---- STRATA_GLM_TIMING=1: cudaEvent phase accumulators (dense mixer / expert tail / rest),
    // printed once at destruction; pool_host accumulates the host-side miss staging (memcpy from
    // the mmap + issue), which shows up as stream idle inside the expert tail.
    bool timing_ = false;
    bool tail_rec_ = false;   // per layer: the MoE tail recorded b and c (dense layers do not)
    void *tev_a_ = nullptr, *tev_b_ = nullptr, *tev_c_ = nullptr, *tev_d_ = nullptr;
    double t_dense_ = 0, t_tail_ = 0, t_rest_ = 0, t_pool_host_ = 0;
    int64_t t_tokens_ = 0;
    uint8_t* dev_xq_ = nullptr;                            // one q8_1 activation row
    void* dev_scratch_ = nullptr;                          // native_expert_grouped scratch
    float* dev_hit_out_ = nullptr;                         // n_exp_used x n_embd expert outputs
    // the grouped plan, host-written and read by the kernels over UVA (host-mapped)
    struct DevPlan {
        unsigned long long ptr[8];
        int32_t start[9];
        int32_t n_groups[1];
        int32_t dst[8];
        int32_t tok[8];
    };
    DevPlan* plan_host_ = nullptr;
    uint8_t* plan_dev_ = nullptr;
    uint8_t* bounce_ = nullptr;                            // pinned staging ring for miss copies
    int bounce_i_ = 0;
    // ---- the miss staging pool (speed pass): one worker per in-flight miss reads the expert's three
    // ranges IN PARALLEL (the NVMe at multi-queue rate instead of one faulting memcpy on the main
    // thread), then the main thread DMAs each staged blob to its slot with a single pinned copy.
    // STRATA_GLM_NO_STAGE=1 keeps the serial pageable path (A/B).
    bool stage_on_ = false;
    struct StageJob {
        const uint8_t* src[3] = {nullptr, nullptr, nullptr};
        uint64_t len[3] = {0, 0, 0};
        uint8_t* buf = nullptr;
    };
    std::vector<StageJob> stage_q_;
    std::vector<std::thread> stage_workers_;
    std::mutex stage_mu_;
    std::condition_variable stage_cv_, stage_done_cv_;
    int stage_pending_ = 0;
    bool stage_quit_ = false;
    std::vector<uint8_t*> stage_bufs_;                     // one pinned buffer per job row (n_exp_used <= 8)
    uint8_t* staged_dst_[8] = {};                          // per-entry: where the DMA goes once staged
    uint64_t staged_len_[8] = {};
    void stage_worker();
    void stage_join();
    void pool_ensure(int layer, const int* ids, int k);
    int32_t pool_evict(int cls);
    bool pack_moe_tail_device(std::string& err, const std::vector<int>& ids,
                              const std::vector<float>& rw, float* A, int layer);

    // ---- the fast path (src/core/glm_fast_path.cu)
    struct FastState;
    FastState* fast_ = nullptr;
    bool fast_mode_ = false;                               // decided at load (STRATA_GLM_SLOW=1 turns it off)
    bool host_logits_ = true;
    int last_tok_ = -1;                                    // the greedy argmax of the last fast forward
    std::deque<int32_t> force_q_;                          // force_next() / force_seq()
    const std::vector<int64_t>* guard_ids_ = nullptr;      // end_guard()
    int32_t guard_tok_ = -1;
    bool guard_fired_ = false;
    int32_t end_guard_pick(int32_t sampled);
    std::map<std::string, const uint16_t*> w16_;           // the pack's big BF16 rows, kept BF16 (fast mode)
    std::map<std::string, const void*> w8_;                // glm5's Q8_0 rows kept Q8_0 (STRATA_GLM_DENSE_Q8, fast mode)
    std::map<std::string, const int*> wi32_;               // I32 tensors (deepseek4's hash-routing tables)
    // ---- deepseek4 (src/core/dsv4_fast_path.cu)
    void* dsv4_state_ = nullptr;                           // the window rings, compressed caches and compressor rings
    size_t dsv4_state_bytes_ = 0;
    // deepseek4: the raw window ring's rows - the window (n_swa) plus kSpecSlack, so a verify window's rejected tokens
    // (positions past the committed ones) never overwrite a key a later query still reads: rolling them back is then
    // only the position (the compressor rings and compressed caches are rewritten in order anyway)
    static constexpr int kSpecSlack = 4;
    int ds_ring_ = 0;
    bool dsv4_alloc(std::string& err);                     // in fast_setup: the caches for max_ctx_, carved per layer
    bool dsv4_bind(std::string& err);                      // in fast_setup: the per-layer weights
    bool fast_dsv4_attn(int il, int64_t p, std::string& err);   // one attention mixer (x -> mixer) at position p
    bool dsv4_head(std::string& err);                      // the hyper-connection head -> head_x / head_xq
    // ---- deepseek41 (src/core/dsv41_fast_path.cu): the token path, prompt chunks (their attention a token at a time),
    // snapshots and verify windows (dsv4_verify.cu, the batched kind); no layer splits
    void* dsv41_state_ = nullptr;                          // the window rings, the 4 KV slots and the scratch
    bool vl_bias_ok_ = false;                              // exp_probs_b_vl is on the device (dsv41_alloc)
    void* v41_eng_ = nullptr;                              // the host's Engram state (dsv41_fast_path.cu V41Engram)
    bool dsv41_bind(std::string& err);
    bool dsv41_alloc(std::string& err);
    bool fast_dsv41_attn(int il, int64_t p, std::string& err);
    bool dsv41_engram_token(int32_t tok, std::string& err);   // before the layers: hash, read and decode both tables'
                                                              // rows of this token, upload them
    // layer il's Engram add into the streams R (rows: that token's 24 x 256 rows on the device; null: the token path's)
    bool dsv41_engram_add(int il, float* R, std::string& err, const float* rows = nullptr, cudaStream_t s = nullptr);
    bool dsv41_engram_hash(const int32_t* toks, int64_t p0, int T, std::string& err,   // the prompt path: a chunk's
                           const int32_t* ahead = nullptr, int T_ahead = 0);   // (and the next chunk's, prefetched)
    bool dsv41_engram_prompt(int il, int T, float* R, std::string& err);   // row numbers, then each Engram layer's adds
    void dsv41_reset();                                    // a new sequence: the Engram history, the pending pairs
    void dsv41_engram_report() const;                      // STRATA_GLM_TIMING: the Engram rows and the cache's share
    bool dsv41_engram_ok(std::string& err);                // after a token: the async reader's failure, if any, is this token's
    bool dsv41_head(std::string& err);                     // the collapse with the last sublayer's pre -> head_x / xq
    // ---- the refusal-direction edit (src/core/glm_abl.cu; STRATA_GLM_ABL / --uncensor).  Each is a no-op without an
    // edit, past its layers or at a zero strength.  abl_apply: an ABL1 edit's sites - rows of a layer's attention
    // (which 0) or shared-expert (1) output, ld floats apart.  abl_out: an ABL2 edit's - the attention (0) or the
    // whole FFN (1: routed + shared + the CPU lane's) output.  abl_res: an ABL2 edit on rows going into the residual
    // whole (the embedding, a residual after Engram's add).  abl_cap: --abl-build's capture of the residual entering
    // layer `slot` (slot n_layers: the last layer's output), `streams` streams of n_embd averaged
    void abl_apply(int il, int which, float* x, int rows, int ld, cudaStream_t s);
    void abl_out(int il, int which, float* x, int rows, int ld, cudaStream_t s);
    void abl_res(float* x, int rows, int ld, cudaStream_t s);
    void abl_cap(int slot, const float* R, int streams, cudaStream_t s);
    bool abl_full() const;   // an ABL2 edit is loaded: the CPU lane's answer joins the FFN output before the read
    // ---- qwen35moe (src/core/qwen35_fast_path.cu): the token path (the prompt path, the verify windows and the
    // conversation snapshot are elsewhere; no layer splits yet)
    void* qwen35_state_ = nullptr;                         // the GDN states, conv states, KV caches and activations
    int q35_kv_ = 0;   // --kv (STRATA_GLM_KV_Q35): the attention layers' cache format, qwen35_fast.hpp's kv (0 FP16)
    // this GPU's attention caches (qwen35_alloc): their bytes, and what --kv int8 would take where it applies and the
    // caches are FP16 (else 0) - the start-up hint when FP16 KV pushes weights out of VRAM (q35_kv_hint)
    size_t q35_kv_b_ = 0, q35_kv8_b_ = 0;
    void q35_kv_hint(const char* what, double pushed_gb) const;
    // ---- elastic KV (src/core/glm_elastic.cu, --kv-elastic / --kv-min): the full-context caches on CUDA's virtual
    // memory API, mapped as positions are reached; the unmapped part lent to the expert pool's elastic ranks or the
    // dense FFN's elastic layers and taken back when the KV grows
    struct ElasticKv;
    ElasticKv* ekv_ = nullptr;
    static int64_t ekv_min_tokens();                       // --kv-min (STRATA_GLM_KV_MIN), 32768
    bool ekv_on() const;
    size_t ekv_lendable() const;                           // the reserved KV bytes not mapped (what the weights may take)
    size_t ekv_lent() const;                               // ... and what the weights hold of them
    bool ekv_kv_setup(const std::vector<size_t>& rows, std::vector<void*>& ptrs);   // qwen35_alloc: false = fixed
    bool ekv_kv_map(int64_t tokens, std::string& err);
    bool ekv_kv_all(std::string& err);                     // nothing borrows: map the KV whole
    bool ekv_pool_layout(std::vector<int>& slots, const std::vector<size_t>& stride, size_t guard, bool zero,
                         int cap_slots, std::vector<uint8_t*>& base, std::string& err);
    bool ekv_pool_ranks(int r, std::string& err);
    int ekv_pool_core(int il) const;                       // a layer's core slots under elastic ranks, else -1
    void ekv_lent_set(size_t bytes);
    void ekv_reset_mark();
    void ekv_destroy();
    bool q35d_elastic_shrink(size_t keep_bytes, std::string& err);   // the dense FFN's elastic layers down to keep_bytes
    bool q35d_elastic_grow(std::string& err);                        // ... and back up to what the budget leaves
    /// elastic KV: positions below `end` are about to be written - map them (the weights give back the room); a no-op
    /// when the caches are fixed
    bool kv_ensure(int64_t end, std::string& err);
    std::string ekv_status() const;                        // "" when fixed
    // the prompt path's lending generalised: every slot take(il, s) picks leaves the decode (its expert moves over a
    // colder resident, parks in the RAM tier or goes) and is left in `state` (kLent: the tail; kElastic: elastic KV)
    bool lend_slots(const std::function<bool(int, int)>& take, char state, uint64_t& moved, uint64_t& dropped,
                    std::string& err);
    // ---- hy4 (src/core/hy4_fast_path.cu): the token path; its weights, caches and activations live in hy4_
    void* hy4_ = nullptr;
    bool hy4_setup(std::string& err);                      // fast_setup: bind, allocate, open the Engram table
    bool hy4_embed(int32_t token, int64_t p, const float* img, std::string& err);   // the row -> state_'s 4 streams
    bool fast_hy4_layers(int64_t p, std::string& err);
    bool fast_hy4_moe(int il, std::string& err);           // x -> ffn (the latent MoE through the tiers)
    bool hy4_head(std::string& err);                       // the collapse + output_norm -> hy4_head_x / _xq
    const float* hy4_head_x() const;
    const void* hy4_head_xq() const;
    void hy4_reset();
    void hy4_destroy();                                    // fast_destroy: the arena, the buffers, the table's file
    bool hy4_snap_copy(bool restore);                      // the window rings + the open indexer groups <-> a copy
    // the batched prompt path's hy4 half (hy4_fast_path.cu): glm_prefill.cu's prefill_half runs the latent MoE through
    // its expert machinery between hy4_prompt_layer (the Engram adds, the attention, the FFN's read and the latent) and
    // hy4_prompt_ffn_out (routed_up, the shared expert, post_ffw_norm, the streams' add).  STRATA_HY4_PREFILL=0: none
    struct Hy4PromptIo {
        void* blas = nullptr;                              // the prompt path's cublasHandle_t (on the compute stream)
        uint16_t* w16 = nullptr;                           // the FP16 weight scratch
        int64_t w16_elems = 0;
        float* x = nullptr;                                // T x n_embd: a MoE layer's latent (the experts' input)
        uint16_t* x16 = nullptr;
        float* ffn = nullptr;                              // T x n_embd: the routed experts' sum
        uint8_t* keep = nullptr;                           // the chunk's rows across its layers (hy4_prompt_bytes)
        uint8_t* uni = nullptr;                            // a layer's scratch (the MoE's buffers between)
        size_t uni_bytes = 0;
        float* emb_h = nullptr;                            // pinned staging (emb_floats floats)
        size_t emb_floats = 0;
        // STRATA_GLM_PREFILL_PROF: the prompt path's phase marks (an event on the stream, name = the phase just ended)
        void (*mark)(void* ctx, const char* name, void* stream) = nullptr;
        void* mark_ctx = nullptr;
    };
    bool hy4_prompt_ok(std::string& why) const;
    /// a chunk of T tokens: {the rows kept across its layers, a layer's scratch} in bytes
    std::pair<size_t, size_t> hy4_prompt_bytes(size_t T) const;
    bool hy4_prompt_embed(const int32_t* toks, int64_t p0, int T, const Hy4PromptIo& io, std::string& err);
    bool hy4_prompt_layer(int il, int64_t p0, int T, bool skip_output, const Hy4PromptIo& io, std::string& err);
    bool hy4_prompt_router(int il, int T, float* logits, const Hy4PromptIo& io, std::string& err);
    bool hy4_prompt_shared(int il, int T, float* sg, float* su, uint16_t* sh16, int sub, const Hy4PromptIo& io,
                           std::string& err);
    bool hy4_prompt_ffn_out(int il, int T, const Hy4PromptIo& io, std::string& err);
    // the batched prompt path's inkling half (inkling_fast_path.cu): prefill_half runs the 48 layer applications in a
    // token's order (ink_app_phys), each through ink_prompt_layer (the depth mixes, the controllers, the attention, the
    // FFN's read into io.x / io.x16), the MoE of prefill_half's machinery, then ink_prompt_ffn_out (the FFN's post conv
    // and add; the last application also the head's mix rows).  The prompt's KV, conv states, slots are the token path's
    bool ink_prompt_ok(std::string& why) const;
    std::pair<size_t, size_t> ink_prompt_bytes(size_t T) const;
    bool ink_prompt_embed(int T, const Hy4PromptIo& io, std::string& err);    // io.x (the embedding rows) -> h
    bool ink_prompt_layer(int ia, int64_t p0, int T, const Hy4PromptIo& io, std::string& err);
    bool ink_prompt_ffn_out(int ia, int T, const Hy4PromptIo& io, std::string& err);
    const float* ink_prompt_final() const;                                      // the last chunk's head rows (T x E)
    void qwen35_bind(int il, std::string& missing);        // fast_setup's layer loop: this layer's weights
    bool qwen35_alloc(std::string& err);
    void qwen35_reset();                                   // a new sequence: the states zeroed (on the fast stream)
    bool fast_qwen35_layers(int64_t p, std::string& err);  // fast_layers for this arch, head_x / head_xq at the end
    bool fast_qwen35_gdn(int il, std::string& err);        // x / xq -> mixer
    bool fast_qwen35_attn(int il, int64_t p, std::string& err);
    bool fast_laguna_attn(int il, int64_t p, std::string& err);   // laguna's attention layer (x / xq -> mixer)
    // ---- inkling (inkling_fast_path.cu): the physical layer's weights (from qwen35_bind), the slots and scratch (from
    // qwen35_alloc, inside qwen35_state_), the token path's 48 layer applications and the head's mix (head_x / head_xq),
    // a new sequence's zeroed conv states, the snapshot's runs (conv states, window rings)
    void ink_bind(int il, std::string& missing);
    bool ink_alloc(size_t& bytes, bool place, uint8_t*& at, std::string& err);
    bool fast_ink_layers(int64_t p, std::string& err);
    void ink_reset();
    void ink_snap_runs(std::vector<std::pair<void*, size_t>>& out) const;
    void ink_free();
    // inkling's verify window (STRATA_INK_DRAFT=1, opt-in): the token path's 48 layer applications over T columns at
    // once - every column the bits a token alone gets (glm_mvn, rms_q8_rows, the attention's splits a row as the
    // token's, the conv states read but left: the window's inputs kept a slot, ink_win_commit moves the states past
    // the kept columns at spec_rewind).  io: the window's buffers (dsv4_verify.cu's) - emb the columns' embedding
    // rows, x / xq the MoE's input and ffn its output (verify_moe), head_x / head_xq the head's input (out)
    struct InkWinIo {
        const float* emb = nullptr;
        float *x = nullptr, *ffn = nullptr, *head_x = nullptr;
        void *xq = nullptr, *head_xq = nullptr;
        std::function<bool(std::string&)> tail;   // (the head's launches after the layers: in the window's graph too)
    };
    static bool ink_draft_env();   // STRATA_INK_DRAFT (default 0)
    bool ink_window(const std::vector<int32_t>& toks, int64_t p0, const InkWinIo& io, std::string& err);
    void ink_win_commit(int64_t pos);   // (no window pending: nothing)
    // a CUDA graph was captured since the last call (inkling's token / window graphs): the step's time is not the
    // step's cost - the callers' cost tables leave it out
    bool graph_captured() {
        const bool c = graph_cap_;
        graph_cap_ = false;
        return c;
    }
    bool graph_cap_ = false;
    int64_t ink_win_p0_ = -1;           // the pending window's first position and width
    int ink_win_T_ = 0;
    // a verify window's columns (toks[0..T) at p0..): their Engram rows read and uploaded before the layers (the n-gram
    // history advances over them), then column t's add at layer il into its streams R
    bool dsv41_engram_window(const int32_t* toks, int64_t p0, int T, std::string& err);
    bool dsv41_engram_window_add(int il, int t, float* R, std::string& err, cudaStream_t s = nullptr);
    void* vb_ = nullptr;                                   // the verify window's buffers (dsv4_verify.cu)
    bool verify_alloc(std::string& err);
    // columns tb .. tb + n - 1 on stream s (the batched window: all of them on the compute stream; the pipelined one:
    // a token on its own stream; verify_moe's route waits for wait_ev and records done_ev when they are set)
    bool verify_attn(int il, int64_t p0, int tb, int n, cudaStream_t s, std::string& err);
    bool verify_attn_v41(int il, int64_t p0, int tb, int n, cudaStream_t s, std::string& err);   // deepseek41's
    void fast_dump_feat(int il, const float* R);   // STRATA_DS4_DUMP_FEAT (the DSpark probe)
    // ---- DSpark, DeepSeek V4 Flash's block drafter (src/core/dspark.cu, STRATA_DS4_DSPARK=<its GGUF>)
    void* dsp_ = nullptr;
    bool dspark_load(std::string& err);            // in fast_setup, before the expert pool takes the VRAM left
    // DeepSeek V4 0731 without STRATA_DS4_DSPARK: the drafter GGUF where setup puts it (<models>/bleysg/...), found
    // from the pack's folders up; "" when there is none.  dsp_auto_: the drafter was loaded that way (only when the
    // VRAM pool beside it still holds every expert: dspark_auto_keep)
    std::string dspark_find() const;
    bool dsp_auto_ = false;
    bool dspark_auto_keep(const void* fast_state, double drafter_bytes) const;   // (glm_fast_path.cu)
    void dspark_destroy();
    size_t dspark_vram_bytes() const;              // the drafter's device memory (the RAM tier is that much smaller)
    void dspark_reset();                           // a new sequence / a restored snapshot: an empty context
    // a target layer's output streams R (4 x n_embd) -> the feature of column col (the token path's: 0); the prompt
    // path's chunk of T rows -> its last rows'
    void dspark_capture(int il_out, const float* R, int col, cudaStream_t s);
    void dspark_capture_rows(int il_out, const float* R, int T, cudaStream_t s);
    // positions p0 .. p0 + n - 1 into the context rings, from the columns' features (prompt: the prompt rows')
    bool dspark_commit(int64_t p0, int n, bool prompt, cudaStream_t s);
    void dspark_window(int64_t p0, int T);         // verify: its columns wait for spec_rewind's position
    void dspark_settle();                          // ... which commits the kept ones
    // ---- DFlash 2, Qwen3.8-Flash-Next's block drafter (src/core/dflash2.cu, STRATA_Q4X_DFLASH=<its GGUF>; the dense
    // qwen35's - Qwen3.8-27B - with STRATA_Q35_DFLASH): the same
    // hooks as DSpark's (dspark_settle / dspark_window / dspark_reset / dspark_draft call these when it is loaded)
    void* dfl_ = nullptr;
    bool dflash_load(std::string& err);
    void dflash_destroy();
    void dflash_reset();
    void dflash_forget();                          // (a slot of another conversation: no context)
    void dflash_report() const;
    // the residual entering target layer il (hc streams of n_embd, rows rs floats apart) -> the features of n rows:
    // the token path's / a window's columns 0..n-1 (prompt false), or a prompt chunk's last rows (prompt true, R the
    // chunk's T = n rows: its last 2048 kept)
    void dflash_capture(int il, const float* R, int n, size_t rs, bool prompt, cudaStream_t s);
    bool dflash_commit(int64_t p0, int n, bool prompt, cudaStream_t s);
    bool dflash_flush(cudaStream_t s);
    void dflash_settle();
    void dflash_window(int64_t p0, int T);
    bool dflash_draft(int32_t y, int32_t* drafts, float* conf, std::string& err);
    // ---- Inkling 2 Mini's DSpark drafter (src/core/ink_dspark.cu, STRATA_INK_DSPARK=<its GGUF>, arch dflash): the
    // same hooks (dspark_settle / dspark_window / dspark_reset / dspark_draft call these when it is loaded).  Its
    // context: the residual entering layer applications 5 / 23 / 41 and the head's normed row (visit 48) of every
    // position - ink_tap from the token path, the verify window and the prompt path, inkd_commit at the position
    void* inkd_ = nullptr;
    bool inkd_load(std::string& err);
    void inkd_destroy();
    void inkd_reset();
    void inkd_report() const;
    // rows of tap `visit` (5, 23, 41 the residual entering that application, 48 the head's row - normed with norm_w
    // when not null): n rows ld floats apart, the token path's / a window's columns 0..n-1 or a prompt chunk's T = n
    // (pend: a single token's tap - its row the deferred rows' next, from the device word inkd_tok_begin sets)
    void ink_tap(int visit, const float* x, int64_t ld, int n, const float* norm_w, bool prompt, cudaStream_t s,
                 bool pend = false);
    // the single tokens' rows deferred (STRATA_INK_DSPARK_DEFER): inkd_tok_begin before the token's taps, inkd_tok_end
    // after them (the old per-token commit when off); inkd_flush commits the deferred rows (a draft, a window, a prompt)
    void inkd_tok_begin(int64_t p, const int64_t* pos_d);   // (pos_d: the token graph's position word, else null)
    bool inkd_tok_end(int64_t p, cudaStream_t s);
    bool inkd_flush();
    bool inkd_commit(int64_t p0, int n, bool prompt, cudaStream_t s);
    void inkd_settle(int64_t pos = -1);   // (the kept columns: up to pos, default pos_)
    void inkd_window(int64_t p0, int T);
    uint64_t inkd_acc_gen() const;   // (the taps' accumulators' generation: the token graphs hold their address)
    void inkd_tok_taps(int n = 1);   // (a replayed token / window graph of n rows: the taps' row counts its capture set)
    bool inkd_draft(int32_t y, int32_t* drafts, float* conf, std::string& err);
    bool ink_dspark_on() const { return inkd_ != nullptr; }
    int inkd_max() const;   // (drafts a step at most: 7, STRATA_INK_DSPARK_N)
    bool verify_moe(int il, int tb, int n, const std::vector<int32_t>& toks, cudaStream_t s, cudaEvent_t wait_ev,
                    cudaEvent_t done_ev, std::string& err, bool router_done = false, bool graphed = false);
    bool verify_moe_router(int il, int tb, int n, cudaStream_t s, std::string& err);   // (its first launch)
    static bool verify_sh_late();
    void verify_report() const;   // (the q4x window's graph use, at the end)
    // one part's share of a window (verify): its layers on the columns' residuals in its own buffers, then, a part
    // before the last, the columns' residuals into the next part's (the host hop), else the head, the logits and the
    // argmax; the last part records its buffers' done event
    bool verify_part(const std::vector<int32_t>& toks, int64_t p0, std::string& err);
    bool verify_part_v41(const std::vector<int32_t>& toks, int64_t p0, std::string& err);   // deepseek41's (one part)
    bool verify_part_q35(const std::vector<int32_t>& toks, int64_t p0, std::string& err);   // qwen35moe's (one part)
    bool verify_part_ink(const std::vector<int32_t>& toks, int64_t p0, std::string& err);   // inkling's (one part)
    void verify_rewind_q35(int64_t pos);   // qwen35moe: the recurrent states back to after the kept columns
    // glm5-next (GLM-5.3-Flash) on one part, STRATA_GLM_WINDOW=1 (dsv4_verify.cu): the token path's layers over the
    // columns - the dense GEMVs one mvn_mv launch (mv()'s bits), the KDA columns through the token path's own
    // kda_prep / kda_rec in order (the state after column 0 kept: verify_rewind_glm restores it and replays the kept
    // columns from the window's saved inputs), the DSA rows written in position order before the reads
    static bool glm_window_env();
    static bool glm_window_nextn_env();   // (the window's drafter is the NextN block: STRATA_GLM_WINDOW=1, unified memory)
    bool glm_window() const { return g_.arch == 0 && !g_.v41 && glm_window_env(); }
    bool verify_part_glm(const std::vector<int32_t>& toks, int64_t p0, std::string& err);
    void verify_rewind_glm(int64_t pos);
    bool glm_mvn(const void* jobs, int n, int T, cudaStream_t s);   // gf::MvJob[n] (STRATA_GLM_MVN_CHECK)
    // qwen4exp's verify-window GEMVs: glm_mvn (each column mv()'s sums: the drafts' tokens equal the token path's)
    // or gf::mvn (the weights read once for all columns) - STRATA_Q4X_VERIFY_EXACT, on by default on CUDA, off on HIP
    // (HIP's glm_mvn is a mv() a column: the dense weights read once per column, issue #42)
    bool q4x_mvn(const void* jobs, int n, int T, cudaStream_t s);
    // deepseek4 / deepseek41's verify-window GEMVs: glm_mvn unless STRATA_DS4_VERIFY_EXACT=0 (as q4x_mvn: default on
    // CUDA, off on HIP; STRATA_DETERMINISTIC sets 1) - gf::mvn's one warp a row against the token path's split Q6_K
    // rows made 3-column windows part from the plain answer (V4 0731 at token 16, 2026-11-14)
    bool ds_mvn(const void* jobs, int n, int T, cudaStream_t s);
    // the dense qwen35's (Qwen3.8-27B) verify-window GEMVs: glm_mvn unless STRATA_Q35_VERIFY_EXACT=0 (as q4x_mvn:
    // default on with CUDA, off with HIP); the MoE qwen35 (Ornith) and Laguna keep gf::mvn
    bool q35_mvn(const void* jobs, int n, int T, cudaStream_t s);
    bool glm_window_check(const std::vector<int32_t>& toks, int64_t p0, std::string& err);   // STRATA_GLM_WINDOW_CHECK
    void kda_copy_layer(int il, int li, bool restore, float* bak = nullptr);   // one KDA layer's S + conv <-> kda_bak_
    // the NextN block (blk.<n_layers>) as the drafter on one GPU (qwen35_fast_path.cu, behind the q35_mtp_* hooks)
    bool glm_mtp_rows(int64_t p0, int n, const float* h_rows, size_t h_stride, const int32_t* next, std::string& err);
    bool glm_mtp_step(int64_t p, const int* tok_dev, const float* h, float* conf_out, int* tok_out, std::string& err);
    float *glm_mt_kv_ = nullptr, *glm_mt_ik_ = nullptr, *glm_mt_ig_ = nullptr;
    int* glm_mt_ids_ = nullptr;
    int* glm_ids_h_ = nullptr;                             // (pinned: the rows' next tokens, staged)
    cudaEvent_t glm_ids_ev_ = nullptr;
    bool verify_part_q4x(const std::vector<int32_t>& toks, int64_t p0, std::string& err);   // qwen4exp's (one part)
    bool q4x_mix_cols(float* R, int T, const float* norm, const WSlot& down, const WSlot& up, const WSlot* inject,
                      const float* out, const float* inj_prev, float* inj_out, float* x, void* xq, std::string& err);
    // qwen35moe's NextN block as the drafter (behind the dspark_* hooks; qwen35_fast_path.cu): a position's cache row
    // needs its normed hidden and the NEXT token, so a position waits in q35_pend_pos_ until that token is known
    bool q35_mtp_on() const;
    bool q35_mtp_pend_commit(int32_t next, std::string& err);
    void q35_mtp_pend_set(int64_t p, const float* h);
    bool q35_mtp_rows(int64_t p0, int n, const float* h_rows, size_t h_stride, const int32_t* next, std::string& err);
    bool q35_mtp_step(int64_t p, const int* tok_dev, const float* h, float* conf_out, int* tok_out, std::string& err);
    bool q35_mtp_draft(int32_t y, int32_t* drafts, float* conf, std::string& err);
    int64_t q35_pend_pos_ = -1, q35_row_done_ = -1;
    // speculative sampling's state (spec_sampling_set): the request's sampler, the drafts' and the window's kept
    // lists (device, then pinned host copies; kDistMax a row)
    strata::kernels::SamplerParams ss_sp_{};
    bool ss_on_ = false, ss_drafted_ = false, ss_win_ok_ = false;
    int ss_k_ = 0;
    uint64_t ss_win_c0_ = 0, ss_win_gen_ = 0;
    int* ss_q_ids_d_ = nullptr;
    float* ss_q_p_d_ = nullptr;
    int* ss_q_n_d_ = nullptr;
    int* ss_q_ids_h_ = nullptr;
    float* ss_q_p_h_ = nullptr;
    int* ss_q_n_h_ = nullptr;
    int* ss_p_ids_d_ = nullptr;
    float* ss_p_p_d_ = nullptr;
    int* ss_p_n_d_ = nullptr;
    int* ss_p_ids_h_ = nullptr;
    float* ss_p_p_h_ = nullptr;
    int* ss_p_n_h_ = nullptr;
    bool ss_alloc();
    // qwen4exp's draft layer (q4x_drafter.cu, STRATA_Q4X_MTP=<tools/mtp_rt.py's dir>): the same pending-row scheme, the
    // rows' input the trunk's final residual (4 streams) instead of a normed hidden
    void* q4d_ = nullptr;
    bool q4x_mtp_load(const std::string& dir, std::string& err);
    bool q4x_mtp_cache(uint16_t** kc, uint16_t** vc, size_t* row_bytes) const;   // (its KV caches: the slots)
    bool q4x_mtp_forward(int64_t p0, int n, const float* Rin, size_t r_stride, const int* tok, bool full, float* conf_out,
                         int* tok_out, std::string& err);
    bool q4x_mtp_rows(int64_t p0, int n, const float* Rin, size_t r_stride, const int32_t* next, std::string& err);
    const float* q4x_mtp_residual() const;
    bool q4x_mtp_on() const { return g_.q4x && chain_mtp_on(); }
    // the drafter here or on a later part (qwen4exp's draft layer sits on a split's last part: it reads the final
    // residual and the head)
    bool chain_mtp_on() const { return q35_mtp_on() || (next_port_ != nullptr && next_port_->mtp_on()); }
    // the conversation snapshot's qwen35moe state (pinned host memory, not VRAM): each delta-net layer's state and conv
    // rows, the drafter's pending hidden and its two positions - the KV caches are written a row a position, so the
    // position alone restores them
    void* q35_snap_h_ = nullptr;
    std::string ple_gguf_;                                 // qwen4exp: the shard holding per_layer_token_embd
    bool q4x_ple_setup(std::string& err);
    bool q4x_ple_issue(int32_t token, int64_t p, std::string& err);
    void q4x_ple_spec(int32_t token, int64_t p);
    bool q4x_ple_apply(std::string& err);
    bool q4x_qsa_setup(std::string& err);
    // Qwen3.8-27B QSA (qsa27.hpp): the block keys' buffers; false only on an allocation failure.  On, it sets
    // FastState::q27 (off: --kv quantized caches - the selection's attention reads FP16 rows)
    bool q27_qsa_setup(std::string& err);
    /// A Qwen3.8-27B QSA attention layer's selected attention for rows p0 .. p0 + n - 1 (q [n][n_head][head_dim], the
    /// indexer queries iq [n][4][128] raw - normed and rotated here at ipos [n][4], steps [n][4] qsa27::step_fill's
    /// records, device), every row past the dense window: their scores, top-k, own block and the attention into ao,
    /// kQ4Grp rows a pass.  The keys must be appended already.
    bool q27_attn_rows(int il, int64_t p0, int n, const float* q, float* iq, const int32_t* ipos, const int32_t* steps,
                       float* ao, cudaStream_t s, std::string& err);
    int64_t q35_snap_pend_ = -1, q35_snap_done_ = -1;
    bool q35_snap_copy(bool restore);
    size_t q35_snap_bytes() const;                         // (q35_snap_h_'s size)
    // the device runs a qwen35 slot of n positions holds after its header and the snapshot, in file order
    void q35_slot_runs(int64_t n, std::vector<std::pair<void*, size_t>>& out) const;
    bool q4x_mix(const float* norm, const WSlot& down, const WSlot& up, const WSlot* inject, const float* out,
                 const float* inj_prev, float* inj_out, float* x, void* xq, std::string& err);
    bool q4x_mix5(const float* norm, const WSlot& down, const WSlot& up, const WSlot* inject, const float* out,
                  const float* inj_prev, float* inj_out, float* x, void* xq, std::string& err);
    bool fast_q4x_layers(int64_t p, std::string& err);
    bool fast_loop_layers(int64_t p, std::string& err);   // sensenova_loop (qwen35_fast_path.cu)
    const Glm5Model* verify_tail() const;
    // the conversation snapshot's deepseek4 state: every layer's window ring and compressor rings (the compressed
    // caches are append-only by position - a row is written exactly when it becomes visible - so the position alone
    // restores them, as glm5-next's DSA caches)
    void* dsv4_snap_ = nullptr;
    bool dsv4_snap_copy(bool restore);
    // the prompt path's attention mixer over a sub-batch of tn tokens at positions p0.. (glm_prefill.cu)
    bool prefill_ds4_mixer(int il, int64_t p0, int tn, const uint16_t* x16, const float* xs, float* mixer,
                           bool skip_output, void* uni, std::string& err);
    // deepseek41's: the projections batched, the caches' order a token at a time (glm_prefill.cu)
    bool prefill_v41_mixer(int il, int64_t p0, int tn, int row0, const uint16_t* x16, const float* xs, float* mixer,
                           bool skip_output, void* uni, std::string& err);
    // ... and one Engram block's adds as a GEMM (false: the token path's GEMV a token at a time)
    bool prefill_v41_engram_block(int il, int bn, int kd, const float* rows, float* R, std::string& err);
    bool prefill_v41_engram_gemm() const;
    bool fast_setup(std::string& err);
    void fast_destroy();
    bool forward_fast(const std::vector<int32_t>& tokens, std::vector<float>& logits_out, std::string& err);
    bool fast_token(int32_t token, std::string& err);
    void fast_gap_mark(std::chrono::steady_clock::time_point tw);   // STRATA_GLM_GAP_PROF: a token is ready
    bool fast_layers(int64_t p, bool hop_in, std::string& err);
    bool fast_dsa(int il, int64_t p, std::string& err);     // one DSA mixer (x -> mixer)
    bool fast_moe(int il, bool& pf_pending, std::string& err);   // one MoE FFN (x -> ffn)
    void fast_service();
    void fast_tail_skip_setup();   // STRATA_GLM_TAIL_SKIP: the rank the trunk's routes leave missing experts out from
    int fast_skip_from(int il) const;   // that rank for layer il's route (the draft block's own, 8 = none)
    float fast_skip_w(int il) const;    // STRATA_GLM_TAIL_SKIP_W: the weight share below which it leaves them out
    // gate / up / down of one expert from the shards (or its chunk-th of n_chunks pieces)
    void fast_read_part(int il, int e, int role, uint8_t* blob, int chunk = 0, int n_chunks = 1);
    void fast_read_expert_x3(int il, int e, uint8_t* blob);   // a whole EXL3 expert in as few reads as it allows
    void fast_ahead_route(int il, const int* ids, unsigned int miss, const short (*ahead)[16]);   // [glmf::kMaxK]   // LOOKAHEAD
    void fast_ahead_reader();
    bool fast_boundary(std::string& err, bool early = false);   // between tokens: apply finished promotions
    int fast_sample(strata::kernels::SamplerParams& sp, std::string& err);   // this half's logits, sampled on its stream
    int64_t ram_budget_ = -1;                              // bytes of pinned RAM tier for this half (-1: derive)
    float* snap_ = nullptr;                                // the saved KDA states + conv histories (this half)
    // the DSA indexer's key / gate caches (ik, ig) are read only to pool each kpool-position block, so the fast path
    // keeps a RING of ik_ring_ positions (a prompt sub-batch, <= 8192, + one open pool; the NextN block's cache fill goes
    // in pieces of the ring) - 8 KB/position/layer, 1 GB per card saved at 128K; a snapshot keeps the open pool's rows
    // (snap_pool_: per DSA layer kpool-1 ik rows, then kpool-1 ig rows), since later positions overwrite their slots
    int ik_ring_ = 0;
    // the fast path's latent cache in INT8 records (--kv int8 / STRATA_GLM_KV_INT8=1; glm_fast.hpp lat8_rec_bytes):
    // 544 bytes a position and layer against FP16's 1024
    bool lat_q8_ = false;
    float* snap_pool_ = nullptr;
    std::vector<int> dsa_layers() const;                   // this half's DSA layers (NextN block first), slot order
    void snap_pool_copy(bool restore);
    int64_t snap_pos_ = -1;
    std::string pack_dir_;
    bool fast_warm(std::string& err);                      // load-time: stream every expert into the tiers
    bool fast_cpu_lane_setup(std::string& err);            // STRATA_GLM_CPU_LANE: the pool, its calibration, the split
    // gs (NVFP4 layers): each expert's gate/up second-level scale (its down scale is folded into w by the caller)
    // fx (the exact combine, gf::CpuAnswer::fx): the experts' weighted rows summed in fixed point instead of out; true
    // when it was written (the device's arithmetic, lane_q81 - not EXL3, not the iGPU lane)
    bool fast_cpu_experts(int il, int ne, const uint8_t* const* blob, const float* w, const float* x, float* out,
                          const void* xq = nullptr, const float* gs = nullptr, long long* fx = nullptr);
    // a batched verify window's n columns of layer il at once on the CPU threads: column t's ne[t] experts at
    // blob / w [8 t + i], its x[t], its sum to out[t] - an expert several columns route decoded once for them
    // (fx[t]: as fast_cpu_experts' - true when every column's was written)
    bool fast_cpu_experts_multi(int il, int n, const int* ne, const uint8_t* const* blob, const float* w,
                                const float* const* x, float* const* out, const void* const* xq = nullptr,
                                const float* gs = nullptr, long long* const* fx = nullptr);   // (gs at [8 t + i] as w)
    void fast_igpu_setup();   // the iGPU lane (STRATA_GLM_IGPU, src/igpu)
    // ---- the NextN (MTP) draft block (the last half carries it; layer index n_layers)
    int mtp_il_ = -1;                                      // its layer index here, -1: not on this half
    int mtp_src_ = -1;                                     // the shard of STRATA_GLM_MTP_GGUF's block, -1: the model's
    int lt_ = 0;                                           // the tier layers: [l0_, lt_) = the trunk's + the draft's
    void* mtp_arena_ = nullptr;
    bool load_mtp(const std::vector<std::unique_ptr<strata::GgufFile>>& gfs, std::string& err);
    bool fast_mtp(int64_t p, int32_t next_tok, std::string& err);   // the draft block at position p -> mtp_tok
public:
    /// The NextN draft for the token after next: after forward() of position p, given the token chosen for p + 1,
    /// returns the block's greedy proposal for p + 2 (-1: no draft block).  Fills the block's caches at p.
    int mtp_draft(int32_t next_tok, std::string& err);
    bool has_mtp() const;
    bool kv_int8() const { return lat_q8_; }   // the latent cache in INT8 (STRATA_GLM_KV_INT8=1)
    /// Images (the vision path): rows of n_embd floats that stand in for the token embeddings at these absolute
    /// positions - the prompt's <|image|> tokens, in order; an empty call clears them.  Read wherever a token is
    /// embedded (the prompt path and the token path); the draft block keeps the token embedding (drafts only).
    void set_image_rows(std::vector<int64_t> positions, std::vector<float> rows) {
        img_pos_ = std::move(positions);
        img_rows_ = std::move(rows);
    }
    /// Is position p an image row (V4.1: its router bias is exp_probs_b_vl and it gets no Engram)?  False at once when
    /// the request has no image.
    bool is_image_pos(int64_t p) const { return !img_pos_.empty() && image_row(p) != nullptr; }
    /// qwen35 / qwen35moe images: the cells' multimodal rotary positions (Qwen's interleaved M-RoPE, sections 11/11/10:
    /// strata/kernels/mrope.hpp), `tab` = cells x (t, h, w) from cell 0 - an image's cells share t and take h = t + row,
    /// w = t + column, the text after it continues at t + max(rows, columns).  Every device of a layer split gets it; the
    /// attention layers' rope kernels read it while it is set.  An empty `tab` clears it (each kernel takes the cell as
    /// its position again: text requests compute exactly as without images).  Between requests only.
    bool set_mrope(const std::vector<int32_t>& tab, std::string& err);
    bool images_ok(std::string& why) const;   // can this model take image rows (GENI)?
    bool has_image_rows() const { return !img_pos_.empty(); }
    /// deepseek41: exp_probs_b_vl is loaded (STRATA_GLM_VL_BIAS, or <pack>/exp_probs_b_vl.f32); an image request needs it
    bool has_vl_bias() const { return vl_bias_ok_; }
    const float* image_row(int64_t p) const {
        const auto it = std::lower_bound(img_pos_.begin(), img_pos_.end(), p);
        if (it == img_pos_.end() || *it != p) return nullptr;
        // (hy4: rows of the residual width; its n_embd is the experts' latent)
        return img_rows_.data() + (size_t) (it - img_pos_.begin()) * (size_t) (g_.is_hy4() ? g_.hy_dim : g_.n_embd);
    }
private:
    std::vector<int64_t> img_pos_;                         // ascending
    std::vector<float> img_rows_;
    int32_t* mrope_d_ = nullptr;                           // set_mrope's device table (this half's device)
    std::vector<int32_t> mrope_h_;                         // hy4: set_mrope's table on the host (cells x (t, h, w))
    size_t mrope_cap_ = 0;                                 // ... its int32 count
public:
    /// The pipelined speculative decode (src/core/glm_fast_path.cu): needs a two-half split whose tail carries the
    /// NextN block.  After the prompt's last forward(), emits up to max_new tokens (emit returns false to stop) - the
    /// same tokens the token-at-a-time loop would produce for the same samples.  The sequence state is left for the
    /// next request to restore (snapshot) or reset.
    bool spec_ready() const;
    /// deepseek4's verify window (src/core/dsv4_verify.cu): `toks` (1..kVerifyMax) through every layer at once from
    /// position(), each token's logits kept; position() advances by toks.size().  By default the tokens run staggered,
    /// a CUDA stream each (a token's attention of a layer overlaps the previous token's RAM-tier experts of it);
    /// STRATA_DS4_SPEC_PIPE=0 batches them on one stream (the dense GEMVs read once, every route issued before the
    /// first answer is awaited).  verify_argmax(i): the greedy token after toks[i];
    /// verify_sample(i): the sampler on those logits.  spec_rewind(pos): drop the window's tail (positions >= pos):
    /// only the position moves (kSpecSlack keeps the window ring's older keys).  A layer split runs the window part
    /// after part (each part's columns hop to the next through pinned host memory, as a token's residual does).
    static constexpr int kVerifyMax = 8;
    bool verify_ready() const;
    /// The window's buffers allocated now (every part), before a decode chooses the speculative loop: false when they
    /// do not fit the VRAM left (the decode then goes a token at a time instead of failing at its first window)
    bool verify_warm(std::string& err);
    /// The refusal-direction edit (src/core/glm_abl.cu) - uncensoring at run time, on every write into the residual
    /// (or, an ABL1 file, the attention / shared-expert outputs it names).  abl_load: an ABL1 / ABL2 file (--uncensor;
    /// STRATA_GLM_ABL); abl_set: an ABL2 edit from memory (--abl-build tries its candidates); abl_clear drops it.
    /// abl_enable: the per-request switch (on after a load).  abl_capture: --abl-build's residuals - the next
    /// forward() of one token records the residual entering every layer and the last one's output, read with
    /// abl_captured ([n_layers + 1][n_embd]).  A layer split: every part gets the edit; the capture is single-part.
    bool abl_load(const std::string& path, std::string& err);
    bool abl_set(const std::vector<float>& q, int rank, const std::vector<float>& strengths, float emb, std::string& err);
    void abl_clear();
    bool abl_loaded() const;
    void abl_enable(bool on);
    bool abl_capture(bool on, std::string& err);
    bool abl_captured(std::vector<float>& out);
    bool verify(const std::vector<int32_t>& toks, std::string& err);
    int verify_argmax(int i) const;
    int verify_sample(int i, strata::kernels::SamplerParams& sp, std::string& err);
    bool verify_logits(int i, std::vector<float>& out, std::string& err);   // column i's logits to the host
    void spec_rewind(int64_t pos);                         // (DSpark: the kept columns into its context)
    /// DSpark (src/core/dspark.cu): loaded with STRATA_DS4_DSPARK=<the drafter GGUF>; its context follows every
    /// token, window (the kept columns) and prompt chunk by itself.  dspark_draft(y): the block at position() with y,
    /// the pending token -> 5 drafts and their confidences (host; waits for the device).  dspark_upload: the expert
    /// cache's copies the last draft planned - after the window is queued, so they run under it.
    bool dspark_ready() const { return dsp_ != nullptr || dfl_ != nullptr || inkd_ != nullptr || chain_mtp_on(); }
    bool dflash_on() const { return dfl_ != nullptr; }   // DFlash2 (dflash2.cu): its walk applied the threshold
    // the dense qwen35 with both drafters (DFlash 2 and its NextN block): dspark_draft runs DFlash 2 while this is set,
    // the NextN chain otherwise (the decode picks a step's drafter by their measured rates)
    bool dfl_pick_ = true;
    bool dfl_thin_ = false;   // (the decode: DFlash 2 not drafting for a while - keep its newest context rows, commit none)
    bool dflash_with_mtp() const { return dfl_ != nullptr && q35_mtp_on(); }
    bool dflash_cjk(int32_t t) const;   // (a piece with kana / CJK ideographs, from the drafter's vocab)
    int dflash_max() const;   // (drafts a step at most: the GGUF's block_size - 1, or STRATA_Q35_DFLASH_N)
    bool dspark_draft(int32_t y, int32_t* drafts, float* conf, std::string& err);
    /// Speculative sampling (spec_sampling.hpp; qwen35's NextN drafter on one part): with a request's sampler set,
    /// the next dspark_draft SAMPLES its drafts from the drafter's distribution under that sampler (draws keyed by
    /// sp's seed and counter, a different stream from the target's) and keeps each one's kept list (q), and the next
    /// window's verify_sample(0) keeps every column's (p).  nullptr: off - argmax drafts and no lists, as before.
    void spec_sampling_set(const strata::kernels::SamplerParams* sp);
    bool spec_sampling_drafted() const { return ss_drafted_; }   // the last dspark_draft's drafts were sampled
    bool draft_dist(int k, const int*& ids, const float*& p, int& n) const;    // draft k's q (host)
    bool window_dist(int i, const int*& ids, const float*& p, int& n) const;   // column i's p (host)
    void dspark_upload();
    void dspark_report() const;
    bool decode_spec(strata::kernels::SamplerParams& sp, int64_t max_new, const std::function<bool(int)>& emit,
                     int64_t& produced, std::string& err);
    uint64_t spec_steps_ = 0, spec_hits_ = 0;              // speculative positions, accepted
    double spec_t_[4] = {0, 0, 0, 0};                      // STRATA_GLM_SPEC_PROF host phases of spec_head/tail
    int64_t spec_n_ = 0;
    cudaEvent_t spec_ev_[3] = {nullptr, nullptr, nullptr}; // ... and the device span of each token
    bool spec_ev_live_ = false;
    double spec_dev_ms_ = 0, spec_gap_ms_ = 0;
    int64_t spec_dev_n_ = 0;
    float* kda_bak_ = nullptr;                             // the head's saved recurrent states
    float* spec_hop_h_[2] = {nullptr, nullptr};            // pinned hop slots by position parity
    cudaEvent_t spec_ev_hop_[2] = {nullptr, nullptr};
    bool spec_kda_copy(bool restore);
    // the speculative decode on a split of more than two parts: the HEAD GROUP is this part and the ones after it
    // up to spec_head_last() (STRATA_GLM_SPEC_HEAD=<parts>, default half), the TAIL GROUP the rest; each group runs
    // its parts one after another and the two groups overlap as the two halves of a two-part split do
    Glm5Model* spec_head_last();
    Glm5Model* spec_tail_last();
    // this part's residual into the next part B (one host hop through this part's pinned hop_h), then B's layers at p
    bool hop_to_next(Glm5Model* B, int64_t p, std::string& err);
    bool spec_head(int64_t p, int32_t token, std::string& err);
    bool spec_tail(Glm5Model* head, int64_t p, std::string& err);
    // ---- the batched prompt path (src/core/glm_prefill.cu)
    struct PrefillState;
    PrefillState* pf_ = nullptr;
    // ---- the dense tier (issue #53, STRATA_GLM_PREFILL_DENSE_TIER; glm_prefill.cu): while a prompt longer than the
    // chunk runs, the layers' dense rows in w_arena_ hold the prompt path's buffers too (a longer chunk: fewer passes
    // over the experts), each layer's rows carried back into one of two layer buffers just before it runs, and every
    // row put back once the prompt ends (the decode reads the arena as it was).  load_pack notes each row's place in
    // the arena and, for a row uploaded as the GGUF stores it, its bytes in the shard mapping (shard -1: none - the
    // tier keeps a host copy of that row)
    struct DtRow {
        int layer = -1;                  // blk.<layer>, -1: not a layer's row
        uint64_t off = 0, bytes = 0;     // in w_arena_
        int shard = -1;                  // pack_shards_[shard].base + src: the bytes (-1: no mapping)
        uint64_t src = 0;
    };
    std::vector<DtRow> dt_rows_;
    bool dt_exl3_ = false;               // an EXL3 matrix in the arena (its WSlot is a host table): no tier
    struct DenseTier;
    DenseTier* dt_ = nullptr;
    bool dt_failed_ = false;             // dt_prepare refused (said why once): no tier for the session
    int dt_mode() const;                 // STRATA_GLM_PREFILL_DENSE_TIER: 0 off, 1 long prompts, 2 every prompt (test)
    bool dt_prepare(std::string& err);   // the first prompt that may take it: spans, sources, buffers, the worker
    int dt_chunk();                      // the longest chunk the tier lays out (0: none longer than the plain one)
    bool dt_fits(int T, int& place) const;   // ... for T: which buffers go to the dense rows (dt_chunk, prefill_carve)
    bool dt_layer(int il, int il_next, std::string& err);   // prefill_half: layer il's rows in place, il_next's coming
    bool dt_restore(std::string& err);   // after the prompt: every borrowed byte and pointer back
    void dt_destroy();
    // a long deepseek41 prompt's skip (prefill_run): the chunks before the cut stop at this layer - its attention (its
    // compressed KV and index keys, which the layers after it read) and no FFN, no layer after it (-1: every layer)
    int pf_kv_cut_ = -1;
    // deepseek41's prompt path: the next chunk's tokens (prefill_run), whose Engram rows are prefetched behind the
    // chunk's own (dsv41_engram_hash; empty: the last chunk, or STRATA_DS41_ENGRAM_AHEAD=0)
    std::vector<int32_t> pf_ahead_toks_;
    int64_t pf_ahead_p0_ = 0;
    // deepseek41's prompt path: the row's own top-k selection and candidate mask (fast_dsv41_attn reads and writes
    // these instead of the token path's ds_sel / v41_keep - the prompt runs a layer over every row before the next
    // layer, and a layer that is no index source reads the selection the last index source made for the same row)
    int* pf_sel_ = nullptr;
    unsigned char* pf_keep_ = nullptr;
    size_t prefill_v41_row() const;   // those two per row, in bytes (0: not deepseek41)
    bool prefill_setup(std::string& err, int per_pool = 0);   // in fast_setup: sizes, host buffers (no device memory);
                                                           // per_pool > 0: against the pool fast_setup carved
    bool prefill_setup_own(std::string& err);              // ... a dense qwen35: and its own device region (no pool)
    bool fast_dense_ffn(int il, std::string& err);         // a dense SwiGLU FFN (x / xq -> ffn): the dense qwen35
    // ---- the dense qwen35 on a GPU smaller than its file (src/core/qwen35_dense_ram.cu): load_pack leaves the trunk's
    // FFN weights in the shard mapping, q35d_place (fast_setup) puts as many layers' FFN in VRAM as the budget leaves
    // and the rest in pinned RAM, where the CPU computes them (decode, verify windows) and the prompt path streams
    // them to the GPU
    struct Q35dRam;
    Q35dRam* q35d_ = nullptr;
    bool q35d_place(std::string& err);
    bool q35d_defer_on() const;                            // load_pack: leave the trunk's FFN for q35d_place
    void q35d_defer(int il, int role, const uint8_t* src, size_t bytes, int type);   // role 0 gate, 1 up, 2 down
    bool q35d_ram(int il) const;                           // this layer's FFN lives in RAM
    size_t q35d_pending() const;                           // FFN bytes load_pack left for q35d_place
    // nt columns of q8_1 x (xq, contiguous) through layer il's FFN on the CPU into out (nt x n_embd floats), in
    // stream order on s
    // a RAM layer's FFN: the CPU's rows posted, gpu(kg) - the caller's launches over the first kg FF rows into out
    // (a split layer: kg > 0) - then the CPU's answer into out (added to the GPU's)
    bool q35d_cpu_ffn(int il, const void* xq, int nt, float* out, cudaStream_t s, std::string& err,
                      const std::function<bool(int)>& gpu = {});
    int q35d_gpu_rows(int il) const;
    // the loan (a card the FFN does not fit): whole layers' FFN live in the prompt path's buffers between prompts -
    // q35d_evict (prefill_lend) gives the buffers back to the prompt, q35d_restore (prefill_return) uploads them again
    bool q35d_prompt_ram(int il) const;                    // the prompt path stages this layer's FFN
    void q35d_evict(size_t region_used);   // (the region's first bytes the prompt takes: the layers past them stay)
    bool q35d_restore(std::string& err);
    std::vector<std::pair<uint8_t*, size_t>> prefill_loan_blocks() const;   // the dense prompt path's scratch
    // the prompt path: layer il's FFN weights in a VRAM staging buffer (gate / up / down into w), in order on s;
    // q35d_unstage after the layer's products (the buffer may be refilled)
    bool q35d_stage(int il, cudaStream_t s, WSlot* w, std::string& err);
    void q35d_unstage(int il, cudaStream_t s);
    void q35d_destroy();
    // the head: load_pack gives it its own allocation (q35d_head_set); a card that even every FFN in RAM leaves too
    // little room keeps only its first rows in VRAM and the CPU computes the rest (q35d_place) - q35d_head then runs
    // T columns of q8_1 x (xq, contiguous) into logits (T x n_vocab floats) on s; false with err empty: not split (the
    // caller's own launches)
    void q35d_head_set(const uint8_t* src, size_t bytes, int type, void* dev);
    bool q35d_head_split() const;
    // (xf: the normed rows as floats, T x n_embd - a BF16 / F32 head's VRAM rows read them; null for the others)
    bool q35d_head(const void* xq, int T, float* logits, cudaStream_t s, std::string& err, const float* xf = nullptr);
    size_t prefill_borrow_bytes() const;                   // device bytes the prompt path borrows from the pool's tail
    int prefill_keep() const;                              // slots a layer the pool keeps from a prompt (issue #23)
    size_t prefill_trim_prestage(size_t max_borrow);       // ... at most this many: a smaller prestage buffer (the new bytes)
    bool prefill_bind(uint8_t* region, size_t bytes, std::string& err);   // its buffers inside the borrowed region
    size_t prefill_carve(int T);                           // ... laid out for chunks of T (the bytes it uses)
    bool prefill_lend(std::string& err);                    // the tail slots -> the prompt path (drops their experts)
    bool prefill_return(std::string& err);                  // ... and back to the expert pool
    void prefill_destroy();
    void prefill_cap(int T, const char* why);              // its pinned staging for chunks of at most T
    void prefill_settle(double pinned_share);              // --prefill auto's lend cap from the pinned share (85/90%)
    std::pair<size_t, size_t> prefill_bytes_for(size_t T) const;   // a chunk's device buffers {kept rows, scratch}
    // the pinned host bytes a part's prompt path takes at its start (the token rows and the disk landing ring) and the
    // residual rows a part that hands on adds - this part's sizes
    struct PinnedPrompt {
        size_t staging = 0, hop = 0;
    };
    PinnedPrompt prefill_pinned() const;
    static size_t pool_avail(size_t free_b, size_t total_b, bool q4x = false);   // the expert pool's bytes from the free VRAM
    size_t pool_avail_uma(size_t free_b, size_t total_b) const;   // (unified memory + STRATA_GLM_VRAM_GB: as fast_setup)
    bool lend_tail(size_t limit, uint64_t& moved, uint64_t& dropped, std::string& err);
    bool vis_lend_ok_ = false;                             // this half's tail can go to the vision encoder
    bool vis_lent_ = false;                                // ... and is with it now (freed)
    bool prefill_half(int64_t p0, int T, std::string& err,
                      const int32_t* next_ids = nullptr);   // this half's layers over a chunk (rows in pf_->R)
    int32_t prefill_next_ = -1;                            // the token after the prompt's last prefilled position
    /// The prompt path: `tokens` at positions [position(), position() + n) through every layer in chunks, layer by
    /// layer (GEMMs for the projections, MMQ for the routed experts), leaving the state the token path would have
    /// left; no logits.  false with `err` empty when the fast path has no prompt path (feed them to forward()).
    bool prefill(const std::vector<int32_t>& tokens, std::string& err, int32_t next_token = -1);
    bool prefill_run(const std::vector<int32_t>& tokens, std::string& err);   // prefill() between lend and return
    int prefill_chunk() const;                             // tokens per chunk (0: no prompt path)
    const char* prefill_off_reason() const;                // why prefill_chunk() is 0 (the line glm prefill printed)
    /// Called (from any thread) when a chunk has gone through every layer: (positions done, total); false cancels
    /// the prompt (prefill then returns false with err "cancelled" and the sequence must be reset).
    std::function<bool(int64_t, int64_t)> prefill_progress;
    // Optional V4.1 single-device consumer of each prefilled row, in position order.
    std::function<bool(int64_t, const std::vector<float>&, std::string&)> prefill_logits;
    /// Debug: every half's sequence state (KDA states + conv histories, DSA caches) to <path>.<half>.bin, with the
    /// per-layer offsets in <path>.<half>.txt - two runs that should agree are compared per layer and component.
    bool dump_state(const std::string& path);

private:
    // scratch offsets, in floats into sc_
    int64_t sc_mixed = 0, sc_x = 0, sc_pre = 0, sc_post = 0, sc_comb = 0, sc_inv = 0, sc_proj = 0;
    int64_t sc_emb = 0;
    int64_t sc_conv[3] = {0, 0, 0};
    int64_t sc_g = 0, sc_g1 = 0, sc_beta = 0, sc_tmp = 0, sc_scan = 0, sc_g2 = 0, sc_gated = 0;
    int64_t sc_qr = 0, sc_kv = 0, sc_q = 0, sc_qabs = 0, sc_iq = 0, sc_iw = 0;
    int64_t sc_score = 0, sc_cells = 0, sc_attn = 0, sc_gate = 0, sc_up = 0, sc_h = 0, sc_dn = 0;
    int64_t sc_moe = 0, sc_sh = 0, sc_ffn = 0, sc_mixer = 0, sc_head = 0, sc_logits = 0;
    int64_t sc_ids = 0, sc_rw = 0;
};

/// What a part with these layer kinds lends the prompt path for a chunk of T tokens, its prestage buffer aside
/// (src/core/glm_prefill.cu) - the layer split search's startability gate.
size_t glm_prefill_lend_bytes(const Glm5Geometry& g, size_t T, bool has_kda, bool has_dsa, bool has_dense,
                              bool has_moe, bool mtp, int64_t max_ctx, size_t gstride);

}  // namespace strata::core
