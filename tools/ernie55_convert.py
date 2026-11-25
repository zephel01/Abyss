"""tools/ernie55_convert.py - baidu/ERNIE-5.5-Flash-Preview from its Hugging Face safetensors (FP8 e4m3 128x128 blocks,
MXFP4 routed experts) straight to the "MX" GGUF Abyss reads: the routed experts in ggml MXFP4 (type 39, the same bits
re-packed), everything else BF16 / F32.  Own code only (numpy): no llama.cpp, gguf-py, torch or safetensors.

usage: ernie55_convert.py <hf folder> <out dir> [--max-shard-gb=48] [--only-main] [--only-draft] [--check-only]
  <out dir>/ERNIE-5.5-Flash-Preview-MX-0000i-of-0000n.gguf   the model, split like llama.cpp's gguf-split
                                                              (shard 1: the keys only; then <= --max-shard-gb each)
  <out dir>/dspark/ERNIE-5.5-Flash-Preview-DSpark-MX.gguf     the DSpark draft head (model.mtp.*), its own file
  --check-only   reads every FP8 tensor and the expert scales and checks the conversion is exact; writes nothing

Why arch deepseek4 (not a new arch): an ERNIE layer is a DeepSeek-V4-Flash layer (the same E 4096, 43 layers, 64 heads
of 512, q_lora 1024, o_groups 8 x 1024, window 128, hc 4 / sinkhorn 20, HCA compression 128 with the same YaRN), so
Glm5Geometry::from_gguf_dsv4 already reads every shared hparam and dsv4_bind every shared tensor name; the engine adds
an ernie flag (deepseek4.ernie.* keys) the way MemT-1.0-Flash added deepseek4.memt.*.  An engine without the ERNIE
path refuses the file: expert_gating_func is 2 (sigmoid; deepseek4 only implements 4) - it cannot run it wrongly.

The layout (worklog 2026-11-25_ernie55-port-survey.md section 5; the GGUF shape is ne0 first, the row length):
  - the MLA + DSA layers (7, 14, 21, 28, 35, 42: compress_ratios -2) are listed in deepseek4.ernie.mla_layers and
    get compress ratio 0 in deepseek4.attention.compress_ratios (0 / 4 / 128 is what from_gguf_dsv4 accepts)
  - FP8 weights: W[i, j] = e4m3[i, j] * scale_inv[i / 128, j / 128]; every scale is a power of two, so the product is
    exact in BF16 (3 mantissa bits into 7) - checked tensor by tensor (the low 16 bits of the float32 product are 0)
  - MXFP4 experts: HF stores a row as int8 bytes, byte b = element 2b (low nibble) | element 2b+1 (high nibble), and
    one e8m0 per 32 elements (weight_scale_inv); ggml's block_mxfp4 is {e, qs[16]} with qs[j] = element j | element
    j+16 << 4.  The 4-bit codes (E2M1, sign in bit 3) and the e8m0 are the same in both, so the re-pack only moves
    nibbles (test_ernie55_convert.py round-trips it against a dequant of the HF layout)
  - the hyper-connection mapping_proj [16384, 24] is stored [in, out]: transposed to [24 rows of 16384]; alpha_pre /
    alpha_post / alpha_res -> hc_*_scale [3] in that order (mhc_pre's scale[0..2])
  - BF16 / F32 vectors (norms, sinks, biases, VHA U / V, MultiMax) widen to F32 (lossless); BF16 matrices stay BF16
"""
from __future__ import annotations

import json
import os
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_io as gio  # noqa: E402

NAME = "ERNIE-5.5-Flash-Preview"
PRE = "ernie55"          # tokenizer.ggml.pre (tools/strata_tokenizer.py ERNIE55_PATTERNS)
MXFP4 = 39


# ------------------------------------------------------------------------------------------------ number formats
def e4m3_table() -> np.ndarray:
    """The 256 FP8 e4m3fn values (bias 7, no infinities, S.1111.111 = NaN) as float32."""
    out = np.empty(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 0xF, b & 7
        if e == 15 and m == 7:
            out[b] = np.nan
        elif e == 0:
            out[b] = s * (m / 8.0) * 2.0 ** -6
        else:
            out[b] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return out


E4M3 = e4m3_table()
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)


def fp8_to_bf16(w: np.ndarray, scale: np.ndarray, name: str = "") -> np.ndarray:
    """[N, K] e4m3 bytes x [ceil(N/128), ceil(K/128)] float32 power-of-two scales -> BF16 bits [N, K], exactly (raises
    if a scale is not a power of two or a product does not fit BF16 exactly, or a NaN code appears)."""
    n, k = w.shape
    m, e = np.frexp(scale.astype(np.float64))
    if not np.all(m == 0.5):
        raise ValueError("%s: a block scale is not a power of two" % name)
    s = np.repeat(np.repeat(scale.astype(np.float32), 128, 0)[:n], 128, 1)[:, :k]
    v = E4M3[w] * s                                    # exact in float32 (3 mantissa bits times 2^e)
    if np.isnan(v).any():
        raise ValueError("%s: an e4m3 NaN code" % name)
    u = v.view(np.uint32)
    if np.any(u & 0xFFFF):
        raise ValueError("%s: a value is not exact in BF16" % name)
    return (u >> 16).astype(np.uint16)


def hf_mxfp4_dequant(packed: np.ndarray, scales: np.ndarray, hi_first: bool = False) -> np.ndarray:
    """The HF layout's values as float64: packed [N, K/2] bytes, scales [N, K/32] e8m0 bytes -> [N, K]."""
    p = packed.view(np.uint8)
    lo, hi = p & 0xF, p >> 4
    codes = np.stack([hi, lo] if hi_first else [lo, hi], axis=-1).reshape(p.shape[0], -1)
    sc = np.ldexp(1.0, scales.view(np.uint8).astype(np.int64) - 127)
    return E2M1[codes] * np.repeat(sc, 32, axis=1)


def mxfp4_repack(packed: np.ndarray, scales: np.ndarray, hi_first: bool = False) -> np.ndarray:
    """HF MXFP4 [N, K/2] int8 + [N, K/32] e8m0 -> ggml block_mxfp4 bytes [N, K/32 * 17] (bit for bit the same values)."""
    p = packed.view(np.uint8)
    n, kh = p.shape
    nb = kh // 16
    lo, hi = p & 0xF, p >> 4
    codes = np.stack([hi, lo] if hi_first else [lo, hi], axis=-1).reshape(n, nb, 32)   # element order
    qs = codes[:, :, :16] | (codes[:, :, 16:] << 4)
    out = np.empty((n, nb, 17), dtype=np.uint8)
    out[:, :, 0] = scales.view(np.uint8).reshape(n, nb)
    out[:, :, 1:] = qs
    return out.reshape(n, nb * 17)


def ggml_mxfp4_dequant(raw: np.ndarray) -> np.ndarray:
    """ggml's dequantize_row_mxfp4, written from its definition: kvalues_mxfp4 (twice the E2M1 values) times
    2^(e - 128) (GGML_E8M0_TO_FP32_HALF), qs[j] low nibble = element j, high = element j + 16.  float64, flat."""
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, 17)
    kv = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float64)
    d = np.ldexp(1.0, b[:, 0].astype(np.int64) - 128)[:, None]
    q = b[:, 1:]
    return (np.concatenate([kv[q & 0xF], kv[q >> 4]], axis=1) * d).reshape(-1)


# ------------------------------------------------------------------------------------------------ safetensors
class Source:
    """Every tensor of the folder's *.safetensors (and flashindexer/): name -> (path, dtype, shape, offset, nbytes)."""

    def __init__(self, folder: Path):
        self.t: dict[str, tuple] = {}
        files = sorted(folder.glob("*.safetensors")) + sorted((folder / "flashindexer").glob("*.safetensors"))
        for f in files:
            with open(f, "rb") as h:
                n = struct.unpack("<Q", h.read(8))[0]
                hdr = json.loads(h.read(n))
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                if k in self.t:
                    raise ValueError("%s twice (%s, %s)" % (k, self.t[k][0], f))
                self.t[k] = (f, v["dtype"], [int(x) for x in v["shape"]], 8 + n + a, b - a)
        self._fd: dict[Path, int] = {}

    def has(self, name: str) -> bool:
        return name in self.t

    def key(self, name: str):
        f, _, _, off, _ = self.t[name]
        return (str(f), off)

    def raw(self, name: str) -> np.ndarray:
        f, _, _, off, nb = self.t[name]
        fd = self._fd.get(f)
        if fd is None:
            fd = self._fd[f] = os.open(f, os.O_RDONLY)
        buf = bytearray(nb)
        mv, got = memoryview(buf), 0
        while got < nb:
            r = os.preadv(fd, [mv[got:]], off + got)
            if r <= 0:
                raise IOError("%s: short read" % name)
            got += r
        return np.frombuffer(buf, dtype=np.uint8)

    def array(self, name: str) -> np.ndarray:
        _, dt, shape, _, _ = self.t[name]
        np_dt = {"BF16": "<u2", "F32": "<f4", "F8_E4M3": "u1", "I8": "u1", "F8_E8M0": "u1"}[dt]
        return self.raw(name).view(np_dt).reshape(shape)

    def dtype(self, name: str) -> str:
        return self.t[name][1]

    def shape(self, name: str):
        return self.t[name][2]

    def f32(self, name: str) -> np.ndarray:
        dt = self.dtype(name)
        a = self.array(name)
        if dt == "BF16":
            return (a.astype(np.uint32) << 16).view(np.float32)
        if dt == "F32":
            return a.astype(np.float32)
        raise ValueError("%s: %s is not a float vector" % (name, dt))

    def bf16(self, name: str) -> np.ndarray:
        """A weight matrix as BF16 bits: BF16 as stored, FP8 through its block scales (exact)."""
        dt = self.dtype(name)
        if dt == "BF16":
            return self.array(name)
        if dt == "F8_E4M3":
            sc = name + "_scale_inv" if not name.endswith(".weight") else name + "_scale_inv"
            return fp8_to_bf16(self.array(name), self.f32(sc), name)
        if dt == "F32":
            return gio.to_bf16(self.array(name))
        raise ValueError("%s: %s" % (name, dt))


# ------------------------------------------------------------------------------------------------ the tensor list
class T:
    """One output tensor: GGUF name, GGUF shape (ne0 first), type, the source names it reads (for ordering), a maker."""
    __slots__ = ("name", "shape", "type", "srcs", "make")

    def __init__(self, name, shape, t, srcs, make):
        self.name, self.shape, self.type, self.srcs, self.make = name, [int(x) for x in shape], t, srcs, make

    @property
    def nbytes(self) -> int:
        return gio.nbytes(self.type, self.shape)


def _mat(st: Source, out: str, src: str) -> T:
    """A weight matrix [rows, cols] (numpy) -> BF16, GGUF shape [cols, rows]."""
    r, c = st.shape(src)
    return T(out, [c, r], 30, [src], lambda: st.bf16(src))


def _vec(st: Source, out: str, src: str) -> T:
    """A vector / small table -> F32 (BF16 widened exactly), GGUF shape = reversed numpy shape."""
    return T(out, list(reversed(st.shape(src))), 0, [src], lambda: st.f32(src))


def _f32_t(st: Source, out: str, src: str) -> T:
    """An F32 [in, out] matrix stored transposed (hc mapping_proj / hc_head_fn) -> F32 [out rows of in]."""
    i, o = st.shape(src)
    return T(out, [i, o], 0, [src], lambda: np.ascontiguousarray(st.f32(src).T))


def _hc(st: Source, B: str, L: str, kind: str, path: str) -> list[T]:
    P = L + path + "."
    scale = [P + "alpha_pre", P + "alpha_post", P + "alpha_res"]
    return [_f32_t(st, B + "hc_%s_fn.weight" % kind, P + "mapping_proj.weight"),
            _vec(st, B + "hc_%s_base.weight" % kind, P + "bias"),
            T(B + "hc_%s_scale.weight" % kind, [3], 0, scale,
              lambda: np.concatenate([st.f32(s).reshape(-1) for s in scale]))]


class ExpertCache:
    """A layer's 512 x (w1, w3, w2) read in one pass in file order, re-packed; handed out tensor by tensor."""

    def __init__(self, st: Source, n_expert: int, hi_first: bool = False):
        self.st, self.ne, self.hi_first = st, n_expert, hi_first
        self.prefix = None
        self.out: dict[str, np.ndarray] = {}

    def get(self, prefix: str, proj: str) -> np.ndarray:
        if self.prefix != prefix:
            self.out.clear()
            self.prefix = prefix
            names = []
            for e in range(self.ne):
                for w in ("w1", "w3", "w2"):
                    names += [(prefix + "%d.%s.weight" % (e, w), e, w, 0), (prefix + "%d.%s.weight_scale_inv" % (e, w), e, w, 1)]
            names.sort(key=lambda x: self.st.key(x[0]))
            parts: dict[tuple, list] = {}
            for n, e, w, is_scale in names:
                parts.setdefault((e, w), [None, None])[is_scale] = self.st.array(n)
            for w in ("w1", "w3", "w2"):
                rows = []
                for e in range(self.ne):
                    p, s = parts.pop((e, w))
                    if s.shape[1] * 16 != p.shape[1]:
                        raise ValueError("%s%d.%s: %s packed vs %s scales" % (prefix, e, w, p.shape, s.shape))
                    if np.any(s == 0xFF):
                        raise ValueError("%s%d.%s: an e8m0 NaN scale (0xFF)" % (prefix, e, w))
                    rows.append(mxfp4_repack(p, s, self.hi_first))
                self.out[w] = np.stack(rows)
        return self.out.pop(proj)


def tensors(st: Source, cfg: dict, hi_first: bool = False) -> list[T]:
    """The main model's tensors in the order they are written (by layer, then by source position)."""
    E, NE = cfg["hidden_size"], cfg["n_routed_experts"]
    I_exp, LAT = cfg["moe_intermediate_size"], cfg["moe_latent_size"]
    nl = cfg["num_hidden_layers"]
    mla = [i for i, r in enumerate(cfg["compress_ratios"]) if r == -2]
    ec = ExpertCache(st, NE, hi_first)
    out = [_mat(st, "token_embd.weight", "model.embed_tokens.weight")]
    for il in range(nl):
        L, B = "model.layers.%d." % il, "blk.%d." % il
        A = L + "self_attn."
        lay = [_vec(st, B + "attn_norm.weight", L + "input_layernorm.weight"),
               _vec(st, B + "ffn_norm.weight", L + "post_attention_layernorm.weight")]
        lay += _hc(st, B, L, "attn", "self_attention_hyper_connection")
        lay += _hc(st, B, L, "ffn", "mlp_hyper_connection")
        lay += [_mat(st, B + "attn_gate.weight", A + "gate_proj.weight"),
                _vec(st, B + "attn_vha_u.weight", A + "vha_postmix_U"),
                _vec(st, B + "attn_vha_v.weight", A + "vha_postmix_V")]
        if il in mla:
            C = A + "core_attention."
            lay += [_mat(st, B + "attn_q_a.weight", A + "q_a_proj.weight"),
                    _vec(st, B + "attn_q_a_norm.weight", A + "q_a_layernorm.weight"),
                    _mat(st, B + "attn_q_b.weight", A + "q_b_proj.weight"),
                    _mat(st, B + "attn_kv_a_mqa.weight", A + "kv_a_proj_with_mqa.weight"),
                    _vec(st, B + "attn_kv_a_norm.weight", A + "kv_a_layernorm.weight"),
                    _mat(st, B + "attn_kv_b.weight", A + "kv_b_proj.weight"),
                    _mat(st, B + "attn_output.weight", A + "o_proj.weight"),
                    _vec(st, B + "attn_sinks.weight", C + "softmax_offset"),
                    _mat(st, B + "indexer.attn_q_b.weight", C + "indexer.wq_b.weight"),
                    _mat(st, B + "indexer.attn_k.weight", C + "indexer.wk.weight"),
                    _vec(st, B + "indexer.k_norm.weight", C + "indexer.k_norm.weight"),
                    _vec(st, B + "indexer.k_norm.bias", C + "indexer.k_norm.bias"),
                    # weights_proj [64, 4096] as F32 (lossless), the layout V4's indexer.proj has
                    T(B + "indexer.proj.weight", [E, 64], 0, [C + "indexer.weights_proj.weight"],
                      lambda s=C + "indexer.weights_proj.weight": st.f32(s)),
                    # FlashIndexer (off by default upstream): the learned head gate and its bias
                    _mat(st, B + "indexer.flash_gate.weight", C + "indexer.linear_gate.weight"),
                    _vec(st, B + "indexer.flash_gate.bias", C + "indexer.flashindexer_bias")]
        else:
            C = A + "core_attention."
            lay += [_mat(st, B + "attn_q_a.weight", A + "linear_q_down_proj.weight"),
                    _vec(st, B + "attn_q_a_norm.weight", A + "q_layernorm.weight"),
                    _mat(st, B + "attn_q_b.weight", A + "linear_q_up_proj.weight"),
                    _mat(st, B + "attn_kv.weight", A + "linear_kv_proj.weight"),
                    _vec(st, B + "attn_kv_a_norm.weight", A + "kv_layernorm.weight"),
                    _mat(st, B + "attn_output_a.weight", A + "linear_o_group_proj"),
                    _mat(st, B + "attn_output_b.weight", A + "o_proj.weight"),
                    _vec(st, B + "attn_sinks.weight", C + "attn_sink"),
                    _mat(st, B + "attn_compressor_kv.weight", C + "compressor.linear_wkv.weight"),
                    _mat(st, B + "attn_compressor_gate.weight", C + "compressor.linear_wgate.weight"),
                    _vec(st, B + "attn_compressor_ape.weight", C + "compressor.ape"),
                    _vec(st, B + "attn_compressor_norm.weight", C + "compressor.norm.weight")]
        if il < cfg["first_k_dense_replace"]:
            M = L + "mlp."
            lay += [_mat(st, B + "ffn_gate.weight", M + "gate_proj.weight"),
                    _mat(st, B + "ffn_up.weight", M + "up_proj.weight"),
                    _mat(st, B + "ffn_down.weight", M + "down_proj.weight")]
        else:
            lay += moe_tensors(st, B, L + "block_sparse_moe.", ec, E, NE, I_exp, LAT)
        lay.sort(key=lambda t: st.key(t.srcs[0]))
        out += lay
    tail = [_f32_t(st, "output_hc_fn.weight", "model.mhc_contract.hc_head_fn"),
            _vec(st, "output_hc_base.weight", "model.mhc_contract.hc_head_base"),
            _vec(st, "output_hc_scale.weight", "model.mhc_contract.hc_head_scale"),
            _vec(st, "output_norm.weight", "model.norm.weight"),
            _mat(st, "output.weight", "lm_head.weight"),
            _vec(st, "output_multimax_r.weight", "model.lm_head.multimax_ranges"),
            _vec(st, "output_multimax_t.weight", "model.lm_head.multimax_ts")]
    tail.sort(key=lambda t: st.key(t.srcs[0]))
    return out + tail


def moe_tensors(st, B, M, ec, E, NE, I_exp, LAT) -> list[T]:
    X = M + "experts."
    first = X + "0.w1.weight"
    lay = [_mat(st, B + "ffn_gate_inp.weight", M + "gate.weight"),
           _mat(st, B + "ffn_gate_inp_1.weight", M + "gate.weight_1"),
           _vec(st, B + "exp_probs_b.bias", M + "e_score_correction_bias"),
           _vec(st, B + "ffn_exp_scale.weight", M + "gate.routed_scaling_factor_param"),
           _mat(st, B + "ffn_latent_down.weight", M + "fc1_latent_proj.weight"),
           _vec(st, B + "ffn_latent_norm.weight", M + "latent_norm.weight"),
           _mat(st, B + "ffn_latent_up.weight", M + "fc2_latent_proj.weight"),
           _mat(st, B + "ffn_gate_shexp.weight", M + "shared_experts.w1.weight"),
           _mat(st, B + "ffn_up_shexp.weight", M + "shared_experts.w3.weight"),
           _mat(st, B + "ffn_down_shexp.weight", M + "shared_experts.w2.weight")]
    # the three expert tensors sort at the layer's first expert byte, gate, up, down in that order
    lay += [T(B + "ffn_gate_exps.weight", [LAT, I_exp, NE], MXFP4, [first], lambda: ec.get(X, "w1")),
            T(B + "ffn_up_exps.weight", [LAT, I_exp, NE], MXFP4, [first], lambda: ec.get(X, "w3")),
            T(B + "ffn_down_exps.weight", [I_exp, LAT, NE], MXFP4, [first], lambda: ec.get(X, "w2"))]
    return lay


def draft_tensors(st: Source, cfg: dict, hi_first: bool = False) -> list[T]:
    """The DSpark draft head (model.mtp.*) under V4 DSpark's names (dspark.S.*, dspark.main_proj ...)."""
    E, NE = cfg["hidden_size"], cfg["n_routed_experts"]
    I_exp, LAT = cfg["moe_intermediate_size"], cfg["moe_latent_size"]
    ec = ExpertCache(st, NE, hi_first)
    out = []
    n = 0
    while st.has("model.mtp.%d.input_layernorm.weight" % n):
        n += 1
    for s in range(n):
        L, B = "model.mtp.%d." % s, "dspark.%d." % s
        A = L + "self_attn."
        lay = [_vec(st, B + "attn_norm.weight", L + "input_layernorm.weight"),
               _vec(st, B + "ffn_norm.weight", L + "post_attention_layernorm.weight")]
        lay += _hc(st, B, L, "attn", "self_attention_hyper_connection")
        lay += _hc(st, B, L, "ffn", "mlp_hyper_connection")
        lay += [_mat(st, B + "attn_q_a.weight", A + "linear_q_down_proj.weight"),
                _vec(st, B + "attn_q_a_norm.weight", A + "q_layernorm.weight"),
                _mat(st, B + "attn_q_b.weight", A + "linear_q_up_proj.weight"),
                _mat(st, B + "attn_kv.weight", A + "linear_kv_proj.weight"),
                _vec(st, B + "attn_kv_a_norm.weight", A + "kv_layernorm.weight"),
                _mat(st, B + "attn_output_a.weight", A + "linear_o_group_proj"),
                _mat(st, B + "attn_output_b.weight", A + "o_proj.weight"),
                _vec(st, B + "attn_sinks.weight", A + "core_attention.attn_sink"),
                _mat(st, B + "attn_gate.weight", A + "gate_proj.weight"),
                _vec(st, B + "attn_vha_u.weight", A + "vha_postmix_U"),
                _vec(st, B + "attn_vha_v.weight", A + "vha_postmix_V")]
        lay += moe_tensors(st, B, L + "block_sparse_moe.", ec, E, NE, I_exp, LAT)
        lay.sort(key=lambda t: st.key(t.srcs[0]))
        out += lay
    P = "model.mtp."
    w1 = P + "markov_head.markov_w1.weight"
    tail = [_mat(st, "dspark.main_proj.weight", P + "feature_projection.weight"),
            _vec(st, "dspark.main_norm.weight", P + "feature_norm.weight"),
            # markov_w1 [256, vocab] (a Linear vocab -> 256): stored as a row per token, like V4 DSpark's
            T("dspark.markov_w1.weight", [st.shape(w1)[0], st.shape(w1)[1]], 30, [w1],
              lambda: np.ascontiguousarray(st.bf16(w1).T)),
            _mat(st, "dspark.markov_w2.weight", P + "markov_head.markov_w2.weight"),
            _vec(st, "dspark.conf_proj.weight", P + "confidence_head.projection.weight"),
            _f32_t(st, "dspark.hc_head_fn.weight", P + "mhc_contract.hc_head_fn"),
            _vec(st, "dspark.hc_head_base.weight", P + "mhc_contract.hc_head_base"),
            _vec(st, "dspark.hc_head_scale.weight", P + "mhc_contract.hc_head_scale"),
            _vec(st, "dspark.norm.weight", P + "norm.weight")]
    tail.sort(key=lambda t: st.key(t.srcs[0]))
    return out + tail, n


# ------------------------------------------------------------------------------------------------ metadata
def tokenizer_keys(w: gio.Writer, hf: Path) -> dict:
    tj = json.loads((hf / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = tj["model"]["vocab"]
    n = len(vocab)
    toks = [None] * n
    for s, i in vocab.items():
        toks[i] = s
    types = [1] * n
    added = {a["id"]: a for a in tj["added_tokens"]}
    for i, a in added.items():
        if toks[i] != a["content"]:
            raise ValueError("added token %d: %r vs vocab %r" % (i, a["content"], toks[i]))
        # CONTROL for the special ones; the tags the model writes in its text (<ernie:think>, <ernie:tool_call> ...,
        # special=False in tokenizer.json) USER_DEFINED.  The server's encode_spans keeps both out of message text
        # (tokenization_ernie.py: only markers the renderer inserts become control ids)
        types[i] = 3 if a["special"] else 4
    if None in toks:
        raise ValueError("vocab has holes")
    merges = [" ".join(m) if isinstance(m, list) else m for m in tj["model"]["merges"]]
    for m in merges:
        if len(m.split(" ")) != 2:
            raise ValueError("merge %r does not split into a pair" % m)
    gen = json.loads((hf / "generation_config.json").read_text())
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", PRE)
    w.add_array("tokenizer.ggml.tokens", toks, gio.STRING)
    w.add_array("tokenizer.ggml.token_type", types, gio.I32)
    w.add_array("tokenizer.ggml.merges", merges, gio.STRING)
    # <|ERNIE_MASK|> (2) starts every sequence (tokenizer.json's post_processor, the chat template's first token) and
    # ends one (generation_config eos [2, 8]); <|ernie:im_end|> (8) ends a turn.  <|ERNIE_BEGIN|> (1) is config's bos
    # but nothing emits it.  add_bos false: the template writes the 2 itself
    w.add_uint32("tokenizer.ggml.bos_token_id", 2)
    w.add_uint32("tokenizer.ggml.eos_token_id", 2)
    w.add_uint32("tokenizer.ggml.eot_token_id", 8)
    w.add_uint32("tokenizer.ggml.padding_token_id", int(gen.get("pad_token_id", 0)))
    w.add_uint32("tokenizer.ggml.unknown_token_id", 0)
    w.add_bool("tokenizer.ggml.add_bos_token", False)
    w.add_bool("tokenizer.ggml.add_eos_token", False)
    w.add_string("tokenizer.chat_template", (Path(__file__).resolve().parent / "ernie55_chat_template.jinja").read_text())
    return {"n_vocab": n, "n_merges": len(merges), "n_control": types.count(3), "n_user": types.count(4)}


def model_keys(w: gio.Writer, cfg: dict, gen: dict) -> None:
    p = "deepseek4."
    mla = [i for i, r in enumerate(cfg["compress_ratios"]) if r == -2]
    ys = cfg["rope_scaling"]
    w.add_string("general.name", NAME)
    w.add_string("general.basename", NAME)
    w.add_string("general.license", "apache-2.0")
    w.add_string("general.base_model.0.repo_url", "https://huggingface.co/baidu/" + NAME)
    w.add_string("general.quantized_by", "Abyss tools/ernie55_convert.py (MX: experts MXFP4 as shipped, rest BF16)")
    w.add_float32("general.sampling.top_p", float(gen.get("top_p", 0.95)))
    w.add_float32("general.sampling.temp", float(gen.get("temperature", 1.0)))
    w.add_uint32(p + "block_count", cfg["num_hidden_layers"])
    w.add_uint32(p + "context_length", cfg["max_position_embeddings"])
    w.add_uint32(p + "embedding_length", cfg["hidden_size"])
    w.add_uint32(p + "vocab_size", cfg["vocab_size"])
    w.add_uint32(p + "attention.head_count", cfg["num_attention_heads"])
    w.add_uint32(p + "attention.head_count_kv", cfg["num_key_value_heads"])
    w.add_uint32(p + "attention.key_length", cfg["head_dim"])
    w.add_uint32(p + "attention.value_length", cfg["v_head_dim"])
    w.add_uint32(p + "attention.q_lora_rank", cfg["q_lora_rank"])
    w.add_uint32(p + "attention.sliding_window", cfg["sliding_window"])
    w.add_uint32(p + "attention.output_group_count", cfg["o_groups"])
    w.add_uint32(p + "attention.output_lora_rank", cfg["o_lora_rank"])
    w.add_float32(p + "attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"])
    w.add_array(p + "attention.compress_ratios", [0 if r == -2 else int(r) for r in cfg["compress_ratios"]], gio.I32)
    w.add_float32(p + "attention.compress_rope_freq_base", cfg["compress_rope_theta"])
    w.add_uint32(p + "attention.indexer.head_count", cfg["index_n_heads"])
    w.add_uint32(p + "attention.indexer.key_length", cfg["index_head_dim"])
    w.add_uint32(p + "attention.indexer.top_k", cfg["index_topk"])
    w.add_uint32(p + "rope.dimension_count", cfg["qk_rope_head_dim"])
    w.add_float32(p + "rope.freq_base", float(cfg["rope_theta"]))           # plain RoPE (the MLA layers)
    w.add_string(p + "rope.scaling.type", ys["type"])                      # YaRN: the HCA layers only
    w.add_float32(p + "rope.scaling.factor", float(ys["factor"]))
    w.add_uint32(p + "rope.scaling.original_context_length", ys["original_max_position_embeddings"])
    w.add_float32(p + "rope.scaling.yarn_beta_fast", float(ys["beta_fast"]))
    w.add_float32(p + "rope.scaling.yarn_beta_slow", float(ys["beta_slow"]))
    w.add_uint32(p + "expert_count", cfg["n_routed_experts"])
    w.add_uint32(p + "expert_used_count", cfg["num_experts_per_tok"])
    w.add_uint32(p + "expert_feed_forward_length", cfg["moe_intermediate_size"])
    w.add_uint32(p + "expert_shared_count", cfg["n_shared_experts"])
    w.add_uint32(p + "expert_gating_func", 2)          # sigmoid (deepseek4 implements 4 only: an old engine refuses)
    w.add_float32(p + "expert_weights_scale", 1.0)     # the per-expert learned scale is blk.N.ffn_exp_scale
    w.add_bool(p + "expert_weights_norm", bool(cfg["norm_topk_prob"]))
    w.add_uint32(p + "leading_dense_block_count", cfg["first_k_dense_replace"])
    w.add_uint32(p + "feed_forward_length", cfg["intermediate_size"])
    w.add_uint32(p + "hash_layer_count", 0)
    w.add_uint32(p + "nextn_predict_layers", 0)
    w.add_uint32(p + "hyper_connection.count", cfg["hc_mult"])
    w.add_uint32(p + "hyper_connection.sinkhorn_iterations", cfg["hc_sinkhorn_iters"])
    w.add_float32(p + "hyper_connection.epsilon", cfg["hc_eps"])
    e = p + "ernie."
    w.add_uint32(e + "version", 1)                     # this layout; bump when it changes
    w.add_array(e + "mla_layers", mla, gio.I32)
    w.add_uint32(e + "mla.head_count", cfg["hybrid_mla_num_attention_heads"])
    w.add_uint32(e + "mla.q_lora_rank", cfg["hybrid_mla_q_lora_rank"])
    w.add_uint32(e + "mla.kv_lora_rank", cfg["hybrid_mla_kv_lora_rank"])
    w.add_uint32(e + "mla.qk_nope_head_dim", cfg["hybrid_mla_qk_nope_head_dim"])
    w.add_uint32(e + "mla.qk_rope_head_dim", cfg["hybrid_mla_qk_rope_head_dim"])
    w.add_uint32(e + "mla.v_head_dim", cfg["hybrid_mla_v_head_dim"])
    w.add_float32(e + "mla.rope_freq_base", float(cfg["rope_theta"]))
    w.add_uint32(e + "vha_rank", 16)
    w.add_float32(e + "situ_beta", cfg["activation_situ_beta"])
    w.add_float32(e + "situ_linear_beta", cfg["activation_situ_linear_beta"])
    w.add_uint32(e + "router_count", 2)
    w.add_uint32(e + "latent_size", cfg["moe_latent_size"])
    w.add_bool(e + "latent_norm", bool(cfg["latent_moe_use_norm"]))
    w.add_bool(e + "hc_rms_eps_outside_sqrt", True)    # mHC pre: 1 / (sqrt(mean) + eps) (both reference stacks)
    w.add_bool(e + "multimax", cfg.get("multimax") == "lm_head")
    w.add_bool(e + "attn_gate_sigmoid_bf16", True)     # sigmoid(gate) rounded to BF16 before the multiply (both stacks)


def draft_keys(w: gio.Writer, cfg: dict, n_stage: int) -> None:
    p = "deepseek4.dspark."
    w.add_string("general.name", NAME + " DSpark")
    w.add_uint32(p + "layer_count", n_stage)
    w.add_uint32(p + "block_size", cfg["dspark_block_size"])
    w.add_uint32(p + "markov_rank", cfg["dspark_markov_rank"])
    w.add_uint32(p + "noise_token_id", cfg["dspark_noise_token_id"])
    w.add_uint32(p + "expert_count", cfg["n_routed_experts"])
    w.add_array(p + "target_layers", cfg["dspark_target_layer_ids"], gio.I32)
    e = p + "ernie."
    w.add_uint32(e + "version", 1)
    w.add_uint32(e + "expert_used_count", cfg["num_experts_per_tok"])
    w.add_uint32(e + "latent_size", cfg["moe_latent_size"])
    w.add_float32(e + "situ_beta", cfg["activation_situ_beta"])
    w.add_float32(e + "situ_linear_beta", cfg["activation_situ_linear_beta"])
    w.add_uint32(e + "router_count", 2)
    w.add_uint32(e + "vha_rank", 16)
    w.add_bool(e + "multimax", True)


# ------------------------------------------------------------------------------------------------ writing
def write_split(ts: list[T], out_dir: Path, stem: str, keys, arch: str, max_bytes: int, log) -> list[Path]:
    """llama.cpp gguf-split's layout: shard 1 the keys and no tensors, then shards of whole tensors up to max_bytes."""
    groups, cur, size = [], [], 0
    for t in ts:
        b = -(-t.nbytes // 32) * 32
        if cur and size + b > max_bytes:
            groups.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += b
    if cur:
        groups.append(cur)
    n = len(groups) + 1
    paths = [out_dir / ("%s-%05d-of-%05d.gguf" % (stem, i + 1, n)) for i in range(n)]
    split = lambda w, i: (w.add("split.no", gio.U16, i), w.add("split.tensors.count", gio.I32, len(ts)),  # noqa: E731
                          w.add("split.count", gio.U16, n))
    w = gio.Writer(paths[0], arch=arch)
    keys(w)
    split(w, 0)
    w.write()
    for i, g in enumerate(groups):
        w = gio.Writer(paths[i + 1], arch=arch)
        split(w, i + 1)
        for t in g:
            w.add_tensor(t.name, t.shape, t.type, (lambda t=t: _bytes(t)))
        t0 = time.time()
        w.write(progress=log)
        log("shard %d/%d written (%.1f GB, %.0f s)" % (i + 2, n, paths[i + 1].stat().st_size / 1e9, time.time() - t0))
    return paths


def _bytes(t: T) -> np.ndarray:
    a = t.make()
    if t.type == 0:
        a = np.ascontiguousarray(a, dtype="<f4")
    return np.ascontiguousarray(a).reshape(-1).view(np.uint8)


def check_only(st: Source, log) -> int:
    """Every FP8 tensor through fp8_to_bf16 (raises when not exact) and every expert scale checked for 0xFF."""
    n_fp8 = n_sc = 0
    lo, hi = 255, 0
    for name in sorted(st.t, key=st.key):
        dt = st.dtype(name)
        if dt == "F8_E4M3":
            st.bf16(name)
            n_fp8 += 1
        elif dt == "F8_E8M0":
            s = st.array(name)
            if np.any(s == 0xFF):
                raise ValueError("%s: 0xFF scale" % name)
            lo, hi = min(lo, int(s.min())), max(hi, int(s.max()))
            n_sc += 1
    log("FP8 tensors exact in BF16: %d / %d; expert scale tensors %d, e8m0 range %d..%d" % (n_fp8, n_fp8, n_sc, lo, hi))
    return 0


def main() -> int:
    flags = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:] if a.startswith("--"))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        return 2
    hf, out = Path(args[0]), Path(args[1])
    cfg = json.loads((hf / "config.json").read_text())
    gen = json.loads((hf / "generation_config.json").read_text())
    qc = cfg.get("quantization_config", {})
    if qc.get("fmt") != "e4m3" or qc.get("expert_dtype") != "mxfp4" or qc.get("weight_block_size") != [128, 128]:
        sys.exit("expected FP8 e4m3 128x128 + MXFP4 experts, got %s" % qc)
    st = Source(hf)
    t0 = time.time()

    def log(msg):
        print("[%6.0f s] %s" % (time.time() - t0, msg), flush=True)

    if "--check-only" in flags:
        return check_only(st, log)
    out.mkdir(parents=True, exist_ok=True)
    max_bytes = int(float(flags.get("--max-shard-gb") or 48) * 1e9)
    if "--only-draft" not in flags:
        ts = tensors(st, cfg)
        info = {}

        def keys(w):
            model_keys(w, cfg, gen)
            info.update(tokenizer_keys(w, hf))
        log("main: %d tensors, %.2f GB" % (len(ts), sum(t.nbytes for t in ts) / 1e9))
        paths = write_split(ts, out, NAME + "-MX", keys, "deepseek4", max_bytes,
                            lambda m: log(m) if " " in m else None)
        log("main written: %s; tokenizer %s" % ([p.name for p in paths], info))
    if "--only-main" not in flags:
        ts, n_stage = draft_tensors(st, cfg)
        d = out / "dspark"
        d.mkdir(exist_ok=True)
        w = gio.Writer(d / (NAME + "-DSpark-MX.gguf"), arch="deepseek4-dspark")
        draft_keys(w, cfg, n_stage)
        for t in ts:
            w.add_tensor(t.name, t.shape, t.type, (lambda t=t: _bytes(t)))
        w.write()
        log("draft written: %d stages, %d tensors, %.2f GB" % (n_stage, len(ts), w.path.stat().st_size / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
