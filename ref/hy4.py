"""ref/hy4.py - a Hy4-flash (GGUF arch hy_v4_flash) reference forward pass in numpy, one op at a time.

Written for Abyss from Tencent's model definition (the HYV4Flash* modules of the transformers pull request the
tencent/Hy4-flash README points at, and the checkpoint's config.json) - read to understand the semantics, not
transcribed.  It runs on a GGUF's own (dequantized) weights, so it is the yardstick the engine's token path is
compared to: same weights, float64/float32 numpy arithmetic, no fusion, no caches.

The pass is TEACHER-FORCED over a whole token sequence (text only: every position's three M-RoPE planes are the
token's own position, so the interleaved M-RoPE reduces to ordinary interleaved-pair RoPE).

What one layer does (layer il of 42; the hidden state is hc = 4 streams of n_embd = 3072):

  * Engram (layers 1, 11, 21, 31 of the GGUF - the config's one-based 2, 12, 22, 32): the token and the two before
    it (the pad id before the sequence start) are mapped through the normalized-vocabulary map, mixed with three
    odd multipliers and xor-ed into a 2-gram and a 3-gram hash; 8 heads per order take the hash modulo their prime
    and read one 96-wide FP8 row each from the table; the 16 rows are RMS-normalized per head, concatenated (1536),
    projected to 4 keys + 1 value (engram_kv + engram_kv_b); stream c gets
        sigmoid(signed_sqrt(<norm(key_c), norm(R_c)> * w_c / sqrt(n_embd))) * main + scale_c * main,
    main = pre_gamma * norm(value)
  * the attention-side hyper-connection: mixes = fn @ flat(R) * rsqrt(mean(flat^2) + eps); pre = sigmoid(...) +
    hc_eps; post = magnitude * sigmoid(...) + hc_eps; x = sum_c pre_c R_c (no stream-mixing matrix)
  * attn_norm, gated MLA: q through a 1024 LoRA (q_a_norm), 192 nope + 64 rope dims per head; one shared latent
    (512, kv_a_norm) + one shared 64-dim rope key; per head K = k_b^T latent, V = v_b latent; scale 1/16; a learned
    sink logit per head in the softmax; the output times sigmoid(attn_gate x) per head; attn_output; then
    post_attention_norm (the sandwich norm).  Layers 0, 1, 5, 9, ... 41: 32 heads, RoPE base 1e7, the compressed
    indexer once the context passes 2048; the others: 64 heads, a 512-key sliding window, RoPE base 1e4
  * R_c += post_c * attn
  * the FFN-side hyper-connection the same way, ffn_norm, then layer 0 a dense SwiGLU (no clamp), the others the
    latent MoE: a sigmoid router (exp_probs_b for the choice only, top 16, renormalized, x 3.998) on x, the routed
    experts on ffn_routed_down x (1536 wide; gate clamped <= 10, up to [-10, 10]), ffn_routed_up back, plus the
    shared expert on x; then post_ffw_norm
  * R_c += post_c * ffn
  * the head: a hyper-connection collapse (output_hc_*), output_norm, output (F32 head in the original)

Usage (STRATA_GGUF_PY must name a gguf-py for the IQ types):
    python ref/hy4.py SHARD1.gguf --tokens 1,2,3 [--layers N] [--dump DIR]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
import gguf_io  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402

F32 = np.float32
P = "hy_v4_flash."


def rms(x: np.ndarray, w: np.ndarray | None, eps: float) -> np.ndarray:
    x = x.astype(np.float64)
    y = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    if w is not None:
        y = y * w
    return y.astype(F32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def e4m3(codes: np.ndarray) -> np.ndarray:
    """OCP FP8 E4M3 (fn: bias 7, no infinities) codes -> float32."""
    c = codes.astype(np.int32)
    e, m = (c >> 3) & 15, c & 7
    v = np.where(e == 0, m * 2.0 ** -9, (8 + m) * np.power(2.0, e - 10))
    return np.where(c & 128, -v, v).astype(F32)


def rope_pairs(x: np.ndarray, pos: np.ndarray, base: float) -> np.ndarray:
    """Interleaved-pair RoPE over the last axis (d even): pair j = (2j, 2j+1) turns by pos * base^(-2j/d)."""
    d = x.shape[-1]
    inv = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float64) / d))
    ang = pos[:, None].astype(np.float64) * inv[None, :]          # (T, d/2)
    c, s = np.cos(ang), np.sin(ang)
    shape = [1] * x.ndim
    shape[0], shape[-1] = ang.shape[0], ang.shape[1]
    c, s = c.reshape(shape), s.reshape(shape)
    x0, x1 = x[..., 0::2].astype(np.float64), x[..., 1::2].astype(np.float64)
    out = np.empty(x.shape, dtype=np.float64)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x1 * c + x0 * s
    return out.astype(F32)


class Hy4Ref:
    def __init__(self, path: str):
        p = pathlib.Path(path)
        m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", p.name)
        paths = [p] if not m else [p.with_name("%s-%05d-of-%s.gguf" % (m.group(1), i, m.group(3)))
                                   for i in range(1, int(m.group(3)) + 1)]
        self.where = {}
        self.meta = None
        for sp in paths:
            g = GGUFFile(sp)
            mm = np.memmap(sp, dtype=np.uint8, mode="r")
            if self.meta is None:
                self.meta = g.metadata
            for t in g.tensors:
                self.where[t.name] = (g, t, mm)
        md = self.meta
        self.n_embd = int(md[P + "embedding_length"])
        self.n_layers = int(md[P + "block_count"])
        self.heads = [int(h) for h in md[P + "attention.head_count"]]
        self.swa = [bool(b) for b in md[P + "attention.sliding_window_pattern"]]
        self.window = int(md[P + "attention.sliding_window"])
        self.eps = float(md[P + "attention.layer_norm_rms_epsilon"])
        self.hc = int(md[P + "hyper_connection.count"])
        self.hc_eps = float(md[P + "hyper_connection.epsilon"])
        self.hc_mag = float(md[P + "hyper_connection.magnitude"])
        self.base_dsa = float(md[P + "rope.freq_base"])
        self.base_swa = float(md[P + "rope.freq_base_swa"])
        self.n_rot = int(md[P + "rope.dimension_count"])
        self.kv_lora = int(md[P + "attention.kv_lora_rank"])
        self.qk_head = int(md[P + "attention.key_length_mla"])     # 256 = 192 nope + 64 rope
        self.v_head = int(md[P + "attention.value_length_mla"])
        self.n_exp_used = int(md[P + "expert_used_count"])
        self.w_scale = float(md[P + "expert_weights_scale"])
        self.dense_lead = int(md[P + "leading_dense_block_count"])
        self.idx_heads = int(md[P + "attention.indexer.head_count"])
        self.idx_key = int(md[P + "attention.indexer.key_length"])
        self.idx_top_k = int(md[P + "attention.indexer.top_k"])
        self.kpool = int(md[P + "attention.indexer.kpool"])
        self.swiglu = [float(v) for v in md[P + "swiglu_clamp_exp"]]
        self.eng_layers = [int(v) for v in md[P + "ple.layers"]]
        self.eng_dim = int(md[P + "embedding_length_per_layer_input"])
        self.eng_heads = int(md[P + "ple.heads_per_ngram"])
        self.eng_ngram = int(md[P + "ple.ngram_size"])
        self.eng_mult = [int(v) for v in md[P + "ple.layer_multipliers"]]
        self.eng_off = [int(v) for v in md[P + "ple.head_offsets"]]
        self.eng_size = [int(v) for v in md[P + "ple.head_vocab_sizes"]]
        self.eng_pad = int(md[P + "engram.pad_token_id"])
        self.tmap = np.asarray(md[P + "engram.token_map"], dtype=np.int64)
        # the Engram norms' epsilon (HF's nn.RMSNorm without eps: the input dtype's - bf16 2^-7 in a bf16 run);
        # the FP8 rows carry no scale in the GGUF, so their per-head norm takes a negligible one
        self.eng_eps = float(os.environ.get("HY4_ENG_EPS", str(2.0 ** -7)))
        self.eng_row_eps = float(os.environ.get("HY4_ENG_ROW_EPS", "1e-12"))
        self.eng_vfirst = os.environ.get("HY4_ENG_VFIRST", "0") == "1"
        self._cache = {}
        self.dumps = {}

    # ---- weights: (out, in) float32, as y = W @ x
    def t(self, name: str) -> np.ndarray:
        if name in self._cache:
            return self._cache[name]
        g, t, mm = self.where[name]
        n = t.expected_bytes()
        raw = mm[g.data_start + t.offset: g.data_start + t.offset + n]
        v = gguf_io.dequantize_any(np.asarray(raw), t.type_id).astype(F32).reshape(tuple(reversed(t.shape)))
        if v.size < 64 << 20:
            self._cache[name] = v
        return v

    def lt(self, il: int, base: str) -> np.ndarray:
        return self.t("blk.%d.%s" % (il, base))

    def has(self, il: int, base: str) -> bool:
        return ("blk.%d.%s" % (il, base)) in self.where

    def expert(self, name: str, e: int) -> np.ndarray:
        key = (name, e)
        if key in self._cache:
            return self._cache[key]
        g, t, mm = self.where[name]
        ne0, ne1, ne2 = t.shape
        rb = gguf_io.row_bytes(t.type_id, ne0)
        start = g.data_start + t.offset + e * ne1 * rb
        raw = mm[start: start + ne1 * rb]
        v = gguf_io.dequantize_any(np.asarray(raw), t.type_id).astype(F32).reshape(ne1, ne0)
        self._cache[key] = v
        return v

    def eng_rows(self, rows: np.ndarray) -> np.ndarray:
        g, t, mm = self.where["per_layer_token_embd.weight"]
        d = int(t.shape[0])
        base = g.data_start + t.offset
        out = np.empty((rows.size, d), dtype=F32)
        for k, r in enumerate(rows.reshape(-1)):
            out[k] = e4m3(np.asarray(mm[base + int(r) * d: base + (int(r) + 1) * d]))
        return out.reshape(rows.shape + (d,))

    def cb(self, name: str, x) -> None:
        self.dumps[name] = np.array(x, dtype=F32, copy=True)

    # ---- the hyper-connections
    def hc_pre(self, R, fn, scale, base):
        T, hc, E = R.shape
        flat = R.reshape(T, hc * E).astype(np.float64)
        inv = 1.0 / np.sqrt((flat * flat).mean(axis=1, keepdims=True) + self.eps)
        mixes = (flat @ fn.T.astype(np.float64)) * inv                 # (T, 2 hc)
        pre = sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + self.hc_eps
        post = self.hc_mag * sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc]) + self.hc_eps
        x = (pre[:, :, None] * R).sum(axis=1)
        return x.astype(F32), post.astype(F32)

    def head(self, R):
        T, hc, E = R.shape
        flat = R.reshape(T, hc * E).astype(np.float64)
        inv = 1.0 / np.sqrt((flat * flat).mean(axis=1, keepdims=True) + self.eps)
        mixes = (flat @ self.t("output_hc_fn.weight").T.astype(np.float64)) * inv
        pre = sigmoid(mixes * self.t("output_hc_scale.weight")[0] + self.t("output_hc_base.weight")) + self.hc_eps
        return (pre[:, :, None] * R).sum(axis=1).astype(F32)

    # ---- Engram
    def engram_rows(self, il: int, tokens: np.ndarray) -> np.ndarray:
        """(T, 16) table rows of layer il's heads for the token sequence (positions 0..T-1)."""
        li = self.eng_layers.index(il)
        T = tokens.size
        cpad = int(self.tmap[self.eng_pad])
        c = np.where(tokens < self.tmap.size, self.tmap[np.minimum(tokens, self.tmap.size - 1)], cpad)
        mult = self.eng_mult[3 * li: 3 * li + 3]
        mixed = []
        for k in range(self.eng_ngram):
            ck = np.full(T, cpad, dtype=np.int64)
            ck[k:] = c[:T - k] if k else c
            mixed.append([int(v) * mult[k] for v in ck])            # python ints: exact
        rows = np.empty((T, 2 * self.eng_heads), dtype=np.int64)
        for t in range(T):
            for n in range(2, self.eng_ngram + 1):
                mix = mixed[0][t]
                for k in range(1, n):
                    mix ^= mixed[k][t]
                assert 0 <= mix < 2 ** 63, "the n-gram mix left int64"
                for j in range(self.eng_heads):
                    h = (n - 2) * self.eng_heads + j
                    gh = 16 * li + h
                    rows[t, h] = self.eng_off[gh] + mix % self.eng_size[gh]
        return rows

    def engram(self, il: int, R: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        rows = self.engram_rows(il, tokens)
        emb = self.eng_rows(rows)                                     # (T, 16, 96)
        lnw = self.lt(il, "engram_lookup_norm.weight")                # (16, 96)
        emb = rms(emb, None, self.eng_row_eps) * lnw[None]
        e = emb.reshape(tokens.size, -1)                              # (T, 1536)
        kv = e.astype(np.float64) @ self.lt(il, "engram_kv.weight").T.astype(np.float64)
        b = self.lt(il, "engram_kv_b.weight").reshape(-1)
        kv = kv + b[None, :]
        E, hc = self.n_embd, self.hc
        if self.eng_vfirst:
            val, keys = kv[:, :E], kv[:, E:].reshape(-1, hc, E)
        else:
            keys, val = kv[:, :hc * E].reshape(-1, hc, E), kv[:, hc * E:]
        gw = self.lt(il, "engram_gate.weight")                        # (4, E): the key and query norms' weights
        sc = self.lt(il, "engram_hc_scale.weight")                    # (4, E)
        pg = self.lt(il, "engram_pre_gamma.weight")
        main = rms(val, None, self.eng_eps) * pg[None]
        out = np.zeros_like(R)
        for c in range(hc):
            kn = rms(keys[:, c], None, self.eng_eps).astype(np.float64)
            qn = rms(R[:, c], None, self.eng_eps).astype(np.float64)
            g = (kn * qn * gw[c]).sum(axis=1) / np.sqrt(E)
            g = np.sqrt(np.maximum(np.abs(g), 1e-6)) * np.sign(g)
            out[:, c] = (sigmoid(g)[:, None] * main + sc[c][None] * main).astype(F32)
        keep = (tokens != 120120) & (tokens != 120124) & (tokens != self.eng_pad)
        out[~keep] = 0.0
        self.cb("engram-%d" % il, out)
        return out

    # ---- the compressed indexer: (T, T) bool, True where a query may NOT attend (past the causal mask)
    def indexer_block(self, il, x, qr, pos):
        T = x.shape[0]
        if T <= self.idx_top_k:
            return np.zeros((T, T), dtype=bool)
        kp = self.kpool
        tk = rms(x @ self.lt(il, "indexer.attn_k.weight").T, self.lt(il, "indexer.k_norm.weight"), self.eps)
        G = T // kp
        grouped = tk[:G * kp].reshape(G, kp, self.idx_key).astype(np.float64)
        vals = grouped @ self.lt(il, "indexer_compressor_value.weight").T
        gates = grouped @ self.lt(il, "indexer_compressor_gate.weight").T + self.lt(il, "indexer_compressor_ape.weight")
        gates = np.exp(gates - gates.max(axis=1, keepdims=True))
        gates = gates / gates.sum(axis=1, keepdims=True)
        pooled = rms((vals * gates).sum(axis=1), self.lt(il, "indexer_compressor_norm.weight"), self.eps)
        q = (qr @ self.lt(il, "indexer.attn_q_b.weight").T).reshape(T, self.idx_heads, self.idx_key)
        nd = self.idx_key - self.n_rot
        q = np.concatenate([q[..., :nd], rope_pairs(q[..., nd:], pos, self.base_dsa)], axis=-1)
        k = np.concatenate([pooled[:, :nd], rope_pairs(pooled[:, nd:], pos[:G * kp:kp], self.base_dsa)], axis=-1)
        w = (x @ self.lt(il, "indexer.proj.weight").T).astype(np.float64) * (self.idx_heads ** -0.5 * self.idx_key ** -0.5)
        sc = np.maximum(np.einsum("thd,gd->thg", q.astype(np.float64), k.astype(np.float64)), 0.0)
        logit = np.einsum("th,thg->tg", w, sc)
        own = np.arange(T) // kp
        logit[np.arange(G)[None, :] >= own[:, None]] = -np.inf
        n_top = min(self.idx_top_k // kp - 1, G)
        blocked = np.ones((T, T), dtype=bool)
        for t in range(T):
            order = np.argsort(-logit[t], kind="stable")[:n_top]
            sel = np.concatenate([order, [own[t]]])
            cells = (sel[:, None] * kp + np.arange(kp)[None, :]).reshape(-1)
            blocked[t, np.minimum(cells, T - 1)] = False
        return blocked

    # ---- attention
    def attention(self, il, x, pos):
        T = x.shape[0]
        nh = self.heads[il]
        nope = self.qk_head - self.n_rot
        base = self.base_swa if self.swa[il] else self.base_dsa
        qr = rms(x @ self.lt(il, "attn_q_a.weight").T, self.lt(il, "attn_q_a_norm.weight"), self.eps)
        q = (qr @ self.lt(il, "attn_q_b.weight").T).reshape(T, nh, self.qk_head)
        q_nope, q_rot = q[..., :nope], rope_pairs(q[..., nope:], pos, base)
        ckv = x @ self.lt(il, "attn_kv_a_mqa.weight").T
        lat = rms(ckv[:, :self.kv_lora], self.lt(il, "attn_kv_a_norm.weight"), self.eps)
        k_rot = rope_pairs(ckv[:, self.kv_lora:], pos, base)
        kb = self.lt(il, "attn_k_b.weight").reshape(nh, self.kv_lora, nope)      # q_nope . k = (kb q) . latent
        vb = self.lt(il, "attn_v_b.weight").reshape(nh, self.v_head, self.kv_lora)
        self.cb("lat-%d" % il, lat)
        self.cb("krot-%d" % il, k_rot)
        q_abs = np.einsum("hrd,thd->thr", kb.astype(np.float64), q_nope.astype(np.float64))
        s = np.einsum("thr,sr->hts", q_abs, lat.astype(np.float64)) + \
            np.einsum("thd,sd->hts", q_rot.astype(np.float64), k_rot.astype(np.float64))
        s *= self.qk_head ** -0.5
        tt, ss = np.arange(T)[:, None], np.arange(T)[None, :]
        blocked = ss > tt
        if self.swa[il]:
            blocked |= ss <= tt - self.window
        else:
            blocked |= self.indexer_block(il, x, qr, pos)
        s = np.where(blocked[None], -np.inf, s)
        sink = self.lt(il, "attn_sinks.weight").astype(np.float64)
        mx = np.maximum(s.max(axis=2), sink[:, None])                            # (h, t)
        p = np.exp(s - mx[:, :, None])
        den = p.sum(axis=2) + np.exp(sink[:, None] - mx)
        p = p / den[:, :, None]
        ctx = np.einsum("hts,sr->thr", p, lat.astype(np.float64))               # (T, h, 512)
        o = np.einsum("hvr,thr->thv", vb.astype(np.float64), ctx)               # (T, h, 256)
        gate = sigmoid((x @ self.lt(il, "attn_gate.weight").T).reshape(T, nh, self.v_head).astype(np.float64))
        o = (o * gate).reshape(T, nh * self.v_head).astype(F32)
        out = o @ self.lt(il, "attn_output.weight").T
        self.cb("attn_raw-%d" % il, out)
        return rms(out, self.lt(il, "post_attention_norm.weight"), self.eps)

    # ---- FFN
    def ffn(self, il, x):
        if il < self.dense_lead:
            g = x @ self.lt(il, "ffn_gate.weight").T
            u = x @ self.lt(il, "ffn_up.weight").T
            y = (silu(g.astype(np.float64)) * u).astype(F32) @ self.lt(il, "ffn_down.weight").T
            return rms(y, self.lt(il, "post_ffw_norm.weight"), self.eps)
        T = x.shape[0]
        logits = x.astype(np.float64) @ self.lt(il, "ffn_gate_inp.weight").T.astype(np.float64)
        probs = sigmoid(logits)
        choice = probs + self.lt(il, "exp_probs_b.bias")[None, :]
        xl = x @ self.lt(il, "ffn_routed_down.weight").T                          # (T, 1536)
        self.cb("rlog-%d" % il, logits.astype(F32))
        self.cb("rlat-%d" % il, xl)
        lim = self.swiglu[il]
        y = np.zeros_like(xl, dtype=np.float64)
        sel_all = np.empty((T, self.n_exp_used), dtype=np.int64)
        for t in range(T):
            sel = np.argsort(-choice[t], kind="stable")[:self.n_exp_used]
            sel_all[t] = sel
            w = probs[t, sel]
            w = w / (w.sum() + 1e-20) * self.w_scale
            for i, e in enumerate(sel):
                g = self.expert("blk.%d.ffn_gate_exps.weight" % il, e) @ xl[t]
                u = self.expert("blk.%d.ffn_up_exps.weight" % il, e) @ xl[t]
                if lim > 0:
                    g = np.minimum(g, lim)
                    u = np.clip(u, -lim, lim)
                h = (silu(g.astype(np.float64)) * u).astype(F32)
                y[t] += w[i] * (self.expert("blk.%d.ffn_down_exps.weight" % il, e) @ h)
        self.dumps["sel-%d" % il] = sel_all
        self.cb("rsum-%d" % il, y.astype(F32))
        routed = y.astype(F32) @ self.lt(il, "ffn_routed_up.weight").T
        g = x @ self.lt(il, "ffn_gate_shexp.weight").T
        u = x @ self.lt(il, "ffn_up_shexp.weight").T
        sh = (silu(g.astype(np.float64)) * u).astype(F32) @ self.lt(il, "ffn_down_shexp.weight").T
        self.cb("rup-%d" % il, routed)
        self.cb("shx-%d" % il, sh)
        return rms(routed + sh, self.lt(il, "post_ffw_norm.weight"), self.eps)

    def forward(self, tokens, n_layers: int | None = None, logits: bool = True):
        tokens = np.asarray(tokens, dtype=np.int64)
        T = tokens.size
        pos = np.arange(T)
        emb = self.t("token_embd.weight")[tokens]
        R = np.repeat(emb[:, None, :], self.hc, axis=1).astype(F32)
        L = self.n_layers if n_layers is None else n_layers
        for il in range(L):
            if il in self.eng_layers:
                R = R + self.engram(il, R, tokens)
            x, post = self.hc_pre(R, self.lt(il, "hc_attn_fn.weight"), self.lt(il, "hc_attn_scale.weight"),
                                  self.lt(il, "hc_attn_base.weight"))
            x = rms(x, self.lt(il, "attn_norm.weight"), self.eps)
            self.cb("attn_norm-%d" % il, x)
            a = self.attention(il, x, pos)
            self.cb("attn_out-%d" % il, a)
            R = (R + post[:, :, None] * a[:, None, :]).astype(F32)
            x, post = self.hc_pre(R, self.lt(il, "hc_ffn_fn.weight"), self.lt(il, "hc_ffn_scale.weight"),
                                  self.lt(il, "hc_ffn_base.weight"))
            x = rms(x, self.lt(il, "ffn_norm.weight"), self.eps)
            self.cb("ffn_norm-%d" % il, x)
            f = self.ffn(il, x)
            self.cb("ffn_out-%d" % il, f)
            R = (R + post[:, :, None] * f[:, None, :]).astype(F32)
            self.cb("l_out-%d" % il, R)
            print("layer %d done" % il, file=sys.stderr, flush=True)
        if L < self.n_layers or not logits:
            return None
        h = rms(self.head(R), self.t("output_norm.weight"), self.eps)
        self.cb("result_norm", h)
        out = h @ self.t("output.weight").T
        self.cb("logits", out)
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--tokens", required=True, help="comma-separated token ids")
    ap.add_argument("--layers", type=int, default=None, help="run only the first N layers (no logits)")
    ap.add_argument("--dump", default=None, help="write every recorded tensor as <name>.npy into this directory")
    a = ap.parse_args()
    m = Hy4Ref(a.gguf)
    toks = [int(v) for v in a.tokens.split(",") if v]
    out = m.forward(toks, a.layers)
    if out is not None:
        for t in range(len(toks)):
            top = np.argsort(-out[t])[:5]
            print(t, toks[t], "->", list(map(int, top)), [round(float(out[t, i]), 3) for i in top])
    if a.dump:
        d = pathlib.Path(a.dump)
        d.mkdir(parents=True, exist_ok=True)
        for k, v in m.dumps.items():
            np.save(d / (k + ".npy"), v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
