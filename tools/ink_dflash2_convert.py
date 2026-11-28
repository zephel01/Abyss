"""tools/ink_dflash2_convert.py - Inkling 2 Mini's DFlash 2 drafter (incoai/Inkling-2-Mini-DFlash2, Apache-2.0, one BF16
model.safetensors + config.json) to the GGUF Abyss's drafter path reads (src/core/ink_dflash2.cu), with no llama.cpp,
gguf-py, safetensors or torch.

usage: ink_dflash2_convert.py <hf folder> <out.gguf> [--outtype=q8_0|bf16]

The names are llama.cpp's dflash ones (arch "dflash", as the Qwen3.8 DFlash 2 GGUFs Abyss reads): blk.N.attn_q / _k / _v
/ _output, ffn_gate / _up / _down, attn_conv_proj / ffn_conv_proj (the dynamic kernels' projections, rows [side][tap]
[group]), attn_conv_base / ffn_conv_base ([side][tap][channel], F32), attn_norm / ffn_norm / attn_q_norm / attn_k_norm,
fc (the 4 taps -> the width), enc.output_norm (hidden_norm), output_norm (norm), selector_hidden /
selector_predecessor / selector_successor (the candidate selector).  Every tensor is copied as it is (the norms are
plain weights, no 1 + w; NEOX RoPE as stored); the 2-D weights become --outtype (default Q8_0, ggml's reference
quantizer in numpy), the vectors and conv bases F32.

dflash.target_layers holds the target's own tap ids [5, 23, 41, 48] (Inkling's layer visits 5 / 23 / 41 - the residual
entering them - and 48, the final normed row: the DSpark drafter's taps), not llama.cpp's id + 1.
dflash.selector_softcap_mode "after_correction": the selector's score is cap tanh((logit + correction) / cap).
"""
from __future__ import annotations

import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_io as gio  # noqa: E402

TAPS = [5, 23, 41, 48]


class SafeTensors:
    """The tensors of a folder's *.safetensors files, read as memmaps (BF16 / F32)."""

    def __init__(self, folder: Path):
        self.t = {}
        for f in sorted(folder.glob("*.safetensors")):
            with open(f, "rb") as h:
                n = struct.unpack("<Q", h.read(8))[0]
                hdr = json.loads(h.read(n))
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.t[k] = (f, v["dtype"], v["shape"], 8 + n + a, b - a)

    def f32(self, name: str) -> np.ndarray:
        f, dt, shape, off, nb = self.t[name]
        if dt == "BF16":
            u = np.memmap(f, dtype="<u2", mode="r", offset=off, shape=(nb // 2,))
            return (u.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if dt == "F32":
            return np.array(np.memmap(f, dtype="<f4", mode="r", offset=off, shape=(nb // 4,))).reshape(shape)
        raise ValueError("%s: dtype %s" % (name, dt))


def check(cfg: dict) -> None:
    dc = cfg.get("dflash_config", {})
    want = {"hidden_size": 2304, "num_hidden_layers": 5, "num_attention_heads": 32, "num_key_value_heads": 8,
            "head_dim": 128}
    bad = [k for k, v in want.items() if cfg.get(k) != v]
    if cfg.get("architectures") != ["DFlash2DraftModel"]:
        bad.append("architectures")
    if dc.get("target_layer_ids") != TAPS:
        bad.append("dflash_config.target_layer_ids")
    for k, v in {"block_size": 8, "conv_kernel_size": 2, "conv_group_size": 16, "selector_rank": 256,
                 "selector_top_k": 16}.items():
        if dc.get(k) != v:
            bad.append("dflash_config." + k)
    if cfg.get("sliding_window") != 2048 or cfg.get("is_causal", False):
        bad.append("sliding_window / is_causal")
    if bad:
        raise SystemExit("not Inkling 2 Mini's DFlash 2 drafter (config.json: %s)" % ", ".join(bad))


def tensors(st: SafeTensors, cfg: dict):
    """(gguf name, numpy-order float32 data function, kind): kind 'w' a 2-D weight (--outtype), 'f' F32."""
    yield "fc.weight", lambda: st.f32("fc.weight"), "w"
    yield "enc.output_norm.weight", lambda: st.f32("hidden_norm.weight"), "f"
    yield "output_norm.weight", lambda: st.f32("norm.weight"), "f"
    yield "selector_hidden.weight", lambda: st.f32("candidate_selector.hidden_projection.weight"), "w"
    yield "selector_predecessor.weight", lambda: st.f32("candidate_selector.predecessor_codebook"), "w"
    yield "selector_successor.weight", lambda: st.f32("candidate_selector.successor_codebook"), "w"
    for il in range(cfg["num_hidden_layers"]):
        L, B = "layers.%d." % il, "blk.%d." % il
        for src, dst in (("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
                         ("self_attn.o_proj", "attn_output"), ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"),
                         ("mlp.down_proj", "ffn_down"), ("attention_conv.kernel_projection", "attn_conv_proj"),
                         ("mlp_conv.kernel_projection", "ffn_conv_proj")):
            yield B + dst + ".weight", (lambda n=L + src + ".weight": st.f32(n)), "w"
        yield B + "attn_conv_base", (lambda n=L + "attention_conv.base_kernel": st.f32(n)), "f"
        yield B + "ffn_conv_base", (lambda n=L + "mlp_conv.base_kernel": st.f32(n)), "f"
        for src, dst in (("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm"),
                         ("self_attn.q_norm", "attn_q_norm"), ("self_attn.k_norm", "attn_k_norm")):
            yield B + dst + ".weight", (lambda n=L + src + ".weight": st.f32(n)), "f"


def main() -> int:
    flags = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:] if a.startswith("--"))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        return 2
    src, out = Path(args[0]), Path(args[1])
    wt = {"q8_0": 8, "bf16": 30}.get(flags.get("--outtype", "q8_0").lower())
    if wt is None:
        raise SystemExit("--outtype: q8_0 or bf16")
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    check(cfg)
    dc = cfg["dflash_config"]
    st = SafeTensors(src)
    w = gio.Writer(out, arch="dflash")
    w.add_string("general.name", "Inkling-2-Mini-DFlash2")
    w.add_string("general.license", "apache-2.0")
    w.add_string("general.source.repo", "incoai/Inkling-2-Mini-DFlash2")
    a = "dflash."
    w.add_string(a + "target_arch", "inkling")
    w.add_uint32(a + "block_count", cfg["num_hidden_layers"])
    w.add_uint32(a + "embedding_length", cfg["hidden_size"])
    w.add_uint32(a + "feed_forward_length", cfg["intermediate_size"])
    w.add_uint32(a + "attention.head_count", cfg["num_attention_heads"])
    w.add_uint32(a + "attention.head_count_kv", cfg["num_key_value_heads"])
    w.add_uint32(a + "attention.key_length", cfg["head_dim"])
    w.add_uint32(a + "attention.value_length", cfg["head_dim"])
    w.add_float32(a + "attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"])
    w.add_uint32(a + "attention.sliding_window", cfg["sliding_window"])
    w.add_bool(a + "attention.causal", False)
    w.add_float32(a + "rope.freq_base", cfg.get("rope_parameters", {}).get("rope_theta", 10000.0))
    w.add_uint32(a + "rope.dimension_count", cfg["head_dim"])
    w.add_uint32(a + "block_size", dc["block_size"])
    w.add_uint32(a + "conv_kernel_size", dc["conv_kernel_size"])
    w.add_uint32(a + "conv_group_size", dc["conv_group_size"])
    w.add_uint32(a + "selector_rank", dc["selector_rank"])
    w.add_uint32(a + "selector_top_k", dc["selector_top_k"])
    w.add_float32(a + "final_logit_softcapping", dc.get("final_logit_softcapping", 30.0))
    w.add_string(a + "selector_softcap_mode", dc.get("selector_softcap_mode", "before_correction"))
    w.add_array(a + "target_layers", TAPS, gio.I32)
    w.add_uint32("tokenizer.ggml.mask_token_id", dc["mask_token_id"])
    total = 0
    for name, fn, kind in tensors(st, cfg):
        x = fn()
        shape = list(reversed(x.shape))   # (GGUF order: the row length first)
        t = wt if kind == "w" and x.ndim == 2 else 0
        data = gio.quantize(np.ascontiguousarray(x, dtype=np.float32), t)
        w.add_tensor(name, shape, t, data)
        total += data.nbytes
    t0 = time.time()
    w.write()
    print("%s: %d tensors, %.1f MB (%s weights), %.0f s" % (out, len(w.tensors), total / 1e6, gio.type_name(wt),
                                                          time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
