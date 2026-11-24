"""tools/memt_convert.py - MemT-1.0-Nano (MemTensor, a Qwen3.6-35B-A3B backbone + native memory blocks) from its
Hugging Face safetensors straight to a GGUF Abyss reads, with no llama.cpp, gguf-py, safetensors or torch.

usage: memt_convert.py <hf folder> <out.gguf> --meta=<a qwen35moe GGUF> [--types-from=<GGUF>] [--outtype=q8_0|bf16]
                       [--gpuq=<libgpuq_cuda.so>] [--only=<regex>]
  --meta        a GGUF of the same family (Ornith-1.5-35B-A3B's) whose keys and tokenizer arrays are copied: the
                tokenizer is checked token by token against the model's tokenizer.json (ids that tokenizer.json lacks
                become [PAD<id>], type 5), the chat template is the model's chat_template.jinja
  --types-from  each backbone tensor gets the type the same-named tensor has in that GGUF (e.g. a Q4_K_M of the same
                shapes); k-quants go through trench-gguf's GPU quantizer (--gpuq), byte-identical to ggml's.
                Without it every 2-D backbone weight is --outtype (default q8_0, ggml's reference Q8_0 in numpy)
  The memory blocks stay BF16 (projections) / F32 (vectors): blk.N.memt_in.weight [2048 -> 20992] (rows q | k | v |
  b | w | f | g), memt_f_down / memt_g_down [256 -> 4096], memt_decay_bias [4096], memt_decay_scale_log [16],
  memt_o_norm.weight [256]; keys qwen35moe.memt.*.

The backbone's layout follows the Qwen3.5 GGUF convention Abyss's qwen35moe path reads (written down here from the
math, worklog 2026-11-25_memt-nano.md): the zero-centred RMSNorm weights stored as 1 + w (in float32; the gated
delta-net's output norm is a plain weight), A_log as -exp(A_log), the delta-net's value heads reordered from grouped by
key head (k0: v0 v1, k1: v0 v1 ...) to tiled (v0 of every key head, then v1 ...), the conv kernel [channels, 4], the
fused expert gate_up split into gate (first half) and up.
"""
from __future__ import annotations

import ctypes
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

P = "model.language_model."


# ------------------------------------------------------------------------------------------------ safetensors
class SafeTensors:
    """The tensors of a folder's *.safetensors files: name -> (file, dtype, shape, offset); read as a memmap."""

    def __init__(self, folder: Path):
        self.t = {}
        for f in sorted(folder.glob("*.safetensors")):
            with open(f, "rb") as h:
                n = struct.unpack("<Q", h.read(8))[0]
                hdr = json.loads(h.read(n))
            base = 8 + n
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.t[k] = (f, v["dtype"], v["shape"], base + a, b - a)

    def f32(self, name: str) -> np.ndarray:
        f, dt, shape, off, nb = self.t[name]
        if dt == "BF16":
            u = np.memmap(f, dtype="<u2", mode="r", offset=off, shape=(nb // 2,))
            return (u.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if dt == "F32":
            return np.array(np.memmap(f, dtype="<f4", mode="r", offset=off, shape=(nb // 4,))).reshape(shape)
        raise ValueError("%s: dtype %s" % (name, dt))

    def bf16_bits(self, name: str) -> np.ndarray:
        f, dt, shape, off, nb = self.t[name]
        assert dt == "BF16", name
        return np.array(np.memmap(f, dtype="<u2", mode="r", offset=off, shape=(nb // 2,))).reshape(shape)


# ------------------------------------------------------------------------------------------------ layout
def tiled_v(x: np.ndarray, axis: int, n_k: int, per_k: int, hd: int) -> np.ndarray:
    """Value heads grouped by key head -> tiled order along axis (the size there is n_k * per_k * hd)."""
    x = np.moveaxis(x, axis, 0)
    rest = x.shape[1:]
    x = x.reshape(n_k, per_k, hd, *rest).swapaxes(0, 1).reshape(n_k * per_k * hd, *rest)
    return np.ascontiguousarray(np.moveaxis(x, 0, axis))


def backbone(st: SafeTensors, cfg: dict):
    """Yields (gguf name, float32 array in numpy order = GGUF shape reversed, kind) for the backbone; kind 'w' a
    weight matrix (quantized), 'f' kept F32."""
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    kd, vd = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    per = nv // nk
    qk = 2 * nk * kd
    yield "token_embd.weight", lambda: st.f32(P + "embed_tokens.weight"), "w"
    for il, lt in enumerate(cfg["layer_types"]):
        L, B = P + "layers.%d." % il, "blk.%d." % il
        yield B + "attn_norm.weight", (lambda L=L: 1.0 + st.f32(L + "input_layernorm.weight")), "f"
        yield B + "post_attention_norm.weight", (lambda L=L: 1.0 + st.f32(L + "post_attention_layernorm.weight")), "f"
        if lt == "linear_attention":
            A = L + "linear_attn."

            def qkv(A=A):
                w = st.f32(A + "in_proj_qkv.weight")
                return np.concatenate([w[:qk], tiled_v(w[qk:], 0, nk, per, vd)], 0)

            def conv(A=A):
                w = st.f32(A + "conv1d.weight").reshape(-1, cfg["linear_conv_kernel_dim"])
                return np.concatenate([w[:qk], tiled_v(w[qk:], 0, nk, per, vd)], 0)

            yield B + "attn_qkv.weight", qkv, "w"
            yield B + "attn_gate.weight", (lambda A=A: tiled_v(st.f32(A + "in_proj_z.weight"), 0, nk, per, vd)), "w"
            yield B + "ssm_alpha.weight", (lambda A=A: tiled_v(st.f32(A + "in_proj_a.weight"), 0, nk, per, 1)), "w"
            yield B + "ssm_beta.weight", (lambda A=A: tiled_v(st.f32(A + "in_proj_b.weight"), 0, nk, per, 1)), "w"
            yield B + "ssm_conv1d.weight", conv, "f"
            yield B + "ssm_dt.bias", (lambda A=A: tiled_v(st.f32(A + "dt_bias"), 0, nk, per, 1)), "f"
            yield B + "ssm_a", (lambda A=A: -np.exp(tiled_v(st.f32(A + "A_log"), 0, nk, per, 1))), "f"
            yield B + "ssm_norm.weight", (lambda A=A: st.f32(A + "norm.weight")), "f"
            yield B + "ssm_out.weight", (lambda A=A: tiled_v(st.f32(A + "out_proj.weight"), 1, nk, per, vd)), "w"
        else:
            A = L + "self_attn."
            yield B + "attn_q.weight", (lambda A=A: st.f32(A + "q_proj.weight")), "w"
            yield B + "attn_k.weight", (lambda A=A: st.f32(A + "k_proj.weight")), "w"
            yield B + "attn_v.weight", (lambda A=A: st.f32(A + "v_proj.weight")), "w"
            yield B + "attn_output.weight", (lambda A=A: st.f32(A + "o_proj.weight")), "w"
            yield B + "attn_q_norm.weight", (lambda A=A: 1.0 + st.f32(A + "q_norm.weight")), "f"
            yield B + "attn_k_norm.weight", (lambda A=A: 1.0 + st.f32(A + "k_norm.weight")), "f"
        M = L + "mlp."
        I = cfg["moe_intermediate_size"]
        yield B + "ffn_gate_inp.weight", (lambda M=M: st.f32(M + "gate.weight")), "f"
        yield B + "ffn_gate_exps.weight", (lambda M=M: st.f32(M + "experts.gate_up_proj")[:, :I, :]), "w"
        yield B + "ffn_up_exps.weight", (lambda M=M: st.f32(M + "experts.gate_up_proj")[:, I:, :]), "w"
        yield B + "ffn_down_exps.weight", (lambda M=M: st.f32(M + "experts.down_proj")), "w"
        yield B + "ffn_gate_inp_shexp.weight", (lambda M=M: st.f32(M + "shared_expert_gate.weight").reshape(-1)), "f"
        yield B + "ffn_gate_shexp.weight", (lambda M=M: st.f32(M + "shared_expert.gate_proj.weight")), "w"
        yield B + "ffn_up_shexp.weight", (lambda M=M: st.f32(M + "shared_expert.up_proj.weight")), "w"
        yield B + "ffn_down_shexp.weight", (lambda M=M: st.f32(M + "shared_expert.down_proj.weight")), "w"
    yield "output_norm.weight", lambda: 1.0 + st.f32(P + "norm.weight"), "f"
    yield "output.weight", lambda: st.f32("lm_head.weight"), "w"


def memory(st: SafeTensors, cfg: dict):
    """(gguf name, numpy-order shape, type, data function) of the memory blocks: BF16 bits as stored, F32 vectors."""
    for il, lt in enumerate(cfg["layer_types"]):
        if lt != "full_attention":
            continue
        M, B = P + "layers.%d.self_attn.memt." % il, "blk.%d." % il
        for src, dst in (("in_proj.weight", "memt_in.weight"), ("f_down.weight", "memt_f_down.weight"),
                         ("g_down.weight", "memt_g_down.weight")):
            shape = st.t[M + src][2]
            yield B + dst, shape, "BF16", (lambda M=M, src=src: st.bf16_bits(M + src))
        for src, dst in (("decay_bias", "memt_decay_bias"), ("decay_scale_log", "memt_decay_scale_log"),
                         ("o_norm.weight", "memt_o_norm.weight")):
            shape = st.t[M + src][2]
            yield B + dst, shape, "F32", (lambda M=M, src=src: st.f32(M + src).astype(np.float32))


# ------------------------------------------------------------------------------------------------ quantizing
class GpuQ:
    def __init__(self, lib: str):
        self.q = ctypes.CDLL(lib)
        self.q.gpuq_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.q.gpuq_quantize.restype = ctypes.c_size_t
        self.q.gpuq_quantize.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                         ctypes.c_int64, ctypes.c_void_p]
        rc = self.q.gpuq_init(os.path.join(os.path.dirname(lib), "gpuq_tables.bin").encode(), 0)
        if rc != 0:
            sys.exit("gpuq_init %d" % rc)

    def __call__(self, x: np.ndarray, t: int) -> np.ndarray:
        n_per_row = x.shape[-1]
        rows = x.size // n_per_row
        out = np.empty(gio.nbytes(t, [n_per_row, rows]), dtype=np.uint8)
        step = max(1, (1 << 28) // (n_per_row * 4))   # (1 GiB of floats a call)
        rb = gio.row_bytes(t, n_per_row)
        x2 = x.reshape(rows, n_per_row)
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            c = np.ascontiguousarray(x2[r0:r1], dtype=np.float32)
            n = self.q.gpuq_quantize(t, c.ctypes.data, out[r0 * rb:].ctypes.data, r1 - r0, n_per_row, None)
            if n != (r1 - r0) * rb:
                sys.exit("gpuq_quantize: %d bytes for %d" % (n, (r1 - r0) * rb))
        return out


def q8_0(x: np.ndarray) -> np.ndarray:
    x2 = x.reshape(-1, x.shape[-1])
    step = max(1, (1 << 26) // x2.shape[1])
    return np.concatenate([gio.quantize(x2[r:r + step], 8) for r in range(0, x2.shape[0], step)])


def main() -> int:
    flags = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:] if a.startswith("--"))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2 or "--meta" not in flags:
        print(__doc__)
        return 2
    hf, out = Path(args[0]), Path(args[1])
    full = json.loads((hf / "config.json").read_text())
    cfg = full["backbone_configs"]["text_config"]
    mem = full["memt"]
    st = SafeTensors(hf)
    meta = gio.Reader(flags["--meta"])
    types = {}
    if flags.get("--types-from"):
        tr = gio.Reader(flags["--types-from"])
        types = {t.name: t.type for t in (tr.tensors if isinstance(tr.tensors, list) else tr.tensors.values())}
    dense = {"q8_0": 8, "bf16": 30}[flags.get("--outtype") or "q8_0"]
    gq = GpuQ(flags["--gpuq"]) if flags.get("--gpuq") else None
    only = re.compile(flags["--only"]) if flags.get("--only") else None

    w = gio.Writer(out, arch="qwen35moe")
    nl = cfg["num_hidden_layers"]
    skip = {"general.name", "general.basename", "general.size_label", "general.license", "general.license.link",
            "general.tags", "general.file_type", "general.sampling.top_k", "general.sampling.top_p",
            "general.sampling.temp", "qwen35moe.block_count", "qwen35moe.nextn_predict_layers",
            "tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.chat_template"}
    w.add_string("general.name", "MemT-1.0-Nano")
    w.add_string("general.license", "apache-2.0")
    gen = json.loads((hf / "generation_config.json").read_text())
    w.add_int32("general.sampling.top_k", gen.get("top_k", 40))
    w.add_float32("general.sampling.top_p", gen.get("top_p", 0.95))
    w.add_float32("general.sampling.temp", gen.get("temperature", 0.7))
    w.add_uint32("qwen35moe.block_count", nl)
    w.add_uint32("qwen35moe.nextn_predict_layers", 0)
    gio.copy_fields(meta, w, skip=lambda k: k in skip)
    # the hparams that must equal the meta file's (the same family, the same shapes)
    need = {"qwen35moe.embedding_length": cfg["hidden_size"], "qwen35moe.attention.head_count": cfg["num_attention_heads"],
            "qwen35moe.attention.head_count_kv": cfg["num_key_value_heads"], "qwen35moe.expert_count": cfg["num_experts"],
            "qwen35moe.expert_used_count": cfg["num_experts_per_tok"], "qwen35moe.attention.key_length": cfg["head_dim"],
            "qwen35moe.expert_feed_forward_length": cfg["moe_intermediate_size"],
            "qwen35moe.expert_shared_feed_forward_length": cfg["shared_expert_intermediate_size"],
            "qwen35moe.ssm.state_size": cfg["linear_key_head_dim"], "qwen35moe.ssm.group_count": cfg["linear_num_key_heads"],
            "qwen35moe.ssm.time_step_rank": cfg["linear_num_value_heads"],
            "qwen35moe.ssm.inner_size": cfg["linear_value_head_dim"] * cfg["linear_num_value_heads"],
            "qwen35moe.full_attention_interval": cfg["full_attention_interval"]}
    for k, v in need.items():
        if int(meta.fields[k].value) != v:
            sys.exit("%s: %s in the meta file, %s in config.json" % (k, meta.fields[k].value, v))
    if abs(float(meta.fields["qwen35moe.rope.freq_base"].value) - cfg["rope_parameters"]["rope_theta"]) > 1:
        sys.exit("rope_theta differs")
    # the tokenizer: the meta file's arrays where tokenizer.json has the same token, [PAD<id>] where it has none
    toks = list(meta.fields["tokenizer.ggml.tokens"].value)
    ttype = [int(x) for x in meta.fields["tokenizer.ggml.token_type"].value]
    tj = json.loads((hf / "tokenizer.json").read_text())
    inv = {i: t for t, i in tj["model"]["vocab"].items()}
    for a in tj["added_tokens"]:
        inv[a["id"]] = a["content"]
    merges = [" ".join(m) if isinstance(m, list) else m for m in tj["model"]["merges"]]
    if merges != list(meta.fields["tokenizer.ggml.merges"].value):
        sys.exit("merges differ from the meta file's")
    n_pad = 0
    for i in range(len(toks)):
        if i in inv:
            if inv[i] != toks[i]:
                sys.exit("token %d: %r vs %r" % (i, inv[i], toks[i]))
        elif not toks[i].startswith("[PAD"):
            toks[i], ttype[i] = "[PAD%d]" % i, 5
            n_pad += 1
    print("tokenizer: %d tokens, %d not in tokenizer.json made [PAD]" % (len(toks), n_pad))
    w.add_array("tokenizer.ggml.tokens", toks, gio.STRING)
    w.add_array("tokenizer.ggml.token_type", ttype, gio.I32)
    w.add_string("tokenizer.chat_template", (hf / "chat_template.jinja").read_text())
    w.add_float32("qwen35moe.memt.softcap", float(mem["output_gate_softcap_limit"]) if mem["output_gate_softcap_enabled"] else 0.0)
    w.add_uint32("qwen35moe.memt.chunk_tokens", int(mem["chunk_tokens"]))
    w.add_uint32("qwen35moe.memt.head_count", int(mem["num_memory_heads"]))
    w.add_float32("qwen35moe.memt.norm_eps", float(cfg["rms_norm_eps"]))

    t0 = time.time()
    for name, fn, kind in backbone(st, cfg):
        if only and not only.search(name):
            continue
        if kind == "f":   # (small: read now)
            arr = np.ascontiguousarray(fn(), dtype=np.float32)
            w.add_tensor(name, list(reversed(arr.shape)), 0, arr)
            continue
        t = types.get(name, dense)
        shp = _shape(st, name, cfg)

        def data(fn=fn, t=t, name=name):
            x = np.asarray(fn(), dtype=np.float32)
            if t == 8 and gq is None:
                b = q8_0(x)
            elif t in (10, 11, 12, 13, 14, 8):
                if gq is None:
                    sys.exit("%s: type %s needs --gpuq" % (name, gio.type_name(t)))
                b = gq(x, t)
            elif t == 30:
                b = gio.to_bf16(x).view(np.uint8)
            elif t == 0:
                b = x.view(np.uint8).reshape(-1)
            else:
                sys.exit("%s: type %s" % (name, gio.type_name(t)))
            print("  %-34s %-5s %7.1f s" % (name, gio.type_name(t), time.time() - t0), flush=True)
            return b
        w.add_tensor(name, list(reversed(shp)), t, data)
    for name, shape, ty, fn in memory(st, cfg):
        if only and not only.search(name):
            continue
        w.add_tensor(name, list(reversed(shape)), ty, fn)
    w.write()
    print("wrote %s (%.1f GB) in %.0f s" % (out, out.stat().st_size / 1e9, time.time() - t0))
    return 0


def _shape(st: SafeTensors, name: str, cfg: dict):
    """The numpy-order shape of a backbone weight, from the source header."""
    m = re.match(r"blk\.(\d+)\.(.*)", name)
    if not m:
        src = {"token_embd.weight": P + "embed_tokens.weight", "output.weight": "lm_head.weight"}[name]
        return st.t[src][2]
    il, n = int(m.group(1)), m.group(2)
    L = P + "layers.%d." % il
    I = cfg["moe_intermediate_size"]
    srcs = {"attn_qkv.weight": "linear_attn.in_proj_qkv.weight", "attn_gate.weight": "linear_attn.in_proj_z.weight",
            "ssm_alpha.weight": "linear_attn.in_proj_a.weight", "ssm_beta.weight": "linear_attn.in_proj_b.weight",
            "ssm_out.weight": "linear_attn.out_proj.weight", "attn_q.weight": "self_attn.q_proj.weight",
            "attn_k.weight": "self_attn.k_proj.weight", "attn_v.weight": "self_attn.v_proj.weight",
            "attn_output.weight": "self_attn.o_proj.weight", "ffn_down_exps.weight": "mlp.experts.down_proj",
            "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
            "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
            "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight"}
    if n in ("ffn_gate_exps.weight", "ffn_up_exps.weight"):
        e, _, h = st.t[L + "mlp.experts.gate_up_proj"][2]
        return [e, I, h]
    return st.t[L + srcs[n]][2]


if __name__ == "__main__":
    sys.exit(main())
