"""tools/memt_flash.py - MemT-1.0-Flash (MemTensor: DeepSeek-V4-Flash-0731 post-trained + native memory blocks) as an
overlay over a DeepSeek V4 Flash 0731 GGUF, with no llama.cpp, gguf-py, safetensors or torch.

usage: memt_flash.py <memt folder> <base shard 1> <out dir> [--outtype=q8_0|bf16] [--backbone=<non-expert dir>]
                     [--gpuq=<libgpuq_cuda.so>] [--imatrix=<imatrix.gguf>]
  <memt folder>   holds memt-0000N-of-0000M.safetensors (the memory blocks) and config.json
  <base shard 1>  a V4 Flash 0731 GGUF's shard 1 (e.g. DeepSeek-V4-Flash-0731-L97-00001-of-00003.gguf)
  <out dir>       overlay-00001-of-0000N.gguf: the base's shard-1 keys, deepseek4.memt.* keys and the memory tensors
                  (plus, with --backbone, MemT's own non-expert backbone tensors in the base's types); overlay-00002..
                  are symlinks to the base's shards.  Pack it with `iq_pack.py --gguf <out dir>/overlay-00001-of-..
                  --compat-bf16 --first-wins`: a tensor in several shards is taken from the first.
  --backbone      a folder of MemT-1.0-Flash's non-expert backbone tensors as range reads (recipes/memt-flash/
                  get_nonexpert.sh: runs/<shard>_<start>.bin, hdr/memt_<shard>.hdr.json + .len): every non-expert
                  tensor of the base is replaced by MemT's (same type as the base's; k-quants by --gpuq, trench-gguf's
                  GPU quantizer, byte-identical to ggml's; Q8_0 by gguf_io).  The experts stay the base's: MemT's
                  experts round back to 0731's MXFP4 codes exactly (worklog 2026-11-25_memt-flash.md section 1).

The memory tensors of layer N (GGUF shapes, ne0 first; the math in include/strata/kernels/memt.hpp):
  blk.N.memt_in.weight      [4096, 11264]  rows q 1024 | k 8x512 | v 8x512 | b 512 | w 512 | f 512 | g 512
  blk.N.memt_q_norm.weight  [1024] F32     the q latent's RMSNorm
  blk.N.memt_q_up.weight    [1024, 32768]
  blk.N.memt_k_norm.weight / memt_v_norm.weight [4096] F32   the 8 groups' RMSNorms (512 each)
  blk.N.memt_k_up.weight / memt_v_up.weight     [512, 32768] rows g*4096 .. : group g's expansion to its 8 heads
  blk.N.memt_b_up / memt_w_up / memt_f_down / memt_g_down .weight [512, 32768]
  blk.N.memt_g_bias [32768] F32, memt_decay_bias [32768] F32, memt_decay_scale_log [64] F32, memt_o_norm.weight [512] F32
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
from memt_convert import SafeTensors, GpuQ, q8_0  # noqa: E402


def memory_tensors(st: SafeTensors, mem: dict):
    """(gguf name, GGUF shape, kind 'w' (a matrix: --outtype) | 'f' (F32), data function -> float32 array)"""
    for il in mem["layers"]:
        M, B = "model.layers.%d.self_attn.memt." % il, "blk.%d." % il

        def mat(src, M=M):
            return lambda: st.f32(M + src).reshape(-1, st.t[M + src][2][-1])

        def vec(src, M=M):
            return lambda: st.f32(M + src).reshape(-1).astype(np.float32)

        sh = lambda src, M=M: st.t[M + src][2]  # noqa: E731
        yield B + "memt_in.weight", [sh("in_proj.weight")[1], sh("in_proj.weight")[0]], "w", mat("in_proj.weight")
        yield B + "memt_q_norm.weight", [sh("q_norm.weight")[0]], "f", vec("q_norm.weight")
        yield B + "memt_q_up.weight", [sh("q_up.weight")[1], sh("q_up.weight")[0]], "w", mat("q_up.weight")
        for n in ("k", "v"):
            g, out, r = sh(n + "_group_up")
            yield B + "memt_%s_norm.weight" % n, [g * r], "f", vec(n + "_group_norm")
            yield B + "memt_%s_up.weight" % n, [r, g * out], "w", mat(n + "_group_up")
        for n in ("b_up", "w_up", "f_down", "g_down"):
            o, r = sh(n + ".weight")
            yield B + "memt_%s.weight" % n, [r, o], "w", mat(n + ".weight")
        yield B + "memt_g_bias", [sh("g_down.bias")[0]], "f", vec("g_down.bias")
        yield B + "memt_decay_bias", [sh("decay_bias")[0]], "f", vec("decay_bias")
        yield B + "memt_decay_scale_log", [sh("decay_scale_log")[0]], "f", vec("decay_scale_log")
        yield B + "memt_o_norm.weight", [sh("o_norm.weight")[0]], "f", vec("o_norm.weight")


# ------------------------------------------------------------------------------------------------ the backbone (B)
class RangeTensors:
    """MemT-1.0-Flash's non-expert tensors from get_nonexpert.sh's range reads: name -> float32 array."""

    def __init__(self, d: Path):
        self.t = {}
        runs = []
        for f in (d / "runs").glob("*.bin"):
            s, a = f.stem.split("_")
            runs.append((s, int(a), int(a) + f.stat().st_size, f))
        for h in sorted((d / "hdr").glob("memt_*.hdr.json")):
            s = h.name[5:10]
            base = 8 + struct.unpack("<Q", (d / "hdr" / ("memt_%s.len" % s)).read_bytes())[0]
            for k, v in json.loads(h.read_text()).items():
                if k == "__metadata__" or ".mlp.experts." in k:
                    continue
                a, b = v["data_offsets"]
                for rs, r0, r1, f in runs:
                    if rs == s and r0 <= base + a and base + b <= r1:
                        self.t[k] = (f, v["dtype"], v["shape"], base + a - r0, b - a)
                        break
                else:
                    raise SystemExit("%s: not in the range reads" % k)

    def f32(self, name: str) -> np.ndarray:
        f, dt, shape, off, nb = self.t[name]
        if dt == "BF16":
            u = np.memmap(f, dtype="<u2", mode="r", offset=off, shape=(nb // 2,))
            return (u.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if dt == "F32":
            return np.array(np.memmap(f, dtype="<f4", mode="r", offset=off, shape=(nb // 4,))).reshape(shape)
        if dt == "I32":
            return np.array(np.memmap(f, dtype="<i4", mode="r", offset=off, shape=(nb // 4,))).reshape(shape)
        raise ValueError("%s: dtype %s" % (name, dt))


# GGUF name (blk.N. stripped) -> MemT name under model.layers.N. (the same tensor; the values are used as stored)
LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q_a.weight": "self_attn.q_a_proj.weight",
    "attn_q_a_norm.weight": "self_attn.q_a_norm.weight",
    "attn_q_b.weight": "self_attn.q_b_proj.weight",
    "attn_kv.weight": "self_attn.kv_proj.weight",
    "attn_kv_a_norm.weight": "self_attn.kv_norm.weight",
    "attn_output_a.weight": "self_attn.o_a_proj.weight",
    "attn_output_b.weight": "self_attn.o_b_proj.weight",
    "attn_sinks.weight": "self_attn.sinks",
    "attn_compressor_kv.weight": "self_attn.compressor.kv_proj.weight",
    "attn_compressor_gate.weight": "self_attn.compressor.gate_proj.weight",
    "attn_compressor_ape.weight": "self_attn.compressor.position_bias",
    "attn_compressor_norm.weight": "self_attn.compressor.kv_norm.weight",
    "indexer.attn_q_b.weight": "self_attn.compressor.indexer.q_b_proj.weight",
    "indexer_compressor_kv.weight": "self_attn.compressor.indexer.kv_proj.weight",
    "indexer_compressor_gate.weight": "self_attn.compressor.indexer.gate_proj.weight",
    "indexer_compressor_ape.weight": "self_attn.compressor.indexer.position_bias",
    "indexer_compressor_norm.weight": "self_attn.compressor.indexer.kv_norm.weight",
    "indexer.proj.weight": "self_attn.compressor.indexer.scorer.weights_proj.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "exp_probs_b.bias": "mlp.gate.e_score_correction_bias",
    "ffn_gate_tid2eid.weight": "mlp.gate.tid2eid",
    "ffn_gate_shexp.weight": "mlp.shared_experts.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_experts.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_experts.down_proj.weight",
    "hc_attn_fn.weight": "attn_hc.fn",
    "hc_attn_base.weight": "attn_hc.base",
    "hc_attn_scale.weight": "attn_hc.scale",
    "hc_ffn_fn.weight": "ffn_hc.fn",
    "hc_ffn_base.weight": "ffn_hc.base",
    "hc_ffn_scale.weight": "ffn_hc.scale",
}
TOP_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
    "output_hc_fn.weight": "model.hc_head.hc_fn",
    "output_hc_base.weight": "model.hc_head.hc_base",
    "output_hc_scale.weight": "model.hc_head.hc_scale",
}


def memt_name(gname: str):
    if gname in TOP_MAP:
        return TOP_MAP[gname]
    m = re.match(r"blk\.(\d+)\.(.*)$", gname)
    if m and m.group(2) in LAYER_MAP:
        return "model.layers.%s.%s" % (m.group(1), LAYER_MAP[m.group(2)])
    return None


def main() -> int:
    flags = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:] if a.startswith("--"))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 3:
        print(__doc__)
        return 2
    hf, shard1, out_dir = Path(args[0]), Path(args[1]), Path(args[2])
    cfg = json.loads((hf / "config.json").read_text())
    mem = cfg["memt"]
    st = SafeTensors(hf)
    wtype = {"q8_0": 8, "bf16": 30}[flags.get("--outtype") or "q8_0"]
    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", shard1.name)
    if m is None:
        sys.exit("%s is not shard 1 of a split GGUF" % shard1)
    total = int(m.group(2))
    shards = [shard1.parent / ("%s-%05d-of-%05d.gguf" % (m.group(1), i, total)) for i in range(1, total + 1)]
    n_out = total + 1
    out_dir.mkdir(parents=True, exist_ok=True)
    r0 = gio.Reader(shards[0])
    if r0.get("general.architecture") != "deepseek4":
        sys.exit("the base is %s, not deepseek4" % r0.get("general.architecture"))
    w = gio.Writer(out_dir / ("overlay-00001-of-%05d.gguf" % n_out), "deepseek4")
    gio.copy_fields(r0, w, skip=lambda k: k.startswith("split.") or k == "general.name")
    w.add_string("general.name", "MemT-1.0-Flash" + (" (MemT backbone)" if flags.get("--backbone") else
                                                      " (memory over the 0731 backbone)"))
    P = "deepseek4.memt."
    w.add_array(P + "layers", [int(x) for x in mem["layers"]], gio.U32)
    w.add_uint32(P + "head_count", int(mem["num_memory_heads"]))
    w.add_uint32(P + "key_length", int(mem["head_dim"]))
    w.add_uint32(P + "q_rank", int(mem["q_rank"]))
    w.add_uint32(P + "kv_groups", int(mem["kv_groups"]))
    w.add_uint32(P + "kv_rank", int(mem["kv_rank"]))
    w.add_uint32(P + "gate_rank", int(mem["gate_rank"]))
    w.add_uint32(P + "chunk_tokens", int(mem["chunk_tokens"]))
    w.add_float32(P + "softcap", float(mem["output_gate_softcap_limit"]) if mem["output_gate_softcap_enabled"] else 0.0)
    w.add_float32(P + "norm_eps", float(mem["norm_eps"]))
    gq = GpuQ(flags["--gpuq"]) if flags.get("--gpuq") else None
    imat = {}
    if flags.get("--imatrix"):   # llama-quantize's weights: in_sum2 / counts (1 where never seen)
        ir = gio.Reader(flags["--imatrix"])
        # (the tensors are F32 bytes: read them as float32, not converted byte by byte - before this every entry
        # had 4 "counts" and was skipped, so --imatrix changed nothing)
        f32 = lambda x: np.ascontiguousarray(x.data).reshape(-1).view("<f4")  # noqa: E731
        cnt = {x.name[:-7]: f32(x) for x in ir.tensors if x.name.endswith(".counts")}
        for x in ir.tensors:
            if x.name.endswith(".in_sum2"):
                n = x.name[:-8]
                c = cnt[n]
                s = f32(x).reshape(len(c), -1)
                if len(c) == 1:
                    imat[n] = np.ascontiguousarray(np.where(c[0] > 0, s[0] / max(c[0], 1), 1.0), np.float32)
        if not imat:
            sys.exit("%s: no 2-D weight in the imatrix" % flags["--imatrix"])
        print("imatrix: %d weights" % len(imat))
    t0 = time.time()

    def quant(x: np.ndarray, t: int, name: str) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if t == 0:
            return x.reshape(-1).view(np.uint8)
        if t == 30:
            return gio.to_bf16(x).view(np.uint8).reshape(-1)
        if t == 8 and gq is None:
            return q8_0(x)
        if gq is None:
            sys.exit("%s: type %s needs --gpuq" % (name, gio.type_name(t)))
        im = imat.get(name)
        n = x.shape[-1]
        if im is not None and im.size != n:
            # a grouped weight (V4's attn_output_a: rows [g*o_lora, (g+1)*o_lora) take input group g): llama.cpp's
            # imatrix holds one n-wide slice a group, as for a 3-D weight - each group's rows with its own slice
            g = im.size // n
            rows = x.size // n
            if im.size % n or rows % g:
                sys.exit("%s: imatrix %d vs %d columns" % (name, im.size, n))
            x2 = x.reshape(g, rows // g, n)
            return np.concatenate([gq(np.ascontiguousarray(x2[i]), t, np.ascontiguousarray(im[i * n:(i + 1) * n]))
                                   for i in range(g)])
        return gq(x, t, im)

    n_bb = 0
    if flags.get("--backbone"):
        rt = RangeTensors(Path(flags["--backbone"]))
        used = set()
        for sh in shards:
            for t in gio.Reader(sh, data=False).tensors:
                src = memt_name(t.name)
                if src is None:
                    if "_exps." not in t.name:
                        sys.exit("%s: no MemT tensor for it" % t.name)
                    continue
                if src not in rt.t:
                    sys.exit("%s: %s not in the range reads" % (t.name, src))
                used.add(src)
                if int(np.prod(rt.t[src][2])) != int(np.prod(t.shape)):
                    sys.exit("%s: %s elements vs %s" % (t.name, rt.t[src][2], t.shape))

                def data(src=src, t=t):
                    x = rt.f32(src)
                    if t.type == 26:
                        b = x.astype("<i4").reshape(-1).view(np.uint8)
                    else:
                        b = quant(x.reshape(-1, int(t.shape[0])), t.type, t.name)
                    print("  %-40s %-6s %7.1f s" % (t.name, gio.type_name(t.type), time.time() - t0), flush=True)
                    return b
                w.add_tensor(t.name, list(t.shape), t.type, data)
                n_bb += 1
        left = sorted(set(rt.t) - used)
        if left:
            sys.exit("MemT tensors with no place in the base: %s" % left[:8])
    n_mem = 0
    for name, shape, kind, fn in memory_tensors(st, mem):
        t = wtype if kind == "w" else 0

        def data(fn=fn, t=t, name=name):
            b = quant(fn(), t, name)
            print("  %-40s %-6s %7.1f s" % (name, gio.type_name(t), time.time() - t0), flush=True)
            return b
        w.add_tensor(name, shape, t, data)
        n_mem += 1
    w.write()
    for k, sh in enumerate(shards, start=2):
        dst = out_dir / ("overlay-%05d-of-%05d.gguf" % (k, n_out))
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(sh), dst)
    f = out_dir / ("overlay-00001-of-%05d.gguf" % n_out)
    print("overlay: %d memory tensors, %d backbone tensors, %.2f GB, %.0f s -> %s"
          % (n_mem, n_bb, f.stat().st_size / 1e9, time.time() - t0, f))
    return 0


if __name__ == "__main__":
    sys.exit(main())
