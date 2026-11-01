"""tools/iq_pack.py - plan v0.3 P6: a native pack for any of the model files (Q2_0, IQ2_XS, IQ3_XXS).

    python tools/iq_pack.py --gguf <model>-00001-of-00002.gguf --out pack/iq3_xxs            (standalone)
    python tools/iq_pack.py --gguf <model>-00001-of-00002.gguf --base pack/full --out ...    (share dense.bin)

The i-quant experts cannot be re-expressed in the Q2_0 pack form, so this pack keeps every quantized tensor in
its GGUF form:

  experts.bin          per layer, 512 blobs of [gate rows | up rows | down rows], the raw GGUF slices.  Blob
                       size is per layer (the files mix IQ1_M ... IQ3_S gate/up and Q2_0 / IQ4_NL down).
  native_experts.txt   one line per layer: layer gu_type d_type offset blob_bytes
  index.txt            the table the engine loads.  Quantized dense tensors, token_embd and output are served
                       natively from the GGUF by the engine (--native): their rows carry shape only.
  dense.bin            standalone: the BF16/F16/F32 tensors exactly as the GGUF stores them (index kinds 4/5/2).
                       With --base: the base (Q2_0) pack's dense.bin, hard-linked - the float tensors are
                       byte-identical in all three model files (checked) - plus extra.bin for tensors that are
                       float here but quantized in the base pack (blk.1.ple_key).
  tokenizer/           exported from the GGUF (tools/strata_tokenizer.py), with the model's chat template.

Split files: every shard of the model is read (<name>-0000N-of-0000M.gguf beside --gguf), so the layers may be
split anyhow (Swift 1.5's GGUFs put layers 13-47 in shard 2 and the PLE table in shard 1).  A layer whose experts
are not in shard 1 names its shard in native_experts.txt (v3).  Router tensors stored as F32 whose values are
exactly BF16 (Swift 1.5) are written as BF16, the form the engine's router takes; anything else is refused.

For ordinary quants, --compat-bf16 dequantizes the small projections that the engine reads as BF16, using
round-to-nearest-even. This introduces BF16 rounding; it does not reconstruct the original full-precision
weights. Experts, native attention projections, token embeddings and the disk-backed PLE table stay unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G  # noqa: E402

FLOAT = {"BF16", "F32", "F16"}
ROUTERS = ("ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight")
NOT_IN_PACK = {"per_layer_token_embd.weight"}      # the 28.8 GB PLE table: read from its GGUF by the engine
# deepseek41's Engram hash values stored as tensors (ISTA-DASLab's / Strata's layout): write_engram reads them into
# engram.txt / engram_token_map.u32
NOT_IN_PACK |= {"engram_primes.weight", "engram_offsets.weight", "engram_multipliers.weight", "engram_token_map.weight"}
# deepseek41's Engram tables (I8 rows of 264 bytes, 2 x 94 GiB): read row by row from the GGUF by the engine
NOT_IN_PACK_SUFFIX = ("engram_embd.weight", "exp_probs_b_vl.bias")   # (the _vl bias: image tokens only)

# These small projections are read as BF16 by the residual, router, GDN, QSA and PLE kernels.
# Strata's Qwen files already store them that way. Ordinary GGUF quants (including OrcaRouter's IQ3_XXS)
# quantize them too; --compat-bf16 explicitly dequantizes and rounds ONLY these tensors.
BF16_PROJECTIONS = (
    "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
    "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
    "ssm_alpha.weight", "ssm_beta.weight", "indexer.q_proj.weight", "indexer.k_proj.weight",
    "ple_value.weight", *ROUTERS,
)
BF16_OUTPUT = {"output_hc_down.weight", "output_hc_up.weight"}

# The architectures the packer knows.  The BF16-preserve lists are PER ARCHITECTURE because each
# engine reads different projections as high-precision: qwen4exp's hyper-connections use the
# down/up/inject bottleneck names, glm5-next's mHC uses fn/base/scale, and their indexers do not
# share a single tensor name.
GLM5NEXT_BF16 = (
    # the mHC mixes (the released config keeps these unquantized; the runner's glm_hc kernels
    # consume them per token)
    "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight",
    "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight",
    # the indexer - at 1M context the selection IS the model's accuracy
    "indexer.attn_q_b.weight", "indexer.attn_k.weight", "indexer_compressor_gate.weight",
    "indexer_compressor_ape.weight", "indexer.proj.weight", "indexer.k_norm.weight",
    "indexer.k_norm.bias",
    # the router and its noaux_tc selection bias (F32 by config: moe_router_dtype float32)
    "ffn_gate_inp.weight", "exp_probs_b.bias",
    # the KDA gates and norms (gates stay unquantized in the released model); the conv weights
    # seed the whole KDA recurrence, so their quantization noise would accumulate per token
    "ssm_a", "ssm_dt.bias", "ssm_norm.weight", "ssm_beta.weight",
    "ssm_f_a.weight", "ssm_f_b.weight", "ssm_g_a.weight", "ssm_g_b.weight",
    "ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight",
    # the MLA norms and the 3-D absorbed projections that form the attention scores and values
    "attn_q_a_norm.weight", "attn_kv_a_norm.weight",
    "attn_k_b.weight", "attn_v_b.weight",
)
# The ONLY non-expert tensors a glm5-next pack may serve NATIVELY (quantized, read from the
# GGUF by the Phase-4 driver's native reader): the attention projections and the dense/shared
# FFN weights, which are exactly what dynamic recipes quantize on a real file.  Everything
# else quantized fails the pack CLOSED - the packer errors instead of emitting a row the
# engine would reject at load, far from where the mistake was made.
GLM5NEXT_NATIVE_OK = (
    "attn_q_a.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_output.weight",
    # the KDA layers' separate projections (Q5_K in the UD recipe)
    "attn_q.weight", "attn_k.weight", "attn_v.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
    # the UD-IQ1_S recipe stores even the embedding and the head at Q4_K; the pack loader
    # dequantizes them into the F32 arena at load (2x ~1.2 GB quantized)
    "token_embd.weight", "output.weight",
)
# deepseek4 (DeepSeek V4 Flash, docs/deepseek4/SPEC.md): the mHC fn matrices are read as BF16 by the shared hc
# kernels (the GGUFs store them F32 - --compat-bf16 rounds them, as glm5-next's are); every other float tensor stays as
# stored (indexer.proj and the compressors' ape/norm are read as F32)
DEEPSEEK4_BF16 = ("hc_attn_fn.weight", "hc_ffn_fn.weight")
DEEPSEEK4_BF16_GLOBAL = {"output_hc_fn.weight"}
# GGUFs that store the small projections as F16 (antirez's deepseek-v4-gguf, huihui-ai's abliterated copies; unsloth's
# are BF16 / Q8_0 / F32): --compat-bf16 rounds the router and the projections the engine serves by GEMV to BF16, and
# widens the rows it reads as F32 (indexer.proj, the compressors' ape) to F32 - lossless.  token_embd stays F16 (the
# engine dequantizes its rows on the host from the shard).
DEEPSEEK4_F16_TO_BF16 = ("ffn_gate_inp.weight", "attn_compressor_kv.weight", "attn_compressor_gate.weight",
                         "indexer_compressor_kv.weight", "indexer_compressor_gate.weight", "indexer.attn_q_b.weight")
DEEPSEEK4_F16_TO_F32 = ("indexer.proj.weight", "attn_compressor_ape.weight", "indexer_compressor_ape.weight")
DEEPSEEK4_NATIVE_OK = (
    "attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight", "attn_output_a.weight", "attn_output_b.weight",
    "attn_compressor_kv.weight", "attn_compressor_gate.weight",
    "indexer.attn_q_b.weight", "indexer_compressor_kv.weight", "indexer_compressor_gate.weight",
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
    "ffn_gate_tid2eid.weight",            # I32: the hash-routed layers' expert ids, uploaded raw
    "token_embd.weight", "output.weight",
)
# deepseek41 (DeepSeek V4.1 Flash, the antirez/kyuz0 GGUF layout): deepseek4's tensors plus the index keys' projection
# and the Engram projection / norms, all F16 / F32 in that GGUF
DEEPSEEK41_F16_TO_BF16 = DEEPSEEK4_F16_TO_BF16 + ("indexer.attn_k.weight", "engram_kv.weight", "engram_wkv.weight")
# mrfakename's 1017 Q2_K quantizes nearly every matrix: the ones the engine reads as BF16 (the index keys' projection)
# or F32 (indexer.proj) are dequantized at any type; engram_wkv is served by the quantized GEMV
DEEPSEEK41_ANY_TO_BF16 = ("indexer.attn_k.weight",)
DEEPSEEK41_ANY_TO_F32 = ("indexer.proj.weight",)
DEEPSEEK41_NATIVE_OK = DEEPSEEK4_NATIVE_OK + ("engram_wkv.weight", "engram_kv.weight")
# qwen35moe (Qwen3.5-MoE: Ornith-1.5-122B): every non-expert tensor is served as stored - the Q8_0 projections by the
# quantized GEMV, the F32 norms / conv / router / gates / ssm_a as F32 - so nothing is converted; the NextN block
# (blk.{trunk}) is dropped as glm5-next's
QWEN35MOE_NATIVE_OK = (
    "attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ssm_alpha.weight", "ssm_beta.weight",
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
    "token_embd.weight", "output.weight",
)
# qwen4exp for the fast path (--fast): the same, plus the hyper-connections' low-rank projections (Q6_K / Q8_0 as
# stored) and the PLE key / value
QWEN4EXP_FAST_NATIVE_OK = QWEN35MOE_NATIVE_OK + (
    "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
    "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
    "output_hc_down.weight", "output_hc_up.weight", "ple_key.weight", "ple_value.weight",
)
ARCHITECTURES = ("qwen4exp", "glm5-next", "glm5next", "deepseek4", "deepseek41", "qwen35moe")   # unsloth spells glm5-next without the hyphen


def needs_bf16(name: str, type_name: str, arch: str = "qwen4exp") -> bool:
    if arch == "deepseek4":
        return name in DEEPSEEK4_BF16_GLOBAL or (name.startswith("blk.") and name.endswith(DEEPSEEK4_BF16)) or (
            type_name == "F16" and name.startswith("blk.") and name.endswith(DEEPSEEK4_F16_TO_BF16))
    if arch == "deepseek41":
        return name.startswith("blk.") and (name.endswith(DEEPSEEK4_BF16) or name.endswith(DEEPSEEK41_ANY_TO_BF16) or (
            type_name == "F16" and name.endswith(DEEPSEEK41_F16_TO_BF16)))
    if arch in ("glm5-next", "glm5next"):
        # glm5-next has no PLE and no output_hc; its F32 router stays F32 (the runner reads it
        # as F32 by config), so ONLY the listed projections take the BF16 path.
        return name.startswith("blk.") and name.endswith(GLM5NEXT_BF16)
    if arch in ("qwen35moe", "qwen4exp-fast"):
        return False
    if name in BF16_OUTPUT:
        return True
    if not name.startswith("blk."):
        return False
    # The existing native PLE key supports Q2_0 only. Other key encodings use the BF16 path.
    return name.endswith(BF16_PROJECTIONS) or (name == "blk.1.ple_key.weight" and type_name != "Q2_0")


def index_shape(shape) -> tuple[int, int]:
    """The (ne0, ne1) the flat index row records for a tensor of GGUF shape `shape`.

    The engine reads packed tensors by BYTE OFFSET and the kernels by their own geometry, so a
    3-D tensor (glm5-next's absorbed MLA projections and conv weights) packs as the 2-D view
    (ne0*ne1, ne2): GGUF lays element (i0, i1, i2) at i0 + ne0*i1 + ne0*ne1*i2, which is exactly
    row (i0 + ne0*i1) of the flattened view - byte-identical, no repacking."""
    if len(shape) >= 3:
        return int(shape[0]) * int(shape[1]), int(shape[2])
    ne0 = int(shape[0])
    return ne0, (int(shape[1]) if len(shape) > 1 else 0)


def bf16_bytes(raw: np.ndarray, type_name: str) -> bytes:
    from _paths import add_gguf_py
    add_gguf_py()
    from gguf import GGMLQuantizationType as Q, quants
    values = quants.dequantize(raw, Q[type_name])
    if not np.isfinite(values).all():
        raise ValueError("cannot convert non-finite weights to BF16")
    # ggml's round-to-nearest-even conversion, including correct halfway rounding.
    return quants.quantize(values, Q.BF16).tobytes()


def f32_bytes(raw: np.ndarray, type_name: str) -> bytes:
    from _paths import add_gguf_py
    add_gguf_py()
    from gguf import GGMLQuantizationType as Q, quants
    values = quants.dequantize(raw, Q[type_name]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("cannot widen non-finite weights to F32")
    return values.tobytes()


class Model:
    """All shards of one model: name -> (GGUFFile, TensorInfo, memmap, shard path)."""

    def __init__(self, first: pathlib.Path, first_wins: bool = False, extra=()):
        import re
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first.name)
        paths = [first]
        if m:
            total = int(m.group(2))
            paths = [first.with_name(first.name[:m.start()] + "-%05d-of-%05d.gguf" % (i, total))
                     for i in range(1, total + 1)]
        missing = [p for p in paths if not p.is_file()]
        if missing and extra:   # (--engram: a shard left undownloaded because it holds only the Engram table)
            print("missing shards skipped, the Engram tables from %s: %s" % (", ".join(map(str, extra)),
                                                                          ", ".join(p.name for p in missing)))
            paths = [p for p in paths if p.is_file()]
            missing = []
        paths += [pathlib.Path(x) for x in extra]   # (another GGUF read after the shards: a separate Engram file)
        missing = [str(p) for p in missing + paths if not p.is_file()]
        if missing:
            raise FileNotFoundError("missing model shards (wait for the download): " + ", ".join(missing))
        self.paths = paths
        self.where = {}
        for p in paths:
            g = G.GGUFFile(p)
            mm = np.memmap(p, dtype=np.uint8, mode="r")
            for t in g.tensors:
                size = t.expected_bytes()
                if size is None or g.data_start + t.offset + size > mm.size:
                    raise ValueError(f"{p.name}: unsupported or truncated tensor {t.name}")
                if t.name in self.where:
                    if first_wins:   # (a hybrid: shard 1's copy wins - the engine's lookups scan in shard order too)
                        self.shadowed = getattr(self, "shadowed", 0) + 1
                        continue
                    raise ValueError(f"{p.name}: duplicate tensor {t.name}")
                self.where[t.name] = (g, t, mm, p)

    def bytes(self, name) -> np.ndarray:
        g, t, mm, _ = self.where[name]
        return tensor_bytes(mm, g, t)
ROLES = ("gate", "up", "down")
N_EXPERT = 512
ALIGN = 64


def read_index(path: pathlib.Path):
    rows, header = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            header.append(line)
            continue
        f = line.split()
        rows[f[0]] = f
    return header, rows


def tensor_bytes(mm, g, t) -> np.ndarray:
    n = t.expected_bytes()
    return mm[g.data_start + t.offset: g.data_start + t.offset + n]


def is_expert(name: str) -> bool:
    return name.startswith("blk.") and name.endswith(("_exps.weight",))


def index_standalone(src, out, model: Model, compat_bf16: bool = False, arch: str = "qwen4exp",
                     block_count: int | None = None) -> int:
    """Every non-expert tensor of the model: floats into dense.bin as stored (exact-BF16 F32 routers as BF16),
    quantized ones native-only.

    Fail-closed for glm5-next: a non-expert tensor that is neither float nor on the BF16 upgrade
    list may only pass if GLM5NEXT_NATIVE_OK names it (the Phase-4 native reader's set); anything
    else is a hard error.  The NextN/MTP block (blk.{block_count}.*, the runner skips MTP) is
    dropped from the pack entirely."""
    dropped = 0
    if not compat_bf16:
        for name, (_, t, _, _) in model.where.items():
            if needs_bf16(name, t.type_name, arch) and t.type_name != "BF16" and not (
                    t.type_name == "F32" and name.endswith(ROUTERS) and arch == "qwen4exp"):
                print(f"{name} is {t.type_name}, but the engine requires BF16; use --compat-bf16")
                return 1
    rows, at = [], 0
    served = 0
    converted = []
    with open(out / "dense.bin", "wb") as fo:
        for name, (g, t, mm, _) in model.where.items():
            if is_expert(t.name) or t.name in NOT_IN_PACK or t.name.endswith(NOT_IN_PACK_SUFFIX):
                continue
            if block_count is not None and name.startswith("blk."):
                il = name.split(".")[1]
                if il.isdigit() and int(il) >= block_count:
                    dropped += 1
                    continue
            # a trailing ne3 of 1 adds no bytes: some converters write glm5-next's conv weights as
            # [4, 1, 8192, 1] instead of [4, 1, 8192], and index_shape packs both as (4, 8192)
            dims = len(t.shape) - (len(t.shape) == 4 and int(t.shape[3]) == 1)
            if dims == 3 and arch == "deepseek41" and "exps" not in name:
                # (ISTA-DASLab's GSQ-RCO builds keep the grouped output LoRA as [4096, 1024, 8], unsloth's as
                # [4096, 8192]: the same bytes - rows of ne0 - so it packs as the 2-D view (ne0, ne1 * ne2))
                t.shape = [int(t.shape[0]), int(t.shape[1]) * int(t.shape[2])]
                dims = 2
            if dims > 2 and not (arch in ("glm5-next", "glm5next") and dims == 3):
                print("tensor %s has %d dimensions; the index holds two" % (t.name, len(t.shape)))
                return 1
            ne0, ne1 = index_shape(t.shape)
            convert = compat_bf16 and needs_bf16(t.name, t.type_name, arch) and t.type_name != "BF16"
            widen = (compat_bf16 and arch in ("deepseek4", "deepseek41") and t.type_name == "F16" and name.startswith("blk.")
                     and name.endswith(DEEPSEEK4_F16_TO_F32))
            # (BF16 too: unsloth's 1017 GGUFs keep indexer.proj in BF16, exact in F32)
            widen_q = (compat_bf16 and arch == "deepseek41" and t.type_name not in ("F32", "F16") and name.startswith("blk.")
                       and name.endswith(DEEPSEEK41_ANY_TO_F32))
            if t.type_name in FLOAT or convert or widen_q:
                raw = tensor_bytes(mm, g, t).tobytes()
                if widen_q:
                    raw = f32_bytes(np.frombuffer(raw, dtype=np.uint8), t.type_name)
                    kind = "2"
                    converted.append({"name": t.name, "source_type": t.type_name, "bytes": len(raw), "to": "F32"})
                elif widen:
                    raw = np.frombuffer(raw, dtype=np.float16).astype(np.float32).tobytes()
                    kind = "2"
                    converted.append({"name": t.name, "source_type": t.type_name, "bytes": len(raw), "to": "F32"})
                elif convert:
                    raw = bf16_bytes(np.frombuffer(raw, dtype=np.uint8), t.type_name)
                    kind = "4"
                    converted.append({"name": t.name, "source_type": t.type_name, "bytes": len(raw)})
                else:
                    kind = {"BF16": "4", "F16": "5", "F32": "2"}[t.type_name]
                if not convert and arch == "qwen4exp" and t.type_name == "F32" and t.name.endswith(ROUTERS):
                    u = np.frombuffer(raw, dtype=np.uint32)
                    if np.count_nonzero(u & 0xFFFF):
                        print("router %s is F32 with values that are not BF16; the engine's router is BF16" % t.name)
                        return 1
                    raw = (u >> 16).astype(np.uint16).tobytes()     # the exact BF16 values
                    kind = "4"
                rows.append([t.name, "0", kind, str(at), str(len(raw)), "0", str(len(raw)), str(ne0), str(ne1),
                             "0", "0", "1"] + ["0"] * 7)
                fo.write(raw)
                pad = (-len(raw)) % ALIGN
                fo.write(b"\0" * pad)
                at += len(raw) + pad
            elif arch in ("deepseek4", "deepseek41") and not name.endswith(
                    DEEPSEEK41_NATIVE_OK if arch == "deepseek41" else DEEPSEEK4_NATIVE_OK):
                print("%s is %s: not on the BF16 upgrade list nor the deepseek4 native set "
                      "(DEEPSEEK4_NATIVE_OK); refusing to pack it silently" % (t.name, t.type_name))
                return 1
            elif arch in ("glm5-next", "glm5next") and not name.endswith(GLM5NEXT_NATIVE_OK):
                print("%s is %s: not on the BF16 upgrade list nor the glm5-next native set "
                      "(GLM5NEXT_NATIVE_OK); refusing to pack it silently" % (t.name, t.type_name))
                return 1
            elif arch == "qwen35moe" and not name.endswith(QWEN35MOE_NATIVE_OK):
                print("%s is %s: not on the qwen35moe native set (QWEN35MOE_NATIVE_OK); refusing to pack it "
                      "silently" % (t.name, t.type_name))
                return 1
            elif arch == "qwen4exp-fast" and not name.endswith(QWEN4EXP_FAST_NATIVE_OK):
                print("%s is %s: not on the qwen4exp fast-path native set (QWEN4EXP_FAST_NATIVE_OK); refusing to "
                      "pack it silently" % (t.name, t.type_name))
                return 1
            else:
                served += 1
                rows.append([t.name, "0", "0", "0", "0", "0", "0", str(ne0), str(ne1), "8", "0", "32"] + ["0"] * 7)
    if dropped:
        print("NextN/MTP: dropped %d tensors of blk.%d+ (the runner skips MTP; re-pack when a "
              "GLM5-Next draft graph exists)" % (dropped, block_count))
    write_index(out, rows, src, served, 0)
    if compat_bf16:
        (out / "compat-bf16.json").write_text(json.dumps({
            "source": str(src), "rounding": "nearest-even", "tensors": converted,
        }, indent=2) + "\n", encoding="utf-8")
        print("compat-bf16: %d tensors, %.2f GiB; expert and PLE table bytes unchanged"
              % (len(converted), sum(t["bytes"] for t in converted) / 2**30))
    return 0


def write_index(out, rows, src, served, n_extra):
    at = 0
    for r in rows:
        r[5] = str(at)
        at += (int(r[6]) + ALIGN - 1) // ALIGN * ALIGN
    with open(out / "index.txt", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# strata pack index v3 -- generated by tools/iq_pack.py (native experts) from %s\n" % src.name)
        fo.write("# align %d pool %d tensors %d\n" % (ALIGN, at, len(rows)))
        for r in rows:
            fo.write(" ".join(r) + "\n")
    print("index.txt: %d tensors, %d served natively, %d in extra.bin, arena %.2f GiB"
          % (len(rows), served, n_extra, at / 2**30))


def index_from_base(a, src, base, out, g, T, mm) -> int:
    base_src = pathlib.Path(json.loads((base / "manifest.json").read_text(encoding="utf-8"))["source"]["shard1"])
    if not base_src.exists():
        print("cannot find the base pack's shard 1 from its manifest.json")
        return 1
    bg = G.GGUFFile(base_src)
    BT = {t.name: t for t in bg.tensors}
    bmm = np.memmap(base_src, dtype=np.uint8, mode="r")
    header, rows = read_index(base / "index.txt")
    new_rows, extra = [], []
    served = 0
    for name, f in rows.items():
        t, bt = T.get(name), BT.get(name)
        if t is None or bt is None:
            print("tensor %s missing from one of the models" % name)
            return 1
        if t.type_name in FLOAT and bt.type_name in FLOAT:
            if t.type_name != bt.type_name or t.shape != bt.shape or \
                    not np.array_equal(tensor_bytes(mm, g, t), tensor_bytes(bmm, bg, bt)):
                print("float tensor %s differs from the base model; this pack cannot reuse its dense.bin" % name)
                return 1
            new_rows.append(list(f))
        elif t.type_name in FLOAT:
            if t.type_name != "BF16":
                print("unexpected float type %s for %s" % (t.type_name, name))
                return 1
            nbytes = t.expected_bytes()
            off = sum(len(b) + (-len(b)) % ALIGN for b in extra)
            extra.append(tensor_bytes(mm, g, t).tobytes())
            # file 3 = extra.bin, raw BF16 (index kind 4)
            new_rows.append([name, "3", "4", str(off), str(nbytes), "0", str(nbytes), f[7], f[8]] + ["0"] * 10)
        else:
            served += 1
            new_rows.append([name, f[1], "0", "0", "0", "0", "0", f[7], f[8], "8", "0", "32"] + ["0"] * 7)
    write_index(out, new_rows, src, served, len(extra))
    with open(out / "extra.bin", "wb") as fo:
        for b in extra:
            fo.write(b)
            fo.write(b"\0" * ((-len(b)) % ALIGN))
    dense = out / "dense.bin"
    if not dense.exists():
        try:
            os.link(base / "dense.bin", dense)
        except OSError:
            shutil.copyfile(base / "dense.bin", dense)
    if (base / "tokenizer").exists() and not (out / "tokenizer").exists():
        shutil.copytree(base / "tokenizer", out / "tokenizer")
    return 0


def write_vl_bias(out, g, model: Model) -> int:
    """deepseek41: the router bias of the image rows (blk.N.exp_probs_b_vl.bias, n_expert F32 a layer) as one raw
    exp_probs_b_vl.f32 (n_layers x n_expert x 4 bytes, layer order; a layer without one - the hash-routed ones, which
    never use a bias - holds zeros).  The engine reads it for an image request (STRATA_GLM_VL_BIAS names another file).
    A GGUF without it (the converters that drop it) writes nothing, as before."""
    import re
    ids = sorted(int(mo.group(1)) for n in model.where for mo in [re.fullmatch(r"blk\.(\d+)\.exp_probs_b_vl\.bias", n)] if mo)
    if not ids:
        return 0
    n_layers = int(g.metadata.get("deepseek41.block_count", ids[-1] + 1))
    rows = []
    ne = None
    for il in range(n_layers):
        name = "blk.%d.exp_probs_b_vl.bias" % il
        if name not in model.where:
            rows.append(None)
            continue
        v = np.asarray(model.bytes(name)).view(np.float32)
        if ne is None:
            ne = v.size
        if v.size != ne:
            print("deepseek41: %s has %d values, not %d" % (name, v.size, ne))
            return 1
        rows.append(v)
    arr = np.stack([r if r is not None else np.zeros(ne, np.float32) for r in rows]).astype("<f4")
    arr.tofile(out / "exp_probs_b_vl.f32")
    print("deepseek41: exp_probs_b_vl.f32 (%d layers x %d experts; %d layers carry one)" % (n_layers, ne, len(ids)))
    return 0


def write_engram(out, g, model: Model) -> int:
    """deepseek41: what the engine needs to read the Engram rows from the GGUF (worklog 2026-10-18_deepseek41-spec.md
    section 3): engram.txt (the row format, each table's absolute offset and row count, the primes and multipliers)
    and the token map as raw uint32 (engram_token_map.u32; the engine's GGUF reader keeps only an array's first
    items).  Layouts: ds4's (e4m3 + e8m0 rows of 264 bytes), llama.cpp's converter (Q8_0 rows of 256, a table a layer)
    and llama.cpp PR 30817's (unsloth: deepseek41.ple.* keys, one per_layer_token_embd table)."""
    m = g.metadata
    p = "deepseek41.engram."
    llama = "deepseek41.block_count" in m
    if not llama and m.get(p + "encoding") != "e4m3_e8m0_32_row264":
        print("deepseek41: engram.encoding %r (only e4m3_e8m0_32_row264 is implemented)" % m.get(p + "encoding"))
        return 1
    # llama.cpp PR 30817 (unsloth's 1017 GGUFs): the hash keys under deepseek41.ple.*, both tables in one
    # per_layer_token_embd.weight (the second table's rows after the first's: offsets run on across both)
    ple = "deepseek41.ple.layers" in m
    if ple:
        q = "deepseek41.ple."
        layers = list(m[q + "layers"])
        primes, mult = list(m[q + "head_vocab_sizes"]), list(m[q + "layer_multipliers"])
        tmap = m[p + "token_map"]
        if int(m.get(q + "heads_per_ngram", 0)) != 8 or int(m.get(q + "ngram_size", 0)) != 4 or \
                int(m.get("deepseek41.embedding_length_per_layer_input", 0)) != 256:
            print("deepseek41: ple heads_per_ngram / ngram_size / embedding_length_per_layer_input are not 8 / 4 / 256")
            return 1
    else:   # (Project Maya's V4.1 GGUFs name the keys engram.layers / head_dim / pad_token_id)
        layers = list(m[p + "layer_ids"] if p + "layer_ids" in m else m[p + "layers"])
        # (ISTA-DASLab's / Strata's layout: the primes, offsets, multipliers and token map as I64 / I32 tensors,
        # [24, 2] / [24, 2] / [4, 2] - a table's values contiguous - and the head shape under deepseek41.engram_*)
        def tens(name, dtype):
            return [int(x) for x in model.bytes(name).view(dtype)] if name in model.where else None
        primes = list(m[p + "primes"]) if p + "primes" in m else tens("engram_primes.weight", np.int64)
        mult = list(m[p + "multipliers"]) if p + "multipliers" in m else tens("engram_multipliers.weight", np.int64)
        tmap = m[p + "token_map"] if p + "token_map" in m else tens("engram_token_map.weight", np.int32)
        if p + "offsets" not in m and "engram_offsets.weight" in model.where:
            m = dict(m)
            m[p + "offsets"] = tens("engram_offsets.weight", np.int64)
            for k in ("head_count", "key_length", "max_ngram_size", "pad_token_id"):
                alt = {"head_count": "deepseek41.engram_n_heads", "key_length": "deepseek41.engram_head_dim",
                       "max_ngram_size": "deepseek41.engram_max_ngram_size",
                       "pad_token_id": "deepseek41.engram_pad_token_id"}[k]
                if p + k not in m and alt in m:
                    m[p + k] = m[alt]
        if primes is None or mult is None or tmap is None:
            print("deepseek41: no Engram primes / multipliers / token map in the metadata or as tensors")
            return 1
    if len(layers) != 2 or len(primes) != 48 or len(mult) != 8:
        print("deepseek41: unexpected engram metadata (layers %d, primes %d, multipliers %d)"
              % (len(layers), len(primes), len(mult)))
        return 1
    if llama and not ple and (int(m.get(p + "head_count", 0)) != 8 or
                              int(m.get(p + "key_length", m.get(p + "head_dim", 0))) != 256 or
                              int(m.get(p + "max_ngram_size", 0)) != 4):
        print("deepseek41: engram head_count / key_length / max_ngram_size are not 8 / 256 / 4")
        return 1
    base = [0, 0]   # (ple: each table's first row in the one tensor)
    if llama:
        offs = list(m["deepseek41.ple.head_offsets" if ple else p + "offsets"])
        if len(offs) != 48:
            print("deepseek41: %d Engram head offsets, not 48" % len(offs))
            return 1
        run = 0
        for k in range(2):
            if not ple:
                run = 0
            base[k] = run
            for c in range(24):
                if int(offs[24 * k + c]) != run:
                    print("deepseek41: the Engram head offsets are not the running sum of the primes")
                    return 1
                run += int(primes[24 * k + c])
    fmt, lines = "row264 264", []
    # the tables as one per_layer_token_embd though the model names a table a layer (another GGUF's copy, --engram):
    # table k's rows begin after the tables before it
    one = not ple and llama and "blk.%d.engram_embd.weight" % layers[0] not in model.where and \
        "per_layer_token_embd.weight" in model.where
    if one:
        base = [sum(int(x) for x in primes[:24 * k]) for k in range(2)]
    for k, il in enumerate(layers):
        name = "per_layer_token_embd.weight" if ple or one else "blk.%d.engram_embd.weight" % il
        if name not in model.where:
            print("deepseek41: %s is missing" % name)
            return 1
        gg, t, _, path = model.where[name]
        need = sum(int(x) for x in primes[24 * k:24 * k + 24])
        rows = int(t.shape[1]) - base[k]
        if one:
            rows = min(rows, need) if k == 0 else rows
        if llama or t.type_name != "I8":   # any ggml row type (Q8_0 in the PR's converter, Q2_K in mrfakename's Q2_K,
            # IQ4_NL in unsloth's); the engine decodes with ggml's to_float
            rb = t.expected_bytes() // max(int(t.shape[1]), 1) if t.expected_bytes() else 0
            ok = int(t.shape[0]) == 256 and rows >= need and rb > 0 and rb <= 272
            fmt = "ggml %d %d" % (t.type_id, rb)
            ok = ok and (llama or rows == int(m[p + "rows"][k]))
        else:
            ok = t.type_name == "I8" and int(t.shape[0]) == 264 and rows == need == int(m[p + "rows"][k])
        if not ok:
            print("deepseek41: %s is %s %s for %d rows of buckets" % (name, t.type_name, list(t.shape), need))
            return 1
        if ple or one:
            rows = need   # (the first table's rows end where the second's begin)
        lines.append("table %d %d %d %d %s" % (k, il, gg.data_start + t.offset + base[k] * (rb if ple or one else 0), rows,
                                               pathlib.Path(path).resolve()))
    lines.insert(0, "format " + fmt)
    lines.append("primes " + " ".join(str(int(x)) for x in primes))
    lines.append("multipliers " + " ".join(str(int(x)) for x in mult))
    lines.append("pad %d" % int(m[p + ("pad_token_id" if ple or p + "pad_id" not in m else "pad_id")]))
    lines.append("vocab %d" % len(tmap))
    (out / "engram.txt").write_text("# deepseek41 Engram: format <name> <row bytes>; table <k> <layer> <abs offset> "
                                    "<rows> <file>\n" + "\n".join(lines) + "\n", encoding="utf-8")
    np.asarray(tmap, dtype=np.int64).astype(np.uint32).tofile(out / "engram_token_map.u32")
    print("deepseek41: engram.txt + engram_token_map.u32 (%d tokens, %s rows); the tables stay in the GGUF"
          % (len(tmap), fmt))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the model's shard 1")
    ap.add_argument("--base", help="optional: a Q2_0 canonical pack whose dense.bin holds the shared float tensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-experts", action="store_true", help="rewrite the index only")
    ap.add_argument("--first-wins", action="store_true",
                    help="a hybrid shard set: a tensor in several shards is taken from the first (e.g. shard 1 = "
                         "another file's dense weights, the rest = the experts' GGUF)")
    ap.add_argument("--engram", help="deepseek41: the GGUF holding the Engram tables when they are not in the shards "
                                     "(blk.N.engram_embd, or per_layer_token_embd as unsloth's 1017 shards have them)")
    ap.add_argument("--compat-bf16", action="store_true",
                    help="dequantize small non-native projections to BF16 for ordinary Qwen4Exp GGUFs "
                         "(rounds weights; leaves experts and the PLE table unchanged)")
    ap.add_argument("--fast", action="store_true",
                    help="qwen4exp: a pack for the fast path (--glm-pack) - nothing converted to BF16")
    ap.add_argument("--experts-bin", action="store_true",
                    help="also write experts.bin (the engine otherwise reads the experts from the GGUF itself)")
    a = ap.parse_args()
    if a.compat_bf16 and a.base:
        ap.error("--compat-bf16 cannot reuse --base dense weights")
    # HF snapshot files are symlinks to hash-named blobs. Keep the shard filename for discovery.
    src = pathlib.Path(a.gguf).absolute()
    base = pathlib.Path(a.base).resolve() if a.base else None
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    g = G.GGUFFile(src)
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    arch = str(g.metadata.get("general.architecture", ""))
    if arch not in ARCHITECTURES:
        print("general.architecture is %r; the packer knows %s" % (arch, " and ".join(ARCHITECTURES)))
        return 1
    if arch == "deepseek4":
        prefix = "deepseek4."
        print("deepseek4: packing for the Glm5Model runner's deepseek4 path (quantized dense served natively)")
    if arch == "deepseek41":
        prefix = "deepseek41."
        print("deepseek41: packing for the Glm5Model runner's deepseek4 path, V4.1 layout (the Engram tables stay "
              "in the GGUF)")
    if arch in ("glm5-next", "glm5next"):
        # two spellings in the wild: llama.cpp's canonical "glm5-next" (llama-arch.cpp L156) and
        # unsloth's "glm5next" - the KV PREFIX follows the same spelling, so it is read per file
        prefix = "glm5-next." if arch == "glm5-next" else "glm5next."
        print("%s: packing for the Glm5Model runner (quantized dense served natively; the driver "
              "gate remains closed until the pack wiring lands - docs/maya/GLM5-FLASH.md §9)" % arch)
    if arch == "qwen35moe":
        prefix = "qwen35moe."
        print("qwen35moe: packing for the Glm5Model runner's qwen35moe path (every dense tensor served as stored)")
    block_count = None
    trunk = None
    if arch in ("glm5-next", "glm5next", "deepseek4", "deepseek41", "qwen35moe"):
        bc = g.metadata.get(prefix + "block_count", g.metadata.get(prefix + "num_hidden_layers"))
        if bc is None:
            print("%s: %sblock_count is missing; cannot tell the trunk from the NextN block" % (arch, prefix))
            return 1
        block_count = int(bc)
        # the NextN/MTP layer counts INTO block_count in the wild (unsloth ships 46 with
        # nextn_predict_layers 1): the servable trunk is what is left after subtracting it
        nextn = int(g.metadata.get(prefix + "nextn_predict_layers", 0) or 0)
        trunk = block_count - nextn
        if trunk < 1:
            print("%s: block_count %d - nextn %d leaves no trunk" % (arch, block_count, nextn))
            return 1
    # deepseek41 with the Engram tables in a GGUF of their own (ISTA-DASLab's GSQ-RCO builds, Strata's layout:
    # deepseek41.engram.file names it, beside shard 1): --engram, else that file - or any GGUF holding the same tables,
    # e.g. unsloth's 1017 shard with per_layer_token_embd (the same IQ4_NL bytes: no second 110 GB download)
    extra = []
    eng_file = a.engram or (str(src.parent / str(g.metadata["deepseek41.engram.file"]))
                            if "deepseek41.engram.file" in g.metadata else None)
    if eng_file:
        if not pathlib.Path(eng_file).is_file():
            print("deepseek41: the Engram tables' file %s is missing (download it, or pass --engram <a GGUF with "
                  "them>)" % eng_file)
            return 1
        extra.append(eng_file)
        print("deepseek41: Engram tables from %s" % eng_file)
    model = Model(src, first_wins=a.first_wins, extra=extra)
    if getattr(model, "shadowed", 0):
        print("first-wins: %d tensors of the later shards shadowed by earlier ones" % model.shadowed)
    if arch == "deepseek4" and trunk is not None and trunk < block_count and \
            "blk.%d.hc_attn_fn.weight" % (block_count - 1) in model.where:
        # antirez's deepseek-v4-gguf (huihui-ai's copies too) says nextn_predict_layers 1 but ships no draft block:
        # the last block has the hyper-connections a NextN block lacks, so it is a trunk layer
        print("%s: nextn_predict_layers %d but blk.%d is a trunk layer (no draft block in the file)"
              % (arch, block_count - trunk, block_count - 1))
        trunk = block_count
    if trunk is not None:
        print("%s: block_count %d -> trunk layers 0..%d" % (arch, block_count, trunk - 1))
    T = {n: w[1] for n, w in model.where.items()}
    if len(model.paths) > 1:
        print("model shards: " + ", ".join(p.name for p in model.paths))
    if a.base:
        if any(w[3] != src for w in model.where.values() if not w[1].name in NOT_IN_PACK):
            print("--base needs a model whose tensors are all in shard 1")
            return 1
        rc = index_from_base(a, src, base, out, g, {t.name: t for t in g.tensors}, mm)
    else:
        # --fast (qwen4exp): the fast path's pack - every tensor as the GGUF stores it, the F32 routers too
        rc = index_standalone(src, out, model, a.compat_bf16, "qwen4exp-fast" if a.fast and arch == "qwen4exp" else arch,
                              trunk)
    if rc:
        return rc
    if arch == "deepseek41":
        rc = write_engram(out, g, model) or write_vl_bias(out, g, model)
        if rc:
            return rc
    if not (out / "tokenizer" / "vocab.json").exists() or not (out / "tokenizer" / "chat_template.jinja").exists():
        subprocess.run([sys.executable, str(HERE / "strata_tokenizer.py"), "--gguf", str(src), "--out", str(out)],
                       check=True)   # writes <out>/tokenizer/

    # ---- the experts
    if trunk is not None:
        # the trunk only: the NextN/MTP layer ships its own experts (a full extra layer) and the
        # runner skips MTP - deriving n_layers from the max expert index would pack the NextN
        # layer as a servable layer the driver must never see
        n_layers = trunk
        trunk_experts = [n for n in T if n.endswith("_exps.weight")
                         and n.startswith("blk.") and 0 <= int(n.split(".")[1]) < trunk]
        if not trunk_experts:
            print("%s: no routed experts in the trunk; the pack has nothing to serve" % arch)
            return 1
        dropped_experts = sum(1 for n in T if n.startswith("blk.%d." % trunk) and n.endswith("_exps.weight"))
        if dropped_experts:
            print("%s: %d NextN expert tensors present (blk.%d.*); the trunk keeps layers 0..%d"
                  % (arch, dropped_experts, trunk, trunk - 1))
    else:
        n_layers = 1 + max(int(n.split(".")[1]) for n in T if n.startswith("blk.") and n.endswith("_exps.weight"))
    # the MoE layers only: the leading dense blocks ship no router and no experts
    if arch in ("glm5-next", "glm5next", "deepseek4", "deepseek41", "qwen35moe"):
        lead = int(g.metadata.get(prefix + "leading_dense_block_count", 0) or 0)
    else:
        lead = int(g.metadata.get("qwen4exp.leading_dense_block_count", 0) or 0)
    moe_layers = [l for l in range(n_layers) if l >= lead]
    first_moe = moe_layers[0]
    n_expert = int(T["blk.%d.ffn_gate_inp.weight" % first_moe].shape[1])  # router rows = experts kept
    if any(int(T["blk.%d.ffn_gate_inp.weight" % l].shape[1]) != n_expert for l in moe_layers):
        print("the routers disagree on the expert count; a per-layer pruned model cannot be packed")
        return 1
    layout, offset = [], 0
    for l in moe_layers:
        ts = [T["blk.%d.ffn_%s_exps.weight" % (l, r)] for r in ROLES]
        per = [t.expected_bytes() // n_expert for t in ts]
        if per[0] != per[1] or ts[0].type_name != ts[1].type_name:
            print("layer %d: gate and up differ in type" % l)
            return 1
        blob = per[0] + per[1] + per[2]
        layout.append((l, ts[0].type_id, ts[2].type_id, offset, blob, ts))
        offset += blob * n_expert
    with open(out / "native_experts.txt", "w", encoding="utf-8", newline="\n") as fo:
        straddle = any(len({w[3] for w in (model.where[t.name] for t in ts)}) != 1
                       for _, _, _, _, _, ts in layout)
        version = "v4" if straddle else "v3"
        fo.write("# strata native experts %s: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
                 "[shard | gate_shard up_shard down_shard] "
                 "(n_expert %d, total %d; absolute offsets in %s, or in the named shard(s) beside it)\n"
                 % (version, n_expert, offset, src.name))
        for l, gt, dt, off, blob, ts in layout:
            ws = [model.where[t.name] for t in ts]
            shards = [w[3] for w in ws]
            # each role's offset is absolute in ITS OWN named shard: the data_start of the
            # file that holds the tensor + the tensor's data-section offset.  (Using the
            # gate role's file for every role put layer 25's down 7616 bytes low in the
            # UD-IQ1_S pack: gate/up sat in one shard, down in another.)
            offs = [w[0].data_start + t.offset for w, t in zip(ws, ts)]
            if len(set(shards)) == 1:
                # v3 row: one shard for the whole blob (omitted when it is shard 1)
                line = "%d %d %d %d %d %d %d %d" % (l, gt, dt, off, blob, *offs)
                fo.write(line + ("" if shards[0] == src else " " + shards[0].name) + "\n")
            else:
                # v4 row: the layer straddles shards (a shard boundary inside a layer); each
                # role names its own file - shard 1 by name too: the engine splits the row on
                # whitespace, so an empty name would vanish and shift the others.  The engine
                # reads every role from its own shard (gate_shard / up_shard / down_shard) and
                # checks each offset against that shard's directory at load, so gate and up may
                # sit in different files too (a split by size can cut between them).
                names = [s.name for s in shards]
                fo.write("%d %d %d %d %d %d %d %d %s\n"
                         % (l, gt, dt, off, blob, *offs, " ".join(names)))
    if a.skip_experts or not a.experts_bin:
        if (out / "experts.bin").exists() and not a.experts_bin:
            print("note: %s/experts.bin exists; the engine reads it instead of the GGUF" % out)
        return 0
    path = out / "experts.bin"
    if path.exists() and path.stat().st_size == offset:
        print("experts.bin exists with the right size; not rewritten")
        return 0
    with open(path, "wb") as fo:
        for l, gt, dt, off, blob, ts in layout:
            parts = [model.bytes(t.name).reshape(n_expert, -1) for t in ts]
            chunk = np.concatenate(parts, axis=1)          # (n_expert, blob): gate | up | down per expert
            assert chunk.shape == (n_expert, blob)
            fo.write(chunk.tobytes())
            if l % 8 == 0:
                print("  layer %2d  %-8s/%-7s blob %8d  at %.2f GiB" % (l, ts[0].type_name, ts[2].type_name, blob,
                                                                        off / 2**30), flush=True)
    print("experts.bin: %d layers, %.2f GiB" % (n_layers, offset / 2**30))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
