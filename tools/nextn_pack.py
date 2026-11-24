#!/usr/bin/env python3
"""nextn_pack.py - the NextN (MTP) block of a split GGUF into one small GGUF the engine can draft with.

    .venv/bin/python tools/nextn_pack.py --gguf <first shard> --out <nextn.gguf> [--experts q4_k] [--ggml-lib <libggml-base.so>]

The block's tensors (blk.<block_count - nextn_predict_layers>.*) are copied as stored, except the routed experts, which
are requantized to a type the engine's expert kernels serve (the default Q4_K; q5_k, q6_k; the block's own are often
Q8_0, as in unsloth's Ornith-1.5-122B GGUFs, and Q8_0 experts have no decode kernel yet) through ggml's quantizer
(ggml_quantize_chunk of a libggml-base shared library: trench-gguf's or llama.cpp's build, LIBGGML or --ggml-lib;
the only step here that needs llama.cpp - reading and writing the GGUF is tools/gguf_io.py). The model's general.* and
<arch>.* keys come along so the file reads as the same model. Load it with STRATA_GLM_MTP_GGUF=<nextn.gguf> (the
engine looks the block up there first; the trunk, the head and the embedding stay in the model's shards).
"""
import argparse
import os
import sys
import time

import numpy as np


def q4_0_to_q5_0(data) -> np.ndarray:
    """Q4_0 blocks -> Q5_0 blocks with the same values: a Q4_0 block's d with codes c - 8 (c in 0..15) is Q5_0's d
    with codes c + 8 - 16; Q5_0 keeps the low 4 bits in the same nibbles and the fifth bit of element j in qh bit j."""
    b4 = np.asarray(data, dtype=np.uint8).reshape(-1, 18)
    d, qs = b4[:, :2], b4[:, 2:]
    lo = (qs & 15).astype(np.uint16) + 8
    hi = (qs >> 4).astype(np.uint16) + 8
    codes = np.concatenate([lo, hi], axis=1)           # [blocks][32]: elements 0..15 then 16..31
    bit = ((codes >> 4) & 1).astype(np.uint32)
    qh = (bit << np.arange(32, dtype=np.uint32)).sum(axis=1, dtype=np.uint32)
    out = np.empty((b4.shape[0], 22), dtype=np.uint8)
    out[:, :2] = d
    out[:, 2:6] = qh.astype("<u4").view(np.uint8).reshape(-1, 4)
    out[:, 6:] = ((lo & 15) | ((hi & 15) << 4)).astype(np.uint8)
    return out.reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="the model's first shard (or its only file)")
    ap.add_argument("--out", required=True, help="the NextN GGUF to write")
    ap.add_argument("--experts", default="q4_k", help="the routed experts' type: q4_k (default), q5_k, q6_k, q8_0 (as "
                    "stored), or q5_0 (from Q4_0 experts: the same values, repacked here - no ggml)")
    ap.add_argument("--ggml-lib", default="", help="libggml-base.so (.dylib / ggml-base.dll) with ggml_quantize_chunk")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gguf_io as G

    want = {"q4_k": 12, "q5_k": 13, "q6_k": 14, "q8_0": 8, "q5_0": 6}.get(a.experts.lower())
    if want is None:
        print("--experts: q4_k, q5_k, q6_k, q8_0 or q5_0")
        return 2
    first = G.Reader(a.gguf)
    arch = first.get("general.architecture")
    block_count = int(first.get(arch + ".block_count", 0))
    nextn = int(first.get(arch + ".nextn_predict_layers", 0))
    if block_count <= 0 or nextn <= 0:
        print("%s: block_count %d, nextn_predict_layers %d - no NextN block" % (arch, block_count, nextn))
        return 1
    L = block_count - nextn
    P = "blk.%d." % L
    # the shards: split.count files named ...-0000i-of-0000n.gguf
    paths = [a.gguf]
    n_split = int(first.get("split.count", 1))
    if n_split > 1:
        base = os.path.basename(a.gguf)
        i = base.rfind("-00001-of-")
        if i < 0:
            print("the shard name does not carry -00001-of-")
            return 1
        for k in range(2, n_split + 1):
            paths.append(os.path.join(os.path.dirname(a.gguf), base[:i] + "-%05d-of-%05d.gguf" % (k, n_split)))
    readers = [first] + [G.Reader(p) for p in paths[1:]]
    tensors = [t for r in readers for t in r.tensors if t.name.startswith(P)]
    if not any(t.name == P + "nextn.eh_proj.weight" for t in tensors):
        print("%s: no %snextn.eh_proj.weight in the shards" % (arch, P))
        return 1
    # ggml's quantizer, when an expert tensor changes type (a NeedsLlamaCpp error otherwise)
    lib = None
    if want == 6 and any(t.name.endswith("_exps.weight") and t.type not in (2, 6) for t in tensors):
        print("--experts q5_0 repacks Q4_0 experts only")
        return 1
    if want != 6 and any(t.name.endswith("_exps.weight") and t.type != want for t in tensors):
        try:
            lib = G.ggml_lib(a.ggml_lib)
        except G.NeedsLlamaCpp as e:
            print("%s; --experts q8_0 copies Q8_0 experts as stored" % e)
            return 1
    w = G.Writer(a.out, arch)
    # the model's keys (not the tokenizer's, not the split's); a scalar integer is written as u32 / i32 / u64, as
    # this tool always has
    for key, f in first.fields.items():
        if key == "general.architecture" or key.startswith("tokenizer.") or key.startswith("split.") or key.startswith("GGUF."):
            continue
        if not (key.startswith("general.") or key.startswith(arch + ".")):
            continue
        vt = f.vtype
        if vt == G.STRING:
            w.add_string(key, f.value)
        elif vt in (G.U32, G.I32, G.U64, G.I64, G.U8, G.I8, G.U16, G.I16):
            w.add_uint32(key, f.value) if vt == G.U32 else w.add_int32(key, f.value) if vt == G.I32 else \
                w.add_uint64(key, f.value)
        elif vt == G.F32V:
            w.add_float32(key, f.value)
        elif vt == G.BOOL:
            w.add_bool(key, f.value)
        elif vt == G.ARRAY:
            if f.etype == G.STRING:
                w.add_array(key, list(f.value))
            elif f.etype == G.BOOL:
                w.add_array(key, [bool(x) for x in f.value])
            elif f.etype == G.F32V:
                w.add_array(key, [float(x) for x in f.value])
            else:
                w.add_array(key, [int(x) for x in f.value])
    total = 0
    for t in tensors:
        data, qt = t.data, t.type
        if t.name.endswith("_exps.weight") and qt == 2 and want == 6:
            data = q4_0_to_q5_0(data)
            print("%s: Q4_0 -> Q5_0 (the same values), %.1f MB" % (t.name, data.nbytes / 1e6), flush=True)
            qt = 6
        elif t.name.endswith("_exps.weight") and qt != want:
            t0 = time.time()
            n_per_row = int(t.shape[0])
            f32 = np.ascontiguousarray(G.dequantize(data, qt), dtype=np.float32)
            nrows = int(f32.size // n_per_row)
            out = np.empty(nrows * G.row_bytes(want, n_per_row), dtype=np.uint8)
            wrote = lib.ggml_quantize_chunk(want, f32.ctypes.data, out.ctypes.data, 0, nrows, n_per_row, None)
            if wrote != out.nbytes:
                print("%s: ggml_quantize_chunk wrote %d bytes, expected %d" % (t.name, wrote, out.nbytes))
                return 1
            data = out
            del f32
            print("%s: %s -> %s, %.1f MB, %.0f s" % (t.name, G.type_name(qt), G.type_name(want), data.nbytes / 1e6,
                                                    time.time() - t0), flush=True)
            qt = want
        else:
            print("%s: %s as stored, %.1f MB" % (t.name, G.type_name(qt), data.nbytes / 1e6), flush=True)
        w.add_tensor(t.name, t.shape, qt, data)
        total += data.nbytes
    w.write()
    print("%s: %d tensors of blk.%d, %.2f GB -> %s" % (arch, len(tensors), L, total / 1e9, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
