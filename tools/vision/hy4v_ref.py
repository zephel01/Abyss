"""tools/vision/hy4v_ref.py - the reference for the hy4v encoder (Tencent Hy4-flash's vision tower) in numpy, on the
mmproj's own weights, and the vocab-only GGUF strata-vision opens beside it.

Written for Abyss from transformers' HYV4VL vision modules (the tower Hy4-flash's config builds): read for the
semantics, not transcribed.  The tower: 16x16 patches (conv with bias) in raster order + the learned 128 x 128
position grid resampled bilinearly (align_corners=False, scale_factor (n + 0.1) / 128) -> 27 blocks (LayerNorm,
full attention with biases, LayerNorm, fc1 / GELU (erf) / fc2) -> RMSNorm -> 2x2 conv to 4C + GELU -> 1x1 conv to
the text width -> one image_newline row after each merged row.  The image is taken at its own size, which must be
a multiple of 32 within the processor's pixel limits (the processor would not resize it either).

    python hy4v_ref.py image --out test.png [--w 512 --h 384]       # a test picture
    python hy4v_ref.py ref --mmproj mmproj.gguf --image test.png --out ref.npy
    python hy4v_ref.py vocab --gguf <model>-00001-of-0000N.gguf --out vocab.gguf [--gguf-py DIR]
    python hy4v_ref.py compare --ref ref.npy --sve out.sve
"""
from __future__ import annotations

import argparse
import math
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import gguf_io  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402


def cmd_image(a):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (a.w, a.h), (236, 232, 222))
    d = ImageDraw.Draw(im)
    d.rectangle([30, 40, 200, 180], fill=(200, 40, 40))
    d.ellipse([240, 60, 420, 240], fill=(30, 90, 200))
    d.polygon([(60, a.h - 20), (180, 200), (300, a.h - 20)], fill=(40, 160, 70))
    d.text((330, a.h - 60), "HY4 42", fill=(10, 10, 10))
    im.save(a.out)
    print(a.out)


class Weights:
    def __init__(self, path):
        self.g = GGUFFile(path)
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        self.by = {t.name: t for t in self.g.tensors}

    def __call__(self, name):
        t = self.by[name]
        raw = self.mm[self.g.data_start + t.offset: self.g.data_start + t.offset + t.expected_bytes()]
        return gguf_io.dequantize_any(np.asarray(raw), t.type_id).astype(np.float64).reshape(tuple(reversed(t.shape)))


def layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu(x):
    """GELU with the exact erf (transformers' "gelu"), elementwise through math.erf."""
    return 0.5 * x * (1.0 + np.frompyfunc(math.erf, 1, 1)(x / math.sqrt(2.0)).astype(np.float64))


def axis_weights(n, edge=128):
    w = np.zeros((n, edge))
    scale = (n + 0.1) / edge
    for o in range(n):
        src = max((o + 0.5) / scale - 0.5, 0.0)
        i0 = min(int(math.floor(src)), edge - 1)
        i1 = min(i0 + 1, edge - 1)
        l1 = src - i0
        w[o, i0] += 1.0 - l1
        w[o, i1] += l1
    return w


def cmd_ref(a):
    from PIL import Image
    W = Weights(a.mmproj)
    md = W.g.metadata
    P = int(md["clip.vision.patch_size"])
    heads = int(md["clip.vision.attention.head_count"])
    eps = float(md["clip.vision.attention.layer_norm_epsilon"])
    n_layer = int(md["clip.vision.block_count"])
    im = np.asarray(Image.open(a.image).convert("RGB"), dtype=np.float64) / 255.0
    H, Wd = im.shape[:2]
    if H % (2 * P) or Wd % (2 * P):
        raise SystemExit(f"{a.image}: {Wd}x{H} is not a multiple of {2 * P} (the reference does not resize)")
    x = (im - 0.5) / 0.5                                                   # mean = std = 0.5
    gh, gw = H // P, Wd // P
    patches = x.reshape(gh, P, gw, P, 3).transpose(0, 2, 4, 1, 3).reshape(gh * gw, 3 * P * P)   # (c, ky, kx)
    k = W("v.patch_embd.weight")                                           # (C, 3, P, P)
    C = k.shape[0]
    h = patches @ k.reshape(C, -1).T + W("v.patch_embd.bias")
    pe = W("v.position_embd.weight").reshape(128, 128, C)                  # [y][x]
    pe = np.einsum("yi,xj,ijc->yxc", axis_weights(gh), axis_weights(gw), pe).reshape(gh * gw, C)
    h = h + pe
    d = C // heads
    for il in range(n_layer):
        b = f"v.blk.{il}."
        y = layer_norm(h, W(b + "ln1.weight"), W(b + "ln1.bias"), eps)
        q = (y @ W(b + "attn_q.weight").T + W(b + "attn_q.bias")).reshape(-1, heads, d).transpose(1, 0, 2)
        kk = (y @ W(b + "attn_k.weight").T + W(b + "attn_k.bias")).reshape(-1, heads, d).transpose(1, 0, 2)
        v = (y @ W(b + "attn_v.weight").T + W(b + "attn_v.bias")).reshape(-1, heads, d).transpose(1, 0, 2)
        s = q @ kk.transpose(0, 2, 1) / math.sqrt(d)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = (s @ v).transpose(1, 0, 2).reshape(-1, C)
        h = h + o @ W(b + "attn_out.weight").T + W(b + "attn_out.bias")
        y = layer_norm(h, W(b + "ln2.weight"), W(b + "ln2.bias"), eps)
        y = gelu(y @ W(b + "ffn_up.weight").T + W(b + "ffn_up.bias"))
        h = h + y @ W(b + "ffn_down.weight").T + W(b + "ffn_down.bias")
        print(f"block {il} done", file=sys.stderr, flush=True)
    h = h / np.sqrt((h * h).mean(-1, keepdims=True) + 1e-5) * W("mm.pre_norm.weight")
    g = h.reshape(gh // 2, 2, gw // 2, 2, C).transpose(0, 2, 4, 1, 3).reshape(gh // 2, gw // 2, C * 4)   # (c, ky, kx)
    k0 = W("mm.0.weight")                                                   # (4C, C, 2, 2)
    m = gelu(g @ k0.reshape(k0.shape[0], -1).T + W("mm.0.bias"))
    k2 = W("mm.2.weight")                                                   # (out, 4C, 1, 1)
    out = m @ k2.reshape(k2.shape[0], -1).T + W("mm.2.bias")                # (rows, cols, out)
    nl = np.broadcast_to(W("v.image_newline"), (gh // 2, 1, out.shape[-1]))
    out = np.concatenate([out, nl], axis=1).reshape(-1, out.shape[-1])
    np.save(a.out, out.astype(np.float32))
    print(f"{a.out}: {out.shape[0]} rows ({gw // 2 + 1} x {gh // 2}) x {out.shape[1]}")


def cmd_vocab(a):
    """A vocab-only GGUF under an architecture the pinned llama.cpp knows, with M-RoPE (qwen2vl), so mtmd lays the
    image out as a grid: the tokenizer as stored, its pre-tokenizer named 'default' (strata-vision tokenizes only the
    media marker; the server tokenizes prompts itself)."""
    if a.gguf_py:
        sys.path.insert(0, a.gguf_py)
    import gguf
    r = gguf.GGUFReader(a.gguf)
    w = gguf.GGUFWriter(a.out, arch="qwen2vl")
    w.add_string("general.name", "Hy4-flash vocabulary (for strata-vision)")
    w.add_uint32("qwen2vl.context_length", 65536)
    w.add_uint32("qwen2vl.embedding_length", 3072)
    w.add_uint32("qwen2vl.block_count", 1)
    w.add_uint32("qwen2vl.feed_forward_length", 3072)
    w.add_uint32("qwen2vl.attention.head_count", 24)
    w.add_float32("qwen2vl.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_array("qwen2vl.rope.dimension_sections", [12, 10, 10, 0])
    for k, f in r.fields.items():
        if not k.startswith("tokenizer."):
            continue
        t = f.types[0]
        v = f.contents()
        if k == "tokenizer.ggml.pre":
            v = "default"
        if t == gguf.GGUFValueType.ARRAY:
            w.add_array(k, v)
        elif t == gguf.GGUFValueType.STRING:
            w.add_string(k, v)
        elif t == gguf.GGUFValueType.BOOL:
            w.add_bool(k, v)
        elif t == gguf.GGUFValueType.UINT32:
            w.add_uint32(k, v)
        elif t == gguf.GGUFValueType.INT32:
            w.add_int32(k, v)
        else:
            w.add_key_value(k, v, t)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(a.out)


def cmd_compare(a):
    ref = np.load(a.ref).astype(np.float64)
    raw = open(a.sve, "rb").read()
    magic, n, nx, ny, n_embd = struct.unpack("<5i", raw[:20])
    got = np.frombuffer(raw[20:20 + 4 * n * n_embd], dtype=np.float32).reshape(n, n_embd).astype(np.float64)
    if got.shape != ref.shape:
        raise SystemExit(f"shape {got.shape} vs reference {ref.shape} (grid {nx}x{ny})")
    cos = (got * ref).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1) + 1e-30)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"{n} tokens x {n_embd} (grid {nx}x{ny}): cosine per token min {cos.min():.5f} mean {cos.mean():.6f}, "
          f"relative error {rel:.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("image"); p.add_argument("--out", required=True)
    p.add_argument("--w", type=int, default=512); p.add_argument("--h", type=int, default=384)
    p = sub.add_parser("ref"); p.add_argument("--mmproj", required=True); p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("vocab"); p.add_argument("--gguf", required=True); p.add_argument("--out", required=True)
    p.add_argument("--gguf-py", default=None)
    p = sub.add_parser("compare"); p.add_argument("--ref", required=True); p.add_argument("--sve", required=True)
    a = ap.parse_args()
    {"image": cmd_image, "ref": cmd_ref, "vocab": cmd_vocab, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
