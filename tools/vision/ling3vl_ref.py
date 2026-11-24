"""tools/vision/ling3vl_ref.py - the float64 numpy reference for Ling-3.2-air's image encoder (inclusionAI's
BailingMoeV3VL: a Qwen3-style ViT without DeepStack + the language projector), on the mmproj's own weights.

Written for Abyss from the HF model's code (modeling_bailing_moe_v3_vl.py, image_processing_bailingmm2.py) - read for
the semantics, not transcribed, and nothing of it is imported or run.  The weights are read with Abyss's own
tools/gguf_io.py; `hfcheck` reads the HF safetensors with the few lines below (no safetensors package).

  preprocessing: RGB, smart_resize (both sides multiples of 32, min 4096 / max cap * 1024 pixels, the aspect kept),
      PIL bicubic resize, x / 255, (x - mean) / std with the processor's CLIP mean / std; the picture is one frame
      repeated (temporal patch 2), so the 3-D patch conv is the 2-D conv with the two time slices' kernels added.
  tower: 16x16 patches (+ bias) + the learned 48 x 48 position grid resampled bilinearly with corners aligned
      (linspace(0, 47, n)) -> 27 blocks: LayerNorm (eps 1e-6), qkv (+ bias), 2-D rotary on head_dim 72 (NEOX halves
      of 36: the first 18 frequencies take the patch row, the next 18 its column, theta 10000^(-2j/36)), full
      attention scale 72^-0.5, out proj; LayerNorm, fc1, GELU (tanh), fc2 -> the merger's LayerNorm per patch (no
      merger projection: disable_merger_proj) -> 2x2 patches concatenated in row-major order (4608) -> linear_proj:
      Linear 4608 -> 2560, GELU (erf), Linear 2560 -> 2560.
  out: one row per 2x2 block, raster order over the merged grid (nx = columns / 2, ny = rows / 2).

    python ling3vl_ref.py images --out DIR                          # the test pictures (shapes, text, chart)
    python ling3vl_ref.py ref --mmproj mmproj.gguf --image x.png --out ref.npy [--cap 1024]
    python ling3vl_ref.py prep --image x.png --out x.lvi [--cap 1024]   # abyss-vision's input, by hand
    python ling3vl_ref.py compare --ref ref.npy --sve out.sve
    python ling3vl_ref.py hfcheck --mmproj mmproj.gguf --hf DIR      # every mmproj tensor vs the HF checkpoint
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import gguf_io  # noqa: E402

MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
STD = np.array([0.26862954, 0.26130258, 0.27577711])
PATCH, MERGE, FACTOR = 16, 2, 32


def smart_resize(h: int, w: int, factor: int = FACTOR, min_pixels: int = 4096, max_pixels: int = 4096 * 1024):
    if max(h, w) / min(h, w) > 200:
        raise ValueError("the aspect ratio must stay under 200")
    hb, wb = round(h / factor) * factor, round(w / factor) * factor
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def load_rgb(path, cap: int):
    """-> uint8 [h][w][3], resized as the HF processor does."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    rh, rw = smart_resize(h, w, max_pixels=cap * FACTOR * FACTOR)
    if (rw, rh) != (w, h):
        im = im.resize((rw, rh), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


class Weights:
    def __init__(self, path):
        self.r = gguf_io.Reader(path)

    def __call__(self, name):
        t = self.r.by_name[name]
        v = gguf_io.dequantize(t.data, t.type).astype(np.float64)
        return v.reshape(list(reversed([int(x) for x in t.shape])))


def layer_norm(x, w, b, eps=1e-6):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_erf(x):
    from math import erf
    return 0.5 * x * (1 + np.vectorize(erf)(x / math.sqrt(2)))


def pos_grid(pe, gh, gw):
    """The 48 x 48 learned grid resampled to gh x gw (bilinear, corners aligned) -> [gh * gw][C], raster order."""
    side = int(round(math.sqrt(pe.shape[0])))
    hi, wi = np.linspace(0, side - 1, gh), np.linspace(0, side - 1, gw)
    hf, wf = hi.astype(np.int64), wi.astype(np.int64)
    hc, wc = np.minimum(hf + 1, side - 1), np.minimum(wf + 1, side - 1)
    dh, dw = (hi - hf)[:, None, None], (wi - wf)[None, :, None]
    g = pe.reshape(side, side, -1)
    out = (g[hf][:, wf] * (1 - dh) * (1 - dw) + g[hf][:, wc] * (1 - dh) * dw +
           g[hc][:, wf] * dh * (1 - dw) + g[hc][:, wc] * dh * dw)
    return out.reshape(gh * gw, -1)


def rope2d(x, ys, xs):
    """x [n][heads][72]: NEOX halves of 36, frequencies 0..17 by the row, 18..35 by the column."""
    d = x.shape[-1]
    half = d // 2
    inv = 1.0 / 10000.0 ** (np.arange(0, half, 2) / half)          # 18
    ang = np.concatenate([ys[:, None] * inv, xs[:, None] * inv], 1)  # [n][36]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., :half], x[..., half:]
    return np.concatenate([a * c - b * s, b * c + a * s], -1)


def encode(W: Weights, rgb: np.ndarray, log=print):
    h, w, _ = rgb.shape
    gh, gw = h // PATCH, w // PATCH
    x = (rgb.astype(np.float64) / 255.0 - MEAN) / STD                        # [h][w][3]
    p = x.reshape(gh, PATCH, gw, PATCH, 3).transpose(0, 2, 4, 1, 3).reshape(gh * gw, 3 * PATCH * PATCH)
    k = (W("v.patch_embd.weight") + W("v.patch_embd.weight.1")).reshape(-1, 3 * PATCH * PATCH)   # [C][c,kh,kw]
    hs = p @ k.T + W("v.patch_embd.bias")
    hs = hs + pos_grid(W("v.position_embd.weight"), gh, gw)
    ys, xs = np.repeat(np.arange(gh), gw).astype(np.float64), np.tile(np.arange(gw), gh).astype(np.float64)
    nl = int(W.r.get("clip.vision.block_count"))
    nh = int(W.r.get("clip.vision.attention.head_count"))
    C = hs.shape[1]
    dh = C // nh
    for il in range(nl):
        P = f"v.blk.{il}."
        y = layer_norm(hs, W(P + "ln1.weight"), W(P + "ln1.bias"))
        qkv = y @ W(P + "attn_qkv.weight").T + W(P + "attn_qkv.bias")
        q, kk, v = (qkv[:, i * C:(i + 1) * C].reshape(-1, nh, dh) for i in range(3))
        q, kk = rope2d(q, ys, xs), rope2d(kk, ys, xs)
        o = np.empty_like(q)
        for hh in range(nh):
            sc = q[:, hh] @ kk[:, hh].T * dh ** -0.5
            sc = np.exp(sc - sc.max(1, keepdims=True))
            o[:, hh] = (sc / sc.sum(1, keepdims=True)) @ v[:, hh]
        hs = hs + o.reshape(-1, C) @ W(P + "attn_out.weight").T + W(P + "attn_out.bias")
        y = layer_norm(hs, W(P + "ln2.weight"), W(P + "ln2.bias"))
        y = gelu_tanh(y @ W(P + "ffn_up.weight").T + W(P + "ffn_up.bias"))
        hs = hs + y @ W(P + "ffn_down.weight").T + W(P + "ffn_down.bias")
        if log and (il % 9 == 8 or il == nl - 1):
            log(f"  block {il + 1}/{nl}")
    hs = layer_norm(hs, W("mm.input_norm.weight"), W("mm.input_norm.bias"))
    m = hs.reshape(gh // 2, 2, gw // 2, 2, C).transpose(0, 2, 1, 3, 4).reshape(-1, 4 * C)   # (dy, dx) row-major
    m = gelu_erf(m @ W("mm.0.weight").T + W("mm.0.bias"))
    m = m @ W("mm.2.weight").T + W("mm.2.bias")
    return m, gw // 2, gh // 2


def cmd_ref(a):
    rgb = load_rgb(a.image, a.cap)
    W = Weights(a.mmproj)
    print(f"{a.image}: {rgb.shape[1]}x{rgb.shape[0]} pixels -> {rgb.shape[1] // 32}x{rgb.shape[0] // 32} rows")
    emb, nx, ny = encode(W, rgb)
    np.save(a.out, emb)
    pathlib.Path(a.out).with_suffix(".grid.json").write_text(json.dumps({"nx": nx, "ny": ny}))
    print(f"wrote {a.out}: {emb.shape[0]} rows x {emb.shape[1]} (grid {nx}x{ny})")


def cmd_prep(a):
    """The picture as abyss-vision reads it (the server writes the same: Vision.prepare in serve/server.py)."""
    rgb = load_rgb(a.image, a.cap)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<3i", 0x3149564C, rgb.shape[1], rgb.shape[0]))
        f.write(rgb.tobytes())
    print(f"wrote {a.out}: {rgb.shape[1]}x{rgb.shape[0]}")


def read_sve(path):
    b = pathlib.Path(path).read_bytes()
    magic, n, nx, ny, e = struct.unpack("<5i", b[:20])
    assert magic == 0x31455653, "not a strata-vision record"
    return np.frombuffer(b[20:20 + 4 * n * e], dtype=np.float32).reshape(n, e).astype(np.float64), nx, ny


def cmd_compare(a):
    ref = np.load(a.ref)
    got, nx, ny = read_sve(a.sve)
    assert got.shape == ref.shape, f"shapes differ: {got.shape} vs {ref.shape}"
    cos = (got * ref).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1) + 1e-30)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"{got.shape[0]} rows x {got.shape[1]} (grid {nx}x{ny}): cosine per row min {cos.min():.6f} mean "
          f"{cos.mean():.6f}, relative error {rel:.2e}")


def hf_tensors(d: pathlib.Path):
    idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    files = {}
    for k, f in idx.items():
        if k.startswith(("model.visual.", "linear_proj.")):
            files.setdefault(f, []).append(k)
    out = {}
    for f, keys in files.items():
        with open(d / f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            base = 8 + n
            for k in keys:
                m = hdr[k]
                s, e = m["data_offsets"]
                fh.seek(base + s)
                raw = np.frombuffer(fh.read(e - s), dtype=np.uint16)
                assert m["dtype"] == "BF16", (k, m["dtype"])
                out[k] = (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(m["shape"])
    return out


def cmd_hfcheck(a):
    W = Weights(a.mmproj)
    hf = hf_tensors(pathlib.Path(a.hf))
    nl = int(W.r.get("clip.vision.block_count"))
    pairs = [("v.patch_embd.weight", lambda: hf["model.visual.patch_embed.proj.weight"][:, :, 0]),
             ("v.patch_embd.weight.1", lambda: hf["model.visual.patch_embed.proj.weight"][:, :, 1]),
             ("v.patch_embd.bias", lambda: hf["model.visual.patch_embed.proj.bias"]),
             ("v.position_embd.weight", lambda: hf["model.visual.pos_embed.weight"]),
             ("mm.input_norm.weight", lambda: hf["model.visual.merger.norm.weight"]),
             ("mm.input_norm.bias", lambda: hf["model.visual.merger.norm.bias"])]
    for i in (0, 2):
        for s in ("weight", "bias"):
            pairs.append((f"mm.{i}.{s}", lambda i=i, s=s: hf[f"linear_proj.{i}.{s}"]))
    names = {"ln1": "norm1", "ln2": "norm2", "attn_qkv": "attn.qkv", "attn_out": "attn.proj",
             "ffn_up": "mlp.linear_fc1", "ffn_down": "mlp.linear_fc2"}
    for il in range(nl):
        for g, h in names.items():
            for s in ("weight", "bias"):
                pairs.append((f"v.blk.{il}.{g}.{s}", lambda il=il, h=h, s=s: hf[f"model.visual.blocks.{il}.{h}.{s}"]))
    worst, seen = 0.0, set()
    for g, fn in pairs:
        mine, theirs = W(g), fn()
        seen.add(g)
        assert mine.size == theirs.size, (g, mine.shape, theirs.shape)
        d = np.abs(mine.reshape(-1) - theirs.reshape(-1)).max() / (np.abs(theirs).max() + 1e-30)
        worst = max(worst, d)
        if d > 1e-3:
            print(f"MISMATCH {g}: max |diff| / max |w| = {d:.3e}")
    left = [t.name for t in W.r.tensors if t.name not in seen]
    print(f"{len(pairs)} tensors compared, worst max |diff| / max |w| = {worst:.3e}; mmproj tensors not compared: "
          f"{left or 'none'}")


def cmd_images(a):
    from PIL import Image, ImageDraw, ImageFont
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ja = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    # shapes: 4 red circles, 2 blue squares, 1 green triangle
    im = Image.new("RGB", (640, 448), (245, 245, 240))
    d = ImageDraw.Draw(im)
    for cx, cy in ((90, 100), (250, 80), (420, 120), (560, 330)):
        d.ellipse([cx - 45, cy - 45, cx + 45, cy + 45], fill=(210, 40, 40))
    for x0, y0 in ((80, 260), (300, 300)):
        d.rectangle([x0, y0, x0 + 100, y0 + 100], fill=(40, 80, 210))
    d.polygon([(470, 200), (400, 300), (540, 300)], fill=(40, 160, 70))
    im.save(out / "shapes.png")
    # text: Japanese and English lines
    im = Image.new("RGB", (768, 320), (255, 255, 255))
    d = ImageDraw.Draw(im)
    f1, f2 = ImageFont.truetype(ja, 44), ImageFont.truetype(ja, 36)
    d.text((30, 30), "東京駅から新大阪駅まで", font=f1, fill=(0, 0, 0))
    d.text((30, 100), "新幹線で約2時間30分", font=f1, fill=(0, 0, 0))
    d.text((30, 190), "Total: 14,800 yen (one way)", font=f2, fill=(20, 20, 120))
    im.save(out / "text_ja.png")
    # a bar chart
    im = Image.new("RGB", (800, 480), (255, 255, 255))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(ja, 26)
    d.text((220, 15), "Monthly sales (units)", font=f, fill=(0, 0, 0))
    vals = (("Jan", 120, (70, 130, 180)), ("Feb", 260, (220, 120, 40)), ("Mar", 180, (60, 160, 90)))
    for i, (lab, v, c) in enumerate(vals):
        x0 = 140 + i * 210
        d.rectangle([x0, 420 - v, x0 + 120, 420], fill=c)
        d.text((x0 + 35, 428), lab, font=f, fill=(0, 0, 0))
        d.text((x0 + 35, 420 - v - 34), str(v), font=f, fill=(0, 0, 0))
    d.line([(100, 420), (760, 420)], fill=(0, 0, 0), width=2)
    im.save(out / "chart.png")
    print("wrote", ", ".join(str(out / n) for n in ("shapes.png", "text_ja.png", "chart.png")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ref")
    p.add_argument("--mmproj", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cap", type=int, default=4096, help="image token cap (max pixels = cap * 1024)")
    p = sub.add_parser("prep")
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cap", type=int, default=4096)
    p = sub.add_parser("compare")
    p.add_argument("--ref", required=True)
    p.add_argument("--sve", required=True)
    p = sub.add_parser("hfcheck")
    p.add_argument("--mmproj", required=True)
    p.add_argument("--hf", required=True)
    p = sub.add_parser("images")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"ref": cmd_ref, "prep": cmd_prep, "compare": cmd_compare, "hfcheck": cmd_hfcheck, "images": cmd_images}[a.cmd](a)


if __name__ == "__main__":
    main()
