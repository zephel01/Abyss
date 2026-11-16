"""A small overlay shard over a split GGUF: a few dense tensors re-quantized and/or edited, the model itself untouched.

usage: dense_overlay.py <in shard 1> <out dir> [--type=q5_k] [--from=q6_k[,...]] [--abl=<edit.npz>] [<regex> ...]
  <out dir>/overlay-00001-of-0000N.gguf holds shard 1's metadata plus the changed tensors; overlay-00002.. are symlinks
  to the model's shards 1..N-1.  Pack it with `iq_pack.py --gguf <out dir>/overlay-00001-of-0000N.gguf --first-wins`:
  a tensor in several shards is taken from the first, so the changed copies shadow the model's (the engine looks
  tensors up in shard order too).  The overlay costs only the changed tensors' bytes (DeepSeek V4.1 UD-IQ1_S: 1.4 GB
  for the abliteration edit, 3.2 GB for its dense Q6_K -> Q5_K) instead of another copy of the model.
  <regex> ...    tensors of a --from type (default Q6_K) re-quantized to --type (default Q5_K): dequantized by
                 tools/gguf_io.py, quantized by ggml_quantize_chunk (libggml-base.so, no imatrix), so the two
                 roundings add up
  --abl=<npz>    a refusal-direction edit as mrfakename/DeepSeek-V4.1-Flash-1017-abliterated ships it
                 (abliteration-edit.npz: Q [n_embd, r] orthonormal, scales [layer, (attn_output_b, ffn_down_shexp)]):
                 W' = W - s Q Q^T W on the rows (the residual writes) of blk.N.attn_output_b / blk.N.ffn_down_shexp,
                 re-quantized to the tensor's own type unless a <regex> also matches it
LIBGGML (a libggml-base shared library: the one step that needs llama.cpp) as for requant_dense.py.  Quality and speed of the overlays made this way: docs/deepseek41/README.md.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_io as G  # noqa: E402

ABL = re.compile(r"^blk\.(\d+)\.(attn_output_b|ffn_down_shexp)\.weight$")


def main() -> int:
    opts = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2 or (len(args) < 3 and "--abl" not in opts):
        print(__doc__)
        return 2
    shard1, out_dir = args[0], args[1]
    pats = [re.compile(p) for p in args[2:]]
    new_type = G.TYPE_ID[opts.get("--type", "q5_k").upper()]
    src_types = {G.TYPE_ID[n.upper()] for n in opts.get("--from", "q6_k").split(",")}
    abl = None
    if "--abl" in opts:
        z = np.load(opts["--abl"])
        abl = (z["Q"].astype(np.float64), z["scales"])

    def abl_scale(name):
        m = ABL.match(name) if abl is not None else None
        if m is None or int(m.group(1)) >= abl[1].shape[0]:
            return 0.0
        return float(abl[1][int(m.group(1)), 0 if m.group(2) == "attn_output_b" else 1])

    try:
        lib = G.ggml_lib()
    except G.NeedsLlamaCpp as e:
        print(e)
        return 1
    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", os.path.basename(shard1))
    if m is None:
        print("%s is not shard 1 of a split GGUF (<name>-00001-of-0000N.gguf)" % shard1)
        return 2
    total = int(m.group(2))
    shards = [os.path.join(os.path.dirname(shard1), "%s-%05d-of-%05d.gguf" % (m.group(1), i, total))
              for i in range(1, total + 1)]
    n_out = total + 1
    os.makedirs(out_dir, exist_ok=True)
    r0 = G.Reader(shards[0])
    w = G.Writer(os.path.join(out_dir, "overlay-00001-of-%05d.gguf" % n_out), r0.get("general.architecture"))
    G.copy_fields(r0, w, skip=lambda k: k.startswith("split."))
    plan, old, new = [], 0, 0
    for sh in shards:
        for t in G.Reader(sh).tensors:
            hit = t.type in src_types and any(p.search(t.name) for p in pats)
            sc = abl_scale(t.name)
            if not hit and sc == 0.0:
                continue
            ty = new_type if hit else t.type
            _, bs, ts = G.TYPES[ty]
            ne0 = int(t.shape[0])
            ne1 = int(np.prod(t.shape[1:])) if len(t.shape) > 1 else 1
            if ne0 % bs:
                print("  %s: ne0 %d not a multiple of %d: left as is" % (t.name, ne0, bs))
                continue
            if sc != 0.0 and ne1 != abl[0].shape[0]:
                print("  %s: %d rows, the edit has %d: left as is" % (t.name, ne1, abl[0].shape[0]))
                continue
            nb = ne1 * (ne0 // bs) * ts
            plan.append((t, ne0, ne1, nb, ty, sc))
            old += t.nbytes
            new += nb
    n_abl = [0]

    def make(t, ne0, ne1, nb, ty, sc):
        def run():
            f32 = np.ascontiguousarray(G.dequantize(t.data, t.type).reshape(ne1, ne0), dtype=np.float32)
            if sc != 0.0:   # the rows are the residual (n_embd outputs): the edit takes the subspace out of what it writes
                Qm = abl[0]
                f32 = np.ascontiguousarray(f32 - (sc * (Qm @ (Qm.T @ f32.astype(np.float64)))).astype(np.float32))
                n_abl[0] += 1
            dst = np.empty(nb, dtype=np.uint8)
            got = lib.ggml_quantize_chunk(int(ty), f32.ctypes.data, dst.ctypes.data, 0, ne1, ne0, None)
            if got != nb:
                raise RuntimeError("ggml_quantize_chunk wrote %d bytes, expected %d for %s" % (got, nb, t.name))
            return dst
        return run
    for t, ne0, ne1, nb, ty, sc in plan:
        w.add_tensor(t.name, [ne0, ne1], ty, make(t, ne0, ne1, nb, ty, sc))
    w.write()
    n_abl = n_abl[0]
    for k, sh in enumerate(shards, start=2):
        dst = os.path.join(out_dir, "overlay-%05d-of-%05d.gguf" % (k, n_out))
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(sh), dst)
    print("overlay: %d tensors (%d edited), %.2f GB -> %.2f GB, over %d shards -> %s"
          % (len(plan), n_abl, old / 1e9, new / 1e9, total, out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
