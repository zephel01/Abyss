"""Re-quantizes a few dense tensors of a (split) GGUF and copies everything else byte for byte.

usage: requant_dense.py <in shard 1> <out dir> <type> <regex> [<regex> ...] [--check] [--from=<type>[,<type>...]]
  e.g. requant_dense.py .../UD-Q2_K_XL/DeepSeek-...-00001-of-00003.gguf .../UD-Q2_K_XL-attn-q6k q6_k \
         'blk\\.\\d+\\.attn_q_b\\.weight' 'blk\\.\\d+\\.attn_output_[ab]\\.weight'
Each input shard becomes one output shard of the same name (the split metadata is copied as is).  The matched
tensors are dequantized by tools/gguf_io.py (the plain types: Q8_0, the k-quants, F16/BF16/F32) and quantized by
ggml_quantize_chunk (libggml-base.so: no imatrix for the k-quants); llama-quantize cannot do this on an i-quant model
without an imatrix, even for tensors it leaves alone.  The quantizer is the one step that needs llama.cpp: LIBGGML (or
trench-gguf's / llama.cpp's build, found by gguf_io.ggml_lib) names the libggml-base shared library.  Examples and the quality / speed of the packs made this way: docs/MANUAL.md,
"A lighter dense pack".
Only Q8_0 tensors are re-quantized unless --from names the source types (e.g. --from=q6_k for GLM-5.3-Flash AJ, whose
dense weights are all Q6_K): a k-quant source is dequantized first, so its rounding error and the new one add up.
"""
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_io as G  # noqa: E402


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--check" and not a.startswith("--from=")]
    check = "--check" in sys.argv
    src_types = {G.TYPE_ID["Q8_0"]}
    for a in sys.argv[1:]:
        if a.startswith("--from="):
            src_types = {G.TYPE_ID[n.upper()] for n in a[len("--from="):].split(",")}
    if len(args) < 4:
        print(__doc__)
        return 2
    shard1, out_dir, type_name = args[0], args[1], args[2].upper()
    pats = [re.compile(p) for p in args[3:]]
    new_type = G.TYPE_ID[type_name]
    try:
        lib = None if check else G.ggml_lib()
    except G.NeedsLlamaCpp as e:
        print(e)
        return 1
    _, bs, ts = G.TYPES[new_type]
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", os.path.basename(shard1))
    shards = [shard1] if m is None else [
        os.path.join(os.path.dirname(shard1), "%s-%05d-of-%s.gguf" % (m.group(1), i, m.group(3)))
        for i in range(1, int(m.group(3)) + 1)]
    os.makedirs(out_dir, exist_ok=True)
    total_old = total_new = 0
    arch = None
    for sh in shards:
        r = G.Reader(sh)
        if "general.architecture" in r.fields:   # (the later shards of a split carry only the split keys)
            arch = r.get("general.architecture")
        out = os.path.join(out_dir, os.path.basename(sh))
        print("%s -> %s" % (sh, out), flush=True)
        plan = []
        for t in r.tensors:
            hit = any(p.search(t.name) for p in pats)
            if hit and t.type not in src_types:
                print("  %s is %s, not %s: left as is" % (t.name, t.type_name,
                                                         "/".join(sorted(G.type_name(q) for q in src_types))))
                hit = False
            if hit:
                ne0, ne1 = int(t.shape[0]), int(t.shape[1]) if len(t.shape) > 1 else 1
                if ne0 % bs != 0:
                    print("  %s: ne0 %d not a multiple of %d: left as is" % (t.name, ne0, bs))
                    hit = False
            if hit:
                nbytes = ne1 * (ne0 // bs) * ts
                plan.append((t, ne0, ne1, nbytes))
                total_old += t.nbytes
                total_new += nbytes
                print("  %-40s %s %9.2f MiB -> %s %9.2f MiB" % (t.name, t.type_name, t.nbytes / 2**20,
                                                             type_name, nbytes / 2**20))
            else:
                plan.append((t, 0, 0, 0))
        if check:
            continue
        # (the architecture key goes into every output shard, the later ones too, as these files always had it)
        w = G.Writer(out, arch)
        G.copy_fields(r, w)
        t0 = time.time()
        done = [0]

        def requant(t, ne0, ne1, nbytes):
            def make():
                f32 = np.ascontiguousarray(G.dequantize(t.data, t.type).reshape(ne1, ne0), dtype=np.float32)
                dst = np.empty(nbytes, dtype=np.uint8)
                got = lib.ggml_quantize_chunk(new_type, f32.ctypes.data, dst.ctypes.data, 0, ne1, ne0, None)
                if got != nbytes:
                    raise RuntimeError("ggml_quantize_chunk wrote %d bytes, expected %d for %s" % (got, nbytes, t.name))
                return dst
            return make

        def progress(name):
            n = sizes[name]
            done[0] += n
            if int(done[0] / 2**30) != int((done[0] - n) / 2**30):
                print("  %.1f GiB written, %.0f s" % (done[0] / 2**30, time.time() - t0), flush=True)
        sizes = {}
        for t, ne0, ne1, nbytes in plan:
            sizes[t.name] = t.nbytes
            if nbytes:
                w.add_tensor(t.name, [ne0, ne1], new_type, requant(t, ne0, ne1, nbytes))
            else:
                w.add_tensor(t.name, t.shape, t.type, t.data)
        w.write(progress)
    print("re-quantized %.2f GiB -> %.2f GiB%s" % (total_old / 2**30, total_new / 2**30, " (check only)" if check else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
