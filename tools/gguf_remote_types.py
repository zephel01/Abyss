#!/usr/bin/env python3
"""The routed experts' types of GGUF sets, read from the files' headers only - over HTTP range requests from a
Hugging Face repo (no download), or from local shards.

  tools/gguf_remote_types.py --repo unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ4_XS UD-Q3_K_XL ...
  tools/gguf_remote_types.py --local /path/to/UD-IQ4_XS          (a downloaded set: compare with the fixture)

Prints one line per MoE layer: "<set> <layer> <gate> <up> <down>" (ggml type names), the form of
tools/fixtures/unsloth_q38_ud31_types.txt, which tools/test_unsloth_types.py checks against the engine's type lists
(GPU expert kernels, prompt MMQ, the CPU lane's test table) and iq_pack.py.  When a quant maker re-uploads, run this
first: a new type or pairing shows up here before anyone downloads 90 GB.  A token comes from $HF_TOKEN or the hf
CLI's token file (public repos need none).
"""
import argparse
import json
import os
from pathlib import Path
import re
import struct
import sys
import urllib.parse
import urllib.request

# ggml type ids -> names (ggml.h; the ones GGUF files carry)
TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K",
              12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
              20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 30: "BF16", 39: "MXFP4",
              40: "NVFP4", 41: "Q1_0", 42: "Q2_0"}
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_EXPERT = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def _token():
    t = os.environ.get("HF_TOKEN")
    p = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    return t or (p.read_text().strip() if p.exists() else None)


def _get(url, first=None, last=None):
    h = {"User-Agent": "abyss-gguf-remote-types"}
    tok = _token()
    if tok:
        h["Authorization"] = "Bearer " + tok
    if first is not None:
        h["Range"] = "bytes=%d-%d" % (first, last)
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
        return r.read()


class _Reader:
    """Sequential reads over a source that hands out byte ranges (fetch(first, n) -> bytes)."""

    def __init__(self, fetch):
        self.fetch, self.buf, self.pos = fetch, b"", 0

    def take(self, n):
        while self.pos + n > len(self.buf):
            more = self.fetch(len(self.buf), max(4 << 20, self.pos + n - len(self.buf)))
            if not more:
                raise EOFError("GGUF header runs past the end of the file")
            self.buf += more
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def num(self, f):
        return struct.unpack("<" + f, self.take(struct.calcsize(f)))[0]

    def string(self):
        return self.take(self.num("Q"))


def _skip_value(r, t):
    if t == 8:
        r.string()
    elif t == 9:
        at, n = r.num("I"), r.num("Q")
        if at == 8:
            for _ in range(n):
                r.string()
        else:
            r.take(struct.calcsize(_SCALAR[at]) * n)
    else:
        r.take(struct.calcsize(_SCALAR[t]))


def tensor_types(fetch):
    """{tensor name: ggml type id} of one GGUF file's header."""
    r = _Reader(fetch)
    if r.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    r.num("I")
    n_tensors, n_kv = r.num("Q"), r.num("Q")
    for _ in range(n_kv):
        r.string()
        _skip_value(r, r.num("I"))
    out = {}
    for _ in range(n_tensors):
        name = r.string().decode("utf-8", "replace")
        r.take(8 * r.num("I"))   # dims
        out[name] = r.num("I")
        r.num("Q")               # offset
    return out


def remote_set(repo, name, revision="main"):
    tree = json.loads(_get("https://huggingface.co/api/models/%s/tree/%s/%s"
                           % (repo, urllib.parse.quote(revision, safe=""), urllib.parse.quote(name))))
    files = sorted(e["path"] for e in tree if e["path"].endswith(".gguf"))
    if not files:
        raise SystemExit("%s: no .gguf files under %s/%s" % (repo, revision, name))
    types = {}
    for p in files:
        url = "https://huggingface.co/%s/resolve/%s/%s" % (repo, urllib.parse.quote(revision, safe=""),
                                                          urllib.parse.quote(p))
        types.update(tensor_types(lambda first, n, u=url: _get(u, first, first + n - 1)))
    return types


def local_set(path):
    types = {}
    for p in sorted(Path(path).glob("*.gguf")):
        with open(p, "rb") as f:
            def fetch(first, n, f=f):
                f.seek(first)
                return f.read(n)
            types.update(tensor_types(fetch))
    return types


def layer_lines(set_name, types):
    roles = {}
    for name, t in types.items():
        m = _EXPERT.match(name)
        if m:
            roles.setdefault(int(m.group(1)), {})[m.group(2)] = TYPE_NAMES.get(t, str(t))
    return ["%s %d %s %s %s" % (set_name, l, roles[l]["gate"], roles[l]["up"], roles[l]["down"]) for l in sorted(roles)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", help="Hugging Face repo id (sets are its top-level folders)")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--local", help="a local set's folder instead of --repo")
    ap.add_argument("sets", nargs="*")
    a = ap.parse_args()
    if a.local:
        name = a.sets[0] if a.sets else Path(a.local).name
        print("\n".join(layer_lines(name, local_set(a.local))))
        return 0
    if not a.repo or not a.sets:
        ap.error("--repo and at least one set (or --local)")
    for s in a.sets:
        print("\n".join(layer_lines(s, remote_set(a.repo, s, a.revision))))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
