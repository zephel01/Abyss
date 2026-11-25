#!/usr/bin/env python3
"""tools/memt_flash_fetch.py - the parts of MemT-1.0-Flash (MemTensor/MemT-1.0-Flash on Hugging Face, MIT) that
tools/memt_flash.py needs, without the 576 GB checkpoint: the memory blocks (memt-*.safetensors, 7.2 GB) whole, and only
the NON-EXPERT bytes of the 43 model shards (14.7 GB) through HTTP range requests.  Standard library only.

usage: memt_flash_fetch.py <out dir> [--repo=MemTensor/MemT-1.0-Flash] [--revision=main] [--plan]

<out dir> then holds config.json, model.safetensors.index.json, memt-0000N-of-0000M.safetensors, hdr/memt_<shard>.len
and .hdr.json (each model shard's safetensors header) and runs/<shard>_<start>.bin (a shard's bytes [start, start +
size): adjacent non-expert tensors - every tensor whose name has no ".mlp.experts." - in one read).  It is both
memt_flash.py's <memt folder> and its --backbone folder:

  memt_flash.py <out dir> <V4 Flash 0731 GGUF shard 1> <overlay dir> --outtype=q8_0 --backbone=<out dir> --gpuq=... ...

A run again resumes: complete files are kept, a part file is fetched again.  --plan: print the reads (shard, first
byte, last byte, bytes, tensors) and their total after the headers, fetch nothing else.  The token: HF_TOKEN, else
~/.cache/huggingface/token (none: anonymous).  HF_ENDPOINT moves the host (a mirror).
"""
from __future__ import annotations

import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 8 << 20


def token() -> str | None:
    t = os.environ.get("HF_TOKEN")
    if t:
        return t.strip()
    p = Path.home() / ".cache" / "huggingface" / "token"
    try:
        return p.read_text().strip() or None
    except OSError:
        return None


class Repo:
    def __init__(self, repo: str, rev: str):
        self.base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/") + f"/{repo}/resolve/{rev}/"
        self.tok = token()
        self.bytes = 0

    def _open(self, name: str, a: int | None = None, b: int | None = None):
        req = urllib.request.Request(self.base + name)
        if self.tok:
            req.add_header("Authorization", "Bearer " + self.tok)
        if a is not None:
            req.add_header("Range", f"bytes={a}-{b}")
        r = urllib.request.urlopen(req, timeout=120)
        if a is not None and r.status != 206:
            r.close()
            raise OSError(f"{name}: the server ignored the range (HTTP {r.status})")
        return r

    def get(self, name: str, out: Path, a: int | None = None, b: int | None = None, size: int | None = None):
        """name (bytes a..b inclusive, or all) -> out, through out.part; size: the bytes it must have."""
        if out.exists() and (size is None or out.stat().st_size == size):
            return
        part = out.with_name(out.name + ".part")
        for attempt in range(1, 6):
            try:
                with self._open(name, a, b) as r, open(part, "wb") as f:
                    if size is None and r.headers.get("Content-Length", "").isdigit():
                        size = int(r.headers["Content-Length"])
                    while True:
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        f.write(buf)
                        self.bytes += len(buf)
                n = part.stat().st_size
                if size is not None and n != size:
                    raise OSError(f"{name}: {n} bytes, {size} wanted")
                part.replace(out)
                return
            except (OSError, urllib.error.URLError) as e:
                print(f"  {name}: {e} (attempt {attempt} of 5)", file=sys.stderr, flush=True)
                time.sleep(min(60, 5 * attempt))
        raise SystemExit(f"{name}: could not fetch it")


def plan(hdr: dict, base: int) -> list[tuple[int, int, int]]:
    """A shard's header -> its non-expert runs (first byte, end byte exclusive, tensors), adjacent tensors merged."""
    ts = sorted((v["data_offsets"][0] + base, v["data_offsets"][1] + base, k) for k, v in hdr.items() if k != "__metadata__")
    runs: list[list[int]] = []
    for a, b, k in ts:
        if ".mlp.experts." in k:
            continue
        if runs and runs[-1][1] == a:
            runs[-1][1] = b
            runs[-1][2] += 1
        else:
            runs.append([a, b, 1])
    return [(a, b, n) for a, b, n in runs]


def main(argv: list[str]) -> int:
    pos = [a for a in argv if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in argv if a.startswith("--"))
    if len(pos) != 1 or set(opt) - {"repo", "revision", "plan"}:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(pos[0])
    (out / "hdr").mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    R = Repo(opt.get("repo", "MemTensor/MemT-1.0-Flash"), opt.get("revision", "main"))
    t0 = time.time()
    for f in ("config.json", "model.safetensors.index.json"):
        R.get(f, out / f)
    wm = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted(set(wm.values()))
    memt = [f for f in files if f.startswith("memt-")]
    model = [f for f in files if not f.startswith("memt-")]
    if not memt or not model:
        raise SystemExit("model.safetensors.index.json: no memt-* or no model shards - not MemT-1.0-Flash?")
    reads = []
    for f in model:
        s = f.split("-")[1]   # (model-00001-of-00043.safetensors -> 00001)
        ln, hj = out / "hdr" / f"memt_{s}.len", out / "hdr" / f"memt_{s}.hdr.json"
        R.get(f, ln, 0, 7, 8)
        n = struct.unpack("<Q", ln.read_bytes())[0]
        R.get(f, hj, 8, 8 + n - 1, n)
        for a, b, k in plan(json.loads(hj.read_text()), 8 + n):
            reads.append((f, s, a, b, k))
    total = sum(b - a for _, _, a, b, _ in reads)
    if "plan" in opt:
        for f, s, a, b, k in reads:
            print(f"{s}\t{a}\t{b - 1}\t{b - a}\t{k}")
        print(f"# {len(reads)} reads, {total} bytes ({total / 1e9:.2f} GB) of {len(model)} shards", file=sys.stderr)
        return 0
    print(f"the memory blocks: {', '.join(memt)}", flush=True)
    for f in memt:
        R.get(f, out / f)
    print(f"the non-expert backbone: {len(reads)} range reads, {total / 1e9:.2f} GB", flush=True)
    for i, (f, s, a, b, k) in enumerate(reads, 1):
        R.get(f, out / "runs" / f"{s}_{a}.bin", a, b - 1, b - a)
        print(f"  {i}/{len(reads)} {f} [{a}, {b}) {k} tensors", flush=True)
    print(f"done: {R.bytes / 1e9:.2f} GB fetched in {time.time() - t0:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
