#!/usr/bin/env python3
"""tools/bench_ref.py - the quality check's reference for one model: `./abyss.sh --bench` scores the same tokens
on another PC and compares (KL) its log-probs with these.

    .venv/bin/python tools/bench_ref.py <config.json> [--tokens 512] [--k 32] [--check] [--variant <tag>] [--beta]

Runs the config's engine on a known-good PC in the score mode (STRATA_GLM_SCORE, teacher-forced through the one-token
decode path), keeps each position's top-k log-probs and writes bench/ref/<model>.json (the token ids, the GGUF set's
size, the score) and bench/ref/<model>.topk (the engine's STRATA_GLM_SCORE_TOPK format: int32 k, then per scored
position k int32 ids, best first, and their k float32 log-probs).  <model> is the GGUF set's name without its shard
suffix (abyss.model_key), so a copy with other files - another quant - finds no reference.  --variant <tag> writes
bench/ref/<model>.<tag>.json / .topk instead, for a second copy under the same name (an overlay of the stock files):
the check picks the reference whose GGUF size matches.  --check scores once more
against the new reference (the KL should be ~0: the run-to-run floor).  --beta marks the model's support as beta:
--bench then asks for the report whatever the numbers.

The text is the start of README.md and README.ja.md, English then Japanese; its token ids are stored, so later edits
of the READMEs do not change a reference."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--variant", default="", help="write <model>.<variant>.json: another copy under the same name")
    ap.add_argument("--beta", action="store_true", help="mark the model's support as beta: --bench asks for the report")
    a = ap.parse_args()
    import numpy as np
    import abyss as A
    import serve.server as SV
    import strata_tokenizer as ST
    cfg = A.read_json(a.config)
    if cfg.get("backend") == "hip":
        A.select_build_backend("hip")
    args = cfg.get("args") or []
    pack = Path(args[args.index("--glm-pack") + 1]) if "--glm-pack" in args[:-1] else None
    name, size = A.model_key(pack)
    if not name:
        print("no GGUF set beside the config's pack", file=sys.stderr)
        return 1
    tok = ST.Tokenizer.from_dir(Path(cfg.get("tokenizer") or pack / "tokenizer"))
    en = (ROOT / "README.md").read_text(encoding="utf-8")
    ja = (ROOT / "README.ja.md").read_text(encoding="utf-8")
    ids = (tok.encode(en[:6000])[:a.tokens // 2] + tok.encode(ja[:6000]))[:a.tokens]
    out = ROOT / "bench" / "ref"
    out.mkdir(parents=True, exist_ok=True)
    cmd_args = list(SV.engine_args(cfg))
    if "--max-context" in cmd_args[:-1]:
        cmd_args[cmd_args.index("--max-context") + 1] = str(max(2048, len(ids) + 64))

    def score(extra: dict) -> str:
        env = {k: v for k, v in SV.child_env(cfg).items() if not k.startswith(A.BENCH_LOSSY)}
        env.update(STRATA_GLM_SCORE="0", **extra)
        p = subprocess.run([cfg["exe"]] + cmd_args + ["--tokens", ",".join(map(str, ids))], env=env,
                           cwd=cfg.get("cwd") or str(ROOT), capture_output=True, text=True)
        line = next((x for x in (p.stdout or "").splitlines() if x.startswith("SCORE")), "")
        if not line:
            sys.exit("no SCORE line: " + ((p.stderr or "").strip().splitlines() or ["?"])[-1])
        return line

    with tempfile.TemporaryDirectory() as tmp:
        dump = Path(tmp) / "dump.bin"
        line = score({"STRATA_GLM_SCORE_DUMP": str(dump)})
        print(line)
        n = int(re.search(r"SCORE (\d+) tokens", line).group(1))
        lp = np.fromfile(dump, dtype=np.float32)
        if lp.size % n:
            sys.exit(f"the dump holds {lp.size} floats, not a multiple of {n} positions")
        lp = lp.reshape(n, -1)
        top = np.argsort(-lp, axis=1)[:, :a.k].astype(np.int32)
        val = np.take_along_axis(lp, top, axis=1).astype(np.float32)
    stem = f"{name}.{a.variant}" if a.variant else name
    topk = out / f"{stem}.topk"
    with open(topk, "wb") as f:
        np.array([a.k], dtype=np.int32).tofile(f)
        for i in range(n):
            top[i].tofile(f)
            val[i].tofile(f)
    m = re.search(r"nll ([\d.]+) .*?top1 ([\d.]+)%", line)
    gpu = ", ".join(g["name"] for g in (A.hip_gpus(cfg) if cfg.get("backend") == "hip" else A.S.gpus())) or "?"
    dense = pack / "dense.bin"
    ref = {"model": name, "gguf_bytes": size, "dense_bytes": dense.stat().st_size if dense.exists() else 0, "skip": 0, "k": a.k, "topk": topk.name, "ids": ids,
           "nll": float(m.group(1)), "top1": float(m.group(2)),
           "made_with": f"Abyss v{(ROOT / 'VERSION').read_text().strip()}, {gpu}, {Path(a.config).name}"}
    if a.beta:
        ref["beta"] = True
    (out / f"{stem}.json").write_text(json.dumps(ref) + "\n", encoding="utf-8")
    print(f"written: {out / (stem + '.json')} and {topk.name} ({os.path.getsize(topk) / 1024:.0f} KB)")
    if a.check:
        print("check:", score({"STRATA_GLM_SCORE_TOPK": str(topk)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
