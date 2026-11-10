#!/usr/bin/env python3
"""spec_exact_check.py <config.json> [--mtp DIR] [--dflash GGUF] [--reps N] [--max-new N] [--prompts a,b] [--no-det]

Greedy decoding with drafts must give the token path's tokens: a draft only decides how many tokens a verify window
checks at once, never which token comes next.  One engine (--serve) a mode - "none" (no drafter: n-gram windows only,
if any), "mtp" (STRATA_Q4X_MTP=<dir>, the model's draft layer), "dflash" (STRATA_Q4X_DFLASH=<gguf>, DFlash 2) - each
prompt N greedy answers with the mode's drafts, then one GEN spec=0 (token by token, no windows).  Every answer must
equal its engine's spec=0 answer, and the spec=0 answers must be the same in every engine.  STRATA_DETERMINISTIC=1 by
default (the CPU / iGPU lanes and tier moves follow wall-clock timing and the CPU lane's experts round apart from the
GPU's, so without it two runs can part at a near-tie; --no-det checks the normal mode anyway).  Exit 0: all equal.

Qwen3.8-Flash-Next (2026-11-10): the verify window's dense GEMVs were gf::mvn - one warp a row where the token path's
mv() splits a small Q8_0 matrix four warps a row - so the windows parted from spec=0 at tokens 7 / 29; mvn_mv since."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402

SWE = '''Here is a Python module with a bug:

```python
import re
from dataclasses import dataclass

@dataclass
class Version:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, s: str) -> "Version":
        m = re.match(r"(\\d+)\\.(\\d+)\\.(\\d+)", s)
        if not m:
            raise ValueError(f"bad version: {s}")
        return cls(int(m.group(1)), int(m.group(2)), int(m.group(2)))

    def __lt__(self, other: "Version") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
```

Issue: `max([Version.parse("1.2.3"), Version.parse("1.2.10")])` gives 1.2.2. Find the bug, explain it in one sentence,
then output the corrected class.'''

PROMPTS = {
    "code": "Write a Python class implementing an LRU cache with get and put in O(1), with type hints, docstrings and a "
            "few unit tests using unittest.",
    "list": "List the 47 prefectures of Japan in order from north to south as a numbered markdown list, each line in "
            "the form `N. Name (Capital city)`.",
    "ja": "日本の四季の移り変わりと、それぞれの季節の行事や食べ物について、やさしい日本語で 600 字ほどの文章を書いてください。",
    "swe": SWE,
}


def first_diff(x, y):
    j = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), None)
    return min(len(x), len(y)) if j is None and len(x) != len(y) else j


def run_engine(exe, args, cwd, env, ids, reps, max_new, log):
    p = subprocess.Popen([exe, "--serve", *args], cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=open(log, "w"), text=True, bufsize=1)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit(f"the engine ended before READY (see {log})")

    def gen(i, k=""):
        p.stdin.write(f"GEN {max_new} temperature=0 {k}{','.join(map(str, i))}\n")
        p.stdin.flush()
        toks, done = [], []
        for line in p.stdout:
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE") or line.startswith("ERR"):
                done = line.split()
                break
        dms = float(done[4]) if done and done[0] == "DONE" and len(done) > 4 else 0.0
        return {"tokens": toks, "tps": round(len(toks) / dms * 1000, 2) if dms else 0.0, "err": done[:1] != ["DONE"]}

    out = {k: {"on": [], "ref": None} for k in ids}
    for _ in range(reps):
        for k, i in ids.items():
            out[k]["on"].append(gen(i))
    for k, i in ids.items():
        out[k]["ref"] = gen(i, "spec=0 ")
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=120)
    except subprocess.TimeoutExpired:
        p.kill()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--exe")
    ap.add_argument("--mtp", help="the draft layer's directory (STRATA_Q4X_MTP)")
    ap.add_argument("--dflash", help="the DFlash 2 drafter GGUF (STRATA_Q4X_DFLASH)")
    ap.add_argument("--no-none", action="store_true", help="skip the no-drafter engine")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--prompts", default=",".join(PROMPTS), help="names from the built-in set, or ids:<file>")
    ap.add_argument("--no-det", action="store_true", help="the normal (timing-driven) mode, no STRATA_DETERMINISTIC")
    ap.add_argument("--log-dir", default="/tmp")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tdir = Path(cfg["tokenizer"])
    tok = ST.Tokenizer.from_dir(tdir)
    tpl = ChatTemplate(tdir / "chat_template.jinja")
    ids = {}
    for k in a.prompts.split(","):
        if k.startswith("ids:"):
            ids[Path(k[4:]).stem] = [int(x) for x in Path(k[4:]).read_text().replace(",", " ").split()]
        else:
            ids[k] = tok.encode(tpl.render([{"role": "user", "content": PROMPTS[k]}], enable_thinking=False),
                                parse_special=True)
    base = dict(os.environ)
    base.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    for k in ("STRATA_Q4X_MTP", "STRATA_Q4X_DFLASH", "STRATA_GLM_NO_MTP"):
        base.pop(k, None)
    if cfg.get("lib_dirs"):
        base["LD_LIBRARY_PATH"] = ":".join(cfg["lib_dirs"] + [base.get("LD_LIBRARY_PATH", "")])
    base.setdefault("CUDA_VISIBLE_DEVICES", ",".join(map(str, cfg.get("gpu", [0]))))
    if not a.no_det:
        base["STRATA_DETERMINISTIC"] = "1"
    modes = [] if a.no_none else [("none", {})]
    if a.mtp:
        modes.append(("mtp", {"STRATA_Q4X_MTP": a.mtp}))
    if a.dflash:
        modes.append(("dflash", {"STRATA_Q4X_DFLASH": a.dflash}))
    res = {}
    for name, extra in modes:
        log = str(Path(a.log_dir) / f"spec-exact-{name}.engine.log")
        res[name] = run_engine(a.exe or cfg["exe"], cfg["args"], cfg.get("cwd"), dict(base, **extra), ids, a.reps,
                               a.max_new, log)
    bad = 0
    ref0 = {k: res[modes[0][0]][k]["ref"]["tokens"] for k in ids}
    for name, _ in modes:
        for k in ids:
            r = res[name][k]
            ref = r["ref"]["tokens"]
            notes = []
            for n, o in enumerate(r["on"]):
                j = first_diff(o["tokens"], ref)
                if o["err"] or j is not None:
                    bad += 1
                    notes.append(f"rep {n}: {'ERR' if o['err'] else f'DIFF at {j}'}")
            j = first_diff(ref, ref0[k])
            if j is not None:
                bad += 1
                notes.append(f"spec=0 parts from {modes[0][0]}'s at {j}")
            tps = sorted(o["tps"] for o in r["on"])
            print(f"{name:6s} {k[:12]:12s} n={len(ref):4d}  {tps[len(tps) // 2]:7.2f} tok/s (spec=0 {r['ref']['tps']:.2f})"
                  f"  {'same' if not notes else '; '.join(notes)}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res))
    print("all answers equal the token path's" if bad == 0 else f"{bad} answers differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
