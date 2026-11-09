"""serve/terminal_view.py - the Abyss Terminal's View for the web Monitor (GET /metrics?view=1 -> "view").

The numbers the Terminal derives (tools/terminal_model.derive: the token's time split and the bottleneck verdict with
--bench's rules, the PCIe link against its best, the drafter's acceptance, the banners ...) are computed here by the
same code, from the server's own /metrics and /health and its engine log - so the browser shows them without
re-deriving anything.  On top of the View: `bottleneck` (the verdict, its fix, the token's time as stacked segments
and which Monitor card it points at) and `diagnose` (the Terminal's "what is slowing you down" lines).
Standard library only, like the Terminal's modules.
"""
from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
import terminal_draw as D  # noqa: E402
import terminal_local as T  # noqa: E402
import terminal_model as M  # noqa: E402

# the View's keys the page does not need again (GET /metrics has them: requests, totals, the series)
DROP = ("requests", "totals", "_keyed", "url", "error", "age", "remote")
# the verdict -> the Monitor card it points at
TARGET = {"ssd": "disk", "cpu": "cpu", "pcie": "pcie", "gpu": "gpu"}


def token_split(tt: dict | None) -> dict | None:
    """A token's time as the Terminal's TOKEN TIME bar splits it: the SSD waits (the GPU stops for them), the waits
    for the RAM tier's experts (the misses that are not SSD waits: the lane's answer or a PCIe copy), and the rest
    (the GPU's own work).  The lane runs alongside the token - its own line (`parallel`), never added in."""
    if not tt or not tt.get("ms"):
        return None
    ms = float(tt["ms"])
    disk = max(0.0, min(float(tt.get("disk_ms") or 0.0), ms))
    miss = tt.get("miss_ms")
    ram = max(0.0, min(float(miss) - disk, ms - disk)) if miss is not None else 0.0
    gpu = max(0.0, ms - disk - ram)
    segs = [{"key": "ssd", "label": "SSD wait", "ms": disk}, {"key": "ram", "label": "RAM-tier wait", "ms": ram},
            {"key": "gpu", "label": "GPU", "ms": gpu}]
    for s in segs:
        s["share"] = s["ms"] / ms
    segs = [s for s in segs if s["ms"] > 0]
    lane = float(tt.get("lane_ms") or 0.0)
    return {"ms": ms, "segments": segs, "biggest": max(segs, key=lambda s: s["ms"])["key"] if segs else None,
            "parallel": {"key": "lane", "label": tt.get("lane_name") or "CPU lane", "ms": lane, "share": lane / ms}}


def bottleneck(v: dict) -> dict | None:
    """The verdict (bench_rules, through derive's token_time), its fix, how much of a token it takes, the card it
    points at and a tone: live and over half the token -> "error", live -> "warn", from the last answer -> "info"."""
    tt = v.get("tt")
    if not tt or not tt.get("verdict"):
        return None
    split = token_split(tt)
    k = tt["verdict"]
    ms = tt["ms"] or 1e-9
    share = {"ssd": tt["disk_ms"] / ms, "cpu": tt["lane_ms"] / ms}.get(k)
    if share is None and split:
        seg = {s["key"]: s for s in split["segments"]}
        share = (seg.get("ram") or {}).get("share") if k == "pcie" else (seg.get("gpu") or {}).get("share")
    label = D.VERDICT.get(k, k.upper())
    if k == "cpu" and tt.get("lane_name"):
        label = tt["lane_name"].upper()
    live = tt["source"] == "live"
    return {"verdict": k, "label": label, "what": tt.get("what") or "", "moves": tt.get("moves") or "",
            "evidence": list(tt.get("evidence") or []), "source": tt["source"], "source_text": D.tt_source(tt),
            "share": share, "target": "igpu" if k == "cpu" and tt.get("lane_name") == "iGPU lane" else TARGET.get(k),
            "tone": ("error" if (share or 0) >= 0.5 else "warn") if live else "info", "split": split}


def view_json(metrics: dict, health: dict, local: dict | None = None, log=None, seen: dict | None = None,
              now: float | None = None) -> dict:
    """derive()'s View of this server's own readings (a Link as the Terminal's poll would leave it), made JSON-ready."""
    link = types.SimpleNamespace(metrics=metrics, health=health, state="live", error="", base=None, key=None,
                                 last_ok=time.time() if now is None else now)
    v = M.derive(link, local or {}, log, now=now, seen=seen)
    out = {k: x for k, x in v.items() if k not in DROP}
    for k in ("decode", "prompt", "cpu", "ram", "disk"):
        if isinstance(out.get(k), dict):
            out[k] = {a: b for a, b in out[k].items() if a != "spark"}
    if isinstance(out.get("gpu"), dict):
        out["gpu"] = {a: b for a, b in out["gpu"].items() if a != "spark"}
    if isinstance(out.get("tiers"), dict):
        out["tiers"] = {a: b for a, b in out["tiers"].items() if a != "history"}
    for c in out.get("cards") or []:
        c["short"] = M.short_gpu(c.get("name") or "")
    out["banners"] = [list(b) for b in v.get("banners") or []]
    out["badges"] = [list(b) for b in v.get("badges") or []]
    out["bottleneck"] = bottleneck(v)
    out["diagnose"] = [list(x) for x in D.diagnose(v)] if v.get("has_metrics") else []
    out["state_key"] = D.state_key(v)
    return out


class TerminalView:
    """One per server: the engine log followed (its "glm stat" lines, the lane, the cards of a split, warnings), the
    run config's facts (badges) and the best PCIe link seen while working - kept between polls, as the Terminal keeps
    them between frames."""

    def __init__(self, svc):
        self.svc = svc
        self.lock = threading.Lock()
        self.seen: dict = {}
        self.log = None
        self._log_path = None
        self._cfg = None
        self._cfg_path = None

    def local(self) -> dict:
        path = getattr(self.svc, "config_path", None)
        if path != self._cfg_path:
            self._cfg_path = path
            try:
                info = T.config_info(Path(path)) if path else None
            except (OSError, ValueError, TypeError):
                info = None
            # (only what the header and badges read: no key, no paths)
            self._cfg = {k: info.get(k) for k in ("file", "stem", "display", "quant", "model_name", "ctx", "backend",
                                                   "env", "gpus")} if info else {}
        return {"remote": False, "is_local": True, "pid_alive": True, "bench": T.bench_summary(ROOT),
                "version": T.version(ROOT), "config": self._cfg or {}}

    def engine_log(self):
        path = getattr(self.svc, "log_path", None)
        if path != self._log_path:
            self._log_path = path
            self.log = M.EngineLog(path, getattr(self.svc, "gpu_indices", None) or [0]) if path else None
        if self.log is not None:
            self.log.read()
        return self.log

    def view(self, metrics: dict, health: dict) -> dict:
        with self.lock:
            return view_json(metrics, health, self.local(), self.engine_log(), self.seen)
