"""What limits this PC: a token's time split into its parts and, per part, what change would move it and by about how
much - for ./abyss.sh --check (abyss-check.txt) and --bench (abyss-bench.txt).  Measured numbers only: the engine's
own counters ("glm stat", "glm gap prof"), its start lines (the expert pool, the RAM tier, the CPU lane's and the
PCIe link's cost of an expert), the PCIe link it ran on and the SSD's read rate.  Every "what if" is an estimate and
is printed as one.  Standard library only; nothing runs on import.

The VRAM estimate replays the engine's own routes (STRATA_GLM_ROUTE_LOG, written while the check runs) through a
least-recently-used cache of each size.  The engine's own placement adapts more slowly than an ideal LRU: on the
2026-11-05 cache simulation of Qwen3.8 (12 / 16 / 24 GB) its misses fell by about the square root of the LRU's ratio,
so the estimate takes that: misses(new) = misses(now) * sqrt(lru(new) / lru(now)), and says the hit rate first.  The
speed: the time a miss costs is the measured one (the CPU lane's, PCIe's and the SSD's time a token over the misses a
token; decode ms = base + k x misses held at 32 GB).  It failed at 16 GB: when the CPU lane takes nearly every miss
(alongside the GPU), a hit saves only lane time and the promotions that bring it cost link time - a higher hit rate
decoded slower there (97.9 -> 93.4 tok/s), so a first draft gave a range from no gain.  Checked before v1.3.112
against more VRAM measured (Qwen3.8 16G64G without the drafter, 16 -> 32 GB: hit 82% estimated / 82% measured, speed
+21% estimated / +26% measured), the misses' measured cost is a fair estimate, a little low: one number again.
"""
from __future__ import annotations

import collections
import math
import re
from pathlib import Path

POOL_LINE = re.compile(r"layers \[(\d+),(\d+)\) expert pool ([\d.]+) GB, (\d+)-(\d+) slots/layer \((\d+) total\)")
RAM_LINE = re.compile(r"RAM tier ([\d.]+) GB pinned, (\d+) slots")
COST_LINE = re.compile(r"an expert ([\d.]+) ms on the CPU vs ([\d.]+) ms over PCIe")
NATIVE_HEAD = re.compile(r"n_expert (\d+), total (\d+)")
PCIE_LANE_GBS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.97, 5: 3.94, 6: 7.88}   # per lane and direction (abyss.py's too)
NVME_GEN4_GBS = 6.0       # a typical PCIe Gen4 NVMe SSD's sequential read
SMALL = 0.10              # a part under this share of a token: no change needed
LANE_MOST = 0.8           # the lane takes this share of the misses or more: the VRAM line says so
VRAM_STEPS = (8, 16)      # GB more VRAM the estimate is made for
ROUTE_LINES = 150_000     # the routes the VRAM estimate replays at most (the last ones: the warmest pool)


def engine_facts(log_lines, pack: Path | None = None) -> dict:
    """The engine's start lines: layers, expert pool (GiB, slots), RAM tier (GiB, slots), the lane's and PCIe's cost
    of an expert (ms); the model's experts (count, bytes, bytes of one) from the pack's native_experts.txt."""
    f = {"layers": 0, "pool_gb": 0.0, "slots": 0, "ram_gb": 0.0, "ram_slots": 0}
    for x in log_lines:
        m = POOL_LINE.search(x)
        if m:
            f["layers"] += int(m.group(2)) - int(m.group(1))
            f["pool_gb"] += float(m.group(3))
            f["slots"] += int(m.group(6))
        m = RAM_LINE.search(x)
        if m:
            f["ram_gb"] += float(m.group(1))
            f["ram_slots"] += int(m.group(2))
        m = COST_LINE.search(x)
        if m and "cpu_ms" not in f:
            f["cpu_ms"], f["pcie_ms"] = float(m.group(1)), float(m.group(2))
    native = Path(pack) / "native_experts.txt" if pack else None
    if native is not None and native.is_file():
        try:
            lines = native.read_text(encoding="utf-8", errors="replace").splitlines()
            m = NATIVE_HEAD.search(lines[0]) if lines else None
            rows = [r.split() for r in lines[1:] if r.strip() and not r.startswith("#")]
            if m and rows:
                f["n_expert"], f["experts_bytes"] = int(m.group(1)), int(m.group(2))
                f["model_layers"] = len(rows)
                f["expert_bytes"] = f["experts_bytes"] / (f["n_expert"] * len(rows))
        except (OSError, ValueError, IndexError):
            pass
    if "expert_bytes" not in f and f["slots"]:
        f["expert_bytes"] = f["pool_gb"] * 2**30 / f["slots"]
    return f


def lru_misses(routes, layers: int, cap: int) -> int:
    """Misses of a least-recently-used cache of `cap` experts in each layer over `routes` [(layer, [experts])]."""
    caches = [collections.OrderedDict() for _ in range(layers)]
    miss = 0
    for il, ids in routes:
        c = caches[il]
        for e in ids:
            if e in c:
                c.move_to_end(e)
            else:
                miss += 1
                c[e] = True
                if len(c) > cap:
                    c.popitem(last=False)
    return miss


def read_routes(paths, layers: int, limit: int = ROUTE_LINES) -> list:
    """The last `limit` routes of the engine's route logs ("layer e0 .. eK-1 fetch miss cpu" a line), the layers below
    `layers` only (not the drafter's block)."""
    out = collections.deque(maxlen=limit)
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    v = line.split()
                    if len(v) < 5:
                        continue
                    try:
                        il = int(v[0])
                        if 0 <= il < layers:
                            out.append((il, [int(e) for e in v[1:-3]]))
                    except ValueError:
                        continue
        except OSError:
            continue
    return list(out)


def vram_ratio(routes, layers: int, cap_now: int, cap_new: int, n_expert: int | None) -> float | None:
    """misses(new) / misses(now) for the engine's placement, from an LRU replay (see the module's note); None when the
    routes are too few to say."""
    if len(routes) < 20 * max(1, layers) or cap_now <= 0:
        return None
    if n_expert and cap_new >= n_expert:
        return 0.0
    now = lru_misses(routes, layers, cap_now)
    if now <= 0:
        return None
    new = lru_misses(routes, layers, cap_new)
    return math.sqrt(new / now)


def link_now(chain) -> tuple:
    """(generation, width, the GPU's max generation, max width) of a pcie_chain: the narrowest link on the way."""
    if not chain:
        return None
    return min(c[0] for c in chain), min(c[1] for c in chain), chain[0][2], chain[0][3]


def split_token(rec: dict, facts: dict) -> dict:
    """A token's ms split into parts (estimates: the parts overlap a little): SSD waits, the CPU lane, PCIe fetches,
    host gaps (with the GPU timeline's profile) and the GPU's own compute (the rest)."""
    t = rec["ms"]
    parts = {"ssd": rec.get("disk_ms", 0.0), "lane": rec.get("lane_ms", 0.0),
             "pcie": rec.get("ram_fetch", 0.0) * facts.get("pcie_ms", 0.0),
             "gaps": rec.get("gpu_idle", 0.0)}
    side = sum(parts.values())
    if side > t > 0:                                  # (they overlap: scale them into the token)
        parts = {k: v * t / side for k, v in parts.items()}
        side = t
    parts["gpu"] = max(0.0, t - side)
    parts["gaps_measured"] = "gpu_idle" in rec
    return parts


def what_limits(rec: dict | None, facts: dict, chain=None, disk_gbs: float | None = None, routes=None,
                vram_gb: float | None = None, lane_name: str = "CPU lane", unified: bool = False) -> list:
    """The "What limits this PC" lines from a decode record without the drafter (bench_rules.bench_records), the
    engine's facts (engine_facts), the GPU's PCIe chain (abyss.pcie_chain / nvidia_links), the SSD's read GB/s, the
    engine's routes (read_routes) and the GPU's VRAM (GB).  Every estimate says it is one.  unified: an integrated
    GPU (DGX Spark / GB10) - no PCIe link, its "VRAM" is the expert pool's share of the one memory."""
    if not rec or rec.get("ms", 0) <= 0:
        return ["(no decode statistics from the engine: nothing to split)"]
    t = rec["ms"]
    tps = 1000.0 / t
    p = split_token(rec, facts)
    pct = lambda v: f"{v / t:.0%}"
    out = [f"A token without the drafter: {t:.1f} ms ({tps:.1f} tokens/s). Where it goes (estimates; the parts overlap "
           "a little):",
           f"- GPU compute: {p['gpu']:.1f} ms ({pct(p['gpu'])})" + ("" if p["gaps_measured"] else
                                                                      " (host gaps included: not profiled in this run)"),
           f"- {lane_name}: {p['lane']:.1f} ms ({pct(p['lane'])}), {rec.get('lane', 0):.0f} experts a token",
           f"- {'RAM-tier copies' if unified else 'PCIe fetches'}: {p['pcie']:.1f} ms ({pct(p['pcie'])}), {rec.get('ram_fetch', 0):.1f} experts a token",
           f"- SSD waits: {p['ssd']:.1f} ms ({pct(p['ssd'])}), {rec.get('disk_reads', 0):.1f} reads a token"]
    if p["gaps_measured"]:
        out.append(f"- host gaps (the GPU idle between tokens): {p['gaps']:.1f} ms ({pct(p['gaps'])})")
    ideas = []                                        # (gain in tokens/s, line)
    misses = rec.get("lane", 0.0) + rec.get("ram_fetch", 0.0) + rec.get("disk_reads", 0.0)
    miss_ms = p["lane"] + p["pcie"] + p["ssd"]
    lane_pe = p["lane"] / rec["lane"] if rec.get("lane", 0) > 0 else facts.get("cpu_ms")
    lane_share = rec.get("lane", 0.0) / misses if misses > 0 else 0.0   # the misses the lane computes

    def est(new_ms):
        new_ms = max(new_ms, 0.2 * t)
        return 1000.0 / new_ms, 1000.0 / new_ms - tps

    # PCIe: the link it ran on vs an x16 slot of the GPU's own generation
    ln = None if unified else link_now(chain)
    if unified:
        ideas.append((0.0, "PCIe: none - the GPU shares the system memory (unified memory): no link to change"))
    elif ln and facts.get("pcie_ms") and facts.get("expert_bytes"):
        g, w, gm, wm = ln
        rate = facts["expert_bytes"] / (facts["pcie_ms"] / 1000.0) / 1e9
        full = PCIE_LANE_GBS.get(gm, 0) * max(wm, 16)
        cur = PCIE_LANE_GBS.get(g, 0) * w
        head = f"PCIe: the GPU's link ran at Gen{g} x{w} (an expert copy measured {rate:.1f} GB/s)"
        if cur <= 0 or full <= cur * 1.05:
            ideas.append((0.0, f"{head} - already the GPU's full link: no change needed"))
        else:
            k = cur / full
            pcie_new = facts["pcie_ms"] * k
            lane_cost = min(lane_pe or pcie_new, pcie_new)   # (the engine sends a miss the cheaper way)
            new_ms = t - p["pcie"] - p["lane"] + rec.get("ram_fetch", 0) * pcie_new + rec.get("lane", 0) * lane_cost
            if new_ms >= t * 0.98:
                ideas.append((0.0, f"{head} - an x16 Gen{gm} slot (~{full:.0f} GB/s) would change little here: the "
                                   f"{lane_name} already does the missed experts faster than such a copy"))
            else:
                v, d = est(new_ms)
                ideas.append((d, f"{head} - estimate: an x16 Gen{gm} slot (~{full:.0f} GB/s nominal) would cut an "
                                 f"expert copy to ~{pcie_new:.2f} ms, ~{v:.0f} tokens/s ({d / tps:+.0%})"))
    elif p["pcie"] < SMALL * t:
        ideas.append((0.0, "PCIe: under a tenth of a token - no change needed"))
    # VRAM: more slots a layer for the experts, from the engine's own routes
    hit = rec.get("vram_hit")
    if hit is not None and (hit >= 97.0 or miss_ms < SMALL * t):
        ideas.append((0.0, f"VRAM: holds {hit:.0f}% of the experts a token uses - no change needed"))
    elif hit is not None and facts.get("slots") and facts.get("layers") and facts.get("expert_bytes") and routes:
        L = facts["layers"]
        cap = facts["slots"] // L
        done = False
        for gb in VRAM_STEPS:
            add = int(gb * 1e9 / facts["expert_bytes"] / L)
            r = vram_ratio(routes, L, cap, cap + add, facts.get("n_expert"))
            if r is None:
                break
            new_miss = misses * r
            new_ms = t - (misses - new_miss) * (miss_ms / misses if misses else 0.0)
            v, d = est(new_ms)
            more = (" for the expert pool (unified memory: STRATA_GLM_VRAM_GB, the share of the memory it gets)"
                    if unified else f" ({vram_gb:.0f} -> {vram_gb + gb:.0f} GB class)" if vram_gb else "")
            head = (f"VRAM: holds {hit:.0f}% of the experts a token uses - estimate: with {gb} GB more{more} "
                    f"~{100 - (100 - hit) * r:.0f}%")
            if lane_share >= LANE_MOST:
                # the CPU lane takes nearly every miss, in parallel with the GPU.  Checked against measurements
                # (v1.3.112, Qwen3.8 16G64G without the drafter, 16 -> 32 GB): the hit rate estimated 82%, measured
                # 82%; the speed estimated +21%, measured +26% (71.4 -> 90.0 tok/s) - so the misses' measured cost is
                # a fair, slightly low estimate (the earlier "from no gain" range was too cautious)
                ideas.append((d, f"{head}; speed ~{v:.0f} tokens/s ({d / tps:+.0%}) - the {lane_name} takes "
                                 f"{lane_share:.0%} of the misses; measured on one model, the real gain was a little "
                                 "larger than this estimate"))
            else:
                ideas.append((d, f"{head}; speed ~{v:.0f} tokens/s ({d / tps:+.0%})"))
            done = True
        if not done:
            ideas.append((0.0, f"VRAM: holds {hit:.0f}% of the experts a token uses (too few routes logged to "
                               "estimate more)"))
    elif hit is not None:
        ideas.append((0.0, f"VRAM: holds {hit:.0f}% of the experts a token uses (not estimated: no route log)"))
    # RAM: does the RAM tier hold every expert the GPU does not?
    reads = rec.get("disk_reads", 0.0)
    if reads < 0.1:
        ideas.append((0.0, "RAM: the RAM tier holds every expert the GPU does not (no SSD reads while writing) - "
                           "no change needed"))
    else:
        need = ""
        if facts.get("experts_bytes"):
            gb = facts["experts_bytes"] / 2**30 - facts.get("pool_gb", 0) - facts.get("ram_gb", 0)
            if gb > 0:
                need = f" about {math.ceil(gb / 8) * 8:.0f} GB more RAM for the tier"
        new_ms = t - p["ssd"] + reads * (lane_pe or 0.0)
        v, d = est(new_ms)
        ideas.append((d, f"RAM: {reads:.1f} experts a token come from the SSD - estimate: with{need or ' more RAM'} "
                         f"(the whole model in GPU + RAM) the SSD waits go, ~{v:.0f} tokens/s ({d / tps:+.0%})"))
    # SSD: its read rate vs a typical NVMe Gen4 drive
    if p["ssd"] < SMALL * t:
        ideas.append((0.0, f"SSD: {'read ' + format(disk_gbs, '.1f') + ' GB/s; ' if disk_gbs else ''}waits under "
                           "a tenth of a token - no change needed"))
    elif disk_gbs and disk_gbs < NVME_GEN4_GBS * 0.85:
        new_ms = t - p["ssd"] * (1 - disk_gbs / NVME_GEN4_GBS)
        v, d = est(new_ms)
        ideas.append((d, f"SSD: read {disk_gbs:.1f} GB/s - estimate: a typical NVMe Gen4 SSD (~{NVME_GEN4_GBS:.0f} "
                         f"GB/s) ~{v:.0f} tokens/s ({d / tps:+.0%})"))
    else:
        ideas.append((0.0, f"SSD: read {disk_gbs:.1f} GB/s, as fast as a typical NVMe Gen4 SSD: more RAM (above) is "
                           "what removes its waits" if disk_gbs else "SSD: read rate not measured here"))
    # the GPU itself and the host
    if p["gpu"] >= 0.5 * t:
        ideas.append((0.0, f"GPU compute: {pct(p['gpu'])} of a token - the GPU itself is the main part; a faster GPU "
                           "class scales it (not estimated)"))
    if p["gaps_measured"] and p["gaps"] >= SMALL * t:
        ideas.append((0.0, f"host gaps: {pct(p['gaps'])} of a token - the CPU's single-thread speed and power mode "
                           "set it (not estimated)"))
    ideas.sort(key=lambda x: -x[0])
    out += ["", "What would change it (largest estimated gain first; tokens/s without the drafter - with it a VRAM "
                "gain can be larger: its verify windows give the lane more work):"]
    out += [f"{i}. {line}" for i, (_, line) in enumerate(ideas, 1)]
    return out
