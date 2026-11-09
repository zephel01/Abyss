"""How Abyss Terminal looks: every view drawn as lines of (text, style) pieces from a View (terminal_model.derive) -
no Textual, no Rich - so the screen (abyss_terminal.py, through Rich) and --plain (ANSI, standard library only) draw
the same thing, and the tests can measure it at 80x24 and 40x12 without either.

The look: near-black, neon cyan for the frames and the numbers that matter, hot pink for the bottleneck and alerts,
acid green for good, amber for warnings; thin frames with bright corners; a two-row ABYSS logo; block-character
sparklines and bars; numbers right-aligned.  A style is a palette role and flags: "accent b" = cyan bold,
"alert r" = pink reversed (b bold, i italic, r reverse, u underline, d dim).
"""
from __future__ import annotations

import re
import time

from terminal_model import SWAP_GROWING, memory_tight, pretty_name

# ------------------------------------------------------------------------------------------------ the palettes
PALETTES = {
    "dark": {"bg": "#05070d", "ink": "#d7f9ff", "soft": "#9fc3cc", "muted": "#5f8794", "faint": "#22343d",
             "edge": "#1d6474", "accent": "#00f0ff", "accent2": "#00a8c0", "alert": "#ff2e88", "magenta": "#ff3df2",
             "good": "#a3ff12", "warn": "#ffb000", "info": "#4fb7ff", "violet": "#9d7bff", "panel": "#0a1018",
             "hilite": "#0f2a33"},
    "light": {"bg": "#f3f9fb", "ink": "#0b2530", "soft": "#2e5563", "muted": "#5d7f8a", "faint": "#bfd2d8",
              "edge": "#8fb3bd", "accent": "#007f99", "accent2": "#00687d", "alert": "#d4006a", "magenta": "#a3009a",
              "good": "#3d8a00", "warn": "#a86400", "info": "#0063b0", "violet": "#6046d0", "panel": "#e6f1f4",
              "hilite": "#d3e9ef"},
}
THEMES = ("dark", "light", "mono")
LAYOUTS = ("speed", "devices")         # the Overview: A big numbers + tiles, B every device in its own panel
VIEWS = (("1", "OVERVIEW", "OVR"), ("2", "SPEED", "SPD"), ("3", "MEMORY", "MEM"), ("4", "REQUESTS", "REQ"),
         ("5", "HARDWARE", "HW"), ("6", "LOGS", "LOG"), ("7", "CONNECT", "CON"), ("8", "MODELS", "MOD"))
BLOCKS = "▁▂▃▄▅▆▇█"
LOGO = ("▄▀█ █▄▄ █▄█ █▀ █▀", "█▀█ █▄█  █  ▄█ ▄█")
# a seven-segment-like digit font of half blocks, three rows (the hero numbers)
DIG = {"0": ("█▀█", "█ █", "▀▀▀"), "1": ("▀█ ", " █ ", "▀▀▀"), "2": ("▀▀█", "█▀▀", "▀▀▀"), "3": ("▀▀█", " ▀█", "▀▀▀"),
       "4": ("█ █", "▀▀█", "  ▀"), "5": ("█▀▀", "▀▀█", "▀▀▀"), "6": ("█▀▀", "█▀█", "▀▀▀"), "7": ("▀▀█", "  █", "  ▀"),
       "8": ("█▀█", "█▀█", "▀▀▀"), "9": ("█▀█", "▀▀█", "▀▀▀"), ".": (" ", " ", "▀"), "-": ("   ", "▀▀▀", "   "),
       " ": (" ", " ", " ")}
FINISH = {"stop": ("DONE", "good"), "length": ("MAX TOK", "soft"), "cancel": ("STOPPED", "warn"),
          "disconnect": ("CLOSED", "warn"), "error": ("ERROR", "alert")}
API = {"web": "chat", "openai": "openai", "anthropic": "anthr"}
ROLE = {"main": "MAIN", "split": "SPLIT", "helper": "HELPER", "igpu": "iGPU"}
VERDICT = {"ssd": "SSD", "cpu": "CPU LANE", "pcie": "PCIe", "gpu": "GPU"}
DASH = "--"


# ------------------------------------------------------------------------------------------------ lines
class Line(list):
    """A line: [(text, style)].  width = its cells (every character the views use is one cell wide)."""

    def add(self, text, style: str = "ink") -> "Line":
        if text:
            self.append((str(text), style))
        return self

    def cat(self, other) -> "Line":
        self.extend(other)
        return self

    @property
    def width(self) -> int:
        return sum(len(t) for t, _ in self)

    @property
    def plain(self) -> str:
        return "".join(t for t, _ in self)


def L(*pieces) -> Line:
    """Line("text", "style", "text", "style", ...) - pairs; a lone last text is "ink"."""
    out = Line()
    it = iter(pieces)
    for t in it:
        out.add(t, next(it, "ink"))
    return out


def fit(line: Line, w: int, fill_style: str = "ink") -> Line:
    """`line` cut or padded to exactly `w` cells (a cut ends in …)."""
    if w <= 0:
        return Line()
    if line.width <= w:
        out = Line(line)
        if line.width < w:
            out.add(" " * (w - line.width), fill_style)
        return out
    out, n = Line(), 0
    for t, s in line:
        if n + len(t) >= w:
            out.add(t[:w - n - 1] + "…", s)
            n = w
            break
        out.add(t, s)
        n += len(t)
    return fit(out, w) if n < w else out


def right(line: Line, w: int) -> Line:
    return fit(Line([(" " * max(0, w - line.width), "ink")]).cat(line), w)


def spread(left: Line, rt: Line, w: int) -> Line:
    """`left` and `rt` on one line of `w`, `rt` against the right edge (dropped when it does not fit)."""
    if left.width + rt.width + 1 > w:
        return fit(left, w)
    return Line(left).add(" " * (w - left.width - rt.width)).cat(rt)


# ------------------------------------------------------------------------------------------------ numbers
def f(x, d: int = 0, unit: str = "") -> str:
    if x is None:
        return DASH
    s = f"{x:,.{d}f}"
    return s + unit


def pct(x, d: int = 0) -> str:
    return DASH if x is None else f"{x * 100:.{d}f}%"


def gib(b, d: int = 1) -> str:
    return DASH if b is None else f"{b / 2**30:,.{d}f}"


def rate(mb) -> str:
    if mb is None:
        return DASH
    return f"{mb / 1024:.2f} GB/s" if mb >= 1000 else f"{mb:,.{1 if mb < 10 else 0}f} MB/s"


def toks(n) -> str:
    if n is None:
        return DASH
    n = float(n)
    return f"{n / 1024:.0f}K" if n >= 1024 and n % 1024 == 0 else f"{n / 1000:.1f}K" if n >= 1000 else f"{n:.0f}"


def dur(s) -> str:
    if s is None:
        return DASH
    s = int(s)
    return f"{s}s" if s < 60 else f"{s // 60}m {s % 60:02d}s" if s < 3600 else f"{s // 3600}h {s % 3600 // 60:02d}m"


def tone_temp(t, junction=False) -> str:
    if t is None:
        return "muted"
    hot, warm = (100, 90) if junction else (87, 80)
    return "alert b" if t >= hot else "warn" if t >= warm else "ink"


# ------------------------------------------------------------------------------------------------ pieces
def bar(frac, w: int, style: str = "accent", rest: str = "faint", ch: str = "█", empty: str = "░") -> Line:
    if w <= 0:
        return Line()
    if frac is None:
        return L(empty * w, rest)
    n = round(min(max(frac, 0.0), 1.0) * w)
    return L(ch * n, style, empty * (w - n), rest)


def spark(values, w: int, style: str = "accent", top=None) -> Line:
    """A sparkline, newest on the right; a value None draws a faint gap (missing, not 0)."""
    if w <= 0:
        return Line()
    v = list(values or [])
    if 2 <= len(v) < w // 2:                       # (a wide box: each sample over several columns)
        v = [v[int(i * len(v) / w)] for i in range(w)]
    if len(v) > w:
        step = len(v) / w
        v = [max((x for x in v[int(i * step):max(int(i * step) + 1, int((i + 1) * step))] if x is not None),
                 default=None) for i in range(w)]
    nums = [x for x in v if x is not None]
    hi = max([top or 0.0, *nums, 1e-9])
    out = L(" " * (w - len(v)), "faint")
    for x in v:
        if x is None:
            out.add("·", "faint")
        elif x <= 0:
            out.add("▁", "faint")
        else:
            out.add(BLOCKS[min(7, int(x / hi * 7 + 0.5))], style)
    return out


def chart(values, w: int, rows: int, style: str = "accent") -> list:
    """A bar chart `rows` tall (eighths of a row per block), newest on the right, its peak labelled; None = a gap."""
    v = list(values or [])
    lw = 7
    cw = max(1, w - lw)
    if 2 <= len(v) < cw:
        v = [v[int(i * len(v) / cw)] for i in range(cw)]
    v = v[-cw:]
    nums = [x for x in v if x is not None]
    hi = max(nums + [1e-9])
    lo = min(nums) * 0.85 if nums and min(nums) > 0.4 * hi else 0.0   # (a steady line: its range, not a block)
    span = max(hi - lo, 1e-9)
    out = []
    for r in range(rows):
        level_lo = (rows - 1 - r) * 8
        label = f"{hi:6.1f} " if r == 0 else f"{lo:6.1f} " if r == rows - 1 else " " * lw
        ln = L(label, "muted")
        ln.add(" " * (cw - len(v)))
        tone = style if r < (rows + 1) // 2 else "accent2"
        for x in v:
            if x is None:
                ln.add("·" if r == rows - 1 else " ", "faint")
                continue
            n = int(round((x - lo) / span * (rows - 1) * 8)) + 8 - level_lo   # (the bottom row always drawn)
            ln.add("█" if n >= 8 else BLOCKS[n - 1] if n > 0 else " ", tone)
        out.append(ln)
    return out


def big(text: str) -> list:
    """`text` (digits . -) in the three-row digit font: three strings."""
    rows = ["", "", ""]
    for i, c in enumerate(text):
        g = DIG.get(c, DIG[" "])
        for r in range(3):
            rows[r] += ("" if i == 0 else " ") + g[r]
    return rows


def bignum(x, d: int = 1) -> str:
    if x is None:
        return "--"
    return f"{x:.{d}f}" if x < 100 else f"{x:.0f}"


def box(title: str, body: list, w: int, tag: Line | str | None = None, tone: str = "edge",
        title_style: str = "accent b", corner: str = "accent", h: int | None = None) -> list:
    """A frame `w` wide around `body` (lines cut / padded to w - 4), the title set into its top edge (┌─ TITLE ──
    tag ─┐), bright corners; `h`: its whole height (body padded or cut)."""
    inner = max(0, w - 4)
    if isinstance(tag, str):
        tag = L(tag, "muted")
    top = L("┌", corner, "─", tone)
    if title:
        top.add(" ", tone).add(title[:max(0, w - 6)], title_style).add(" ", tone)
    room = w - 1 - top.width
    if tag is not None and tag.width and tag.width + 4 <= room:
        top.add("─" * (room - tag.width - 3), tone).add(" ", tone).cat(tag).add(" ", tone).add("─", tone)
    else:
        top.add("─" * max(0, room), tone)
    top.add("┐", corner)
    rows = list(body)
    if h is not None:
        rows = (rows + [Line()] * max(0, h - 2 - len(rows)))[:max(0, h - 2)]
    out = [fit(top, w)]
    for r in rows:
        out.append(L("│", tone, " ").cat(fit(r, inner)).add(" ").add("│", tone))
    out.append(L("└", corner, "─" * max(0, w - 2), tone, "┘", corner))
    return out


def hjoin(blocks: list, widths: list, gap: int = 1) -> list:
    """Blocks of lines side by side (each cut / padded to its width), as tall as the tallest."""
    hgt = max((len(b) for b in blocks), default=0)
    out = []
    for i in range(hgt):
        ln = Line()
        for k, (b, w) in enumerate(zip(blocks, widths)):
            if k:
                ln.add(" " * gap)
            ln.cat(fit(b[i] if i < len(b) else Line(), w))
        out.append(ln)
    return out


def split(w: int, parts: list, gap: int = 1) -> list:
    """Widths for len(parts) columns sharing `w` (parts: their weights), gaps included."""
    room = w - gap * (len(parts) - 1)
    tot = sum(parts)
    ws = [max(1, room * p // tot) for p in parts]
    ws[-1] += room - sum(ws)
    return ws


def kv(k: str, val: str, unit: str = "", style: str = "ink b", kw: int = 0) -> Line:
    return L(k.ljust(kw), "muted", val, style if val != DASH else "muted", unit, "muted")


def pill(text: str, style: str) -> Line:
    return L(f" {text} ", style + " r")


# ------------------------------------------------------------------------------------------------ the frame
STATE = {   # the state pill: (glyph, text, style)
    "connecting": ("◌", "CONNECTING", "accent2"), "loading": ("◐", "LOADING", "warn"),
    "idle": ("●", "IDLE", "accent2"), "reading": ("▶", "READING PROMPT", "info"),
    "generating": ("▶", "GENERATING", "good"), "queued": ("≡", "QUEUED", "warn"),
    "reloading": ("◐", "RELOADING", "warn"), "updating": ("◐", "UPDATING", "warn"),
    "error": ("✖", "ERROR", "alert"), "stale": ("◇", "STALE", "warn"),
    "disconnected": ("○", "DISCONNECTED", "alert"), "auth": ("⊘", "AUTH REQUIRED", "alert"),
    "not_abyss": ("?", "NOT ABYSS", "warn"), "stopped": ("■", "STOPPED", "muted"),
}


def state_key(v: dict) -> str:
    c = v["conn"]
    if c in ("live", "not_abyss") and v.get("state"):
        return v["state"] if v["state"] in STATE else "idle"
    return c if c in STATE else "connecting"


def state_pill(v: dict) -> Line:
    k = state_key(v)
    glyph, text, style = STATE[k]
    extra = ""
    if k == "loading" and v.get("loading") and v["loading"].get("frac") is not None:
        extra = f" {v['loading']['frac'] * 100:.0f}%"
    elif k == "reading" and v["prompt"].get("total"):
        extra = f" {toks(v['prompt'].get('read'))}/{toks(v['prompt']['total'])}"
    elif k == "generating" and v["decode"].get("generated") is not None:
        extra = f" {v['decode']['generated']}" + (f"/{v['decode']['max_tokens']}" if v["decode"].get("max_tokens")
                                                   else "")
    elif k == "stale" and v.get("age") is not None:
        extra = f" {v['age']:.0f}s"
    if v.get("queued") and k in ("reading", "generating"):
        extra += f" +{v['queued']}q"
    return L(f" {glyph} {text}{extra} ", style + " b r")


def logo(row: int, tick: int = 0) -> Line:
    """The ABYSS logo's row: cyan into magenta column by column, with one static glitch slice."""
    s = LOGO[row]
    out = Line()
    for i, ch in enumerate(s):
        style = "accent b" if i < 8 else "accent2 b" if i < 12 else "magenta b"
        if row == 1 and 4 <= i <= 6:
            style = "magenta b"                    # (the glitch: a slice of the B pushed into pink)
        out.add(ch, style)
    return out


SHORT_STATE = {"GENERATING": "GEN", "READING PROMPT": "READING", "DISCONNECTED": "OFFLINE", "AUTH REQUIRED": "AUTH",
               "CONNECTING": "CONN", "RELOADING": "RELOAD", "UPDATING": "UPDATE"}


def header(v: dict, w: int, tick: int = 0, anim: bool = True) -> list:
    """Two lines: the logo, the model with its quant and the badges, the state; the config, context, backend, server,
    versions and the connection.  Narrow, the name gives way first - never the quant, the badges or the state."""
    small = w < 60
    pill_ = state_pill(v)
    if w < 80:                                 # (narrow: the state's short name)
        k = state_key(v)
        glyph, text, style = STATE[k]
        pill_ = L(f" {glyph} {SHORT_STATE.get(text, text)} ", style + " b r")
    logo_w = 0 if small else len(LOGO[0]) + 2
    room = w - logo_w - pill_.width - 1
    quant = Line()
    if v.get("quant"):
        quant.add(" ").add(v["quant"], "accent b")
    marks = Line()
    for text, tone in v.get("badges") or []:
        marks.add(" ").cat(L(f"[{text}]", ("alert b" if tone == "alert" else "violet")))
    name_w = max(4, room - quant.width - (marks.width if v["name"] and len(v["name"]) + quant.width + marks.width
                                          <= room else 0))
    l1 = Line() if small else logo(0).add("  ")
    l1.cat(fit(L(v["name"], "ink b"), min(len(v["name"]), name_w))).cat(quant)
    l2 = Line() if small else logo(1).add("  ")
    if len(v["name"]) + quant.width + marks.width <= room:
        l1.cat(marks)
    elif marks.width:
        l2.cat(Line(marks[1:]) if marks and marks[0][0] == " " else marks).add(" ")   # (the badges: line 2, first)
    meta = [x for x in (v.get("config"), f"ctx {toks(v['ctx'])}" if v.get("ctx") else None, v.get("backend"),
                        (v.get("url") or "").replace("http://", "").replace("https://", "https "),
                        f"Abyss v{v['version']}" if v.get("version") else None) if x]
    l2.add(" · ".join(meta), "muted")
    live = v["conn"] == "live"
    dot = L("●", ("good b" if (tick % 2 == 0 or not anim) else "faint") if live else "faint")
    age = v.get("age")
    conn = Line().cat(dot).add(" LIVE " if live else f" {v['conn'].upper().replace('_', ' ')} ",
                               "good" if live else "muted")
    if age is not None:
        conn.add(f"{age:.1f}s" if age < 10 else dur(age), "muted")

    def pin(left, rt):                         # (the state is never dropped: the left side gives way)
        if rt.width + 8 > w:
            return fit(rt, w)
        return fit(left, w - rt.width - 1).add(" ").cat(rt)
    return [pin(l1, pill_), pin(l2, conn)]


def tabs(view: int, w: int) -> Line:
    long_ = sum(len(n) + 4 for _, n, _ in VIEWS) <= w
    short = sum(len(s) + 4 for _, _, s in VIEWS) <= w
    out = Line()
    if long_ or short:
        for i, (k, n, s) in enumerate(VIEWS):
            name = n if long_ else s
            out.cat(L(f" {k} {name} ", "accent b r") if i == view else L(f" {k}", "accent2", f" {name} ", "muted"))
            out.add(" " if i < len(VIEWS) - 1 else "")
        rest = w - out.width
        return out.add("━" * max(0, rest), "edge") if rest > 0 else fit(out, w)
    lo = max(0, min(view - 1, len(VIEWS) - 3))   # (very narrow: the chosen view and its neighbours)
    out.add("‹ " if lo > 0 else "  ", "muted")
    for i in range(lo, min(len(VIEWS), lo + 3)):
        k, n, s = VIEWS[i]
        out.cat(L(f" {k} {s} ", "accent b r") if i == view else L(f" {k} {s} ", "muted"))
    out.add(" ›" if lo + 3 < len(VIEWS) else "", "muted")
    return fit(out, w)


FOOT_KEYS = (("1-8", "views"), ("←→", ""), ("l", "layout"), ("r", "refresh"), ("space", "freeze"), ("t", "theme"),
             ("L", "temp log"),
             ("?", "help"), ("q", "quit"))


def footer(v: dict, w: int, frozen: bool = False, message: str = "", recording: bool = False,
           keys=FOOT_KEYS, layout: str | None = None) -> Line:
    left = Line()
    if layout:
        left.add(f"▣ {layout.upper()} ", "accent b r").add(" ")
    for k, what in keys:
        piece = L(k, "accent b", (" " + what) if what else "", "muted", "  ")
        if left.width + piece.width > w - 22:
            break
        left.cat(piece)
    rt = Line()
    if message:
        rt.add(message, "warn b")
        rt.add("  ")
    if recording:
        rt.add("● REC ", "alert b")
    if frozen:
        rt.add(" FROZEN ", "warn b r")
    if message and left.width + rt.width > w:
        left = Line()
    return spread(left, rt, w)


def banner_lines(v: dict, w: int) -> list:
    out = []
    for text, tone in v.get("banners") or []:
        style = "alert" if tone == "alert" else "warn"
        out.append(fit(L(" ▲ ", style + " b r", " " + text, style + " b"), w))
    return out


# ------------------------------------------------------------------------------------------------ overview
def speed_tile(v: dict, which: str, w: int, compact: bool = False, h: int | None = None) -> list:
    """The decode or the prompt hero: the number in big, what it is (live / last request), ms, the sparkline."""
    d = v[which]
    val = d.get("value")
    inner = w - 4
    if which == "decode":
        sub = L(d["label"] or "no answer yet", "good b" if d["label"] == "live" else "soft")
        if d.get("ms"):
            sub.add(f"  {d['ms']:.1f} ms/tok", "muted")
        extra = L("mean ", "muted", f(d.get("mean"), 1), "ink", "  last ", "muted", f(d.get("last"), 1), "ink")
        bench = (v.get("bench") or {}).get("decode")
        if bench:
            extra.add("  bench ", "muted").add(f"{bench:.1f}", "violet")
        style, sp = "accent", d.get("spark")
    else:
        sub = L(d["label"] or "no prompt yet", "info b" if d["label"] == "live" else "soft")
        if d.get("total") and v.get("state") == "reading":
            sub.add(f"  {toks(d.get('read'))} / {toks(d['total'])}", "ink")
            if d.get("reused"):
                sub.add(f"  reused {toks(d['reused'])}", "good")
            if d.get("disk_ms") is not None:
                sub.add(f"  SSD {d['disk_ms'] / 1000:.1f} s", "warn")
        extra = Line()
        if v.get("state") == "reading" and d.get("total"):
            frac = (d.get("read") or 0) / max(1, d["total"])
            extra = bar(frac, max(4, inner - 6), "info").add(f" {frac * 100:3.0f}%", "info")
        else:
            bench = (v.get("bench") or {}).get("prompt8k")
            extra.add("last ", "muted").add(f(d.get("last"), 0), "ink")
            if bench:
                extra.add("  bench 8k ", "muted").add(f"{bench:.0f}", "violet")
        style, sp = "info", d.get("spark")
    title = "DECODE" if which == "decode" else "PROMPT"
    if compact:
        line = L(f"{title:<7}", "muted", f"{bignum(val):>6}", f"{style} b", " tok/s ", "muted").cat(sub)
        return [line]
    digits = big(bignum(val))
    unit = [L("tok/s", "muted b"), sub, Line()]
    rows = []
    for r in range(3):
        rows.append(L(digits[r], (style + " b") if val is not None else "faint").add("  ").cat(unit[r]))
    rows.append(spark(sp, inner, style))
    rows.append(extra)
    tag = L("◆ live", "good b") if d["label"] == "live" else None
    return box(title, rows, w, tag=tag, h=h)


def tt_source(tt: dict) -> str:
    """What the token's time was measured on: the live window, this answer, the last answer's STAT, the log."""
    if tt["source"] == "live":
        return "live · last ~64 tokens" if tt.get("window") else "live · this answer"
    if tt["source"] == "stat":
        return "last answer · engine STAT"
    return f"last {tt['n']} answers · engine log"


def tt_bar(tt: dict, w: int) -> list:
    """The token's time as one stacked bar: SSD waits (the GPU stops for them) then the rest (the GPU, the PCIe
    copies inside it); the bottleneck's part in pink.  The lane runs alongside - its own line, never added in."""
    ms = tt["ms"] or 1e-9
    disk = min(tt["disk_ms"], ms)
    v = tt["verdict"]
    nd = round(w * disk / ms) if disk > 0 else 0
    if disk > 0 and nd == 0:
        nd = 1
    ng = w - nd
    ssd_style = "alert" if v == "ssd" else "warn"
    gpu_style = "alert" if v in ("gpu", "pcie") else "accent"
    def seg(n, label, style):
        if n <= 0:
            return Line()
        text = label if len(label) + 2 <= n else label.split()[0] if len(label.split()[0]) + 2 <= n else ""
        text = f" {text}".ljust(n) if text else " " * n
        return L(text, style + " b r")
    out = [seg(nd, f"SSD WAIT {disk:.1f} ms", ssd_style).cat(seg(ng, f"GPU {ms - disk:.1f} ms", gpu_style))]
    lane = tt["lane_ms"]
    nl = min(w, round(w * lane / ms)) if lane else 0
    out.append(L("▔" * nl, "alert" if v == "cpu" else "violet", " " * (w - nl), "faint"))
    return out


def tt_tile(v: dict, w: int, compact: bool = False) -> list:
    tt = v.get("tt")
    inner = w - 4
    if not tt:
        rows = [L("SSD wait ", "muted", DASH, "muted", "   lane ", "muted", DASH, "muted", "   PCIe ", "muted", DASH,
                  "muted", "   GPU ", "muted", DASH, "muted"),
                bar(None, inner),
                L(v.get("tt_why") or "", "faint i")]
        tiers = v["tiers"]
        if tiers.get("vram_hit") is not None:
            rows.append(L("live: VRAM hit ", "muted", pct(tiers["vram_hit"], 1), "ink b", "  RAM→GPU ", "muted",
                          f(tiers.get("ram_fetch"), 2), "ink b", "/tok  SSD reads ", "muted", f(tiers.get("disk"), 2),
                          "ink b", "/tok", "muted"))
        return rows if compact else box("TOKEN TIME", rows, w, tag=L("--", "muted"))
    ms = tt["ms"]
    src = tt_source(tt)
    head = L("a token ", "muted", f"{ms:.1f} ms", "ink b", f"  ({1000 / ms:.1f} tok/s)", "muted")
    rows = [spread(head, L(src, "faint i"), inner)]
    b = tt_bar(tt, inner)
    v_ = tt["verdict"]
    legend = L("■", "alert" if v_ == "ssd" else "warn", f" SSD wait {tt['disk_ms']:.1f} ms ", "ink",
               f"{tt['disk_ms'] / ms:.0%}", "muted", "  ■", "alert" if v_ in ("gpu", "pcie") else "accent",
               f" GPU {ms - tt['disk_ms']:.1f} ms", "ink", "  ▔", "alert" if v_ == "cpu" else "violet",
               f" {tt['lane_name']} {tt['lane_ms']:.1f} ms ", "ink", f"{tt['lane_ms'] / ms:.0%} parallel", "muted")
    pcie_ = L("PCIe ", "muted", f"{tt['ram_fetch']:.1f}", "alert b" if v_ == "pcie" else "ink b", " fetch/tok", "muted",
              "  VRAM hit ", "muted", f"{tt['vram_hit']:.0f}%", "ink b", "  SSD ", "muted",
              f"{tt['disk_reads']:.1f}", "ink b", " reads/tok", "muted")
    if tt.get("tail_skip"):
        pcie_.add("  skip ", "muted").add(f"{tt['tail_skip']:.1f}", "alert b")
    verdict = L(" ▶ BOTTLENECK ", "alert b r", " " + VERDICT.get(v_, "?") + " ", "alert b").add(tt["what"], "soft")
    rows += b + [legend, pcie_, verdict]
    if not compact:
        rows.append(L("  fix: ", "muted", tt["moves"], "soft i"))
    return rows if compact else box("TOKEN TIME", rows, w, tag=L(VERDICT.get(v_, "?"), "alert b"))


def tiles(v: dict, w: int) -> list:
    """The hardware tiles: GPU load (the busiest card), VRAM, GPU temp, power, PCIe, CPU, disk."""
    g, p, c, dk = v["gpu"], v.get("pcie"), v["cpu"], v["disk"]
    t = []
    t.append(("GPU LOAD", L(f(g.get("busiest"), 0, "%"), "accent b"),
              L(f"GPU{g.get('busiest_card')} busiest of {g['n']}" if g.get("n", 0) > 1 else "model card", "muted"),
              g.get("spark"), "accent"))
    mu = g.get("cards_mem_used") if g.get("cards_mem_used") is not None else g.get("mem_used")
    mt = g.get("cards_mem_total") or g.get("mem_total")
    vram = L(gib(mu, 0 if (mt or 0) >= 100 * 2**30 else 1), "accent b", f"/{gib(mt, 0)} GB", "muted")
    frac = mu / mt if mu is not None and mt else None
    t.append(("VRAM", vram, L("pool ", "muted", f(g.get("pool_gb"), 1, " GB"), "ink"), ("bar", frac), "accent"))
    temp = L(f(g.get("temp"), 0, "°C"), tone_temp(g.get("temp")) + " b")
    temp.add("  jct ", "muted").add(f(g.get("junction"), 0, "°C"), tone_temp(g.get("junction"), True))
    t.append(("GPU TEMP", temp, L(f"GPU{g.get('hottest_card')} hottest" if g.get("n", 0) > 1 else "the card", "muted"),
              ("bar", (g.get("junction") or g.get("temp") or 0) / 110 if g.get("temp") is not None else None),
              tone_temp(g.get("junction"), True).split()[0] if g.get("junction") is not None else
              tone_temp(g.get("temp")).split()[0]))
    pw = L(f(g.get("power"), 0, " W"), "accent b", f" / {f(g.get('power_limit'), 0)} W", "muted")
    pf = g["power"] / g["power_limit"] if g.get("power") is not None and g.get("power_limit") else None
    t.append(("POWER", pw, L("cards in total", "muted"), ("bar", pf), "warn"))
    if p:
        style = "alert b" if p.get("dropped") else "accent b"
        link = L(f"Gen{p['gen']:.0f} x{p['width']:.0f}" if p.get("width") else f"Gen{p['gen']:.0f}", style)
        if g.get("n", 0) > 1 and g.get("slowest_link_card") is not None:
            link.add(f" GPU{g['slowest_link_card']}", "muted")
        mx = L("max ", "muted", f"Gen{p['gen_max']:.0f}" if p.get("gen_max") else DASH, "ink",
               f" x{p['width_max']:.0f}" if p.get("width_max") else "", "ink",
               "  " + rate(p.get("rx_mb")) if p.get("rx_mb") is not None else "", "muted")
        sub = L("▲ DROPPED while working", "alert b") if p.get("dropped") else \
            L("while working" if p.get("working") else "idle (cards may drop)", "faint i")
        t.append(("PCIe", link, mx, ("line", sub), None))
    else:
        t.append(("PCIe", L(DASH, "muted"), L("no link reading", "faint i"), None, None))
    cpu = L(f(c.get("pct"), 0, "%"), "accent b", f"  {f(c.get('temp'), 0, '°C')}", tone_temp(c.get("temp")))
    csub = L(f"{c['cores'] or '?'}c/{c['threads'] or '?'}t" if c.get("threads") else "", "muted",
             f" {c['mhz'] / 1000:.1f}GHz" if c.get("mhz") else "", "muted",
             f" {f(c.get('power'), 0)}W" if c.get("power") is not None else "", "muted")
    t.append(("CPU", cpu, csub, c.get("spark"), "violet"))
    dl = L(rate(dk.get("read_mb")), "accent b", " read", "muted")
    ds = Line()
    mir = [d for d in dk.get("drives") or [] if "mirror" in d["roles"] or "model" in d["roles"]]
    if len(mir) > 1:
        dl = L(rate(dk.get("shard_mb")), "accent b", f" {len(mir)} SSDs", "muted")
        for i, d in enumerate(mir):
            ds.add(f"SSD{i + 1} ", "violet" if "mirror" in d["roles"] else "accent2")
            ds.add(f"{pct(d.get('share'))} ", "ink")
    elif dk.get("mirror_stat") is not None:
        ds.add("SSD1 ", "accent2").add(f"{pct(1 - dk['mirror_stat'])} ", "ink").add("SSD2 ", "violet") \
            .add(f"{pct(dk['mirror_stat'])} ", "ink").add("engine", "muted")
    elif mir:
        ds.add(f"{mir[0]['name']} ", "muted").add(rate(mir[0].get("read_mb")), "ink")
    else:
        ds.add("write " + rate(dk.get("write_mb")), "muted")
    t.append(("DISK", dl, ds, dk.get("spark"), "info"))
    n = max(1, min(len(t), w // 22))
    rows = []
    for i in range(0, len(t), n):
        chunk = t[i:i + n]
        ws = split(w, [1] * len(chunk))
        blocks = []
        for (title, main, sub, extra, style), tw in zip(chunk, ws):
            inner = tw - 4
            third = Line()
            if isinstance(extra, tuple) and extra[0] == "bar":
                third = bar(extra[1], inner, style or "accent")
            elif isinstance(extra, tuple) and extra[0] == "line":
                third = extra[1]
            elif extra is not None:
                third = spark(extra, inner, style or "accent", top=100 if title in ("GPU LOAD", "CPU") else None)
            blocks.append(box(title, [main, sub, third], tw, title_style="soft b"))
        rows += hjoin(blocks, ws)
    return rows


def tile_line(v: dict, w: int) -> list:
    """The tiles in two lines (narrow terminals)."""
    g, p, c, dk = v["gpu"], v.get("pcie"), v["cpu"], v["disk"]
    if (g.get("n") or 0) > 1 or len(v["cards"]) > 2:
        b = L("CPU ", "muted", f(c.get("pct"), 0, "%"), "accent b", f" {f(c.get('temp'), 0, '°C')}  ",
              tone_temp(c.get("temp")))
        return [gpu_summary(v, w), fit(b.cat(disk_summary(v, w - b.width)), w)]
    a = L("GPU ", "muted", f(g.get("busiest"), 0, "%"), "accent b", "  VRAM ", "muted", gib(g.get("mem_used")),
          "accent b", f"/{gib(g.get('mem_total'), 0)}G", "muted", "  ", "", f(g.get("temp"), 0, "°C"),
          tone_temp(g.get("temp")) + " b", "/", "muted", f(g.get("junction"), 0, "°C"),
          tone_temp(g.get("junction"), True), "  ", "", f(g.get("power"), 0, "W"), "accent b")
    b = L("PCIe ", "muted")
    if p:
        b.add(f"Gen{p['gen']:.0f} x{(p.get('width') or 0):.0f}", "alert b" if p.get("dropped") else "accent b")
        if p.get("gen_max"):
            b.add(f"/{p['gen_max']:.0f}x{(p.get('width_max') or 0):.0f}", "muted")
    else:
        b.add(DASH, "muted")
    b.add("  CPU ", "muted").add(f(c.get("pct"), 0, "%"), "accent b").add(f" {f(c.get('temp'), 0, '°C')}",
                                                                         tone_temp(c.get("temp")))
    b.add("  DISK ", "muted").add(rate(dk.get("read_mb")), "accent b")
    return [fit(a, w), fit(b, w)]


def meter(label: str, frac, w: int, value: Line, style: str = "accent", lw: int = 16, vw: int = 0) -> Line:
    bw = max(4, w - lw - max(vw, value.width) - 2)
    return L(label.ljust(lw), "muted").cat(bar(frac, bw, style)).add("  ").cat(value)


def memory_bars(v: dict, w: int) -> list:
    """Context fill, experts in VRAM, the RAM tier, system RAM (swap, pressure), the drafter - the bars all as wide."""
    ctx, t, r, dr = v["context"], v["tiers"], v["ram"], v["drafter"]
    lw = 15 if w >= 60 else 9
    items = []
    cf = ctx["used"] / ctx["max"] if ctx.get("used") is not None and ctx.get("max") else None
    cv = L(toks(ctx.get("used")), "ink b", f" / {toks(ctx.get('max'))}", "muted")
    if ctx.get("reused"):
        cv.add(f"  reused {toks(ctx['reused'])}", "good")
    if not ctx.get("live") and ctx.get("used") is not None:
        cv.add("  last", "faint i")
    items.append(("Context", cf, cv, "info"))
    vu, vs = t.get("vram_used"), t.get("vram_slots")
    vv = L(f"{f(vu)}/{f(vs)}" if vs else DASH, "ink b", f"  {f(t.get('vram_gb'), 1)} GB", "muted")
    if v.get("experts") and vu is not None:
        vv.add(f"  {vu / v['experts']:.0%} of all", "muted")
    items.append(("Experts VRAM", vu / vs if vu is not None and vs else None, vv, "accent"))
    ru, rs = t.get("ram_used"), t.get("ram_slots")
    rv = L(f"{f(ru)}/{f(rs)}" if rs else DASH, "ink b", f"  {f(t.get('ram_gb'), 1)} GB", "muted")
    items.append(("RAM tier", ru / rs if ru is not None and rs else None, rv, "violet"))
    rf = r["used"] / r["total"] if r.get("used") is not None and r.get("total") else None
    sv = L(gib(r.get("used")), "ink b", f"/{gib(r.get('total'), 0)} GB", "muted")
    if r.get("swap_total"):
        sv.add("  swap ", "muted").add(gib(r.get("swap_used")), "warn b" if (r.get("swap_growth") or 0) >= SWAP_GROWING else "ink")
    if r.get("psi_some") is not None:
        sv.add("  psi ", "muted").add(f"{r['psi_some']:.0f}/{(r.get('psi_full') or 0):.0f}",
                                      "alert b" if (r.get("psi_full") or 0) >= 5 else
                                      "warn" if r["psi_some"] >= 10 else "ink")
    items.append(("System RAM", rf, sv, "alert" if rf and rf > 0.97 else "info"))
    if dr.get("kind") or dr.get("total") is not None:
        live = dr.get("live") is not None
        dv = L(pct(dr["live"] if live else dr.get("last")), "good b", " live" if live else " last", "muted",
               f"  {pct(dr.get('total'))}", "ink", " total", "muted",
               "  tok/step ", "muted", f(dr.get("step"), 2) if dr.get("step") is not None else DASH,
               "ink" if dr.get("step") is not None else "muted")
        items.append((f"Drafter {(dr.get('kind') or '').upper()}"[:lw - 1], dr.get("total"), dv, "good"))
    vw = min(max(x[2].width for x in items), max(0, w - lw - 12))
    return [meter(lab, fr, w, val, sty, lw, vw) for lab, fr, val, sty in items]


def link_text(c: dict, own_max: bool = True) -> str:
    """A card's working PCIe link, with its own max (each card its own: a mixed box shows each card's)."""
    if c.get("pcie_gen") is None:
        return DASH
    t = f"G{c['pcie_gen']:.0f}x{(c.get('pcie_width') or 0):.0f}"
    if own_max and c.get("pcie_gen_max"):
        t += f"/{c['pcie_gen_max']:.0f}x{(c.get('pcie_width_max') or 0):.0f}"
    return t


def link_tone(c: dict) -> str:
    return "alert b" if c.get("drop") or c.get("weak") else "warn" if c.get("behind") else "ink"


def layers_text(c: dict) -> str:
    lay = c.get("layers")
    return f"L{lay[0]}-{lay[1] - 1}" if lay else DASH


def card_flags(c: dict) -> Line:
    out = Line()
    if c.get("drop"):
        fg, fw = c.get("drop_from") or (0, 0)
        out.add(f"▲ DROPPED from G{(fg or 0):.0f}x{(fw or 0):.0f} ", "alert b")
    if c.get("weak"):
        out.add("▲ WEAK LINK - holds the split back ", "alert b")
    elif c.get("behind"):
        out.add("behind its twins' link ", "warn")
    thr = [t for t in (c.get("throttle") or []) if t not in ("idle", "gpu_idle")]
    if thr:
        out.add("throttle " + ", ".join(thr), "warn")
    return out


def card_rows(v: dict, w: int, full: bool = False) -> list:
    """One row a card: role, name, load, VRAM, layers and expert pool (a layer split), temps (edge / junction /
    memory), clocks, power against its own limit, PCIe against its own max, its share of the token (phase 2), source.
    The busiest card is starred; a card holding the split back (dropped / weak link) is pink."""
    cols = [("ROLE", 7, "l"), ("CARD", 0, "l"), ("LOAD", 5, "r"), ("VRAM GB", 11, "r"), ("LAYERS", 7, "r"),
            ("POOL", 5, "r"), ("TEMP °C", 7, "r"), ("MHz", 11 if full else 5, "r"), ("W", 7, "r"),
            ("PCIe", 12, "r")]
    if full:
        cols.insert(7, ("MEM°C", 5, "r"))
        cols += [("TOK%", 4, "r"), ("SOURCE", 12, "l")]
    namew = max(6, w - sum(c[1] + 1 for c in cols))
    for drop in ("SOURCE", "TOK%", "MEM°C", "POOL", "LAYERS", "MHz", "W"):   # (narrow: the least needed go first)
        if namew >= 14:
            break
        cols = [c for c in cols if c[0] != drop]
        namew = max(6, w - sum(c[1] + 1 for c in cols))
    cols = [(n, cw or namew, a) for n, cw, a in cols]

    def row(cells):
        ln = Line()
        for (name, cw, a), (text, style) in zip(cols, cells):
            t = text[:cw]
            ln.add((t.rjust(cw) if a == "r" else t.ljust(cw)) + " ", style)
        return ln
    out = [row([(n, "muted b") for n, _, _ in cols])]
    if not v["cards"]:
        out.append(L("no GPU readings (the server read none)", "faint i"))
    for c in v["cards"]:
        role = c.get("role") or "main"
        util = c.get("util")
        hot = max([x for x in (c.get("temp"), (c.get("temp_junction") or 0) - 13) if x is not None], default=None)
        bad = c.get("drop") or c.get("weak")
        cell = {"ROLE": (("★" if c.get("busiest") else " ") + ROLE.get(role, role.upper()),
                         "alert b" if bad else "violet b" if role in ("helper", "igpu") else "accent b"),
                "CARD": ((("" if c.get("index") is None else f"{c['index']} ") + short(c.get("name") or "")),
                         "alert" if bad else "ink b" if c.get("busiest") else "ink"),
                "LOAD": (f(util, 0, "%"), "accent b r" if c.get("busiest") else "accent b" if (util or 0) >= 80
                         else "ink"),
                "VRAM GB": (f"{gib(c.get('mem_used'))}/{gib(c.get('mem_total'), 0)}" if c.get("mem_total") else
                            ("shared" if c.get("shared_memory") else DASH), "ink"),
                "LAYERS": (layers_text(c), "ink" if c.get("layers") else "muted"),
                "POOL": (f(c.get("pool_gb"), 1), "ink" if c.get("pool_gb") is not None else "muted"),
                "TEMP °C": (f"{f(c.get('temp'), 0)}/{f(c.get('temp_junction'), 0)}", tone_temp(hot)),
                "MEM°C": (f(c.get("temp_mem"), 0), tone_temp(c.get("temp_mem"))),
                "MHz": (f"{f(c.get('clock_mhz'))}" + (f"/{f(c.get('mem_clock_mhz'))}" if full else ""), "ink"),
                "W": (f"{f(c.get('power'), 0)}/{f(c.get('power_limit'), 0)}" if c.get("power_limit")
                      else f(c.get("power"), 0), "ink"),
                "PCIe": (link_text(c), link_tone(c)),
                "TOK%": (pct(c.get("tt_share")), "muted" if c.get("tt_share") is None else "ink"),
                "SOURCE": ("+".join(c["source"]) if isinstance(c.get("source"), list) else str(c.get("source") or DASH),
                           "faint")}
        out.append(row([cell[n] for n, _, _ in cols]))
        flags = card_flags(c)
        if full and flags.width:
            out.append(L("         ", "", flags.plain, "alert b" if bad else "warn"))
    return out


# ------------------------------------------------------------------------------------------------ devices
def gpu_summary(v: dict, w: int) -> Line:
    """Many cards in one line: how many, the busiest, VRAM of the model's cards, the hottest, the slowest link."""
    g = v["gpu"]
    byi = {c.get("index"): c for c in v["cards"]}
    n = g.get("n") or len(v["cards"])
    tight = w < 100
    ln = L(f"{n} GPU{'s' if n != 1 else ''}", "accent b")
    b = byi.get(g.get("busiest_card"))
    if b is not None and n > 1:
        ln.add(" · busiest " if not tight else " · busy ", "muted").add(f"GPU{b.get('index')} {f(b.get('util'), 0, '%')}",
                                                                         "accent b")
    elif v["cards"]:
        ln.add(f" {f(v['cards'][0].get('util'), 0, '%')}", "accent b")
    if g.get("cards_mem_total"):
        ln.add(" · VRAM ", "muted").add(f"{gib(g.get('cards_mem_used'), 0)}/{gib(g['cards_mem_total'], 0)}"
                                        + ("G" if tight else " GB"), "ink b")
    hc = byi.get(g.get("hottest_card"))
    if hc is not None and n > 1:
        ln.add(" · hottest " if not tight else " · hot ", "muted").add(
            f"GPU{hc.get('index')} {f(hc.get('temp'), 0, '°C')}", tone_temp(hc.get("temp")))
    sc = byi.get(g.get("slowest_link_card"))
    if sc is not None and n > 1:
        ln.add(" · lowest link " if not tight else " · link ", "muted").add(
            f"GPU{sc.get('index')} {link_text(sc, False)}", link_tone(sc))
    return fit(ln, w)


def disk_summary(v: dict, w: int) -> Line:
    dk = v["disk"]
    shard = [d for d in dk.get("drives") or [] if d.get("share") is not None or "model" in d["roles"]
             or "mirror" in d["roles"]]
    n = len(shard)
    ln = L(f"{n} SSD{'s' if n != 1 else ''} " if n else "disk ", "info b",
           rate(dk.get("shard_mb") if dk.get("shard_mb") is not None else dk.get("read_mb")), "ink b")
    if n > 1:
        ln.add("  ", "")
        for i, d in enumerate(shard):
            ln.add(f"SSD{i + 1} ", "violet" if "mirror" in d["roles"] else "accent2").add(f"{pct(d.get('share'))}  ",
                                                                                            "ink")
    return fit(ln, w)


def gpu_panel(c: dict, v: dict, w: int, h: int) -> list:
    inner = w - 4
    role = c.get("role") or "main"
    bad = c.get("drop") or c.get("weak")
    title = f"GPU{c.get('index')} {short(c.get('name') or '')}"
    tag = L(("★ " if c.get("busiest") else "") + ROLE.get(role, role.upper()),
            "alert b" if bad else "accent b r" if c.get("busiest") else "violet b" if role in ("helper", "igpu")
            else "accent b")
    vw = 14
    rows = [L("load".ljust(6), "muted").cat(bar((c.get("util") or 0) / 100 if c.get("util") is not None else None,
                                                max(3, inner - 6 - 6), "accent")).add(f"{f(c.get('util'), 0, '%'):>6}",
                                                                                    "accent b")]
    if c.get("mem_total"):
        fr = (c.get("mem_used") or 0) / c["mem_total"]
        rows.append(L("vram".ljust(6), "muted").cat(bar(fr, max(3, inner - 6 - vw), "info"))
                    .add(f"{gib(c.get('mem_used')):>6}/{gib(c['mem_total'], 0)} GB".rjust(vw), "ink"))
    else:
        rows.append(L("vram  ", "muted", "shared with the CPU" if c.get("shared_memory") else DASH, "muted"))
    if role == "igpu":
        rows.append(L("RAM read ", "muted", rate(c.get("dram_read_mb")), "ink b", "  lane ", "muted",
                      (v.get("lane") or {}).get("name") or DASH, "violet"))
    else:
        rows.append(L("layers ", "muted", layers_text(c), "ink b" if c.get("layers") else "muted", "  pool ", "muted",
                      f(c.get("pool_gb"), 1, " GB") if c.get("pool_gb") is not None else DASH,
                      "ink b" if c.get("pool_gb") is not None else "muted", "  tok ", "muted", pct(c.get("tt_share")),
                      "muted" if c.get("tt_share") is None else "ink b"))
    rows.append(L(f(c.get("temp"), 0, "°C"), tone_temp(c.get("temp")) + " b", " jct ", "muted",
                  f(c.get("temp_junction"), 0), tone_temp(c.get("temp_junction"), True), " mem ", "muted",
                  f(c.get("temp_mem"), 0), tone_temp(c.get("temp_mem")), f"  {f(c.get('clock_mhz'))} MHz", "muted"))
    pw = L(f(c.get("power"), 0, " W"), "warn b", f"/{f(c.get('power_limit'), 0)}" if c.get("power_limit") else "",
           "muted")
    if role != "igpu":
        pw.add("  PCIe ", "muted").add(link_text(c), link_tone(c))
        if c.get("pcie_rx_mb") is not None:
            pw.add(f" {rate(c['pcie_rx_mb'])}", "muted")
    rows.append(pw)
    fl = card_flags(c)
    rows.append(fl if fl.width else L(" ".join(c["source"]) if isinstance(c.get("source"), list) else "", "faint"))
    title = title[:max(4, w - tag.width - 9)]
    return box(title, rows, w, tag=tag, h=h, tone="alert" if bad else "accent" if c.get("busiest") else "edge",
               corner="alert" if bad else "accent")


def cpu_panel(v: dict, w: int, h: int) -> list:
    c = v["cpu"]
    inner = w - 4
    rows = [L("load".ljust(6), "muted").cat(bar((c.get("pct") or 0) / 100 if c.get("pct") is not None else None,
                                                max(3, inner - 12), "violet")).add(f"{f(c.get('pct'), 0, '%'):>6}",
                                                                                 "violet b"),
            L(f(c.get("temp"), 0, "°C"), tone_temp(c.get("temp")) + " b", f"  {f(c.get('mhz'), 0)}", "ink",
              f"/{f(c.get('mhz_max'), 0)} MHz", "muted", f"  {f(c.get('power'), 0)} W", "warn" if c.get("power")
              is not None else "muted"),
            L(f"{c.get('cores') or '?'}c/{c.get('threads') or '?'}t", "ink", "  lane ", "muted",
              f"{c['lane_threads']:.0f} threads" if c.get("lane_threads") else DASH, "ink"),
            L("ISA ", "muted", "/".join((c.get("isa") or [])[:4]) or DASH, "violet"),
            spark(c.get("per_core") or c.get("spark"), inner, "violet", top=100)]
    return box("CPU " + short(c.get("name") or "")[:max(0, w - 10)], rows, w, h=h, tag=L("CPU", "violet b"))


def ssd_panel(d: dict, i: int, v: dict, w: int, h: int) -> list:
    inner = w - 4
    dk = v["disk"]
    top = max([x.get("read_mb") or 0 for x in dk.get("drives") or []] + [((v.get("bench") or {}).get("ssd") or 0) * 1024,
                                                                        1.0])
    roles = "/".join(d["roles"]) or "other"
    rows = [L("read".ljust(6), "muted").cat(bar((d.get("read_mb") or 0) / top if d.get("read_mb") is not None else None,
                                                max(3, inner - 6 - 11), "info"))
            .add(f"{rate(d.get('read_mb')):>11}", "info b")]
    if d.get("share") is not None:
        rows.append(L("share".ljust(6), "muted").cat(bar(d["share"], max(3, inner - 6 - 11),
                                                         "violet" if "mirror" in d["roles"] else "accent2"))
                    .add(f"{pct(d['share']):>11}", "ink b"))
        rows.append(L("of the expert reads from disk", "faint i"))
    else:
        rows.append(L("not a model drive", "faint i"))
    rows.append(L("write ", "muted", rate(d.get("write_mb")), "ink"))
    rows.append(L(", ".join(d.get("paths") or [])[:inner] or DASH, "faint"))
    return box(f"SSD{i + 1} {d['name']}", rows, w, h=h,
               tag=L(roles.upper(), "violet b" if "mirror" in d["roles"] else "accent2 b"))


def device_panels(v: dict, w: int, h_panel: int = 8) -> list:
    """Every device its own panel: each GPU (main / split / helper / iGPU), the CPU, each drive - in a grid that wraps
    (as many columns as fit at >= 36 wide)."""
    panels = [("gpu", c) for c in v["cards"]] + [("cpu", None)] + \
        [("ssd", (i, d)) for i, d in enumerate(v["disk"].get("drives") or [])]
    cols = max(1, min(len(panels), w // 36))
    out = []
    for k in range(0, len(panels), cols):
        chunk = panels[k:k + cols]
        ws = split(w, [1] * cols)[:len(chunk)]
        blocks = []
        for (kind, x), pw in zip(chunk, ws):
            if kind == "gpu":
                blocks.append(gpu_panel(x, v, pw, h_panel))
            elif kind == "cpu":
                blocks.append(cpu_panel(v, pw, h_panel))
            else:
                blocks.append(ssd_panel(x[1], x[0], v, pw, h_panel))
        out += hjoin(blocks, ws)
    return out


def device_line(kind: str, x, v: dict, w: int) -> Line:
    """One device in one line (the compact Devices layout)."""
    if kind == "gpu":
        c = x
        bad = c.get("drop") or c.get("weak")
        ln = L(("★" if c.get("busiest") else " ") + f"GPU{c.get('index')} ", "alert b" if bad else "accent b",
               f"{ROLE.get(c.get('role'), '?'):<6}", "violet" if c.get("role") in ("helper", "igpu") else "muted",
               f"{f(c.get('util'), 0, '%'):>4} ", "accent b r" if c.get("busiest") else "accent b",
               (f"{gib(c.get('mem_used'), 0)}/{gib(c.get('mem_total'), 0)}G " if c.get("mem_total") else "shared "),
               "ink", f"{f(c.get('temp'), 0, '°C')} ", tone_temp(c.get("temp")),
               f"{layers_text(c)} " if c.get("layers") else "", "muted",
               link_text(c, False) if c.get("role") != "igpu" else "", link_tone(c))
        return fit(ln.add(" " + short(c.get("name") or ""), "faint"), w)
    if kind == "cpu":
        c = v["cpu"]
        return fit(L(" CPU  ", "violet b", f"{f(c.get('pct'), 0, '%'):>10} ", "violet b",
                     f"{f(c.get('temp'), 0, '°C')} ", tone_temp(c.get("temp")), f"{f(c.get('power'), 0)} W  ", "muted",
                     short(c.get("name") or ""), "faint"), w)
    i, d = x
    return fit(L(f" SSD{i + 1} ", "info b", f"{('/'.join(d['roles']) or 'other'):<7}", "muted",
                 f"{rate(d.get('read_mb')):>10} ", "info b", f"share {pct(d.get('share'))}  " if d.get("share")
                 is not None else "", "ink", d["name"], "faint"), w)


def speed_strip(v: dict, w: int) -> list:
    """The Devices layout's header strip: decode and prompt tok/s, the token's time as one bar with the verdict."""
    inner = w - 4
    d, p = v["decode"], v["prompt"]
    a = L("DECODE ", "muted b", f(d.get("value"), 1), "accent b", " tok/s ", "muted",
          d.get("label") or "no answer yet", "good" if d.get("label") == "live" else "soft",
          "    PROMPT ", "muted b", f(p.get("value"), 0), "info b", " tok/s ", "muted", p.get("label") or "", "soft")
    ctx = v["context"]
    if ctx.get("used") is not None and ctx.get("max"):
        a.add("    ctx ", "muted").add(f"{toks(ctx['used'])}/{toks(ctx['max'])}", "ink")
    if v["drafter"].get("total") is not None:
        a.add("    drafts ", "muted").add(pct(v["drafter"]["total"]), "good b")
    a.add("    ", "").cat(spark(d.get("spark"), max(0, inner - a.width - 4), "accent"))
    rows = [a]
    tt = v.get("tt")
    if tt:
        lw = 18
        barw = max(8, inner - lw - 2)
        rows += [tt_bar(tt, barw)[0].add("  ").cat(L(" ▶ " + VERDICT.get(tt["verdict"], "?") + " ", "alert b r"))]
        rows.append(L(f"a token {tt['ms']:.1f} ms · SSD wait {tt['disk_ms']:.1f} · GPU {tt['ms'] - tt['disk_ms']:.1f} · "
                      f"{tt['lane_name']} {tt['lane_ms']:.1f} (parallel) · ", "muted", tt["what"], "soft"))
    else:
        rows.append(L("token time ", "muted", DASH, "muted", "  " + (v.get("tt_why") or ""), "faint i"))
    return box("SPEED", rows, w, tag=L("◆ live", "good b") if d.get("label") == "live" else None)


def devices_overview(v: dict, w: int, h: int, ui: dict) -> list:
    """Layout B "Devices": a small speed strip, then every device in its own panel (the multi-GPU view)."""
    out = banner_lines(v, w)
    if w >= 110 and h >= 30:
        out += speed_strip(v, w)
        out += device_panels(v, w)
        if h - len(out) >= 16:                    # (room: the tiers too)
            out += box("MEMORY · TIERS", memory_bars(v, w - 4), w)
        room = h - len(out) - 3
        if room >= 3:
            out += box("RECENT REQUESTS", req_rows(v, w - 4, room), w,
                       tag=L(f"{(v.get('totals') or {}).get('requests', 0)} since start", "muted"))
        return out
    inner = w - 4
    rows = speed_tile(v, "decode", w, True) + speed_tile(v, "prompt", w, True)
    tt = v.get("tt")
    rows.append(L("token ", "muted").cat(tt_bar(tt, max(6, inner - 16))[0]).add(f" {VERDICT.get(tt['verdict'], '?')}",
                                                                             "alert b")
                if tt else L("token time ", "muted", DASH, "muted", "  " + (v.get("tt_why") or ""), "faint i"))
    out += box("SPEED", rows, w)
    devs = [("gpu", c) for c in v["cards"]] + [("cpu", None)] + \
        [("ssd", (i, d)) for i, d in enumerate(v["disk"].get("drives") or [])]
    room = max(1, h - len(out) - 4)
    rows = [gpu_summary(v, inner), disk_summary(v, inner)]
    if len(devs) <= room - 2:
        rows += [device_line(k, x, v, inner) for k, x in devs]
    else:                                      # (too many: the ones that matter - busiest, hottest, lowest link)
        g = v["gpu"]
        byi = {c.get("index"): c for c in v["cards"]}
        keep = []
        for i in (g.get("busiest_card"), g.get("hottest_card"), g.get("slowest_link_card"), *g.get("weak_cards", []),
                  *g.get("dropped_cards", [])):
            if i in byi and byi[i] not in keep:
                keep.append(byi[i])
        keep = keep[:max(0, room - 3)]
        rest = [c for c in v["cards"] if c not in keep]
        keep += rest[:max(0, room - 3 - len(keep))]
        keep.sort(key=lambda c: v["cards"].index(c))
        rows += [device_line("gpu", c, v, inner) for c in keep]
        rows.append(L(f"+{len(devs) - len(keep)} more · ", "muted", "5", "accent b",
                      " Hardware lists every device", "muted"))
    out += box("DEVICES", rows, w, tag=L(f"{len(devs)} devices", "muted"))
    return out


def short(name: str) -> str:
    for a in ("NVIDIA ", "GeForce ", "AMD ", "Radeon ", "(TM)", " Graphics", "Tesla ", "-PCIE", "-SXM2"):
        name = name.replace(a, "")
    name = re.sub(r"\s+\d+-Core Processor|\s+w/ .*$|\s+Processor$", "", name)
    return " ".join(name.split())


def cpu_line(v: dict, w: int) -> Line:
    c = v["cpu"]
    ln = L("CPU ", "accent b", short(c.get("name") or "")[:28], "ink", "  ", "", f(c.get("pct"), 0, "%"), "ink b",
           f"  {f(c.get('mhz'), 0)} MHz", "muted", "  ", "", f(c.get("temp"), 0, "°C"), tone_temp(c.get("temp")),
           f"  {f(c.get('power'), 0)} W", "muted")
    if c.get("isa"):
        ln.add("  ISA ", "muted").add("/".join(c["isa"][:4]), "violet")
    if c.get("lane_threads"):
        ln.add(f"  lane {c['lane_threads']:.0f}t", "muted")
    return fit(ln, w)


def req_rows(v: dict, w: int, n: int, sel: int | None = None, log=None, full: bool = False) -> list:
    cols = [("TIME", 8, "l"), ("STATUS", 8, "l"), ("PROMPT", 7, "r"), ("REUSED", 7, "r"), ("OUT", 6, "r"),
            ("PP T/S", 7, "r"), ("TG T/S", 7, "r"), ("TIME", 8, "r"), ("DRAFT", 6, "r")]
    if full:
        cols.insert(1, ("API", 6, "l"))
        cols.append(("HIT", 6, "r"))
    while sum(c[1] + 1 for c in cols) > w and len(cols) > 4:
        cols.pop()

    def row(cells, base=""):
        ln = Line()
        for (name, cw, a), (text, style) in zip(cols, cells):
            t = text[:cw]
            ln.add((t.rjust(cw) if a == "r" else t.ljust(cw)) + " ", (style + base).strip())
        return ln
    out = [row([(c[0], "muted b") for c in cols])]
    reqs = v.get("requests") or []
    if not reqs:
        out.append(L("no requests yet - the API: ", "faint i", f"{v.get('url') or ''}/v1", "accent"))
    for i, r in enumerate(reqs[:n]):
        status, st = FINISH.get(r.get("finish"), (str(r.get("finish") or "?").upper()[:8], "soft"))
        pt, ru = r.get("prompt_tokens"), r.get("reused")
        da, do = r.get("drafts_accepted"), r.get("drafts_offered")
        cells = [(time.strftime("%H:%M:%S", time.localtime(r["time"])) if r.get("time") else DASH, "muted")]
        if full:
            cells.append((API.get(r.get("api"), str(r.get("api") or DASH)), "soft"))
        cells += [(status, st + " b"), (toks(pt), "ink"),
                  (f"{ru / pt:.0%}" if ru and pt else ("0%" if ru == 0 else DASH), "good" if ru else "muted"),
                  (toks(r.get("output_tokens")), "ink"), (f(r.get("prefill_tok_s"), 0), "info b"),
                  (f(r.get("decode_tok_s"), 1), "accent b"), (dur(r.get("duration_s")), "muted"),
                  (f"{da / do:.0%}" if da is not None and do else DASH, "good")]
        if full:
            cells.append((pct(r.get("hit_rate")), "ink"))
        out.append(row(cells, " r" if sel == i else ""))
    return out


def overview(v: dict, w: int, h: int, ui: dict) -> list:
    out = banner_lines(v, w)
    if v["conn"] in ("loading", "stopped", "auth", "connecting", "disconnected", "error") and not v["has_metrics"]:
        return out + status_panel(v, w, h, ui)
    if ui.get("layout") == "devices":
        return devices_overview(v, w, h, ui)
    rich_ = w >= 110 and h >= 30
    if rich_:
        ws = split(w, [27, 27, 46])
        tt = tt_tile(v, ws[2])
        out += hjoin([speed_tile(v, "decode", ws[0], h=len(tt)), speed_tile(v, "prompt", ws[1], h=len(tt)), tt], ws)
        out += tiles(v, w)
        lw = split(w, [45, 55])
        mem = box("MEMORY · TIERS", memory_bars(v, lw[0] - 4), lw[0])
        crd = box("CARDS", card_rows(v, lw[1] - 4) + [cpu_line(v, lw[1] - 4)], lw[1],
                  tag=L(f"{len(v['cards'])} device{'s' if len(v['cards']) != 1 else ''}", "muted"))
        out += hjoin([mem, crd], lw)
        nreq = len(v.get("requests") or []) or 1
        if h - len(out) - (nreq + 3) >= 7:            # (room to spare: the last minute's decode as a chart)
            rows = max(3, min(6, h - len(out) - (nreq + 3) - 2))
            out += box("THROUGHPUT · LAST 60 s", chart(v["decode"].get("spark"), w - 4, rows, "accent"), w,
                       tag=L("decode tok/s", "muted"))
        room = max(3, h - len(out) - 3)
        out += box("RECENT REQUESTS", req_rows(v, w - 4, room), w,
                   tag=L(f"{(v.get('totals') or {}).get('requests', 0)} since start", "muted"))
        return out
    # compact (80x24): speed + token time in one box, the tiles in two lines, the bars, the requests
    inner = w - 4
    rows = speed_tile(v, "decode", w, True) + speed_tile(v, "prompt", w, True)
    rows += tt_tile(v, w, True) if v.get("tt") else [L("token time ", "muted", DASH, "muted", "  ", "",
                                                       v.get("tt_why") or "", "faint i")]
    out += box("SPEED", rows, w, tag=L("◆ live", "good b") if v["decode"]["label"] == "live" else None)
    out += tile_line(v, w)
    out += box("MEMORY", memory_bars(v, inner), w)
    out += box("RECENT", req_rows(v, inner, 5), w)
    return out


def status_panel(v: dict, w: int, h: int, ui: dict) -> list:
    """No numbers to show yet: connecting, loading (the warm-up), stopped, auth, disconnected, error."""
    k = state_key(v)
    glyph, text, style = STATE[k]
    rows = [L(f"{glyph} {text}", style + " b"), Line()]
    if k == "loading":
        ld = v.get("loading") or {}
        frac = ld.get("frac")
        rows.append(bar(frac, max(10, w - 14), "warn").add(f"  {pct(frac)}", "warn b"))
        el = L("elapsed ", "muted", dur(ld.get("elapsed")), "ink")
        if ld.get("step"):
            el.add("   stage ", "muted").add(ld["step"], "warn b")
        if ld.get("gb") and ld["gb"][1]:
            el.add("   warm ", "muted").add(f"{ld['gb'][0]:.1f} / {ld['gb'][1]:.1f} GB", "ink")
        rows.append(el)
        rows.append(L(ld.get("stage") or "the engine reads the model and pins its RAM tier", "soft"))
        if v.get("remote"):
            rows.append(L("(the server's own progress, from its /health)", "faint i"))
        rows.append(Line())
        rows.append(L("The PC can feel slow while the model loads: up to all but ~6 GB of the free RAM is pinned.",
                      "faint i"))
    elif k == "stopped":
        cfg = ui.get("config_file")
        rows.append(L("The server is not running.", "soft"))
        rows.append(L("Enter", "accent b", f" start {cfg}" if cfg else " start the last config", "ink",
                      "   8", "accent b", " pick another in Models", "ink"))
    elif k == "auth":
        rows.append(L("/metrics needs the API key.", "soft"))
        rows.append(L("Start with ", "muted", "--key-env NAME", "accent b", " (the environment variable that holds the "
                      "key; the key itself is never stored), then press ", "muted", "r", "accent b"))
    elif k == "disconnected":
        rows.append(L(f"No answer from {v.get('url')}: {v.get('error') or ''}", "soft"))
        rows.append(L("Retrying with back-off (up to 10 s). " + ("A remote server starting up looks the same until it "
                                                                  "is ready." if v.get("remote") else ""), "muted"))
    elif k == "error":
        rows.append(L("The engine stopped with an error. Its last line:", "soft"))
        rows.append(L(ui.get("last_error") or "", "alert"))
        rows.append(L("6", "accent b", " Logs shows the rest", "muted"))
    else:
        rows.append(L(f"Connecting to {v.get('url')} ...", "soft"))
    return box(text, rows, w, tone="edge", title_style=style + " b")


# ------------------------------------------------------------------------------------------------ speed
def speed_view(v: dict, w: int, h: int, ui: dict) -> list:
    out = banner_lines(v, w)
    inner = w - 4
    d, p = v["decode"], v["prompt"]
    rows = [L("decode (live tok/s, last 60 s)  ", "muted", f(d.get("value"), 1), "accent b", f"  {d['label'] or ''}",
              "soft"), spark(d.get("spark"), inner, "accent"),
            L("prompt (prefill tok/s)  ", "muted", f(p.get("value"), 0), "info b", f"  {p['label'] or ''}", "soft"),
            spark(p.get("spark"), inner, "info")]
    reqs = [r for r in (v.get("requests") or []) if r.get("decode_tok_s") or r.get("prefill_tok_s")]
    if reqs:
        top_d = max((r.get("decode_tok_s") or 0) for r in reqs) or 1
        top_p = max((r.get("prefill_tok_s") or 0) for r in reqs) or 1
        rows.append(L("per request (newest first)", "muted b"))
        bw = min(40, max(6, (inner - 32) // 2))
        for r in reqs[:min(8, max(2, h // 6))]:
            rows.append(L(time.strftime("%H:%M:%S ", time.localtime(r["time"])) if r.get("time") else "", "muted")
                        .add("TG ", "muted").cat(bar((r.get("decode_tok_s") or 0) / top_d, bw, "accent"))
                        .add(f" {f(r.get('decode_tok_s'), 1):>6}", "accent b").add("  PP ", "muted")
                        .cat(bar((r.get("prefill_tok_s") or 0) / top_p, bw, "info"))
                        .add(f" {f(r.get('prefill_tok_s'), 0):>6}", "info b"))
    out += box("SPEED · DECODE / PROMPT", rows, w)
    tt = v.get("tt")
    rows = []
    if tt:
        ms = tt["ms"]
        bw = max(10, inner - 34)
        rows.append(L("a token ", "muted", f"{ms:.1f} ms", "ink b", f" ({1000 / ms:.1f} tok/s) - the GPU's token = 100%; "
                      "the lane runs alongside it, never added", "muted"))
        rows.append(L("SSD wait   ".ljust(14), "muted").cat(bar(tt["disk_ms"] / ms, bw, "alert" if tt["verdict"] == "ssd"
                                                                 else "warn"))
                    .add(f" {tt['disk_ms']:6.1f} ms {tt['disk_ms'] / ms:4.0%}", "ink"))
        rows.append(L(f"{tt['lane_name']:<14}", "muted").cat(bar(min(1, tt["lane_ms"] / ms), bw,
                                                                  "alert" if tt["verdict"] == "cpu" else "violet"))
                    .add(f" {tt['lane_ms']:6.1f} ms {tt['lane_ms'] / ms:4.0%}", "ink"))
        rows.append(L("PCIe fetches".ljust(14), "muted").cat(bar(min(1, tt["ram_fetch"] / 20), bw,
                                                                  "alert" if tt["verdict"] == "pcie" else "accent"))
                    .add(f" {tt['ram_fetch']:6.1f} /tok (pcie >= 10)", "ink"))
        rows.append(Line())
        rows.append(L(" ▶ BOTTLENECK ", "alert b r", " " + VERDICT.get(tt["verdict"], "?") + "  ", "alert b",
                      tt["what"], "soft"))
        rows.append(L("   what moves it: ", "muted", tt["moves"], "soft i"))
        rows.append(L("   evidence (the --bench rules, ", "muted", "same code", "accent", "):", "muted"))
        rows += [L("   · " + x, "soft") for x in tt["evidence"]]
        if tt.get("miss_ms") is not None:
            rows.append(L("   misses: ", "muted", f"{tt['miss_ms']:.1f} ms", "ink", " a token waiting for experts not "
                          "in VRAM (SSD waits are part of it)", "muted"))
        tag = L(tt_source(tt), "faint i")
    else:
        rows.append(L("SSD wait / lane busy ms per token: ", "muted", DASH, "muted", "   ", "", v.get("tt_why") or "",
                      "faint i"))
        rows.append(L("A 1.3.80+ engine sends them in its STAT line every 8 tokens (live, the last ~64 tokens); older "
                      "ones only in the engine log after each answer.", "faint"))
        tag = L("--", "muted")
    out += box("WHERE A TOKEN'S TIME GOES", rows, w, tag=tag)
    t, dk, dr, ln = v["tiers"], v["disk"], v["drafter"], v["lane"]
    lw = split(w, [1, 1])
    ssd = [kv("expert reads ", f(tt["disk_reads"], 1) if tt else f(t.get("disk"), 2), "/tok", kw=14),
           kv("waits ", f(tt["disk_waits"], 1) if tt else DASH, "/tok", kw=14),
           kv("wait ms ", f(tt["disk_ms"], 1) if tt else DASH, " ms/tok", kw=14)]
    for dv in dk.get("drives") or []:
        ssd.append(L(f"{'/'.join(dv['roles']) or 'drive'} ", "violet", dv["name"] + " ", "muted",
                     rate(dv.get("read_mb")), "ink b"))
    if dk.get("mirror_stat") is not None:
        a, b = dk["mirror_mb"]
        ssd.append(L("mirror split (engine) ", "muted", pct(dk["mirror_stat"]), "good b",
                     f"  SSD2 read {b / 1024:.1f} of {(a + b) / 1024:.1f} GB this answer", "muted"))
    elif dk.get("mirror_share") is not None:
        ssd.append(L("mirror share (live MB/s) ", "muted", pct(dk["mirror_share"]), "good b"))
    if dk.get("mirror_end"):
        a, b = dk["mirror_end"]
        ssd.append(L("mirror at the last end ", "muted", f"{b / max(1e-9, a + b):.0%}", "ink b",
                     f" ({b:.1f} of {a + b:.1f} GB)", "muted"))
    exp_mib = ln.get("expert_mib")
    lane = [kv("kind ", (ln.get("name") or DASH) + (f" ({ln['state']})" if ln.get("state") else ""), kw=14),
            kv("threads ", f(v["cpu"].get("lane_threads")), kw=14),
            kv("PCIe share ", f(ln.get("share"), 2), kw=14),
            kv("experts ", f(tt["lane"], 1) if tt else DASH, "/tok", kw=14),
            kv("busy ", f(tt["lane_ms"], 1) if tt else DASH, " ms/tok", kw=14),
            L("bandwidth".ljust(14), "muted", f(ln.get("gbs"), 1) if ln.get("gbs") is not None else DASH, "ink b",
              " GB/s", "muted", f"  est.: experts/tok × {exp_mib:.1f} MiB ÷ busy ms" if ln.get("gbs") is not None
              and exp_mib else ("  (an expert's MiB and the lane's ms come from a 1.3.80+ engine)" if tt else ""),
              "faint i")]
    out += hjoin([box("SSD", ssd, lw[0]), box("LANE", lane, lw[1])], lw)
    tiers_ = [kv("VRAM hit ", pct(t.get("vram_hit"), 1), kw=16),
              kv("RAM→GPU ", f(t.get("ram_fetch"), 2), "/tok", kw=16),
              kv("promotions ", f(t.get("promo"), 2), "/tok", kw=16),
              kv("RAM of misses ", pct(t["ram_fetch"] / (t["ram_fetch"] + t["disk"])) if t.get("ram_fetch") is not None
                 and t.get("disk") is not None and t["ram_fetch"] + t["disk"] > 0 else DASH, kw=16),
              kv("tail skip ", f(tt["tail_skip"], 1) if tt else DASH, "/tok", kw=16)]
    drafter = [kv("kind ", (dr.get("kind") or ("--" if dr.get("total") is not None else "none")).upper(), kw=14),
               kv("live ", pct(dr.get("live")), kw=14) if dr.get("live") is not None else
               kv("last request ", pct(dr.get("last")), kw=14),
               kv("total ", pct(dr.get("total")), f"  ({f(dr.get('accepted'))} of {f(dr.get('offered'))})", kw=14),
               L("tokens/step".ljust(14), "muted", f(dr.get("step"), 2) if dr.get("step") is not None else DASH,
                 "ink b", f"  {dr.get('step_label') or ''}" if dr.get("step") is not None else
                 "  (a 1.3.80+ engine counts the verify steps)", "faint i")]
    out += hjoin([box("VRAM POOL · RAM TIER", tiers_, lw[0]), box("DRAFTER", drafter, lw[1])], lw)
    b = v.get("bench")
    if b:
        rows = [L(b["line"], "soft")]
        if b.get("decode") and v["decode"].get("last"):
            diff = v["decode"]["last"] / b["decode"] - 1
            rows.append(L("last answer vs bench decode: ", "muted", f"{diff:+.0%}", "good b" if diff >= -0.05 else
                          "warn b", f" ({v['decode']['last']:.1f} vs {b['decode']:.1f} tok/s; y copies the line)",
                          "muted"))
        out += box("LAST --bench", rows, w, tag=L(b.get("config") or "", "faint"))
    return out


# ------------------------------------------------------------------------------------------------ memory
def memory_view(v: dict, w: int, h: int, ui: dict) -> list:
    out = banner_lines(v, w)
    inner = w - 4
    t, r = v["tiers"], v["ram"]
    total = v.get("experts")
    vu, ru = t.get("vram_used"), t.get("ram_used")
    rows = []
    if total and vu is not None and ru is not None:
        whole = max(total, vu + ru)
        a = round(inner * vu / whole)
        b_ = min(inner - a, round(inner * ru / whole))
        rows.append(L("█" * a, "accent", "█" * b_, "violet", "█" * (inner - a - b_), "warn"))
        rows.append(L("■", "accent", f" VRAM pool {f(vu)} experts · {f(t.get('vram_gb'), 1)} GB   ", "ink",
                      "■", "violet", f" RAM tier {f(ru)} · {f(t.get('ram_gb'), 1)} GB   ", "ink",
                      "■", "warn", f" SSD only {f(max(0, total - vu - ru))}", "ink"))
    else:
        rows.append(bar(None, inner))
        rows.append(L("where the experts live: the engine's STAT lines (fast path, while it writes); ", "faint i",
                      f"{f(total)} experts", "muted"))
    out += box("WHERE THE EXPERTS LIVE", rows, w)
    gcards = [c for c in v["cards"] if c.get("mem_total")]
    if gcards:
        crow = []
        for c in gcards:
            val = L(f"{gib(c.get('mem_used'))}/{gib(c['mem_total'], 0)} GB", "ink b", "  pool ", "muted",
                    f(c.get("pool_gb"), 1), "ink" if c.get("pool_gb") is not None else "muted", "  ", "",
                    layers_text(c), "ink" if c.get("layers") else "muted")
            crow.append(meter(f"{'★' if c.get('busiest') else ' '}GPU{c.get('index')} {short(c.get('name') or '')}"[:15],
                              (c.get("mem_used") or 0) / c["mem_total"], inner, val, "accent", 16, 30))
        if not any(c.get("pool_gb") is not None for c in gcards):
            crow.append(L("pool GB / layers per card: ", "muted", DASH, "muted", "  from the engine log on this PC "
                          "(a remote server: engine counter, phase 2)", "faint i"))
        out += box("VRAM PER CARD", crow, w, tag=L(f"{len(gcards)} card{'s' if len(gcards) != 1 else ''}", "muted"))
    ctx = v["context"]
    kvr = [kv("KV cache ", f(ctx.get("kv_gb"), 2), " GB", kw=14), kv("KV context ", toks(ctx.get("kv_ctx")), kw=14),
           kv("max context ", toks(ctx.get("max")), kw=14)]
    ramr = [kv("used ", gib(r.get("used")), f" / {gib(r.get('total'), 0)} GB", kw=14),
            kv("RAM tier ", f(t.get("ram_gb"), 1), " GB" + (f"  ({t['ram_gb'] * 2**30 / r['total']:.0%} of RAM, "
                                                             "default cap 73%)" if t.get("ram_gb") and r.get("total")
                                                             else ""), kw=14),
            kv("page cache ", gib(r.get("cached")), " GB", kw=14),
            kv("swap ", gib(r.get("swap_used")), f" / {gib(r.get('swap_total'))} GB", kw=14),
            kv("PSI some/full ", f"{f(r.get('psi_some'), 1)} / {f(r.get('psi_full'), 1)}", " % of 10 s", kw=14)]
    lw = split(w, [2, 3])
    out += hjoin([box("KV · CONTEXT", kvr, lw[0]), box("SYSTEM RAM", ramr, lw[1])], lw)
    out += box("RAM (last 60 s)", [spark(r.get("spark"), inner, "info")], w)
    diag = diagnose(v)
    out += box("WHAT IS SLOWING YOU DOWN", [L(x, s) for x, s in diag], w, title_style="alert b")
    return out


def diagnose(v: dict) -> list:
    """The low-spec line: the verdict and what to do on this PC, in one or two lines."""
    tt = v.get("tt")
    out = []
    r = v["ram"]
    if (r.get("psi_full") or 0) >= 5 or memory_tight(r):
        out.append(("Memory pressure: the RAM tier and the page cache fight. Lower STRATA_GLM_RAM_GB or close apps.",
                    "alert b"))
    if not tt:
        out.append(("Not known yet: the per-token ms come from the engine log after an answer "
                    "(or, from phase 2, the live STAT line).", "muted"))
        return out
    if tt["verdict"] == "ssd":
        full = v["tiers"].get("ram_used") and v["tiers"].get("ram_slots") and \
            v["tiers"]["ram_used"] >= 0.97 * v["tiers"]["ram_slots"]
        out.append((f"SSD-bound: {tt['disk_ms']:.1f} ms of every {tt['ms']:.1f} ms token waits for the SSD"
                    + (" and the RAM tier is full" if full else "") + ".", "alert b"))
        smallest = "IQ1" in (v.get("quant") or "")
        out.append(("A second SSD (STRATA_GLM_MIRROR, docs/SSD2) cut the waits ~20%; "
                    + ("more RAM for the RAM tier" if smallest else "a smaller quant (e.g. UD-IQ1_M) or more RAM")
                    + "; or the tail skip (STRATA_GLM_TAIL_SKIP, lossy).", "soft"))
    else:
        out.append((f"{VERDICT.get(tt['verdict'], '?')}: {tt['what']}.", "warn b"))
        out.append((f"What moves it: {tt['moves']}", "soft"))
    return out


# ------------------------------------------------------------------------------------------------ requests
def requests_view(v: dict, w: int, h: int, ui: dict) -> list:
    out = banner_lines(v, w)
    sel = ui.get("sel", 0)
    reqs = v.get("requests") or []
    n = max(3, h - 16)
    out += box("REQUESTS", req_rows(v, w - 4, n, sel, full=w >= 100), w,
               tag=L(f"{len(reqs)} shown · ↑↓ select", "muted"))
    if 0 <= sel < len(reqs):
        r = reqs[sel]
        rows = [L("prompt ", "muted", f"{f(r.get('prompt_ms'), 0)} ms", "ink b", "  decode ", "muted",
                  f"{f(r.get('decode_ms'), 0)} ms", "ink b", "  engine tokens ", "muted", f(r.get("engine_generated")),
                  "ink b", "  API ", "muted", str(r.get("api") or DASH), "ink",
                  "  projection ", "muted", DASH if r.get("projection") is None else str(r["projection"]), "ink")]
        log = ui.get("log")
        rec = log.match(r) if log is not None else None
        if rec:
            ms = rec["ms"]
            rows.append(L("glm stat ", "accent b", f"{ms:.1f} ms/tok", "ink b", "  SSD wait ", "muted",
                          f"{rec['disk_ms']:.1f} ms ({rec['disk_ms'] / ms:.0%})", "ink", "  lane ", "muted",
                          f"{rec['lane_ms']:.1f} ms", "ink", "  PCIe ", "muted", f"{rec['ram_fetch']:.1f}/tok", "ink",
                          "  VRAM hit ", "muted", f"{rec['vram_hit']:.0f}%", "ink"))
        else:
            rows.append(L("glm stat: ", "muted", DASH, "muted", "  " + ("matched by time for answers since the "
                          "Terminal opened (local engine log)" if log is not None else LOCAL_ONLY_HINT), "faint i"))
        out += box("DETAIL", rows, w)
    tot = v.get("totals") or {}
    pm, dm = tot.get("prompt_ms") or 0, tot.get("decode_ms") or 0
    rows = [L("requests ", "muted", f(tot.get("requests")), "ink b", "  prompt tokens ", "muted",
              toks(tot.get("prompt_tokens")), "ink b", "  reused ", "muted", toks(tot.get("reused")), "good b",
              f" ({tot['reused'] / tot['prompt_tokens']:.0%})" if tot.get("prompt_tokens") else "", "good",
              "  output ", "muted", toks(tot.get("output_tokens")), "ink b"),
            L("mean prefill ", "muted", f(((tot.get("prompt_tokens") or 0) - (tot.get("reused") or 0)) / (pm / 1000), 0)
              if pm else DASH, "info b", " tok/s", "muted", "  mean decode ", "muted",
              f((tot.get("output_tokens") or 0) / (dm / 1000), 1) if dm else DASH, "accent b", " tok/s", "muted")]
    if v.get("state") in ("reading", "generating"):
        rows.append(L("now ", "muted").cat(state_pill(v)).add(f"  {dur(v.get('elapsed'))}", "muted"))
    out += box("TOTALS SINCE START", rows, w)
    return out


LOCAL_ONLY_HINT = "local only (the engine log on the PC that runs the server)"


# ------------------------------------------------------------------------------------------------ hardware
def hardware_view(v: dict, w: int, h: int, ui: dict) -> list:
    out = banner_lines(v, w)
    inner = w - 4
    out += box("CARDS", card_rows(v, inner, full=w >= 100), w,
               tag=L("edge/junction °C · core/mem MHz · W/limit", "faint"))
    p = v.get("pcie")
    rows = []
    many = len([c for c in v["cards"] if c.get("pcie_gen") is not None]) > 1
    if many:
        for c in v["cards"]:
            if c.get("pcie_gen") is None:
                continue
            rows.append(L(f"GPU{c.get('index')} ", "accent b", f"{short(c.get('name') or '')[:18]:<19}", "ink",
                          f"{link_text(c, False):>7}", link_tone(c), "  max ", "muted",
                          f"G{(c.get('pcie_gen_max') or 0):.0f}x{(c.get('pcie_width_max') or 0):.0f}" if
                          c.get("pcie_gen_max") else DASH, "ink", f"  {f(c.get('link_gbs'), 1)} GB/s", "muted",
                          f"  →GPU {rate(c.get('pcie_rx_mb'))}  ", "muted").cat(card_flags(c)))
    if p and not many:
        rows.append(L("working link ", "muted", f"Gen{p['gen']:.0f} x{(p.get('width') or 0):.0f}",
                      "alert b" if p.get("dropped") else "accent b", "   card max ", "muted",
                      f"Gen{(p.get('gen_max') or 0):.0f} x{(p.get('width_max') or 0):.0f}", "ink",
                      "   →GPU ", "muted", rate(p.get("rx_mb")), "ink"))
    if p:
        rows.append(L("judged only while reading / generating (an idle card drops to Gen1 by itself); "
                      "AMD: the narrowest link up to the root port; no MB/s counter on RDNA 3/4", "faint i"))
    if not p:
        rows.append(L("no PCIe reading (AMD on Windows, or no GPU found)", "faint i"))
    out += box("PCIe", rows, w)
    c = v["cpu"]
    rows = [cpu_line(v, inner)]
    if c.get("per_core"):
        rows.append(L("cores ", "muted").cat(spark(c["per_core"], min(inner - 6, len(c["per_core"])), "violet",
                                                   top=100)))
    for cl in c.get("clusters") or []:
        rows.append(L(f"cluster cpus {cl['cpus'][0]}-{cl['cpus'][-1]} ", "muted", f(cl.get("mhz")), "ink b",
                      f" / {f(cl.get('max_mhz'))} MHz", "muted"))
    rows.append(L("governor ", "muted", str(c.get("governor") or DASH), "ink", "  EPP ", "muted",
                  str(c.get("epp") or DASH), "ink", "  power ", "muted",
                  f(c.get("power"), 1, " W") if c.get("power") is not None else
                  ("RAPL root-only (see MANUAL)" if c.get("power_blocked") else DASH), "ink"))
    out += box("CPU", rows, w)
    dk = v["disk"]
    rows = [L("all disks ", "muted", rate(dk.get("read_mb")), "accent b", " read  ", "muted", rate(dk.get("write_mb")),
              "ink", " write", "muted", "   model drive free ", "muted", gib(dk.get("free"), 0), "ink",
              f" / {gib(dk.get('total'), 0)} GB", "muted")]
    for d in dk.get("drives") or []:
        rows.append(L(f"{('/'.join(d['roles']) or 'other'):<12}", "violet b", f"{d['name']:<16}", "ink",
                      f"{rate(d.get('read_mb')):>12}", "accent b", " read  ", "muted",
                      f"share {pct(d.get('share')):>4}  " if d.get("share") is not None else " " * 12, "ink",
                      ", ".join(d.get("paths") or [])[:max(0, inner - 62)], "faint"))
    b = v.get("bench")
    if b and b.get("ssd"):
        rows.append(L("--bench SSD read ", "muted", f"{b['ssd']:.1f} GB/s", "violet"))
    out += box("STORAGE", rows, w)
    rec = ui.get("thermal")
    out += [fit(L("L", "accent b", " temperature log ", "muted",
                  ("● recording " + str(rec)) if rec else "off (CSV every 2 s into bench/thermal/)",
                  "alert b" if rec else "faint"), w)]
    return out


# ------------------------------------------------------------------------------------------------ logs
LOG_STYLE = {"stat": "accent", "spec": "good", "warn": "warn", "error": "alert b", "watchdog": "alert b r",
             "note": "violet", "load": "info", "info": "soft", "server": "muted"}


def logs_view(v: dict, w: int, h: int, ui: dict) -> list:
    log = ui.get("log")
    if log is None:
        return box("LOGS", [L("The engine log is " + LOCAL_ONLY_HINT + ".", "soft"),
                            L("A 1.3.80+ server sends its tail over GET /api/log (the API key is needed); this one "
                              "answers none.", "faint i")], w)
    flt = (ui.get("filter") or "").lower()
    lines = [(k, x) for k, x in log.lines if not flt or flt in x.lower() or flt == k]
    n = max(3, h - 3)
    off = ui.get("scroll", 0)
    view = lines[max(0, len(lines) - n - off):len(lines) - off]
    rows = [L(f"{k[:5]:<6}", LOG_STYLE.get(k, "soft") + " b" if k != "info" else "faint", x,
              LOG_STYLE.get(k, "soft")) for k, x in view]
    tag = L(str(log.path or ("remote /api/log · " + (log.remote_path or "")) if log.path or log.remote_path else "")[-50:],
            "faint")
    title = "ENGINE LOG" + (f" · filter '{flt}'" if flt else "")
    return box(title, rows or [L("(nothing yet)", "faint i")], w, tag=tag)


# ------------------------------------------------------------------------------------------------ connect
def connect_view(v: dict, w: int, h: int, ui: dict) -> list:
    base = (v.get("url") or "http://127.0.0.1:8080").rstrip("/")
    keyed = (ui.get("key_hint") or "")
    rows = [L("web app      ", "muted", base + "/", "accent b"),
            L("OpenAI       ", "muted", base + "/v1", "accent b", "   any model name", "faint"),
            L("Anthropic    ", "muted", base + "/v1/messages", "accent b"),
            L("API key      ", "muted", keyed or "none", "ink" if keyed else "muted"), Line(),
            L("example      ", "muted", f"curl {base}/v1/chat/completions -H 'Content-Type: application/json' "
              + ("-H \"Authorization: Bearer $ABYSS_API_KEY\" " if keyed else "")
              + "-d '{\"model\":\"abyss\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'", "soft"), Line(),
            L("c", "accent b", " copy the API address   ", "muted", "h", "accent b", " check it (/v1/models)   ",
              "muted", "b", "accent b", " open the web app", "muted")]
    if ui.get("check"):
        rows.append(L(ui["check"], "good b" if ui["check"].startswith("ok") else "alert b"))
    rows.append(Line())
    rows.append(L("From another PC: Tailscale, or ", "faint i", "ssh -L 8082:127.0.0.1:8082 host", "accent",
                  " then --url http://127.0.0.1:8082 (Abyss has no TLS: a key over plain http is readable).",
                  "faint i"))
    return box("CONNECT", rows, w)


# ------------------------------------------------------------------------------------------------ models
def models_view(v: dict, w: int, h: int, ui: dict) -> list:
    out = banner_lines(v, w)
    cfgs = ui.get("configs")
    if cfgs is None:
        return out + box("MODELS", [L("The run configs are " + LOCAL_ONLY_HINT + ".", "soft"),
                                    L("A remote server is read-only here.", "faint i")], w)
    sel = ui.get("sel", 0)
    inner = w - 4
    cfgw = min(34, inner - 82) if inner >= 110 else 0     # (wide: the config's file too - names repeat)
    namew = max(16, inner - 46 - (cfgw + 1 if cfgw else 0))
    rows = [L(f"{'':2}{'MODEL':<{namew}} " + (f"{'CONFIG':<{cfgw}} " if cfgw else "") +
              f"{'CTX':>5} {'GPU':>5} {'PORT':>5} {'BACK':>4}  STATE", "muted b")]
    bench = v.get("bench") or {}
    for i, c in enumerate(cfgs):
        name, quant = pretty_name(c.get("display"), c.get("quant"), c.get("model_name") or c["stem"])
        label = f"{name} {quant or ''}".strip()
        state = ("SERVING" if c.get("serving") else "STARTING" if c.get("running") else
                 ("MISSING " + ",".join(c["missing"])) if c.get("missing") else "")
        st = "good b" if state == "SERVING" else "warn b" if state == "STARTING" else "alert" if state else "muted"
        mark = "▶ " if i == sel else "  "
        base = " r" if i == sel else ""
        ln = L(mark, "accent b", f"{label[:namew]:<{namew}} ", ("ink b" + base).strip(),
               f"{c['stem'][:cfgw]:<{cfgw}} " if cfgw else "", ("faint" + base).strip(),
               f"{toks(c.get('ctx')):>5} ", ("muted" + base).strip(),
               f"{','.join(str(g) for g in c.get('gpus') or []):>5} ", ("muted" + base).strip(),
               f"{c.get('port'):>5} ", ("muted" + base).strip(), f"{c.get('backend', '')[:4].upper():>4}  ",
               ("muted" + base).strip(), state, st)
        rows.append(ln)
        if i == sel:
            sub = L("    ", "", c["file"], "faint")
            if bench.get("config") == c["file"]:
                sub.add(f"   bench: decode {f(bench.get('decode'), 1)} tok/s · prompt 8k {f(bench.get('prompt8k'), 0)}"
                        f" · bottleneck {bench.get('bottleneck') or DASH}", "violet")
            rows.append(sub)
    if not cfgs:
        rows.append(L("No run config here yet: ./abyss.sh sets one up.", "soft"))
    rows.append(Line())
    rows.append(L("Enter", "accent b", " start (stops the running one first)   ", "muted", "s", "accent b", " stop   ",
                  "muted", "B", "accent b", " run --bench   ", "muted", "y", "accent b", " copy the bench line", "muted"))
    out += box("MODELS", rows, w, tag=L(f"{len(cfgs)} configs", "muted"))
    return out


def help_lines(w: int) -> list:
    rows = [L("KEYS", "accent b"),
            L("1-8 ←→", "accent b", "  views        ", "muted", "↑↓ PgUp PgDn", "accent b", "  rows / scroll", "muted"),
            L("r", "accent b", "  poll now (also after AUTH REQUIRED)   ", "muted", "space", "accent b",
              "  freeze the screen", "muted"),
            L("b", "accent b", "  open the web app   ", "muted", "y", "accent b", "  copy the --bench line   ", "muted",
              "c", "accent b", "  copy the API address", "muted"),
            L("Enter / s / B", "accent b", "  (Models) start / stop / bench   ", "muted", "L", "accent b",
              "  temperature log", "muted"),
            L("l", "accent b", "  Overview layout: Speed (big numbers) / Devices (every GPU, SSD and the CPU in its own "
              "panel)", "muted"),
            L("/", "accent b", "  (Logs) filter   ", "muted", "h", "accent b", "  (Connect) check the API   ", "muted",
              "t", "accent b", "  theme dark / light / mono   ", "muted", "q", "accent b", "  quit", "muted"), Line(),
            L("NUMBERS", "accent b"),
            L("--", "ink b", "  not reported (missing is never shown as 0). A tile that names a 1.3.80+ engine or server "
              "waits for a counter this one does not send.", "soft"),
            L("live", "good b", " = the last few seconds; ", "soft", "last request", "ink b", " = the previous answer; ",
              "soft", "mean", "ink b", " = since the request began.", "soft"),
            L("TOKEN TIME's source", "ink b", ": 'live · last ~64 tokens' = the engine's STAT window while it writes; "
              "'last answer · engine STAT' = its final numbers; 'engine log' = the per-answer line (older engines).",
              "soft"),
            L("TOKEN TIME", "ink b", ": SSD waits stop the GPU (serial); the lane runs alongside (parallel) - never "
              "added. The verdict uses ./abyss.sh --bench's rules (tools/bench_rules.py).", "soft"),
            L("GPU LOAD", "ink b", " is the busiest model card; the iGPU lane and a helper card have their own rows.",
              "soft")]
    return box("HELP · ? or Esc closes", rows, w)


DRAW = (overview, speed_view, memory_view, requests_view, hardware_view, logs_view, connect_view, models_view)


def screen(v: dict, w: int, h: int, ui: dict) -> list:
    """The whole screen (--plain draws this): header, tabs, the view, footer - h lines of w."""
    tick = ui.get("tick", 0)
    top = header(v, w, tick, ui.get("anim", True)) + [tabs(ui.get("view", 0), w)]
    body_h = max(1, h - len(top) - 1)
    body = help_lines(w) if ui.get("help") else DRAW[ui.get("view", 0)](v, w, body_h, ui)
    body = [fit(x, w) for x in body[ui.get("scroll_body", 0):]][:body_h]
    body += [fit(Line(), w)] * (body_h - len(body))
    return top + body + [footer(v, w, ui.get("frozen"), ui.get("message", ""), bool(ui.get("thermal")),
                                layout=ui.get("layout") if w >= 60 else None)]


# ------------------------------------------------------------------------------------------------ plain output
FLAGS = {"b": "1", "d": "2", "i": "3", "u": "4", "r": "7"}


def ansi(lines: list, theme: str = "dark", color: bool = True) -> str:
    """Lines as text with ANSI colours (24-bit; none with color False / NO_COLOR / theme "mono")."""
    if not color:
        return "\n".join(ln.plain for ln in lines)
    pal = PALETTES.get(theme)                 # (None: "mono" - bold / reverse only, no colours)
    out = []
    for ln in lines:
        s = []
        for text, style in ln:
            role, *flags = (style or "ink").split()
            codes = []
            if pal:
                hexc = pal.get(role, pal["ink"])
                codes.append(f"38;2;{int(hexc[1:3], 16)};{int(hexc[3:5], 16)};{int(hexc[5:7], 16)}")
            codes += [FLAGS[c] for fl in flags for c in fl if c in FLAGS]
            s.append(f"\x1b[{';'.join(codes)}m{text}\x1b[0m" if codes else text)
        out.append("".join(s))
    return "\n".join(out)
