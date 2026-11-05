"""./abyss.sh --check: is the installed model stable and right on this PC?  It starts the model's server the way the
dashboard does (serve/server.py with the installed config), on a free port of this machine only, and runs real work
through its API: long prompts with a needle up to the configured context (the drafter on), a multi-turn session that
reuses the conversation, Japanese, small code tasks run against tests, one tool-call round, and one greedy answer with
and without the drafter.  Each step passes or fails on its own; a crash keeps the engine log's last lines.  Writes
abyss-check.txt (./abyss.sh --report includes it) with no host name, user name or folder path.  Nothing is sent.

The parts that need no GPU (the steps for a context, the graders, the scrubbing, the report) are plain functions,
tested in tools/test_check.py.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

QUICK, DEFAULT, LONG = "quick", "default", "long"
ANSWER_ROOM = 1024        # tokens a long prompt leaves for the question and the answer
LONG_SIZES = {QUICK: (8192, 24576), DEFAULT: (8192, 24576, 32768), LONG: (8192, 24576, 32768, 65536, 131072)}
NEEDLE = "4729-LUMEN"
NEEDLE_LINE = f"(A note for the reader: the access code for the archive room is {NEEDLE}.)"
NEEDLE_Q = ("\n\nQuestion: according to the text above, what is the access code for the archive room? "
            "Reply with the code only.")
THINK_OFF = {"chat_template_kwargs": {"enable_thinking": False}}
SAMPLED = {"temperature": 0.6, "top_p": 0.95, "seed": 1}     # the real-use steps: as people run it, fixed seed

JA_TASKS = [
    ("ja-explain", "AIの「Mixture of Experts（MoE）」とは何か、中学生にもわかるように3文で説明してください。", None),
    ("ja-arith", "りんごは1個120円、みかんは1個45円です。りんごを3個とみかんを7個買って1000円札で払うと、"
                 "おつりはいくらですか。", "325"),
]
CODE_TASKS = [
    ("code-lru", "Write a Python class LRUCache(capacity) with get(key) (returns -1 when missing) and put(key, value), "
                 "evicting the least recently used key when full. O(1) operations.",
     "c=LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3); assert c.get(2)==-1; c.put(4,4)\n"
     "assert c.get(1)==-1; assert c.get(3)==3; assert c.get(4)==4\nc=LRUCache(1); c.put(1,1); c.put(1,5); "
     "assert c.get(1)==5"),
    ("code-intervals", "Write a Python function merge_intervals(xs) that merges overlapping [start, end] intervals "
                       "(touching ones too) and returns them sorted.",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\n"
     "assert merge_intervals([[1,4],[4,5]])==[[1,5]]\nassert merge_intervals([])==[]\n"
     "assert merge_intervals([[5,7],[1,2]])==[[1,2],[5,7]]"),
    ("code-hiragana", "Pythonで、文字列中のカタカナ（全角）をひらがなに変換する関数 to_hiragana(s) を書いてください。"
                      "カタカナ以外の文字（長音記号「ー」を含む）はそのまま残してください。",
     "assert to_hiragana('カタカナ')=='かたかな'\nassert to_hiragana('コーヒーとABC')=='こーひーとABC'\n"
     "assert to_hiragana('漢字ひらがな')=='漢字ひらがな'"),
]
TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "The current weather in a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "the city's name"}},
                   "required": ["city"]}}}
TOOL_RESULT = {"city": "Tokyo", "temp_c": 18, "sky": "light rain"}
# the session: two facts in the first turn, questions about them, a recall and a Japanese summary at the end
FACTS = ("I'm writing a small command-line tool named Hoshizora that backs up photos from a phone to a home server. "
         "Its status page will listen on port 8642. Please keep these two facts in mind.")
TURNS = [
    ("Which file format would you suggest for its settings file, and why? Two sentences.",
     ("json", "yaml", "toml", "ini", "config", "setting")),
    ("How should it detect photos it has already backed up? Two sentences.", ("hash", "checksum", "sha", "md5",
                                                                               "duplicate", "name", "date", "size")),
    ("What should it do when the server is not reachable? Two sentences.", ("retry", "later", "queue", "wait", "log",
                                                                            "offline", "again")),
    ("Suggest three short subcommands for it.", ("backup", "status", "restore", "sync", "list", "init", "config")),
    ("How could the status page show progress? Two sentences.", ("progress", "percent", "%", "bar", "count", "page")),
    ("Which Python standard-library module would you use for the command-line options?", ("argparse",)),
    ("How should it store its log files? Two sentences.", ("log", "file", "rotat", "folder", "directory")),
    ("What is one risk when backing up over Wi-Fi, and how to reduce it? Two sentences.",
     ("interrupt", "drop", "slow", "resume", "partial", "connection", "battery", "encrypt", "secur")),
    ("Remind me: what is the name of my tool and the port of its status page? One sentence.", ("hoshizora", "8642")),
    ("ここまでの話を日本語で2文でまとめてください。", None),
]
CONTEXT_TEXT = ("Some background from the project's notes, for context only:\n\n")


# ------------------------------------------------------------------------------------------------ which steps
def long_sizes(ctx: int, mode: str) -> list:
    """The long prompts' sizes (tokens) for a context: those of the mode that fit with room for the answer, and (not
    --quick) the whole context less that room when it is bigger than the last of them (up to 64K; --long: all of
    it)."""
    sizes = [n for n in LONG_SIZES[mode] if n + ANSWER_ROOM <= ctx]
    if mode != QUICK:
        top = min((ctx - ANSWER_ROOM) // 1024 * 1024, 65536 if mode == DEFAULT else ctx)
        if top > (sizes[-1] if sizes else 0) + 2048:
            sizes.append(top)
    return sizes


def plan(ctx: int, mode: str, tools: bool) -> list:
    """[(step name, what it runs)] in the order they run."""
    steps = [("drafter", "one greedy answer with and without the drafter (same text expected)")]
    ja = JA_TASKS[:1] if mode == QUICK else JA_TASKS
    steps += [(k, "Japanese answer") for k, _, _ in ja]
    code = CODE_TASKS[:1] if mode == QUICK else CODE_TASKS
    steps += [(k, "code, run against tests") for k, _, _ in code]
    if tools:
        steps.append(("tool-call", "one tool call and the answer after its result"))
    steps += [(f"long-{n // 1024}k", f"a {n}-token prompt with a needle (drafter on)") for n in long_sizes(ctx, mode)]
    steps.append(("conversation", f"{len(TURNS) + 1} turns that reuse the conversation"))
    return steps


# ------------------------------------------------------------------------------------------------ graders
def kana(s: str) -> int:
    return len(re.findall(r"[\u3040-\u30ff]", s))


def japanese_share(s: str) -> float:
    """Japanese characters (kana, kanji, Japanese punctuation) over all letters."""
    jp = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff\u3000-\u303f\uff01-\uff60]", s))
    other = len(re.findall(r"[A-Za-z]", s))
    return jp / max(1, jp + other)


def looping(s: str, size: int = 60, times: int = 6) -> bool:
    """True when one stretch of `size` characters comes back `times` times or more (an answer stuck in a loop)."""
    s = re.sub(r"\s+", " ", s)
    if len(s) < size * times:
        return False
    seen = {}
    for i in range(0, len(s) - size):
        k = s[i:i + size]
        seen[k] = seen.get(k, 0) + 1
        if seen[k] >= times:
            return True
    return False


def broken(s: str) -> str:
    """Why an answer looks broken ("" = it does not): empty, the replacement character, a loop."""
    if not s.strip():
        return "empty answer"
    if "\ufffd" in s:
        return "broken characters (U+FFFD) in the answer"
    if looping(s):
        return "the answer repeats itself (a loop)"
    return ""


def grade_japanese(answer: str, want: str | None) -> tuple:
    """(passed, why) for a Japanese answer: not broken, mostly Japanese, the expected text when there is one."""
    why = broken(answer)
    if why:
        return False, why
    if kana(answer) < 10 or japanese_share(answer) < 0.5:
        return False, f"not Japanese enough (kana {kana(answer)}, Japanese {japanese_share(answer):.0%})"
    if want and want not in answer:
        return False, f"the answer lacks {want!r}"
    return True, f"Japanese {japanese_share(answer):.0%}, kana {kana(answer)}"


NO_NET = ("import socket as _s\n"
          "def _no(*a, **k):\n    raise OSError('no network in the check')\n"
          "_s.socket = _no; _s.create_connection = _no\n")


def code_blocks(answer: str) -> str:
    """The answer's Python: its fenced blocks that define something (the last definition winning), else its last
    block."""
    bl = re.findall(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", answer, re.S)
    if not bl:
        return ""
    defs = [b for b in bl if re.search(r"^\s*(def|class) ", b, re.M)] or bl[-1:]
    return "\n\n".join(defs)


def grade_code(answer: str, test: str, timeout: float = 20.0) -> tuple:
    """(passed, why): the answer's code and the test in a temp folder, run with this Python (isolated, no network,
    a timeout)."""
    code = code_blocks(answer)
    if not code:
        return False, "no Python code block in the answer"
    d = tempfile.mkdtemp(prefix="abyss-check-")
    try:
        f = Path(d) / "t.py"
        f.write_text(NO_NET + code + "\n\n" + test + "\nprint('CHECK-PASS')\n", encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            p = subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return False, f"the code ran over {timeout:.0f} s"
        if "CHECK-PASS" in (p.stdout or ""):
            return True, "the tests pass"
        last = ((p.stderr or "").strip().splitlines() or ["no output"])[-1]
        return False, f"the tests fail: {last[:160]}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def grade_tool_call(msg: dict) -> tuple:
    """(passed, why, the call) for the first answer of the tool step: one get_weather call with JSON arguments that
    name Tokyo."""
    calls = msg.get("tool_calls") or []
    if not calls:
        return False, "no tool call (the answer: " + (msg.get("content") or "")[:120].replace("\n", " ") + ")", None
    c = calls[0]
    fn = c.get("function") or {}
    if fn.get("name") != "get_weather":
        return False, f"called {fn.get('name')!r}, not get_weather", c
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except ValueError:
        return False, f"the arguments are not JSON: {str(fn.get('arguments'))[:120]}", c
    if "tokyo" not in str(args.get("city", "")).lower():
        return False, f"the arguments do not name Tokyo: {args}", c
    return True, "get_weather({\"city\": \"Tokyo\"})", c


def grade_turn(answer: str, words) -> tuple:
    """(passed, why) for a conversation turn: not broken, and on topic (one of its words; all of them for the recall;
    Japanese for the summary)."""
    if words is None:
        return grade_japanese(answer, None)
    why = broken(answer)
    if why:
        return False, why
    low = answer.lower()
    if words == ("hoshizora", "8642"):
        miss = [w for w in words if w not in low]
        return (not miss), ("recalled both facts" if not miss else f"forgot {', '.join(miss)}")
    return (any(w in low for w in words)), ("on topic" if any(w in low for w in words) else "off topic")


SPEC_EARLY = 12   # as --bench's drafter check: parting before this many tokens is a fault unless a near-tie


def drafter_compare(a: str, b: str, encode=None) -> tuple:
    """(status, text) for the drafter step's two greedy answers, judged in tokens like --bench's drafter check: the
    same, or parting at token SPEC_EARLY or later (a near-tie in the sums: PASS).  An earlier parting is WARN - the
    server gives no logits to tell a near-tie there, --bench does.  `encode`: the model's tokenizer (the answers are
    its greedy tokens, detokenized: encoding them again gives the same ids); without it, characters / 4."""
    if a == b:
        return "PASS", "the same answer with and without the drafter"
    ta, tb = (encode(a), encode(b)) if encode else (None, None)
    if ta is not None:
        i = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]), min(len(ta), len(tb)))
        n = max(len(ta), len(tb))
    else:
        c = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        i, n = c // 4, max(len(a), len(b)) // 4
    if i >= SPEC_EARLY:
        return "PASS", f"the same first {i} of {n} tokens with and without the drafter (a near-tie later on: fine)"
    return "WARN", (f"parted at token {i}: this early it can be a near-tie or a drafter fault - {ME()} --bench "
                    "checks the two tokens' scores")


def ME() -> str:
    return "START-ABYSS.bat" if os.name == "nt" else "./abyss.sh"


# ------------------------------------------------------------------------------------------------ privacy
def scrub(text: str, paths=(), names=()) -> str:
    """The text without folder paths, the user's name or this machine's name: the given folders (longest first) and
    the home folder become <...>, then any other absolute path keeps its last part only (/a/b/c.gguf -> .../c.gguf),
    then the names (3+ characters, whole words) become <user> / <host>."""
    home = str(Path.home())
    subs = sorted({(str(p), tag) for p, tag in paths if p and len(str(p)) > 3}, key=lambda x: -len(x[0]))
    for p, tag in subs:
        text = text.replace(p, tag)
    if len(home) > 3:
        text = text.replace(home, "~")
    # other absolute paths: POSIX /x/y/..., Windows C:\x\y\... (UNC too)
    text = re.sub(r"(?<![\w.~<>/-])~?/(?:[^\s/:'\"`<>|,;()\[\]]+/)+", ".../", text)
    text = re.sub(r"(?:\b[A-Za-z]:|\\\\[^\s\\]+)\\(?:[^\s\\:'\"<>|]+\\)+", "...\\\\", text)
    for n, tag in names:
        if n and len(n) >= 3:
            text = re.sub(rf"(?<![\w]){re.escape(n)}(?![\w])", tag, text, flags=re.I)
    return text


def identity() -> list:
    """[(name, tag)] of the user's and this machine's names, for scrub."""
    out = []
    for v in (os.environ.get("USER"), os.environ.get("USERNAME"), os.environ.get("LOGNAME"), Path.home().name):
        if v:
            out.append((v, "<user>"))
    for v in (socket.gethostname(), platform.node(), os.environ.get("COMPUTERNAME"), os.environ.get("HOSTNAME")):
        if v:
            out.append((v.split(".")[0], "<host>"))
            out.append((v, "<host>"))
    return out


# ------------------------------------------------------------------------------------------------ the report
STATUS_ORDER = {"FAIL": 0, "WARN": 1, "PASS": 2, "SKIP": 3, "NOT RUN": 4}


def overall(results) -> str:
    """FAIL when a step failed or did not run (the engine stopped), WARN when one warned, else PASS."""
    st = {r["status"] for r in results}
    if "FAIL" in st or "NOT RUN" in st:
        return "FAIL"
    return "WARN" if "WARN" in st else "PASS"


def fmt(v, spec=".1f", none="-") -> str:
    return none if v is None else format(v, spec)


def render(head: dict, results: list, limits: list, failures: list, answers: list, engine_lines: list) -> str:
    """abyss-check.txt's text (not scrubbed yet): the summary line, this PC, one row a step, what limits this PC,
    the failures with the engine log's last lines, and the answers' openings."""
    res = overall(results)
    n_ok = sum(r["status"] in ("PASS", "SKIP") for r in results)
    dec = [r["decode"] for r in results if r.get("decode")]
    pre = {r["name"]: r["prefill"] for r in results if r.get("prefill") and r["name"].startswith("long-")}
    link = "memory unified" if head.get("unified") else f"PCIe {head['pcie']}"
    summary = (f"Abyss v{head['version']} | {head['model']} | {head['gpus']} | {link} | RAM "
               f"{head['ram']} | check: {res} ({n_ok} of {len(results)} steps)"
               + (f" | decode {sorted(dec)[len(dec) // 2]:.1f} tok/s (median)" if dec else "")
               + "".join(f" | prompt {k[5:]} {v:.0f} tok/s" for k, v in list(pre.items())[:2]))
    lines = ["### Abyss check", "", summary, "",
             f"- date: {head['date']}; mode: {head['mode']}; setup: {head['setup']}, context {head['context']}, "
             f"backend {head['backend']}, drafter {head['drafter']}",
             f"- model: {head['model']} ({head['model_detail']})",
             f"- GPUs: {head['gpus']}; driver {head['driver']}",
             ("- memory: unified (the GPU and the CPU share the RAM; no PCIe link)" if head.get("unified") else
              f"- PCIe while working: {head['pcie']}"),
             f"- CPU {head['cpu']}; RAM {head['ram']}; OS {head['os']}; model SSD reads {head['ssd']}",
             f"- Abyss v{head['version']}, commit {head['commit']}; loaded in {head['load_s']}",
             "",
             "| step | result | time | decode tok/s | prompt tok/s | reused | VRAM hit | note |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['status']} | {fmt(r.get('secs'), '.0f')} s | {fmt(r.get('decode'))} | "
                     f"{fmt(r.get('prefill'), '.0f')} | {fmt(r.get('reused'), 'd')} | "
                     f"{fmt(r.get('vram_hit'), '.0f')}{'%' if r.get('vram_hit') is not None else ''} | "
                     f"{r.get('note', '').replace('|', '/')} |")
    if limits:
        lines += ["", "#### What limits this PC (estimates from this run's measurements)", ""] + limits
    for f in failures:
        lines += ["", f"#### {f['step']}: {f['status']}", "", f"- {f['why']}"]
        if f.get("log"):
            lines += ["", "the engine log's last lines:", "", "```"] + f["log"] + ["```"]
        if f.get("server"):
            lines += ["", "the server's last lines:", "", "```"] + f["server"] + ["```"]
    if answers:
        lines += ["", "<details><summary>the answers' openings</summary>", ""]
        lines += [f"- {k}: {a[:240]!r}" for k, a in answers]
        lines += ["", "</details>"]
    if engine_lines:
        lines += ["", "<details><summary>engine lines</summary>", "", "```"] + engine_lines[-40:] + ["```", "",
                                                                                                    "</details>"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------ the run
class Server:
    """serve/server.py on a free port of this machine, with a copy of the config whose log is abyss-check.log."""

    def __init__(self, root: Path, cfg: dict, env: dict, out_log: Path):
        self.root, self.cfg = root, cfg
        self.dir = Path(tempfile.mkdtemp(prefix="abyss-check-"))
        self.cfg_path = self.dir / "check.json"
        self.cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.out = open(out_log, "w", encoding="utf-8")
        kw = {"start_new_session": True} if os.name != "nt" else \
            {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
        self.p = subprocess.Popen([sys.executable, str(root / "serve" / "server.py"), "--engine", "strata", "--config",
                                   str(self.cfg_path), "--port", str(self.port), "--host", "127.0.0.1"],
                                  stdout=self.out, stderr=subprocess.STDOUT, env=env, cwd=str(root), **kw)

    def alive(self) -> bool:
        return self.p.poll() is None

    def wait_ready(self, timeout: float) -> str:
        """"" once /health answers 200; else why not (the server ended, or the time ran out)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not self.alive():
                return f"the server ended while loading (exit code {self.p.returncode})"
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return ""
            except (OSError, ValueError):
                pass
            time.sleep(2)
        return f"the model did not load within {timeout / 60:.0f} minutes"

    def chat(self, body: dict, timeout: float = 3600) -> dict:
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"HTTP {e.code}: {body}") from None

    def stop(self):
        if self.alive():
            try:
                if os.name != "nt":
                    os.killpg(self.p.pid, 15)
                else:
                    self.p.terminate()
                self.p.wait(timeout=90)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    if os.name != "nt":
                        os.killpg(self.p.pid, 9)
                    else:
                        self.p.kill()
                except OSError:
                    pass
        self.out.close()
        shutil.rmtree(self.dir, ignore_errors=True)


class LogTail:
    """The engine log's lines written since the last mark."""

    def __init__(self, path: Path):
        self.path, self.pos = path, 0

    def mark(self):
        try:
            self.pos = self.path.stat().st_size
        except OSError:
            self.pos = 0

    def since(self) -> list:
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                return f.read().decode("utf-8", errors="replace").splitlines()
        except OSError:
            return []

    def last(self, n: int = 40) -> list:
        try:
            with open(self.path, "rb") as f:
                f.seek(max(0, f.seek(0, os.SEEK_END) - 64 * 1024))
                lines = f.read().decode("utf-8", errors="replace").splitlines()
        except OSError:
            return []
        return [x for x in lines if not x.startswith("T ")][-n:]


def server_tail(path: Path, n: int = 12) -> list:
    """The server's last lines (its own messages: an engine that stopped, a request it refused)."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def filler(root: Path) -> str:
    """The long prompts' text: this folder's English docs (the .ja ones and the CHANGELOG left out)."""
    files = [root / "README.md"] + sorted(p for p in (root / "docs").rglob("*.md") if not p.name.endswith(".ja.md"))
    return "\n\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files if p.exists())


def needle_prompt(text: str, n_chars: int, offset: int, depth: float) -> str:
    """`n_chars` of `text` (repeated as needed) from `offset`, the needle at `depth` of it, at a paragraph break."""
    if not text:
        text = "The quick brown fox jumps over the lazy dog. " * 100
    body = (text * (n_chars // max(1, len(text)) + 2))[offset % len(text):][:n_chars]
    at = int(len(body) * depth)
    br = body.rfind("\n\n", 0, at)
    at = br if br > 0 else at
    return body[:at] + "\n\n" + NEEDLE_LINE + "\n\n" + body[at:] + NEEDLE_Q


def run(root: Path, cfg_path: Path, cfg: dict, mode: str, version: str, hooks: dict) -> int:
    """The check (see the module's note).  `hooks`: what abyss.py provides - say / ok / warn, the GPU list and link
    watch, the SSD read rate, the model's name, the speed lines, mem_gb, cpu name."""
    say, ok, warn = hooks["say"], hooks["ok"], hooks["warn"]
    import bench_rules as BR
    import pc_limits as PL
    args = cfg.get("args") or []
    ctx = int(args[args.index("--max-context") + 1]) if "--max-context" in args[:-1] else 32768
    pack = Path(args[args.index("--glm-pack") + 1]) if "--glm-pack" in args[:-1] else None
    tpl = Path(cfg.get("tokenizer") or (pack / "tokenizer" if pack else "")) / "chat_template.jinja"
    has_tools = (not tpl.exists()) or "tool" in tpl.read_text(encoding="utf-8", errors="replace")
    steps = plan(ctx, mode, has_tools)
    log_path = root / "abyss-check.log"
    out_log = root / "abyss-check-server.log"
    for p in (log_path, out_log):
        try:
            p.unlink()
        except OSError:
            pass
    route_dir = Path(tempfile.mkdtemp(prefix="abyss-check-routes-"))
    ccfg = dict(cfg, log=str(log_path))
    ccfg.setdefault("cwd", str(root))
    ccfg.pop("api_key", None)
    ccfg.pop("mcp_servers", None)
    env = dict(os.environ, STRATA_GLM_GAP_PROF="1", STRATA_GLM_ROUTE_LOG=str(route_dir / "route"), PYTHONUNBUFFERED="1",
               ABYSS_NO_BROWSER="1")
    env.pop("STRATA_API_KEY", None)
    say(f"  {len(steps)} steps ({mode}): " + ", ".join(s for s, _ in steps))
    say("  loading the model first (a minute or a few) ...")
    t_load = time.time()
    srv = Server(root, ccfg, env, out_log)
    tail = LogTail(log_path)
    results, failures, answers = [], [], []
    watch = hooks["links"](cfg)
    plain_rec, load_s = None, None
    try:
        why = srv.wait_ready(3600)
        load_s = time.time() - t_load
        if why:
            failures.append({"step": "load", "status": "FAIL", "why": why, "log": tail.last(),
                             "server": server_tail(out_log)})
            results.append({"name": "load", "status": "FAIL", "secs": load_s, "note": why})
            for s, _ in steps:
                results.append({"name": s, "status": "NOT RUN", "note": "the model did not load"})
        else:
            ok(f"loaded in {load_s:.0f} s")
            stopped = False
            tail.mark()
            try:                                                             # warm-up: the first answer is slower
                srv.chat(dict(messages=[{"role": "user", "content": "Say hello in five languages."}], max_tokens=64,
                              temperature=0, **THINK_OFF))
            except Exception as e:
                failures.append({"step": "warm-up", "status": "FAIL", "why": str(e)[:300], "log": tail.last(),
                                 "server": server_tail(out_log)})
                results.append({"name": "warm-up", "status": "FAIL", "note": str(e)[:200]})
                stopped = not srv.alive() or bool(srv.wait_ready(900))
            text = filler(root)
            ratio = max(2.0, min(6.0, hooks["chars_per_token"](text[:200000])))
            for i, (name, _) in enumerate(steps):
                if stopped:
                    results.append({"name": name, "status": "NOT RUN", "note": "the engine stopped before it"})
                    continue
                tail.mark()
                t0 = time.time()
                r = {"name": name}
                try:
                    st, note, extra = step(srv, name, ctx, text, ratio, i, hooks.get("count_tokens"),
                                           hooks.get("encode"))
                    r.update(status=st, note=note, **{k: v for k, v in extra.items() if k != "answer"})
                    if extra.get("answer") is not None:
                        answers.append((name, extra["answer"]))
                    if extra.get("plain"):
                        recs = BR.bench_records([x for x in tail.since() if "glm stat" in x or "glm gap" in x
                                                 or "glm spec" in x])
                        plain_rec = recs[-1] if recs else None
                except Exception as e:                   # (a crash, an HTTP error, a timeout: the step failed)
                    r.update(status="FAIL", note=str(e)[:200])
                r["secs"] = time.time() - t0
                recs = BR.bench_records(tail.since())
                hits = [x["vram_hit"] for x in recs if x.get("ms", 0) >= 1.0]
                if hits:
                    r["vram_hit"] = sum(hits) / len(hits)
                results.append(r)
                mark = {"PASS": ok, "SKIP": say}.get(r["status"], warn)
                mark(f"{name}: {r['status']} - {r.get('note', '')}" + (f", {r['decode']:.1f} tok/s" if r.get("decode")
                                                                       else ""))
                if r["status"] == "FAIL":
                    gone = not srv.alive() or any("engine stopped" in x or "exit code" in x for x in tail.since())
                    failures.append({"step": name, "status": "FAIL", "why": r.get("note", ""),
                                     "log": tail.last(), "server": server_tail(out_log)})
                    if not srv.alive():
                        stopped = True
                    elif gone and srv.wait_ready(900):
                        stopped = True
    except KeyboardInterrupt:
        warn("stopped by Ctrl+C: the report has the steps that ran")
        results.append({"name": "(stopped)", "status": "NOT RUN", "note": "Ctrl+C"})
    finally:
        links = watch.stop()
        chains = dict(getattr(watch, "best", {}) or {})
        srv.stop()
    log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines() if log_path.exists() else []
    facts = PL.engine_facts(log_lines, pack)
    routes = PL.read_routes(sorted(route_dir.glob("route*")), facts.get("layers") or 0)
    shutil.rmtree(route_dir, ignore_errors=True)
    gpus = hooks["gpus"](cfg)
    uma = bool(gpus) and all(g.get("integrated") for g in gpus)   # (DGX Spark / GB10: no PCIe link to its memory)
    disk = hooks["disk"](pack)
    vram = (cfg.get("env") or {}).get("STRATA_GLM_VRAM_GB") or os.environ.get("STRATA_GLM_VRAM_GB")
    try:                                                 # (a VRAM cap in the config: the size the model runs in)
        vram = float(vram) if vram else (sum(g["vram_gb"] for g in gpus) if gpus else None)
    except ValueError:
        vram = sum(g["vram_gb"] for g in gpus) if gpus else None
    limits = PL.what_limits(plain_rec, facts, None if uma else next(iter(chains.values()), None), disk, routes, vram,
                            BR.lane_name(log_lines), unified=uma) if plain_rec else \
        ["(not split: the drafter step did not run to the end)"]
    total, _ = hooks["mem_gb"]()
    drafter = next((r.get("drafter") for r in results if r.get("drafter")), "?")
    head = {"version": version, "model": hooks["model"](pack, cfg, cfg_path)[0],
            "model_detail": hooks["model"](pack, cfg, cfg_path)[1], "gpus": ", ".join(
                f"{g['name']} {g['vram_gb']:.0f} GB" + (" unified" if g.get("integrated") else "") for g in gpus) or "?",
            "driver": hooks["driver"](cfg), "unified": uma,
            "pcie": "none (unified memory)" if uma else "; ".join(f"{k} {v}" for k, v in links.items()) or "?",
            "ram": f"{total:.0f} GB", "cpu": hooks["cpu"](), "os": hooks["os"](),
            "ssd": f"{disk:.1f} GB/s" if disk else "? (not measured)", "date": time.strftime("%Y-%m-%d %H:%M"),
            "mode": mode, "setup": cfg_path.name, "context": ctx, "backend": cfg.get("backend", "cuda"),
            "drafter": drafter, "commit": hooks["commit"](),
            "load_s": f"{load_s:.0f} s" if load_s is not None else "?"}
    text = render(head, results, limits, failures, answers, hooks["speed_lines"](log_lines))
    paths = [(root, "<abyss>"), (cfg_path.parent, "<abyss>")]
    if pack is not None:
        paths += [(pack, "<model>/" + pack.name), (pack.parent, "<model>"), (pack.parent.parent, "<models>")]
    for k in ("exe", "tokenizer", "cwd"):
        if cfg.get(k):
            paths.append((Path(cfg[k]).parent, "<" + k + ">"))
    text = scrub(text, paths, identity())
    out = root / "abyss-check.txt"
    out.write_text(text, encoding="utf-8")
    say()
    say("  " + text.splitlines()[2])
    (warn if overall(results) == "FAIL" else ok)(
        f"written: {out} - {ME()} --report adds it to abyss-report.txt for an issue "
        "(https://github.com/zephel01/Abyss/issues/new?template=bug.md)")
    return 1 if overall(results) == "FAIL" else 0     # (a WARN is a note, not a fault)


def timing(j: dict) -> dict:
    """decode / prompt tok/s, the reused tokens and the drafts of a response."""
    t = j.get("timings") or {}
    return {"decode": t.get("predicted_per_second"), "prefill": t.get("prompt_per_second"),
            "reused": t.get("cache_n"), "drafts": (t.get("draft_n"), t.get("draft_n_accepted"))}


def message(j: dict) -> dict:
    return ((j.get("choices") or [{}])[0].get("message")) or {}


def step(srv: Server, name: str, ctx: int, text: str, ratio: float, i: int, count=None, encode=None) -> tuple:
    """(status, note, extra) of one step; `count`: the model's tokens in a text (the long prompts' size)."""
    if name == "drafter":
        q = "Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring."
        base = dict(messages=[{"role": "user", "content": q}], max_tokens=256, temperature=0, **THINK_OFF)
        on = srv.chat(base)
        off = srv.chat(dict(base, strata_tune={"spec": 0}))
        a, b = message(on).get("content") or "", message(off).get("content") or ""
        st, note = drafter_compare(a, b, encode)
        t_on, t_off = timing(on), timing(off)
        dn, da = t_on["drafts"]
        drafter = (f"on ({da} of {dn} drafts accepted)" if dn else "off (no drafts offered)")
        if b and broken(b):
            st, note = "FAIL", "the answer without the drafter: " + broken(b)
        elif a and broken(a):
            st, note = "FAIL", "the answer with the drafter: " + broken(a)
        if t_on["decode"] and t_off["decode"]:
            note += f"; {t_off['decode']:.1f} tok/s without, {t_on['decode']:.1f} with"
        return st, note, {"decode": t_on["decode"], "answer": a, "plain": True, "drafter": drafter}
    for k, q, want in JA_TASKS:
        if name == k:
            j = srv.chat(dict(messages=[{"role": "user", "content": q}], max_tokens=4096, **SAMPLED))
            a = message(j).get("content") or ""
            okk, why = grade_japanese(a, want)
            if okk and (j.get("choices") or [{}])[0].get("finish_reason") == "length":
                return "WARN", why + "; cut at the token limit", dict(timing(j), answer=a)
            return ("PASS" if okk else "FAIL"), why, dict(timing(j), answer=a)
    for k, q, test in CODE_TASKS:
        if name == k:
            j = srv.chat(dict(messages=[{"role": "user", "content": q}], max_tokens=6144, **SAMPLED))
            a = message(j).get("content") or ""
            why = broken(a)
            okk, note = (False, why) if why else grade_code(a, test)
            return ("PASS" if okk else "FAIL"), note, dict(timing(j), answer=a)
    if name == "tool-call":
        msgs = [{"role": "user", "content": "What is the weather in Tokyo right now? Use the get_weather tool."}]
        j = srv.chat(dict(messages=msgs, tools=[TOOL], max_tokens=4096, **SAMPLED))
        okk, why, call = grade_tool_call(message(j))
        if not okk:
            return "FAIL", why, dict(timing(j), answer=json.dumps(message(j), ensure_ascii=False))
        msgs += [{"role": "assistant", "content": message(j).get("content") or "", "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": call.get("id", "call_0"), "content": json.dumps(TOOL_RESULT)}]
        j2 = srv.chat(dict(messages=msgs, tools=[TOOL], max_tokens=4096, **SAMPLED))
        a = message(j2).get("content") or ""
        if broken(a) or "18" not in a:
            return "FAIL", f"{why}, then the answer after the result " + (broken(a) or "lacks the 18 degrees"), \
                dict(timing(j2), answer=a)
        return "PASS", f"{why}, then an answer with the result", dict(timing(j2), answer=a)
    if name.startswith("long-"):
        n = int(name[5:-1]) * 1024
        depth = (0.3, 0.5, 0.7)[i % 3]
        chars = int(n * ratio * 0.97)
        prompt = needle_prompt(text, chars, i * 7919, depth)
        if count is not None:                             # (this stretch of the docs: its own characters a token)
            got = count(prompt)
            if got > 0:
                prompt = needle_prompt(text, int(chars * n * 0.97 / got), i * 7919, depth)
        j = srv.chat(dict(messages=[{"role": "user", "content": prompt}], max_tokens=128, temperature=0, **THINK_OFF),
                     timeout=7200)
        a = message(j).get("content") or ""
        t = timing(j)
        got = int((j.get("usage") or {}).get("prompt_tokens") or 0)
        note = f"{got} prompt tokens, needle at {depth:.0%}: "
        if NEEDLE in a:
            return "PASS", note + "found", dict(t, answer=a)
        return "FAIL", note + ("not found" if a.strip() else "empty answer"), dict(t, answer=a)
    if name == "conversation":
        intro = FACTS + "\n\n" + CONTEXT_TEXT + text[:6000]
        msgs = [{"role": "user", "content": intro + "\n\nReply with one short sentence."}]
        bad, reuse, prompt_speeds, decs, last = [], [], [], [], ""
        for k, (q, words) in enumerate([("", ("hoshizora", "photo", "8642", "backup", "back up", "noted", "got it",
                                              "understood", "keep"))] + TURNS):
            if k:
                msgs.append({"role": "user", "content": q})
            j = srv.chat(dict(messages=msgs, max_tokens=1500, **SAMPLED))
            a = message(j).get("content") or ""
            t = timing(j)
            if k:
                reuse.append(t["reused"] or 0)
            if t["decode"]:
                decs.append(t["decode"])
            okk, why = grade_turn(a, words)
            if not okk:
                bad.append(f"turn {k + 1}: {why}")
            msgs.append({"role": "assistant", "content": a})
            last = a
        hits = sum(1 for x in reuse if x > 0)
        note = (f"{len(msgs) // 2} turns, conversation reused on {hits} of {len(reuse)} later turns"
                + (f" (mean {sum(reuse) / max(1, hits):.0f} tokens)" if hits else ""))
        extra = {"decode": sorted(decs)[len(decs) // 2] if decs else None, "reused": max(reuse) if reuse else None,
                 "answer": last}
        if bad:
            return "FAIL", note + "; " + "; ".join(bad[:3]), extra
        if hits < len(reuse) / 2:
            return "WARN", note + " - expected on most turns", extra
        return "PASS", note, extra
    return "SKIP", "unknown step", {}
