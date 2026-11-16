"""serve/lang_ban.py - the answer's language profile and OpenAI logit_bias, as the engine's bias sets.

A request names a profile with `"lang": "ja-en"` (STRATA_LANG: the server's default; "off" / "" none) - the scripts
the answer may use (tools/token_classes.py: PROFILES, and the per-tokenizer class tables in data/token_classes/).
Every token of another script gets -inf at the sampler: a Japanese answer cannot pick up a Chinese word.  The profile
applies to the ANSWER by default; `"lang_think": true` (STRATA_LANG_THINK=1) applies it to the reasoning as well.
`logit_bias` ({"<token id>": -100..100}, OpenAI's; -100 bans) applies to the whole output.

Special tokens (<think>, <tool_call>, the turn's end, ...) and the server's stop / think / argument ids are never
banned, by a profile or a logit_bias.

The engine gets each set once (`BIAS <name> <id:value,...>`, the name a hash of its contents) and a request names
them on its GEN line: `bias=<whole output>` and `bias_answer=<from the end of the reasoning on>`.
"""
from __future__ import annotations

import hashlib
import os
import sys
import threading
from pathlib import Path

_TOOLS = str(Path(__file__).resolve().parents[1] / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
import token_classes as TC  # noqa: E402

OFF = ("", "off", "none", "0")


def default_profile() -> str:
    return os.environ.get("STRATA_LANG", "").strip()


def default_think() -> bool:
    return os.environ.get("STRATA_LANG_THINK", "0").strip() not in OFF


def profile_names() -> list[str]:
    return sorted(TC.PROFILES)


def check_request(req: dict, n_vocab: int) -> None:
    """ValueError (-> 400) for a malformed lang / lang_think / logit_bias."""
    lang = req.get("lang")
    if lang is not None:
        if not isinstance(lang, str) or (lang.strip() not in OFF and lang.strip() not in TC.PROFILES):
            raise ValueError(f"lang: one of {', '.join(profile_names())}, or \"off\"")
    lt = req.get("lang_think")
    if lt is not None and not isinstance(lt, bool):
        raise ValueError("lang_think: true or false")
    lb = req.get("logit_bias")
    if lb is not None:
        if not isinstance(lb, dict):
            raise ValueError("logit_bias: an object of token id -> bias (-100..100)")
        for k, v in lb.items():
            try:
                i = int(k)
            except (TypeError, ValueError):
                raise ValueError(f"logit_bias: {k!r} is not a token id") from None
            if not 0 <= i < n_vocab:
                raise ValueError(f"logit_bias: token id {i} outside the vocabulary (0..{n_vocab - 1})")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not -100 <= v <= 100:
                raise ValueError(f"logit_bias[{k}]: a number -100..100")


class LangBan:
    """One tokenizer's class table and the banned ids per profile (computed once each)."""

    def __init__(self, tokenizer, keep_ids=()):
        self.tok = tokenizer
        self.keep = set(int(i) for i in keep_ids if i is not None)
        self._table = None
        self.table_source = None
        self._ids: dict[str, list[int]] = {}
        self._lock = threading.Lock()

    @property
    def n_vocab(self) -> int:
        return len(self.tok.tokens)

    def table(self):
        if self._table is None:
            self._table, self.table_source = TC.load_table(self.tok)
        return self._table

    def banned(self, profile: str) -> list[int]:
        with self._lock:
            got = self._ids.get(profile)
            if got is None:
                got = self._ids[profile] = TC.banned_ids(self.tok, self.table(), profile, keep=self.keep)
            return got

    def never_banned(self, i: int) -> bool:
        return i in self.keep or TC.is_special(self.tok, i)

    def sets(self, req: dict) -> tuple[dict | None, dict | None]:
        """(whole output, from the answer on): {id: value} each (value None = banned), None when empty."""
        lang = req.get("lang")
        lang = default_profile() if lang is None else lang.strip()
        lang = "" if lang in OFF else lang
        if lang and lang not in TC.PROFILES:
            raise ValueError(f"lang {lang!r} (STRATA_LANG?): one of {', '.join(profile_names())}, or off")
        think = req.get("lang_think")
        think = default_think() if think is None else think
        bias: dict[int, float | None] = {}
        for k, v in (req.get("logit_bias") or {}).items():
            i, v = int(k), float(v)
            if v == 0 or (v <= -100 and self.never_banned(i)):
                continue
            bias[i] = None if v <= -100 else v
        prof = {i: None for i in self.banned(lang)} if lang else {}
        whole = {**bias, **prof} if think else bias
        answer = None if think or not prof else {**bias, **prof}
        return whole or None, answer


def spec(s: dict) -> str:
    return ",".join(f"{i}:{'ban' if v is None else repr(float(v))}" for i, v in sorted(s.items()))


def set_name(sp: str) -> str:
    return "b" + hashlib.sha1(sp.encode()).hexdigest()[:16]


def register(engine, s: dict | None) -> str | None:
    """The engine's name for set `s`, sent to it first if this engine process does not have it yet (the caller holds
    the service's FIFO: no request is reading the engine's lines)."""
    if not s:
        return None
    sp = spec(s)
    name = set_name(sp)
    proc = getattr(engine, "proc", None)
    pid = getattr(proc, "pid", None)
    reg = getattr(engine, "_bias_sets", None)
    if reg is None or reg[0] != pid:
        reg = (pid, set())
        engine._bias_sets = reg
    if name not in reg[1]:
        cmd = getattr(engine, "command", None)
        if cmd is None:
            raise ValueError("this engine does not take logit biases (lang / logit_bias)")
        try:
            cmd(f"BIAS {name} {sp}", "BIASOK", timeout=120.0)
        except ValueError as e:
            raise ValueError(f"the engine refused the logit bias set: {e} "
                             "(an engine older than the language profiles has no BIAS)") from None
        reg[1].add(name)
    return name
