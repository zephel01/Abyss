#!/usr/bin/env python3
"""./abyss.sh --connect <tool>: the settings that point a coding tool at the Abyss server that runs now.

    ./abyss.sh --connect claude              print how to start Claude Code against this server
    ./abyss.sh --connect opencode            print the OpenCode provider block for this server
    ./abyss.sh --connect <tool> --write      write them (a launcher script / opencode.json), asked first, a backup kept
    ./abyss.sh --connect <tool> --check      send one prompt and one tool call the way the tool does, and say what came back
    ./abyss.sh --connect <tool> --url http://192.168.1.20:8082 --key-env ABYSS_KEY   a server on another PC

The values come from the server itself (GET /v1/models: the model's name and its context) and from its run config (the
address, the API key), so they match what runs.  Nothing is written without --write, and --write changes no global
setting of the tool: Claude Code gets a launcher script next to this folder's configs (run-claude-<config>.sh) that
sets its variables for that run only; OpenCode gets a provider "abyss" merged into its config file, the old file kept
as <file>.bak-<time>.  The API key is never written into a file: the launcher and the provider read it from an
environment variable (ABYSS_API_KEY).  Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import terminal_local as T  # noqa: E402

ROOT = T.ROOT
WIN = T.WIN
ME = T.ME
KEY_VAR = "ABYSS_API_KEY"
TOOLS = ("claude", "opencode")
# Claude Code's own prompt (its system prompt, the tool list) is ~20K tokens before the first word of a task; below
# this the conversation is compacted almost at once
CLAUDE_MIN_CTX = 32768


# ------------------------------------------------------------------------------------------------ the server
def _get(url: str, key: str | None, timeout: float = 5.0):
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace") or "null")


def _post(url: str, body: dict, key: str | None, headers: dict | None = None, timeout: float = 600.0):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def find_server(url: str | None, key_env: str | None, config: Path | None) -> dict:
    """{"url", "key", "model", "ctx", "config", "remote"} of the server to connect to, or {"error": ...}."""
    if url:
        base = url.rstrip("/")
        if "://" not in base:
            base = "http://" + base
        key = os.environ.get(key_env) if key_env else None
        if key_env and not key:
            return {"error": f"${key_env} is empty: set it to the server's API key (or leave --key-env out)"}
        cands = [{"url": base, "key": key, "config": None, "remote": True}]
    else:
        paths = [Path(config).resolve()] if config else T.list_configs(ROOT)
        if not paths:
            return {"error": f"no run config here: {ME} sets one up"}
        # a server started by the Terminal has a PID file; one started by ./abyss.sh or run-*.sh does not, so every
        # config's address is asked (the most recently used first)
        running = [c for c, _ in T.running(ROOT)] if not config else []
        order = running + [p for p in paths if p not in running]
        cands = []
        for p in order:
            info = T.config_info(p)
            cands.append({"url": info["url"], "key": info.get("key"), "config": p, "remote": False,
                          "cfg_ctx": info.get("ctx"), "cfg_model": info.get("model_name")})
    last = None
    for c in cands:
        try:
            models = _get(c["url"] + "/v1/models", c["key"], timeout=3.0)
        except urllib.error.HTTPError as e:
            last = (f"{c['url']} answered {e.code}" +
                    (": the API key is missing or wrong" + ("" if c["remote"] else " (the config's api_key)") if e.code == 401 else ""))
            continue
        except (OSError, ValueError) as e:
            last = f"{c['url']}: {getattr(e, 'reason', e)}"
            continue
        data = (models or {}).get("data") or []
        if not data:
            last = f"{c['url']}: the server is up but no model is loaded yet (still starting?)"
            continue
        m = data[0]
        c["model"] = m.get("id") or c.get("cfg_model") or "abyss"
        c["ctx"] = int(((m.get("meta") or {}).get("n_ctx")) or c.get("cfg_ctx") or 0) or None
        return c
    if last and " 401" in last:
        return {"error": last + (f" - pass --key-env <variable holding the key>" if url else "")}
    where = "that address" if url else ("that config's address" if config else "any installed config's address")
    return {"error": f"no Abyss server answers at {where}" + (f" ({last})" if last else "") +
                     f" - start it first ({ME}, or the Terminal's Models view)"}


# ------------------------------------------------------------------------------------------------ Claude Code
def claude_env(s: dict) -> dict:
    """The variables Claude Code needs for this server (the key as a reference to $ABYSS_API_KEY)."""
    env = {
        "ANTHROPIC_BASE_URL": s["url"],
        # every model slot to the one local model: the main one and the small one for background calls
        "ANTHROPIC_MODEL": s["model"],
        "ANTHROPIC_DEFAULT_OPUS_MODEL": s["model"],
        "ANTHROPIC_DEFAULT_SONNET_MODEL": s["model"],
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": s["model"],      # (also the background calls)
        "CLAUDE_CODE_SUBAGENT_MODEL": s["model"],
        # no anthropic-beta headers and beta-only body fields: a local server has no use for them
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
    if s.get("ctx"):
        # Claude Code assumes 200K for a model it does not know: the real context, so it compacts in time
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(s["ctx"])
        # the answer's room: a quarter of the context, at most 32K (the server keeps 1K+ for the answer anyway)
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max(4096, min(32000, s["ctx"] // 4)))
    return env


def claude_launcher(s: dict) -> tuple[Path, str]:
    stem = s["config"].stem if s.get("config") else "remote"
    env = claude_env(s)
    if WIN:
        path = ROOT / f"run-claude-{stem}.bat"
        lines = ["@echo off", "rem Claude Code against the Abyss server (./abyss.sh --connect claude --write)"]
        lines += [f"set {k}={v}" for k, v in env.items()]
        lines += [f"if defined {KEY_VAR} set ANTHROPIC_AUTH_TOKEN=%{KEY_VAR}%",
                  "if not defined ANTHROPIC_AUTH_TOKEN set ANTHROPIC_AUTH_TOKEN=abyss", "claude %*"]
        return path, "\r\n".join(lines) + "\r\n"
    path = ROOT / f"run-claude-{stem}.sh"
    lines = ["#!/bin/sh", "# Claude Code against the Abyss server (./abyss.sh --connect claude --write); these settings",
             "# hold for this run only - your ~/.claude settings are not changed"]
    lines += [f"export {k}='{v}'" for k, v in env.items()]
    lines += [f"# the server's API key, if it has one: export {KEY_VAR}=... before running this",
              f"export ANTHROPIC_AUTH_TOKEN=\"${{{KEY_VAR}:-abyss}}\"", "unset ANTHROPIC_API_KEY",
              'exec claude "$@"']
    return path, "\n".join(lines) + "\n"


def claude_print(s: dict) -> list:
    out = [f"Claude Code -> {s['url']} (model {s['model']}" + (f", context {s['ctx']:,} tokens)" if s.get("ctx") else ")"), ""]
    out.append("For one session, in the shell you start it from:")
    out.append("")
    for k, v in claude_env(s).items():
        out.append(f"  export {k}='{v}'")
    out.append(f"  export ANTHROPIC_AUTH_TOKEN=\"${{{KEY_VAR}:-abyss}}\"   # the server's key, or any word when it has none")
    out.append("  unset ANTHROPIC_API_KEY")
    out.append("  claude")
    out.append("")
    path, _ = claude_launcher(s)
    out.append(f"Or `{ME} --connect claude --write` writes that as {path.name} (start Claude Code with it; your")
    out.append("~/.claude settings stay as they are).")
    out += claude_notes(s)
    return out


def claude_notes(s: dict) -> list:
    notes = ["", "Good to know:",
             "  - Claude Code sends its own instructions and tool list (~20K tokens) with every request: the first",
             "    answer of a session waits for that prompt (Abyss keeps it for the next turns).",
             "  - ANTHROPIC_AUTH_TOKEN is sent as 'Authorization: Bearer'; ANTHROPIC_API_KEY would make Claude Code ask",
             "    whether to use it, so it is left unset here.",
             "  - The settings hold for that shell / that launcher only; ~/.claude/settings.json is not changed (its"
             " \"env\" would win over the shell's variables, so take ANTHROPIC_* out of it if you set them there)."]
    if str(s.get("model", "")).startswith("claude-"):
        notes.insert(1, f"  ! the model name '{s['model']}' starts with claude-: Claude Code then takes it for one of"
                        " its own models and ignores CLAUDE_CODE_MAX_CONTEXT_TOKENS. Give the config another"
                        " \"model_name\".")
    if s.get("ctx") and s["ctx"] < CLAUDE_MIN_CTX:
        notes.insert(1, f"  ! the server's context is {s['ctx']:,} tokens: Claude Code needs {CLAUDE_MIN_CTX:,}+ "
                        f"(its own prompt is ~20K). Set up again with a larger --context.")
    return notes


# ------------------------------------------------------------------------------------------------ OpenCode
def opencode_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    if WIN and not os.environ.get("XDG_CONFIG_HOME"):
        base = Path.home() / ".config"
    return base / "opencode" / "opencode.json"


def opencode_provider(s: dict) -> dict:
    model = {"name": f"{s['model']} (Abyss)"}
    if s.get("ctx"):
        model["limit"] = {"context": s["ctx"], "output": max(4096, min(32000, s["ctx"] // 4))}
    return {"npm": "@ai-sdk/openai-compatible", "name": "Abyss",
            "options": {"baseURL": s["url"] + "/v1", "apiKey": "{env:" + KEY_VAR + "}"},
            "models": {s["model"]: model}}


def opencode_print(s: dict) -> list:
    block = {"$schema": "https://opencode.ai/config.json", "provider": {"abyss": opencode_provider(s)},
             "model": f"abyss/{s['model']}"}
    out = [f"OpenCode -> {s['url']}/v1 (model {s['model']}" + (f", context {s['ctx']:,} tokens)" if s.get("ctx") else ")"), "",
           f"Merge this into {opencode_path()} (or an opencode.json in your project):", ""]
    out += ["  " + x for x in json.dumps(block, indent=2, ensure_ascii=False).splitlines()]
    out += ["", f"then: export {KEY_VAR}=<the server's key>   (any word when the server has none), and run opencode;",
            f"pick abyss/{s['model']} with /models if another model is the default.",
            f"Or `{ME} --connect opencode --write` merges it for you (a backup of the old file is kept).",
            "", "Good to know:",
            "  - limit.context is the server's context; OpenCode compacts the conversation before it fills.",
            "  - The model thinks before it answers (DeepSeek); the thinking arrives as reasoning_content."]
    return out


# ------------------------------------------------------------------------------------------------ --write
def _confirm(question: str, yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        print(f"{question} [y/N] (no terminal to ask: pass --yes)")
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    b = path.with_name(path.name + time.strftime(".bak-%Y%m%d-%H%M%S"))
    shutil.copy2(path, b)
    return b


def write_claude(s: dict, yes: bool) -> int:
    path, text = claude_launcher(s)
    print(f"Writes {path}" + (" (replacing the one there; a backup is kept)" if path.exists() else "") + ":")
    print("".join("  " + x + "\n" for x in text.splitlines()), end="")
    if not _confirm("Write it?", yes):
        print("Nothing written.")
        return 1
    b = _backup(path)
    path.write_text(text, encoding="utf-8")
    if not WIN:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Written: {path}" + (f" (the old one: {b.name})" if b else ""))
    print(f"Start Claude Code with: {'' if WIN else './'}{path.name}" + ("" if s.get("key") is None else
          f"   (export {KEY_VAR}=<the server's key> first)"))
    return 0


def write_opencode(s: dict, yes: bool, path: Path | None = None) -> int:
    path = Path(path) if path else opencode_path()
    cur = {}
    if path.exists():
        try:
            cur = json.loads(path.read_text(encoding="utf-8-sig"))
        except ValueError:
            print(f"{path} is not plain JSON (comments?): not changed. Merge the block from "
                  f"`{ME} --connect opencode` by hand.")
            return 1
        if not isinstance(cur, dict):
            print(f"{path}: not a JSON object - not changed")
            return 1
    new = dict(cur)
    new.setdefault("$schema", "https://opencode.ai/config.json")
    prov = dict(new.get("provider") or {})
    prov["abyss"] = opencode_provider(s)
    new["provider"] = prov
    set_default = "model" not in cur
    if set_default:
        new["model"] = f"abyss/{s['model']}"
    print(f"Sets the provider \"abyss\" in {path}" + (" (new file)" if not path.exists() else " (a backup is kept)") +
          (f" and makes abyss/{s['model']} the default model" if set_default else
           f"; your default model ({cur.get('model')}) stays - pick abyss/{s['model']} with /models") + ":")
    print("".join("  " + x + "\n" for x in json.dumps({"provider": {"abyss": prov["abyss"]}}, indent=2,
                                                       ensure_ascii=False).splitlines()), end="")
    if not _confirm("Write it?", yes):
        print("Nothing written.")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    b = _backup(path)
    path.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Written: {path}" + (f" (the old one: {b.name})" if b else ""))
    print(f"Then: export {KEY_VAR}=<the server's key> (any word when it has none) and run opencode.")
    return 0


# ------------------------------------------------------------------------------------------------ --check
WEATHER_TOOL = {"name": "get_weather", "description": "Current weather of a city.",
                "params": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}


def check(tool: str, s: dict, timeout: float = 600.0) -> int:
    """One plain prompt and one tool call in the tool's own dialect (Claude Code: Anthropic /v1/messages with its
    headers; OpenCode: OpenAI /v1/chat/completions with tools) -> 0 when both came back right."""
    print(f"Checking {s['url']} the way {'Claude Code' if tool == 'claude' else 'OpenCode'} talks to it "
          f"(model {s['model']}) - the first answer can take a while (the prompt, the thinking) ...")
    ok = True
    for what in ("plain", "tool"):
        t0 = time.time()
        try:
            if tool == "claude":
                body = {"model": s["model"], "max_tokens": 4096, "stream": False,
                        "messages": [{"role": "user", "content":
                                      "Reply with the single word OK." if what == "plain" else
                                      "What is the weather in Tokyo right now? Use the tool."}]}
                if what == "tool":
                    body["tools"] = [{"name": WEATHER_TOOL["name"], "description": WEATHER_TOOL["description"],
                                      "input_schema": WEATHER_TOOL["params"]}]
                r = _post(s["url"] + "/v1/messages?beta=true", body, s.get("key"),
                          {"anthropic-version": "2023-06-01", "x-api-key": s.get("key") or "abyss"}, timeout)
                blocks = r.get("content") or []
                text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
                calls = [b for b in blocks if b.get("type") == "tool_use"]
                got = (calls[0].get("name"), calls[0].get("input")) if calls else None
            else:
                body = {"model": s["model"], "max_tokens": 4096, "stream": False,
                        "messages": [{"role": "user", "content":
                                      "Reply with the single word OK." if what == "plain" else
                                      "What is the weather in Tokyo right now? Use the tool."}]}
                if what == "tool":
                    body["tools"] = [{"type": "function", "function": {
                        "name": WEATHER_TOOL["name"], "description": WEATHER_TOOL["description"],
                        "parameters": WEATHER_TOOL["params"]}}]
                r = _post(s["url"] + "/v1/chat/completions", body, s.get("key"), None, timeout)
                msg = ((r.get("choices") or [{}])[0].get("message")) or {}
                text = (msg.get("content") or "").strip()
                calls = msg.get("tool_calls") or []
                got = None
                if calls:
                    fn = calls[0].get("function") or {}
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        args = fn.get("arguments")
                    got = (fn.get("name"), args)
        except urllib.error.HTTPError as e:
            print(f"  {what}: the server answered {e.code}: {e.read()[:300].decode('utf-8', 'replace')}")
            ok = False
            continue
        except (OSError, ValueError) as e:
            print(f"  {what}: {getattr(e, 'reason', e)}")
            ok = False
            continue
        secs = time.time() - t0
        if what == "plain":
            good = "ok" in text.lower()
            print(f"  plain prompt: {'OK' if good else 'unexpected'} in {secs:.1f} s - {text[:80]!r}")
        else:
            good = bool(got) and got[0] == WEATHER_TOOL["name"] and isinstance(got[1], dict) and \
                "tokyo" in json.dumps(got[1]).lower()
            print(f"  tool call: {'OK' if good else 'not made' if not got else 'unexpected'} in {secs:.1f} s - "
                  + (f"{got[0]}({json.dumps(got[1], ensure_ascii=False)})" if got else repr(text[:80])))
        ok = ok and good
    print("Works: start the tool with the settings above." if ok else
          "Something is off (above). The server's log (the config's \"log\") has the request.")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tool", choices=TOOLS)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--url")
    ap.add_argument("--key-env")
    ap.add_argument("--config", type=Path)
    a = ap.parse_args(argv)
    s = find_server(a.url, a.key_env, a.config)
    if "error" in s:
        print(s["error"])
        return 1
    if a.check:
        return check(a.tool, s)
    if a.write:
        return write_claude(s, a.yes) if a.tool == "claude" else write_opencode(s, a.yes)
    print("\n".join(claude_print(s) if a.tool == "claude" else opencode_print(s)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
