"""serve/frontend.py - plan v0.3 P8: the request/response layer between API clients and the engine.

Everything here is text: no model, no GPU. The engine consumes token ids and produces text deltas; this module

  * normalizes OpenAI Chat Completions and Anthropic Messages requests into the chat template's message form,
  * renders the model's own Jinja chat template (pack `tokenizer/chat_template.jinja`) exactly as Hugging Face
    does (checked against the pack's `chat_golden.json`, 10 cases incl. thinking and tools),
  * parses the streamed output incrementally into reasoning (`<think>...</think>`), content and tool calls in
    either template's XML form - GLM's (`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`)
    or Qwen's (`<tool_call><function=NAME><parameter=P>VALUE</parameter>...</function></tool_call>`) - never
    emitting a partial tag to the client.

Design note: plan v0.3 describes a single C++ process. The frontend is Python first because the template is Jinja
and the formats change often; the engine boundary is token ids in, text deltas out, which keeps a later C++ port
(or a pybind wrapper) a local change.
"""
from __future__ import annotations

import base64
import codecs
import json
import math
import os
import re
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment


# ------------------------------------------------------------------------------------------------ template
class _GenerationTag(jinja2.ext.Extension):
    """{% generation %} ... {% endgeneration %} (transformers' marker of the assistant's tokens for training masks, in
    Laguna-S-2.1's template): rendered as its body, as apply_chat_template does without return_assistant_tokens_mask."""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.Scope(body, lineno=lineno)


_PAST_THINK_DROP = re.compile(r"\{%-?\s*if\s+loop\.index0\s*>\s*ns\.last_query_index\s*-?%\}")


def keep_past_thinking(source: str) -> str:
    """Qwen3.5-family templates from before 2026-11 (Ornith-1.5-122B, Qwen3.5-9B QSA GGUFs) render an assistant turn
    before the last user message without its <think> block; the newer ones keep it unless preserve_thinking is false.
    Dropped, turn 2's prompt is no longer turn 1's prompt plus the new tokens, and the delta-net layers cannot roll
    back to where they differ: every turn read the whole conversation again (issue #52: 234K tokens, ~600 s a turn).
    An older template gets the newer test; STRATA_KEEP_PAST_THINK=0 keeps the template as it is."""
    if os.environ.get("STRATA_KEEP_PAST_THINK") == "0" or "preserve_thinking" in source:
        return source
    return _PAST_THINK_DROP.sub("{%- if preserve_thinking is undefined or preserve_thinking is true or "
                                "loop.index0 > ns.last_query_index %}", source)


def ink_tool_names(messages: list[dict]) -> list[dict]:
    """Inkling's template heads a tool result with the tool's name (<|message_tool|>NAME), from the message's name or
    its tool_call_id; the requests' tool messages carry neither here, so each result gets the name of the call it
    answers - the calls of the assistant turn before it, in order (both APIs keep results in the calls' order)."""
    out, names = [], []
    for m in messages:
        if m.get("role") == "assistant":
            names = [(c.get("function") or {}).get("name") or "" for c in m.get("tool_calls") or []]
        elif m.get("role") == "tool" and not m.get("name") and names:
            m = {**m, "name": names.pop(0)}
        out.append(m)
    return out


class ChatTemplate:
    """The model's chat template, rendered with the same Jinja settings as transformers' apply_chat_template."""

    def __init__(self, path: str | Path, bos_token: str = "", dsml41: bool = False):
        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols", _GenerationTag])
        env.filters["tojson"] = tojson
        env.filters["from_json"] = json.loads           # DeepSeek V4's template parses tool-call arguments
        env.globals["raise_exception"] = raise_exception
        self.source = keep_past_thinking(Path(path).read_text(encoding="utf-8"))
        # DeepSeek V4.1 writes <｜DSML｜ calls> (a space after the token); a V4.1 pack whose template spells V4's
        # <｜DSML｜tool_calls> (ISTA-DASLab's GSQ-RCO GGUFs) shows the model V4.1's spelling instead, so the model
        # sees the form it writes.  Only the template's own strings change, never a message's.
        tree = env.parse(self.source)
        self.dsml41 = dsml41 and respell_dsml41(tree)
        self.template = env.from_string(tree)
        self.bos_token = bos_token
        # DeepSeek V4 (the DSML tool-call template): thinking is the template's `thinking` (off unless asked), with
        # reasoning_effort "high" / "max" on top.  The server's levels map to: none -> off, low -> thinking,
        # high -> thinking + high, max -> thinking + max; a request without a level thinks (as llama.cpp serves it).
        self.dsv4 = "\uff5cDSML\uff5c" in self.source or "｜DSML｜" in self.source
        # GLM-5.x: the template knows reasoning_effort "low" and "high" only - anything else, an absent one included,
        # is "Max", its most verbose - and it always opens <think> (no enable_thinking).  The server's levels are
        # GLM's own (low / high / max, or thinking off: the think block closed the way the template writes a turn
        # without reasoning).  Another model's template gets them in its own words (Qwen: low / medium / xhigh).
        self.glm_effort = "reasoning_effort in ['low', 'high']" in self.source
        # DeepSeek V4.1 1017's template: reasoning_effort low / medium / high / max (50 / 62 / 75 / 100) or an integer
        # 1..100, "high" when absent - the client's own value goes through (not the four levels above: its "medium"
        # was sent as "high" and "low" as none, so both thought at the template's high)
        self.v41_effort = "reasoning_effort is integer" in self.source and "effort_map" in self.source
        # DeepSeek V4 0731 (the DSML template, not V4.1: no Engram pack, not V4.1's effort map) - the model that answers
        # a Japanese question in Chinese without a system prompt (ja_system_prompt below)
        self.dsv4_0731 = self.dsv4 and not dsml41 and not self.v41_effort
        # Tencent Hy4-flash: reasoning_effort "no_think" (thinking off), "high" (the default) and, in the GGUFs'
        # template, "max" - any other value is an error there; the level goes into the system turn as
        # <｜reasoning_mode｜>reasoning_effort:<level>, the think tags are <think:opensource> / </think:opensource>
        self.hy4 = "reasoning_mode" in self.source and "'no_think'" in self.source
        self.hy4_max = self.hy4 and "'max'" in self.source
        # Tencent Hy-MT3 (the same family's template, plain <think> tags - HYTK '' - and no 'max'): a translation model
        # whose template defaults to no_think; a request without a level keeps that default (Hy4's is high)
        self.hymt = self.hy4 and "set HYTK = ''" in self.source and not self.hy4_max
        # Inkling 2: reasoning_effort none / minimal / low / medium / high / xhigh / max (0 .. 0.99, 0.9 when absent)
        # goes into a system message "Thinking effort level: <n>"; thinking off is "none" (the model may still think
        # briefly - the server reads its messages either way)
        self.ink = "Thinking effort level" in self.source and "effort_map" in self.source
        # Ling-3.x (Bailing V3): <role>HUMAN</role> turns; its processor writes a newline after each picture's
        # <|vision_end|> (BailingMM2Processor._expand_image_tokens), which the template itself does not
        self.ling = "<role>HUMAN</role>" in self.source and "<|role_end|>" in self.source
        # KRAFTON Raon-1: a ChatML template that knows thinking on / off only (enable_thinking); the reasoning effort is
        # the engine's - a steering vector's alpha per request (GEN effort=, raon_effort below), not prompt text
        self.raon = "raon-1" in self.source and "<|im_start|>" in self.source

    def render(self, messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = True,
               **kwargs) -> str:
        off = False
        if self.dsv4:
            off = kwargs.pop("enable_thinking", True) is False
            level = kwargs.pop("reasoning_effort", None)
            raw = kwargs.pop("effort_raw", None)
            kwargs["thinking"] = not off and level != "none"
            if self.v41_effort:
                kwargs["reasoning_effort"] = (raw if raw in ("low", "medium", "high", "max") or isinstance(raw, int)
                                              else level if level in ("low", "high", "max") else None)
            else:
                kwargs["reasoning_effort"] = level if level in ("high", "max") else None
            kwargs.setdefault("bos_token", self.bos_token)
            # the template reads OpenAI's form ({"type": "function", "function": {...}}); the server's are flat
            if tools:
                tools = [t if t.get("type") == "function" and "function" in t else {"type": "function", "function": t}
                         for t in tools]
        elif self.hy4:
            kwargs.pop("effort_raw", None)
            off = kwargs.pop("enable_thinking", True) is False
            level = kwargs.pop("reasoning_effort", None)
            kwargs["reasoning_effort"] = ("no_think" if off or level in ("none", "no_think") or (self.hymt and level is None)
                                          else "low" if self.hymt and level == "low" else
                                          "max" if level == "max" and self.hy4_max else "high")
            off = False   # (the template closes its own think block for no_think)
        elif self.ink:
            raw = kwargs.pop("effort_raw", None)
            off = kwargs.pop("enable_thinking", True) is False
            level = kwargs.pop("reasoning_effort", None)
            ok = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
            # ("minimal" comes as thinking off plus its raw name: the template's 0.1, not "none")
            kwargs["reasoning_effort"] = ("minimal" if raw == "minimal" else "none" if off else
                                          raw if raw in ok else level if level in ok else None)
            off = False   # (no think block to close: the effort level says it)
            messages = ink_tool_names(messages)
        elif self.raon:   # (the effort goes to the engine: raon_effort; the template takes enable_thinking only)
            kwargs.pop("effort_raw", None)
            kwargs.pop("reasoning_effort", None)
        elif self.glm_effort:
            kwargs.pop("effort_raw", None)
            off = kwargs.pop("enable_thinking", True) is False
            level = kwargs.pop("reasoning_effort", None)
            kwargs["reasoning_effort"] = "low" if off else level if level in ("low", "high") else "max"
        elif kwargs.get("reasoning_effort") in QWEN_EFFORT:
            kwargs.pop("effort_raw", None)
            kwargs["reasoning_effort"] = QWEN_EFFORT[kwargs["reasoning_effort"]]
        else:
            kwargs.pop("effort_raw", None)
        out = self.template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                   **kwargs)
        if off and add_generation_prompt and out.endswith("<think>"):
            out += "</think>"
        return out


# ------------------------------------------------------------------------------------------------ requests
def _text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in
                   ("text", "input_text", None))


IMAGE_PARTS = ("image_url", "input_image", "image")

# Thinking levels: GLM-5.x's own - Off, Low, High and Max (its template's default, the most thorough).  Clients spell
# them many ways and everything maps onto those four: OpenAI's "medium" (GLM has none) is High, "xhigh" is Max.
EFFORT = {"none": None, "off": None, "minimal": None, "disabled": None, "false": None,
          "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max", "maximum": "max"}
QWEN_EFFORT = {"low": "low", "high": "medium", "max": "xhigh"}   # the same levels in a Qwen3.x template's words


def effort_level(value) -> str | None:
    """A client's spelling -> the level's name (none / low / high / max); None when none is given or it is unknown."""
    if value is None or value == "":
        return None
    if value is False:
        return "none"
    if value is True:
        return None
    key = str(value).strip().lower()
    if key.isdigit():   # DeepSeek V4.1's 1..100 scale: the nearest level (low 50, high 75, max 100)
        n = int(key)
        return ("low" if n <= 56 else "high" if n <= 87 else "max") if 1 <= n <= 100 else None
    return (EFFORT[key] or "none") if key in EFFORT else None


def effort_kwargs(value) -> dict:
    """A reasoning effort as given by a client -> the template's kwargs.  Unknown values are a 400, not a crash."""
    if value is None or value == "":
        return {}
    if value is False:
        return {"enable_thinking": False}
    key = str(value).strip().lower()
    # an integer 1..100 (DeepSeek V4.1's own scale: low 50, medium 62, high 75, max 100): that template gets it as is
    # (effort_raw), every other one the nearest of the four levels
    if isinstance(value, int) and not isinstance(value, bool) or key.isdigit():
        n = int(key)
        if not 1 <= n <= 100:
            raise ValueError(f"reasoning effort {value!r}: an integer must be 1..100")
        return {"reasoning_effort": "low" if n <= 56 else "high" if n <= 87 else "max", "effort_raw": n}
    if key not in EFFORT:
        raise ValueError(f"unknown reasoning effort {value!r}: use none, low, medium, high, max or 1..100")
    level = EFFORT[key]
    if key == "minimal":   # thinking off for every model but Inkling 2, whose template has its own "minimal" (0.1)
        return {"enable_thinking": False, "effort_raw": key}
    return {"enable_thinking": False} if level is None else {"reasoning_effort": level, "effort_raw": key}


RAON_EFFORTS = ("low", "medium", "high", "xhigh", "max")   # raon1.reasoning_effort.names (alphas 0.3 .. 0)


def raon_effort(kwargs: dict) -> str | None:
    """Raon-1: the template kwargs' thinking level -> the engine's GEN effort= value (its steering alpha): thinking
    off "none" (the model's 0.1), the client's own name when Raon has it (medium, xhigh), else the level (low / high /
    max); None when nothing is asked (the engine's default, high).  The template still gets enable_thinking."""
    if kwargs.get("enable_thinking") is False:
        return "none"
    raw = kwargs.get("effort_raw")
    if isinstance(raw, str) and raw in RAON_EFFORTS:
        return raw
    if raw == "maximum":
        return "max"
    level = kwargs.get("reasoning_effort")
    return level if level in RAON_EFFORTS else None


def budget_effort(tokens) -> dict:
    """Anthropic's thinking budget (budget_tokens) -> a level: under 2K Low, under 8K High, else Max."""
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": "low" if n < 2048 else "high" if n < 8192 else "max"}


def _has_image(content) -> bool:
    return isinstance(content, list) and any(isinstance(p, dict) and p.get("type") in IMAGE_PARTS for p in content)


def _image_source(part: dict) -> str:
    """An image part's source as one string: a data: URL, an http(s) URL or a local file path.
    OpenAI: {"type": "image_url", "image_url": {"url": ...}} (or "image_url": "..."), Responses-style
    {"type": "input_image", "image_url": ...}; Anthropic: {"type": "image", "source": {"type": "base64",
    "media_type": ..., "data": ...}} or {"source": {"type": "url", "url": ...}}."""
    if part.get("type") == "image":
        src = part.get("source") or {}
        if src.get("type") == "base64":
            return f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
        return src.get("url") or src.get("path") or ""
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    return url or ""


def _parts_of(content):
    """Message content for the template: a string when there is no image (unchanged behaviour), otherwise the
    template's list form - text items and image items, in order - whose image items carry their source."""
    if not _has_image(content):
        return _text_of(content)
    items = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in IMAGE_PARTS:
            items.append({"type": "image", "source": _image_source(part)})
        elif part.get("type") in ("text", "input_text", None) and "text" in part:
            items.append({"type": "text", "text": part.get("text", "")})
    return items


def images_of(messages: list[dict]) -> list[str]:
    """The image sources of the rendered conversation, in prompt order (the template renders one
    <|vision_start|><|image_pad|><|vision_end|> per image item, message by message)."""
    return [item["source"] for m in messages if isinstance(m.get("content"), list)
            for item in m["content"] if item.get("type") == "image"]


# DeepSeek V4.1: an image is one <｜deepseek_image｜> token in the prompt (the official template writes it for an image
# block, its parts joined by a blank line).  The GGUF templates (ISTA's, unsloth's) print a message's content as it is, so
# a list content has to become text first; the server swaps each image for V41_IMAGE_SENTINEL and, once the prompt is
# rendered, puts the image marker token in its place (a "<｜deepseek_image｜>" typed inside a message stays text).
V41_IMAGE = "<｜deepseek_image｜>"
V41_IMAGE_SENTINEL = "\ue000image\ue000"


def v41_images_as_text(messages: list[dict]) -> list[dict]:
    """Every list content -> one string: the text parts and an image sentinel per image, joined with a blank line."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts = [V41_IMAGE_SENTINEL if p.get("type") == "image" else p.get("text", "")
                     for p in c if isinstance(p, dict) and (p.get("type") == "image" or "text" in p)]
            m = dict(m, content="\n\n".join(parts))
        out.append(m)
    return out


SPECIAL_MARK = "\ue001"


class ContentGuard:
    """A special token's text inside a message stays text.  The prompt is encoded with the special tokens parsed, so
    a `<|im_end|>`, `</think>` or `<｜deepseek_image｜>` typed in a message, pasted with source code or returned by a
    tool became the real token: a turn ended mid-message, or an image's place with no image (DeepSeek V4.1's official
    template refuses the marker in content outright).  Before the template runs, each special token's text inside the
    messages (contents, reasoning, tool-call arguments) and the tool definitions becomes a private-use sentinel; after
    it, restore() puts the text back and says where, and the tokenizer encodes those places as plain text while the
    markers the template writes stay special.

    Kept as they are, in a past assistant turn's content: the model's own tags (`model_tags`: the tokenizer's
    USER_DEFINED tokens - <think> / </think>, <tool_call>, GLM's <arg_key>, DeepSeek's ｜DSML｜), which the model writes
    as those tokens, so a client that sends a turn back as "<think>...</think>answer" or with its calls written in the
    text gets the tokens the model wrote (and a template that splits reasoning off at </think> still can); also
    <think> / </think> there when the template splits at them (`split_think`, for a tokenizer without types), and
    <tool_response> ... </tool_response> around a whole user message (Qwen's template tells a tool result sent as a
    user turn by them).  The CONTROL tokens (<|im_end|>, <｜User｜>, ...) are text everywhere in a message.
    STRATA_RAW_SPECIALS=1 turns all of it off (Service._prompt_ids): the prompt encoded as rendered, as before."""

    THINK = ("<think>", "</think>")
    TOOL_RESPONSE = ("<tool_response>", "</tool_response>")

    def __init__(self, specials, split_think: bool = False, model_tags=()):
        self.assistant_keep = frozenset(model_tags) | (frozenset(self.THINK) if split_think else frozenset())
        # the sentinels' own characters too: a private-use character typed in a message can never pose as one
        lits = sorted(set(specials) | {SPECIAL_MARK, V41_IMAGE_SENTINEL[0]}, key=len, reverse=True)
        self.lits = lits
        self.index = {s: i for i, s in enumerate(lits)}
        self.pat = re.compile("|".join(re.escape(s) for s in lits))
        self.back = re.compile(SPECIAL_MARK + r"(\d+)" + SPECIAL_MARK)

    def text(self, s, keep=()):
        if not isinstance(s, str):
            return s
        return self.pat.sub(lambda m: m.group(0) if m.group(0) in keep else
                            f"{SPECIAL_MARK}{self.index[m.group(0)]}{SPECIAL_MARK}", s)

    def value(self, v):
        """Every string inside a JSON value (tool-call arguments, a tool's schema); keys are names and stay."""
        if isinstance(v, str):
            return self.text(v)
        if isinstance(v, dict):
            return {k: self.value(x) for k, x in v.items()}
        if isinstance(v, list):
            return [self.value(x) for x in v]
        return v

    def content(self, c, keep=()):
        if isinstance(c, list):            # the template's list form: text items and image items
            return [dict(p, text=self.text(p["text"], keep)) if isinstance(p, dict) and isinstance(p.get("text"), str)
                    else p for p in c]
        return self.text(c, keep)

    def messages(self, messages: list[dict], tools: list[dict] | None):
        out = []
        for m in messages:
            m = dict(m)
            c = m.get("content")
            keep = ()
            if m.get("role") == "assistant":
                keep = self.assistant_keep
            elif (m.get("role") == "user" and isinstance(c, str) and c.startswith(self.TOOL_RESPONSE[0])
                  and c.endswith(self.TOOL_RESPONSE[1])):
                keep = self.TOOL_RESPONSE
            if "content" in m:
                m["content"] = self.content(c, keep)
            for k in ("reasoning_content", "thinking"):
                if isinstance(m.get(k), str):
                    m[k] = self.text(m[k])
            if m.get("tool_calls"):
                m["tool_calls"] = [dict(tc, function=dict(tc["function"],
                                                          arguments=self.value(tc["function"].get("arguments"))))
                                   if isinstance(tc, dict) and isinstance(tc.get("function"), dict) else tc
                                   for tc in m["tool_calls"]]
            out.append(m)
        return out, (self.value(tools) if tools else tools)

    def restore(self, prompt: str) -> tuple[str, list[tuple[int, int]]]:
        """The rendered prompt with the messages' text back, and the (start, end) ranges of that text."""
        parts, spans, pos, n = [], [], 0, 0
        for m in self.back.finditer(prompt):
            k = int(m.group(1))
            if k >= len(self.lits):
                continue
            parts.append(prompt[pos:m.start()])
            n += m.start() - pos
            parts.append(self.lits[k])
            spans.append((n, n + len(self.lits[k])))
            n += len(self.lits[k])
            pos = m.end()
        if not spans:
            return prompt, []
        parts.append(prompt[pos:])
        return "".join(parts), spans


def _late_system_to_user(messages: list[dict]) -> list[dict]:
    """The chat template takes a system message only at the start ("System message must be at the beginning").
    Clients also send them mid-conversation - Claude Code's hook context as {"role": "system"} after the first user
    turn, some OpenAI clients a late "developer" message (issue #56) - so those become user messages, in place:
    merging them into the first one would change the prompt's start and cost the conversation cache every turn.
    A late system message right after a user message goes in just before that user message (after the turn's tool
    results): the user's own words stay the last thing the model reads.  Claude Code's compaction request ends with
    the user's "summarize, do NOT call any tools" and then a system note (files read before, deferred tools, agent
    types); read last, the note made Qwen3.8-27B call Read instead of summarizing ("summarization produced empty
    response").  The same place every turn, so the conversation cache still matches."""
    out: list[dict] = []
    words = -1                                       # index of the last user message with the user's own words
    for i, m in enumerate(messages):
        if m.get("role") != "system" or i == 0:
            words = len(out) if m.get("role") == "user" else -1
            out.append(m)
        elif words >= 0 and words == len(out) - 1:   # right after it: in before it (several keep their order)
            out.insert(words, dict(m, role="user"))
            words += 1
        else:
            out.append(dict(m, role="user"))
    return out


# DeepSeek V4 0731 answers a Japanese question in Chinese now and then - the original weights too, the answer itself
# right: 8 Japanese prompts x 6 seeds (temperature 0.6), Chinese answers 14 / 48 for the original (UD-Q8_K_XL) and
# 16 / 48 for a 2-bit file; with this one system line 1 / 48 for both.  A language profile (lang ja-en) alone did not
# do it (12 / 48: the model wrote Chinese in the kanji Japanese shares).  Measured: Abyss-tools recipes/v4-0731-zh.
JA_SYSPROMPT = "日本語で答えてください。"
_KANA = re.compile(r"[\u3041-\u30ff]")


def ja_system_prompt(messages: list[dict], env: dict | None = None) -> list[dict]:
    """A request without a system message whose last user message has kana: JA_SYSPROMPT first, as the system
    message.  STRATA_JA_SYSPROMPT=0 (off / none / false) keeps the messages as they are.  Only for DeepSeek V4 0731
    (the caller checks ChatTemplate.dsv4_0731)."""
    import os
    v = (env if env is not None else os.environ).get("STRATA_JA_SYSPROMPT", "1").strip().lower()
    if v in ("0", "off", "none", "false", "no"):
        return messages
    if any(m.get("role") in ("system", "developer") for m in messages):
        return messages
    last = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last is None or not _KANA.search(_text_of(last.get("content"))):
        return messages
    return [{"role": "system", "content": JA_SYSPROMPT}] + list(messages)


# A thinking block's `signature` (the Anthropic API): what the server writes there is the block's reasoning exactly as
# the model wrote it (zlib + base64url behind SIGNATURE_TAG).  A client sends the block back with it, and the turn is
# rendered from that text - not from the block's `thinking`, which a client may have trimmed, re-wrapped or left empty
# ("display": "omitted") - so the re-rendered prompt is the one the model saw and the conversation cache still holds
# it.  A signature that is not ours (or does not decode) leaves the block's own text.
SIGNATURE_TAG = "abyss1:"


def thinking_signature(text: str) -> str:
    return SIGNATURE_TAG + base64.urlsafe_b64encode(zlib.compress(text.encode("utf-8"), 6)).decode("ascii")


def signature_text(sig) -> str | None:
    """The reasoning a signature of ours carries; None for any other signature."""
    if not isinstance(sig, str) or not sig.startswith(SIGNATURE_TAG):
        return None
    try:
        return zlib.decompress(base64.urlsafe_b64decode(sig[len(SIGNATURE_TAG):].encode("ascii"))).decode("utf-8")
    except (ValueError, zlib.error, UnicodeDecodeError):
        return None


def _has_tool_calls(messages: list[dict]) -> bool:
    return any(m.get("tool_calls") or m.get("role") == "tool" for m in messages)


# tool_choice "none" in a conversation that holds tool calls: the model sees the earlier calls and their results but
# no tools, and may write another call anyway.  A last user turn says the reply is prose (STRATA_TOOL_NONE_NOTE=0
# leaves it out); a call the model still starts is never sent (OutputParser no_calls: the answer ends there).  It goes
# at the END, so the conversation cache still holds everything before it.
TOOL_NONE_NOTE = "No tools are available for this reply: answer in prose, without any tool call."


def _tool_none(messages: list[dict], kwargs: dict) -> list[dict]:
    import os
    kwargs["_no_calls"] = True
    if _has_tool_calls(messages) and os.environ.get("STRATA_TOOL_NONE_NOTE", "1") != "0":
        messages = messages + [{"role": "user", "content": TOOL_NONE_NOTE}]
    return messages


def openai_to_messages(req: dict) -> tuple[list[dict], list[dict] | None, dict]:
    """OpenAI Chat Completions -> (template messages, template tools, template kwargs)."""
    messages = []
    for m in req.get("messages", []):
        role = m.get("role")
        if role == "developer":
            role = "system"
        out = {"role": role, "content": _parts_of(m.get("content")) if role in ("user", "tool", "assistant") else _text_of(m.get("content"))}
        if m.get("reasoning_content"):
            out["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = c.get("function", c)
                args = fn.get("arguments")
                if isinstance(args, str):               # the template requires a mapping, not a JSON string
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except ValueError:                  # (broken JSON in the client's history: kept as given)
                        args = {"raw": fn.get("arguments")}
                if args and not isinstance(args, dict):  # (a list or a number: not a mapping either)
                    args = {"raw": fn.get("arguments")}
                calls.append({"function": {"name": fn.get("name"), "arguments": args or {}}})
            out["tool_calls"] = calls
        if role == "assistant" and not out["content"] and not out.get("tool_calls") and not out.get("reasoning_content"):
            continue   # an empty turn is not rendered: a history of them teaches the model to answer with nothing
        messages.append(out)
    tools = [t.get("function", t) if isinstance(t, dict) and t.get("type") == "function" else t
             for t in req.get("tools") or []] or None
    tc = req.get("tool_choice")
    kwargs = {}
    if tc == "none":   # the tools stay declared to the client, but none is offered to the model
        tools = None
        messages = _tool_none(messages, kwargs)
    # OpenAI Chat Completions: "reasoning_effort"; Responses style: "reasoning": {"effort": ...}
    reasoning = req.get("reasoning") if isinstance(req.get("reasoning"), dict) else {}
    kwargs.update(effort_kwargs(req.get("reasoning_effort") or reasoning.get("effort")))
    # the vLLM / llama.cpp convention: {"chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "low"}}
    for k, v in (req.get("chat_template_kwargs") or {}).items():
        if k == "enable_thinking" and not v:
            kwargs = {"enable_thinking": False}
        elif k == "reasoning_effort" and "enable_thinking" not in kwargs:
            kwargs.update(effort_kwargs(v))
    # "required" / a named function: the answer starts with the call (Service.prepare; from Strata #790)
    if tools and tc == "required":
        kwargs["_force_tool"] = ""
    elif tools and isinstance(tc, dict) and tc.get("type") == "function" and (tc.get("function") or {}).get("name"):
        kwargs["_force_tool"] = tc["function"]["name"]
    return _late_system_to_user(messages), tools, kwargs


def _tool_reference_text(block: dict, by_name: dict) -> str:
    """A tool_reference block (Claude Code's ToolSearch result: a deferred tool is loaded) -> the tool's definition as
    text, where the model reads it: the system prompt keeps only the tools that were not deferred, so it stays the
    same while tools are loaded mid-conversation (and the conversation cache keeps it)."""
    name = block.get("tool_name") or block.get("name") or ""
    t = by_name.get(name)
    if t is None:
        return f"Tool {name} loaded."
    fn = {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {})}
    return f"Tool {name} loaded:\n" + json.dumps({"type": "function", "function": fn}, ensure_ascii=False)


def _anthropic_tools(req: dict) -> tuple[list[dict], list[dict]]:
    """-> (the tools the prompt shows, every tool a call may name).  A tool with "defer_loading": true (Claude Code's
    deferred tools, found with its ToolSearch) is not in the prompt until a tool_reference loads it; a server tool
    (a "type" other than "custom": web_search_..., code_execution_...) runs on Anthropic's side and is left out."""
    shown, every = [], []
    for t in req.get("tools") or []:
        if not isinstance(t, dict) or not t.get("name") or t.get("type") not in (None, "custom"):
            continue
        d = {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {})}
        every.append(d)
        if t.get("defer_loading") is not True:
            shown.append(d)
    return shown, every


def _result_content(content, by_name: dict):
    """A tool_result's content for the template: its tool_reference blocks become the tools' definitions (text)."""
    if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_reference" for b in content):
        content = [{"type": "text", "text": _tool_reference_text(b, by_name)} if isinstance(b, dict) and
                   b.get("type") == "tool_reference" else b for b in content]
    return _parts_of(content)


def anthropic_to_messages(req: dict) -> tuple[list[dict], list[dict] | None, dict]:
    """Anthropic Messages -> (template messages, template tools, template kwargs).  kwargs["_parser_tools"]: every
    tool a call may name (the deferred ones too); kwargs["_no_calls"]: tool_choice none.  Block types this server does
    not use (redacted_thinking, server tool results, documents, ...) are skipped, never a 400."""
    messages = []
    system = req.get("system")
    if system:
        messages.append({"role": "system", "content": _text_of(system)})
    shown, every = _anthropic_tools(req)
    by_name = {t["name"]: t for t in every}
    for m in req.get("messages", []):
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant", "system"):
            continue
        content = m.get("content")
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": content})
            continue
        content = [b for b in content or [] if isinstance(b, dict)]
        if m.get("role") == "user" and _has_image(content) and not any(
                b.get("type") == "tool_result" for b in content):
            messages.append({"role": "user", "content": _parts_of(content)})
            continue
        text, reasoning, calls, rest = [], [], [], []   # rest: the text and image blocks, in order (#41)
        for block in content:
            kind = block.get("type")
            if kind == "text":
                text.append(block.get("text", ""))
                rest.append(block)
            elif kind == "tool_reference":
                t = _tool_reference_text(block, by_name)
                text.append(t)
                rest.append({"type": "text", "text": t})
            elif kind in IMAGE_PARTS:
                rest.append(block)
            elif kind == "thinking":
                raw = signature_text(block.get("signature"))
                reasoning.append(raw if raw is not None else block.get("thinking", "") or "")
            elif kind == "tool_use":
                calls.append({"function": {"name": block.get("name"), "arguments": block.get("input") or {}}})
            elif kind == "tool_result":   # an image tool's result keeps its images, as role "tool" does on /v1/chat (#41)
                messages.append({"role": "tool", "content": _result_content(block.get("content"), by_name)})
        images = m.get("role") == "user" and _has_image(rest)
        if text or calls or reasoning or images:
            out = {"role": m["role"], "content": _parts_of(rest) if images else "".join(text)}
            if reasoning:
                out["reasoning_content"] = "".join(reasoning)
            if calls:
                out["tool_calls"] = calls
            messages.append(out)
    tools = shown or None
    kwargs = {}
    if every:
        kwargs["_parser_tools"] = every
    if isinstance(req.get("tool_choice"), dict) and req["tool_choice"].get("type") == "none":
        tools = None
        messages = _tool_none(messages, kwargs)
    # Anthropic: "thinking": {"type": "disabled"} or {"type": "enabled", "budget_tokens": N} or {"type": "adaptive"};
    # "output_config": {"effort": "low" | "medium" | "high"}
    thinking = req.get("thinking")
    effort = (req.get("output_config") or {}).get("effort") if isinstance(req.get("output_config"), dict) else None
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        kwargs["enable_thinking"] = False
    elif effort:
        kwargs.update(effort_kwargs(effort))
    elif isinstance(thinking, dict) and thinking.get("budget_tokens"):
        kwargs.update(budget_effort(thinking["budget_tokens"]))
    tc = req.get("tool_choice") if isinstance(req.get("tool_choice"), dict) else {}
    if tools and tc.get("type") == "any":            # a call is required: the answer starts with it
        kwargs["_force_tool"] = ""
    elif tools and tc.get("type") == "tool" and tc.get("name"):
        kwargs["_force_tool"] = tc["name"]
    return _late_system_to_user(messages), tools, kwargs


# ------------------------------------------------------------------------------------------------ output parser
@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:24])


@dataclass
class Event:
    kind: str                     # "reasoning" | "content" | "tool_call"
    text: str = ""
    call: ToolCall | None = None


THINK_END = "</think>"
THINK_END_HY4 = "</think:opensource>"   # Hy4-flash's end-of-reasoning token
CALL_START = "<tool_call>"
CALL_END = "</tool_call>"
FUNC_START = "<function="                       # Qwen's form: <function=NAME><parameter=P>V</parameter>...</function>
ARG_KEY, ARG_KEY_END = "<arg_key>", "</arg_key>"   # GLM's form: NAME<arg_key>K</arg_key><arg_value>V</arg_value>...
ARG_VAL, ARG_VAL_END = "<arg_value>", "</arg_value>"
# DeepSeek V4's DSML form: one block holds every call of the turn -
#   <｜DSML｜tool_calls>
#   <｜DSML｜invoke name="NAME">
#   <｜DSML｜parameter name="P" string="true|false">VALUE</｜DSML｜parameter>
#   </｜DSML｜invoke>
#   </｜DSML｜tool_calls>
# string="true" values are written as they are, the others as JSON (the chat template's own encoding).
DSML_CALLS, DSML_CALLS_END = "<\uff5cDSML\uff5ctool_calls>", "</\uff5cDSML\uff5ctool_calls>"
# DeepSeek V4.1 writes a space after the DSML token and `calls` for the block (<｜DSML｜ calls>, <｜DSML｜ invoke ...>,
# <｜DSML｜ parameter ...>), whatever spelling the pack's template shows; both are read.
DSML41_CALLS, DSML41_CALLS_END = "<\uff5cDSML\uff5c calls>", "</\uff5cDSML\uff5c calls>"
DSML_ENDS = {DSML_CALLS: DSML_CALLS_END, DSML41_CALLS: DSML41_CALLS_END}


def _dsml41_text(text: str, after_token: bool = False) -> str:
    """V4's DSML tags in a string -> V4.1's spelling; after_token: the string follows the token itself."""
    import re
    head = "^" if after_token else "(?<=\uff5cDSML\uff5c)"
    text = re.sub(head + "tool_calls>", " calls>", text)
    return re.sub(head + r"(invoke|parameter)\b", r" \1", text)


def respell_dsml41(tree) -> bool:
    """Respell a parsed chat template's DSML tags in V4.1's form, in place: the string constants that hold the token
    and the ones that follow a variable assigned the token (`'<' + dsml_token + 'invoke name="'`).  True when one
    changed.  Messages are values, not constants, so a message's text is never touched."""
    from jinja2 import nodes
    from collections import Counter
    sets = Counter(a.target.name for a in tree.find_all(nodes.Assign) if isinstance(a.target, nodes.Name)
                   and isinstance(a.node, nodes.Const) and a.node.value == "\uff5cDSML\uff5c")
    stores = Counter(n.name for n in tree.find_all(nodes.Name) if n.ctx in ("store", "param"))
    names = {k for k, c in sets.items() if stores[k] == c}     # never given another value (a loop, a macro, a set)

    def token(n) -> bool:                               # the expression ends with the token variable
        while isinstance(n, (nodes.Add, nodes.Concat, nodes.Filter)):
            n = n.right if isinstance(n, nodes.Add) else n.nodes[-1] if isinstance(n, nodes.Concat) else n.node
        return isinstance(n, nodes.Name) and n.name in names

    def consts(n):                                      # a constant, or the constants of a conditional's branches
        if isinstance(n, nodes.Const):
            yield n
        elif isinstance(n, nodes.CondExpr):
            yield from consts(n.expr1)
            if n.expr2 is not None:
                yield from consts(n.expr2)

    changed = False

    def fix(n, after_token: bool) -> None:
        nonlocal changed
        attr = "data" if isinstance(n, nodes.TemplateData) else "value"
        old = getattr(n, attr)
        if isinstance(old, str):
            new = _dsml41_text(old, after_token)
            if new != old:
                setattr(n, attr, new)
                changed = True

    for n in tree.find_all((nodes.Const, nodes.TemplateData)):
        fix(n, False)
    for n in tree.find_all(nodes.Add):
        if token(n.left):
            for c in consts(n.right):
                fix(c, True)
    for n in tree.find_all(nodes.Concat):
        for a, b in zip(n.nodes, n.nodes[1:]):
            if token(a):
                for c in consts(b):
                    fix(c, True)
    return changed


def parse_dsml_calls(body: str, schemas: dict | None = None) -> list[ToolCall]:
    """The body of a DSML tool_calls block -> its calls, in order.  `schemas` (tool name -> declared tool) lets a
    boolean parameter written `True` / `False` (or with string="true") come out as a bool."""
    import re
    calls = []
    for m in re.finditer(r'<\uff5cDSML\uff5c\s*invoke\s+name="([^"]*)"\s*>(.*?)</\uff5cDSML\uff5c\s*invoke>', body, re.S):
        args = {}
        props = (((schemas or {}).get(m.group(1)) or {}).get("parameters") or {}).get("properties") or {}
        for pm in re.finditer(r'<\uff5cDSML\uff5c\s*parameter\s+name="([^"]*)"(?:\s+string="(true|false)")?\s*>(.*?)'
                              r'</\uff5cDSML\uff5c\s*parameter>', m.group(2), re.S):
            value, types = pm.group(3), _declared_types(props.get(pm.group(1)))
            b = _as_bool(value, types)
            if b is not None:
                args[pm.group(1)] = b
            elif pm.group(2) == "true":
                args[pm.group(1)] = value
            else:                                   # string="false": JSON, unless the schema says string
                args[pm.group(1)] = _arg_value(value, types)
        calls.append(ToolCall(name=m.group(1), arguments=args))
    if not calls:
        raise ValueError("malformed DSML tool calls: " + body[:80])
    return calls


# An XML call some models drift into (Inkling 2 among them, its own form being a JSON call message):
#   <function_calls>                                      (optional, also inside <tool_call>)
#   <invoke name="NAME">
#   <parameter name="P" string="true">VALUE</parameter>   (string="true|false" optional, as DSML's)
#   </invoke>
#   </function_calls>
XML_CALLS, XML_CALLS_END = "<function_calls>", "</function_calls>"
XML_INVOKE, XML_INVOKE_END = "<invoke", "</invoke>"
_XML_INVOKE_OPEN = re.compile(r'<invoke\s+name\s*=\s*"([^"]*)"\s*>')
_XML_FIND = re.compile(r"<function_calls>|<invoke(?=\s|\Z)")
_XML_INVOKE = re.compile(r'<invoke\s+name\s*=\s*"([^"]*)"\s*>(.*?)</invoke>', re.S)
_XML_PARAM = re.compile(r'<parameter\s+name\s*=\s*"([^"]*)"(?:\s+string\s*=\s*"(true|false)")?\s*>(.*?)</parameter>',
                        re.S)


def parse_xml_calls(body: str, schemas: dict | None = None) -> list[ToolCall]:
    """The complete <invoke> elements of `body` -> their calls, in order; [] when there is none or one names a tool
    the request did not declare (`schemas`: tool name -> declared tool; with none declared nothing is a call - this
    is a fallback form, and a model quoting XML must not hand the client a call).  A value loses one leading and one
    trailing newline (the layout around it); string="true" keeps it as written, else the schema decides (_arg_value)."""
    if not schemas:
        return []
    calls = []
    for m in _XML_INVOKE.finditer(body):
        if m.group(1) not in schemas:
            return []
        props = ((schemas.get(m.group(1)) or {}).get("parameters") or {}).get("properties") or {}
        args = {}
        for pm in _XML_PARAM.finditer(m.group(2)):
            value, types = pm.group(3), _declared_types(props.get(pm.group(1)))
            if value.startswith("\n"):
                value = value[1:]
            if value.endswith("\n"):
                value = value[:-1]
            b = _as_bool(value, types)
            args[pm.group(1)] = b if b is not None else value if pm.group(2) == "true" else _arg_value(value, types)
        calls.append(ToolCall(name=m.group(1), arguments=args))
    return calls


_SCHEMA_DEPTH = 8          # anyOf / oneOf nesting looked into; deeper is "no schema"


def _declared_types(schema, depth: int = 0):
    """The JSON types a parameter's schema allows, as a frozenset of names - from its "type" (one name or a list) or
    gathered through anyOf / oneOf (nested, up to _SCHEMA_DEPTH): Pydantic writes Optional[bool] as
    {"anyOf": [{"type": "boolean"}, {"type": "null"}]}.  None when it cannot be known (no schema, a branch with a
    $ref or without a type, too deep): the value is then read as before, JSON when it parses (Strata 5e965ec57's
    idea).  A bare type name (a string) is also taken, for the callers that had only that."""
    if isinstance(schema, str):
        return frozenset((schema,))
    if not isinstance(schema, dict) or depth > _SCHEMA_DEPTH:
        return None
    t = schema.get("type")
    if isinstance(t, str):
        return frozenset((t,))
    if isinstance(t, (list, tuple)):
        names = frozenset(x for x in t if isinstance(x, str))
        return names or None
    branches = []
    for key in ("anyOf", "oneOf"):
        if isinstance(schema.get(key), list):
            branches += schema[key]
    if not branches:
        return None
    out = set()
    for branch in branches:
        got = _declared_types(branch, depth + 1)
        if got is None:
            return None
        out |= got
    return frozenset(out) or None


def _is_string(types) -> bool:
    """The parameter is a string (or a string or null): its text is the value as written, never JSON-decoded."""
    return bool(types) and "string" in types and types <= {"string", "null"}


def _fits(value, types) -> bool:
    """A JSON-decoded value is one of the declared types."""
    for t in types:
        if (t == "string" and isinstance(value, str) or t == "null" and value is None or
                t == "boolean" and isinstance(value, bool) or t == "object" and isinstance(value, dict) or
                t == "array" and isinstance(value, list) or
                t == "integer" and isinstance(value, int) and not isinstance(value, bool) or
                t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool)):
            return True
    return False


def _as_bool(value, declared):
    """`True` / `False` (Python's spelling, any case, also a JSON string "true") for a parameter the schema declares
    boolean and not string -> the JSON bool; None otherwise.  Models write `<parameter=x>True</parameter>` with
    thinking off, and the client got the string "True".  `declared`: _declared_types() of the schema (or the schema,
    or a type name)."""
    types = declared if isinstance(declared, frozenset) else _declared_types(declared)
    if not types or "boolean" not in types or "string" in types or not isinstance(value, str):
        return None
    word = value.strip().lower()
    if len(word) >= 2 and word[0] == word[-1] and word[0] in "\"'":
        word = word[1:-1].strip()
    return {"true": True, "false": False}.get(word)


def _arg_value(value: str, declared) -> object:
    """A parameter's text as the client gets it.  `declared`: _declared_types() of its schema (or the schema).
    - a string (or string-or-null) parameter: as written ("42" stays "42", and so does `null`: a stream sends a
      string value's characters as they come, so the whole call must say the same)
    - another type that allows null: `null` is None (JSON)
    - a boolean one also takes `True` / `False` in any case (and the quoted words)
    - otherwise JSON-decoded when it parses (numbers, booleans, objects, arrays), else as written - except that a
      parameter that also allows a string keeps its text when the decoded value is none of the declared types
      (["string", "integer"]: "42" -> 42, "true" stays "true")."""
    types = declared if isinstance(declared, frozenset) else _declared_types(declared)
    if _is_string(types):
        return value
    b = _as_bool(value, types)
    if b is not None:
        return b
    try:
        v = json.loads(value)
    except ValueError:
        n = _as_number(value, types)
        return value if n is None else n
    if types and "string" in types and not _fits(v, types):
        return value
    return v


_LOOSE_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _as_number(value, types):
    """A number JSON does not take - a leading `+`, leading zeros (`010`), a bare `.5` or `5.` - as a number, for a
    parameter the schema declares integer or number and not string; None otherwise.  Quantized models write them
    now and then, and the client got the string "+10".  An integer parameter takes a whole value only (`010` -> 10,
    `+3.0` -> 3); `1.5` there stays text, as before."""
    if not types or "string" in types or not ({"integer", "number"} & types) or not isinstance(value, str):
        return None
    word = value.strip()
    if not _LOOSE_NUMBER.fullmatch(word):
        return None
    try:
        f = float(word)
    except ValueError:
        return None
    if not math.isfinite(f):
        return None
    whole = "." not in word and "e" not in word.lower()
    if whole:
        return int(word)
    if "number" in types:
        return f
    return int(f) if f.is_integer() else None


def call_name(body: str) -> str:
    """The tool name of a call body (the text between <tool_call> and </tool_call>), in either form."""
    body = body.strip()
    if body.startswith(FUNC_START):
        return body[len(FUNC_START):].split(">", 1)[0]
    return body.split("<", 1)[0].strip()


_CALL_OPEN_SLIP = re.compile(r"<(?:=|function\s*:\s*|function\s+)([A-Za-z_][A-Za-z0-9_.\-]*)\s*>")
_CALL_OPEN_NAME = re.compile(r"<([A-Za-z_][A-Za-z0-9_.\-]*)>")


def fix_call_open(body: str, names=()) -> str:
    """A Qwen-style call body whose opening tag slipped, rewritten as <function=NAME>: `<=NAME>`, `<function:NAME>`
    or `<function NAME>`, and `<NAME>` when NAME is one of the declared tools (`names`) - quantized Qwen models write
    these now and then, and the call was returned as text.  Anything else is left as it is."""
    b = body.lstrip()
    if b.startswith(FUNC_START):
        return body
    m = _CALL_OPEN_SLIP.match(b)
    if m is None:
        m = _CALL_OPEN_NAME.match(b)
        if m is None or m.group(1) not in names:
            return body
    return FUNC_START + m.group(1) + ">" + b[m.end():]


# A Qwen-form value that holds the call's own closing tags (a document about tool calls, written with a tool call):
# `</parameter></function></tool_call>` inside a call body is the value's text - the body reaches past such a triple
# only when what follows it closes the call again before any new <tool_call> (OutputParser._qwen_close)
_QWEN_INNER_CLOSE = re.compile(r"</parameter>\s*</function>\s*</tool_call>")
_QWEN_BODY_TAIL = re.compile(r"</parameter>\s*</function>\s*\Z")


def _close_prefix(s: str) -> bool:
    """`s` may still grow into `</function></tool_call>` (spaces between): a streamed value cannot end before it."""
    t = s.lstrip()
    if not t.startswith("</function>"):
        return "</function>".startswith(t)
    u = t[len("</function>"):].lstrip()
    return "</tool_call>".startswith(u) and u != "</tool_call>"


def parse_tool_call(body: str, schema: dict | None = None) -> ToolCall:
    """A call body -> ToolCall, in GLM's form (`NAME<arg_key>K</arg_key><arg_value>V</arg_value>...`, what GLM-5.x
    writes: issue #5) or Qwen's (`<function=NAME>\\n<parameter=P>\\nVALUE\\n</parameter>...</function>`).  Values are
    JSON-decoded when the tool's schema says the parameter is not a string (or, without a schema, when they parse as
    JSON objects/arrays/numbers/booleans)."""
    body = body.strip()
    props = ((schema or {}).get("parameters") or {}).get("properties") or {}
    if not body.startswith(FUNC_START):
        name = call_name(body)
        rest = body[len(body.split("<", 1)[0]):]
        if not name or not (rest == "" or rest.lstrip().startswith(ARG_KEY)):
            raise ValueError("malformed tool call: " + body[:80])
        args = {}
        while ARG_KEY in rest:
            rest = rest[rest.index(ARG_KEY) + len(ARG_KEY):]
            end = rest.find(ARG_KEY_END)
            if end < 0:
                raise ValueError("malformed tool call: " + body[:80])
            key, rest = rest[:end].strip(), rest[end + len(ARG_KEY_END):]
            v0 = rest.find(ARG_VAL)
            if v0 < 0 or rest[:v0].strip():
                raise ValueError("malformed tool call: " + body[:80])
            rest = rest[v0 + len(ARG_VAL):]
            end = rest.find(ARG_VAL_END)
            value = rest[:end] if end >= 0 else rest
            rest = rest[end + len(ARG_VAL_END):] if end >= 0 else ""
            args[key] = _arg_value(value, _declared_types(props.get(key)))
        return ToolCall(name=name, arguments=args)
    if ">" not in body:
        raise ValueError("malformed tool call: " + body[:80])
    name = body[len(FUNC_START):body.index(">")]
    rest = body[body.index(">") + 1:]
    args = {}
    while "<parameter=" in rest:
        rest = rest[rest.index("<parameter=") + len("<parameter="):]
        close = rest.find(">")
        if close < 0 or "<" in rest[:close]:   # an unclosed <parameter= tag: malformed like the other cases
            raise ValueError("malformed tool call: " + body[:80])
        pname = rest[:close]
        rest = rest[close + 1:]
        end = rest.find("</parameter>")
        while end >= 0 and _QWEN_INNER_CLOSE.match(rest, end):   # (a closing triple inside the value: its text)
            end = rest.find("</parameter>", end + 1)
        value = rest[:end] if end >= 0 else rest
        rest = rest[end + len("</parameter>"):] if end >= 0 else ""
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        args[pname] = _arg_value(value, _declared_types(props.get(pname)))
    return ToolCall(name=name, arguments=args)


# Inkling 2's tool call is a model message of its own: <|message_model|>NAME<|content_invoke_tool_json|>
# {"name": NAME, "args": {...}}<|end_message|> (its chat template writes past calls so, the arguments an object)
INK_CALL = "<|content_invoke_tool_json|>"


def ink_call_open(name: str) -> str:
    """A required call's opening in Inkling's form (the prompt ends with it; InklingCalls reads the rest)."""
    return "<|message_model|>" + name + INK_CALL


class InkArgStream:
    """The streaming view of an Inkling 2 call body ({"name": NAME, "args": {...}}), read as it is written.  The body
    is followed with a small incremental JSON reader: once "name" (a string) and then "args" / "arguments" /
    "parameters" (an object) are read, on_start(name) is asked whether to announce the call; inside the arguments each
    complete piece goes out re-encoded as json.dumps(args, ensure_ascii=False) would write it (string values
    character by character, other scalars whole once complete, `, ` and `: ` between), so the pieces joined are the
    same text as the call a held body gives.  A body of any other shape is never announced ("hold": the whole body
    decides at its end, as before).  A body that goes wrong after the announcement, or ends early, is closed as far
    as it came (close())."""

    ARGS_KEYS = ("args", "arguments", "parameters")
    LIT = frozenset("0123456789+-.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")   # a scalar's characters
    WS = " \t\r\n"

    def __init__(self, on_start):
        self.on_start = on_start
        self.buf, self.phase = "", "top"   # top -> k1 -> c1 -> v1 -> s1 -> k2 -> c2 -> v2 -> args -> after | hold
        self.name = ""
        self.stack = []                    # open containers in the arguments: [kind "o"/"a", state, key text]
        self.mode = None                   # None | "str" (a string value) | "lit" (a scalar being read)
        self.lit = ""
        self.out = []                      # every piece sent: the arguments' text so far

    @property
    def started(self) -> bool:
        return self.phase in ("args", "after", "broken")

    def text(self) -> str:
        return "".join(self.out)

    def feed(self, s: str) -> list[str]:
        """More of the body: the argument pieces to send (none before the call is announced, or after it closed)."""
        if self.phase in ("hold", "after", "broken"):
            return []
        self.buf += s
        pieces = []
        try:
            i = self._run(pieces)
        except ValueError:               # (the pieces up to it are whole JSON and are sent; close() ends the rest)
            self.phase = "broken" if self.started else "hold"
            i = len(self.buf)
        self.buf = self.buf[i:]
        self.out += pieces
        return pieces

    # -- the reader ----------------------------------------------------------------------------------------------
    def _ws(self, i: int) -> int:
        b = self.buf
        while i < len(b) and b[i] in self.WS:
            i += 1
        return i

    def _string_end(self, i: int) -> int:
        """buf[i] is a string's opening quote: the index past its closing quote, or -1 when it has not come."""
        b, j = self.buf, i + 1
        while True:
            q, e = b.find('"', j), b.find("\\", j)
            if q < 0:
                return -1
            if 0 <= e < q:
                j = e + 2
                if j > len(b):
                    return -1
                continue
            return q + 1

    def _whole_string(self, i: int):
        """(the string at buf[i] decoded, the index past it), or None while incomplete."""
        end = self._string_end(i)
        if end < 0:
            return None
        v = json.loads(self.buf[i:end])
        return v, end

    def _run(self, pieces: list) -> int:
        b, i = self.buf, 0
        while True:
            if self.phase == "args":
                return self._args(i, pieces)
            if self.phase == "after":
                return len(b)
            i = self._ws(i)
            if i >= len(b):
                return i
            c, ph = b[i], self.phase
            if ph == "top":
                if c != "{":
                    raise ValueError
                self.phase, i = "k1", i + 1
            elif ph in ("k1", "k2"):
                if c != '"':
                    raise ValueError
                got = self._whole_string(i)
                if got is None:
                    return i
                key, i = got
                if (ph == "k1" and key != "name") or (ph == "k2" and key not in self.ARGS_KEYS):
                    raise ValueError
                self.phase = "c1" if ph == "k1" else "c2"
            elif ph in ("c1", "c2"):
                if c != ":":
                    raise ValueError
                self.phase, i = ("v1" if ph == "c1" else "v2"), i + 1
            elif ph == "v1":
                if c != '"':
                    raise ValueError
                got = self._whole_string(i)
                if got is None:
                    return i
                self.name, i = got
                self.phase = "s1"
            elif ph == "s1":
                if c != ",":
                    raise ValueError
                self.phase, i = "k2", i + 1
            elif ph == "v2":
                if c != "{" or not self.on_start(self.name):
                    raise ValueError
                self.phase, i = "args", i + 1
                self.stack = [["o", "k0"]]
                pieces.append("{")

    def _value_done(self) -> None:
        if self.stack:
            self.stack[-1][1] = "e"

    def _args(self, i: int, pieces: list) -> int:
        """Inside the arguments object: consume what can be, returning the index to keep from."""
        b = self.buf
        while True:
            if self.mode == "str":
                i = self._str(i, pieces)
                if self.mode == "str":
                    return i
                continue
            if self.mode == "lit":
                j = i
                while j < len(b) and b[j] in self.LIT:
                    j += 1
                self.lit += b[i:j]
                if j >= len(b):
                    return j
                pieces.append(json.dumps(json.loads(self.lit), ensure_ascii=False))
                self.mode, self.lit, i = None, "", j
                self._value_done()
                continue
            if not self.stack:
                self.phase = "after"
                return len(b)
            i = self._ws(i)
            if i >= len(b):
                return i
            c = b[i]
            top = self.stack[-1]
            kind, st = top[0], top[1]
            if kind == "o" and st in ("k0", "k"):
                if c == "}" and st == "k0":
                    self.stack.pop()
                    pieces.append("}")
                    self._value_done()
                    i += 1
                    continue
                if c != '"':
                    raise ValueError
                got = self._whole_string(i)
                if got is None:
                    return i
                key, i = got
                top[1] = "colon"
                top[2:] = [(", " if st == "k" else "") + json.dumps(key, ensure_ascii=False) + ": "]
                continue
            if kind == "o" and st == "colon":
                if c != ":":
                    raise ValueError
                pieces.append(top[2])
                top[1], i = "v", i + 1
                continue
            if kind == "o" and st == "e" or kind == "a" and st == "e":
                if c == ",":
                    top[1] = "k" if kind == "o" else "v"
                    i += 1
                    continue
                if c == ("}" if kind == "o" else "]"):
                    self.stack.pop()
                    pieces.append(c)
                    self._value_done()
                    i += 1
                    continue
                raise ValueError
            # a value: after an object's colon, or an array's start / comma
            if kind == "a" and st == "v0" and c == "]":
                self.stack.pop()
                pieces.append("]")
                self._value_done()
                i += 1
                continue
            if c not in "{[\"" and c not in self.LIT:
                raise ValueError
            if kind == "a" and st == "v":
                pieces.append(", ")
            top[1] = "in"                       # (a value is being read: done -> "e", _value_done)
            if c == "{":
                self.stack.append(["o", "k0"])
                pieces.append("{")
                i += 1
            elif c == "[":
                self.stack.append(["a", "v0"])
                pieces.append("[")
                i += 1
            elif c == '"':
                self.mode = "str"
                pieces.append('"')
                i += 1
            else:
                self.mode, self.lit = "lit", ""

    def _str(self, i: int, pieces: list) -> int:
        """Inside a string value: its decoded characters so far, re-encoded; the index to keep from."""
        b, n = self.buf, len(self.buf)
        while i < n:
            j = i
            while j < n and b[j] not in '"\\' and b[j] >= " ":
                j += 1
            if j > i:
                pieces.append(b[i:j])           # (plain characters: json.dumps writes them as they are)
                i = j
                continue
            c = b[i]
            if c == '"':
                pieces.append('"')
                self.mode = None
                self._value_done()
                return i + 1
            if c != "\\":
                raise ValueError                # (a raw control character: not JSON)
            if i + 1 >= n:
                return i
            if b[i + 1] != "u":
                esc = b[i:i + 2]
                end = i + 2
            else:
                if i + 6 > n:
                    return i
                end = i + 6
                cp = int(b[i + 2:i + 6], 16) if all(x in "0123456789abcdefABCDEF" for x in b[i + 2:i + 6]) else -1
                if cp < 0:
                    raise ValueError
                if 0xD800 <= cp < 0xDC00:       # a high surrogate: its low half may follow as a second escape
                    if i + 12 > n and b[i + 6:].startswith("\\u"[:n - i - 6]):
                        return i
                    if b[i + 6:i + 8] == "\\u":
                        lo = b[i + 8:i + 12]
                        if all(x in "0123456789abcdefABCDEF" for x in lo) and 0xDC00 <= int(lo, 16) < 0xE000:
                            end = i + 12
                esc = b[i:end]
            pieces.append(json.dumps(json.loads('"' + esc + '"'), ensure_ascii=False)[1:-1])
            i = end
        return i

    def close(self) -> str:
        """The body ended early (or went wrong) after the announcement: the text that closes what is open, so the
        pieces sent parse as JSON; the reader is done."""
        if not self.started or self.phase == "after":
            self.phase = "after" if self.started else "hold"
            return ""
        tail = ""
        if self.mode == "str":
            tail += '"'
            self._value_done()
        elif self.mode == "lit":
            try:
                tail += json.dumps(json.loads(self.lit), ensure_ascii=False)
            except ValueError:
                tail += "null"
            self._value_done()
        self.mode = None
        for kind, st, *_ in reversed(self.stack):
            if kind == "o" and st == "v":
                tail += "null"                  # (a key sent with its colon, its value never came)
            tail += "}" if kind == "o" else "]"
        self.stack, self.phase = [], "after"
        self.out.append(tail)
        return tail


class InklingCalls:
    """Inkling 2's call messages out of the token stream (the markers are special tokens, so this works on ids, ahead
    of the text parser): a <|message_model|> holds its header (the tool's name) until the message's kind is known; a
    text / thinking message goes on as before, an invoke message's JSON is read to its <|end_message|> and becomes one
    tool_call event (the existing call shape) - or text when it is not a call to a declared tool or not valid JSON.
    While the JSON is written it is also read as it comes (InkArgStream): a body in the template's shape is announced
    ("tool_start") once its name and the start of its arguments are read, and its arguments go out piece by piece
    ("tool_args"), as the text parser's stream_tools calls do - a 4K-token file write is no longer silent until its
    end.  Joined, the pieces are the text of the arguments the whole body gives."""

    HDR_MAX = 64                     # header tokens at most (a tool's name): more is not a header - text

    def __init__(self, tok, mm: int, th: int, tx: int, em: int, invoke: int, parser: "OutputParser",
                 invoke_text: int | None = None, error: int | None = None):
        self.tok, self.mm, self.th, self.tx, self.em, self.invoke = tok, mm, th, tx, em, invoke
        # <|content_invoke_tool_text|> (a call written as text, not JSON: no call the API can carry - its message is
        # read to the end and goes out as text) and <|content_tool_error|> (a text message: the caller shows it)
        self.invoke_text, self.error = invoke_text, error
        self.parser = parser
        self.state, self.hdr, self.body, self.name, self.marker = "idle", [], [], "", INK_CALL
        self.stream, self.scall, self.dec = None, None, None

    def start_call(self, name: str, marker: str = INK_CALL) -> None:
        """The prompt ended inside a call's header (a required call, ink_call_open): its JSON comes next."""
        self.state, self.body, self.name, self.marker = "json", [], name, marker
        self.parser.state, self.parser.lead = "content", True   # (a call message: any reasoning is over)
        self.scall = None
        # the streaming view: JSON calls only, and not under tool_choice "none" (that call is refused at its end)
        self.stream = InkArgStream(self._announce) if marker == INK_CALL and not self.parser.no_calls else None
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace") \
            if self.stream is not None and hasattr(self.tok, "token_bytes") else None
        self.sent = 0                    # (a tokenizer without token_bytes: the decoded body's characters fed so far)

    def _announce(self, name: str) -> bool:
        """InkArgStream read the name and the start of the arguments: announce the call if it is one to run."""
        nm = name or self.name
        if not nm or not self.parser._declared(nm):
            return False                 # (not a call this request can take: the whole body decides at its end)
        self.scall = ToolCall(name=nm, arguments={})
        return True

    def _stream_text(self, t: int) -> str:
        if self.dec is not None:
            return self.dec.decode(self.tok.token_bytes(t))
        text = self.tok.decode(self.body)
        if text.endswith("\ufffd"):
            return ""                    # (a character still split across tokens)
        delta, self.sent = text[self.sent:], len(text)
        return delta

    def _stream_push(self, t: int) -> list[Event]:
        """One more body id through the streaming view: the call's announcement and argument pieces, if any."""
        if self.stream is None or self.stream.phase in ("hold", "after", "broken"):
            return []
        before = self.scall
        pieces = self.stream.feed(self._stream_text(t))
        out = [Event("tool_start", call=self.scall)] if self.scall is not None and before is None else []
        return out + [Event("tool_args", p, call=self.scall) for p in pieces if p]

    def push(self, t: int) -> list[Event] | None:
        """One generated id.  None: not this reader's - the caller handles it as before; a list: taken (the events
        to send, often none)."""
        if self.state == "idle":
            if t == self.mm:
                self.state, self.hdr = "hdr", []
                return []
            return None
        if self.state == "hdr":
            if t == self.invoke or (t == self.invoke_text and t is not None):
                self.start_call(self.tok.decode(self.hdr).strip(),
                                INK_CALL if t == self.invoke else "<|content_invoke_tool_text|>")
                return []
            if t == self.mm:                 # (an empty message: a new header)
                self.hdr = []
                return []
            if t in (self.th, self.tx, self.em) or (t == self.error and t is not None):
                self.state, self.hdr = "idle", []   # (a text / thinking message: its header, if any, is not text)
                return None
            self.hdr.append(t)
            if len(self.hdr) > self.HDR_MAX:
                text, self.state, self.hdr = self.tok.decode(self.hdr), "idle", []
                return self.parser.feed(text)
            return []
        if t == self.em:
            return self._close()
        self.body.append(t)
        return self._stream_push(t)

    def _close(self, cut: bool = False) -> list[Event]:
        """The call message ended (cut: the output ended inside it, no <|end_message|>)."""
        text, name, marker = self.tok.decode(self.body), self.name, self.marker
        self.state, self.body, self.name, self.marker = "idle", [], "", INK_CALL
        stream, scall = self.stream, self.scall
        self.stream, self.scall, self.dec = None, None, None
        if scall is not None:
            # announced while it was written: the call is what was streamed (closed as far as it came, when the
            # output ended early or the body went wrong after the start) - the pieces sent are never taken back
            tail = stream.close()
            out = [Event("tool_args", tail, call=scall)] if tail else []
            try:
                scall.arguments = json.loads(stream.text())
            except ValueError:           # (cannot happen: the pieces are json.dumps's)
                scall.arguments = {}
            self.parser.state, self.parser.lead = "content", True
            return out + [Event("tool_call", call=scall)]
        call = None
        try:
            obj = json.loads(text) if marker == INK_CALL else None
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            # the template writes "args"; a model may write "arguments" / "parameters", or the object as a JSON string
            args = next((obj[k] for k in ("args", "arguments", "parameters") if k in obj), {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    args = None
            nm = obj.get("name") if isinstance(obj.get("name"), str) and obj.get("name") else name
            if isinstance(args, dict) and nm and self.parser._declared(nm):
                call = ToolCall(name=nm, arguments=args)
        # a call message whose body is the XML form (<invoke name=..><parameter name=..>), not JSON
        calls = [call] if call is not None else parse_xml_calls(text, self.parser.schemas)
        if not calls and cut and marker == INK_CALL:
            return []                        # (a JSON call the output ended inside: never the marker as text)
        if not calls:                        # (not a call this request can take: the text, as the parser would)
            return [Event("content", name + marker + text)]
        if self.parser.no_calls:
            return [Event("calls_refused")]
        self.parser.state, self.parser.lead = "content", True
        return [Event("tool_call", call=c) for c in calls]

    def finish(self) -> list[Event]:
        """The output ended inside a call message (no <|end_message|>): read what came.  A call already announced
        ends as it was streamed (closed where it stopped; the turn's finish says why, e.g. "length"); one that was
        not is a call if what came is one, else dropped - its marker never reaches the answer's text."""
        return self._close(cut=True) if self.state == "json" else []


class OutputParser:
    """Incremental parser of the model's text. Feed deltas; get events. A tag split across deltas is held back
    until it is complete, so clients never see `<tool_` or `</thi`."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False,
                 think_token: bool = False, call_ends_think: bool = False, no_calls: bool = False,
                 xml_calls: bool = False):
        self.state = "reasoning" if thinking else "content"
        # xml_calls (Inkling 2): the answer may write a call in the XML form (<invoke name=..>, bare, in
        # <function_calls> or in <tool_call>) - a call when it names a declared tool, else text (parse_xml_calls)
        self.xml_calls = xml_calls
        self.xml_end = XML_INVOKE_END                   # what ends the XML call being read
        # no_calls (tool_choice "none"): a call the model starts anyway is neither a call nor text - one
        # "calls_refused" event, and everything after it is dropped (the caller ends the answer there)
        self.no_calls = no_calls
        # call_ends_think: a <tool_call> at the top level of the reasoning ends it, as </think> does - the Qwen
        # templates' response_template (2026-11-07: "</think>|(?=<tool_call>)"), for a model that starts its call
        # without closing its thinking.  One inside a code fence or inline code of the reasoning stays reasoning.
        self.call_ends_think = call_ends_think
        self.dsml_tag = DSML_CALLS                      # the block opening seen last (V4 or V4.1 spelling)
        # think_token: the model ends its reasoning with a SPECIAL token the caller reports (feed(..., think_end=True));
        # the text "</think>" inside the reasoning is then the model writing about the tag (DeepSeek V4.1 explaining
        # a chat template), not the end of it
        self.think_token = think_token
        self.buf = ""
        self.lead = False
        self.schemas = {t.get("name"): t for t in tools or []}
        # stream_tools: a tool call is also reported while it is being written - "tool_start" (its name and id) as
        # soon as the name is known, then "tool_args" pieces of its JSON arguments (string parameters character by
        # character; other types whole, once complete) - before the final "tool_call".  Without it, a client sees
        # nothing until the call is complete, which for a large file write can be many minutes.
        self.stream_tools = stream_tools
        # the answer text that has gone out, followed to tell a call from a quoted one (Strata #1058): the open code
        # fence, the current line, and the backticks of the current paragraph
        self.fence, self.line, self.ticks = "", "", 0
        # characters looked at for an end tag or a streamed value, in all: a test hook - a long tool argument fed in
        # small pieces must cost about its length, not its length times the pieces (Maya #626's idea: each new piece
        # searched the whole call again, and a Write of a 100 KB file took minutes of CPU on the way out)
        self.scanned = 0
        self._reset_scan()
        self._reset_body()

    def _track(self, text: str) -> str:
        """Follow answer text that goes out (see fence above).  Returns the text."""
        parts = text.split("\n")
        for k, part in enumerate(parts):
            if k < len(parts) - 1:
                line, self.line = self.line + part, ""
                s = line.lstrip()
                if self.fence:
                    if s.startswith(self.fence * 3):
                        self.fence = ""
                elif s[:3] in ("```", "~~~") and (s[0] * 3) not in s[3:]:
                    self.fence = s[0]
                elif not s:
                    self.ticks = 0
                else:
                    self.ticks = self._carry(line)
            else:
                self.line += part
        return text

    def _carry(self, line: str) -> int:
        """The backticks a finished line leaves open for the next line of its paragraph: 1 when inline code is
        still open at the end of the line, else 0.  A backtick left open with more text after it on the same line
        is a stray mark ("Let me use `run_commands to check."), not code that goes on: carrying it made every
        tool call after it in the paragraph quoted text, and the agent got a turn without its calls.  A backtick at
        the end of a line (only spaces after it) still opens inline code that goes on to the next line."""
        if (self.ticks + line.count("`")) % 2 == 0:
            return 0
        return 1 if not line[line.rfind("`") + 1:].strip() else 0

    def _in_code(self) -> bool:
        """The answer so far leaves the next character inside a code fence or inline code."""
        return bool(self.fence) or (self.ticks + self.line.count("`")) % 2 == 1

    def _reset_body(self):
        """The body of the call being read (the text after <tool_call> or a DSML opening) is kept as pieces, not grown
        as one string, and its end tag is looked for in each new piece plus the tail that a tag split across pieces
        needs - the whole body is joined once, when the end is there."""
        self.body: list[str] = []    # the pieces
        self.blen = 0                # their length
        self.win = ""                # the body's tail not yet known to be free of the end tag
        self.seen = 0                # where `win` starts in the body
        self.bend = -1               # where the end tag starts in the body; -1: not there yet
        # a Qwen-form call whose body ends with </parameter></function> at a </tool_call>: that close may be text of
        # the last value (_qwen_close).  qmode None: not looked at yet, "real": the call ends there, "wait": decided
        # by what follows (qtail, searched from qpos); qforce: the output ended - the close is real
        self.qmode, self.qtail, self.qpos, self.qforce = None, "", 0, False

    def _body_end(self) -> str:
        return DSML_ENDS[self.dsml_tag] if self.state == "dsml" else self.xml_end if self.state == "xml" else CALL_END

    def _body_push(self, text: str) -> None:
        """A piece of the call body (see _reset_body).  With stream_tools the scanner gets it too."""
        self.body.append(text)
        self.blen += len(text)
        if self.bend >= 0:
            self.qtail += text       # (past the end tag: what decides a Qwen-form close, see _qwen_close)
            return
        if self.stream_tools and self.state == "call":
            self.srest += text       # (the scanner sees the body only, never past its end tag)
            self.sfed += len(text)
        if not text:
            return
        end = self._body_end()
        self.win += text
        self.scanned += len(self.win)
        k = self.win.find(end)
        if k >= 0:
            self.bend = self.seen + k
            self.qtail = self.win[k + len(end):]
            if self.stream_tools and self.state == "call" and self.sfed > self.bend:
                over = self.sfed - self.bend
                self.srest = self.srest[:len(self.srest) - over] if len(self.srest) >= over else ""
                self.sfed = self.bend
            return
        keep = len(end) - 1          # a suffix this long may be the start of the end tag
        if len(self.win) > keep:
            self.seen += len(self.win) - keep
            self.win = self.win[-keep:]

    def _body_take(self) -> tuple[str, str]:
        """(the body before its end tag, the text after the tag); the body is forgotten."""
        whole = "".join(self.body)
        body, rest = whole[:self.bend], whole[self.bend + len(self._body_end()):]
        self._reset_body()
        return body, rest

    def _body_text(self) -> str:
        return "".join(self.body)

    def _reset_scan(self):
        self.srest = ""              # the call body not yet consumed by the scanner (only the body: never past its end)
        self.sfed = 0                # the body's characters given to srest so far
        self.sfinal = False          # srest reaches the body's end (a closing triple at its end is the call's own)
        self.sraw = 0                # how far `srest` was searched for the end of a raw value
        self.ss = "name"             # name -> between -> str|raw -> ... -> done
        self.scall = None            # the ToolCall being streamed (its id is reused by the final event)
        self.sfirst = True
        self.sval_started = False
        self.sdeclared = {}
        self.spname = None
        self.glm = False             # the call is in GLM's form (NAME<arg_key>..), not Qwen's (<function=NAME>..)
        self.vend = "</parameter>"   # what ends the value being streamed

    def _declared(self, name: str) -> bool:
        """A call runs only for a tool the request declared: a model quoting a call, or one that writes a stray
        <tool_call> token in its thinking, must not hand the client a call to a tool that does not exist.  A request
        that declares no tools gets a call back as before (the client decides)."""
        return not self.schemas or name in self.schemas

    def _start_call(self, name: str, out: list, args) -> bool:
        if not self._declared(name):
            return False                  # not announced: the whole body decides at its end (text)
        self.scall = ToolCall(name=name, arguments={})
        props = ((self.schemas.get(name) or {}).get("parameters") or {}).get("properties") or {}
        self.sdeclared = {k: _declared_types(v) for k, v in props.items()}
        out.append(Event("tool_start", call=self.scall))
        args("{")
        return True

    def _scan(self) -> list[Event]:
        """Advance the streaming view of the call body (see stream_tools): self.srest holds what the scanner has not
        consumed yet, so each piece is looked at about once however many pieces the call comes in."""
        out = []

        def args(s):
            if s:
                out.append(Event("tool_args", s, call=self.scall))

        def take(n):
            self.srest = self.srest[n:]
        while True:
            rest = self.srest
            if self.ss == "name":
                lead = rest.lstrip()
                if not lead or FUNC_START.startswith(lead):
                    return out          # which form it is: not known yet
                if lead.startswith(FUNC_START):
                    a = rest.find(FUNC_START)
                    b = rest.find(">", a + len(FUNC_START))
                    if b < 0:
                        return out
                    if not self._start_call(rest[a + len(FUNC_START):b], out, args):
                        self.ss = "done"
                        return out
                    take(b + 1)
                    self.ss = "between"
                else:
                    # GLM: the name runs to the first tag (or the line's end); until one has arrived it may go on
                    b = min((i for i in (rest.find("<"), rest.find("\n", len(rest) - len(lead))) if i >= 0),
                            default=-1)
                    if b < 0 or not rest[:b].strip():
                        return out      # (no name before a tag: malformed - the whole call's parse says so)
                    if not self._start_call(rest[:b].strip(), out, args):
                        self.ss = "done"
                        return out
                    take(b)
                    self.glm, self.vend, self.ss = True, ARG_VAL_END, "between"
            elif self.ss == "between" and self.glm:
                stripped = rest.lstrip()
                if not stripped.startswith(ARG_KEY):
                    return out          # the next pair still arriving (or the call's end: the caller closes it)
                k1 = stripped.find(ARG_KEY_END)
                if k1 < 0:
                    return out
                after = stripped[k1 + len(ARG_KEY_END):]
                v0 = len(after) - len(after.lstrip())
                if not after[v0:].startswith(ARG_VAL):
                    return out
                pname = stripped[len(ARG_KEY):k1].strip()
                args(("" if self.sfirst else ",") + json.dumps(pname) + ":")
                self.sfirst = False
                take((len(rest) - len(stripped)) + k1 + len(ARG_KEY_END) + v0 + len(ARG_VAL))
                if _is_string(self.sdeclared.get(pname)):
                    args('"')
                    self.ss, self.sval_started = "str", True     # GLM's values are written as they are
                else:
                    self.ss, self.sraw, self.spname = "raw", 0, pname
            elif self.ss == "between":
                stripped = rest.lstrip()
                take(len(rest) - len(stripped))
                if stripped.startswith("<parameter="):
                    b = stripped.find(">")
                    if b < 0:
                        return out
                    pname = stripped[11:b]
                    args(("" if self.sfirst else ",") + json.dumps(pname) + ":")
                    self.sfirst = False
                    take(b + 1)
                    if _is_string(self.sdeclared.get(pname)):
                        args('"')
                        self.ss, self.sval_started = "str", False
                    else:
                        self.ss, self.sraw, self.spname = "raw", 0, pname
                elif stripped.startswith("</function>"):
                    args("}")
                    take(len("</function>"))
                    self.ss = "done"
                else:
                    return out          # a tag still arriving (or trailing text): wait
            elif self.ss == "str":
                if not self.sval_started:
                    if not rest:
                        return out
                    if rest[0] == "\n":          # the value's leading newline is not part of it
                        take(1)
                        rest = rest[1:]
                    self.sval_started = True
                self.scanned += len(rest)
                end = rest.find(self.vend)
                if end >= 0 and not self.glm:
                    inner = _QWEN_INNER_CLOSE.match(rest, end)
                    if inner:                    # the call's closing tags inside the value: its text (_qwen_close)
                        args(json.dumps(rest[:inner.end()])[1:-1])
                        take(inner.end())
                        continue
                    if not self.sfinal and _close_prefix(rest[end + len(self.vend):]):
                        safe = end - 1 if end > 0 and rest[end - 1] == "\n" else end
                        if safe > 0:             # (may still be one: the value up to the tag goes out)
                            args(json.dumps(rest[:safe])[1:-1])
                            take(safe)
                        return out
                if end >= 0:
                    value = rest[:end]
                    if value.endswith("\n") and not self.glm:
                        value = value[:-1]
                    args(json.dumps(value)[1:-1] + '"')
                    take(end + len(self.vend))
                    self.ss = "between"
                    continue
                safe = len(rest) - self._hold(rest, (self.vend,))
                if safe > 0 and rest[safe - 1] == "\n" and not self.glm:   # may be the newline before </parameter>
                    safe -= 1
                if safe > 0:
                    args(json.dumps(rest[:safe])[1:-1])
                    take(safe)         # (what stays is at most the tag's length: the next piece is searched with it)
                return out
            elif self.ss == "raw":
                # a value of another type goes out whole, once complete: searched from where the last look ended
                self.scanned += len(rest) - self.sraw
                end = rest.find(self.vend, self.sraw)
                if end < 0:
                    self.sraw = max(0, len(rest) - len(self.vend) + 1)
                    return out
                value = rest[:end]
                if not self.glm:
                    if value.startswith("\n"):
                        value = value[1:]
                    if value.endswith("\n"):
                        value = value[:-1]
                v = _arg_value(value, self.sdeclared.get(self.spname))
                args(json.dumps(v, ensure_ascii=False))
                take(end + len(self.vend))
                self.ss, self.sraw = "between", 0
            else:
                return out

    def _close_scan(self) -> list[Event]:
        """A streamed call that ended without a clean </function>: close its JSON so clients can still parse it."""
        out = []
        if self.scall is None or self.ss == "done":
            return out
        tail = ""
        if self.ss == "str":
            tail += '"'
        elif self.ss == "raw":
            tail += json.dumps(self.srest.strip("\n"))
        tail += "}"
        out.append(Event("tool_args", tail, call=self.scall))
        self.ss = "done"
        return out

    def _think_track(self, text: str) -> str:
        """Reasoning text that goes out: followed for code fences when a <tool_call> can end the reasoning."""
        return self._track(text) if self.call_ends_think else text

    def _think_keep(self, tags: tuple[str, ...]) -> int:
        """What the reasoning holds back when a <tool_call> can end it: a partial tag AND the newlines before it (they
        are dropped if a call follows, so streamed and whole outputs agree)."""
        j = len(self.buf) - self._hold(self.buf, tags)
        while j > 0 and self.buf[j - 1] == "\n":
            j -= 1
        return len(self.buf) - j

    def _think_call(self) -> tuple[int, bool]:
        """(where a <tool_call> at the top level of the reasoning starts in buf, pending) (call_ends_think), or
        (-1, False).  Only a call to a declared tool ends the reasoning: pending means its name has not arrived yet
        (hold the text from there); one naming no declared tool stays reasoning.  Only the text before </think>
        counts (THINK_END in the text form; with think_token the end comes as its own token)."""
        end = len(self.buf) if self.think_token else self.buf.find(THINK_END)
        if end < 0:
            end = len(self.buf)
        snap, at = (self.fence, self.line, self.ticks), 0
        try:
            while True:
                c = self.buf.find(CALL_START, at, end)
                if c < 0:
                    return -1, False
                self._track(self.buf[at:c])
                if not self._in_code():
                    name = self._think_call_name(self.buf[c + len(CALL_START):])
                    if name is None:
                        return c, True
                    if self._declared(name):
                        return c, False
                self._track(CALL_START)
                at = c + len(CALL_START)
        finally:
            self.fence, self.line, self.ticks = snap

    _THINK_NAME_HOLD = 256          # characters after a <tool_call> in the reasoning that may still be its opening

    def _think_call_name(self, tail: str) -> str | None:
        """The tool name after a <tool_call> in the reasoning, "" when what follows is no call opening, or None
        while it may still become one."""
        t = tail.lstrip()
        m = re.match(r"<function=([A-Za-z_][A-Za-z0-9_.\-]*)>", t) or _CALL_OPEN_SLIP.match(t) or \
            _CALL_OPEN_NAME.match(t)
        if m:
            return m.group(1)
        if len(tail) < self._THINK_NAME_HOLD and (not t or (t.startswith("<") and ">" not in t)):
            return None
        return ""

    def _hold(self, text: str, tags: tuple[str, ...]) -> int:
        """Length of the longest suffix of `text` that is a proper prefix of one of `tags`."""
        best = 0
        for tag in tags:
            for n in range(1, len(tag)):
                if text.endswith(tag[:n]):
                    best = max(best, n)
        return best

    def feed(self, delta: str, think_end: bool = False) -> list[Event]:
        if self.state == "refused":
            return []
        if self.state in ("call", "dsml", "xml"):
            self._body_push(delta)       # (the text before a call is short: it goes out as it comes; a call's body is not)
        else:
            self.buf += delta
        out: list[Event] = []
        while True:
            if self.state == "reasoning" and self.call_ends_think:
                c, pending = self._think_call()
                if pending and self.think_token and think_end:
                    c, pending = -1, False      # the reasoning ended before the name came: no call
                if pending:                     # a call's name still arriving: the text before it goes out
                    j = c
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("reasoning", self._think_track(self.buf[:j])))
                        self.buf = self.buf[j:]
                    return out
                if c >= 0:
                    text = self.buf[:c].rstrip("\n")
                    if text:
                        out.append(Event("reasoning", text))
                    self.buf = self.buf[c:]
                    self.state, self.lead = "content", True
                    self.fence, self.line, self.ticks = "", "", 0
                    think_end = False
                    continue
            if self.state == "reasoning" and self.think_token:
                if not think_end:   # (no tag to hold back: the end comes as its own token)
                    keep = self._think_keep((CALL_START,)) if self.call_ends_think else 0
                    if len(self.buf) > keep:
                        out.append(Event("reasoning", self._think_track(self.buf[:len(self.buf) - keep])))
                        self.buf = self.buf[len(self.buf) - keep:]
                    return out
                # the special token's text ends this delta (Hy4-flash spells it </think:opensource>)
                tag = THINK_END_HY4 if THINK_END_HY4 in self.buf else THINK_END
                i = self.buf.rfind(tag)
                if i < 0:
                    i = len(self.buf)
                if i:
                    out.append(Event("reasoning", self.buf[:i]))
                self.buf = self.buf[i + len(tag):] if self.buf[i:].startswith(tag) else ""
                self.state, self.lead = "content", True
                self.fence, self.line, self.ticks = "", "", 0
                think_end = False
            elif self.state == "reasoning":
                i = self.buf.find(THINK_END)
                if i < 0:
                    keep = self._think_keep((THINK_END, CALL_START)) if self.call_ends_think else \
                        self._hold(self.buf, (THINK_END,))
                    if len(self.buf) > keep:
                        out.append(Event("reasoning", self._think_track(self.buf[:len(self.buf) - keep])))
                        self.buf = self.buf[len(self.buf) - keep:]
                    return out
                if i:
                    out.append(Event("reasoning", self.buf[:i]))
                self.buf = self.buf[i + len(THINK_END):]
                self.state, self.lead = "content", True
                self.fence, self.line, self.ticks = "", "", 0
            elif self.state == "content":
                if self.lead:                                   # newlines right after </think> or a call
                    stripped = self.buf.lstrip("\n")
                    if not stripped:
                        self.buf = ""
                        return out
                    self.buf, self.lead = stripped, False
                hits = [(i, "call", CALL_START) for i in (self.buf.find(CALL_START),) if i >= 0]
                hits += [(d, "dsml", t) for t in DSML_ENDS for d in (self.buf.find(t),) if d >= 0]
                xml = self.xml_calls and bool(self.schemas)
                if xml:
                    m = _XML_FIND.search(self.buf)
                    if m:
                        hits.append((m.start(), "xml", XML_CALLS if m.group(0) == XML_CALLS else XML_INVOKE))
                if not hits:
                    # Hold a partial tag AND the newlines before it: if a tool call follows, they are dropped,
                    # so emitting them early would make streamed and whole outputs differ.
                    tags = (CALL_START, DSML_CALLS, DSML41_CALLS) + ((XML_CALLS, XML_INVOKE + " ") if xml else ())
                    j = len(self.buf) - self._hold(self.buf, tags)
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self._track(self.buf[:j])))
                        self.buf = self.buf[j:]
                    return out
                i, kind, tag = min(hits)
                # Strata #1058: a <tool_call> inside a code fence or inline code of the answer is a quoted example -
                # text, never a call (an example `rm -rf` must not run).  The text before it decides, so streamed and
                # whole outputs agree; one at the top level, mid-sentence too, is a call as before.
                snap = (self.fence, self.line, self.ticks)
                self._track(self.buf[:i])
                in_code = self._in_code()
                self.fence, self.line, self.ticks = snap
                if in_code:
                    out.append(Event("content", self._track(self.buf[:i + len(tag)])))
                    self.buf = self.buf[i + len(tag):]
                    continue
                if kind == "xml" and tag == XML_INVOKE:
                    # a bare <invoke name="NAME">: a call only when its opening is complete and names a declared tool
                    m = _XML_INVOKE_OPEN.match(self.buf, i)
                    if m is None and ">" not in self.buf[i:] and len(self.buf) - i < self._THINK_NAME_HOLD:
                        j = i                   # (the opening still arriving: the text before it goes out)
                        while j > 0 and self.buf[j - 1] == "\n":
                            j -= 1
                        if j > 0:
                            out.append(Event("content", self._track(self.buf[:j])))
                            self.buf = self.buf[j:]
                        return out
                    if m is None or m.group(1) not in self.schemas:
                        k = i + len(tag) if m is None else m.end()
                        out.append(Event("content", self._track(self.buf[:k])))
                        self.buf = self.buf[k:]
                        continue
                if i and self.buf[:i].strip():
                    out.append(Event("content", self._track(self.buf[:i].rstrip("\n"))))
                # the call ends the line before it (its text went out without the newline): nothing of that line is
                # open - the call was at the top level - so a stray backtick there does not quote the next call
                self.line, self.ticks = "", 0
                if self.no_calls:
                    out.append(Event("calls_refused"))
                    self.state, self.buf = "refused", ""
                    return out
                self.state = kind
                if kind == "dsml":
                    self.dsml_tag = tag
                if kind == "xml":               # (the body keeps its opening: parse_xml_calls reads the <invoke>s)
                    self.xml_end = XML_CALLS_END if tag == XML_CALLS else XML_INVOKE_END
                    rest, self.buf = self.buf[i:], ""
                else:
                    rest, self.buf = self.buf[i + len(tag):], ""
                self._body_push(rest)
            elif self.state == "xml":
                if self.bend < 0:
                    return out
                end = self.xml_end
                body, self.buf = self._body_take()
                calls = parse_xml_calls(body + end, self.schemas)
                if not calls:                       # (a malformed one is text, never a failed request)
                    out.append(Event("content", self._track(body + end)))
                for call in calls:
                    out.append(Event("tool_call", call=call))
                self.state, self.lead = "content", True
            elif self.state == "dsml":
                if self.bend < 0:
                    return out
                body, self.buf = self._body_take()
                try:
                    calls = parse_dsml_calls(body, self.schemas)
                except ValueError:                  # a malformed block is returned as text, never a failed request
                    calls = None
                    out.append(Event("content", self._track(self.dsml_tag + body)))
                for call in calls or []:
                    out.append(Event("tool_call", call=call))
                self.state, self.lead = "content", True
            else:
                i = self.bend
                if i >= 0 and self._qwen_close() is None:
                    if self.stream_tools:
                        out += self._scan()
                    return out                    # (is that close the call's or its last value's: wait)
                i = self.bend
                if self.stream_tools:
                    if i >= 0:                    # scan only the body: the scanner never sees past its end tag
                        self.sfinal = True
                        out += self._scan()
                        out += self._close_scan()
                    else:
                        out += self._scan()
                if i < 0:
                    return out
                body, self.buf = self._body_take()
                fixed = fix_call_open(body, self.schemas)
                try:
                    call = parse_tool_call(fixed, self.schemas.get(call_name(fixed)))
                except ValueError:                  # a malformed call is returned as text, never a failed request
                    call = None
                if call is not None and not self._declared(call.name):
                    call = None                     # a call to a tool the request did not declare: text
                xml_calls = parse_xml_calls(body, self.schemas) if call is None and self.scall is None and \
                    self.xml_calls else []
                if xml_calls:                       # <tool_call><function_calls><invoke name=..>..</tool_call>
                    out += [Event("tool_call", call=c) for c in xml_calls]
                elif call is None and self.scall is None:
                    out.append(Event("content", self._track(CALL_START + body)))
                elif call is None:                  # already announced while streaming: close it as scanned
                    out.append(Event("tool_call", call=self.scall))
                else:
                    if self.scall is not None:
                        call.id = self.scall.id
                    out.append(Event("tool_call", call=call))
                self._reset_scan()
                self.state, self.lead = "content", True

    def in_open_call(self) -> bool:
        """Inside a tool call whose end tag has not come (a Qwen-form close still being decided is not open: the
        call's tags are all there, and an end-of-turn token after them ends the turn)."""
        return self.state in ("call", "dsml", "xml") and self.bend < 0

    def _qwen_close(self) -> bool | None:
        """At a </tool_call> (self.bend) of a Qwen-form call whose body ends with </parameter></function>: True when
        that close is the call's, None while unknown.  The close is text of the last value (a document about tool
        calls) when what follows closes the call again - </parameter></function></tool_call> - before any new
        <tool_call>: the body then reaches to that later close (and it is looked at the same way).  A new <tool_call>
        or the end of the output (qforce) makes it the call's own; so does any other body (GLM's form, no value)."""
        while True:
            if self.qmode is None:
                whole = "".join(self.body)
                head = fix_call_open(whole[:512], self.schemas).lstrip()
                tail = whole[max(0, self.bend - 512):self.bend]
                self.qmode = "wait" if head.startswith(FUNC_START) and _QWEN_BODY_TAIL.search(tail) else "real"
            if self.qmode == "real" or self.qforce:
                return True
            t = self.qtail
            lo = max(0, self.qpos - len(CALL_END))
            c = t.find(CALL_START, lo)
            e = t.find(CALL_END, lo)
            while e >= 0 and (c < 0 or e < c):
                if _QWEN_BODY_TAIL.search(t, max(0, e - 512), e):
                    break
                e = t.find(CALL_END, e + 1)
            else:
                if c >= 0:
                    return True
                self.qpos = len(t)
                return None
            # the close at bend was inside the value: the body goes on to the later one
            nb = self.bend + len(CALL_END) + e
            if self.stream_tools:
                self.srest += "".join(self.body)[self.sfed:nb]
                self.sfed = nb
            self.bend, self.qtail, self.qpos = nb, t[e + len(CALL_END):], 0

    def finish(self) -> list[Event]:
        """End of generation: flush whatever is held (an unterminated tool call is returned as content)."""
        out = []
        if self.state == "refused":
            return out
        if self.state == "call" and self.bend >= 0:   # a Qwen-form close still undecided: the output ended, it is real
            self.qforce = True
            out += self.feed("")
            return out + self.finish()
        if self.state == "call" and self.stream_tools and self.scall is not None:
            out += self._scan()                 # the output ended inside a call that was already announced
            out += self._close_scan()
            out.append(Event("tool_call", call=self.scall))
            self._reset_body()
            self._reset_scan()
            return out
        body = self._body_text() if self.state in ("call", "dsml", "xml") else self.buf
        if self.state in ("call", "xml") and self.xml_calls:
            # the output ended inside an XML call (no </invoke> / </function_calls> / </tool_call> after it): its
            # complete <invoke>s are calls, what follows the last one text
            calls = parse_xml_calls(body, self.schemas)
            if calls:
                out += [Event("tool_call", call=c) for c in calls]
                tail = body[body.rfind(XML_INVOKE_END) + len(XML_INVOKE_END):]
                tail = tail.replace(XML_CALLS_END, "").strip()
                if tail:
                    out.append(Event("content", self._track(tail)))
                self.buf = ""
                self._reset_body()
                self.state = "content"
                return out
        if body or self.state == "dsml":               # (a block that ended at its opening tag stays text too)
            kind = {"reasoning": "reasoning", "content": "content"}.get(self.state, "content")
            text = CALL_START + body if self.state == "call" else self.dsml_tag + body if self.state == "dsml" else body
            out.append(Event(kind, text))
            self.buf = ""
            self._reset_body()
        return out


def reasoning_tail_calls(reasoning: str, tools: list[dict] | None, ends: tuple[str, ...] = (),
                         xml_calls: bool = False) -> list[Event]:
    """The tool calls of a call block the reasoning ENDS with (DeepSeek V4 / V4.1's DSML block, or <tool_call> ...
    </tool_call>), for a turn that then ended with no answer and no call (Service.run): the model wrote its call in its
    thinking and stopped there.  Only a block at the very end counts (whitespace and the turn's end-marker texts
    `ends` - kept end tokens - may follow it); a call written earlier and thought over again is not one.  A block
    that does not parse, or names no declared tool, gives no calls."""
    if not tools:
        return []
    t = reasoning.rstrip()
    while True:                                   # (the end markers a soft end kept as text, and spaces between)
        e = next((s for s in ends if s and t.endswith(s)), None)
        if e is None:
            break
        t = t[:-len(e)].rstrip()
    for opening, close in ((DSML_CALLS, DSML_CALLS_END), (DSML41_CALLS, DSML41_CALLS_END), (CALL_START, CALL_END)):
        if t.endswith(close):
            i = t.rfind(opening)
            if i < 0:
                return []
            p = OutputParser(thinking=False, tools=tools, xml_calls=xml_calls)
            names = {x.get("name") for x in tools}
            return [ev for ev in p.feed(t[i:]) + p.finish() if ev.kind == "tool_call" and ev.call.name in names]
    return []
