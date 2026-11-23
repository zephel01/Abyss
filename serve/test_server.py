"""serve/test_server.py - the max tokens budget over both APIs, against the mock engine (no GPU, no pack).

    python -m unittest serve.test_server -v
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import CTX_SLACK, ByteTokenizer, carry_embeddings, EngineDied, MockEngine, engine_model_path, Service, StrataEngine, request_timings, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096
ANSWER = "x" * 2000                              # longer than the old 1024 fallback: one token per byte


class RecordingEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_max_new = max_new
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class MaxTokens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingEngine(tok, "</think>\n\n" + ANSWER, max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def call(self, api, text="hi", **budget):
        """-> (status, body, prompt tokens, completion tokens); `budget` is merged into the request as given."""
        msgs = [{"role": "user", "content": text}]
        if api == "openai":
            s, b = self.post("/v1/chat/completions", {"model": "m", "messages": msgs, **budget})
            u = b.get("usage", {})
            return s, b, u.get("prompt_tokens"), u.get("completion_tokens")
        s, b = self.post("/v1/messages", {"model": "m", "messages": msgs, **budget})
        u = b.get("usage", {})
        return s, b, u.get("input_tokens"), u.get("output_tokens")

    def test_unset_budget_is_the_rest_of_the_context(self):
        cases = {"openai": [{"max_tokens": -1}, {"max_tokens": 0}, {}, {"max_tokens": None},
                            {"max_completion_tokens": -1}, {"max_completion_tokens": None, "max_tokens": None}],
                 "anthropic": [{"max_tokens": -1}, {"max_tokens": 0}, {}, {"max_tokens": None}]}
        for api, budgets in cases.items():
            for budget in budgets:
                with self.subTest(api=api, budget=budget):
                    s, b, pt, ct = self.call(api, **budget)
                    self.assertEqual(s, 200, b)
                    self.assertEqual(self.engine.last_max_new, CTX - CTX_SLACK - pt)
                    self.assertGreater(ct, 1024)          # the whole answer, not cut at the old 1024 fallback

    def test_explicit_budget_is_honoured(self):
        for api, budget in [("openai", {"max_tokens": 50}), ("openai", {"max_completion_tokens": 50}),
                            ("openai", {"max_completion_tokens": 50, "max_tokens": 9}),
                            ("anthropic", {"max_tokens": 50}), ("openai", {"max_tokens": 1500}),
                            ("anthropic", {"max_tokens": 1500})]:
            with self.subTest(api=api, budget=budget):
                want = budget.get("max_completion_tokens") or budget["max_tokens"]
                s, b, _, ct = self.call(api, **budget)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, want)
                self.assertEqual(ct, want)

    def test_explicit_budget_over_the_context_is_rejected(self):
        s, b, _, _ = self.call("openai", max_tokens=CTX)
        self.assertEqual(s, 400)
        self.assertIn("exceeds the context", b["error"]["message"])

    def test_anthropic_budget_over_the_context_runs_with_the_room(self):
        """Anthropic's max_tokens is a required ceiling: Claude Code sends its model's maximum (128K) whatever the
        context.  Unless fit_max_tokens is false, the request runs with what the context leaves (and says so)."""
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            s, b, pt, _ = self.call("anthropic", max_tokens=128000)
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.last_max_new, CTX - CTX_SLACK - pt)
        self.assertIn("max tokens 128,000 is more than the context leaves", out.getvalue())
        try:
            self.svc.fit_max_tokens = False                 # "fit_max_tokens": false in the config: refused
            s, b, _, _ = self.call("anthropic", max_tokens=CTX)
            self.assertEqual(s, 400)
            self.assertIn("exceeds the context", b["error"]["message"])
        finally:
            self.svc.fit_max_tokens = None

    def test_clamp_below_the_floor_is_prompt_too_long(self):
        """Strata 33cfc83aa's idea: a max_tokens clamped below min(1024, context / 4) left no room for an answer or a
        tool call.  It is refused in the Anthropic API's context-overflow shape, which Claude Code compacts on."""
        import contextlib
        import io
        import re
        _, _, pt0, _ = self.call("anthropic", max_tokens=1)
        overhead = pt0 - len("hi")
        floor = min(1024, CTX // 4)
        for room, ok in ((floor - 1, False), (floor, True)):
            with self.subTest(room=room):
                text = "y" * (CTX - CTX_SLACK - overhead - room)
                with contextlib.redirect_stdout(io.StringIO()):
                    s, b, pt, _ = self.call("anthropic", text=text, max_tokens=128000)
                if ok:
                    self.assertEqual(s, 200, b)
                    self.assertEqual(self.engine.last_max_new, room)
                    continue
                self.assertEqual(s, 400, b)
                self.assertEqual(b["type"], "error")
                self.assertEqual(b["error"]["type"], "invalid_request_error")
                m = re.match(r"prompt is too long: (\d+) tokens > (\d+) maximum", b["error"]["message"])
                self.assertIsNotNone(m, b)
                self.assertEqual(int(m.group(2)), CTX - CTX_SLACK - floor)
                self.assertGreater(int(m.group(1)), int(m.group(2)))
        os.environ["STRATA_FIT_FLOOR"] = "0"                # off: clamped down to what is left, as before
        try:
            text = "y" * (CTX - CTX_SLACK - overhead - 5)
            with contextlib.redirect_stdout(io.StringIO()):
                s, b, _, _ = self.call("anthropic", text=text, max_tokens=128000)
            self.assertEqual(s, 200, b)
            self.assertEqual(self.engine.last_max_new, 5)
        finally:
            del os.environ["STRATA_FIT_FLOOR"]

    def test_no_room_is_prompt_too_long_too(self):
        _, _, pt0, _ = self.call("anthropic", max_tokens=1)
        text = "y" * (CTX - CTX_SLACK - (pt0 - len("hi")))
        s, b, _, _ = self.call("anthropic", text=text)
        self.assertEqual(s, 400, b)
        self.assertTrue(b["error"]["message"].startswith("prompt is too long: "), b)
        self.assertIn("no room to answer", b["error"]["message"])

    def test_unset_budget_with_a_near_full_prompt(self):
        _, _, pt0, _ = self.call("openai", max_tokens=1)
        overhead = pt0 - len("hi")                  # the template's tokens around the user text
        for api in ("openai", "anthropic"):
            _, _, pa, _ = self.call(api, max_tokens=1)
            over = pa - pt0                          # the Anthropic template may differ slightly
            with self.subTest(api=api, room=5):     # a few tokens left: the budget is exactly those
                text = "y" * (CTX - CTX_SLACK - overhead - over - 5)
                s, b, pt, ct = self.call(api, text=text, max_tokens=-1)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, 5)
                self.assertEqual(ct, 5)
            with self.subTest(api=api, room=0):     # nothing left: rejected, not truncated
                text = "y" * (CTX - CTX_SLACK - overhead - over)
                s, b, _, _ = self.call(api, text=text)
                self.assertEqual(s, 400, b)
                self.assertIn("no room to answer", b["error"]["message"])

    def test_debug_log_shows_the_resolved_budget(self):
        import contextlib
        import io
        os.environ["STRATA_DEBUG"] = "1"
        try:
            for api in ("openai", "anthropic"):
                with self.subTest(api=api):
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        _, _, pt, _ = self.call(api, max_tokens=-1)
                    self.assertIn(f"max_new={CTX - CTX_SLACK - pt} ", out.getvalue())
        finally:
            del os.environ["STRATA_DEBUG"]


class FitMaxTokens(unittest.TestCase):
    """PR #24: --fit-max-tokens clamps an explicit budget that overshoots the context instead of a 400."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingEngine(tok, "</think>\n\n" + ANSWER, max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), fit_max_tokens=True)
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    post = MaxTokens.post
    call = MaxTokens.call

    def test_overshoot_is_clamped_to_the_room(self):
        for api in ("openai", "anthropic"):
            with self.subTest(api=api):
                s, b, pt, ct = self.call(api, max_tokens=CTX)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, CTX - CTX_SLACK - pt)

    def test_a_budget_that_fits_is_unchanged(self):
        s, b, _, ct = self.call("openai", max_tokens=50)
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.last_max_new, 50)

    def test_openai_below_the_floor_is_context_length_exceeded(self):
        import contextlib
        import io
        _, _, pt0, _ = self.call("openai", max_tokens=1)
        text = "y" * (CTX - CTX_SLACK - (pt0 - len("hi")) - 100)
        with contextlib.redirect_stdout(io.StringIO()):
            s, b, _, _ = self.call("openai", text=text, max_tokens=CTX)
        self.assertEqual(s, 400, b)
        self.assertEqual(b["error"]["code"], "context_length_exceeded")
        self.assertTrue(b["error"]["message"].startswith("prompt is too long: "), b)
        s, b, _, _ = self.call("openai", text=text, max_tokens=50)     # a budget that fits still runs
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.last_max_new, 50)

    def test_no_room_is_still_a_400(self):
        _, _, pt0, _ = self.call("openai", max_tokens=1)
        overhead = pt0 - len("hi")
        s, b, _, _ = self.call("openai", text="y" * (CTX - CTX_SLACK - overhead), max_tokens=100)
        self.assertEqual(s, 400, b)
        self.assertIn("no room to answer", b["error"]["message"])


class ImageMarkers(unittest.TestCase):
    """#150: the text "<|image_pad|>" inside a message is text, not an image's place."""

    class FakeVision:
        def __init__(self, d):
            self.dir = Path(d)
            self.rows = self.dir / "img.sve"
            self.rows.write_bytes(b"rows")

        owed = False   # (server.Vision: GPU memory an on-demand encode borrowed and has not given back)

        def encode(self, source):
            return self.rows, 3

        def active(self):
            import contextlib
            return contextlib.nullcontext(self)

    def test_literal_marker_with_an_image(self):
        import tempfile
        tok = ByteTokenizer()
        with tempfile.TemporaryDirectory() as d:
            svc = Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"),
                          vision=self.FakeVision(d))
            pad = tok.encode("<|image_pad|>", parse_special=True)[0]
            for text in ("the docs say <|image_pad|> marks an image", "plain"):
                with self.subTest(text=text):
                    msgs = [{"role": "user", "content": [{"type": "text", "text": text},
                                                         {"type": "image", "source": "x.png"}]}]
                    ids, _, _ = svc.prepare(msgs, None, {})
                    self.assertEqual(ids.count(pad), 3)          # the image's three rows, nothing else
                    self.assertIn("<|image_pad|> marks" if "docs" in text else "plain", tok.decode(ids))
            svc.embeddings.path.unlink(missing_ok=True)


class ToolResultImages(unittest.TestCase):
    """#41: an image inside a tool result (Anthropic tool_result blocks, OpenAI role "tool" image parts) reaches the
    prompt as an image, through the whole path (conversion, the model's template, the image rows); a server without the
    image encoder says so (a 400), it never drops the picture silently."""

    class GlmTok(ByteTokenizer):
        SPECIALS = ByteTokenizer.SPECIALS + ["<|begin_of_image|>", "<|image|>", "<|end_of_image|>"]

    IMG = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo"}}
    TOOLS = [{"name": "read_image", "description": "Read an image file",
              "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}]

    def anthropic(self, result, extra=()):
        from serve.frontend import anthropic_to_messages
        return anthropic_to_messages({"tools": self.TOOLS, "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Open check.png"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read_image",
                                               "input": {"path": "check.png"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": result}, *extra]}]})

    def openai(self, parts):
        from serve.frontend import openai_to_messages
        return openai_to_messages({"messages": [
            {"role": "user", "content": "Open check.png"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "read_image", "arguments": "{\"path\": \"check.png\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": parts}]})

    def rows(self, converted, template, tok, vision=True):
        """-> (the prompt's image rows, the prompt as text): every image is 3 rows (FakeVision)."""
        import tempfile
        msgs, tools, kwargs = converted
        with tempfile.TemporaryDirectory() as d:
            svc = Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate(template),
                          vision=ImageMarkers.FakeVision(d) if vision else None)
            glm = "<|begin_of_image|>" in svc.template.source
            pad = tok.encode("<|image|>" if glm else "<|image_pad|>", parse_special=True)[0]
            ids, _, _ = svc.prepare(msgs, tools, kwargs)
            if svc.embeddings.path:
                svc.embeddings.path.unlink(missing_ok=True)
            return ids.count(pad), tok.decode(ids)

    def templates(self):
        return [("qwen", ROOT / "serve/chat_template.jinja", ByteTokenizer()),
                ("glm", ROOT / "tools/glm_chat_template.jinja", self.GlmTok())]

    def test_anthropic_tool_result_images(self):
        for name, tpl, tok in self.templates():
            with self.subTest(template=name):
                self.assertEqual(self.rows(self.anthropic([self.IMG]), tpl, tok)[0], 3)
                self.assertEqual(self.rows(self.anthropic([self.IMG, self.IMG]), tpl, tok)[0], 6)

    def test_anthropic_mixed_text_and_images(self):
        for name, tpl, tok in self.templates():
            with self.subTest(template=name):
                n, text = self.rows(self.anthropic(
                    [{"type": "text", "text": "Read image file check.png [image/png, 640x360]"}, self.IMG],
                    extra=[{"type": "text", "text": "and this one"}, self.IMG]), tpl, tok)
                self.assertEqual(n, 6)   # the tool's image and the top-level one next to it
                self.assertIn("Read image file check.png", text)
                self.assertIn("and this one", text)
                self.assertLess(text.index("Read image file"), text.index("and this one"))
                n, text = self.rows(self.anthropic("plain result"), tpl, tok)   # text only: unchanged
                self.assertEqual(n, 0)
                self.assertIn("plain result", text)

    def test_openai_tool_message_images(self):
        url = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"}}
        for name, tpl, tok in self.templates():
            with self.subTest(template=name):
                n, text = self.rows(self.openai([{"type": "text", "text": "Screenshot"}, url]), tpl, tok)
                self.assertEqual(n, 3)
                self.assertIn("Screenshot", text)

    def test_without_the_encoder_a_clear_error(self):
        for name, tpl, tok in self.templates():
            with self.subTest(template=name):
                with self.assertRaisesRegex(ValueError, "without the vision encoder"):
                    self.rows(self.anthropic([self.IMG]), tpl, tok, vision=False)


class V41ImageMarkers(unittest.TestCase):
    """DeepSeek V4.1: an image is one <｜deepseek_image｜> token in the prompt, expanded to the encoder's n rows; the same
    text typed inside a message stays text.  (A GGUF template prints a message's content as it is.)"""

    class V41Tok(ByteTokenizer):
        SPECIALS = ByteTokenizer.SPECIALS + ["<｜deepseek_image｜>"]

    TEMPLATE = ("{#｜DSML｜#}{%- for m in messages -%}<｜User｜>{{ m['content'] }}{%- endfor -%}"
                "{%- if add_generation_prompt -%}<｜Assistant｜>{%- endif -%}")

    def test_one_marker_becomes_n_rows(self):
        import tempfile
        tok = self.V41Tok()
        with tempfile.TemporaryDirectory() as d:
            tpl = Path(d) / "chat_template.jinja"
            tpl.write_text(self.TEMPLATE, encoding="utf-8")
            svc = Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate(tpl),
                          vision=ImageMarkers.FakeVision(d))
            self.assertTrue(svc.template.dsv4)
            pad = tok.encode("<｜deepseek_image｜>", parse_special=True)[0]
            for text in ("the docs say <｜deepseek_image｜> marks an image", "plain"):
                with self.subTest(text=text):
                    msgs = [{"role": "user", "content": [{"type": "image", "source": "x.png"},
                                                         {"type": "text", "text": text}]}]
                    ids, _, _ = svc.prepare(msgs, None, {})
                    self.assertEqual(ids.count(pad), 3)          # the image's three rows, nothing else
                    self.assertIn("<｜deepseek_image｜> marks" if "docs" in text else "plain", tok.decode(ids))
                    self.assertNotIn(None, ids)
                    # the rows come first (the image block precedes the text), after the user's marker
                    first = ids.index(pad)
                    self.assertEqual(ids[first:first + 3], [pad] * 3)
            svc.embeddings.path.unlink(missing_ok=True)


class V41OfficialTemplateImages(unittest.TestCase):
    """DeepSeek V4.1 with the official template, which writes <｜deepseek_image｜> for an image block itself (and refuses
    the marker typed in a message): the image still becomes the encoder's n rows, not a 400."""

    TEMPLATE = ('{#｜DSML｜#}{%- set IMG = "<｜deepseek_image｜>" -%}'
                "{%- for m in messages -%}{%- set c = m['content'] -%}"
                "{%- if c is string and IMG in c -%}{{ raise_exception('image token in content') }}{%- endif -%}"
                "{%- if c is not string -%}{%- set sp = namespace(parts=[]) -%}{%- for b in c -%}"
                "{%- if b.type == 'text' -%}{%- set sp.parts = sp.parts + [b.text] -%}"
                "{%- elif b.type in ['image', 'image_url'] -%}{%- set sp.parts = sp.parts + [IMG] -%}{%- endif -%}"
                "{%- endfor -%}{%- set c = sp.parts | join('\\n\\n') -%}{%- endif -%}"
                "<｜User｜>{{ c }}{%- endfor -%}{%- if add_generation_prompt -%}<｜Assistant｜>{%- endif -%}")

    def test_an_image_block_becomes_n_rows(self):
        import tempfile
        tok = V41ImageMarkers.V41Tok()
        with tempfile.TemporaryDirectory() as d:
            tpl = Path(d) / "chat_template.jinja"
            tpl.write_text(self.TEMPLATE, encoding="utf-8")
            svc = Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate(tpl),
                          vision=ImageMarkers.FakeVision(d))
            self.assertTrue(svc.template.dsv4)
            self.assertIn("<｜deepseek_image｜>", svc.template.source)
            pad = tok.encode("<｜deepseek_image｜>", parse_special=True)[0]
            for n in (1, 2):
                with self.subTest(images=n):
                    blocks = [{"type": "image", "source": "x.png"}] * n + [{"type": "text", "text": "what is it"}]
                    ids, _, _ = svc.prepare([{"role": "user", "content": blocks}], None, {})
                    self.assertEqual(ids.count(pad), 3 * n)      # each image's three rows
                    self.assertNotIn(None, ids)
                    self.assertIn("what is it", tok.decode(ids))
            svc.embeddings.path.unlink(missing_ok=True)


class V4JapaneseSystemPrompt(unittest.TestCase):
    """DeepSeek V4 0731 answers a Japanese question in Chinese without a system prompt: a Japanese request with none gets
    JA_SYSPROMPT as its system message (V4 0731 only; STRATA_JA_SYSPROMPT=0 turns it off)."""

    V4 = ("{#｜DSML｜#}{%- for m in messages -%}{%- if m['role'] == 'system' -%}[SYS]{{ m['content'] }}"
          "{%- else -%}<｜User｜>{{ m['content'] }}{%- endif -%}{%- endfor -%}"
          "{%- if add_generation_prompt -%}<｜Assistant｜>{%- endif -%}")
    V41 = "{#｜DSML｜ reasoning_effort is integer effort_map #}" + V4
    PLAIN = V4.replace("{#｜DSML｜#}", "")

    def prompt(self, source, messages, env=None, dsml41=False):
        import tempfile
        from unittest import mock
        tok = ByteTokenizer()
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, env or {}):
            os.environ.pop("STRATA_JA_SYSPROMPT", None) if env is None else None
            tpl = Path(d) / "chat_template.jinja"
            tpl.write_text(source, encoding="utf-8")
            svc = Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate(tpl, dsml41=dsml41))
            ids, _, _ = svc.prepare(messages, None, {})
            return tok.decode(ids)

    def test_helper(self):
        from serve.frontend import JA_SYSPROMPT, ja_system_prompt
        ja = [{"role": "user", "content": "富士山の高さは？"}]
        self.assertEqual(ja_system_prompt(ja, env={}), [{"role": "system", "content": JA_SYSPROMPT}] + ja)
        parts = [{"role": "user", "content": [{"type": "text", "text": "これは何ですか"}]}]
        self.assertEqual(ja_system_prompt(parts, env={})[0]["content"], JA_SYSPROMPT)
        for msgs in ([{"role": "system", "content": "be brief"}] + ja,           # the client's own system message
                     [{"role": "developer", "content": "x"}] + ja,
                     [{"role": "user", "content": "How tall is Mt. Fuji?"}],      # English
                     [{"role": "user", "content": "北京是中国的首都吗"}],          # Chinese: no kana
                     ja + [{"role": "assistant", "content": "3776 m"}, {"role": "user", "content": "and K2?"}],
                     []):
            with self.subTest(msgs=msgs):
                self.assertIs(ja_system_prompt(msgs, env={}), msgs)
        for off in ("0", "off", "false"):
            self.assertIs(ja_system_prompt(ja, env={"STRATA_JA_SYSPROMPT": off}), ja)

    def test_v4_0731_only(self):
        from serve.frontend import JA_SYSPROMPT
        ja = [{"role": "user", "content": "日本の三大随筆は？"}]
        self.assertIn("[SYS]" + JA_SYSPROMPT, self.prompt(self.V4, ja))
        self.assertNotIn("[SYS]", self.prompt(self.V4, [{"role": "user", "content": "hello"}]))
        self.assertNotIn("[SYS]", self.prompt(self.V4, ja, env={"STRATA_JA_SYSPROMPT": "0"}))
        self.assertNotIn("[SYS]", self.prompt(self.V41, ja))                 # V4.1's template
        self.assertNotIn("[SYS]", self.prompt(self.V4, ja, dsml41=True))     # a V4.1 pack (engram.txt)
        self.assertNotIn("[SYS]", self.prompt(self.PLAIN, ja))               # not DeepSeek V4
        own = [{"role": "system", "content": "簡潔に"}] + ja
        self.assertEqual(self.prompt(self.V4, own).count("[SYS]"), 1)


class GlmThinkingLevels(unittest.TestCase):
    """GLM-5.x's template knows reasoning_effort low / high and is "Max" otherwise (no effort given included), and it
    always opens <think>: the server's levels map onto low / high / max, and thinking off closes the think block."""

    SRC = ("[gMASK]{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and "
           "reasoning_effort in ['low', 'high'] else 'max' -%}E={{ effective_reasoning_effort }}"
           "{%- if add_generation_prompt -%}<|assistant|>{{- '<think>' -}}{%- endif -%}")

    def render(self, **kwargs):
        import tempfile
        from serve.frontend import effort_kwargs
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "chat_template.jinja"
            p.write_text(self.SRC, encoding="utf-8")
            t = ChatTemplate(p)
            self.assertTrue(t.glm_effort)
            return t.render([{"role": "user", "content": "hi"}], **effort_kwargs(kwargs.get("effort")))

    def test_levels(self):
        # GLM's own names: Off, Low, High, Max (no effort given: the template's Max)
        self.assertEqual(self.render(), "[gMASK]E=max<|assistant|><think>")
        self.assertEqual(self.render(effort="none"), "[gMASK]E=low<|assistant|><think></think>")
        self.assertEqual(self.render(effort="low"), "[gMASK]E=low<|assistant|><think>")
        self.assertEqual(self.render(effort="high"), "[gMASK]E=high<|assistant|><think>")
        self.assertEqual(self.render(effort="max"), "[gMASK]E=max<|assistant|><think>")

    def test_other_spellings(self):
        # OpenAI's medium (GLM has none) is High; xhigh / maximum are Max; minimal is off
        self.assertEqual(self.render(effort="medium"), "[gMASK]E=high<|assistant|><think>")
        self.assertEqual(self.render(effort="xhigh"), "[gMASK]E=max<|assistant|><think>")
        self.assertEqual(self.render(effort="minimal"), "[gMASK]E=low<|assistant|><think></think>")

    def test_level_names_and_budgets(self):
        from serve.frontend import budget_effort, effort_level
        self.assertEqual([effort_level(v) for v in ("none", "low", "medium", "high", "max", "xhigh", False, "?")],
                         ["none", "low", "high", "high", "max", "max", "none", None])
        self.assertEqual([budget_effort(n)["reasoning_effort"] for n in (1024, 4096, 16384)], ["low", "high", "max"])


class InklingThinkingLevels(unittest.TestCase):
    """Inkling 2's template maps reasoning_effort none / minimal / low / ... to 0.0 / 0.1 / 0.2 / ... (0.9 when
    absent): "minimal" is the template's 0.1, not thinking off (which every other model keeps)."""

    SRC = ('{%- set effort_map = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, '
           '"xhigh": 0.99, "max": 0.99} -%}'
           "{%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}"
           "Thinking effort level: {{ effort_map[eff] if eff is string else eff }}")

    def render(self, effort=None):
        import tempfile
        from serve.frontend import effort_kwargs
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "chat_template.jinja"
            p.write_text(self.SRC, encoding="utf-8")
            t = ChatTemplate(p)
            self.assertTrue(t.ink)
            return t.render([{"role": "user", "content": "hi"}], **effort_kwargs(effort))

    def test_levels(self):
        self.assertEqual(self.render(), "Thinking effort level: 0.9")
        self.assertEqual(self.render("none"), "Thinking effort level: 0.0")
        self.assertEqual(self.render("minimal"), "Thinking effort level: 0.1")
        self.assertEqual(self.render("low"), "Thinking effort level: 0.2")
        self.assertEqual(self.render("medium"), "Thinking effort level: 0.7")
        self.assertEqual(self.render("max"), "Thinking effort level: 0.99")
        self.assertEqual(self.render(False), "Thinking effort level: 0.0")

    def test_minimal_is_off_elsewhere(self):
        from serve.frontend import effort_kwargs
        self.assertIs(effort_kwargs("minimal")["enable_thinking"], False)
        self.assertEqual(V41ThinkingLevels("test_levels").render("minimal")[:1], "C")   # DeepSeek: thinking off


class V41ThinkingLevels(unittest.TestCase):
    """DeepSeek V4.1 1017's template takes reasoning_effort low / medium / high / max or an integer 1..100 (high when
    absent): the client's own value reaches it; other templates get the nearest of the server's levels."""

    SRC = ("{%- if thinking is defined and thinking -%}T{%- else -%}C{%- endif -%}"
           "{%- if reasoning_effort is not defined or reasoning_effort is none -%}{%- set reasoning_effort = 'high' -%}"
           "{%- endif -%}{%- set effort_map = {'low': 50, 'medium': 62, 'high': 75, 'max': 100} -%}"
           "{%- if reasoning_effort is string and reasoning_effort in effort_map -%}{{ effort_map[reasoning_effort] }}"
           "{%- elif reasoning_effort is integer -%}{{ reasoning_effort }}{%- endif -%}\uff5cDSML\uff5c")

    def render(self, effort=None):
        import tempfile
        from serve.frontend import effort_kwargs
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "chat_template.jinja"
            p.write_text(self.SRC, encoding="utf-8")
            t = ChatTemplate(p)
            self.assertTrue(t.dsv4 and t.v41_effort)
            return t.render([{"role": "user", "content": "hi"}], **effort_kwargs(effort))[:4]

    def test_levels(self):
        self.assertEqual(self.render(), "T75\uff5c")             # (thinks, the template's high)
        self.assertEqual(self.render("low"), "T50\uff5c")
        self.assertEqual(self.render("medium"), "T62\uff5c")
        self.assertEqual(self.render("high"), "T75\uff5c")
        self.assertEqual(self.render("max"), "T100")
        self.assertEqual(self.render("xhigh"), "T100")
        self.assertEqual(self.render(40), "T40\uff5c")
        self.assertEqual(self.render("90"), "T90\uff5c")
        self.assertTrue(self.render("none").startswith("C"))

    def test_integers_elsewhere(self):
        from serve.frontend import effort_kwargs
        self.assertEqual(effort_kwargs(50)["reasoning_effort"], "low")
        self.assertEqual(effort_kwargs(75)["reasoning_effort"], "high")
        self.assertEqual(effort_kwargs(100)["reasoning_effort"], "max")
        with self.assertRaises(ValueError):
            effort_kwargs(0)
        with self.assertRaises(ValueError):
            effort_kwargs(101)


class SpelledStopMarker(unittest.TestCase):
    """GLM-5.3 sometimes writes its turn's end, <|user|>, out of ordinary pieces: the answer must end there, and a
    '<|' that is not a marker must still reach the client."""

    class GlmTok(ByteTokenizer):
        SPECIALS = ByteTokenizer.SPECIALS + ["<|user|>"]

    class Spelled(MockEngine):
        def __init__(self, tok, text):
            super().__init__(tok, "", max_context=CTX)
            self.script = list(text.encode("utf-8"))          # byte pieces only: no special id ever arrives

    def answer(self, text):
        tok = self.GlmTok()
        svc = Service(self.Spelled(tok, text), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        out, done = "", None
        for kind, v in svc.run(tok.encode("hi"), False, None, 200, {}, threading.Event()):
            if kind == "event":
                out += v.text
            elif kind == "done":
                done = v
        return out, done["finish"]

    def test_marker_ends_the_answer(self):
        self.assertEqual(self.answer("71 is the largest.<|user|>The user asked"), ("71 is the largest.", "stop"))

    def test_lookalikes_pass(self):
        self.assertEqual(self.answer("a <|b|> and x<|use"), ("a <|b|> and x<|use", "length"))


class ModelOpensThink(unittest.TestCase):
    """A template that leaves <think> to the model (MiMo-V2.6-Distill-Qwen-9B's ends the prompt at
    "<|im_start|>assistant\n"): the model's own first <think> is not reasoning text, and an answer that does not start
    with it is content.  A prompt that opens the reasoning itself keeps the old behaviour."""

    class ThinkTok(ByteTokenizer):
        SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>"]

    def answer(self, script, prompt):
        tok = self.ThinkTok()
        svc = Service(MockEngine(tok, script, max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        out = {"reasoning": "", "content": ""}
        for kind, v in svc.run(tok.encode(prompt, parse_special=True), True, None, 200, {}, threading.Event()):
            if kind == "event" and v.kind in out:
                out[v.kind] += v.text
        return out["reasoning"], out["content"]

    def test_model_written_think(self):
        self.assertEqual(self.answer("<think>plan</think>Tokyo.", "<|im_start|>assistant\n"), ("plan", "Tokyo."))

    def test_no_think_is_content(self):
        self.assertEqual(self.answer("Tokyo.", "<|im_start|>assistant\n"), ("", "Tokyo."))

    def test_prompt_opened_think(self):
        self.assertEqual(self.answer("plan</think>Tokyo.", "<|im_start|>assistant\n<think>\n"), ("plan", "Tokyo."))


class InklingThinkingOff(unittest.TestCase):
    """Inkling 2 with thinking off (reasoning_effort none) still writes a reasoning message: it must not reach the
    answer, and a turn without one is all content."""

    class InkTok(ByteTokenizer):
        SPECIALS = ByteTokenizer.SPECIALS + ["<|message_model|>", "<|content_thinking|>", "<|content_text|>",
                                             "<|end_message|>", "<|content_model_end_sampling|>"]

    def answer(self, script):
        tok = self.InkTok()
        svc = Service(MockEngine(tok, script, max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        out = {"reasoning": "", "content": ""}
        for kind, v in svc.run(tok.encode("hi"), False, None, 200, {}, threading.Event()):
            if kind == "event" and v.kind in out:
                out[v.kind] += v.text
        return out["reasoning"], out["content"]

    def test_reasoning_message_with_thinking_off(self):
        self.assertEqual(self.answer("<|message_model|><|content_thinking|>plan<|end_message|>"
                                     "<|message_model|><|content_text|>Tokyo.<|end_message|>"
                                     "<|content_model_end_sampling|>"), ("plan", "Tokyo."))

    def test_no_reasoning_message(self):
        self.assertEqual(self.answer("<|message_model|><|content_text|>Tokyo.<|end_message|>"
                                     "<|content_model_end_sampling|>"), ("", "Tokyo."))


class InklingToolCalls(unittest.TestCase):
    """Inkling 2's call messages (<|message_model|>NAME<|content_invoke_tool_json|>{"name","args"}<|end_message|>)
    become the server's tool calls; anything else in that shape stays text."""

    class InkTok(ByteTokenizer):
        SPECIALS = ByteTokenizer.SPECIALS + ["<|message_model|>", "<|content_thinking|>", "<|content_text|>",
                                             "<|end_message|>", "<|content_model_end_sampling|>",
                                             "<|content_invoke_tool_json|>", "<|content_invoke_tool_text|>",
                                             "<|content_tool_error|>"]

    TOOLS = [{"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]

    def answer(self, script, tools=TOOLS, prompt="hi", sampling=None):
        tok = self.InkTok()
        svc = Service(MockEngine(tok, script, max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        out = {"reasoning": "", "content": "", "calls": []}
        for kind, v in svc.run(tok.encode(prompt, parse_special=True), True, tools, 200, sampling or {},
                               threading.Event()):
            if kind == "event" and v.kind == "tool_call":
                out["calls"].append((v.call.name, v.call.arguments))
            elif kind == "event" and v.kind in out:
                out[v.kind] += v.text
        return out

    CALL = ('<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Tokyo"}}'
            '<|end_message|>')

    def test_call_after_reasoning(self):
        out = self.answer("<|message_model|><|content_thinking|>look it up<|end_message|>" + self.CALL +
                          "<|content_model_end_sampling|>")
        self.assertEqual(out, {"reasoning": "look it up", "content": "", "calls": [("get_weather", {"city": "Tokyo"})]})

    def test_text_then_two_calls(self):
        out = self.answer("<|message_model|><|content_text|>Checking.<|end_message|>" + self.CALL +
                          self.CALL.replace("Tokyo", "Osaka") + "<|content_model_end_sampling|>")
        self.assertEqual(out["content"], "Checking.")
        self.assertEqual(out["calls"], [("get_weather", {"city": "Tokyo"}), ("get_weather", {"city": "Osaka"})])

    def test_bad_json_and_unknown_tool_are_text(self):
        out = self.answer('<|message_model|>get_weather<|content_invoke_tool_json|>{"name":<|end_message|>'
                          "<|content_model_end_sampling|>")
        self.assertEqual(out["calls"], [])
        self.assertEqual(out["content"], 'get_weather<|content_invoke_tool_json|>{"name":')
        out = self.answer(self.CALL.replace("get_weather", "rm_rf") + "<|content_model_end_sampling|>")
        self.assertEqual(out["calls"], [])

    def test_unterminated_call_at_the_end(self):
        out = self.answer(self.CALL.replace("<|end_message|>", ""))
        self.assertEqual(out["calls"], [("get_weather", {"city": "Tokyo"})])

    def test_required_call_opened_in_the_prompt(self):
        from serve.frontend import ink_call_open
        out = self.answer('{"name":"get_weather","args":{"city":"Kyoto"}}<|end_message|><|content_model_end_sampling|>',
                          prompt="hi" + ink_call_open("get_weather"), sampling={"_prefill": ink_call_open("get_weather")})
        self.assertEqual(out["calls"], [("get_weather", {"city": "Kyoto"})])

    def test_tool_results_named_for_the_template(self):
        from serve.frontend import ink_tool_names, openai_to_messages
        msgs, _, _ = openai_to_messages({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'}},
                {"id": "b", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "sunny"},
            {"role": "tool", "tool_call_id": "b", "content": "noon"}]})
        named = ink_tool_names(msgs)
        self.assertEqual([m.get("name") for m in named if m["role"] == "tool"], ["get_weather", "get_time"])
        self.assertEqual(named[1]["tool_calls"][0]["function"]["arguments"], {"city": "Tokyo"})   # an object
        self.assertNotIn("name", msgs[2])                                                          # (a copy)

    def test_thinking_after_text_or_call_is_reasoning(self):
        # the messages come in any order: a thinking message after the answer's text or a call is reasoning too
        out = self.answer("<|message_model|><|content_text|>Checking.<|end_message|>"
                          "<|message_model|><|content_thinking|>now the call<|end_message|>" + self.CALL +
                          "<|message_model|><|content_thinking|>and done<|end_message|>"
                          "<|message_model|><|content_text|>Sunny.<|end_message|><|content_model_end_sampling|>")
        self.assertEqual(out, {"reasoning": "now the calland done", "content": "Checking.Sunny.",
                               "calls": [("get_weather", {"city": "Tokyo"})]})

    def test_other_argument_keys(self):
        for key in ("arguments", "parameters"):
            out = self.answer(self.CALL.replace('"args"', '"' + key + '"') + "<|content_model_end_sampling|>")
            self.assertEqual(out["calls"], [("get_weather", {"city": "Tokyo"})], key)
        out = self.answer(self.CALL.replace('{"city":"Tokyo"}', '"{\\"city\\":\\"Tokyo\\"}"') +
                          "<|content_model_end_sampling|>")
        self.assertEqual(out["calls"], [("get_weather", {"city": "Tokyo"})])
        out = self.answer(self.CALL.replace('{"city":"Tokyo"}', '"Tokyo"') + "<|content_model_end_sampling|>")
        self.assertEqual(out["calls"], [])                                    # (a plain string: not arguments)

    def test_xml_call_in_a_text_message(self):
        # Inkling drifts into the XML form (Basalt #540: 18 of 50 answers): a call for a declared tool, else text
        xml = '<invoke name="get_weather">\n<parameter name="city">Tokyo</parameter>\n</invoke>'
        out = self.answer("<|message_model|><|content_text|>Checking.\n" + xml + "<|end_message|>"
                          "<|content_model_end_sampling|>")
        self.assertEqual(out, {"reasoning": "", "content": "Checking.", "calls": [("get_weather", {"city": "Tokyo"})]})
        out = self.answer("<|message_model|><|content_text|><tool_call>\n<function_calls>\n" + xml +
                          "\n</function_calls>\n</tool_call><|end_message|><|content_model_end_sampling|>")
        self.assertEqual(out["calls"], [("get_weather", {"city": "Tokyo"})])
        out = self.answer("<|message_model|><|content_text|>" + xml.replace("get_weather", "rm_rf") +
                          "<|end_message|><|content_model_end_sampling|>")
        self.assertEqual(out["calls"], [])

    def test_xml_call_in_a_call_message(self):
        xml = '<invoke name="get_weather">\n<parameter name="city">Osaka</parameter>\n</invoke>'
        for marker in ("<|content_invoke_tool_json|>", "<|content_invoke_tool_text|>"):
            out = self.answer("<|message_model|>get_weather" + marker + xml + "<|end_message|>"
                              "<|content_model_end_sampling|>")
            self.assertEqual(out["calls"], [("get_weather", {"city": "Osaka"})], marker)

    def test_tool_text_and_tool_error_messages_are_seen(self):
        # <|content_invoke_tool_text|> (a call written as text) and <|content_tool_error|> are message kinds the
        # template never writes: what the model puts in them is visible text, never dropped as a header
        out = self.answer("<|message_model|>get_weather<|content_invoke_tool_text|>Tokyo<|end_message|>"
                          "<|content_model_end_sampling|>")
        self.assertEqual(out, {"reasoning": "", "content": "get_weather<|content_invoke_tool_text|>Tokyo", "calls": []})
        out = self.answer("<|message_model|><|content_tool_error|>no such city<|end_message|>"
                          "<|content_model_end_sampling|>")
        self.assertEqual(out, {"reasoning": "", "content": "no such city", "calls": []})


class ClientShapes(unittest.TestCase):
    """What real clients send: Claude Code posts /v1/messages?beta=true (issue #55) and puts hook context into the
    conversation as a mid-conversation system message (issue #56); some OpenAI clients send a late developer message."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingPrompt(tok, "</think>\n\n2", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def prompt_text(self):
        return bytes(i for i in self.engine.last_ids if i < 256).decode("utf-8", "replace")

    def test_query_string(self):
        body = {"model": "x", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]}
        for path in ("/v1/messages?beta=true", "/v1/chat/completions?api-version=1", "/v1/messages/?beta=true"):
            status, b = self.post(path, body)
            self.assertEqual(status, 200, (path, b))
        status, _ = self.post("/v1/nothing?beta=true", body)
        self.assertEqual(status, 404)

    def test_anthropic_mid_conversation_system(self):
        status, b = self.post("/v1/messages?beta=true", {
            "model": "x", "max_tokens": 50,
            "system": [{"type": "text", "text": "You are terse."}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "1+1? digits only"}]},
                {"role": "system", "content": [{"type": "text", "text": "<system-reminder>answer in digits</system-reminder>"}]}]})
        self.assertEqual(status, 200, b)
        text = self.prompt_text()
        self.assertIn("You are terse.", text)
        self.assertIn("<system-reminder>answer in digits</system-reminder>", text)
        self.assertLess(text.index("You are terse."), text.index("1+1?"))          # the first system stays first
        self.assertLess(text.index("answer in digits"), text.index("1+1?"))        # the late one goes before the words

    def test_late_system_before_the_user_words(self):
        """Claude Code's compaction request: tool results, the user's "summarize, no tools", then a system note.  The
        note goes in after the tool results and before the words, which stay last; each turn's prompt is the start
        of the next one's (the conversation cache still matches)."""
        from serve.frontend import anthropic_to_messages
        call = {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]}
        turn1 = [{"role": "user", "content": "read f1"}, {"role": "system", "content": "env note"}]
        turn2 = turn1 + [call, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "R1"},
                                                            {"type": "text", "text": "summarize, no tools"}]},
                         {"role": "system", "content": "files read before"}, {"role": "system", "content": "tool list"}]
        m1, _, _ = anthropic_to_messages({"system": "S", "messages": turn1})
        m2, _, _ = anthropic_to_messages({"system": "S", "messages": turn2})
        self.assertEqual([m["content"] for m in m1], ["S", "env note", "read f1"])
        self.assertEqual([(m["role"], m["content"]) for m in m2[4:]],
                         [("tool", "R1"), ("user", "files read before"), ("user", "tool list"),
                          ("user", "summarize, no tools")])
        self.assertEqual(m2[:len(m1)], m1)

    def test_old_qwen35_template_keeps_past_thinking(self):
        """Issue #52: a Qwen3.5-family template from before 2026-11 dropped the <think> of an assistant turn before
        the last user message; turn 2 must start with turn 1's prompt and answer (STRATA_KEEP_PAST_THINK=0 = old)."""
        import tempfile
        from serve.frontend import ChatTemplate
        src = ("{%- set ns = namespace(last_query_index=messages|length - 1) %}"
               "{%- for message in messages[::-1] %}{%- if message.role == 'user' %}"
               "{%- set ns.last_query_index = messages|length - 1 - loop.index0 %}{%- break %}{%- endif %}{%- endfor %}"
               "{%- for message in messages %}{%- if message.role == 'assistant' %}"
               "{%- if loop.index0 > ns.last_query_index %}"
               "{{- '<a><think>' + message.reasoning_content + '</think>' + message.content }}"
               "{%- else %}{{- '<a>' + message.content }}{%- endif %}"
               "{%- else %}{{- '<' + message.role + '>' + message.content }}{%- endif %}{%- endfor %}"
               "{%- if add_generation_prompt %}{{- '<a><think>' }}{%- endif %}")
        with tempfile.NamedTemporaryFile("w", suffix=".jinja", delete=False) as f:
            f.write(src)
        self.addCleanup(os.unlink, f.name)
        m1 = [{"role": "user", "content": "q1"}]
        m2 = m1 + [{"role": "assistant", "content": "A1", "reasoning_content": "R1"}, {"role": "user", "content": "q2"}]
        t = ChatTemplate(f.name)
        self.assertTrue(t.render(m2).startswith(t.render(m1) + "R1</think>A1"))
        os.environ["STRATA_KEEP_PAST_THINK"] = "0"
        self.addCleanup(os.environ.pop, "STRATA_KEEP_PAST_THINK", None)
        t = ChatTemplate(f.name)
        self.assertFalse(t.render(m2).startswith(t.render(m1) + "R1</think>A1"))

    def test_openai_late_developer_and_system(self):
        status, b = self.post("/v1/chat/completions", {
            "model": "x", "max_tokens": 50,
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hello"},
                         {"role": "assistant", "content": "hi"}, {"role": "developer", "content": "Now use digits."},
                         {"role": "system", "content": "Also this."}, {"role": "user", "content": "1+1?"}]})
        self.assertEqual(status, 200, b)
        text = self.prompt_text()
        for part in ("Be brief.", "Now use digits.", "Also this.", "1+1?"):
            self.assertIn(part, text)

    def test_leading_system_unchanged(self):
        from serve.frontend import anthropic_to_messages, openai_to_messages
        msgs, _, _ = openai_to_messages({"messages": [{"role": "developer", "content": "D"}, {"role": "user", "content": "u"}]})
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        msgs, _, _ = anthropic_to_messages({"system": "S", "messages": [{"role": "user", "content": "u"}]})
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])

    def test_anthropic_tool_result_images(self):
        """#41: an image inside a tool_result block (an image tool, a browser screenshot) reaches the template, and so
        does a top-level image next to a tool_result; text-only tool results stay plain strings."""
        from serve.frontend import anthropic_to_messages, images_of
        img = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo"}}
        req = {"messages": [
            {"role": "user", "content": [{"type": "text", "text": "Open check.png"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read_image", "input": {"path": "c"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "Read image"}, img]},
                {"type": "tool_result", "tool_use_id": "t2", "content": "plain"},
                {"type": "text", "text": "and this one"}, img]}]}
        msgs, _, _ = anthropic_to_messages(req)
        tools = [m for m in msgs if m["role"] == "tool"]
        self.assertEqual(tools[0]["content"][0], {"type": "text", "text": "Read image"})
        self.assertEqual(tools[0]["content"][1]["type"], "image")
        self.assertEqual(tools[1]["content"], "plain")
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertEqual([i["type"] for i in msgs[-1]["content"]], ["text", "image"])
        self.assertEqual(len(images_of(msgs)), 2)


class SamplingKeys(unittest.TestCase):
    """The GEN line's sampling keys: top_k 0 ("off") or wider than the engine's 64 get the widest list, 64 (they used
    to fall back to the engine default 20); a penalty always carries its window."""

    def keys(self, **sampling):
        return StrataEngine.sampling_keys(sampling).split()

    def test_top_k(self):
        self.assertIn("top_k=10", self.keys(temperature=0.7, top_k=10))
        self.assertIn("top_k=64", self.keys(temperature=0.7, top_k=64))
        self.assertIn("top_k=64", self.keys(temperature=0.7, top_k=0))
        self.assertIn("top_k=64", self.keys(temperature=0.7, top_k=100))
        for bad in (-1, True, 2.5, "20"):
            self.assertFalse([k for k in self.keys(temperature=0.7, top_k=bad) if k.startswith("top_k=")], bad)

    def test_tune_keys(self):
        k = self.keys(temperature=0, strata_tune={"pcie_frac": 0.2, "spec_min_p": 0.7})
        self.assertIn("pcie_frac=0.2", k)
        self.assertIn("spec_min_p=0.7", k)
        bad = self.keys(strata_tune={"pcie_frac": 3, "spec_min_p": True, "pool_workers": 2})
        self.assertFalse([x for x in bad if x.split("=")[0] in ("pcie_frac", "spec_min_p", "pool_workers")])
        # ./abyss.sh --check's drafter step: one answer without the drafter
        self.assertIn("spec=0", self.keys(temperature=0, strata_tune={"spec": 0}))
        for sp in (True, 2, 0.0, "0"):
            self.assertFalse([x for x in self.keys(strata_tune={"spec": sp}) if x.startswith("spec=")], sp)

    def test_think_keys(self):
        # the open reasoning block goes with or without a budget; the end guard only with whitespace ids
        k = self.keys(temperature=0.7, _think_budget=0, _think_end=151668)
        self.assertIn("think_end=151668", k)
        self.assertFalse([x for x in k if x.startswith(("think_budget=", "end_guard=", "ws_ids="))])
        k = self.keys(temperature=0, _think_budget=4096, _think_end=151668, _ws_ids=[198, 271], _end_ids=[151645])
        self.assertIn("think_budget=4096", k)
        self.assertIn("end_guard=1", k)
        self.assertIn("ws_ids=198,271", k)
        self.assertIn("end_ids=151645", k)
        self.assertEqual(self.keys(temperature=0.7, _think_budget=4096, _ws_ids=[198]), ["temperature=0.7"])   # no </think>
        # the bridge a forced </think> comes after (issue #36): with the open reasoning only
        k = self.keys(temperature=0.7, _think_budget=750, _think_end=151668, _think_close=[271, 40, 614])
        self.assertIn("think_close=271,40,614", k)
        self.assertFalse([x for x in self.keys(temperature=0.7, _think_close=[271]) if x.startswith("think_close=")])
        self.assertFalse([x for x in self.keys(temperature=0.7, _think_end=151668, _think_close=[])
                          if x.startswith("think_close=")])
        # the tool-argument tags (GLM's <arg_value> </arg_value>); an empty list says the vocabulary has none
        self.assertIn("arg_ids=154849,154850", self.keys(temperature=0, _arg_ids=[154849, 154850]))
        self.assertIn("arg_ids=", self.keys(temperature=0, _arg_ids=[]))
        self.assertFalse([x for x in self.keys(temperature=0) if x.startswith("arg_ids")])
        from serve.server import question_language
        self.assertEqual(question_language("寝坊した上司への謝罪メールの件名を提案してください。"), "Japanese")
        self.assertEqual(question_language("请用中文解释三省两指南。"), "Chinese")
        self.assertEqual(question_language("한국어로 설명해 주세요"), "Korean")
        self.assertIsNone(question_language("Explain the 3 ministries guidelines."))

    def test_uncensor_key(self):
        # the refusal-direction edit's per-request switch: a bool only, greedy and sampled lines alike
        self.assertIn("abl=0", self.keys(temperature=0, uncensor=False))
        self.assertIn("abl=1", self.keys(temperature=0.7, uncensor=True))
        for bad in (None, 1, "true"):
            self.assertFalse([k for k in self.keys(temperature=0.7, uncensor=bad) if k.startswith("abl=")], bad)

    def test_uncensor_arg(self):
        from serve.server import engine_args
        self.assertEqual(engine_args({"args": ["--glm-pack", "p"], "uncensor": "m.abl"})[-2:], ["--uncensor", "m.abl"])
        self.assertNotIn("--uncensor", engine_args({"args": ["--glm-pack", "p"]}))

    def test_penalty_window(self):
        self.assertIn("penalty_last_n=64", self.keys(presence_penalty=1.5))
        self.assertIn("penalty_last_n=4096", self.keys(repetition_penalty=1.1, penalty_last_n=4096))
        self.assertFalse([k for k in self.keys(temperature=0.7) if k.startswith("penalty")])


class ShippedPrior(unittest.TestCase):
    """A Qwen3.8-Flash-Next pack without expert_counts.txt gets the shipped routing profile (STRATA_GLM_PRIOR)."""

    def test_match(self):
        import tempfile
        from serve.server import child_env, shipped_prior
        with tempfile.TemporaryDirectory() as d:
            pack = Path(d)
            (pack / "index.txt").write_text("# strata pack index v3\nblk.0.hc_attn_down.weight 0\noutput_hc_down.weight 0\n")
            (pack / "native_experts.txt").write_text("# strata native experts v3: ... (n_expert 512, total 1)\n")
            cfg = {"args": ["--glm-pack", str(pack)]}
            prior = shipped_prior(cfg)
            self.assertTrue(prior.endswith("data/expert_counts/qwen38-flash-next.txt"), prior)
            self.assertTrue(Path(prior).exists())
            self.assertEqual(child_env(cfg)["STRATA_GLM_PRIOR"], prior)
            self.assertEqual(child_env({**cfg, "env": {"STRATA_GLM_PRIOR": "/x"}})["STRATA_GLM_PRIOR"], "/x")
            (pack / "expert_counts.txt").write_text("0 1 2\n")      # the pack's own profile wins
            self.assertEqual(shipped_prior(cfg), "")
            (pack / "expert_counts.txt").unlink()
            (pack / "index.txt").write_text("blk.0.attn_q.weight 0\n")   # another model: none
            self.assertEqual(shipped_prior(cfg), "")
        self.assertEqual(shipped_prior({"args": ["--model", "x.gguf"]}), "")


class GpuChoice(unittest.TestCase):
    """Issue #51: the config's \"gpu\" reaches the engine as CUDA_VISIBLE_DEVICES, numbered like nvidia-smi."""

    def test_env(self):
        from serve.server import child_env
        env = child_env({"gpu": 1})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        plain = child_env({})                     # no choice: the environment as it was (existing installs)
        self.assertEqual(plain.get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_VISIBLE_DEVICES"))
        self.assertEqual(plain.get("CUDA_DEVICE_ORDER"), os.environ.get("CUDA_DEVICE_ORDER"))

    def test_hip_igpu_lane(self):
        """A HIP engine on one discrete Radeon of an APU: the iGPU is listed after it, for the iGPU lane."""
        from unittest import mock
        from serve import server
        with mock.patch.object(server, "amd_igpu_index", return_value=1):
            self.assertEqual(server.child_env({"gpu": [0], "backend": "hip"})["HIP_VISIBLE_DEVICES"], "0,1")
            self.assertEqual(server.child_env({"gpu": [1], "backend": "hip"})["HIP_VISIBLE_DEVICES"], "1")
            off = {"gpu": [0], "backend": "hip", "env": {"STRATA_GLM_IGPU": "0"}}
            self.assertEqual(server.child_env(off)["HIP_VISIBLE_DEVICES"], "0")
            split = {"gpu": [0, 2], "backend": "hip"}   # a layer split numbers its later GPUs itself
            self.assertEqual(server.child_env(split)["HIP_VISIBLE_DEVICES"], "0,2")
        with mock.patch.object(server, "amd_igpu_index", return_value=None):
            self.assertEqual(server.child_env({"gpu": [0], "backend": "hip"})["HIP_VISIBLE_DEVICES"], "0")

    def test_hip_hw_queues(self):
        """HIP engines get two hardware queues unless the shell or the config sets them (R9700 prompts +33-48%)."""
        from unittest import mock
        from serve import server
        with mock.patch.object(server, "amd_igpu_index", return_value=None):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GPU_MAX_HW_QUEUES", None)
                self.assertEqual(server.child_env({"gpu": [0], "backend": "hip"})["GPU_MAX_HW_QUEUES"], "2")
                own = {"gpu": [0], "backend": "hip", "env": {"GPU_MAX_HW_QUEUES": "4"}}
                self.assertEqual(server.child_env(own)["GPU_MAX_HW_QUEUES"], "4")   # the config's wins
                self.assertNotIn("GPU_MAX_HW_QUEUES", server.child_env({"gpu": [0]}))   # CUDA: untouched
            with mock.patch.dict(os.environ, {"GPU_MAX_HW_QUEUES": "1"}):
                self.assertEqual(server.child_env({"gpu": [0], "backend": "hip"})["GPU_MAX_HW_QUEUES"], "1")   # the shell's

    def test_hip_no_hipblaslt_on_windows(self):
        """Windows HIP engines run rocBLAS without hipBLASLt unless the shell or the config sets it (#49)."""
        from unittest import mock
        from serve import server
        with mock.patch.object(server, "amd_igpu_index", return_value=None), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROCBLAS_USE_HIPBLASLT", None)
            hip = {"gpu": [0], "backend": "hip"}
            if os.name != "nt":
                self.assertNotIn("ROCBLAS_USE_HIPBLASLT", server.child_env(hip))   # Linux: untouched
            with mock.patch.object(server, "_WINDOWS_TEST", True):
                self.assertEqual(server.child_env(hip)["ROCBLAS_USE_HIPBLASLT"], "0")
                own = {**hip, "env": {"ROCBLAS_USE_HIPBLASLT": "1"}}
                self.assertEqual(server.child_env(own)["ROCBLAS_USE_HIPBLASLT"], "1")   # the config's wins
                self.assertNotIn("ROCBLAS_USE_HIPBLASLT", server.child_env({"gpu": [0]}))   # CUDA: untouched
                with mock.patch.dict(os.environ, {"ROCBLAS_USE_HIPBLASLT": "1"}):
                    self.assertEqual(server.child_env(hip)["ROCBLAS_USE_HIPBLASLT"], "1")   # the shell's

    def test_amd_igpu_index(self):
        import tempfile
        from serve.server import amd_igpu_index
        with tempfile.TemporaryDirectory() as d:
            for n, props in {"0": "simd_count 0\ngfx_target_version 0", "1": "simd_count 128\ngfx_target_version 120001",
                             "2": "simd_count 80\ngfx_target_version 110501"}.items():
                os.makedirs(os.path.join(d, n))
                with open(os.path.join(d, n, "properties"), "w") as f:
                    f.write(props + "\n")
            self.assertEqual(amd_igpu_index(d), 1)
            self.assertIsNone(amd_igpu_index(os.path.join(d, "missing")))

    def fake_kfd(self, d, nodes):
        for n, props in nodes.items():
            os.makedirs(os.path.join(d, n))
            with open(os.path.join(d, n, "properties"), "w") as f:
                f.write(props + "\n")

    def test_amd_gpu_nodes(self):
        import tempfile
        from serve.server import amd_gpu_nodes
        with tempfile.TemporaryDirectory() as d:
            self.fake_kfd(d, {"0": "simd_count 0\ngfx_target_version 0",
                              "1": "simd_count 128\ngfx_target_version 120001\nlocation_id 50944\ndomain 0",
                              "2": "simd_count 80\ngfx_target_version 110501\nlocation_id 51200\ndomain 0"})
            self.assertEqual(amd_gpu_nodes(d), [
                {"index": 0, "arch": "gfx1201", "igpu": False, "pci": "0000:c7:00.0"},
                {"index": 1, "arch": "gfx1151", "igpu": True, "pci": "0000:c8:00.0"}])

    def test_hip_gpu_pci(self):
        """The config's PCI addresses name the cards when HIP numbers them otherwise (an eGPU found after the iGPU)."""
        import contextlib
        import io
        from unittest import mock
        from serve import server
        swapped = [{"index": 0, "arch": "gfx1151", "igpu": True, "pci": "0000:c8:00.0"},
                   {"index": 1, "arch": "gfx1201", "igpu": False, "pci": "0000:c7:00.0"}]
        cfg = {"gpu": [0], "gpu_pci": ["0000:c7:00.0"], "backend": "hip"}
        with mock.patch.object(server, "amd_gpu_nodes", return_value=swapped), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(server.hip_gpu_list(cfg), [1])
            self.assertEqual(server.child_env(cfg)["HIP_VISIBLE_DEVICES"], "1,0")   # the iGPU lane after it
            self.assertIn("HIP number 1 now", out.getvalue())
            gone = dict(cfg, gpu_pci=["0000:03:00.0"])                           # card not there: the numbers
            self.assertEqual(server.hip_gpu_list(gone), [0])
            self.assertIn("WARNING: no AMD GPU at 0000:03:00.0", out.getvalue())
        with mock.patch.object(server, "amd_gpu_nodes", return_value=[]):        # no KFD (Windows): the numbers
            self.assertEqual(server.hip_gpu_list(cfg), [0])
        self.assertEqual(server.hip_gpu_list({"gpu": [0, 2]}), [0, 2])           # older configs: as before


class RecordingPrompt(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_ids = list(ids)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class DyingEngine(MockEngine):
    """Issue #27: an engine that dies after a few tokens of its first answer, and comes back when restarted."""

    def __init__(self, tok, script, max_context):
        super().__init__(tok, script, max_context=max_context)
        self.dead, self.restarts, self.die_after = False, 0, 5

    def alive(self):
        return not self.dead

    def restart(self):
        self.dead, self.die_after = False, None
        self.restarts += 1

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
            if self.die_after is not None and i == self.die_after:
                self.dead = True
                raise EngineDied("the engine stopped unexpectedly (exit code -9)")
            yield t


class CudaContextLost(unittest.TestCase):
    """An ERR naming a sticky CUDA error ends the engine (2026-10-19: V4.1 past 16K tokens, then 125 empty answers)."""

    def test_classification(self):
        from serve.server import cuda_context_lost
        self.assertTrue(cuda_context_lost("glm token completion: an illegal memory access was encountered"))
        self.assertTrue(cuda_context_lost("glm lending copies: unspecified launch failure"))
        self.assertFalse(cuda_context_lost("the prompt (70000 tokens) is longer than the context"))
        self.assertFalse(cuda_context_lost("an image must be a data: URL"))


class EngineDeath(unittest.TestCase):
    """Issue #27: a dead engine is an error (not "length"), and the next request starts it again."""

    def test_error_then_restart(self):
        tok = ByteTokenizer()
        eng = DyingEngine(tok, "</think>\n\n" + ANSWER, max_context=CTX)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            def post(body):
                req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=30) as r:
                        return r.status, r.read().decode()
                except urllib.error.HTTPError as e:
                    with e:
                        return e.code, e.read().decode()
            msgs = [{"role": "user", "content": "hi"}]
            code, text = post({"model": "m", "messages": msgs, "max_tokens": 50, "stream": True})
            self.assertEqual(code, 200)
            self.assertIn('"error"', text)
            self.assertIn("stopped unexpectedly", text)
            self.assertTrue(text.rstrip().endswith("data: [DONE]"))
            self.assertEqual(svc.metrics()["requests"][0]["finish"], "error")
            code, text = post({"model": "m", "messages": msgs, "max_tokens": 50})
            self.assertEqual(code, 200, text)
            self.assertEqual(eng.restarts, 1)
            self.assertEqual(json.loads(text)["usage"]["completion_tokens"], 50)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_engine_err_mid_stream(self):
        """The engine's ERR line after the stream started reaches the client as an error event (it used to be a
        400 written into the open stream, which clients read as an empty answer)."""
        class ErrEngine(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                yield None                                  # a prompt-progress heartbeat: the stream has started
                raise ValueError("verify: layer 31 never rang (an illegal memory access was encountered)")

        tok = ByteTokenizer()
        svc = Service(ErrEngine(tok, ANSWER, max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            for path, body in [("/v1/chat/completions", {"model": "m", "stream": True, "max_tokens": 20,
                                                          "messages": [{"role": "user", "content": "hi"}]}),
                               ("/v1/messages", {"model": "m", "stream": True, "max_tokens": 20,
                                                 "messages": [{"role": "user", "content": "hi"}]})]:
                req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    text = r.read().decode()
                self.assertIn("illegal memory access", text, path)
                self.assertNotIn("HTTP/1", text, path)
                self.assertEqual(svc.metrics()["requests"][0]["finish"], "error")
        finally:
            httpd.shutdown()
            httpd.server_close()


class LiveRate(unittest.TestCase):
    """The Monitor's Speed readout: live.tok_s is a rate, and a request that never got a DONE keeps no counters.

    It used to be `generated / (now - first_token)` - the mean since the first token, whose first sample is
    1/elapsed.  Against a paced engine that reads five-digit numbers for the first instant of every answer and
    undershoots for the first second after that.  It is now the rate over the last RATE_WINDOW_S, with the mean
    still available as `live.tok_s_mean` for anyone who wants it."""

    PACE_S = 0.02                    # 50 tokens/s: a 30-token answer takes about 0.6 s
    TOKENS = 30

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, "x" * self.TOKENS, max_context=CTX, delay_s=self.PACE_S)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def metrics(self):
        with urllib.request.urlopen(self.base + "/metrics", timeout=10) as r:
            return json.loads(r.read())

    def test_prefill_rate_excludes_cached_tokens(self):
        import io
        import queue
        from types import SimpleNamespace
        engine = StrataEngine.__new__(StrataEngine)
        engine.proc = SimpleNamespace(stdin=io.StringIO())
        engine.lines = queue.Queue()
        engine.can_stop = False
        engine.max_context = 262144
        engine.prefill_tok_s_mean = 9999.0
        engine.lines.put("PP 10000 12000 2000 1000.0")  # 8000 cached, 2000 newly read in two seconds
        engine.lines.put("DONE 1 12000 4000 10 stop 0 0 8000")
        gen = engine.generate([1], 1, {}, threading.Event())
        self.assertIsNone(next(gen))
        self.assertEqual(engine.progress, (10000, 12000))
        self.assertEqual(engine.prefill_tok_s_mean, 1000.0)
        self.svc.engine = engine
        self.svc.status.update(busy=True, first_token=None)
        self.assertEqual(self.metrics()["live"]["prefill_tok_s_mean"], 1000.0)
        self.assertNotIn("prefill_tok_s", self.metrics()["live"])
        self.assertEqual(self.svc._prefill_tok_s_mean(), 1000.0)
        self.svc.status.update(first_token=time.time(), generated=1)
        self.assertEqual(self.svc._prefill_tok_s_mean(), 0.0)
        self.assertEqual(list(gen), [])
        timings = request_timings(12000, 1, engine.last)
        self.assertEqual(timings["prompt_per_second"], 1000.0)
        engine.lines.put("PP 8000 12000")
        engine.lines.put("DONE 0 12000 0 0 stop 0 0 12000")
        gen = engine.generate([1], 1, {}, threading.Event())
        next(gen)
        self.assertIsNone(engine.prefill_tok_s_mean)
        list(gen)
        self.svc.status["busy"] = False
        self.assertIsNone(self.metrics()["live"]["prefill_tok_s_mean"])

    def test_the_live_number_is_a_rate(self):
        live_samples, stop = [], threading.Event()

        def poll():                                   # what the Monitor polls, at 10 ms
            while not stop.is_set():
                live = self.metrics()["live"]
                if live["state"] == "generating" and live["tok_s"] is not None:
                    live_samples.append((live["generated"], live["tok_s"], live["tok_s_mean"]))
                time.sleep(0.01)

        body = json.dumps({"model": "m", "max_tokens": self.TOKENS, "temperature": 0,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        watcher = threading.Thread(target=poll, daemon=True)
        watcher.start()
        t0 = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(self.base + "/v1/chat/completions", data=body,
                                                               headers={"Content-Type": "application/json"}),
                                        timeout=30) as r:
                usage = json.loads(r.read())["usage"]
        finally:
            stop.set()
            watcher.join(2)
        true_rate = usage["completion_tokens"] / (time.time() - t0)
        self.assertGreaterEqual(len(live_samples), 3, "too few live readings to judge the readout")
        self.assertLess(max(s for _g, s, _m in live_samples), 4 * true_rate,
                        f"live.tok_s peaked at {max(s for _g, s, _m in live_samples):.1f} tok/s "
                        f"for a {true_rate:.1f} tok/s engine")
        self.assertEqual(self.metrics()["live"]["state"], "idle")
        self.assertIsNone(self.metrics()["live"]["tok_s"])

    def test_a_request_without_a_done_keeps_no_engine_counters(self):
        """An engine that dies mid-answer: the previous request's `last` must not become this row's decode rate."""
        class HalfDead(MockEngine):
            last = {"generated": 99, "prompt_tokens": 9, "prompt_ms": 10.0, "decode_ms": 100.0, "finish": "stop"}

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
                    if i == 3:
                        raise EngineDied("the engine stopped unexpectedly (exit code -9)")
                    yield t

        tok = ByteTokenizer()
        svc = Service(HalfDead(tok, "x" * self.TOKENS, max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            body = json.dumps({"model": "m", "max_tokens": self.TOKENS, "stream": True,
                               "messages": [{"role": "user", "content": "hi"}]}).encode()
            with urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=body,
                                                               headers={"Content-Type": "application/json"}),
                                        timeout=30) as r:
                text = r.read().decode()
            self.assertIn('"error"', text)
            row = svc.metrics()["requests"][0]
            self.assertEqual(row["finish"], "error")
            self.assertEqual(row["output_tokens"], 3)
            self.assertIsNone(row["decode_tok_s"], "the previous request's counters were recorded as this one's")
            self.assertIsNone(row["engine_generated"])
        finally:
            httpd.shutdown()
            httpd.server_close()


class SharedSettings(unittest.TestCase):
    """The web app's "Use for other apps too": POST /settings makes its Chat settings every client's defaults."""

    @classmethod
    def setUpClass(cls):
        import tempfile

        class Sampled(RecordingEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                self.last_sampling = dict(sampling or {})
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        tok = ByteTokenizer()
        cls.engine = Sampled(tok, "</think>\n\nhello", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.tmp = tempfile.TemporaryDirectory()
        cls.svc.shared_path = os.path.join(cls.tmp.name, "strata-x.shared-settings.json")
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()

    def req(self, path, body, headers=None, raw=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        r = urllib.request.Request(self.base + path, data=raw if raw is not None else json.dumps(body).encode(), headers=h)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def chat(self, **extra):
        return self.req("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra})

    def tearDown(self):
        self.svc.set_shared(None)

    def test_other_apps_get_the_chat_settings(self):
        d = {"temperature": 0.3, "top_p": 0.9, "top_k": 10, "min_p": 0.05, "seed": 7, "max_tokens": 77,
             "reasoning_effort": "low", "experimental_speed_projection": False}
        code, b = self.req("/settings", {"defaults": d})
        self.assertEqual(code, 200, b)
        self.assertTrue(b["shared"])
        self.assertTrue(os.path.exists(self.svc.shared_path))
        code, _ = self.chat()                                        # a client that sets nothing
        self.assertEqual(code, 200)
        got = self.engine.last_sampling
        for k in ("temperature", "top_p", "top_k", "min_p", "seed", "experimental_speed_projection"):
            self.assertEqual(got[k], d[k], k)
        self.assertEqual(self.engine.last_max_new, 77)
        code, _ = self.chat(temperature=0.9, max_tokens=5)          # its own values win
        self.assertEqual(self.engine.last_sampling["temperature"], 0.9)
        self.assertEqual(self.engine.last_max_new, 5)
        r = self.svc.with_shared({"messages": []}, "openai")
        self.assertEqual(r["reasoning_effort"], "low")
        self.assertEqual(self.svc.with_shared({"reasoning_effort": "high"}, "openai")["reasoning_effort"], "high")
        for own in (False, 0, ""):   # a request's own "off" is kept
            self.assertEqual(self.svc.with_shared({"reasoning_effort": own}, "openai")["reasoning_effort"], own)
        self.assertEqual(self.svc.with_shared({}, "anthropic")["output_config"], {"effort": "low"})

    def test_off_again(self):
        self.req("/settings", {"defaults": {"temperature": 0.3}})
        code, b = self.req("/settings", {"defaults": None})
        self.assertEqual((code, b["shared"]), (200, False))
        self.assertFalse(os.path.exists(self.svc.shared_path))
        self.chat()
        self.assertNotIn("temperature", self.engine.last_sampling)

    def test_only_strata_s_own_page_may_set_them(self):
        code, _ = self.req("/settings", None, {"Content-Type": "text/plain"}, raw=b'{"defaults": {"temperature": 1}}')
        self.assertEqual(code, 415)
        code, _ = self.req("/settings", {"defaults": {"temperature": 1}}, {"Origin": "http://evil.example"})
        self.assertEqual(code, 403)
        code, b = self.req("/settings", {"defaults": {"temperature": 9}})
        self.assertEqual(code, 400)
        self.assertIn("temperature", b["error"]["message"])
        self.assertEqual(self.svc.shared, {})
        host = self.base.split("://", 1)[1]
        code, _ = self.req("/settings", {"defaults": {"temperature": 1}}, {"Origin": "http://" + host})
        self.assertEqual(code, 200)

    def test_they_need_the_key_when_one_is_set(self):
        self.svc.api_key = "secret"
        try:
            self.assertEqual(self.req("/settings", {"defaults": {"temperature": 1}})[0], 401)
            self.assertEqual(self.req("/settings", {"defaults": {"temperature": 1}},
                                      {"Authorization": "Bearer secret"})[0], 200)
        finally:
            self.svc.api_key = ""


class WebApp(unittest.TestCase):
    """The web app (PR #22's dashboard idea, rebuilt): its page and files, and GET /metrics."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(RecordingEngine(tok, "</think>\n\nhello", max_context=CTX), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def get(self, path, headers=None):
        req = urllib.request.Request(self.base + path, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    def test_page_and_files(self):
        code, ctype, body = self.get("/")
        self.assertEqual(code, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"\"web/app.js\"", body)   # relative since #82 (works behind a path-prefixed proxy)
        for path, want in (("/web/app.js", "javascript"), ("/web/app.css", "text/css"), ("/web/tokens.css", "text/css"),
                           ("/web/components.css", "text/css"), ("/web/sprite.svg", "image/svg+xml")):
            with self.subTest(path=path):
                code, ctype, _ = self.get(path)
                self.assertEqual(code, 200)
                self.assertIn(want, ctype)

    def test_only_the_app_files_are_served(self):
        for path in ("/web/..%2Fserver.py", "/web/index.html", "/web/test.py", "/fonts/..%2F..%2Fsetup.py",
                     "/fonts/missing.woff2", "/fonts/x.ttf"):
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404)

    def test_metrics(self):
        data = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}).encode()
        urllib.request.urlopen(urllib.request.Request(self.base + "/v1/chat/completions", data=data,
                                                      headers={"Content-Type": "application/json"}), timeout=10).read()
        code, ctype, body = self.get("/metrics")
        self.assertEqual(code, 200)
        m = json.loads(body)
        for key in ("engine", "live", "requests", "hardware", "hardware_static", "history"):
            self.assertIn(key, m)
        self.assertEqual(m["engine"]["max_context"], CTX)
        self.assertEqual(m["live"]["state"], "idle")
        self.assertEqual(m["requests"][0]["output_tokens"], 5)

    def test_model_discovery_and_props(self):
        svc = self.svc
        previous = svc.engine.max_context, svc.vision, svc.sampling_defaults, svc.shared
        try:
            svc.engine.max_context = 262144
            svc.sampling_defaults = {"temperature": 1.0, "repetition_penalty": 1.1}
            svc.shared = {"temperature": 0.7, "max_tokens": 4096}
            for vision in (None, object()):
                svc.vision = vision
                for path in ("/models", "/v1/models"):
                    code, _, body = self.get(path)
                    self.assertEqual(code, 200)
                    models = json.loads(body)["data"]
                    self.assertEqual(len(models), 1)
                    model = models[0]
                    self.assertEqual(model["id"], svc.model)
                    self.assertEqual(model["status"]["value"], "loaded")
                    self.assertEqual(model["meta"]["n_ctx"], 262144)
                    self.assertEqual(model["architecture"]["input_modalities"],
                                     ["text", "image"] if vision else ["text"])
                code, _, body = self.get("/props?model=" + svc.model + "&autoload=false")
                self.assertEqual(code, 200)
                props = json.loads(body)
                self.assertEqual(props["default_generation_settings"]["n_ctx"], 262144)
                self.assertEqual(props["default_generation_settings"]["params"],
                                 {"temperature": 0.7, "repeat_penalty": 1.1, "n_predict": 4096})
                self.assertEqual(props["chat_template"], (ROOT / "serve/chat_template.jinja").read_text(encoding="utf-8"))
                self.assertEqual(props["modalities"]["vision"], vision is not None)
                self.assertEqual(props["total_slots"], 1)
                self.assertFalse(props["models_autoload"])
            svc.shared = {}
            props = json.loads(self.get("/props")[2])
            self.assertEqual(props["default_generation_settings"]["params"]["n_predict"], -1)
            self.assertEqual(self.get("/props?model=not-loaded&autoload=true")[0], 404)
        finally:
            svc.engine.max_context, svc.vision, svc.sampling_defaults, svc.shared = previous

    def test_discovery_needs_the_api_key(self):
        self.svc.api_key = "secret"
        try:
            for path in ("/models", "/v1/models", "/props", "/slots"):
                self.assertEqual(self.get(path)[0], 401)
                self.assertEqual(self.get(path, {"Authorization": "Bearer secret"})[0], 200)
        finally:
            self.svc.api_key = ""

    def test_build_model_path_and_slot_status(self):
        engine = self.svc.engine
        engine.model_path = "models/example.gguf"
        engine.info = {"version": "0.1.21"}
        try:
            props = json.loads(self.get("/props")[2])
            self.assertEqual(props["model_path"], engine.model_path)
            self.assertEqual(props["build_info"], "Strata 0.1.21")
            for busy in (True, False):
                with self.svc.status_lock:
                    self.svc.status["busy"] = busy
                code, _, body = self.get("/slots")
                self.assertEqual(code, 200)
                self.assertEqual(json.loads(body), [{"id": 0, "n_ctx": CTX, "is_processing": busy}])
        finally:
            with self.svc.status_lock:
                self.svc.status["busy"] = False
            del engine.model_path, engine.info
        props = json.loads(self.get("/props")[2])
        self.assertNotIn("build_info", props)
        self.assertNotIn("model_path", props)

    def test_engine_model_path_reads_the_pack_flags(self):
        self.assertEqual(engine_model_path(["--glm-pack", "/m/pack", "--max-context", "65536"]), "/m/pack")
        self.assertEqual(engine_model_path(["--pack", "/m/pack-dsv4"]), "/m/pack-dsv4")
        self.assertEqual(engine_model_path(["--native", "/m/a.gguf", "--pack", "/m/p"]), "/m/a.gguf")
        self.assertEqual(engine_model_path(["--max-context", "65536"]), "pack/full")

    def test_discovery_does_not_restart_a_dead_engine(self):
        self.svc.engine.alive = lambda: False
        try:
            for path in ("/models", "/v1/models"):
                code, _, body = self.get(path)
                self.assertEqual(code, 200)
                self.assertEqual(json.loads(body)["data"], [])
            self.assertEqual(self.get("/props")[0], 503)
            self.assertEqual(json.loads(self.get("/slots")[2]), [])
        finally:
            del self.svc.engine.alive

    def test_metrics_need_the_key_when_one_is_set(self):
        self.svc.api_key = "secret"
        try:
            self.assertEqual(self.get("/metrics")[0], 401)
            self.assertEqual(self.get("/metrics", {"Authorization": "Bearer secret"})[0], 200)
            self.assertEqual(self.get("/")[0], 200)                  # the page itself asks for the key
        finally:
            self.svc.api_key = ""


class ClockedEngine(MockEngine):
    """The mock engine with StrataEngine's clock: `last` as the engine's DONE line gives it, the conversation cache
    holding the first REUSED tokens of every prompt."""
    REUSED = 5

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        n = 0
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                n += 1
                yield t
        finally:          # as StrataEngine reads its DONE line: also when the server closes the request at a stop token
            self.last = {"generated": n, "prompt_tokens": len(ids), "prompt_ms": 40.0, "decode_ms": 20.0 * n,
                         "finish": "stop", "reused": min(self.REUSED, len(ids)), "hits": 9, "lookups": 10}


class UsageAndStatus(unittest.TestCase):
    """What clients read besides the text: the part of the prompt the conversation cache held (OpenAI's
    prompt_tokens_details.cached_tokens, Anthropic's cache_read_input_tokens), llama.cpp's timings, GET /v1/status."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = ClockedEngine(tok, "</think>\n\nok", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def request(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()

    def chat(self, path, stream=False):
        body = {"model": "x", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}], "stream": stream}
        status, raw = self.request(path, body)
        self.assertEqual(status, 200)
        if not stream:
            return json.loads(raw)
        return [json.loads(line[6:]) for line in raw.decode().splitlines()
                if line.startswith("data: {")]

    def test_openai(self):
        b = self.chat("/v1/chat/completions")
        u, t = b["usage"], b["timings"]
        self.assertEqual(u["prompt_tokens_details"]["cached_tokens"], ClockedEngine.REUSED)
        self.assertEqual(t["cache_n"], ClockedEngine.REUSED)
        self.assertEqual(t["prompt_n"] + t["cache_n"], u["prompt_tokens"])
        self.assertEqual(t["predicted_n"], u["completion_tokens"])
        self.assertAlmostEqual(t["prompt_per_second"], t["prompt_n"] / 0.040, delta=0.1)
        self.assertAlmostEqual(t["predicted_per_second"], 50.0, delta=0.1)            # 20 ms a token

    def test_openai_stream(self):
        last = self.chat("/v1/chat/completions", stream=True)[-1]
        self.assertEqual(last["usage"]["prompt_tokens_details"]["cached_tokens"], ClockedEngine.REUSED)
        self.assertEqual(last["timings"]["cache_n"], ClockedEngine.REUSED)

    def test_anthropic(self):
        u = self.chat("/v1/messages")["usage"]
        self.assertEqual(u["cache_read_input_tokens"], ClockedEngine.REUSED)
        self.assertEqual(u["input_tokens"] + u["cache_read_input_tokens"], len(self.engine.last_prompt))
        self.assertGreater(u["output_tokens"], 0)

    def test_v1_status(self):
        self.chat("/v1/chat/completions")
        status, raw = self.request("/v1/status")
        self.assertEqual(status, 200)
        s = json.loads(raw)
        self.assertEqual(s["model"], self.svc.model)
        self.assertEqual(s["context"]["max_positions"], CTX)
        self.assertEqual(s["concurrency"]["serving"], 1)
        self.assertFalse(s["vision"]["available"])
        self.assertEqual(s["activity"]["in_flight"], 0)
        self.assertGreaterEqual(s["activity"]["requests"], 1)
        self.assertEqual(s["last_timings"]["cache_n"], ClockedEngine.REUSED)
        self.assertIn("at", s["last_timings"])

    def test_no_clock(self):
        """An engine without a clock (MockEngine): no timings, nothing cached."""
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nok", max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            data = json.dumps({"model": "x", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                b = json.loads(r.read())
            self.assertNotIn("timings", b)
            self.assertEqual(b["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
            self.assertIsNone(svc.v1_status()["last_timings"])
        finally:
            httpd.shutdown()
            httpd.server_close()


class TimingsDrafts(unittest.TestCase):
    """`timings` carries the speculative draft counts (PR #83's fields) only when the engine reported them."""

    def test_draft_fields(self):
        base = {"prompt_ms": 100.0, "decode_ms": 200.0, "generated": 20, "reused": 4}
        t = request_timings(24, 20, dict(base, drafts_offered=15, drafts_accepted=11))
        self.assertEqual((t["draft_n"], t["draft_n_accepted"]), (15, 11))
        self.assertEqual((t["prompt_n"], t["cache_n"]), (20, 4))
        self.assertNotIn("draft_n", request_timings(24, 20, base))
        self.assertIsNone(request_timings(24, 20, {}))


class StallWatchdog(unittest.TestCase):
    """A silent engine is ended only when it also used no CPU, disk or GPU in the window (from Strata #1317)."""

    def check(self, first, second, gpu=False):
        e = StrataEngine.__new__(StrataEngine)            # no process: its activity samples are given
        samples = iter([first, second])
        e._activity = lambda: next(samples)
        e.gpu_busy = lambda: gpu
        state = {}
        self.assertIsNone(e._stall_check(time.monotonic(), state))           # the baseline, soon after a line
        return e._stall_check(time.monotonic() - 1000.0, state), state       # long silent since then

    def test_stuck(self):
        why, _ = self.check((10.0, 1000), (10.2, 1000 + 4096))
        self.assertIn("stuck", why)

    def test_working(self):
        for second in ((30.0, 0), (10.0, 50 << 20)):                        # CPU time, disk bytes
            why, state = self.check((10.0, 0), second)
            self.assertIsNone(why, second)
            self.assertEqual(state["base"], second)                          # the next window starts here
        self.assertIsNone(self.check((10.0, 0), (10.0, 0), gpu=True)[0])

    def test_without_psutil_nothing_is_ended(self):
        self.assertIsNone(self.check(None, None)[0])

    def test_the_config_env_reaches_the_server(self):
        import serve.server as S
        saved = {k: os.environ.get(k) for k in S.SERVER_ENV}
        try:
            S.apply_server_env({"env": {"STRATA_ENGINE_STALL_S": "0", "STRATA_HTTP_BACKLOG": "64"}})
            self.assertEqual((S.ENGINE_STALL_S, S.Server.request_queue_size), (0.0, 64))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            S.apply_server_env({})


class RequestBodies(unittest.TestCase):
    """Bodies refused before they are read, a chunked body, a malformed one, several API keys and the listen backlog
    (from Strata 0.1.41)."""

    MSG = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(RecordingEngine(tok, "</think>\n\nhello", max_context=CTX), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def raw(self, headers: str, body: bytes = b"") -> tuple[int, dict]:
        """One POST written by hand (headers urllib would not send) -> (status, JSON answer)."""
        import socket
        head = ("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n" + headers +
                "\r\n").encode()
        with socket.create_connection(("127.0.0.1", self.port), timeout=30) as s:
            s.sendall(head + body)
            data = b""
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
        status, _, rest = data.partition(b"\r\n\r\n")
        return int(status.split()[1]), json.loads(rest or b"{}")

    def test_a_body_over_the_limit_is_refused_unread(self):
        code, b = self.raw(f"Content-Length: {300 << 20}\r\n")
        self.assertEqual(code, 413)
        self.assertIn("MiB", b["error"]["message"])

    def test_a_bad_content_length(self):
        self.assertEqual(self.raw("Content-Length: abc\r\n")[0], 400)
        self.assertEqual(self.raw("Content-Length: -5\r\n")[0], 400)

    def test_a_chunked_body(self):
        body = b"%x\r\n%s\r\n0\r\n\r\n" % (len(self.MSG), self.MSG)
        code, b = self.raw("Transfer-Encoding: chunked\r\n", body)
        self.assertEqual(code, 200, b)
        self.assertIn("hello", json.dumps(b))
        self.assertEqual(self.raw("Transfer-Encoding: chunked\r\n", b"zz\r\nab\r\n0\r\n\r\n")[0], 400)

    def test_a_malformed_body_is_a_400(self):
        self.assertEqual(self.raw("Content-Length: 2\r\n", b"[]")[0], 400)
        bad = json.dumps({"model": "m", "messages": [{"role": "user", "content": {"no": "list"}}, 7]}).encode()
        code, b = self.raw(f"Content-Length: {len(bad)}\r\n", bad)
        self.assertEqual(code, 400, b)

    def test_several_api_keys(self):
        self.svc.api_key = "k1, k2"
        try:
            for key, want in (("k1", 200), ("k2", 200), ("k3", 401), ("k1, k2", 401)):
                code, _ = self.raw(f"Authorization: Bearer {key}\r\nContent-Length: {len(self.MSG)}\r\n", self.MSG)
                self.assertEqual(code, want, key)
            self.svc.api_key = ["a,b"]                     # a list in the config: a key with a comma in it
            code, _ = self.raw(f"Authorization: Bearer a,b\r\nContent-Length: {len(self.MSG)}\r\n", self.MSG)
            self.assertEqual(code, 200)
        finally:
            self.svc.api_key = ""

    def test_the_listen_backlog(self):
        self.assertGreaterEqual(self.httpd.request_queue_size, 256)

class TestCarryEmbeddings(unittest.TestCase):
    def test_the_job_thread_sees_the_request_threads_image_file(self):
        local = threading.local()
        local.path = "req-1.sve"                       # set by the request thread while it prepares the prompt
        seen = []

        def chunks():
            seen.append(getattr(local, "path", None))  # what Service.generate reads when the engine call starts
            yield "chunk"

        gen = carry_embeddings(local, local.path, chunks())
        worker = threading.Thread(target=lambda: list(gen))   # the strata_resume job thread
        worker.start()
        worker.join()
        self.assertEqual(seen, ["req-1.sve"])


class VisionFootprint(unittest.TestCase):
    """The on-demand encoder's measurement: a failed or malformed one, or a damaged vision-memory.json, is no
    measurement (the encoder runs on the CPU), never a server that does not start."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "mmproj.gguf").write_bytes(b"x")
        self.cache = d / "vision-memory.json"
        self.vcfg = {"exe": "strata-vision", "mmproj": str(d / "mmproj.gguf"), "model": "vocab.gguf",
                     "max_tokens": 1024}

    def tearDown(self):
        self.tmp.cleanup()

    def measure(self, returncode, stdout):
        from unittest import mock
        from serve.server import vision_footprint
        done = mock.Mock(returncode=returncode, stdout=stdout)
        with mock.patch("serve.server.subprocess.run", return_value=done) as run:
            got = vision_footprint(self.vcfg, {}, self.cache, 0)
        return got, run.call_count

    def test_a_failed_warm_up_is_no_measurement(self):
        self.cache.write_text("[]")                    # not the dict it writes
        self.assertEqual(self.measure(1, "ERR the vision encoder's warm-up encoded nothing\n"), (None, 1))

    def test_a_malformed_mem_line_is_no_measurement(self):
        self.assertEqual(self.measure(0, "MEM 12\n"), (None, 1))
        self.assertEqual(self.measure(0, "MEM a b c\n"), (None, 1))

    def test_a_damaged_cache_entry_is_measured_again(self):
        self.measure(0, "MEM 1000 2000 100000\n")      # the key it writes
        key = next(iter(json.loads(self.cache.read_text())))
        self.cache.write_text(json.dumps({key: {"bytes": "1", "total": 100000}}))
        self.assertEqual(self.measure(0, "MEM 1000 2000 100000\n"), ((1024, 3000 + (256 << 20)), 1))
        self.assertEqual(self.measure(0, ""), ((1024, 3000 + (256 << 20)), 0))   # now from the cache


class StallLimit(unittest.TestCase):
    """Strata 99b1b27b1's idea: a client that sends a streaming request and stops reading filled the send buffer, the
    write never returned and the engine's turn was held for good - every later request waited.  With a stall limit
    (STRATA_HTTP_STALL_S) the write fails, the answer is cancelled as on a disconnect, and the next request runs.  A
    request body that stops arriving is a 408."""

    @classmethod
    def setUpClass(cls):
        cls.old = os.environ.get("STRATA_HTTP_STALL_S")
        os.environ["STRATA_HTTP_STALL_S"] = "1"
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, "</think>\n\n" + "x" * 60000, max_context=131072)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        if cls.old is None:
            os.environ.pop("STRATA_HTTP_STALL_S", None)
        else:
            os.environ["STRATA_HTTP_STALL_S"] = cls.old

    def raw(self, head: bytes, body: bytes = b""):
        import socket
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)   # a small window: the server's buffer fills soon
        s.connect(("127.0.0.1", self.port))
        s.sendall(head + body)
        return s

    def test_a_stream_nobody_reads_lets_the_next_request_run(self):
        import contextlib
        import io
        body = json.dumps({"model": "m", "stream": True, "max_tokens": 60000,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            s = self.raw(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                         b"Content-Length: %d\r\n\r\n" % len(body), body)
            try:
                time.sleep(0.5)                                         # the stream has started; nobody reads it
                req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/messages", json.dumps(
                    {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}).encode(),
                    {"Content-Type": "application/json"})
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=60) as r:
                    self.assertEqual(r.status, 200)
                    self.assertEqual(json.loads(r.read())["usage"]["output_tokens"], 5)
                self.assertLess(time.time() - t0, 50)
            finally:
                s.close()
        self.assertIn("the client stopped reading for 1 s", out.getvalue())

    def test_a_body_that_stops_arriving_is_a_408(self):
        s = self.raw(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     b"Content-Length: 100\r\n\r\n", b'{"model"')
        try:
            s.settimeout(30)
            got = b""
            while True:
                piece = s.recv(65536)
                if not piece:
                    break
                got += piece
        finally:
            s.close()
        self.assertTrue(got.startswith(b"HTTP/1.0 408"), got[:80])
        self.assertIn(b"stopped arriving", got)

    def test_a_silent_connection_is_closed(self):
        s = self.raw(b"")
        try:
            s.settimeout(30)
            t0 = time.time()
            self.assertEqual(s.recv(100), b"")                          # closed by the server, not left open
            self.assertLess(time.time() - t0, 20)
        finally:
            s.close()


class DeathNote(unittest.TestCase):
    """Why the engine ended: an abort or a crash is not called a RAM shortage (Maya #1007's idea)."""

    def test_abort_quotes_its_line(self):
        from serve.server import engine_death_note
        tail = "glm fast: loading\nabyss: src/core/x.cu:12: void f(): Assertion `n > 0' failed.\n"
        for rc in (-6, 3):
            note = engine_death_note(rc, tail)
            self.assertIn("aborted", note)
            self.assertIn("Assertion `n > 0' failed", note)
            self.assertNotIn("RAM:", note)
        self.assertIn("aborted", engine_death_note(-6, "no marks here\n"))
        self.assertIn("aborted", engine_death_note(None, "terminate called after throwing an instance of 'x'\n"))

    def test_crash(self):
        from serve.server import engine_death_note
        for rc in (-11, 0xC0000005):
            self.assertIn("crashed", engine_death_note(rc, "glm fast: decode\n"))

    def test_killed_is_ram_and_watchdog_wins(self):
        from serve.server import engine_death_note
        self.assertIn("running out of RAM", engine_death_note(-9, "glm fast: decode\n"))
        self.assertIn("running out of RAM", engine_death_note(None, ""))
        # a kill by the OOM killer is RAM even when an earlier line mentioned an assertion
        self.assertIn("running out of RAM", engine_death_note(-9, "Assertion checks on\n"))
        self.assertIn("stopped itself", engine_death_note(-6, "x\nwatchdog: no progress (issue #29)\n"))


class EmptyAnswerContent(unittest.TestCase):
    """OpenAI non-streamed: an answer with neither text nor calls has content "", not null (Strata #5249's case)."""

    @staticmethod
    def chunk(delta, finish=None):
        return {"id": "c", "created": 1, "model": "m", "usage": {},
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    def test_empty(self):
        from serve.server import openai_collect
        out = openai_collect([self.chunk({"role": "assistant"}), self.chunk({}, "stop")])
        self.assertEqual(out["choices"][0]["message"]["content"], "")

    def test_calls_keep_null_and_text_stays(self):
        from serve.server import openai_collect
        call = {"index": 0, "id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        out = openai_collect([self.chunk({"tool_calls": [call]}), self.chunk({}, "tool_calls")])
        self.assertIsNone(out["choices"][0]["message"]["content"])
        out = openai_collect([self.chunk({"content": "hi"}), self.chunk({}, "stop")])
        self.assertEqual(out["choices"][0]["message"]["content"], "hi")


if __name__ == "__main__":
    unittest.main()
