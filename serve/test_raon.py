"""serve/test_raon.py - KRAFTON Raon-1: thinking on / off in the template, the reasoning effort as the engine's GEN
effort= key (its steering alpha, per request), its tool calls (Qwen's form) - no GPU, no pack.

    python -m unittest serve.test_raon -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, OutputParser, effort_kwargs, raon_effort  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine  # noqa: E402

# a stand-in with the parts of Raon-1's template the server keys on (its comment, ChatML, enable_thinking)
SRC = ("{#- Text-only ChatML SFT protocol for raon-1-mini. -#}"
       "{%- for m in messages %}{{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endfor %}"
       "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
       "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think></think>\\n\\n' }}"
       "{%- else %}{{- '<think>\\n' }}{%- endif %}{%- endif %}"
       "{{- reasoning_effort if reasoning_effort is defined else '' }}")
TOOLS = [{"name": "get_weather", "description": "w", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}]


class RaonTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        p = Path(cls.dir.name) / "chat_template.jinja"
        p.write_text(SRC, encoding="utf-8")
        cls.path = p
        cls.t = ChatTemplate(p)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def render(self, effort=None):
        return self.t.render([{"role": "user", "content": "hi"}], **effort_kwargs(effort))

    def test_detected(self):
        self.assertTrue(self.t.raon)
        self.assertFalse(self.t.ink or self.t.glm_effort or self.t.dsv4 or self.t.ling)

    def test_thinking_on_off(self):
        self.assertTrue(self.render().endswith("<|im_start|>assistant\n<think>\n"))
        self.assertTrue(self.render("high").endswith("<think>\n"))   # (no effort text reaches the template)
        for off in ("none", "minimal", False):
            self.assertTrue(self.render(off).endswith("<think></think>\n\n"), off)

    def test_effort_names(self):
        cases = {None: None, "none": "none", "minimal": "none", False: "none", "low": "low", "medium": "medium",
                 "high": "high", "xhigh": "xhigh", "max": "max", "maximum": "max", 40: "low", 75: "high", 100: "max"}
        for given, want in cases.items():
            self.assertEqual(raon_effort(effort_kwargs(given)), want, given)

    def test_gen_key(self):
        keys = StrataEngine.sampling_keys
        self.assertIn(" effort=medium", keys({"temperature": 0, "_effort": "medium"}))
        self.assertIn(" effort=none", keys({"temperature": 0.6, "_effort": "none"}))
        self.assertNotIn("effort=", keys({"temperature": 0}))
        self.assertNotIn("effort=", keys({"temperature": 0, "_effort": "x y"}))   # (never a broken GEN line)

    def test_prepare(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, self.t)
        for effort, want in ((None, None), ("low", "low"), ("none", "none"), ("xhigh", "xhigh")):
            kw = dict(effort_kwargs(effort))
            svc.prepare([{"role": "user", "content": "hi"}], None, kw)
            self.assertEqual(kw.get("_effort"), want, effort)
        kw = {}
        svc.default_effort = "low"   # (the run config's level, when the request names none)
        svc.prepare([{"role": "user", "content": "hi"}], None, kw)
        self.assertEqual(kw.get("_effort"), "low")

    def test_other_templates(self):
        tok = ByteTokenizer()
        root = Path(__file__).resolve().parents[1]
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(root / "serve/chat_template.jinja"))
        kw = dict(effort_kwargs("low"))
        svc.prepare([{"role": "user", "content": "hi"}], None, kw)
        self.assertIsNone(kw.get("_effort"))   # (no effort= key for another model)

    def test_tool_call(self):
        p = OutputParser(thinking=True, tools=TOOLS, stream_tools=True)
        text = ("Check.</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n"
                "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>")
        ev = []
        for i in range(0, len(text), 7):
            ev += p.feed(text[i:i + 7])
        ev += p.finish()
        calls = [e.call for e in ev if e.kind == "tool_call" and e.call is not None]
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0].name, calls[0].arguments), ("get_weather", {"city": "Tokyo", "days": 2}))


if __name__ == "__main__":
    unittest.main()
