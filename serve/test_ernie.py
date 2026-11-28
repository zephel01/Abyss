"""serve/test_ernie.py - Baidu ERNIE 5.5 (tools/ernie55_chat_template.jinja, the template the pack carries): the
template detected, the server's thinking levels in its words (low / medium / high, max -> high, none -> the think block
closed), and the reasoning read up to </ernie:think> - no GPU, no pack.

    python -m unittest serve.test_ernie -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate, OutputParser, effort_kwargs  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402


class ErnieTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = ChatTemplate(ROOT / "tools" / "ernie55_chat_template.jinja")

    def render(self, **kw):
        return self.t.render([{"role": "user", "content": "hi"}], **kw)

    def test_detected(self):
        self.assertTrue(self.t.ernie)
        self.assertFalse(self.t.hy4 or self.t.giga or self.t.glm_effort)

    def test_default_high(self):
        out = self.render()
        self.assertTrue(out.startswith("<|ERNIE_MASK|><|ernie:im_start|>system\n"))
        self.assertIn("reasoning_effort=high", out)
        self.assertTrue(out.endswith("<|ernie:im_start|>assistant\n<ernie:think>"))

    def test_levels(self):
        self.assertIn("reasoning_effort=low", self.render(**effort_kwargs("low")))
        self.assertIn("reasoning_effort=medium", self.render(**effort_kwargs("medium")))
        self.assertIn("reasoning_effort=high", self.render(**effort_kwargs("high")))
        self.assertIn("reasoning_effort=high", self.render(**effort_kwargs("max")))   # (no max: its high)

    def test_off(self):
        for kw in (effort_kwargs("none"), {"enable_thinking": False}):
            out = self.render(**kw)
            self.assertNotIn("reasoning_effort=", out)
            self.assertTrue(out.endswith("<ernie:think></ernie:think>"))

    def test_server_thinking_flag(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, self.t)
        m = [{"role": "user", "content": "hi"}]
        self.assertTrue(svc.prepare(m, None, {})[1])
        self.assertFalse(svc.prepare(m, None, {"enable_thinking": False})[1])

    def test_reasoning_ends_at_its_tag(self):
        # the end comes as its own token (think_token): its text </ernie:think> is not part of the answer
        p = OutputParser(thinking=True, think_token=True)
        ev = p.feed("Let me think.") + p.feed("</ernie:think>", think_end=True) + p.feed("Paris.") + p.finish()
        self.assertEqual("".join(e.text for e in ev if e.kind == "reasoning"), "Let me think.")
        self.assertEqual("".join(e.text for e in ev if e.kind == "content"), "Paris.")


if __name__ == "__main__":
    unittest.main()
