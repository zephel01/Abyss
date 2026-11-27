"""serve/test_giga.py - ai-sage GigaChat4: the template detected, thinking on / off through its own `reasoning`
variable (from reasoning_effort, enable_thinking or the card's chat_template_kwargs.reasoning), the bos token, the
turn end <|message_sep|>\\n\\n as a stop - no GPU, no pack.

    python -m unittest serve.test_giga -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, OutputParser, effort_kwargs, openai_to_messages  # noqa: E402

# a stand-in with the parts of GigaChat4's template the server keys on (chatml v5's separators, `reasoning`)
SRC = ("{{- bos_token }}{%- if reasoning is not defined or reasoning is none %}{%- set reasoning = True %}{%- endif %}"
       "{%- for m in messages %}{{- m.role + '<|role_sep|>\\n' + m.content + '<|message_sep|>\\n\\n' }}{%- endfor %}"
       "{%- if add_generation_prompt %}{{- 'assistant<|role_sep|>\\n' }}"
       "{%- if not reasoning %}{{- '<think></think>\\n\\n' }}{%- endif %}{%- endif %}")


class GigaTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        p = Path(cls.dir.name) / "chat_template.jinja"
        p.write_text(SRC, encoding="utf-8")
        cls.t = ChatTemplate(p, bos_token="<s>")

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def render(self, **kw):
        return self.t.render([{"role": "user", "content": "hi"}], **kw)

    def test_detected(self):
        self.assertTrue(self.t.giga)
        self.assertFalse(self.t.raon or self.t.ink or self.t.dsv4 or self.t.ling)

    def test_thinking_on_off(self):
        on = self.render()
        self.assertTrue(on.startswith("<s>user<|role_sep|>\nhi<|message_sep|>\n\n"))
        self.assertTrue(on.endswith("assistant<|role_sep|>\n"))
        self.assertTrue(self.render(**effort_kwargs("high")).endswith("assistant<|role_sep|>\n"))
        for off in ("none", False):
            self.assertTrue(self.render(**effort_kwargs(off)).endswith("<think></think>\n\n"), off)
        self.assertTrue(self.render(enable_thinking=False).endswith("<think></think>\n\n"))

    def test_card_kwargs(self):
        # the model card's request: "chat_template_kwargs": {"reasoning": false}
        for v, off in ((False, True), (True, False)):
            _, _, kw = openai_to_messages({"messages": [{"role": "user", "content": "hi"}],
                                           "chat_template_kwargs": {"reasoning": v}})
            self.assertEqual(self.render(**kw).endswith("<think></think>\n\n"), off, v)

    def test_tool_call(self):
        # GigaChat4's GCML block (DeepSeek's DSML form with its own token), streamed in small pieces
        tools = [{"name": "get_weather", "description": "w", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}]
        p = OutputParser(thinking=False, tools=tools, stream_tools=True)
        text = ('<\uff5cGCML\uff5ctool_calls>\n<\uff5cGCML\uff5cinvoke name="get_weather">\n'
                '<\uff5cGCML\uff5cparameter name="city" string="true">Tokyo</\uff5cGCML\uff5cparameter>\n'
                '<\uff5cGCML\uff5cparameter name="days" string="false">2</\uff5cGCML\uff5cparameter>\n'
                '</\uff5cGCML\uff5cinvoke>\n</\uff5cGCML\uff5ctool_calls>')
        ev = []
        for i in range(0, len(text), 5):
            ev += p.feed(text[i:i + 5])
        ev += p.finish()
        calls = [e.call for e in ev if e.kind == "tool_call" and e.call is not None]
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0].name, calls[0].arguments), ("get_weather", {"city": "Tokyo", "days": 2}))
        self.assertFalse([e for e in ev if e.kind == "content" and e.text.strip()])


if __name__ == "__main__":
    unittest.main()
