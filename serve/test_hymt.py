"""Tencent Hy-MT3's chat template (the GGUF's, trimmed to the parts the server's levels touch): a request without a
reasoning level keeps the template's own default no_think (Hy4-flash's default is high), low / high pass through."""
import tempfile
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service

TEMPLATE = """{%- set HYTK = '' %}
{%- set bos_token = '<｜hy_begin▁of▁sentence｜>' %}
{%- set reasoning_mode_token = '<｜reasoning_mode｜>' %}
{%- if not reasoning_effort is defined or reasoning_effort not in ['high', 'low', 'no_think'] %}
    {%- set reasoning_effort = 'no_think' %}
{%- endif %}
{{- bos_token }}{{- reasoning_mode_token + 'reasoning_effort:' + reasoning_effort }}
{%- for message in messages %}{{- '<｜hy_User｜>' + message['content'] }}{%- endfor %}
{%- if add_generation_prompt %}
    {%- if reasoning_effort in ['low', 'high'] %}{{- '<｜hy_Assistant｜><think>' }}
    {%- else %}{{- '<｜hy_Assistant｜><think></think>' }}{%- endif %}
{%- endif %}"""


class HyMT3Template(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        p = Path(d) / "chat_template.jinja"
        p.write_text(TEMPLATE, encoding="utf-8")
        self.t = ChatTemplate(p)

    def test_detected(self):
        self.assertTrue(self.t.hy4)
        self.assertTrue(self.t.hymt)

    def test_default_no_think(self):
        out = self.t.render([{"role": "user", "content": "Translate: hi"}])
        self.assertIn("reasoning_effort:no_think", out)
        self.assertTrue(out.endswith("<think></think>"))

    def test_levels(self):
        m = [{"role": "user", "content": "x"}]
        self.assertIn("reasoning_effort:high", self.t.render(m, reasoning_effort="high"))
        self.assertIn("reasoning_effort:low", self.t.render(m, reasoning_effort="low"))
        self.assertIn("reasoning_effort:no_think", self.t.render(m, enable_thinking=False, reasoning_effort="high"))

    def test_server_reads_the_default_as_content(self):
        # the template's no_think prompt ends with <think></think>: the answer is content, not reasoning (before the
        # fix the server read it as reasoning, its end guard closed it and the model wrote the answer again)
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, self.t)
        m = [{"role": "user", "content": "Translate: hi"}]
        self.assertFalse(svc.prepare(m, None, {})[1])
        self.assertFalse(svc.prepare(m, None, {"enable_thinking": False})[1])
        self.assertTrue(svc.prepare(m, None, {"reasoning_effort": "high"})[1])
        self.assertTrue(svc.prepare(m, None, {"reasoning_effort": "low"})[1])


if __name__ == "__main__":
    unittest.main()
