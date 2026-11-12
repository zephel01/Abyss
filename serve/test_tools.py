"""Tool calls in GLM-5.x's own form (issue #5: every GLM call was "malformed") and Qwen's, streamed and whole, and a
call quoted in a code fence or inline code staying text (Strata #1058), and DeepSeek's DSML in V4's and V4.1's spelling.
    python -m unittest serve.test_tools"""
from __future__ import annotations

import json
import sys
import time
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, OutputParser, parse_tool_call  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [
    {"name": "get_weather", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"},
        "where": {"type": "object"}, "note": {"type": "string"}}}},
    {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}},
]
GLM = ("<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n<arg_key>days</arg_key>"
       "\n<arg_value>3</arg_value>\n<arg_key>metric</arg_key>\n<arg_value>true</arg_value>\n<arg_key>where</arg_key>"
       "\n<arg_value>{\"lat\": 48.9, \"lon\": 2.35}</arg_value>\n<arg_key>note</arg_key>\n<arg_value>123</arg_value>"
       "\n</tool_call>")
GLM_ARGS = {"city": "Paris", "days": 3, "metric": True, "where": {"lat": 48.9, "lon": 2.35}, "note": "123"}
QWEN = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>"
        "\n</function>\n</tool_call>")


def run(text: str, pieces: int | None = None, tools=TOOLS) -> list:
    """The parser's events for `text` after </think>, fed whole (pieces=None) or `pieces` characters at a time."""
    p = OutputParser(thinking=True, tools=tools, stream_tools=True)
    text = "Let me look.</think>\n\n" + text
    out = []
    step = pieces or len(text)
    for i in range(0, len(text), step):
        out += p.feed(text[i:i + step])
    return out + p.finish()


def calls(events) -> list:
    return [e.call for e in events if e.kind == "tool_call"]


def content(events) -> str:
    return "".join(e.text for e in events if e.kind == "content")


class SlippedCallOpening(unittest.TestCase):
    """Qwen-style calls whose opening tag slipped (<=NAME>, <function:NAME>, <NAME> of a declared tool) are calls."""

    def test_slips(self):
        for opening in ("<=get_weather>", "<function:get_weather>", "<function get_weather>", "<get_weather>"):
            for pieces in (None, 1, 7):
                text = "<tool_call>\n" + opening + "\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
                with self.subTest(opening=opening, pieces=pieces):
                    got = calls(run(text, pieces))
                    self.assertEqual([(c.name, c.arguments.get("city")) for c in got], [("get_weather", "Paris")])

    def test_undeclared_name_stays_text(self):
        text = "<tool_call>\n<rm_rf>\n<parameter=path>\n/\n</parameter>\n</function>\n</tool_call>"
        self.assertEqual(calls(run(text)), [])
        self.assertIn("rm_rf", content(run(text)))


class GlmCalls(unittest.TestCase):
    def test_whole(self):
        c = parse_tool_call(GLM[len("<tool_call>"):-len("</tool_call>")], TOOLS[0])
        self.assertEqual((c.name, c.arguments), ("get_weather", GLM_ARGS))

    def test_without_a_schema_and_without_arguments(self):
        c = parse_tool_call("get_weather<arg_key>days</arg_key><arg_value>3</arg_value>")
        self.assertEqual(c.arguments, {"days": 3})
        self.assertEqual(parse_tool_call("now").arguments, {})

    def test_malformed(self):
        for body in ("<arg_key>a</arg_key><arg_value>1</arg_value>", "f<arg_key>a<arg_value>1</arg_value>",
                     "f junk <b>"):
            with self.assertRaises(ValueError, msg=body):
                parse_tool_call(body)

    def test_unclosed_parameter(self):
        bad = "<function=get_weather>\n<parameter=city\nParis\n</parameter>\n</function>"
        with self.assertRaises(ValueError):
            parse_tool_call(bad)
        for stream in (True, False):
            for pieces in (None, 1, 5):
                p = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream)
                text = "Let me check.\n<tool_call>\n" + bad + "\n</tool_call>\nDone."
                chunks = [text] if pieces is None else [text[i:i + pieces] for i in range(0, len(text), pieces)]
                ev = []
                for c in chunks:
                    ev += p.feed(c)
                ev += p.finish()
                self.assertTrue(ev, msg=(stream, pieces))
                self.assertFalse([e for e in ev if e.kind == "tool_call" and e.call and e.call.arguments.get("city") == "Paris"])

    def test_streamed_as_whole(self):
        for pieces in (None, 1, 3, 7):
            ev = run("Checking the weather.\n\n" + GLM, pieces)
            got = calls(ev)
            self.assertEqual(len(got), 1, pieces)
            self.assertEqual((got[0].name, got[0].arguments), ("get_weather", GLM_ARGS), pieces)
            starts = [e for e in ev if e.kind == "tool_start"]
            self.assertEqual([s.call.name for s in starts], ["get_weather"], pieces)
            streamed = "".join(e.text for e in ev if e.kind == "tool_args")
            self.assertEqual(json.loads(streamed), GLM_ARGS, pieces)       # the pieces add up to the arguments
            self.assertEqual(starts[0].call.id, got[0].id)
            self.assertEqual(content(ev), "Checking the weather.", pieces)

    def test_a_string_value_keeps_its_newlines(self):
        body = "bash<arg_key>command</arg_key><arg_value>\nls -la\n</arg_value>"
        self.assertEqual(parse_tool_call(body, TOOLS[1]).arguments["command"], "\nls -la\n")
        for pieces in (None, 1):
            ev = run("<tool_call>" + body + "</tool_call>", pieces)
            streamed = "".join(e.text for e in ev if e.kind == "tool_args")
            self.assertEqual(json.loads(streamed), {"command": "\nls -la\n"}, pieces)

    def test_two_calls(self):
        two = GLM + "\n<tool_call>bash<arg_key>command</arg_key><arg_value>date</arg_value></tool_call>"
        for pieces in (None, 2):
            got = calls(run(two, pieces))
            self.assertEqual([c.name for c in got], ["get_weather", "bash"], pieces)
            self.assertEqual(got[1].arguments, {"command": "date"})

    def test_qwen_form_still_works(self):
        for pieces in (None, 1, 5):
            ev = run(QWEN, pieces)
            got = calls(ev)
            self.assertEqual((got[0].name, got[0].arguments), ("get_weather", {"city": "Paris", "days": 3}), pieces)
            streamed = "".join(e.text for e in ev if e.kind == "tool_args")
            self.assertEqual(json.loads(streamed), {"city": "Paris", "days": 3}, pieces)


DSML_V4 = ("<\uff5cDSML\uff5ctool_calls>\n<\uff5cDSML\uff5cinvoke name=\"get_weather\">\n"
           "<\uff5cDSML\uff5cparameter name=\"city\" string=\"true\">Paris</\uff5cDSML\uff5cparameter>\n"
           "<\uff5cDSML\uff5cparameter name=\"days\" string=\"false\">3</\uff5cDSML\uff5cparameter>\n"
           "</\uff5cDSML\uff5cinvoke>\n</\uff5cDSML\uff5ctool_calls>")
DSML_V41 = ("<\uff5cDSML\uff5c calls>\n<\uff5cDSML\uff5c invoke name=\"get_weather\">\n"
            "<\uff5cDSML\uff5c parameter name=\"city\" string=\"true\">Paris</\uff5cDSML\uff5c parameter>\n"
            "<\uff5cDSML\uff5c parameter name=\"days\" string=\"false\">3</\uff5cDSML\uff5c parameter>\n"
            "</\uff5cDSML\uff5c invoke>\n</\uff5cDSML\uff5c calls>")


class DsmlCalls(unittest.TestCase):
    """DeepSeek V4 writes <｜DSML｜tool_calls>; V4.1 writes <｜DSML｜ calls> with a space after the token (its invokes
    and parameters too).  Before, V4.1's calls came back as text and agents stopped at the first tool call."""

    def calls(self, text, pieces=None):
        out = run(text, pieces)
        self.assertEqual([e.kind for e in out if e.kind == "content"], [])
        return [(e.call.name, e.call.arguments) for e in out if e.kind == "tool_call"]

    def test_both_spellings_whole_and_streamed(self):
        for text in (DSML_V4, DSML_V41):
            for pieces in (None, 1, 7):
                self.assertEqual(self.calls(text, pieces), [("get_weather", {"city": "Paris", "days": 3})])

    def test_unterminated_block_is_text_with_its_own_tag(self):
        out = run(DSML_V41[:60])
        self.assertTrue("".join(e.text for e in out if e.kind == "content").startswith("<\uff5cDSML\uff5c calls>"))

    def test_opening_tag_alone_stays_text(self):
        for text in (DSML_V4[:DSML_V4.index(">") + 1], DSML_V41[:DSML_V41.index(">") + 1]):
            out = run(text)
            self.assertEqual("".join(e.text for e in out if e.kind == "content"), text)

    def test_template_respelled_for_a_v41_pack(self):
        # both ways a template spells the tags: the token written out, and a variable holding it (the ISTA GGUFs)
        src = ("{%- set dsml_token = '\\uff5cDSML\\uff5c' -%}Write <\uff5cDSML\uff5ctool_calls> blocks.\n"
               "{{ obj.dsml_token ~ 'parameter' }} {{ \"dsml_token + 'invoke'\" }}\n"
               "{%- set t = '\uff5cDSML\uff5c' -%}{%- set t = 'x' -%}{{ t + 'invoke' }}|"
               "{{ '<' + (dsml_token|string) + 'tool_calls>' }}{{ '<' ~ dsml_token ~ ('invoke' if 1 else 'parameter') }}|"
               "{%- for m in messages -%}{%- if m.tool_calls -%}{{ '<' + dsml_token + 'tool_calls>' }}"
               "{%- for c in m.tool_calls -%}{{ '<' + dsml_token + 'invoke name=\"' + c.function.name + '\">' }}"
               "{{ '<' + dsml_token + 'parameter name=\"p\">v</' + dsml_token + 'parameter>' }}"
               "{{ '</' + dsml_token + 'invoke>' }}{%- endfor -%}{{ '</' + dsml_token + 'tool_calls>' }}"
               "{%- else -%}{{ m.content }}{%- endif -%}{%- endfor -%}")
        quoted = "say <\uff5cDSML\uff5ctool_calls> and <\uff5cDSML\uff5cparameterized>"
        msgs = [{"role": "user", "content": quoted},
                {"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": {}}}]}]
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "chat_template.jinja"
            p.write_text(src, encoding="utf-8")
            v4, v41 = ChatTemplate(p), ChatTemplate(p, dsml41=True)
            out4 = v4.render(msgs, add_generation_prompt=False, obj={"dsml_token": "x_"})
            out41 = v41.render(msgs, add_generation_prompt=False, obj={"dsml_token": "x_"})
            p.write_text(src.replace("tool_calls>", " calls>").replace("+ 'invoke", "+ ' invoke")
                         .replace("+ 'parameter", "+ ' parameter").replace("('invoke' if 1 else 'parameter')",
                                                                           "(' invoke' if 1 else ' parameter')"),
                         encoding="utf-8")
            again = ChatTemplate(p, dsml41=True)            # already V4.1's spelling: nothing to change
        self.assertEqual((v4.dsml41, v41.dsml41, again.dsml41), (False, True, False))
        self.assertIn("x_parameter dsml_token + 'invoke'xinvoke|", out41)   # an attribute, a string, a variable set twice
        self.assertIn("|<\uff5cDSML\uff5c calls><\uff5cDSML\uff5c invoke|", out41)  # through a filter and a conditional
        self.assertIn("<\uff5cDSML\uff5ctool_calls><\uff5cDSML\uff5cinvoke name=\"bash\">", out4)
        self.assertIn("Write <\uff5cDSML\uff5c calls> blocks.", out41)
        self.assertIn("<\uff5cDSML\uff5c calls><\uff5cDSML\uff5c invoke name=\"bash\"><\uff5cDSML\uff5c parameter "
                      "name=\"p\">v</\uff5cDSML\uff5c parameter></\uff5cDSML\uff5c invoke></\uff5cDSML\uff5c calls>", out41)
        self.assertIn(quoted, out41)                    # a message's own text is never respelled


class GenerationTag(unittest.TestCase):
    def test_laguna_generation_block_renders_its_body(self):
        # Laguna-S-2.1's template wraps the assistant's turn in transformers' {% generation %} marker
        src = ("{%- for m in messages -%}{%- if m.role == 'assistant' -%}<assistant>{%- generation -%}{{ m.content }}"
               "</assistant>{%- endgeneration -%}{%- else -%}<user>{{ m.content }}</user>{%- endif -%}{%- endfor -%}"
               "{%- if add_generation_prompt -%}<assistant>{%- endif -%}")
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "chat_template.jinja"
            p.write_text(src, encoding="utf-8")
            out = ChatTemplate(p).render([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                          {"role": "user", "content": "c"}])
        self.assertEqual(out, "<user>a</user><assistant>b</assistant><user>c</user><assistant>")


class ThinkToken(unittest.TestCase):
    """think_token: the reasoning ends at the special token only - DeepSeek V4.1 wrote `</think>` while explaining a
    chat template, and its reasoning's second half went out as the answer."""

    def text(self, events, kind):
        return "".join(e.text for e in events if e.kind == kind)

    def test_the_tag_written_inside_the_reasoning(self):
        p = OutputParser(thinking=True, think_token=True)
        ev = p.feed("It appends `</think>` if thinking is off.")
        ev += p.feed(" Done.")
        ev += p.feed("</think>", think_end=True)
        ev += p.feed("\n\nThe answer.")
        ev += p.finish()
        self.assertEqual(self.text(ev, "reasoning"), "It appends `</think>` if thinking is off. Done.")
        self.assertEqual(self.text(ev, "content"), "The answer.")

    def test_the_end_token_with_held_text(self):
        p = OutputParser(thinking=True, think_token=True)
        ev = p.feed("plan</think>", think_end=True) + p.feed("answer") + p.finish()
        self.assertEqual(self.text(ev, "reasoning"), "plan")
        self.assertEqual(self.text(ev, "content"), "answer")

    def test_text_mode_unchanged(self):
        ev = OutputParser(thinking=True).feed("plan</think>answer")
        self.assertEqual(self.text(ev, "reasoning"), "plan")


class QuotedCalls(unittest.TestCase):
    """A call quoted in the answer's code fence or inline code is text; at the top level it is a call."""

    def test_in_a_fence(self):
        for fence in ("```", "~~~", "```xml"):
            text = f"The format looks like this:\n\n{fence}\n" + GLM + "\n" + fence[:3] + "\n\nThat's all."
            for pieces in (None, 1, 4):
                ev = run(text, pieces)
                self.assertEqual(calls(ev), [], (fence, pieces))
                self.assertIn("<tool_call>get_weather", content(ev))
                self.assertTrue(content(ev).endswith("That's all."), (fence, pieces))

    def test_in_inline_code(self):
        text = "Write `<tool_call>bash<arg_key>command</arg_key><arg_value>rm -rf build</arg_value></tool_call>` to run it."
        for pieces in (None, 1, 3):
            ev = run(text, pieces)
            self.assertEqual(calls(ev), [], pieces)
            self.assertEqual(content(ev), text, pieces)

    def test_after_a_closed_fence_and_mid_sentence(self):
        text = "Example:\n```\n<tool_call>x</tool_call>\n```\nNow for real. " + GLM
        for pieces in (None, 1):
            got = calls(run(text, pieces))
            self.assertEqual([c.name for c in got], ["get_weather"], pieces)


class HttpToolCalls(unittest.TestCase):
    """Through the server: a GLM call reaches an OpenAI and an Anthropic client as a tool call."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "Let me look.</think>\n\n" + GLM, max_context=8192), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        r = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read())

    def test_openai(self):
        tools = [{"type": "function", "function": t} for t in TOOLS]
        code, b = self.post("/v1/chat/completions", {"model": "m", "tools": tools,
                                                      "messages": [{"role": "user", "content": "weather?"}]})
        self.assertEqual(code, 200, b)
        msg = b["choices"][0]["message"]
        self.assertEqual(b["choices"][0]["finish_reason"], "tool_calls")
        fn = msg["tool_calls"][0]["function"]
        self.assertEqual((fn["name"], json.loads(fn["arguments"])), ("get_weather", GLM_ARGS))

    def test_anthropic(self):
        tools = [{"name": t["name"], "input_schema": t["parameters"]} for t in TOOLS]
        code, b = self.post("/v1/messages", {"model": "m", "max_tokens": 4000, "tools": tools,
                                             "messages": [{"role": "user", "content": "weather?"}]})
        self.assertEqual(code, 200, b)
        use = [c for c in b["content"] if c["type"] == "tool_use"]
        self.assertEqual((use[0]["name"], use[0]["input"]), ("get_weather", GLM_ARGS))
        self.assertEqual(b["stop_reason"], "tool_use")


class ClientStops(unittest.TestCase):
    """The client's stop strings (OpenAI "stop", Anthropic "stop_sequences") end the answer - not the reasoning - and
    tool_choice "none" offers no tools; an empty assistant turn is not rendered (from Strata 0.1.40)."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, "Thinking: END is a word.</think>\n\nHello there END and more", max_context=8192)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.tok = tok

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    post = HttpToolCalls.post

    def test_openai_stop(self):
        msgs = [{"role": "user", "content": "hi"}]
        for stop in ("END", ["nope", "END"]):
            code, b = self.post("/v1/chat/completions", {"model": "m", "messages": msgs, "stop": stop})
            self.assertEqual(code, 200, b)
            self.assertEqual(b["choices"][0]["message"]["content"].strip(), "Hello there")
            self.assertEqual(b["choices"][0]["finish_reason"], "stop")
            self.assertIn("END is a word", b["choices"][0]["message"].get("reasoning_content", ""))  # not cut there

    def test_openai_stop_streamed(self):
        r = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(
            {"model": "m", "stream": True, "stop": ["END"], "messages": [{"role": "user", "content": "hi"}]}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=30) as resp:
            body = resp.read().decode()
        text = "".join((json.loads(line[6:])["choices"][0]["delta"].get("content") or "")
                       for line in body.splitlines() if line.startswith("data: {"))
        self.assertEqual(text.strip(), "Hello there")
        self.assertNotIn("EN", text.replace("Hello there", ""))

    def test_anthropic_stop_sequence(self):
        code, b = self.post("/v1/messages", {"model": "m", "max_tokens": 200, "stop_sequences": ["END"],
                                             "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 200, b)
        self.assertEqual((b["stop_reason"], b["stop_sequence"]), ("stop_sequence", "END"))
        self.assertEqual("".join(c.get("text", "") for c in b["content"] if c["type"] == "text").strip(), "Hello there")
        code, b = self.post("/v1/messages", {"model": "m", "max_tokens": 200,
                                             "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual((b["stop_reason"], b["stop_sequence"]), ("end_turn", None))

    def test_tool_choice_none(self):
        tools = [{"type": "function", "function": TOOLS[0]}]
        self.post("/v1/chat/completions", {"model": "m", "tools": tools, "tool_choice": "none",
                                           "messages": [{"role": "user", "content": "hi"}]})
        self.assertNotIn("get_weather", self.tok.decode(self.engine.last_prompt))
        self.post("/v1/chat/completions", {"model": "m", "tools": tools, "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn("get_weather", self.tok.decode(self.engine.last_prompt))

    def test_empty_assistant_turns_are_dropped(self):
        from serve.frontend import openai_to_messages
        msgs, _, _ = openai_to_messages({"messages": [{"role": "user", "content": "a"},
                                                      {"role": "assistant", "content": ""},
                                                      {"role": "user", "content": "b"}]})
        self.assertEqual([m["role"] for m in msgs], ["user", "user"])


class JsonMode(unittest.TestCase):
    """response_format json_object: the JSON alone, out of a code fence or the prose around it (Strata #762)."""

    def test_extraction(self):
        from serve.server import json_from_text
        cases = [('{"a": 1}', '{"a": 1}'), ('Sure:\n```json\n{"a": [1, 2]}\n```\nDone.', '{"a": [1, 2]}'),
                 ('Here it is: {"ok": true, "n": {"x": "}"}} - enjoy', '{"ok": true, "n": {"x": "}"}}'),
                 ('[1, 2, 3]', '[1, 2, 3]'), ('no json here', 'no json here')]
        for given, want in cases:
            self.assertEqual(json_from_text(given), want, given)

    def test_over_http(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, '</think>\n\nSure! ```json\n{"city": "Paris"}\n```', max_context=8192), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
        code, b = HttpToolCalls.post(self, "/v1/chat/completions", {
            "model": "m", "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": "json"}]})
        self.assertEqual(json.loads(b["choices"][0]["message"]["content"]), {"city": "Paris"})


class ForcedToolCalls(unittest.TestCase):
    """tool_choice "required" / a named function (Anthropic "any" / "tool"): the answer starts with the call - the
    prompt ends with its opening, thinking off - and the model's continuation is the call (from Strata #790)."""

    BODY = "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"

    def server(self, script):
        tok = ByteTokenizer()
        engine = MockEngine(tok, script, max_context=8192)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"
        return engine, tok

    post = HttpToolCalls.post

    def test_named(self):
        engine, tok = self.server(self.BODY)
        tools = [{"type": "function", "function": TOOLS[0]}]
        code, b = self.post("/v1/chat/completions", {"model": "m", "tools": tools, "messages": [
            {"role": "user", "content": "hi"}], "tool_choice": {"type": "function", "function": {"name": "get_weather"}}})
        self.assertEqual(code, 200, b)
        self.assertTrue(tok.decode(engine.last_prompt).endswith("<tool_call>\n<function=get_weather>\n"))
        fn = b["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual((fn["name"], json.loads(fn["arguments"])), ("get_weather", {"city": "Paris"}))

    def test_required_and_anthropic_any(self):
        engine, tok = self.server("<function=get_weather>\n" + self.BODY)
        tools = [{"type": "function", "function": TOOLS[0]}]
        code, b = self.post("/v1/chat/completions", {"model": "m", "tools": tools, "tool_choice": "required",
                                                      "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(b["choices"][0]["finish_reason"], "tool_calls", b)
        self.assertTrue(tok.decode(engine.last_prompt).endswith("<tool_call>\n"))
        atools = [{"name": "get_weather", "input_schema": TOOLS[0]["parameters"]}]
        code, b = self.post("/v1/messages", {"model": "m", "max_tokens": 300, "tools": atools,
                                             "tool_choice": {"type": "any"},
                                             "messages": [{"role": "user", "content": "hi"}]})
        use = [c for c in b["content"] if c["type"] == "tool_use"]
        self.assertEqual((b["stop_reason"], use[0]["name"], use[0]["input"]), ("tool_use", "get_weather", {"city": "Paris"}))

    def test_auto_adds_nothing(self):
        engine, tok = self.server("</think>\n\nhello")
        tools = [{"type": "function", "function": TOOLS[0]}]
        self.post("/v1/chat/completions", {"model": "m", "tools": tools, "tool_choice": "auto",
                                           "messages": [{"role": "user", "content": "hi"}]})
        self.assertFalse(tok.decode(engine.last_prompt).rstrip().endswith("<tool_call>"))


DSML_LONG = ("<\uff5cDSML\uff5c calls>\n<\uff5cDSML\uff5c invoke name=\"bash\">\n"
             "<\uff5cDSML\uff5c parameter name=\"command\" string=\"true\">%s</\uff5cDSML\uff5c parameter>\n"
             "</\uff5cDSML\uff5c invoke>\n</\uff5cDSML\uff5c calls>")
QWEN_LONG = "<tool_call>\n<function=bash>\n<parameter=command>\n%s\n</parameter>\n</function>\n</tool_call>"
GLM_LONG = "<tool_call>bash<arg_key>command</arg_key><arg_value>%s</arg_value></tool_call>"
QWEN_RAW_LONG = "<tool_call>\n<function=get_weather>\n<parameter=where>\n%s\n</parameter>\n</function>\n</tool_call>"


def merged(events) -> list:
    """Events with neighbouring pieces of the same text kind joined and call ids dropped: what a client sees, however
    the text was cut into pieces."""
    out = []
    for e in events:
        call = None if e.call is None else (e.call.name, json.dumps(e.call.arguments, sort_keys=True))
        if out and out[-1][0] == e.kind and e.kind in ("reasoning", "content", "tool_args") and out[-1][2] == call:
            out[-1] = (e.kind, out[-1][1] + e.text, call)
        else:
            out.append((e.kind, e.text, call))
    return out


def fed(text: str, pieces, stream_tools=True, think_token=False) -> tuple:
    """(merged events, the parser) for `text` fed in `pieces` (an int: that many characters at a time; a list of
    cut offsets)."""
    p = OutputParser(thinking=True, tools=TOOLS, stream_tools=stream_tools, think_token=think_token)
    cuts = list(range(0, len(text), pieces)) if isinstance(pieces, int) else [0] + list(pieces)
    out = []
    for a, b in zip(cuts, cuts[1:] + [len(text)]):
        out += p.feed(text[a:b])
    return merged(out + p.finish()), p


class LongStreamedArguments(unittest.TestCase):
    """A long tool argument (an agent's Write of a whole file) streamed in small pieces: the same events as the text
    fed whole, in time that grows with the length, not with the length times the pieces (each piece used to search
    the whole call from its start for the end tag - Maya #626's finding)."""

    FORMS = {"qwen": QWEN_LONG, "glm": GLM_LONG, "dsml": DSML_LONG, "qwen-raw": QWEN_RAW_LONG}

    def long_text(self, form) -> str:
        if form == "qwen-raw":
            value = "[" + ", ".join(str(i) for i in range(40000)) + "]"            # a 200 KB array (a non-string value)
        else:
            value = ("def f(x):\n    return x * 2  # a line of a file being written\n" * 4000)
        return "Writing the file.</think>\n\n" + self.FORMS[form] % value[:200_000] + "\nDone."

    def test_same_events_as_whole_and_linear_work(self):
        for form in self.FORMS:
            for stream_tools in (True, False):
                text = self.long_text(form)
                whole, _ = fed(text, len(text), stream_tools)
                for step in (8, 64):
                    with self.subTest(form=form, stream_tools=stream_tools, step=step):
                        t0 = time.perf_counter()
                        got, p = fed(text, step, stream_tools)
                        took = time.perf_counter() - t0
                        self.assertEqual(got, whole)
                        self.assertEqual([k for k, _, _ in got if k == "tool_call"], ["tool_call"])
                        if stream_tools and form != "dsml":          # (a DSML block is reported once it is whole)
                            streamed = "".join(t for k, t, _ in got if k == "tool_args")
                            final = next(c for k, _, c in got if k == "tool_call")
                            self.assertEqual(json.loads(streamed), json.loads(final[1]))
                        # the characters looked at: each piece plus the overlap a split tag needs (11-15 characters), by
                        # the end-tag search and the streamed value's - about 2 x (step + overlap) / step x the length
                        self.assertLess(p.scanned, len(text) * (2 * (step + 16) / step + 1), (form, step, p.scanned))
                        self.assertLess(took, 5.0, (form, step, took))   # (the old search: ~1 s at step 8, growing n^2)

    def test_quadratic_growth_is_gone(self):
        text = self.long_text("qwen")
        half = text[:len(text) // 2]
        _, p_half = fed(half, 8)
        _, p_full = fed(text, 8)
        self.assertLess(p_full.scanned, 2.5 * p_half.scanned)      # twice the text: about twice the work


class SplitMarkers(unittest.TestCase):
    """Every tag the parser knows cut at every offset between two pieces (and three): the events of the whole text,
    in every form (GLM, Qwen, DSML V4 and V4.1, a quoted call, an unterminated one, a think token)."""

    TEXTS = {
        "glm": "Checking.</think>\n\nNow.\n\n" + GLM + "\nAfter.",
        "qwen": "Checking.</think>\n\n" + QWEN,
        "dsml-v4": "Checking.</think>\n\n" + DSML_V4,
        "dsml-v41": "Checking.</think>\n\n" + DSML_V41 + "\nAfter.",
        "two-calls": "T</think>" + GLM + "\n<tool_call>bash<arg_key>command</arg_key><arg_value>date</arg_value></tool_call>",
        "quoted": "T</think>Write `" + GLM + "` to run it, or:\n```\n" + QWEN + "\n```\nThat is all.",
        "unterminated-qwen": "T</think>" + QWEN[:QWEN.index("</parameter>") + 5],
        "unterminated-glm": "T</think>" + GLM[:GLM.index("</arg_value>") + 7],
        "unterminated-dsml": "T</think>" + DSML_V41[:60],
        "no-call": "T</think>a plain answer with a < sign and `code` in it\n",
        "glm-newlines": "T</think><tool_call>bash<arg_key>command</arg_key><arg_value>\nls -la\n</arg_value></tool_call>",
    }

    def test_every_offset(self):
        for name, text in self.TEXTS.items():
            for stream_tools in (True, False):
                whole, _ = fed(text, len(text), stream_tools)
                for cut in range(1, len(text)):
                    got, _ = fed(text, [cut], stream_tools)
                    self.assertEqual(got, whole, (name, stream_tools, cut))
                for step in (1, 2, 3, 5, 7, 11, 13):
                    got, _ = fed(text, step, stream_tools)
                    self.assertEqual(got, whole, (name, stream_tools, step))

    def test_three_pieces_around_the_end_tags(self):
        for name, text in self.TEXTS.items():
            whole, _ = fed(text, len(text))
            for tag in ("</tool_call>", "</arg_value>", "</parameter>", "</\uff5cDSML\uff5c calls>", "</think>"):
                at = text.find(tag)
                if at < 0:
                    continue
                for a in range(max(1, at - 2), at + len(tag)):
                    for b in range(a + 1, min(len(text), at + len(tag) + 2)):
                        got, _ = fed(text, [a, b])
                        self.assertEqual(got, whole, (name, tag, a, b))

    def test_think_token(self):
        text = "It writes `</think>` itself. Done.</think>\n\n" + GLM
        p = OutputParser(thinking=True, tools=TOOLS, stream_tools=True, think_token=True)
        end = text.index("Done.</think>") + len("Done.</think>")
        out = p.feed(text[:end - 8]) + p.feed(text[end - 8:end], think_end=True)
        for i in range(end, len(text), 3):
            out += p.feed(text[i:i + 3])
        out += p.finish()
        got = merged(out)
        self.assertEqual(got[0], ("reasoning", "It writes `</think>` itself. Done.", None))
        self.assertEqual([k for k, _, _ in got if k == "tool_call"], ["tool_call"])


class StreamSockets(unittest.TestCase):
    def test_no_nagle_on_the_server_connections(self):
        from serve.server import make_handler
        handler = make_handler(Service(MockEngine(ByteTokenizer(), "", max_context=512), ByteTokenizer(),
                                       ChatTemplate(ROOT / "serve/chat_template.jinja")))
        self.assertTrue(handler.disable_nagle_algorithm)     # (socketserver sets TCP_NODELAY on each connection)


if __name__ == "__main__":
    unittest.main()
