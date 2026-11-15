"""serve/test_agent_compat.py - agent-facing behaviour against scripted engines (no GPU, no pack):

  * the answer room: a capped request that is still thinking gets its reasoning closed in time to answer;
  * an end-of-turn token inside the reasoning or an open tool call is kept as text and the answer goes on;
  * usage says who closed the reasoning (OpenAI completion_tokens_details, Anthropic output_tokens_details);
  * tool_choice "none" never hands out a call or its markup;
  * Claude Code's request shapes: mid-conversation system messages, deferred tools and tool_reference results,
    unknown blocks, thinking signatures that carry the reasoning back, cache_control, count_tokens.

    python -m unittest serve.test_agent_compat -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import (TOOL_NONE_NOTE, ChatTemplate, OutputParser, anthropic_to_messages,  # noqa: E402
                            signature_text, thinking_signature)
from serve.server import IM_END, ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 16384


class ThinkTok(ByteTokenizer):
    """The byte tokenizer with the reasoning tags as special tokens (every model family Abyss serves has </think> as one
    token: Qwen3.8 248069, GLM 154842, DeepSeek V4 / V4.1 128822, Laguna 19)."""
    SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>"]


class BudgetEngine:
    """Thinks `think` tokens unless the request's think budget closes it first - the engine's way: at the budget,
    the bridge (think_close) then </think> are forced - then answers `answer`.  Records every run's sampling."""

    def __init__(self, tok, think: int, answer: str = "The answer.", notes=None):
        self.tok, self.think, self.answer, self.max_context = tok, think, answer, CTX
        self.end_id = tok.encode("</think>", parse_special=True)[0]
        self.runs, self.notes0 = [], notes or []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.runs.append(dict(sampling or {}, _max_new=max_new))
        self.notes = list(self.notes0)
        budget = int((sampling or {}).get("_think_budget") or 0)
        out, produced = [], 0
        while produced < self.think and not (budget and produced >= budget):
            out.append(ord("r"))
            produced += 1
        if budget and produced >= budget:
            out += list((sampling or {}).get("_think_close") or [])
        out.append(self.end_id)
        out += self.tok.encode("\n\n" + self.answer) + self.tok.encode(IM_END, parse_special=True)
        for t in out[:max_new]:
            yield t


class Base(unittest.TestCase):
    engine = None
    tok = ByteTokenizer()
    svc_kwargs: dict = {}

    @classmethod
    def setUpClass(cls):
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), **cls.svc_kwargs)
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

    def quiet_post(self, path, body):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            r = self.post(path, body)
        self.log = out.getvalue()
        return r


class AnswerRoom(Base):
    """Thinking on: the reasoning is closed when a quarter of max tokens is left (at least 1024, at most half), so a
    capped request ends with an answer - with or without the run config's thinking budget.  STRATA_ANSWER_ROOM=0: no
    budget at all (with the config's budget 0, as here)."""
    tok = ThinkTok()
    engine = BudgetEngine(tok, think=6000)

    def ask(self, api="openai", **kw):
        msgs = [{"role": "user", "content": "think hard"}]
        if api == "openai":
            return self.quiet_post("/v1/chat/completions", {"model": "m", "messages": msgs, **kw})
        return self.quiet_post("/v1/messages", {"model": "m", "messages": msgs, **kw})

    def test_capped_request_answers(self):
        self.svc.think_bridge = "\nStop.\n"
        try:
            s, b = self.ask(max_tokens=4000)
        finally:
            self.svc.think_bridge = None
        self.assertEqual(s, 200, b)
        run = self.engine.runs[-1]
        self.assertEqual(run["_think_budget"], 4000 - 1024)          # max(1/4, 1024) of 4000 left for the answer
        self.assertEqual(run["_think_close"], self.tok.encode("\nStop.\n"))   # the bridge goes before </think>
        self.assertTrue(b["choices"][0]["message"]["reasoning_content"].endswith("r\nStop.\n"))
        m = b["choices"][0]["message"]
        self.assertEqual(m["content"], "The answer.")
        self.assertEqual(b["choices"][0]["finish_reason"], "stop")
        d = b["usage"]["completion_tokens_details"]
        self.assertEqual(d["reasoning_closed_by"], "answer_room")
        self.assertGreater(d["reasoning_tokens"], 2976)
        self.assertIn("answer room", self.log)

    def test_large_cap_reserves_a_quarter(self):
        s, b = self.ask(max_tokens=12000)
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.runs[-1]["_think_budget"], 9000)
        self.assertEqual(b["usage"]["completion_tokens_details"]["reasoning_closed_by"], "model")   # 6000 < 9000

    def test_small_cap_keeps_half_for_thinking(self):
        s, b = self.ask(max_tokens=800)
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.runs[-1]["_think_budget"], 400)

    def test_unlimited_request_leaves_a_quarter_of_the_context(self):
        s, b = self.ask()
        self.assertEqual(s, 200, b)
        run = self.engine.runs[-1]
        self.assertEqual(run["_think_budget"], run["_max_new"] - run["_max_new"] // 4)
        self.assertEqual(b["usage"]["completion_tokens_details"]["reasoning_closed_by"], "model")

    def test_opt_out(self):
        os.environ["STRATA_ANSWER_ROOM"] = "0"
        try:
            s, b = self.ask(max_tokens=4000)
        finally:
            del os.environ["STRATA_ANSWER_ROOM"]
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.runs[-1]["_think_budget"], 0)
        self.assertEqual(b["choices"][0]["finish_reason"], "length")   # thought to the end: no answer
        self.assertEqual(b["usage"]["completion_tokens_details"]["reasoning_closed_by"], "unclosed")

    def test_run_config_budget_smaller_wins(self):
        self.svc.think_budget = 1000
        try:
            s, b = self.ask(max_tokens=4000)
        finally:
            self.svc.think_budget = 0
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.runs[-1]["_think_budget"], 1000)
        self.assertEqual(b["usage"]["completion_tokens_details"]["reasoning_closed_by"], "budget")

    def test_anthropic_usage_details(self):
        s, b = self.ask("anthropic", max_tokens=4000)
        self.assertEqual(s, 200, b)
        self.assertEqual([c["type"] for c in b["content"]], ["thinking", "text"])
        self.assertEqual(b["usage"]["output_tokens_details"]["reasoning_closed_by"], "answer_room")
        self.assertEqual(b["stop_reason"], "end_turn")

    def test_thinking_off_has_no_details(self):
        s, b = self.ask(max_tokens=4000, reasoning_effort="none")
        self.assertEqual(s, 200, b)
        self.assertNotIn("completion_tokens_details", b["usage"])

    def test_engine_stop_ids_go_along(self):
        self.ask(max_tokens=100)
        self.assertEqual(self.engine.runs[-1]["_stop_ids"], sorted(self.svc.stop_ids))


class ClosedByGuard(Base):
    """The engine's NOTE lines say the end guard closed the reasoning (an end token inside it became </think>)."""
    tok = ThinkTok()
    engine = BudgetEngine(tok, think=50, notes=["the turn ended inside the reasoning: the model was kept going"])

    def test_end_guard(self):
        s, b = self.quiet_post("/v1/chat/completions", {"model": "m", "max_tokens": 4000,
                                                         "messages": [{"role": "user", "content": "q"}]})
        self.assertEqual(s, 200, b)
        self.assertEqual(b["usage"]["completion_tokens_details"]["reasoning_closed_by"], "end_guard")


class SoftEnd(Base):
    """An end-of-turn token inside an open tool call: kept as text, and the answer goes on - from where the engine
    stopped, as a new run of the prompt plus the answer so far."""
    TOOLS = [{"type": "function", "function": {"name": "write", "parameters": {
        "type": "object", "properties": {"content": {"type": "string"}}}}}]
    engine = MockEngine(ByteTokenizer(), ["</think>\n\n<tool_call>\n<function=write>\n<parameter=content>\nabc",
                                          "def\n</parameter>\n</function>\n</tool_call>"], max_context=CTX)

    def ask(self, **kw):
        self.engine.turns = 0
        return self.quiet_post("/v1/chat/completions", {"model": "m", "max_tokens": 500, "tools": self.TOOLS,
                                                         "messages": [{"role": "user", "content": "write"}], **kw})

    def test_end_token_in_a_call_is_text(self):
        s, b = self.ask()
        self.assertEqual(s, 200, b)
        calls = b["choices"][0]["message"]["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"content": "abc<|im_end|>def"})
        self.assertEqual(b["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(self.engine.turns, 2)
        first = self.engine.scripts[0]
        self.assertEqual(self.engine.last_prompt[-len(first):], first)   # the second run: prompt + answer so far
        self.assertEqual(b["usage"]["completion_tokens"], len(first) + len(self.engine.scripts[1]))
        self.assertIn("kept as text", self.log)

    def test_opt_out(self):
        os.environ["STRATA_SOFT_END"] = "0"
        try:
            s, b = self.ask()
        finally:
            del os.environ["STRATA_SOFT_END"]
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.turns, 1)
        calls = b["choices"][0]["message"]["tool_calls"]
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"content": "abc"})   # closed as it stood


class SoftEndCap(Base):
    engine = MockEngine(ByteTokenizer(), ["</think>\n\n<tool_call>\n<function=write>\n<parameter=content>\na", "b",
                                          "c", "d"], max_context=CTX)

    def test_cap(self):
        os.environ["STRATA_SOFT_END"] = "2"
        try:
            s, b = self.quiet_post("/v1/chat/completions", {"model": "m", "max_tokens": 500, "tools": SoftEnd.TOOLS,
                                                             "messages": [{"role": "user", "content": "w"}]})
        finally:
            del os.environ["STRATA_SOFT_END"]
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.turns, 3)                           # two kept, the third ends it
        calls = b["choices"][0]["message"]["tool_calls"]
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"content": "a<|im_end|>b<|im_end|>c"})


class PlainEndStillEnds(Base):
    engine = MockEngine(ByteTokenizer(), "</think>\n\nHello.", max_context=CTX)

    def test_ordinary_answer(self):
        s, b = self.quiet_post("/v1/chat/completions", {"model": "m", "max_tokens": 500,
                                                         "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 200, b)
        self.assertEqual(b["choices"][0]["message"]["content"], "Hello.")
        self.assertEqual(b["choices"][0]["finish_reason"], "stop")
        self.assertNotIn("kept as text", self.log)


class RecordingPrompt(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_ids = list(ids)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


def prompt_text(ids):
    return ByteTokenizer().decode(ids)


class ToolChoiceNone(Base):
    """tool_choice "none" with tool calls in the conversation: a closing user note asks for prose, and a call the
    model writes anyway is never sent - neither as a call nor as its markup."""
    engine = RecordingPrompt(ByteTokenizer(), "</think>\n\nIt says 23.\n\n<tool_call>\n<function=read_file>\n"
                             "<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>", max_context=CTX)
    TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}}}}]
    HISTORY = [{"role": "user", "content": "read a.py"},
               {"role": "assistant", "content": "", "tool_calls": [
                   {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]},
               {"role": "tool", "tool_call_id": "c1", "content": "print(23)"}]

    def test_openai(self):
        s, b = self.quiet_post("/v1/chat/completions", {"model": "m", "max_tokens": 500, "tools": self.TOOLS,
                                                         "tool_choice": "none", "messages": self.HISTORY})
        self.assertEqual(s, 200, b)
        m = b["choices"][0]["message"]
        self.assertEqual(m["content"], "It says 23.")
        self.assertNotIn("tool_calls", m)
        self.assertEqual(b["choices"][0]["finish_reason"], "stop")
        text = prompt_text(self.engine.last_ids)
        self.assertIn(TOOL_NONE_NOTE, text)
        self.assertLess(text.index("print(23)"), text.index(TOOL_NONE_NOTE))   # at the end: the cache keeps the rest
        self.assertIn("tool_choice none", self.log)

    def test_openai_stream_has_no_markup(self):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(
            {"model": "m", "max_tokens": 500, "tools": self.TOOLS, "tool_choice": "none", "stream": True,
             "messages": self.HISTORY}).encode(), headers={"Content-Type": "application/json"})
        with contextlib.redirect_stdout(io.StringIO()), urllib.request.urlopen(req, timeout=30) as r:
            chunks = [json.loads(x[6:]) for x in r.read().decode().splitlines() if x.startswith("data: {")]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        self.assertFalse([d for d in deltas if d.get("tool_calls")])
        text = "".join(d.get("content") or "" for d in deltas)
        self.assertEqual(text, "It says 23.")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_anthropic(self):
        s, b = self.quiet_post("/v1/messages", {
            "model": "m", "max_tokens": 500, "tool_choice": {"type": "none"},
            "tools": [{"name": "read_file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
            "messages": [{"role": "user", "content": "read a.py"},
                         {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read_file",
                                                            "input": {"path": "a.py"}}]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                                       "content": "print(23)"}]}]})
        self.assertEqual(s, 200, b)
        self.assertEqual([c["type"] for c in b["content"]], ["text"])
        self.assertEqual(b["content"][0]["text"], "It says 23.")
        self.assertEqual(b["stop_reason"], "end_turn")

    def test_no_note_without_calls_in_the_conversation(self):
        s, b = self.quiet_post("/v1/chat/completions", {"model": "m", "max_tokens": 500, "tools": self.TOOLS,
                                                         "tool_choice": "none",
                                                         "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 200, b)
        self.assertNotIn(TOOL_NONE_NOTE, prompt_text(self.engine.last_ids))

    def test_parser_refuses(self):
        p = OutputParser(thinking=False, no_calls=True)
        ev = p.feed("ok <tool_call>\n<function=x>") + p.feed("</function></tool_call> more") + p.finish()
        self.assertEqual([(e.kind, e.text) for e in ev], [("content", "ok "), ("calls_refused", "")])


# Claude Code 2.1.323's request, as captured (bench/agentfx/cc): system blocks with cache_control, the environment as a
# system message after the first user turn, adaptive thinking with display omitted, context_management, metadata, a
# deferred tool, and after ToolSearch a tool_result holding a tool_reference
def claude_code_request(turn: int) -> dict:
    tools = [{"name": "Read", "description": "Read a file.", "input_schema": {
                 "type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}},
             {"name": "ToolSearch", "description": "Load deferred tools.", "input_schema": {
                 "type": "object", "properties": {"query": {"type": "string"}}}},
             {"name": "DeferredToolPlaceholder", "description": "never call", "input_schema": {
                 "type": "object", "properties": {}}, "defer_loading": True}]
    if turn >= 2:
        tools.insert(2, {"name": "WebFetch", "description": "Fetch a URL.", "input_schema": {
            "type": "object", "properties": {"url": {"type": "string"}}}, "defer_loading": True})
    msgs = [{"role": "user", "content": [{"type": "text", "text": "<system-reminder>\nCLAUDE.md says hi\n</system-reminder>"},
                                         {"type": "text", "text": "read hello.txt", "cache_control": {"type": "ephemeral"}}]},
            {"role": "system", "content": [{"type": "text", "text": "# Environment\nThe following deferred tools are now "
                                            "available via ToolSearch:\nWebFetch", "cache_control": {"type": "ephemeral"}}]}]
    if turn >= 2:
        msgs += [{"role": "assistant", "content": [
                     {"type": "thinking", "thinking": "I should load WebFetch.\n", "signature": "Ex0123notours"},
                     {"type": "tool_use", "id": "toolu_1", "name": "ToolSearch", "input": {"query": "select:WebFetch"}}]},
                 {"role": "user", "content": [
                     {"type": "tool_result", "tool_use_id": "toolu_1",
                      "content": [{"type": "tool_reference", "tool_name": "WebFetch"}]},
                     {"type": "text", "text": "Tool loaded.", "cache_control": {"type": "ephemeral"}}]}]
    return {"model": "claude-opus-5-5", "max_tokens": 128000, "stream": False,
            "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.323.f9d; cc_entrypoint=sdk-cli;"},
                       {"type": "text", "text": "You are a Claude agent.", "cache_control": {"type": "ephemeral"}},
                       {"type": "text", "text": "\nYou help with code.", "cache_control": {"type": "ephemeral"}}],
            "messages": msgs, "tools": tools,
            "metadata": {"user_id": '{"device_id":"d","account_uuid":"","session_id":"s-1"}'},
            "thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "medium"},
            "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}}


class ClaudeCode(Base):
    tok = ThinkTok()
    engine = RecordingPrompt(ThinkTok(), "I will read it.</think>\n\n<tool_call>\n<function=Read>\n<parameter=file_path>\n"
                             "hello.txt\n</parameter>\n</function>\n</tool_call>", max_context=CTX)

    def text(self):
        return self.tok.decode(self.engine.last_ids)

    def test_session_turns(self):
        s, b = self.quiet_post("/v1/messages?beta=true", claude_code_request(1))
        self.assertEqual(s, 200, b)                                     # 128K max_tokens over a 16K context: runs
        self.assertEqual([c["type"] for c in b["content"]], ["thinking", "tool_use"])
        self.assertEqual(b["content"][1]["input"], {"file_path": "hello.txt"})
        t1 = self.text()
        self.assertNotIn("DeferredToolPlaceholder", t1)                 # deferred tools are not in the prompt
        self.assertIn('"name": "Read"', t1)
        s, b = self.quiet_post("/v1/messages?beta=true", claude_code_request(2))
        self.assertEqual(s, 200, b)
        t2 = self.text()
        self.assertIn("Tool WebFetch loaded:", t2)                      # the definition where ToolSearch loaded it
        self.assertIn('"url"', t2)
        self.assertNotIn("Fetch a URL.", t2[:t2.index("read hello.txt")])
        # the prompt up to the end of the first turn is the same: a loaded tool does not change the system prompt
        head = t1[:t1.index("<|im_start|>assistant")]
        self.assertTrue(t2.startswith(head))

    def test_calls_to_deferred_tools_are_calls(self):
        _, tools, kw = anthropic_to_messages(claude_code_request(2))
        self.assertEqual([t["name"] for t in tools], ["Read", "ToolSearch"])
        self.assertEqual([t["name"] for t in kw["_parser_tools"]], ["Read", "ToolSearch", "WebFetch",
                                                                     "DeferredToolPlaceholder"])

    def test_unknown_blocks_and_server_tools_are_skipped(self):
        req = claude_code_request(1)
        req["messages"][0]["content"] += ["stray", {"type": "document", "source": {"type": "text", "data": "x"}},
                                          {"type": "redacted_thinking", "data": "abc"}]
        req["messages"].insert(1, {"role": "assistant", "content": [{"type": "server_tool_use", "id": "s", "name":
                                                                     "web_search", "input": {}}, {"type": "text",
                                                                                                  "text": "ok"}]})
        req["tools"].append({"type": "web_search_20250305", "name": "web_search", "max_uses": 3})
        s, b = self.quiet_post("/v1/messages", req)
        self.assertEqual(s, 200, b)
        self.assertNotIn("web_search", self.text())

    def test_signature_round_trip(self):
        req = claude_code_request(1)
        s, b = self.quiet_post("/v1/messages", req)
        self.assertEqual(s, 200, b)
        th = b["content"][0]
        self.assertEqual(signature_text(th["signature"]), th["thinking"])
        self.assertEqual(th["thinking"], "I will read it.")
        follow = [{"role": "assistant", "content": [th, {"type": "tool_use", "id": b["content"][1]["id"], "name": "Read",
                                                         "input": b["content"][1]["input"]}]},
                  {"role": "user", "content": [{"type": "tool_result", "tool_use_id": b["content"][1]["id"],
                                                "content": "hello world"}]}]
        self.quiet_post("/v1/messages", dict(req, messages=req["messages"] + follow))
        exact = self.engine.last_ids
        for changed in ("", "  I will read it.\n\n", "trimmed by the client"):   # display omitted, re-wrapped, edited
            with self.subTest(thinking=changed):
                f = [dict(follow[0], content=[dict(th, thinking=changed)] + follow[0]["content"][1:])] + follow[1:]
                s, b = self.quiet_post("/v1/messages", dict(req, messages=req["messages"] + f))
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_ids, exact)
        # a signature that is not ours: the block's own text
        f = [dict(follow[0], content=[dict(th, thinking="other", signature="Ex0123")] + follow[0]["content"][1:])]
        self.quiet_post("/v1/messages", dict(req, messages=req["messages"] + f + follow[1:]))
        self.assertIn("<think>\nother\n</think>", self.text())

    def test_stream_carries_the_signature(self):
        req = dict(claude_code_request(1), stream=True)
        r = urllib.request.Request(self.base + "/v1/messages", data=json.dumps(req).encode(),
                                   headers={"Content-Type": "application/json"})
        with contextlib.redirect_stdout(io.StringIO()), urllib.request.urlopen(r, timeout=30) as f:
            events = [json.loads(line[6:]) for line in f.read().decode().splitlines() if line.startswith("data: ")]
        sig = [e for e in events if e.get("type") == "content_block_delta" and e["delta"]["type"] == "signature_delta"]
        self.assertEqual(len(sig), 1)
        self.assertEqual(sig[0]["index"], 0)
        stop0 = next(i for i, e in enumerate(events) if e.get("type") == "content_block_stop" and e["index"] == 0)
        self.assertLess(events.index(sig[0]), stop0)                    # before the thinking block closes
        self.assertEqual(signature_text(sig[0]["delta"]["signature"]), "I will read it.")

    def test_count_tokens(self):
        req = claude_code_request(2)
        s, b = self.quiet_post("/v1/messages/count_tokens", req)
        self.assertEqual(s, 200, b)
        self.quiet_post("/v1/messages", req)
        self.assertEqual(b["input_tokens"], len(self.engine.last_ids))

    def test_signature_helpers(self):
        for t in ("", "plain", "日本語の推論\n" * 50):
            self.assertEqual(signature_text(thinking_signature(t)), t)
        for bad in (None, 5, "Ex0", "abyss1:!!!", "abyss1:" + "QUJD"):
            self.assertIsNone(signature_text(bad))


if __name__ == "__main__":
    unittest.main()
