"""Inkling 2's call messages streamed while they are written (InklingCalls + InkArgStream): the call is announced once
its name and the start of its arguments are read, its arguments go out in pieces whose joined text is the held
call's json.dumps, a body of another shape is held to its end as before, and a call the output ends inside is closed
as it came (never the marker as text)."""
from __future__ import annotations

import json
import random
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import INK_CALL, ChatTemplate, InkArgStream, InklingCalls, OutputParser  # noqa: E402
import serve.server as SV  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MM, TH, TX, EM, INV, INVT, ERR = range(1000, 1007)
SPECIAL = {MM: "<|message_model|>", TH: "<|content_thinking|>", TX: "<|content_text|>", EM: "<|end_message|>",
           INV: INK_CALL, INVT: "<|content_invoke_tool_text|>", ERR: "<|content_tool_error|>"}
TOOLS = [{"name": "write_file", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "content": {"type": "string"}, "opts": {"type": "object"}}}},
         {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]


class ChunkTok:
    """A tokenizer whose ids are text chunks of the test's choosing (cut anywhere), or byte chunks (token_bytes: a
    character split across ids, as a real tokenizer's byte tokens do)."""

    def __init__(self, as_bytes: bool = False):
        self.pieces: list = []
        self.as_bytes = as_bytes
        if as_bytes:
            self.token_bytes = lambda t: self.pieces[t]

    def ids(self, chunks) -> list[int]:
        out = []
        for c in chunks:
            if isinstance(c, int):
                out.append(c)
                continue
            self.pieces.append(c.encode() if self.as_bytes and isinstance(c, str) else c)
            out.append(len(self.pieces) - 1)
        return out

    def decode(self, ids) -> str:
        raw = b"" if self.as_bytes else ""
        for t in ids:
            if t in SPECIAL:
                raw += SPECIAL[t].encode() if self.as_bytes else SPECIAL[t]
            else:
                raw += self.pieces[t]
        return raw.decode("utf-8", errors="replace") if self.as_bytes else raw


class InkTok(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + list(SPECIAL.values()) + ["<|content_model_end_sampling|>"]


def split(text: str, cuts) -> list[str]:
    """text cut at the given character positions (a list), or into pieces of 1..cuts characters at random (an int
    seed after the size: (size, seed))."""
    if isinstance(cuts, tuple):
        size, seed = cuts
        rnd, out, i = random.Random(seed), [], 0
        while i < len(text):
            j = i + rnd.randrange(1, size + 1)
            out.append(text[i:j])
            i = j
        return out
    cuts = [0] + sorted(cuts) + [len(text)]
    return [text[a:b] for a, b in zip(cuts, cuts[1:]) if b > a]


def run(chunks, tools=TOOLS, as_bytes=False, no_calls=False, end=True):
    """The events InklingCalls makes of the chunks (ints are the special ids); end: the output ends there."""
    tok = ChunkTok(as_bytes)
    parser = OutputParser(thinking=False, tools=tools, stream_tools=True, xml_calls=True, no_calls=no_calls)
    ink = InklingCalls(tok, MM, TH, TX, EM, INV, parser, INVT, ERR)
    ids = tok.ids(chunks)
    evs = []
    for t in ids:
        got = ink.push(t)
        if got is None:
            evs += parser.feed(tok.decode([t]))
        else:
            evs += got
    if end:
        evs += ink.finish() + parser.finish()
    return evs


def summary(evs):
    """(the calls as (name, arguments), the streamed arguments text per call id, content text, kinds in order)."""
    calls = [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]
    streamed = {}
    for e in evs:
        if e.kind == "tool_start":
            streamed[e.call.id] = ""
        elif e.kind == "tool_args":
            streamed[e.call.id] += e.text
    content = "".join(e.text for e in evs if e.kind == "content")
    return calls, list(streamed.values()), content, [e.kind for e in evs]


def call_msg(body: str, name="write_file", cuts=(3, 0)) -> list:
    return [MM] + split(name, [len(name) // 2]) + [INV] + split(body, cuts) + [EM]


class Streaming(unittest.TestCase):
    ARGS = {"path": "src/a \"b\".py", "content": 'print("hi\\\\n")\n\tx = "\\u00e9 é 日本 😀"\n' * 3,
            "opts": {"mode": [1, 2.5, -3e-7, True, None, {"deep": ["x", {}]}], "empty": {}, "list": []}}

    def body(self, **kw):
        return json.dumps({"name": "write_file", "args": self.ARGS}, **kw)

    def check(self, chunks, args=None, **kw):
        args = self.ARGS if args is None else args
        calls, streamed, content, kinds = summary(run(chunks, **kw))
        self.assertEqual(calls, [("write_file", args)])
        self.assertEqual(streamed, [json.dumps(args, ensure_ascii=False)])   # (the held call's own text, joined)
        self.assertEqual(content, "")
        self.assertEqual(kinds[0], "tool_start")
        self.assertGreater(kinds.count("tool_args"), 3)
        return kinds

    def test_awkward_splits(self):
        for kw in ({}, {"ensure_ascii": False}, {"separators": (",", ":")}, {"indent": 2}):
            body = self.body(**kw)
            for size in (1, 2, 3, 7):
                for seed in range(3):
                    with self.subTest(kw=kw, size=size, seed=seed):
                        self.check(call_msg(body, cuts=(size, seed)))

    def test_cut_inside_escapes(self):
        body = self.body()                           # (ASCII: \" \\ \n é and a surrogate pair 😀)
        for marker in ('\\"', "\\\\", "\\n", "\\u00e9", "\\ud83d\\ude00"):
            k = body.index(marker)
            for off in range(1, len(marker)):
                with self.subTest(marker=marker, off=off):
                    self.check(call_msg(body, cuts=[k + off]))

    def test_utf8_split_across_ids(self):
        body = self.body(ensure_ascii=False).encode()
        k = body.index("日".encode())
        chunks = [MM, "write_file", INV, body[:k + 1], body[k + 1:k + 2], body[k + 2:], EM]
        self.check(chunks, as_bytes=True)

    def test_name_split_and_announced_early(self):
        body = self.body()
        evs = run([MM, "wri", "te_", "file", INV] + split(body, (4, 1)), end=False)   # (no end yet)
        self.assertEqual(evs[0].kind, "tool_start")
        self.assertEqual(evs[0].call.name, "write_file")
        self.assertIn("tool_args", [e.kind for e in evs])
        self.assertNotIn("tool_call", [e.kind for e in evs])

    def test_two_calls(self):
        a = json.dumps({"name": "get_weather", "args": {"city": "Tokyo"}})
        b = json.dumps({"name": "get_weather", "args": {"city": "Osaka"}})
        evs = run(call_msg(a, "get_weather", (2, 0)) + call_msg(b, "get_weather", (2, 1)))
        calls, streamed, content, kinds = summary(evs)
        self.assertEqual(calls, [("get_weather", {"city": "Tokyo"}), ("get_weather", {"city": "Osaka"})])
        self.assertEqual(streamed, ['{"city": "Tokyo"}', '{"city": "Osaka"}'])
        ids = [e.call.id for e in evs if e.kind in ("tool_start", "tool_call")]
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[2], ids[3])
        self.assertNotEqual(ids[0], ids[2])

    def test_other_argument_keys_and_empty_args(self):
        for key in ("arguments", "parameters"):
            with self.subTest(key=key):
                self.check(call_msg(self.body().replace('"args"', '"' + key + '"')))
        calls, streamed, _, _ = summary(run(call_msg('{"name": "get_weather", "args": {}}', "get_weather")))
        self.assertEqual((calls, streamed), ([("get_weather", {})], ["{}"]))

    def test_cut_by_max_tokens_after_args_started(self):
        body = self.body()
        for frac in (0.3, 0.5, 0.9):
            k = int(len(body) * frac)
            with self.subTest(frac=frac):
                evs = run([MM, "write_file", INV] + split(body[:k], (5, 2)))     # (the output ends: no end mark)
                calls, streamed, content, kinds = summary(evs)
                self.assertEqual(content, "")
                self.assertEqual(len(calls), 1)
                self.assertEqual(json.loads(streamed[0]), calls[0][1])           # (closed: parses, same call)
                self.assertTrue(json.dumps(self.ARGS, ensure_ascii=False).startswith(streamed[0][:-8]))

    def test_cut_before_args_drops_the_marker(self):
        for head in ('', '{"na', '{"name": "write_file"', '{"name": "write_file", "args"'):
            with self.subTest(head=head):
                calls, streamed, content, _ = summary(run([MM, "write_file", INV] + split(head, (3, 0))))
                self.assertEqual((calls, streamed), ([], []))
                self.assertNotIn(INK_CALL, content)
                self.assertEqual(content, "")

    def test_complete_body_without_end_mark_is_a_call(self):
        calls, streamed, _, _ = summary(run([MM, "write_file", INV] + split(self.body(), (6, 0))))
        self.assertEqual(calls, [("write_file", self.ARGS)])

    def test_other_shapes_are_held(self):
        """Bodies the reader does not expect are never announced: the whole body decides at its end, as before."""
        held = [('{"args": {"city": "Kyoto"}, "name": "get_weather"}', [("get_weather", {"city": "Kyoto"})], ""),
                ('{"name": "get_weather", "args": "{\\"city\\": \\"Nara\\"}"}',
                 [("get_weather", {"city": "Nara"})], ""),
                ('<invoke name="get_weather">\n<parameter name="city">Kobe</parameter>\n</invoke>',
                 [("get_weather", {"city": "Kobe"})], ""),
                ('{"name":', [], 'get_weather' + INK_CALL + '{"name":'),
                ('{"name": "rm_rf", "args": {"x": 1}}', [],
                 'get_weather' + INK_CALL + '{"name": "rm_rf", "args": {"x": 1}}')]
        for body, want_calls, want_content in held:
            with self.subTest(body=body):
                evs = run(call_msg(body, "get_weather", (2, 0)))
                calls, streamed, content, kinds = summary(evs)
                self.assertEqual(streamed, [])
                self.assertNotIn("tool_args", kinds)
                self.assertEqual(calls, want_calls)
                self.assertEqual(content, want_content)

    def test_tool_choice_none_is_not_streamed(self):
        evs = run(call_msg(self.body(), cuts=(3, 0)), no_calls=True)
        self.assertEqual([e.kind for e in evs], ["calls_refused"])

    def test_malformed_after_the_start_closes_the_call(self):
        """A body that goes wrong after the call went out is closed where it went wrong (the pieces sent are never
        taken back) - the client gets valid JSON and the same call as the server reports."""
        body = '{"name": "write_file", "args": {"path": "a.py", "content": "x" oops, "more": 1}}'
        calls, streamed, content, _ = summary(run(call_msg(body, cuts=(3, 0))))
        self.assertEqual(calls, [("write_file", {"path": "a.py", "content": "x"})])
        self.assertEqual(streamed, [json.dumps(calls[0][1])])
        self.assertEqual(content, "")

    def test_reader_fuzz(self):
        """Random arguments, any spacing, cut anywhere: joined pieces == json.dumps(args), and a cut body closes to
        valid JSON."""
        rnd = random.Random(7)

        def value(d=0):
            k = rnd.randrange(7 if d < 3 else 4)
            if k == 0:
                return rnd.choice([0, -3, 12345678901234567890, 1.5, -2.5e-7, 1e300, True, False, None])
            if k in (1, 2, 3):
                return "".join(rnd.choice('a"\\\n\té日😀/\x01  ') for _ in range(rnd.randrange(10)))
            if k in (4, 5):
                return [value(d + 1) for _ in range(rnd.randrange(4))]
            return {"k%d%s" % (i, rnd.choice(["", '"', "ü"])): value(d + 1) for i in range(rnd.randrange(4))}
        for n in range(400):
            args = {"k%d" % i: value() for i in range(rnd.randrange(5))}
            text = json.dumps({"name": "w", "args": args}, ensure_ascii=rnd.random() < 0.5,
                              separators=rnd.choice([None, (",", ":"), (" , ", " : ")]))
            st, i, sent = InkArgStream(lambda nm: True), 0, []
            while i < len(text):
                j = i + rnd.randrange(1, 9)
                sent += st.feed(text[i:j])
                i = j
            self.assertEqual("".join(sent), json.dumps(args, ensure_ascii=False), text)
            cut = InkArgStream(lambda nm: True)
            cut.feed(text[:rnd.randrange(len(text))])
            cut.close()
            if cut.started:
                json.loads(cut.text())
            k = rnd.randrange(len(text))                 # (one character spoilt: whatever went out still closes)
            bad = InkArgStream(lambda nm: True)
            bad.feed(text[:k] + rnd.choice('x}],:"\\\x02') + text[k + 1:])
            bad.close()
            if bad.started:
                json.loads(bad.text())


class ServerShapes(unittest.TestCase):
    """Through the server's stream forms: OpenAI tool_calls deltas and Anthropic input_json_delta pieces, the
    non-streamed answers (collected from the same pieces) byte-identical to the held call's, and a cut call."""

    ARGS = {"path": "notes/日本.md", "content": "line \"1\"\n\\path\\ é\n" * 20}
    TOOLS = [{"name": "write_file", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}}}}]

    def script(self):
        return ("<|message_model|><|content_text|>Writing.<|end_message|><|message_model|>write_file" + INK_CALL +
                json.dumps({"name": "write_file", "args": self.ARGS}) + "<|end_message|><|content_model_end_sampling|>")

    def svc(self):
        tok = InkTok()
        return tok, Service(MockEngine(tok, self.script(), max_context=8192), tok,
                            ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def openai(self, max_new=4000):
        tok, svc = self.svc()
        ids = tok.encode("hi")
        chunks = list(SV.openai_chunks(svc, {"stream": True}, ids, False, self.TOOLS, max_new, threading.Event()))
        return [c for c in chunks if c is not None]

    def anthropic(self, max_new=4000):
        tok, svc = self.svc()
        ids = tok.encode("hi")
        return [e for e in SV.anthropic_events(svc, {}, ids, False, self.TOOLS, max_new, threading.Event())
                if e is not None]

    def test_openai_streamed_and_collected(self):
        chunks = self.openai()
        pieces = [tc for c in chunks for tc in c["choices"][0]["delta"].get("tool_calls") or []]
        self.assertGreater(len(pieces), 20)                              # (streamed, not one held chunk)
        self.assertEqual(pieces[0]["function"], {"name": "write_file", "arguments": ""})
        content, reasoning, by_index = [], [], {}
        last = SV._openai_collect(iter(chunks), content, reasoning, by_index, [])
        self.assertEqual(list(by_index), [0])
        self.assertEqual(by_index[0]["function"]["arguments"], json.dumps(self.ARGS, ensure_ascii=False))
        self.assertEqual(by_index[0]["id"], pieces[0]["id"])
        self.assertEqual("".join(content), "Writing.")
        self.assertEqual(last["choices"][0]["finish_reason"], "tool_calls")

    def test_anthropic_streamed_and_collected(self):
        events = self.anthropic()
        deltas = [e["delta"]["partial_json"] for n, e in events
                  if n == "content_block_delta" and e["delta"]["type"] == "input_json_delta"]
        self.assertGreater(len(deltas), 20)
        self.assertEqual("".join(deltas), json.dumps(self.ARGS, ensure_ascii=False))
        msg = SV.anthropic_collect(iter(events))
        self.assertEqual([b["type"] for b in msg["content"]], ["text", "tool_use"])
        self.assertEqual(msg["content"][1]["input"], self.ARGS)
        self.assertEqual(msg["content"][1]["name"], "write_file")
        self.assertEqual(msg["stop_reason"], "tool_use")

    def test_cut_by_max_tokens(self):
        head = len(InkTok().encode(self.script().split(INK_CALL)[0], parse_special=True)) + 1
        for max_new in (head + 60, head + 300):
            with self.subTest(max_new=max_new):
                chunks = self.openai(max_new)
                content, reasoning, by_index = [], [], {}
                last = SV._openai_collect(iter(chunks), content, reasoning, by_index, [])
                self.assertEqual("".join(content), "Writing.")
                self.assertEqual(last["choices"][0]["finish_reason"], "length")
                args = json.loads(by_index[0]["function"]["arguments"])
                self.assertEqual(set(args) - set(self.ARGS), set())
                msg = SV.anthropic_collect(iter(self.anthropic(max_new)))
                self.assertEqual(msg["stop_reason"], "max_tokens")
                self.assertEqual(msg["content"][1]["input"], args)
        # cut before the arguments start: no call, and no marker in the text
        chunks = self.openai(head + 5)
        content, reasoning, by_index = [], [], {}
        last = SV._openai_collect(iter(chunks), content, reasoning, by_index, [])
        self.assertEqual(("".join(content), by_index), ("Writing.", {}))
        self.assertEqual(last["choices"][0]["finish_reason"], "length")


if __name__ == "__main__":
    unittest.main()
