"""The conversation slots' keys (include/strata/program/conv_slots.hpp): the server tells the engine which namespace
(the API key) and which session a request's conversation belongs to - GEN keys slot_ns= / slot_sess= - so one key's
conversations set aside in the engine are never taken back by another key's requests.

    python -m unittest serve.test_slots
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import IM_END, ByteTokenizer, Service, StrataEngine, serve, slot_hint  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Hint(unittest.TestCase):
    def test_no_keys_one_namespace(self):
        ns, sess = slot_hint({"X-Claude-Code-Session-Id": "abc"}, {}, "")
        self.assertEqual(ns, "")
        self.assertRegex(sess, r"^[0-9a-f]{16}$")
        self.assertEqual(slot_hint({"X-Claude-Code-Session-Id": "abc"}, {}, "")[1], sess)
        self.assertNotEqual(slot_hint({"X-Claude-Code-Session-Id": "abd"}, {}, "")[1], sess)

    def test_key_namespaces(self):
        a = slot_hint({"Authorization": "Bearer key-one"}, {}, "key-one,key-two")[0]
        b = slot_hint({"x-api-key": "key-two"}, {}, "key-one,key-two")[0]
        self.assertRegex(a, r"^[0-9a-f]{16}$")
        self.assertNotEqual(a, b)
        self.assertEqual(slot_hint({"x-api-key": "key-one"}, {}, ["key-one", "key-two"])[0], a)
        self.assertNotIn("key-one", a)

    def test_session_from_metadata_and_user(self):
        md = {"metadata": {"user_id": json.dumps({"device_id": "d", "session_id": "s-1"})}}
        s1 = slot_hint({}, md, "")[1]
        self.assertEqual(s1, slot_hint({"X-Claude-Code-Session-Id": "s-1"}, {}, "")[1])
        self.assertRegex(slot_hint({}, {"user": "u7"}, "")[1], r"^[0-9a-f]{16}$")
        self.assertEqual(slot_hint({}, {}, ""), ("", ""))


class Keys(unittest.TestCase):
    def test_keys_on_the_gen_line(self):
        k = StrataEngine.sampling_keys({"temperature": 0, "_slot_ns": "00ab", "_slot_sess": "ff01"}).split()
        self.assertIn("slot_ns=00ab", k)
        self.assertIn("slot_sess=ff01", k)
        k = StrataEngine.sampling_keys({"temperature": 0.7, "_slot_ns": "00ab"}).split()
        self.assertIn("slot_ns=00ab", k)

    def test_absent_or_bad_values_send_nothing(self):
        self.assertEqual(StrataEngine.sampling_keys({"temperature": 0}), " temperature=0")
        k = StrataEngine.sampling_keys({"temperature": 0, "_slot_ns": "", "_slot_sess": "x y=1"})
        self.assertNotIn("slot_", k)


class Recorder:
    """An engine that answers "ok." and records each run's sampling."""

    def __init__(self, tok):
        self.tok, self.max_context, self.runs = tok, 16384, []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.runs.append(dict(sampling or {}))
        yield from self.tok.encode("ok.") + self.tok.encode(IM_END, parse_special=True)


class OverHttp(unittest.TestCase):
    def test_requests_carry_their_namespace_and_session(self):
        tok = ByteTokenizer()
        eng = Recorder(tok)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.api_key = "key-one,key-two"
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            def post(path, body, headers):
                r = urllib.request.Request(base + path, json.dumps(body).encode(),
                                           {"Content-Type": "application/json", **headers})
                with urllib.request.urlopen(r, timeout=30) as f:
                    return json.loads(f.read())

            msg = {"model": "m", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}],
                   "thinking": {"type": "disabled"}, "reasoning_effort": "none"}
            post("/v1/messages", msg, {"x-api-key": "key-one", "X-Claude-Code-Session-Id": "main"})
            post("/v1/chat/completions", msg, {"Authorization": "Bearer key-two"})
        finally:
            httpd.shutdown()
            httpd.server_close()
        a, b = eng.runs[0], eng.runs[-1]   # (each request's runs carry its keys: the first's, the last's)
        self.assertEqual(a["_slot_ns"], slot_hint({"x-api-key": "key-one"}, {}, svc.api_key)[0])
        self.assertEqual(a["_slot_sess"], slot_hint({"X-Claude-Code-Session-Id": "main"}, {}, "")[1])
        self.assertEqual(b["_slot_ns"], slot_hint({"x-api-key": "key-two"}, {}, svc.api_key)[0])
        self.assertNotEqual(a["_slot_ns"], b["_slot_ns"])
        self.assertEqual(b["_slot_sess"], "")


if __name__ == "__main__":
    unittest.main()
