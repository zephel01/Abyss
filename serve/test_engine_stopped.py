"""Answers the engine stopped itself (its ERR, or the process died after the answer began): HTTP 422, type
engine_error, with the partial text - not a 5xx, which the OpenAI and Anthropic SDKs send again by themselves (the
same long answer ran a second time).  503 stays for an engine that died before writing anything."""
import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, EngineDied, EngineError, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096


class StoppingEngine(MockEngine):
    """Writes `after` tokens of its answer, then fails the way `mode` says ("err": the engine's ERR line, "died": the
    process ended)."""
    mode, after = "err", 30

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
            if i == self.after:
                if self.mode == "died":
                    raise EngineDied("the engine stopped unexpectedly (exit code -9)")
                raise EngineError("glm token completion: the sample failed")
            yield t


class EngineStoppedAnswers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = StoppingEngine(tok, "thought</think>\n\nThe answer is forty-two.", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.engine.mode, self.engine.after = "err", 30

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read().decode()

    OA = ("/v1/chat/completions", {"model": "m", "max_tokens": 200, "messages": [{"role": "user", "content": "q"}]})
    AN = ("/v1/messages", {"model": "m", "max_tokens": 200, "messages": [{"role": "user", "content": "q"}],
                           "thinking": {"type": "enabled", "budget_tokens": 1024}})

    def test_openai_err_is_422_with_partial(self):
        code, text = self.post(*self.OA)
        self.assertEqual(code, 422, text)
        err = json.loads(text)["error"]
        self.assertEqual((err["type"], err["code"]), ("engine_error", "engine_error"))
        self.assertIn("sample failed", err["message"])
        self.assertEqual(err["partial_reasoning"], "thought")       # what the 30 tokens wrote (one a byte)
        self.assertTrue(err["partial_text"].startswith("The answer"), err["partial_text"])
        self.assertEqual(self.svc.metrics()["requests"][0]["finish"], "error")

    def test_anthropic_err_is_422_anthropic_shaped(self):
        code, text = self.post(*self.AN)
        self.assertEqual(code, 422, text)
        b = json.loads(text)
        self.assertEqual(b["type"], "error")
        self.assertEqual(b["error"]["type"], "engine_error")
        self.assertEqual(b["error"]["partial_reasoning"], "thought")
        self.assertTrue(b["error"]["partial_text"].startswith("The answer"), b["error"]["partial_text"])

    def test_died_mid_answer_is_422_before_any_token_503(self):
        self.engine.mode = "died"
        code, text = self.post(*self.OA)
        self.assertEqual(code, 422, text)
        self.assertIn("stopped unexpectedly", json.loads(text)["error"]["message"])
        self.engine.after = 0                              # nothing written: the engine is simply not running
        code, text = self.post(*self.OA)
        self.assertEqual(code, 503, text)

    def test_err_before_any_token_is_422_too(self):
        self.engine.after = 0
        for path, body in (self.OA, self.AN):
            code, text = self.post(path, body)
            self.assertEqual(code, 422, (path, text))

    def test_streaming_ends_with_engine_error(self):
        for path, body in (self.OA, self.AN):
            code, text = self.post(path, {**body, "stream": True})
            self.assertEqual(code, 200, path)
            self.assertIn('"engine_error"', text, path)
            self.assertIn("sample failed", text, path)
        self.assertTrue(text.count("event: error") == 1)


if __name__ == "__main__":
    unittest.main()
