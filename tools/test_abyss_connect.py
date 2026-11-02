"""tools/abyss_connect.py against a small fake server (no engine): finding the server, the printed settings, --write
for both tools (merge, backup, nothing written on no), --check in both dialects."""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abyss_connect as C  # noqa: E402


class _Fake(BaseHTTPRequestHandler):
    key = "k1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _ok(self):
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {self.key}" or self.headers.get("x-api-key") == self.key

    def do_GET(self):
        if not self._ok():
            return self._send(401, {"error": {"message": "missing or wrong API key"}})
        if self.path == "/v1/models":
            return self._send(200, {"data": [{"id": "m1", "meta": {"n_ctx": 65536}}]})
        self._send(404, {})

    def do_POST(self):
        if not self._ok():
            return self._send(401, {})
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tool = bool(req.get("tools"))
        if self.path.startswith("/v1/messages"):
            content = ([{"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Tokyo"}}] if tool
                       else [{"type": "text", "text": "OK"}])
            return self._send(200, {"content": content})
        msg = ({"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "get_weather", "arguments": json.dumps({"city": "Tokyo"})}}]} if tool else {"content": "OK"})
        self._send(200, {"choices": [{"message": msg}]})


class ConnectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), _Fake)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.cfg = self.d / "abyss-m1.json"
        port = self.srv.server_address[1]
        self.cfg.write_text(json.dumps({"exe": "x", "args": ["--glm-pack", "p", "--max-context", "4096"],
                                        "port": port, "api_key": "k1", "model_name": "m1"}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_find_by_config_and_url(self):
        s = C.find_server(None, None, self.cfg)
        self.assertEqual((s["url"], s["model"], s["ctx"], s["key"]), (self.url, "m1", 65536, "k1"))
        with mock.patch.dict(os.environ, {"K": "k1"}):
            s = C.find_server(self.url, "K", None)
        self.assertEqual(s["model"], "m1")
        s = C.find_server(self.url, None, None)
        self.assertIn("401", s["error"])

    def test_claude_settings_never_hold_the_key(self):
        s = C.find_server(None, None, self.cfg)
        text = "\n".join(C.claude_print(s))
        self.assertIn(f"ANTHROPIC_BASE_URL='{self.url}'", text)
        self.assertIn("ANTHROPIC_MODEL='m1'", text)
        self.assertNotIn("k1", text)
        with mock.patch.object(C, "ROOT", self.d):
            path, body = C.claude_launcher(s)
            self.assertNotIn("k1", body)
            self.assertEqual(C.write_claude(s, yes=True), 0)
            self.assertTrue(path.exists())
            if not C.WIN:
                self.assertTrue(os.access(path, os.X_OK))

    def test_opencode_merge_keeps_the_rest(self):
        s = C.find_server(None, None, self.cfg)
        oc = self.d / "opencode.json"
        oc.write_text(json.dumps({"model": "other/x", "provider": {"other": {"name": "o"}}}))
        self.assertEqual(C.write_opencode(s, yes=True, path=oc), 0)
        new = json.loads(oc.read_text())
        self.assertEqual(new["model"], "other/x")                 # the user's default stays
        self.assertEqual(new["provider"]["other"], {"name": "o"})
        ab = new["provider"]["abyss"]
        self.assertEqual(ab["options"]["baseURL"], self.url + "/v1")
        self.assertEqual(ab["options"]["apiKey"], "{env:ABYSS_API_KEY}")
        self.assertEqual(ab["models"]["m1"]["limit"]["context"], 65536)
        self.assertEqual(len(list(self.d.glob("opencode.json.bak-*"))), 1)

    def test_opencode_not_plain_json_is_left_alone(self):
        s = C.find_server(None, None, self.cfg)
        oc = self.d / "opencode.json"
        oc.write_text('{ // a comment\n "model": "x" }')
        self.assertEqual(C.write_opencode(s, yes=True, path=oc), 1)
        self.assertIn("// a comment", oc.read_text())

    def test_no_answer_writes_nothing(self):
        s = C.find_server(None, None, self.cfg)
        oc = self.d / "opencode.json"
        with mock.patch("sys.stdin.isatty", return_value=False):
            self.assertEqual(C.write_opencode(s, yes=False, path=oc), 1)
        self.assertFalse(oc.exists())

    def test_check_both_dialects(self):
        s = C.find_server(None, None, self.cfg)
        self.assertEqual(C.check("claude", s, timeout=10), 0)
        self.assertEqual(C.check("opencode", s, timeout=10), 0)


if __name__ == "__main__":
    unittest.main()
