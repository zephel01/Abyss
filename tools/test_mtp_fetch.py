"""tools/mtp_fetch.py's range checks: a server that ignores Range (200 + the whole file) or answers another range
must stop the fetch, a proper 206 answer passes, and a tensor whose sha256 differs from the known one is refused."""
import http.server
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtp_fetch  # noqa: E402

BODY = bytes(range(256)) * 16   # 4096 bytes


class _Handler(http.server.BaseHTTPRequestHandler):
    mode = "ok"

    def log_message(self, *a):
        pass

    def do_GET(self):
        rng = self.headers.get("Range", "")
        a, b = (int(x) for x in rng.split("=")[1].split("-"))
        if self.mode == "full":          # a proxy that drops Range: 200 and the whole file
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY)
            return
        if self.mode == "shifted":       # 206, but another range than asked
            a, b = a + 1, b + 1
        self.send_response(206)
        self.send_header("Content-Range", "bytes %d-%d/%d" % (a, b, len(BODY)))
        self.send_header("Content-Length", str(b - a + 1))
        self.end_headers()
        self.wfile.write(BODY[a:b + 1])


class RangeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = "http://127.0.0.1:%d/shard.safetensors" % cls.srv.server_port

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def get(self, mode, a, b):
        _Handler.mode = mode
        return mtp_fetch.get(self.url, a, b, retries=1)

    def test_206_with_the_asked_range_passes(self):
        self.assertEqual(self.get("ok", 100, 199), BODY[100:200])

    def test_200_full_body_is_refused(self):
        with self.assertRaises(mtp_fetch.RangeError):
            self.get("full", 100, 199)

    def test_wrong_content_range_is_refused(self):
        with self.assertRaises(mtp_fetch.RangeError):
            self.get("shifted", 100, 199)

    def test_check_range(self):
        mtp_fetch.check_range(206, "bytes 5-9/100", 5, 9)
        with self.assertRaises(mtp_fetch.RangeError):
            mtp_fetch.check_range(206, None, 5, 9)
        with self.assertRaises(mtp_fetch.RangeError):
            mtp_fetch.check_range(200, "bytes 5-9/100", 5, 9)

    def test_known_sha256_mismatch_is_refused(self):
        _Handler.mode = "ok"
        with tempfile.TemporaryDirectory() as d:
            old = mtp_fetch.REPO
            mtp_fetch.REPO = self.url.rsplit("/", 1)[0] + "/"
            try:
                inv = {"tensors": [{"name": "mtp.x", "shard": "shard.safetensors", "dtype": "BF16", "shape": [8],
                                    "start": 16, "end": 31, "bytes": 16}]}
                Path(d, "mtp-inventory.json").write_text(json.dumps(inv))
                Path(d, "want.json").write_text(json.dumps([{"name": "mtp.x", "sha256": "0" * 64}]))
                with self.assertRaises(SystemExit):
                    mtp_fetch.fetch(d, None, os.path.join(d, "want.json"))
                self.assertFalse(Path(d, "tensors", "mtp.x.bin").exists())
                mtp_fetch.fetch(d, None)   # (no known hash: fetched and recorded)
                self.assertEqual(Path(d, "tensors", "mtp.x.bin").read_bytes(), BODY[16:32])
            finally:
                mtp_fetch.REPO = old


if __name__ == "__main__":
    unittest.main()
