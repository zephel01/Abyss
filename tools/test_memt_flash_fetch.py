"""tools/memt_flash_fetch.py against a local range-serving HTTP server on made-up shards (no network): the memory
files whole, only the non-expert bytes of the model shards (adjacent tensors in one read), a second run fetches
nothing, a server that ignores Range is refused."""
import json
import os
import struct
import sys
import tempfile
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memt_flash_fetch as F  # noqa: E402


def shard(tensors):
    """[(name, nbytes)] -> safetensors bytes (data = the tensor's index repeated)"""
    h, off, data = {}, 0, b""
    for i, (n, nb) in enumerate(tensors):
        h[n] = {"dtype": "U8", "shape": [nb], "data_offsets": [off, off + nb]}
        data += bytes([i + 1]) * nb
        off += nb
    hj = json.dumps(h).encode()
    return struct.pack("<Q", len(hj)) + hj + data, 8 + len(hj)


class Handler(BaseHTTPRequestHandler):
    files: dict = {}
    ranges = True
    log: list = []

    def do_GET(self):
        name = self.path.split("/resolve/main/", 1)[-1]
        if name not in self.files:
            self.send_error(404)
            return
        body = self.files[name]
        rg = self.headers.get("Range")
        if rg and self.ranges:
            a, b = (int(x) for x in rg.split("=")[1].split("-"))
            part = body[a:b + 1]
            self.send_response(206)
        else:
            part = body
            self.send_response(200)
        self.log.append((name, rg))
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        self.wfile.write(part)

    def log_message(self, *a):
        pass


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.s1, self.b1 = shard([("model.embed.weight", 100), ("model.layers.0.attn.w", 50),
                                  ("model.layers.0.mlp.experts.0.w1", 400), ("model.layers.0.mlp.gate.weight", 30)])
        self.s2, self.b2 = shard([("model.layers.1.mlp.experts.1.w1", 300), ("model.norm.weight", 20)])
        self.memt = b"MEMT" * 64
        wm = {"model.embed.weight": "model-00001-of-00002.safetensors", "model.norm.weight": "model-00002-of-00002.safetensors",
              "model.layers.4.memt.q": "memt-00001-of-00001.safetensors"}
        Handler.files = {"config.json": b"{}", "model.safetensors.index.json": json.dumps({"weight_map": wm}).encode(),
                         "model-00001-of-00002.safetensors": self.s1, "model-00002-of-00002.safetensors": self.s2,
                         "memt-00001-of-00001.safetensors": self.memt}
        Handler.ranges = True
        Handler.log = []
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.env = dict(os.environ)
        os.environ["HF_ENDPOINT"] = f"http://127.0.0.1:{self.srv.server_address[1]}"
        os.environ["HF_TOKEN"] = "x"
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        os.environ.clear()
        os.environ.update(self.env)
        self.tmp.cleanup()

    def test_fetch(self):
        out = Path(self.tmp.name) / "m"
        self.assertEqual(F.main([str(out)]), 0)
        self.assertEqual((out / "memt-00001-of-00001.safetensors").read_bytes(), self.memt)
        b1 = self.b1
        # shard 1: embed + attn (adjacent) in one read, the gate after the experts in another; shard 2: the norm
        want = {f"00001_{b1}.bin": self.s1[b1:b1 + 150], f"00001_{b1 + 550}.bin": self.s1[b1 + 550:b1 + 580],
                f"00002_{self.b2 + 300}.bin": self.s2[self.b2 + 300:]}
        got = {f.name: f.read_bytes() for f in (out / "runs").glob("*.bin")}
        self.assertEqual(got, want)
        self.assertEqual(struct.unpack("<Q", (out / "hdr" / "memt_00001.len").read_bytes())[0], b1 - 8)
        n = len(Handler.log)
        self.assertEqual(F.main([str(out)]), 0)   # (resumed: everything is there)
        self.assertEqual(len(Handler.log), n)

    def test_no_range_refused(self):
        Handler.ranges = False
        with mock.patch.object(F.time, "sleep", lambda s: None), self.assertRaises(SystemExit):
            F.main([str(Path(self.tmp.name) / "m")])


if __name__ == "__main__":
    unittest.main()
