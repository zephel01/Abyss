"""serve/test_memt.py - MemT memory requests against the mock engine (no GPU, no pack): the session rules - the BOS
dropped from the prompt, the written turn and a document once the memory holds something, kept while it is empty.

    python -m unittest serve.test_memt -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BOS = 2   # (ByteTokenizer has no specials: one is named the BOS here)


class MemtEngine(MockEngine):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.lines, self.prompts = [], []

    def command(self, line, expect, timeout=60.0):
        self.lines.append(line)
        if line.startswith("MEMT SAVE "):
            Path(line[len("MEMT SAVE "):]).write_bytes(b"x")
        return "MEMT 1 0 0.0"

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class MemtBos(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        os.environ["STRATA_MEMT_DIR"] = self.dir.name
        tok = ByteTokenizer()
        tok.special_ids = {"tokenizer.ggml.bos_token_id": BOS}
        self.engine = MemtEngine(tok, "</think>\n\nok", max_context=4096)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def tearDown(self):
        os.environ.pop("STRATA_MEMT_DIR", None)
        self.dir.cleanup()

    def turn(self, ids, commit=True):
        msgs = [{"role": "user", "content": "hello"}]
        m = self.svc.memt_request({"memory": {"id": "cat", "commit": commit}}, msgs, {})
        for _ in self.svc.run(list(ids), False, None, 8, {"_memt": m}, threading.Event()):
            pass
        return m

    def test_bos_dropped_once_the_memory_holds_something(self):
        m = self.turn([BOS, 65, 66])
        self.assertFalse(m["present"])
        self.assertEqual(self.engine.prompts[-1][0], BOS)          # an empty memory: the BOS stays
        self.assertTrue(any(x.startswith("MEMT COMMIT ") for x in self.engine.lines))
        m = self.turn([BOS, 65, 66])
        self.assertTrue(m["present"])
        self.assertEqual(self.engine.prompts[-1], [65, 66])        # the memory holds the first turn: no BOS
        self.assertNotEqual(m["replay"][:1], [BOS])

    def test_document_without_bos_on_a_written_memory(self):
        self.svc.template.bos_token = chr(BOS)
        (Path(self.dir.name) / "cat.safetensors").write_bytes(b"x")
        self.svc.memt_document("cat", "some text")
        doc = [x for x in self.engine.lines if x.startswith("MEMT DOC ")][-1]
        self.assertNotEqual(doc[len("MEMT DOC "):].split(",")[0], str(BOS))


if __name__ == "__main__":
    unittest.main()
