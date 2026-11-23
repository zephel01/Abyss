"""tools/test_tokenizer_inkling.py - Inkling 2's tokenizer sets ignore_merges.

Inkling 2's tokenizer.json sets `ignore_merges`: a pre-token piece that is itself a vocabulary entry is that one
token, even when the merges cannot build it.  The o200k vocabulary has 337 such entries (chffff, Aleksandr, VBORw,
kwargs, について, ファイル, ...); without the flag each came out as 2-4 tokens the model was never trained on.

The tiny-vocab tests run everywhere.  The model test runs when an Inkling 2 GGUF is on disk (STRATA_INKLING_GGUF, or
the usual Abyss-data path): it derives every entry the merges cannot build from the GGUF's own vocab and merges, and
checks each one encodes to its own id - the id ignore_merges emits by definition.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from strata_tokenizer import BYTE_TO_UNICODE, Tokenizer, split_sequential  # noqa: E402


def tiny(pre):
    # the byte alphabet, two merges, and two entries the merges cannot build
    tokens = list(BYTE_TO_UNICODE.values()) + ["ab", "abc", "Xyz", "zz"]
    return Tokenizer(tokens, ["a b", "ab c"], pre=pre)


class TinyVocab(unittest.TestCase):
    def test_flag_set_for_inkling2(self):
        self.assertTrue(tiny("inkling2").ignore_merges)
        for pre in ("qwen35", "joyai-llm", "laguna", "hy_v4"):
            self.assertFalse(tiny(pre).ignore_merges, pre)

    def test_unbuildable_entry_is_one_token(self):
        tk = tiny("inkling2")
        self.assertEqual(tk.encode("Xyz"), [tk.ids["Xyz"]])           # no merge builds it
        self.assertEqual(tk.encode("abc"), [tk.ids["abc"]])           # the merges build it too: same id
        # a piece that is not an entry still goes through BPE
        self.assertEqual(tk.encode("abcd"), [tk.ids["abc"], tk.ids["d"]])
        q = tiny("qwen35")
        self.assertEqual(len(q.encode("Xyz")), 3)                     # the old behaviour, kept for qwen35

    def test_split_is_still_gpt4o(self):
        # ignore_merges must not switch Inkling to the GLM split: digits group by three, a cased word keeps its case run
        tk = tiny("inkling2")
        self.assertIsNotNone(tk._seq)
        self.assertEqual(split_sequential("1234567 Xyz", tk._seq), ["123", "456", "7", " Xyz"])

    def test_old_pack_gets_the_flag(self):
        # packs written before the fix say "ignore_merges": false; the flag comes from `pre`, not from the file
        tk = tiny("inkling2")
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d)
            (p / "vocab.json").write_text(json.dumps(tk.ids, ensure_ascii=False), encoding="utf-8")
            (p / "merges.txt").write_text("a b\nab c", encoding="utf-8")
            (p / "tokenizer.json").write_text(json.dumps({"pre": "inkling2", "ignore_merges": False}), encoding="utf-8")
            got = Tokenizer.from_dir(p)
        self.assertTrue(got.ignore_merges)
        self.assertEqual(got.encode("Xyz"), [got.ids["Xyz"]])


def _gguf():
    c = os.environ.get("STRATA_INKLING_GGUF") or \
        "/home/zephel01/llm/apps/Abyss-data/models/inkling2mini/unsloth/Inkling-2-Mini-UD-Q4_K_XL.gguf"
    return c if os.path.exists(c) else None


@unittest.skipUnless(_gguf(), "no Inkling 2 GGUF on disk (STRATA_INKLING_GGUF)")
class RealVocab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tk = Tokenizer.from_gguf(_gguf())

    def test_known_entries(self):
        # ids read from the GGUF vocab (Inkling 2 Mini, 200,058 entries)
        for text, i in (("chffff", 37812), ("Aleksandr", 49919), ("VBORw", 61018), ("kwargs", 87550),
                        ("について", 91149), ("ください", 103413), ("ファイル", 124300), (" Graphics", 192621)):
            self.assertEqual(self.tk.encode(text), [i], text)

    def test_every_unbuildable_entry(self):
        tk = self.tk
        u2b = {v: k for k, v in BYTE_TO_UNICODE.items()}
        n = 0
        for i, t in enumerate(tk.tokens):
            if tk.token_types and tk.token_types[i] in (3, 4):
                continue
            try:
                raw = bytes(u2b[c] for c in t).decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                continue
            if split_sequential(raw, tk._seq) != [raw] or tk._bpe(t) == [t]:
                continue
            n += 1
            self.assertEqual(tk.encode(raw), [i], raw)
        self.assertEqual(n, 337)

    def test_sentence(self):
        # the pieces around an entry are untouched: a data URL keeps VBORw as one token
        ids = self.tk.encode("data:image/png;base64,iVBORw0KGgo")
        self.assertIn(61018, ids)


if __name__ == "__main__":
    unittest.main()
