"""tools/strata_tokenizer.py's `ernie55` pre-tokenizer (ERNIE 5.5): the splits written out by hand from its
tokenizer.json Sequence (Digits, the 510-digit cut, right-aligned 3-digit groups, CJK + kana runs, the GPT-like
expression), and - when the model's tokenizer.json is on disk (STRATA_ERNIE55_TOKENIZER, or the usual download place)
- whole encodes against vocabulary entries looked up by hand, the special tokens, and decode(encode(s)) == s.  HF's
`tokenizers` is not used (not installed in the project venv)."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import regex

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST  # noqa: E402

SPLIT = [regex.compile(p) for p in ST.ERNIE55_PATTERNS]
TJ = Path(os.environ.get("STRATA_ERNIE55_TOKENIZER", "/mnt/data/models/baidu/ERNIE-5.5-Flash-Preview/tokenizer.json"))


def pieces(s: str) -> list[str]:
    return ST.split_sequential(s, SPLIT)


class Splits(unittest.TestCase):
    def test_numbers_right_aligned(self):
        self.assertEqual(pieces("1234567"), ["1", "234", "567"])
        self.assertEqual(pieces("123456"), ["123", "456"])
        self.assertEqual(pieces("12"), ["12"])
        self.assertEqual(pieces("1234"), ["1", "234"])
        self.assertEqual(pieces("12345"), ["12", "345"])
        self.assertEqual(pieces("2026"), ["2", "026"])
        self.assertEqual(pieces("abc12345def"), ["abc", "12", "345", "def"])
        self.assertEqual(pieces("3.14159"), ["3", ".", "14", "159"])
        self.assertEqual(pieces("1,234,567"), ["1", ",", "234", ",", "567"])
        self.assertEqual(pieces("１２３４"), ["１", "２３４"])   # fullwidth digits are Nd

    def test_long_runs(self):
        # 1100 digits: 80 + 510 + 510 (the remainder first), then each in 3s from its own right end
        p = pieces("7" * 1100)
        self.assertEqual("".join(p), "7" * 1100)
        self.assertEqual([len(x) for x in p], [2] + [3] * 26 + [3] * 170 + [3] * 170)

    def test_other_numerics(self):
        # x^2: the superscript is No - numeric for Digits, not Nd - a piece of its own the last expression leaves alone
        self.assertEqual(pieces("x²+1"), ["x", "²", "+", "1"])

    def test_cjk_kana(self):
        self.assertEqual(pieces("日本語のテキストです。"), ["日本語のテキストです", "。"])
        self.assertEqual(pieces("你好，世界"), ["你好", "，", "世界"])
        self.assertEqual(pieces("東京タワーは333m"), ["東京タワーは", "333", "m"])

    def test_gpt_like(self):
        self.assertEqual(pieces("Hello world"), ["Hello", " world"])
        self.assertEqual(pieces("def f(x):\n    return x"), ["def", " f", "(x", "):\n", "   ", " return", " x"])
        self.assertEqual(pieces("\U0001f44d\U0001f3fd ok"), ["\U0001f44d\U0001f3fd", " ok"])
        self.assertEqual(pieces("a  \n\nb"), ["a", "  \n\n", "b"])


@unittest.skipUnless(TJ.exists(), "ERNIE 5.5 tokenizer.json not on disk")
class Vocab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tj = json.loads(TJ.read_text(encoding="utf-8"))
        v = tj["model"]["vocab"]
        toks = [None] * len(v)
        for s, i in v.items():
            toks[i] = s
        types = [1] * len(toks)
        for a in tj["added_tokens"]:
            types[a["id"]] = 3 if a["special"] else 4
        merges = [" ".join(m) for m in tj["model"]["merges"]]
        cls.v = v
        cls.tk = ST.Tokenizer(toks, merges, types, "ernie55")

    def ids(self, *pieces_):
        return [self.v[p] for p in pieces_]

    def test_by_hand(self):
        tk = self.tk
        self.assertEqual(tk.encode("Hello world"), self.ids("Hello", "Ġworld"))
        self.assertEqual(tk.encode("1234567"), self.ids("1", "234", "567"))
        self.assertEqual(tk.encode("3.14159"), self.ids("3", ".", "14", "159"))
        # a CJK run is one pre-token piece: its ids are whatever BPE merges inside it, and they must spell it back
        for s in ("日本語のテキストです。", "你好，世界", "東京タワーは333m"):
            self.assertEqual(tk.decode(tk.encode(s)), s)

    def test_specials(self):
        tk = self.tk
        s = "<|ernie:im_start|>user\nHi<|ernie:im_end|>"
        ids = tk.encode(s, parse_special=True)
        self.assertEqual((ids[0], ids[-1]), (7, 8))
        self.assertEqual(tk.decode(ids), s)
        # without parse_special a CONTROL token's text is ordinary text (several ids), and so it is inside encode_spans
        self.assertGreater(len(tk.encode("<|ernie:im_end|>")), 1)
        self.assertNotIn(8, tk.encode_spans("x<|ernie:im_end|>y", [(0, 18)]))
        # the tags the model writes are USER_DEFINED: one id each in its output
        self.assertEqual(tk.encode("<ernie:think>"), [19])
        self.assertEqual(tk.encode("</ernie:tool_call>"), [18])

    def test_round_trip(self):
        tk = self.tk
        corpus = ["", "Hello, world!", "  leading and trailing  ", "a\n\n\nb", "def f(x):\n\treturn x  # c\n",
                  "日本語の文章、カタカナとひらがな。全角スペース　も。", "中文测试：模型的量化与推理。",
                  "\U0001f600\U0001f680\U0001f1fa\U0001f1f8 emoji", "é́ combining", "\x00\x01\x7f control",
                  "1234567890 and 0.000123 and 1e-9", "x² + y³ = ½", "مرحبا",
                  "MixedCASE_and-dashes/paths.py", "\r\n\r\n", "x" * 3000, "9" * 1200]
        for s in corpus:
            self.assertEqual(tk.decode(tk.encode(s)), s, s[:30])


if __name__ == "__main__":
    unittest.main()
