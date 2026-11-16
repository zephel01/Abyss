"""tools/test_token_classes.py - the token classes and the language profiles on a tiny byte-level vocabulary."""
from __future__ import annotations

import gzip
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import token_classes as TC  # noqa: E402


class TinyTok:
    """tokens as raw byte strings; types 3 for the specials."""

    def __init__(self, pieces, specials=()):
        self.raw = [p if isinstance(p, bytes) else p.encode("utf-8") for p in pieces] + \
                   [s.encode() for s in specials]
        self.tokens = [r.decode("latin-1") for r in self.raw]
        self.token_types = [1] * len(pieces) + [3] * len(specials)

    def token_bytes(self, i):
        return self.raw[i]


class CharClassTests(unittest.TestCase):
    def test_scripts(self):
        self.assertEqual(TC.char_class("a"), TC.LATIN)
        self.assertEqual(TC.char_class("é"), TC.LATIN)
        self.assertEqual(TC.char_class("Ａ"), TC.LATIN)
        self.assertEqual(TC.char_class("あ"), TC.KANA)
        self.assertEqual(TC.char_class("ー"), TC.KANA)
        self.assertEqual(TC.char_class("ｱ"), TC.KANA)
        self.assertEqual(TC.char_class("한"), TC.HANGUL)
        self.assertEqual(TC.char_class("Д"), TC.CYRILLIC)
        self.assertEqual(TC.char_class("α"), TC.GREEK)
        for c in "1。、！?  \n😀%":
            self.assertEqual(TC.char_class(c), 0, c)

    def test_han_jis(self):
        # 誤 (JIS) / 误 (simplified, not in JIS X 0213) / 這 (JIS) / 这 (not)
        self.assertTrue(TC.char_class("誤") & TC.HAN_JIS)
        self.assertTrue(TC.char_class("误") & TC.HAN_NONJIS)
        self.assertTrue(TC.char_class("這") & TC.HAN_JIS)
        self.assertTrue(TC.char_class("这") & TC.HAN_NONJIS)
        self.assertTrue(TC.char_class("这") & TC.HAN_SIMP)
        self.assertTrue(TC.char_class("們") & TC.HAN_TRAD)
        # a JIS X 0213 plane-2 / non-0208 character Japanese uses (𠮟 U+20B9F) stays JIS
        self.assertTrue(TC.char_class("\U00020B9F") & TC.HAN_JIS)

    def test_partial(self):
        lead, mid, tail = TC.split_utf8("误".encode()[:2])
        self.assertEqual((lead, mid, tail), (0, "", "误".encode()[:2]))
        lead, mid, tail = TC.split_utf8("误".encode()[1:] + "a".encode())
        self.assertEqual((lead, mid, tail), (2, "a", b""))
        self.assertTrue(TC.token_class("误".encode()[:2]) & TC.PARTIAL)
        # a Hangul-only lead pair: every completion is Hangul
        self.assertEqual({c & TC.SCRIPTS for c in TC.fragment_classes("한".encode()[:2])}, {TC.HANGUL})


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.tok = TinyTok(["the", "誤って", "误", "这样", "皮膚", "ます", "한국", "Привет", "α", "123", "。",
                            "这".encode()[:2], "한".encode()[:2], b"\x99", "両性具有", "误".encode()[:2]], specials=["<think>"])
        self.table = TC.build_table(self.tok)

    def names(self, profile):
        return [self.tok.raw[i].decode("utf-8", "replace") for i in TC.banned_ids(self.tok, self.table, profile)]

    def test_ja_en(self):
        got = self.names("ja-en")
        for w in ("误", "这样", "한국", "Привет"):
            self.assertIn(w, got)
        for w in ("the", "誤って", "皮膚", "ます", "α", "123", "。", "<think>", "両性具有"):
            self.assertNotIn(w, got)
        ban = TC.banned_ids(self.tok, self.table, "ja-en")
        self.assertNotIn(11, ban)   # 这's first two bytes also begin JIS characters (迎, 近, 返): kept
        self.assertIn(15, ban)      # 误's: U+8BC0..8BFF, simplified speech-radical characters only
        self.assertIn(12, ban)      # a Hangul lead pair: every character it begins is banned
        self.assertNotIn(13, ban)   # a continuation byte: unknowable here, kept

    def test_zh_en_and_ko_en(self):
        zh = self.names("zh-en")
        self.assertIn("ます", zh)
        self.assertIn("誤って", zh)   # (kana inside)
        self.assertNotIn("误", zh)
        self.assertNotIn("皮膚", zh)
        ko = self.names("ko-en")
        self.assertNotIn("한국", ko)
        self.assertIn("皮膚", ko)

    def test_keep_and_specials(self):
        ban = TC.banned_ids(self.tok, self.table, "en", keep={0, 5})
        self.assertNotIn(5, ban)
        self.assertNotIn(len(self.tok.raw) - 1, ban)   # the special
        self.assertIn(1, ban)

    def test_table_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            fp = TC.fingerprint(self.tok)
            p = TC.save_table(self.table, "tiny", fp, len(self.tok.raw), d)
            raw = gzip.decompress(p.read_bytes())
            self.assertEqual(len(raw), 2 * len(self.tok.raw))
            self.assertEqual(json.loads((d / "index.json").read_text())["tiny"]["fingerprint"], fp)
            got, src = TC.load_table(self.tok, d)
            self.assertEqual(src, "tiny")
            self.assertEqual(list(got), list(self.table))

    def test_text_chars(self):
        self.assertEqual(TC.text_chars_banned("これは误って这样です", "ja-en"), ["误", "这", "样"])
        self.assertEqual(TC.text_chars_banned("両性具有", "ja-en"), [])

    def test_shipped_index(self):
        idx = json.loads((TC.DATA / "index.json").read_text())
        for name in ("qwen35", "glm5", "deepseek4"):
            self.assertIn(name, idx)
            self.assertTrue((TC.DATA / idx[name]["file"]).exists())


if __name__ == "__main__":
    unittest.main()
