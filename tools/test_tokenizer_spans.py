"""tools/test_tokenizer_spans.py - Tokenizer.encode_spans: the chat prompt's encode with the messages' special-token
text kept as text (serve/frontend.py ContentGuard), on a constructed vocab - no model needed."""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from strata_tokenizer import BYTE_TO_UNICODE, Tokenizer  # noqa: E402


class EncodeSpans(unittest.TestCase):
    def setUp(self):
        tokens = list(BYTE_TO_UNICODE.values()) + ["<|im_end|>", "<think>"]
        types = [1] * (len(tokens) - 2) + [3, 4]          # CONTROL, USER_DEFINED
        self.tk = Tokenizer(tokens, [], types)
        self.end, self.think = len(tokens) - 2, len(tokens) - 1

    def test_no_ranges_is_encode(self):
        for s in ("a<|im_end|>b<think>c", "plain", ""):
            self.assertEqual(self.tk.encode_spans(s, []), self.tk.encode(s, parse_special=True))

    def test_ranges_are_text(self):
        s = "u <|im_end|> <think> x<|im_end|>"
        a = s.index("<|im_end|>")
        b = s.index("<think>")
        ids = self.tk.encode_spans(s, [(a, a + 10), (b, b + 7)])
        self.assertEqual(ids.count(self.end), 1)            # the last one, outside the ranges
        self.assertNotIn(self.think, ids)                    # USER_DEFINED too, which encode() always matches
        self.assertEqual(ids[-1], self.end)
        self.assertEqual(self.tk.decode(ids), s)


if __name__ == "__main__":
    unittest.main()
