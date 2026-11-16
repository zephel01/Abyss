"""serve/test_lang_ban.py - lang / lang_think / logit_bias -> the engine's bias sets and GEN keys."""
from __future__ import annotations

import os
import unittest
from unittest import mock

from serve import lang_ban
from serve.server import StrataEngine
from test_token_classes import TinyTok  # noqa: E402 (tools/ is on the path: serve.server puts it there)


class FakeEngine:
    def __init__(self, reply="BIASOK"):
        self.lines, self.reply = [], reply

        class P:
            pid = 1234
        self.proc = P()

    def command(self, line, expect, timeout=60.0):
        self.lines.append(line)
        if not self.reply.startswith(expect):
            raise ValueError(self.reply)
        return self.reply


class LangBanTests(unittest.TestCase):
    def setUp(self):
        self.tok = TinyTok(["the", "誤", "误", "ます", "한국"], specials=["</think>"])
        self.lb = lang_ban.LangBan(self.tok, keep_ids={0})

    def test_check_request(self):
        lang_ban.check_request({"lang": "ja-en", "lang_think": True, "logit_bias": {"3": -100}}, 6)
        for bad in ({"lang": "xx"}, {"lang_think": "yes"}, {"logit_bias": {"9": -100}}, {"logit_bias": {"1": 101}},
                    {"logit_bias": {"a": 1}}, {"logit_bias": [1]}):
            with self.assertRaises(ValueError, msg=bad):
                lang_ban.check_request(bad, 6)

    def test_answer_only_default(self):
        with mock.patch.dict(os.environ, {"STRATA_LANG": "", "STRATA_LANG_THINK": ""}):
            whole, answer = self.lb.sets({"lang": "ja-en"})
            self.assertIsNone(whole)
            self.assertEqual(answer, {2: None, 4: None})
            whole, answer = self.lb.sets({"lang": "ja-en", "lang_think": True})
            self.assertEqual(whole, {2: None, 4: None})
            self.assertIsNone(answer)
            self.assertEqual(self.lb.sets({}), (None, None))

    def test_env_default_and_logit_bias(self):
        with mock.patch.dict(os.environ, {"STRATA_LANG": "zh-en", "STRATA_LANG_THINK": "0"}):
            whole, answer = self.lb.sets({"logit_bias": {"0": -100, "5": -100, "1": 5, "4": 0}})
            # 0 is a kept id and 5 a special: never banned; 4 (0) is nothing
            self.assertEqual(whole, {1: 5.0})
            self.assertEqual(answer, {1: 5.0, 3: None, 4: None})
            self.assertEqual(self.lb.sets({"lang": "off"}), (None, None))

    def test_register_once_per_process(self):
        eng = FakeEngine()
        s = {2: None, 3: -1.5}
        n1 = lang_ban.register(eng, s)
        n2 = lang_ban.register(eng, s)
        self.assertEqual(n1, n2)
        self.assertEqual(eng.lines, [f"BIAS {n1} 2:ban,3:-1.5"])
        eng.proc.pid = 99                       # a restarted engine has no sets
        lang_ban.register(eng, s)
        self.assertEqual(len(eng.lines), 2)
        with self.assertRaises(ValueError):
            lang_ban.register(FakeEngine("ERR expected: GEN"), s)

    def test_gen_keys(self):
        keys = StrataEngine.sampling_keys({"temperature": 0, "_bias": "babc", "_bias_answer": "bdef"})
        self.assertIn(" bias=babc", keys)
        self.assertIn(" bias_answer=bdef", keys)
        self.assertNotIn("bias", StrataEngine.sampling_keys({"temperature": 0.6}))


if __name__ == "__main__":
    unittest.main()
