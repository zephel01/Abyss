"""tools/ernie55_convert.py: the MXFP4 re-pack (HF int8 pairs + e8m0 -> ggml block_mxfp4) round-trips against a numpy
dequant of the HF layout bit for bit; the FP8 e4m3 table and the exact FP8 -> BF16 step; the split writer's layout
(shard 1 the keys only, the split.* keys, every tensor once); the chat template against strings written out by hand
from baidu's encoding_ernie.py (read, not run).  No model needed."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ernie55_convert as C  # noqa: E402
import gguf_io as G  # noqa: E402


class Mxfp4(unittest.TestCase):
    def test_known_values(self):
        # byte 0x21: element 0 = code 1 (0.5), element 1 = code 2 (1.0); scale 127 = 2^0; 0xF7: 6.0 then -6.0
        packed = np.zeros((1, 16), np.uint8)
        packed[0, 0], packed[0, 1] = 0x21, 0xF7
        sc = np.full((1, 1), 127, np.uint8)
        v = C.hf_mxfp4_dequant(packed, sc)[0]
        self.assertEqual(list(v[:4]), [0.5, 1.0, 6.0, -6.0])
        g = C.mxfp4_repack(packed, sc)
        self.assertEqual(g.shape, (1, 17))
        self.assertEqual(g[0, 0], 127)
        self.assertEqual(list(C.ggml_mxfp4_dequant(g)[:4]), [0.5, 1.0, 6.0, -6.0])
        sc2 = np.full((1, 1), 120, np.uint8)             # 2^-7
        self.assertEqual(C.hf_mxfp4_dequant(packed, sc2)[0, 1], 2.0 ** -7)

    def test_round_trip_random(self):
        rng = np.random.default_rng(5)
        for n, k in ((3, 64), (16, 2048), (2048, 128)):
            packed = rng.integers(-128, 128, size=(n, k // 2), dtype=np.int8)
            sc = rng.integers(100, 140, size=(n, k // 32), dtype=np.uint8)
            want = C.hf_mxfp4_dequant(packed, sc)
            raw = C.mxfp4_repack(packed, sc)
            self.assertEqual(raw.nbytes, G.nbytes(39, [k, n]))
            got = C.ggml_mxfp4_dequant(raw).reshape(n, k)
            np.testing.assert_array_equal(got, want)               # every value, bit for bit (float64, exact)
            np.testing.assert_array_equal(G.dequantize(raw, 39).reshape(n, k), want.astype(np.float32))
            # the other nibble order is a different tensor (the order is a real choice, settled by the forward)
            other = C.ggml_mxfp4_dequant(C.mxfp4_repack(packed, sc, hi_first=True)).reshape(n, k)
            self.assertFalse(np.array_equal(other, want))

    def test_bytes_are_a_permutation(self):
        # the re-pack keeps every code: per 32-element block the multiset of nibbles is the same
        rng = np.random.default_rng(1)
        packed = rng.integers(0, 256, size=(4, 64), dtype=np.uint8)
        sc = np.full((4, 4), 127, np.uint8)
        raw = C.mxfp4_repack(packed, sc).reshape(4, 4, 17)
        src = packed.reshape(4, 4, 16)
        for i in range(4):
            for b in range(4):
                a = sorted(list(src[i, b] & 15) + list(src[i, b] >> 4))
                q = raw[i, b, 1:]
                self.assertEqual(a, sorted(list(q & 15) + list(q >> 4)))


class Fp8(unittest.TestCase):
    def test_table(self):
        t = C.E4M3
        self.assertEqual(t[0x38], 1.0)
        self.assertEqual(t[0x7E], 448.0)
        self.assertEqual(t[0x01], 2.0 ** -9)
        self.assertEqual(t[0x08], 2.0 ** -6)
        self.assertEqual(t[0xB8], -1.0)
        self.assertTrue(np.isnan(t[0x7F]) and np.isnan(t[0xFF]))
        self.assertEqual(int(np.isnan(t).sum()), 2)

    def test_exact_bf16(self):
        rng = np.random.default_rng(2)
        w = rng.integers(0, 256, size=(200, 300), dtype=np.uint8)
        w[(w & 0x7F) == 0x7F] = 0x38
        sc = np.ldexp(1.0, rng.integers(-20, 4, size=(2, 3))).astype(np.float32)
        b = C.fp8_to_bf16(w, sc)
        got = (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        want = C.E4M3[w].astype(np.float64) * np.repeat(np.repeat(sc.astype(np.float64), 128, 0)[:200], 128, 1)[:, :300]
        np.testing.assert_array_equal(got, want)
        with self.assertRaises(ValueError):
            C.fp8_to_bf16(w, sc * np.float32(1.5))
        w[0, 0] = 0x7F
        with self.assertRaises(ValueError):
            C.fp8_to_bf16(w, sc)


class Split(unittest.TestCase):
    def test_layout(self):
        ts = [C.T("a.weight", [32, 4], 30, ["a"], lambda: np.arange(128, dtype=np.uint16)),
              C.T("b.weight", [64, 2], 0, ["b"], lambda: np.ones(128, np.float32)),
              C.T("c.weight", [32, 2, 3], 39, ["c"], lambda: np.zeros(6 * 17, np.uint8))]
        with tempfile.TemporaryDirectory() as d:
            paths = C.write_split(ts, Path(d), "m", lambda w: w.add_uint32("x.k", 7), "deepseek4", 800, lambda m: None)
            self.assertEqual(len(paths), 3)
            r0 = G.Reader(paths[0])
            self.assertEqual(len(r0.tensors), 0)
            self.assertEqual(r0.get("x.k"), 7)
            self.assertEqual((r0.get("split.no"), r0.get("split.count"), r0.get("split.tensors.count")), (0, 3, 3))
            names = []
            for i, p in enumerate(paths[1:]):
                r = G.Reader(p)
                self.assertEqual(r.get("general.architecture"), "deepseek4")
                self.assertEqual(r.get("split.no"), i + 1)
                self.assertNotIn("x.k", r.fields)
                names += [t.name for t in r.tensors]
            self.assertEqual(names, ["a.weight", "b.weight", "c.weight"])
            r = G.Reader(paths[1])
            np.testing.assert_array_equal(r.by_name["a.weight"].data.view("<u2"), np.arange(128))


class Template(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import jinja2
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError:
            raise unittest.SkipTest("jinja2 is not installed")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: \
            json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        env.filters["from_json"] = json.loads
        cls.t = env.from_string((Path(__file__).resolve().parent / "ernie55_chat_template.jinja").read_text())
        del jinja2

    def r(self, msgs, **kw):
        return self.t.render(messages=msgs, add_generation_prompt=True, **kw)

    def test_user(self):
        self.assertEqual(self.r([{"role": "user", "content": "Hi"}]),
                         "<|ERNIE_MASK|><|ernie:im_start|>system\n<ernie:global_setting>\nreasoning_effort=high\n"
                         "</ernie:global_setting><|ernie:im_end|>\n<|ernie:im_start|>user\nHi<|ernie:im_end|>\n"
                         "<|ernie:im_start|>assistant\n<ernie:think>")

    def test_system_thinking_off(self):
        self.assertEqual(self.r([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
                                enable_thinking=False),
                         "<|ERNIE_MASK|><|ernie:im_start|>system\n<ernie:system_setting>\nBe brief.\n"
                         "</ernie:system_setting>\n<ernie:global_setting>\n</ernie:global_setting><|ernie:im_end|>\n"
                         "<|ernie:im_start|>user\nHi<|ernie:im_end|>\n<|ernie:im_start|>assistant\n"
                         "<ernie:think></ernie:think>")

    def test_tools_and_history(self):
        tools = [{"type": "function", "function": {"name": "w", "parameters": {"type": "object"}}}]
        msgs = [{"role": "user", "content": "Paris?"},
                {"role": "assistant", "content": "", "reasoning_content": "look up",
                 "tool_calls": [{"function": {"name": "w", "arguments": "{\"days\": 2, \"city\": \"Paris\"}"}}]},
                {"role": "tool", "content": "sunny"}, {"role": "tool", "content": "warm"}]
        self.assertEqual(self.r(msgs, tools=tools, reasoning_effort="low"),
                         "<|ERNIE_MASK|><|ernie:im_start|>system\n<ernie:global_setting>\nreasoning_effort=low\n"
                         "parallel_tool_calls=True\n</ernie:global_setting>\n<ernie:tool_list>\n"
                         "[{\"function\": {\"name\": \"w\", \"parameters\": {\"type\": \"object\"}}, \"type\": \"function\"}]"
                         "\n</ernie:tool_list><|ernie:im_end|>\n"
                         "<|ernie:im_start|>user\nParis?<|ernie:im_end|>\n"
                         "<|ernie:im_start|>assistant\n<ernie:think>look up</ernie:think><ernie:tool_call>\n"
                         "<invoke name=\"w\"><parameter name=\"city\">Paris</parameter><parameter name=\"days\">2"
                         "</parameter></invoke>\n</ernie:tool_call><|ernie:im_end|>\n"
                         "<|ernie:im_start|>tool\n<ernie:tool_output>sunny</ernie:tool_output>\n<ernie:tool_output>warm"
                         "</ernie:tool_output><|ernie:im_end|>\n<|ernie:im_start|>assistant\n<ernie:think>")

    def test_inline_think(self):
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "<ernie:think>x</ernie:think>y"},
                {"role": "user", "content": "b"}]
        self.assertIn("<|ernie:im_start|>assistant\n<ernie:think>x</ernie:think>y<|ernie:im_end|>\n"
                      "<|ernie:im_start|>user\nb<|ernie:im_end|>", self.r(msgs))


if __name__ == "__main__":
    unittest.main()
