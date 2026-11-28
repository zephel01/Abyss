"""tools/test_iq_pack_ernie5.py - iq_pack's ernie5_5 rewrites (Xylolabs' ERNIE-5.5 GGUF layout -> ours): the router
split into its two BF16 halves, and k_b / v_b (Q8_0) joined into an F32 attn_kv_b [heads][nope + v][latent]."""
from __future__ import annotations

import pathlib
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_io  # noqa: E402
import iq_pack  # noqa: E402


class FakeModel:
    """name -> (None, TensorInfo-like, bytes); bytes(name) the raw tensor bytes, as iq_pack.Model's."""

    def __init__(self):
        self.where, self.raw = {}, {}

    def add(self, name, shape, type_name, raw):
        self.where[name] = (None, types.SimpleNamespace(name=name, shape=shape, type_name=type_name), None, None)
        self.raw[name] = np.frombuffer(raw, dtype=np.uint8)

    def bytes(self, name):
        return self.raw[name]


def collect():
    rows = []

    def put(nm, raw, kind, ne0, ne1):
        rows.append((nm, raw, kind, ne0, ne1))
    return rows, put


class Ernie5Rewrite(unittest.TestCase):
    def test_router_split(self):
        rng = np.random.default_rng(1)
        ne0, ne1 = 64, 16   # n_embd, 2 x n_expert
        w = gguf_io.to_bf16(rng.standard_normal((ne1, ne0)).astype(np.float32))   # BF16-exact values
        f = (w.astype(np.uint32) << 16).view(np.float32)
        m = FakeModel()
        m.add("blk.1.ffn_gate_inp.weight", [ne0, ne1], "F32", f.tobytes())
        rows, put = collect()
        self.assertEqual(iq_pack.ernie5_rewrite("blk.1.ffn_gate_inp.weight", m.where["blk.1.ffn_gate_inp.weight"][1], m,
                                                put, []), 0)
        self.assertEqual([r[0] for r in rows], ["blk.1.ffn_gate_inp.weight", "blk.1.ffn_gate_inp_1.weight"])
        self.assertEqual(rows[0][1], w[:8].tobytes())
        self.assertEqual(rows[1][1], w[8:].tobytes())
        self.assertEqual((rows[0][2], rows[0][3], rows[0][4]), ("4", ne0, 8))

    def test_router_not_bf16_refused(self):
        m = FakeModel()
        m.add("blk.1.ffn_gate_inp.weight", [32, 4], "F32", np.full((4, 32), 0.1, np.float32).tobytes())
        rows, put = collect()
        self.assertEqual(iq_pack.ernie5_rewrite("blk.1.ffn_gate_inp.weight", m.where["blk.1.ffn_gate_inp.weight"][1], m,
                                                put, []), 1)

    def test_kv_b_join(self):
        rng = np.random.default_rng(2)
        H, nope, V, L = 2, 32, 64, 96
        kb = rng.standard_normal((H, L, nope)).astype(np.float32)   # numpy [h][c][n]: ne0 nope, ne1 latent, ne2 heads
        vb = rng.standard_normal((H, V, L)).astype(np.float32)      # [h][d][c]
        m = FakeModel()
        m.add("blk.7.attn_k_b.weight", [nope, L, H], "Q8_0", gguf_io.quantize(kb, "Q8_0").tobytes())
        m.add("blk.7.attn_v_b.weight", [L, V, H], "Q8_0", gguf_io.quantize(vb, "Q8_0").tobytes())
        rows, put = collect()
        tk = m.where["blk.7.attn_k_b.weight"][1]
        self.assertEqual(iq_pack.ernie5_rewrite("blk.7.attn_v_b.weight", m.where["blk.7.attn_v_b.weight"][1], m, put, []), 0)
        self.assertEqual(rows, [])   # (v_b goes with k_b)
        self.assertEqual(iq_pack.ernie5_rewrite("blk.7.attn_k_b.weight", tk, m, put, []), 0)
        (nm, raw, kind, ne0, ne1), = rows
        self.assertEqual((nm, kind, ne0, ne1), ("blk.7.attn_kv_b.weight", "2", L, H * (nope + V)))
        kv = np.frombuffer(raw, dtype=np.float32).reshape(H, nope + V, L)
        kq = gguf_io.dequantize(m.raw["blk.7.attn_k_b.weight"], "Q8_0").reshape(H, L, nope)
        vq = gguf_io.dequantize(m.raw["blk.7.attn_v_b.weight"], "Q8_0").reshape(H, V, L)
        np.testing.assert_array_equal(kv[:, :nope, :], kq.transpose(0, 2, 1))   # exact: Q8_0 values are F32 values
        np.testing.assert_array_equal(kv[:, nope:, :], vq)
        self.assertLess(np.abs(kv[:, :nope, :] - kb.transpose(0, 2, 1)).max(), 0.05)

    def test_needs_bf16_hc_fn_only(self):
        self.assertTrue(iq_pack.needs_bf16("blk.3.hc_attn_fn.weight", "F32", "ernie5_5"))
        self.assertTrue(iq_pack.needs_bf16("output_hc_fn.weight", "F32", "ernie5_5"))
        self.assertFalse(iq_pack.needs_bf16("blk.3.attn_vha.weight", "F32", "ernie5_5"))
        self.assertFalse(iq_pack.needs_bf16("blk.3.ffn_gate_inp.weight", "F32", "ernie5_5"))


if __name__ == "__main__":
    unittest.main()
