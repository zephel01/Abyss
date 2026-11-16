"""tools/gguf_io.py: the reader and writer round trip, the plain dequantizers and BF16 rounding; against llama.cpp's
gguf-py (the oracle the pack tools' files were written with) when STRATA_GGUF_PY names it: the same calls give the
same file bytes, the dequantizers the same float bits."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gguf_io as G  # noqa: E402

PLAIN = ["F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]


def oracle():
    p = os.environ.get("STRATA_GGUF_PY")
    if not p or not (Path(p) / "gguf").is_dir():
        return None
    if p not in sys.path:
        sys.path.insert(0, p)
    import gguf
    return gguf


def random_blocks(rng, t: int, nblocks: int) -> np.ndarray:
    _, bl, sz = G.TYPES[t]
    if t == 0:
        return rng.standard_normal(nblocks).astype("<f4").view(np.uint8)
    if t in (1, 30):
        v = rng.standard_normal(nblocks).astype(np.float32)
        return (v.astype("<f2") if t == 1 else G.to_bf16(v)).view(np.uint8)
    b = rng.integers(0, 256, size=(nblocks, sz), dtype=np.uint8)
    scale = lambda: rng.uniform(1e-3, 2e-2, size=(nblocks, 1)).astype("<f2").view(np.uint8)   # noqa: E731
    pos = {2: [0], 3: [0, 2], 6: [0], 7: [0, 2], 8: [0], 10: [80, 82], 11: [108], 12: [0, 2], 13: [0, 2], 14: [208]}[t]
    for o in pos:
        b[:, o:o + 2] = scale()
    return b.reshape(-1)


class GGUFIO(unittest.TestCase):
    def build(self, path, mod=None):
        w = G.Writer(path, "testarch") if mod is None else mod.GGUFWriter(str(path), "testarch")
        if mod is None:
            w.add_string("general.name", "t")
            w.add_uint32("testarch.block_count", 3)
            w.add_int32("testarch.i", -5)
            w.add_uint64("testarch.u64", 1 << 40)
            w.add_float32("testarch.eps", 1e-6)
            w.add_bool("testarch.flag", True)
            w.add_array("tokenizer.ggml.tokens", ["a", "bc", "日本"])
            w.add_array("testarch.ints", [1, 2, 3])
            w.add_array("testarch.floats", [0.5, 1.5])
            w.add_array("testarch.bools", [True, False, True])
        else:
            w.add_string("general.name", "t")
            w.add_uint32("testarch.block_count", 3)
            w.add_int32("testarch.i", -5)
            w.add_uint64("testarch.u64", 1 << 40)
            w.add_float32("testarch.eps", 1e-6)
            w.add_bool("testarch.flag", True)
            w.add_array("tokenizer.ggml.tokens", ["a", "bc", "日本"])
            w.add_array("testarch.ints", [1, 2, 3])
            w.add_array("testarch.floats", [0.5, 1.5])
            w.add_array("testarch.bools", [True, False, True])
        rng = np.random.default_rng(1)
        f32 = rng.standard_normal((3, 64)).astype(np.float32)
        q8 = random_blocks(rng, 8, 3 * 4).reshape(3, 4 * 34)
        q6 = random_blocks(rng, 14, 5).reshape(5, 210)
        bf = G.to_bf16(rng.standard_normal((2, 7)).astype(np.float32))
        if mod is None:
            w.add_tensor("a.weight", [64, 3], "F32", f32)
            w.add_tensor("b.weight", [128, 3], "Q8_0", q8)
            w.add_tensor("c.weight", [256, 5], "Q6_K", q6)
            w.add_tensor("d.weight", [7, 2], "BF16", bf)
            w.write()
        else:
            Q = mod.GGMLQuantizationType
            w.add_tensor("a.weight", f32)
            w.add_tensor("b.weight", q8, raw_dtype=Q.Q8_0)
            w.add_tensor("c.weight", q6, raw_dtype=Q.Q6_K)
            w.add_tensor("d.weight", bf, raw_dtype=Q.BF16)
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()
        return f32, q8, q6, bf

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.gguf"
            f32, q8, q6, bf = self.build(p)
            r = G.Reader(p)
            self.assertEqual(r.get("general.architecture"), "testarch")
            self.assertEqual(r.get("testarch.u64"), 1 << 40)
            self.assertEqual(r.get("tokenizer.ggml.tokens"), ["a", "bc", "日本"])
            self.assertEqual(r.get("testarch.ints"), [1, 2, 3])
            self.assertEqual(r.fields["testarch.ints"].etype, G.I32)
            self.assertEqual([t.name for t in r.tensors], ["a.weight", "b.weight", "c.weight", "d.weight"])
            self.assertTrue(np.array_equal(r.by_name["a.weight"].data.view("<f4").reshape(3, 64), f32))
            self.assertEqual(r.by_name["b.weight"].data.tobytes(), q8.tobytes())
            self.assertEqual(r.by_name["c.weight"].data.tobytes(), q6.tobytes())
            self.assertEqual(r.by_name["d.weight"].data.tobytes(), bf.tobytes())
            # a copy through copy_fields + the same tensors is the same file
            w = G.Writer(Path(d) / "y.gguf", "testarch")
            G.copy_fields(r, w)
            for t in r.tensors:
                w.add_tensor(t.name, t.shape, t.type, t.data)
            w.write()
            self.assertEqual((Path(d) / "y.gguf").read_bytes(), p.read_bytes())

    def test_dequant_q8_0_and_bf16(self):
        b = np.zeros(34, dtype=np.uint8)
        b[0:2] = np.array([0.5], dtype="<f2").view(np.uint8)
        b[2:] = np.arange(-16, 16, dtype=np.int8).view(np.uint8)
        self.assertTrue(np.array_equal(G.dequantize(b, "Q8_0"), np.arange(-16, 16, dtype=np.float32) * 0.5))
        v = np.array([1.0, 1.00390625, 1.01171875, -2.5], dtype=np.float32)   # ties go to even
        self.assertEqual(G.to_bf16(v).tolist(), [0x3F80, 0x3F80, 0x3F82, 0xC020])
        with self.assertRaises(G.NeedsLlamaCpp):
            G.dequantize(np.zeros(66, dtype=np.uint8), "IQ2_XXS")

    @unittest.skipIf(oracle() is None, "STRATA_GGUF_PY not set (the gguf-py oracle)")
    def test_against_gguf_py(self):
        gguf = oracle()
        with tempfile.TemporaryDirectory() as d:
            self.build(Path(d) / "own.gguf")
            self.build(Path(d) / "ref.gguf", gguf)
            self.assertEqual((Path(d) / "own.gguf").read_bytes(), (Path(d) / "ref.gguf").read_bytes())
        rng = np.random.default_rng(3)
        for name in PLAIN:
            t = G.TYPE_ID[name]
            raw = random_blocks(rng, t, 512 if G.TYPES[t][1] == 1 else 96)
            if name in ("F32", "F16", "BF16"):
                ref = gguf.quants.dequantize(raw.reshape(1, -1), gguf.GGMLQuantizationType[name]) if name == "BF16" else \
                    raw.view({"F32": "<f4", "F16": "<f2"}[name]).astype(np.float32)
            else:
                ref = gguf.quants.dequantize(raw.reshape(-1, G.TYPES[t][2]), gguf.GGMLQuantizationType[name])
            got = G.dequantize(raw, t)
            self.assertEqual(got.astype(np.float32).tobytes(), np.asarray(ref, dtype=np.float32).reshape(-1).tobytes(),
                             name)
        v = rng.standard_normal(4096).astype(np.float32) * 3
        v[:32] = np.arange(-16, 16) * 0.5                     # ties for the Q8_0 rounding (half away from zero)
        for name in ("F32", "F16", "BF16", "Q8_0"):
            ref = gguf.quants.quantize(v, gguf.GGMLQuantizationType[name])
            self.assertEqual(G.quantize(v, name).tobytes(), np.asarray(ref).tobytes(), name)


if __name__ == "__main__":
    unittest.main()
