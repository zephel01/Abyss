"""Inkling pack tests: Q8_0 loop controllers widen to F32 exactly; run .venv/bin/python -m unittest discover -s tools
-p test_iq_pack_inkling.py."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from the repo root too

import gguf_io as G
import iq_pack

F32, Q8_0 = 0, 8


def write_gguf(path, tensors):
    writer = G.Writer(path, "inkling2_loop")
    for name, values, kind in tensors:
        writer.add_tensor(name, list(reversed(np.shape(values))), kind, G.quantize(values, kind))
    writer.write()


def pack(root, tag, tensors):
    source, out = root / (tag + ".gguf"), root / tag
    out.mkdir()
    write_gguf(source, tensors)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        rc = iq_pack.index_standalone(source, out, iq_pack.Model(source), False, "inkling2_loop")
    return rc, out, log.getvalue()


class InklingControllerTests(unittest.TestCase):
    def test_q8_controllers_pack_as_their_f32_values(self):
        rng = np.random.default_rng(7)
        down = rng.standard_normal((4, 64)).astype(np.float32)    # (a 4608 x 64 down, cut to 4 rows)
        gain = rng.standard_normal((2, 96)).astype(np.float32)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            q = [("blk.6.loop_q_ctrl_down.weight", down, Q8_0), ("blk.6.loop_ffn_ctrl_gain.weight", gain, Q8_0),
                 ("blk.6.attn_q_clip.weight", gain[0], F32)]
            rc, out_q, _ = pack(root, "q8", q)
            self.assertEqual(rc, 0)
            # the same values as an F32 source: what Q8_0 holds, dequantized
            exact = [(n, G.dequantize(G.quantize(v, k), "Q8_0").reshape(v.shape) if k == Q8_0 else v, F32)
                     for n, v, k in q]
            rc, out_f, _ = pack(root, "f32", exact)
            self.assertEqual(rc, 0)
            self.assertEqual((out_q / "dense.bin").read_bytes(), (out_f / "dense.bin").read_bytes())
            _, rq = iq_pack.read_index(out_q / "index.txt")
            _, rf = iq_pack.read_index(out_f / "index.txt")
            for name, *_ in q:
                self.assertEqual(rq[name], rf[name])
            got = np.frombuffer((out_q / "dense.bin").read_bytes()[:down.size * 4], np.float32).reshape(down.shape)
            np.testing.assert_allclose(got, down, atol=np.abs(down).max() / 127)   # (Q8_0's rounding, no more)

    def test_other_q8_tensors_still_refused(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            for name in ("blk.6.loop_msg_query.weight", "blk.0.ffn_gate_inp.weight", "blk.6.loop_q_ctrl_bias.weight"):
                rc, _, log = pack(Path(tmp), name.replace(".", "_"), [(name, np.ones((2, 32), np.float32), Q8_0)])
                self.assertEqual(rc, 1, name)
                self.assertIn("refusing to pack it silently", log)


if __name__ == "__main__":
    unittest.main()
