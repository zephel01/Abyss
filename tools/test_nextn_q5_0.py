"""nextn_pack.py --experts q5_0: Q4_0 blocks repacked into Q5_0 dequantize to the same values (gguf_io's own
dequantizers, no ggml)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_io as G  # noqa: E402
import nextn_pack  # noqa: E402


class Q4ToQ5(unittest.TestCase):
    def test_same_values(self):
        rng = np.random.default_rng(7)
        n = 64
        b = np.empty((n, 18), dtype=np.uint8)
        b[:, :2] = np.float16(rng.uniform(-0.05, 0.05, n)).view(np.uint8).reshape(n, 2)
        b[:, 2:] = rng.integers(0, 256, (n, 16), dtype=np.uint8)
        b[0, 2:] = 0x00          # every code at the bottom (-8) and the top (+7)
        b[1, 2:] = 0xFF
        q5 = nextn_pack.q4_0_to_q5_0(b.reshape(-1))
        self.assertEqual(q5.nbytes, n * 22)
        np.testing.assert_array_equal(G.dequantize(b.reshape(-1), 2), G.dequantize(q5, 6))


if __name__ == "__main__":
    unittest.main()
