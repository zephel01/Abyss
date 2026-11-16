"""exlr8 conversion tests (tools/exlr8_convert.py and its codebook in tools/exl3_pack.py); run
.venv/bin/python -m unittest discover -s tools -p test_exlr8.py."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from the repo root too

import gguf_io as G
import exl3_pack
import exlr8_convert as C


class E4M3(unittest.TestCase):
    def test_rounding_is_the_nearest_e4m3_value_ties_to_even(self):
        tab = C.E4M3[:127]                       # the non-negative finite values, ascending; codes 0..126
        rng = np.random.default_rng(1)
        v = np.concatenate([rng.normal(0, 1, 20000), rng.uniform(-500, 500, 2000), rng.uniform(-0.03, 0.03, 4000),
                            tab, -tab, (tab[1:] + tab[:-1]) / 2]).astype(np.float32)
        got = exl3_pack.e4m3_round(v)
        a = np.minimum(np.abs(v), 448.0)
        i = np.clip(np.searchsorted(tab, a), 1, 126)
        lo, hi = tab[i - 1], tab[i]
        pick_hi = (hi - a < a - lo) | ((hi - a == a - lo) & (i % 2 == 0))   # a tie goes to the even code
        want = np.copysign(np.where(pick_hi, hi, lo), v).astype(np.float32)
        np.testing.assert_array_equal(got, want)


class Tiles(unittest.TestCase):
    def split(self, full: np.ndarray, side_in: bool) -> list[np.ndarray]:
        """The exlr8 halves of a trellis [k/16][n/16][32 K bytes] (the card's tile orders)."""
        kt, nt, b = full.shape
        if side_in:   # entry g = rows 128 g..: tile (b, r, c) at 64 b + 8 r + c
            return [full[8 * g:8 * g + 8].reshape(8, nt // 8, 8, b).transpose(1, 0, 2, 3).reshape(-1)
                    for g in range(kt // 8)]
        return [full[:, 8 * g:8 * g + 8].reshape(-1) for g in range(nt // 8)]   # tile (k16, c) at 8 k16 + c

    def test_halves_go_back_to_the_exllamav3_tile_grid(self):
        rng = np.random.default_rng(2)
        for K in (2, 3):
            for side_in in (False, True):
                full = rng.integers(0, 256, (32, 16, 32 * K), dtype=np.uint8)
                got = C.trellis_from_halves(self.split(full, side_in), K, 512, 256, side_in)
                np.testing.assert_array_equal(got, full)


class Dense(unittest.TestCase):
    def test_mxfp8_is_e4m3_times_its_power_of_two(self):
        fp8 = np.array([[0x38, 0xB8, 0x40, 0x01] * 8], dtype=np.uint8)   # 1, -1, 2, 2^-9
        e8 = np.array([[127 + 3]], dtype=np.uint8)
        w = C.decode_dense([C.MXFP8, 1, 32, 0, 0, 0, 0, 0], {"fp8": fp8, "e8m0": e8})
        np.testing.assert_array_equal(w[0, :4], np.array([8, -8, 16, 2.0 ** -6], dtype=np.float32))

    def test_q8_0_matches_the_reference_quantizer(self):
        a = np.random.default_rng(3).normal(0, 0.05, (4, 256)).astype(np.float32)
        self.assertEqual(exl3_pack.to_q8_0(a), G.quantize(a, "Q8_0").tobytes())


if __name__ == "__main__":
    unittest.main()
