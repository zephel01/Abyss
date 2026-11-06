"""unsloth's UD 3.1 Qwen3.8 sets, from their headers only (tools/fixtures/unsloth_q38_ud31_types.txt, written by
tools/gguf_remote_types.py): every routed-expert type and gate/up + down pairing must be one the engine's lists
carry - the GPU expert kernels (decode and the verify window), the prompt's MMQ (its type switch and both builds'
instance lists), the mixed gate/up pairs, the CPU lane's test table (native_split_test.cpp checks those pairs
against ggml's dots) - and iq_pack.py must pack each set's layer types.  No download, no GPU.
Run .venv/bin/python -m unittest discover -s tools -p test_unsloth_types.py."""
import contextlib
import io
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from _paths import add_gguf_py
add_gguf_py()
from gguf import GGUFWriter, GGMLQuantizationType as Q, GGML_QUANT_SIZES
import iq_pack
from gguf_remote_types import TYPE_NAMES

FIXTURE = HERE / "fixtures" / "unsloth_q38_ud31_types.txt"
IDS = {v: k for k, v in TYPE_NAMES.items()}


def fixture():
    sets = {}
    for line in FIXTURE.read_text().splitlines():
        if line and not line.startswith("#"):
            s, layer, g, u, d = line.split()
            sets.setdefault(s, []).append((int(layer), IDS[g], IDS[u], IDS[d]))
    return sets


def src(rel):
    return (ROOT / rel).read_text()


class UnslothUd31Types(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sets = fixture()
        cls.types = {t for layers in cls.sets.values() for _, g, u, d in layers for t in (g, u, d)}

    def test_fixture_is_whole(self):
        self.assertEqual(set(self.sets), {"UD-IQ1_S", "UD-IQ1_M", "UD-Q2_0", "UD-IQ2_XS", "UD-Q2_K_XL", "UD-IQ3_XXS",
                                          "UD-Q3_K_XL", "UD-IQ4_XS"})
        for s, layers in self.sets.items():
            self.assertEqual([l for l, *_ in layers], list(range(48)), s)

    def test_gpu_expert_kernels_carry_every_type(self):
        cu = src("src/kernels/cuda/glm_fast.cu")
        gu = {int(t) for t in re.findall(r"GLMF_GU\((\d+)\)", cu)}
        dn = {int(t) for t in re.findall(r"GLMF_DN\((\d+)\)", cu)}
        body = re.search(r"bool moe_supported\(int type\) \{(.*?)\n\}", cu, re.S).group(1)
        ok = {int(t) for t in re.findall(r"type == (\d+)", body)}
        pairs = {(int(a), int(b)) for a, b in re.findall(r"X\((\d+), (\d+)\)",
                                                        re.search(r"#define GLMF_GU_PAIRS\(X\)(.*)", cu).group(1))}
        for s, layers in self.sets.items():
            for l, g, u, d in layers:
                where = "%s layer %d" % (s, l)
                self.assertIn(g, gu & ok, where)
                self.assertIn(d, dn & ok, where)
                if g != u:
                    self.assertIn((g, u), pairs, where + ": mixed gate/up")

    def test_prompt_mmq_carries_every_type(self):
        # an expert type outside MMQ falls to the per-expert FP16 decode on the prompt path (correct, much slower):
        # UD-IQ4_XS's 33 Q5_0 down layers were that until Q5_0 joined the list
        mmq = src("src/prefill/moe_mmq.cu")
        body = re.search(r"bool supported\(int t\) \{(.*?)\n\}", mmq, re.S).group(1)
        sup = {getattr(Q, n) for n in re.findall(r"case GGML_TYPE_(\w+):", body)}
        disp = {getattr(Q, n) for n in re.findall(r"case GGML_TYPE_(\w+): mul_mat_q_case", mmq)}
        lists = re.findall(r"foreach\(_t ([a-z0-9_ ]+)\)", src("CMakeLists.txt"))
        lists = [{getattr(Q, n.upper()) for n in l.split()} for l in lists if "iq2_xxs" in l]
        self.assertEqual(len(lists), 2, "the CUDA and the HIP build's MMQ instance lists")
        for t in self.types:
            self.assertIn(t, sup, TYPE_NAMES[t])
            self.assertIn(t, disp, TYPE_NAMES[t])
            for l in lists:
                self.assertIn(t, l, TYPE_NAMES[t])

    def test_cpu_lane_test_table_covers_every_pair(self):
        test = src("src/kernels/cpu/native_split_test.cpp")
        table = re.search(r"BEGIN unsloth-ud31-combos(.*?)END unsloth-ud31-combos", test, re.S).group(1)
        combos = {(getattr(Q, a), getattr(Q, b))
                  for a, b in re.findall(r"\{GGML_TYPE_(\w+), GGML_TYPE_(\w+)\}", table)}
        for s, layers in self.sets.items():
            for l, g, u, d in layers:
                if g == u:   # (mixed layers: native_split_test's own UD-IQ1_S case)
                    self.assertIn((g, d), combos, "%s layer %d" % (s, l))

    def test_iq_pack_packs_each_sets_types(self):
        # a small qwen4exp GGUF per set with the set's distinct layer types (8 experts, n_embd 256, n_ff 64; the
        # bytes are zeros - the pack copies offsets, not values): native_experts.txt names each layer's types and
        # sizes its blob from the types' block sizes
        E, FF, NE = 256, 64, 8

        def raw(t, rows, cols):
            blk, size = GGML_QUANT_SIZES[Q(t)]
            return np.zeros((NE, rows, cols // blk * size), np.uint8)

        for s, layers in self.sets.items():
            seen, pick = set(), []
            for l, g, u, d in layers:
                if (g, u, d) not in seen:
                    seen.add((g, u, d))
                    pick.append((g, u, d))
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                root = Path(tmp)
                first = root / "model-00001-of-00001.gguf"
                w = GGUFWriter(first, "qwen4exp")
                for i, (g, u, d) in enumerate(pick):
                    w.add_tensor("blk.%d.ffn_gate_inp.weight" % i, np.ones((NE, E), np.float32))
                    w.add_tensor("blk.%d.ffn_gate_exps.weight" % i, raw(g, FF, E), raw_dtype=Q(g))
                    w.add_tensor("blk.%d.ffn_up_exps.weight" % i, raw(u, FF, E), raw_dtype=Q(u))
                    w.add_tensor("blk.%d.ffn_down_exps.weight" % i, raw(d, E, FF), raw_dtype=Q(d))
                w.write_header_to_file()
                w.write_kv_data_to_file()
                w.write_tensors_to_file()
                w.close()
                out = root / "pack"
                (out / "tokenizer").mkdir(parents=True)
                for name in ["vocab.json", "chat_template.jinja"]:
                    (out / "tokenizer" / name).touch()
                with patch.object(sys, "argv", ["iq_pack.py", "--gguf", str(first), "--out", str(out), "--fast"]):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(iq_pack.main(), 0, s)
                rows = [r.split() for r in (out / "native_experts.txt").read_text().splitlines()
                        if r and not r.startswith("#")]
                self.assertEqual(len(rows), len(pick), s)
                for row, (g, u, d) in zip(rows, pick):
                    self.assertEqual(row[1], str(g) if g == u else "%d,%d" % (g, u), s)
                    self.assertEqual(int(row[2]), d, s)
                    per = sum(raw(t, r, c)[0].nbytes for t, r, c in ((g, FF, E), (u, FF, E), (d, E, FF)))
                    self.assertEqual(int(row[4]), per, s)


if __name__ == "__main__":
    unittest.main()
