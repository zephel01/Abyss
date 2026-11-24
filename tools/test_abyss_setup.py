"""tools/abyss_setup.py's Abyss-tools folder: made from ops/abyss-tools with Abyss's path, the scripts follow the
template (a changed one kept as .bak), the README is the user's once written.  The DSpark drafter's step: downloaded
(here a local file), checked and named in the config's env, kept, taken out.  Another toolkit than build/ was
configured with: configured from scratch (CMake drops every other -D otherwise, issue #1); a kept config takes the
engine's lib_dirs.  No GPU, no network.

    python -m unittest tools.test_abyss_setup
"""
import hashlib
import json
import sys
import tempfile
import unittest
import urllib.error
from argparse import Namespace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abyss_setup as A  # noqa: E402


class AbyssTools(unittest.TestCase):
    def test_made_from_the_templates(self):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "Abyss-tools"
            A.make_tools(t)
            for name in ("abyss.sh", "common.sh", "README.md"):
                self.assertTrue((t / name).exists(), name)
            self.assertTrue((t / "logs").is_dir())
            common = (t / "common.sh").read_text(encoding="utf-8")
            self.assertIn(f'ABYSS="${{ABYSS_DIR:-{A.ROOT}}}"', common)
            self.assertNotIn("@ABYSS@", common)
            self.assertTrue((t / "abyss.sh").stat().st_mode & 0o111)

    def test_again_updates_the_scripts_and_keeps_the_readme(self):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "Abyss-tools"
            A.make_tools(t)
            (t / "README.md").write_text("my notes\n", encoding="utf-8")
            (t / "abyss.sh").write_text("old\n", encoding="utf-8")
            A.make_tools(t)
            self.assertEqual((t / "README.md").read_text(encoding="utf-8"), "my notes\n")
            self.assertEqual((t / "abyss.sh.bak").read_text(encoding="utf-8"), "old\n")
            self.assertEqual((t / "abyss.sh").read_text(encoding="utf-8"),
                             (A.TOOL_TEMPLATES / "abyss.sh").read_text(encoding="utf-8"))
            self.assertFalse((t / "common.sh.bak").exists())   # unchanged: no backup

    def test_the_tested_quant_is_known_offline(self):
        files = A.KNOWN["UD-Q2_K_XL"]
        self.assertEqual(len(files), 3)
        self.assertTrue(all(len(sha) == 64 for _, _, sha in files))



class PickQuant(unittest.TestCase):
    """fit_quant / pc_memory / choose_quant: the PC-size file for this PC's VRAM + RAM, with the sizes GPUs and RAM
    report (a 16 GB card 15.9, a 64 GB PC ~62)."""
    PCS = {"12/32": (11.99, 31.1), "16/64": (15.92, 62.5), "24/64": (23.99, 62.5), "32/96": (31.84, 93.9),
           "32/128": (31.84, 125.6), "32/123": (31.84, 123.0)}

    def pick(self, key, pc):
        vram, ram = self.PCS[pc]
        return A.fit_quant(key, vram, ram)

    def test_qwen38(self):
        self.assertEqual(self.pick("qwen38", "12/32")["quant"], "12G32G")
        self.assertEqual(self.pick("qwen38", "16/64")["quant"], "16G64G")
        self.assertEqual(self.pick("qwen38", "16/64")["others"], ["12G32G", "24G64G"])
        self.assertEqual(self.pick("qwen38", "24/64")["quant"], "24G64G")
        self.assertEqual(self.pick("qwen38", "32/128")["quant"], "24G64G")

    def test_v4(self):
        r = self.pick("v4", "12/32")
        self.assertEqual((r["quant"], r["fits"]), ("16G64G", False))   # the smallest, and why
        self.assertEqual(self.pick("v4", "16/64")["quant"], "16G64G")
        self.assertEqual(self.pick("v4", "24/64")["quant"], "16G64G")
        self.assertEqual(self.pick("v4", "32/96")["quant"], "32G96G")
        self.assertEqual(self.pick("v4", "32/128")["quant"], "32G96G")

    def test_glm(self):
        self.assertEqual(self.pick("glm", "16/64"), {"quant": "32G96G", "fits": False, "others": ["32G128G"],
                                                     "extra": None})
        self.assertEqual(self.pick("glm", "32/96")["quant"], "32G96G")
        self.assertEqual(self.pick("glm", "32/128")["quant"], "32G128G")
        self.assertEqual(self.pick("glm", "32/123")["quant"], "32G128G")   # 128 GB with an iGPU's carve-out

    def test_ornith_keeps_the_122b_and_names_the_397b(self):
        r = self.pick("ornith", "12/32")
        self.assertEqual((r["quant"], r["fits"], r["extra"]), ("IQ3_XXS-B", False, None))
        r = self.pick("ornith", "16/64")
        self.assertEqual((r["quant"], r["fits"], r["extra"]), ("IQ3_XXS-B", True, "16G64G"))
        self.assertEqual(r["others"], ["24G64G"])                         # the 397B ones are not among them
        self.assertEqual(self.pick("ornith", "24/64")["quant"], "24G64G")
        r = self.pick("ornith", "32/128")
        self.assertEqual((r["quant"], r["extra"]), ("24G64G", "32G128G"))

    def test_models_without_pc_sizes(self):
        for key in ("v41", "ornith35", "qwen35", "qwen38-27b", "qwen35-9b-qsa", "ornith-9b-qsa", "mimo-9b",
                    "sensenova-loop-9b", "mimo-9b-qsa"):
            self.assertIsNone(A.fit_quant(key, 32, 96))

    def test_no_gpu(self):
        vram, ram, shared = A.pc_memory([], 62.5)
        self.assertEqual((vram, shared), (0.0, False))
        r = A.fit_quant("qwen38", vram, ram, shared)
        self.assertEqual((r["quant"], r["fits"]), ("12G32G", False))

    def test_apu_counts_the_memory_once(self):
        # a Ryzen AI Max+ 395 with 128 GB: a 512 MB carve-out (~125 GB to the OS) or 96 GB of it (~31 GB to the OS)
        for vram_gb, ram in ((0.5, 125.0), (96.0, 31.0)):
            g = {"index": 0, "name": "AMD Radeon 8060S Graphics", "arch": "gfx1151", "vram_gb": vram_gb,
                 "vendor": "amd", "driver": "amdgpu"}
            vram, total, shared = A.pc_memory([g], ram)
            self.assertEqual((vram, shared), (0.0, True))
            self.assertAlmostEqual(total, ram + vram_gb)
            self.assertEqual(A.fit_quant("qwen38", vram, total, shared)["quant"], "24G64G")
            self.assertEqual(A.fit_quant("v4", vram, total, shared)["quant"], "32G96G")
            self.assertEqual(A.fit_quant("glm", vram, total, shared)["quant"], "32G96G")
            r = A.fit_quant("ornith", vram, total, shared)
            self.assertEqual((r["quant"], r["extra"]), ("24G64G", "16G64G"))
        # DGX Spark: its "VRAM" is the RAM itself
        spark = {"index": 0, "name": "NVIDIA GB10", "arch": "121", "vram_gb": 119.7, "integrated": True}
        self.assertEqual(A.pc_memory([spark], 119.7), (0.0, 119.7, True))

    def test_split_cards_add_up(self):
        cards = [{"index": 0, "name": "RTX 3090", "vram_gb": 23.99}, {"index": 1, "name": "RTX 3090", "vram_gb": 23.99}]
        vram, _, shared = A.pc_memory(cards, 62.5)
        self.assertEqual((round(vram), shared), (48, False))

    def choose(self, key, quant=None, yes=True, name=None, have=(), ram=62.5, vram=15.92):
        a = Namespace(quant=quant, yes=yes, name=name)
        said = []
        with tempfile.TemporaryDirectory() as d, mock.patch.object(A, "ROOT", Path(d)), \
                mock.patch.object(A.S, "ram_gb", return_value=ram), \
                mock.patch.object(A.S, "say", side_effect=lambda *x: said.append(" ".join(map(str, x)))):
            for q in have:
                (Path(d) / f"abyss-{A.MODELS[key]['prefix']}-{q.lower()}.json").write_text("{}", encoding="utf-8")
            got = A.choose_quant(a, key, [{"index": 0, "name": "GPU", "vram_gb": vram}])
        return got, "\n".join(said)

    def test_choose_takes_the_fit_with_yes(self):
        got, said = self.choose("qwen38")
        self.assertEqual(got, "16G64G")
        self.assertIn("this PC: 16 GB GPU + 62 GB RAM -> --quant 16G64G (the others: 12G32G, 24G64G)", said)
        got, said = self.choose("ornith")
        self.assertEqual(got, "IQ3_XXS-B")
        self.assertIn("Ornith-1.5-397B fits this PC too: --quant 16G64G", said)

    def test_explicit_quant_wins(self):
        self.assertEqual(self.choose("qwen38", quant="12G32G"), ("12G32G", ""))

    def test_v4_keeps_its_default_and_names_the_fit(self):
        got, said = self.choose("v4", ram=93.9, vram=31.84)
        self.assertEqual(got, "UD-Q2_K_XL")
        self.assertIn("--quant 32G96G is the file made for this size", said)

    def test_a_file_set_up_before_is_kept(self):
        got, said = self.choose("qwen38", have=("16G64G",), vram=23.99)
        self.assertEqual(got, "16G64G")
        self.assertIn("already set up here: 16G64G", said)
        self.assertEqual(self.choose("qwen38", have=("16G64G",), vram=23.99, name="mine")[0], "24G64G")

    def test_asks_with_the_fit_as_default(self):
        with mock.patch("builtins.input", return_value=""):
            self.assertEqual(self.choose("qwen38", yes=False, vram=23.99)[0], "24G64G")
        with mock.patch("builtins.input", return_value="12g32g"):
            self.assertEqual(self.choose("qwen38", yes=False)[0], "12G32G")


class DSparkStep(unittest.TestCase):
    """dspark_step against a config in a temporary folder, the drafter 'downloaded' from a local file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.data = d / "Abyss-data"
        self.cfg = d / "abyss-dsv4-test.json"
        self.cfg.write_text(json.dumps({"exe": "strata", "env": {"GPU_MAX_HW_QUEUES": "2"}}), encoding="utf-8")
        src = d / "src" / "drafter.gguf"
        src.parent.mkdir()
        src.write_bytes(b"GGUF drafter" * 100)
        sha = hashlib.sha256(src.read_bytes()).hexdigest()
        self.patches = [mock.patch.object(A, "DSPARK_URL", str(src.parent) + "/{file}"),
                        mock.patch.object(A, "DSPARK_FILE", ("drafter.gguf", src.stat().st_size, sha))]
        for pt in self.patches:
            pt.start()
        self.gpu = [{"index": 0, "vram_gb": 32.0}]
        self.dst = self.data / "models" / A.DSPARK_REPO / "drafter.gguf"

    def tearDown(self):
        for pt in self.patches:
            pt.stop()
        self.tmp.cleanup()

    def run_step(self, fresh=False, had=None, gpus=None, **kw):
        a = Namespace(dspark=kw.get("dspark", False), no_dspark=kw.get("no_dspark", False), yes=True)
        A.dspark_step(a, self.data, self.cfg, fresh, had, gpus or self.gpu, "hip")
        return json.loads(self.cfg.read_text(encoding="utf-8"))["env"]

    def test_a_new_config_asks_and_the_default_is_no(self):
        env = self.run_step(fresh=True)
        self.assertNotIn("STRATA_DS4_DSPARK", env)
        self.assertFalse(self.dst.exists())

    def test_dspark_downloads_checks_and_turns_it_on(self):
        env = self.run_step(dspark=True)
        self.assertEqual(env["STRATA_DS4_DSPARK"], str(self.dst))
        self.assertEqual(env["GPU_MAX_HW_QUEUES"], "2")   # the rest of the env stays
        self.assertTrue(self.dst.exists())
        self.assertTrue(self.dst.with_name("drafter.gguf.sha256").exists())

    def test_a_config_that_has_it_keeps_it_and_no_dspark_takes_it_out(self):
        self.run_step(dspark=True)
        self.assertIn("STRATA_DS4_DSPARK", self.run_step())
        env = self.run_step(no_dspark=True)
        self.assertNotIn("STRATA_DS4_DSPARK", env)
        self.assertTrue(self.dst.exists())   # the file stays

    def test_rewrite_config_keeps_the_drafter_it_had(self):
        self.run_step(dspark=True)
        self.cfg.write_text(json.dumps({"exe": "strata", "env": {}}), encoding="utf-8")   # written again
        env = self.run_step(fresh=True, had=str(self.dst))
        self.assertEqual(env["STRATA_DS4_DSPARK"], str(self.dst))

    def test_not_with_a_layer_split(self):
        env = self.run_step(dspark=True, gpus=self.gpu * 2)
        self.assertNotIn("STRATA_DS4_DSPARK", env)
        self.assertFalse(self.dst.exists())

    def test_a_wrong_file_stops(self):
        with mock.patch.object(A, "DSPARK_FILE", ("drafter.gguf", 1200, "0" * 64)):
            with self.assertRaises(SystemExit):
                self.run_step(dspark=True)
        self.assertFalse(self.dst.with_name("drafter.gguf.sha256").exists())


class InkDSparkStep(unittest.TestCase):
    """ink_dspark_step (Inkling 2 Mini's DSpark drafter) against a config in a temporary folder: nothing downloaded;
    the engine finds a file on disk itself (the config does not name it), --no-dspark writes STRATA_INK_DSPARK=0, a
    config that names the file setup found loses the line, one that names another file keeps it; CUDA, one GPU."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.mdir = d / "Abyss-data" / "models" / "unsloth" / "Inkling-2-Mini-GGUF" / "UD-Q4_K_XL"
        self.mdir.mkdir(parents=True)
        self.cfg = d / "abyss-inkling2mini-q4_k_xl.json"
        self.cfg.write_text(json.dumps({"exe": "abyss", "env": {"STRATA_LANG": "ja-en"}}), encoding="utf-8")
        self.gpu = [{"index": 0, "vram_gb": 32.0}]
        self.dst = self.mdir.parent / "dspark" / A.INK_DSPARK_NAME

    def tearDown(self):
        self.tmp.cleanup()

    def run_step(self, gpus=None, backend="cuda", **kw):
        a = Namespace(dspark=False, no_dspark=kw.get("no_dspark", False), yes=True)
        with mock.patch.object(A.S, "download", side_effect=AssertionError("no download")):
            A.ink_dspark_step(a, self.mdir, self.cfg, gpus or self.gpu, backend)
        return json.loads(self.cfg.read_text(encoding="utf-8"))["env"]

    def name(self, v):
        conf = json.loads(self.cfg.read_text(encoding="utf-8"))
        conf["env"]["STRATA_INK_DSPARK"] = v
        self.cfg.write_text(json.dumps(conf), encoding="utf-8")

    def put(self):
        self.dst.parent.mkdir(parents=True, exist_ok=True)
        self.dst.write_bytes(b"GGUF")

    def test_off_without_the_file(self):
        self.assertNotIn("STRATA_INK_DSPARK", self.run_step())

    def test_on_when_the_file_is_there_the_engine_finds_it(self):
        self.put()
        env = self.run_step()
        self.assertNotIn("STRATA_INK_DSPARK", env)   # (unset: the engine's auto mode)
        self.assertEqual(env["STRATA_LANG"], "ja-en")   # the rest of the env stays
        self.assertEqual(A.ink_dspark_found(self.mdir), self.dst)

    def test_the_file_beside_the_model_folder(self):
        f = self.mdir.parent / A.INK_DSPARK_NAME
        f.write_bytes(b"GGUF")
        self.assertEqual(A.ink_dspark_found(self.mdir), f)
        self.assertNotIn("STRATA_INK_DSPARK", self.run_step())

    def test_the_abliterated_model_uses_unsloths_drafter(self):
        # (--model inkling2mini-abl: orcarouter's repo has no drafter; unsloth's folder beside it serves it)
        self.put()
        self.mdir = self.mdir.parents[2] / "orcarouter" / "Inkling-2-Mini-abliterated-GGUF" / "Mix-16G"
        self.mdir.mkdir(parents=True)
        self.assertEqual(A.ink_dspark_found(self.mdir), self.dst)
        self.assertNotIn("STRATA_INK_DSPARK", self.run_step())
        m = A.MODELS["inkling2mini-abl"]
        self.assertEqual(m["prefix_tmpl"].format(q=m["quant"]), "Inkling-2-Mini-abliterated-Mix-16G.")
        self.assertIn("Mix-12G", m["quants"])

    def test_an_old_config_naming_the_found_file_loses_the_line(self):
        # (v1.3.195-v1.3.199 setups wrote the path: loaded whatever the VRAM; now the engine's check applies)
        self.put()
        self.name(str(self.dst))
        self.assertNotIn("STRATA_INK_DSPARK", self.run_step())

    def test_a_config_that_names_another_file_keeps_it_and_no_dspark_writes_0(self):
        self.name("/elsewhere/d.gguf")
        self.put()
        self.assertEqual(self.run_step()["STRATA_INK_DSPARK"], "/elsewhere/d.gguf")
        self.assertEqual(self.run_step(no_dspark=True)["STRATA_INK_DSPARK"], "0")
        self.assertTrue(self.dst.exists())   # the file stays
        self.assertEqual(self.run_step(no_dspark=True)["STRATA_INK_DSPARK"], "0")
        self.assertEqual(self.run_step()["STRATA_INK_DSPARK"], "0")   # (off stays off)

    def test_not_with_a_layer_split_or_amd(self):
        self.put()
        self.name(str(self.dst))
        self.assertNotIn("STRATA_INK_DSPARK", self.run_step(gpus=self.gpu * 2))
        self.name("/elsewhere/d.gguf")
        self.assertNotIn("STRATA_INK_DSPARK", self.run_step(backend="hip"))

    def drafter(self, f: Path, names) -> Path:
        import numpy as np
        import gguf_io
        f.parent.mkdir(parents=True, exist_ok=True)
        w = gguf_io.Writer(f, arch="dflash")
        for n in names:
            w.add_tensor(n, [4], "F32", np.zeros(4, np.float32))
        return w.write()

    def test_the_old_file_wins_and_the_newer_layout_is_named(self):
        # (unsloth, 2026-11-23: dspark-Inkling-2-Mini-Q8_0.gguf became the newer dflash layout, the old one _Old)
        new = self.drafter(self.dst, ["markov_w1.weight", "markov_w2.weight", "markov_w2.bias", "mask_embd.weight"])
        self.assertTrue(A.ink_dspark_new_layout(new))
        with mock.patch.object(A.S, "warn") as warn:
            self.assertNotIn("STRATA_INK_DSPARK", self.run_step())
        self.assertIn(A.INK_DSPARK_OLD, warn.call_args[0][0])
        old = self.drafter(self.dst.with_name(A.INK_DSPARK_OLD), ["markov_w1.weight", "dspark_fuse.weight"])
        self.assertFalse(A.ink_dspark_new_layout(old))
        self.assertEqual(A.ink_dspark_found(self.mdir), old)
        with mock.patch.object(A.S, "warn") as warn:
            self.assertNotIn("STRATA_INK_DSPARK", self.run_step())
        warn.assert_not_called()


class DFlash27Step(unittest.TestCase):
    """dflash27_step (Qwen3.8-27B's DFlash 2 drafter) against a config in a temporary folder, the drafter 'downloaded'
    from a local file: on by default where it fits, --no-dflash takes it out, --dflash sets it up where it does not."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.data = d / "Abyss-data"
        self.cfg = d / "abyss-qwen38-27b-test.json"
        self.cfg.write_text(json.dumps({"exe": "abyss", "env": {"STRATA_LANG": "ja-en"}}), encoding="utf-8")
        src = d / "src" / "drafter27.gguf"
        src.parent.mkdir()
        src.write_bytes(b"GGUF dflash" * 100)
        sha = hashlib.sha256(src.read_bytes()).hexdigest()
        self.patches = [mock.patch.object(A, "DSPARK_URL", str(src.parent) + "/{file}"),
                        mock.patch.object(A, "DFLASH27_FILE", ("drafter27.gguf", src.stat().st_size, sha))]
        for pt in self.patches:
            pt.start()
        self.gpu = [{"index": 0, "vram_gb": 32.0}]
        self.dst = self.data / "models" / A.DFLASH27_REPO / "drafter27.gguf"

    def tearDown(self):
        for pt in self.patches:
            pt.stop()
        self.tmp.cleanup()

    def run_step(self, fits=True, gpus=None, **kw):
        a = Namespace(dflash=kw.get("dflash", False), no_dflash=kw.get("no_dflash", False), yes=True)
        A.dflash27_step(a, self.data, self.cfg, gpus or self.gpu, fits)
        return json.loads(self.cfg.read_text(encoding="utf-8"))["env"]

    def test_on_by_default_where_it_fits(self):
        env = self.run_step()
        self.assertEqual(env["STRATA_Q35_DFLASH"], str(self.dst))
        self.assertEqual(env["STRATA_LANG"], "ja-en")   # the rest of the env stays
        self.assertTrue(self.dst.with_name("drafter27.gguf.sha256").exists())

    def test_off_where_it_does_not_fit_unless_dflash(self):
        self.assertNotIn("STRATA_Q35_DFLASH", self.run_step(fits=False))
        self.assertFalse(self.dst.exists())
        self.assertEqual(self.run_step(fits=False, dflash=True)["STRATA_Q35_DFLASH"], str(self.dst))

    def test_no_dflash_skips_it_and_takes_it_out(self):
        self.assertNotIn("STRATA_Q35_DFLASH", self.run_step(no_dflash=True))
        self.assertFalse(self.dst.exists())
        self.run_step()
        env = self.run_step(no_dflash=True)
        self.assertNotIn("STRATA_Q35_DFLASH", env)
        self.assertTrue(self.dst.exists())   # the file stays

    def test_not_with_a_layer_split(self):
        env = self.run_step(gpus=self.gpu * 2)
        self.assertNotIn("STRATA_Q35_DFLASH", env)
        self.assertFalse(self.dst.exists())

    def test_a_wrong_file_stops(self):
        with mock.patch.object(A, "DFLASH27_FILE", ("drafter27.gguf", 1100, "0" * 64)):
            with self.assertRaises(SystemExit):
                self.run_step()
        self.assertFalse(self.dst.with_name("drafter27.gguf.sha256").exists())

    PINNED = A.DFLASH27_FILE   # (read before setUp patches it)

    def test_the_pinned_file(self):
        name, size, sha = self.PINNED
        self.assertEqual((name, size), ("Qwen3.8-27B-DFlash2-Q4_K_M.gguf", 1143006816))
        self.assertTrue(sha.startswith("1a25c568") and len(sha) == 64)


class ToolkitSwitch(unittest.TestCase):
    """Setup again after a switch to another ROCm / CUDA toolkit (issue #1: ROCm 7.0 -> 10)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "build").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def cache(self, **entries):
        (self.root / "build" / "CMakeCache.txt").write_text(
            "# This is the CMakeCache file.\n" + "".join(f"{k}:STRING={v}\n" for k, v in entries.items()),
            encoding="utf-8")

    def test_fresh_only_when_the_cache_names_another_compiler(self):
        build = self.root / "build"

        def fresh(clang):
            return A.S.cmake_fresh(build, ["cmake", "-DSTRATA_ENABLE_HIP=ON", f"-DCMAKE_HIP_COMPILER={clang}"])
        rocm10 = "/opt/rocm/core-10.0/llvm/bin/clang++"
        self.assertEqual(fresh(rocm10), [])   # no cache yet
        self.cache(CMAKE_HIP_COMPILER="/home/u/rocm_sdk/lib/llvm/bin/clang++")
        self.assertEqual(fresh(rocm10), ["--fresh"])
        self.assertEqual(A.S.cmake_fresh(build, ["cmake", "-DSTRATA_ENABLE_HIP=ON"]), [])   # no compiler given
        self.cache(CMAKE_HIP_COMPILER=rocm10)
        self.assertEqual(fresh(rocm10), [])
        self.assertEqual(fresh("/opt/rocm/core-10.0//llvm/bin/./clang++"), [])   # the same path written otherwise
        self.assertEqual(fresh("/opt/rocm/llvm/bin/clang++"), ["--fresh"])   # (a link: compared as written)
        self.cache(CMAKE_CUDA_COMPILER="/usr/local/cuda-13.0/bin/nvcc", CMAKE_CUDA_HOST_COMPILER="/usr/bin/g++-14")
        self.assertEqual(A.S.cmake_fresh(build, ["-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc",
                                                 "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13"]), ["--fresh"])

    def build(self, nvcc_arg=None):
        gpus = [{"arch": "120", "name": "RTX 5090"}]
        with mock.patch.object(A, "ROOT", self.root), mock.patch.object(A.S, "run") as run:
            A.build_engine(None, gpus, nvcc_arg)
        return [[str(c) for c in call.args[0]] for call in run.call_args_list]

    def test_another_nvcc_configures_build_from_scratch(self):
        (self.root / "build" / "build.ninja").touch()
        nvcc = self.root / "cuda-12.9" / "bin" / "nvcc"
        nvcc.parent.mkdir(parents=True)
        nvcc.touch()
        self.cache(CMAKE_CUDA_COMPILER="/usr/local/cuda-13.0/bin/nvcc", CMAKE_CUDA_ARCHITECTURES="120a")
        conf, build = self.build(str(nvcc))
        self.assertIn("--fresh", conf)
        for want in ("-DSTRATA_ENABLE_CUDA=ON", f"-DCMAKE_CUDA_COMPILER={nvcc}", "-DCMAKE_CUDA_ARCHITECTURES=120a"):
            self.assertIn(want, conf)
        self.assertIn("--build", build)
        # the toolkit build/ has (by --nvcc, or none given): only the build
        self.cache(CMAKE_CUDA_COMPILER=str(nvcc), CMAKE_CUDA_ARCHITECTURES="120a")
        self.assertEqual(len(self.build(str(nvcc))), 1)
        self.cache(CMAKE_CUDA_COMPILER="/usr/local/cuda-13.0/bin/nvcc", CMAKE_CUDA_ARCHITECTURES="120a")
        self.assertEqual(len(self.build()), 1)

    def test_a_kept_config_takes_the_engines_lib_dirs(self):
        cfg = self.root / "abyss-dsv4-r9700.json"
        mine = {"exe": "build-hip/strata", "backend": "hip", "lib_dirs": ["/home/u/rocm_sdk/lib"], "port": 8082,
                "env": {"GPU_MAX_HW_QUEUES": "2", "STRATA_GLM_RAM_GB": "80"}}
        cfg.write_text(json.dumps(mine), encoding="utf-8")
        extra = {"exe": "build-hip/strata", "backend": "hip", "lib_dirs": ["/opt/rocm/core-10.0/lib"], "env": {}}
        with mock.patch.object(A, "ROOT", self.root):
            A.write_config("dsv4-r9700", self.root / "pack", 65536, 8080, False, None, extra)
            got = json.loads(cfg.read_text(encoding="utf-8"))
            self.assertEqual(got, dict(mine, lib_dirs=["/opt/rocm/core-10.0/lib"]))   # the rest is the user's
            # (an engine stamp without lib_dirs: what the config has stays)
            A.write_config("dsv4-r9700", self.root / "pack", 65536, 8080, False, None, dict(extra, lib_dirs=[]))
            self.assertEqual(json.loads(cfg.read_text(encoding="utf-8"))["lib_dirs"], ["/opt/rocm/core-10.0/lib"])


    def test_rewrite_config_keeps_the_server_settings(self):
        # (Maya #1089) the model's part is written again; port, API key, host and the other server keys stay
        cfg = self.root / "abyss-dsv41-132g.json"
        mine = {"exe": "old/strata", "args": ["--glm-pack", "old"], "port": 9001, "host": "0.0.0.0", "api_key": "sk-x",
                "allow_shutdown": False, "mcp_servers": {"fs": {"command": "mcp-fs"}}, "thinking_budget": 4096,
                "env": {"STRATA_GLM_RAM_GB": "80"}}
        cfg.write_text(json.dumps(mine), encoding="utf-8")
        m = A.MODELS["v41"]
        with mock.patch.object(A, "ROOT", self.root):
            _, run = A.write_config("dsv41-132g", self.root / "pack", 65536, None, True, None, {"env": dict(m["env"])}, m)
            got = json.loads(cfg.read_text(encoding="utf-8"))
            for k in ("port", "host", "api_key", "allow_shutdown", "mcp_servers", "thinking_budget"):
                self.assertEqual(got[k], mine[k], k)
            self.assertEqual(got["args"][:2], ["--glm-pack", str(self.root / "pack")])   # the model's part: new
            self.assertNotIn("STRATA_GLM_RAM_GB", got["env"])
            self.assertIn('"--port" "9001"', run.read_text(encoding="utf-8"))
            # a port given on the command line wins; a new config without one: 8082
            A.write_config("dsv41-132g", self.root / "pack", 65536, 8090, True, None, None, m)
            got = json.loads(cfg.read_text(encoding="utf-8"))
            self.assertEqual((got["port"], got["api_key"]), (8090, "sk-x"))
            cfg.unlink()
            A.write_config("dsv41-132g", self.root / "pack", 65536, None, True, None, None, m)
            got = json.loads(cfg.read_text(encoding="utf-8"))
            self.assertEqual(got["port"], 8082)
            self.assertNotIn("api_key", got)

    def test_v41_config_has_its_name_and_settings(self):
        m = A.MODELS["v41"]
        extra = {"env": dict(m["env"], GPU_MAX_HW_QUEUES="2")}
        with mock.patch.object(A, "ROOT", self.root):
            cfg, run = A.write_config("dsv41-132g", self.root / "pack", 65536, 8082, False, None, extra, m)
            got = json.loads(cfg.read_text(encoding="utf-8"))
        self.assertEqual(got["model_name"], "deepseek-v4.1-flash-1017")
        self.assertEqual(got["env"]["STRATA_GLM_TAIL_SKIP"], "4")
        self.assertEqual(got["env"]["GPU_MAX_HW_QUEUES"], "2")
        self.assertIn("DeepSeek V4.1 Flash 1017", run.read_text(encoding="utf-8"))
        self.assertEqual(m["quants"], ("132G", "139G"))

    def test_root_level_repo_files_by_prefix(self):
        items = [{"type": "file", "path": p, "size": 1, "lfs": {"oid": "x"}} for p in (
            "Ornith-1.5-122B-IQ3_XXS-B-00002-of-00002.gguf", "Ornith-1.5-122B-IQ3_XXS-B-00001-of-00002.gguf",
            "imatrix-ornith122b.gguf", "README.md")]
        m = A.MODELS["ornith"]

        class R:
            def read(self):
                return json.dumps(items).encode()
        with mock.patch.object(A.urllib.request, "urlopen", return_value=R()):
            got = A.quant_files(m["quant"], m["repo"], m["folder"], m["prefix_file"])
        self.assertEqual([g[0] for g in got], ["Ornith-1.5-122B-IQ3_XXS-B-00001-of-00002.gguf",
                                               "Ornith-1.5-122B-IQ3_XXS-B-00002-of-00002.gguf"])
        with mock.patch.object(A, "ROOT", self.root):
            cfg, _ = A.write_config("ornith-iq3_xxs-b", self.root / "pack", 65536, 8082, False, None, None, m)
        self.assertEqual(json.loads(cfg.read_text(encoding="utf-8"))["sampling"]["top_k"], 20)



class DenseAllInVram(unittest.TestCase):
    """--model qwen38-27b (Qwen3.8-27B, the whole file in one GPU's VRAM): the context a GPU holds, a stop on a GPU too
    small (before any download or build), one GPU only, unsloth's file picked by its prefix."""
    KEY = "qwen38-27b"

    def check(self, vram, context=65536, gpus=1, backend="cuda", shared=False, ram=96.0):
        a = Namespace(context=context)
        said = []
        cards = [{"index": i, "name": "GPU", "vram_gb": vram} for i in range(gpus)]
        with mock.patch.object(A.S, "say", side_effect=lambda *x: said.append(" ".join(map(str, x)))):
            try:
                got = A.dense_gpu_check(a, self.KEY, cards, backend, shared, ram)
            except SystemExit:
                got = "stopped"
        return got, "\n".join(said)

    def test_the_context_a_gpu_holds(self):
        f = A.dense_context
        self.assertIsNone(f(self.KEY, 15.92, 8192))
        self.assertIsNone(f(self.KEY, 19.99, 8192))   # (8K ran with 19.1 GB free: no room for the desktop)
        self.assertEqual(f(self.KEY, 23.99, 65536), 32768)
        self.assertEqual(f(self.KEY, 23.99, 16384), 16384)
        self.assertEqual(f(self.KEY, 31.84, 65536), 65536)
        self.assertEqual(f(self.KEY, 31.84, 131072), 131072)
        self.assertEqual(f(self.KEY, 31.84, 262144), 131072)
        self.assertEqual(f(self.KEY, 47.99, 262144), 262144)
        self.assertEqual(f(self.KEY, 95.0, 1 << 20), 262144)   # (longer than measured: the longest)

    def test_the_context_a_smaller_gpu_holds_with_the_ffn_in_ram(self):
        f = A.dense_split_context
        self.assertIsNone(f(self.KEY, 7.99, 8192))
        self.assertEqual(f(self.KEY, 11.99, 65536), 8192)
        self.assertEqual(f(self.KEY, 15.92, 65536), 32768)
        self.assertEqual(f(self.KEY, 23.99, 65536), 65536)

    def test_a_small_gpu_stops_with_the_reason(self):
        got, said = self.check(5.99)
        self.assertEqual(got, "stopped")
        self.assertIn("needs a GPU with 8 GB or more", said)
        self.assertIn("--model qwen38", said)

    def test_an_8_gb_gpu_gets_8k_with_the_head_partly_on_the_cpu(self):
        got, said = self.check(7.99)
        self.assertEqual(got, 8192)
        self.assertIn("part of the FFN in RAM", said)

    def test_a_12_16_gb_gpu_runs_with_the_ffn_partly_in_ram(self):
        got, said = self.check(11.99)
        self.assertEqual(got, 8192)
        self.assertIn("part of the FFN in RAM", said)
        got, said = self.check(15.92, context=32768)
        self.assertEqual(got, 32768)
        got, said = self.check(15.92, ram=16.0)
        self.assertEqual(got, "stopped")
        self.assertIn("needs 32 GB of RAM", said)

    def test_a_24_gb_gpu_gets_32k_all_in_vram_or_64k_with_the_ffn_in_ram(self):
        got, said = self.check(23.99, context=32768)
        self.assertEqual(got, 32768)
        self.assertIn("the whole file in VRAM", said)
        got, said = self.check(23.99)   # (64K: all in VRAM needs ~26 GB, so the FFN partly in RAM)
        self.assertEqual(got, 65536)
        self.assertIn("part of the FFN in RAM", said)
        self.assertEqual(self.check(31.84), (65536, self.check(31.84)[1]))

    def test_one_gpu_only(self):
        self.assertEqual(self.check(23.99, gpus=2)[0], "stopped")

    def test_shared_memory_and_amd_go_on_with_a_note(self):
        got, said = self.check(0.0, shared=True)
        self.assertEqual(got, 65536)
        self.assertIn("untested", said)
        got, said = self.check(31.84, backend="hip")
        self.assertEqual(got, 65536)
        self.assertIn("on AMD it is untested", said)

    def test_the_file_and_the_config(self):
        m = A.MODELS[self.KEY]
        self.assertEqual((m["repo"], m["quant"]), ("unsloth/Qwen3.8-27B-GGUF", "UD-Q4_K_XL"))
        items = [{"type": "file", "path": p, "size": 1, "lfs": {"oid": "x"}} for p in (
            "Qwen3.8-27B-UD-Q4_K_XL.gguf", "Qwen3.8-27B-UD-Q4_K_M.gguf", "Qwen3.8-27B-Q4_0.gguf", "mmproj-F16.gguf",
            "imatrix_unsloth.gguf", "README.md")]

        class R:
            def read(self):
                return json.dumps(items).encode()
        with mock.patch.object(A.urllib.request, "urlopen", return_value=R()):
            got = A.quant_files(m["quant"], m["repo"], m["folder"], m["prefix_tmpl"].format(q=m["quant"]))
        self.assertEqual([g[0] for g in got], ["Qwen3.8-27B-UD-Q4_K_XL.gguf"])
        with tempfile.TemporaryDirectory() as d, mock.patch.object(A, "ROOT", Path(d)):
            cfg, run = A.write_config("qwen38-27b-q4_k_xl", Path(d) / "pack", 32768, 8082, False, None, None, m)
            got = json.loads(cfg.read_text(encoding="utf-8"))
            self.assertIn("Qwen3.8-27B", run.read_text(encoding="utf-8"))
        self.assertEqual(got["model_name"], "qwen3.8-27b")
        self.assertEqual(got["args"][got["args"].index("--max-context") + 1], "32768")
        self.assertEqual(got["sampling"]["top_k"], 20)
        self.assertNotIn("env", got)   # (no RAM tier: nothing of it is in RAM)


class DenseOtherFiles(unittest.TestCase):
    """--model qwen38-27b --quant <any file of unsloth's repo>: the VRAM a context needs moves with the file's size, the
    PC-size proposal names the file a GPU holds (the default stays UD-Q4_K_XL), a file without the NextN block gets
    unsloth's separate one (size + SHA-256 checked), an unknown name stops."""
    KEY = "qwen38-27b"

    def test_every_file_of_the_repo_is_offered(self):
        m = A.MODELS[self.KEY]
        self.assertEqual(m["quant"], "UD-Q4_K_XL")
        for q in ("UD-IQ1_S", "UD-IQ2_XXS", "UD-IQ3_XXS", "UD-Q6_K", "Q4_0", "Q4_1", "Q8_0", "UD-Q8_K_XL", "BF16"):
            self.assertIn(q, m["quants"])
        no_nextn = sorted(q for q, (_, nx) in m["quant_gb"].items() if not nx)
        self.assertEqual(no_nextn, ["UD-IQ1_M", "UD-IQ1_S", "UD-IQ2_S", "UD-IQ2_XS", "UD-IQ2_XXS"])

    def test_the_vram_moves_with_the_file(self):
        self.assertAlmostEqual(A.dense_need(self.KEY, 32768, "UD-Q4_K_XL"), 23.5)
        self.assertAlmostEqual(A.dense_need(self.KEY, 32768, "UD-Q6_K"), 23.5 + 21.98 - 17.66)
        self.assertEqual(A.dense_context(self.KEY, 31.84, 65536, "UD-Q6_K"), 65536)
        self.assertEqual(A.dense_context(self.KEY, 31.84, 65536, "Q8_0"), None)   # (29 GB + caches: not all in VRAM)
        self.assertEqual(A.dense_context(self.KEY, 11.99, 8192, "UD-IQ2_XXS"), 8192)

    def test_the_pc_size_proposal(self):
        self.assertEqual(A.dense_pick(self.KEY, 31.84), "UD-Q6_K")
        self.assertEqual(A.dense_pick(self.KEY, 23.99), "UD-Q4_K_XL")
        self.assertEqual(A.dense_pick(self.KEY, 15.92), "UD-IQ3_XXS")
        self.assertEqual(A.dense_pick(self.KEY, 11.99), "UD-IQ2_XXS")
        self.assertIsNone(A.dense_pick(self.KEY, 7.99))
        a = Namespace(quant=None, yes=True, name=None)
        with mock.patch.object(A.S, "ram_gb", return_value=96.0), mock.patch.object(A.S, "say"):
            got = A.choose_quant(a, self.KEY, [{"index": 0, "name": "GPU", "vram_gb": 31.84}])
        self.assertEqual(got, "UD-Q4_K_XL")   # (--yes: the default stays)
        with self.assertRaises(SystemExit), mock.patch.object(A.S, "say"):
            A.choose_quant(Namespace(quant="Q3_K_M", yes=True, name=None), self.KEY, [])

    def test_an_8_gb_gpu_and_a_large_file(self):
        a = Namespace(context=8192)
        cards = [{"index": 0, "name": "GPU", "vram_gb": 7.99}]
        with mock.patch.object(A.S, "say"):
            self.assertEqual(A.dense_gpu_check(a, self.KEY, cards, "cuda", False, 96.0, quant="UD-IQ2_XXS"), 8192)
            with self.assertRaises(SystemExit):
                A.dense_gpu_check(a, self.KEY, cards, "cuda", False, 96.0, quant="Q8_0")

    def test_the_side_files_are_size_and_sha_checked(self):
        body = b"x" * 100
        sha = hashlib.sha256(body).hexdigest()
        items = [{"type": "file", "path": "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "size": 100, "lfs": {"oid": sha}}]

        class R:
            def read(self):
                return json.dumps(items).encode()
        a = Namespace(download_model=True, yes=True)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(A.urllib.request, "urlopen", return_value=R()), \
                mock.patch.object(A.S, "say"), mock.patch.object(A.S, "ok"):
            def good(url, dst, what=None):
                Path(dst).write_bytes(body)
            with mock.patch.object(A.S, "download", side_effect=good) as dl:
                got = A.side_file(a, "unsloth/Qwen3.8-27B-GGUF", A.DENSE27_MTP, Path(d), "block")
                self.assertEqual(got, Path(d) / "MTP" / "mtp-Qwen3.8-27B-Q4_0.gguf")
                self.assertIn("MTP/mtp-Qwen3.8-27B-Q4_0.gguf", dl.call_args[0][0])
            got.unlink()
            for f in Path(d, "MTP").glob("*.sha256"):
                f.unlink()

            def bad(url, dst, what=None):   # (a proxy's other bytes: right size, wrong content)
                Path(dst).write_bytes(b"y" * 100)
            with mock.patch.object(A.S, "download", side_effect=bad), self.assertRaises(SystemExit):
                A.side_file(a, "unsloth/Qwen3.8-27B-GGUF", A.DENSE27_MTP, Path(d), "block")


class Qwen35NineBQsa(unittest.TestCase):
    """--model qwen35-9b-qsa (zephel01's Qwen3.5-9B QSA, the whole file in one GPU): zephel01's repo, Q4_K_M by default,
    each file with its NextN block (no MTP/ side file), the measured 11 GB for every context up to 128K, a stop under it
    (before any download or build), text only."""
    KEY = "qwen35-9b-qsa"

    def check(self, vram, context=65536, quant=None):
        a = Namespace(context=context)
        said = []
        cards = [{"index": 0, "name": "GPU", "vram_gb": vram}]
        with mock.patch.object(A.S, "say", side_effect=lambda *x: said.append(" ".join(map(str, x)))):
            try:
                got = A.dense_gpu_check(a, self.KEY, cards, "cuda", False, 32.0, quant=quant)
            except SystemExit:
                got = "stopped"
        return got, "\n".join(said)

    def test_the_entry(self):
        m = A.MODELS[self.KEY]
        self.assertEqual((m["repo"], m["quant"], m["quants"]), ("zephel01/Qwen3.5-9B-QSA-GGUF", "Q4_K_M", ("Q4_K_M", "Q8_0")))
        self.assertTrue(all(nx for _, nx in m["quant_gb"].values()))   # (no separate NextN block to fetch)
        self.assertNotIn("ffn_gb", m)                                  # (not measured with the FFN in RAM by design)
        self.assertEqual(m["env"], {})                                 # (the engine reads the QSA layers from the file)

    def test_the_context_a_gpu_holds(self):
        f = A.dense_context
        self.assertEqual(f(self.KEY, 11.99, 131072), 131072)          # (a 12 GB card: 128K, measured with an 11 GB cap)
        self.assertEqual(f(self.KEY, 10.75, 65536), 65536)            # (an 11 GB card)
        self.assertIsNone(f(self.KEY, 9.99, 8192))                    # (10 GB: not measured)
        self.assertEqual(f(self.KEY, 31.84, 262144), 131072)          # (longer than measured: 128K)
        self.assertAlmostEqual(A.dense_need(self.KEY, 32768, "Q8_0"), 11.0 + 9.86 - 5.85)
        self.assertEqual(f(self.KEY, 15.92, 131072, "Q8_0"), 131072)
        self.assertIsNone(f(self.KEY, 11.99, 8192, "Q8_0"))

    def test_the_setup_check(self):
        got, said = self.check(11.99)
        self.assertEqual(got, 65536)
        self.assertIn("the whole file in VRAM", said)
        got, said = self.check(11.99, context=131072)
        self.assertEqual(got, 131072)
        got, said = self.check(31.84, context=262144)
        self.assertEqual(got, 131072)
        got, said = self.check(7.99)
        self.assertEqual(got, "stopped")
        self.assertIn("needs a GPU with 11 GB or more", said)
        self.assertNotIn("FFN", said)
        self.assertEqual(self.check(11.99, quant="Q8_0")[0], "stopped")

    def test_the_pc_size_proposal(self):
        self.assertEqual(A.dense_pick(self.KEY, 31.84), "Q8_0")
        self.assertEqual(A.dense_pick(self.KEY, 11.99), "Q4_K_M")
        self.assertIsNone(A.dense_pick(self.KEY, 7.99))
        a = Namespace(quant=None, yes=True, name=None)
        with mock.patch.object(A.S, "ram_gb", return_value=32.0), mock.patch.object(A.S, "say"):
            got = A.choose_quant(a, self.KEY, [{"index": 0, "name": "GPU", "vram_gb": 31.84}])
        self.assertEqual(got, "Q4_K_M")   # (--yes: the default stays)

    def test_the_file_and_the_config(self):
        m = A.MODELS[self.KEY]
        items = [{"type": "file", "path": p, "size": 1, "lfs": {"oid": "x"}} for p in (
            "Qwen3.5-9B-QSA-Q4_K_M.gguf", "Qwen3.5-9B-QSA-Q8_0.gguf", "README.md", ".gitattributes")]

        class R:
            def read(self):
                return json.dumps(items).encode()
        for q in m["quants"]:
            with mock.patch.object(A.urllib.request, "urlopen", return_value=R()):
                got = A.quant_files(q, m["repo"], m["folder"], m["prefix_tmpl"].format(q=q))
            self.assertEqual([g[0] for g in got], [f"Qwen3.5-9B-QSA-{q}.gguf"])
        with tempfile.TemporaryDirectory() as d, mock.patch.object(A, "ROOT", Path(d)):
            cfg, run = A.write_config("qwen35-9b-qsa-q4_k_m", Path(d) / "pack", 131072, 8082, False, None, None, m)
            got = json.loads(cfg.read_text(encoding="utf-8"))
            self.assertIn("Qwen3.5-9B QSA", run.read_text(encoding="utf-8"))
        self.assertEqual(got["model_name"], "qwen3.5-9b-qsa")
        self.assertEqual(got["args"][got["args"].index("--max-context") + 1], "131072")
        self.assertNotIn("env", got)
        self.assertNotIn("vision", got)



class OrnithNineBQsa(Qwen35NineBQsa):
    """--model ornith-9b-qsa (zephel01's Ornith-1.5-9B QSA): every check of Qwen3.5-9B QSA again (same shapes, same VRAM
    table), with Ornith's repo, file names and model name"""
    KEY = "ornith-9b-qsa"

    def test_the_entry(self):
        m = A.MODELS[self.KEY]
        self.assertEqual((m["repo"], m["quant"], m["quants"]), ("zephel01/Ornith-1.5-9B-QSA-GGUF", "Q4_K_M", ("Q4_K_M", "Q8_0")))
        self.assertTrue(all(nx for _, nx in m["quant_gb"].values()))
        self.assertEqual(m["env"], {})
        self.assertEqual(m["vram_ctx"], A.MODELS["qwen35-9b-qsa"]["vram_ctx"])
        self.assertEqual(m["gpu_min"], A.MODELS["qwen35-9b-qsa"]["gpu_min"])
        self.assertEqual(m["sampling"], {"temperature": 0.6, "top_p": 0.95, "top_k": 20})

    def test_the_file_and_the_config(self):
        m = A.MODELS[self.KEY]
        items = [{"type": "file", "path": p, "size": 1, "lfs": {"oid": "x"}} for p in (
            "Ornith-1.5-9B-QSA-Q4_K_M.gguf", "Ornith-1.5-9B-QSA-Q8_0.gguf", "README.md", ".gitattributes")]

        class R:
            def read(self):
                return json.dumps(items).encode()
        for q in m["quants"]:
            with mock.patch.object(A.urllib.request, "urlopen", return_value=R()):
                got = A.quant_files(q, m["repo"], m["folder"], m["prefix_tmpl"].format(q=q))
            self.assertEqual([g[0] for g in got], [f"Ornith-1.5-9B-QSA-{q}.gguf"])
        with tempfile.TemporaryDirectory() as d, mock.patch.object(A, "ROOT", Path(d)):
            cfg, run = A.write_config("ornith-9b-qsa-q4_k_m", Path(d) / "pack", 131072, 8082, False, None, None, m)
            got = json.loads(cfg.read_text(encoding="utf-8"))
            self.assertIn("Ornith-1.5-9B QSA", run.read_text(encoding="utf-8"))
        self.assertEqual(got["model_name"], "ornith-1.5-9b-qsa")
        self.assertEqual(got["args"][got["args"].index("--max-context") + 1], "131072")
        self.assertNotIn("env", got)
        self.assertNotIn("vision", got)


class SensenovaLoopNineB(unittest.TestCase):
    """--model sensenova-loop-9b (guideboardlabs' SenseNova-Loop-9B GGUFs, the whole file in one GPU): its own VRAM table
    (the capped runs: 10 / 12 / 14 GB for 32K / 64K / 128K), four files, the GGUF's sampling, n-gram drafts, text only."""
    KEY = "sensenova-loop-9b"

    def check(self, vram, context=65536, quant=None):
        a = Namespace(context=context)
        said = []
        cards = [{"index": 0, "name": "GPU", "vram_gb": vram}]
        with mock.patch.object(A.S, "say", side_effect=lambda *x: said.append(" ".join(map(str, x)))):
            try:
                got = A.dense_gpu_check(a, self.KEY, cards, "cuda", False, 32.0, quant=quant)
            except SystemExit:
                got = "stopped"
        return got, "\n".join(said)

    def test_the_entry(self):
        m = A.MODELS[self.KEY]
        self.assertEqual((m["repo"], m["quant"], m["quants"]),
                         ("guideboardlabs/SenseNova-Loop-9B-GGUF", "Q4_K_M", ("Q3_K_M", "Q4_0", "Q4_K_M", "Q8_0")))
        self.assertTrue(all(nx for _, nx in m["quant_gb"].values()))   # (no draft block to fetch: none exists)
        self.assertEqual(m["env"], {})
        self.assertEqual(m["sampling"], {"temperature": 1.0, "top_p": 0.95})
        self.assertEqual(m["gpu_min"], 10)
        self.assertNotIn("ffn_gb", m)

    def test_the_context_a_gpu_holds(self):
        f = A.dense_context
        self.assertEqual(f(self.KEY, 9.6, 32768), 32768)              # (a 10 GB card: 32K, measured with a 10 GB cap)
        self.assertEqual(f(self.KEY, 11.99, 131072), 65536)           # (12 GB: 64K)
        self.assertEqual(f(self.KEY, 15.92, 131072), 131072)          # (16 GB: 128K)
        self.assertEqual(f(self.KEY, 31.84, 262144), 131072)          # (longer than measured: 128K)
        self.assertIsNone(f(self.KEY, 7.99, 8192))
        self.assertAlmostEqual(A.dense_need(self.KEY, 131072, "Q8_0"), 14.0 + 9.54 - 5.63)
        self.assertEqual(f(self.KEY, 11.99, 32768, "Q3_K_M"), 32768)

    def test_the_setup_check(self):
        self.assertEqual(self.check(11.99)[0], 65536)
        self.assertEqual(self.check(15.92, context=131072)[0], 131072)
        got, said = self.check(7.99)
        self.assertEqual(got, "stopped")
        self.assertIn("needs a GPU with 10 GB or more", said)

    def test_the_pc_size_proposal(self):
        self.assertEqual(A.dense_pick(self.KEY, 31.84), "Q8_0")
        self.assertEqual(A.dense_pick(self.KEY, 9.6), "Q4_K_M")

    def test_the_file_and_the_config(self):
        m = A.MODELS[self.KEY]
        sizes = {"Q3_K_M": 4642406752, "Q4_0": 5325947232, "Q4_K_M": 5628723552, "Q8_0": 9540092256}
        items = [{"type": "file", "path": f"SenseNova-Loop-9B-{q}.gguf", "size": n, "lfs": {"oid": q.lower()}}
                 for q, n in sizes.items()] + [{"type": "file", "path": p, "size": 1} for p in ("README.md", ".gitattributes")]

        class R:
            def read(self):
                return json.dumps(items).encode()
        for q in m["quants"]:
            with mock.patch.object(A.urllib.request, "urlopen", return_value=R()):
                got = A.quant_files(q, m["repo"], m["folder"], m["prefix_tmpl"].format(q=q))
            self.assertEqual(got, [(f"SenseNova-Loop-9B-{q}.gguf", sizes[q], q.lower())])
            self.assertAlmostEqual(m["quant_gb"][q][0], sizes[q] / 1e9, places=2)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(A, "ROOT", Path(d)):
            cfg, run = A.write_config("sensenova-loop-9b-q4_k_m", Path(d) / "pack", 65536, 8082, False, None, None, m)
            got = json.loads(cfg.read_text(encoding="utf-8"))
            self.assertIn("SenseNova-Loop-9B", run.read_text(encoding="utf-8"))
        self.assertEqual(got["model_name"], "sensenova-loop-9b")
        self.assertEqual(got["sampling"], {"temperature": 1.0, "top_p": 0.95})
        self.assertNotIn("env", got)
        self.assertNotIn("vision", got)


class MimoNineB(Qwen35NineBQsa):
    """--model mimo-9b (bartowski's MiMo-V2.6-Distill-Qwen-9B GGUFs): the checks of Qwen3.5-9B QSA (same shapes, same
    VRAM table) with bartowski's repo and file names, and Qwen3.5-9B's NextN block borrowed into MTP/ - or no draft
    block, without stopping, when its repo does not have it"""
    KEY = "mimo-9b"

    def test_the_entry(self):
        m = A.MODELS[self.KEY]
        self.assertEqual((m["repo"], m["quant"], m["quants"]),
                         ("bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF", "Q4_K_M", ("Q4_K_M", "Q8_0")))
        self.assertEqual(m["env"], {})
        self.assertEqual(m["vram_ctx"], A.MODELS["qwen35-9b-qsa"]["vram_ctx"])
        self.assertEqual(m["gpu_min"], 11)
        self.assertEqual(m["sampling"], {"temperature": 0.6, "top_p": 0.95, "top_k": 20})

    def test_the_context_a_gpu_holds(self):
        f = A.dense_context
        self.assertEqual(f(self.KEY, 11.99, 131072), 131072)
        self.assertIsNone(f(self.KEY, 9.99, 8192))
        self.assertAlmostEqual(A.dense_need(self.KEY, 32768, "Q8_0"), 11.0 + 9.55 - 5.84)
        self.assertEqual(f(self.KEY, 15.92, 131072, "Q8_0"), 131072)
        self.assertIsNone(f(self.KEY, 11.99, 8192, "Q8_0"))

    def test_the_file_and_the_config(self):
        m = A.MODELS[self.KEY]
        items = [{"type": "file", "path": p, "size": 1, "lfs": {"oid": "x"}} for p in (
            "MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf", "MiMo-V2.6-Distill-Qwen-9B-Q4_K_L.gguf",
            "MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf", "MiMo-V2.6-Distill-Qwen-9B-imatrix.gguf",
            "mmproj-MiMo-V2.6-Distill-Qwen-9B-f16.gguf", "README.md")]

        class R:
            def read(self):
                return json.dumps(items).encode()
        for q in m["quants"]:
            with mock.patch.object(A.urllib.request, "urlopen", return_value=R()):
                got = A.quant_files(q, m["repo"], m["folder"], m["prefix_tmpl"].format(q=q))
            self.assertEqual([g[0] for g in got], [f"MiMo-V2.6-Distill-Qwen-9B-{q}.gguf"])
        with tempfile.TemporaryDirectory() as d, mock.patch.object(A, "ROOT", Path(d)):
            cfg, run = A.write_config("mimo-9b-q4_k_m", Path(d) / "pack", 131072, 8082, False, None, None, m)
            got = json.loads(cfg.read_text(encoding="utf-8"))
            self.assertIn("MiMo-V2.6-Distill-Qwen-9B", run.read_text(encoding="utf-8"))
        self.assertEqual(got["model_name"], "mimo-v2.6-distill-qwen-9b")
        self.assertEqual(got["sampling"], {"temperature": 0.6, "top_p": 0.95, "top_k": 20})
        self.assertNotIn("env", got)

    def test_the_borrowed_draft_block(self):
        body = b"block"
        sha = hashlib.sha256(body).hexdigest()
        items = [{"type": "file", "path": A.MIMO9_MTP, "size": len(body), "lfs": {"oid": sha}}]

        class R:
            def read(self):
                return json.dumps(items).encode()
        a = Namespace(download_model=True, yes=True)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(A.S, "say"), mock.patch.object(A.S, "ok"), \
                mock.patch.object(A.urllib.request, "urlopen", return_value=R()), \
                mock.patch.object(A.S, "download", side_effect=lambda url, dst, name: Path(dst).write_bytes(body)) as dl:
            got = A.borrowed_mtp(a, A.MIMO9_MTP_REPO, A.MIMO9_MTP, Path(d), "MiMo")
            self.assertEqual(got, Path(d) / "MTP" / A.MIMO9_MTP)       # (where the engine takes it by itself)
            self.assertIn(A.MIMO9_MTP_REPO, dl.call_args[0][0])
            self.assertEqual(got.read_bytes(), body)

    def test_no_draft_block_does_not_stop(self):
        said = []
        a = Namespace(download_model=True, yes=True)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(A.S, "ok", side_effect=lambda *x: said.append(x[0])), \
                mock.patch.object(A.urllib.request, "urlopen", side_effect=urllib.error.HTTPError("u", 404, "nf", {}, None)):
            got = A.borrowed_mtp(a, A.MIMO9_MTP_REPO, A.MIMO9_MTP, Path(d), "MiMo")
        self.assertIsNone(got)
        self.assertIn("runs without it", said[-1])


class MimoNineBQsa(MimoNineB):
    """--model mimo-9b-qsa (zephel01's MiMo-V2.6-Distill-Qwen-9B QSA: bartowski's files with Qwen3.5-9B QSA's indexer): the
    checks of mimo-9b (the same VRAM table, the same borrowed draft block) with zephel01's repo and file names"""
    KEY = "mimo-9b-qsa"

    def test_the_entry(self):
        m = A.MODELS[self.KEY]
        self.assertEqual((m["repo"], m["quant"], m["quants"]),
                         ("zephel01/MiMo-V2.6-Distill-Qwen-9B-QSA-GGUF", "Q4_K_M", ("Q4_K_M", "Q8_0")))
        self.assertEqual(m["env"], {})                                 # (the engine reads the QSA layers from the file)
        self.assertEqual(m["vram_ctx"], {32768: 11.0, 131072: 12.0})   # (measured capped; 128K at 11 GB is FFN-in-RAM slow)
        self.assertEqual(m["gpu_min"], 11)
        self.assertEqual(m["sampling"], A.MODELS["mimo-9b"]["sampling"])

    def test_the_context_a_gpu_holds(self):
        f = A.dense_context
        self.assertEqual(f(self.KEY, 11.99, 131072), 131072)
        self.assertEqual(f(self.KEY, 10.9, 131072), 32768)           # an 11 GB card: 32K, not 128K
        self.assertIsNone(f(self.KEY, 9.99, 8192))
        self.assertAlmostEqual(A.dense_need(self.KEY, 32768, "Q8_0"), 11.0 + 9.62 - 5.91)
        self.assertEqual(f(self.KEY, 15.92, 131072, "Q8_0"), 131072)
        self.assertIsNone(f(self.KEY, 11.99, 8192, "Q8_0"))

    def test_the_file_and_the_config(self):
        m = A.MODELS[self.KEY]
        items = [{"type": "file", "path": p, "size": 1, "lfs": {"oid": "x"}} for p in (
            "MiMo-V2.6-Distill-Qwen-9B-QSA-Q4_K_M.gguf", "MiMo-V2.6-Distill-Qwen-9B-QSA-Q8_0.gguf", "README.md",
            "LICENSE", "LICENSE-Qwen3.5-9B")]

        class R:
            def read(self):
                return json.dumps(items).encode()
        for q in m["quants"]:
            with mock.patch.object(A.urllib.request, "urlopen", return_value=R()):
                got = A.quant_files(q, m["repo"], m["folder"], m["prefix_tmpl"].format(q=q))
            self.assertEqual([g[0] for g in got], [f"MiMo-V2.6-Distill-Qwen-9B-QSA-{q}.gguf"])
        with tempfile.TemporaryDirectory() as d, mock.patch.object(A, "ROOT", Path(d)):
            cfg, run = A.write_config("mimo-9b-qsa-q4_k_m", Path(d) / "pack", 131072, 8082, False, None, None, m)
            got = json.loads(cfg.read_text(encoding="utf-8"))
            self.assertIn("MiMo-V2.6-Distill-Qwen-9B QSA", run.read_text(encoding="utf-8"))
        self.assertEqual(got["model_name"], "mimo-v2.6-distill-qwen-9b-qsa")
        self.assertNotIn("env", got)


class TestVariants(unittest.TestCase):
    """--model qwen38 / glm / ornith: every quant the setup offers names a variant with a repo and a file prefix"""

    def test_variants_cover_the_quants(self):
        for name, m in A.MODELS.items():
            if "variants" not in m:
                continue
            if m["quants"] is not None:   # (v4: any folder of unsloth's repo, or one of zephel01's sizes)
                self.assertEqual(set(m["quants"]), set(m["variants"]), name)
                self.assertIn(m["quant"], m["variants"], name)
            for q, v in m["variants"].items():
                full = {**m, **v}
                self.assertTrue(full.get("repo", "").startswith("zephel01/"), (name, q))
                self.assertTrue(full.get("prefix_file"), (name, q))
                self.assertGreater(full.get("gb", 0), 0, (name, q))


class LingAir(unittest.TestCase):
    """--model ling32air: bartowski's repo, a quant's folder of shards (Q4_K_M, the default) or a top-level file (the
    small ones) resolved the way main() asks quant_files; the config's sampling the model card's (1.0 / 0.95 / 20)."""

    def files(self, q, listing):
        m = A.MODELS["ling32air"]
        m = {**m, "prefix_file": m["prefix_tmpl"].format(q=q)}
        folder = (m.get("quant_folders") or {}).get(q, m.get("folder"))
        prefix = m["prefix_file"] if q not in m["quant_folders"] else m["prefix_file"].rstrip(".") + "-"
        seen = []

        class R:
            def read(self):
                return json.dumps([{"type": "file", "path": p, "size": 1, "lfs": {"oid": "x"}} for p in listing]).encode()

        def urlopen(req, timeout=0):
            seen.append(req.full_url)
            return R()
        with mock.patch.object(A.urllib.request, "urlopen", side_effect=urlopen):
            got = A.quant_files(q, m["repo"], folder, prefix)
        return [g[0] for g in got], seen[0]

    def test_the_default_shards(self):
        m = A.MODELS["ling32air"]
        self.assertEqual((m["repo"], m["quant"]), ("bartowski/Ling-3.2-air-GGUF", "Q4_K_M"))
        shards = ["Ling-3.2-air-Q4_K_M/Ling-3.2-air-Q4_K_M-0000%d-of-00003.gguf" % i for i in (1, 2, 3)]
        got, url = self.files("Q4_K_M", shards)
        self.assertEqual(got, shards)
        self.assertTrue(url.endswith("/tree/main/Ling-3.2-air-Q4_K_M"), url)

    def test_a_top_level_file(self):
        got, url = self.files("IQ2_XXS", ["Ling-3.2-air-IQ2_XXS.gguf", "Ling-3.2-air-IQ2_XS.gguf",
                                          "Ling-3.2-air-imatrix.gguf", "mmproj-Ling-3.2-air-f16.gguf"])
        self.assertEqual(got, ["Ling-3.2-air-IQ2_XXS.gguf"])

    def test_the_config(self):
        m = A.MODELS["ling32air"]
        with tempfile.TemporaryDirectory() as d, mock.patch.object(A, "ROOT", Path(d)):
            cfg, _ = A.write_config("ling32air-q4_k_m", Path(d) / "pack", 65536, 8082, False, None, None, m)
            got = json.loads(cfg.read_text(encoding="utf-8"))
        self.assertEqual(got["model_name"], "ling-3.2-air")
        self.assertEqual((got["sampling"]["temperature"], got["sampling"]["top_p"], got["sampling"]["top_k"]),
                         (1.0, 0.95, 20))


class VenvLayout(unittest.TestCase):
    """The .venv's programs where Windows puts them (issue #39): Scripts/<name>.exe, bin/<name> elsewhere."""

    def test_windows_and_posix_paths(self):
        with mock.patch.object(A, "ROOT", Path("/abyss")):
            self.assertEqual(A.venv_bin("cmake", windows=True), Path("/abyss/.venv/Scripts/cmake.exe"))
            self.assertEqual(A.venv_bin("python", windows=False), Path("/abyss/.venv/bin/python"))

    def test_build_engine_uses_the_windows_layout(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(A, "ROOT", Path(d)), \
                mock.patch.object(A, "WIN", True), mock.patch.object(A.S, "run") as run, \
                mock.patch.object(A, "configured_archs", return_value=None):
            (Path(d) / "build").mkdir()
            nvcc = Path(d) / "nvcc"
            nvcc.touch()
            A.build_engine(None, [{"arch": "89", "name": "RTX 4090"}], str(nvcc))
            calls = [[str(c) for c in call.args[0]] for call in run.call_args_list]
        scripts = str(Path(d) / ".venv" / "Scripts")
        self.assertEqual(calls[0][0], str(Path(scripts) / "cmake.exe"))
        self.assertIn(f"-DCMAKE_MAKE_PROGRAM={Path(scripts) / 'ninja.exe'}", calls[0])
        self.assertEqual(calls[-1][0], str(Path(scripts) / "cmake.exe"))


class LinkMirror(unittest.TestCase):
    """#56: a plain mirror of the Hugging Face repo (a quant's file at the repo's top level) is linked into setup's
    quant folder instead of stopping at "the model is not downloaded"."""

    def test_links_a_flat_mirror(self):
        with tempfile.TemporaryDirectory() as d:
            flat, mdir = Path(d) / "unsloth" / "Inkling-2-Mini-GGUF", Path(d) / "unsloth" / "Inkling-2-Mini-GGUF" / "UD-Q4_K_XL"
            flat.mkdir(parents=True)
            (flat / "Inkling-2-Mini-UD-Q4_K_XL.gguf").write_bytes(b"GGUF")
            self.assertTrue(A.link_mirror(flat, mdir, ["Inkling-2-Mini-UD-Q4_K_XL.gguf"]))
            link = mdir / "Inkling-2-Mini-UD-Q4_K_XL.gguf"
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.read_bytes(), b"GGUF")
            self.assertTrue((flat / "Inkling-2-Mini-UD-Q4_K_XL.gguf").is_file())   # (the mirror is left as it is)

    def test_links_nothing_when_a_file_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            flat, mdir = Path(d) / "r", Path(d) / "r" / "Q4_K_M"
            (flat / "Q4_K_M").mkdir(parents=True)
            (flat / "Q4_K_M" / "a-00001-of-00002.gguf").write_bytes(b"1")
            self.assertFalse(A.link_mirror(flat, mdir, ["Q4_K_M/a-00001-of-00002.gguf", "Q4_K_M/a-00002-of-00002.gguf"]))
            self.assertFalse((mdir / "a-00002-of-00002.gguf").exists())
            self.assertFalse((mdir / "a-00001-of-00002.gguf").is_symlink())


if __name__ == "__main__":
    unittest.main()
