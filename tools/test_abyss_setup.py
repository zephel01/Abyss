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
        for key in ("v41", "ornith35", "qwen35", "qwen38-27b"):
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
        got, said = self.check(7.99)
        self.assertEqual(got, "stopped")
        self.assertIn("needs a GPU with 12 GB or more", said)
        self.assertIn("--model qwen38", said)

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


if __name__ == "__main__":
    unittest.main()
