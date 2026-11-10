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
