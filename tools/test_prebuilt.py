"""tools/prebuilt.py and the setups' prebuilt path: the archive for this OS, the driver's verdict (below 580 stop,
580-609 warn, 610+ ok), SHA256SUMS, the release looked up (this version's, else the latest), the archive checked and
unpacked into build/ (a wrong SHA-256 or an unsafe path refused), and when the setups take it (no nvcc / compiler or
--prebuilt; a toolkit that is there keeps compiling).  No GPU, no network: the "release" is a local folder.

    python -m unittest tools.test_prebuilt
"""
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prebuilt as P  # noqa: E402


def make_tar(path: Path, files: dict, top="abyss-engine-cuda-linux-x64"):
    with tarfile.open(path, "w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}" if top else name)
            info.size = len(data)
            info.mode = 0o644
            t.addfile(info, io.BytesIO(data))


def release(d: Path, name: str, files: dict, sha=None) -> Path:
    rel = d / "release"
    rel.mkdir(exist_ok=True)
    if name.endswith(".zip"):
        with zipfile.ZipFile(rel / name, "w") as z:
            for n, data in files.items():
                z.writestr(f"abyss-engine-cuda-windows-x64/{n}", data)
    else:
        make_tar(rel / name, files)
    real = hashlib.sha256((rel / name).read_bytes()).hexdigest()
    (rel / P.SUMS).write_text(f"{sha or real}  {name}\n", encoding="utf-8")
    return rel


class Pieces(unittest.TestCase):
    def test_asset_names(self):
        self.assertEqual(P.asset_name("linux", "x86_64"), "abyss-engine-cuda-linux-x64.tar.gz")
        self.assertEqual(P.asset_name("windows", "AMD64"), "abyss-engine-cuda-windows-x64.zip")
        self.assertIsNone(P.asset_name("linux", "aarch64"))
        self.assertIsNone(P.asset_name("darwin", "x86_64"))

    def test_driver_verdict(self):
        self.assertEqual(P.driver_verdict("579.99")[0], "stop")
        self.assertEqual(P.driver_verdict("575.64.05")[0], "stop")
        self.assertEqual(P.driver_verdict("580.65.06")[0], "warn")
        self.assertEqual(P.driver_verdict("609.10")[0], "warn")
        self.assertEqual(P.driver_verdict("610.43.02")[0], "ok")
        self.assertEqual(P.driver_verdict("615.71.10")[0], "ok")
        self.assertEqual(P.driver_verdict(None)[0], "stop")
        stop = P.driver_verdict("570.1")[1]
        self.assertIn("580", stop)
        self.assertIn(P.DRIVER_PAGE, stop)
        self.assertIn("untested", P.driver_verdict("590.1")[1])

    def test_driver_version_takes_the_oldest(self):
        self.assertEqual(P.driver_version("615.71.10\n580.65.06\n"), "580.65.06")
        self.assertEqual(P.driver_version("610.43.02\n"), "610.43.02")
        self.assertIsNone(P.driver_version("NVIDIA-SMI has failed\n"))

    def test_sums_and_archs(self):
        a, b = "a" * 64, "B" * 64
        self.assertEqual(P.parse_sums(f"{a}  x.tar.gz\n{b} *y.zip\nnoise\n"), {"x.tar.gz": a, "y.zip": b.lower()})
        self.assertEqual(P.unsupported([86, 89, 120]), [])
        self.assertEqual(P.unsupported(["75", "120a", "121a"]), [75, 121])

    def test_want_prebuilt(self):
        self.assertFalse(P.want_prebuilt(False, "/usr/local/cuda/bin/nvcc", True))
        self.assertTrue(P.want_prebuilt(True, "/usr/local/cuda/bin/nvcc", True))
        self.assertTrue(P.want_prebuilt(False, None, True))
        self.assertTrue(P.want_prebuilt(False, "/usr/local/cuda/bin/nvcc", False))

    def test_release_lookup_falls_back_to_the_latest(self):
        name = "abyss-engine-cuda-linux-x64.tar.gz"
        seen = []

        def get(url):
            seen.append(url)
            if "/download/v9.9.9/" in url:
                raise OSError("404")
            return f"{'c' * 64}  {name}\n".encode()
        with mock.patch.dict(os.environ, {P.ENV_URL: ""}):
            base, sha = P.find_release(name, "9.9.9", get=get)
        self.assertEqual(base, "https://github.com/zephel01/Abyss/releases/latest/download")
        self.assertEqual(sha, "c" * 64)
        self.assertIn("v9.9.9", seen[0])


class Install(unittest.TestCase):
    NAME = "abyss-engine-cuda-linux-x64.tar.gz"
    FILES = {"strata": b"\x7fELF engine", "abyss_gpu_tune": b"tune", "libcublas.so.13": b"lib",
             "PREBUILT.json": json.dumps({"version": "1.3.999", "archs": [86, 89, 120], "cuda": "13.3"}).encode()}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_install(self, rel: Path, name=None):
        with mock.patch.dict(os.environ, {P.ENV_URL: str(rel)}), \
                mock.patch.object(P, "asset_name", return_value=name or self.NAME):
            return P.install(self.d / "build", ver="1.3.999", log=lambda *x: None)

    def test_unpacked_where_the_build_would_be(self):
        build = self.d / "build"
        build.mkdir()
        (build / "strata").write_bytes(b"old engine")
        man = self.run_install(release(self.d, self.NAME, self.FILES))
        self.assertEqual((build / "strata").read_bytes(), b"\x7fELF engine")
        self.assertEqual((build / "libcublas.so.13").read_bytes(), b"lib")
        if os.name != "nt":
            self.assertTrue(os.access(build / "strata", os.X_OK))
        self.assertEqual(man["version"], "1.3.999")
        self.assertEqual(len(man["sha256"]), 64)
        self.assertFalse((build / ".prebuilt-download").exists())
        self.assertFalse((build / ".prebuilt-unpack").exists())
        st = P.stamp_fields(man)
        self.assertEqual((st["prebuilt"], st["archs"], st["cuda"]), ("1.3.999", [86, 89, 120], "13.3"))
        # installed for this checkout: not downloaded again, even when the release is an older version
        with mock.patch.object(P, "version", return_value="1.4.0"):
            self.assertTrue(P.current({**st, "prebuilt_for": "1.4.0"}))
            self.assertFalse(P.current({**st, "prebuilt_for": "1.3.999"}))
            self.assertFalse(P.current({"src": "x"}))          # a compiled engine

    def test_windows_zip(self):
        name = "abyss-engine-cuda-windows-x64.zip"
        files = {"strata.exe": b"MZ engine", "cublasLt64_13.dll": b"dll", "PREBUILT.json": b"{}"}
        self.run_install(release(self.d, name, files), name)
        self.assertEqual((self.d / "build" / "strata.exe").read_bytes(), b"MZ engine")

    def test_wrong_sha_is_refused_and_deleted(self):
        rel = release(self.d, self.NAME, self.FILES, sha="0" * 64)
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            self.run_install(rel)
        self.assertFalse((self.d / "build" / "strata").exists())
        self.assertFalse((self.d / "build" / ".prebuilt-download" / self.NAME).exists())

    def test_missing_from_the_release(self):
        rel = self.d / "release"
        rel.mkdir()
        (rel / P.SUMS).write_text(f"{'a' * 64}  something-else.zip\n")
        with self.assertRaisesRegex(RuntimeError, "no release"):
            self.run_install(rel)

    def test_unsafe_paths_refused(self):
        rel = self.d / "release"
        rel.mkdir()
        make_tar(rel / self.NAME, {"../../evil": b"x", "strata": b"e"}, top="")
        (rel / P.SUMS).write_text(hashlib.sha256((rel / self.NAME).read_bytes()).hexdigest() + f"  {self.NAME}\n")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.run_install(rel)
        self.assertFalse((self.d / "evil").exists())

    def test_no_engine_in_it(self):
        with self.assertRaisesRegex(RuntimeError, "has no strata"):
            self.run_install(release(self.d, self.NAME, {"README": b"x"}))


class Setups(unittest.TestCase):
    """When the setups take the prebuilt engine."""
    def test_abyss_py_choice(self):
        import abyss as M
        a = Namespace(prebuilt=False, host_compiler=None)
        with mock.patch.object(M, "compiler_found", return_value=True), \
                mock.patch.object(P, "asset_name", return_value="x.tar.gz"):
            # a toolkit that builds for the GPU: compile (the default)
            self.assertFalse(M.prebuilt_choice(a, [120], "/usr/local/cuda/bin/nvcc", (13, 3), (12, 8), None))
            # none, or one too old for RTX 50: prebuilt
            self.assertTrue(M.prebuilt_choice(a, [120], None, None, (12, 8), None))
            self.assertTrue(M.prebuilt_choice(a, [120], "/usr/bin/nvcc", (12, 0), (12, 8), None))
            # --prebuilt with a toolkit there
            self.assertTrue(M.prebuilt_choice(Namespace(prebuilt=True, host_compiler=None), [86, 89], "/x/nvcc",
                                              (13, 3), (12, 0), None))
            # a GPU the archive has no code for: compiled (the missing toolkit is listed as before) ...
            with mock.patch.object(M, "say"):
                self.assertFalse(M.prebuilt_choice(a, [75], None, None, (12, 0), None))
            # ... and --prebuilt says why
            with self.assertRaises(SystemExit):
                with mock.patch("setup.UI", None), mock.patch("builtins.print"):
                    M.prebuilt_choice(Namespace(prebuilt=True, host_compiler=None), [70], None, None, (12, 0), None)
        with mock.patch.object(M, "compiler_found", return_value=False), \
                mock.patch.object(P, "asset_name", return_value="x.zip"):
            # no Visual Studio / g++: prebuilt even with nvcc
            self.assertTrue(M.prebuilt_choice(a, [86], "/x/nvcc", (13, 3), (12, 0), None))

    def test_abyss_setup_choice(self):
        import abyss_setup as A
        gpus = [{"arch": "120", "name": "RTX 5090", "index": 0}]
        a = Namespace(prebuilt=False, nvcc=None)
        with mock.patch.object(A, "find_nvcc", return_value="/usr/local/cuda/bin/nvcc"), \
                mock.patch.object(A.shutil, "which", return_value="/usr/bin/g++"), \
                mock.patch.object(A.S, "find_vcvars", return_value="vcvars64.bat"):
            self.assertFalse(A.use_prebuilt(a, gpus))
        with mock.patch.object(A, "find_nvcc", return_value=None), \
                mock.patch.object(A.shutil, "which", return_value="/usr/bin/g++"), \
                mock.patch.object(A.S, "find_vcvars", return_value="vcvars64.bat"), \
                mock.patch.object(P, "asset_name", return_value="x.tar.gz"), \
                mock.patch.object(A.S, "ok"), mock.patch.object(A.S, "warn") as warn:
            with mock.patch.object(P, "driver_version", return_value="615.71.10"):
                self.assertTrue(A.use_prebuilt(a, gpus))
            with mock.patch.object(P, "driver_version", return_value="590.1"):
                self.assertTrue(A.use_prebuilt(a, gpus))
                self.assertIn("610", warn.call_args[0][0])
            with mock.patch.object(P, "driver_version", return_value="575.1"), \
                    mock.patch.object(A.S, "fail", side_effect=SystemExit(1)) as fail:
                with self.assertRaises(SystemExit):
                    A.use_prebuilt(a, gpus)
                self.assertIn("580", fail.call_args[0][0])
            with mock.patch.object(A.S, "fail", side_effect=SystemExit(1)) as fail:
                with self.assertRaises(SystemExit):
                    A.use_prebuilt(a, [{"arch": "75", "name": "RTX 2080", "index": 0}])
                self.assertIn("7.5", fail.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
