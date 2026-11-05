"""The DGX Spark (GB10: an ARM CPU and an integrated GPU sharing one memory) as the tools see it - from the text its
nvidia-smi, /proc/cpuinfo and lscpu print, no GPU and no ARM machine needed (the ARM CI job runs these too).

    python -m unittest tools.test_arm
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402

# nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version on a DGX Spark: no memory.total
GB10 = "0, NVIDIA GB10, [N/A], 12.1, 580.95.05\n"
# /proc/cpuinfo on aarch64: "Features" in place of "flags", no "model name"
ARM_CPUINFO = ("processor\t: 0\nBogoMIPS\t: 2000.00\nFeatures\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics "
               "fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit "
               "uscat ilrcpc flagm sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint "
               "svei8mm svebf16 i8mm bf16 dgh bti\nCPU implementer\t: 0x41\nCPU architecture: 8\nCPU variant\t: 0x0\n"
               "CPU part\t: 0xd85\nCPU revision\t: 1\n\n")
LSCPU = ("Architecture:             aarch64\n  CPU op-mode(s):         64-bit\nCPU(s):                   20\n"
         "Vendor ID:                ARM\n  Model name:             Cortex-X925\n    Thread(s) per core:   1\n"
         "  Model name:             Cortex-A725\n    Thread(s) per core:   1\n")


class SmiTests(unittest.TestCase):
    def test_gb10_is_an_integrated_gpu_with_the_system_memory(self):
        with mock.patch.object(S, "out", return_value=GB10), mock.patch.object(S, "_mem_total_gib", return_value=119.7):
            found = S.gpus()
        self.assertEqual(len(found), 1)
        g = found[0]
        self.assertEqual((g["name"], g["arch"], g["integrated"]), ("NVIDIA GB10", "121", True))
        self.assertAlmostEqual(g["vram_gb"], 119.7)

    def test_a_discrete_card_without_memory_is_left_out(self):
        smi = "0, NVIDIA GeForce RTX 5090, 32607, 12.0, 610.43.02\n1, NVIDIA GeForce RTX 3090, [N/A], 8.6, 610.43.02\n"
        with mock.patch.object(S, "out", return_value=smi):
            found = S.gpus()
        self.assertEqual([g["name"] for g in found], ["NVIDIA GeForce RTX 5090"])
        self.assertNotIn("integrated", found[0])
        self.assertAlmostEqual(found[0]["vram_gb"], 32607 / 1024.0)

    def test_cuda_arch_for_gb10(self):
        import abyss_setup as AS
        self.assertEqual(AS.cuda_archs(None, [{"arch": "121"}]), "121a")
        self.assertEqual(AS.cuda_archs(None, [{"arch": "86"}, {"arch": "120"}]), "86;120a")


class CpuTests(unittest.TestCase):
    def test_arm_cpuinfo(self):
        if S.WIN:
            self.skipTest("the /proc/cpuinfo path is Linux's")
        with mock.patch("builtins.open", mock.mock_open(read_data=ARM_CPUINFO)), \
                mock.patch.object(S, "out", return_value=LSCPU):
            name, avx2, avx512 = S.cpu_info()
        self.assertEqual(name, "ARM Cortex-X925 + Cortex-A725")
        self.assertFalse(avx2)
        self.assertFalse(avx512)

    def test_arm_name_without_lscpu(self):
        self.assertEqual(S._arm_cpu_name(""), "")
        self.assertEqual(S._arm_cpu_name("  Model name: Neoverse-V2\n"), "Neoverse-V2")


class AbyssTests(unittest.TestCase):
    def setUp(self):
        import abyss as A
        self.A = A

    def test_cuda_floor_for_sm_121(self):
        self.assertEqual(self.A.nvcc_range([121])[0], (12, 9))
        self.assertEqual(self.A.nvcc_range([120])[0], (12, 8))
        self.assertEqual(self.A.nvcc_range([86])[0], (12, 0))

    def test_label_says_unified(self):
        g = {"index": 0, "name": "NVIDIA GB10", "vram_gb": 119.7, "arch": "121", "integrated": True}
        self.assertIn("unified memory", self.A.gpu_label(g))
        self.assertNotIn("unified", self.A.gpu_label(dict(g, integrated=False)))

    def test_sbsa_libraries(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            lib = Path(d) / "targets" / "sbsa-linux" / "lib"
            lib.mkdir(parents=True)
            (Path(d) / "bin").mkdir()
            self.assertIn(str(lib), self.A.cuda_lib_dirs(str(Path(d) / "bin" / "nvcc")))


class CheckTests(unittest.TestCase):
    def test_report_on_unified_memory(self):
        import abyss_check as C
        import pc_limits as PL
        import bench_rules as BR
        from test_check import STAT, GAP, START
        rec = BR.bench_records([STAT, GAP])[0]
        lines = PL.what_limits(rec, PL.engine_facts(START), None, 2.0, [], 119.0, unified=True)
        txt = "\n".join(lines)
        self.assertIn("PCIe: none - the GPU shares the system memory", txt)
        self.assertIn("RAM-tier copies", txt)
        self.assertNotIn("x16", txt)
        head = {k: "x" for k in ("version", "model", "model_detail", "gpus", "driver", "pcie", "ram", "cpu", "os",
                                 "ssd", "date", "mode", "setup", "context", "backend", "drafter", "commit", "load_s")}
        head["unified"] = True
        out = C.render(head, [{"name": "drafter", "status": "PASS", "secs": 3.0, "decode": 47.0, "note": "same"}],
                       lines, [], [], [])
        self.assertIn("| memory unified | RAM", out)
        self.assertIn("- memory: unified", out)
        self.assertNotIn("PCIe while working", out)


if __name__ == "__main__":
    unittest.main()
