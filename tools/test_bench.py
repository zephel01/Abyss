"""./abyss.sh --bench: the engine's per-token counters, the bottleneck verdict, the PCIe chain and the model key."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import abyss as A

# an R9700 decode line of Ornith-1.5-122B (abyss-ornith-r9700l.log, 2026-10-23)
STAT = ("glm stat: decode 15.30 ms/tok (65.38 tok/s) | vram hit 52.13% | ram fetches 0.09/tok | disk 0.00 reads/tok "
        "in 0.00 waits/tok, 0.00 ms/tok | promotions 0.07/tok | cpu lane 235.20/tok in 6.94 ms/tok | tail skip "
        "0.00/tok | prompt 2.8 ms/tok | vram 4210/4361 ram 8327/9765")
GAP = ("glm gap prof (300 calls, ms/call): host after token 0.100 (sampler 0.020 x 1.00) | boundary sync 0.010 "
       "wait 0.020 scan 0.010 | embed 0.010 | spin 0.200 | GPU busy 14.000, idle between tokens 1.000 (300), "
       "of it before the next token's first command 0.300")


def stat(ms=20.0, disk_ms=0.0, lane_ms=5.0, fetch=0.5, skip=0.0):
    return {"ms": ms, "vram_hit": 50.0, "ram_fetch": fetch, "disk_reads": 1.0, "disk_waits": 1.0, "disk_ms": disk_ms,
            "promotions": 0.0, "lane": 200.0, "lane_ms": lane_ms, "tail_skip": skip, "prompt_ms": 3.0}


class BenchTests(unittest.TestCase):
    def test_records_follow_requests_with_spec_and_gap_lines(self):
        recs = A.bench_records(["glm fast: loaded", STAT, GAP, "glm spec: 338 of 470 drafts accepted (71.9%)",
                                STAT.replace("decode 15.30", "decode 30.00")])
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0]["ms"], 15.30)
        self.assertEqual(recs[0]["lane_ms"], 6.94)
        self.assertEqual(recs[0]["vram_hit"], 52.13)
        self.assertEqual(recs[0]["spec"], (338, 470))
        self.assertEqual((recs[0]["gpu_busy"], recs[0]["gpu_idle"]), (14.0, 1.0))
        self.assertEqual(recs[1]["ms"], 30.0)
        self.assertNotIn("spec", recs[1])

    def test_mmq_optimized_from_the_engine_line(self):
        """#49: the MMQ kernels' build state from the engine's line (None without it)."""
        self.assertTrue(A.mmq_optimized(["x", "glm prefill: CUDA0 MMQ kernels built optimized"]))
        self.assertFalse(A.mmq_optimized(["glm prefill: CUDA0 MMQ kernels built WITHOUT optimization (-O0): ..."]))
        self.assertIsNone(A.mmq_optimized(["glm prefill: CUDA0 chunks of 2048 tokens"]))

    def test_plan_share_from_the_cpu_lane_line(self):
        line = ("glm fast: CUDA0 CPU lane: 10 threads, an expert 0.357 ms on the CPU vs 0.322 ms over PCIe -> of 0..8 "
                "RAM-tier experts the CPU takes {}")
        self.assertEqual(A.plan_share([line.format("0 0 1 1 2 2 3 3 4")]), 0.5)    # issue #14: what share 0.5 gives
        self.assertEqual(A.plan_share([line.format("0 1 1 2 3 4 4 5 6")]), 0.25)
        self.assertAlmostEqual(A.plan_share([line.format("0 1 2 2 3 4 4 5 6")]), 0.2065, places=3)  # no sweep share
        self.assertEqual(A.plan_share(["glm fast: CUDA0 CPU lane off: an expert 0.5 ms on the CPU vs 0.1 ms"]), 1.0)
        self.assertIsNone(A.plan_share(["glm fast: loaded"]))

    def test_hip_advice_does_not_name_calibrate_as_the_fix(self):
        for k in ("cpu", "pcie"):
            self.assertIn("STRATA_GLM_PCIE_SHARE", A.BENCH_ADVICE_HIP[k])
            self.assertIn("CUDA-only", A.BENCH_ADVICE_HIP["cpu"])

    def test_the_rules_are_shared_with_abyss_terminal(self):   # (tools/bench_rules.py: one place for both)
        import bench_rules as R
        self.assertIs(A.bench_verdict, R.bench_verdict)
        self.assertIs(A.BENCH_ADVICE, R.BENCH_ADVICE)
        self.assertEqual(R.advice("cpu", "hip"), (R.BENCH_ADVICE["cpu"][0], R.BENCH_ADVICE_HIP["cpu"]))
        self.assertEqual(R.advice("gpu", "cuda"), R.BENCH_ADVICE["gpu"])
        self.assertEqual(R.advice("", "cuda"), ("", ""))
        self.assertEqual(R.lane_name(["glm fast: CUDA0 iGPU lane: an expert 0.042 ms vs 0.085 on the CPU's 16 "
                                      "threads -> the iGPU computes the lane"]), "iGPU lane")
        self.assertEqual(R.lane_name(["glm fast: loaded"]), "CPU lane")

    def test_sweep_compares_with_the_engines_best_run(self):
        # issue #19: own 24.8 (cold, 6.8 SSD reads a token) / 44.8 (warm, 0.06); share 0 42.6 is no win
        shares = [(0.0, 42.6), (0.25, 39.4), (0.5, 32.3), (0.75, 32.3), (1.0, 32.1)]
        out = "\n".join(A.sweep_report(shares, [24.8, 44.8], [6.81, 0.06], 0.21, None, "CPU lane", "share {s:.2f}"))
        self.assertNotIn("beats the engine's own choice", out)
        self.assertIn("still read the SSD 6.8 times a token", out)
        self.assertIn("24.8 / 44.8 tokens/s", out)
        # a share well above both own runs is still named, with how to set it
        out = "\n".join(A.sweep_report([(0.0, 58.4), (1.0, 12.2)], [30.0, 31.0], [0.2, 0.2], 0.5, None, "CPU lane",
                                       "S={s:.2f}"))
        self.assertIn("share 0.00 beats the engine's own choice (31.0 tokens/s)", out)
        self.assertIn("S=0.00", out)
        self.assertNotIn("still read the SSD", out)
        # no own runs: compared with the decode speed given
        out = "\n".join(A.sweep_report([(0.0, 20.0), (1.0, 19.5)], [], [], None, 19.8, "CPU lane", "{s}"))
        self.assertIn("barely moves the speed", out)

    def test_verdicts(self):
        self.assertEqual(A.bench_verdict([stat(disk_ms=8.0)])[0], "ssd")
        self.assertEqual(A.bench_verdict([stat(lane_ms=18.0)])[0], "cpu")
        self.assertEqual(A.bench_verdict([stat(fetch=25.0)])[0], "pcie")
        self.assertEqual(A.bench_verdict([stat()])[0], "gpu")
        busy = dict(stat(fetch=25.0), gpu_busy=19.0, gpu_idle=1.0)   # the GPU never waits: it is the limit
        self.assertEqual(A.bench_verdict([busy])[0], "gpu")
        self.assertEqual(A.bench_verdict([])[0], "")
        ev = A.bench_verdict([dict(stat(skip=2.0), spec=(7, 10))])[1]
        self.assertTrue(any("70% of its drafts" in x for x in ev))
        self.assertTrue(any("tail skip on" in x for x in ev))
        for k in ("ssd", "cpu", "pcie", "gpu"):
            self.assertEqual(len(A.BENCH_ADVICE[k]), 2)

    @unittest.skipIf(os.name == "nt", "sysfs and symlinks: Linux")
    def test_pcie_chain_takes_the_narrowest_link(self):
        # an AMD card: its own switch shows Gen5 x16, the OCuLink root port Gen4 x4
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "sys/devices/pci0000:00/0000:00:03.1"
            gpu = root / "0000:c5:00.0/0000:c6:00.0/0000:c7:00.0"
            gpu.mkdir(parents=True)
            for d, cur, w, mx, mw in ((root, "16.0", 4, "16.0", 8), (root / "0000:c5:00.0", "16.0", 4, "32.0", 16),
                                      (root / "0000:c5:00.0/0000:c6:00.0", "32.0", 16, "32.0", 16),
                                      (gpu, "32.0", 16, "32.0", 16)):
                (d / "current_link_speed").write_text(f"{cur} GT/s PCIe\n")
                (d / "current_link_width").write_text(f"{w}\n")
                (d / "max_link_speed").write_text(f"{mx} GT/s PCIe\n")
                (d / "max_link_width").write_text(f"{mw}\n")
            devs = Path(t) / "bus"
            devs.mkdir()
            os.symlink(gpu, devs / "0000:c7:00.0")
            chain = A.pcie_chain("0000:c7:00.0", str(devs))
            self.assertEqual(len(chain), 4)
            self.assertEqual(A.pcie_link(chain), "Gen4 x4 (~7.9 GB/s), the GPU can do Gen5 x16")
            self.assertEqual(A.pcie_chain("0000:00:00.0", str(devs)), [])
        self.assertEqual(A.pcie_link([]), "?")

    def test_model_key_is_the_shard_set(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "pack").mkdir()
            for n, b in (("M-UD-Q4_K_XL-00001-of-00002.gguf", 10), ("M-UD-Q4_K_XL-00002-of-00002.gguf", 20),
                         ("M-nextn-q4k.gguf", 5)):
                (d / n).write_bytes(b"x" * b)
            self.assertEqual(A.model_key(d / "pack"), ("M-UD-Q4_K_XL", 30))
            (d / "M-UD-Q4_K_XL-00001-of-00002.gguf").unlink()
            (d / "M-UD-Q4_K_XL-00002-of-00002.gguf").unlink()
            self.assertEqual(A.model_key(d / "pack"), ("M-nextn-q4k", 5))
        self.assertEqual(A.model_key(None), ("", 0))

    def test_nvidia_links_parse_nvidia_smi(self):
        from unittest.mock import patch
        smi = "0, 4, 4, 5, 16\n1, 3, 16, 4, 16\nbad line\n"
        with patch.object(A.subprocess, "run", return_value=A.subprocess.CompletedProcess([], 0, smi, "")):
            links = A.nvidia_links([0])
        self.assertEqual(links, {0: [(4, 4, 5, 16)]})
        self.assertEqual(A.pcie_link(links[0]), "Gen4 x4 (~7.9 GB/s), the GPU can do Gen5 x16")
        with patch.object(A.subprocess, "run", side_effect=OSError("no nvidia-smi")):
            self.assertEqual(A.nvidia_links([0]), {})

    def test_disk_read_measures_a_big_file(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "m-00001-of-00001.gguf"
            with open(f, "wb") as w:
                w.truncate(300 << 20)
            v = A.disk_read_gbs([f], gib=0.05)
            self.assertTrue(v is None or v > 0)
            self.assertIsNone(A.disk_read_gbs([Path(t) / "missing.gguf"]))

    def test_drafter_check_finds_the_first_difference(self):   # (greedy with and without speculative decoding)
        self.assertIsNone(A.first_difference(["1", "2", "3"], ["1", "2", "3"]))
        self.assertEqual(A.first_difference(["1", "2", "3"], ["1", "5", "3"]), 1)
        self.assertEqual(A.first_difference(["1", "2"], ["1", "2", "3"]), 2)
        self.assertIsNone(A.first_difference([], []))

    def test_quality_without_a_reference_is_none(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "pack").mkdir()
            (d / "Nothing-00001-of-00001.gguf").write_bytes(b"x")
            self.assertIsNone(A.bench_quality({}, d / "pack", None))

    def test_quality_picks_the_reference_of_this_copy(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "pack").mkdir()
            (d / "M-00001-of-00001.gguf").write_bytes(b"x" * 30)
            ref = d / "bench" / "ref"
            ref.mkdir(parents=True)
            for stem, size in (("M", 99), ("M.overlay", 30)):   # issue #15: stock files + an overlay, one name
                (ref / f"{stem}.json").write_text(f'{{"gguf_bytes": {size}, "topk": "{stem}.topk", "ids": [1, 2]}}')
                (ref / f"{stem}.topk").write_bytes(b"k")
            sv = SimpleNamespace(child_env=lambda cfg: {}, engine_args=lambda cfg: [])
            out = A.subprocess.CompletedProcess([], 0, "SCORE 2 tokens: nll 1.0 top1 50.0% KL 0.004 (max 0.1), "
                                                       "same top token 99.0%", "")
            with patch.object(A, "ROOT", d), patch.object(A.subprocess, "run", return_value=out) as run:
                q = A.bench_quality({"exe": "strata"}, d / "pack", sv)
            self.assertEqual(q["verdict"], "OK")
            self.assertTrue(run.call_args.kwargs["env"]["STRATA_GLM_SCORE_TOPK"].endswith("M.overlay.topk"))
            (d / "M-00001-of-00001.gguf").write_bytes(b"x" * 31)                      # neither copy
            with patch.object(A, "ROOT", d):
                self.assertIn("skipped", A.bench_quality({"exe": "strata"}, d / "pack", sv))


if __name__ == "__main__":
    unittest.main()
