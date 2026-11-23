"""A config's "gpu" naming cards by UUID: the engine's listing parsed, UUIDs turned into today's numbers, and the
visible-devices variable set from them as from plain numbers (the listing mocked: no engine, no GPU)."""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server  # noqa: E402

LISTING = """# abyss --list-gpus: CUDA devices as the run numbers them (index, uuid, pci, name)
0\tGPU-aa4f7ecd-6b3f-4a93-4813-7195effc94f4\t0000:03:00.0\tNVIDIA GeForce RTX 3090
1\tGPU-88044da7-d824-7056-3fbf-40d39acad923\t0000:c5:00.0\tNVIDIA GeForce RTX 5090
"""
CARDS = server.parse_gpu_listing(LISTING)


class GpuUuid(unittest.TestCase):
    def setUp(self):
        server._GPU_RESOLVED.clear()

    def test_parse_listing(self):
        self.assertEqual(CARDS[1], {"index": 1, "uuid": "GPU-88044da7-d824-7056-3fbf-40d39acad923",
                                    "pci": "0000:c5:00.0", "name": "NVIDIA GeForce RTX 5090"})
        self.assertEqual(server.parse_gpu_listing("# no CUDA device (no CUDA-capable device)\n"), [])
        self.assertEqual(server.parse_gpu_listing("0\t-\t0000:03:00.0\tAMD Radeon\n")[0]["uuid"], "")

    def test_resolve(self):
        r = server.resolve_gpu_items
        self.assertEqual(r(["GPU-88044da7-d824-7056-3fbf-40d39acad923"], CARDS), [1])
        self.assertEqual(r(["GPU-88044da7"], CARDS), [1])                    # a unique prefix
        self.assertEqual(r(["gpu-88044DA7-d824"], CARDS), [1])               # any case, with dashes
        self.assertEqual(r(["aa4f7ecd", 1], CARDS), [0, 1])                  # without GPU-, mixed with a number
        self.assertEqual(r([0, "2"], None), [0, 2])                          # numbers only: no listing needed
        amd = server.parse_gpu_listing("0\tGPU-1f5e3a9b0c2d4e6f\t0000:c7:00.0\tAMD Radeon AI PRO R9700\n")
        self.assertEqual(r(["0x1f5e3a9b0c2d4e6f"], amd), [0])                # rocm-smi --showuniqueid's form

    def test_resolve_errors(self):
        r = server.resolve_gpu_items
        with self.assertRaises(server.GpuSelectError) as e:
            r(["GPU-deadbeef"], CARDS)
        self.assertIn("matches no GPU", str(e.exception))
        self.assertIn("RTX 5090  UUID GPU-88044da7", str(e.exception))      # the cards found, listed
        with self.assertRaises(server.GpuSelectError) as e:
            r(["GPU-"], CARDS)                                               # (empty: names nothing)
        both = server.parse_gpu_listing("0\tGPU-12345678-aaaa\t0000:03:00.0\tA\n1\tGPU-12345678-bbbb\t0000:04:00.0\tB\n")
        with self.assertRaises(server.GpuSelectError) as e:
            r(["GPU-1234"], both)
        self.assertIn("matches several GPUs", str(e.exception))
        with self.assertRaises(server.GpuSelectError) as e:
            r(["GPU-1234"], server.parse_gpu_listing("0\t-\t0000:03:00.0\tAMD Radeon RX 9060 XT\n"))
        self.assertIn("report no UUID", str(e.exception))
        with self.assertRaises(server.GpuSelectError) as e:
            r(["GPU-1234"], None)
        self.assertIn("could not list", str(e.exception))

    def test_gpu_list_and_env(self):
        cfg = {"exe": "/x/abyss", "gpu": ["GPU-88044da7"]}
        with mock.patch.object(server, "list_gpus", return_value=CARDS) as ls, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(server.gpu_list(cfg), [1])
            env = server.child_env(cfg)
            self.assertEqual(server.gpu_list(dict(cfg, gpu="GPU-aa4f7ecd,1")), [0, 1])   # --gpu's string form
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(ls.call_count, 2)                                   # once per config in this process
        self.assertIn("GPU GPU-88044da7 is NVIDIA GeForce RTX 5090 (number 1 now)", out.getvalue())
        with mock.patch.object(server, "list_gpus") as ls:                   # numbers: as before, no listing
            self.assertEqual(server.gpu_list({"gpu": [0, 2]}), [0, 2])
            self.assertEqual(server.gpu_list({"gpu": 3}), [3])
            self.assertEqual(server.gpu_list({}), [])
        ls.assert_not_called()

    def test_hip_uuid_skips_gpu_pci(self):
        """A UUID names the card itself: the PCI-address renumbering (gpu_pci) is not applied on top."""
        cfg = {"exe": "/x/abyss", "gpu": ["GPU-1f5e"], "gpu_pci": ["0000:03:00.0"], "backend": "hip"}
        amd = server.parse_gpu_listing("0\tGPU-77aa\t0000:c8:00.0\tiGPU\n1\tGPU-1f5e3a9b\t0000:c7:00.0\tR9700\n")
        with mock.patch.object(server, "list_gpus", return_value=amd), \
                mock.patch.object(server, "amd_gpu_nodes") as nodes, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(server.hip_gpu_list(cfg), [1])
        nodes.assert_not_called()

    def test_list_gpus_runs_engine(self):
        """The listing runs the engine with every card visible, numbered in PCI bus order."""
        done = mock.Mock(returncode=0, stdout=LISTING)
        with mock.patch.object(server.subprocess, "run", return_value=done) as run, \
                mock.patch.dict(server.os.environ, {"CUDA_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "1"}):
            self.assertEqual(server.list_gpus("/x/abyss"), CARDS)
        args, kw = run.call_args
        self.assertEqual(args[0], ["/x/abyss", "--list-gpus"])
        self.assertNotIn("CUDA_VISIBLE_DEVICES", kw["env"])
        self.assertNotIn("HIP_VISIBLE_DEVICES", kw["env"])
        self.assertEqual(kw["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        with mock.patch.object(server.subprocess, "run", side_effect=OSError("gone")):
            self.assertIsNone(server.list_gpus("/x/abyss"))


if __name__ == "__main__":
    unittest.main()
