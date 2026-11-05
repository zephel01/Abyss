"""./abyss.sh --check (tools/abyss_check.py) and "What limits this PC" (tools/pc_limits.py): the parts that need no GPU -
the steps a context gets, the graders, the report's privacy, the limits' split and estimates."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abyss_check as C
import pc_limits as PL


class StepsTests(unittest.TestCase):
    def test_long_prompts_follow_the_context(self):
        self.assertEqual(C.long_sizes(32768, C.QUICK), [8192, 24576])
        self.assertEqual(C.long_sizes(16384, C.QUICK), [8192])
        self.assertEqual(C.long_sizes(32768, C.DEFAULT), [8192, 24576, 31744])     # the context's top, room kept
        self.assertEqual(C.long_sizes(65536, C.DEFAULT), [8192, 24576, 32768, 64512])
        self.assertEqual(C.long_sizes(262144, C.DEFAULT), [8192, 24576, 32768, 65536])   # (no more than 64K)
        self.assertEqual(C.long_sizes(262144, C.LONG), [8192, 24576, 32768, 65536, 131072, 261120])
        self.assertEqual(C.long_sizes(8192, C.DEFAULT), [7168])
        for ctx in (8192, 32768, 65536, 262144):
            for mode in (C.QUICK, C.DEFAULT, C.LONG):
                self.assertTrue(all(n + C.ANSWER_ROOM <= ctx for n in C.long_sizes(ctx, mode)))

    def test_plan(self):
        quick = [s for s, _ in C.plan(32768, C.QUICK, True)]
        self.assertEqual(quick, ["drafter", "ja-explain", "code-lru", "tool-call", "long-8k", "long-24k",
                                 "conversation"])
        full = [s for s, _ in C.plan(65536, C.DEFAULT, False)]
        self.assertNotIn("tool-call", full)                                   # (a template without tools)
        self.assertEqual([s for s in full if s.startswith("code-")], ["code-lru", "code-intervals", "code-hiragana"])
        self.assertIn("long-63k", full)
        self.assertGreaterEqual(len(C.TURNS) + 1, 10)                        # a 10+ turn session


class GraderTests(unittest.TestCase):
    def test_japanese(self):
        self.assertTrue(C.grade_japanese("MoEは、たくさんの専門家の中から必要なものだけを選んで使う仕組みです。", None)[0])
        self.assertFalse(C.grade_japanese("MoE is a mixture of experts model.", None)[0])
        self.assertFalse(C.grade_japanese("", None)[0])
        self.assertFalse(C.grade_japanese("おつりは３２５円\ufffdです。ひらがなをもっとたくさんかきます。", "325")[0])
        self.assertTrue(C.grade_japanese("おつりは325円です。りんごは360円、みかんは315円になります。", "325")[0])
        self.assertFalse(C.grade_japanese("おつりは300円です。りんごは360円、みかんは315円になります。", "325")[0])

    def test_loops(self):
        self.assertTrue(C.looping("the same sentence again and again here. " * 12))
        self.assertFalse(C.looping("A short answer."))
        self.assertIn("loop", C.broken("ありがとうございます。それではまた。" * 20))

    def test_code_runs_against_the_tests(self):
        good = "Here it is:\n```python\ndef add(a, b):\n    return a + b\n```\nUsage:\n```python\nprint(add(1, 2))\n```"
        self.assertEqual(C.grade_code(good, "assert add(2, 3) == 5"), (True, "the tests pass"))
        ok, why = C.grade_code(good, "assert add(2, 3) == 6")
        self.assertFalse(ok)
        self.assertIn("AssertionError", why)
        self.assertFalse(C.grade_code("no code here", "pass")[0])
        net = "```python\nimport socket\ndef f():\n    return socket.create_connection(('example.com', 80))\n```"
        self.assertFalse(C.grade_code(net, "f()")[0])                        # (no network in the check)
        slow = "```python\ndef f():\n    while True: pass\n```"
        self.assertEqual(C.grade_code(slow, "f()", timeout=1), (False, "the code ran over 1 s"))

    def test_tool_call(self):
        call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'}}
        self.assertTrue(C.grade_tool_call({"tool_calls": [call]})[0])
        self.assertFalse(C.grade_tool_call({"content": "It is sunny."})[0])
        bad = dict(call, function={"name": "get_weather", "arguments": "{city: Tokyo"})
        self.assertIn("not JSON", C.grade_tool_call({"tool_calls": [bad]})[1])
        other = dict(call, function={"name": "get_weather", "arguments": '{"city": "Osaka"}'})
        self.assertFalse(C.grade_tool_call({"tool_calls": [other]})[0])

    def test_turns(self):
        self.assertTrue(C.grade_turn("I would use TOML for the settings.", ("json", "toml"))[0])
        self.assertFalse(C.grade_turn("Bananas are yellow.", ("json", "toml"))[0])
        self.assertTrue(C.grade_turn("Your tool is Hoshizora and its page is on port 8642.", ("hoshizora", "8642"))[0])
        self.assertEqual(C.grade_turn("Your tool is Hoshizora.", ("hoshizora", "8642")), (False, "forgot 8642"))

    def test_drafter_compare_in_tokens(self):
        enc = lambda s: s.split()                                         # (a word a token)
        self.assertEqual(C.drafter_compare("abc", "abc", enc)[0], "PASS")
        head = " ".join(f"w{k}" for k in range(12))
        self.assertEqual(C.drafter_compare(head + " a", head + " b", enc)[0], "PASS")   # token 12: a near-tie
        st, note = C.drafter_compare("w0 w1 a", "w0 w1 b", enc)
        self.assertEqual((st, "token 2" in note), ("WARN", True))
        # many characters but few tokens: still early (the 07:55 run parted at character 50)
        self.assertEqual(C.drafter_compare("x" * 300 + " a", "x" * 300 + " b", enc)[0], "WARN")
        self.assertEqual(C.drafter_compare("x" * 300 + "a", "x" * 300 + "b")[0], "PASS")   # (no tokenizer: chars/4)

    def test_warn_is_not_a_failure_exit(self):
        self.assertEqual(C.overall([{"status": "PASS"}, {"status": "WARN"}]), "WARN")

    def test_needle_prompt(self):
        p = C.needle_prompt("para one.\n\npara two.\n\npara three.\n\n" * 50, 1000, 13, 0.5)
        self.assertIn(C.NEEDLE_LINE, p)
        self.assertTrue(p.endswith(C.NEEDLE_Q))
        self.assertLess(abs(p.index(C.NEEDLE_LINE) - 500), 60)


class PrivacyTests(unittest.TestCase):
    def test_scrub_paths_and_names(self):
        home = str(Path.home())
        text = (f"glm fast: the warm-up follows {home}/llm/Abyss/data/expert_counts/q.txt\n"
                "model folder /mnt/models/zephel01/Qwen3.8-M16b/pack-fast/dense.bin\n"
                "loaded C:\\Users\\Taro\\Abyss-data\\models\\Q\\x.gguf on DESKTOP-ABC123 by taro\n"
                "tokens/s 1/2 and 4/8 stay; ratio a/b too")
        out = C.scrub(text, [("/mnt/models/zephel01/Qwen3.8-M16b", "<model>")],
                      [("taro", "<user>"), ("DESKTOP-ABC123", "<host>")])
        self.assertNotIn(home, out)
        self.assertNotIn("zephel01", out)
        self.assertNotIn("Users", out)
        self.assertNotIn("Taro", out)
        self.assertNotIn("DESKTOP-ABC123", out)
        self.assertIn("<model>/pack-fast/dense.bin", out)
        self.assertIn("x.gguf", out)
        self.assertIn(".../q.txt", out)
        self.assertIn("1/2 and 4/8 stay", out)

    def test_short_names_are_not_scrubbed(self):   # (a user "ai" must not eat every "ai" in the text)
        self.assertEqual(C.scrub("said ai", [], [("ai", "<user>")]), "said ai")

    def test_render_and_overall(self):
        head = {k: "x" for k in ("version", "model", "model_detail", "gpus", "driver", "pcie", "ram", "cpu", "os",
                                 "ssd", "date", "mode", "setup", "context", "backend", "drafter", "commit", "load_s")}
        res = [{"name": "drafter", "status": "PASS", "secs": 3.0, "decode": 100.0, "note": "same"},
               {"name": "long-24k", "status": "FAIL", "secs": 9.0, "note": "HTTP 500: engine stopped"},
               {"name": "conversation", "status": "NOT RUN", "note": "the engine stopped before it"}]
        self.assertEqual(C.overall(res), "FAIL")
        self.assertEqual(C.overall(res[:1]), "PASS")
        self.assertEqual(C.overall(res[:1] + [{"name": "a", "status": "WARN"}]), "WARN")
        txt = C.render(head, res, ["A token: 9 ms"], [{"step": "long-24k", "status": "FAIL", "why": "HTTP 500",
                                                       "log": ["ERR prompt scratch overrun"], "server": ["x"]}],
                       [("drafter", "def f")], ["glm stat: decode 9 ms"])
        self.assertIn("check: FAIL (1 of 3 steps)", txt)
        self.assertIn("| long-24k | FAIL |", txt)
        self.assertIn("ERR prompt scratch overrun", txt)
        self.assertIn("What limits this PC", txt)


STAT = ("glm stat: decode 20.00 ms/tok (50.00 tok/s) | vram hit 60.00% | ram fetches 4.00/tok | disk 3.00 reads/tok in "
        "3.00 waits/tok, 6.00 ms/tok | promotions 1.00/tok | cpu lane 100.00/tok in 4.00 ms/tok | tail skip 0.00/tok | "
        "prompt 2.0 ms/tok | vram 100/100 ram 100/100")
GAP = ("glm gap prof (300 calls, ms/call): host after token 0.100 (sampler 0.020 x 1.00) | boundary sync 0.010 "
       "wait 0.020 scan 0.010 | embed 0.010 | spin 0.200 | GPU busy 18.000, idle between tokens 1.000 (300), "
       "of it before the next token's first command 0.300")
START = ["glm fast: CUDA0 layers [0,4) expert pool 0.10 GB, 20-30 slots/layer (100 total)",
         "glm fast: CUDA0 RAM tier 0.20 GB pinned, 100 slots",
         "glm fast: CUDA0 CPU lane: 8 threads, an expert 0.040 ms on the CPU vs 0.300 ms over PCIe -> of 0..8 RAM-tier "
         "experts the CPU takes 0 1 2 3 4 5 6 7 8"]


class LimitsTests(unittest.TestCase):
    def setUp(self):
        import bench_rules as BR
        self.rec = BR.bench_records([STAT, GAP])[0]

    def test_facts_from_the_start_lines_and_the_pack(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "native_experts.txt").write_text(
                "# strata native experts v3: layer ... (n_expert 64, total 268435456; absolute offsets)\n"
                + "".join(f"{i} 18 20 0 1048576 0 0 0 x.gguf\n" for i in range(4)))
            f = PL.engine_facts(START, Path(d))
        self.assertEqual((f["layers"], f["slots"], f["ram_slots"]), (4, 100, 100))
        self.assertEqual((f["cpu_ms"], f["pcie_ms"]), (0.04, 0.3))
        self.assertEqual((f["n_expert"], f["model_layers"]), (64, 4))
        self.assertEqual(f["expert_bytes"], 268435456 / 256)

    def test_split_adds_up_to_the_token(self):
        f = PL.engine_facts(START)
        p = PL.split_token(self.rec, f)
        self.assertAlmostEqual(p["pcie"], 1.2)
        self.assertAlmostEqual(p["ssd"] + p["lane"] + p["pcie"] + p["gaps"] + p["gpu"], 20.0)
        self.assertTrue(p["gaps_measured"])
        over = dict(self.rec, disk_ms=30.0)                   # (parts that overlap: scaled into the token)
        q = PL.split_token(over, f)
        self.assertAlmostEqual(q["ssd"] + q["lane"] + q["pcie"] + q["gaps"], 20.0)
        self.assertEqual(q["gpu"], 0.0)

    def test_lru_and_the_vram_ratio(self):
        routes = [(0, [1, 2]), (0, [3, 1]), (0, [2, 3])] * 50
        self.assertEqual(PL.lru_misses(routes, 1, 3), 3)
        self.assertGreater(PL.lru_misses(routes, 1, 2), 3)
        self.assertEqual(PL.vram_ratio(routes, 1, 2, 3, None), (3 / PL.lru_misses(routes, 1, 2)) ** 0.5)
        self.assertEqual(PL.vram_ratio(routes, 1, 2, 64, 64), 0.0)        # (every expert fits)
        self.assertIsNone(PL.vram_ratio(routes[:5], 1, 2, 3, None))     # (too few routes)

    def test_what_limits_orders_by_gain_and_marks_estimates(self):
        import random
        rnd = random.Random(1)
        routes = [(il, rnd.sample(range(64), 8)) for _ in range(400) for il in range(4)]
        f = PL.engine_facts(START)
        f.update(n_expert=64, experts_bytes=64 * 4 * 2**20, expert_bytes=2**20)
        lines = PL.what_limits(self.rec, f, [(4, 4, 4, 16)], 2.0, routes, 16.0)
        txt = "\n".join(lines)
        self.assertIn("What would change it", txt)
        self.assertIn("x16 Gen4", txt)
        self.assertIn("RAM: 3.0 experts a token come from the SSD", txt)
        self.assertIn("SSD: read 2.0 GB/s - estimate", txt)
        # the lane takes 93% of the misses: the hit rate first, then the speed with the lane's share named
        self.assertRegex(txt, r"VRAM: holds 60% .* ~\d+%; speed ~\d+ tokens/s \([+-]\d+%\) - the CPU lane takes 9\d% of the misses")
        mostly_pcie = dict(self.rec, lane=5.0, lane_ms=0.5, ram_fetch=60.0)
        self.assertRegex("\n".join(PL.what_limits(mostly_pcie, f, [(4, 4, 4, 16)], 2.0, routes, 16.0)),
                         r"VRAM: holds 60% .* ~\d+%; speed ~\d+ tokens/s")
        self.assertIn("estimate", txt)
        self.assertNotRegex(txt, r"(?i)nvidia|geforce|radeon|samsung")   # (sizes and classes only, no products)
        gains = [ln for ln in lines if ln[:2] in ("1.", "2.")]
        self.assertEqual(len(gains), 2)

    def test_nothing_to_change(self):
        rec = dict(self.rec, vram_hit=99.0, ram_fetch=0.0, disk_reads=0.0, disk_ms=0.0, lane=1.0, lane_ms=0.1)
        txt = "\n".join(PL.what_limits(rec, PL.engine_facts(START), [(5, 16, 5, 16)], 7.0, [], 32.0))
        self.assertIn("already the GPU's full link: no change needed", txt)
        self.assertIn("VRAM: holds 99% of the experts a token uses - no change needed", txt)
        self.assertIn("RAM: the RAM tier holds every expert", txt)
        self.assertIn("GPU compute", txt)

    def test_no_record(self):
        self.assertIn("no decode statistics", PL.what_limits(None, {})[0])


class DispatchTests(unittest.TestCase):
    """--check: the model check when a model is installed; before the setup (or with --setup / --gpu) the PC check."""

    def run_main(self, argv, installed):
        import os
        from unittest import mock
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import abyss as A
        cfg = Path("/x/abyss-q38.json")
        with mock.patch.object(sys, "argv", ["abyss.py"] + argv), mock.patch.dict(os.environ, {"MAYA_SH": "1"}), \
                mock.patch.object(A, "all_configs", return_value=[cfg] if installed else []), \
                mock.patch.object(A, "configs", return_value=[]), \
                mock.patch.object(A, "read_json", return_value={}), \
                mock.patch.object(A, "model_check", return_value=7) as mc, \
                mock.patch.object(A, "check_pc", return_value={"gpus": [{"index": 0}]}) as pc, \
                mock.patch.object(A, "say"):
            rc = A.main()
        return rc, mc, pc

    def test_installed_runs_the_model_check(self):
        rc, mc, pc = self.run_main(["--check", "--quick"], True)
        self.assertEqual(rc, 7)
        self.assertEqual(mc.call_args[0][2], "quick")
        pc.assert_not_called()
        self.assertEqual(self.run_main(["--check", "--long"], True)[1].call_args[0][2], "long")
        self.assertEqual(self.run_main(["--check"], True)[1].call_args[0][2], "default")

    def test_nothing_installed_or_setup_flags_check_the_pc(self):
        for argv, installed in ((["--check"], False), (["--check", "--setup"], True), (["--check", "--gpu", "0"], True)):
            rc, mc, pc = self.run_main(argv, installed)
            mc.assert_not_called()
            pc.assert_called_once()


if __name__ == "__main__":
    unittest.main()
