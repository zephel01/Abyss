"""tools/test_abyss_terminal.py - Abyss Terminal (abyss_terminal.py, terminal_model/draw/local/app.py) against a fake
Abyss server (terminal_fixtures.FakeServer: the /metrics of DeepSeek V4.1 UD-IQ2_XXS on an R9700 + iGPU lane + two
SSDs, a 16 GB + 64 GB PC, a 5090 + 3090 split, 5090 + 3090 + 2x V100, 8 and 10 RTX 3090s, a server loading, one
down) and fake engine logs: the connection's
states, the numbers (missing stays --, never 0), --bench's verdict from the log, every view drawn to the exact size at
160x50, 80x24 and 40x12, the detached start / stop with its PID file, and the screen with Textual's pilot (skipped
without Textual).  No GPU, no engine.

    ABYSS_TERMINAL_SHOTS=docs/terminal python -m unittest tools.test_abyss_terminal   (also writes the SVG screenshots)
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_STATE = tempfile.TemporaryDirectory()                      # (never the user's own ~/.config/abyss/terminal.json)
os.environ["ABYSS_TERMINAL_STATE"] = str(Path(_STATE.name) / "terminal.json")
import abyss_terminal as A  # noqa: E402
import bench_rules as R  # noqa: E402
import terminal_draw as D  # noqa: E402
import terminal_fixtures as F  # noqa: E402
import terminal_local as T  # noqa: E402
import terminal_model as M  # noqa: E402
try:
    import terminal_app as TA
except ImportError:
    TA = None
needs_textual = unittest.skipIf(TA is None, "Textual is not installed (./abyss.sh --setup installs it)")


class FakeLink:
    """A Link as after a poll of a scenario (no HTTP)."""

    def __init__(self, sc, key=None, base="http://127.0.0.1:8082"):
        self.metrics, self.health = F.metrics(sc), F.health(sc)
        self.state = "live" if self.metrics else "loading" if (self.health or {}).get("status") == "loading" \
            else "disconnected"
        self.error = "" if self.metrics or self.state == "loading" else "[Errno 111] Connection refused"
        self.base, self.key = base, key
        self.last_ok = time.time() - 0.3 if self.metrics else None
        self.ever_ok = bool(self.metrics)


def log_of(sc):
    if sc not in F.LOGS:
        return None
    lg = M.EngineLog(None)
    for x in F.LOGS[sc]:
        lg.feed(x, None)
    return lg


def view(sc, seen=None, **kw):
    remote = sc in ("split", "multi8", "starting")
    link = FakeLink(sc, key="k" if remote else None,
                    base="http://192.168.1.20:8082" if remote else "http://127.0.0.1:8082")
    return M.derive(link, F.local(sc), log_of(sc), seen=seen if seen is not None else {}, **kw)


class Names(unittest.TestCase):
    def test_the_name_always_carries_its_quant(self):
        self.assertEqual(M.pretty_name("DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS", "UD-IQ2_XXS"),
                         ("DeepSeek V4.1 Flash 1017", "UD-IQ2_XXS"))
        self.assertEqual(M.pretty_name("Ornith-1.5-122B-IQ3_XXS-B", "IQ3_XXS"), ("Ornith 1.5 122B", "IQ3_XXS-B"))
        self.assertEqual(M.pretty_name(None, None, "glm-5.3-flash"), ("glm-5.3-flash", None))
        self.assertEqual(M.pretty_name("Qwen3.8-Flash-Next-UD-Q4_K_XL", None), ("Qwen3.8 Flash Next", "UD-Q4_K_XL"))

    def test_badges_for_the_settings_that_change_the_output(self):
        b = dict(M.badges({"uncensor": 1, "mtp": 1}, {"STRATA_GLM_TAIL_SKIP": "4", "STRATA_GLM_DEFER": "1",
                                                       "STRATA_GLM_MIRROR": "/mnt/ssd2/x"}))
        self.assertEqual(b["UNCENSOR"], "alert")
        self.assertEqual(b["TAIL SKIP 4"], "alert")
        self.assertEqual(b["DEFER"], "alert")
        self.assertIn("MIRROR", b)
        self.assertIn("MTP", b)
        self.assertEqual(M.badges({"uncensor": 0, "mtp": 0}), [])

    def test_server_strings_lose_control_and_bidi_characters(self):
        self.assertEqual(M.clean("a\x1b[31mb\u202ec\nd"), "a[31mbcd")


class Connection(unittest.TestCase):
    def test_live_then_stale_then_disconnected(self):
        srv = F.FakeServer("r9700")
        link = M.Link(srv.url, timeout=2)
        self.assertEqual(link.state, "connecting")
        self.assertEqual(link.poll(), "live")
        self.assertEqual(link.metrics["engine"]["quant"], "UD-IQ2_XXS")
        srv.down()
        self.assertEqual(link.poll(), "stale")             # (the values are kept)
        self.assertIsNotNone(link.metrics)
        link.poll()
        link.poll()
        self.assertEqual(link.state, "disconnected")
        self.assertGreater(link.next_delay(1.0), 1.5)       # back-off
        link.fails = 20
        self.assertLessEqual(link.next_delay(1.0), M.BACKOFF_MAX_S * 1.15)

    def test_never_answered_is_disconnected_not_stale(self):
        srv = F.FakeServer("r9700")
        srv.down()
        self.assertEqual(M.Link(srv.url, timeout=1).poll(), "disconnected")

    def test_auth_required_stops_polling_until_asked(self):
        with F.FakeServer("split", key="sekrit") as srv:
            link = M.Link(srv.url)
            self.assertEqual(link.poll(), "auth")
            self.assertEqual(link.next_delay(1.0), float("inf"))
            link.key = "sekrit"
            self.assertEqual(link.poll(), "live")
            self.assertIn("Bearer", "Bearer sekrit")

    def test_another_server_is_not_abyss(self):
        with F.FakeServer("r9700", service="strata") as srv:
            self.assertEqual(M.Link(srv.url).poll(), "not_abyss")

    def test_poller_kick_and_the_requests_view(self):
        with F.FakeServer("r9700") as srv:
            p = M.Poller(M.Link(srv.url), interval=5).start()
            deadline = time.time() + 5
            while p.polls < 1 and time.time() < deadline:
                time.sleep(0.02)
            p.all_requests = True
            p.kick()
            while p.polls < 2 and time.time() < deadline:
                time.sleep(0.02)
            p.stop()
            self.assertIn("/metrics?requests=all", srv.hits)

    def test_a_server_still_loading_is_loading_not_disconnected(self):
        with F.FakeServer("starting", key="k") as srv:
            link = M.Link(srv.url, key="k")
            self.assertEqual(link.poll(), "loading")                  # (/health 200 "loading"; /metrics not asked)
            self.assertEqual((link.fails, link.error, link.health["load"]["pct"]), (0, "", 42.0))
            self.assertNotIn("/metrics", srv.hits)
            self.assertGreaterEqual(link.next_delay(1.0), 2.0)
            srv.scenario = "r9700"
            self.assertEqual(link.poll(), "live")

    def test_the_remote_log_comes_over_api_log(self):
        text = "\n".join(F.LOG_R9700) + "\n"
        with F.FakeServer("r9700", key="k", log_text=text) as srv:
            os.environ["T_KEY"] = "k"
            s = A.Session(url=srv.url, key_env="T_KEY", poll=False)
            s.link.poll()
            self.assertIsNone(s.log)
            n = s.fetch_remote_log()
            self.assertGreater(n, 0)
            self.assertEqual(s.log.remote_next, len(text.encode()))
            self.assertEqual(s.log.lane, "iGPU lane")
            self.assertTrue(s.log.decode_recs())                      # the "glm stat" records: the Speed view's
            self.assertEqual(s.fetch_remote_log(), 0)                 # (nothing new)
            v = s.view_data()
            self.assertEqual(v["tt"]["source"], "live")
            text2 = "\n".join(x.plain for x in D.logs_view(v, 120, 30, s.ui()))
            self.assertIn("remote /api/log", text2)
            self.assertIn("glm stat", text2)
            s.close()
        with F.FakeServer("r9700", key="k") as srv:                   # an older server: no /api/log, the hint stays
            s = A.Session(url=srv.url, key_env="T_KEY", poll=False)
            s.link.poll()
            self.assertEqual(s.fetch_remote_log(), 0)
            self.assertIsNone(s.log)
            s.close()

    def test_shutdown_goes_to_the_server(self):
        with F.FakeServer("r9700") as srv:
            with tempfile.TemporaryDirectory() as d:
                cfg = Path(d) / "abyss-x.json"
                cfg.write_text("{}")
                msg = T.stop(cfg, M.Link(srv.url))
            self.assertEqual(srv.shutdowns, [{}])
            self.assertIn("stopping", msg)


class Numbers(unittest.TestCase):
    def test_r9700_live_decode_the_ssd_verdict_the_lane_and_the_mirror(self):
        v = view("r9700")
        self.assertEqual((v["conn"], v["state"]), ("live", "generating"))
        self.assertEqual((v["name"], v["quant"]), ("DeepSeek V4.1 Flash 1017", "UD-IQ2_XXS"))
        self.assertEqual(v["decode"]["label"], "live")
        self.assertEqual(v["decode"]["value"], 24.6)
        self.assertEqual(v["prompt"]["label"], "last request")
        tt = v["tt"]
        # a 1.3.80+ engine: the live STAT's window (the last ~64 tokens), the lane's kind from INFO
        self.assertEqual((tt["source"], tt["window"], tt["verdict"], tt["lane_name"]), ("live", True, "ssd", "iGPU lane"))
        self.assertEqual((tt["ms"], tt["disk_ms"], tt["lane_ms"], tt["spec"]), (40.1, 17.5, 1.8, (301, 410)))
        self.assertAlmostEqual(v["disk"]["mirror_share"], 2660 / 5920, places=3)
        self.assertAlmostEqual(v["disk"]["mirror_stat"], 952.0 / (1210.5 + 952.0), places=3)   # the engine's own count
        self.assertAlmostEqual(v["lane"]["gbs"], 3.0 * 9.4 * 2**20 / 0.0018 / 1e9, places=2)   # 3 experts x 9.4 MiB / 1.8 ms
        self.assertEqual(v["lane"]["kind"], "igpu")
        self.assertAlmostEqual(v["drafter"]["step"], 412 / 160, places=3)                         # tokens a verify step
        self.assertEqual(v["drafter"]["step_label"], "live")
        self.assertAlmostEqual(v["drafter"]["live"], 301 / 410, places=3)
        self.assertEqual(v["gpu"]["busiest"], 93)                # the iGPU lane's 64% is not the model's card
        self.assertEqual([c["role"] for c in v["cards"]], ["main", "igpu"])
        self.assertEqual(v["drafter"]["total"], 214 / 402)
        self.assertIn(("UNCENSOR", "alert"), v["badges"])

    def test_bench_only_for_the_model_running(self):
        # the r9700 fixture's --bench measured its own config: kept; another model's bench (an Ornith run left in
        # abyss-bench.txt while Qwen3.8 serves) is not shown beside this one
        self.assertEqual(view("r9700")["bench"]["decode"], 25.1)
        other = {"line": "Abyss v1.3.122 | Ornith-1.5-397B-32G128G | ...", "decode": 44.2, "prompt8k": 681.0,
                 "model": "Ornith-1.5-397B-32G128G", "config": "cfg_pc8.json"}
        cfg = {"file": "qwen38.cfg.json", "display": "Qwen3.8-Flash-Next-IQ3_S"}
        self.assertIsNone(M.bench_for(other, cfg, {"display_name": "Qwen3.8-Flash-Next-IQ3_S"}))
        self.assertIsNone(M.bench_for(other, {}, {}))                              # (nothing known here: hidden)
        self.assertIs(M.bench_for(other, {"file": "cfg_pc8.json"}, {}), other)      # the same config
        self.assertIs(M.bench_for(other, {}, {"display_name": "Ornith-1.5-397B-32G128G"}), other)   # the same model
        local = dict(F.local("r9700"), bench=other)
        v = M.derive(FakeLink("r9700"), local, log_of("r9700"), seen={})
        self.assertIsNone(v["bench"])
        self.assertNotIn("bench", "".join(str(x) for ln in D.speed_tile(v, "decode", 40) for x in ln))

    def test_the_drafter_s_kind(self):
        self.assertEqual(M.drafter_kind({"drafter": "none", "mtp": 1}), "mtp")     # the engine's "none": its own head
        self.assertEqual(M.drafter_kind({"drafter": "none", "mtp": 0}), None)
        self.assertEqual(M.drafter_kind({"drafter": "dspark"}), "dspark")
        self.assertEqual(M.drafter_kind({"drafter": "NextN"}), "nextn")
        self.assertEqual(M.drafter_kind({}), None)

    def test_old_swap_is_not_memory_pressure(self):
        seen = {}
        r = {"total": 123 * 2**30, "used": 11 * 2**30, "psi_some": 0.0, "psi_full": 0.0, "swap_used": 2.2 * 2**30}
        now = 1000.0
        for i in range(5):                                       # 2.2 GB of swap, still, PSI 0 / 0: no warning
            r["swap_growth"] = M.swap_growth(r["swap_used"], seen, now + i)
        v = {"gpu": {}, "cards": [], "pcie": None, "ram": r, "conn": "live", "remote": False, "url": None}
        self.assertEqual(r["swap_growth"], 0.0)
        self.assertFalse(M.memory_tight(r))
        self.assertFalse(any(t.lower().startswith(("memory is tight", "memory pressure")) for t, _ in M.banners(v, None, now + 5)))
        r["swap_growth"] = M.swap_growth(r["swap_used"] + 600 * 2**20, seen, now + 10)   # swap growing now: warned
        r["swap_used"] += 600 * 2**20
        self.assertTrue(M.memory_tight(r))
        self.assertTrue(any(t.startswith("memory is tight") for t, _ in M.banners(v, None, now + 10)))
        r["swap_growth"] = None                                   # PSI says tasks wait: warned too
        r["psi_some"] = 25.0
        self.assertTrue(M.memory_tight(r))
        self.assertIsNone(M.swap_growth(1.0, None, now))         # (no history kept: unknown)

    def test_the_verdict_is_bench_s_own(self):
        lg = log_of("r9700")
        v = view("r9700")
        self.assertEqual(v["tt"]["verdict"], R.bench_verdict(lg.decode_recs(), "iGPU lane")[0])
        self.assertEqual(v["tt"]["what"], R.advice("ssd", "hip")[0])

    def test_an_older_engine_falls_back_to_the_log(self):
        link = FakeLink("r9700")
        for k in list(link.metrics["tiers"]["now"]):          # (a 1.3.79 STAT: the six old keys only)
            if k in F.STAT_KEYS[6:]:
                del link.metrics["tiers"]["now"][k]
        for k in ("lane", "drafter", "expert_mib", "tail_skip", "defer", "mirror", "helper"):
            link.metrics["engine"].pop(k, None)
        for r in link.metrics["requests"]:
            r.pop("spec_rounds", None)
        v = M.derive(link, F.local("r9700"), log_of("r9700"), seen={})
        self.assertEqual((v["tt"]["source"], v["tt"]["verdict"], v["tt"]["lane_name"]), ("log", "ssd", "iGPU lane"))
        self.assertAlmostEqual(v["tt"]["disk_ms"], 12.8, places=1)
        self.assertIsNone(v["lane"]["gbs"])
        self.assertIsNone(v["disk"]["mirror_stat"])
        self.assertIsNone(v["drafter"]["step"])
        v = M.derive(link, {"remote": True}, None, seen={})   # ... and remote without the log: the hint says why
        self.assertIsNone(v["tt"])
        self.assertIn("1.3.80", v["tt_why"])

    def test_idle_shows_the_last_answer_s_stat_not_the_window(self):
        v = view("lowspec")
        self.assertEqual((v["tt"]["source"], v["tt"]["window"], v["tt"]["ms"], v["tt"]["disk_ms"]),
                         ("stat", False, 49.0, 24.6))
        self.assertEqual(v["tt"]["verdict"], "ssd")
        self.assertIn("last answer", D.tt_source(v["tt"]))

    def test_remote_split_has_no_token_time_yet_and_says_why(self):
        v = view("split")
        self.assertIsNone(v["tt"])
        self.assertIn("1.3.80", v["tt_why"])
        self.assertEqual(v["state"], "reading")
        self.assertEqual(v["prompt"]["label"], "live")
        self.assertEqual((v["prompt"]["reused"], v["prompt"]["disk_ms"]), (2048, 1840.0))   # the PP line's new fields
        self.assertIn("reused 2K  SSD 1.8 s", "\n".join(x.plain for x in D.speed_tile(v, "prompt", 70)))
        self.assertEqual(v["gpu"]["busiest"], 97)
        self.assertEqual((v["pcie"]["gen"], v["pcie"]["width"]), (3, 4))
        self.assertTrue(any("plain http" in t for t, _ in v["banners"]))
        self.assertTrue(any("sw_power_cap" in t for t, _ in v["banners"]))

    def test_phase2_stat_keys_make_the_verdict_live(self):
        link = FakeLink("split")
        link.metrics["tiers"]["now"].update(disk_ms=0.5, lane_ms=14.9, lane=180.0)
        v = M.derive(link, {"remote": True}, None, seen={})
        self.assertEqual((v["tt"]["source"], v["tt"]["verdict"]), ("stat", "cpu"))   # (reading: the last answer's)
        link.metrics["live"]["state"] = "generating"
        v = M.derive(link, {"remote": True}, None, seen={})
        self.assertEqual((v["tt"]["source"], v["tt"]["window"], v["tt"]["verdict"]), ("live", False, "cpu"))
        link.metrics["tiers"]["now"].update(w_ms_tok=16.0, w_disk_ms=0.2, w_lane_ms=3.0)
        v = M.derive(link, {"remote": True}, None, seen={})
        self.assertEqual((v["tt"]["window"], v["tt"]["verdict"], v["tt"]["lane_ms"]), (True, "gpu", 3.0))

    def test_a_remote_server_still_loading_says_how_far(self):
        v = view("starting")
        self.assertEqual((v["conn"], v["remote"]), ("loading", True))
        self.assertEqual((v["loading"]["frac"], v["loading"]["step"], v["loading"]["elapsed"], v["loading"]["gb"]),
                         (0.42, "warming", 61.0, (37.0, 88.0)))
        text = "\n".join(x.plain for x in D.overview(v, 100, 30, {}))
        self.assertIn("LOADING", text)
        self.assertIn("42%", text)
        self.assertIn("37.0 / 88.0 GB", text)
        self.assertIn("LOADING 42%", "\n".join(x.plain for x in D.header(v, 120)))

    def test_missing_is_dashes_never_zero(self):
        link = FakeLink("lowspec")
        m = link.metrics
        m["hardware"] = {"cpu": None}
        m["hardware_static"] = {}
        m["history"] = {}
        v = M.derive(link, {}, None, seen={})
        self.assertIsNone(v["gpu"]["busiest"])
        self.assertIsNone(v["gpu"]["power"])
        self.assertIsNone(v["pcie"])
        text = "\n".join(x.plain for x in D.tiles(v, 160))
        self.assertIn("--", text)
        self.assertNotIn(" 0%", text)
        self.assertNotIn("0 W", text)

    def test_lowspec_idle_shows_the_last_request_and_memory_trouble(self):
        v = view("lowspec")
        self.assertEqual((v["state"], v["decode"]["label"]), ("idle", "last request"))
        self.assertEqual(v["tt"]["verdict"], "ssd")
        self.assertTrue(any("memory is tight" in t for t, _ in v["banners"]))
        self.assertFalse(v["pcie"]["dropped"])                   # (idle Gen1: not judged)
        self.assertIn("SSD-bound", D.diagnose(v)[1][0])

    def test_a_pcie_drop_while_working_is_an_alert(self):
        seen = {}
        link = FakeLink("split")
        m = link.metrics
        for c in m["hardware"]["gpus"]:
            c.update(pcie_gen=4, pcie_width=4)
        v = M.derive(link, {"remote": True}, None, seen=seen)
        self.assertFalse(v["pcie"]["dropped"])
        for c in m["hardware"]["gpus"]:
            c.update(pcie_gen=1)
        v = M.derive(link, {"remote": True}, None, seen=seen)
        self.assertTrue(v["pcie"]["dropped"])
        self.assertTrue(any("PCIe fell to Gen1" in t for t, _ in v["banners"]))
        m["live"]["state"] = "idle"
        self.assertFalse(M.derive(link, {"remote": True}, None, seen=seen)["pcie"]["dropped"])

    def test_states_loading_stopped_error(self):
        v = view("loading")
        self.assertEqual(v["conn"], "loading")
        self.assertAlmostEqual(v["loading"]["frac"], 37 / 88, places=3)
        self.assertEqual(view("down")["conn"], "stopped")
        lg = M.EngineLog(None)
        lg.feed("ERR CUDA error: an illegal memory access was encountered", time.time())
        down = FakeLink("down")
        self.assertEqual(M.derive(down, {"is_local": True, "pid_alive": False}, lg)["conn"], "error")
        self.assertEqual(M.derive(down, {"remote": True}, None)["conn"], "disconnected")


class Log(unittest.TestCase):
    def test_followed_from_the_last_start_and_matched_by_time(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "engine.log"
            p.write_text("\n".join(F.LOG_LOWSPEC + F.LOG_R9700) + "\n")
            lg = M.EngineLog(p)
            lg.read()
            self.assertEqual(lg.lane, "iGPU lane")                 # (the last start's lane, not the one before)
            self.assertEqual(len(lg.recs), 6)                      # (but every request's stat line)
            self.assertTrue(all(r["t"] is None for r in lg.recs))
            with open(p, "a") as f:
                f.write(F._stat(30.0, 70.0, 0.0, 2.0, 2.0, 3.0, 50.0, 4.0) + "\nglm spec: 3 of 4 drafts accepted\n")
                f.write("half a line")
            t_end = time.time()
            self.assertEqual(lg.read(), 2)
            self.assertEqual(lg.recs[-1]["spec"], (3, 4))
            self.assertIs(lg.match({"time": t_end - 10, "duration_s": 10}), lg.recs[-1])
            self.assertIsNone(lg.match({"time": t_end - 600, "duration_s": 10}))
            p.write_text("short\n")                                # (rotated)
            lg.read()
            self.assertEqual(lg.last_line, "short")

    def test_mirror_split_and_lane_failure(self):
        lg = M.EngineLog(None)
        lg.feed("glm fast mirror CUDA0: 120.5 GB from the shards' drive, 98.0 GB from the mirror (45%)")
        lg.feed("glm fast: CUDA0 iGPU lane failed: hipErrorOutOfMemory - the CPU computes the lane", time.time())
        self.assertEqual(lg.mirror, (120.5, 98.0))
        self.assertEqual((lg.lane, lg.lane_state), ("CPU lane", "failed"))


class Drawing(unittest.TestCase):
    def test_every_view_every_scenario_exactly_fits(self):
        for sc in F.SCENARIOS:
            v = view(sc)
            for w, h in ((160, 50), (80, 24), (40, 12), (110, 30)):
                for i, layout in [(i, "speed") for i in range(len(D.VIEWS))] + [(0, "devices")]:
                    ui = {"view": i, "log": log_of(sc), "configs": F.CONFIGS, "sel": 1, "layout": layout}
                    lines = D.screen(v, w, h, ui)
                    self.assertEqual(len(lines), h, (sc, w, h, i, layout))
                    bad = [n for n, ln in enumerate(lines) if ln.width != w]
                    self.assertEqual(bad, [], (sc, w, h, i, layout, [lines[n].plain for n in bad]))
                lines = D.screen(v, w, h, {"view": 0, "help": True})
                self.assertEqual(len(lines), h)

    def test_the_header_names_the_model_with_its_quant_and_state(self):
        text = "\n".join(x.plain for x in D.header(view("r9700"), 160))
        self.assertIn("DeepSeek V4.1 Flash 1017 UD-IQ2_XXS", text)
        self.assertIn("[UNCENSOR]", text)
        self.assertIn("GENERATING 412/2048", text)
        for sc, word in (("loading", "LOADING 42%"), ("down", "STOPPED"), ("split", "READING PROMPT 6K/16K")):
            self.assertIn(word, "\n".join(x.plain for x in D.header(view(sc), 160)))

    def test_the_bottleneck_is_named_on_the_overview(self):
        text = "\n".join(x.plain for x in D.overview(view("r9700"), 160, 46, {}))
        self.assertIn("BOTTLENECK", text)
        self.assertIn("SSD WAIT 17.5 ms", text)          # the live window's
        self.assertIn("live · last ~64 tokens", text)
        self.assertIn("THROUGHPUT", text)
        self.assertIn("tok/step 2.58", text)
        text = "\n".join(x.plain for x in D.speed_view(view("r9700"), 160, 50, {}))
        self.assertIn("mirror split (engine) 44%", text)
        self.assertIn("GB/s", text)
        self.assertIn("16.4 GB/s", text)
        self.assertIn("tokens/step   2.58", text)

    def test_plain_output_colour_mono_and_none(self):
        lines = D.screen(view("r9700"), 80, 24, {"view": 0})
        self.assertIn("\x1b[38;2;", D.ansi(lines, "dark"))
        mono = D.ansi(lines, "mono")
        self.assertNotIn("38;2", mono)
        self.assertIn("\x1b[1;7m", mono)                           # (bold / reverse still)
        self.assertNotIn("\x1b", D.ansi(lines, "dark", color=False))


class Local(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, name, **kw):
        c = {"exe": str(self.root / "strata"), "args": ["--glm-pack", str(self.root / "m" / "pack"), "--max-context",
                                                         "65536"], "port": 8082, "backend": "hip",
             "log": str(self.root / f"{name}.log"), **kw}
        p = self.root / f"{name}.json"
        p.write_text(json.dumps(c))
        return p

    def test_configs_their_names_and_missing_files(self):
        (self.root / "m" / "pack").mkdir(parents=True)
        (self.root / "m" / "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00003.gguf").write_bytes(b"x")
        (self.root / "strata").write_bytes(b"x")
        a = self.config("abyss-a")
        (self.root / "abyss-a.shared-settings.json").write_text("{}")
        (self.root / "notes.json").write_text("{}")
        self.assertEqual(T.list_configs(self.root), [a])
        info = T.config_info(a)
        self.assertEqual((info["display"], info["quant"], info["ctx"], info["url"]),
                         ("DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS", "UD-IQ2_XXS", 65536, "http://127.0.0.1:8082"))
        self.assertEqual(info["missing"], [])
        os.remove(self.root / "strata")
        self.assertEqual(T.config_info(a)["missing"], ["engine"])

    def test_a_stale_pid_file_is_dropped(self):
        a = self.config("abyss-a")
        T.pid_path(a).write_text(json.dumps({"pid": os.getpid()}))   # (alive, but not an Abyss start of it)
        self.assertIsNone(T.read_pid(a))
        self.assertFalse(T.pid_path(a).exists())

    @unittest.skipIf(os.name == "nt", "the detached start's process group is tested on POSIX")
    def test_detached_start_and_stop(self):
        a = self.config("abyss-a")
        (self.root / "abyss.py").write_text(textwrap.dedent("""
            import signal, sys, time
            print("fake abyss", sys.argv[1:], flush=True)
            signal.signal(signal.SIGINT, lambda *x: sys.exit(0))
            time.sleep(60)
        """))
        info = T.start_detached(a, root=self.root)
        try:
            for _ in range(100):   # (until the child is up: a slow or emulated machine takes over 0.5 s)
                log = T.server_log_path(a)
                if log.exists() and "fake abyss" in log.read_text():
                    break
                time.sleep(0.1)
            self.assertTrue(T.alive(info["pid"]))
            self.assertEqual(os.getsid(info["pid"]), info["pid"])     # (a session of its own: survives our exit)
            self.assertEqual(T.read_pid(a)["pid"], info["pid"])
            self.assertEqual([c for c, _ in T.running(self.root)], [a])
            msg = T.stop(a, None, wait=10)
            self.assertEqual(msg, "stopped")
            self.assertFalse(T.pid_path(a).exists())
            self.assertIn("fake abyss", T.server_log_path(a).read_text())
            self.assertIn("--plain", T.server_log_path(a).read_text())
        finally:
            T.force_stop(info["pid"])

    def test_bench_summary_and_thermal_log(self):
        (self.root / "abyss-bench.txt").write_text("### Abyss benchmark\n\n" + F.local("r9700")["bench"]["line"] +
                                                   "\n\n- date: 2026-10-29 21:00; setup: abyss-dsv41-iq2xxs-r9700l.json,"
                                                   " context 65536\n")
        b = T.bench_summary(self.root)
        self.assertEqual((b["decode"], b["prompt8k"], b["bottleneck"], b["config"]),
                         (25.1, 781.0, "ssd", "abyss-dsv41-iq2xxs-r9700l.json"))
        th = T.ThermalLog(self.root)
        p = th.toggle()
        th.row(view("r9700"))
        th.toggle()
        rows = p.read_text().splitlines()
        self.assertTrue(rows[0].startswith("time,state,tok_s,cpu_c"))
        self.assertIn("g0_main_junction_c", rows[0])
        self.assertEqual(len(rows), 2)


class Cli(unittest.TestCase):
    def test_once_and_json_from_a_demo(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            A.main(["--demo", "r9700", "--json"], textual=False)
        v = json.loads(out.getvalue())
        self.assertEqual((v["conn"], v["quant"], v["tt"]["verdict"]), ("live", "UD-IQ2_XXS", "ssd"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            A.main(["--demo", "lowspec", "--once"], textual=False)
        self.assertIn("UD-IQ1_M", out.getvalue())

    def test_plain_http_with_a_key_warns(self):
        self.assertIn("ssh -L", A.remote_warning("http://192.168.1.20:8082", "ABYSS_API_KEY"))
        self.assertIsNone(A.remote_warning("http://127.0.0.1:8082", "ABYSS_API_KEY"))
        self.assertIsNone(A.remote_warning("https://abyss.example.ts.net", "ABYSS_API_KEY"))
        self.assertIsNone(A.remote_warning("http://192.168.1.20:8082", None))

    def test_the_key_comes_from_the_named_variable_only(self):
        with F.FakeServer("split", key="sekrit") as srv:
            os.environ["ABYSS_TEST_KEY"] = "sekrit"
            try:
                s = A.Session(url=srv.url, key_env="ABYSS_TEST_KEY", poll=False)
                self.assertEqual(s.link.poll(), "live")
                self.assertTrue(s.remote)
                self.assertIn("read-only", s.can_manage())
                self.assertNotIn("sekrit", json.dumps(s.ui(), default=str))
            finally:
                del os.environ["ABYSS_TEST_KEY"]


class ManyDevices(unittest.TestCase):
    """Several GPUs are the usual case (2-10 cards): per card its role, layers, pool, link against its own max; the
    busiest starred, a weak / dropped link in pink; any number of drives with their share of the expert reads."""

    def test_mixed_cards_each_with_its_own_max_and_limit(self):
        v = view("multi4")
        by = {c["index"]: c for c in v["cards"]}
        self.assertEqual(v["gpu"]["busiest_card"], 2)
        self.assertTrue(by[2]["busiest"])
        self.assertEqual((by[0]["layers"], by[0]["pool_gb"]), ((0, 14), 24.6))
        self.assertTrue(by[3]["behind"])           # (a V100 on x8, its twin on x16)
        self.assertFalse(any(c["weak"] for c in v["cards"]))
        text = "\n".join(x.plain for x in D.card_rows(v, 156, full=True))
        for want in ("G4x4/5x16", "G4x16/4x16", "G3x8/3x16", "430/575", "330/350", "240/250", "L0-13", "L36-42",
                     "V100-32GB"):
            self.assertIn(want, text)
        self.assertIsNone(by[0]["tt_share"])       # (phase 2: --, never 0)
        self.assertIn("--", text)

    def test_a_riser_at_x1_is_the_weak_link(self):
        v = view("multi8")
        self.assertEqual(v["gpu"]["weak_cards"], [7])
        self.assertEqual(v["gpu"]["slowest_link_card"], 7)
        self.assertTrue(any("GPU7 on Gen1 x1" in t for t, _ in v["banners"]))
        self.assertIsNone(v["cards"][0]["layers"])     # (remote: no engine log)

    def test_a_card_dropping_while_working_is_named(self):
        seen = {}
        view("multi4", seen=seen)
        m = F.metrics("multi4")
        m["hardware"]["gpus"][1]["pcie_gen"] = 1
        link = FakeLink("multi4")
        link.metrics = m
        v = M.derive(link, F.local("multi4"), log_of("multi4"), seen=seen)
        self.assertEqual(v["gpu"]["dropped_cards"], [1])
        self.assertTrue(v["pcie"]["dropped"])
        self.assertTrue(any(t.startswith("GPU1 PCIe fell to Gen1 x16") for t, _ in v["banners"]))

    def test_drives_of_any_number_share_the_reads(self):
        d = view("multi8")["disk"]
        self.assertEqual([x["name"] for x in d["drives"]], ["nvme0n1", "nvme1n1", "sda"])
        self.assertAlmostEqual(d["drives"][0]["share"] + d["drives"][1]["share"], 1.0)
        self.assertIsNone(d["drives"][2]["share"])
        self.assertEqual(d["shard_n"], 2)

    def test_ten_cards_wrap_wide_and_summarise_narrow(self):
        v = view("multi10")
        wide = "\n".join(x.plain for x in D.screen(v, 160, 50, {"view": 0, "layout": "devices"}))
        body = "\n".join(x.plain for x in D.devices_overview(v, 160, 200, {}))
        for i in range(10):
            self.assertIn(f"GPU{i} RTX 3090", body)
        self.assertIn("SSD2 nvme1n1", body)
        self.assertIn("CPU EPYC 7413", body)
        self.assertIn("▣ DEVICES", wide)
        narrow = "\n".join(x.plain for x in D.screen(v, 80, 24, {"view": 0, "layout": "devices"}))
        self.assertIn("10 GPUs · busy GPU3 95%", narrow)
        self.assertIn("link GPU9 G2x1", narrow)
        self.assertIn("more · 5 Hardware", narrow)
        speed = "\n".join(x.plain for x in D.screen(v, 80, 24, {"view": 0, "layout": "speed"}))
        self.assertIn("10 GPUs", speed)
        self.assertIn("2 SSDs", speed)

    def test_memory_and_hardware_list_every_card(self):
        v = view("multi10")
        mem = "\n".join(x.plain for x in D.memory_view(v, 160, 80, {}))
        hw = "\n".join(x.plain for x in D.hardware_view(v, 160, 80, {}))
        for i in range(10):
            self.assertIn(f"GPU{i} ", mem)
            self.assertIn(f"GPU{i} ", hw)
        self.assertIn("WEAK LINK", hw)


class Layout(unittest.TestCase):
    def setUp(self):
        with contextlib.suppress(FileNotFoundError):
            A.state_path().unlink()

    def session(self, demo, **kw):
        s = A.Session(demo=demo, poll=False, **kw)
        self.addCleanup(s.close)
        return s

    def test_three_or_more_cards_start_on_devices(self):
        self.assertEqual(self.session("multi4").pick_layout(), "devices")
        self.assertEqual(self.session("multi10").pick_layout(), "devices")
        self.assertEqual(self.session("r9700").pick_layout(), "speed")
        self.assertEqual(self.session("multi8").pick_layout(view("multi8")), "devices")   # (remote: the server's)

    def test_l_switches_and_is_remembered(self):
        s = self.session("r9700")
        self.assertEqual(s.cycle_layout(), "devices")
        self.assertEqual(json.loads(A.state_path().read_text())["layout"], "devices")
        self.assertEqual(self.session("r9700").pick_layout(), "devices")
        self.assertEqual(self.session("multi4", layout="speed").pick_layout(), "speed")   # (--layout wins)

    def test_a_broken_state_file_is_ignored(self):
        A.state_path().parent.mkdir(parents=True, exist_ok=True)
        A.state_path().write_text("{not json")
        self.assertEqual(self.session("r9700").pick_layout(), "speed")


SHOTS = [("r9700", 0), ("r9700", 1), ("r9700", 2), ("r9700", 3), ("r9700", 4), ("r9700", 5), ("r9700", 6),
         ("r9700", 7), ("lowspec", 0), ("lowspec", 2), ("split", 0), ("split", 4), ("loading", 0), ("down", 0),
         ("multi4", 4), ("multi10", 2), ("multi10", 4), ("starting", 0)]


@needs_textual
class Screen(unittest.TestCase):
    def run_app(self, coro, sc="r9700", size=(160, 50), layout="speed"):
        s = A.Session(demo=sc, interval=1.0, layout=layout)

        async def main():
            app = TA.TerminalApp(s, anim=False)
            async with app.run_test(size=size) as pilot:
                for _ in range(40):                                   # (the first poll)
                    if s.poller.polls:
                        break
                    await pilot.pause(0.05)
                app.redraw()
                await pilot.pause(0.1)
                await coro(app, pilot)
        try:
            asyncio.run(main())
        finally:
            s.close()

    def test_views_by_key_and_the_text_on_screen(self):
        async def go(app, pilot):
            self.assertIn("UD-IQ2_XXS", app.plain())
            self.assertIn("TOKEN TIME", app.plain())
            await pilot.press("2")
            self.assertIn("WHERE A TOKEN'S TIME GOES", app.plain())
            await pilot.press("right")
            self.assertIn("WHERE THE EXPERTS LIVE", app.plain())
            await pilot.press("8")
            self.assertIn("MODELS", app.plain())
            await pilot.press("question_mark")
            self.assertIn("KEYS", app.plain())
            await pilot.press("escape")
            await pilot.press("space")
            self.assertIn("FROZEN", app.plain())
        self.run_app(go)

    def test_80x24_and_40x12(self):
        for size in ((80, 24), (40, 12)):
            async def go(app, pilot):
                head = app.plain().splitlines()[0]
                self.assertLessEqual(len(head), size[0])
                self.assertIn("UD-IQ2_XXS" if size[0] >= 60 else "DeepSeek", app.plain())
            self.run_app(go, size=size)

    def test_quit_asks_when_this_terminal_started_the_server(self):
        async def go(app, pilot):
            app.s.started_by_us = True
            await pilot.press("q")
            self.assertEqual(type(app.screen).__name__, "Choice")
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause(0.1)
            self.assertEqual(app.result, "stop")
        self.run_app(go)

    def test_l_switches_the_overview_layout(self):
        async def go(app, pilot):
            self.assertIn("▣ SPEED", app.plain())
            await pilot.press("l")
            await pilot.pause(0.1)
            self.assertIn("▣ DEVICES", app.plain())
            self.assertIn("GPU0", app.plain())
        self.run_app(go)

    def test_quit_without_asking_otherwise(self):
        async def go(app, pilot):
            await pilot.press("q")
            await pilot.pause(0.1)
            self.assertEqual(app.result, "keep")
        self.run_app(go)

    @unittest.skipUnless(os.environ.get("ABYSS_TERMINAL_SHOTS"), "ABYSS_TERMINAL_SHOTS=<folder> writes the screenshots")
    def test_screenshots(self):
        out = Path(os.environ["ABYSS_TERMINAL_SHOTS"])
        out.mkdir(parents=True, exist_ok=True)
        for size in ((160, 50), (80, 24)):
            for sc, i in SHOTS:
                if size == (80, 24) and i not in (0, 1, 3, 7):
                    continue

                async def go(app, pilot, sc=sc, i=i):
                    app.action_go(i)
                    await pilot.pause(0.1)
                    name = D.VIEWS[i][1].lower()
                    app.save_screenshot(f"{sc}-{name}-{size[0]}x{size[1]}.svg", str(out))
                self.run_app(go, sc=sc, size=size)
            for sc in ("r9700", "split", "multi4", "multi8", "multi10", "lowspec"):    # (both Overview layouts)
                for layout in D.LAYOUTS:
                    async def go(app, pilot, sc=sc, layout=layout):
                        await pilot.pause(0.1)
                        app.save_screenshot(f"{sc}-layout-{layout}-{size[0]}x{size[1]}.svg", str(out))
                    self.run_app(go, sc=sc, size=size, layout=layout)


def tearDownModule():
    _STATE.cleanup()


if __name__ == "__main__":
    unittest.main()
