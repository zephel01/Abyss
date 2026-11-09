"""serve/test_dashboard.py - what the web app's Settings and Monitor ask the server for: the context size (its range, a
reload, the 503 while it reloads, the run config it is saved into), the engine's measured context cost, and the report.
Against the mock engine (no GPU, no pack).

    python -m unittest serve.test_dashboard -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve import server as SV  # noqa: E402
from serve.server import CONTEXT_MIN, MAYA_VERSION, ByteTokenizer, MockEngine, Service, StrataEngine, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Dashboard(unittest.TestCase):
    def setUp(self):
        os.environ["STRATA_MOCK_RELOAD_S"] = "0.3"
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, "</think>\n\nhello", max_context=32768)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Path(self.tmp.name) / "maya-test.json"
        self.cfg.write_text(json.dumps({"exe": "strata", "args": ["--glm-pack", "/x/pack", "--max-context", "32768"],
                                        "env": {"STRATA_GLM_RAM_GB": "20", "MY_API_KEY": "secret"}}, indent=1))
        self.svc.config_path = str(self.cfg)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def req(self, method, path, body=None, headers=None):
        h = {"Content-Type": "application/json"} if body is not None else {}
        h.update(headers or {})
        r = urllib.request.Request(self.base + path, method=method, headers=h,
                                   data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                raw = resp.read()
                return resp.status, resp.headers, raw
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.headers, e.read()

    def wait_reload(self):
        for _ in range(100):
            s, _, b = self.req("GET", "/api/context")
            rl = json.loads(b)["reload"]
            if rl and rl["state"] in ("done", "failed"):
                return json.loads(b)
            time.sleep(0.05)
        self.fail("the reload did not end")

    def test_health_has_the_version(self):
        s, _, b = self.req("GET", "/health")
        self.assertEqual(json.loads(b)["version"], MAYA_VERSION)

    def test_context_range(self):
        s, _, b = self.req("GET", "/api/context")
        j = json.loads(b)
        self.assertEqual((s, j["context"], j["min"]), (200, 32768, CONTEXT_MIN))
        self.assertGreaterEqual(j["max"], 32768)

    def test_reload_changes_the_context_and_saves_it(self):
        s, _, b = self.req("POST", "/api/context", {"max_context": 65536})
        self.assertEqual(s, 202)
        j = self.wait_reload()
        self.assertEqual((j["reload"]["state"], j["context"]), ("done", 65536))
        args = json.loads(self.cfg.read_text())["args"]
        self.assertEqual(args[args.index("--max-context") + 1], "65536")
        self.assertEqual(json.loads(self.cfg.read_text())["env"]["STRATA_GLM_RAM_GB"], "20")   # the rest kept

    def test_chat_gets_503_while_reloading(self):
        os.environ["STRATA_MOCK_RELOAD_S"] = "1.5"
        self.req("POST", "/api/context", {"max_context": 16384})
        s, h, b = self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 503)
        self.assertEqual(h.get("Retry-After"), "30")
        self.assertIn("reloading", json.loads(b)["error"]["message"])
        s, _, b = self.req("GET", "/health")
        self.assertEqual(json.loads(b)["status"], "reloading")
        self.wait_reload()
        s, _, _ = self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 200)

    def test_out_of_range_and_foreign_pages_are_refused(self):
        s, _, _ = self.req("POST", "/api/context", {"max_context": 100})
        self.assertEqual(s, 400)
        s, _, _ = self.req("POST", "/api/context", {"max_context": 65536}, {"Origin": "http://evil.example"})
        self.assertEqual(s, 403)
        r = urllib.request.Request(self.base + "/api/context", method="POST", data=b"max_context=65536",
                                   headers={"Content-Type": "application/x-www-form-urlencoded"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(r, timeout=10)
        self.assertEqual(cm.exception.code, 415)
        cm.exception.close()

    def test_report_leaves_secrets_out(self):
        s, h, b = self.req("GET", "/api/report")
        text = b.decode()
        self.assertEqual(s, 200)
        self.assertTrue(text.startswith("Abyss "), text[:80])
        self.assertIn("STRATA_GLM_RAM_GB", text)
        self.assertNotIn("secret", text)

    def test_token_count_matches_the_prompt(self):
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "Hi! How can I help?"}, {"role": "user", "content": "count me"}]
        s, _, b = self.req("POST", "/api/tokens", {"messages": msgs, "reasoning_effort": "high"})
        j = json.loads(b)
        from serve.frontend import openai_to_messages
        m, tools, kw = openai_to_messages({"messages": msgs, "reasoning_effort": "high"})
        ids, _, _ = self.svc.prepare(m, tools, kw)
        self.assertEqual((s, j["tokens"], j["approximate"]), (200, len(ids), False))
        s, _, b = self.req("POST", "/api/tokens", {"messages": msgs + [{"role": "user", "content": [
            {"type": "text", "text": "and this"}, {"type": "image_url", "image_url": {"url": "data:,"}}]}]})
        self.assertTrue(json.loads(b)["approximate"])

    def test_abyss_terminal_reads_this_metrics(self):
        """Abyss Terminal (tools/terminal_model.py) against this server's own /metrics: attached, the model named,
        every view drawn to its size - the server and the Terminal agree on the fields."""
        sys.path.insert(0, str(ROOT / "tools"))
        import terminal_draw as D
        import terminal_model as M
        self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        link = M.Link(self.base)
        self.assertEqual(link.poll(), "live")
        v = M.derive(link, {"remote": True}, None, seen={})
        self.assertEqual((v["conn"], v["state"]), ("live", "idle"))
        self.assertEqual(v["ctx"], 32768)
        self.assertEqual(v["requests"][0]["api"], "openai")
        for i in range(len(D.VIEWS)):
            lines = D.screen(v, 80, 24, {"view": i})
            self.assertEqual({ln.width for ln in lines}, {80})

    def test_requests_record_their_api(self):
        self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        s, _, b = self.req("GET", "/metrics")
        self.assertEqual(json.loads(b)["requests"][0]["api"], "openai")


class ContextCost(unittest.TestCase):
    """The GLM engine's start lines give what its context costs in VRAM, summed over its GPUs."""

    def test_parsed_from_the_start_log(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.write_text("old start\nglm fast: CUDA0 VRAM before the expert pool: 4.0 of 31.7 GB in use - dense weights 3.1, "
                           "state/KV (16384 ctx) 0.40, activations 0.1\n"
                           "this start\nglm fast: CUDA0 VRAM before the expert pool: 4.4 of 31.7 GB in use - dense weights 3.1, "
                           "state/KV (32768 ctx) 0.66, activations 0.1\n"
                           "glm fast: CUDA1 VRAM before the expert pool: 4.3 of 31.7 GB in use - dense weights 2.9, "
                           "state/KV (32768 ctx) 0.65, activations 0.1\n")
            eng = StrataEngine.__new__(StrataEngine)
            eng.log_path, eng.info = str(log), {}
            eng._context_cost(log.read_text().index("this start"))
            self.assertEqual((eng.info["kv_ctx"], eng.info["kv_gb"]), (32768, 1.31))

    def test_set_context_rewrites_the_command(self):
        eng = StrataEngine.__new__(StrataEngine)
        eng.spawn = ("strata", ["--glm-pack", "p", "--max-context", "32768"], None, None, None)
        eng.set_context(131072)
        self.assertEqual(eng.spawn[1], ["--glm-pack", "p", "--max-context", "131072"])


class GpuRelease(unittest.TestCase):
    """A reload starts the new engine only once the GPUs have dropped the old one (its memory is freed after it ends)."""

    def test_waits_until_no_gpu_lists_the_engine(self):
        from unittest import mock
        from serve import server as SV, telemetry as T
        seen = [{4242, 7}, {4242, 7}, {7}]
        with mock.patch.object(T, "gpu_pids", side_effect=lambda: seen.pop(0) if len(seen) > 1 else seen[0]), \
                mock.patch.object(SV.time, "sleep") as sleep:
            SV.wait_gpu_release(4242, timeout=30)
        self.assertEqual(seen, [{7}])
        self.assertEqual(sleep.call_count, 2)

    def test_without_nvml_a_short_wait(self):
        from unittest import mock
        from serve import server as SV, telemetry as T
        with mock.patch.object(T, "gpu_pids", return_value=None), mock.patch.object(SV.time, "sleep") as sleep:
            SV.wait_gpu_release(4242)
        sleep.assert_called_once()


class AmdTelemetry(unittest.TestCase):  # (serve/test_telemetry.py has the rest of the readings)
    """Issue #14: a HIP run reads its AMD card from sysfs, not NVML's card 0 (an NVIDIA display card beside it)."""

    def test_reads_the_amd_card_from_sysfs(self):
        from serve import telemetry as T
        with tempfile.TemporaryDirectory() as t:
            dev = Path(t, "0000:c7:00.0")
            (dev / "hwmon" / "hwmon3").mkdir(parents=True)
            for name, v in (("gpu_busy_percent", "87"), ("mem_info_vram_used", str(30 << 30)),
                            ("mem_info_vram_total", str(32 << 30)), ("current_link_speed", "16.0 GT/s PCIe"),
                            ("max_link_speed", "32.0 GT/s PCIe"), ("current_link_width", "4"),
                            ("hwmon/hwmon3/temp1_input", "61000"), ("hwmon/hwmon3/power1_average", "212000000"),
                            ("hwmon/hwmon3/power1_cap", "300000000")):
                (dev / name).write_text(v + "\n")
            g = T._AmdSysfs("0000:c7:00.0", "gfx1201", base=t)
            self.assertTrue(g.ok())
            self.assertEqual(g.name(), "AMD Radeon (gfx1201)")      # no product_name file: the arch
            r = g.read()
            self.assertEqual((r["util"], r["mem_total"], r["temp"], r["power"], r["power_limit"]),
                             (87, 32 << 30, 61.0, 212.0, 300.0))
            self.assertEqual((r["pcie_gen"], r["pcie_gen_max"], r["pcie_width"]), (4, 5, 4))
            (dev / "product_name").write_text("AMD Radeon AI PRO R9700\n")
            self.assertEqual(g.name(), "AMD Radeon AI PRO R9700")
            self.assertFalse(T._AmdSysfs("0000:01:00.0", base=t).ok())   # not an amdgpu card
            self.assertFalse(T._AmdSysfs(None).ok())

    def test_a_hip_run_never_opens_nvml(self):
        from unittest import mock
        from serve import telemetry as T
        with mock.patch.object(T, "_Nvml") as nvml:
            tel = T.Telemetry(gpu_index=0, amd=[(0, None, "gfx1201")])
        nvml.assert_not_called()
        self.assertIsNone(tel.static["gpu_name"])



class TerminalFields(unittest.TestCase):
    """Phase 1.5 of Abyss Terminal: what /metrics adds for it - the run's name with its quant, drafts per request and in
    total, every STAT key's series, and the cards' roles from the run config."""

    def test_label_drafts_and_tier_series(self):
        from serve import server as SV
        tok = ByteTokenizer()

        class Drafting(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                try:
                    yield from super().generate(ids, max_new, sampling, cancel, embeddings)
                finally:                                      # (the server closes it at the end-of-turn token)
                    self.last = {"generated": 3, "prompt_tokens": len(ids), "prompt_ms": 10.0, "decode_ms": 30.0,
                                 "finish": "stop", "drafts_accepted": 5, "drafts_offered": 8, "reused": 0}

        eng = Drafting(tok, "</think>\n\nhi")
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.label = {"display_name": "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS", "quant": "UD-IQ2_XXS",
                     "config": "abyss-dsv41.json", "backend": "hip"}
        for _ in range(2):
            list(svc.run(tok.encode("hello"), False, None, 16, {}, SV.threading.Event()))
        eng.stat_history = [{"tok_s": 30.0, "disk_ms": 12.5, "lane": "igpu", "time": 1.0}, {"tok_s": 31.0, "time": 2.0}]
        eng.stat = eng.stat_history[-1]
        m = svc.metrics()
        self.assertEqual((m["engine"]["display_name"], m["engine"]["quant"], m["engine"]["backend"]),
                         ("DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS", "UD-IQ2_XXS", "hip"))
        self.assertEqual((m["requests"][0]["drafts_accepted"], m["requests"][0]["drafts_offered"]), (5, 8))
        self.assertEqual((m["totals"]["drafts_accepted"], m["totals"]["drafts_offered"]), (10, 16))
        h = m["tiers"]["history"]
        self.assertEqual(h["disk_ms"], [12.5, None])          # a new numeric key gets its series, None where absent
        self.assertEqual(h["tok_s"], [30.0, 31.0])
        self.assertNotIn("lane", h)                           # text keys stay in tiers.now only
        self.assertIn("promo", h)                             # the six old series are always there

    def test_roles_igpu_and_drives_from_the_config(self):
        from unittest import mock
        from serve import server as SV
        self.assertEqual(SV.gpu_roles({"env": {"STRATA_GLM_EXPERT_HELPER": "1"}}, [0, 1]), {0: "main", 1: "helper"})
        self.assertEqual(SV.gpu_roles({}, [0, 2]), {0: "main", 2: "split"})
        nodes = [{"index": 0, "arch": "gfx1201", "igpu": False, "pci": "0000:c7:00.0"},
                 {"index": 1, "arch": "gfx1151", "igpu": True, "pci": "0000:c8:00.0"}]
        with mock.patch.object(SV, "amd_gpu_nodes", return_value=nodes):
            self.assertEqual(SV.igpu_lane_device({"backend": "hip"}, [0]), (1, "0000:c8:00.0", "gfx1151"))
            self.assertEqual(SV.igpu_lane_device({}, [0]), (None, "0000:c8:00.0", "gfx1151"))     # CUDA: no HIP number
            self.assertIsNone(SV.igpu_lane_device({"backend": "hip", "env": {"STRATA_GLM_IGPU": "0"}}, [0]))
            self.assertIsNone(SV.igpu_lane_device({"backend": "hip"}, [1]))                       # the model's own card
        self.assertEqual(SV.model_drives({"env": {"STRATA_GLM_MIRROR": "/ssd2/m:/ssd3/m"}}, "/m"),
                         [("model", "/m"), ("mirror", "/ssd2/m"), ("mirror", "/ssd3/m")])

    def test_model_label(self):
        from serve import server as SV
        with tempfile.TemporaryDirectory() as d:
            for i in (1, 2):
                Path(d, f"DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-0000{i}-of-00002.gguf").write_bytes(b"x")
            self.assertEqual(SV.model_label(d), ("DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS", "UD-IQ2_XXS"))
        with tempfile.TemporaryDirectory() as d:
            Path(d, "Qwen3.5-122B-IQ3_XXS-B2.gguf").write_bytes(b"x")
            self.assertEqual(SV.model_label(d), ("Qwen3.5-122B-IQ3_XXS-B2", "IQ3_XXS"))
            self.assertEqual(SV.model_label(Path(d, "none")), (None, None))
        self.assertEqual(SV.model_label(None), (None, None))


class Shutdown(unittest.TestCase):
    """POST /api/shutdown: allowed with an API key or from this PC, as JSON from no foreign page; the request in flight is
    cut first, and an engine that does not stop it is killed."""

    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nhello"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, body=b"{}", headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        r = urllib.request.Request(self.base + "/api/shutdown", method="POST", data=body, headers=h)
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read() or b"{}")

    def test_from_this_pc_without_a_key(self):
        s, j = self.post(b'{"force": true}')
        self.assertEqual((s, j["shutdown"]["force"], j["shutdown"]["request_in_flight"]), (202, True, False))
        self.assertTrue(self.svc.quit.is_set())
        self.assertTrue(self.svc.quit_force)

    def test_refused(self):
        from unittest import mock
        from serve import server as SV
        self.assertEqual(self.post(headers={"Content-Type": "text/plain"})[0], 415)        # a form can't send it
        self.assertEqual(self.post(headers={"Origin": "http://evil.example"})[0], 403)    # nor another site's page
        with mock.patch.object(SV, "is_loopback", return_value=False):
            s, j = self.post()
            self.assertEqual(s, 403)
            self.assertIn("API key", j["error"]["message"])
            self.svc.api_key = "k"
            self.assertEqual(self.post()[0], 401)                                          # a key: it must be sent
            self.assertEqual(self.post(headers={"Authorization": "Bearer k"})[0], 202)     # ... and then from anywhere
        self.svc.quit.clear()
        self.svc.allow_shutdown = False
        self.assertEqual(self.post(headers={"Authorization": "Bearer k"})[0], 403)
        self.assertFalse(self.svc.quit.is_set())

    def test_loopback(self):
        from serve.server import is_loopback
        for a, want in (("127.0.0.1", True), ("127.8.0.1", True), ("::1", True), ("::ffff:127.0.0.1", True),
                        ("100.64.1.2", False), ("192.168.1.5", False), ("fe80::1%eth0", False), ("x", False)):
            self.assertEqual(is_loopback(a), want, a)


class CutInFlight(unittest.TestCase):
    class Engine:
        def __init__(self, svc, stops):
            self.svc, self.stops, self.ended = svc, stops, None

        def stop(self):
            if self.stops:                                    # a working engine: the request ends at its next step
                self.svc.fifo.release()

        def _end(self, why):                                  # a killed engine: the request ends with an error
            self.ended = why
            if not self.stops:
                self.svc.fifo.release()

    def svc(self):
        from serve import server as SV
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "x"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.fifo.acquire()                                    # a request holds the engine
        svc.status["busy"] = True
        return SV, svc

    def test_a_working_engine_stops_the_request(self):
        SV, svc = self.svc()
        eng = self.Engine(svc, stops=True)
        self.assertEqual(SV.cut_in_flight(svc, eng, wait_s=2), "stopped")
        self.assertIsNone(eng.ended)

    def test_a_runaway_engine_is_killed(self):
        SV, svc = self.svc()
        eng = self.Engine(svc, stops=False)
        t = time.time()
        self.assertEqual(SV.cut_in_flight(svc, eng, wait_s=0.3), "killed")
        self.assertIsNotNone(eng.ended)
        self.assertLess(time.time() - t, 5)

    def test_idle_and_force(self):
        from serve import server as SV
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "x"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        eng = self.Engine(svc, stops=True)
        self.assertEqual(SV.cut_in_flight(svc, eng), "idle")
        self.assertEqual(SV.cut_in_flight(svc, eng, force=True), "killed")       # force: ended even when idle
        self.assertIsNotNone(eng.ended)


class ShutdownProcess(unittest.TestCase):
    """The whole server ends on POST /api/shutdown (the mock engine: no GPU)."""

    def test_the_server_exits(self):
        import socket
        import subprocess
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        p = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "mock", "--port", str(port)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT))
        try:
            for _ in range(100):                       # (the port answers "loading" before the engine is in)
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as h:
                        if json.loads(h.read()).get("status") == "ok":
                            break
                except OSError:
                    pass
                time.sleep(0.1)
            r = urllib.request.Request(f"http://127.0.0.1:{port}/api/shutdown", method="POST", data=b"{}",
                                       headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=10) as resp:
                self.assertEqual(resp.status, 202)
            self.assertEqual(p.wait(timeout=20), 0)
            self.assertIn("asked over HTTP", p.stdout.read())
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
            p.stdout.close()

class EarlyBind(unittest.TestCase):
    """The port is bound before the engine loads (Abyss Terminal phase 2, H1): GET /health says "loading" and how
    far, everything else gets 503 + Retry-After, and serve(svc, httpd=...) puts the real handler in on the same
    socket."""

    def setUp(self):
        self.httpd = SV.bind_early("127.0.0.1", 0, {"model": "dsv41", "api_key": "k", "config": "abyss-x.json"})
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def get(self, path, method="GET"):
        r = urllib.request.Request(self.base + path, method=method, data=b"{}" if method == "POST" else None,
                                   headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                return resp.status, resp.headers, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.headers, json.loads(e.read() or b"{}")

    def test_loading_then_the_real_server_on_the_same_socket(self):
        SV.LOADING.update(stage="weights", text="starting the engine: reading the model's weights", parts={},
                          started=time.time() - 12)
        s, _, h = self.get("/health")
        self.assertEqual((s, h["status"], h["service"], h["model"], h["api_key"], h["config"]),
                         (503, "loading", "abyss", "dsv41", True, "abyss-x.json"))
        self.assertEqual(h["load"]["stage"], "weights")
        self.assertIsNone(h["load"]["pct"])
        self.assertGreaterEqual(h["load"]["elapsed_s"], 12)
        s, hd, _ = self.get("/health")
        self.assertEqual(hd.get("Retry-After"), "5")   # (a "200 = ready" health check waits; Abyss Terminal reads the body)
        SV.load_line("glm fast: CUDA0 warming the expert tiers 50% (44.0 of 88.0 GB, 9 s)")
        SV.load_line("glm fast: CUDA1 warming the expert tiers 25% (3.0 of 12.0 GB, 2 s)")
        s, _, h = self.get("/health")
        self.assertEqual((h["load"]["stage"], h["load"]["pct"], h["load"]["done_gb"], h["load"]["total_gb"]),
                         ("warming", 47.0, 47.0, 100.0))
        for path, method in (("/metrics", "GET"), ("/v1/models", "GET"), ("/v1/chat/completions", "POST"),
                             ("/api/shutdown", "POST")):
            s, hd, body = self.get(path, method)
            self.assertEqual((path, s, hd.get("Retry-After")), (path, 503, "15"))
            self.assertIn("loading", body["error"]["message"])
        # the engine is ready: the real handler takes over, the socket stays
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nhello"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        same = serve(svc, httpd=self.httpd)
        self.assertIs(same, self.httpd)
        s, _, h = self.get("/health")
        self.assertEqual((s, h["status"], h["load"]), (200, "ok", None))
        s, _, m = self.get("/metrics")
        self.assertEqual((s, m["live"]["state"]), (200, "idle"))

    def test_a_taken_port_is_an_error(self):
        with self.assertRaises(OSError):
            SV.bind_early("127.0.0.1", self.httpd.server_address[1])


class LogTail(unittest.TestCase):
    """GET /api/log (Abyss Terminal on another PC): the engine log's tail by byte offset, whole lines, secrets masked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "engine.log"
        self.path.write_text("glm fast: CUDA0 VRAM before the expert pool\n"
                             "server api_key=hunter22 Bearer tok-abcdef\n"
                             "glm stat: decode 41.5 ms/tok (24.1 tok/s)\n"
                             "last line without newline", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_tail_then_continue_by_offset(self):
        size = self.path.stat().st_size
        t = SV.log_tail(str(self.path), None, 70, "hunter22")
        self.assertEqual((t["size"], t["more"]), (size, True))
        self.assertTrue(t["text"].startswith("glm stat"), t)            # (a tail starts at a whole line)
        self.assertEqual(t["next"], size - len("last line without newline"))
        self.assertNotIn("hunter22", SV.log_tail(str(self.path), 0, 4096, "hunter22")["text"])
        whole = SV.log_tail(str(self.path), 0, 4096, "")
        self.assertIn("api_key=***", whole["text"])                      # (named keys masked without the value known)
        self.assertIn("Bearer ***", whole["text"])
        self.assertEqual(whole["next"], size - len("last line without newline"))   # (an unfinished line waits)
        rest = SV.log_tail(str(self.path), whole["next"], 4096, "")
        self.assertEqual((rest["text"], rest["more"]), ("last line without newline", False))
        self.assertIn("error", SV.log_tail(None, None, 4096, ""))
        self.assertIn("error", SV.log_tail(str(self.path) + ".missing", None, 4096, ""))

    def test_the_endpoint_needs_the_key(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nhello"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.api_key, svc.log_path = "k1", str(self.path)
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            r = urllib.request.Request(base + "/api/log?max=4096")
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(r, timeout=10)
            cm.exception.close()
            self.assertEqual(cm.exception.code, 401)
            r = urllib.request.Request(base + "/api/log?since=0&max=4096", headers={"Authorization": "Bearer k1"})
            with urllib.request.urlopen(r, timeout=10) as resp:
                t = json.loads(resp.read())
            self.assertEqual(t["path"], "engine.log")
            self.assertIn("glm stat", t["text"])
            self.assertNotIn("hunter22", t["text"])
        finally:
            httpd.shutdown()
            httpd.server_close()


class Phase2Lines(unittest.TestCase):
    """The engine's phase-2 lines parsed (no engine run): STAT's new keys are numbers in `stat` and get a series in
    /metrics, DONE's 11th column, PP's reused tokens and disk wait - and the older engine's shorter lines still go."""

    def engine(self):
        e = StrataEngine.__new__(StrataEngine)
        e.last, e.info, e.max_context = {}, {}, 4096
        return e

    def test_stat_new_and_old(self):
        e = self.engine()
        e._parse_stat("STAT tok_s=24.10 ms_tok=41.50 vram_hit=0.6356 ram_fetch=0.000 disk=5.840 promo=6.020 "
                      "vram_used=2271 vram_slots=2386 ram_used=6258 ram_slots=6301 ram_gb=66.1 vram_gb=21.9 "
                      "disk_ms=18.200 disk_waits=4.100 miss_ms=19.000 lane=3.000 lane_ms=1.900 tail_skip=0.000 "
                      "mirror_mb0=1200.5 mirror_mb1=980.0 w_ms_tok=40.10 w_disk_ms=17.500 w_lane_ms=1.800 "
                      "spec_acc=120 spec_offered=200 spec_rounds=90")
        self.assertEqual((e.stat["disk_ms"], e.stat["lane_ms"], e.stat["mirror_mb1"], e.stat["w_ms_tok"],
                          e.stat["spec_rounds"]), (18.2, 1.9, 980.0, 40.1, 90.0))
        e._parse_stat("STAT tok_s=20.00 ms_tok=50.00 vram_hit=0.5 ram_fetch=1.0 disk=2.0 promo=1.0 vram_used=1 "
                      "vram_slots=2 ram_used=3 ram_slots=4 ram_gb=1.0 vram_gb=2.0")   # (an older engine's line)
        self.assertNotIn("disk_ms", e.stat)
        tok = ByteTokenizer()
        svc = Service(e, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.status = {"busy": False, "queued": 0}
        tiers = svc.metrics()["tiers"]
        self.assertEqual(tiers["history"]["disk_ms"], [18.2, None])      # (the key's series: None where absent)
        self.assertEqual(tiers["history"]["ms_tok"], [41.5, 50.0])

    def test_done_and_pp(self):
        e = self.engine()
        e._parse_done("DONE 412 5210 6670.0 17100.0 stop 301 410 4800 2500 3900 150")
        self.assertEqual((e.last["drafts_accepted"], e.last["reused"], e.last["lookups"], e.last["spec_rounds"]),
                         (301, 4800, 3900, 150))
        e._parse_done("DONE 412 5210 6670.0 17100.0 stop 301 410 4800 2500 3900")
        self.assertNotIn("spec_rounds", e.last)
        e._parse_pp("PP 2048 5210 2600.0 787.0 1024 310.5")
        self.assertEqual((e.progress, e.prefill_tok_s_mean, e.prompt_reused, e.prompt_disk_ms),
                         ((2048, 5210), 787.0, 1024, 310.5))
        e._parse_pp("PP 3072 5210 3900.0 785.0")                          # (an older engine's line)
        self.assertEqual((e.progress, e.prompt_reused, e.prompt_disk_ms), ((3072, 5210), None, None))


class TerminalViewTest(unittest.TestCase):
    """GET /metrics?view=1: the Abyss Terminal's View of the server's own readings (serve/terminal_view.py), the
    numbers the Monitor shows beside /metrics' - the token's time, the bottleneck and the card it points at."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import terminal_fixtures as F
        import terminal_model as M
        self.F, self.M = F, M

    def fixture_view(self, name):
        from serve.terminal_view import view_json
        log = None
        if name in self.F.LOGS:
            log = self.M.EngineLog(None)
            for x in self.F.LOGS[name]:
                log.feed(x, None)
        v = view_json(self.F.metrics(name), self.F.health(name), self.F.local(name), log, {})
        json.dumps(v)                                          # (JSON-ready)
        return v

    def test_bottleneck_of_an_ssd_bound_run(self):
        v = self.fixture_view("r9700")
        b = v["bottleneck"]
        self.assertEqual((b["verdict"], b["label"], b["target"], b["source"], b["tone"]), ("ssd", "SSD", "disk", "live", "warn"))
        self.assertAlmostEqual(b["share"], 17.5 / 40.1, places=3)
        self.assertIn("RAM tier", b["what"])
        self.assertTrue(b["moves"])
        sp = b["split"]
        self.assertAlmostEqual(sum(s["ms"] for s in sp["segments"]), sp["ms"], places=6)
        self.assertEqual([s["key"] for s in sp["segments"]], ["ssd", "ram", "gpu"])
        self.assertEqual(sp["parallel"]["label"], "iGPU lane")
        self.assertEqual((v["name"], v["quant"], v["state_key"]), ("DeepSeek V4.1 Flash 1017", "UD-IQ2_XXS", "generating"))
        self.assertIn(["UNCENSOR", "alert"], v["badges"])
        self.assertEqual([c["short"] for c in v["cards"]], ["AI PRO R9700", "8060S"])
        for gone in ("requests", "totals", "_keyed"):
            self.assertNotIn(gone, v)

    def test_bench_drafter_and_old_swap_in_the_view(self):
        from serve.terminal_view import view_json
        other = {"line": "Abyss v1.3.122 | Ornith-1.5-397B-32G128G | ...", "decode": 44.2, "prompt8k": 681.0,
                 "model": "Ornith-1.5-397B-32G128G", "config": "cfg_pc8.json"}
        m = self.F.metrics("r9700")
        m["engine"] = dict(m["engine"], drafter="none", mtp=1)
        m["hardware"] = dict(m["hardware"], swap_used=int(2.2 * 2**30), psi_mem_some=0.0, psi_mem_full=0.0,
                             ram_used=11 * 2**30, ram_total=123 * 2**30)
        seen = {}
        for _ in range(3):
            v = view_json(m, self.F.health("r9700"), dict(self.F.local("r9700"), bench=other), None, seen)
        self.assertIsNone(v["bench"])                            # another model's --bench: not shown
        self.assertEqual(v["drafter"]["kind"], "mtp")            # the engine's "none" is not the drafter's name
        self.assertFalse(any(t.lower().startswith(("memory is tight", "memory pressure")) for t, _ in v["banners"]))

    def test_token_split_and_tone(self):
        from serve.terminal_view import token_split, bottleneck
        self.assertIsNone(token_split(None))
        sp = token_split({"ms": 50.0, "disk_ms": 30.0, "miss_ms": None, "lane_ms": 0.0, "lane_name": "CPU lane"})
        self.assertEqual([(s["key"], s["ms"]) for s in sp["segments"]], [("ssd", 30.0), ("gpu", 20.0)])
        self.assertEqual(sp["biggest"], "ssd")
        tt = {"ms": 50.0, "disk_ms": 30.0, "lane_ms": 0.0, "miss_ms": None, "verdict": "ssd", "source": "stat",
              "lane_name": "CPU lane", "what": "x", "moves": "y", "evidence": [], "window": False, "n": 1}
        self.assertEqual(bottleneck({"tt": tt})["tone"], "info")     # (the last answer's: not live)
        self.assertEqual(bottleneck({"tt": dict(tt, source="live")})["tone"], "error")
        self.assertIsNone(bottleneck({"tt": None}))

    def test_the_server_sends_it_with_view_1(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nhello", max_context=32768), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            with urllib.request.urlopen(base + "/metrics?view=1", timeout=10) as r:
                m = json.loads(r.read())
            with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
                plain = json.loads(r.read())
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertNotIn("view", plain)
        v = m["view"]
        # a mock engine: no STAT, no log - the View says so instead of inventing numbers
        self.assertEqual((v["state_key"], v["tt"], v["bottleneck"], v["ctx"]), ("idle", None, None, 32768.0))
        self.assertEqual(v["tt_why"], "no answer written yet")
        self.assertEqual(v["version"], MAYA_VERSION)


if __name__ == "__main__":
    unittest.main()
