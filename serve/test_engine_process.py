"""StrataEngine against a fake engine process (a small Python script that speaks the serve protocol): a restart asks
QUIT and waits (the kept conversation is written) instead of killing; close() gives up on an engine that ignores
QUIT; a stopped request whose engine goes silent is ended within STRATA_ENGINE_DRAIN_S; an engine that never says
READY fails the start within STRATA_ENGINE_START_S; the vision encoder's answers have a limit; the server's own env
numbers are checked; GET /status shows the text being generated only with the API key."""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as S  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, Vision, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# the fake engine: READY, then GEN -> T lines and DONE; QUIT ends it (exit 0) unless FAKE_MODE=ignore-quit;
# STOP ends the request at the next token (FAKE_MODE=hang-after-stop: it goes silent instead); FAKE_MODE=no-ready
# never says READY.  Every line it gets is appended to FAKE_LOG.
FAKE_ENGINE = r"""#!/usr/bin/env python3
import os, queue, sys, threading, time
mode = os.environ.get("FAKE_MODE", "")
log = open(os.environ["FAKE_LOG"], "a")
if mode == "no-ready":
    time.sleep(30)
    sys.exit(0)
if mode == "warm-fault":   # (#66) the warm-up's GPU fault while STRATA_GLM_WARM_RETRY < FAKE_FAULTS: RETRY, exit 1
    tries = int(os.environ["STRATA_GLM_WARM_RETRY"])
    reserve = int(os.environ.get("STRATA_GLM_RESERVE_MB", "1536"))
    log.write(f"start:{reserve}:{tries}\n"); log.flush()
    if tries < int(os.environ.get("FAKE_FAULTS", "1")):
        print(f"RETRY STRATA_GLM_RESERVE_MB={reserve + 2048} STRATA_GLM_WARM_RETRY={tries + 1}", flush=True)
        sys.exit(1)
print("INFO engine=fake kv=1", flush=True)
print("READY 4096 stop", flush=True)
box = queue.Queue()
def reader():
    for line in sys.stdin:
        log.write(line); log.flush()
        box.put(line)
    box.put(None)
threading.Thread(target=reader, daemon=True).start()
while True:
    line = box.get()
    if line is None:
        sys.exit(0)
    if line.startswith("QUIT"):
        if mode == "ignore-quit":
            continue
        print("NOTE conversation kept", flush=True)
        sys.exit(0)
    if line.startswith("GEN"):
        n = int(line.split()[1])
        for i in range(n):
            try:
                nxt = box.get(timeout=0.02)
            except queue.Empty:
                nxt = None
            if nxt is not None and nxt.startswith("STOP"):
                if mode == "hang-after-stop":
                    time.sleep(30)
                    sys.exit(0)
                print(f"DONE {i} 1 1.0 1.0 cancel", flush=True)
                break
            if nxt is not None and nxt.startswith("QUIT"):
                sys.exit(0)
            print(f"T {100 + i}", flush=True)
        else:
            print(f"DONE {n} 1 1.0 1.0 stop", flush=True)
"""

# the fake vision encoder: READY, then no answer to ENC
FAKE_VISION = r"""#!/usr/bin/env python3
import sys, time
print("READY", flush=True)
for line in sys.stdin:
    time.sleep(30)
"""


def script(dir: Path, name: str, text: str) -> str:
    p = dir / name
    p.write_text(text)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


@unittest.skipIf(os.name == "nt", "the fake engine is a #! script")
class FakeEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fake-engine-"))
        self.exe = script(self.tmp, "fake-strata", FAKE_ENGINE)
        self.log = self.tmp / "lines.txt"
        self.log.write_text("")
        self.patches = [mock.patch.object(S, "wait_gpu_release", lambda pid, timeout=60.0: None),
                        mock.patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.tmp / "cache")})]   # (run markers)
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        for e in getattr(self, "engines", []):
            try:
                e.proc.kill()
            except OSError:
                pass

    def engine(self, mode="", **env):
        e = StrataEngine(self.exe, ["--glm-pack", str(self.tmp / "m" / "pack.txt")], env={**os.environ, "FAKE_MODE": mode, "FAKE_LOG": str(self.log),
                                                                                 **{k: str(v) for k, v in env.items()}})
        self.engines = getattr(self, "engines", []) + [e]
        return e

    def lines(self) -> list[str]:
        return self.log.read_text().split()

    def test_ready_and_a_request(self):
        e = self.engine()
        self.assertEqual((e.max_context, e.can_stop, e.info["engine"]), (4096, True, "fake"))
        self.assertEqual(list(e.generate([1, 2], 3, {}, threading.Event())), [100, 101, 102])
        self.assertEqual(e.last["finish"], "stop")

    @unittest.skipUnless(sys.platform.startswith("linux"), "the systemd-oomd run marker is Linux only")
    def test_run_marker_life_cycle(self):
        import contextlib
        import io
        with mock.patch.dict(os.environ, {"STRATA_OOMD_GUARD": "1", "STRATA_ENGINE_QUIT_S": "10"}), \
                mock.patch.object(S, "oomd_kills_since", lambda ts, timeout=5.0: []), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            S.OWN_MARKERS.clear()
            e = self.engine()
            m = e.run_marker
            self.assertEqual(json.loads(m.read_text())["pid"], e.proc.pid)
            e.proc.kill()                                   # died under this server: its restart is not reported
            e.proc.wait()
            e.restart()
            self.assertEqual(json.loads(m.read_text())["pid"], e.proc.pid)
            e.close()                                       # a clean stop: no marker
            self.assertFalse(m.exists())
            e2 = self.engine()                              # a new server: the previous engine was killed
            e2.proc.kill()
            e2.proc.wait()
            S.OWN_MARKERS.clear()
            e3 = self.engine()
            e3.close()
            self.assertFalse(m.exists())
        said = [ln for ln in out.getvalue().splitlines() if "previous engine" in ln]
        self.assertEqual(len(said), 1, out.getvalue())
        self.assertTrue(said[0].endswith("did not end cleanly"))

    def test_restart_asks_quit_first(self):
        e = self.engine()
        first = e.proc
        e.set_context(2048)
        with mock.patch.dict(os.environ, {"STRATA_ENGINE_QUIT_S": "10"}):
            e.restart()
        self.assertIn("QUIT", self.lines(), "the running engine was not asked to end")
        self.assertEqual(first.wait(timeout=5), 0, "QUIT should end it with exit 0, a kill would not")
        self.assertTrue(e.alive())
        self.assertEqual(e.spawn[1][-2:], ["--max-context", "2048"])
        self.assertEqual(list(e.generate([1], 2, {}, threading.Event())), [100, 101])

    def test_restart_after_death_does_not_wait(self):
        e = self.engine()
        e.proc.kill()
        e.proc.wait(timeout=5)
        time.sleep(0.2)                                   # the pump sees the EOF
        self.assertFalse(e.alive())
        t0 = time.monotonic()
        e.restart()
        self.assertLess(time.monotonic() - t0, 10)
        self.assertTrue(e.alive())

    def test_close_kills_an_engine_that_ignores_quit(self):
        e = self.engine("ignore-quit")
        with mock.patch.dict(os.environ, {"STRATA_ENGINE_QUIT_S": "1"}):
            t0 = time.monotonic()
            e.close()
        self.assertIsNotNone(e.proc.wait(timeout=5))
        self.assertLess(time.monotonic() - t0, 10)

    def test_a_stopped_request_whose_engine_goes_silent_is_ended(self):
        e = self.engine("hang-after-stop")
        with mock.patch.dict(os.environ, {"STRATA_ENGINE_DRAIN_S": "2"}), mock.patch.object(S, "ENGINE_STALL_S", 0):
            gen = e.generate([1], 1000, {}, threading.Event())
            self.assertEqual(next(gen), 100)
            t0 = time.monotonic()
            gen.close()                                   # the consumer stops early: STOP, then the drain
        self.assertLess(time.monotonic() - t0, 30)
        self.assertIn("STOP", self.lines())
        self.assertFalse(e.alive(), "the silent engine should have been ended")

    def test_a_cancelled_request_drains_to_done(self):
        e = self.engine()
        gen = e.generate([1], 1000, {}, threading.Event())
        self.assertEqual(next(gen), 100)
        gen.close()
        self.assertEqual(e.last["finish"], "cancel")
        self.assertTrue(e.alive())
        self.assertEqual(list(e.generate([1], 1, {}, threading.Event())), [100])

    def test_no_ready_within_the_limit_fails_the_start(self):
        with mock.patch.dict(os.environ, {"STRATA_ENGINE_START_S": "1"}):
            t0 = time.monotonic()
            with self.assertRaises(RuntimeError) as cm:
                self.engine("no-ready")
        self.assertIn("STRATA_ENGINE_START_S", str(cm.exception))
        self.assertLess(time.monotonic() - t0, 10)

    def test_a_warm_up_gpu_fault_starts_the_engine_again_with_more_vram_free(self):   # #66
        e = self.engine("warm-fault", STRATA_GLM_RESERVE_MB=3072, FAKE_FAULTS=1)
        self.assertEqual(e.max_context, 4096)
        self.assertEqual(self.lines(), ["start:3072:0", "start:5120:1"])
        self.assertEqual((e.spawn[4]["STRATA_GLM_RESERVE_MB"], e.spawn[4]["STRATA_GLM_WARM_RETRY"]), ("5120", "1"))
        e.restart()                                     # a later start keeps the larger reserve and the count
        self.assertEqual(self.lines()[-1], "start:5120:1")

    def test_at_most_two_warm_up_retries(self):
        with self.assertRaises(RuntimeError):
            self.engine("warm-fault", STRATA_GLM_RESERVE_MB=3072, FAKE_FAULTS=5)
        self.assertEqual(self.lines(), ["start:3072:0", "start:5120:1", "start:7168:2"])

    def test_no_warm_up_retry_when_the_user_sized_the_pool(self):
        with self.assertRaises(RuntimeError):
            self.engine("warm-fault", STRATA_GLM_POOL_GB="3.5", FAKE_FAULTS=1)
        self.assertEqual(self.lines(), ["start:1536:0"])

    def test_a_retry_line_asking_for_anything_else_is_ignored(self):
        self.assertIsNone(S.warm_retry_env("RETRY LD_PRELOAD=/x STRATA_GLM_WARM_RETRY=1", {}))
        self.assertIsNone(S.warm_retry_env("RETRY STRATA_GLM_RESERVE_MB=5120 STRATA_GLM_WARM_RETRY=3", {}))
        self.assertIsNone(S.warm_retry_env("RETRY STRATA_GLM_RESERVE_MB=-1 STRATA_GLM_WARM_RETRY=1", {}))
        self.assertEqual(S.warm_retry_env("RETRY STRATA_GLM_RESERVE_MB=5120 STRATA_GLM_WARM_RETRY=1", None),
                         {"STRATA_GLM_RESERVE_MB": "5120", S.WARM_RETRY_ENV: "1"})

    def test_the_service_reloads_the_context_through_quit(self):
        e = self.engine()
        first = e.proc
        svc = Service(e, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.trained_context = 16384
        with mock.patch.dict(os.environ, {"STRATA_ENGINE_QUIT_S": "10"}):
            svc.set_context(8192)
            for _ in range(100):
                if not svc.reloading():
                    break
                time.sleep(0.1)
        self.assertFalse(svc.reloading())
        self.assertEqual(svc.reload["state"], "done", svc.reload)
        self.assertEqual(first.wait(timeout=5), 0)
        self.assertIn("QUIT", self.lines())
        self.assertEqual(e.max_context, 4096)             # (the fake always says 4096; the command carries 8192)
        self.assertEqual(e.spawn[1][-2:], ["--max-context", "8192"])


@unittest.skipIf(os.name == "nt", "the fake encoder is a #! script")
class VisionTimeout(unittest.TestCase):
    def test_an_encoder_that_never_answers(self):
        tmp = Path(tempfile.mkdtemp(prefix="fake-vision-"))
        exe = script(tmp, "fake-vision", FAKE_VISION)
        v = Vision({"exe": exe, "mmproj": "x", "model": "y"}, start=False)
        bmp = "data:image/bmp;base64," + __import__("base64").b64encode(b"BM" + bytes(60)).decode()
        with mock.patch.dict(os.environ, {"STRATA_VISION_ENCODE_S": "1"}):
            t0 = time.monotonic()
            with self.assertRaises(ValueError) as cm:
                v.encode(bmp)
        self.assertIn("did not answer", str(cm.exception))
        self.assertLess(time.monotonic() - t0, 15)
        self.assertIsNone(v.proc)                          # ended: the next picture starts it again
        v.close()


class ServerEnv(unittest.TestCase):
    def test_bad_numbers_are_dropped_and_the_late_keys_arrive(self):
        keys = S.SERVER_ENV + S.SERVER_ENV_LATE
        saved = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                os.environ.pop(k, None)
            os.environ["STRATA_ENGINE_QUIT_S"] = "50s"
            with mock.patch("builtins.print") as said:
                S.apply_server_env({"env": {"STRATA_GLM_THINK_RESERVE": "2048.0", "STRATA_ENGINE_WRAP_S": "abc",
                                            "STRATA_ENGINE_STALL_S": "0", "STRATA_HTTP_BACKLOG": 64}})
            self.assertEqual(os.environ.get("STRATA_GLM_THINK_RESERVE"), "2048")      # an int where int() reads it
            self.assertNotIn("STRATA_ENGINE_WRAP_S", os.environ)
            self.assertNotIn("STRATA_ENGINE_QUIT_S", os.environ)
            self.assertEqual((S.ENGINE_STALL_S, S.Server.request_queue_size), (0.0, 64))
            told = " ".join(str(c.args[0]) for c in said.call_args_list)
            self.assertIn("STRATA_ENGINE_WRAP_S='abc' is not a number", told)
            self.assertIn("STRATA_ENGINE_QUIT_S='50s' is not a number", told)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            S.apply_server_env({})


class ServerEnvLang(unittest.TestCase):
    def test_the_language_profile_from_a_config_reaches_the_server(self):   # #50
        from serve import lang_ban
        keys = S.SERVER_ENV_TEXT
        saved = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                os.environ.pop(k, None)
            with mock.patch("builtins.print") as said:
                S.apply_server_env({"env": {"STRATA_LANG": "ja-en", "STRATA_LANG_THINK": "1"}})
            self.assertEqual(lang_ban.default_profile(), "ja-en")
            self.assertTrue(lang_ban.default_think())
            told = " ".join(str(c.args[0]) for c in said.call_args_list)
            self.assertNotIn("STRATA_LANG", told)        # text, not checked as a number
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class StatusBehindTheKey(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "hello"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, headers=None):
        req = urllib.request.Request(self.base + "/status", headers=headers or {})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())

    def test_the_text_being_generated_needs_the_key(self):
        with self.svc.status_lock:
            self.svc.status.update(busy=True, phase="answering", tail="the secret draft", prompt_tokens=12,
                                   started=time.time(), first_token=None)
        self.svc.api_key = "k"
        try:
            code, s = self.get()
            self.assertEqual(code, 200)
            self.assertEqual(set(s), {"busy", "queued"})
            code, s = self.get({"Authorization": "Bearer k"})
            self.assertEqual((code, s["tail"], s["phase"]), (200, "the secret draft", "answering"))
            code, s = self.get({"x-api-key": "k"})
            self.assertEqual(s["prompt_tokens"], 12)
        finally:
            self.svc.api_key = ""
            with self.svc.status_lock:
                self.svc.status.update(busy=False, tail="")
        code, s = self.get()
        self.assertEqual((code, s["busy"]), (200, False))   # no key configured: everything, as before


if __name__ == "__main__":
    unittest.main()
