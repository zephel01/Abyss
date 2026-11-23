"""--parallel (READY ... par=N): StrataEngine against a fake engine that runs requests side by side and answers with
"@<rid> ..." lines - each request gets its own tokens while another runs, STOP <rid> ends that one alone, control
lines (BIAS) still get their answer; and EngineGate lets N requests in together in arrival order while an exclusive
holder (a reload) waits for them and keeps new ones out."""
from __future__ import annotations

import os
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as S  # noqa: E402
from serve.server import EngineGate, StrataEngine  # noqa: E402

# the fake engine: READY 4096 stop par=4; GEN <n> rid=<r> ... -> "@r T <1000*r + i>" every 20 ms, interleaved across
# requests, then "@r DONE"; STOP <r> ends r ("cancel"); BIAS -> BIASOK; QUIT ends it.  Lines it gets go to FAKE_LOG.
FAKE = r"""#!/usr/bin/env python3
import os, sys, threading, time
log = open(os.environ["FAKE_LOG"], "a")
print("READY 4096 stop par=4", flush=True)
reqs, stop, lock = {}, set(), threading.Lock()
def reader():
    for line in sys.stdin:
        log.write(line); log.flush()
        f = line.split()
        with lock:
            if f[0] == "GEN":
                rid = int(next(x for x in f if x.startswith("rid="))[4:])
                reqs[rid] = [int(f[1]), 0]
            elif f[0] == "STOP" and len(f) > 1:
                stop.add(int(f[1]))
            elif f[0] == "BIAS":
                print(f"BIASOK {f[1]} 1", flush=True)
            elif f[0] == "QUIT":
                os._exit(0)
    os._exit(0)
threading.Thread(target=reader, daemon=True).start()
while True:
    time.sleep(0.02)
    with lock:
        for rid, st in list(reqs.items()):
            if rid in stop:
                print(f"@{rid} DONE {st[1]} 1 1.0 1.0 cancel", flush=True)
                del reqs[rid]
                continue
            print(f"@{rid} T {1000 * rid + st[1]}", flush=True)
            st[1] += 1
            if st[1] >= st[0]:
                print(f"@{rid} DONE {st[1]} 1 1.0 1.0 stop", flush=True)
                del reqs[rid]
"""


@unittest.skipIf(os.name == "nt", "the fake engine is a #! script")
class ParallelEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fake-par-"))
        p = self.tmp / "fake-strata"
        p.write_text(FAKE)
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
        self.log = self.tmp / "lines.txt"
        self.log.write_text("")
        self.patches = [mock.patch.object(S, "wait_gpu_release", lambda pid, timeout=60.0: None),
                        mock.patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.tmp / "cache")})]
        for x in self.patches:
            x.start()
        self.e = StrataEngine(str(p), ["--glm-pack", str(self.tmp / "pack.txt")],
                              env={**os.environ, "FAKE_LOG": str(self.log)})

    def tearDown(self):
        for x in self.patches:
            x.stop()
        try:
            self.e.proc.kill()
        except OSError:
            pass

    def test_two_requests_side_by_side(self):
        self.assertEqual(self.e.parallel, 4)
        out, lasts = {}, {}

        def run(k, n):
            out[k] = [t for t in self.e.generate([1, 2], n, {}, threading.Event()) if t is not None]
            lasts[k] = self.e.last

        th = [threading.Thread(target=run, args=(k, n)) for k, n in (("a", 6), ("b", 4))]
        t0 = time.monotonic()
        for t in th:
            t.start()
        for t in th:
            t.join(10)
        # each its own tokens (rid r's are 1000*r + i), and together: about the time of the longer one alone
        ra, rb = out["a"][0] // 1000, out["b"][0] // 1000
        self.assertEqual(out["a"], [1000 * ra + i for i in range(6)])
        self.assertEqual(out["b"], [1000 * rb + i for i in range(4)])
        self.assertNotEqual(ra, rb)
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual((lasts["a"]["generated"], lasts["b"]["generated"]), (6, 4))   # each its own DONE

    def test_cancel_one_leaves_the_other(self):
        cancel = threading.Event()
        got = {}

        def long():
            got["long"] = [t for t in self.e.generate([1], 30, {}, threading.Event()) if t is not None]

        th = threading.Thread(target=long)
        th.start()
        short = []
        for t in self.e.generate([1], 100, {}, cancel):
            if t is not None:
                short.append(t)
            if len(short) == 3:
                cancel.set()
        th.join(10)
        self.assertEqual(len(got["long"]), 30, "the other request was cut too")
        self.assertIn(f"STOP {short[0] // 1000}", self.log.read_text().splitlines())
        self.assertNotIn("STOP", self.log.read_text().splitlines())   # (no bare STOP: that stops all)

    def test_control_line_while_a_request_runs(self):
        got = []
        th = threading.Thread(target=lambda: got.extend(self.e.generate([1], 20, {}, threading.Event())))
        th.start()
        time.sleep(0.1)
        self.assertTrue(self.e.command("BIAS x 1:ban", "BIASOK", timeout=5).startswith("BIASOK"))
        th.join(10)
        self.assertEqual(len([t for t in got if t is not None]), 20)


class ConfigTests(unittest.TestCase):
    def test_parallel_key_to_engine_args(self):
        base = {"args": ["--glm-pack", "p"]}
        self.assertEqual(S.engine_args(base), ["--glm-pack", "p"])
        self.assertEqual(S.engine_args({**base, "parallel": 1}), ["--glm-pack", "p"])
        self.assertEqual(S.engine_args({**base, "parallel": 4}), ["--glm-pack", "p", "--parallel", "4"])
        self.assertEqual(S.engine_args({**base, "parallel": "auto"}), ["--glm-pack", "p", "--parallel", "auto"])
        given = {"args": ["--glm-pack", "p", "--parallel", "2"], "parallel": 4}   # (the args' own wins)
        self.assertEqual(S.engine_args(given), ["--glm-pack", "p", "--parallel", "2"])


class GateTests(unittest.TestCase):
    def test_shared_up_to_slots_in_order(self):
        g = EngineGate(lambda: 2)
        inside, peak, order, lock = [0], [0], [], threading.Lock()

        def req(k):
            with g.shared():
                with lock:
                    inside[0] += 1
                    peak[0] = max(peak[0], inside[0])
                    order.append(k)
                time.sleep(0.05)
                with lock:
                    inside[0] -= 1

        th = []
        for k in range(6):
            th.append(threading.Thread(target=req, args=(k,)))
            th[-1].start()
            time.sleep(0.005)
        for t in th:
            t.join(5)
        self.assertEqual(peak[0], 2)
        self.assertEqual(order, list(range(6)))

    def test_one_slot_is_a_lock(self):
        g = EngineGate()
        with g.shared():
            self.assertFalse(g.acquire(timeout=0.05))
        self.assertTrue(g.acquire(timeout=0.05))
        g.release()

    def test_exclusive_waits_and_holds_new_requests(self):
        g = EngineGate(lambda: 4)
        events = []
        started = threading.Event()

        def req(k, hold):
            with g.shared():
                events.append(("in", k))
                started.set()
                time.sleep(hold)
            events.append(("out", k))

        a = threading.Thread(target=req, args=(0, 0.2))
        a.start()
        started.wait(2)

        def reload():
            with g:
                events.append(("reload", None))

        r = threading.Thread(target=reload)
        r.start()
        time.sleep(0.05)
        b = threading.Thread(target=req, args=(1, 0.0))   # comes after the reload asked: waits for it
        b.start()
        for t in (a, r, b):
            t.join(5)
        self.assertEqual(events, [("in", 0), ("out", 0), ("reload", None), ("in", 1), ("out", 1)])


if __name__ == "__main__":
    unittest.main()
