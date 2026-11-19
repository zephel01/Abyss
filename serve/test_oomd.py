"""serve/test_oomd.py - the systemd-oomd guard (Linux): the model's page cache is dropped only when RAM is short and
the user slice holds file cache; the run marker says at the next start how the previous engine ended (systemd-oomd's
journal line, else "did not end cleanly"), is removed on a clean stop, and an engine restarted by the same server
(issue #27) is not reported.  /proc/meminfo, the cgroup files and journalctl are faked; no GPU, no engine.

    python -m unittest serve.test_oomd -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as S  # noqa: E402
from serve.server import (GIB, meminfo, model_files, release_model_cache, report_previous_run,  # noqa: E402
                          run_marker_path, user_slice_file_cache, write_run_marker, clear_run_marker)

ON = {"STRATA_OOMD_GUARD": "1"}


def meminfo_text(avail_kb: int, cached_kb: int, shmem_kb: int = 0) -> str:
    return (f"MemTotal:       98000000 kB\nMemFree:          100000 kB\nMemAvailable:   {avail_kb} kB\n"
            f"Cached:         {cached_kb} kB\nShmem:          {shmem_kb} kB\n")


class Tmp(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory(prefix="oomd-")
        self.tmp = Path(self._td.name)
        self.addCleanup(self._td.cleanup)
        S.OWN_MARKERS.clear()
        self.addCleanup(S.OWN_MARKERS.clear)

    def write(self, name: str, text: str) -> str:
        p = self.tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return str(p)


class MeminfoAndCgroup(Tmp):
    def test_meminfo_in_bytes(self):
        m = meminfo(self.write("mi", meminfo_text(1000, 2000, 10)))
        self.assertEqual((m["MemAvailable"], m["Cached"], m["Shmem"]), (1000 * 1024, 2000 * 1024, 10 * 1024))
        self.assertEqual(meminfo(str(self.tmp / "none")), {})

    def test_user_slice_file_from_the_ancestor(self):
        cg = self.write("cgroup", "0::/user.slice/user-1000.slice/user@1000.service/app.slice/x.scope\n")
        self.write("sys/user.slice/user-1000.slice/memory.stat", "anon 5\nfile 7340032000\nkernel 3\n")
        self.write("sys/user.slice/user-1000.slice/user@1000.service/memory.stat", "file 1\n")
        self.assertEqual(user_slice_file_cache(cg, str(self.tmp / "sys"), str(self.tmp / "none")), 7340032000)

    def test_falls_back_to_meminfo_cached_minus_shmem(self):
        cg = self.write("cgroup", "0::/system.slice/abyss.service\n")      # not in a user slice
        mi = self.write("mi", meminfo_text(1, 3 * 1024 * 1024, 1024 * 1024))
        self.assertEqual(user_slice_file_cache(cg, str(self.tmp / "sys"), mi), 2 * GIB)
        self.assertIsNone(user_slice_file_cache(str(self.tmp / "none"), str(self.tmp / "sys"), str(self.tmp / "x")))


class ReleaseCache(Tmp):
    def setUp(self):
        super().setUp()
        self.pack = self.tmp / "pack"
        (self.pack / "sub").mkdir(parents=True)
        (self.pack / "a.bin").write_bytes(b"\0" * 4000)
        (self.pack / "sub" / "b.bin").write_bytes(b"\0" * 6000)       # 10 000 bytes in all
        self.dropped = []
        self.p = mock.patch.object(S, "drop_file_cache", lambda files: self.dropped.extend(files))
        self.p.start()
        self.addCleanup(self.p.stop)

    def run_it(self, avail_bytes, cache, env=ON, linux=True):
        mi = self.write("mi", meminfo_text(avail_bytes // 1024, 5000))
        with mock.patch.dict(os.environ, env), mock.patch.object(S.sys, "platform", "linux" if linux else "win32"), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            r = release_model_cache(str(self.pack), mi, cache=lambda: cache)
        return r, out.getvalue()

    def test_walks_a_pack_or_takes_one_file(self):
        self.assertEqual(sorted(Path(f).name for f in model_files(str(self.pack))), ["a.bin", "b.bin"])
        self.assertEqual(model_files(str(self.pack / "a.bin")), [str(self.pack / "a.bin")])
        self.assertEqual(model_files(str(self.tmp / "missing")), [])

    def test_drops_when_ram_is_short_and_the_slice_has_cache(self):
        r, out = self.run_it(7000, 2 * GIB)                 # 7000 < 0.8 x 10000
        self.assertIsNotNone(r)
        self.assertEqual(len(self.dropped), 2)
        self.assertIn("[abyss] memory: released", out)
        self.assertIn("systemd-oomd", out)

    def test_left_alone_when_ram_is_enough(self):
        r, out = self.run_it(9000, 2 * GIB)                 # 9000 >= 8000
        self.assertIsNone(r)
        self.assertEqual((self.dropped, out), ([], ""))

    def test_left_alone_under_1_gib_of_cache_or_unknown(self):
        self.assertIsNone(self.run_it(100, GIB - 1)[0])
        self.assertIsNone(self.run_it(100, None)[0])
        self.assertEqual(self.dropped, [])

    def test_off_by_env_and_off_linux(self):
        self.assertIsNone(self.run_it(100, 2 * GIB, env={"STRATA_OOMD_GUARD": "0"})[0])
        self.assertIsNone(self.run_it(100, 2 * GIB, linux=False)[0])
        self.assertEqual(self.dropped, [])

    def test_never_raises(self):
        with mock.patch.object(S, "model_files", side_effect=RuntimeError("boom")), mock.patch.dict(os.environ, ON):
            self.assertIsNone(release_model_cache(str(self.pack)))

    @unittest.skipUnless(hasattr(os, "posix_fadvise"), "posix_fadvise is Linux / POSIX only")
    def test_real_fadvise_on_a_temp_file(self):
        self.p.stop()
        S.drop_file_cache([str(self.pack / "a.bin"), str(self.tmp / "missing")])   # no error, missing skipped
        self.p.start()


class RunMarker(Tmp):
    def journal(self, lines):
        self.asked = []

        def j(ts):
            self.asked.append(ts)
            return lines
        return j

    def report(self, marker, lines):
        with mock.patch.dict(os.environ, ON), mock.patch.object(S.sys, "platform", "linux"), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            r = report_previous_run(marker, self.journal(lines))
        return r, out.getvalue()

    def dead_marker(self, started):
        m = run_marker_path(str(self.tmp / "abyss.log"), "/m/pack")
        m.write_text(json.dumps({"pid": 999999999, "started": started, "model": "/m/pack"}))
        return m

    def test_place(self):
        self.assertEqual(run_marker_path("/x/cfg.log", "p"), Path("/x/cfg.log.engine-run.json"))
        with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.tmp)}):
            p = run_marker_path(None, "/m/pack")
        self.assertEqual(p.parent, self.tmp / "abyss")
        self.assertTrue(p.name.startswith("engine-run-"))

    def test_oomd_kill_is_named(self):
        m = self.dead_marker(1700000000.5)
        kill = ("Killed /user.slice/user-1000.slice/user@1000.service/app.slice/x.scope due to memory pressure for "
                "/user.slice/user-1000.slice/user@1000.service being 67.52% > 50.00% for > 20s with reclaim activity")
        r, out = self.report(m, ["Killed /old.scope due to memory pressure", kill])
        self.assertEqual(self.asked, [1700000000.5])
        self.assertIn("was stopped by systemd-oomd (memory pressure): " + kill, r)
        self.assertIn("lower STRATA_GLM_RAM_GB", r)
        self.assertEqual(out.strip(), r)
        self.assertFalse(m.exists())

    def test_no_journal_line_is_not_clean(self):
        m = self.dead_marker(time.time() - 60)
        r, _ = self.report(m, [])
        self.assertTrue(r.endswith("did not end cleanly"))
        self.assertIn("the previous engine (started ", r)
        self.assertFalse(m.exists())

    def test_running_engine_or_no_marker_is_quiet(self):
        m = run_marker_path(str(self.tmp / "abyss.log"), "/m/pack")
        self.assertEqual(self.report(m, ["Killed x"]), (None, ""))
        m.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))   # alive
        self.assertEqual(self.report(m, ["Killed x"]), (None, ""))
        self.assertTrue(m.exists())
        m.write_text("not json")
        self.assertEqual(self.report(m, []), (None, ""))

    def test_own_restart_is_not_reported(self):
        m = run_marker_path(str(self.tmp / "abyss.log"), "/m/pack")
        write_run_marker(m, 999999999, "/m/pack")           # this server's engine, since died (issue #27 restart)
        self.assertEqual(self.report(m, ["Killed x"]), (None, ""))
        self.assertTrue(m.exists())
        clear_run_marker(m)
        self.assertFalse(m.exists())
        clear_run_marker(m)                                  # (twice: no error)

    def test_journal_reader_uses_since_and_tag(self):
        calls = []

        class R:
            stdout = "Considered 11 cgroups for killing\nKilled /a.scope due to memory pressure\n"

        def run(cmd, **kw):
            calls.append(cmd)
            return R()
        with mock.patch.object(S.subprocess, "run", run):
            self.assertEqual(S.oomd_kills_since(1234.9), ["Killed /a.scope due to memory pressure"])
        self.assertEqual(calls[0][:7], ["journalctl", "--since", "@1234", "-t", "systemd-oomd", "-q", "--no-pager"])

        def missing(cmd, **kw):
            raise FileNotFoundError("journalctl")
        with mock.patch.object(S.subprocess, "run", missing):
            self.assertEqual(S.oomd_kills_since(0), [])


if __name__ == "__main__":
    unittest.main()
