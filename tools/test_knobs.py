"""include/strata/core/knobs.hpp - the engine's number-reading and bounds helpers, run on the host without CUDA:
a table of env_num cases (unset, a typo, below / above the range, a plain value) and draft_kmax at the context's end.
Needs a C++ compiler (skipped without one)."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")

DRIVER = r"""
#include "strata/core/knobs.hpp"
#include <cstdio>
using namespace strata::core;
int main() {
    // name default lo hi raw -> value changed
    struct Row { const char* raw; double def, lo, hi; };
    const Row rows[] = {{nullptr, 6, 0, 100}, {"", 6, 0, 100}, {"abc", 6, 0, 100}, {"1,0", 0.73, 0.05, 0.95},
                        {"-3", 256, 0, 65536}, {"200", 15, 0, 64}, {"0.5", 0.73, 0.05, 0.95}, {"1.0", 0.73, 0.05, 0.95},
                        {"16", 6, 0, 4096}, {" 12 ", 6, 0, 4096}, {"1e3", 6, 0, 4096}};
    for (const Row& r : rows) {
        bool changed = false;
        const double v = env_num("STRATA_TEST_KNOB", r.def, r.lo, r.hi, r.raw != nullptr ? r.raw : "", &changed);
        std::printf("%s|%g|%d\n", r.raw != nullptr ? r.raw : "<unset>", v, changed ? 1 : 0);
    }
    const long long ks[][3] = {{5, 0, 100}, {5, 95, 100}, {5, 96, 100}, {5, 99, 100}, {5, 100, 100}, {5, 101, 100},
                               {3, 97, 100}, {5, -1, 100}, {0, 10, 100}};
    for (const auto& k : ks) std::printf("k:%lld,%lld,%lld=%d\n", k[0], k[1], k[2], draft_kmax((int) k[0], k[1], k[2]));
    const char* sw[] = {"", " ", "0", "1", "off", "OFF", "false", "No", " 0 ", "on", "yes", "true", "2", "x"};
    for (const char* v : sw) std::printf("s:%s=%d%d\n", v, env_on("STRATA_TEST_SW", false, v) ? 1 : 0,
                                         env_on("STRATA_TEST_SW", true, v) ? 1 : 0);
    std::printf("s:<unset>=%d%d\n", env_on("STRATA_TEST_SW_UNSET_X", false) ? 1 : 0,
                env_on("STRATA_TEST_SW_UNSET_X", true) ? 1 : 0);
    return 0;
}
"""


@unittest.skipUnless(CXX, "needs a C++ compiler")
class KnobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="knobs-"))
        src = cls.tmp / "driver.cpp"
        src.write_text(DRIVER)
        r = subprocess.run([CXX, "-std=c++20", "-Wall", "-Wextra", "-I", str(ROOT / "include"), str(src), "-o",
                            str(cls.tmp / "driver")], capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError("knobs.hpp did not compile:\n" + r.stderr)
        p = subprocess.run([str(cls.tmp / "driver")], capture_output=True, text=True, timeout=30)
        cls.out, cls.err = p.stdout.splitlines(), p.stderr

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_env_num_table(self):
        got = {ln.split("|")[0]: tuple(ln.split("|")[1:]) for ln in self.out if "|" in ln}
        want = {"<unset>": ("6", "0"),      # unset: the default, nothing said
                "": ("6", "0"),             # empty: the same
                "abc": ("6", "1"),          # a typo: the default, said
                "1,0": ("0.73", "1"),       # a comma decimal: not a number, the default, said
                "-3": ("0", "1"),           # negative sizes: 0
                "200": ("64", "1"),         # too many threads: the top
                "0.5": ("0.5", "0"),
                "1.0": ("0.95", "1"),       # the RAM share that would lift the cap
                "16": ("16", "0"),
                " 12 ": ("12", "0"),
                "1e3": ("1000", "0")}
        for k, v in want.items():
            self.assertEqual(got[k], v, k)
        self.assertIn("STRATA_TEST_KNOB=abc is not a number: using 6", self.err)
        self.assertIn("STRATA_TEST_KNOB=1,0 is not a number: using 0.73", self.err)
        self.assertIn("STRATA_TEST_KNOB=200 is outside 0..64: using 64", self.err)

    def test_env_on_reads_the_value(self):
        got = dict(ln[2:].rsplit("=", 1) for ln in self.out if ln.startswith("s:"))
        for v in ("", " ", "0", "off", "OFF", "false", "No", " 0 "):
            self.assertEqual(got[v], "00", repr(v))     # off, whatever the default
        for v in ("1", "on", "yes", "true", "2", "x"):
            self.assertEqual(got[v], "11", repr(v))     # on
        self.assertEqual(got["<unset>"], "01")          # unset: the default

    def test_draft_kmax_stops_at_the_context(self):
        got = dict(ln[2:].split("=") for ln in self.out if ln.startswith("k:"))
        self.assertEqual(got["5,0,100"], "5")
        self.assertEqual(got["5,95,100"], "5")     # rows 95..99 fit
        self.assertEqual(got["5,96,100"], "4")     # 96..99
        self.assertEqual(got["5,99,100"], "1")     # the last row
        self.assertEqual(got["5,100,100"], "0")    # nothing left
        self.assertEqual(got["5,101,100"], "0")
        self.assertEqual(got["3,97,100"], "3")
        self.assertEqual(got["5,-1,100"], "0")
        self.assertEqual(got["0,10,100"], "0")


if __name__ == "__main__":
    unittest.main()
