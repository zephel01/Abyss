"""tools/glm_big_fixture.py - generate the big runner fixture GGUF (45 layers, seed 777).

A ctest FIXTURE (see CMakeLists.txt): glm_model_big_test needs a 1.2 GB F32 GGUF that is far too
large to commit, so the fixture test generates it into the build tree at configure-known path.
CPU-only, ~30 s.  The expected logits committed under data/glm5-big-dumps/ were produced from
THIS geometry by ref/glm.py (f64 oracle, 512 teacher-forced tokens, (i % 511) + 1).
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:
    import numpy  # noqa: F401,E402  (glm_synth_gguf's one dependency)
except ImportError:
    print("SKIPPED: numpy is not installed for %s (the repo's .venv has it: CMake uses it when present)" % sys.executable)
    sys.exit(77)
import glm_synth_gguf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if out.exists() and out.stat().st_size > 10 ** 9:
        print("fixture exists, keeping: %s (%.2f GB)" % (out, out.stat().st_size / 2 ** 30))
        return 0
    md, ts = glm_synth_gguf.build_big(a.seed, vocab=False)
    glm_synth_gguf.write_gguf(out, md, ts)
    glm_synth_gguf.selftest_big(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
