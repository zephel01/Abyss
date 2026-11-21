# Contributing to Abyss

Reports and pull requests are welcome. Abyss is a fork of [Project Maya](https://github.com/mw00/project-maya):
an issue or a change that is about Maya's own parts (GLM-5.3-Flash, the installer, AMD) may belong upstream there.

## DeepSeek V4: a speed report, a bug, a new machine

Open an issue with:

- GPU(s) (model, VRAM, PCIe as `nvidia-smi` shows it), CPU, RAM, disk, CUDA version;
- the model file (repository and quant) and `--max-context`;
- your config (`abyss-*.json`, without an API key) and the engine log's `glm fast:`, `glm prefill:`, `deepseek4:` and
  `glm stat:` lines (the log is the config's `"log"`);
- what you did and what you saw (the request, the answer or the error).

An output problem is best shown against llama.cpp on the same GGUF: `tools/dsv4_logits_dump.cpp` and
`STRATA_GLM_SCORE_REF` ([docs/deepseek4/PORT.md](docs/deepseek4/PORT.md)).

The sections below are Maya's and apply to its GLM-5.3-Flash setup.

## A speed report or a new machine

Open an issue with:

- **`./abyss.sh --bench --deep`** (Abyss stopped), and paste `abyss-bench.txt` (the
  [speed report](https://github.com/zephel01/Abyss/issues/new?template=bench.md) template). It has the speeds, where a
  token's time goes, the likely bottleneck and a quality check; [docs/BENCH.md](docs/BENCH.md) explains each line.
  `./abyss.sh --report` adds the setup and the engine log (`abyss-report.txt`, includes the bench). Windows:
  `START-ABYSS.bat --bench --deep` / `--report`.
- **A build from source without the installer** is fine too. Send the same things by hand:
  - the exact engine command and environment (`STRATA_GLM_*` settings);
  - GPUs (model, VRAM, PCIe generation and width as `nvidia-smi` shows them), CPU, RAM, disk;
  - model and context length;
  - the engine log's `glm fast:`, `glm split`, `glm stat:` and `glm spec:` lines.

  For numbers comparable with the README's table, measure the way `--bench` does: three questions of 256 tokens,
  greedy, thinking off, one short warm-up first; prefill (reading the prompt) at 2048 and 8192 tokens.

Name your Abyss version (`VERSION`, or the commit). Say what you measured and what you didn't.

## A bug

Run `./abyss.sh --report` right after the failure and attach `abyss-report.txt`. If the engine stopped, add the end
of its log (the `.log` next to your `abyss-*.json` / `maya-*.json`). Wrong or garbled output: `./abyss.sh --bench`
too - its quality check tells a broken kernel on your GPU from a prompt problem.

## A pull request

- **Base it on `main`** and keep it to one topic.
- **Measure before and after** on your hardware, same harness, and put the numbers in the description. If the
  output changes (it can: which expert runs on the CPU or the GPU changes the last bits), say so.
- **New settings** are `STRATA_GLM_*` environment variables, off or at today's behaviour by default unless measured
  better, with a row in the README's Tuning table.
- **AMD (HIP) and other backends:** keep NVIDIA's code path unchanged (`#if defined(STRATA_USE_HIP)`). We can't test
  AMD ourselves, so we merge AMD changes on your measurements once NVIDIA builds and behaves the same.
- **A new CUDA call needs its AMD mapping.** The AMD build compiles the same sources with
  `include/strata/hip_compat/` forced in. A CUDA runtime or cuBLAS name without a `#define` there breaks every AMD build,
  as v1.0.21 did. `python tools/check_hip_compat.py` names any missing one.
- **Every branch push and pull request runs the checks** (GitHub Actions, `.github/workflows/tests.yml`), with no GPU needed:
  - the AMD-mapping check;
  - the server tests;
  - the installer and tools tests.
  
  A pull request is merged only when they pass.
- **A release is built on its `ci-v<version>` branch before the tag:** a push there runs every workflow (tests,
  arm-build, prebuilt with the Windows and Linux CUDA archives). The tag then runs only prebuilt, which attaches that
  branch run's archives to the release (checked against their SHA-256) instead of building them again; it builds them
  itself only when neither that run nor main's run of the same commit has them (artifacts are kept 14 days). Pushes
  to main run tests, arm-build and prebuilt (whose Windows job replaces windows-build there; windows-build is now
  by hand only).
- **Before merging, we build every NVIDIA target** and run on Tesla V100s (one GPU and two):
  - the engine's parity tests (`glm_*_parity --selftest`);
  - a decode / prefill A/B;
  - the server end to end.
- **Areas where help is welcome:**
  - decode speed when every expert fits in VRAM (kernel launches and host waits dominate there);
  - Windows with NVIDIA GPUs;
  - AMD;
  - consumer GPUs from 12 to 32 GB;
  - docs.

Every merged change ships in a release (`CHANGELOG.md`), with credit.
