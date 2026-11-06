# Measuring Abyss on your PC (`--bench`, `--check`)

[日本語](BENCH.ja.md)

Abyss runs models that do not fit in GPU memory: each token's experts come from GPU memory, from a RAM tier, or from
the SSD, and some are computed by a CPU (or iGPU) lane. Which of those limits the speed depends on the PC - the GPU,
its PCIe slot, the RAM, the SSD, the CPU. We can only measure our own PCs, so the benchmark is built to be run on
yours and pasted into an issue: it gives the same numbers on every machine, says where a token's time goes, and checks
that the PC computes what a known-good run did.

## Running it

Stop Abyss first (the benchmark needs the GPU memory it holds), then:

```
./abyss.sh --bench                 # a few minutes: speed, where the time goes, the quality check
./abyss.sh --bench --deep          # + the GPU timeline and a PCIe / lane share sweep (finds the bottleneck)
./abyss.sh --bench --full          # + three runs of each question and a 32k prompt
./abyss.sh --bench --config abyss-ornith-r9700l.json    # another installed model than the last one used
```

Windows: `START-ABYSS.bat --bench` (the same options). The results go to `abyss-bench.txt` in the Abyss folder
(`--report` includes it). Nothing is sent anywhere, and the file has no host name, user name or folder path - read it,
then paste it into a [speed report](https://github.com/zephel01/Abyss/issues/new?template=bench.md).

What it runs, through the engine alone (the way the dashboard starts it):

| step | what | why |
|---|---|---|
| decode | three fixed questions, 256 tokens each, greedy, thinking off (after one warm-up answer) | the same work on every PC, so tok/s compare |
| prefill | prompts of 2k and 8k tokens of the Abyss docs (32k with `--full`) | prompt reading speed; on a model that does not fit GPU + RAM, each prompt reads the disk-only experts once whatever its length, so 2k can read slower per token than 8k (issue #15) |
| counters | the engine's per-request line (`glm stat`) of each answer | where a token's time goes |
| PCIe | the GPU's link, read while the engine works | links drop to Gen1 when idle, so an idle reading lies |
| SSD | 2 GB of the model's own GGUF, read from the disk (past the page cache) | the speed the engine's SSD tier gets |
| quality | 512 known tokens scored against `bench/ref/<model>` (a known-good run) | a broken kernel on this GPU shows as a large KL |
| `--deep` | the GPU timeline (`STRATA_GLM_GAP_PROF=1`) and five answers at PCIe shares 0 / 0.25 / 0.5 / 0.75 / 1 | what moves the speed here |

## Reading the result

The first line is the summary - one line to compare PCs (docs/MODELS.md's tables come from lines like it):

```
Abyss v1.3.55 | Ornith-1.5-122B-UD-Q4_K_XL | AMD Radeon AI PRO R9700 32 GB | PCIe Gen4 x4 (~7.9 GB/s), the GPU can do
Gen5 x16 | RAM 94 GB | SSD 4.9 GB/s | decode 56.4 tok/s | prompt 8k 960 tok/s | quality OK | bottleneck: gpu
```

### Where a token's time goes

```
- a token: 17.8 ms (56.2 tokens/s)
- SSD waits: 0.0 ms a token (0%), 0.0 expert reads a token
- iGPU lane: busy 7.1 ms a token (40%), 246 experts a token
- GPU memory: holds 47% of the experts a token uses; 0.4 a token come over PCIe from RAM
- drafter: 76% of its drafts accepted
```

| line | what it is | what a bad value means |
|---|---|---|
| a token | decode time per token (with the drafter: per accepted token) | - |
| SSD waits | time the GPU waited for experts read from the SSD | a quarter of a token or more: the model does not fit GPU + RAM |
| CPU / iGPU lane | time the lane was busy with the experts it computes (in parallel with the GPU) | close to the whole token: the lane is the critical path |
| GPU memory | the share of a token's experts already in GPU memory, and those copied from RAM over PCIe | many copies a token: PCIe carries the token |
| drafter | the share of the drafter's guesses the model kept | low: the drafter costs more than it saves (`--dspark` off, or another drafter) |
| tail skip | lossy setting on (its KL is in docs/MODELS.md) | - |

### The verdict

The benchmark names the likely bottleneck from those counters (rules of thumb, checked on an R9700 and an RTX 5090):

| verdict | rule | what moves it |
|---|---|---|
| **ssd** | SSD waits >= 25% of a token | a smaller quant that fits GPU + RAM, more RAM for the RAM tier (`STRATA_GLM_RAM_GB`), a faster SSD; or, changing the answers a little, the disk tail skip (`STRATA_GLM_TAIL_SKIP=4` + `STRATA_GLM_TAIL_SKIP_DISK=1`: DeepSeek V4.1 UD-IQ2_XXS on an R9700 +34%, KL +0.007) |
| **cpu** | the lane busy >= 80% of a token | `./abyss.sh --calibrate` (the lane's share and threads; AMD: the sweep's best share as `STRATA_GLM_PCIE_SHARE`), the CPU's power mode, the iGPU lane on a Radeon APU |
| **pcie** | >= 10 experts a token copied over PCIe | a better slot (Gen4 x16 carries 4x an x4 / OCuLink link), `--calibrate` (AMD: `STRATA_GLM_PCIE_SHARE`) |
| **gpu** | none of the above (or, with `--deep` and no drafter, the GPU busy >= 85%) | `./abyss.sh --gpu-tune`, and a report: the kernels for this GPU may have room |

More RAM for the RAM tier moves expert reads off the SSD, but the experts it holds are run by the CPU lane, so the
lane's time goes up by part of what the SSD waits go down. On a DDR4 desktop (123 GB, Gen5 x16, no iGPU, DeepSeek
V4.1) `STRATA_GLM_RAM_GB=88` against 80 cut the SSD waits from 11.1 to 8.7 ms a token but raised the lane from 9.4 to
10.6 ms; decode went 18.0 -> 18.6 tokens/s (+3%). Compare both lines before and after, not the SSD line alone.

An SSD-bound run (DeepSeek V4.1 Flash 1017 UD-IQ2_XXS, 300 GB, on the same PC) reads like this:

```
- a token: 50.1 ms (20.0 tokens/s)
- SSD waits: 19.5 ms a token (39%), 8.5 expert reads a token
- iGPU lane: busy 6.4 ms a token (13%), 96 experts a token
- likely bottleneck: the SSD: the experts that fit neither in GPU memory nor in the RAM tier are read from disk every token
```

"gpu" is also what is left when nothing else stands out - the memory tiers keep up, so the GPU's own kernels (and the
host between tokens) are what is left to speed up. That is where a report from a GPU we do not have helps most.

### `--deep`: the share sweep

The engine splits the experts that are not in GPU memory between the lane (computed on the CPU / iGPU, from RAM) and
PCIe (copied to the GPU). `--deep` answers the same question at five shares, with the engine's own share before and
after them (the sweep runs last, on warmer tiers and shorter answers than the decode above, so it is compared with
those two runs, not with the decode). Every run writes the same answer, so an untimed run first brings that answer's
experts into the tiers, and a share has to beat the faster of the engine's two runs (issue #19: a still-cold first run
read the SSD 6.8 times a token against 0.4 later, and the mean of the two made share 0 look 22% faster than the engine's
own share, which was 5% faster than it once warm). If the first run still read the SSD much more than the last, the
sweep says so: then the early shares were slowed by the disk, not by their share:

```
  share 0.00:   58.4 tokens/s  <- fastest
  share 0.25:   23.3 tokens/s
  ...
  share 1.00:   12.2 tokens/s
  the engine's own choice is as fast as the best share; share 1.00 is 4.8x slower - PCIe is the narrow path here
```

- The fastest share well above the engine's own share: a lead, not a setting yet. The sweep runs after the engine's
  own share has filled the GPU's pool (the experts that go over PCIe are promoted into VRAM, and a lower share
  promotes fewer), so a lower share set at startup can be slower than the engine's own: on a DDR4 desktop, Ornith's
  share 0 gave 34.7 tokens/s in the sweep and 21 from the start, against 24.7 for the engine's own (issue #14). Try it
  in the config's `"env"` as `"STRATA_GLM_PCIE_SHARE": "<share>"` and keep it only if a plain `--bench` gets faster.
- The engine already the fastest, and the curve steep: one path is much narrower - here PCIe (Gen4 x4 over OCuLink):
  every expert moved to it costs more than the lane's work.
- A flat curve: neither PCIe nor the lane limits the speed; look at the GPU (or the SSD).

### The quality check

```
quality: OK (KL 0.0078 vs a known-good run, same top token 98.2%)
```

The engine scores 512 tokens of known text (English and Japanese) one token at a time and compares each position's
top-32 log-probs with those the same model gave on our PC (`bench/ref/<model>.json` / `.topk`, made by
`tools/bench_ref.py`). The same PC gives 0 to ~0.01 (which experts the GPU and which the lane computes
varies run to run, and the two round differently); another GPU's arithmetic adds a little. **OK** is up to 0.02,
**slightly off** up to 0.05, **OFF** above - then this PC computes something the known-good run did not (a kernel bug
on that GPU, issue #9 style): please report it.

**slightly off** is expected on a smaller GPU: the references were made on a 32 GB card, and a 16 GB one leaves many
more experts to the CPU lane, which rounds differently (issue #15; here, GLM-5.3 Maya-S-v2 limited to 16 GB: KL 0.000
-> 0.025, V4.1 UD-IQ2_XXS: 0.008 -> 0.012). Only **OFF** points to a fault.

- The check needs the same model files: another quant, or a pack with an overlay (`tools/dense_overlay.py`), says
  "skipped" instead.
- Settings that change the numbers on purpose (`STRATA_GLM_TAIL_SKIP*`, `STRATA_GLM_ABL`, `STRATA_GLM_DEFER`) are
  left out of the scoring run, and the result says so.
- References ship for: Ornith-1.5-122B UD-Q4_K_XL, DeepSeek V4.1 Flash 1017 UD-IQ2_XXS, DeepSeek V4 Flash 0731
  UD-Q2_K_XL, GLM-5.3-Flash Maya-S-v2 IQ2_XXS.

### The drafter check

```
drafter check (greedy, the first answer): the same first 140 of 256 tokens with and without speculative decoding (a late near-tie: fine)
```

The first answer is asked again token by token (no drafts, no verify windows). A draft only decides how many tokens a
window checks, never which come, so the two answers should agree. A window adds its sums in another order than the
token path, so a near-tie can still part them later in the answer; parting within the first 12 tokens (**TOO EARLY**)
means the drafter or the window computes something wrong - please report it (v1.3.83 fixed one such bug: on CUDA the
verify windows lost their experts' weights).

## What limits this PC

`abyss-bench.txt` and `abyss-check.txt` end with a section built from the run's own measurements: the answer written
without the drafter (the drafter check's second answer) split into its parts, then what would change each part, the
largest estimated gain first.

```
A token without the drafter: 9.1 ms (110.5 tokens/s). Where it goes (estimates; the parts overlap a little):
- GPU compute: 6.0 ms (66%)
- CPU lane: 2.8 ms (31%), 142 experts a token
- PCIe fetches: 0.0 ms (0%), 0.0 experts a token
- SSD waits: 0.0 ms (0%), 0.0 reads a token
- host gaps (the GPU idle between tokens): 0.3 ms (3%)

What would change it (largest estimated gain first; tokens/s without the drafter - with it a VRAM gain can be larger: its verify windows give the lane more work):
1. VRAM: holds 64% of the experts a token uses - estimate: with 16 GB more (16 -> 32 GB class) ~82%; speed ~96 tokens/s (+21%) - the CPU lane takes 100% of the misses; measured on one model, the real gain was a little larger than this estimate
2. VRAM: holds 64% of the experts a token uses - estimate: with 8 GB more (16 -> 24 GB class) ~74%; speed ~89 tokens/s (+11%) - the CPU lane takes 100% of the misses; measured on one model, the real gain was a little larger than this estimate
3. PCIe: the GPU's link ran at Gen4 x4 (an expert copy measured 6.7 GB/s) - an x16 Gen5 slot (~63 GB/s) would change little here: the CPU lane already does the missed experts faster than such a copy
4. RAM: the RAM tier holds every expert the GPU does not (no SSD reads while writing) - no change needed
5. SSD: read 6.2 GB/s; waits under a tenth of a token - no change needed
```

| part | measured | the estimate |
|---|---|---|
| GPU compute | the rest of the token | not estimated: a faster GPU class scales it |
| CPU / iGPU lane | its busy time a token (`glm stat`) | moves with the VRAM and PCIe lines below |
| PCIe fetches | experts copied a token x the engine's own cost of one copy (its start line) | the same copy over an x16 slot of the GPU's generation (nominal rate vs the link it ran at); a miss goes the cheaper way, lane or copy |
| SSD waits | the engine's wait time a token | RAM: the model in GPU + RAM removes them (the size missing is shown); SSD: its read rate vs a typical NVMe Gen4 drive (~6 GB/s) |
| host gaps | the GPU idle between tokens (`--check`, `--bench --deep`) | not estimated: the CPU's single-thread speed and power mode set it |
| VRAM | the share of a token's experts already in VRAM | the engine's own routes (written while it runs) replayed through a cache of each size; misses x their measured cost |

The VRAM estimate: the routes replayed through a least-recently-used cache per layer, at the slots a layer has now
and with 8 / 16 GB more. The engine's placement adapts more slowly than an ideal cache - on a simulation of
Qwen3.8 at 12 / 16 / 24 GB its misses fell by about the square root of the ideal cache's ratio - so the estimate
takes that root. It gives the hit rate first; the speed follows from the misses' measured cost. Checked against a measurement (v1.3.112, Qwen3.8 16G64G without the drafter, 16 -> 32 GB): the hit rate estimated 82%, measured 82%; the speed estimated +21%, measured +26% (71.4 -> 90.0 tok/s) - the real gain was a little larger. With the drafter the gain is larger still: 89.2 -> 131.3 tok/s (+47%) on the same model. A part under a tenth of a token says "no change needed". They are estimates: the parts overlap, and
another model or another kind of work moves them. The section names sizes and classes only, never a product.

## The stability check (`--check`)

`--bench` measures speed; `--check` runs the installed model the way it is used and checks that it stays up and answers
right. Its output is what we look at first in a bug report, and we run the same check before a release.

```
./abyss.sh --check                 # ~15-30 minutes
./abyss.sh --check --quick         # ~5 minutes
./abyss.sh --check --long          # + the biggest prompts the context allows
./abyss.sh --check --config abyss-q38-16g64g.json   # another installed model than the last one used
```

Stop Abyss first. Windows: `START-ABYSS.bat --check`. Before the setup (or with `--setup`, `--gpu`, `--gpus`)
`--check` checks the PC only, as before. The check starts the model's own server (`serve/server.py` with the
installed config, as the dashboard does) on a free port of this machine only, and sends it:

| step | what | passes when |
|---|---|---|
| drafter | one greedy answer (thinking off) with and without the drafter | the same tokens; parting at token 12 or later is a near-tie in the sums (fine); earlier warns - `--bench` scores the two tokens |
| ja-explain, ja-arith | Japanese questions (`--quick`: the first) | Japanese, not broken (no U+FFFD, no loop); the arithmetic answer right |
| code-* | small Python tasks (`--quick`: one) | the answer's code passes its tests (run here in a temp folder, no network, 20 s at most) |
| tool-call | one `get_weather` call, then the answer after its result (when the model's template has tools) | a call with JSON arguments, then an answer with the result |
| long-* | prompts of 8K, 24K, 32K tokens and the context's top (up to 64K; `--long`: 64K, 128K, the whole context) with a needle, the drafter on | no crash, the needle found |
| conversation | 11 turns about one topic, a recall of two facts from the first turn, a Japanese summary | no crash, on topic, both facts recalled; the conversation reused on most turns |

Each row also has its decode and prompt tok/s, the tokens reused and the VRAM hit. A failed step keeps the engine log's
and the server's last lines in the file; a crash stops the steps after it (marked NOT RUN). The exit code is 1 only when a step FAILs (a crash, a wrong answer, a needle not found, code that fails its tests); a WARN is a note and exits 0. The file also has this PC
(GPUs, VRAM, driver, the PCIe link while working, CPU, RAM, OS, the SSD's read rate), the Abyss version and the model
(its folder's name and quant), and "What limits this PC". `abyss-check.txt` is in the Abyss folder, `--report` includes
it; the engine log is `abyss-check.log`. Nothing is sent, and the file has no host name, user name or folder path.
For a problem, open a [bug report](https://github.com/zephel01/Abyss/issues/new?template=bug.md) with
`abyss-report.txt` attached.

## Platforms

| | Linux, AMD | Linux, NVIDIA | Linux ARM, DGX Spark | Windows, NVIDIA | Windows, AMD |
|---|---|---|---|---|---|
| speed, counters, verdict, quality, `--deep` | yes | yes | yes (v1.3.113) | yes | yes |
| PCIe link | yes (sysfs, the narrowest link up to the root port) | yes (nvidia-smi) | – ("memory unified": the GPU shares the system memory) | yes (nvidia-smi) | no |
| SSD speed | yes | yes | yes | yes (unbuffered read) | yes (unbuffered read) |

An AMD card shows its own internal switch as x16; the benchmark walks up to the root port, so an OCuLink x4 or a
chipset slot shows as what it is.

## What else helps a report

- `./abyss.sh --check` (above): crashes and wrong answers on this PC; `--report` includes `abyss-check.txt`.
- `./abyss.sh --gpu-tune` (a minute, no model): the per-GPU kernel choices; paste `abyss-gpu-tune.txt` too.
- `./abyss.sh --report`: the PC, the setup and the engine log's speed lines (`abyss-report.txt`, includes the
  benchmark and the check).
- Before / after a change: run `--bench` twice and paste both - the counters show what the change moved.
