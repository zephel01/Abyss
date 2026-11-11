# Changelog

Every release is on GitHub (Releases) with these notes; every published change moves the last number. Update: the
dashboard's About > Update (from v1.0.18), or `git pull`, then `./setup.sh` (Windows: `START-MAYA.bat`) - it recompiles
only what changed and starts; the model is not downloaded again.

Abyss numbers its own releases from v1.1.0 (zephel01/Abyss). v1.0.31 and older below are Project Maya's, which Abyss
was copied from; DeepSeek V4 support (the `deepseek4` port) came on top of v1.0.31.

## v1.3.134 - 2026-11-12

Ornith-1.5 (122B / 397B) takes a smaller K/V cache, so a 16 GB card runs Ornith-1.5-397B at a 128K context (#44).

- **`--kv q4_0` / `k8v4` / `int8` on the qwen35moe path (Ornith-1.5).** The 4-bit and 8-bit caches were only on
  Qwen3.8's older attention path; on Ornith `--kv` was read for GLM-5.3's `int8` alone and `q4_0` / `k8v4` were
  silently ignored, so its attention layers always kept fp16 K/V - while the pool-shortage error suggested
  `--kv q4_0`. Now `q4_0` (K and V Hadamard-rotated, then ggml's q4_0: 144 B a row against 512), `k8v4` (int8 K +
  rotated q4_0 V) and `int8` (272 B) are written by the token, prompt, verify-window and NextN paths and read
  directly by the split decode kernel, the WMMA prompt kernel and the verify window; the freed VRAM goes to the
  expert pool, and the split cost model counts the smaller cache. At `--max-context 131072` Ornith-1.5-397B's cache
  is 4.00 GB in fp16, 2.12 / 1.62 / 1.12 GB in int8 / k8v4 / q4_0: capped at 16 GB (`STRATA_GLM_VRAM_GB=16`, 46 GB
  RAM tier) fp16 stops with `not enough VRAM for the expert pool ... (81 MB short)` while all three start and answer.
  Ornith-1.5-122B 24G64G capped at 16 GB with a 128K context: experts get 7.05 GB instead of 4.71 (1,470 slots
  instead of 980), decode 26.3 -> 34.9 tok/s, a 32K prompt 746 -> 1,654 tok/s. At 32 GB decode is unchanged
  (4K-64K) and prompts are 5-9% slower at 32K-64K (the WMMA tiles turn the codes back into FP16).
  KL against fp16 (122B 24G64G, 511 scored tokens): after an 8K English context q4_0 0.019 (same top token 93.9%),
  k8v4 0.013, int8 0.008; after 32K of code 0.007 / 0.005 / 0.003; after an 8K Japanese context 0.031 (91.6%) /
  0.021 / 0.014; nll within 0.013 of fp16 everywhere. A wrong `--kv` value with `--glm-pack` now stops with an error
  instead of being ignored. Default stays fp16 (the same tokens as v1.3.133); Qwen3.8's fast path keeps fp16.
- `setup --quant` help lists Ornith's `32G128G` (Ornith-1.5-397B).
- Manual (EN/JA): `--kv` row in Settings and "when VRAM is short" in the Ornith section.

Not run on an AMD card: the HIP build (gfx1201 / gfx1151) compiles and links.

## v1.3.133 - 2026-11-12

GLM-5.3 and DeepSeek V4 / V4.1 decode faster on NVIDIA, the same answers; and v1.3.132 builds on Windows again.
RTX 5090 (OCuLink Gen4 x4) + Ryzen AI Max+ 395, 123 GB RAM, iGPU off, 4K prompt + 512 tokens, greedy, median of 3
runs alternating with v1.3.132:

| model | v1.3.132 | v1.3.133 |
|---|---:|---:|
| DeepSeek V4 0731 UD-Q2_K_XL | 58.3 tok/s | **60.7** (+4.2%) |
| GLM-5.3-Flash Maya-S-v2 IQ2_XXS | 46.8 | **49.1** (+5.0%) |
| DeepSeek V4.1 Compact JA256 (32G96G) | 50.0 | **52.2** (+4.4%) |
| DeepSeek V4.1 1017 UD-IQ2_XXS (SSD-bound) | 21.4 | 21.6 (+1.1%) |
| Qwen3.8-Flash-Next 16G64G | 128.0 | 127.0 (unchanged path) |

- **Tier promotions cross the link while the CPU lane computes (GLM-5.3, DeepSeek V4 / V4.1, CUDA).** An nsys trace
  of V4 0731 showed the PCIe link ~96% idle for the ~215 us a layer between the route and the CPU lane's answer,
  while the background moves queued after the answer ran into the next layer's route (12 -> 25 us; 2 MiB pieces
  added ~60 us a layer, which is why more moves had been slower). Now, when the lane takes a route's ne RAM-tier
  experts, its service thread lets ne x 512 KiB - 128 KiB of the queued promotions go at once on a second copy stream
  (`STRATA_GLM_BG_LANE=<KiB a lane expert>:<KiB off>[:<kinds>[:<max KiB>]]`, default `512:128:1` there, `0` off).
  VRAM hits V4 72.7 -> 77.0%, GLM-5.3 61.6 -> 66.6%; CPU lane V4 7.0 -> 6.2 ms a token, GLM-5.3 11.8 -> 10.7. With a
  24 GB pool V4 +8.1%, GLM-5.3 +4.0%; 16 GB +0.5% / +2.8%; 32K prompts GLM-5.3 +4.6%, V4 +0.6%; V4 + DSpark
  unchanged. Not for Qwen3.8 (its ~30 us lane experts leave no gap: -7.5%) or HIP (not measured on AMD).
  `STRATA_GLM_BG_IDLE=<KiB>:<kinds>:<kinds>` (experimental) sends a fixed amount after the route instead.
- **Windows: the CUDA build passes again.** v1.3.132's `qwen35_fast.cu` includes `<cooperative_groups.h>`, whose CUDA 13
  CCCL stops with `#error` under MSVC's traditional preprocessor; CUDA files are now compiled with
  `-Xcompiler=/Zc:preprocessor` under MSVC.
- Manual (EN/JA): `STRATA_GLM_BG_LANE`, `STRATA_GLM_BG_IDLE`.

Answers: `STRATA_DETERMINISTIC=1` greedy md5 equal to v1.3.132 on V4 0731, GLM-5.3 and V4.1; with the exact lane and
moves on (`STRATA_GLM_CPU_EXACT=1 STRATA_GLM_LANE_XC=1`) V4 and GLM-5.3 give v1.3.132's tokens with 4-10 points
different hits. AMD: the changed HIP files compile for gfx1201 / gfx1151; not run on an AMD card.

## v1.3.132 - 2026-11-12

Decode speed on NVIDIA for all four models, the same answers. RTX 5090 (OCuLink Gen4 x4) + Ryzen AI Max+ 395, 123 GB
RAM, iGPU off, 4K prompt + 512 tokens, greedy, median of 3 runs against v1.3.131 (Qwen3.8: 3 prompts x 3 runs):

| model | v1.3.131 | v1.3.132 |
|---|---:|---:|
| Qwen3.8-Flash-Next UD 3.1 UD-IQ3_XXS, no drafter | 95.4 tok/s | **142.2** (+49.0%) |
| Qwen3.8-Flash-Next UD 3.1 UD-IQ3_XXS, MTP | 143.2 | **197.6** (+38.0%) |
| DeepSeek V4 0731 UD-Q2_K_XL | 53.7 | **58.2** (+8.3%) |
| GLM-5.3-Flash Maya-S-v2 IQ2_XXS | 43.8 | **46.0** (+5.1%) |
| DeepSeek V4.1 Compact JA256 (32G96G) | 47.3 | **49.0** (+3.6%) |
| DeepSeek V4.1 1017 UD-IQ2_XXS (SSD-bound) | 20.0 | 20.3 (+1.4%, within noise) |
| Ornith-1.5-122B 24G64G | 59.0 | 60.4 (+2%, within noise) |

- **The background tier moves keep what the text is using (GLM-5.3, DeepSeek V4 / V4.1, Qwen3.8).** They ranked
  experts by integer route counts halved every 32K routes, which swapped often and followed the current text poorly.
  They now rank by counts that decay by 0.995 every token boundary (about 200 tokens of memory); a RAM-tier expert
  replaces its layer's coldest resident when its count passes that one's by 2, at most 8 swaps a boundary
  (`STRATA_GLM_BG_DECAY` / `_N` / `_MARGIN`, `0` = the old counts). Weights move, the arithmetic does not: VRAM hits
  V4 66 -> 72%, GLM-5.3 57 -> 62%, Qwen3.8 UD 3.1 93 -> 97.5%, and the CPU lane takes 10-20% fewer experts a token.
  Qwen3.8 alone: 128.5 -> 136.1 tok/s (MTP 185.5 -> 196.6); with a 16 GB pool +5.9% / +11.1%; on a CPU without AVX-512
  (AVX2 rows) +8.3% / +11.7%, because the CPU lane does a third of the work it did. GLM-5.3 / V4 with a 24 GB pool
  +3-4%, 16 GB unchanged; 32K prompts +3%; V4 + DSpark unchanged. On HIP only Qwen3.8 uses it (the others not measured
  on AMD).
- **Qwen3.8 on RTX 50 cards: programmatic dependent launch (PDL) on by default** (discrete sm_120+; it was on for the
  DGX Spark only): a kernel starts while the one before finishes. UD 3.1 4K + 512: 128.0 -> 137.8 tok/s without a
  drafter, 178.6 -> 184.7 with MTP; 16G64G 121.5 -> 129.0 / 168.9 -> 175.1. `STRATA_GLM_PDL=0` turns it off. The MTP
  verify window's per-layer CUDA graphs are captured with PDL edges too (`STRATA_Q4X_GRAPH_PDL`; same speed, kept for
  the next window work). GLM-5.3 / V4 / V4.1 measured no gain and keep PDL off on a discrete card.
- **Qwen3.8 decode, smaller pieces (CUDA, each the same bits):**
  - the delta-net middle (alpha / beta, conv + SiLU, the recurrence) in one launch after the qkv / z GEMV
    (`STRATA_Q4X_GDN_MID`; sm_90+; `STRATA_Q4X_GDN_MID_PF=1` also pulls `ssm_out` into L2, off: slower);
  - the hyper-connection mix fused for the Q8_0 down / up + F32 inject of the unsloth UD 3.1 and our 16G64G packs
    (was five launches);
  - the route's top-k by ranks over 8 warps (8.8k -> 7.8k cycles in the route kernel), QSA top-k digit search by 256
    threads (14.5 -> 7.2 us a query at 4.6K cells), the step record by a kernel instead of two copies queued behind the
    expert copies, the indexer rope a thread a pair, argmax over 128 blocks (17 -> 4.2 us a token), the next token's
    boundary while the head runs (209 -> 16 us gap), the greedy token's PLE rows prefetched;
  - CPU lane (AVX-512): IQ2_S 17.4 -> 11.3 ns, IQ3_XXS 7.0 -> 5.8 ns, IQ4_NL down 10.6 -> 5.6 ns a block, the window's
    columns quantized and summed on the pool;
  - background copies paced at 512 KiB a MoE layer (was 1 MiB: the next layer's route writes waited behind it), and
    700 MB of VRAM kept free on every CUDA card (was 1.5 GB over 12 GB: Qwen3.8 has no DSpark; 432 more expert slots
    on a 5090). GLM-5.3 / V4 / V4.1 keep 1 MiB and 1.5 GB (512 KiB undid the decayed counts' gain on their 7-8 MB
    experts; 1.5 GB is v1.3.82's fix for V4.1 + DSpark at 64K).
  Together on UD 3.1 (merged build against v1.3.130): 97.6 -> 131.2 tok/s without a drafter, 147.1 -> 186.1 with MTP.
- **Qwen3.8: the same tokens from run to run with default settings for prompts up to ~2,000 tokens.** The exact
  prompt's light RAM-tier experts went to MMQ under `STRATA_GLM_PREFILL_CPU=0` (which `STRATA_DETERMINISTIC` sets), so
  a second request's numbers followed which experts the first one had moved into VRAM; and UD 3.1's four layers the
  exact prompt cannot take (Q2_0 down weights) kept the CPU's timed share. Both fixed: three processes and one with
  MTP give the same tokens as `STRATA_DETERMINISTIC=1`. Longer prompt chunks keep the timed share and can still part.
- Debug: `STRATA_GLM_CPU_LANE_BITS=<n>` compares one RAM-tier expert's CPU-lane rows with the GPU's as bits on n
  random inputs per layer (UD 3.1: 0 of 1,474,560 rows differ); `STRATA_GLM_ROUTE_LOG` records the starting residents.
- Manual (EN/JA): `STRATA_GLM_BG_DECAY` / `_N` / `_MARGIN`, `STRATA_Q4X_GRAPH_PDL`, `STRATA_Q4X_GDN_MID` / `_PF`; the new
  defaults of `STRATA_GLM_PDL`, `STRATA_GLM_BG_PACE_KB` and `STRATA_GLM_RESERVE_MB`.

Answers: `STRATA_DETERMINISTIC=1` greedy md5 equal to v1.3.131 on Qwen3.8 (UD 3.1, 16G64G, GSQ IQ3_S; no drafter and
MTP; code / Japanese / 4K x 256), GLM-5.3, V4 0731 and V4.1. Known: on GLM-5.3 and V4.1 the deterministic tokens
still depend on the VRAM pool's size (the prompt's light kernel takes only resident experts), as in earlier releases.
AMD: the changed HIP files compile for gfx1201 / gfx1151; not run on an AMD card (none attached).

## v1.3.131 - 2026-11-11

A fix release for two AMD reports - thank you, hondaru (#42) and tim6300243 (#43):

- **Qwen3.8 with drafts (MTP, DFlash 2) on AMD GPUs (HIP) decodes at its pre-v1.3.125 speed again.** v1.3.125 gave
  every dense GEMV of Qwen3.8's verify window (the hyper-connection mixes, delta-net / attention projections,
  `ssm_out` / `attn_output`, the router and shared expert, the output head) the token path's sums, so greedy answers
  with drafts equal the ones without. On CUDA that GEMV reads the weights once for all columns; on HIP it ran one GEMV
  a column, so each verify window read the ~4 GB of dense weights and the head once per column. An AI PRO R9700 on
  Qwen3.8 IQ3_XXS + MTP went 87.0 -> 67.2 tok/s. HIP now keeps the shared-read GEMVs of v1.3.124
  (`STRATA_Q4X_VERIFY_EXACT`: `1` on CUDA as before, `0` on HIP; `STRATA_DETERMINISTIC` sets `1`). On HIP, greedy
  answers with drafts can again part from the ones without late in a long answer, as up to v1.3.124. CUDA is
  unchanged.
- **HIP: one host read less per CPU-lane wait.** The wait read the exact-combine mask even with the exact combine off
  (HIP's default); it now reads it only with the combine on.
- **Windows + AMD: the HIP build passes with Visual Studio's MSVC 14.51** (#43, RX 9060 XT; also #2 / #3). MSVC
  14.51's `<cmath>` makes `isgreater` .. `isunordered`, `isnormal` and `signbit` `constexpr` (under HIP implicitly
  `__host__ __device__`), and TheRock 7.14.1's Clang 23 then fails CMake's HIP compiler test ("`__device__` function
  'isgreater' cannot overload `__host__ __device__` function"). LLVM fixed it by including
  `__clang_cuda_math_forward_declares.h` before `<cmath>` in `__clang_hip_runtime_wrapper.h` (01d3932); no Windows ROCm
  SDK has that yet (it comes with ROCm 10.1). Setup on Windows now makes the same one-line move in the SDK's own copy
  of that header when MSVC is 14.50 or newer, once, keeping the original as `.abyss-orig`; `--check` only warns,
  `ABYSS_NO_ROCM_HEADER_FIX=1` leaves the SDK alone. README, the manual and [docs/amd](docs/amd/README.md) (EN/JA) give
  the manual step for an SDK setup cannot write to.
- Manual (EN/JA): `STRATA_Q4X_VERIFY_EXACT`.

Not run on an AMD card or on Windows here (none attached today): the HIP files compile, the header move was checked against the reported error with ROCm's Clang 23 on Linux, and the CUDA answers are unchanged.

## v1.3.130 - 2026-11-11

Four findings from LarsDrakblod's RTX 4090 field report (#39) - thank you:

- **`--vram-reserve-mib` now works on every model.** The flag was parsed and honored by the qwen4exp path's
  `--expert-cache auto`, but the GLM runner's expert pool (Qwen3.8, GLM-5.3, DeepSeek V4.1) read only
  `STRATA_GLM_RESERVE_MB`, so the flag in a Qwen3.8 config did nothing. The flag now sets that reserve too (also the
  verified runner's pool, which had 1536 MB written in). With both set, the flag wins and the log says so.
- **Windows build**: `glm_fast.cu`'s `__builtin_memcpy` (which MSVC lacks) is a plain `memcpy`.
- **Windows setup**: `tools/abyss_setup.py` finds CMake, Ninja and Python in `.venv/Scripts/` (Windows' venv layout)
  instead of `.venv/bin/`.
- **Manual**: `STRATA_DETERMINISTIC` is a test switch, not a runtime setting - it freezes the expert tiers, and on a
  PCIe-bound RTX 4090 it cost 3.7x (54.1 -> 14.6 tok/s); for the same answers in normal use Qwen3.8 has the exact lane
  combine (`STRATA_GLM_LANE_XC`). The DFlash 2 note adds the 4090's tie with the MTP layer (55.6 vs 56.4 tok/s at
  86-89% vs 78% acceptance): on a PCIe-bound card a better drafter does not change each window's expert fetches.

- **Qwen3.8 with the iGPU lane no longer decodes slower than without it** (Ryzen AI Max+ 395 + RTX 5090, Qwen3.8
  16G64G, v1.3.129: iGPU on MTP 132 / default 89 tok/s, iGPU off 141 / 94; with this the iGPU-on box takes the
  CPU lane, as iGPU off). The start-up timing that picks the iGPU or
  the CPU for the RAM-tier lane saw a near tie on Qwen3.8 (an expert 0.022 ms vs 0.019-0.025 on the CPU) and flipped
  run to run, and every run that picked the iGPU was slower. The iGPU now takes the lane only when it times at least
  1.5x faster (`STRATA_GLM_IGPU_MARGIN`; GLM-5.3, V4 and V4.1 time it 1.9-2.7x faster and keep it), and never with
  the exact combine (`STRATA_GLM_LANE_XC`, Qwen3.8's default): the iGPU's sums are not the GPU's bits, so the
  same-answers-every-run combine needs the CPU lane. `STRATA_GLM_IGPU=1` still forces the iGPU.

- **Opt-in: an expert's SSD chunks all read at once** (`STRATA_GLM_DISK_PAR=1`, off by default until measured). While
  answering, an expert a token misses is read as 24 chunks (3 parts x `STRATA_GLM_READ_CHUNKS` 8) by 9 threads, about
  three rounds; the switch gives these reads a pool of their own with a thread a chunk (`<n>`: n threads). Only the
  order of the reads changes, so the answers are the same. fio on the 6.7 GB/s NVMe: 9 reads at once 4.64 GB/s, 24 at
  once 5.10. `STRATA_GLM_TIMING=1` now also prints the disk waits fio-style (rate while waiting, mean / p50 / p90 / p99
  of a one-expert wait), with the switch on or off.
- **A GPU lost during the start-up fill now says so.** The warm-up checks its copies to VRAM and the table upload and
  names the CUDA error with a hint (Xid in the kernel log, reboot, reseat an external link), instead of "the warmed
  expert tables did not upload" (an RTX 5090 on OCuLink fell off the bus there on 2026-11-10).

- **A bigger automatic RAM tier on machines with 100 GiB of RAM or more** (Linux, a discrete GPU, no
  `STRATA_GLM_RAM_GB`). The cap was 73% of the RAM (`STRATA_GLM_RAM_FRAC`), set on a 94 GB machine where pinning
  more thrashed; the room the page cache, the server and the desktop need does not grow with the RAM, so a 123.5 GiB
  machine left 34 GB unpinned (88.7 GB tier). From 100 GiB of RAM the cap is now the larger of 73% and the RAM less
  24 GB (`STRATA_GLM_RAM_KEEP_GB`): 99.5 GB there (227 GB on a 251 GiB machine), still below the free RAM less
  `STRATA_GLM_RAM_HEADROOM_GB`. DeepSeek V4.1 UD-IQ2_XXS on an RTX 5090 (iGPU off): 30.2 -> 34.5 tok/s. 64 and 96 GB
  machines keep the 73% cap byte for byte (45.6 / 68.6 GB on 62.5 / 94 GiB); setting `STRATA_GLM_RAM_KEEP_GB` uses the
  keep there too. If other programs grow by more than ~8 GB after the engine starts, set `STRATA_GLM_RAM_KEEP_GB=32`.
  `STRATA_GLM_RAM_FRAC` set on its own keeps the old cap; unified memory (DGX Spark, APUs) and Windows are unchanged.
  The `RAM tier ... pinned` log line now says how its budget was set.

## v1.3.129 - 2026-11-10

- **GLM-5.3 and DeepSeek V4.1 decode back to v1.3.123's speed.** `STRATA_GLM_CPU_EXACT` (v1.3.124: the CPU lane in
  the GPU's arithmetic) was on for every model on x86, and its rows for GLM's and V4.1's quant types are slower than
  ggml-cpu's: on an RTX 5090 (RAM tier 60 GB, iGPU off) GLM-5.3 Maya-S-v2 went 44.6 -> 38.2 tok/s and V4.1 UD-IQ2_XXS
  21.4 -> 20.2. It is now on by default for Qwen3.8 only, whose same-answers-every-run combine needs it; `=1` turns it
  on for the others. GLM-5.3: 38.2 -> 44.0 tok/s, V4.1: 20.2 -> 21.3; Qwen3.8 unchanged. Found by re-measuring each
  week's last release on the same machine.

## v1.3.128 - 2026-11-10

- **Windows build fix**: v1.3.127's exact lane sum (`q81::dev_sum`) was guarded by `__FMA__`, which MSVC does not
  define with `/arch:AVX2`, so `iq_avx512_vbmi.cpp` did not compile on Windows. Linux is unchanged.
- **Qwen3.8: same answers run to run now also with 16 GB of VRAM** (the exact lane combine is on for every VRAM
  size on x86 CUDA). With 16 GB it was ~4% slower in v1.3.127: the GPU read the CPU's int64 partial sums over PCIe,
  8 bytes a row against 4. The kernel now also posts its own int64 sum to the host; when it is there by the time the
  CPU finishes, the CPU adds both, rounds once and sends floats (else the int64 path as before; the same bits either
  way), and the CPU's fixed-point sum is vectorized. RTX 5090, Qwen3.8 16G64G at 16 GB: 96.4 -> 103.0 tok/s (the
  older behaviour 100.6), runs that parted 0 of 20 at 16 GB and at 32 GB.

## v1.3.127 - 2026-11-10

- **Qwen3.8: the same greedy answers run to run in the normal mode** (no `STRATA_DETERMINISTIC`), on x86 CUDA with
  24 GB of VRAM or more. The CPU lane's share of the experts follows timing; its experts now give the GPU's bits
  exactly (the GPU's float order and its exp in the SiLU) and the layer's expert sum is taken in fixed point (each
  weighted row to a 64-bit integer of 2^-34, rounded to a float once), so it does not matter which experts the CPU
  computed (`STRATA_GLM_LANE_XC`). The prompt's small experts follow the same arithmetic on either side
  (`STRATA_GLM_PREFILL_XROWS`, 32 rows). RTX 5090, Qwen3.8 16G64G, MTP on, 5 prompts: runs that parted 15 of 20 -> 0
  of 20; decode within noise. Prompt chunks longer than about 2,000 tokens can still part. With 16 GB of VRAM decode
  was ~4% slower, so it stays off there for now (`STRATA_GLM_LANE_XC=1` turns it on); HIP, ARM and the other models
  are off by default (`=1` on the other models: not measured yet).

## v1.3.126 - 2026-11-10

- **Anthropic `/v1/messages`: images inside `tool_result` blocks reach the model (#41).** An image tool's result (a file
  reader, a browser screenshot) went through the text-only path and lost its images; it goes through the same parts
  path as role "tool" on `/v1/chat/completions` now (text-only results stay plain strings). A top-level image next to
  a `tool_result` in the same user message is kept too. Reported with the fix by gyroid-eth.
- **DFlash 2 draft step 1.96 -> 1.14 ms** (RTX 5090, Qwen3.8), the drafts bit for bit the same: the drafter's
  attention reads each K/V chunk once into shared memory for the four q heads of a KV head (95 -> ~10 us a layer),
  RoPE angles once a draft, q8_1 bytes written by the kernels that make the values (31 launches gone), and the whole
  step as one CUDA graph (`STRATA_Q4X_DFLASH_GRAPH=0` launches kernel by kernel). Qwen3.8 16G64G: code 159.3 -> 165.1
  tok/s (MTP 153.5), lists 153.6 -> 159.8 (MTP 158.7); Japanese prose and bug fixes unchanged (MTP still faster
  there: it is the acceptance, not the step). MTP stays the default. Answers with `STRATA_DETERMINISTIC=1` still
  equal the token path's for no drafter / MTP / DFlash 2.

## v1.3.125 - 2026-11-10

- **Qwen3.8: greedy answers with drafts (MTP, DFlash 2) are the same tokens as without.** A few small Q8_0 matrices
  in the q4x verify window (router / shared expert, the shared expert's gate, the hyper-connection's down / up, the
  PLE key / value) went through gf::mvn - one warp a row - where the token path's mv() splits them four warps a row;
  the sums parted at near-ties (token 7 of a code prompt, token 29 of a Japanese one). They go through glm_mvn
  (mv()'s sums a column) now. The prompt path's MMQ stream-k split its K by the tile count, which moved with the
  expert placement; `STRATA_MMQ_WHOLE_K=1` (third_party/ggml/mmq-whole-k.patch, on by default for Qwen3.8) runs a tile
  a block. With `STRATA_DETERMINISTIC=1`, 45 answers (code / list / Japanese / bug fix / 4K prompt, 3 runs each)
  without a drafter, with MTP and with DFlash 2 equal the token path's; same speed. `tools/spec_exact_check.py`
  checks it. In the normal mode the CPU lane's share follows timing, so runs can still part slightly.
- **Qwen3.8 DFlash 2 drafter (opt-in, `STRATA_Q4X_DFLASH=<gguf>`).** `./setup-deepseek.sh --model qwen38 --dflash`
  downloads it (0.65 GB, SHA-256 checked); `--no-dflash` takes it out. RTX 5090, Qwen3.8 16G64G: code 152.3 tok/s
  (MTP 140.2), lists 142.1 (137.2), Japanese prose 91.9 (105.7), bug fixes 122.0 (128.2) - MTP stays the default.

## v1.3.124 - 2026-11-10

- **The CPU lane computes in the GPU's arithmetic (x86).** For the RAM-tier experts the CPU lane quantized
  activations its own way (one scale per 256 values), so an expert gave numbers 2-3% apart on the CPU and on the GPU.
  It now uses the GPU kernels' 8-bit activations (one scale per 32) and their integer sums: the difference is about
  1e-7, checked on all 20 quant types (`lane_q81_test`). On Ornith-1.5-397B the first token of the same question
  changed in 2 runs of 48 (was 4 of 24). Same speed (RTX 5090: Ornith-397B 42.9 tok/s, Qwen3.8 84.5 tok/s, both
  within run-to-run noise). `STRATA_GLM_CPU_EXACT=0` brings back the old dots; ARM CPUs (DGX Spark) keep them.
- **Monitor and Terminal fixes** (found using v1.3.123's new view on real models): the "bench" numbers show only for
  the running model's `--bench` (another model's run was shown); the drafter row says MTP / NextN / DSpark (or just
  "Drafter") instead of "Drafter (NONE)"; "memory is tight" needs PSI some at 10% or more, or swap growing by 256 MB
  in the last minute - not old, idle swap; while idle (the last answer's verdict) the highlighted card is blue with a
  "LAST ANSWER" tag instead of a red alarm.

## v1.3.123 - 2026-11-09

- **`./abyss.sh --check --finance`: Japanese finance rules on synthetic accounts.** 17 questions: FX / CFD / crypto
  margin, 信用取引 margin and margin calls, swap points and dividend adjustments kept apart from price P/L,
  listed-stock tax, NISA limits, a foreign-currency deposit's yield; three where a needed figure (a collateral
  haircut, a loss-cut rate, a swap rate) is missing and the right answer says so; one that asks for a firm buy / sell
  call. Every question states the rule it uses, so a model is not marked down for a rule that changed after its
  training; the rules sit in one table in `tools/finance_check.py` with their sources and dates. Only what does not
  depend on the market is graded, from the final `答え:` lines. The score is "n of 17 items as the rules say" in
  `abyss-finance.txt` - not investment advice, and not a measure of how good a model is at finance. It works against
  any OpenAI-compatible server too (`python tools/finance_check.py --url ...`).
- **The web Monitor shows what Abyss Terminal shows, with the bottleneck at a glance.** A banner names what holds a
  token back (SSD, GPU, CPU lane, PCIe...) with how much of the token and a fix; a token's time as one stacked bar
  (the bottleneck's part marked); the card that is the bottleneck (Disk, CPU, PCIe, GPU) outlined; plus the model's
  badges and live state, decode / prompt mean / last / bench, memory tiers (context, VRAM, RAM tier, system RAM,
  drafter), the cards table and recent requests with the drafter's share. The numbers come from the Terminal's own
  code on the server (`GET /metrics?view=1`), not a second copy in the browser.
- **The dashboard says Abyss, not Maya.** The page title, the name at the top left, the message box, the version and
  update notes, and the server's log lines (`[abyss]`).
- **Ornith-1.5-397B 32G128G is on Hugging Face:**
  [zephel01/Ornith-1.5-397B-IQ2_XXS-32G128G-GGUF](https://huggingface.co/zephel01/Ornith-1.5-397B-IQ2_XXS-32G128G-GGUF)
  (`./setup-deepseek.sh --model ornith --quant 32G128G`). Known: in Abyss its first tokens can differ between
  identical greedy runs - the CPU lane computes the experts it takes with slightly different rounding from the GPU
  (activations quantized per 256 values instead of per 32), and the set it takes changes every token. A fix (the
  CPU lane with the GPU's rounding) is in progress; `STRATA_GLM_CPU_LANE=0` gives identical runs but is much slower.

## v1.3.122 - 2026-11-09

- **Ornith-1.5-397B: 10 experts a token.** Its GGUF (qwen35moe) routes 10 of 512 experts a token; the loader took
  at most 8 and refused the file ("expert_used_count are off"). It now takes up to 16, as Qwen3.8 already did. On an
  RTX 5090 + 128 GB RAM our 114 GB IQ2_XXS build runs at **42.0 tok/s** (8K prompt 882 tok/s, NextN drafter 79%
  accepted). `./setup-deepseek.sh --model ornith --quant 32G128G` sets it up (the Hugging Face repo
  zephel01/Ornith-1.5-397B-IQ2_XXS-32G128G-GGUF goes up next).
- **Three buffers sized for 8 experts, now for up to 16:** the next-layer prediction statistics (they wrote past
  their end with 10 experts, Qwen3.8 included - statistics only, the tokens were not affected), the #24 expert
  helper's single-column buffer, and the iGPU lane's single-column buffer.
- **`--model glm` setup: the pack step works again.** v1.3.121 built the GLM pack without `--compat-bf16`, so the
  setup stopped at the pack (`hc_attn_base is F32`).
- **`--bench` checks our GLM-5.3-Flash files too (#40).** bench/ref now has references for GLM-5.3-Flash-IQ3_XXS-32G128G
  (Q120d) and GLM-5.3-Flash-IQ2_XS-32G96G (S1), so the bench prints a KL against a known-good run.
- **Docs: `120a` on RTX 50 (#40).** The hand-build section of the manual has the note in a box, with the ptxas message
  (`mma with block scale not supported on .target sm_120`) and `121a` for DGX Spark.

## v1.3.121 - 2026-11-09

- **ISTA-DASLab's GSQ-RCO IQ1_M (DeepSeek V4.1) reads prompts at full speed again.** Its layers keep IQ1_S gate / up
  experts beside IQ1_M down experts; the prompt path's expert decoder did not list IQ1_S, so such a layer fell back to
  one token at a time (~5 tok/s). IQ1_S is now decoded like the other i-quants: an 8K prompt 5 -> **234 tok/s**
  (RTX 5090 capped at 12 GB + 128 GB RAM), decode unchanged (26.7 tok/s). The prompt path against one token at a time over 463 positions: KL
  0.0017, the same top token 97.8%.
- **One-command setup for every own GGUF.** `./setup-deepseek.sh --model qwen38` (`--quant 12G32G`, `16G64G` - the
  default - or `24G64G`), `--model glm` (`32G96G` or `32G128G`), `--model ornith --quant 24G64G`, and our DeepSeek V4
  0731 files as `--quant 16G64G` / `32G64G` / `32G96G`. The quant names the PC size and picks the Hugging Face repo;
  the setup says when the PC is smaller than the file was made for, writes the RAM tier the model card was measured
  with (64 GB PC: 46 GB), and leaves the DSpark drafter off for our V4 files (faster without it on a discrete GPU).
- **README: pick a model for your PC first.** A table of our GGUFs by GPU and RAM size with the setup command for
  each, then the quick start, requirements and the disk space each model needs (GGUF + pack + drafter); our files
  under their current Hugging Face names (the earlier names in a note); the news at the end.
- **`--calibrate` and temperature 0 in the docs (#39).** The setup ends by pointing to `./abyss.sh --calibrate` (the
  CPU lane / PCIe split for this PC: +12% on an RTX 4090 + 96 GB in #39), and the manual has a section on it. For coding
  agents the manual notes that temperature 0 (`"sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1}`) raises the
  drafter's acceptance (76 -> 86%, ~+20% in #39); the default stays the model card's sampling. `--spec-min-p 0.5` as a
  default is still being measured.

## v1.3.120 - 2026-11-08

- **Windows builds again (#31).** Since v1.3.109 `src/core/hop_link.cu` included `sys/socket.h` and `unistd.h`, which
  Windows does not have, so a stock Windows build failed. They are now behind `#ifndef _WIN32`. On Windows
  `STRATA_GLM_HOP=sock` (the in-process socket hop of a two-GPU split) logs one line and uses the host-memory hop,
  the default; nothing else changes. Linux and DGX Spark builds are unchanged.

## v1.3.119 - 2026-11-07

- **GLM-5.3-Flash's one-GPU verify window is faster on DGX Spark.** Three changes to v1.3.118's window, each with the
  same sums as before: the absorbed attention k_b / v_b GEMVs of all columns run in one launch each, every weight read
  once (`STRATA_GLM_MLA_COLS`); column 0 writes its KDA state straight into the rewind copy and the next column reads
  it from there - no 2 x 146 MB of state copies a window (`STRATA_GLM_KDA_OOP`); and the column-batched expert
  launches put the column first in the grid, so an expert two columns share is read from memory once and from L2 the
  second time (`STRATA_GLM_COLS_ORDER`, CUDA). DGX Spark (GB10), 4K prompt, median of 3 x 512 tokens, 512 greedy
  tokens identical to v1.3.118:
  - our GLM-5.3-Flash S1 (94 GB): 24.3 -> **27.4 tok/s (+13%)**;
  - unsloth UD-IQ2_XXS: 22.9 -> **24.2 tok/s (+6%)**.
- **DeepSeek V4 / V4.1 windows and the DSpark drafter take the same launch order** (`STRATA_GLM_COLS_ORDER`, one
  switch for every model): DGX Spark, DeepSeek V4 0731 + DSpark drafter 23.74 -> **25.22 tok/s (+6%)**, the same
  tokens. RTX 5090: 48.9 -> 50.7 tok/s (within noise), the same tokens.
- **Unified memory (DGX Spark): the prompt path's tail goes on top of a pool that holds every expert**, as on a
  discrete card since v1.3.104 (`STRATA_GLM_TAIL_ON_TOP=0`: the old layout). A prompt no longer takes 20-46 slots a
  layer from the experts, so the decode after it stops re-reading experts from the SSD (Qwen3.8 after a 32K prompt:
  1.3 a token -> 0); the pool takes the tail's few GB more of the shared memory. The experts that sat on the SSD now
  come from memory, so greedy tokens differ from v1.3.118's on the Spark (with `=0` they are v1.3.118's: checked for
  Qwen3.8 MTP off at 4K / 32K, GLM S1 and V4 0731 + DSpark). Spark, the whole release, 4K: GLM S1 28.6 tok/s, V4
  0731 + DSpark 26.0.
- **Qwen3.8 on DGX Spark: the decode after a long prompt +13%.** ISTA IQ3_XXS with the MTP drafter, median of 3 x 512
  tokens: 4K prompt 54.7 -> **55.8 tok/s (+2%)**, 32K prompt 47.5 -> **53.4 tok/s (+13%)** - the tail above and:
  - a verify window's routes, one per column, run in one launch (`STRATA_GLM_ROUTE_COLS`, on for Qwen3.8; it was
    48 x T launches a window, ~13 us each on a Spark) - the same plans in the same order, the same tokens;
  - the window's dense GEMVs start with programmatic dependent launch and pull their weight rows into L2 before they
    wait, as the token path's do (`STRATA_GLM_MVN_PDL`, CUDA; the same tokens): Spark 4K 52.5 -> 55.8 tok/s, 32K
    51.5 -> 53.4 of the gain; RTX 5090 the same speed.
- Build switches `-DGLMF_GU_DEC_WARPS=8|16|32` and `-DGLMF_DOWN_ROWS=2|4|8` (defaults 16 / 4, as before): the decode's
  expert gate/up warps a block and down rows a block, for per-GPU tuning; every choice gives the same bits.
- `STRATA_Q35_MTP_Q4=1` (experiment, off): Qwen3.8's MTP drafter reads 4-bit copies of its Q8_0 projections and head,
  made at load - a faster draft step, fewer drafts kept (DGX Spark 87.6% -> ~80%).
- Same-token checks on the RTX 5090 (`STRATA_DETERMINISTIC=1`, 512 tokens): GLM-5.3-Flash S1 (token path and
  `STRATA_GLM_WINDOW=1`), DeepSeek V4 0731 + DSpark, and Qwen3.8 ISTA IQ3_S (MTP off at 4K and 32K, MTP on at 4K)
  give v1.3.118's tokens. Qwen3.8 was compared with the expert pool pinned (`STRATA_GLM_POOL_GB`): another engine
  build can leave one expert slot more or less in VRAM, which by itself changes greedy tokens on a card with a RAM
  tier.

## v1.3.118 - 2026-11-07

- **GLM-5.3-Flash on DGX Spark: speculative decode on one GPU, the same tokens** (`STRATA_GLM_WINDOW`). Until now GLM
  could only speculate across two GPUs; on one, drafts just added work (a Spark: 18.9 tok/s without, 16.6 with the NextN
  drafter). Now the model's own NextN block drafts up to 3 tokens and one batched verify window checks them: the dense
  weights are read once for all columns, the KDA (recurrent) layers' state is kept after the first column and the
  accepted columns are replayed after a rejection. DGX Spark (GB10), 4K prompt, median of 3 x 512 tokens:
  - our GLM-5.3-Flash S1 (94 GB): 21.4 -> **24.3 tok/s (+14%)**, drafts accepted 74%;
  - unsloth UD-IQ2_XXS: 19.8 -> **22.9 tok/s (+16%)**.
  The window gives the same tokens as one token at a time (512 greedy tokens identical on the Spark for both files;
  in-run checks: 499 columns re-run on the token path, logits and the KDA / DSA state 0 differences; 125 rewinds, 0
  differences). On by default where the GPU shares the host's memory (CUDA, integrated: DGX Spark); off on a discrete
  card, where the CPU lane computes the RAM tier's experts and a window is break-even (RTX 5090: 69.0 vs 69.6 tok/s at
  2 drafts, 65.7 at 3). `STRATA_GLM_WINDOW=1` / `0` force it. Note: loading the NextN block takes VRAM from the expert
  pool, which by itself can change greedy output a little on a card with a RAM tier (not on a Spark).
- **GLM-5.3-Flash reads its Q8_0 dense weights as Q8_0** (`STRATA_GLM_DENSE_Q8`, on): the attention k_b / v_b, the
  indexer's, the KDA f / g / beta projections and the hc weights were widened to BF16 at load; now the GGUF's own blocks
  are read (llama.cpp's product). 0.37 GB less read a token, KL 0.2708 -> 0.2667 (S1, 10 sequences); DGX Spark
  19.2 -> 19.7 tok/s (UD-IQ2_XXS), 20.9 -> 21.4 (S1); RTX 5090 63.9 -> 66.1. `0`: the BF16 copy as before.
- Greedy verify windows copy the column's logits to the row the end-of-turn guard reads (it read a stale row).
- `STRATA_GLM_KDA_SPLIT=1` (off): the KDA recurrence in 4 column blocks a head - the same tokens, no gain measured.

## v1.3.117 - 2026-11-07

- **DeepSeek V4.1 gets its own thinking levels**: `reasoning_effort` `low` / `medium` / `high` / `max` (the template's
  effort 50 / 62 / 75 / 100) or an integer 1-100 now reach V4.1 1017's template as it expects; it used to get
  DeepSeek V4's levels. Other models map an integer to the nearest level. The server's shared defaults take the same
  values.
- **Qwen3.8's PLE table in Q5_1 or BF16** besides IQ4_NL and Q8_0 (`STRATA_Q4X_PLE_GGUF=<file>`). 16G64G (M16b) on an
  RTX 5090, KL against NVFP4: IQ4_NL 0.0826 -> Q5_1 0.0732 (Q8_0 0.0714), same speed; the table is read row by row,
  so the bigger type costs disk (+9.6 GB), not VRAM.
- **Verify windows, same tokens, less waiting**: with temperature > 0 a window's columns are drawn in one launch and
  one wait instead of a copy, a launch and a wait each (`STRATA_GLM_SAMPLE_COLS=0`: the old way); DeepSeek V4.1's
  window reads its Engram rows in the background as tokens do (the window's Engram wait 0.39 -> 0 ms,
  `STRATA_DS41_ENGRAM_WINDOW_ASYNC=0`: the old way); the DSpark drafter's expert uploads start before the window's
  wait instead of after it.
- GLM-5.3-Flash's Q4_K token embedding is gathered on the GPU (bit-identical to ggml) instead of dequantized on the
  host and copied.
- **A note when a prompt goes token by token**: the batched prompt path can decline a prompt (too short, or a part
  lent to the vision encoder); the engine now says so and why. For the slow long prompts through the server on
  DGX Spark.
- `STRATA_DETERMINISTIC=1` (a test mode, slower): the choices that follow wall-clock timing are switched off,
  including the pack's `expert_usage.txt`, so two runs of one request give the same tokens - for checking that a
  change does not change them.
- Opt-in, off by default: `STRATA_DS4_NGRAM_LONG=1` (n-gram drafts from the longest earlier match; Qwen3.8 copy task
  +8%, DeepSeek V4 0731 -12%) and a reworked `STRATA_DS4_SPEC_COST=1` (same speed as off on V4 0731).
- `./abyss.sh --gpu-tune` also reports the Q4_K / Q5_K / Q8_0 dense GEMV split (1 / 2 / 4 warps a row); nothing is
  written from it yet.

## v1.3.116 - 2026-11-06

- **A reasoning block closed by the thinking budget no longer spills into the answer (#36, thanks @hondaru)**: the
  budget (and the loop guard) used to force a bare `</think>` in the middle of a thought, and Qwen3.8 now and then went
  on thinking in English as the visible answer. It now closes with one short sentence first - the thinking stops here
  and the answer follows in the question's language, named when the question's script shows it (Japanese, Chinese,
  Korean) - then `</think>`. Qwen3.8 12G32G L448, RTX 5090, temperature 1.0, three of the reported Japanese prompts with
  `max_tokens` 1000 (the budget closed 70 of 75 replies): English as the answer 2 of 75 -> 0 of 75; the
  3-ministries prompt no longer runs into the cap (8 -> 0 of 25). The sentence shows in `reasoning_content`.
  `STRATA_GLM_THINK_BRIDGE=<text>` sets another sentence, `0` the bare `</think>` as before.

## v1.3.115 - 2026-11-06

- **unsloth's other UD 3.1 sets of Qwen3.8-Flash-Next run** (beta): UD-IQ4_XS, UD-Q3_K_XL, UD-IQ3_XXS, UD-Q2_K_XL,
  UD-Q2_0 and UD-IQ1_M. Their Q5_0 down experts (UD-IQ4_XS: 33 layers, UD-Q3_K_XL: 20) now go through the prompt's
  MMQ instead of one expert at a time in FP16 (UD-IQ4_XS 8K prompt 1,365-1,564 -> 1,769-1,776 tok/s). RTX 5090:
  KL 0.077 (UD-IQ4_XS) to 0.190 (UD-IQ1_M), each within 0.001 of llama.cpp on the same file; 117-152 tok/s with the
  MTP drafter at 32 GB; `./abyss.sh --check --quick` 7 / 7 for all six ([docs/qwen38](docs/qwen38/README.md)).
  `tools/gguf_remote_types.py` reads a set's expert types from its headers without a download, and
  `tools/test_unsloth_types.py` checks the engine takes every type and pairing of these sets.
- **An integrated GPU (DGX Spark) no longer starts on memory another process holds**: when other processes hold more
  than 16 GB of the shared memory and the experts no longer fit because of it, the engine stops at load and names the
  biggest ones. A second engine left running had made one start on 20 GB of free memory and the machine ran out of
  memory. `STRATA_GLM_UMA_SHARE=1` starts anyway, `STRATA_GLM_UMA_OTHERS_GB` sets the limit; a model bigger than the
  machine with nothing else running is not stopped.
- **Saved conversations are written to disk in the background**: the next prompt no longer waits for the file. On a
  machine with little free RAM for the page cache the 0.25 GB write took 5-50 s before the prompt started.
- `./abyss.sh --gpu-tune` finds an engine built by hand in another folder (the configs' `exe`, e.g. `build-cuda`)
  instead of stopping with "not built here yet" (issue #37).
- **Docs caught up** (English and Japanese): DGX Spark results for GLM-5.3-Flash (19-20 tok/s), DeepSeek V4 0731
  (21.0, 22.8 with DSpark) and V4.1 against llama.cpp on the same box, and the open issue (GLM's long prompts through
  the server ~120-130 tok/s on the Spark only); our Qwen3.8 16G64G / 12G32G builds; the NVIDIA results that several
  pages still called "not built / not measured" (Qwen3.8, Ornith, DSpark); V4.1's one-command setup, images and
  DSpark in the MANUAL; the Qwen3.8 two-GPU split result (#35); `/health`'s 503 while loading.

## v1.3.114 - 2026-11-06

- **unsloth's UD 3.1 UD-IQ2_XS of Qwen3.8-Flash-Next (75.8 GB) runs** (beta). One of its layers has Q5_0 down
  experts, which the GPU expert kernels did not take; the engine stopped at load (`unresolved weights: blk.4.expert
  types 22/22/6`). The Q5_0 dot sums the 5-bit values as signed integers, as ggml-cpu does (the layer's error matches
  its neighbours'). RTX 5090, 4K prompt, median of 3 x 512 tokens: 134.2 tok/s with the MTP drafter at 32 GB, 71.7 at
  16 GB; KL 0.1359 / same top token 86.4% (llama.cpp on the same file 0.1356 / 86.6%); `./abyss.sh --check --quick`
  7 / 7. Make the pack with `tools/iq_pack.py --fast` ([docs/qwen38](docs/qwen38/README.md)).
- `./abyss.sh --check`: a short right answer to the Japanese arithmetic task passes (it failed as "not Japanese
  enough" with 9 kana on a DGX Spark, issue #37); an answer without the number or in English still fails.

## v1.3.113 - 2026-11-05

- **NVIDIA DGX Spark (GB10, ARM, unified memory) support** (beta,
  [docs/DGX-SPARK.md](docs/DGX-SPARK.md)). The engine builds for aarch64 + sm_121 (CMake picks the ARM CPU's own
  ggml-cpu flags from `/proc/cpuinfo`; no x86 kernels are built - packs with Q2_0 experts need AVX and say so). Measured
  on a rented DGX Spark with Qwen3.8 16G64G: **47.05 tok/s with the MTP drafter** (CLI, 4K, median of 3 x 512 tokens;
  31.2 without), 43-54 tok/s through the server, 8K prompts ~1,040 tok/s, 24K-262K prompts with the drafter to the
  end. DeepSeek V4.1 / V4 and GLM-5.3-Flash are not tested on a real Spark yet.
  - **Programmatic dependent launch** on an integrated GPU: the decode's next kernel starts while the one before
    finishes. GB10, Qwen3.8 16G64G: 43.4 -> 47.05 tok/s with the drafter (with the graphs), 25.0 -> 31.2 without; the
    same tokens. Off on discrete cards (RTX 5090: no change). `STRATA_GLM_PDL`, `STRATA_GLM_PDL_PREFETCH`.
  - **One memory budget** for the expert pool and the pinned RAM tier on an integrated GPU (`MemAvailable`, page cache
    counted as free, less `STRATA_GLM_UMA_HEADROOM_GB`, 8 GB on CUDA); `STRATA_GLM_UMA=0` keeps the discrete-card
    rules. RAM-tier experts are copied into the pool's slots as on a PC (32 GB budget: 45.8 tok/s, read in place
    34.9 - `STRATA_GLM_UMA_DIRECT=1` for tests), and the CPU lane is off there.
  - **Prompts under a VRAM cap on unified memory**: with `STRATA_GLM_VRAM_GB` the prompt path counted the page cache
    and other processes as used, saw no pool and read the prompt token by token (GB10 at 16 GB: 3.8K tokens 129 s ->
    6.5 s).
  - **The tools on ARM**: `./abyss.sh --check`, `--bench` and `--gpu-tune` take nvidia-smi's `[N/A]` memory (GB10:
    the system memory, shared), ARM's `/proc/cpuinfo` (no `flags`; the CPU named from lscpu) and say "memory
    unified" instead of a PCIe link; CUDA 12.9+ is asked for sm_121.
  - `STRATA_GLM_DISK_BUFFERED=1` (a test): the SSD's expert reads through the page cache instead of O_DIRECT.
  - CI: a new `arm-build` workflow on GitHub's ARM runners - the Python tests (with GB10-like nvidia-smi and ARM
    cpuinfo inputs), the CPU-only build and its tests, and the CUDA engine compiled for sm_121a.
- RTX 5090: the same tokens and speed as v1.3.112 (PDL and the unified-memory paths are off on a discrete card).

## v1.3.112 - 2026-11-05

- **Qwen3.8-Flash-Next answers faster with the MTP drafter**, the same tokens. The speculative verify window now
  replays each delta-net layer from a CUDA graph, waits for the CPU lane and fetches the columns' experts in one
  launch instead of one per column (launches a token 453 -> 254), and swaps experts into VRAM also while the CPU lane
  takes part of the RAM-tier experts (with a quicker swap rule when the VRAM pool holds half of the experts or more).
  RTX 5090 (PCIe Gen4 x4) + 94 GB RAM, 4K prompt, median of 3 x 512 tokens, v1.3.111 -> v1.3.112 the same day:
  - Swift 1.5 IQ2_XS, 32 GB: 158.6 -> **180.6 tok/s (+14%)** with the drafter (VRAM hits 92.6 -> 95.6%), 115.7 -> 118.6
    without; 16 GB: 79.3 -> 80.7.
  - Qwen3.8 16G64G: 32 GB 129.3 -> 131.3 (+1.5%; 16K / 32K prompts: +1.6% / +2.8%); 16 GB 89.2 -> **96.4 (+8%)**.
    Without the drafter: unchanged (90.0 / 70.7 -> 71.4).
  - Switches: `STRATA_GLM_GRAPH`, `STRATA_GLM_GRAPH_MOE`, `STRATA_GLM_CPU_WAIT_COLS`, `STRATA_GLM_FETCH_COLS`,
    `STRATA_GLM_BG_SPLIT` ([MANUAL](docs/MANUAL.md)). The graphs are CUDA only.
- **unsloth's UD 3.1 UD-IQ1_S of Qwen3.8-Flash-Next (66.4 GB) loads** (beta). One layer stores its gate and up experts
  in different types (IQ2_S / IQ2_XXS), which the pack and the kernels did not take. `tools/iq_pack.py` now writes such
  a layer's types as `G,U` (other packs stay byte for byte the same), and the decode, verify window, prompt path, CPU
  lane and streamed fetches compute each role in its own type; the iGPU lane and a helper GPU leave that layer to the
  CPU. The GGUF's bytes are read as they are: KL to the NVFP4 teacher 0.2007 (llama.cpp on the same file 0.2012),
  same top token 83.2% ([docs/qwen38](docs/qwen38/README.md#unsloth-ud-iq1_s-ud-31-v13112-beta)). Re-make its pack
  after updating.
- **`./abyss.sh --check`: a stability check of the installed model** (15-30 minutes; `--quick` ~5, `--long` adds the
  biggest prompts). Through the model's own server: drafter on and off, Japanese, code run against tests, a tool call,
  8K-64K prompts with a needle, an 11-turn conversation. `abyss-check.txt` (no host, user or folder names; nothing
  sent) ends with "What limits this PC", an estimate of which part holds a token back - also in `--bench`.
  Attach it to the new [bug report](https://github.com/zephel01/Abyss/issues/new?template=bug.md) form. Before the
  setup, `--check` still checks the PC only ([BENCH](docs/BENCH.md#the-stability-check---check)).

## v1.3.111 - 2026-11-05

- **Qwen3.8 with the MTP drafter no longer breaks on long prompts.** From about 20K tokens in one prompt chunk, the
  prompt path's drafter step wrote past its scratch region into the experts kept in VRAM for decoding, and the first
  speculative window failed with `glm router: ... invalid expert ID or non-finite score` (seen with Swift 1.5 IQ2_XS
  from 24K; other Qwen3.8 builds were exposed the same way). The step now works on groups of rows that fit; the output
  is bit-identical and no extra VRAM is lent.

## v1.3.110 - 2026-11-05

- **Two fixes for the Qwen3.8 two-GPU split (BETA), reported in #35** (thanks to tjm8874 for the real two-card runs):
  - A hand split is now checked against the model's own layer count. Before, it was checked against the default
    45-layer shape before the GGUF was read, so `--layer-split 46` on the 48-layer Qwen3.8 was refused.
  - When `--layer-split auto` decides that one card alone is faster, that card keeps its own entry of
    `STRATA_GLM_VRAM_GB_PARTS`, the budget its prediction used. Before, it took the first entry (with `16,32` it
    predicted with 32 GB and then loaded with 16).
- Known: on a 16 GB + 16 GB GeForce pair, `auto` still picks the first card alone, while a split at layer 32 measured
  22% faster on long requests (#35). The split's cost model will be refitted with those numbers; until then, set the
  split by hand (`--layer-split 32`).

## v1.3.109 - 2026-11-05

- **Qwen3.8-Flash-Next on two GPUs (BETA)**: `--gpus 0,1` now splits Qwen3.8 (qwen4exp) builds across the cards' layers -
  the token path, prompts (serial and pipelined), the speculative verify window and the MTP drafter, which sits on the
  last card (it reads the final residual and the head; that card also keeps a pinned copy of token_embd, +497 MB of
  RAM). `--layer-split auto` counts Qwen3.8's delta-net states, K / V caches and QSA keys at the context and the drafter
  on the last card; the first boundary is layer 2 or later. `STRATA_GLM_VRAM_GB_PARTS=a,b` caps each card's VRAM.
  Checked on one RTX 5090 seen as two devices (16G64G build, split at layer 24): KL to the NVFP4 teacher the same as
  unsplit (0.170 / 0.176), real-use tasks the same (Japanese 5/5, code 4/5, 8K needle), drafts accepted 84.7% (unsplit
  83.2-86.1%); `auto` with two 15 GB caps picks layer 26. **A real two-card speed has not been measured** - reports
  are welcome through the bench issue form. Conversation reuse (snapshots) is off under a split.
- **The single-GPU Qwen3.8 drafter now sees its prompt rows' next tokens**: prompt chunks passed no next-token ids to the
  MTP drafter, so its draft cache after a prompt was built from the wrong rows. Accepted drafts are within run-to-run
  noise on the checked prompts (83.2% vs 86.1%).
- **One swappable hand-over between split parts** (`STRATA_GLM_HOP=host|peer|sock`, default host as before), and the
  Qwen3.8 split drives the next card through one interface (the groundwork for a part in another process or on another
  PC). Other models' splits keep their paths; DeepSeek V4 0731 split vs unsplit on one card: same top-1 (89.18%), nll
  0.5435 / 0.5398.

## v1.3.108 - 2026-11-04

- **Thinking no longer collapses to one token when a request's `max_tokens` is 1024 or less (#33, thanks @hondaru)**:
  the server leaves the answer at least 1024 tokens of the output (`STRATA_GLM_THINK_RESERVE`), and with
  `max_tokens` <= 1024 that left the reasoning a single token - the model answered without thinking. On the 1-bit
  Qwen3.8 12G32G L448 build that made it echo the prompt back now and then (the reported dose-calculation prompt at
  temperature 0.3, `max_tokens` 800: 1 of 8 answers, 6 of 8 with the drafters off). The reserve is now capped at half
  of `max_tokens` (800 -> 400 tokens of thinking, 400 for the answer; 2048 and above unchanged). Same prompt after the
  fix: 0 of 8 echoes, 8 of 8 with the right dose, on both L448 and a 2-bit build.
- **Qwen3.8 GGUFs with Q8_0 down experts work on the prompt path again (unsloth UD-IQ4_XS)**: the prompt's expert
  matrices are read in 512-value tiles, which run past a 640-wide down row; the expert slots had no room for that
  over-read when the down matrix is Q8_0, so the tail was read from the next slot and layer 2 came out NaN (any prompt
  of ~100 tokens and more: "invalid expert ID or non-finite score" and a 400). The slots now reserve the over-read
  (only where the rounding left too little: other layouts keep their sizes). Checked on an RTX 5090: UD-IQ4_XS with
  100- and 8,000-token prompts (8K at ~1,600 tok/s), the prompt path's first 26 tokens equal to the token path's on a
  2,000-token prompt; UD-IQ3_XXS and the NVFP4 pack give the same tokens as v1.3.107.

## v1.3.107 - 2026-11-04

- **Long prompts no longer fail or crash when the CPU share's pinned buffers do not allocate (#31, thanks @LarsDrakblod)**:
  on a Windows host whose RAM tier pins most of its RAM (RTX 4090, 96 GB, `STRATA_GLM_RAM_GB=72`), the prompt's CPU
  share asked for more pinned memory as prompts grew; from ~20K tokens the allocation failed and the request came back
  as a 400, and from ~40K the engine died (0xC0000005). Two faults: a failed allocation kept the new buffer size with a
  freed pointer, which the next layer wrote through (the crash); and a failure ended the prompt instead of moving the
  CPU's experts to PCIe. Now a failed allocation leaves every buffer null, the buffers are taken before the CPU's
  share is planned, and when they do not allocate - or would leave under 2 GB of RAM free - the process's prompts
  stage every expert over PCIe from then on, with one log line (as `STRATA_GLM_PREFILL_CPU=0`, which the report found
  avoided both). `STRATA_GLM_PREFILL_PIN_MB=<n>` caps a buffer's size. Checked on an RTX 5090 with DeepSeek V4 0731
  UD-Q2_K_XL, the failure forced with `STRATA_GLM_PREFILL_PIN_MB=64`: 15K / 30K / 44K-token prompts all answered
  (1,538 / 2,675 / 2,142 tok/s), and the default setup unchanged (1,284-2,731 / 3,120 / 2,999). The Windows limit itself
  is not reproduced here.

## v1.3.106 - 2026-11-04

- **End-of-turn guard against empty answers (Qwen3.8 and other thinking models on the GLM path)**: a heavily quantized
  model can close its reasoning and end the turn at once, or end it inside the reasoning - the client gets an empty
  answer. The server now asks the engine to keep it going once each: a turn ended inside the reasoning closes the
  reasoning instead, and a turn ended right after it with nothing but whitespace takes the position's most likely
  other token. Qwen3.8-Flash-Next 12G32G (448 experts, first LDLQ build), 2 x 11 real-use prompts: the guard fired
  once, on a summary that had come back empty, and the answer was a right summary. `STRATA_GLM_END_GUARD=0` turns it
  off. The engine watches the end tokens
  the server stops on (Qwen3.8's `<|im_end|>` on this path; its own list is GLM's).
- **Tool calls whose opening tag slipped are calls (Qwen style)**: `<=NAME>`, `<function:NAME>`, `<function NAME>`, and
  `<NAME>` when NAME is a declared tool, are read as `<function=NAME>`; they came back as text before. An undeclared
  `<NAME>` stays text.
- **IQ1_S experts on the CPU lane ~18% faster for one token**: the grid indices and the delta sums of a block are built
  in vectors instead of 32 scalar steps; bit for bit ggml-cpu's dot (checked at start, and in `native_rows_exact_test`).
  Qwen3.8 12G32G, a gate/up expert from DRAM on a Ryzen AI Max+ 395: 1 thread 1142 -> 944 us, 8 threads 174 -> 142.
- **Qwen3.8's MTP drafter with a pruned main model**: the drafter's files hold 512 experts and were refused for a model
  that keeps fewer (448 / 384); they load now (the main model checks what the drafter proposes). A drafter that does
  not fit at all is now skipped with a log line instead of stopping the engine.
- **`reasoning_effort` of false / 0 / "" in a request is kept**: the dashboard's shared thinking level used to
  overwrite it, so a request could not turn thinking off.

## v1.3.105 - 2026-11-04

- **DeepSeek V4.1 image questions with ISTA-DASLab's new-format GGUF (PR #29, thanks @rS-alonewolf)**: its chat template
  writes `<｜deepseek_image｜>` for an image block itself, which the server took as "the template handles images" and
  so sent no image rows - a question with an image came back as a 400. The server now places the image rows itself for
  every V4.1 template; templates that do not write the marker behave as before. A test with one and two images on the
  new-format template is added (it fails on v1.3.104); checked on an RTX 5090 with both templates.

## v1.3.104 - 2026-11-03

- **Prompts no longer lose their CPU share to a timing cliff (#24)**: the prompt path gave the CPU none of its staged
  experts whenever one copy over PCIe timed under two CPU experts at start-up. On a PCIe Gen5 x8 host that line sat
  inside the CPU timing's own jitter: processes that read 0.155 ms an expert ran 16K prompts at ~2,700 tok/s, those
  that read 0.159-0.169 ms at ~2,050 for their whole life (the same as `STRATA_GLM_PREFILL_CPU=0`). The gate is gone;
  the plan already takes an expert only while the host's time stays under the copy stream's, and corrects itself as
  it goes. Checked on one RTX 5090 with GLM-5.3-Flash UD-IQ2_XXS, iGPU lane off: with 1 CPU thread (the old gate off)
  10K-token prompts went 321-346 -> 375-388 tok/s; with 16 threads (gate on) 516-524 -> 519-524, unchanged.
- **A card that holds the whole model keeps it through a prompt**: the prompt path borrows a tail of the expert pool
  (~3.5-4 GB) while it runs, and that tail used to be cut from the experts' own slots - so even a 72 GB card that holds
  every expert dropped some to RAM for each prompt and ran them on the CPU lane after it. When VRAM holds every expert,
  the spares and the tail, the tail now goes on top, and a lent slot's expert moves to a free slot instead of evicting
  a colder one. Qwen3.8-35B-A3B on an RTX 5090: VRAM hits 99.1-99.4% -> 100%, CPU lane 2-3 experts a token -> 0,
  10K-token prompts 5,148-5,675 -> 5,921-6,414 tok/s, decode after them +2-6%. `STRATA_GLM_TAIL_ON_TOP=0` keeps the
  old layout.

## v1.3.103 - 2026-11-03

- **CPU lane calibration taken again when the CPU is busy (#24)**: the start-up timing of an expert on the CPU decides
  the decode's CPU / PCIe split for the whole process. With a build running beside the engine it read 0.923 ms where
  the quiet box takes 0.155, and the split gave the CPU 2 of 8 RAM-tier experts instead of 6 (a first decode at 54
  tok/s instead of 83). When the timing looks disturbed - other tasks running on a quarter of the CPUs (the engine's
  own threads not counted), or the slowest of its 48 runs over 3x the median - it is now taken again up to 3 times,
  1 s apart, and the lowest kept; a CPU still busy after that is named in the log. `STRATA_GLM_CPU_CAL_RETRY=<n>`
  (default 3, `0` = one timing). The prompt path's own split recovers by itself once the CPU is free (checked on one
  RTX 5090: 32 busy processes through start-up and the first prompt, 449 tok/s, then 520-526 as on a quiet start).
- Docs: DeepSeek V4.1 Compact JA256 (207 GB, 256 of 384 experts) in README, MODELS, MANUAL and docs/deepseek41.

## v1.3.102 - 2026-11-03

- **Expert helper resync: the next request waits about two copies (#24)**: v1.3.101's rounds of 16 still kept a next
  request 6-8 ms (Gen5 x8), because a round's copies - and the demotions the next boundary queued for the spares it
  used - were queued together and waited for at the end. The resync now lands each copy before it checks again, runs
  rounds of 2, and waits for the decode's last copies while checking; a stopped resync leaves the spare refill to the
  next request's own boundary. On one RTX 5090 (Gen4 x4) the longest gap between checks went from 18-28 ms to
  3.5-5 ms, with the same promotions per second. The log line adds `longest gap between next-request checks`.

## v1.3.101 - 2026-11-03

NVFP4 experts, and a shorter wait for the next request after a helper resync.
- **NVFP4 (NVIDIA's FP4 checkpoints)**: expert rows in NVFP4 with their per-expert second scale (`ffn_*_exps.scale`)
  in the decode kernels, the CPU lane and the prompt path (Blackwell's FP4 tensor cores), and a Q8_0 PLE table.
  Checked with [magnitudedev's Qwen3.8-Flash-Next NVFP4](https://huggingface.co/magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF)
  on an RTX 5090 + 96 GB: KL 0.015 / 0.018 (token / prompt path) against llama.cpp, prompts 2,515 tok/s at 32K,
  swe-bench 3 tasks 105.3 tok/s with the MTP drafter, Japanese and a 7-turn tool session fine (beta,
  [docs/qwen38](docs/qwen38/README.md#nvfp4-nvidias-fp4-checkpoints-v13101-beta)).
- **Expert helper resync stops within a few ms (#24)**: a resync round promoted ~120 experts (~42 ms over PCIe) before
  it looked for the next request; rounds are now at most 16 promotions and resume where the last one stopped. The
  budget (`STRATA_GLM_EXPERT_HELPER_RESYNC`) and the default (off) are unchanged.

## v1.3.100 - 2026-11-02

Our own GGUFs in the one-command setup, and the setup's check of a metadata-only first shard.
- **`./setup-deepseek.sh --model v41`**: DeepSeek V4.1 Flash 1017 with zephel01's LDLQ GGUF, made for a 32 GB GPU +
  96 GB RAM ([the model card](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-LDLQ-32G96G-GGUF), BETA):
  132G by default (speed: the types of unsloth's UD-IQ1_M with LDLQ rounding, KL 0.194, coding tasks 67.5 tok/s on an
  RTX 5090 + 8060S lane), `--quant 139G` for quality (layers 35-39 at IQ3_XXS, KL 0.182, 57.9 tok/s). The download is
  275.8 / 283.0 GB (the experts 132.2 / 139.4 GB, the Engram table 135.2 GB, which is read from the SSD a few rows at a
  time). The config gets the card's everyday settings (`STRATA_GLM_TAIL_SKIP=4`, `_DISK=1`,
  `STRATA_GLM_ROUTE_AFFINITY_RAM=0.08`, 8 disk readers) and, below 160 GB of RAM, a RAM tier of the RAM less 34 GB
  (`STRATA_GLM_RAM_GB`; 60 on a 94 GB PC, as measured): without it the page cache fills and systemd-oomd stops the
  server. The DSpark drafter is not set up for V4.1 (Lucebox's V4.1 drafter repository is no longer on Hugging Face).
  The files also run in upstream llama.cpp (`deepseek41`, checked on ac8e1694d).
- **`./setup-deepseek.sh --model ornith` / `--model qwen35`**: zephel01's 16 GB GPU + 64 GB RAM quants of
  Ornith-1.5-122B (IQ3_XXS-B, 51.7 GB; its NextN drafter is read from the second shard) and Qwen3.5-122B-A10B
  (IQ3_XXS-B2, 51 GB), with the model cards' sampling (temperature 0.6, top_p 0.95, top_k 20). Checked on the RTX 5090
  from the written config: Ornith decode 116 tok/s, 75% of the NextN drafts accepted.
- **Setup: a first shard of metadata alone passes the size check.** llama-gguf-split's first shard holds no tensors
  and ends before the aligned data start (unsloth's and zephel01's V4.1 shard 1 of 6: 5,780,868 bytes, the data would
  start at 5,780,896); `setup.py`'s check took it for 28 bytes short and stopped. It is now whole at its header's end,
  as `abyss.py` already counted it (#83).
- **Docs**: MANUAL (EN + JA) one-command setup, docs/deepseek41 and docs/ornith; README introduces the own GGUFs and
  asks for per-PC reports (`./abyss.sh --bench --deep`, the bench issue form).

## v1.3.99 - 2026-11-02

- **Expert helper RESYNC stops when the next request arrives** (#24). In #24's two-card run, RESYNC=96/192 raised the
  primary's VRAM hit (67.7 -> 76-80%) but barely moved the next Prefill (-0.4 to +1.4%) and its 33-106 ms wait made
  back-to-back requests 0.7-2.1% slower. The resync now checks for a waiting request between rounds and stops there,
  so it only uses idle time between requests and a large budget (e.g. 8192) costs back-to-back requests nothing. One
  RTX 5090: with an 8 s pause it ran to the end (210 promotions, 0.7 s); requests sent right after DONE stopped it
  (110 / 59 promotions); answers normal. Default still 0 (off).

## v1.3.98 - 2026-11-02

- **`./abyss.sh --connect claude|opencode`**: the settings that point Claude Code or OpenCode at the server that runs
  now, read from the server (model name, context) and its config (address, key). `--write` writes a Claude Code
  launcher (`run-claude-<config>.sh`, settings for that run only) or merges provider `abyss` into OpenCode's
  `opencode.json` (asked first, backup kept); `--check` sends one prompt and one tool call the way the tool does;
  `--url` / `--key-env` for a server on another PC. Claude Code gets its context (`CLAUDE_CODE_MAX_CONTEXT_TOKENS`)
  so it compacts in time. The API key is never written into a file. docs/MANUAL.md "Coding tools".

## v1.3.97 - 2026-11-02

- **Expert helper: catch the VRAM tier up between requests** (#24). v1.3.96's `PROMOTE` brought the Prefill back
  (+7%) but cost Decode (-6 to -10%) on two RTX PRO cards, because the promotions ran during Decode. New opt-in
  `STRATA_GLM_EXPERT_HELPER_RESYNC=N` (default 0): after a helper decode, once `DONE` is out, up to N of the hottest
  RAM-held experts are promoted into VRAM (spares refilled from the coldest residents). Decode is untouched; the next
  prompt starts with a VRAM tier that followed the conversation. One RTX 5090 (same-GPU helper): answers normal,
  helper vs exact reference RMS 0.022 (CPU lane 0.023), 50-192 promotions in 0.2-0.5 s; the two-GPU effect is to be
  measured in #24.

## v1.3.96 - 2026-11-02

- **Expert helper: optional promotion of hot helper experts** (#24). With the helper on, its experts were never
  promoted into primary VRAM, so Decode promotions fell from 2.2-2.5 to 0.3-0.5 per token and the following prompt
  ran about 10% slower (16K prompt: ~2,950-3,030 tok/s off vs ~2,700-2,720 on, after a Decode). New opt-in
  `STRATA_GLM_EXPERT_HELPER_PROMOTE=N` (default 0, unchanged behaviour): a helper-owned expert whose route count
  reaches N is fetched into a free VRAM spare instead, at most one per route. Checked on one RTX 5090 (same-GPU
  helper): answers normal, helper vs exact reference RMS 0.018 (CPU lane 0.020), no Prefill slowdown. The effect on
  two GPUs is still to be measured in #24.

## v1.3.95 - 2026-11-02

- **Expert helper check: RMS and an exact reference** (#24). `STRATA_GLM_EXPERT_HELPER_CHECK=1` printed only the
  worst column's largest difference, which grows with the number of columns checked (0.034 at 2K columns, 0.056 at
  20K in the #24 report). It now also prints the RMS difference against the CPU lane, and on every 256th column an
  exact float reference (dequantized weights, float activations) with the helper's and the CPU lane's distance to it.
  On one RTX 5090: helper vs CPU RMS 1e-4, and both 0.022 RMS from the exact result (worst column 0.026 both) - the
  helper's difference is ordinary quantized arithmetic. Debug output only; no change to decoding.

## v1.3.94 - 2026-11-02

The second-GPU expert helper also works with the DSpark drafter, and the V4.1 guide names a pick for 32 GB + 96 GB.

- **Expert helper with DSpark** (#24, item 1). `STRATA_GLM_EXPERT_HELPER` was turned off under speculative decoding,
  so a second GPU added nothing to DSpark decode. Now a verify window's helper-owned experts go to the helper GPU in one
  call for all the window's columns while the CPU lane computes the rest at the same time; a helper error recomputes
  the whole window on the CPU. Layer graphs still turn it off. Debug: `STRATA_GLM_EXPERT_HELPER_SAME_GPU=1` puts the
  helper on the primary GPU (a correctness check on one GPU), `STRATA_GLM_EXPERT_HELPER_CHECK=1` compares every
  helper-served window with the CPU's answer. [docs/EXPERT_HELPER.md](docs/EXPERT_HELPER.md)
- **docs/deepseek41: 32 GB GPU + 96 GB RAM -> UD-IQ1_M.** Real-use numbers against UD-IQ2_XXS (harder graded tasks
  10/10 both, decode 52.5 vs 39.2 tok/s, 32K prompt 1,020 vs 833 tok/s) and `--uncensor` on UD-IQ1_M (refusals
  92% -> 0%, MMLU 90% -> 90%, KL 0.018).

Checked: CUDA build; unsloth V4.1 UD-IQ2_XXS + DSpark on one RTX 5090 with the helper on the same GPU and the check on:
7,336 window columns through the helper, no fallback, largest difference from the CPU 3.8% of the largest value (the
GPU vs CPU expert arithmetic; a mixed-up column would be ~100%), answers normal. The speed on two GPUs is for #24's
reporter to measure; with the helper unset nothing changes.

## v1.3.93 - 2026-11-02

DeepSeek V4.1 reads images (a contributed change).

- **DeepSeek V4.1 image input** (PR #28, thanks @rS-alonewolf). With ISTA-DASLab's V4.1 GSQ-RCO mmproj, image positions
  pick their experts with the image router bias (`exp_probs_b_vl`) and add no Engram, as llama.cpp does; the server
  expands V4.1's image placeholder. The `vision` entry is written by hand (setup does not install it for V4.1), and
  `STRATA_GLM_VL_BIAS` names the bias file (or `exp_probs_b_vl.f32` beside the pack, which `tools/iq_pack.py` now
  writes). Without the bias the model starts and image requests are refused. Text-only requests are unchanged.
  How to set it up: [docs/SETTINGS.md](docs/SETTINGS.md#images-on-deepseek-v41).

Checked: CUDA build; unsloth V4.1 UD-IQ2_XXS without images on an RTX 5090 (TAIL_SKIP, affinity, DSpark), alternated
with v1.3.92 twice each: swe-bench 3 tasks 50.1 / 50.0 tok/s (v1.3.92) vs 52.2 / 51.0, answers normal; the serve and
tools tests.

## v1.3.92 - 2026-11-02

Qwen3.5-MoE models with a NextN drafter (Ornith-1.5-122B, Qwen3.5-122B, Qwen3.8-35B-A3B) no longer crash at the
first token.

- **qwen35moe NextN drafter: a crash at the first draft** (since v1.3.87). The verify-window row prefetch added for
  Qwen3.8-Flash-Next (qwen4exp) read its token history on every qwen35 model, but the history is kept only on
  qwen4exp; on the others the empty list was read and the engine stopped with a segmentation fault right after the
  first token. The history is now read only where the prefetch runs. Qwen3.8-Flash-Next is unchanged.
- **Qwen3.8-35B-A3B Distill** (empero-ai Q4_K_M, 21.7 GB) runs: RTX 5090, swe-bench 3 tasks 399 tok/s with its NextN
  drafter (85% of drafts accepted); VRAM capped at 16 GB 297 tok/s. Listed in the README's model table.

Checked: CUDA build, Qwen3.8-35B-A3B Q4_K_M on an RTX 5090 (32 GB and 16 GB VRAM, a Japanese answer, swe-bench 3
tasks), the serve and tools tests.

## v1.3.91 - 2026-11-01

Qwen3.8-Flash-Next from unsloth's UD files reads the prompt in batches again; two contributed fixes.

- **unsloth Qwen3.8 UD files: the prompt path** (issue #25, thanks @LarsDrakblod). unsloth keeps the GDN layers'
  `ssm_alpha` / `ssm_beta` rows F32 (ISTA's files have them BF16); the prompt path's weight check refused F32, so every
  prompt was read token by token. The GEMM already widens F32 rows to FP16, so the check now lets them through on
  qwen35 / qwen4exp. unsloth UD-IQ1_S, RTX 5090 + Radeon 8060S lane, the same 8K / 32K prompts:
  8K 90 -> 2,373 tok/s, 32K 90 -> 3,165 tok/s (32K: 6 min -> 10 s to the first token); the needle found in both.
- **`--uncensor` and `STRATA_GLM_ABL` both set: an error** instead of one silently replacing the other; deferral
  (`STRATA_GLM_DEFER`) stays off while an ABL2 edit is loaded (PR #26, thanks @rS-alonewolf).
- **`LOGPOS`** (opt-in, `STRATA_GLM_LOGPOS=<new file>`): the engine command writes each position's target and top-k
  log probabilities, for comparing quants and edits position by position; `STRATA_GLM_LOGPOS_PREFILL=1` scores
  through the prompt path on a single-GPU DeepSeek V4.1 (PR #27, thanks @rS-alonewolf). Off by default.

Checked: CUDA build, the serve and tools tests, the engine's CPU tests (those needing a local model pack skipped).

## v1.3.90 - 2026-11-01

A malformed tool call no longer fails the request; a Q5_K dense recipe for DeepSeek V4.1 UD-IQ2_XXS.

- **A malformed tool call comes back as text.** A call with an unclosed `<parameter=` tag, or a DeepSeek DSML block
  that does not parse, raised inside the streaming parser and ended the request with an error; small quants and the
  uncensor edit make such output more likely. Now the block is returned as ordinary text (a call already announced
  while streaming is closed as scanned). A regression test covers the tag whole and in pieces, streamed or not.
- **docs/deepseek41: UD-IQ2_XXS with Q5_K dense** (opt-in, recommended on a 32 GB card): the layers' Q8_0 / Q6_K
  projections re-quantized to Q5_K as a 4.4 GB overlay (283 tensors, 6.26 -> 4.36 GB), the 1.9 GB going to the VRAM
  tier. RTX 5090 + Radeon 8060S lane, the everyday settings + DSpark, swe-bench 3 tasks alternated 4 times:
  47.5 -> 50.5 tok/s (+6%); 8 held-out texts with exact routing: KL 0.1420 -> 0.1456. Three copy-paste commands.

Checked: the serve tests (206) and the tools tests. Engine code is unchanged from v1.3.89.

## v1.3.89 - 2026-11-01

DeepSeek V4.1 Flash 1017 with the DSpark drafter and an iGPU lane: the lane takes a verify window's columns in one
call. RTX 5090 (PCIe Gen4 x4) + Radeon 8060S lane, unsloth UD-IQ2_XXS, the everyday settings + DSpark, swe-bench 3
tasks, the v1.3.88 and v1.3.89 builds alternated 4 times each:

| | v1.3.88 | v1.3.89 |
| --- | ---: | ---: |
| swe-bench decode | 46.3-50.0 tok/s (mean 48.6) | 49.8-53.0 tok/s (mean 51.0) |
| the lane, an expert | 60-62 us | 54-57 us |
| a token, the same task | | -3-5% |

- **The iGPU lane computes a window's columns in one call**, each of its experts read once for all the columns that
  route to it (a window's columns share ~20% of their lane experts) instead of a lane call a column. The same bits as
  before (14,000 columns checked against the one-column call, 0 differ); 0 crashes in 8 runs.
  `STRATA_GLM_IGPU_COLS=0` is the old way, `ABYSS_IGPU_COLS_UNION=0` keeps one call without the shared reads.
- **Qwen3.8-Flash-Next decode**: the QSA layers' attention, token by token and in the MTP verify windows, reads its
  cells in place, a window's columns in one launch (the same bits). RTX 5090, 2,051 cells a layer: a 4-column window
  88 -> 24 us; --bench decode about +2.6% (inside the bench's drift). `STRATA_Q4X_QSA_DECODE=0` is the old way; off
  on HIP.
- docs/deepseek41: **keep the iGPU lane on** when the box has one (it is the default): with DSpark, swe-bench 3 tasks
  lane off 37.6-37.9 tok/s, on 46.6-50.5 (+25-33%). The lookahead (`STRATA_GLM_AHEAD`) re-measured on a ~6 GB/s NVMe
  is still slower (47.9-49.3 off, 39.5-46.0 on) and stays off.
- HIP: the lane library (built for the 8060S) carries the new call; the AMD engine build was not built or run on an AMD card for this release.

## v1.3.88 - 2026-11-01

Qwen3.8-Flash-Next reads long prompts much faster: RTX 5090 (PCIe Gen4 x4), ISTA-DASLab GSQ-RCO IQ3_S, the v1.3.87
and v1.3.88 builds alternated (`./abyss.sh --bench --full`, ctx 65536):

| prompt | v1.3.87 | v1.3.88 |
| --- | ---: | ---: |
| 2K tokens | 1061-1120 tok/s | 1145-1202 tok/s (+5%) |
| 8K tokens | 1477-1561 tok/s | 2240-2376 tok/s (+52%) |
| 32K tokens | 1702-1720 tok/s | 3002-3027 tok/s (+76%) |

A 45.5K-token prompt (needle test) 26-29 s -> 14-17 s, the needle found 3 of 3 on both. Decode unchanged; quality KL
0.0022 as before.

- **The prompt's sparse attention (the 12 QSA layers, rows past 2,051 cells) runs 64 rows a launch** instead of a
  gather and an attention launch a row and a layer, each row reading its own selected cells straight from the KV cache,
  one block the six heads of a KV group (one K / V read for six heads). The same bits as before (`qsa_rows_parity`:
  8K / 32K contexts, 4-2,051-cell selections). The indexer's block keys are appended a sub-batch at a time in two
  launches instead of one a token (the same bytes). The stage took 2,000 ms of a 12K-token run, now 346.
  `STRATA_Q4X_QSA_BATCH=0` is the old way; off by default on HIP (it compiles; not run on an AMD card).
- **The delta-net middle (q/k norm, gates, recurrence, output norm, q8_1) runs in one launch** on the token path and
  three in one in the verify windows, the same bits (`gdn_core_parity`). About -0.09 ms a token; the decode speed
  moved within the noise. `STRATA_Q4X_GDN_FUSE=0` is the old way; off by default on HIP.
- **`--bench`'s drafter check no longer calls a near-tie "TOO EARLY"**: when the answers with and without the drafter
  part within the first 12 tokens, the token path scores both tokens after the same start (LOGP); within 1 nat is an
  early near-tie (fine), farther apart is still TOO EARLY. On the 5090 the parts measured 0.01-0.13 nats apart, a
  wrong token 18-28 nats below.
- Not checked on an AI PRO R9700 (not in the PC).

## v1.3.87 - 2026-10-31

Qwen3.8-Flash-Next with its MTP drafter decodes about 12% faster. RTX 5090 (PCIe Gen4 x4) with ISTA-DASLab GSQ-RCO
IQ3_S, ctx 32K, the v1.3.86 and v1.3.87 builds alternated in one session (4 greedy answers of 256 tokens, twice):

| | v1.3.86 | v1.3.87 |
| --- | ---: | ---: |
| `./abyss.sh --bench` decode (3 questions, MTP drafter on) | 116.1 tok/s | 128.5-129.5 tok/s |
| 4 answers, drafter on (mean) | ~116 | ~130 |
| token by token (`GEN spec=0`), the 4 answers' mean | ~106 | ~107 (unchanged) |
| quality check (KL against bench/ref) | 0.0022 | 0.0022 |

- **The MTP verify windows' hyper-connection mixes run in two launches instead of five, with the same bits**, as the
  token path's since v1.3.86 (`hc4_fused_parity` now also checks windows of 1-8 columns byte for byte). The five
  launches read every column's normed streams (40 KB each) once a row and they did not fit in L1: a window's 97 mixes
  took 2.6 ms at 2 columns and 5.5 ms at 5; now 1.8 and 2.6. Drafter on +7.7%. `STRATA_Q4X_HC_FUSE=2` keeps the token
  path fused and the windows on five launches (for A/B); off by default on HIP, as before.
- **The MoE route kernel picks its top-k with the GPU's warp reductions** (NVIDIA, sm_80 and newer) and writes the
  request to the CPU side a thread an entry. The same experts in the same order as before (checked against the old
  selection over 64-512 experts, 1-16 picks, ties, -inf and +-0), so every model's output is unchanged; a route takes
  ~45k -> ~36.5k GPU cycles. Qwen3.8: drafter on and token by token both +2%. Every model routes through this kernel
  (GLM, DeepSeek V4 / V4.1, Ornith, Qwen3.8); HIP keeps the old selection.
- **Qwen3.8 reads a verify window's PLE table rows while the drafter runs**: each window token's 16 rows are
  prefetched into the row cache as soon as the drafter knows the token, so the window waits for its rows 0.1-0.25 ms
  instead of 0.8-1.1 ms (in the engine a 4 KB SSD read takes ~0.3 ms, not 0.1, beside the CPU lane). Drafter on +3.1%.
  `STRATA_Q4X_PLE_PREFETCH=0` turns it off. The window's read no longer starts a thread.
- Other models (the route kernel change), on the 5090 against the v1.3.86 build:
  GLM-5.3-Flash Maya-S-v2 IQ2_XXS (ctx 64K): decode 64.3 / 64.0 tok/s against 64.2, 8K prompt 1087 / 1091 against
  1087, quality KL 0.039-0.040 against 0.049 (the same "slightly off" as v1.3.86's).
  DeepSeek V4.1 UD-IQ2_XXS (ctx 64K, no DSpark): decode 22.9 / 20.7 tok/s against 21.6, 8K prompt 360 / 319 against
  356 (SSD-bound: the second run met a slow SSD spell, 2.7 GB/s), KL 0.0081 / 0.0082 against 0.0080.
  `--bench`'s drafter check sometimes parts at the first tokens on both builds (a near-tie that moves with where the
  experts sit): with expert swaps pinned, V4.1 parted at token 0 in 1 of 3 runs on this build and 1 of 3 on v1.3.86's,
  GLM at token 7 in 2 of 3 here and 1 of 4 on v1.3.86's. The route kernel's whole output (request block, window column
  blocks, canaries) is byte-identical to v1.3.86's in a parity test at k = 6 / 8 / 10 and 1-6 window columns.
- HIP is unchanged by default (it compiles); not checked on an AI PRO R9700 (not in the PC).

## v1.3.86 - 2026-10-31

Qwen3.8-Flash-Next decodes about a third faster. RTX 5090 (PCIe Gen4 x4) with ISTA-DASLab GSQ-RCO IQ3_S, ctx 32K:

| | v1.3.85 | v1.3.86 |
| --- | ---: | ---: |
| `./abyss.sh --bench` decode (3 questions, MTP drafter on) | 87.6 tok/s | 116.1 tok/s |
| 2 prose answers, drafter on (256 tokens, greedy) | ~84 / ~80 | ~114 / ~111 |
| 2 code answers, drafter on | ~88 / ~98 | ~118 / ~128 |
| token by token (`GEN spec=0`), the 4 answers' mean | ~79 | ~106 |
| VRAM hits | 60-77% | 80-94% |
| CPU lane | 217 experts a token (6.0 ms) | 109 (3.1 ms) |
| quality check (KL against bench/ref) | 0.0019 | 0.0022 |

- **A routing profile for Qwen3.8 ships with Abyss** (`data/expert_counts/qwen38-flash-next.txt`: 36 prompts of
  prose, code, math and Japanese, 8.2M routes). Its packs have no profile of their own, so the experts went into VRAM
  in id order (experts 0-249 of every layer). The server now points `STRATA_GLM_PRIOR` at the shipped profile for a
  Qwen3.8 pack (any quant, matched by its tensors) without an `expert_counts.txt`: the experts warm in its order, and
  each layer gets a VRAM share by its routing (205-322 slots instead of 250 each). A pack's own profile still wins.
- **Qwen3.8 swaps experts between RAM and VRAM sooner**: up to 16 moves a token boundary, a swap at 2 x the resident's
  count + 8 (others keep 8 and 4 x + 16, tuned for DeepSeek V4's 8 MB experts; Qwen3.8's are ~2 MB, ~0.3 ms a copy).
  The swap rule alone: ~88 -> ~100 tok/s drafter on; the profile alone ~115; both ~118.
- **Qwen3.8's hyper-connection mixes (97 a token) run in two launches instead of five, with the same bits**
  (`hc4_fused_parity`, `-DSTRATA_PARITY_HC4=ON`, compares every output byte for byte). The mixes take 2.98 -> 2.21 ms a
  token; token by token ~+8%. The verify windows still use the five launches. Off by default on HIP (HIP sums the
  down and inject rows in another order); `STRATA_Q4X_HC_FUSE=0` / `1` switches it.
- The KL moves from 0.0019 to 0.0022 because the CPU now computes other experts than before (CPU and GPU round
  differently), as between two runs. Other models are unchanged: GLM-5.3-Flash Maya-S-v2 on the 5090, decode 64.1
  tok/s, 8K prompt 1090, as the v1.3.85 build (63.2 / 1100). Not checked on an AI PRO R9700 (not in the PC).

## v1.3.85 - 2026-10-31

- `tools/abl_prompts.py --cyber N` and `--extra-harmful FILE [--extra-n N]`: force malware / hacking prompts, or your
  own (one a line, `#` comments, JSON lines with a `"prompt"`), into the build's harmful side, for a kind of request
  still refused after `--abl-build`. Qwen3.8 ISTA IQ3_S: `--cyber 30 --abl-rank 4` took JailbreakBench refusals from
  11.1% to 1.2% (Malware / Hacking 4 of 5 -> 0), math 43 / 50, text KL 0.027-0.057. docs/UNCENSOR: "Adding your own
  prompts".

## v1.3.84 - 2026-10-31

`--abl-build` and `--uncensor` now work on **Qwen3.8-Flash-Next** (qwen4exp, the fast path). Before, the build
measured zeros on that path and its file did nothing.

- The build captures the residual on the fast path (at each layer's entry and after the head mix, as the other models).
- The per-layer embedding (PLE) rows are edited as a residual write on the token path, the prompt path and the verify
  window, as V4.1's Engram. Off without a file; with a file, `"uncensor": false` per request still gives the unedited
  model.
- ISTA-DASLab GSQ-RCO IQ3_S on the RTX 5090, layers 39+43+36, rank 3 (31 KB): harmful check answers refusing 23 -> 3
  of 24. Capability set (120 items, greedy) with the edit vs without: math 45/50 vs 42/50, code 30/30, Japanese 10/10,
  multiple choice 20/20 on both, MMLU-500 84.4% vs 84.8% (McNemar p 0.75). A full comparison with downloaded
  uncensored releases is still running.

## v1.3.83 - 2026-10-31

Speculative decoding on CUDA gave wrong answers since v1.3.74. Fixed. With Qwen3.8-Flash-Next's MTP drafter on an RTX
5090 (ISTA IQ3_S), 120 capability items + 30 everyday tasks, greedy: **math 3/50 -> 43/50** (drafter off: 43/50),
code 30/30, looping answers 2 of 130 (off: 3), **100.4 tok/s against 72.0 with the drafter off**.

- **The route kernel wrote past a plan array and zeroed the verify window's expert weights.** v1.3.74 let it clear up
  to 16 entries of `fetch_src`, which holds max(8, k); in a window column's block `plan_w` comes right after it, so
  every window column on CUDA ran without its routed experts (only the shared expert and the CPU lane's share). The
  token path's padding hid it there. This hit every model's windows: the Qwen3.8 MTP drafter, DeepSeek V4.1 with
  DSpark (its answers parted from the no-draft ones at token 2-6, now at 16 or later or not at all), and the n-gram
  windows. Not checked on HIP (no AMD card in the PC right now).
- **Qwen3.8's batched windows: one scratch slot set a column** (up to 6 columns, as V4.1's). Columns 3-5 shared a
  set, so a fetch for one could overwrite another's experts before the batched launch read them. Any model's batched
  launch now needs every column in its own set.
- **Qwen3.8 past 2,051 cells: a rejected window column no longer leaves the attention indexer's key tail changed**
  (the tail is put back and the kept columns' keys added again).
- `./abyss.sh --bench` has a **drafter check**: the first answer again token by token, and where it parts
  ([docs/BENCH.md](docs/BENCH.md#the-drafter-check)). Over the `--serve` wire, `GEN ... spec=0` asks one answer
  without speculative decoding; `tools/mtp_greedy_check.py` compares many.

## v1.3.82 - 2026-10-31

Small VRAM (issue #23: an RTX A2000 Laptop 8 GB + 64 GB RAM, PCIe Gen4 x8), and CUDA's VRAM reserve on big cards.
Checked on an RTX 5090 given the reporter's pool (`STRATA_GLM_POOL_GB`): Ornith-1.5-122B IQ3_XXS-B with a 1.45 GB
pool reads a prompt at **356 tok/s (2K) / 347 (8K) instead of 46** (token by token), decode 45.5 tok/s, KL 0.0073
against a 16 GB run either way; GLM-5.3-Flash Maya-S-v2 with a 1.2 GB pool (4 slots a layer) **starts** where it was
refused, decode 9.1 tok/s (SSD-bound), KL 0.039 (a 32 GB run: 0.036-0.044). Nothing changes on a 32 GB card's
GLM-5.3-Flash run (decode 64.5 / 64.6 tok/s, 8K prompt 1113 / 1105, no new log lines). One page for all of it:
[docs/SMALL-VRAM.md](docs/SMALL-VRAM.md) ([日本語](docs/SMALL-VRAM.ja.md)).

- **CUDA keeps 1.5 GB of VRAM free beside the expert pool on cards over 12 GB** (`STRATA_GLM_RESERVE_MB`, as HIP
  since v1.3.68; 700 MB before). An RTX 5090 running DeepSeek V4.1 UD-IQ2_XXS at `--max-context 65536` ran out of
  VRAM: `--bench` stopped in the prompt attention (`dsv4_fast attend: out of memory`) and a coding run with the
  DSpark drafter in its MMQ buffers. With 1.5 GB: `--bench` decode 23.1 tok/s, 8K prompt 357, KL 0.0086, and 8
  coding tasks with DSpark at 64.9 tok/s, no out-of-memory. GLM-5.3-Flash on the 5090 loses 3 pool slots of 78 (decode
  64.1 tok/s, 8K prompt 1099: within the run-to-run spread). A card of 12 GB or less keeps 700 MB: its pool is a few
  slots a layer (below).

- **The expert pool may hold fewer slots a layer than a route** (GLM-5.3-Flash / Ornith: 8). The floor is now the
  spares and one resident (4; `STRATA_GLM_POOL_FLOOR`): a miss lands in a spare or its own scratch slot and the CPU
  lane computes its share from RAM, none of which needs the route to fit - the reporter ran 6 of 8 that way, KL 0
  against the engine's own reference. The refusal below the floor says how many MB are short for the floor and for
  a whole route, and what frees them (`--max-context`, `--kv q4_0`, the IQ4_XS dense pack); a pool between the two
  says so once at start. The expert-count slot split no longer asks a layer for more slots than the pool has.
- **Prompts in chunks on a pool that could lend none**: the prompt path used to leave every layer a route, the
  spares and 2 (13 slots), so a pool of 9 (Ornith-1.5-122B IQ3_XXS-B on 8 GB) read the prompt token by token at
  16 tok/s. Such a pool now keeps 4 slots a layer while a prompt runs (`STRATA_GLM_PREFILL_KEEP`) and lends the
  rest - ~800 MB on Ornith, chunks of ~2,000 tokens; the decode's misses refill the tail after the prompt. The
  "can lend no prompt chunk" line now says what the smallest chunk needs and what the pool can lend. Measured on
  the 5090 with the reporter's pool (8 slots a layer, chunks of 768): 46 -> 356 tok/s. A pool at the floor (4) has
  nothing to lend and still reads token by token.
- **The CPU lane takes every RAM-tier expert when it is much faster than the link** (`STRATA_GLM_LANE_LEAN`, 4x):
  the measured split kept 1 of 8 on the link at 16.8 tok/s where the `--deep` sweep's share 0 ran 18.5 (both lanes
  read the same DRAM; the link copy sits on the GPU's stream). The lean takes effect once the pool has settled (the
  `:settle` switch of v1.3.70, automatically) and steps back when the VRAM hits fall, so a prompt's misses still
  refill the pool first (issue #14). Cards whose split already sends every expert to the CPU (Gen4 x4 / OCuLink)
  and cards whose link is close to the CPU (an R9700 on Gen5 x16, 0.311 vs 0.499 ms) are unchanged.
- **bench/ref**: a quality reference for Ornith-1.5-122B IQ3_XXS-B (zephel01's 16G64G quant), so `--bench` checks it.
- **Docs**: GLM-5.3-Flash Maya-S-v2 on an 8 GB card with its dense tensors at IQ4_XS (`tools/requant_dense.py
  --from=q6_k`: which tensors, 5.6 -> 3.6 GiB, the pack and config lines), in docs/SMALL-VRAM.md (EN + JA).

## v1.3.81 - 2026-10-30

- `/health` answers **503** (with `Retry-After: 5`) while the model loads, the body still `"status": "loading"` and how
  far: a client that reads 200 as "ready" (a llama-swap health check, a script's wait loop) now waits for the engine.
  Abyss Terminal reads the 503's body and shows the loading progress as before. 200 once the engine is ready, as
  before v1.3.80.

## v1.3.80 - 2026-10-30

Not yet run on a GPU: the engine's new STAT / INFO fields and the early bind pass the HIP compiler's syntax check and
the host tests (serve 204, tools 263, Terminal 51); the live check on an engine (R9700 / RTX 5090) follows. `/health`
answers 200 with `"status": "loading"` while the model loads (503 from v1.3.81).

**Abyss Terminal (phase 1): `./abyss.sh --terminal` watches and runs the server from a terminal - this PC's or
another's - and closing it leaves the model serving ([MANUAL](docs/MANUAL.md#abyss-terminal-abysssh---terminal)).**

- Eight views: Overview (decode and prompt tok/s in big, where a token's time goes as one bar with `--bench`'s verdict
  and advice, GPU / VRAM / temperature / power / PCIe while working / CPU / each SSD of a mirror, context, experts in
  VRAM, the RAM tier, RAM with swap and memory pressure, the drafter, every card with its role, the last requests),
  Speed, Memory, Requests, Hardware, Logs, Connect, Models (start / stop / switch a config, run `--bench`).
- Two Overview layouts, `l` or `--layout speed|devices`, remembered per user: Speed (the big numbers and tiles) and
  Devices (every GPU, the CPU and every drive in a panel of its own, a grid that wraps to 10 cards; the default for a
  config with 3+ cards until one is chosen).
- Several GPUs as the usual case: per card the layers and expert pool of a layer split (from the engine log), power
  and PCIe against the card's own limit and max (a 5090 + 3090 + V100 box shows each), the busiest starred, a card
  whose link dropped or sits under half the others' (an x1 riser) in pink with a banner; any number of drives with
  their share of the expert reads; at 80x24 one summary line plus the cards that matter.
- The server it starts runs detached (PID file, output beside the config); quit asks keep serving or stop. A stop is
  `POST /api/shutdown`, else Ctrl+C to its process group.
- Remote: `--url` with `--key-env NAME` (the key read from that variable, never stored), read-only, a warning for a key
  over plain http. `--plain` (no Textual), `--once`, `--json`, `--demo <scenario>` (recorded readings, no server).
- `--bench`'s bottleneck rules and advice moved to `tools/bench_rules.py`, which both use (the bench's output is
  unchanged). A value not reported is `--`, never 0; the live per-token ms wait for the engine's next STAT keys
  (phase 2) and come from the engine log on this PC until then.

**Server side of Abyss Terminal (phase 1.5): `/metrics` names the run and reads the whole machine, and
`POST /api/shutdown` stops a runaway engine from another program ([MANUAL](docs/MANUAL.md#using-it)).**

- `POST /api/shutdown` (JSON body, `{"force": true}` optional): the request in progress is cut, an engine that does
  not stop it within `STRATA_SHUTDOWN_WAIT_S` (15 s) is killed, then the server exits. Allowed with the `api_key`, or
  from this PC (loopback) when there is none; never from a foreign web page; `"allow_shutdown": false` turns it off.
  Over Tailscale set an `api_key` (behind `tailscale serve` every client looks like loopback).
- AMD PCIe link: the narrowest link up to the root port, as `--bench` reports it - an AI PRO R9700 on OCuLink now reads
  Gen4 x4 (it read the card's internal Gen5 x16). AMD's PCIe MB/s stays empty: RDNA 3/4 has no counter (amd-smi says
  N/A too).
- GPUs are read through backends tried in order per card, each filling what the ones before left empty, with
  `source` naming them: NVML, else `nvidia-smi`; AMD sysfs + the binary `gpu_metrics` table (throttle reasons,
  junction / memory temperatures, an APU's RAM traffic), else `amd-smi` / `rocm-smi`; Windows' GPU performance
  counters as the last resort. `hardware.gpus` lists one card too, with its `role` (`main` / `split` / `helper` /
  `igpu`, the iGPU lane's Radeon); the `gpu_*` totals and the stall watchdog count only the model's cards. New per
  card: `clock_mhz`, `mem_clock_mhz`, `temp_junction`, `temp_mem`, `throttle`, `pcie_width_max`, `name`.
- CPU (`serve/telemetry_cpu.py`): load and clock per core, clusters (cpufreq policies: big.LITTLE, a Ryzen's CCDs),
  frequency driver / governor / EPP, temperature (k10temp, coretemp, Arm thermal zones), power (RAPL, amd_energy,
  hwmon, an AMD APU's power table - RAPL is root-only on most distributions), Intel throttle counts, and the
  instruction sets the CPU lane uses (AVX-512 / VBMI / AVX-VNNI, NEON / SVE).
- `hardware.drives`: each model / `STRATA_GLM_MIRROR` drive's own MB/s (Linux); swap, page cache and memory pressure
  (PSI).
- `engine.display_name` / `quant` / `config` / `backend`; drafts accepted / offered per request and in `totals`;
  `tiers.history` carries every numeric STAT key (keys a newer engine adds get their series).
- A value that cannot be read is `null` (shown as `--`), never 0.

**Abyss Terminal phase 2: the token's time live from the engine, a server that answers while it loads, the engine log
over HTTP.** The engine change is printf of counters it already keeps (STAT every 8 tokens, INFO once, PP, DONE);
not yet run on a GPU here (the R9700 was busy): the syntax is checked for HIP, the parsers against the lines.

- Engine: `STAT` carries `disk_ms disk_waits miss_ms lane lane_ms tail_skip` (the "glm stat" line's per-token numbers
  while the answer is written), `mirror_mb0 / mirror_mb1` (the mirror's bytes per drive), a window over the last ~64
  tokens (`w_ms_tok w_disk_ms w_lane_ms`) and the drafter's `spec_acc spec_offered spec_rounds`; `INFO` adds
  `lane=cpu|igpu|exl3|off drafter=mtp|nextn|dspark|none expert_mib tail_skip defer mirror helper`; `PP` the reused
  tokens and the prompt's SSD wait so far; `DONE` an 11th column, the verify steps. Older engines' shorter lines are
  read as before.
- Server: the port is bound before the engine starts - `GET /health` answers `status: "loading"` with `load`
  (stage, text, percent and GB of the warm-up, elapsed seconds; the same under `reloading` during a context change),
  everything else 503 + `Retry-After: 15` until the model is ready; one listening socket for the whole life.
  `GET /api/log?since=<offset>&max=<bytes>` (the key needed) sends the engine log's tail by byte offset, whole lines,
  the API key's values masked. `/metrics`: `requests[].spec_rounds`, `live.prompt_reused / prompt_disk_ms`.
- Terminal: the Token time tile and the Speed view show the live window while an answer is written (`live · last ~64
  tokens`), the last answer's final STAT when idle, the engine log for older engines; the lane's kind from INFO and
  its estimated GB/s; the mirror's split as the engine counts it; the drafter's live acceptance and tokens a verify
  step; the prompt's reused tokens and SSD wait while reading; a remote server's loading progress; its engine log
  over `/api/log` (Logs and the per-request breakdown work remotely). `--demo starting`. Tests for every new field
  and for engines and servers without them (`serve/test_dashboard.py`, `tools/test_abyss_terminal.py`).

## v1.3.79 - 2026-10-30

**Two items of the 2026-10-30 Strata/Maya review (no GPU needed; the ideas are Maya #626 and #419, the code is
Abyss's own).** Host-side changes only (the server's parser and the setup's calibration store); the engine is the
same as v1.3.78. Not yet seen on hardware: the PCIe link as an RTX 5090 on OCuLink reports it (this weekend).

- Streamed tool calls: a call's body was one growing string and every new piece searched it from the start for the
  end tag (and the streamed value's end), so an agent's Write of a 200 KB file cost about a second of CPU at 8
  characters a piece, growing with the square of the length. The body is kept as pieces, the end tag is looked for
  in the new piece plus the tail a tag split across pieces needs, and the streaming scanner consumes what it has
  reported: 20-100x faster on the 200 KB case, the events byte for byte the same (checked against the old parser on
  7,680 chunkings of 24 texts in every form - GLM, Qwen, DSML V4 and V4.1, quoted and unterminated calls).
- The server's connections set `TCP_NODELAY`: a stream's one-token SSE pieces no longer wait for the delayed ACK of the
  piece before (up to 40 ms each on a slow link).
- `--calibrate`'s store (`~/.config/project-maya/calibration.json`) keys its results by the GPU's PCIe link (what the
  card and its slot can do together) and the driver version too: a card moved between an x16 slot and an OCuLink x4,
  or a driver update, used to reuse the tuning measured on the other. An older store is read and its entries are
  stale: not used, and the setup says once, in one line, what the saved tuning was measured on, what it is now, and
  that `./abyss.sh --calibrate` measures again. The nearest-context lookup still works on the same slot and driver.

## v1.3.78 - 2026-10-30

**Bug-fix pack (the 2026-10-30 review's week-1 items, no GPU needed): a prompt-path out-of-range read, the server's
engine stop paths, waits with limits, the SSD mirror's copy check, bounds on the memory knobs, and the docs' stale
"not yet" lines.** Checked on the AI PRO R9700 (HIP): `./abyss.sh --bench` matches the known-good runs
(DeepSeek V4.1 UD-IQ2_XXS KL 0.0036, Ornith-1.5-122B UD-Q4_K_XL 0.0058; GLM-5.3-Flash Maya-S-v2 0.0316, the same
"slightly off" as v1.3.69's 0.0337) at the usual speeds; a 32,768-token V4.1 prompt (583 tok/s) with the mirror on a
second SSD (43.7% of the reads from it, decode 22.3 tok/s); Qwen3.8 on the CLI with `--max-context 512` stops with
"the sequence exceeds max_ctx" at the end, no out-of-range write. Not checked here: the RTX 5090 (CUDA) and the RDNA
sibling chips (no card).

- Prompt path: the MMQ row-id table (`iota`) is allocated and filled 512 entries past `T * n_exp_used`. `mul_mat_q`
  loads a whole tile of ids before it checks the row bounds, so the last tile read past the table - a long prompt's
  (or a split's) "illegal memory access", an out-of-range read under compute-sanitizer (Maya #645's `IDS_PAD`, the same idea).
- Qwen3.5 / Qwen3.8 drafters: the chain of draft steps stops where the context ends (`draft_kmax`); it wrote KV rows
  past `--max-context` when the context was nearly full (the server's margin of 8 hid it, the CLI had none). DSpark
  keeps its rows in a ring and was not affected.
- DeepSeek V4.1 Engram: a failed background row read fails its own token (the step ends with the error) and is then
  forgotten; before, the flag stayed set and every later V4.1 request failed with it, after one transient EIO.
- `STRATA_GLM_MIRROR`: a copy is accepted after its size, first and last MiB **and 32 blocks of 256 KiB spread over
  the file** match (~10 MiB a shard; a copy that stopped part way, or a file only fallocated to the size, passed the old
  head/tail check and `read_slice` then returned other bytes). A read interrupted by a signal is done again instead of
  turning the mirror off for the run. Reads that leave the O_DIRECT path are counted and reported with the mirror's
  timing line. The file layer is `src/core/shard_io.cpp` now, with a host-only test (`tools/test_shard_io.py`).
- Numbers from the environment: `env_num` (include/strata/core/knobs.hpp) reads them - a value that is not a number
  (`1,0`, `50s`) gives the default and a line, one outside its range is clamped and said. `STRATA_GLM_RAM_FRAC` is held
  to 0.05..0.95 on a discrete card (1.0 lifted the cap: the thrash case); `STRATA_GLM_RESERVE_MB` / `POOL_GB` / `VRAM_GB`
  cannot go negative; `STRATA_DS41_ENGRAM_THREADS` is bounded at 64, `_CACHE_MB` at 64 GB; `STRATA_WATCHDOG_S` with a
  typo keeps its default instead of turning the watchdog off; `STRATA_GLM_RAM_HEADROOM_GB` has one default in one place
  (6 GB, 16 with unified memory; the two sizing sites had their own).
- Server (`serve/server.py`): `StrataEngine.stop` was defined twice and the later one (STOP the request) hid the first
  (QUIT, kill, wait for the GPU memory), so a context change from the dashboard killed the engine outright and a
  conversation kept with `STRATA_GLM_SLOT_KEEP` was lost; a restart now asks QUIT first (up to `STRATA_ENGINE_QUIT_S`).
  Waits have limits with a line that says which: the start's READY (`STRATA_ENGINE_START_S`, 30 min), a stopped
  request's drain to DONE (`STRATA_ENGINE_DRAIN_S`, 120 s of silence; the stall watchdog applies there too), the image
  encoder's start and each picture (`STRATA_VISION_START_S` / `_ENCODE_S`, 600 / 120 s). The config's `"env"`
  `STRATA_GLM_THINK_RESERVE` / `STRATA_ENGINE_WRAP_S` / `STRATA_ENGINE_QUIT_S` reach the server (MANUAL listed
  THINK_RESERVE as a config setting; they went to the engine only), and every server-side number is checked at start
  instead of a traceback at shutdown. `GET /status` without the configured API key shows `busy` / `queued` only: the
  tail of the text being generated, the prompt's size and the tool being called need the key (they went out to anyone
  on the network with `host: 0.0.0.0`).
- About > Updates: a zip install (no git origin) looked at Project Maya's releases; the fallback is zephel01/Abyss, as
  are the watchdog's report line and the setup screen's link.
- AMD: the RDNA sibling chips - RX 7800 XT / 7700 XT (gfx1101), RX 7600 (gfx1102), Radeon 890M (gfx1150), RX 9060
  (gfx1200) - take their family's expert kernels (they fell back to the legacy ones, 20-30% slower a call, the same
  sums; the device guards already named them) and setup builds for the one it finds next to the three validated
  targets. Not validated on a real card: please report.
- Tests (CPU only, in CI): the fake-engine tests (`serve/test_engine_process.py`: QUIT on restart, an ignored QUIT, the
  drain and start limits, a context reload, the vision limit, `/status` with and without the key, the env numbers),
  `tools/test_shard_io.py`, `tools/test_knobs.py` (the env table, `draft_kmax`), and a docs lint
  (`tools/test_docs_env.py`): every setting the code reads is in MANUAL / SETTINGS or on `tools/env_undocumented.txt`
  (237 debug and experimental names today - the list only shrinks), EN and JA name the same settings and config keys,
  every config key is in SETTINGS.
- Docs: MANUAL / README / docs/deepseek41 no longer say V4.1 has no speculative decoding (DSpark, v1.3.53) or that its
  prompt attention runs a token at a time (batched in v1.3.60); the swe-bench "first 35 of 60" rows say they are a
  partial v1.3.31 run with the full run still owed; the server's settings (`STRATA_ENGINE_STALL_S` / `_WRAP_S` /
  `_QUIT_S`, `STRATA_HTTP_BACKLOG`, `STRATA_MAX_BODY_MIB`) and the new limits are in MANUAL's table; SETTINGS names
  `installer` and vision `threads`; docs/amd lists the sibling Radeons.
- Not done from the review's list (reasons): `tail_skip_disk` before the 90% early return - with the tail skip off
  there is nothing starved, so a RAM fill default of 0 is right; DSpark's drafter cannot overrun (a ring of 256 rows).

## v1.3.77 - 2026-10-29

**Two SSDs without a RAID: copy the model's shards to a second NVMe and `STRATA_GLM_MIRROR=<folder>` reads every
expert from whichever drive will finish first ([docs/SSD2](docs/SSD2.md), 日本語: [SSD2.ja.md](docs/SSD2.ja.md)).**
For PCs that cannot take more RAM but have a free M.2 slot.

DeepSeek V4.1 Flash 1017 unsloth UD-IQ2_XXS, AI PRO R9700 + the iGPU lane, drive 1 a 6.2 GB/s internal NVMe, drive 2
an NVMe in a USB4 enclosure (3.5 GB/s); medians of 2 sequences x 3 rounds, one and two drives alternating:

| GPU + RAM | a token (1 → 2 drives) | SSD wait a token | read from drive 2 |
| --- | --- | --- | --- |
| 32 GB + 96 GB | 51.4 → 49.2 ms (+4.5%) | 19.7 → 15.7 ms (−20%) | 37-45% |
| 32 GB + 64 GB | 71.9 → 59.3 ms (+21%) | 27.9 → 22.2 ms (−20%) | 38-45% |
| 16 GB + 64 GB | 92.0 → 77.7 ms (+18%) | 42.8 → 34.0 ms (−21%) | 39-47% |

- Each read picks the drive by (reads in flight + 1) / its recent speed, measured again on every read; one read in 32
  goes to the other drive so a drive that looked slow once is measured again (without it one run read 0.0% from the
  mirror). The start-up fill of RAM and VRAM reads from both too (5.2 → 6.9 GB/s).
- A copy is used only when its name, size and first and last MiB match; a failed read from it is read again from the
  first drive and the mirror is then left out. The output does not change. Linux (O_DIRECT reads).
- `STRATA_GLM_DISK_LANE=1` is slower with it (and without it) on this PC: do not combine them.
- As far as we checked (Strata v0.2.7.1, Project Maya v1.1.36, llama.cpp, ik_llama.cpp, KTransformers), no other
  engine reads the same file from two SSDs while answering; Maya's `MAYA_DS4_LOAD_INTERLEAVE` speeds up the start-up
  copy of shards split over two drives.

**DeepSeek V4.1: unsloth's re-upload of 2026-10-28 measured** (revision `85f81fb970add013033f25e94006713ca2c0aaf2`;
[docs/deepseek41](docs/deepseek41/README.md#ggufs-that-load)):

| file | size | KL (old file) | swe-bench 8 tasks, 32 GB + 96 GB / 32 GB + 64 GB |
| --- | ---: | --- | --- |
| UD-IQ2_XXS - the pick for quality | 300 GB | 0.141 (0.188) | 33.9 / 29.5 tok/s |
| UD-IQ1_M - the pick for speed | 250 GB | 0.199 (0.270) | 40.6 / 30.5 tok/s |
| UD-IQ1_S - runs, not recommended | 233 GB | 0.257 (0.36) | - |
| ISTA-DASLab GSQ-RCO IQ2_XXS (new layout) - runs, not recommended | 138 GB + Engram | 0.315 | - |

swe-bench rows: tail skip 4 + disk, affinity 0.08, the DSpark drafter (threshold 0.6), iGPU lane off.
`./abyss.sh --bench`'s quality check has references for the new UD-IQ2_XXS and UD-IQ1_M (the re-upload is 55 bytes
smaller, so the old UD-IQ2_XXS reference skipped the check on new files; it stays for the old files as
`*.old1020.json`).

## v1.3.76 - 2026-10-28

**Qwen3.8-Flash-Next: the popular ISTA-DASLab and unsloth GGUFs on the fast path, as beta - which file fits which PC:
[docs/qwen38](docs/qwen38/README.md#other-ggufs-beta) (日本語: [README.ja.md](docs/qwen38/README.ja.md)).**

| File | Resident | vs Q8_0 (KL / same top) | Abyss vs llama.cpp (KL) | R9700, prose / code |
| --- | ---: | --- | --- | --- |
| ISTA-DASLab GSQ-RCO IQ3_XXS | 47.0 GB | 0.398 / 71.6% | 0.019-0.022 | 66 / 84 tok/s |
| ISTA-DASLab GSQ-RCO IQ3_S | 54.8 GB | 0.282 / 76.5% | 0.026 | 51 / 77 tok/s |
| unsloth UD-IQ4_XS | 64.9 GB | 0.191 / 80.4% | 0.038-0.043 | 57 / 74 tok/s |
| unsloth UD-Q4_K_XL | 82.5 GB | 0.100 / 86.4% | 0.029-0.037 | 55 / 71 tok/s |

- New in the engine: Q2_0, Q5_1 and Q8_0 expert rows; IQ4_XS / Q5_1 / Q2_0 dense rows in the verify window; Q5_1 on
  the prompt path; BF16 hyper-connection, `ssm_alpha` / `ssm_beta` and PLE key / value rows, and F32-widened injects
  on the prompt path (ISTA keeps those in BF16).
- Beta: each file was checked on one PC only (correctness against llama.cpp on the same file, a short speed run with
  the MTP drafter, and 16 GB + 64 GB / 24 GB + 64 GB caps). `./abyss.sh --bench` on these files ends with a request to
  post the report (`tools/bench_ref.py --beta` marks a reference so); the bench issue template asks for the file's
  folder.
- Capped to 16 GB of VRAM + a 46 GB RAM tier (ISTA) or 24 GB + 46 GB (UD-IQ4_XS) there were no SSD reads and prose
  stayed at 45-61 tok/s, code 54-77 on this PC - which has an iGPU lane for the RAM-tier experts; without one expect less.
- Other models: V4 gives bit-identical token-path logits to v1.3.75.

## v1.3.75 - 2026-10-28

**An opt-in second-GPU expert helper for DeepSeek V4.1's decode on CUDA (PR #22 by tjm8874, following #21):
[docs/EXPERT_HELPER.md](docs/EXPERT_HELPER.md).**

- `STRATA_GLM_EXPERT_HELPER=<gpu>` keeps a pool of RAM-tier experts (`STRATA_GLM_EXPERT_HELPER_GIB`, default 13 GiB)
  on a second visible GPU and computes them there during decode, instead of on the CPU lane. Prompts, attention,
  Engram and the head stay on the primary GPU. Off by default.
- Measured by the contributor (RTX PRO 4500 32 GB + RTX 5070 Ti 16 GB, PCIe 5.0 x8 each, 16K prompt / 2K answer):
  decode 45.2 -> 49.8 tok/s (+10%) with the primary's full memory, 28.8 -> 36.8 tok/s (+28%) with the primary capped
  at 16 GiB.
- A missing GPU, a pool that does not fit or a helper error leaves the CPU lane in charge; an error mid-route
  recomputes the whole route on the CPU and turns the helper off. Speculative decoding and layer graphs keep it off.
  Short requests can lose more to placing the experts than they gain, hence off by default.
- Scope: an unsplit native V4.1 pack on CUDA with a CPU lane (no iGPU lane). HIP builds it but leaves it off.
- Checked here before merging (AI PRO R9700): no conflicts with v1.3.74, the HIP build passes, and with the helper
  unset V4.1 and V4 give bit-identical token-path logits to v1.3.74; `STRATA_GLM_EXPERT_HELPER=1` on HIP prints
  "disabled" and gives the same output. Not yet run on CUDA here.
- MANUAL.ja and README.ja list it too.

## v1.3.74 - 2026-10-28

**Qwen3.8-Flash-Next (qwen4exp) on the fast path (`--glm-pack`), with its prompt path, verify window and MTP drafter:
[docs/qwen38](docs/qwen38/README.md) (日本語: [README.ja.md](docs/qwen38/README.ja.md)).**

- Measured on an AI PRO R9700 with orcarouter's Uncensored IQ3_XXS against the older engine path: prompts at 30K
  217 -> 1,037 tok/s; answers 52 -> 64-65 tok/s on prose and 66-85 -> 86 tok/s writing code (100-107 copying code);
  a 31K prompt's passphrase found; a conversation's next turn reuses the cache (10,802 of 10,876 tokens, 0.28 s).
  KL 0.026 (token path) / 0.035 (prompt path) against llama.cpp, whose own HIP-vs-CPU difference is 0.045.
- `tools/iq_pack.py --fast` packs the model for the fast path (nothing converted to BF16). The drafter is the older
  path's MTP runtime files, `STRATA_Q4X_MTP=<dir>` (experts kept in Q2_0 in VRAM, ~1 GB); its defaults are 4 drafts at
  confidence 0.75 (`STRATA_Q35_MTP_K`, `STRATA_Q35_MTP_MIN`).
- The pieces: the hyper-connections (their q8_1 written in the same kernels), the gated delta-net with a sigmoid
  output gate, layer 1's PLE (read from the SSD on a thread, kept in the snapshot and rewound after a window), QSA's
  block selection past 2,051 cells, top-k up to 16, IQ4_NL experts, i-quant projections in the multi-column GEMV.
- Fixed on the way: the prompt path's MMQ read past a 640-wide down matrix into a slot's unset tail (NaN rows; the tail
  is now zeroed, only when the width is not a multiple of 512); the prompt attention kernel skipped a 12-head group.
- Other models: token path bit-identical to v1.3.73 on V4.1, V4, GLM-5.3-Flash and Ornith; prompt path within its
  run-to-run spread.
- NVIDIA: this path has not been built or measured there yet.

## v1.3.73 - 2026-10-28

**Docs only: the config files on one page, [docs/SETTINGS.md](docs/SETTINGS.md) (日本語:
[docs/SETTINGS.ja.md](docs/SETTINGS.ja.md)), written for a first-time user.**

- Setup and config told apart: setup (build, model, pack) runs once and writes the config file and start script; the
  config is what you change afterwards.
- What to change where (context, sampling, port - config and start script both -, access from other PCs, GPUs,
  uncensoring, engine tuning), where the files are with their real names (`abyss-<name>.json`, `run-abyss-<name>.sh`,
  `maya-<model>.json`, the logs, `.shared-settings.json`), finding the config in use (`abyss.sh models`), changing one
  step by step (backup, JSON rules, restart, which log to read), every key the server reads, the keys Abyss writes
  itself (dashboard context, setup, `--rewrite-config`, `--gpu-tune`, `--calibrate`) and the other files.
- MANUAL's "Configure and start" field table, docs/UNCENSOR's "which file" and the README's docs table link to it.
  docs/UNCENSOR now names the file to edit (`abyss-<name>.json`), where the key goes and how to restart.
- No code changes.

## v1.3.72 - 2026-10-28

**Docs only: everything about uncensoring on one page, [docs/UNCENSOR.md](docs/UNCENSOR.md) (日本語:
[docs/UNCENSOR.ja.md](docs/UNCENSOR.ja.md)).**

- Which way to use (the run-time `--uncensor`, an abliteration's own ABL1 edit, the `--abl` overlay shard,
  abliterated GGUFs) compared in one table; how the edit works; turning it on and switching it per request
  (config, request, `POST /settings`, `GEN abl=`); making the file for your model (`tools/abl_prompts.py` and every
  `--abl-build` option, with a sample log); checking a file (text KL with the score path, held-out refusals); the
  measured results with file sizes and the harmful answers' breakdown; the limits; the `.abl` file format.
- The MANUAL section, the refusal rows in MODELS, the abliteration notes in docs/deepseek41 and the README lines now
  link to it instead of repeating it.
- No code changes.

## v1.3.71 - 2026-10-28

**Uncensoring at run time (opt-in): `--uncensor <file>` takes a model's refusal direction out of the residual stream
while it runs - no new GGUF, switchable per request.**

- The edit removes a few directions from every write into the residual stream: the embedding, each layer's attention
  output, the whole FFN output (routed + shared experts + the CPU lane's answer) and V4.1's Engram add (`w - s Q Q^T w`,
  Arditi et al.'s directional ablation). The routed experts on the SSD are not touched. It runs on the token path, the
  prompt path and the speculative verify windows of GLM, DeepSeek V4 / V4.1 and the qwen35moe models.
- Turning it on: `"uncensor": "<file.abl>"` in the config (engine: `--uncensor <file>` or `STRATA_GLM_ABL=<file>`). A
  request's `"uncensor": false` runs without it, `true` with it (engine: `GEN ... abl=0|1`); it is also one of the shared
  settings other apps get (`POST /settings`). A conversation computed with the other setting is not reused. While a file is loaded the
  CPU lane's hc3 fusion and `STRATA_GLM_DEFER` are off. mrfakename's ABL1 files (attention and shared-expert outputs
  only) still load.
- Off is unchanged: with the file loaded and switched off (`STRATA_GLM_ABL_ON=0`) the score path's logits match a run
  without the file bit for bit (V4.1, KL 0, the same CPU-lane count); speed with it on stays within run-to-run noise.
- `--abl-build <out.abl> --abl-prompts <file>` makes the file (16-82 KB) from the model itself in under an hour:
  harmful minus harmless mean residuals per layer, made orthogonal to the harmless means (without that the direction
  carries the residual's huge fixed channels and breaks the model), then candidate layers tried on held-out prompts.
  `--abl-gen N` judges a candidate by N generated tokens (GLM opens with a heading and refuses in its second line),
  `--abl-rank`, `--abl-scale`, `--abl-range`, `--abl-candidates`, `--abl-max-kl`, `--abl-layer`, `--abl-dirs` (reuse a
  build's measurement). `tools/abl_prompts.py` writes the prompt file from public sets (AdvBench, MaliciousInstruct,
  ForbiddenQuestions, SimpleSafetyTests, JailbreakBench, XSTest), mixed evenly, never the evaluation's prompts.
- Measured on an AI PRO R9700 with 101 held-out prompts (66 harmful, 35 harmless), greedy:

  | Model | Harmful refused (off -> on) | Harmless refused (off -> on) | Text KL |
  | --- | :---: | :---: | :---: |
  | DeepSeek V4.1 Flash 1017 UD-IQ2_XXS | 86% -> 3% | 17% -> 3% | 0.023 |
  | DeepSeek V4 Flash 0731 UD-Q2_K_XL attn-q5k | 85% -> 6% | 11% -> 3% | 0.013 |
  | GLM-5.3-Flash Maya-S-v2-IQ2_XXS | 88% -> 8% | 11% -> 3% | 0.053 |

  For comparison, orcarouter's abliterated V4.1 Mix-256G (rebuilt weights) has a text KL of 0.083. docs/MANUAL has the
  steps; the edit removes refusals, it adds no knowledge.

## v1.3.70 - 2026-10-27

**DeepSeek V4.1: one Engram worker instead of one a token (#20), and a session-wide late share for the CPU lane (#14).**

- The async Engram reader (`STRATA_DS41_ENGRAM_ASYNC`, on by default) started a new waiting thread on every decode
  token: it was detached right after starting, and a detached `std::thread` is never joinable, which is what the start
  test checked. The threads piled up for as long as the engine ran and each token woke all of them, so decode slowed
  answer by answer. A flag now starts the worker once. Measured on an AI PRO R9700 (UD-IQ2_XXS, one server session,
  eight 256-token answers): 44 -> 2,188 threads and 16.1 tok/s on average before, 45 threads and 20.6 tok/s (+28%)
  after; the reporter's RTX PRO 4500 went 40.3 -> 37.5 -> 33.0 tok/s over three 2K answers before and held 45 after.
  Long-session V4.1 numbers measured before this release were low by this.
- `STRATA_GLM_PCIE_SHARE_LATE=<share>:settle[:<window>[:<gain>[:<drop>]]]`: the late share follows the pool instead of
  the request - every `<window>` token steps (default 128) the VRAM hit rate is measured, the lane moves to the late
  share once a window gains less than `<gain>` points (default 1), and back when the rate falls `<drop>` points (default
  5) below that level. The CPU lane is now kept even when the start share is 1 if a late share is set. On an R9700
  (Ornith IQ3_XXS-B, 16 GB config, 8 answers) share 0.5 -> 0 went 11.5 -> 18.1 tok/s (+57%) and share 1 -> 0 7.8 ->
  10.8 (+38%) against the fixed share; share 0 from the start (35.9) stays the best there, so it is for boxes where the
  `--deep` sweep picks a share above 0 (docs/MANUAL).
- README: the supported-models table lists the other GGUFs that run (Ornith-1.5-122B, Qwen3.5-122B-A10B, the
  orcarouter V4.1 abliterated mix, Qwen3.8-Flash-Next).

## v1.3.69 - 2026-10-27

**DeepSeek V4.1 UD-IQ2_XXS: where its ~20 tok/s goes, the speed setting for it, and a read-chunk fix.**

- `STRATA_GLM_READ_CHUNKS` above 10 broke SSD-read experts: a streamed expert's ready word has a bit per part and
  chunk (3 x 16 = 48 of 32 bits), so the device copied chunks that had not landed and the routes read garbage
  ("invalid expert ID"). It is capped at 10 now (default 8 unchanged; 10 was no faster on an R9700).
- Measured on an AI PRO R9700 + 8060S lane (UD-IQ2_XXS, 300 GB, NM790 SSD, 4 sequences): a token is 48.5-53.4 ms, ~20
  ms of it SSD waits (8-9 reads of 8-12 MB, ~2.1 ms each - near the SSD's speed). ~57% of the experts fit VRAM + RAM;
  a cache study on the routes (LRU, decayed LFU, frequency admission) gave 6.6-6.8 reads a token, the clairvoyant
  bound 3.1, so the eviction policy has little left. rocprofv3: the dense GEMVs run at ~88-90% of the card's memory
  bandwidth (~11 ms a token), the rest is the experts (~5 ms) and small kernels - no large lossless lever left.
- The speed setting: `STRATA_GLM_TAIL_SKIP=4 STRATA_GLM_TAIL_SKIP_DISK=1` 39.8 ms a token (25.1 tok/s, +34%) for KL
  +0.007; `=3` 38.6 ms for +0.020. `--bench`'s "ssd" verdict now names it (docs/BENCH, MANUAL, MODELS).

## v1.3.68 - 2026-10-27

**AMD (HIP): long contexts no longer crawl - 1.5 GB of VRAM kept free beside the expert pool (was 0.7).**

- The first prompt's hipBLASLt GEMMs load their kernels (~0.7 GB of VRAM) after the expert pool has been carved. With
  0.7 GB kept, an AI PRO R9700 at `--max-context 131072` (the full-VRAM config) had ~0.6 GB left before the prompt:
  every prompt GEMM failed to load its kernel and fell back, and an 8K-token prompt took 1,300 s (5 tok/s). At the
  configs' 65,536 it fitted with ~80 MB to spare. HIP now keeps 1.5 GB (`STRATA_GLM_RESERVE_MB`, CUDA unchanged at
  700 MB): the Ornith pool 83 -> 80 slots a layer, decode ~3% slower (UD-Q4_K_XL `--bench` 59.1 / 60.2 -> 56.7 / 58.6
  tok/s), the prompts unchanged. Not checked on CUDA at 131072 yet.
- Long contexts checked on the four models (R9700, the full-VRAM configs at `--max-context 131072`): prompts of
  ~8K / 32K / 64K / 115K tokens with a passphrase on the first line, then asked for it - every model found it at every
  length. Prompt tok/s / decode tok/s after it:
  - Ornith-1.5-122B UD-Q4_K_XL: 966 / 1,478 / 1,119 / 781, decode 49.2 / 48.9 / 47.6 / 38.3
  - DeepSeek V4.1 Flash 1017 UD-IQ2_XXS: 206 (cold disk) / 853 / 1,038 / 1,071, decode 20.0 / 19.5 / 20.8 / 19.7
    (SSD-bound)
  - DeepSeek V4 Flash 0731 UD-Q2_K_XL: 820 / 952 / 872 / 721, decode 45.8 / 44.5 / 42.3 / 39.6
  - GLM-5.3-Flash Maya-S-v2-IQ2_XXS: 597 / 912 / 843 / 810, decode 37.0 / 36.6 / 35.9 / 34.8

## v1.3.67 - 2026-10-27

**Ornith (Qwen3.5-MoE) prompts: the attention on tensor cores - a 115K-token prompt 685 -> 154 s.**

- The prompt path's attention was the plain FP32 kernel: each block of 4 queries walked every key with a shuffle sum a
  key, ~90% of a 115K-token prompt. It now runs on WMMA (rocWMMA on AMD gfx11 / gfx12, nvcuda::wmma on NVIDIA; one
  source): a block takes 8 tokens, a warp one token's 16 heads of a KV group as a 16-row tile, so a K / V tile in shared
  memory serves 128 rows; Q K^T and P V in FP16 with FP32 sums; two passes over the keys (each row's max and sum, then
  P V with P normalized), so the accumulators are never rescaled.
- AI PRO R9700, a 1,024-token sub-batch, a layer: at 32K / 64K / 114K positions 222 / 544 / 918 -> 38 / 85 / 156 ms.
  IQ3_XXS-B on 16 GB of VRAM with no lane: 57.6K / 115K-token prompts 324 / 168 -> 1,076 / 747 tok/s (time to the first
  token 178 / 685 -> 54 / 154 s); decode after them 40.7 / 32.1 tok/s. UD-Q4_K_XL `--bench`: 8K 948 -> 1,348 tok/s, 2K
  695 -> 728, decode 59.1 (unchanged), quality check OK (KL 0.0049).
- Quality: 255 tokens scored after a 24K-token context read through the prompt path, KL 0.0054 against the FP32 kernel
  (same top token 98.8%); two runs of the FP32 kernel differ by 0.0064. The attention output is within 0.08% of each
  row's scale of FP32 (tests/hip/q35_attn_window.cpp). `STRATA_Q35_PATTN=1` goes back to the FP32 kernel.
- Not yet run on an RTX 5090 (the CUDA build compiles for sm_120a and sm_86).

## v1.3.66 - 2026-10-26

**Docs: when the conversation is reused and how to see it (issue #14).**

- docs/MANUAL: the same prompt again reads only its last token (V4.1: 5K and 23K tokens, 0.04 s to the first token,
  no size cap); only the last conversation is kept (another request in between or a restart replaces it); the tokens
  must match from the start; `cached_tokens` / `cache_read_input_tokens` / `timings.cache_n` say whether it was reused.

## v1.3.65 - 2026-10-26

**Ornith (Qwen3.5-MoE): long-context decode ~40% faster at 115K tokens, and the conversation is reused.**

- The decode and verify-window attention runs on a new split kernel: a block takes a chunk of keys for one KV head
  and its rows' heads, a lane's 8 partial dots of 8 keys are summed in one shuffle tree (9 shuffles for 8 keys, not
  5 a key), the softmax runs once a 32-key tile, and the tile's K / V go through shared memory once for the block's 8
  warps. AI PRO R9700, a layer at 115K positions: a token 1.09 -> 0.57 ms, a 3-row window 3.29 -> 1.39 (12 attention
  layers a token); at short contexts a window 0.15 -> 0.02 ms. IQ3_XXS-B on 16 GB of VRAM with no lane, decode after a
  57.6K / 115K-token prompt: 30.5-31.5 / 23.7-24.6 -> 37.9 / 34.0 tok/s (prompts unchanged: their attention is the
  prompt kernel). Short answers unchanged (UD-Q4_K_XL `--bench` 57.1 -> 58.3 / 58.8 tok/s); the quality check KL
  0.0049 (a reorder of the sums alone gives 0.0060: the activations' q8_1 rounding turns any reorder into this).
  `STRATA_Q35_ATTN=1` goes back to the old kernels.
- Conversation reuse: a request that repeats or continues the last one restores the delta-net layers' states and conv
  rows and the NextN drafter's pending row (kept in pinned RAM, ~150 MB, not VRAM) instead of reading the prompt again.
  A ~5.5K-token prompt asked again: 8.1 -> 1.4 s; its next turn 8.0 -> 1.8 s. Conversation slots on disk stay off
  for this model.
- Not yet run on an RTX 5090 (the CUDA build compiles).

## v1.3.64 - 2026-10-26

**AMD runs: the dashboard, the report and the stall watchdog read the AMD card, not an NVIDIA card beside it (issue #14).**

- The server's hardware readings (the Monitor tab, `/metrics`, the dashboard's Copy report) came from NVIDIA's NVML
  only, so a HIP run on a PC with an NVIDIA display card read that card: the report named an RTX 3060 beside an AI
  PRO R9700, and the stall watchdog's "is the GPU busy" sign read the 3060's load. A HIP run now reads its own cards
  from the amdgpu driver's sysfs files (load, VRAM, temperature, power and its cap, PCIe link; no PCIe throughput),
  found by the config's `gpu_pci` / HIP numbers, and never opens NVML. The name is the driver's product_name, else
  "AMD Radeon (<arch>)". CUDA runs are unchanged.
- Windows with HIP has no sysfs: the GPU readings there show as not readable instead of an NVIDIA card's.

## v1.3.63 - 2026-10-26

**DeepSeek V4.1 long prompts: the indexer on WMMA and the candidate blocks batched; a score-buffer overrun fixed.**

- DeepSeek V4.1 prompts, AMD (gfx11 / gfx12): the indexer's 32 heads score on the WMMA kernel deepseek4's 64 used
  (40K prompt: 3.4 -> 1.9 s of GPU time). `STRATA_DS4_IDX=1` goes back to the plain kernel.
- DeepSeek V4.1 prompts: the candidate blocks of the rows that see more than 16K compressed rows (layer 20 on) take a
  launch per group of rows instead of three or four per row (a 60K prompt has ~44K such rows a layer); their scratch
  comes out of the score buffer. `STRATA_GLM_V41_CAND_ROW=1` goes back to a row at a time.
- AI PRO R9700 + iGPU, Maya-XS prompt tok/s at 32K / 60K: 1,093 / 1,028 -> 1,119 / 1,214 (Project Maya v1.1.28: 948 /
  1,235). UD-IQ2_XXS 60K stays ~880: 25 s of its 67 s are SSD waits, which faster compute does not shorten. One 65,536-
  token chunk instead of two of 32,768 is slower (it borrows 15.6 GB of the expert pool: 60K 764). KL of 255 tokens
  after 40K against the token-at-a-time attention: 0.0043 (0.0055 / 0.0059 with either change off - the run-to-run
  band); DeepSeek V4's prompt path 0.0032 against its token path.
- Fix (from v1.3.60): a V4.1 sub-batch of 1-3 tokens past a quarter of the context wrote its indexer scores past the
  score buffer (a ratio-1 row sees the whole context, the buffer held a quarter of it a token); the buffer now holds
  at least 6 rows. It landed in unused prompt-path memory here; a 40,961-token prompt now runs clean.
- tools/test_maya_hip.py / test_maya_models.py: under a system Python `main()` re-ran itself through ./abyss.sh with
  the test's arguments - a real `--bench` that overwrote abyss-bench.txt. The tests now set MAYA_SH.
- Not yet run on an RTX 5090 (the CUDA build compiles; the CUDA side changes only the candidate batching and the
  buffer size).

## v1.3.62 - 2026-10-26

**`--bench --deep`: the share sweep no longer calls a share faster because the disk warmed up during the sweep.**

- Every run of the share sweep writes the same answer, so the first run read the SSD for that answer's experts and the
  later ones did not: on an AM5 desktop with an R9700 on Gen4 x16 and 64 GB (issue #19) the engine's own share ran
  24.8 tokens/s first (6.8 SSD reads a token) and 44.8 last (0.06), and the sweep compared share 0 (42.6) with the
  mean of the two (34.8) and called it the winner. Now an untimed run brings the answer's experts in first, a share
  has to beat the faster of the engine's own two runs, both runs are shown, and when the first still read the SSD much
  more than the last the sweep says the early shares were slowed by the disk. The header also says the sweep's numbers
  run above the decode answers (new text). docs/BENCH(.ja) updated.

## v1.3.61 - 2026-10-26

**Docs: what a larger RAM tier does on a DDR4 desktop.**

- docs/BENCH and the `STRATA_GLM_RAM_GB` row in docs/MANUAL: a larger RAM tier moves expert reads off the SSD, but
  the CPU lane runs those experts, so part of the gain moves to the lane's time. A user's DDR4 desktop (123 GB, Gen5
  x16, no iGPU, DeepSeek V4.1), 80 -> 88 GB: SSD waits 11.1 -> 8.7 ms a token, the lane 9.4 -> 10.6 ms, decode 18.0 ->
  18.6 tokens/s. No code changes.

## v1.3.60 - 2026-10-26

**DeepSeek V4.1 long prompts another 2.5-3.2x faster: the rest of the attention and Engram batched.**

- DeepSeek V4.1 prompts: what v1.3.59 left a token at a time now runs over each 1,024-token sub-batch - the window
  keys (the token path's prep over the batch), the KV source's compressed rows and index keys (a pair's first token
  from the batch or, across sub-batches, the pend ring), q's RoPE, the indexer (scores of every row's visible
  compressed rows in row groups the score buffer holds, the candidate blocks / mask a row at a time past 16K, each
  row's top 512 into its own selection slot) and the attention (deepseek4's batched kernels, taught V4.1's ratios 1 /
  2: every visible row while they fit the top 512, else the row's selection). 32K: the attention 62 -> ~10 s.
- DeepSeek V4.1 prompts: Engram's engram_kv as one FP16 GEMM over each 256-token block instead of a GEMV a token (it
  read its ~160 MB again for every token; 32K: 20 -> 3.9 s).
- AI PRO R9700 + iGPU, prompt tok/s at 2K / 8K / 16K / 32K / 60K: UD-IQ2_XXS 127 / 237 / 292 / 314 / 277 ->
  140 / 363 / 596 / 781 / 896; Maya-XS 209 / 192 / 334 / 327 / 288 -> 478 / 964 / 1,055 / 1,093 / 1,028 (Project
  Maya v1.1.28 on the same file, same client: 159 / 506 / 694 / 948 / 1,235). Decode unchanged. KL of 255 tokens
  after 8K / 20K / 40K contexts against the token-at-a-time attention: 0.0039 / 0.0040 / 0.0038 (the run-to-run floor
  is 0.0032); DeepSeek V4's prompt path, which shares the attention kernels, 0.0030 against its token path.
  `STRATA_GLM_V41_ATTN_TOK=1` / `STRATA_GLM_V41_ENG_GEMM=0` go back to a token at a time.
- Not yet run on an RTX 5090 (the CUDA build compiles; v1.3.58 and v1.3.59 are in the same state).

## v1.3.59 - 2026-10-26

**DeepSeek V4.1 prompts another 2.2-2.7x faster: the attention's projections batched.**

- DeepSeek V4.1 prompts: the attention ran a token at a time, and each token read the layer's ~136 MB of attention
  weights again (~0.24 of its 0.31 ms a layer on an AI PRO R9700; the attention was 9/10 of a 16K prompt). Its
  projections - q_a, kv, the compressor, the indexer's weights, q_b, the grouped output LoRA and wo_b - now run as
  FP16 GEMMs over each 1,024-token sub-batch; what has an order (the window ring, the KV source's compressed row, q's
  RoPE, the indexer with the row's own selection, the attention) stays a token at a time on the token path's kernels.
  The layer the prompt skip cuts at writes only its ring and compressed rows. AI PRO R9700 + iGPU, UD-IQ2_XXS, prompt
  tok/s at 8K / 16K / 32K / 60K: 86 / 101 / 101 / 101 -> 185 / 242 / 273 / 262 (v1.3.57: 66 / 68 / 64 / 57). KL of
  255 tokens after an 8K context against the token-at-a-time attention: 0.0034 (0.0039 without the skip), the same as
  that attention against itself run twice (0.0032). `STRATA_GLM_V41_BATCH=0` goes back to a token at a time; a pack
  with an attention weight type that has no FP16 decode does so by itself, and the engine log says which.
- The serve watchdog: a V4.1 prompt layer beats every 256 rows too. Under heavy memory pressure (a 200 GB copy to a
  network share) one layer of the token-at-a-time attention took 210 s against 5, and the per-layer heartbeat alone
  read that as stuck after 180 s. A big copy can still hold the engine up for minutes at one place (a single
  wait of 191-238 s here); docs/MANUAL now lists `STRATA_WATCHDOG_S` (default 180) for that.

## v1.3.58 - 2026-10-26

**DeepSeek V4.1 long prompts read 1.5-1.8x faster and every prompt row now uses its own selection; a late PCIe share
for issue #14.**

- DeepSeek V4.1 prompts (fix): in a prompt chunk, the layers that read another layer's top-k selection read the one
  chosen for the sub-batch's last row, not their own row's. Each row now keeps its own selection and candidate mask
  across the layers (~16 KB a row of prompt arena). Greedy answers after 16K / 32K prompts now follow the
  token-by-token path's (`--prefill 0`) for their first 151 characters and stay on its content; before the fix they
  left it at the 13th / 29th, in the heading.
- DeepSeek V4.1 prompts (speed): layers 20-39 read the earlier positions only through layer 20's compressed KV and
  index keys, and each looks back one window more than the one before, so only the prompt's last 2,668 rows need them.
  The rows before stop at layer 20's attention (exact up to the experts' summing order). AI PRO R9700 + iGPU,
  UD-IQ2_XXS, prompt tok/s at 8K / 16K / 32K / 60K: 66 / 68 / 64 / 57 -> 86 / 101 / 101 / 101.
  `STRATA_GLM_PROMPT_SKIP=0` turns it off; one device only (not a split); the DSpark drafter's context ring reads the
  prompt's last rows, which still run every layer.
- `--bench --deep` (issue #14, thanks to its reporter): the sweep's lower shares run on a GPU pool the engine's own
  share has already filled, so they can look faster than they are from a start. The sweep and docs/BENCH now say so,
  and the advice is to keep a share only if a plain `--bench` gets faster.
- New `STRATA_GLM_PCIE_SHARE_LATE=<share>[:<steps>]`: each request starts on the engine's share and moves to this one
  after `<steps>` token steps (default 256), so the start of an answer refills the pool first. Off by default; a
  request's own share overrides it.

## v1.3.57 - 2026-10-26

**Ornith's NextN drafter runs on GGUFs whose token_embd the GPU cannot read, two contributed fixes merged.**

- Qwen3.5 / Ornith (issue #18, thanks to its reporter): the NextN drafter reads its token's embedding from the pinned
  token_embd on the GPU, which only Q8_0 / Q5_K / Q6_K tables give; a GGUF with another type (IQ3_S in the Ornith
  IQ3_XXS-B file, llama-quantize's pick for an IQ3_XXS target) or `STRATA_GLM_ZERO_COPY=0` left the drafter off
  without a word, and decode fell back to the n-gram drafter. Such a table now gets a Q8_0 copy for the drafter only
  (773 MB pinned on Ornith); the trunk keeps its exact row. The engine log says which path it took. Ornith IQ3_XXS-B on
  a 16 GB GPU + 64 GB RAM setup: drafts accepted 39% -> 71%, decode 39.5 -> 42.6 tok/s. No file needs downloading again.
- DeepSeek V4.1 tool calls (PR #16, thanks to its author): V4.1 Flash 1017 writes `<｜DSML｜ calls>` (a space after the
  token); the server read only V4's `<｜DSML｜tool_calls>`, so agents stopped at the first call. Both spellings are read,
  whole and streamed, and a V4.1 pack whose chat template spells V4's tags (ISTA-DASLab's GSQ-RCO GGUFs) shows the
  model V4.1's form. Templates that already spell V4.1 (unsloth, orcarouter, Maya-XS) render byte-identically; V4
  packs are untouched.
- RX 7900 XT / XTX (PR #17, thanks to its author): gfx1100 takes the RDNA expert kernels for IQ2_XS 4096x2048 and V4.1's
  5120x2304 gate/up and down (1.23-1.41x per kernel, bit-identical) and a warp a row for Q6_K dense GEMV. Other GPUs
  are unchanged.

## v1.3.56 - 2026-10-25

**Two contributed CPU / Windows speed-ups merged, and `--bench` fixed where reports from 16 GB cards showed it
misleading.**

- CPU lane (PR #13, thanks to its author): AVX-VNNI one-token kernels for IQ2_XXS / IQ3_XXS / IQ3_S experts, picked at
  run time by CPUID on CPUs that have it (Intel 12th gen and later, AMD Zen 5) and built only by compilers that can
  target it (GCC 11+, Clang 12+, MSVC 2022); the one-token loops keep their unroll. Bit-identical with VNNI on, off
  (`STRATA_NO_AVXVNNI=1`) and forced AVX2; rows 1-5% faster than AVX2 where it applies.
- Windows (PR #12, thanks to its author): with `STRATA_GLM_CPU_PIN` set, the pinned CPU-lane workers turn off EcoQoS
  power throttling and run above normal priority - Windows 11 parked them on efficiency settings (reported 18.8 ->
  36.5 tok/s on an i7-14700K). Nothing changes without `STRATA_GLM_CPU_PIN`.
- `--bench --deep` (issues #14, #15): the PCIe / CPU-lane share sweep ran last, on tiers the prompts had warmed and
  128-token answers, and was compared with the 256-token decode above it - so a warm-tier gain read as "share X beats
  the engine's own choice" (1.5-1.7x on SSD-bound models). The engine's own share now runs before and after the five
  shares under the same conditions, the report names its share, and it warns when those two runs differ by more than
  10% (the tiers still warming). On AMD it points to `STRATA_GLM_PCIE_SHARE` in the config's `"env"` - `--calibrate`
  is CUDA-only - instead of `--calibrate`.
- Quality check: DeepSeek V4 0731 UD-Q2_K_XL was "skipped" for everyone with the stock unsloth files - the reference
  had been made from our attention-Q5_K overlay of them. Both now have a reference (`bench/ref/<model>.json` for the
  stock files, `<model>.attn-q5k.json` for the overlay); the check takes the one whose GGUF size matches, and
  `tools/bench_ref.py --variant <tag>` writes another copy's.
- Quality check: "slightly off" (KL 0.02-0.05) now says it is expected on a smaller GPU - a 16 GB card leaves many
  more experts to the CPU lane, which rounds differently from the GPU the reference ran on (GLM-5.3 Maya-S-v2 limited
  to 16 GB here: KL 0.000 -> 0.025; V4.1 UD-IQ2_XXS 0.008 -> 0.012). Only OFF asks for a report.
- docs/BENCH(.ja).md: the sweep's new runs, the AMD advice, the small-GPU KL, and why a 2k prompt can read slower per
  token than an 8k one on a model that does not fit GPU + RAM (each prompt reads the disk-only experts once, whatever
  its length).

## v1.3.55 - 2026-10-25

**`./abyss.sh --bench` measures where a token's time goes on your PC, names the likely bottleneck, and checks the
PC computes what a known-good run did - one file to paste into an issue.**

- `--bench` (rebuilt): the model's own chat template (DeepSeek V4 / V4.1, Ornith and GLM alike - it was GLM's for all),
  then from the engine's per-request counters: SSD waits, the CPU / iGPU lane's busy time, the experts GPU memory holds
  and those copied over PCIe, the drafter's acceptance. A verdict - ssd / cpu / pcie / gpu - with what moves it
  (docs/BENCH.md's table). On our R9700: Ornith-1.5-122B "gpu" (56.4 tok/s, SSD waits 0%), DeepSeek V4.1 UD-IQ2_XXS
  (300 GB) "ssd" (20.1 tok/s, 39% of a token waits on the SSD).
- The PCIe link is read while the engine works (links drop to Gen1 x4 when idle): AMD on Linux from sysfs, the
  narrowest link up to the root port (an AMD card's own switch shows x16 - our OCuLink R9700 is Gen4 x4); NVIDIA on
  Linux and Windows from nvidia-smi. The model SSD's read speed: 2 GB of its GGUF past the page cache (Linux
  `posix_fadvise`, Windows unbuffered).
- A quality check: the engine scores 512 tokens of known text against the top-32 log-probs the same model gave on a
  known-good run (`bench/ref`, made by `tools/bench_ref.py`): OK up to KL 0.02, OFF above 0.05 - a kernel broken on
  that GPU shows there (issue #9 style). References for Ornith-1.5-122B UD-Q4_K_XL, DeepSeek V4.1 UD-IQ2_XXS, V4
  UD-Q2_K_XL and GLM-5.3-Flash Maya-S-v2 IQ2_XXS (128 KB each); the same PC gives 0-0.01. Settings that change the
  numbers on purpose (tail skip, the abliteration edit, deferral) are left out of the scoring run, and another quant or
  an overlay pack is skipped rather than flagged.
- `--bench --deep` finds the bottleneck by experiment: the GPU timeline (`STRATA_GLM_GAP_PROF=1`) and five answers at
  PCIe / lane shares 0 to 1 - whether `--calibrate` would gain, or which path is the narrow one (Ornith on Gen4 x4: all
  over PCIe 4.8x slower than all on the lane). `--bench --full`: three runs of each question and a 32k prompt.
- The result is `abyss-bench.txt`: a one-line summary for docs/MODELS.md's tables, then the details. No host name,
  user name or folder path. A "Speed report" issue template takes it.
- `--bench` and `--report` find setup-deepseek.sh's `abyss-*.json` configs too (they looked for `maya-*.json` only);
  `--report` writes `abyss-report.txt`. The start line says Abyss, not Project Maya.
- Docs: docs/BENCH.md / BENCH.ja.md (how to run it, what each line means, the verdict table, platforms); MANUAL,
  README and CONTRIBUTING point to it.

## v1.3.54 - 2026-10-25

**Bug fixes: the HIP iGPU lane on a PC with no iGPU (issue #11), ROCm 7.2.1 with the lane, RDNA3 grid padding (issue
#9); a docs correction.**

- HIP: the iGPU lane stays off unless an integrated GPU other than the engine's own card is visible. With one discrete
  Radeon (an RX 9070 XT, the APU's iGPU off in the BIOS) the lane library fell back to device 0 - the engine's own
  card - and its calls ran 27-62 ms late; the user measured `STRATA_GLM_IGPU=0` +30-50% decode (issue #11). The CPU
  lane computes instead; with `STRATA_GLM_IGPU=1` the log says why the lane is off.
- ROCm 7.2.1: with the lane on, every request ended "kernel launch(es) failed". The lane's failed `hipHostRegister`
  left its error as the thread's last one, the engine's next launch check counted it, and that count is never reset.
  The lane and the engine's lane-off path now clear the error. The lane also takes ROCm 7.2's "invalid argument" on a
  range the engine already pinned as "already registered" when the range is mapped (not checkable here: no 7.2.1).
  ROCm 7.2.1 runs with `STRATA_GLM_IGPU=0` (Ornith 32-35 tok/s against ~39 on ROCm 10, the user's report).
- HIP grid padding: off by default on gfx11 (issue #9, RX 7900 XT: no TheRock#9300 stall there, the padding within
  noise); `STRATA_GLM_GRID_PAD=1` forces it on.
- DeepSeek V4.1: a shard holding only the Engram table may be left undownloaded when `--engram` names another GGUF with
  the same table (iq_pack.py skips it, the engine skips a missing shard when engram.txt is there): orcarouter's
  abliterated Mix-256G shard 00002 is unsloth's IQ4_NL table byte for byte, 110.6 GB not downloaded twice.
- Docs: v1.3.53 called ISTA's IQ2_XXS the best quality per GB; it compared ISTA's size without the Engram table
  against unsloth's with it. Without the table UD-IQ1_M is 139.5 GB at KL 0.27 against IQ2_XXS's 138 GB at 0.31.
  README corrected. docs/MODELS: issue #11's rows (V4.1 ISTA IQ2_XXS and Ornith on a 16 GB RX 9070 XT, lane on / off,
  ROCm 7.2.1), the #8 rows marked as measured with the lane bug, and the first `--gpu-tune` report (issue #10).

## v1.3.53 - 2026-10-24

**DeepSeek V4.1: ISTA-DASLab's GSQ-RCO GGUFs, the Lucebox DSpark drafter (opt-in), the abliteration edit at run time;
GLM-5.3-Flash: kindlingai's exlr8 K2.**

- DeepSeek V4.1 Flash 1017: ISTA-DASLab's [GSQ-RCO GGUFs](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
  load (IQ2_XXS 138 GB, IQ1_M 121 GB): the Engram hash values as tensors, the Engram table in a separate GGUF
  (`--engram`, or `deepseek41.engram.file`), the 3-D output LoRA. Natively served tensors now upload every row past
  the first dimension (ISTA's [4096, 1024, 8] `attn_output_a` uploaded one group in eight: KL 11). AI PRO R9700 +
  8060S lane, RAM tier 60 GB, 8 held-out sequences: IQ2_XXS KL 0.3137 (unsloth UD-IQ1_S 0.3557), IQ1_M 0.3845.
  swe-bench 8-task subset: IQ2_XXS 30.7 / 30.9 tok/s; with `STRATA_GLM_TAIL_SKIP=3 STRATA_GLM_TAIL_SKIP_DISK=1`
  (lossy, KL 0.3266) 42.2 / 41.3; with the skip and the drafter below 48.3 / 47.3.
- Experts the MMQ prompt kernels do not read (IQ1_M - ISTA's experts) are decoded to FP16 once each and multiplied by
  cuBLAS / hipBLAS, instead of the whole prompt running token by token: ISTA IQ2_XXS 8K prompt 281 -> 109 s (~75
  tok/s; UD-IQ1_S 103 s). The 64 generated tokens match the token path's.
- DeepSeek V4.1: the DSpark drafter (Lucebox's `deepseek41-dflash-draft`, DeepSeek-V4.1-Flash-1017-DSpark-draft-
  Q2K-Q4K.gguf, 4.9 GB), opt-in: `STRATA_DS4_DSPARK=<file> STRATA_DS4_DSPARK_CHAIN=0.6`. swe-bench 8 tasks on UD-IQ1_S
  (R9700 + lane, Lexar NM790 SSD): 40.4 -> 46.7 tok/s over 4 runs (+16%); with the disk tail skip (lossy, KL 0.3573)
  with `STRATA_GLM_ROUTE_AFFINITY_RAM=0.08` 43.7 -> 54.4. Drafts are verified by the model, so the drafter does not change what is sampled. On a slow SSD
  (~3.4 GB/s) it measured no gain - the window's columns wait on the disk.
- `STRATA_DS4_SPEC_COST=1` (opt-in, DeepSeek V4 / V4.1 with DSpark): sizes the verify window by expected kept tokens a
  millisecond. Measured within noise of the fixed chain (46.2 vs 47.5 tok/s), so it stays off.
- `STRATA_GLM_ABL=<file>`: a refusal-direction edit applied at run time (x - sQQ^Tx on the attention and shared-expert
  outputs, no re-quantization, no overlay shard). `tools/abl_export.py` turns mrfakename's
  `abliteration-edit.npz` into the file. V4.1 UD-IQ1_S: KL 0.3557 -> 0.3577 (the v1.3.52 overlay: 0.3563), 16
  refusal-bait prompts refused 12 -> 0.
- GLM-5.3-Flash: kindlingai's [exlr8 v2](https://huggingface.co/kindlingai/glm-5.3-flash-exlr8-k2-k3-k4) checkpoints.
  `tools/exlr8_convert.py --model <dir> --width 2` rewrites the halves as an exllamav3-style EXL3 directory, then
  `tools/exl3_pack.py --model <out>` packs it as for turboderp's EXL3. The EXL3 kernels (device, iGPU lane, CPU lane
  AVX-512 / AVX2 / scalar) take a third codebook: mcg rounded to FP8 E4M3, which exlr8 requires. The FP8 dense matrices
  are BF16 / Q8_0 rows; the NextN block's projections Q8_0. Checked against kindlingai's reference loader (experts,
  shared expert, dense trellis within 7e-8). K2 (86 GB) on the R9700 + 8060S lane: nll 1.61 on glm_ids4k (turboderp
  EXL3 sc_2.10bpw 1.598, Maya-S-v2 IQ2_XXS 1.634; lower is better), ~23 tok/s; K3 (125 GB) nll 1.548 - the best GLM
  quant measured here - at ~11 tok/s, a third of its experts read from the SSD. K4 not tried.
- Fixed: an EXL3 expert read from the SSD was wrong (turboderp's EXL3 too, once its experts outgrow VRAM + RAM). The
  streamed fetch copies each chunk's byte range of a part once its read lands, but the EXL3 reads chunked the trellis
  alone and brought the scales with the first chunk. Reads now cover the kernel's ranges: exlr8 K2 on a 30 GB RAM tier
  KL 1.73 -> 0.206 (as with `STRATA_GLM_STREAM_FETCH=0`), 80.2 -> 75.5 ms a token against that.

## v1.3.52 - 2026-10-24

**DeepSeek V4.1: the token's embedding row read on the GPU, and overlays - changed dense tensors (Q5_K, or a
published abliteration edit) without another copy of the model.**

- unsloth's V4.1 UD-IQ1_S stores `token_embd` as Q6_K, which the GPU row gather did not read, so every token
  dequantized its row on the host and copied it in while the GPU waited. Q6_K rows are now gathered on the GPU from
  the pinned table (518 MB pinned), as Q8_0 / Q5_K were. Bit-identical (8 sequences: KL 0.3557 as before); the GPU's
  idle time between two tokens 0.99 -> 0.55 ms (`STRATA_GLM_GAP_PROF`), the host's embedding step 0.39 -> 0.24 ms.
- `tools/dense_overlay.py` writes only the changed tensors into a small first shard and links the model's shards
  behind it (`iq_pack.py --first-wins` takes the changed copies). On UD-IQ1_S (AI PRO R9700 + 8060S lane):
  - Q5_K for the layers' Q6_K projections (opt-in, lossy): a 3.2 GB overlay, GPU time a token -1.4 ms, swe-bench
    subset 40.1 -> 43.0 tok/s, KL 0.3557 -> 0.3652.
  - `--abl`: mrfakename's DeepSeek-V4.1-Flash-1017-abliterated edit (`abliteration-edit.npz`) on `attn_output_b` /
    `ffn_down_shexp`: a 1.4 GB overlay, KL 0.3563, 16 refusal-bait prompts refused 14 -> 0.
  Recipes and numbers: docs/deepseek41/README.md, "Overlays".

## v1.3.51 - 2026-10-24

**`./abyss.sh --gpu-tune`: the engine's per-GPU kernel choices, measured on this PC's GPU(s) without a model.**

- The dense GEMVs pick their kernel shape from a per-GPU table (`mv_tune`, read from the GPU's architecture at start;
  the log prints `glm fast: dense GEMV tune for device N (...)`). First entry: Q6_K a warp a row on gfx12 (AI PRO
  R9700), where 4 warps a row left the bandwidth unused - rocprof, DeepSeek V4.1 UD-IQ1_S decode: Q6_K 8.78 -> 7.45 ms a
  token (562 GB/s); 8 sequences alternated with v1.3.50: 33.2 / 34.3 -> 32.1 / 33.9 ms a token, KL 0.3533 -> 0.3557
  (summation order only). Other GPUs as before. `STRATA_GLM_MV_Q6K_SPLIT=1/0` overrides.
- `./abyss.sh --gpu-tune` (a new `abyss_gpu_tune` tool, built with the engine) times those choices on each GPU of the
  installed configs, under a minute a GPU and without a model: the device, a device-to-device copy's bandwidth, and
  the Q6_K split over the row counts the models have (the weights cycled through 1 GB so a large cache - the R9700's
  64 MB Infinity Cache - does not fake the bandwidth). Only a 2% win changes a choice; it is written into the
  configs' `env`. The report (`abyss-gpu-tune.txt`) is meant for a GitHub issue (new template "GPU tune report"), so
  the table grows for GPUs we do not have. `setup-deepseek.sh` / `setup.sh` run it after writing the config (never
  fatal).

## v1.3.50 - 2026-10-23

**DeepSeek V4.1 starts without `STRATA_GLM_RAM_GB` again: the RAM tier pins at most 73% of the RAM by default.**

- With no `STRATA_GLM_RAM_GB`, the RAM tier took MemAvailable less 6 GB. DeepSeek V4.1's experts outgrow the RAM, so on
  a 94 GB machine it pinned 82 GB; the disk staging pinned after it then did not allocate (`glm fast: pinned disk
  staging did not allocate`) and the engine did not start - the R9700 V4.1 config as `setup-deepseek.sh` writes it.
  The tier now also leaves room for that staging, and on Linux with a discrete GPU pins at most 73% of the RAM
  (`STRATA_GLM_RAM_FRAC`; on that machine 67-68 GB pinned ran for days, 73.9 GB thrashed). Measured (AI PRO R9700 +
  8060S lane, 94 GB): V4.1 UD-IQ1_S 67.5 GB, starts, 8K prompt and decode as with `STRATA_GLM_RAM_GB=60`; DeepSeek V4
  UD-Q2_K_XL 67.37 -> 67.07 GB, decode 52-56 tok/s as before. `STRATA_GLM_RAM_GB` still sets the size outright.
- DeepSeek V4.1's batched verify windows run their columns' VRAM-resident experts in one launch
  (`STRATA_DS4_WIN_COLS`, as V4's since v1.3.43), with a scratch set per column (up to 6 columns, 8 slots each - ~350 MB
  of VRAM on UD-IQ1_S, taken from the expert pool) so the columns' SSD reads stay apart. Bit-identical
  (`STRATA_DS4_WIN_COLS_CHECK=1`: 260 columns, 0 differ); speed within noise on the R9700, where a V4.1 window column
  is bound by the SSD's bandwidth (~9.5 of its ~20 ms).
- `tools/iq_pack.py` and the loader read Project Maya's V4.1 GGUFs (Maya-S / Maya-XS): `deepseek41.engram.layers`,
  `head_dim` and `pad_token_id` as aliases of the Engram keys.

## v1.3.49 - 2026-10-23

**Ornith's NextN drafter works after a prompt again: 19K tokens in, 40 -> 51 tok/s.**

- The batched prompt path ran the GLM / DeepSeek hyper-connections' last write (`hc_update`) on Ornith too, whose plain
  residual has none, whenever the drafter was loaded. It broke the residual rows the NextN block's caches are built
  from, so after any prompt of 32 tokens or more the drafter attended over garbage, none of its drafts reached the 0.5
  threshold (`drafts kept 0..5: 7460 0 0 0 0 0` in long swe-bench answers) and every token paid its ~2.2 ms for
  nothing. The update is now skipped for Ornith; the trunk's answers were never affected (it reads its own caches).
- Measured (UD-Q4_K_XL, R9700 + 8060S lane, greedy): a 18,955-token prompt then 2,048 tokens 40.3 -> 51.0 tok/s, 1,459
  drafts 78% accepted (v1.3.48: 199, all from n-gram repeats); without drafts 43.3. Short prompts (the token path)
  unchanged: 512-token answers prose 51.7 / 53.1, code 61.7 / 59.3.

## v1.3.48 - 2026-10-23

**Ornith's decode at long contexts: the verify window's attention split into chunks - 19K tokens in, 27 -> 40 tok/s
with the NextN drafter; short answers unchanged.**

- The verify window (a token and its drafts) ran its 12 full-attention layers through the prompt path's kernel, which
  gives a block to every 4 rows and KV group: a window of 2-8 rows got 1-4 blocks, each walking every cached position
  in turn - 25.6 ms a layer at 20K positions (4 rows, AI PRO R9700), and decode fell to ~8 tok/s in long swe-bench
  answers. `q35::attn_window` keeps that block (each key read once for 4 rows x 16 heads) over chunks of 128 positions
  and merges the chunks' partials: 0.69 ms a layer at 20K (37x), 0.33 at 8K, 0.18 at 2K.
- Measured (UD-Q4_K_XL, R9700 + 8060S lane, greedy): a 18,955-token prompt then 2,048 tokens with the NextN drafter
  27.8 / 26.8 -> 40.3 tok/s (without drafts 43.3, n-gram drafts only 44.5: at long contexts the drafter no longer
  pays - next); 512-token answers prose 50.2 / 49.4 -> 51.9 / 52.8, code 54.1 / 56.6 -> 55.5 / 56.0; the drafts' accept
  rate unchanged (71-76%).
- The same sums as before up to their order (the window's attention against attn_prompt: 1.1e-4 relative at most,
  windows of 1-8 rows at positions 0-20,000; `tests/hip/q35_attn_window.cpp`). Greedy answers part from the old
  build's within the first few dozen tokens, as two runs of the same build do.
- The window's scratch: 135 MB of VRAM at a 65,536-token context.

## v1.3.47 - 2026-10-23

**`/props` names the model of a GLM, DeepSeek V4.1 or Ornith config; Ornith's docs use port 8082 like the other
configs.**

- The server's `/props` reported `model_path: "pack/full"` for every config that names its model with `--glm-pack`
  (GLM-5.3-Flash, DeepSeek V4.1, Ornith-1.5-122B): it read only `--native` and `--pack`. It now reads `--glm-pack` too
  (`engine_model_path`, a server test), so benchmark tools that read `/props` see the pack's real path and its quant.
- docs/ornith: the config's example and the start command use port 8082, the port of the other models' configs (one
  model runs at a time).
- Known: with the NextN drafter, Ornith's decode slows down at long contexts (about 42 tok/s at a few thousand tokens,
  about 8 tok/s past 20K): the verify window's full-attention layers run in 1-4 blocks that walk every cached position
  one by one. A split attention for the window is next.

## v1.3.46 - 2026-10-22

**Ornith-1.5-122B (Qwen3.5-MoE) runs on the engine, on HIP and CUDA - a prompt path, a verify window and the GGUF's
NextN block as the drafter; DeepSeek V4.1 gets the AMD speed-ups V4 had (RDNA expert kernels, the CPU lane's IQ1_S
rows, hc3, the verify windows) and unsloth's UD-IQ2_XXS as the recommended file; the prompt router on HIP; the RAM
fill for the disk tail skip (issue #8).**

**Ornith-1.5-122B (Qwen3.5-MoE, llama.cpp's `qwen35moe`): a new model family on the engine, on HIP and CUDA
([docs/ornith/README.md](docs/ornith/README.md)).**
- The packer takes `qwen35moe` GGUFs (every dense tensor served as stored - Q8_0 through the quantized GEMV, F32 as
  F32 - the experts indexed in place, the NextN block left out).
- The token path: 36 gated delta-net layers (the `gdn.hpp` recurrence, a 4-tap conv, L2-normed q / k, per-head
  gates, a SiLU-gated norm) and 12 full-attention layers (32 heads over 2 KV heads, q / k RMS norms, NEOX RoPE on 64
  dims, FP16 caches, a sigmoid output gate), a plain residual through `rms_q8`, softmax top-8 routing and a shared
  expert behind its sigmoid gate on the common MoE path (the pool, the RAM tier, the CPU / iGPU lane).
- The batched prompt path: the same layers inside the prompt chunk loop - row kernels for the residual, the conv
  with its carried state, the recurrence (a block a value head, the state column in registers), the attention (4
  queries x one KV group a block, an online softmax a pair) - with the common MMQ expert path (`route_kernel` takes
  the softmax gating).
- Measured (unsloth UD-Q4_K_XL, AI PRO R9700 + 8060S lane, a 60 GB RAM tier): decode 23-25 ms a token (41-43 tok/s),
  a 575-token prompt at 233 tok/s; KL against llama.cpp's CPU backend 0.024 over 559 positions (the prompt path
  0.026 over 287) - llama.cpp's own CPU and HIP backends differ by 0.027-0.029 there. The CUDA build compiles
  (not run). Not yet: speculative decoding with the NextN block, snapshots, a layer split, images.
- The verify window for the arch (the engine's n-gram speculative decode, one draft a window by default): the
  token path's layers over the window's columns - the dense GEMVs read once, the recurrence and the conv carried
  through the columns, the attention over the caches, the MoE a column - and a rejected tail replayed out of the
  recurrent states. A copy task 23.5 -> 17.4 ms a token (57 tok/s), `STRATA_DS4_SPEC=3` 13.1 (76 tok/s, 97% of the
  drafts accepted); the greedy tokens match the token path's.
- The GGUF's NextN block as the drafter: a position's cache row from its normed hidden and the next token (eh_proj and
  k / v alone, for every prompt and answer position), up to 3 chained drafts with their probabilities, the decode's
  window verifying them. unsloth's block keeps Q8_0 experts, which the expert kernels do not serve: `tools/nextn_pack.py`
  repacks the block with Q4_K experts through ggml's quantizer, `STRATA_GLM_MTP_GGUF` loads it. Code 22.3 -> 16.9 ms
  a token (59 tok/s, 92% accepted), prose 21.6 -> 20.0 (the drafter's threshold 0.5), the copy task 85 tok/s; without
  the iGPU code 35.7 -> 41.7 tok/s, prose 33 -> 39.
- The decode recurrence through a register-column scan (the state read and written once; the GDN phase 3.0 -> 1.8 ms
  a token) and the conv / norms / gates in fewer launches.
- Debug: `GLM_CB_DIR` seam dumps for the arch under llama.cpp's cb names (`GLM_CB_POS=1` a file a position).
- The prompt path's F32 router (DeepSeek V4.1, Ornith) on HIP through the engine's own tile GEMM: rocBLAS's F32
  GEMM took 15.7 ms a layer inside the engine (a third of an Ornith prompt: 575 tokens 233 -> 359 tok/s on an AI PRO
  R9700) against 0.2 ms for the same shape alone; CUDA keeps cuBLAS.

**DeepSeek V4.1 on AMD: the speed-ups V4 had, at V4.1's shapes
([docs/deepseek41/README.md](docs/deepseek41/README.md)).**
- RDNA expert kernels at V4.1's 5120 / 2304 (gfx1201, gfx1151): IQ2_XS / IQ2_S gate/up, IQ2_XXS / IQ2_XS / IQ3_XXS /
  IQ2_S down - an AI PRO R9700 101 / 106 us a gate/up call of 8 experts (125 / 131 before), the down 46-56 us (67-81);
  the Radeon 8060S lane 249 / 273 (261 / 301). Bit-identical sums.
- The CPU lane: IQ1_S multi-token rows in AVX-512 VBMI; IQ2_S one-token rows back to ggml's dot (faster there).
- hc3 at V4.1's width (5120: 40 blocks, its lagged collapse) with the CPU lane's answer added inside the next layer's
  read, as V4's.
- The verify windows (the batched kind) run on V4.1: the n-gram drafts - a copy task 35.9-36.3 -> 27.7-28.2 ms a token
  with one draft, 24.7-24.8 with three (the greedy tokens identical across 0 / 1 / 3 / 5 drafts).
- Measured, 8 held-out sequences, R9700 + 8060S lane, the tail skip (3, disk) on: unsloth UD-IQ2_XXS 41.1 ms a token at
  KL 0.209 (57.8 / 0.186 without the skip), UD-IQ1_M 37.7 / 0.281, UD-IQ1_S 31.3 / 0.359, nazeshinjite Q2 50.7 / 0.243.
  Against the build before: UD-IQ1_S 43.9 -> 36.9 ms with the lane, 52.0 -> 44.0 without; UD-IQ2_XXS 62.7 -> 60.8 /
  69.6 -> 66.6. The docs now recommend unsloth's UD-IQ2_XXS (300 GB) over nazeshinjite's Q2 (366 GB): the same speed
  at KL 0.186 against 0.228.
- `STRATA_DS4_WIN_TIMING=1` (the window's host phases) and profile marks for the window (`v<T>_*`, `vm_*`).

**The RAM fill: with the disk tail skip, the most routed experts that sit on the SSD alone are read into the RAM tier
in the background (issue #8).**
- The disk tail skip (`STRATA_GLM_TAIL_SKIP_DISK=1`) never computes the experts it leaves out, so they were never read
  and never left the SSD: an expert the routes keep placing at rank 4-6 stayed disk-only and was skipped at every
  route - 16-25 skips a token through a whole session while the disk reads had long dropped to ~0.6 a token (#8,
  @sh-Tx2Lym's reading of the code, which was right).
- `STRATA_GLM_RAM_FILL` (default 2 with the disk tail skip, else 0): after each token up to that many of the most
  routed disk-only experts are queued for the lookahead's reader threads, into a free RAM slot past the routes'
  reserve, else over a colder RAM expert when the candidate's route count beats it by `STRATA_GLM_RAM_FILL_MARGIN`
  (4). A fill reads in chunks, each waiting until no route read is in flight (the SSD serves one stream at full
  speed; a first version that read whole experts beside the routes' reads made their waits 2.4x longer). The routes
  see the expert on the disk until a later boundary makes it live.
- DeepSeek V4.1 nazeshinjite Q2 on an AI PRO R9700 + 8060S lane with `STRATA_GLM_TAIL_SKIP=3 STRATA_GLM_TAIL_SKIP_DISK=1`,
  two held-out sequences each way, alternating: 44.4 / 40.8 ms a token against 46.0 / 41.2 without the fill, KL 0.328 /
  0.227 against 0.333 / 0.232, 16% fewer skips, ~7% fewer disk reads (the fills: ~0.6 experts a token, nearly all over
  a colder RAM expert).
- The engine log sums the skips and the fills (`glm fast tail skip CUDA0: ... RAM fill: ...`).

## v1.3.45 - 2026-10-22

**Setup needs HIP 7.14 or newer - ROCm 10 on Linux, AMD's ROCm SDK 7.14 on Windows - and stops on ROCm 7.0-7.2, which
failed with current drivers on two PCs (issues #1 and #8).**
- The two reports: a ROCm 7.0 pip SDK on an AI PRO R9700 ran up to v1.3.37, then failed every kernel launch after a
  kernel update and a move to amdgpu-dkms 6.19.14 (#1); in a ROCm 7.2.1 container on an RX 9070 XT the RAM tier's
  registration failed (`iGPU lane off (hipHostRegister: invalid argument)`, `glm_fast zero_copy: invalid argument`,
  nothing in `dmesg`) and every request failed (#8). ROCm 10 fixed both with the same driver.
- Setup checks `hipcc --version` (it checked for ROCm 7 before) and, below HIP 7.14, stops with the reason and how to
  get ROCm 10 (Windows: the SDK setup installs, 7.14.1). `ABYSS_OLD_ROCM=1` builds with an older ROCm anyway, with a
  warning. ROCm 10's packages are HIP 7.15; amdrocm 7.14 on Linux passes the check but has not been run here.
- docs/amd (EN/JA): the requirement; "Older ROCm (7.0-7.2)" with both reports and the override; with ROCm 10 the
  kernel's own amdgpu worked on an RX 9070 XT (Linux 7.2.11, NixOS, Docker, no `iommu=pt`, #8), amdgpu-dkms stays the
  fix for `failed to pin userptr`; the troubleshooting row for `kernel launch(es) failed`. README and the manual: the
  toolchain lines.
- The engine's source is unchanged from v1.3.44. Thanks to @sh-Tx2Lym for the RX 9070 XT report and the
  ROCm 7.2.1 / 10 comparison on the same host.

## v1.3.44 - 2026-10-22

**Setup after a switch to another ROCm or CUDA toolkit builds the engine for it (issue #1); docs/amd says which ROCm
and driver an R9700 has been checked with.**
- A build folder configured with another compiler (a ROCm 7.0 pip SDK, then setup again with `ROCM_PATH` at ROCm 10):
  CMake deleted its cache by itself but configured again with the new compiler alone. `STRATA_ENABLE_HIP=ON` and the
  other `-D` options were dropped, the build had no GPU target and did nothing, and setup wrote "engine compiled" over
  the engine built for the old ROCm. Setup now configures from scratch (`cmake --fresh`) when the cache names another
  `CMAKE_<LANG>_COMPILER` than the one it passes: the HIP engine (`build-hip/`), the CUDA engine (`build/`: a newly
  installed toolkit, or `--nvcc`) and the vision encoder. Checked with the real CMakeLists on ROCm 10 under its two
  paths (`/opt/rocm/llvm`, `/opt/rocm/core-10.0/llvm`): before, the second configure left `STRATA_ENABLE_HIP=OFF` and
  `STRATA_PREFILL_MMQ=OFF`; now each switch configures from scratch and keeps them on.
- `./setup-deepseek.sh --nvcc` naming another toolkit than `build/` was configured with now configures again (before,
  the new toolkit was ignored unless the GPUs' architectures changed).
- A config that setup keeps takes the engine's `lib_dirs`. ROCm 7 and 10 share the library names (`libamdhip64.so.7`,
  `libhsa-runtime64.so.1`) and the engine has no RUNPATH, so the old ROCm's `lib` first in `LD_LIBRARY_PATH` would
  have loaded the old runtime into the new engine.
- docs/amd: on an R9700 only ROCm 10 has been checked. A user's ROCm 7.0 pip SDK failed every kernel launch after a
  kernel update and a move to amdgpu-dkms 6.19.14, until the engine was built with ROCm 10. Before that move, Ubuntu's
  kernel 7.0.0-38 with its own amdgpu could not pin the 80 GB RAM tier (`failed to pin userptr`), and amdgpu-dkms with
  `iommu=pt` fixed that. New: "Switching to another ROCm", and a troubleshooting row for `kernel launch(es) failed` on
  every request while `/health` answers. The README and the manual say the same in short.
- The engine's source is unchanged from v1.3.43. Thanks to @potatopote-debug for running it down on their R9700 and
  for the report.

## v1.3.43 - 2026-10-22

**DeepSeek's DSpark drafter for DeepSeek V4 Flash 0731 (opt-in, `./setup-deepseek.sh --dspark`): up to 5 drafted
tokens a step, checked in one verify window - on an AI PRO R9700 without an iGPU, swe-bench 6 tasks 48.4 -> 61.9 tok/s
(+28%), 1,024-token code answers 47-49 -> 63-66; with the iGPU lane 67.3.** Measured on an AI PRO R9700 (OCuLink, PCIe
Gen4 x4) in a Ryzen AI Max+ 395 PC with 94 GB RAM, DeepSeek V4 UD-Q2_K_XL with the Q5_K attention pack.
- **The drafter** (`src/core/dspark.cu`, `STRATA_DS4_DSPARK=<GGUF>`): DeepSeek's DSpark block drafter as bleysg's GGUF
  (`DSpark-drafter-Q2K-Q8-MarkovQ8-0731.gguf`, 6.94 GB, MIT) - 3 DeepSeek V4 layers over the target's layer 40-42
  outputs, a rank-256 Markov head and a confidence head. After each token it proposes the next 5 with a confidence
  each; the leading ones at 0.6 or above (`STRATA_DS4_DSPARK_THR`) go through one verify window behind the token, and
  each is kept when the target's own sample agrees - the samples are the target's, only the time per token changes.
  Its dense weights (Q8_0, 0.57 GB) and every expert (Q2_K, 2.1 GB a stage) sit in VRAM: 6.48 GB. Its context: the
  token path, the window's kept columns and a prompt chunk's last 128 rows leave their layer 40-42 outputs, projected
  into each stage's 256-row K = V ring. A draft runs on the target's kernels (mHC, GEMVs, attention, MoE) and five new
  ones (kv_rows, stream_mean, dspark_route, dspark_head, dspark_markov): ~4.4 ms on the R9700 (a stage's attention
  0.45 ms and MoE 0.63 ms, the head + Markov chain 1.12 ms). Its acceptance matches the numpy reference (4.68 against
  4.705 tokens a step on swe-bench t003). One GPU and DeepSeek V4 Flash only: with V4.1, a layer split or the MTP
  block the engine says why and runs on without it.
- **Repeats first:** when the last 3 tokens occurred before, their continuation (up to 5) is the window's drafts and
  the drafter does not run - while the repeats keep being right. Every step checks the repeat's proposal against the
  tokens that came, a share right per position: below 0.75 at the first position the drafter takes over (new code: 5
  repeated drafts were 22% accepted, and cost the drafter's steps their windows), otherwise the repeat's drafts while
  the product of the shares stays >= 0.5 (swe-bench t025's 5 repeated drafts were 53% accepted)
  (`STRATA_DS4_DSPARK_NGRAM`, `_NGRAM_P`, `_NGRAM_CHAIN`).
- **Verify windows of up to 8 columns, pipelined by groups** (`kVerifyMax` 4 -> 8): the pipelined window runs its
  columns in two groups of ceil(T / 2) on the two hardware queues, batched inside a group (`STRATA_DS4_SPEC_GROUP`):
  2 / 4 / 6 / 8 columns 34-36 / 52-56 / 71 / 88-89 ms against the batched window's 41 / 68.5 / 98 / 133.
  `STRATA_DS4_SPEC_PIPE=auto` (HIP's default) times the two kinds per width now (3 windows each at the same width;
  v1.3.40 compared windows of different widths and once kept batched windows at 78 ms against 56). Dense GEMVs of 2-8
  columns in one launch (wider ones in chunks of 8; bit-identical, the same sums a column). A batched window's
  CPU-lane experts of a layer in one call for all its columns (`STRATA_GLM_CPU_MULTI`; 7,683 columns bit-identical to
  a call a column). A group's VRAM-resident experts in one gate/up and one down launch (`gf::moe_gate_up_cols` /
  `moe_down_cols`, gridDim.z = the columns; `STRATA_DS4_WIN_COLS`, `_CHECK=1`: 2,691 columns bit-identical) - the
  windows no faster (34.1 / 51.7 / 70.9 / 88.2 against 34.7 / 52.4 / 70.0 / 88.7 ms), the drafter's MoE 0.69 -> 0.63 ms
  a stage.
- **The RAM tier with the drafter:** the drafter's 6.5 GB come out of the expert pool, and the tier pinned those
  experts instead: on the 94 GB machine 73.9 GB pinned (67.4 without) left 13 GB free, an 8K prompt took 301 s (10.6
  without), the decode ran at 1.6 tok/s and the server's watchdog restarted the engine. Without the drafter and a pool
  as small (`STRATA_GLM_RESERVE_MB=7200`) the same happened (61 s, 1.6 tok/s). With the drafter the tier keeps its size
  without it (67.0 GB here) and the coldest experts come from the SSD: 0.01 reads a token in the decode, an 8K prompt
  11.2 s. `STRATA_GLM_RAM_GB` still sizes the tier by hand.
- **Setup:** `./setup-deepseek.sh --dspark` downloads the drafter (bleysg's repo at the commit measured, SHA-256
  checked) into `Abyss-data/models/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF/` and adds `STRATA_DS4_DSPARK` to the
  config's env; a new config asks (default no), `--no-dspark` takes it out, `--rewrite-config` keeps it, a layer split
  leaves it out.
- **Without the iGPU lane** (through the server; swe-bench 6 tasks sampled at temperature 1.0, +-5% run to run):
  swe-bench 48.4 tok/s (v1.3.42) -> 61.9 (the drafter's threshold 0.6; 0.7: 61.3, 0.8: 59.8), +16-32% a task (their
  mean 62.7 against 50.5). 1,024-token answers (greedy, 2 runs, the same build without the drafter): code 46.6 / 48.6
  -> 63.4 / 66.4, prose 47.2 / 47.5 -> 50.5 / 51.4. Prompts 8K / 16K 850 / 978-1,028 -> 799 / 977-1,000 tok/s (fewer
  experts in VRAM). The same build without the drafter ran swe-bench at 54.5: its t025 / t030 answers came out long
  and repetitive (6,583 / 7,413 tokens, n-gram drafts 83-92% accepted), its short tasks as fast as v1.3.42's.
- **With the iGPU lane:** swe-bench 67.3 tok/s (v1.3.41: 51.7; the drafter's threshold 0.7 in that run), code 68.5 /
  73.2, prose 51.6 / 53.2.
- **Where the time goes now:** a window's column costs ~9 ms, the CPU lane's time for a token's RAM-tier experts (8.4
  ms: ~93 experts at 88 us, the RAM's ~93 GB/s); over a window the GPU is 89% busy (rocprof, a 3.1-column 45.9 ms
  window: experts ~20 ms, dense GEMVs ~19, mHC 13.5 with the lane's wait, attention ~5).
- **Measured and left out:** a smaller drafter expert cache (128 / 64 slots a stage: 3-5 GB of VRAM back, the target's
  VRAM hits +2 points, drafts 4.7 -> 6.2 / 7.1 ms; kept as `STRATA_DS4_DSPARK_EXPERTS`); keeping drafts by the
  product of their confidences (`STRATA_DS4_DSPARK_CHAIN`); repeats alone at adaptive lengths without the drafter
  (`STRATA_DS4_SPEC_ADAPT=1`: t010 48.5, t025 47.0, code 47.2 against the default n-gram decode's 47.7 / 48.7 / 48.5);
  fixed n-gram drafts of 3 or 5 tokens (3 helps t010 and costs code, 5 costs both).
- **The same results:** without the drafter the token path is unchanged - KL 0.005922 (iGPU lane) / 0.005687 (CPU
  lane) against the CUDA reference, as in v1.3.42. With the drafter loaded the target's logits stay within the
  run-to-run spread: KL 0.005472 (fewer experts in VRAM, 64% hits against 70%, so more on the lane). GLM-5.3-Flash
  Maya-S-v2's nll is identical (1.64344 with the iGPU lane, 1.63961 with the CPU lane). Greedy answers drift from the
  run without speculation after a while, with the n-gram decode as well (t003 at its 374th token, code at the 82nd,
  prose at the 5th): the window's sums differ from the token path's in the last bits and the tier moves' timing varies
  run to run. The first DSpark run of t003 matched all 1,024 tokens.
- **Not checked:** the CUDA build compiles; it was not run (no NVIDIA card attached) - neither the drafter nor the
  windows' changes on NVIDIA, which keeps pipelined windows (`STRATA_DS4_SPEC_PIPE=1`).

## v1.3.42 - 2026-10-21

**AMD (HIP) decode: the GPU waits for the CPU lane's answer with one poller, and two launches a layer fewer - DeepSeek
V4 without an iGPU +2.5-3.6% a token, swe-bench +2.5%, bit-identical.** Measured on an AI PRO R9700 (OCuLink, PCIe Gen4
x4) in a Ryzen AI Max+ 395 PC, DeepSeek V4 UD-Q2_K_XL unless named.
- **Polling the answer slowed the CPU lane itself.** While the GPU waits for the CPU lane's answer it reads the
  host-mapped answer over PCIe, and the CPU's memory serves every read while the lane reads its experts from it: 32
  polling blocks made the lane's own time 7.46 -> 8.43 ms a token. One block polls the host now, the others wait on a
  word in device memory (moe_cpu_wait, and hc3 below).
- **The answer added inside the next layer's mHC read** (`STRATA_GLM_HC_CPU`, deepseek4): no wait kernel of its own -
  hc3's 32 blocks read their 512 bytes each instead of the wait kernel's 8 blocks reading all 16 KB, and the paced
  promotion copies still go out after the answer is read.  Single tokens and the verify window's columns.
- **moe_wait and moe_fetch in one launch** (`STRATA_GLM_WAIT_FETCH`): a route without disk misses goes straight to the
  copies; with misses, block 0 applies the host's response first and the other blocks wait for it.
- The route's request x (16 KB into host memory) in 16-byte stores: 3.8 -> 1.6 us.
- **Without the iGPU lane**, one token at a time, 6 alternating starts: prose 44.45 -> 46.05 tok/s (+3.6%), code 46.50
  -> 47.65 (+2.5%); engine 22.51 / 21.51 -> 21.48 / 20.80 ms a token.  swe-bench 6 tasks with drafts (temperature
  1.0, +-5% run to run): 47.2 -> 48.4 tok/s, windows 30.3 -> 29.5 ms, single tokens 22.9 -> 22.2.
- **With the iGPU lane**, 2 alternating starts: prose 50.95 -> 51.85 tok/s, code 52.45 -> 53.65.
- **The same results:** KL 0.005687 (CPU lane) / 0.005922 (iGPU lane) as in v1.3.41; greedy answers with drafts
  (pipelined windows, gate off) identical to v1.3.41's; GLM-5.3-Flash Maya-S-v2's nll identical (1.64344 with the iGPU
  lane, 1.63961 with the CPU lane, no speed change there); DeepSeek V4.1 (nazeshinjite Q2, 8 held-out sequences,
  its SSD reads through the one-launch wait + fetch) KL 0.2280 against 0.2284 before (a run before v1.3.40's
  keep-alive), 58.1 ms a token, 0-7 late iGPU calls a sequence.
- HIP only by default: NVIDIA keeps the separate kernels (`STRATA_GLM_HC_CPU=1`, `STRATA_GLM_WAIT_FETCH=1` turn them on
  there, not measured).  The CUDA build compiles; it was not run (no NVIDIA card attached).
- Measured and left out: merging a verify window's CPU-lane requests (the pipelined window is GPU-bound: 29-31 ms a
  window with the iGPU lane, 30-33 with the CPU lane); ROCm's dispatch settings (`HSA_ALLOCATE_QUEUE_DEV_MEM`,
  `HIP_FORCE_DEV_KERNARG`, `ROC_SYSTEM_SCOPE_SIGNAL=0`, `DEBUG_CLR_AQL_DEV_QUEUE`, `AMD_DIRECT_DISPATCH=0`: a dependent
  kernel takes ~3 us whatever); the CPU lane's threads (12-28 all ~90 us an expert: RAM-bound); huge pages for the RAM
  tier (the CPU reads 120 GB/s either way); the answer pushed into VRAM through the 32 GB BAR (a round trip 16.7 ->
  15.8 us).
- Known, also before this version: answers can differ slightly between runs.  A token boundary makes a background tier
  move live once its copy has landed (`cudaEventQuery`), so whether an expert runs on the GPU (q8_1) or on the CPU lane
  (q8_K) at a given token depends on the copy's timing.

## v1.3.41 - 2026-10-21

**The CPU lane's IQ2 / IQ3_XXS rows in AVX-512 with the grids looked up in registers: decode on one GPU without an
iGPU +8-10%, bit-identical.** Measured on an AI PRO R9700 (OCuLink, PCIe Gen4 x4) in a Ryzen AI Max+ 395 PC (Zen 5,
16 cores, ~113 GB/s from RAM), DeepSeek V4 UD-Q2_K_XL unless named.
- **Why:** without an iGPU the CPU threads compute the RAM-tier experts (the CPU lane). ggml-cpu's AVX2 dots for
  these formats rebuild every weight from table lookups in memory, so the lane read ~65 GB/s of the RAM's ~113: a call
  with one expert took 147 us, each more expert ~124 us.
- **`src/kernels/cpu/iq_avx512_vbmi.cpp`** (IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS, one token or several): the IQ2 grids
  hold only the bytes 8 / 25 / 43 and IQ3_XXS's only 4, 12, ..., 52, 62, so each entry is a 16-bit code. The codes stay
  in registers (IQ2_S's are read from L1) and vpermt2w looks 32 entries up at once; vpmultishiftqb + vpshufb turn them
  into the grid bytes, vpermi2b turns the 7-bit sign indices into ksigns_iq2xs's masks, and maddubs + vpdpwssd take
  ggml's integer steps. A block is decoded once for up to 4 tokens (verify windows, prompt chunks). Each token's
  result is bit-identical to ggml-cpu's dot: checked against it at every start (a mismatch keeps v1.3.40's rows), and
  `native_rows_exact_test` checks 1-11 tokens.
- **An expert** (glm_cpu_bench): one thread 1,175 -> 665 us (an IQ2_XXS gate/up 848 -> 424); 16 threads 109 -> 90-102
  us, memory-bound now (~89 GB/s).
- **Decode without the iGPU lane, one token at a time:** 41.8-42.8 -> 46.1-46.5 tok/s; GLM-5.3-Flash Maya-S-v2 +6-9%,
  DeepSeek V4 huihui Q2 +3-5%.
- **swe-bench, 6 tasks with drafts** (temperature 1.0, +-5% run to run): 49.6 tok/s without the iGPU lane (v1.3.39:
  36.4), 51.7 with it (v1.3.39: 42.4). Prompts without the lane 8K / 16K 844 / 977-1,026 tok/s, about as with it.
- Needs AVX-512 with VNNI and VBMI (Zen 4 / Zen 5, Ice Lake and later: the check the other AVX-512 rows use); other
  CPUs keep v1.3.40's rows. `STRATA_NO_IQ512_VBMI=1` turns these off.
- KL against the reference, 1,000 tokens, with the CPU lane: 0.005687, the same as v1.3.40.
- Measured and left out: more VRAM promotion churn (hits 77-79% but 40-43 tok/s: the PCIe copies hold up the lane's
  host-mapped reads; 4/16 stays); HIP graphs (a launch costs 0.9-1.7 us on the host, and a kernel switch ~2.3 us on
  the GPU either way: only fewer kernels would help). Expert deferral (`STRATA_GLM_DEFER=1`, lossy, off by default)
  without the lane: 53.0-53.2 tok/s at KL 0.059 (the reference's top token 90% of the time, 97% off) - now in the
  manual's row.
- The CUDA build compiles; it was not run (no NVIDIA card attached). The CPU rows are the same code in both builds.

## v1.3.40 - 2026-10-21

**AMD AI PRO R9700: prompts ~2x (ROCm's single hardware queue handled), decode +17% (gfx1201 kernels), the verify
window chosen by timing; the iGPU lane kept awake through DeepSeek V4.1's SSD waits.** Measured on an AI PRO R9700
(OCuLink, PCIe Gen4 x4) + Radeon 8060S lane, DeepSeek V4 UD-Q2_K_XL unless named.
- **Prompts: 8K / 16K 454 / 599 tok/s (v1.3.39, one hardware queue) -> 866-995 / 999-1,011.**
  - The prompt's host lane waited for its expert inputs with a stream sync. With every stream in one hardware queue
    (`GPU_MAX_HW_QUEUES=1`, which R9700 guides set against a MES firmware's idle power) that waited for all the other
    streams' earlier work too, 120-150 ms a layer. It waits for an event after the copies now (the same on CUDA).
  - **`GPU_MAX_HW_QUEUES=2`**: setup writes it into HIP configs, and the server passes it when neither the shell nor
    the config sets it. 8K / 16K: one queue 586-650 / 676-684 tok/s, ROCm's 4 653-737 / 748-756, two 866-995 /
    999-1,011; idle power 12 W on one and two alike. A config from an older setup: add it to `env`, or run setup again.
  - HIP's prompt sub-batch is 1,024 (was 256): +8-10% (DeepSeek V4), +13-15% (GLM-5.3-Flash Maya-S-v2) on its own.
  - The iGPU lane's fused int8 prompt kernel was looked up on HIP device 0 - beside a discrete Radeon, the card - so
    the iGPU's prompt experts ran the one-row kernels. It is looked up on the iGPU now (as beside an RTX 5090).
  - The fused kernel's barriers wait for LDS only on gfx11 / gfx12: no measured change, the same results.
- **Decode** (one token at a time, no drafts): 22.0-22.4 ms a token (44.7-45.5 tok/s) -> 18.9-19.0 (52.2-52.9).
  - **Grid padding** (`STRATA_GLM_GRID_PAD`, HIP): a gfx1201 dispatch whose waves end at, or just below, a multiple
    of the GPU's wave slots holds up the next dispatch by 8-14 us (ROCm/TheRock#9300); DeepSeek V4's 4096-row GEMVs
    and expert downs hit it ~275 times a token. A few blocks that return at once: 22.2 -> 20.2 ms, bit-identical.
  - **A token's small GEMVs of several types in one launch** (`STRATA_GLM_MV_FUSE`, HIP; a small GEMV costs 15-29 us
    on gfx1201 whatever its size): 20.2 -> 19.5-19.9 ms. Not bit-identical to the typed kernels.
  - **The RDNA decode expert kernels on gfx1201** (gfx1100's gate/up and down): ~20% faster each, 19.4-19.5 ->
    18.9-19.0 ms a token, bit-identical (`STRATA_HIP_EXPERTS_LEGACY=1`: the old ones).
  - Without the iGPU lane (the CPU threads compute the RAM-tier experts): 23.4-23.9 ms (41.8-42.8 tok/s).
- **The verify window** (`STRATA_DS4_SPEC_PIPE`): on HIP the first windows time the pipelined and the batched kind (3
  each) and the faster stays. The R9700 keeps the batched one: 2-token windows 30.6 against 44.5 ms with one hardware
  queue (swe-bench 6 tasks 42.3 / 41.9 -> 50.2 / 46.1 tok/s), 14.1 against 14.9 ms a token with two. NVIDIA keeps the
  pipelined window (`auto` times it there too; not checked on an NVIDIA card yet).
- **DeepSeek V4.1: the iGPU lane's keep-alive** runs 500 ms after a route (`STRATA_GLM_IGPU_POKE_MS`, was 20). The SSD
  waits between routes let the iGPU sleep, and its next call took 21-27 ms (the CPU threads took it over):
  nazeshinjite Q2, the slowest of the 8 held-out sequences, 80.4 -> 65.6 ms a token, 123 -> 3 late calls.
- **Setup** keeps the 3 GB desktop reserve only on a GPU with a screen connected: an eGPU beside an APU's desktop has
  none (~380 more DeepSeek V4 expert slots on an R9700).
- **GLM-5.3-Flash Maya-S-v2**: prompts 8K / 16K 417-421 / 539-543 tok/s (one queue, 2026-10-20) -> 644-716 /
  797-843; decode unchanged (34.3-38.3 tok/s).
- KL against the reference, DeepSeek V4, 1,000 tokens: 0.005922 with the iGPU lane (v1.3.39: 0.005195; the one-launch
  GEMVs' sum order - `STRATA_GLM_MV_FUSE=0` gives 0.005195 again; the CUDA build's own 0.005939), 0.005687 with the
  CPU lane (0.0059). The CUDA build compiles; it was not run this time (no NVIDIA card attached) - its kernels are
  unchanged, the prompt's event wait and the keep-alive window apply to it too.
- `STRATA_GLM_SVC_STAT=1` also prints the CPU lane's calls by their expert count (an R9700 host: 147 us for one
  expert, ~124 us for each more - the CPU's IQ2_XS / IQ3_XXS dots, not the call, take the time).
- Docs: MANUAL (the new settings), docs/amd (`GPU_MAX_HW_QUEUES`, two troubleshooting rows).

## v1.3.39 - 2026-10-21

**AMD: a discrete Radeon (AI PRO R9700) with a Ryzen AI Max's iGPU as its lane, +12-46% decode; GPUs named by PCI
address (`gpu_pci`); an AMD install guide (docs/amd).**
- **The iGPU lane under the HIP engine.** On an R9700 + Radeon 8060S the iGPU computes the RAM-tier experts as it does
  beside an RTX 5090 (an expert 0.040 ms against 0.10-0.13 on the CPU): swe 6 tasks, CPU lane -> iGPU lane, DeepSeek V4
  UD-Q2_K_XL 36.4 -> 42.4 tok/s, huihui Q2 33.9 -> 38.1, GLM-5.3-Flash Maya-S-v2 25.8 -> 35.8, Maya-S24 28.2 -> 41.2;
  prompts unchanged. KL against the CUDA build 0.0052 (0.0059 with the CPU lane).
  - `libabyss_igpu.so` is built for HIP engines too, with the engine's architectures as well as the iGPU's: HIP
    refuses a library's kernels on every GPU when any visible GPU has no code in it.
  - The lane picks the integrated GPU itself and keeps the engine's device current outside its calls; memory the
    engine pinned already is read in place.
  - The server lists the iGPU after a single discrete HIP GPU (`HIP_VISIBLE_DEVICES=0,1`; `STRATA_GLM_IGPU=0` or a
    layer split: as before). GLM-5.3-Flash's automatic layer split no longer picks the iGPU (it ran swe at 11.7 tok/s).
- **`gpu_pci`.** HIP numbers the GPUs in the driver's boot order, which flips when an eGPU is attached after boot or
  the buses are renumbered. Setup writes the cards' PCI addresses into the config, and the server maps them to the
  current numbers at each start. Windows and older configs keep the numbers.
- **docs/amd** (English and Japanese): the amdgpu driver, ROCm 10's per-architecture packages (an R9700 needs
  `amdrocm10.0-gfx1201`, or its first prompt hangs in rocBLAS), why `HSA_OVERRIDE_GFX_VERSION` must not be set (ROCr
  fails on gfx12), checks, a discrete Radeon + the iGPU, Windows, troubleshooting. MANUAL and README link it.
- The CUDA build is unchanged in its answers and speed: on an RTX 5090 + Radeon 8060S lane, KL 0.005939 (identical to
  v1.3.38), DeepSeek V4 UD-Q2_K_XL decode 94.7-98.0 tok/s, swe 6 tasks 103.4 (104.8 before), prompts 8K 1,646-1,989 /
  16K 2,599-2,729 - all within the run-to-run spread.

## v1.3.38 - 2026-10-20

**DeepSeek V4.1: unsloth's GGUFs load (llama.cpp PR 30817's layout); UD-IQ1_S (233 GB) runs 32% faster than the
nazeshinjite Q2 for lower quality; a supported-models table in the README.**
- **unsloth/DeepSeek-V4.1-Flash-1017-GGUF** (and other GGUFs from llama.cpp PR 30817, likely Maya-S V4.1): one
  `per_layer_token_embd.weight` Engram table (`deepseek41.ple.*` keys, global head offsets), `engram_gate` with the
  q / k norms folded in, `attention.kv_source_layers` / `attention.indexer.*` keys. `tools/iq_pack.py` writes the pack
  (BF16 `indexer.proj` is widened to F32 like F16); the engine reads the folded gate with a ones vector for the key norm.
  The nazeshinjite and mrfakename GGUFs pack as before (same engram.txt).
- **UD-IQ1_S** on an RTX 5090 + Radeon 8060S lane, tail skip: **21.9 / 22.0 ms a token** (was 32.3 / 32.4 with the
  nazeshinjite Q2; disk waits ~3,750 -> ~2,300); the 8 held-out sequences (exact routing) KL 0.228 -> 0.355, same
  top-1 79.2% -> 75.0%, ppl 9.68 -> 10.80. A server chat answered correctly at 38.5 tok/s. Pack recipe and table in
  [docs](docs/deepseek41/README.md#ggufs-that-load). Only UD-IQ1_S was measured; other unsloth quants share the layout.
- **README / README.ja**: a "Supported models" table (DeepSeek V4, V4.1, GLM-5.3-Flash GGUF / EXL3, with what was
  measured and what is untested).

## v1.3.37 - 2026-10-20

**DeepSeek V4.1: the iGPU lane's Q2_K down 31% faster; `STRATA_GLM_ROUTE_AFFINITY_RAM` measured (0.08: -8-12% a token
for KL +0.003).**
- **The lane's Q2_K down** (nazeshinjite's V4.1 Q2 experts) runs 8 lanes a row on whole 256-value super-blocks, the
  products and min terms in integers: on a Radeon 8060S the down kernel 53.9 -> 37.1 us for two experts (144 -> 209
  GB/s, the gate/up's level), a lane call 170 -> 155 us, V4.1 decode -2% (33.5 -> 32.7 ms a token). Not
  bit-identical to the old dot (another sum order): KL 0.2294 -> 0.2279 on the 8 held-out sequences (exact routing).
  `ABYSS_IGPU_Q2K_NR=0` for the old kernel. DeepSeek V4 and GLM-5.3-Flash are unchanged (KL 0.005939 as before).
- **`STRATA_GLM_ROUTE_AFFINITY_RAM`** (opt-in, from v1.3.36) measured on V4.1 with the tail skip: `0.08` reads 22%
  fewer experts from the SSD, -8-12% a token, KL 0.2406 -> 0.2438, ppl 9.76 -> 9.81; `0.16` -12-20% at KL 0.2501
  ([docs](docs/deepseek41/README.md#options-for-this-model)).
- `STRATA_GLM_ROUTE_PROF=1` (diagnostics): the route kernel's SM cycles per phase. DeepSeek V4 on an RTX 5090, 15 us
  a route: the request's system fence 6.3 us, the plan 3.4, the top-k 2.4 - too little to chase (~1% a token).

## v1.3.36 - 2026-10-19

**EXL3 on the iGPU lane 16% faster; DeepSeek V4.1's Engram reads off the token's path; opt-in route affinity and a
V4.1 Q5_K dense pack.**
- **EXL3 (GLM-5.3-Flash sc_2.10bpw): 22.4 -> 18.8 ms a token** on an RTX 5090 + Radeon 8060S lane. The trellis
  codebook's byte sum is one `v_sad_u8` on HIP (it compiled to ~5 instructions; bit-identical, 19.6 ms), and the decode
  GEMV's dot goes two halves at a time through `v_dot2_f32_f16` on RDNA 3 / 4 (another sum order: nll 1.5913 ->
  1.5959, inside the run-to-run range; `STRATA_EXL3_NO_FDOT2` at build time keeps the old form). The gap to the GGUF
  Maya-S-v2 (15.4 ms) is 1.2x, was 1.5x.
- **DeepSeek V4.1 Engram**: a token's 48 rows are read by a background thread while layer 0 runs and layers 1 / 14
  wait for them on the GPU (`STRATA_DS41_ENGRAM_ASYNC`, on; -0.5 ms a token, same logits); a prompt chunk's rows are
  read into the row cache right after its hash (`STRATA_DS41_ENGRAM_PREFETCH`, 8 threads; a 1K-token prompt's Engram
  1.7-1.9 -> 0.6 s, the prompt -6-7%).
- **The iGPU lane**: a route with SSD reads starts its lane before the reads (`STRATA_GLM_IGPU_EARLY`, on; V4.1 -2%,
  same logits); the lane takes the GPU's q8_1 input from the request (`STRATA_GLM_IGPU_XQ`, on; bit-identical: DeepSeek
  V4 KL 0.005939 as before); the RDNA IQ2_XXS gate/up kernel also runs V4.1's 5120 / 2304 shape (lane -3.5-7%,
  bit-identical; `ABYSS_IGPU_GU16_4096=1` for the old rule).
- **`STRATA_GLM_ROUTE_AFFINITY=<a>`** (opt-in, after Maya #252): a VRAM-resident expert's routing score gets +a, its
  weight stays the router's. GLM-5.3-Flash Maya-S-v2 at `0.04`: 15.4 -> 14.1 ms a token, VRAM hits 56 -> 69%, nll
  1.635 -> 1.631 (KL 0.043 to the default's logits). `_RAM` does the same for the RAM tier (not measured yet).
- **DeepSeek V4.1 Q5_K dense pack** (optional): `attn_q_b` / `attn_output_a` / `attn_output_b` at Q5_K, dense 8.14 ->
  6.50 GB, +177 VRAM slots, 10-15% faster a sequence, KL 0.2416 -> 0.2508 on the 8 held-out sequences
  ([how](docs/deepseek41/README.md#a-lighter-dense-pack-q5_k)). `tools/requant_dense.py` now keeps a GGUF's custom
  alignment (V4.1's 16 KB); before, its output was shifted and `iq_pack.py` refused it.
- V4.1 lookahead: a demand read waits on an in-flight lookahead read inside the streamed fetch
  (`STRATA_GLM_AHEAD_MERGE`, on; only with `STRATA_GLM_AHEAD`, which stays off: 37.4 against 34.9 ms without it on a
  3.4 GB/s SSD).
- Measured and dropped: unrolling the lane's K loop and a multi-expert down kernel (slower), GLM-5.3-Flash
  `STRATA_GLM_DEFER=1` on Maya-S-v2 (-19% but nll +0.063).

## v1.3.35 - 2026-10-19

**EXL3 quants of GLM-5.3-Flash (exllamav3), experimental; the installer says which quant suits the PC.**
- **The engine reads EXL3** - exllamav3's trellis-coded quants, in place from their safetensors - with Project Maya's
  EXL3 support (by @needmorevram, after exllamav3's format and kernels; Maya 25cbbf3, the kernels at v1.1.15): the
  packer (`tools/exl3_pack.py`), the GPU kernels (tensor-core tile decode on RTX 30+, CUDA cores before), the prompt
  path (light experts on the tensor-core kernels, the rest decoded to FP16 for cuBLAS with their rows rotated) and the
  CPU lane's split-K AVX-512 / AVX2 kernels (0.118 ms an expert on 16 Zen 5 threads).
- **The iGPU lane computes EXL3 experts** (new in Abyss): `libabyss_igpu.so` builds the EXL3 kernels for the APU and
  `abyss_igpu_experts_x3` runs a route's RAM-tier experts on the registered RAM tier (0.088 ms an expert on a Radeon
  8060S). An EXL3 expert never reaches the ggml-type kernels; a path without EXL3 kernels uses the CPU's.
- **Measured** (RTX 5090 + Radeon 8060S lane + 94 GB RAM, glm_ids4k's last 1,000 tokens): turboderp's sc_2.10bpw
  (88.3 GB) nll 1.591-1.598, top-1 63.3-63.7%, 22.4-22.7 ms a token; Maya-S-v2-IQ2_XXS (96.5 GB) 1.634 / 62.3% /
  14.8 ms. Better text at 8 GB less, at about two thirds of the decode speed on this lane.
- **The installer** (`./abyss.sh --setup`) offers `EXL3-sc_2.10bpw` (new, pinned to 8d2eca0e) and `EXL3-3.05bpw`
  beside the GGUFs, every file checked by size and sha256, and says what each suits on this PC: recommended (and the
  default) when RTX 30+ cards hold 80% of it, a quality option with an APU iGPU or an AVX-512 CPU for the RAM tier,
  not for a CPU with AVX2 only, or "experts would come from the SSD". Each quant has its own config and start script,
  so a GGUF and an EXL3 install sit side by side. NVIDIA only in the setup.
- **`STRATA_GLM_X3_SPLIT`** (opt-in): the CPU threads compute a share of each EXL3 route while the iGPU computes the
  rest. Slower on a Ryzen AI Max+ 395, whose CPU and iGPU share power and heat (22.7 ms alone vs 23.5-32.6 split).
- GGUF models are unchanged: GLM-5.3-Flash Maya-S-v2 KL 0.000000 against v1.3.34's logits, DeepSeek V4 KL 0.005939 as
  before.
- Docs: [docs/exl3/README.md](docs/exl3/README.md), an EXL3 section in the manuals; exllamav3's MIT notice in
  `third_party/exllamav3/LICENSE`.

## v1.3.34 - 2026-10-19

**DeepSeek V4.1 on Windows: the Engram tables build and read (issue #6).**
- `src/core/dsv41_fast_path.cu` used POSIX calls only (`<unistd.h>`, `O_CLOEXEC`, `posix_fadvise`, `pread`), so it
  did not compile with MSVC / TheRock ROCm, and with them patched over, the `(off_t)` offset cast cut the tables'
  offsets (162 GB / 264 GB into nazeshinjite's Q2) to Windows' 32-bit `off_t`: rows came from the wrong place and the
  engine stopped with "an invalid Engram row (NaN code or scale)". On Windows the tables are now opened with
  `CreateFileA` (`FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED`) and read with an overlapped `ReadFile` at the full
  64-bit offset, each reader thread waiting on its own event - the 16 Engram readers run at once, as the expert reads
  do (a synchronous handle would read one at a time). Linux is unchanged: on Lucebox's held-out sequence 0 the same
  logits as v1.3.33 (KL 0.184819, 49,104 rows read) and the same speed (55.9-56.6 against 56.2-56.8 ms a token,
  cooled SSD, alternating runs). Thanks to @ozz-sun for the analysis and the Windows run (RX 7900 XT, 10.55 tok/s).

## v1.3.33 - 2026-10-19

**Docs: DeepSeek V4.1 and what it changes against V4; the documents brought up to date.** No code change.
- **[docs/deepseek41/SPEC.md](docs/deepseek41/SPEC.md)** (new): V4.1 Flash's forward pass as a delta against V4's
  spec - the layer roles (four shared KV slots from layers 2 / 8 / 14 / 20, eight index sources whose top-512 the
  layers between reuse), the ratio-2 / ratio-1 compressors, the candidate blocks past 16K, the Engram hash, row format
  and gate, the lagged hyper-connection pre, the head without `output_hc_*`, the router without hash layers, the chat
  tokens - each with where Abyss implements it, what it does not run yet, and where ds4 and llama.cpp's PR disagree.
- **[docs/deepseek41/README.md](docs/deepseek41/README.md)**: a V4 / V4.1 comparison table and diagram, how to run it
  (a config with the speed settings, the memory-capped scope), v1.3.31's numbers (8 held-out texts 35.9 ms a token,
  KL 0.241; swe-bench's first 35 tasks 51.6 tok/s, 34 of 35 resolved), the v1.3.31 switches, the options measured
  worse.
- **README / README.ja**: V4.1 in the title, highlights, performance (its own table), how it works (what changes, a
  diagram), correctness, docs and roadmap; the 60-task swe-bench row for V4 (~107 tok/s); the RTX 5090 + RTX 3090
  row at the v1.3.26-1.3.27 split (87-90 tok/s); the Japanese README had no V4.1 at all and still said 82-84 tok/s.
- **MANUAL / MANUAL.ja**: a DeepSeek V4.1 section; settings rows for `STRATA_GLM_STREAM_FETCH`,
  `STRATA_GLM_CPU_PIN` lists, `STRATA_GLM_TAIL_SKIP_DISK`, `STRATA_GLM_DISK_LANE` and the Engram cache / readers;
  V4.1's limits (and that long chats need v1.3.32).
- **docs/deepseek4/PORT.md**: the speed table at v1.3.21-1.3.27, the steps from v1.3.18 to v1.3.28, the open items
  (the layer split has speculative decoding and prestaging since v1.3.26 / v1.3.27).

## v1.3.32 - 2026-10-19

**DeepSeek V4.1 crashed on the 16,385th token of a chat; the server now restarts an engine whose CUDA context is
lost.** Found by a swe-bench run: a 15K-token reasoning ended in `an illegal memory access was encountered` (Xid 13,
an out-of-range address), and the 125 attempts after it came back empty.
- **The cause**: the DSA top-k selection (`dsa_topk.cuh`, shared with DeepSeek V4's indexer, which takes 512 pools)
  held 1,024 winners, and V4.1's candidate blocks ask for up to 2,048 (`cand_blocks`, blocks of 8 tokens). Past 8K
  tokens (more than 1,024 blocks) the selection ran from corrupt ranks - a quality loss nothing reported; past 16K
  (2,048 winners among more blocks) it wrote the selection out of range, and every later launch failed the same way.
  The winners' arrays now hold 2,048 and the rank and output loops cover them; the parity test runs the V4.1 widths
  (1,025 and 2,048 winners, up to 250K pools: 228 of 228 cases match the reference order). Reproduced and fixed with
  24,005 tokens through the decode path (crashed at 16,384 before; 24.9 ms/token after). DeepSeek V4 KL 0.005939 and
  GLM-5.3-Flash S-v2 0.000000 unchanged (their selections take 512 or fewer).
- **serve**: an engine `ERR` naming a CUDA error that ends the context (illegal memory access, unspecified launch
  failure, device-side assert, ...) now ends the engine process, and the next request starts it again (as after a
  crash, issue #27) instead of every request failing until someone restarts the server. The streamed error event is
  unchanged (`{"error": {"type": "server_error", ...}}`, then `[DONE]`).

## v1.3.31 - 2026-10-18

**The iGPU lane's multi-second stalls are gone, and a disk-landed expert reaches VRAM while the SSD is still
reading it.** DeepSeek V4.1 Flash 1017 on RTX 5090 + Radeon 8060S lane + 93 GB RAM (nazeshinjite's Q2,
`STRATA_GLM_TAIL_SKIP=3` + `_DISK=1`, the speed config), Lucebox's 8 held-out sequences:
- **RAM tier in transparent huge pages** (`STRATA_GLM_RAM_THP`, default on, Linux). The lane's APU driver
  re-validates the registered tier's pages when its MMU notifier fires (`svm_range_restore_work hogged CPU` in the
  kernel log); with the tier in the driver's 4 KB pinned pages that took 15-24 s, during which the CPU lane took the
  calls (7 such stalls in 8 x 1K tokens). The tier now comes from an mmap with `MADV_HUGEPAGE` on as much of it as
  the free 2 MB blocks cover (the rest stays 4 KB pages, so the faults never compact memory - that had tripped
  systemd-oomd) and is registered from there, on one NUMA node too. 47.5 of a 60 GB tier in huge pages, the lane
  registers it in 255 ms (was 390), no stall in 17 x 1K tokens; the sequences that had stalled: 47.5 / 40.2 / 32.7
  / 34.6 -> 35.3 / 30.3 / 29.3 / 29.1 ms/token. `STRATA_GLM_RAM_THP=0`: the driver's pages as before.
- **Streamed fetch** (`STRATA_GLM_STREAM_FETCH`, default on). The service thread read a missed expert's 24 chunks
  into its RAM landing slot and only then let the device pull the 10 MB over PCIe, so the pull (~1.5 ms at Gen4 x4)
  ran after the read (~3 ms) on the token's critical path. The device now copies each chunk as it lands (a ready
  bit per chunk), PCIe running behind the SSD; only the last chunk's copy stays exposed. seq0 512 tokens 43.6 ->
  39.5 ms/token, logits identical. `=0`: the whole-blob fetch after the read.
- Together: 8 sequences 35.9 ms/token (27.9 tok/s; was 34.2 at v1.3.29 and 39.3 with the streamed fetch alone,
  both with stalls in the mean), 29-37 ms on 7 of them; one sequence 67 ms with the SSD's reads 2.5x slower for
  that run (disk waits 34 s against 10-15 s, no lane stall - the ORICO NVMe has such spells, and runs slower when
  warm: a sequence re-run back to back went 30.5 -> 39.2 ms/token at 51 -> 56 C). KL 0.2408 against the
  unquantized 1017's top-256 (0.2399 before; the tail skip's choice depends on which experts are on disk at the
  time). DeepSeek V4 KL 0.005939 and GLM-5.3-Flash S-v2 0.000000 unchanged.
- Measured worse and left off: `STRATA_GLM_SEED_PRIOR=1` (LFU counts seeded from the pack's routing profile:
  46.8 -> 48.5 ms/token) and `STRATA_GLM_DEMOTE_TIES=1` (a VRAM victim demoted instead of dropped on equal
  counts: 46.1, RAM re-reads up).

## v1.3.30 - 2026-10-18

**DeepSeek V4.1 chats fixed.** The ds4-layout V4.1 GGUFs (antirez's converter: nazeshinjite's, Lucebox's) mark the
48 added chat tokens as normal tokens, so v1.3.29's server split `<｜User｜>`, `<｜Assistant｜>`, `</think>`,
`｜DSML｜` ... into characters and the model answered with end markers written out as text (`2<｜end▁of▁sentence｜>`,
`Hello!<|end|>`). The tokenizer now restores the types DeepSeek V4's GGUF carries (control for `<｜...｜>` and
`<|EOT|>`, user-defined for `<think>`, `</think>`, `｜DSML｜`, `<dsml:`, `</dsml:`) when it reads a DeepSeek GGUF and
when it loads a pack, so a pack made with v1.3.29 works without re-packing. Greedy answers are now `2`, `Hello`,
`東京`. The quality numbers of v1.3.29 stand: they were measured on token ids, which this did not touch.
Also: the tools tests know the `deepseek41` packer arch (v1.3.29's CI failed on the old count of four).

## v1.3.29 - 2026-10-18

**DeepSeek V4.1 Flash (1017)** runs on the fast path ([docs/deepseek41/README.md](docs/deepseek41/README.md)): the
V4.1 attention (four shared KV slots, ratio-2 / ratio-1 compressed rows, index-source reuse, candidate blocks), the
lagged hyper-connection pre, and the Engram tables read row by row from the GGUF. Decode, the batched prompt path and
conversation reuse; both GGUF layouts (ds4 / antirez and llama.cpp's PR). RTX 5090 + Radeon 8060S lane + 93 GB RAM,
against the unquantized 1017's top-256 on Lucebox's 8 held-out sequences:
- nazeshinjite's Q2 (IQ2_XXS / Q2_K experts, 1017 imatrix): 17.7 tok/s, KL 0.229, top-1 agreement 79.2%.
- Lucebox's Q8_0 dense + 8-bit Engram with mrfakename's Q2_K/Q3_K experts: 8.5 tok/s, KL 0.168, 83.4%.
- The same GGUF as llama.cpp's V4.1 PR: teacher KL 0.639 against its 0.636. A chat's second turn reuses the first:
  44 s -> 4.7 s of prompt for 1,132 tokens, the same text.

Also:
- **serve**: a reasoning ends only at the special `</think>` token (GLM, DeepSeek V4 / V4.1). V4.1 wrote the tag as
  text while explaining a chat template, and the rest of its reasoning went out as the answer.
- **HIP / Windows** (#2, #3, #4): `-O3 -DNDEBUG` on every non-Debug HIP build (a toolchain with an empty
  `CMAKE_HIP_FLAGS_RELEASE` built the kernels at `-O0`: 1.30 tok/s instead of ~45 on an RX 7900 XT);
  `cudaMemcpyKind` in the HIP compat header (v1.3.28 broke the HIP build); no `M_PI`; quieter Windows warnings;
  `setup-deepseek.bat` (Windows' `./setup-deepseek.sh` - `START-ABYSS.bat` installs GLM-5.3-Flash); a tuning hint
  for Radeon cards on Windows; the MSVC 14.51 / TheRock 7.14.1 header workaround in
  [docs/maya/AMD_HIP.md](docs/maya/AMD_HIP.md).
- **CPU lane**: `STRATA_GLM_CPU_PIN=<CPU list>` (e.g. `0-15`) pins the workers with one GPU too, and pinning works on
  Windows (it was Linux-only and only for a split's pools).
- **iGPU lane**: the RDNA dot for the lane's down (IQ2_XXS / IQ3_XXS / IQ2_S) and DeepSeek V4's IQ2_XS gate/up -
  lane time -4-5%, decode about +1%, bit-identical sums; a stall is logged as a range (an idle server made the old
  "finished after" read minutes).
- **New options, off by default**: `STRATA_GLM_TAIL_SKIP_DISK=1` (the tail skip leaves out only disk-only experts),
  `STRATA_GLM_DISK_LANE=1` (a disk read computed by the lane in RAM), `STRATA_GLM_AHEAD_TOP=<n>`,
  `STRATA_GLM_RAM_THP=1` (transparent huge pages for the RAM tier), `iq_pack.py --first-wins`,
  `requant_dense.py --from=<types>`.
- GLM-5.3-Flash and DeepSeek V4 are unchanged: KL 0.000000 (GLM S-v2) and 0.0058 (V4) against their references.

## v1.3.28 - 2026-10-17

Decode right after a prompt is no longer slow.  The prompt borrows part of the VRAM expert pool, and the token
boundaries afterwards copied the experts back in 24 at a time (7-8 MB each); the next token's first host-mapped reads
(the table edits, the embedding row) waited until all of those copies had landed, so the GPU idled ~20 ms a token.
The copies (these refills and the swaps' promotions and demotions) now go out in pieces of up to 1 MB a MoE layer,
after the layer's CPU-lane answer, while the next layer's mixer runs.  RTX 5090 + Radeon 8060S lane:
- GLM-5.3-Flash AJ-IQ2_XXS: the tokens after an 8K / 16K prompt 14-17 ms (were 35-39 ms for 10 / 16 tokens); the
  first 20 tokens 305-352 ms (581-711).
- DeepSeek V4 Flash (q5k pack): 11-13 ms (were 16-20 ms for 20-28 tokens); the first 20 tokens 235-271 ms
  (360-381).
- RTX 5090 + RTX 3090 split 33 / 10 (`STRATA_GLM_SPLIT_KEEP=1`), 8K prompt: 12-14 ms (were 54 ms for 9 tokens);
  the first 20 tokens 260-288 ms (581-750).
- Steady decode unchanged (one card 91-97 tok/s, the split 86-91); KL the same (0.0051-0.0060 against 0.0055-0.0058
  without the pacing, one card and a split).
- `STRATA_GLM_BG_PACE_KB=<KB a layer>` (default 1024); `0`: the copies at the boundary, as before.  The spares'
  demotions (which the lane may wait for) and the KDA graphs (`STRATA_GLM_KDA_GRAPH`) still copy at the boundary.

## v1.3.27 - 2026-10-16

A layer split's prompts prestage the experts too (one card had it; a split part copied each chunk's experts after
the chunk's attention): each part keeps its own ring of staged experts and fills it while the chunk before runs.
RTX 5090 + RTX 3090 split 33 / 10 (`STRATA_GLM_SPLIT_KEEP=1`, RAM tier 66 GB): a 32K prompt 2,274-2,350 tok/s
(v1.3.26: 2,119-2,186, +7-8%), 8K about the same (1,424-1,508 vs 1,435-1,650).  The RTX 5090 alone is unchanged
(32K 3,173-3,334).
- Sizing under a split with `--prefill auto`: a part picks its chunk first (up to 8,192 tokens) and gives what is
  left to the prestage ring (none below 16 slots), so a small card (a V100) does not get smaller chunks for it.
  One card picks as before.  `STRATA_GLM_PRESTAGE=0` turns it off.
- A split's prompts are still about 30% behind the RTX 5090 alone here.

## v1.3.26 - 2026-10-16

The n-gram speculative decode (`STRATA_DS4_SPEC`) runs with a layer split too: the verify window goes through the
parts one after another, each part's last layer writing its columns' residuals, which cross to the next part through
a pinned host buffer (as a token's residual does); the last part runs the head and keeps the logits.  RTX 5090 +
RTX 3090 split 33 / 10: 87.1-90.4 tok/s (a window 16.2 ms, a single token 11.4; 57% of the drafts accepted on code,
as on one card), 89.7-91.1 with `STRATA_GLM_DEFER=1`; the batched window (`STRATA_DS4_SPEC_PIPE=0`) works as well.
The 5090 alone is unchanged (94.5-95.2 tok/s, a window 14.0-14.5 ms) and `--layer-split auto` still picks it here.
- Expert deferral in the window: a part's first layer no longer adds the layer before's part (the part before waited
  for it), as the token path.
- Greedy answers are not a bit-exact
  check here: with the speculation on or off they part after a few hundred tokens on one card too (which device
  computes an expert, the GPU or the iGPU lane, varies from run to run).

## v1.3.25 - 2026-10-16

`--layer-split auto` prices the placements by what a decode token costs on this hardware, and runs on one GPU when
that is faster.  RTX 5090 + RTX 3090 (both PCIe Gen4 x4), the default start with both cards visible: the 5090 alone,
94.2-94.8 tok/s with the n-gram drafts (v1.3.24 split them 26 / 17: 84.4 tok/s).
- Why a split lost: a decode token goes through the parts one after another, the 3090's layers cost 1.64x the
  5090's (its memory bandwidth, STRATA_GLM_PROF), and the second card's VRAM only trims the CPU lane's wait, which the
  iGPU lane already mostly hides; a split also has no n-gram drafts (+3-4% on one card).
- The model: a card reads at 56% of its bus's peak, a miss costs 57% of its bytes' read from RAM (no longer the RAM
  speed split between the parts - they do not run at once), 0.39 ms a boundary; fitted by least squares on splits
  22 / 26 / 30 / 36 and each card alone (predicted 12.19 / 11.70 / 11.38 / 11.33 / 10.87 / 17.48 ms against measured
  12.33 / 11.85 / 11.38 / 11.82 / 10.86 / 18.13).  `STRATA_SPLIT_BW_EFF`, `STRATA_SPLIT_MISS_SHARE`,
  `STRATA_SPLIT_HOP_MS` override the terms; `STRATA_SPLIT_DUMP=1` prints each part's bytes for a refit.
- `STRATA_GLM_SPLIT_KEEP=1` splits anyway (it now picks 33 / 10 here: 88.3-88.7 tok/s); an explicit
  `STRATA_GLM_SPLIT=K` / `"layer_split"` is unchanged.  Fitted on one machine: V100 pairs and GLM's pipelined split are
  not re-measured.

## v1.3.24 - 2026-10-16

Expert deferral (`STRATA_GLM_DEFER=1`) works with a layer split (`STRATA_GLM_SPLIT`) too: each part defers within its
own layers, and a part's last layer waits for its own CPU-lane part before the hop.  Answers: split alone KL 0.0115
(unsplit 0.0119), split + deferral 0.060 (unsplit deferral 0.082 - the part boundaries keep the in-order sum).  No
speed from it on an RTX 5090 + RTX 3090 split: 80.9-81.4 tok/s either way (12.4 ms a token) - with two cards' VRAM
88% of the experts hit VRAM and the CPU lane is 1.7 ms a token, little to hide.  A split on two GPUs runs slower than
the 5090 alone here (81 against 95 tok/s).  Note for one-GPU splits (`STRATA_GLM_SPLIT_SAME=1`): each part sizes its
RAM tier from the free RAM, so give `STRATA_GLM_RAM_GB` - two parts pinned 72 GB of 93 and the system's OOM killer
stopped the session.

## v1.3.23 - 2026-10-16

Expert deferral (`STRATA_GLM_DEFER=1`) covers the verify window too: each column's CPU-lane part joins its streams
after the next layer's attention, as the single-token path's does.  The 2-token window 15.0 -> 13.1 ms, and the
window is worth taking more often again: 1,024-token answers 104.5-106.1 -> 107.7-110.2 tok/s (off: 95-96).  Same
answers as v1.3.22's single-token path (KL 0.08).

## v1.3.22 - 2026-10-16

Expert deferral, opt-in (`STRATA_GLM_DEFER=1`, lossy): the GPU no longer idles for the CPU lane - single tokens
10.9 -> 9.6 ms, +9-10% tok/s on an RTX 5090 + Radeon 8060S, at a cost in the answers (KL 0.08 against the Q8_0
reference; 0.012 without).
- The CPU lane's part of a trunk layer's MoE is waited for after the NEXT layer's attention and written straight into
  the hyper-connection streams with its own layer's write weights (`hc_defer_add`; the weights saved by a kernel,
  not the copy engine).  Layer l + 1's attention misses it, its router and experts see it.  The trunk's last layer, the
  draft layer and the verify window keep the in-order sum.  Added one layer late into the next FFN output instead
  (KTransformers' form) it ran 111 tok/s at KL 0.21 - the write weights differ by layer.  swe-bench L7 (16 tasks x 3)
  with it on: 107.2 tok/s (v1.3.21, off: 100.9), resolved 81.2% (13 of 16; off: 87.5%, 14 - the 3-run spread is 75-88%), quality 83.1 (81.2), 9.4 ms a token averaged over the run (10.1).
- Tried and dropped: a persistent cooperative grid for the iGPU lane (branch `lane-persist`).  The launched lane has no
  fixed cost to remove - its three kernels hide their launches and run at the ~160 GB/s the iGPU reads registered RAM
  at - and the resident grid starved the display's shaders on the APU into a GPU reset.  `bench/lane_bench.cpp`
  (the lane alone, 10 s) stays.

## v1.3.21 - 2026-10-16

The token boundary no longer waits behind the expert moves' DMAs: a single token 11.7 -> 11.1 ms on an RTX 5090 at
Gen4 x4, the same answers.
- A between-tokens profile (`STRATA_GLM_GAP_PROF=1`) showed the GPU idle 0.5-1.0 ms between two tokens while the host's
  own work there was 0.05-0.25 ms: the small copies at a token's start and end (the expert-table edits, the 16 KB
  embedding row, the 4-byte argmax or sampler result) went through the copy engines, queued behind whichever 8 MB
  promotion or demotion was in flight there (1.2 ms each at Gen4 x4, ~0.5 a token). With `STRATA_GLM_PROMOTE=0` the
  idle fell to 0.03-0.1 ms, and the VRAM hit rate with it.
- Now the kernels read the edit sets host-mapped, gather the embedding row from a pinned copy of `token_embd` (Q8_0 /
  Q5_K; 347 MB of host memory for DeepSeek V4 Flash) by a token id they read host-mapped, and write the argmax /
  sampled token into host-mapped memory - no copy engine on the token path. The GPU's idle between tokens 0.5-1.1 ->
  0.11-0.15 ms; 1,024-token answers: a single token 11.69 -> 11.10 ms, the 2-token verify window 14.85 -> 14.61;
  KL against the Q8_0 reference unchanged (0.0117). The embedding row's page-cache stalls (0.2 ms a token on a cold
  shard mapping) go with it. `STRATA_GLM_ZERO_COPY=0` restores the copies. swe-bench L7 (16 tasks x 3): 100.9 tok/s (v1.3.20: 99.0) at 10.1 ms a token averaged over the run (10.8), resolved 87.5% with the same two tasks failing, quality 81.2 (82.0).
- docs: the pack table has the Q4_K row (KL 0.0246, twice Q5_K's, for 2% on a token: where the lighter dense pack
  stops paying); the two variables above in the environment table; the README / MANUAL decode rows (~92-95 tok/s, swe-bench ~100).

## v1.3.20 - 2026-10-15

The Q5_K attention pack is now the measured default: decode ~8% faster than the GGUF as published, at the same
swe-bench quality.
- swe-bench L7 (16 tasks x 3, sampled, RTX 5090 + Radeon 8060S): the Q5_K pack 99.0 tok/s, resolved 87.5% (14/16),
  success rate 83.3%, quality 82.0, combined 75.9 - against the Q6_K pack 93.9 / 81.2% / 81.2% / 81.2 / 74.1 and the
  Q8_0 GGUF 91.3 / 81.2% / 79.2% / 81.0 / 71.5 (the task-level differences are the sampling's). Engine statistics over
  the runs: a single token 10.78 ms (Q6_K 11.26, Q8_0 11.88), the 2-token verify window 14.78 (15.65, 16.29), VRAM hit
  74.6% with 3,225 slots (74.1% / 3,139, 73.2% / 3,010). KL against the Q8_0 reference 0.0119 (v1.3.19's table).
- docs: the four performance tables carry the Q5_K pack's decode figures (1,024-token answers ~90-93 tok/s; the
  swe-bench row is now the 16-task L7 set at ~99); the "lighter dense pack" section says which pack the author runs
  and what each costs. The prompt and 125K rows were not re-measured (v1.3.16-1.3.18, the GGUF as published).
- abyss.py: the HIP check's hipcc lookup as before when no hipcc file exists (the setup tests fake the root and mock
  the version; v1.3.19 made them exit).

## v1.3.19 - 2026-10-15

Lighter attention projections as a documented, measured option; abyss.py finds ROCm 10's package layout.
- `tools/requant_dense.py` (moved in from Abyss-tools/bench) re-quantizes `attn_q_b` / `attn_output_a` /
  `attn_output_b` - 4.6 of the 6.6 GB every decoded token reads - and copies the rest of the shards byte for byte;
  `iq_pack.py` packs the result as usual. docs/MANUAL.md "A lighter dense pack" has the commands and the measurements
  (KL against the engine's Q8_0 reference over 1,000 tokens, CPU lane, one setup for all three): Q8_0 0.0060 (same
  top token 97.0%), Q6_K 0.0082 (95.8%), Q5_K 0.0119 (95.9%, the largest single-token divergence 0.57 against 0.11).
  - Q6_K on the swe-bench L7 set (16 tasks x 3, RTX 5090 + Radeon 8060S): 91.3 -> 93.9 tok/s with the same resolved
    rate (81.2%) and quality score (81.0 -> 81.2) - the pack the author now runs by default.
  - Q5_K on an RTX 5090, 1,024-token answers: a single token 12.13 -> 11.65 ms against Q6_K, the verify window
    15.52 -> 14.86, 86-89 -> 93 tok/s, 3,225 VRAM expert slots (Q6_K 3,139, Q8_0 3,010). Worth more on a card with less
    memory bandwidth (a Radeon AI PRO R9700 reads 1.5 GB less a token); twice Q6_K's quality cost.
- abyss.py (Linux): the ROCm root is also looked for in /opt/rocm/core-<version> (ROCm 10's packages, whose /opt/rocm
  may lack the llvm / lib links) and /opt/rocm-<version>, the compiler under lib/llvm/bin as well as llvm/bin, hipBLAS
  under lib/<triplet>/ as well as lib/. Reported from an Ubuntu 26.04 + AI PRO R9700 build that only passed with
  ROCM_PATH pointed at AMD's pip SDK. (The HIP attention needs rocWMMA's headers; Ubuntu 26.04's librocwmma-dev was
  reported broken - the pip SDK or ROCm's own apt repository has them.)
- Tried and not adopted (branch `lane-multi`): the batched verify window's columns answered by one multi-column iGPU
  lane call (bit-identical to one call a column). The batched window went 22.7-27.7 -> 18.8 ms, the pipelined default
  is 16.0: pipelining hides the lane behind the other column's GPU work, batching exposes it every layer, and a
  two-column lane call is only 10% cheaper than two. `Abyss-tools/worklog/2026-10-15_lane-multi.md`.

## v1.3.18 - 2026-10-15

The prompt path's host experts on the iGPU as one fused int8 tensor-core pass, and a faster Q6_K dense GEMV for the
attention-Q6_K pack. Both from the second survey of Strata's derivatives
(`Abyss-tools/worklog/2026-10-15_strata-derivatives-round2.md`; the measurements in
`Abyss-tools/worklog/2026-10-15_igpu-fused-q6k-split.md`).
- iGPU lane, prompts: the RAM-tier experts the APU computes for a prompt chunk go through a fused gate/up + SwiGLU +
  down kernel on the matrix cores (`src/igpu/igpu_fused.cu`, from Maya v1.1.10's `moe_fused_rdna4.cu` by @boxwrench;
  the weights decoded to int8 and multiplied with `v_wmma_i32_16x16x16_iu8`), instead of the one-row dp4a kernels.
  It covers the IQ2_XS / IQ3_XXS experts of DeepSeek V4 Flash at 4096 / 2048 (the MXFP4 down of layers 26 and 42
  stays on the CPU threads as before). A prompt's host call is ~1.6x faster per row (13.7 -> 8.5 us), so the planner gives
  the iGPU more rows (8K: 185K -> 235K of the chunk's expert rows): RTX 5090 + Radeon 8060S, 8K prompts 1,634 / 1,753 ->
  1,767 / 1,888 tok/s (+8%); 32K unchanged (3,136 / 3,295 -> 3,136 / 3,272). The answers agree with the row kernels
  to 2e-3 relative (41 of 41 calls checked), KL against the reference 0.0053 (row kernels 0.0055).
  `STRATA_GLM_IGPU_FUSED=0`: the row kernels; `ABYSS_IGPU_FUSED_CHECK=<n>`: compare the first n calls against them.
- Q6_K dense GEMVs split a row over 4 warps up to 4,096 blocks (`STRATA_GLM_MV_SPLIT=0`: one warp a row, as before).
  nsys on an RTX 5090 with the attn-q6k pack: wo_a 24.1 -> 18.1 us (1.14 -> 1.52 TB/s), wo_b 23.0 -> 19.8, q_b
  19.2 -> 18.5; the pack's decode 11.52 -> 11.14 ms/token (83.2 / 85.8 -> 86.3 / 88.7 tok/s). The Q8_0 pack is
  unaffected (its only Q6_K GEMV is one shared-expert row block). Why the Q6_K pack had gained only 1-2% before: at
  one warp a row the Q6_K dot ran at 1.14-1.20 TB/s against Q8_0's 1.40-1.52, eating two thirds of its 23% fewer
  bytes - the same finding as Indras-Mirror's on a 4090. The sums are not bit-identical to the one-warp kernel's.
- Bench: `bench/kl_pf.sh` scores with the context through the prompt path (`STRATA_GLM_SCORE_PREFILL=1`), the
  yardstick for prompt-path changes; `bench/nsys_dec_pack.sh` and `bench/ksum_grid.py` profile another pack and sum the
  kernels by type and grid.

## v1.3.17 - 2026-10-15

Documentation brought up to date with v1.3.16. No code change.
- README (English and Japanese): the performance table is v1.3.16's (decode ~82-84 tok/s, prompts 8K / 32K / 125K
  ~1,700 / ~3,200 / ~2,800 tok/s) against llama.cpp at its best settings on the same machine (`-ncmoe 32`: 28-30
  tok/s decode, 78-660 prompt), with the other measured setups (RTX 3090, the RTX 5090 + RTX 3090 split, the Radeon
  8060S); the highlights, requirements, roadmap and badges cover the iGPU lane for prompts, speculative decoding on
  by default, the tensor-core long-context kernels, the layer split, the HIP build and the V100 build.
- docs/MANUAL (both languages): the GPU row, the F16-projection GGUFs, the two-card measurements, the iGPU lane's
  prompt use and call deadline, speculative decoding's default and gate, new settings
  (`STRATA_GLM_IGPU_TIMEOUT_MS`, `STRATA_GLM_PREFILL_IGPU`, `STRATA_GLM_PROMOTE_RATIO` / `_MARGIN`,
  `STRATA_GLM_TAIL_SKIP`, `STRATA_DS4_IDX` / `_ATTN`, `STRATA_DS4_SPEC_GATE`), the limitations.
- docs/deepseek4/PORT.md: the "what is new" table gains the tensor-core kernels, the verify window, the iGPU lane,
  HIP, the split and sm_70; the speed table is current with a release-by-release summary; "not done yet" names what
  is left.

## v1.3.16 - 2026-10-15

DeepSeek V4's prompt processing 3-7% faster at 8K-32K tokens, a few decode-path savings, and a full survey of what
is left (every source file read, 23 candidates ranked and then measured one by one:
`Abyss-tools/worklog/2026-10-15_ds4-speedup-survey.md`). The answers are the same.
- Prompt path: the prestage buffer is carved to the chunk's size (T / 128 slots, 16..160: it took all 160 slots =
  1.3 GB for any chunk from 1,024 tokens, which evicted ~2.8 resident experts a layer at 8K;
  `STRATA_GLM_PRESTAGE_CARVE=<slots>`, 0 = all), the prestaged experts are computed in 8-expert pieces as their copies
  land instead of after the last one (`STRATA_GLM_PRESTAGE_PIECES=0`: after the last), the shared expert's GEMMs run
  after the route's sync so the GPU works while the host plans the expert sets (`STRATA_GLM_PREFILL_SH_LATE=0`:
  before the route), and a chunk's embedding rows are dequantized by the worker threads (`STRATA_GLM_PREFILL_EMB_PAR=0`:
  one thread). RTX 5090 (PCIe Gen4 x4) + Ryzen AI Max+ 395 (Radeon 8060S lane), DeepSeek V4 Flash UD-Q2_K_XL, prompts
  of 8K / 32K / 125K tokens, 2 runs each: 1,546 / 1,680 -> 1,629 / 1,757 tok/s; 2,934 / 3,179 -> 3,133 / 3,274;
  2,800 -> 2,813.
- Decode: the lightning indexer's score and top-k launches are skipped while fewer rows are visible than it keeps
  (positions below 2,047: every row is selected anyway, and the attention reads them in position order;
  `STRATA_DS4_IDX_ALL=1` keeps the launches) - 0.3-0.4 ms a token in a 1,024-token answer. KL against the old path's
  own reference over positions 1,000-4,000: 0.0045, as between two runs of it. The attention output's q8_1 (wo_a's
  input) is written by the combine kernel's epilogue, the same bits, 43 launches fewer (`STRATA_DS4_OQ_FUSE=0`: the
  separate launch). The boundary's table updates go through two pinned sets with events instead of a stream sync a
  flush (`STRATA_GLM_BOUNDARY_SYNC=1`: the sync). Typed GEMVs under 256 blocks run 4 warps a row
  (`STRATA_GLM_MV_SPLIT`; nsys: q_a 4.3 -> 3.0 us, the HCA projections 6.3 -> 5.3; wo_b's 512 blocks stay at ~23.5 us,
  the ceiling of its 8.7 KB rows, and the 512-block Q5_K shared gate/up got slower when split, so only the small ones).
  The n-gram gate's acceptance also learns from the proposals it drops: the dropped draft's first token against the
  single token sampled instead (`STRATA_DS4_SPEC_GATE_LEARN=0`: windows only). 1,024-token answers, prose / code, the
  2nd-3rd of 3 runs: v1.3.15 81.6 / 82.3 -> 82-84 tok/s, within the bench's +-2%.
- Measured and left off (the switches stay): the request's x written at the route kernel's start
  (`STRATA_GLM_ROUTE_EARLY=1` - the system fence, not the store, is what the lane's request waits for); the iGPU lane
  reading x from the registered request ring and writing its sum in place, with a host-mapped completion counter
  (`ABYSS_IGPU_DIRECT=1` - a call is 109-117 us either way: the iGPU's own time, ~20 us + 41 an expert); hc3 without
  its second grid barrier (`STRATA_GLM_HC_GRAM=1` - 8.5 -> 8.1 us, the sums round differently); the batched verify
  window (`STRATA_DS4_SPEC_PIPE=0`: 73 against 90 tok/s on swe-bench prompts, its lane calls serialise); the lane's
  keep-alive off (`STRATA_GLM_IGPU_POKE=0`: -10%); `STRATA_GLM_VRAM_EVICT=lru`, `STRATA_GLM_CPU_WAIT_BLOCKS=16` and
  `STRATA_GLM_PROMOTE_FILL=200` (no change).
- Investigation counters: `STRATA_GLM_SVC_STAT=1` prints the service thread's time per route (the lane call, the
  bookkeeping around it, long spin gaps), `ABYSS_IGPU_POKE_STAT=1` the keep-alive's query and launch times.
- A pack with `attn_q_b` / `attn_output_a` / `attn_output_b` re-quantized to Q6_K (`Abyss-tools/bench/requant_dense.py`
  - llama-quantize refuses an i-quant model's dense tensors without an imatrix) decodes +1-2% with 129 more VRAM
  slots at KL +0.0022 (0.0060 -> 0.0082, both on the CPU lane). Not adopted; a quality call.

## v1.3.15 - 2026-10-14

DeepSeek V4 decodes 3-5% faster with the iGPU / CPU lane: the lane's request goes out earlier in each layer, and the
route kernel is shorter. The answers are the same.
- Each MoE layer ran the router and the shared expert's gate/up as one GEMV before the route. The shared expert does
  not need the route, so its gate/up now runs after it (`STRATA_GLM_SH_LATE=0` gives the old order). The route, and
  with it the request to the iGPU / CPU lane, goes out one GEMV earlier (DeepSeek V4's shared gate/up is 18 MB of
  Q8_0, ~12 us a layer), and the shared expert runs while the lane computes. The same for the n-gram verify windows.
  The sums are the same, only their order on the GPU changes: KL with 4-token verify windows is identical with and
  without it (0.005462).
- The route kernel's plan (the weights, which RAM-tier experts go to the lane, the spare slots for fetched experts)
  was made by one thread walking the route's entries: ~7 of its ~17 us. One lane of a warp now takes each entry,
  with the same decisions.
- Measured on an RTX 5090 (PCIe Gen4 x4) + Ryzen AI Max+ 395 (Radeon 8060S lane), DeepSeek V4 Flash UD-Q2_K_XL,
  greedy, 1,024-token answers, the 2nd and 3rd of 3 runs, prose / code, tok/s: v1.3.14 79.0 / 78.3, the late shared
  expert 81.3 / 81.4, with the parallel plan 81.5 / 81.7, with the verify windows 81.6 / 82.3. KL (3K prompt + 1K
  scored against the RTX 5090 reference) 0.0055-0.0057, as before.
- GLM-5.3-Flash takes the same path (Maya-S v2, the same machine, the 2nd of 2 runs, prose / code): 43.6 / 41.4 ->
  44.6 / 42.8 tok/s.
- Tried and dropped: the CPU threads computing part of each iGPU lane call (the ff columns split between the two).
  The two together read ~230 GB/s against the iGPU's 150 at a 2-expert call, but during decode the CPU's share ran at
  ~0.22 ms an expert (0.105 when timed alone) plus ~40 us a call, so even 1/8 of the columns took as long as the whole
  call on the iGPU: 78 -> 58-65 tok/s.

## v1.3.14 - 2026-10-14

The iGPU lane no longer loses the decode to the CPU lane for a whole session because of one stall at startup.
- At startup the lane is timed against the CPU threads and kept only when faster. When the kernel driver held the
  iGPU's queue during that timing (re-validating the registered RAM tier: kernel log "svm_range_restore_work hogged
  CPU"), the calls past the 20 ms deadline went to the CPU threads, so the "iGPU" median timed the CPU. Seen:
  a call held 11 s right after the start, "an expert 0.129 ms vs 0.124 on the CPU -> the CPU", and DeepSeek V4
  decoded at 41-49 tok/s for the whole session instead of ~80.
- A timing now counts only when every call ran on the iGPU and none took 500 ms. Otherwise the lane is timed again
  once it takes 16 calls in a row, for up to `STRATA_GLM_IGPU_CAL_WAIT_S` (60) seconds in all. If no clean timing
  comes by then, the iGPU keeps the lane: each stalled call goes to the CPU threads anyway, and the lane is back at
  full speed once the stalls stop. The split between the CPU and PCIe is then planned on the CPU's time.
- The stalls came in bursts for a few minutes after a start (0.1-2.8 s each, while the kernel reclaimed and
  compacted memory after the 90 GB tier warm-up) and then stopped.
- Measured on an RTX 5090 (PCIe Gen4 x4) + Ryzen AI Max+ 395, DeepSeek V4 Flash UD-Q2_K_XL, greedy, 1,024-token
  answers, prose / code, tok/s: the session stuck on the CPU lane 49.3 / 41.5; after the change, once the stalls
  stopped, 83.7 / 81.8 and 81.0 / 82.9.
  Tested with `ABYSS_IGPU_TEST_STALL=3000:300 STRATA_GLM_IGPU_CAL_WAIT_S=10`: the lane is timed again, then kept.

## v1.3.13 - 2026-10-14

Decode is faster when the CPU / iGPU lane is on: the expert tiers swap far fewer experts between VRAM and RAM.
DeepSeek V4 decodes 4-12% faster, GLM-5.3-Flash 4-6%.
- The background tier moves swapped a resident expert for a RAM-only one when its use count passed 2 x the
  resident's + 4. The counts halve about every 130 tokens, so they are small and noisy, and DeepSeek V4 swapped
  ~2.7 experts a token. Each swap is two 8 MB copies, and each copy holds the PCIe link for ~1.2 ms (Gen4 x4).
  Behind those copies, the route's ring writes and the CPU lane's answer reads (host-mapped) waited: nsys showed
  the route kernel at 16 us without a copy and 94-133 us with one, and `moe_cpu_wait` at 42 -> 119-154 us.
- The swap rule is now 4 x + 16 (`STRATA_GLM_PROMOTE_RATIO` / `STRATA_GLM_PROMOTE_MARGIN`; 2 / 4 gives the old
  rule). DeepSeek V4: ~0.5 swaps a token, VRAM hit 75% -> 68%, and still faster. Turning the swaps off altogether
  is slower (prose 60 tok/s: the tiers then keep what the prompt left).
- The route reads a device copy of the route error flag instead of the host-mapped one (a PCIe read each layer).
- Measured on an RTX 5090 (PCIe Gen4 x4) + Ryzen AI Max+ 395 (iGPU lane on), greedy, 1,024-token answers to a
  prose and a code prompt, 3 runs each (old -> new, tok/s):
  - DeepSeek V4 Flash UD-Q2_K_XL, n-gram spec on (the default): prose 75.4 -> 78.8, code 70.1 -> 78.5.
    Spec off: prose 76.5 -> 79.1, code 70.4 -> 77.3.
  - GLM-5.3-Flash Maya-S-v2: prose 41.0 -> 42.6, code 38.3 -> 40.6.
  - KL against bench/ref4k_cuda.bin (3K + 1K scored): 0.0058 -> 0.0056 (run-to-run spread ~0.005).
- Looked at and not taken: a Q8_0 GEMV that stages rows through shared memory with 16-byte loads (the current one
  is at 1.4-1.6 TB/s already, <5% to gain), and table edits passed in the launch instead of a staged copy + sync
  (no measurable gain once the copies are few).

## v1.3.12 - 2026-10-14

Speedups taken from Project Maya v1.1.2-v1.1.7 (mw00/project-maya). Abyss was copied from Maya v1.0.31. GLM-5.3-Flash
and DeepSeek V4 both decode about 10% faster, and long GLM prompts run up to 16% faster.
- **Sampler: top_k in linear time** (Maya #92). Before, the sampler re-scanned the vocabulary once for every one of
  the top_k tokens. Now one histogram pass finds the k-th score, the tokens above it are ranked in shared memory, and
  the draw chain runs on one thread. Maya measured 1.5 -> 0.07 ms a sample (top_k 20, RTX 4090); `top_k` 0 means 64,
  which took 11 ms. The picks are the same, token for token (sampler_parity). With n-gram speculative decode on, a
  sampled request samples every verified column, so it gains more there.
- **The prompt path's experts compute the SwiGLU inside the MMQ gate/up product** (Maya #166,
  `third_party/ggml/mmq-fused-swiglu.patch`, applied to a copy of ggml-cuda in the build tree). It writes T x n_ff
  floats instead of T x 2 n_ff, and the separate SwiGLU kernel goes away. glm_mmq_glu_test: 14 types are
  bit-identical where the product does not split K (RTX 5090 and 3090). Where it does split K, the floats round
  differently. GLM KL against the old build is 0.0155; the old build against itself with the iGPU/CPU lanes on is
  0.0172. DeepSeek V4 KL is 0.0043; its run-to-run spread is 0.0046. `STRATA_GLM_MMQ_GLU=0` gives the old kernel,
  bit for bit.
- **CPU experts** (Maya #154, #166):
  - When several tokens share an expert (a prompt's CPU experts, verify windows), the CPU decodes each weight row
    once for all of them (`STRATA_GLM_CPU_MT`). This covers IQ2_XXS/XS/S, IQ3_XXS, IQ3_S, IQ4_XS, Q4_K and Q5_K, which
    includes DeepSeek V4's IQ2_XS gate/up and IQ3_XXS down.
  - The one-token IQ3_S / IQ3_XXS rows are cheaper (`STRATA_GLM_CPU_1T`).
  - Both give the same bits as ggml's dot. Each format is checked at first use, and native_rows_exact_test covers 8
    formats.
- From Maya, smaller items:
  - INT8 latent cache (`--kv int8`): the prompt attention converts its records to FP16 for less, with the same bits
    (glm_lat8_test: 0 differ).
  - A new conversation's state is zeroed on the stream that reads it (Maya #133, Windows + AMD).
  - Windows: the engine opts out of EcoQoS power throttling (Maya #114).
  - Opt-in `STRATA_GLM_VRAM_EVICT=lru` and `STRATA_GLM_DISK_READERS=<n>`.
- Measured on an RTX 5090 + Ryzen AI Max+ 395 (iGPU lane on), old build against new, server, sampled (temperature 1.0,
  top_p 0.95):
  - GLM-5.3-Flash Maya-S-v2, decode at 8K/16K: 51-55 -> 55-61 tok/s. Prompt: 8K 840 -> 875, 16K 1,478 -> 1,543,
    32K 2,073 -> 2,398 tok/s.
  - DeepSeek V4 Flash UD-Q2_K_XL, decode: 60-69 -> 67-77 tok/s. Prompt at 8K/16K: about +3%.
- Not taken (for now): DFlash2, EXL3, `--parallel`, KV streaming and the conversation checkpoints. Notes are in
  Abyss-tools/worklog/2026-10-14_maya-v1.1.7-review.md.

## v1.3.11 - 2026-10-14

On AMD (HIP), the prompt path's attention runs on RDNA's WMMA too (gfx11 / gfx12), as the CUDA path's has since
v1.3.5.
- `attend_batch_wmma_kernel`: a block takes one token and 16 heads, and each of its 8 waves owns 64 of the 512 dims.
  Per tile of 16 keys: the keys go into LDS as FP16, and each wave computes its QK partial with Q as FP16 hi + lo.
  Q is scaled into FP16's range by a power of two per head and wave, as glm_batch.cu's `mla_attn_wmma2_kernel` does.
  The partials are summed for the online softmax, whose max and sum start from the sink. P goes into P.V as FP16
  hi + lo. Then the output is normalised and de-roped.
- Radeon 8060S, 1,024 tokens at 30K: 37.0 -> 15.3 ms per CSA layer, 18.5 -> 10.0 ms per HCA layer. Relative RMS
  against the old kernel: 4e-6. `STRATA_DS4_ATTN=1` keeps the old kernel, and the WMMA check no longer depends on
  `STRATA_DS4_IDX`.
- In the engine (Radeon 8060S alone, a 7K prompt): `ds4_attn` 6.5 -> 3.0 s, prompt 143 -> 154 tok/s. On the APU the
  prompt is bound by the experts (part of them read from the SSD). KL against the RTX 5090 build over the next 1,000
  tokens: 0.0050 (the old kernel: 0.0058).

## v1.3.10 - 2026-10-14

DeepSeek V4 Flash runs on AMD Radeon (HIP, experimental). There is also a race fix in the indexer's top-k select,
which CUDA builds have had since v1.3.3.
- Fix (CUDA and HIP): the select's cut-off search read `s_need` in the same phase in which the thread that found the
  bin rewrites it. A thread of another warp that read it late could take a second bin, so a token's cells came out
  with a slot unset. On a Radeon 8060S this showed up when a token took all of its rows (n_vis 494, top 494): the
  last cell held garbage, and the prompt attention made an illegal access. `need` is now read with the prefix and
  the mask, before the search. No such failure had been seen on CUDA. 2,000 random cases still match the old code
  exactly on both backends. KL on the RTX 5090: 0.0050 (run-to-run spread 0.0050).
- The HIP build links again:
  - MXFP4 (DeepSeek V4's down rows) is added to the HIP MMQ instances.
  - The engine library links the prefill library (`strata::prefill::Gemm`) on HIP.
- `./setup-deepseek.sh --backend hip` sets up an AMD card: abyss.py's HIP checks, ROCm and build for gfx1100,
  gfx1201 and gfx1151 build `build-hip/strata`, and the config gets `"backend": "hip"` with abyss.py's HIP
  settings. It is chosen by itself when there is no NVIDIA GPU.
- On HIP the indexer uses RDNA's WMMA (gfx11 / gfx12; `STRATA_DS4_IDX=1` keeps the older kernels), with Q as FP16
  hi + lo as on CUDA. Radeon 8060S:
  - decode: 1,121 -> 174 us at 31K rows (a 125K context), 95 -> 23 at 2K;
  - prompt: 1,024 tokens at 120K, 214 -> 36.5 ms. Q is converted to FP16 hi / lo in place once, and the waves read
    it without LDS (through LDS, a token at a time: 143 ms);
  - the scores match the SIMT kernels within 3e-6.
- The new top-k select (v1.3.3) now runs on HIP too (wave32 only, which the HIP backend requires): 34 -> 14 us at 2K
  keys, 61 -> 27 at 31K, with the same cells.
- Checked on the Radeon 8060S alone (`build-hip/strata`; a 3K prompt, then 1,000 tokens scored against the RTX 5090
  build): KL 0.0052. The prompt was 94 -> 120 tok/s; the experts dominate a prompt that short. Discrete Radeon cards
  have not been run yet, and the prompt attention still uses the older kernel on HIP.

## v1.3.9 - 2026-10-14

The engine builds for the V100 (sm_70). It has not run on a V100 yet.
- The floor drops from compute capability 7.5 to 7.0: in CMake's architecture check and in the device check at
  start. The code below Ampere already took the fallbacks Turing uses (every arch switch in the kernels is at
  sm_80: mma.sync, ldmatrix, cp.async), so nothing else had to change. The DeepSeek V4 path's BF16 projections run
  as FP32 (`cublasSgemm`), not as cuBLAS BF16 GEMMs, which need sm_80.
- CUDA 13 cannot compile for sm_70, so a V100 needs a CUDA 12 toolkit. `./setup-deepseek.sh --nvcc
  /usr/local/cuda-12.9/bin/nvcc` builds with it. Setup stops with that advice when a chosen GPU is below sm_75 and
  the nvcc it finds is CUDA 13. CUDA 12.9 also builds an RTX 50's `120a`, so a V100 beside an RTX 50 builds too.
- Checked: the whole `strata` target compiles and links for `CMAKE_CUDA_ARCHITECTURES=70` with CUDA 12.9.86
  (conda-forge). The usual CUDA 13.3 build (`86;120a`) is unchanged: 8K prompt 1,548 tok/s, decode 60.7 on the RTX
  5090 + iGPU.
- The tensor-core kernels from v1.3.2-1.3.5 (the indexer's scores, the prompt attention) need sm_80. A V100 keeps
  the older kernels for them, as AMD does.

## v1.3.8 - 2026-10-14

Fix: the iGPU lane was off whenever the server picked a GPU other than nvidia-smi's 0 (`"gpu": [1]`, or a split such
as `[1, 0]`). The server sets `CUDA_VISIBLE_DEVICES` to choose the NVIDIA cards, and ROCm's HIP reads it too when
`HIP_VISIBLE_DEVICES` is unset. So HIP saw no device ("no ROCm-capable device is detected") and the RAM-tier experts
went to the CPU threads. Before it opens the lane, the engine now sets `HIP_VISIBLE_DEVICES` to the iGPU (HIP's 0, or
`STRATA_GLM_IGPU_DEVICE`) when neither it nor `ROCR_VISIBLE_DEVICES` is set.

Measured on the first two-card setup: RTX 5090 (PCIe Gen4 x4) + RTX 3090 in a USB4 enclosure (~3.8 GB/s, 2.1 ms an
expert) + Ryzen AI Max+ 395. Server, second run of each prompt, prompt / decode tok/s:

| setup | 8K | 32K |
|---|---|---|
| RTX 5090 + iGPU | 1,683 / 58.5 | 2,876 / 65.9 |
| RTX 5090 alone | 1,356 / 48.6 | 2,728 / 56.0 |
| RTX 5090 + RTX 3090 + iGPU (split 27/16) | 1,506 / 54.9 | 1,985 / 56.7 |
| RTX 5090 + RTX 3090 | 1,046 / 48.1 | 1,594 / 48.7 |
| RTX 3090 + iGPU | 756 / 37.1 | 1,129 / 37.0 |

- With this fix, the split gains +44% prompt (8K) and +14% decode from the iGPU. It still trails one RTX 5090 with
  the iGPU.
  - VRAM hit rises from 82% to 91%, but the 3090's 16 layers and the hop between the cards cost more than the
    RAM-tier experts saved (decode 15.2 -> 17.6 ms a token).
  - A split has no n-gram speculative decoding.
  - Its prompt path has no prestage (each part's pool is not lent to it).
  - The 3090's USB4 link is slow (2.1 ms an expert against 1.2).
- The USB4 enclosure dropped off the bus once under a 32K prompt ("GPU lost from the bus"). CUDA did not come back
  until a reboot.

## v1.3.7 - 2026-10-14

DeepSeek V4 Flash on several NVIDIA GPUs (layer split): a routing fix, the iGPU lane for every part, and setup for two
cards. Checked as a two-part split on one RTX 5090. Speed on two physical cards has not been measured yet.
- Fix: the prompt path's hash-routed layers (the first 3 of DeepSeek V4) read the chunk's token ids from a buffer that
  only the first part filled. A later part starting below layer 3 routed those layers by whatever its buffer held:
  with a split at layer 2, the KL against the unsplit model was 0.0078 instead of 0.0027. Each later part now gets the
  ids on its own stream before its layers.
- The iGPU lane (the decode's RAM-tier experts, and the prompt's host experts from v1.3.6) now runs with a split too.
  Each part registers its RAM tier with the one lane. A decode call that finds the lane busy goes to the CPU, so it
  never waits; a prompt call waits its turn. Two parts on one GPU, a 7K prompt: 917 -> 1,324 tok/s (+44%), scoring
  4.5 -> 2.6 ms a token.
- `./setup-deepseek.sh --gpus 0,1` (or `--gpus all`) builds for every chosen GPU's architecture (`86;120a` for an RTX
  3090 beside an RTX 5090; a build for other architectures is configured again) and writes `"gpu": [0, 1]`, so the
  server starts with `--layer-split auto`. Without it, setup picks one GPU as before and points to the option when
  it finds more.
- `STRATA_GLM_SPLIT_SAME=1` (test): the parts of a split may share a GPU (with `STRATA_GLM_DEVS=0,0` and
  `STRATA_GLM_POOL_GB` to leave room for both). Splits 22 and 2 on one GPU match the unsplit model: KL 0.0037 /
  0.0027 over 1,000 tokens after a 7K prompt (two unsplit runs: 0.0028).
- docs/MANUAL: a "Several GPUs" section. Limits: a split has no n-gram speculative decoding; sm_70 (V100) does not
  build; AMD is not supported.

## v1.3.6 - 2026-10-14

**The prompt's RAM-tier experts run on the APU's integrated GPU: DeepSeek V4 Flash reads an 8K prompt ~1,310 -> ~1,830
tok/s (+40%), 16K ~1,800 -> ~2,440 (+35%), 32K ~2,990 -> ~3,170 (Ryzen AI Max+ 395 + RTX 5090, one prompt, engine
time).** A short prompt was bound by the experts that live in RAM. Each one either crossed PCIe Gen4 x4 to the RTX
5090 (~1.2 ms) or was computed by the 16 CPU threads (~1.2 ms for ~20 rows), while the iGPU sat idle during prompts.
- `abyss_igpu_rows` (libabyss_igpu.so, interface 2) computes the experts the prompt plan gives the host on the
  iGPU. It uses glm_fast's light row kernels and reads the RAM tier in place: ~20 rows in ~0.22 ms against ~1.2 ms on
  the CPU threads. The kernels gain IQ2_XS (DeepSeek V4's gate/up type) and a per-expert address table. The planner
  costs these experts at the iGPU's rate (its own learned per-row time and a 0.045 ms fixed cost), so it gives the
  iGPU more of them. A call it cannot take, or a layer type without the kernels (the two MXFP4-down layers), goes to
  the CPU threads as before. `STRATA_GLM_PREFILL_IGPU=0` keeps the CPU threads.
- Only the host rows' activations go down to RAM, gathered on the GPU after the plan, instead of the chunk's whole x.
  For a 32K chunk that is ~70 MB instead of 537 MB per layer. The iGPU's cost per row at 32K dropped from 51 to 12 us.
- Fix: the prestage count's adaptation read the prestage events after the next layer's prestage had recorded them
  again. They were never ready, so the count stayed at the buffer's cap (91 experts a layer at 8K) whatever the lanes
  did. It is now measured before that, and it also lowers the count when the copies are the later lane while some
  experts were staged after the plan. With the iGPU, the first prompt starts from 32 experts per 8K tokens (24-40
  measured best); a second prompt starts from what the first learned.
- Server (`/v1/chat/completions`, prompt speed, two runs): 8K 1,565 / 1,638 tok/s (v1.3.5: ~1,170), 32K 2,939 / 2,943
  (v1.3.5: ~2,630). Decode unchanged (62-66 tok/s).
- KL on the prompt path: 3,000 tokens against llama.cpp 0.0089. 60K prompt, last 1,000 tokens against the v1.3.1
  reference: 0.0032 (the run-to-run spread is 0.0030-0.0036). Without the iGPU (`STRATA_GLM_PREFILL_IGPU=0`): 8K
  ~1,330 tok/s, the same as v1.3.5.

## v1.3.5 - 2026-10-14

**DeepSeek V4 Flash reads long prompts faster again: 125K tokens ~1,770 -> ~2,540 tok/s (+44%), 32K ~2,060 ->
~2,630, 64K ~1,970 -> ~2,740.** After v1.3.4 the attention was the largest part of a 125K prompt (`ds4_attn` 31 of 70 s
of GPU time). Its kernel ran on the CUDA cores in FP32, a block per token and 8 heads, so each key was read 8 times
per token.
- The prompt path's attention now runs on tensor cores (`mma.sync` m16n8k16, `ldmatrix.trans` for P.V). A block takes
  one token and 16 heads. Each of its 4 warps owns 128 of the 512 dims of Q, K (= V) and O. For each tile of 32 keys,
  the warps' QK partials meet in shared memory for the online softmax, and P comes back for P.V. Q and P go in as FP16
  hi + lo pairs, so the result keeps ~FP32 accuracy (relative RMS 2e-6 against the old kernel). 4,096 tokens at 120K
  (RTX 5090): 26.7 -> 6.5 ms per CSA layer, 37.1 -> 8.9 ms per HCA layer. sm_80+ CUDA builds; `STRATA_DS4_ATTN=1`
  keeps the old kernel; AMD keeps the old kernel.
- Profile (`STRATA_GLM_PREFILL_PROF`, 125K prompt): `ds4_attn` 31.2 -> 7.9 s, GPU time 70 -> 48 s. The indexer (7.6
  s) and the experts are now about the same size.
- Server, prompt speed: 8K ~1,150 -> 1,170 (the experts dominate there), 32K 2,062 -> 2,632, 64K 1,965 -> 2,738, 125K
  1,769 -> 2,539 tok/s. Decode unchanged.
- KL on the prompt path, against llama.cpp over 3,000 tokens: 0.0091 (v1.3.4: 0.0093). 60K prompt, last 1,000 tokens
  scored against the v1.3.1 reference: 0.0034 (the same as two runs of the old code).

## v1.3.4 - 2026-10-13

**DeepSeek V4 Flash reads long prompts faster: 125K tokens ~1,010 -> ~1,770 tok/s (+75%), 32K ~1,650 -> ~2,060.** At
125K, half the prompt's GPU time went to the lightning indexer's scores (`ds4_index` 61 of 123 s). The kernel was
FP32 on the CUDA cores (tiles of 32 tokens x 32 rows, one head at a time). The top-512 select took under 1% of it.
- The prompt path's scores now run on tensor cores like the decode path's (v1.3.2): `mma.sync` m16n8k16, a warp holds
  2 x 16 K rows as A fragments for a tile of 32 tokens. Each token's Q (64 heads x 128) goes into shared memory as hi +
  lo FP16 halves, double buffered so the next token converts while this one multiplies. 4,096 tokens at 120K: 177 ->
  21.6 ms per CSA layer (RTX 5090, ~190 TFLOPS). Same conditions as v1.3.2: sm_80+ CUDA builds, 64 heads;
  `STRATA_DS4_IDX=1` keeps the old kernels for both paths; AMD keeps the old kernel.
- Profile (`STRATA_GLM_PREFILL_PROF`, 125K prompt): `ds4_index` 61.3 -> 7.8 s, stream time 123 -> 70 s. The attention
  (`ds4_attn`, 31 s) is now the largest part.
- Server, prompt speed old -> new: 8K ~1,150 -> 1,130 (unchanged), 32K 1,651 -> 2,062, 64K ~1,440 -> 1,965, 125K
  1,008 -> 1,769 tok/s. Decode unchanged.
- Accuracy: synthetic 4,096-token batches match the old kernel within 1.7e-6 with the same top-512 sets. KL, prompt
  path: 3,000 tokens against llama.cpp 0.009260 (old 0.009259). 60K prompt, last 1,000 tokens scored against the
  v1.3.1 reference: 0.0034 (two runs of the old code: 0.0034).

## v1.3.3 - 2026-10-13

The indexer's top-k select is faster, with the same result: DeepSeek V4 Flash decodes ~0.3 ms per token faster at any
context length (the CSA layers' `ds_index` 8K 0.75 -> 0.47 ms/token, 125K 1.18 -> 0.85). GLM's DSA select uses the same
code and gets the same speedup.
- `src/kernels/cuda/dsa_topk.cuh` (one block of 1,024 threads; 2K keys 18 -> 12 us, 31K 35 -> 25, RTX 5090):
  - three radix passes of 12, 12 and 8 bits (was four of 8). The cut-off bin is found by a block scan, not a serial
    walk by thread 0.
  - the winners above the cut are collected with coalesced reads, in any order. The keys equal to the cut are too when
    all of them are taken; otherwise the first ones in pool order are taken, as before.
  - each winner's rank is a count of the winners larger than it (the values are distinct). The bitonic sort and its 45
    barriers are gone.
- The same cells as before: 2,000 random cases (up to 40K keys, top 1-1,024, kpool 1 and 4, heavy ties, negatives,
  zeros) match the old code exactly. `glm_dsa_parity`, `glm_layer_parity` and `glm_model_test` pass. KL against
  llama.cpp (3,000 tokens) 0.0092, within run-to-run noise (0.0087-0.0093). HIP builds keep the old version (the new
  one assumes 32-lane warps).

## v1.3.2 - 2026-10-13

**DeepSeek V4 Flash decodes long contexts faster: 125K tokens ~58 -> ~66 tok/s (+14%).** The decode slowed with the
context because of the lightning indexer. Its score kernel (one warp per compressed row, the 64 heads reduced one by
one with shuffles) took 84 us per CSA layer at 125K (31K rows), ~2.7 ms per token over the 21 CSA layers. The expert
side (VRAM hit ~80%, the CPU/iGPU lanes) does not change with the context.
- The scores now run on tensor cores (sm_80+, CUDA builds): K[n x 128] . Q^T as `mma.sync` m16n8k16, a warp per 16 rows
  x the 64 heads. Q goes in as two FP16 halves (hi + lo), so it keeps ~22 bits; the K rows are FP16 already. 84 -> 12
  us at 31K rows, 13 -> 7 at 2K (RTX 5090). `STRATA_DS4_IDX=1` keeps the old kernel (A/B). AMD and pre-sm_80 cards keep
  the old kernel.
- Profile (`STRATA_GLM_PROF`, GPU time, 125K): `ds_index` 2.68 -> 1.18 ms/token, total 17.83 -> 16.29. Most of what is
  left is the top-512 select (~41 us per layer at 31K rows).
- Server, 1,024 generated tokens, two runs each, old -> new: 125K 56.0 / 59.3 -> 65.2 / 66.1 tok/s; 32K 66.9 / 67.1 ->
  69.0 / 69.5; 8K unchanged (65-69). Prompt speed unchanged (the prompt path's indexer is a separate kernel).
- Accuracy: against llama.cpp (3,000 tokens one at a time) KL 0.0087 (old kernel 0.0088), the same top token 94.7%. At
  60K (prefill, then 1,000 tokens scored), new against old KL 0.0030, while two runs of the old kernel differ by
  0.0034 (the CPU/GPU split of the experts varies between runs).
- Measured on the same machine: llama.cpp (bed0a8566, `-ngl 99 -ncmoe 32 -lm none`) decodes this GGUF at 28-30 tok/s
  (0-32K) and reads prompts at 78 tok/s (ubatch 512) to 660 (ubatch 8192).

## v1.3.1 - 2026-10-12

**The iGPU lane no longer freezes a decode for seconds.** v1.3.0's pieces did not prevent it: after the host had been
idle for hours, single lane calls took 0.7-18 s again (five in ~1,000 tokens), and the decode waited for each. The
kernel driver can hold the iGPU's queues that long (re-validating registered memory); the cause outside the engine is
still open (it did not reproduce on demand: refilled page cache, pressure from reading 87 GB, engine restarts).
- A lane call now has a deadline (`STRATA_GLM_IGPU_TIMEOUT_MS`, 20 ms; a call is ~0.05-0.3 ms; 0: wait as before). A late
  call is left running and the CPU lane computes that layer's experts, and every call after it until the late one has
  finished; then the iGPU takes them again. The log says both: `iGPU lane: a call is still running after 20 ms ...` and
  `... the late call finished after N ms; M calls went to the CPU meanwhile`. The CPU's sum is the CPU lane's (within
  rounding of the iGPU's).
- Checked with an injected 3 s hold (`ABYSS_IGPU_TEST_STALL=<ms>:<every n calls>`, test only), 500 sampled tokens:
  20.81 ms/token waiting for it, 16.25 with the deadline (4,967 calls on the CPU meanwhile); 14.85 without a hold, as
  before.
- Measured and left out: writing the between-token table edits and the embedding where the GPU reads them in place
  (no small copies, no host wait): the GPU's reads queue behind the tier's 8 MB expert copies on PCIe Gen4 x4 the same
  way, and 20 swe-bench prompts decoded at 13.05 ms/token against 13.04.

## v1.3.0 - 2026-10-12

DeepSeek V4 Flash writes code ~13% faster by default: the n-gram speculative decoding of v1.2.0 is now on. Measured on
an RTX 5090 on PCIe Gen4 x4 + Ryzen AI Max+ 395 (iGPU lane), unsloth UD-Q2_K_XL, per-request medians:

| | v1.2.2 (`STRATA_DS4_SPEC=0`) | v1.3.0 |
| --- | --- | --- |
| Decode, 20 swe-bench prompts | 14.69 ms/token (68.1 tok/s) | 13.04 (76.7 tok/s) |
| Decode, two ~1,500-token prose answers | 13.82 ms/token (72.4 tok/s) | 13.77 (72.6 tok/s) |

- **Speculative decoding on by default** (`STRATA_DS4_SPEC=1`; `=0` off, `=2`/`3` drafts more), behind a gate: the
  request's acceptance and the window / single-token times as running means, and a proposal is dropped while
  (1 + k p) t_one < t_win (every 16th still runs). Code is accepted 84-85%, prose ~52%; here the gate measured neutral
  (it guards machines where a window costs more, e.g. the CPU lane alone). `STRATA_DS4_SPEC_GATE=0`: no gate.
  `ds4 spec:` in the log counts the gated proposals. The output follows the same distribution as without it.
- **The iGPU lane registers the RAM tier in ~256 MB pieces** (`STRATA_GLM_IGPU_CHUNK_MB`, 0: one range a RAM class).
  The kernel driver keeps a registered range in step with the page tables without pinning it, and a change to any of
  its pages stops the iGPU's queues until the whole range is faulted in again: with one 69 GB range a token waited
  10+ s about once a request on a fragmented host (kernel log: `svm_range_restore_work` /
  `amdgpu_amdkfd_restore_userptr_worker hogged CPU`), and registering it took 0.4-13 s (283 pieces: 0.4-1.3 s). The
  stalls did not reproduce afterwards with either form, so the pieces bound the damage rather than proving a cure; a
  lane call over 500 ms now prints `iGPU lane: a call took N ms`. Decode unchanged (14.70 ms/token).
- Measured and left out: CUDA graphs for the token path. nsys of a sampled decode: 13.92 ms/token, the GPU idle 0.90
  of them - ~976 inter-kernel gaps of 0.5 us (0.50 ms, what a graph could shrink, ~2%) and ~0.4 ms between tokens
  (argmax -> host -> the next token's table edits and embedding), which a graph does not touch.

## v1.2.2 - 2026-10-12

**DeepSeek V4 GGUFs with F16 projections** (antirez's deepseek-v4-gguf, and huihui-ai's abliterated copies such as
`huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF` `DeepSeek-V4-Flash-Q2-0731.gguf`: IQ2_XXS gate/up, Q2_K
down, Q8_0 attention, F16 router and small projections) pack and run:
- `tools/iq_pack.py --compat-bf16` rounds an F16 router and the F16 projections the GEMVs serve (the compressors', the
  indexer's q_b) to BF16, and widens the F16 rows read as F32 (indexer.proj, the compressors' ape) to F32 (lossless).
  token_embd stays F16 (its rows are dequantized on the host). unsloth's packs are unchanged.
- The engine serves a deepseek4 pack's BF16 rows on the BF16 GEMV where a quantized projection was expected, and reads
  F16 rows of dense.bin.
- These files say `nextn_predict_layers 1` but ship no draft block: a last block with hyper-connections is a trunk
  layer (pack and engine), so all 43 layers run.

Measured on an RTX 5090 on PCIe Gen4 x4 + Ryzen AI Max+ 395 with the huihui-ai Q2 file (80.8 GiB): KL against llama.cpp
on the same file 0.0075 over 3000 tokens (unsloth UD-Q2_K_XL: 0.0096), ppl 6.32 on the same text (UD-Q2_K_XL: 6.37);
decode on 20 swe-bench prompts 14.63 ms/token (68.4 tok/s; UD-Q2_K_XL 14.80); VRAM holds 3,397 of its experts (3,010).
The unsloth pack's output is the same as before (KL 0.00001, the CPU lane's split). Set up by hand for now:
`.venv/bin/python tools/iq_pack.py --gguf <file> --out <dir>/pack-dsv4 --compat-bf16`, then a copy of
`abyss-dsv4-q2_k_xl.json` / `run-abyss-dsv4-q2_k_xl.sh` pointing at that pack.

## v1.2.1 - 2026-10-12

**Route tail skip (opt-in, from Strata 0.1.42's `STRATA_ROUTE_TAIL_SKIP`):** `STRATA_GLM_TAIL_SKIP=<r>` leaves a
route's experts from rank r + 1 on (of deepseek4's 6) out when they are not in VRAM: no PCIe pull, no CPU / iGPU lane,
no disk wait; the route's other experts keep the router's weights. `STRATA_GLM_TAIL_SKIP_W=<share>` leaves out (also)
those whose share of the route's probability is below it. The token path and the verify window both; not the
hash-routed layers (ids in hash order) nor the draft block (`STRATA_GLM_MTP_KEEP`, as before). Off when the pool holds
90% or more of the experts (`STRATA_GLM_TAIL_SKIP_MAX_CACHED=<percent>`). It changes the answers, so it is off by
default; `glm stat` shows the skipped experts a token. Measured on an RTX 5090 on PCIe Gen4 x4 + Ryzen AI Max+ 395,
UD-Q2_K_XL (VRAM holds 27.7% of the experts), 20 swe-bench prompts, per-request medians:

| | CPU lane (`STRATA_GLM_IGPU=0`) | iGPU lane (default) | KL vs off, 3000 tokens | ppl |
| --- | --- | --- | --- | --- |
| off | 19.18 ms/token (52.2 tok/s) | 14.80 (67.6) | - | 6.365 |
| `STRATA_GLM_TAIL_SKIP=5` (rank 6) | 17.97 (55.6, +6%) | 14.83 (67.4) | 0.0141 | 6.414 |
| `STRATA_GLM_TAIL_SKIP_W=0.11` | - | 14.42 (69.3, +3%) | 0.0156 | 6.396 |
| `STRATA_GLM_TAIL_SKIP=4` (ranks 5-6) | 16.55 (60.4, +16%) | 13.86 (72.2, +7%) | 0.0226 | 6.409 |

For scale: this engine against llama.cpp on the same file is KL 0.0096; the run twice, 0.000000. With 6 experts a
route each carries more of the token than Strata's 10 do (its KL was 0.003-0.006): try it where the CPU lane is the
bottleneck (no iGPU, a slow CPU), and measure.

## v1.2.0 - 2026-10-11

DeepSeek V4 Flash decodes ~30% faster on a Ryzen AI Max+ 395: its integrated GPU computes the RAM-tier experts that
the CPU threads computed. Opt-in speculative decoding adds ~12% on coding prompts. Measured on an RTX 5090 on PCIe
Gen4 x4 + Ryzen AI Max+ 395 (94 GB RAM), unsloth UD-Q2_K_XL:

| | v1.1.2 | v1.2.0 | v1.2.0 + `STRATA_DS4_SPEC=1` |
| --- | --- | --- | --- |
| Decode, two answers of ~1,500 tokens (greedy) | 17.8 ms/token (56 tok/s) | 13.7 (73 tok/s) | 13.7 (prose: drafts rarely match) |
| Decode, 20 swe-bench prompts (up to 800 tokens) | 19.6 ms/token (51 tok/s) | 15.1 (66 tok/s) | 12.8 (78 tok/s) |
| ... the same, prompt + answer, end to end | 41.5 tok/s | 52.9 tok/s | 59.3 tok/s |
| KL vs llama.cpp, 3000 tokens one by one | 0.0096 | 0.0093 | - |
| Prompt, 5.6K tokens (later requests) | 1,065-1,069 tok/s | 1,053-1,054 tok/s | - |

- **The iGPU lane** (`src/igpu/igpu_lane.cpp`, `libabyss_igpu.so`): the Radeon 8060S reads the same DRAM as the CPU at
  ~240 GB/s (the CPU lane's threads ~75), including memory CUDA pinned. At start the engine registers its RAM tier with
  HIP (69.7 GB in ~0.35 s, nothing copied) and the CPU lane's calls go to the iGPU: glm_fast's expert kernels built
  for gfx1151 by ROCm's `hipcc`, loaded at run time (CUDA and HIP never share a translation unit). 0.046 ms an expert
  against the CPU's 0.107; its sums equal the RTX 5090's (relative L2 0.00000; 0.0137 against a double reference, the
  CPU lane's 0.0265). A down kernel for 1-2 experts (a decode's usual call; the device's has a warp an expert of 8:
  55 -> 18 us) and a no-op kernel every 50 us while a decode runs (a call after 0.3 ms idle took 103 us instead of 57)
  bring a token's lane from 8.6 to 3.7 ms. The start-up log says which lane computes and why; the CPU stays when it
  is faster, or without the library or a HIP device. `STRATA_GLM_IGPU=0` off, `=1` on even when slower.
  The prompt path keeps the CPU threads (and plans with the CPU's own time). GLM-5.3-Flash too: 29.1 -> 18.1 ms a
  token scored one by one; with the iGPU lane off, its logits still equal Maya 1.0.24's (KL 0.000000).
- **Speculative decoding from the text itself** (`STRATA_DS4_SPEC=<k>`, 1-3; off by default): the last 3 tokens'
  (`STRATA_DS4_SPEC_MIN`) latest earlier occurrence in the prompt or the answer proposes the next k, and the token with
  its proposal goes through one verify window (`src/core/dsv4_verify.cu`); a column whose sample matches the proposal
  is kept, the first mismatch's sample is the next token, and the rejected tail is dropped by position (the attention
  ring keeps 4 spare rows). The output is the token-at-a-time loop's (the same sampler on the same logits). The window
  runs as k + 1 staggered CUDA streams: token t + 1's attention of a layer runs while the RAM-tier lane computes token
  t's experts of it (`STRATA_DS4_SPEC_PIPE=0`: one batched stream, 36.8 against 29.1 ms for 2 tokens without the iGPU).
  swe-bench prompts: 77-83% of the drafts accepted, 1.4-1.8 tokens a step; prose ~50%, about even. `k = 1` measured
  the fastest. `STRATA_GLM_SCORE_VERIFY=<T>` scores through T-token windows (KL 0.000000 against the token path with
  every expert on the GPU).
- The CPU lane's and the disk misses' answers are kept 4 deep (a window has several routes of a layer in flight).
- The build makes `libabyss_igpu.so` when ROCm's `hipcc` is found (`-DSTRATA_IGPU_ARCH`, default `gfx1151`;
  `-DSTRATA_IGPU_LANE=OFF` skips it).

## v1.1.2 - 2026-10-11

The CPU lane no longer sends an expert over PCIe on a start whose calibration came out a little slow. GLM-5.3-Flash
(Maya-S-v2-IQ2_XXS, 8 experts a token) on a Ryzen AI Max+ 395 + RTX 5090 on PCIe Gen4 x4 got "of 8 RAM-tier experts
the CPU takes 7" on about one start in three, and decoded 25% slower until the next start (forced, 1,000 tokens:
33.7 ms a token with 7 of 8 on the CPU, 26.9 with all 8; VRAM hits 43% vs 61%).

- **The split is planned with the decode's own PCIe path**: a RAM-tier expert reaches the GPU by `moe_fetch`'s SMs
  reading host memory, not by the copy engine the calibration timed - 1.12 ms against 1.04 an expert on that link.
  The calibration now times both and plans with the slower one. All 8 stay on the CPU up to 0.135 ms an expert
  there (0.125 before; this machine times 0.105-0.126).
- **The CPU's time is the median of 48 runs** instead of the mean of 16.
  `STRATA_GLM_CPU_CAL_VERBOSE=1` prints the calibration's spread and both PCIe timings.
- `STRATA_GLM_PREFILL_VERBOSE=1` also prints what a prompt borrows from the expert pool and how many experts leave
  VRAM for it.

GLM-5.3-Flash runs on Abyss as on Maya: 2,000 tokens scored one by one give the same log-probabilities as Maya
1.0.24's engine (KL 0.000000), at 27.1 ms a token against Maya's 29.3.

## v1.1.1 - 2026-10-11

`/health` says which server this is: `"service": "abyss"` (Strata's says `"strata"`). Abyss and Maya answer the same
`/health` and llama.cpp's `/props` (whose `model_path` is only `pack/full`), so a client could not tell them apart -
a benchmark tool recorded Abyss as llama.cpp. Nothing else changes.

## v1.1.0 - 2026-10-11

DeepSeek V4 Flash writes ~19% faster and reads a 5.6K-token prompt ~55% faster. Measured on an RTX 5090 on PCIe Gen4
x4 + Ryzen AI Max+ 395 (94 GB RAM), unsloth UD-Q2_K_XL, every expert in VRAM or pinned RAM:

| | v1.0 (port) | v1.1.0 |
| --- | --- | --- |
| Decode, two ~1,500-token answers (greedy) | 20.8 ms/token (48 tok/s) | 17.4-17.7 ms/token (57 tok/s) |
| Prompt, 5.6K tokens (first after start / later) | 690 tok/s | 1,018 / 1,075-1,090 tok/s |
| Prompt, 18K tokens | 1,752 tok/s | 1,716-1,734 tok/s |
| KL vs llama.cpp, 3000 tokens one by one (CPU lane on) | 0.0093 | 0.0091 |
| KL vs llama.cpp, 2499 through the prompt path + 500 | 0.0059 | 0.0054 |

- **The token path's attention, split over key chunks** (`dsv4_fast.cu`): every head reads the same 512-wide keys, so a
  block now takes 8 heads and a chunk of keys (one tile read from L2 for all 8) and a combine kernel merges the chunks
  with the sink - one block a head had left most of the GPU idle. 3.2 -> 0.84 ms a token at ~1.4K context; equal to
  the old kernel to 4e-7. `STRATA_DS4_ATTN=1`: the old kernel; `STRATA_DS4_ATTN_CHECK=1`: both, compared now and then.
- **A lighter route kernel** (`glm_fast.cu`, every MoE layer, GLM too): the host-mapped error flag is read without
  stalling the block, the top-k is one warp's (no block barriers a rank), and the route's table entries are read by one
  thread an expert instead of one after another by thread 0.
- **The CPU lane's answer read by 8 blocks with 16-byte loads**: 30 -> 9 us a layer over PCIe Gen4 x4.
  `STRATA_GLM_CPU_WAIT_BLOCKS=<n>` sets the blocks. With the route kernel above: 18.4 -> 17.5 ms a token.
- **The prompt path's CPU rows by the AVX-512 i-quant kernels** (`glm_prefill.cu`): IQ2_XXS/XS/S and IQ3_XXS/S rows of
  an expert with 2+ prompt tokens go through `iq_avx512.cpp` (a weight block decoded once for 4 tokens) instead of
  ggml's one-token dots: 1.2 -> 0.52 ms a token through a DeepSeek V4 expert on one core, 0.12 -> 0.04 ms a row in the
  lane. Only on CPUs with AVX-512 F/BW/VL/VNNI/VBMI (Zen 4/5, Intel Ice Lake and later); the others keep ggml's.
  `STRATA_GLM_PREFILL_IQ512=0`: ggml's.
- **deepseek4 prestages adaptively by default**: with the faster CPU rows the host takes the lightly routed experts that
  the full 160-expert prestage buffer would have copied over PCIe. It starts at 64 experts for a 5.6K-token chunk (the
  whole buffer from ~14K tokens on) and learns from there. `STRATA_GLM_PRESTAGE_ADAPT=0`: the whole buffer, as before.
- `STRATA_GLM_PROF` splits out `ds_index` (the indexer) from `ds_attend`.

Tried and left out: the AVX-512 kernels for ONE token (the decode lane) are slower than ggml's on Zen 5 too (1.6 vs
1.2 ms an expert on one core), and 24 / 32 lane threads instead of 16 were equal / slower (12.1 vs 7.0 ms a token in
the lane at 32: SMT siblings spin against the service thread).

## v1.0.31 - 2026-10-10

The expert prefetch can now pay on a RAM-bound split: three settings choose when its copy starts, how many GPU blocks it
takes and which predictions it copies.

- **The expert prefetch, tunable** (#84 by @sociolog): `STRATA_GLM_PREFETCH_N` copies the next layer's predicted
  experts that are not in VRAM into that layer's spares while a layer computes. As it was, it cost more than it saved
  on a RAM-bound split. Three settings for it; the prefetch stays off by default and nothing else changes:
  - `STRATA_GLM_PREFETCH_AT=route|fetch|cpu`: when the copy starts - at once (as before), after this layer's own
    fetch, or after its CPU-lane answer;
  - `STRATA_GLM_PREFETCH_BLOCKS=<n>`: the copy's GPU blocks (half the SMs before);
  - `STRATA_GLM_PREFETCH_RANK=<n>`: copy only the prediction's first n ranks. Its first guesses are right almost
    always, its 5th-8th mostly not.

  Maya-M on an RTX 3090 + 3060 at 21K context: decode (writing the answer) 19.64 tok/s without the prefetch, 20.22
  with `N=1 AT=fetch BLOCKS=4 RANK=2`, and +4.7% over a 40-turn agent-like conversation. The README has a row for it.

## v1.0.30 - 2026-10-10

AMD GPUs set up on Windows too, Windows reads its experts from the SSD much faster, the CPU lane times itself
correctly at start, and the logs say Maya.

- **AMD on Windows, Strix Halo / Gorgon Halo included (experimental)** (#69): `START-MAYA.bat --backend hip` sets
  Maya up on Windows 10/11 as `./maya.sh --backend hip` does on Linux, for Ryzen AI Max 300 / 400 (Radeon 8050S /
  8060S / 8065S) and the RX 7900 XT / XTX and RX 9070 / R9700.
  - **Setup:**
    - finds the GPUs with ROCm's hipInfo, else Windows' display adapters;
    - uses AMD's HIP SDK for Windows, or offers AMD's ROCm SDK wheels in `.venv` when there is none;
    - compiles the engine with ROCm's clang and Ninja in Visual Studio's environment (2022 or 2026).

    The SDK's HIP runtime goes next to `strata.exe`, so the driver's own copy in System32 is not loaded instead.
  - **The engine on a Windows APU:**
    - The GPU's memory is a fixed carve-out that Windows does not count as RAM, so the engine sizes it like a
      discrete card.
    - An expert pool larger than one allocation goes into several.
    - The engine submits queued GPU work when no route has come for 1 ms, because Windows' HIP runtime holds
      launches until the host waits.

    Linux and NVIDIA are unchanged.
  - **Two HIP checks were wrong on every platform:** `hip_glm_handoff` passed a removed argument, and
    `hip_prefill_mmq_parity` raced its output sentinel.
  - **Checked on a Ryzen AI Max+ PRO 495 / Radeon 8065S under Windows 11:** setup builds and packs Maya-L, every
    expert stays in the GPU pool, answers, long prompts and thinking come out right, and all 15 HIP checks pass.
- **Windows reads experts from the SSD much faster** (#73 by @klvnblst, for #65 and #67):
  - **Unmapping:** while a model file stays mapped, Windows sends every direct read of it down a slower path. The
    engine now unmaps the files once they are loaded, moving the token table (~0.7 GB) to RAM. On the RTX 5090
    Laptop of #67, decode is about 3.5x as fast. `STRATA_GLM_KEEP_MAP=1` keeps the old behaviour.
  - **CPUs without SMT** (Intel Core Ultra 200 and others): the CPU lane uses two thirds of the cores, leaving the
    rest for the disk readers and the service thread.
  - **`--calibrate`:** the CPU lane threads it leaves out sleep instead of spinning.
- **The CPU lane times itself on experts read from RAM** (#70 by @merbanan): its start-up timing reused 4 experts
  that stayed in the CPU's cache, so it took the CPU for faster than it is during decode. It now goes through up to
  64. That gave +6% decode on a Ryzen 9 7950X with a Tesla V100, and the same plan at every start.
- **Maya's name in the logs** (#74):
  - the engine's lines say `maya generate:` / `maya serve:` / `maya:`, and the server's say `[maya]`;
  - the installer no longer shows Strata's name;
  - the Windows stall dump is `maya-stall-<pid>.dmp`.

  Settings (`STRATA_*`), flags, the API, file and binary names are unchanged, and the README keeps the credit.
- **A restart no longer mistakes the dashboard's saved settings for a model's config** (#69). After a settings
  change, `./maya.sh` / `START-MAYA.bat` could pick `maya-<model>.shared-settings.json` and fail.
- **The README's Windows setup, step by step** (#71 by @handmade0octopus):
  - the Visual Studio 2022 Build Tools components, and the difference between `nvidia-smi`'s CUDA version and an
    installed toolkit;
  - copy-paste commands for checking the PC, setting up on an NVMe drive and calibrating;
  - page-file advice that matches what `--check` asks for (about VRAM + 9 GB, since "System managed" can stay too
    small).
- The README has no speed figure under 12 tokens/s.
- Checked:
  - on 1x and 2x Tesla V100: the builds and the parity tests; identical greedy tokens with the CPU lane off, on one
    GPU and on two; decode and prefill (reading the prompt) the same within run-to-run variation (#70 picks a
    different CPU-lane plan at some starts); the CPU thread count changed per request, as `--calibrate` does; the
    server end to end; the setup screen in a real terminal; the logs saying `[maya]` and `maya ...`;
  - on Windows: the engine compiles (MSVC with CUDA 12.8);
  - AMD on Windows: on the author's Ryzen AI Max+ PRO 495;
  - the GitHub checks.

## v1.0.29 - 2026-10-09

Documentation: the README and v1.0.28's entry below describe the model files' new architecture name more plainly.

## v1.0.28 - 2026-10-09

The model files name their architecture with the standard GGUF name, `glm5-next`.

- **The model files say `glm5-next`.** Maya's quants had copied `glm5next`, an early spelling, from the file they
  were made from. Standard GGUF tools don't recognize that spelling.
  - On Hugging Face, the first file of each model is published again with only its header changed. Every byte of
    the model stays where it was, so installed models and their packs keep working, and nothing has to be
    downloaded again. Setup accepts both versions of that file.
  - `tools/gguf_fix_arch.py` renames a file downloaded earlier, in place (only the header).
  - The quantizer writes `glm5-next` from now on, and Maya reads either name.
- Checked:
  - Maya's greedy tokens are the same with the new header (Tesla V100, CPU lane off);
  - the four new files are the old ones byte for byte after the header;
  - the GitHub checks.

## v1.0.27 - 2026-10-09

The setup and Maya get a screen of their own in the terminal, the crash after a long prompt is fixed, a stuck prompt
ends with an error instead of waiting forever, and Maya builds on Windows again.

- **A screen of its own** (#58 by @needmorevram): in a terminal, the setup's 8 steps are tabs, the questions are
  arrow-key menus, and the compile and the download show their progress. Maya then runs on the same screen, with
  a loading view while the experts warm up, a dashboard of the Monitor's numbers and its log.
  - `./setup.sh`, `./maya.sh`, `START-MAYA.bat` and `run-maya-<model>.sh` all open it.
  - Its library (Textual) ships with Maya as 9 pure-Python wheels in `third_party/wheels`, byte-identical to
    PyPI's, and goes into `.venv` with nothing downloaded.
  - `--plain`, a pipe or a service keeps the plain text exactly as before, and `--yes` keeps a plain setup.
- **No more "illegal memory access" after a long prompt, and no more 0xC0000005 on Windows / AMD** (#39 by
  @boxwrench). Three races in the expert tiers, shared by NVIDIA and AMD:
  - the pinned buffers of the expert tables' updates were rewritten while their copy to the GPU was still in
    flight;
  - a background move of an expert could land in a VRAM slot that a prompt was borrowing at the same moment;
  - a route of NaN scores indexed the expert tables. It now fails that request with an error, and the engine
    goes on.
  - Confirmed on an RTX 3090 + 3060 with every RAM-tier expert on the CPU (#50: the 8K prompt after decoding
    crashed, now it runs), and on Windows with an RX 7900 XTX (#53: 5 crashes in 8 sessions, now 0 in 8).
- **A stuck prompt ends** (#40): the engine's watchdog now watches Maya's GLM engine too. A request that finishes
  no prompt layer and no token for 3 minutes ends the engine with an error that names where it stopped (on Windows
  also `strata-stall-<pid>.dmp`, every thread's stack), and the next request starts it again. Before, a prompt that
  stalled with the GPU idle waited forever. `STRATA_WATCHDOG_S` sets the time; `0` turns it off.
- **Windows builds again** (#54 by @jerem91150, the same fix as #55 by @noahark): `--kv int8` called `setenv`,
  which Windows' C library doesn't have.
- **Thinking off stays off** (#54): GLM-5.3's template opens a thinking block even when thinking is off, so the
  answer could land inside it. The server now closes it. This applies to every GPU.
- **AMD**:
  - Windows: `tools\hip\build_maya_windows.bat` builds the engine with TheRock's ROCm wheels (#54).
  - RDNA4 prompts +8-10% (#38 by @boxwrench): a wave32 WMMA attention kernel, the default on R9700 / RX 9070.
  - `STRATA_GLM_DISK_QD` goes up to 32 reads in flight (#54). The default stays 2.
- **One GPU skips the last layer's unused outputs in a prompt** (#61 by @boxwrench): +1% prefill (reading the
  prompt), the same output. `STRATA_GLM_PREFILL_TAIL_SKIP=0` turns it off.
- **Robustness** (#62 by @needmorevram):
  - The embedding rows of F16, BF16, Q2_K, Q4_1 and Q5_1 GGUFs are found (every token read row 0 before). The
    published Maya models are unaffected.
  - A prompt whose staging events cannot be created runs token by token instead of failing.
  - Damaged image-embedding files and vision measurements are refused instead of ending the engine or the
    server.
- **Switch chats while an answer is written** (#66 by @needmorevram): another chat, or a new one, opens; the
  answer goes on in its own chat and shows again when that chat is opened. Esc stops it only in its own chat.
- **A setup for another context keeps your tuning** (#57): it used the engine's defaults until `--calibrate` ran
  again. It now takes the settings tuned at the nearest context.
- Checked on 1x and 2x Tesla V100, and on Windows:
  - the builds and the parity tests, and the engine's Windows build (MSVC with CUDA 12.8);
  - identical greedy tokens with the CPU lane off, on one GPU and on two, and with the KDA graphs on;
  - #62's and #59's tests;
  - #50's sequence (every RAM-tier expert on the CPU, answers, then 2K and 8K prompts) on one GPU and on two;
  - the watchdog: a forced stall ends with an error that names its stage, and the next request is answered. A
    10-second watchdog never fired over long prompts and decode;
  - decode and prefill (reading the prompt) the same at the defaults, prefill +1% on one GPU (#61);
  - the server end to end: answers, a picture, thinking, context reloads, a graceful stop;
  - #58's screen in a real terminal, with its tests on Linux and Windows, and #66 in a browser against the
    running model;
  - the GitHub checks.

## v1.0.26 - 2026-10-09

Documentation.

- **A speed report in the README** (#63 by @npc97): Maya-L on an RTX 4090 24 GB with 192 GB of DDR5 under
  Windows 11. It holds every expert in VRAM or RAM and decodes at 16.4 tokens/s, with prefill at 1371 tokens/s
  on an 8K-token prompt. The Windows section no longer says Maya has not run on a Windows PC with an NVIDIA GPU,
  and notes that CUDA 13 works for RTX 20 and newer.
- **`STRATA_GLM_CPU_PIN` is Linux-only** (#64 by @needmorevram): on Windows the engine pins nothing, and one GPU is
  never pinned anywhere, so `0` changes nothing there. A two-socket Windows PC whose decode differs between starts
  differs in something else; compare the starts' `CPU lane:` log lines.

## v1.0.25 - 2026-10-09

Splits of three GPUs or more use all of the CPU; `--calibrate` measures realistic text and leaves your usage
profile alone; opt-in CUDA graphs for one GPU.

- **One CPU pool for a split of 3+ GPUs** (#60 by @needmorevram):
  - A decode token visits the cards in turn, so a pool per card (`cores / GPUs` threads each) left most of the
    CPU idle. One pool now serves them all, sized as one GPU's.
  - Maya-L on 9 GPUs (8x RTX 5060 Ti + RTX 3090, 2x Xeon Gold 6152): decode 25.7 -> 30.7 tokens/s, an 8K prompt
    1265 -> 1466 tokens/s.
  - One and two GPUs are unchanged. `STRATA_GLM_CPU_SHARED=1` shares the pool on two GPUs too; on 2x Tesla V100
    with one CPU socket it measured 30.0 against 28.5 tokens/s. `0` gives every GPU its own pool.
- **`--calibrate` on realistic text** (#60):
  - Every measurement answers prompts it has not seen before (60 of them). Repeating three prompts had left the
    experts in VRAM fitted to just those answers, so the CPU lane's settings were measured with almost nothing
    on the CPU.
  - The tuning's engine reads and writes a copy of your expert usage file, so its answers no longer change the
    order the next start warms up in.
- **Opt-in CUDA graphs for one GPU** (#59 by @handmade0octopus): `STRATA_GLM_KDA_GRAPH=1` replays the recurrent
  (KDA) decode layers as CUDA graphs. The output is bit-identical.
  - RTX 4090D: +2.35% decode.
  - Tesla V100: the same speed.
  - Off by default; two GPUs, MTP, profiling, prefetch and lookahead keep the direct path.
- Checked on 1x and 2x Tesla V100:
  - the builds and the parity tests;
  - identical greedy tokens with each change at its default, and with the graphs on (CPU lane on and off);
  - #59's two tests: 192 bitwise comparisons on the real model;
  - decode and prefill unchanged at the defaults;
  - a full `--calibrate` run (380 s) that left the usage file byte-for-byte unchanged;
  - the GitHub checks.

## v1.0.24 - 2026-10-09

Better drafts for speculative decode on two GPUs or more.

- **Better drafts** (#48 by @sociolog): the MTP draft block used to skip every expert it found missing from
  VRAM. It now computes the missing ones among its route's 2 best-scored experts and skips the rest. It is the
  default, set with `STRATA_GLM_MTP_KEEP=<n>`: `0` restores the old behaviour, `STRATA_GLM_MTP_MISS=1` fetches
  every missing expert.
  - On 2x Tesla V100: 82-83% of drafts accepted against 76-81%, decode at 128K 28.0-28.2 against
    27.0-28.1 tokens/s, and never slower.
  - On an RTX 3090 + 3060 split: 19.5 -> 20.6 tokens/s.
  - The output stays the model's own: greedy tokens are identical with the CPU lane off.
- Checked on 2x V100: the build, the tokens above, five draft settings over two rounds, and the GitHub checks.

## v1.0.23 - 2026-10-09

- **The AMD build can no longer break unseen.** `tools/check_hip_compat.py` lists every CUDA runtime and cuBLAS
  name the AMD (HIP) build compiles, and fails on one without a mapping in `include/strata/hip_compat/`. It
  flags v1.0.21's three.
- **Checks on every push and pull request:** a GitHub Actions workflow runs that check, the server tests and the
  installer and tools tests. No GPU is needed.
- **Every AMD target builds:** the debug tool `peer_probe` gets its three peer-copy mappings. The installer never
  built it.

## v1.0.22 - 2026-10-09

- **AMD builds again (#52 by @boxwrench):** v1.0.21 did not compile with HIP: #44's split search and CPU pinning use
  three device queries (`cudaDeviceGetPCIBusId`, the memory clock and the memory bus width) that had no HIP mapping.
  NVIDIA builds are unchanged. Built for gfx1100, gfx1201 and gfx1151 (ROCm 7.2.1) and run end to end on a Strix Halo.

## v1.0.21 - 2026-10-09

Eight community pull requests, with every default fitted to the machine it runs on:
- a split search and prompt chunks sized from the expert pool;
- a smaller KV cache at long context (decode 6-8% faster at 128K);
- pictures from the web app read again;
- a loop guard, and a thinking budget that leaves room for the answer;
- speculative decode on three GPUs or more;
- CPUs without AVX2;
- conversations that outlive a restart.

- **The layer split and prompt chunks (#44 by @needmorevram):**
  - `--layer-split auto` prices every placement on 2-4 GPUs before loading (each card's room for experts, the share
    of routes they cover, its memory bandwidth, the host's RAM speed) and takes the fastest.
  - `--prefill auto | N | 0` sizes the prompt chunk from what each card's expert pool can lend: up to 32768 tokens on
    one GPU and 8192 on a split, each the faster there. One V100 reads 26K-token prompts at 562 tok/s; two V100s at
    719.
  - On hosts with two NUMA nodes or more, each GPU's CPU lane runs on CPUs of its own (2x Xeon Gold 6152: decode
    15.4 -> 19.8 tok/s). On one node it stays off, which measured faster there. `STRATA_GLM_CPU_PIN=1/0` forces it.
  - The RAM tier takes huge pages only when free 2 MB blocks cover it, so a second card's tier no longer waits over
    10 minutes in kernel compaction.
- **A smaller KV cache (#41 by @merbanan):**
  - The attention indexer keeps its keys for the last 8,256 positions instead of the whole context, with the same
    output. At 128K that is 1.28 GB more VRAM for experts. Decode: one V100 with Maya-S24 18.2 -> 19.4 tok/s; two
    V100s with Maya-S 26.3 -> 28.2.
  - `STRATA_GLM_KV_INT8=1` (opt-in) stores the attention latents in INT8: another 0.65 GB, 19.6 tok/s on the same V100.
    Its KL divergence from the FP8 model is the same as without it (0.436 against 0.431).
  - Conversation slots saved by earlier versions are read again once.
- **Pictures from the web app (#42 by @tanutanu56):** the engine received them without their image data, so the model
  answered about a black picture. Fixed.
- **Loop guard (#33 by @ksanislo):** a reply that repeats one short pattern for 256 tokens while thinking (1024 while
  answering) is closed: the thinking ends, a tool call's argument ends, or else the turn. `STRATA_GLM_LOOP_THINK` /
  `_ANSWER` set the lengths; `0` = off.
- **Thinking budget (#34 by @ksanislo):** the budget leaves the answer at least a quarter of the request's `max_tokens`
  (and at least 1024 tokens), so a small `max_tokens` no longer ends in thinking with no answer.
- **Speculative decode on three GPUs or more (#31 by @ksanislo):** the parts split into a head and a tail group (4x
  Tesla T4: decode 15.0 -> 22.6 tok/s). Two GPUs: the same as before.
- **CPUs without AVX2 (#32 by @ksanislo):** the engine runs there; its CPU lane uses ggml's kernels. The setup warns
  instead of refusing.
- **Conversations that outlive a restart (#35 by @ksanislo):** `STRATA_GLM_SLOT_KEEP=1` (opt-in) keeps the saved
  conversations across starts. A stop (SIGTERM, Ctrl+C) lets the running answer finish its thinking and answer first.
- **Reloads wait for the GPUs:** a context reload, or a restart after a crash, starts the new engine only once the GPUs
  have freed the old one's memory. The driver releases it a few seconds after the process ends, and the split search
  measures the cards before anything loads. The server waits until no GPU lists the old engine (NVML), or 5 s where it
  can't tell.
- **INT8 shown as such:** with `STRATA_GLM_KV_INT8=1`, the engine reports `kv=int8` and About says "8-bit (INT8)
  attention cache".
- **Housekeeping:**
  - No third-party quant names in the tree.
  - `setup.py` keeps only the helpers Maya's installer uses.
  - The start line names the thinking level GLM gets.
- Checked on 1x and 2x Tesla V100:
  - the 239 Python tests and the CUDA parity tests;
  - greedy tokens, identical with and without the new KV cache;
  - prompts, decode, and KL against FP8;
  - the server end to end: an exact answer, a picture through the web app's path, a small `max_tokens` with
    thinking, the context reload 128K -> 16K -> 128K, and a graceful stop.

## v1.0.20 - 2026-10-09

- Housekeeping: Maya-S24's files on Hugging Face are labelled IQ2_XXS_S, apart from Maya-S; the files themselves are
  unchanged. A Maya-S24 already downloaded keeps its names; older setups need this update to download it.

## v1.0.19 - 2026-10-09

- Housekeeping: Maya-M's and Maya-L's files on Hugging Face are named with their quant label (IQ2_S, IQ3_S); the files
  themselves are unchanged. Models already downloaded keep their names; older setups need this update to download
  them.

## v1.0.18 - 2026-10-09

Maya-L, the closest quant yet to the full FP8 model; GLM's own thinking levels; an exact context meter; and an Update
button in the dashboard.

- **Maya-L** (156.3 GB, `./setup.sh --setup --model Maya-L`): Maya-M's recipe one step up - IQ3_S gate/up experts,
  IQ4_XS down projections, Q5_K in the most sensitive MoE layers, Q6_K attention and shared experts. It keeps 99.2% of
  the FP8 model's zero-shot accuracy (the same score on HellaSwag and PIQA), its KL divergence is 35% below Maya-M's on
  the same engine (0.188 vs 0.291), it picks the FP8 model's next token 90% of the time, and it wrote two
  14,000-token answers without a loop. The most demanding of the four: fastest when VRAM and RAM hold most of its
  156 GB. It is the setup's fourth download, in place of the 3.5-bit community file offered before.
  ([bench/results/MAYA-L.md](bench/results/MAYA-L.md))
- **Thinking levels are GLM's own:** Off / Low / High (the default) / Max in the dashboard, `none` / `low` / `high` /
  `max` in the API (OpenAI's `medium` is High and `xhigh` Max; Anthropic budgets under 2K tokens Low, under 8K High,
  else Max). The page's old "High" asked GLM for Max, its longest thinking; settings saved by the old page are
  translated once.
- **The context meter counts exactly:** while it answers, the request's prompt plus the tokens written so far (the
  server's live count); otherwise the chat and the draft through the model's own template and tokenizer
  (`POST /api/tokens`, nothing run; approximate only with pictures). Both numbers in the same K.
- **Updates from the dashboard:** About says when a new release is out, and its **Update** button downloads the new
  code (git), compiles only the engine files that changed and loads the same model with the same settings - the
  model is not downloaded again. A folder that can't update itself (not a git checkout, files changed by hand, a
  server not started by `./setup.sh` or `START-MAYA.bat`) says why and gives the steps by hand. The check asks GitHub
  for the latest release at most every six hours; `MAYA_UPDATE_CHECK=0` turns it off. (`GET`/`POST /api/update`)
- **Monitor shows each GPU figure once:** with two to four GPUs, every hardware card lists each GPU's own value under
  the total (the GPUs table repeated them); from five GPUs the table comes back and the cards keep their totals.
- **Fixed:** saving the Chat settings for other apps failed with a Min-p above 0 (the Creative preset), so nothing
  was shared.
- Checked: Maya-L against the FP8 model on one Tesla V100 (KL, zero-shot, loop test); the thinking levels rendered
  with GLM's own template; the update against a throwaway git origin (a real fast-forward, then the exit maya.py
  starts the new version on) and every case that blocks it; the dashboard in the browser at desktop and phone widths;
  the Python tests (82, 16 of them new) pass.

## v1.0.17 - 2026-10-09

A new dashboard: many chats, the context size from the settings (the model reloads with it), editing and regenerating,
code previews, and a Monitor and About that say what the server runs.

- **Chats:** every chat is kept in the browser (IndexedDB, with its pictures and attached files) and listed in the
  sidebar with search, rename and delete; "New chat" no longer wipes the last one. On a phone the list slides in from
  the left. A user message can be edited and sent again (the answers after it are replaced), the last answer
  regenerated, and an answer that was streaming when the page reloaded is taken back up from the server.
- **Context size in Settings:** a slider from 4K to the model's trained 1M tokens. It shows what the size costs here,
  from the engine's own measurement (2x V100: 3.3 GB at 128K; "about 249 more experts in VRAM" at 64K), warns past
  256K (untested) and asks before it reloads the model (about a minute on 2x V100). Requests from other apps get a
  503 with Retry-After meanwhile; a size that does not start puts the old one back; the new size is saved in the run
  config. (`GET`/`POST /api/context`, from the page itself only.)
- **Instructions per chat** (a system prompt, optionally for new chats too), sampling presets (Precise, Balanced,
  Creative), min-p, and a guard against closing Settings with unsaved changes.
- **While it answers:** a context meter for the chat, the prompt it read and reused under each answer, a "Latest"
  button, Esc to stop, Ctrl+Shift+O for a new chat.
- **Code:** syntax highlighting (bundled, no CDN), wrap, download, and a sandboxed preview for HTML pages.
- **Monitor:** a reading from the last answer is marked as such (it looked live), idle charts say so instead of a flat
  line, a GPUs table on multi-GPU PCs, the request's source (Chat / OpenAI / Anthropic) and prompt speed per row,
  banners for a reload or a hot GPU, tips from the expert tiers, and **Copy report** (versions, PC, engine settings,
  the engine log's telling lines; no API key).
- **About:** the version (it was missing), the trained context, the KV cache as it is (16-bit; it showed "32-bit"),
  each GPU's PCIe link, the model folder and its free disk, and copy-ready snippets for Claude Code, OpenAI-style
  apps and curl.
- **Phones and access:** no zoom into the message box on iOS, 44 px buttons, safe areas, a status word in the header,
  installable as an app (manifest); focus stays in Settings and dialogs, the arrow keys move the thinking level, and
  reduced motion is honoured. Removed: the Qwen-only speed-projection switch.
- Checked: on 2x Tesla V100 with Maya-S, the context changed 128K -> 96K -> 128K through the page's endpoint (about a
  minute each, chat requests answered 503 meanwhile, the run config's other keys unchanged), then answered exactly;
  the page tried in the browser at desktop and phone widths; the server's tests (102 + 9 new) pass.

## v1.0.16 - 2026-10-08

README: the users' speed table keeps the measurements on current versions.

- The 2x TITAN RTX row (v1.0.6, before the prefill and CPU-lane fixes since) is out of the users' table; the GPU
  requirements list 2x CMP 170HX among the users' machines instead.
- The RAM requirement names the machine it refers to (the 2x V100 in the speed table, 30 GB of RAM).

## v1.0.15 - 2026-10-08

Numbers in long prompts are read correctly, and Maya is faster on consumer GPUs: it measures the PCIe link the way
decode uses it and stops re-reading experts from the SSD that are already on their way to RAM; Maya-S24 for 24 GB
cards; Strix Halo; an idle engine no longer holds a CPU core per GPU.

- **Numbers in prompts are read the way GLM reads them (#27, found by @mab776):** the server split every number in a
  prompt into single digits (Qwen's pre-tokenizer) where GLM groups up to three (`504` is `50|4`, not `5|0|4`), so
  in long prompts numbers came back wrong (`504` -> `5504` from ~12K tokens on, every quant). The server now uses the
  pre-tokenizer the model's tokenizer names; numbers in a 32K and a 128K prompt came back exact (6 of 6, measured by
  @mab776). Nothing to redo: the setup already stored it.
- **Maya-S24 (94.7 GB), for cards of 24 GB or less:** Maya-S with its attention and shared experts in 4-bit (Q4_K)
  instead of 6-bit, so about 1.5 GB more of the card holds experts. One Tesla V100 limited to 24 GB: decode (writing
  the answer) 11.8 -> 13.5 tokens/s (+14%); on the full 32 GB 17.2 -> 19.1 (+11%); prefill the same. It keeps 97.7%
  of the FP8 model's zero-shot accuracy (Maya-S: 97.9%). The setup recommends it when every card has 24 GB or less;
  `--model Maya-S24` picks it anywhere. Details: bench/results/MAYA-S24.md.

- **The CPU / PCIe split on consumer GPUs (#24 by @tanutanu56, found in #13):** at start the engine times expert
  copies over PCIe to decide how many RAM-tier experts the CPU computes. It timed them on an idle GPU, and GeForce
  cards hold the link at Gen1 when idle, so the copies measured a link 3-8x slower than decode sees and the CPU took
  experts PCIe moves faster. The timing now runs with the GPU kept busy. (Tesla cards like the V100 keep the link up:
  unchanged there.)
- **Experts leaving VRAM are no longer read again from the SSD (#24):** an expert moved down to RAM at the end of one
  token, and needed again before its copy had landed, was read from the SSD; it now waits for the copy and reads RAM.
  2x Tesla V100 with Maya-S: decode (writing the answer) 28.4 / 28.9 -> 29.8 / 29.7 tokens/s, SSD reads per token
  3.0 -> 2.2.
- **CPU-lane threads per GPU (#24):** `STRATA_GLM_CPU_LANE<n>` sets GPU n's own (a card in a narrow slot wants more
  than one on a wide link). @tanutanu56's RTX 4070 Ti SUPER (PCIe 3.0 x4) + 5070 Ti, Maya-S: decode 8.1 -> 9.1
  tokens/s with the defaults, 14.6 with the threads and the split tuned for that pair.
- **An idle engine no longer holds a CPU core per GPU:** its tier threads spun waiting for work; they now sleep when
  nothing runs (2x V100: 200% -> 2% of a core idle, one V100: 100% -> 1%). Decode and prefill (reading the prompt)
  are unchanged (`STRATA_GLM_SERVICE_IDLE_MS`; 0 = spin always).
- **Sampled answers (temperature > 0)** draw their token on the engine's own GPU stream instead of syncing the whole
  GPU every token (which also waited for the tier copies in flight): 2x V100, temperature 1.0: 26.7 / 26.6 ->
  26.8 / 27.6 tokens/s. On Windows with an AMD GPU this also fixes answers that lost the prompt after their first
  token (#6, found by jerem91150).
- **The MTP draft block from a file:** `STRATA_GLM_MTP_GGUF=<file>` now also replaces a model's own draft block, and
  `tools/maya_quant/mtp_gguf.py` writes a model's draft block alone as a small GGUF - for quants published without
  one, on two GPUs.
- **RAM-resident tier (#25 by @handmade0octopus, opt-in):** `STRATA_GLM_RAM_RESIDENT=1` sizes the pinned RAM tier to
  hold every expert that isn't in VRAM, so nothing is ever evicted to the SSD (the start refuses if RAM can't hold
  them). On an RTX 4090 D with 128 GB RAM, Maya-S at 256K context, a 38K-token prompt: decode 11.8 -> 22.1 tokens/s
  (with #26's `STRATA_GLM_PROMOTE_MIN=6`), no SSD reads.
- **Fewer one-off promotions (#26 by @handmade0octopus, opt-in):** `STRATA_GLM_PROMOTE_MIN=N` keeps a fetched expert
  in VRAM only when it has been routed at least N times lately, so an expert used once no longer evicts a resident
  one. 4090 D: 49.8 -> 40.9 ms a token warm with N=6. On one AMD R9700 (measured by @boxwrench): decode 23.4 ->
  24.5 tokens/s with N=6, 25.1 with the RAM-resident tier too (no SSD reads), 25.55 with #24 as well (+9%).
- **AMD (experimental, by @boxwrench):** faster decode expert kernels on RDNA3 / RDNA3.5 (#19: RX 7900 XT +8.6%, the
  same tokens), the attention's prompt products in FP16 with a matrix-core attention kernel (#16: Strix Halo +16%
  prefill, RX 7900 XT +4% on 4K-token prompts), and `--bench` / `--report` on AMD (#18; `--config FILE` picks one
  installed setup). #24's PCIe timing runs with the GPU awake on AMD too.
- **Strix Halo (#17 by @boxwrench, experimental):** the Ryzen AI Max+ 395's GPU (gfx1151) shares the PC's memory, so
  the engine sizes its expert pool from the free RAM (MemAvailable less 16 GB) instead of the GPU's reported share.
  When the pool holds every expert it keeps them all and the RAM tier only stages reads, instead of holding a second
  copy. The setup finds a versioned ROCm (`/opt/rocm-X.Y.Z`); an APU runs as one GPU. Measured on a Strix Halo with
  Maya-S: prefill (reading the prompt) ~221 tokens/s on 3.5K-token prompts, decode (writing the answer) 17.4
  tokens/s. A HIP build configured by hand without `-DSTRATA_PREFILL_MMQ=ON` now stops at configure and says so,
  instead of failing to link.
- **Windows: the page file the RAM tier needs (#20, reported by @anbow1):** under Windows the card's memory and the
  pinned RAM tier are both charged to RAM + page file, so a small page file capped the RAM tier while RAM sat free
  (an RTX 5090 with 128 GB RAM: 2.4-4.1 tokens/s with an 8 GB page file, 8.4-11 with a 64 GB one). `--check` now
  asks for at least your VRAM + 8 GB and says what a smaller one costs, and the engine warns at start when the
  commit limit, not the free RAM, caps its RAM tier. (Built and run on Linux here; the Windows part is not compiled
  on Windows yet.)
- **API (from Strata 0.1.40):** the client's stop strings end the answer (OpenAI `stop`, Anthropic `stop_sequences`
  with `stop_reason: stop_sequence`; not inside the reasoning); `tool_choice` works - `none` offers no tools, and
  `required` or a named function (Anthropic `any` / `tool`) starts the answer with the call; `response_format:
  json_object` returns the JSON alone; an empty assistant turn is no longer put back into the prompt. The engine's
  start-up warnings show in the server window.
- **Activations can't turn into NaN in the quantizers (from Strata #1448):** the FP16 scale of an activation block
  overflowed to infinity past ~8.3 million; it is clamped at all four places it is written. Below that the output
  is bit-identical.
- **CONTRIBUTING.md:** what makes a useful speed report, bug report or pull request (#21). The README's table of
  speeds measured by users has 2x CMP 170HX with Maya-M at 61.3 tokens/s (#22, @ZackO2o).
- Checked on 2x Tesla V100: the engine's parity tests pass, the prompt attention matches F32 (1.5e-4), the server's
  102 tests and the 40 setup / tokenizer tests pass, tool calls work end to end; decode and prefill unchanged or faster. NVIDIA code is unchanged by the
  AMD pull requests.

## v1.0.14 - 2026-10-08

Maya runs on two AMD GPUs (experimental), with MTP drafting on the second card, contributed by @boxwrench.

- **Two AMD GPUs (#14 by @boxwrench):** `./maya.sh --backend hip --gpus 0,1` splits the layers across two supported
  cards (RX 7900 XT / XTX, Radeon AI PRO R9700 / RX 9070; they may be different models), and the second card runs
  the MTP block that drafts the next token. Setup puts the card with more VRAM first and writes one hipBLASLt tuning
  table per card's architecture (`STRATA_HIPBLASLT_TUNING` takes a `:`-separated list). Measured by @boxwrench with
  Maya-S on an R9700 + RX 7900 XT: decode (writing the answer) about 33 tokens/s against about 15 on the RX 7900 XT
  alone, prefill (reading the prompt) about 490 tokens/s on a 4K-token prompt against about 414. docs/AMD_MAYA.md has
  the details.
- NVIDIA: unchanged - the change is in the AMD setup and the AMD-only prompt GEMM code; every NVIDIA target builds.
- We have no AMD hardware; these results are the contributor's.

## v1.0.13 - 2026-10-08

Tool calls work: GLM-5.3-Flash's function calls reach your apps and coding agents instead of ending the request.

- **Tool calls (issue #5, reported by @xldistance):** GLM-5.x writes its calls in its own form
  (`<tool_call>NAME<arg_key>...</arg_key><arg_value>...</arg_value></tool_call>`), and the server read only another
  model family's form, so every tool call ended the request with "malformed tool call" - in the chat's tools, MCP,
  and every OpenAI or Anthropic client (Claude Code, agents). Both forms are read now, whole and streamed (the call's
  name as soon as it is written, then its arguments piece by piece), and checked end to end: the call, its result
  in the next turn, both APIs.
- **A call only quoted in the answer stays text:** a `<tool_call>` inside a code block or inline code (an example the
  model shows) is never run - an example `rm -rf` must not reach your shell (from Strata #1058).
- **A stuck engine is restarted:** an engine that says nothing for 90 s and in that time uses no CPU, disk or GPU is
  ended, the request gets an error, and the next request starts it again; a silent engine that is working (a long
  prompt on a slow PC) is never ended (`STRATA_ENGINE_STALL_S`, `0` = off; from Strata #1317).
- **The server, from Strata 0.1.41:** up to 256 connections wait to be accepted (an agent opening many at once got
  "connection reset"; `STRATA_HTTP_BACKLOG`); a request body is checked before it is read (a bad Content-Length is a
  400, one over 256 MiB a 413 - `STRATA_MAX_BODY_MIB`), relays' chunked bodies are read, and a malformed request is
  a 400 with its traceback in the server log instead of a dropped connection; `--api-key` takes several keys
  (`KEY1,KEY2`, or a list in the config).
- `--env` now also reaches these server settings (the rest still go to the engine).
- Checked: 92 server tests (12 new for tool calls, 10 for the rest); the engine is unchanged from v1.0.12.

## v1.0.12 - 2026-10-08

Maya runs on up to 16 GPUs and reads long prompts much faster on one GPU, contributed by @needmorevram.

- **Up to 16 GPUs (#12 by @needmorevram):** `--gpus 0,1,2,...` takes up to 16 GPUs (two before). Two GPUs split the
  layers in the middle as before; with more, each GPU takes a share sized to its free VRAM (`STRATA_GLM_SPLIT` or the
  config's `"layer_split"` pin it). Measured by @needmorevram on nine GPUs (8x RTX 5060 Ti 16 GB + an RTX 3090):
  decode (writing the answer) about 28 tokens/s, prefill (reading the prompt) 1,150-1,210 tokens/s on 8K-token
  prompts.
- **Prefill on one GPU:** bigger prompt chunks (up to 32,768 tokens, sized from the free VRAM); each expert's output
  added in as soon as it's computed, so a chunk holds about twice the tokens; the least-used RAM-tier experts computed
  on the CPU while the GPU loads the rest; experts copied to the GPU while the attention runs; faster sparse-attention
  scoring; a tensor-core attention kernel for Ampere and newer. 1x Tesla V100 with Maya-S: 2K / 8K / 16K-token prompts
  267 / 373 / 362 -> 263 / 578 / 620 tokens/s. 2x V100: 293 / 485 / 553 -> 313 / 541 / 674 tokens/s.
- **Decode with most experts in RAM:** hot RAM-tier experts move into VRAM in the background while it answers
  (`STRATA_GLM_PROMOTE`), and on two-socket machines the RAM tier is spread over both sockets' memory. On
  @needmorevram's RTX 3090: 15.3 -> 18.7-19.9 tokens/s. 1x and 2x V100 with Maya-S: unchanged
  (17.1 and about 27.6 tokens/s).
- **`./maya.sh --calibrate`:** measures decode with a few CPU-lane settings (how many RAM-tier experts go over PCIe
  instead of to the CPU, and how many CPU threads) and keeps one only if it is more than 3% faster (README > Tuning).
- **`--models-dir`** names the folder downloaded models go to (`--data-dir` before; older configs still work).
- **Our follow-ups to #12:**
  - Builds for Volta / Turing (V100, RTX 20) and Windows again: the new attention kernel compiles only for Ampere and
    newer, the page-cache drop is Linux-only, and the AVX-512 kernel converts FP16 in a way MSVC accepts.
  - One GPU beside a 6-core CPU kept its decode speed: the CPU's share of each token is cut into about 48 pieces in
    all (`STRATA_GLM_CPU_SPLIT`). One piece per thread, as submitted, took 1x V100 from 17.0 to 14.3 tokens/s; now
    17.1. Machines with 40 or more threads still stream one piece each.
  - The tokenizer remembers what it has read: a 220K-token agent prompt is tokenized in 0.23 s instead of 0.89 s, and
    in 0.013 s when it is sent again with a new turn (the same token ids).
- **Checked:** the engine's parity tests pass on 1x and 2x V100. On held-out text the model's loss stays within
  v1.0.11's run-to-run noise; on one GPU it now varies a little more from run to run.

## v1.0.11 - 2026-10-08

Maya runs on AMD GPUs (experimental): RX 7900 XT / XTX and Radeon AI PRO R9700 / RX 9070 on Linux, contributed by
@boxwrench.

- **AMD (experimental, #7 by @boxwrench):** `./maya.sh --backend hip` builds Maya's engine with ROCm 7 for RX 7900
  XT / XTX (gfx1100) and Radeon AI PRO R9700 / RX 9070 (gfx1201) - Linux, one GPU, text only. Measured by
  @boxwrench with Maya-S: on the R9700, decode (writing the answer) about 20 tokens/s and prefill (reading the
  prompt) up to 490-560 tokens/s; on the RX 7900 XT, prefill up to about 410 tokens/s. Setup and measurements:
  docs/AMD_MAYA.md.
- **Prefill on AMD:** the prompt projections go through hipBLASLt with tuning tables for each card, and the prompt
  runs in bigger sub-batches on cards with room (`STRATA_GLM_PREFILL_SUB`; NVIDIA keeps 256).
- The groundwork, the GPU-to-CPU signal fix, came in v1.0.7 (#2).
- NVIDIA: unchanged - the same tokens as v1.0.10 on 2x Tesla V100, every target builds.
- We have no AMD hardware ourselves; these results are the contributor's. Tell us how it runs on yours (`--bench`
  and `--report` learn AMD in a follow-up).

## v1.0.10 - 2026-10-08

RTX 20-series cards now read prompts on their tensor cores too.

- **Prefill on Turing (RTX 20-series, Titan RTX, Quadro RTX):** the tensor-core prompt attention now keeps its context
  in registers on these cards, so it fits their 64 KB of shared memory. They had been using the slower F32 kernel.
  2x TITAN RTX, an 18,476-token prompt: 62 s -> 58 s of prefill (reading the prompt). By @dummerjindabin (#10).
  Other cards keep the kernel they had; that variant's arithmetic is only correct on Turing and newer (a V100 got the
  attention wrong with it), so it is used only where the other one doesn't fit. A V100 gives the same tokens as
  before.
- **Prefill for 4-bit (Q4_K) weights:** they are converted for the tensor cores the fast way, as 5- and 6-bit ones
  already were - the same values. A test quant with 4-bit attention, 1x V100: 224 / 291 / 285 ->
  268 / 368 / 358 tokens/s at 2k / 8k / 16k, the same tokens. Maya-S and Maya-M are unchanged.

## v1.0.9 - 2026-10-08

Switching between conversations no longer re-reads them: Maya keeps the last few on the SSD.

- **Conversation slots:** when a request doesn't continue the conversation the engine holds - another chat, or an
  agent tool's side request (a title, a sub-task) - that conversation is first set aside in a file on the SSD (its
  attention caches and recurrent state, about 0.15 GB plus 21 KB a token: 0.2 GB at 3K tokens, 0.8 GB at 30K), and a
  later request that continues it takes it back instead of reading its whole prompt again. On 1x Tesla V100 with
  Maya-S, going back to a 3,500-token conversation after another one: prefill (reading the prompt) 10.2 s -> 2.2 s;
  setting it aside 0.09 s, taking it back 0.06 s. A conversation taken back from its file continues token for token as
  it would from memory, and each one still recalls its own details after others ran in between (one GPU, and the
  two-GPU split with the MTP block). Requests still run one at a time. Up to 4 conversations of 1,024 tokens or more, 16 GB on disk, never
  below 8 GB free; `STRATA_GLM_SLOTS=0` turns it off (README > Tuning).
- **Prefill:** the disk read-ahead buffer grows to 3% of free RAM, up to 96 experts (was 2%, up to 64): on two GPUs
  the second one no longer waits on the SSD between layers (2x Tesla V100 with 30 GB RAM: +1% prefill; with 64 GB RAM,
  which reads the SSD less, unchanged).
- README: a table of speeds measured by users (`--bench`), starting with 2x TITAN RTX.
- `--bench` / `--report`: a prompt-only request's engine line no longer shows a decode speed in every case it has
  none.

## v1.0.8 - 2026-10-08

`--bench` warms up before it measures prefill, so speed reports compare fairly.

- **`--bench`:** one unmeasured prompt is read before the 2k and 8k prefill measurements. The 2k figure used to be
  the first prompt after start-up, with the caches still cold, and could read lower than the 8k one (152 vs 238
  tokens/s on a Titan RTX pair). Reported by @dummerjindabin (#8).
- **`--bench` and `--report`:** a request that only read a prompt shows just its prefill time in the engine lines,
  not a meaningless decode speed (it read "96,000 tokens/s").

## v1.0.7 - 2026-10-08

RTX 20-series cards (Turing) no longer crash on long prompts.

- **Fix (RTX 20-series, Titan RTX, Quadro RTX):** any prompt longer than about a thousand tokens stopped the engine, and
  the server restarted it with the conversation lost. The prefill attention asked for 66 KB of shared memory, more
  than the 64 KB Turing GPUs have, and the failure went unnoticed until the engine crashed. Both attention kernels
  now keep the cached rows in shared memory in their 16-bit form - the same values and arithmetic in half the space
  (34 KB, which every NVIDIA card gives without asking). On Turing, decode at long context also gets the faster
  attention it was silently skipping. Other cards: the same answers token for token, the same speed. Found and
  diagnosed by @dummerjindabin (#8).
- **AMD groundwork (#2, by @boxwrench):** the signals the GPU sends the CPU (the expert requests the CPU lane answers,
  the doorbells) are published past AMD's GPU cache, as in Strata #697, and a new HIP test replays the real routing
  handoff. The Linux AMD setup itself is in review (#7). NVIDIA: unchanged.

## v1.0.6 - 2026-10-08

A standard speed test: `./maya.sh --bench` (Windows: `START-MAYA.bat --bench`).

- **`--bench`** measures the installed model on your machine in a few minutes, with Maya stopped: decode on the same
  three questions everywhere (after a warm-up answer) and prefill at 2k and 8k tokens, through the engine the way the
  dashboard starts it. It writes `maya-bench.txt` with the results and the engine's per-token breakdown; `--report`
  includes it - so speed reports from different machines can be compared.
- README: single-GPU speeds (1x Tesla V100 32 GB, 64 GB RAM: decode up to 19 tokens/s, prefill up to 370 tokens/s).

## v1.0.5 - 2026-10-08

Maya-M is out: the larger quant (116 GB), closer to the full model token by token.

- **Maya-M** on Hugging Face (`Maya-M/`): IQ2_S gate/up experts with error-feedback rounding, IQ3_XXS down
  projections, IQ3_S in the most sensitive layers, from Z.ai's FP8 release with the FP8 model's own statistics,
  calibrated toward tool calls and front-end code. Against the FP8 model: 23% lower KL divergence than Maya-S
  (0.329 vs 0.428), the same next token 86.2% of the time (Maya-S 83.3%), and 97.9% of its zero-shot accuracy - the
  same as Maya-S (multiple-choice tasks do not separate the two). Set it up with
  `./setup.sh --setup --model Maya-M` (Windows: `START-MAYA.bat --setup --model Maya-M`); downloads are checked
  against their published sha256. Maya-S stays the default. Results: `bench/results/MAYA-M.md`.
- **Decode:** the dense matrix-vector products run in kernels compiled for their weight type (Q6_K, Q8_0, Q4_K,
  Q5_K) - 10% faster per product, bit-identical results; it shows on cards that hold most experts in VRAM.
- `--model` now takes that download even when an earlier setup used `--gguf-dir`.

## v1.0.4 - 2026-10-08

Faster prefill again (how fast Maya reads your prompt), and more room for experts at long context.

- **Prefill attention on the tensor cores:** the prompt's attention (the absorbed MLA over each token's selected
  positions) now runs on the GPU's tensor cores, with the next positions loaded while the current ones compute - FP16
  operands with F32 accumulation, as flash attention (within ~1e-3 of before). RTX 20-series cards keep the previous
  kernel automatically. Prefill on 2x Tesla V100 32 GB, prompts of 2k / 8k / 16k / 30k tokens: 273 / 428 / 487 / 506
  -> 286 / 469 / 538 / 561 tokens/s (since v1.0.1: +19% / +42% / +56%). On one of those GPUs: 8k-token prompt 226 ->
  243 tokens/s.
- **The attention cache in FP16:** half the memory, so more of the GPU holds experts - at the default 32K context
  0.66 -> 0.47 GB a GPU, at 128K about 0.8 GB more for experts on each GPU (faster decode at long context).
- Quality: the same long texts scored before and after (6,000 tokens read, the next 400 scored) differ by +0.8% in
  likelihood, less than two runs of the same engine differ from each other (+1.4%) - no measurable change.

## v1.0.3 - 2026-10-07

A one-command report for problems and speeds: `./maya.sh --report` (Windows: `START-MAYA.bat --report`).

- **`--report`** writes `maya-report.txt` in the Maya folder: your GPUs (VRAM, driver, PCIe link), CPU, RAM, disks,
  your Maya setup (model, context, GPUs) and the engine log's speed lines - how the model is split across VRAM, RAM and
  the SSD, and where each token's time goes. Attach it when you report a problem or a speed, so the engine can be
  tuned for your machine. Nothing is sent anywhere; your home folder shows as `~` and no API key is included.

## v1.0.2 - 2026-10-07

Faster prefill (how fast Maya reads your prompt): up to 46% on two GPUs and up to 2x on one.

- **Prefill speed:** the experts a prompt needs that are neither in VRAM nor in RAM are read from the SSD ahead of
  time into a deeper buffer (the reader was keeping the NVMe at about a quarter of its speed), the next layer's
  experts are read while the current one computes, and the prompt is cut into bigger pieces on bigger cards (fewer
  times every expert is fetched). Prefill measured on 2x Tesla V100 32 GB with 30 GB RAM, prompts of 2k / 8k / 16k /
  30k tokens: 240 / 331 / 345 / - -> 273 / 428 / 487 / 506 tokens/s. On one of those GPUs: prefill of an 8k-token
  prompt 115 -> 226 tokens/s. Answers are unchanged in quality (the same text, up to rounding).
- **Support Project Maya:** [buymeacoffee.com/peasantsmith](https://buymeacoffee.com/peasantsmith) (README, and the
  Sponsor button on GitHub).
- README: how Maya grew out of Strata; exported chats are named `maya-chat-*.md`.

## v1.0.1 - 2026-10-07

- **Windows (experimental):** `START-MAYA.bat` sets Maya up the way `./maya.sh` does on Linux - Python, the
  engine and the image encoder compiled with Visual Studio 2022 Build Tools and CUDA 12.8, the model download, the
  dashboard. The engine reads the model's experts with unbuffered parallel reads and sizes its pinned RAM to what
  Windows allows (RAM + page file). It compiles on Windows; it has not been run on a Windows PC with an NVIDIA GPU
  yet - tell us how it runs. See README > Windows.
- **Fix:** a second GPU big enough to hold all of its experts (e.g. 64 GB cards) stopped the start with "the pinned
  RAM tier did not allocate".
- **Fix:** temperature 0 is greedy and deterministic again.

## v1.0.0 - 2026-10-07

The first release.

- **Maya-S** (96.5 GB): Project Maya's compact quant of GLM-5.3-Flash, made from Z.ai's FP8 release. It keeps
  97.9% of the FP8 model's zero-shot accuracy (ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, PIQA).
- The engine: the model's experts tiered across VRAM, pinned RAM and the SSD, one GPU or two that share the layers,
  MTP speculative decoding, a thinking budget, images on demand.
- The installer (`./setup.sh`), the dashboard and the OpenAI / Anthropic compatible API.
