# Qwen3.8-Flash-Next on Abyss

日本語: [README.ja.md](README.ja.md)

Qwen3.8-Flash-Next (llama.cpp's `qwen4exp`) runs on Abyss's fast path (`--glm-pack`) from v1.3.74, with its prompt
path, verify window and MTP drafter. The older engine path ([docs/maya/ORCA.md](../maya/ORCA.md)) still works; the
fast path is several times faster on prompts and faster on answers.

Measured with orcarouter's
[Qwen3.8-Flash-Next-Uncensored-GGUF](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) IQ3_XXS
(two shards, 85.2 GB) on an AI PRO R9700 (32 GB, PCIe Gen4 x4), 94 GB of RAM (`STRATA_GLM_RAM_GB=60`) and the iGPU
lane. NVIDIA: not built or measured on this path yet.

## The model

- **48 layers**: 36 gated delta-net layers (its output gate a sigmoid) and, every fourth layer, 12 attention layers
  (24 query heads of 256 over 2 KV heads, q / k norms, RoPE on 64 dims, a sigmoid output gate).
- **4 residual streams** of 2560 mixed by low-rank hyper-connections (no norm tensors: the mixer is the norm).
- **Layer 1's per-layer embedding (PLE)**: n-gram hashes of the last tokens pick 16 rows of a 28.8 GB table, which
  stays on the SSD (read directly, about 1 ms a token).
- **QSA** on the attention layers: past 2,051 cells each query attends to 512 blocks of 4 cells picked by an indexer
  (up to 2,051 cells it is dense attention).
- **512 experts a layer, 10 routed**, plus a shared expert behind a sigmoid gate, all 640 wide.
- The GGUF has no MTP (draft) block; the drafter is a separate file (below).

## Setup

1. **The pack** (the dense weights as the GGUF stores them; nothing converted to BF16):

   ```sh
   .venv/bin/python tools/iq_pack.py --fast \
     --gguf /path/to/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf \
     --out /path/to/IQ3_XXS/pack-fast
   ```

   Both shards must be complete. The pack is ~300 MB; the experts and the PLE table are read from the GGUF.

2. **The drafter (optional, recommended)**: the MTP runtime files the older path uses, made the same way
   ([docs/maya/ORCA.md](../maya/ORCA.md#preparation): `tools/mtp_fetch.py`, `tools/mtp_pack.py --experts q2_0`,
   `tools/mtp_rt.py`, then `data/draft_vocab.bin` copied in). The directory holds `dense.bin`, `dense.txt`,
   `experts.bin` and `draft_vocab.bin` (~820 MB); the engine keeps them in VRAM (~1 GB with its cache).

3. **The config** (`abyss-<name>.json`, see [docs/SETTINGS.md](../SETTINGS.md)):

   ```json
   {
    "exe": "build-hip/strata",
    "args": ["--glm-pack", "/path/to/IQ3_XXS/pack-fast", "--max-context", "32768", "--prefill", "auto"],
    "tokenizer": "/path/to/IQ3_XXS/pack-fast/tokenizer",
    "model_name": "qwen3.8-flash-next-uncensored-iq3_xxs",
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    "port": 8082,
    "backend": "hip",
    "env": {
     "STRATA_GLM_RAM_GB": "60",
     "STRATA_Q4X_MTP": "/path/to/mtp/rt"
    }
   }
   ```

   Start it with `.venv/bin/python serve/server.py --engine strata --config abyss-<name>.json --port 8082` (the
   `--port` on the command line is the one used).

## Settings

| Variable | Default | What it does |
| --- | --- | --- |
| `STRATA_Q4X_MTP=<dir>` | off | the drafter's runtime files; without it the decode drafts only from repeats (n-gram) |
| `STRATA_Q35_MTP_K` | 4 (this model) | drafts a step at most (1..5) |
| `STRATA_Q35_MTP_MIN` / `STRATA_DS4_DSPARK_THR` | 0.75 (this model) | a draft goes into the verify window only at or above this confidence |
| `STRATA_Q4X_MTP_FULLHEAD=1` | off | the drafter's head over every token instead of `draft_vocab.bin`'s 106K |
| `STRATA_Q4X_DENSE=1` | off | debug: dense attention everywhere instead of QSA's selection |
| `STRATA_Q4X_PLE_SYNC=1` | off | debug: the verify window reads its PLE rows before layer 0 instead of during it |
| `STRATA_GLM_NO_MTP=1` | off | the drafter off even with `STRATA_Q4X_MTP` set |

## Measured (R9700, 2026-10-28)

| | older path (v1.3.73) | fast path (v1.3.74) |
| --- | ---: | ---: |
| prompt, 30K tokens | 217 tok/s | 1,037 tok/s |
| answer, prose (Japanese / English) | 52 tok/s | 64 / 65 tok/s |
| answer, writing code | 66-85 tok/s | 86 tok/s |
| answer, copying code | | 100-107 tok/s |
| answer, no drafter | | 51-53 tok/s |
| a 31K-token prompt's passphrase | found | found |
| a conversation's next turn (10.8K tokens) | reused | reused: 10,802 tokens, 0.28 s |

Prose and code are greedy 300-token answers from the command line; the server's answers vary a little from run to
run (the iGPU lane and where the experts sit). The drafter's windows: 74% of the drafts accepted on prose, 86% on
code; a window costs 30 ms against 18.6 ms for a token alone.

Quality: against llama.cpp on 494 positions past 5,600 tokens, KL 0.026 / top-1 92.3% (the token path) and 0.035 /
90.9% (the prompt path); llama.cpp's own HIP-vs-CPU difference on this model is KL 0.045. Greedy answers with and
without drafts are the same tokens.

## Other GGUFs (beta)

From v1.3.76 the fast path also takes the popular quants below. Each was checked on one PC (the R9700 above): the
engine's log-probs against llama.cpp's on the same file, and a short speed run. **Support for them is beta**: please
run `./abyss.sh --bench` and post the report ([a bench issue](https://github.com/zephel01/Abyss/issues/new?template=bench.md),
with the file's folder name), whatever the numbers; the report says so for these files.

| File | Size | Resident (all but the PLE table) | Quality vs Q8_0 (KL / same top token) | Engine vs llama.cpp (KL) | Speed (R9700) |
| --- | ---: | ---: | --- | --- | --- |
| [ISTA-DASLab GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) IQ3_XXS | 75.8 GB | 47.0 GB | 0.398 / 71.6% | 0.019-0.022 | 66 prose / 84 code (16 GB + 64 GB: 61 / 77) |
| ISTA-DASLab GSQ-RCO IQ3_S | 83.6 GB | 54.8 GB | 0.282 / 76.5% | 0.026 | 51 prose / 77 code (16 GB + 64 GB: 45-57 / 54) |
| [unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) UD-IQ4_XS | 93.7 GB | 64.9 GB | 0.191 / 80.4% | 0.038-0.043 | 57 prose / 74 code (24 GB + 64 GB: 55 / 68) |
| unsloth UD-Q4_K_XL | 111.3 GB | 82.5 GB | 0.100 / 86.4% | 0.029-0.037 | 55 prose / 71 code |
| (for comparison) orcarouter Uncensored IQ3_XXS | 85.2 GB | | | 0.026 (5,600-token run) | see above |

"Quality vs Q8_0" is llama.cpp's own KL against unsloth's Q8_0 on the same 8 chunks of 1,024 tokens (smaller is
better); "Engine vs llama.cpp" is Abyss against llama.cpp on the same file (the port's error, not the quant's).

"Speed" is greedy 300-token answers (prose: an English essay prompt, code: an LRU cache) with the MTP drafter and the
iGPU lane on the whole R9700; the GPU + RAM figures in brackets cap the engine with `STRATA_GLM_VRAM_GB` and
`STRATA_GLM_RAM_GB=46`. Runs vary by several tok/s.

Which to pick: go down this list and take the first file whose resident part fits in your GPU's memory + the RAM
the engine may use (about 75% of the RAM: `STRATA_GLM_RAM_GB`). The PLE table (28.8 GB) stays on the SSD, so the
file's full size does not have to fit.

| Your GPU + RAM | File | Resident |
| --- | --- | ---: |
| 32 GB + 96 GB or more | unsloth UD-Q4_K_XL | 82.5 GB |
| 24 GB + 64 GB or more | **unsloth UD-IQ4_XS** - the best quality per GB measured here; the first pick when it fits | 64.9 GB |
| 16 GB + 64 GB | ISTA IQ3_S (83.6 GB file) | 54.8 GB |
| less (12 GB + 64 GB, 16 GB + 48 GB) | ISTA IQ3_XXS (75.8 GB file) | 47.0 GB |

A file a step too big still runs (what does not fit is read from the SSD each token), only slower; `--bench` shows
how many reads a token waits for.

New in the engine for these files: Q2_0, Q5_1 and Q8_0 expert rows, IQ4_XS / Q5_1 / Q2_0 in the verify window's
dense rows, Q5_1 on the prompt path, and BF16 hyper-connection, `ssm_alpha` / `ssm_beta` and PLE key / value
rows (ISTA keeps those in BF16).

### NVFP4 (NVIDIA's FP4 checkpoints, v1.3.101, beta)

[magnitudedev's NVFP4 GGUF](https://huggingface.co/magnitudedev/Qwen3.8-Flash-Next-NVFP4-GGUF) (rev `4591036`,
127.8 GB in 3 shards): experts in NVFP4 (FP4 values, an FP8 scale every 16, and one more scale per expert), the
rest in Q8_0, the PLE table too (54.4 GB, read from the SSD; resident 73.4 GB). The pack is made as above
(`tools/iq_pack.py --fast --gguf ...-00001-of-00003.gguf`). Checked on an RTX 5090 + 96 GB RAM (+ the 8060S
iGPU lane), MTP drafter on:

| Check | Result |
| --- | --- |
| Engine vs llama.cpp (KL, 640 EN/JA tokens) | 0.015 token by token, 0.018 through the prompt path |
| Prompt speed (needle found) | 8K: 1,054 tok/s, 32K: 2,515 tok/s |
| swe-bench 3 tasks | 105.3 tok/s |
| 10 hard llmbench tasks | 8 of 10 (2 hit the length cap), quality 76.1 |
| Japanese (3 prompts), 7-turn tool session | natural Japanese; tests pass, the final report in Japanese |

The prompt path multiplies NVFP4 with Blackwell's FP4 tensor cores (RTX 50, RTX PRO Blackwell, DGX Spark); other
NVIDIA GPUs and AMD are not checked yet. Reports from other GPUs are welcome (a bench issue as above).

### unsloth UD-IQ1_S (UD 3.1, v1.3.112, beta)

[unsloth's UD 3.1 UD-IQ1_S](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (66.4 GB) stores one layer's gate
and up experts in different types (layer 18: IQ2_S gate, IQ2_XXS up; down Q2_0 everywhere). From v1.3.112 the pack
records such a layer as `G,U` and each role is computed in its own type; older engines refuse the pack with a reason.
Make the pack with v1.3.112's `tools/iq_pack.py --fast` (as above). The iGPU lane and a helper GPU leave that one layer
to the CPU. Checked on an RTX 5090 (PCIe Gen4 x4) + 94 GB RAM, MTP drafter on, iGPU lane off:

| Check | Result |
| --- | --- |
| KL to the NVFP4 teacher (seq 0-9, 2,048 context) | 0.2007, same top token 83.2% (llama.cpp on the same file: 0.2012 / 83.1%) - 16G64G: 0.083 / 89.3% |
| Decode, 4K prompt, 512 tokens (median of 3) | 32 GB: 153.4 tok/s with the drafter, 103.7 without; 16 GB (`STRATA_GLM_VRAM_GB=16`): 78.0 with it, 109.6 with it and the 8060S iGPU lane |
| Prompt | 4K ~1,900 tok/s, 10K (needle) 2,734 tok/s |
| Real use (bench/mgpu/run.sh, 32 GB) | Japanese OK, code 4 / 5, 8K needle found |

Lower quality than the 16G64G build at a smaller file; take it when the larger files do not fit.

### unsloth UD-IQ2_XS (UD 3.1, v1.3.114, beta)

[unsloth's UD 3.1 UD-IQ2_XS](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (75.8 GB) has IQ2_S / IQ2_XXS
gate and up experts (the same type within a layer) and IQ4_NL / Q2_0 down experts, with one Q5_0 down (layer 4).
Engines before v1.3.114 stop at load with `unresolved weights: blk.4.expert types 22/22/6`. Make the pack with
`tools/iq_pack.py --fast` (as above). Checked on an RTX 5090 (PCIe Gen4 x4) + 94 GB RAM (50 GB RAM tier), iGPU lane
off:

| Check | Result |
| --- | --- |
| KL to the NVFP4 teacher (seq 0-9, 2,048 context) | 0.1359, same top token 86.4% (llama.cpp on the same file: 0.1356 / 86.6%) - UD-IQ1_S 0.2007, 16G64G 0.083 |
| Decode, 4K prompt, 512 tokens (median of 3) | 32 GB: 134.2 tok/s with the drafter, 91.3 without; 16 GB (`STRATA_GLM_VRAM_GB=16`): 71.7 with it, 57.9 without |
| Prompt | 8K ~1,660-2,080 tok/s, 24K 3,082 tok/s |
| `./abyss.sh --check --quick` (32 GB) | 7 / 7 pass: same answer with and without the drafter, Japanese, code, tool call, 8K and 24K needles, 11-turn conversation |

Between UD-IQ1_S and the 16G64G build in quality; slower than UD-IQ1_S because less of the larger file fits in VRAM.

## Limits

- One GPU: a split across cards has not been tried with this model, and the drafter loads only when one card runs
  the whole model.
- `--uncensor` has no file for this model (the GGUF used here is already uncensored).
- NVIDIA builds of this path have not been checked.
