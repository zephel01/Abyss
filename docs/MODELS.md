# Model evaluation and reference benchmarks per GPU

A guide to picking a model file and to the speed to expect on your PC. Every number was measured with Abyss and
carries its Abyss version or date. Speeds are tok/s (from ms a token: 1000 / ms). "not measured" means nobody measured
it, "–" means it does not apply. A setting that changes the answers (lossy) has its KL next to it.

日本語: [MODELS.ja.md](MODELS.ja.md)

## How it is measured

| item | what |
| --- | --- |
| V4.1 KL / top-1 / ppl | KL against the unquantized DeepSeek-V4.1-Flash-1017's top-256 log-probs on Lucebox's 8 held-out sequences (8 x 1,023 positions, `eval/` of [Lucebox/...-ROCMFP-GGUF](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF)). Lower is closer to the original model |
| V4 0731 KL | KL against llama.cpp on the same file (3,000 tokens), or against the engine's own reference output (1,000-3,000 tokens). It checks the engine and the settings, not the quantization's own loss |
| GLM nll | `glm_ids4k`: 3,000 tokens in, the nll of the last 1,000 (lower is better). The KL (against `ref4k_glm`) is the difference to another quant's output, not a quality figure |
| Ornith KL | KL against llama.cpp (its CPU backend) on the same GGUF. llama.cpp's own CPU and HIP backends differ by 0.027-0.029 |
| swe-bench | our own coding tasks (llmbench). **60 tasks x 5** (resolved rate and pass@1), **L7 = 16 hard tasks** (x 3 or x 1), **6 tasks** for speed (t003 / t010 / t018 / t025 / t030 / t036, sampled, +-5% run to run), **8 tasks** for V4.1 (from 2026-10-23, greedy). tok/s end to end through the server |
| one token at a time | decode without drafts (or n-gram drafts only); slower than swe-bench |

Test PC: a Ryzen AI Max+ 395 (Radeon 8060S, 94 GB RAM) with an RTX 5090 32 GB or an AI PRO R9700 32 GB on OCuLink (PCIe
Gen4 x4). The model SSD went from an ORICO (~3.4 GB/s) to a Lexar NM790 (~6.7 GB/s) on the night of 2026-10-23. V4.1's
speed depends on the SSD, so its rows say which one.

## 1. Model evaluation

### DeepSeek V4 Flash 0731

Popular (Hugging Face, last 30 days, as of 2026-10-24): huihui-ai's 0731 abliterated Q2 (320K a month), huihui-ai's
0422 ds4 (230K), unsloth's 0731 (120K), 0xSero's EXL3 (110K). Files of this size (~87-100 GB) fit a 32 GB GPU + ~92 GB
of RAM with no SSD reads.

| file (source) | size | quality | speed | memory, roughly | use |
| --- | ---: | --- | --- | --- | --- |
| `DeepSeek-V4-Flash-Q2-0731.gguf` ([huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF), packed with `--compat-bf16`) | 86.8 GB | KL against llama.cpp 0.0075, ppl 6.32 (v1.2.2). swe-bench resolved rate not measured | 5090: swe 6 tasks 91.1 tok/s (v1.3.27). R9700 + iGPU lane: swe 6 tasks 38.1 (v1.3.39) | 32 GB VRAM + ~70 GB RAM | when you want the uncensored model; set up by hand |
| huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF (0422) | ~87-100 GB | not measured | not measured | – | loads: not yet checked |
| `UD-Q2_K_XL` ([unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF), the setup's default; the default pack has Q5_K attention, KL 0.0055 -> 0.0119) | 96.8 GB | KL against llama.cpp 0.0096. swe-bench 60 x 5: 59/60 resolved, pass@1 95.8% (v1.3.27) | 5090: swe 60 tasks 106.7 tok/s (v1.3.27), one token at a time 92-95. R9700 + lane + DSpark: swe 60 tasks 70.2 (the build just before v1.3.43) | 32 GB VRAM + ~70 GB RAM hold every expert | **the default; start here** |
| 0xSero's DGX-Spark EXL3 | – | not measured | – | – | DeepSeek EXL3 does not load yet |
| DSpark drafter `DSpark-drafter-Q2K-Q8-MarkovQ8-0731.gguf` ([bleysg](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF), `--dspark`) | 6.94 GB | the answers do not change (a draft is kept only when the target's own sample agrees) | R9700, swe 6 tasks: no lane 48.4 -> 61.9, with the lane 51.7 -> 67.3 tok/s (v1.3.43) | +6.5 GB VRAM (as many experts move to the SSD) | speed on an R9700; a DGX Spark 21.0 -> 22.8 tok/s (v1.3.114) |

Lossy settings (V4 0731, 5090, v1.2.1 / v1.3.2x): `STRATA_GLM_TAIL_SKIP=4` +7% one token at a time (iGPU lane), KL 0.0226.
`STRATA_GLM_DEFER=1` +12-15%, KL 0.08 (0.012 off), L7 resolved 14 -> 13. `TAIL_SKIP=1` / `=2`: KL not measured.

### DeepSeek V4.1 Flash 1017

Popular (same date): the September V4.1 still has more downloads (1.48M a month; 1017 58K in its first week). Abyss runs
1017. antirez/deepseek-v4.1-flash-gguf (1.34M) mixes September and 1017 files. Among 1017-only quants nazeshinjite is #1.

V4.1's experts do not fit in RAM, so some are read from the SSD every token; a smaller file reads less and runs faster.
"One token at a time" below is the mean over the 8 sequences.

| file (source) | size | quality: KL / top-1 / ppl (8 sequences) | speed | use |
| --- | ---: | --- | --- | --- |
| `Q2` ([nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF](https://huggingface.co/nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF)) | 366 GB | 0.229 / 79.2% / 9.69 (exact routing), with `TAIL_SKIP` 0.241 / 78.9% / 9.75. swe-bench 60 x 5: 60/60 resolved, pass@1 98.4% (v1.3.33) | 5090: swe 60 tasks 50.6 tok/s (v1.3.33, with TAIL_SKIP, ORICO), one token at a time 27.9 (with) / 17.7 (without) (v1.3.31). R9700 + lane: 19.7 (with, v1.3.46, ORICO) | the most quality data; UD-IQ2_XXS's speed with 66 GB more on the SSD |
| antirez/deepseek-v4.1-flash-gguf, its 1017 files | – | not measured | not measured | loads: not yet checked (September files are out of scope) |
| `UD-IQ2_XXS` ([unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)) | 300 GB | 0.186 (without) / 0.209 (with TAIL_SKIP). top-1 / ppl / swe-bench not measured | R9700 + lane, one token at a time: 17.3 (without) / 24.3 (with) (v1.3.46, ORICO) | **quality first** (recommended from v1.3.46) |
| `UD-IQ1_M` (unsloth) | 266 GB | 0.270 (without) / 0.281 (with) | R9700 + lane: 20.5 / 26.5 (v1.3.46, ORICO) | in between |
| `LDLQ-132G` / `LDLQ-139G` ([zephel01/DeepSeek-V4.1-Flash-1017-LDLQ-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-LDLQ-32G96G-GGUF), BETA) | 275.8 / 283.0 GB | 0.194 / 0.182 (exact routing, 8 texts; UD-IQ1_M 0.198). 10 hard swe tasks 10/10 both, quality 77.6 / 81.4 | 5090 + lane, 32 GB + 96 GB (RAM tier 60 GB) + DSpark: swe 3 tasks 67.5 / 57.9 tok/s (UD-IQ1_M 71.8), 10 hard tasks 58.0 / 47.8 (2026-11-02) | our own LDLQ re-quantization for 32 GB + 96 GB: `132G` = speed, `139G` = quality ([details](deepseek41/README.md)) |
| `Compact-JA256` ([zephel01/DeepSeek-V4.1-Flash-1017-Compact-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-Compact-32G96G-GGUF), BETA) | 207.1 GB (experts 88.4 + Engram IQ4_NL 110.6) | 0.383 / 75.2% (exact routing, 8 texts). 10 hard swe tasks 8/10, quality 80.5 | 5090 + lane, 32 GB + 96 GB (RAM tier 60 GB) + DSpark: swe 3 tasks **87.0 tok/s**, 10 hard tasks 75.7, 32K prompt 1,833 tok/s (2026-11-03) | 256 of 384 experts, chosen on ja / zh / en / code / math / tool text: general-purpose, Japanese intact; hard coding: `LDLQ-132G` ([details](deepseek41/README.md)) |
| `GSQ-RCO-Coder` (ISTA-DASLab, 192 of 384 experts) | 182.7 GB | 0.582 / 69.3% (2026-11-03). 10 hard tasks 9/10 | 5090 + lane: swe 3 tasks 97.0 tok/s | code-only expert selection: answers Japanese in Chinese |
| `UD-IQ1_S` (unsloth) | 233 GB | 0.355 / 75.0% / 10.80 (without), 0.359 (with) | 5090: 45.6 (with, v1.3.38, ORICO). R9700 + lane: 27.2 / 31.9 (v1.3.46, ORICO); swe 8 tasks 43.7, + DSpark 54.4 (2026-10-24, NM790, see below) | **speed first**; clearly lower quality |
| `GSQ-RCO-IQ2_XXS` ([ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF), Engram in its own file) | 248.8 GB (backbone 138.2 + Engram 110.6) | 0.3137 (without) / 0.3266 (with TAIL_SKIP) | R9700 + lane, swe 8 tasks: 30.7 / 30.9 (without), 42.2 / 41.3 (with), 48.3 / 47.3 (with + DSpark). 8K prompt 109 s (~75 tok/s) (2026-10-24, NM790) | a small backbone at a lower KL than UD-IQ1_S; needs TAIL_SKIP for 40 tok/s |
| `GSQ-RCO-IQ1_M` (ISTA-DASLab) | 231.3 GB | 0.3845 (2026-10-24) | not measured | not recommended: higher KL than UD-IQ1_S |
| `Q2_K` ([mrfakename/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF)) | 247 GB | teacher KL 0.64 (dense and Engram Q2_K too) | 5090: ~9.7 tok/s (2026-10-18) | not on its own |
| hybrid: Lucebox's Q8_0 dense + 8-bit Engram with mrfakename's experts | 201 GB | 0.168 / 83.4% / 9.25 | 5090: 8.5 tok/s (2026-10-18) | the most accurate, slow |

V4.1 overlays (UD-IQ1_S, R9700 + lane, v1.3.52):

| method | adds | KL (base 0.3557) | speed | for |
| --- | --- | --- | --- | --- |
| Q5_K dense overlay (`tools/dense_overlay.py`, opt-in, lossy) | 3.2 GB | 0.3652, ppl 10.79 -> 10.93 | swe 8 tasks 40.1 -> 43.0 tok/s (n-gram windows) | a little more speed |
| Q5_K + the refusal overlay together | – | 0.3738 | not measured | – |

The refusal edits (`--abl` overlay, `STRATA_GLM_ABL`) and the run-time `--uncensor` are measured in [UNCENSOR.md](UNCENSOR.md).

V4.1 lossy settings (KL on nazeshinjite Q2, 8 sequences): `STRATA_GLM_TAIL_SKIP=3 STRATA_GLM_TAIL_SKIP_DISK=1` 0.229 ->
0.240-0.241 (5090: 17.7 -> ~28-29 tok/s); `=4` 0.235. `STRATA_GLM_ROUTE_AFFINITY_RAM=0.08` 0.2406 -> 0.2438 (-8-12% a
token). `STRATA_GLM_DEFER=1` 0.15 -> 0.28 on one sequence (left off). UD-IQ2_XXS on the R9700 + lane (NM790, 4
sequences, 2026-10-27): exact routing 48.5-53.4 ms a token (~19 tok/s, ~20 ms of it SSD waits), `TAIL_SKIP=4` +
`_DISK=1` 39.8 ms (25.1 tok/s) for KL +0.007, `=3` 38.6 ms for +0.020 - `=4` is the pick there.

V4.1 DSpark drafter (Lucebox's `deepseek41-dflash-draft`, DSpark-draft-Q2K-Q4K, 4.92 GB, ~1.67 GB of VRAM, opt-in,
v1.3.53): the answers do not change. On the R9700 with the NM790, swe 8 tasks +16-19% (UD-IQ1_S: 40.0 ->
47.5, 2026-10-24). No gain on an ORICO-class SSD (2026-10-23).

### GLM-5.3-Flash

Download counts not surveyed. Every file holds all its experts in 32 GB of VRAM + ~60-78 GB of RAM.

| file (source) | size | quality: nll (lower is better) | speed | use |
| --- | ---: | --- | --- | --- |
| Maya-S-v2-IQ2_XXS (Project Maya) | 96.5 GB | 1.634, top-1 62.3%. swe-bench L7: 13/16 resolved (Abyss 1.1.2, 2026-10-11) | 5090: one token at a time ~67 (14.8 ms, 2026-10-19), swe 6 tasks 64.1 (v1.3.27). R9700 + lane: 34.3-38.3 (v1.3.40), swe 6 tasks 35.8 (v1.3.39) | **the default** |
| Maya-S24 | 94.7 GB | not measured on this page's scale | 5090: swe 6 tasks 70.7 (v1.3.27). R9700 + lane: swe 6 tasks 41.2 (v1.3.39) | cards of 24 GB or less |
| EXL3 `sc_2.10bpw` ([turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/sc_2.10bpw), experimental) | 88.3 GB | 1.591-1.598, top-1 63.3-63.7%. KL 0.168 (difference to Maya-S-v2) | 5090: ~53 tok/s (18.8 ms, v1.3.36) | smaller with better text; ~80% of the speed |
| EXL3 `3.05bpw` (turboderp) | 125.3 GB | not measured | not measured | offered by the installer, not measured |
| exlr8 K2 (kindlingai; `tools/exlr8_convert.py --width 2` then `tools/exl3_pack.py`; v1.3.53) | 86 GB | nll 1.608-1.613 (KL 0.208 against `ref4k_glm`) | R9700 + 8060S lane, 60 GB RAM tier: ~22.9 tok/s (43.7 ms, 2026-10-24) | the smallest; speed work to come |
| exlr8 K3 (kindlingai, `--width 3`; v1.3.53) | 125 GB | nll 1.548 (KL 0.212 against `ref4k_glm`) | R9700 + 8060S lane, 60 GB RAM tier: ~11.2 tok/s (89.1 ms, a third of the experts from the SSD, 2026-10-24) | the best GLM quality measured; slow on a 94 GB box |
| exlr8 K4 (kindlingai) | 164 GB | – | – | not tried |

Lossy settings (Maya-S-v2, 5090): `STRATA_GLM_DEFER=1` -19% a token, nll +0.063.

8 GB cards (issue #23, an RTX A2000 Laptop): Maya-S-v2 starts with its dense pack re-quantized to IQ4_XS (5.6 -> 3.6
GiB, the recipe in [Small VRAM](SMALL-VRAM.md)); 4-4.7 tok/s there, bound by a 512 GB SSD's reads.

### Ornith-1.5-122B (Qwen3.5-MoE)

Download counts not surveyed. UD-Q4_K_XL holds all its experts in 32 GB of VRAM + a 60 GB RAM tier: no SSD reads.

| file (source) | size | quality | speed | use |
| --- | ---: | --- | --- | --- |
| `UD-Q4_K_XL` ([unsloth/Ornith-1.5-122B-GGUF](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF)) + the NextN drafter (`tools/nextn_pack.py`) | 79.7 GB + 1.5 GB | KL against llama.cpp 0.024 (v1.3.46). swe-bench L7: 11/16 resolved (v1.3.49) | R9700 + lane: one token at a time 41-43, with NextN code 59 / prose 50 (v1.3.46), 51.0 after 19K tokens (v1.3.49), swe L7 54.9 tok/s (v1.3.49). No lane: code 42 / prose 39 (v1.3.46) | 40-50 tok/s without the SSD |
| `UD-IQ3_XXS` (unsloth) | 49.5 GB | not measured | user report only (#8 below) | 16 GB cards |
| `IQ3_XXS-B` ([zephel01/Ornith-1.5-122B-IQ3_XXS-16G64G-GGUF](https://huggingface.co/zephel01/Ornith-1.5-122B-IQ3_XXS-16G64G-GGUF)) | 51.7 GB | KL about UD-IQ3_XXS's (0.138-0.140); swe-bench 56 of 60 resolved | 5090, the config `--model ornith` writes: 116 tok/s, 75% of the NextN drafts accepted (v1.3.100) | 16 GB GPU + 64 GB RAM |

### Qwen3.8-Flash-Next (qwen4exp)

Details, packing and the drafter: [docs/qwen38](qwen38/README.md). KL is against the NVFP4 teacher (seq 0-9, 2,048
context) unless it says llama.cpp; speeds are a 4K prompt, 512 tokens, median of 3, with the MTP drafter, RTX 5090
(PCIe Gen4 x4) + 94 GB RAM, iGPU lane off.

| file (source) | size | quality | speed | use |
| --- | ---: | --- | --- | --- |
| UD 3.1 `UD-IQ4_XS` ([unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), beta) | 93.7 GB | KL 0.077 (llama.cpp on the same file 0.076) | 32 GB: 117.0, 16 GB: 80.9 (v1.3.115) | quality on 32 GB + 96 GB |
| UD 3.1 `UD-IQ3_XXS` (unsloth, beta) | 83.6 GB | KL 0.099 | 32 GB: 133.0, 16 GB: 87.4 (v1.3.115) | speed and quality together |
| UD 3.1 `UD-IQ1_M` (unsloth, beta) | 68.0 GB | KL 0.190 | 32 GB: 151.8, 16 GB: 77.0 (v1.3.115) | the smallest and fastest |
| `16G64G` ([zephel01/Qwen3.8-Flash-Next-16G64G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-16G64G-GGUF), beta) | see the model card | KL 0.083 | 32 GB: 131.3, 16 GB: 96.4 (v1.3.112); DGX Spark 47.05 (v1.3.113) | 16 GB GPU + 64 GB RAM |
| `12G32G` L448 ([zephel01/Qwen3.8-Flash-Next-12G32G-GGUF](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-12G32G-GGUF), 448 of 512 experts, beta) | 67.5 GB | not on this page's scale | not measured on this page's scale | 12 GB GPU + 32 GB RAM |
| ISTA-DASLab GSQ-RCO `IQ3_S` | 83.6 GB | KL against llama.cpp 0.026 (R9700, v1.3.76) | 5090, ctx 32K: `--bench` 128.5-129.5 (v1.3.87); prompts 8K 2,240-2,376, 32K 3,002-3,027 (v1.3.88) | – |
| magnitudedev `NVFP4` | 127.8 GB | KL against llama.cpp 0.015 / 0.018 (token / prompt path) | 5090 + 96 GB: swe 3 tasks 105.3, 32K prompt 2,515 (v1.3.101) | Blackwell's FP4 prompt path |

## 2. Reference benchmarks per GPU (the author's PC)

RTX 5090 numbers up to v1.3.38 are from **before the 2026-10-23 SSD swap** (ORICO, ~3.4 GB/s); the 5090 is back from
2026-10-31, and the rows from v1.3.82 on are on the NM790. ISTA and UD-IQ1_S on V4.1 are not re-measured on the 5090.

| model / file | item | RTX 5090 + 8060S lane | AI PRO R9700 alone (CPU lane) | R9700 + 8060S lane | RTX 3090 (USB4) + lane |
| --- | --- | --- | --- | --- | --- |
| V4 0731 UD-Q2_K_XL | one token at a time | 92-95 (v1.3.21-27) | 46.1-46.5 (v1.3.41) | 52.2-52.9 (v1.3.40) | 54.4-56.5 (v1.3.27) |
| | swe 6 tasks | 104.8 (v1.3.27) | 48.4, DSpark 61.9 (v1.3.43) | 51.7, DSpark 67.3 (v1.3.43) | 57.9 (v1.3.27) |
| | swe 60 tasks x 5 | 106.7 (v1.3.27) | not measured | DSpark 70.2 (just before v1.3.43) | not measured |
| | prompt 8K | 1,630-1,760 (v1.3.16-18) | 844 (v1.3.41) | 866-995 (v1.3.40) | 765-864 (v1.3.27) |
| V4 0731 huihui Q2 | swe 6 tasks | 91.1 (v1.3.27) | 33.9 (v1.3.39) | 38.1 (v1.3.39) | 49.1 (v1.3.27) |
| V4.1 nazeshinjite Q2 + TAIL_SKIP (KL 0.241) | one token at a time | 27.9 (v1.3.31) | not measured | 19.7 (v1.3.46) | not measured |
| | swe 60 tasks x 5 | 50.6 (v1.3.33) | not measured | not measured | not measured |
| V4.1 UD-IQ1_S + TAIL_SKIP (KL 0.359) | one token at a time | 45.6 (v1.3.38) | not measured | 31.9 (v1.3.46) | not measured |
| | swe 8 tasks (NM790) | not measured | not measured | 43.7 (+ affinity, KL 0.3573), DSpark 54.4 (2026-10-24) | not measured |
| V4.1 ISTA IQ2_XXS + TAIL_SKIP (KL 0.3266) | swe 8 tasks (NM790) | not measured | not measured | 42.2 / 41.3, DSpark 48.3 / 47.3 (2026-10-24) | not measured |
| V4.1 UD-IQ2_XXS, ctx 64K | `--bench` decode / 8K prompt (no DSpark) | 23.1 / 357 (v1.3.82), 22.9 / 360 (v1.3.87) | not measured | not measured | not measured |
| | swe 3 tasks, everyday settings + DSpark | 51.0 (v1.3.89) | not measured | not measured | not measured |
| GLM Maya-S-v2 | one token at a time | 64.7-75.4 (v1.3.27); `--bench` 64.0-64.6 (v1.3.82-87) | not measured | 34.3-38.3 (v1.3.40) | 40.5-42.4 (v1.3.27) |
| | swe 6 tasks | 64.1 (v1.3.27) | 25.8 (v1.3.39) | 35.8 (v1.3.39) | 39.7 (v1.3.27) |
| | prompt 8K | 1,087-1,113 (v1.3.82-87) | not measured | not measured | not measured |
| GLM EXL3 sc_2.10bpw | one token at a time | ~53 (v1.3.36) | not measured | not measured | not measured |
| GLM exlr8 K2 | one token at a time | not measured | not measured | ~22.9 (2026-10-24) | not measured |
| GLM exlr8 K3 | one token at a time | not measured | not measured | ~11.2 (2026-10-24) | not measured |
| Ornith UD-Q4_K_XL + NextN | code / prose | not measured | 42 / 39 (v1.3.46) | 59 / 50 (v1.3.46) | not measured |
| | swe L7 | not measured | not measured | 54.9 (v1.3.49) | not measured |
| Qwen3.8 ISTA IQ3_S + MTP | `--bench` decode | 128.5-129.5 (v1.3.87) | not measured | 51 prose / 77 code (v1.3.76) | not measured |
| Qwen3.8 16G64G + MTP | 4K prompt, 512 tokens | 131.3 at 32 GB, 96.4 at 16 GB (v1.3.112) | not measured | not measured | not measured |

The R9700's V4.1 one-token numbers are 8-sequence means on the ORICO SSD (2026-10-22); the swe 8-task runs are after
the switch to the NM790. UD-IQ1_S on the R9700 alone: ~22.7 tok/s one token at a time (44.0 ms, v1.3.46; the source
does not say whether TAIL_SKIP was on). A 5090 + 3090 layer split is slower than the 5090 alone (87-90 tok/s, around
v1.3.25).

### DGX Spark

A rented DGX Spark (GB10, 128 GB unified memory), Abyss v1.3.113-v1.3.114 (2026-11-05 / 06), beta
([DGX Spark](DGX-SPARK.md)):

| model / file | decode | prompt |
| --- | --- | --- |
| Qwen3.8 16G64G + MTP (v1.3.113) | 47.05 tok/s (CLI, 4K, median of 3 x 512; 31.2 without); 43-54 through the server | 8K ~1,040 tok/s |
| GLM-5.3-Flash Maya-S-v2 IQ2_XXS / unsloth UD-IQ2_XXS | 19-20 tok/s (the MTP drafter gives no gain here) | 8K 730-750 tok/s; 32K from the CLI 675-796 |
| DeepSeek V4 0731 | 21.0 tok/s, 22.8 with DSpark (78% of the drafts accepted) | 8K 814 tok/s |
| DeepSeek V4.1 Compact JA256 (207 GB, does not fit in memory) | ~18 tok/s | not measured |

llama.cpp on the same Spark (8K prompt, 4K of context filled): V4 0731 20.2 tok/s / prompt 360, GLM 19.3 / 321 - Abyss
decodes the same or faster and reads prompts ~2.3x faster. Known (open, seen only on the Spark): through the server,
32K-121K prompts run at ~120-130 tok/s against the CLI's 675-796; being investigated (the RTX 5090 does not show it).
From v1.3.115 the engine does not start when other processes hold more than 16 GB of the shared memory and the experts
no longer fit because of it (`STRATA_GLM_UMA_SHARE=1` starts anyway, `STRATA_GLM_UMA_OTHERS_GB` sets the limit).

## 3. User reports (for reference)

Numbers users reported in GitHub issues. **Measured by the users, not re-checked by the author.** Their methods (prompt
length, a `glm stat` line or end to end) differ from report to report, so they do not compare directly with the
tables above.

| issue | PC | Abyss | model / file | settings | decode tok/s |
| --- | --- | --- | --- | --- | --- |
| #8 | RX 9070 XT 16 GB + Ryzen 7 7800X3D + 64 GB DDR5, PCIe 4.0 NVMe, Linux (Docker, ROCm 10) | v1.3.44 | V4 0731 UD-Q2_K_XL | no TAIL_SKIP, no DSpark | 18.3-18.4 (warm), 16.7 after an 8.8K prompt; prompts 133-148 |
| #8 | same | v1.3.44 / v1.3.46 | same | `TAIL_SKIP=3` + `_DISK=1` (KL not measured on V4; 0.229 -> 0.241 on V4.1) | 23.7-24.1; 29-31 as a topic goes on (v1.3.46's RAM fill: 40% fewer skips, same speed) |
| #8 | same | v1.3.44 | GLM Maya-S-v2 IQ2_XXS | defaults | 10.3-10.6 |
| #8 | same | v1.3.46 | Ornith UD-Q4_K_XL / UD-IQ3_XXS | `TAIL_SKIP=3` + `_DISK=1` (KL not measured), NextN (the build with the bug fixed in v1.3.49), no iGPU | prose 20.8 / 21.9, code 25-29 / 34-39 |
| #11 | same box (the 7800X3D's iGPU off in the BIOS, PCIe Gen5 x16), Linux (ROCm 10 container) | v1.3.53 | V4.1 1017 ISTA GSQ-RCO IQ2_XXS, 128K, `RAM_GB=44` | `TAIL_SKIP=3` + `_DISK=1`; the lane on (the bug) / `STRATA_GLM_IGPU=0` | cold 18.3 / 26.3; same topic 22.9-24.0 / 31.3-31.5; new topic 17.7-17.9 / 18.7-20.9; after a 10K prompt 18.0 / 20.3 |
| #11 | same | v1.3.53 | Ornith UD-IQ3_XXS, 128K, no thinking | the lane on (the bug) / `STRATA_GLM_IGPU=0` | 25.9-28.0 / 39.1-39.3 |
| #11 | same, ROCm 7.2.1 container (`ABYSS_OLD_ROCM=1`) | v1.3.53 | Ornith UD-IQ3_XXS | `STRATA_GLM_IGPU=0` (with the lane on 7.2.1 failed every request: fixed in v1.3.54) | 32-35 (short answers) |
| #5 | RX 7900 XT 20 GB + i7-14700K + 128 GB DDR4, Windows 11 (TheRock 7.14.1) | v1.3.30 -> v1.3.46 | V4 0731 UD-Q2_K_XL, 64K | `TAIL_SKIP=0` | 13.71 -> 12.56 |
| #5 | same | v1.3.30 -> v1.3.46 | same | `TAIL_SKIP=1`, `LANE=8`, `PIN=0-15` (KL not measured; a large quality change) | 28.20 -> 32.86 |
| #5 | same | v1.3.30 -> v1.3.46 | same, 256K | `TAIL_SKIP=4` (KL 0.0226, measured on the 5090) | 17.77 -> 16.40 |
| #5 | same | v1.3.30 -> v1.3.46 | V4.1 nazeshinjite Q2, 85 GB RAM tier | `TAIL_SKIP=4` + `_DISK=1` (KL 0.235) | 10.55 -> 10.12 |
| #9 | same | v1.3.46 + the user's own patches (not upstream) | V4 0731 UD-Q2_K_XL | `TAIL_SKIP=1` / `=4` / `=0` (=1 KL not measured, =4 0.0226) | ~53-55 / ~35 / ~18 (before the patches ~36.8 / 16.5 / 10.7) |
| #9 | same | same | V4.1 nazeshinjite Q2 | `TAIL_SKIP=2` + `_DISK=1` (KL not measured) / `TAIL_SKIP=0` | ~44.0 / ~35.8 |
| #1 | AI PRO R9700 + i5-14400F + 128 GB RAM, Ubuntu (ROCm 7.0, ROCm 10 from v1.3.42) | v1.3.20 | V4 0731 UD-Q2_K_XL | defaults | 28.4-31.1, 36.6 warm |
| #1 | same | v1.3.31 | same, 393K context | defaults | a 291K prompt read at 279 tok/s, decode 19.5 |
| #1 | same | v1.3.35 / v1.3.37 / v1.3.42 | V4.1 nazeshinjite Q2, 80 GB RAM tier | `TAIL_SKIP=3` + `_DISK=1` (KL 0.241) | 24-28 / 30.5 / 22-25; 8K prompt 42-48 -> 60 (v1.3.42) |
| #23 | RTX A2000 Laptop 8 GB + 64 GB RAM, PCIe Gen4 x8 | before v1.3.82 | GLM Maya-S-v2 (IQ4_XS dense pack), Ornith IQ3_XXS-B | an expert pool under a route (6 of 8 slots) | GLM 4-4.7 (bound by a 512 GB SSD); v1.3.82 lets such pools start and read prompts in chunks ([Small VRAM](SMALL-VRAM.md)) |
| #31 | RTX 4090 + 96 GB RAM, Windows | before v1.3.107 | – | `STRATA_GLM_RAM_GB=72` | – (prompts from ~20K tokens failed, from ~40K the engine died; v1.3.107 moves the CPU's share to PCIe when its pinned buffers do not allocate) |
| #35 | two 16 GB GeForce cards | v1.3.109-v1.3.110 | Qwen3.8-Flash-Next, two-GPU split (beta) | `--layer-split 32` against one card | – (the split 22% faster on long requests; `auto` still picks one card: set the split by hand) |
| #37 | DGX Spark | v1.3.113-v1.3.114 | – | `./abyss.sh --check`, `--gpu-tune` | – (a short right Japanese answer failed `--check`'s check, fixed in v1.3.114; `--gpu-tune` finds a hand-built engine from v1.3.115) |

The #8 rows and #11's "lane on" ones were measured with the iGPU lane wrongly running on the 9070 XT itself (no
iGPU visible, the lane fell back to the engine's card: issue #11, fixed in v1.3.54). With it off V4.1 is about 30%
faster and Ornith 40-50%.

`./abyss.sh --gpu-tune` reports (the dense Q6_K GEMV, a call's total in us; lower is faster):

| issue | GPU | Abyss | 4 warps / a warp a row | the engine's choice |
| --- | --- | --- | --- | --- |
| #10 | RX 9070 XT 16 GB (gfx1201), Ryzen 7 7800X3D, 62 GB RAM | v1.3.53 | 868.0 / 537.8 (~590 GB/s on the large matrices) | a warp a row (the measured best, no change) |

### What the reports say

- A 16 GB card with 64 GB of RAM (#8) runs V4 0731 at ~18 tok/s without TAIL_SKIP (the lane-on bug included), and
  V4.1's ISTA IQ2_XXS at 19-31 tok/s with TAIL_SKIP=3 (#11, the lane off).
- ROCm 7.2.1 runs (Ornith 32-35 tok/s against ~39 on ROCm 10 on the same box, #11); setup still asks for ROCm 10 and
  `ABYSS_OLD_ROCM=1` overrides it.
- With 128 GB of RAM (#1, #5) V4.1's Q2 runs too; the SSD and the RAM tier's size set its speed.
- `TAIL_SKIP=1` / `=2` speeds come from settings whose KL nobody measured. To compare quality, use `=3` (V4.1, with
  `_DISK=1`) or `=4` and above.
