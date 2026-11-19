# Long context: how far each model goes, and what to set

[日本語](CONTEXT.ja.md)

`--max-context N` (in a config's `args`) sets the longest conversation the engine holds. This page lists, for every
model Abyss runs, the length the model was trained for, what was actually measured up to 256K, how much memory the
context costs, and the value to set on your PC.

## 1. The model's limit

The limit is the model's own `context_length` from its GGUF. Abyss does not stop you from setting more, but past this
length the model runs on positions it was never trained on: answers are not checked there and are likely to degrade.

| Model | Limit (`--max-context` up to) | How the model gets there |
|---|---|---|
| DeepSeek V4 Flash 0731 | 1,048,576 (1M) | 64K trained, YaRN x16 |
| DeepSeek V4.1 Flash 1017 | 1,048,576 (1M) | 64K trained, YaRN x16 |
| GLM-5.3-Flash | 1,048,576 (1M) | (as stated by the GGUF) |
| Qwen3.8-Flash-Next | 262,144 | native |
| Qwen3.8-27B | 262,144 | native |
| Ornith-1.5-122B / 35B / 397B | 262,144 | native |
| Laguna-S-2.1 | 262,144 | 8K trained, YaRN x32 |
| Qwen3.5-9B QSA, Ornith-1.5-9B QSA, MiMo-V2.6-Distill-Qwen-9B | 262,144 | native |
| SenseNova-Loop-9B | 524,288 (512K) | 256K native, YaRN x2 |

Read from the GGUF files the measured packs were made from (`<arch>.context_length`). 256K (262,144) is within the
limit of every model above; V4, V4.1 and GLM-5.3-Flash go to 1M, SenseNova-Loop-9B to 512K.

## 2. What was measured (RTX 5090 32 GB, 123 GB RAM)

Each run is one request through the server (`/v1/chat/completions`, temperature 0, the model's chat template): real
source code of about 60K / 125K / 250K tokens with three planted secret strings (at 2%, 50% and 97% of the prompt) and
a small planted function to evaluate. "OK" = all three secrets and the function's value right. Prompt = the server's
prompt tok/s, decode = generation tok/s with that context filled. iGPU off, one greedy run each.

| Model (file) | 64K | 128K | 256K |
|---|---|---|---|
| V4 0731 UD-Q2_K_XL | OK / 2,815 / 66.8 | OK / 2,686 / 70.7 | OK / 2,336 / 65.6 |
| V4.1 Compact JA256 | OK / 2,356 / 57.0 | secrets OK, function wrong / 2,678 / 68.9 | secrets OK, function wrong / 4,651 / 67.5 |
| GLM-5.3-Flash Maya-S-v2 IQ2_XXS | OK / 2,247 / 36.0 | OK / 2,299 / 39.9 | OK / 2,127 / 30.0 |
| Qwen3.8-Flash-Next UD3.1 IQ3_XXS | OK / 2,830 / 111.2 | OK / 2,898 / 124.6 | OK / 2,562 / 84.2 |
| Ornith-1.5-122B 24G64G | OK / 2,186 / 53.0 | OK / 1,510 / 60.6 | OK / 1,608 / 41.5 |
| Ornith-1.5-35B-A3B Q4_K_M | OK / 1,888 / 299.8 | OK / 926 / 254.8 | OK / 3,857 / 170.6 |
| Qwen3.8-27B UD-Q4_K_XL, fp16 KV | OK / 1,892 / 73.6 | OK / 1,274 / 51.7 | OK / 1,530 / 25.2 |
| Qwen3.8-27B UD-Q4_K_XL, `--kv int8` | - | - | OK / 1,410 / 38.4 |
| Laguna-S-2.1 UD-IQ4_XS, fp16 KV | OK / 526 / 45.5 | secrets OK, function wrong / 258 / 37.2 | OK or 2 of 3 secrets (varies run to run) / 2,347 / 19.8 |
| Laguna-S-2.1 UD-IQ4_XS, `--kv int8` | - | - | OK / 2,329 / 20.9 |

Cells: result / prompt tok/s / decode tok/s. 64K and 128K: v1.3.150 (2026-11-15); 256K: v1.3.174, the `--kv int8`
rows and the 27B / Laguna fp16 256K rows v1.3.175 (2026-11-19). Prompt speeds of different versions are not
comparable (prompt-path work between them; Laguna's 128K -> 256K jump is that). The V4.1 Compact JA256 function answer
is the same wrong value (196) at 128K and 256K: a weakness of that quant in reading, not of the length.

Every model started and finished at 256K, without errors. Decode slows as the context fills (every token reads the
whole cache); it slows most where the cache pushes weights out of VRAM (Qwen3.8-27B, Laguna - see §3).

## 3. Memory the context takes

The KV cache (what the model keeps of the conversation) is set aside at start for the whole `--max-context`, in VRAM.
What is left after it goes to the model's weights (the expert pool). Measured from the engine's start-up lines:

| Model | KV at 128K | KV at 256K | VRAM used before the experts, 256K |
|---|---|---|---|
| DeepSeek V4 0731 | 0.86 GB | 1.70 GB | 8.91 GB |
| DeepSeek V4.1 1017 | 0.40 GB | 0.79 GB | 8.85 GB |
| GLM-5.3-Flash | 1.78 GB | 3.32 GB | 11.03 GB |
| Qwen3.8-Flash-Next | 3.00 GB | 6.00 GB | 11.35 GB |
| Ornith-1.5-122B | 3.25 GB (0.91 with `--kv q4_0`) | 6.50 GB | 13.21 GB |
| Ornith-1.5-35B | 2.75 GB | 5.50 GB | 8.14 GB |
| Qwen3.8-27B (dense) | 8.50 GB | 17.00 GB (9.03 with `--kv int8`) | 25.93 GB before the FFN (9.93 GB) |
| Laguna-S-2.1 | 6.35 GB | 12.35 GB (6.56 with `--kv int8`) | 17.07 GB |

The KV grows in step with `--max-context` (half the context = about half the KV). "VRAM used before the experts" is
the dense weights + KV + working buffers on the RTX 5090; the working buffers are smaller on smaller cards, so treat it
as an upper bound. The rest of the card holds experts; experts that do not fit are read from RAM (slower, not wrong).

What this means:

- **DeepSeek V4 / V4.1, GLM-5.3-Flash:** their caches are small (MLA / compressed attention). 256K costs 1-3 GB more
  than a short context; the context length barely matters for memory.
- **Qwen3.8-Flash-Next, Ornith:** 3 GB a 128K. 256K takes about 3 GB more of the expert pool than 128K.
- **Qwen3.8-27B, Laguna:** large caches. At 256K on a 32 GB card fp16 KV pushes weights out (27B: 5.2 GB of the FFN to
  RAM, decode 51.7 -> 25.2 tok/s; Laguna: 97 -> 66 expert slots a layer). The engine then prints one `hint:` line with
  what `--kv int8` would free. With `--kv int8` the 27B's whole FFN is back in VRAM (decode 38.4 tok/s).

`--kv int8` / `q4_0` store the cache in fewer bits (a small accuracy cost; KL figures in the
[manual's `--kv` row](MANUAL.md#settings)). The default stays fp16.

## 4. What to set

| Your card | Setting that was measured to work |
|---|---|
| 32 GB (RTX 5090 class) | `--max-context 262144` on every model in §2. Add `--kv int8` for Qwen3.8-27B (decode 25 -> 38 tok/s) |
| 32 GB, V4 / V4.1 / GLM past 256K | allowed up to 1M by the model; not measured here (see the users' reports below) |
| 16 GB | `--max-context 131072` measured on V4.1 Compact JA256, Qwen3.8-Flash-Next and Ornith-1.5-122B (`--kv q4_0`): all right, decode 39.8 / 77.9 / 38.8 tok/s. 256K not measured |

The 16 GB rows are the RTX 5090 capped with `STRATA_GLM_VRAM_GB=16` (RAM still 123 GB), not a real 16 GB card.
On other cards, estimate from §3: the card needs the "VRAM used before the experts" figure (scaled down for a shorter
context by the KV column) plus room for experts; the start-up log prints the real figures, and the engine stops with
`not enough VRAM for the expert pool` when they do not fit - then lower `--max-context` or add `--kv int8` / `q4_0`.

Users' reports: DeepSeek V4 Flash at 393K on a Radeon AI PRO R9700 (a 291K prompt read at 279 tok/s, decode 19.5,
issue #1); more in [MODELS.md §3](MODELS.md#3-user-reports-for-reference).

## 5. Related behaviour

- A prompt that leaves less than min(1024, context / 4) tokens to answer is refused with HTTP 400 "prompt is too long"
  (from v1.3.172; `STRATA_FIT_FLOOR`). Coding agents such as Claude Code then compact the conversation.
