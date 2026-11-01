# DeepSeek V4.1 Flash (1017) on Abyss

From v1.3.29 Abyss runs [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(revision `1bae0e10`) on the same fast path as DeepSeek V4: one GPU, the RAM tier (computed by the CPU or an APU's
iGPU), and the SSD for the experts that fit neither. This page is for running it; the forward pass and every change
against V4 are in [SPEC.md](SPEC.md).

## What is different from V4

| | DeepSeek V4 Flash (0731) | DeepSeek V4.1 Flash (1017) |
| --- | --- | --- |
| Layers / hidden / experts | 43 / 4096 / 256 (top 6) | 40 / 5120 / **384** (top 6), expert width 2304 |
| Compressed attention | every compressed layer has its own compressor (CSA ratio 4 / HCA ratio 128) and, on CSA, its own indexer | **four shared KV slots** written by layers 2 / 8 / 14 / 20 (ratio 2 or 1); 8 index sources pick the top-512 and the layers between reuse it |
| Long contexts | indexer top-512 | + **candidate blocks** from layer 20 past 16K tokens |
| Hyper-connections | the sublayer's own `pre` | `pre` **lags one sublayer**; no `output_hc_*` head |
| Router | layers 0-2 hash-routed | biased top-6 on every layer |
| Engram | - | **two hashed n-gram tables (~101 GB each)** added before layers 1 and 14: 48 rows of 264 bytes a token, read row by row from the GGUF |
| Expert bytes (Q2-class GGUF) | ~93 GB (UD-Q2_K_XL): all fit VRAM + RAM on a 94 GB box | 153 GB (nazeshinjite Q2): **~43% stay on the SSD** |
| Chat | V4 template | `<｜System｜>`, DSML tags with a leading space; the ds4-layout GGUFs mark the chat tokens as normal (handled from v1.3.30) |

What this means for speed: V4 keeps every expert in VRAM + RAM, so its decode is bound by the GPU and the iGPU lane
(~92-95 tok/s). V4.1 reads experts from the SSD (10 MB each, ~3-4 ms at 3.4 GB/s), and how many a token decides the
speed: 13-15 ms a token when it is near zero, ~28 ms past 1.5 a token. A faster NVMe is the cheapest lever on one GPU.

```mermaid
flowchart LR
    T[token] --> H["Engram hash<br/>48 rows"]
    H -->|pread, 12.7 KB| SSD1[("GGUF on NVMe<br/>Engram tables ~200 GB")]
    T --> L["40 layers<br/>attention: own window +<br/>a shared KV slot, top-512"]
    SSD1 --> L
    L --> R{"router<br/>top 6 of 384"}
    R --> V["VRAM<br/>~2,300 experts"]
    R --> P["pinned RAM 60 GB<br/>~6,500 experts<br/>iGPU / CPU lane"]
    R --> D[("NVMe<br/>~6,600 experts<br/>streamed into VRAM")]
```

(The counts are the RTX 5090 + 94 GB box with nazeshinjite's Q2 and a 60 GB RAM tier.)

## GGUFs that load

The packer (`tools/iq_pack.py`) reads the three V4.1 layouts:

| GGUF | layout | experts | notes |
| --- | --- | --- | --- |
| [nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF](https://huggingface.co/nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF) `Q2` | ds4 (antirez) | IQ2_XXS gate/up, Q2_K down, 1017 imatrix (9.95 MB an expert) | Q8_0 dense, FP8 Engram. **The default.** 366 GB |
| [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) `UD-IQ1_S` | llama.cpp PR 30817 | IQ1_S, IQ2_XXS in some layers (~6.9 MB) | Q5_K-Q8_0 dense, IQ4_NL Engram. **The fastest** (v1.3.38): -32% a token, KL 0.355 against 0.228. 233 GB |
| [mrfakename/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF) `Q2_K` | llama.cpp PR (earlier) | Q2_K gate/up, Q3_K down (12.8 MB) | dense and Engram tables are Q2_K too - low quality alone (teacher KL 0.64) |

unsloth's other quants (UD-IQ1_M 250 GB up to UD-Q8_K_XL 508 GB) and Project Maya's
[Maya-S](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF) (178 GB, its Engram tables in a
separate file) use the PR 30817 layout too; they should pack the same way but are untested. Their layout keeps both
Engram tables in one `per_layer_token_embd.weight` and folds the Engram q / k norm weights into one `engram_gate`;
unsloth's `engram/` folder offers the tables in MXFP8 / Q6_K / Q5_K / IQ4_NL (the pack points at whichever file it
was made from).

**unsloth's re-upload of 2026-10-28** (revision `85f81fb970add013033f25e94006713ca2c0aaf2`; v1.3.77). unsloth
replaced UD-IQ2_XXS / UD-IQ1_M / UD-IQ1_S with new quants, clearly better at the same sizes. AI PRO R9700 + Radeon
8060S lane, 8 held-out texts, KL against the unquantized model:

| | size | exact routing: a token / KL / same top | everyday (tail skip 4 + disk, affinity 0.08) | swe-bench 8 tasks, 32 GB + 96 GB / 32 GB + 64 GB |
| --- | ---: | :---: | :---: | :---: |
| unsloth UD-IQ2_XXS (new) | 300 GB | 50.0 ms / **0.141** / 83.7% | 36.8 ms / 0.149 | 33.9 / 29.5 tok/s |
| unsloth UD-IQ1_M (new) | 250 GB | 39.4 ms / 0.199 / 81.5% | 30.9 ms / 0.203 | **40.6** / 30.5 tok/s |
| unsloth UD-IQ1_S (new) | 233 GB | 30.6 ms / 0.257 / 78.7% | - | - |
| ISTA-DASLab GSQ-RCO IQ2_XXS (new 6-shard layout) | 138 GB + Engram | 39.7 ms / 0.315 / 76.5% | - | - |

- The picks: **UD-IQ2_XXS** for quality (KL 0.188 → 0.141 against the old files), **UD-IQ1_M** for speed (0.270 →
  0.199, 40 tok/s on swe-bench with the DSpark drafter on 32 GB + 96 GB).
- UD-IQ1_S and ISTA run but are not recommended: past KL 0.25 the loss shows in the text (our line for a pick).
- swe-bench rows: the everyday settings + the DSpark drafter (`--dspark`, threshold 0.6), iGPU lane off, 8 tasks.
- **Keep the iGPU lane on** when the box has one (the default; `STRATA_GLM_IGPU=0` turns it off): it computes RAM-tier
  experts beside the CPU lane, which is what a verify window waits on. RTX 5090 (OCuLink Gen4 x4) + Radeon 8060S lane,
  UD-IQ2_XXS (new), everyday settings + the DSpark drafter, swe-bench 3 tasks (t010 / t021 / t030), 2026-11-01:
  lane off 37.9 / 37.6 tok/s, **lane on 50.5 / 46.6 tok/s (+25-33%)**.
- ISTA's new layout keeps the Engram table in shard 6 under another name; point a symlink named like shard 6 at
  `Engram-IQ4_NL.gguf` before packing.
- Download the measured files with `--revision 85f81fb970add013033f25e94006713ca2c0aaf2` (files made before 10-28 are
  the old quants). Two SSDs cut the SSD waits by ~20% ([SSD2](../SSD2.md)).

The tables below are the old files (before 2026-10-28).

**unsloth UD-IQ1_S against nazeshinjite Q2** (2026-10-20, RTX 5090 + 60 GB RAM tier, model SSD 3.4 GB/s):

| | nazeshinjite Q2 | unsloth UD-IQ1_S |
| --- | :---: | :---: |
| a token, disk tail skip (seq0 / seq1) | 32.3 / 32.4 ms | **21.9 / 22.0 ms** (45.6 tok/s) |
| disk waits a 1,023-token sequence | 3,747 / 3,796 | 2,293 / 2,331 |
| 8 held-out texts, exact routing: KL / top-1 agreement / ppl | 0.228 / 79.2% / 9.68 | 0.355 / 75.0% / 10.80 |
| the chat (a 62-token prompt, 419 tokens out, first answer after start) | - | 38.5 tok/s |

Its experts are 30% smaller, so fewer of them come from the SSD. The quality loss is clear (every one of the 8 texts,
KL up to 0.51), so it is the pick when speed matters more than accuracy.

**The four files on an AI PRO R9700 + Radeon 8060S lane** (2026-10-22, v1.3.46's build, 60 GB RAM tier, the same SSD;
8 held-out texts, a token in ms / KL against the unquantized model, the disk tail skip `STRATA_GLM_TAIL_SKIP=3
STRATA_GLM_TAIL_SKIP_DISK=1` on and off):

| | size | exact routing | with the tail skip |
| --- | ---: | :---: | :---: |
| unsloth UD-IQ2_XXS | 300 GB | **57.8 / 0.186** | **41.1 / 0.209** |
| nazeshinjite Q2 | 366 GB | - | 50.7 / 0.243 |
| unsloth UD-IQ1_M | 266 GB | 48.7 / 0.270 | 37.7 / 0.281 |
| unsloth UD-IQ1_S | 233 GB | 36.7 / 0.354 | 31.3 / 0.359 |

unsloth's UD-IQ2_XXS is the pick from v1.3.46: nazeshinjite Q2's speed with 66 GB less on the SSD and a clearly lower
KL (0.186 against 0.228). The AMD build now carries V4.1's RDNA expert kernels, the CPU lane's IQ1_S rows and the verify
windows (the n-gram drafts: a copy task 36 -> 28 ms a token with one draft, 25 with three); against v1.3.45's build
UD-IQ1_S went 43.9 -> 36.9 ms a token with the lane (52.0 -> 44.0 without), UD-IQ2_XXS 62.7 -> 60.8 (69.6 -> 66.6).

**ISTA-DASLab's GSQ-RCO GGUFs** (v1.3.53): [ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
IQ2_XXS (138 GB) and IQ1_M (121 GB) keep the Engram tables in a separate GGUF (`deepseek41.engram.file`, ~110 GB).
unsloth's 1017 shard with `per_layer_token_embd` holds the same IQ4_NL bytes, so `--engram` can point at it instead
of a second download. The prompt path decodes the IQ1_M experts to FP16 once each (8K prompt 281 -> 109 s). Same box
and texts as below (AI PRO R9700 + 8060S lane, 60 GB RAM tier, Lexar NM790 SSD, v1.3.53):

| | size (+ Engram) | KL, exact routing | swe-bench 8 tasks |
| --- | ---: | :---: | :---: |
| ISTA IQ2_XXS | 138 GB | **0.3137** | 30.7 / 30.9 tok/s |
| ISTA IQ2_XXS + tail skip (lossy) | | 0.3266 | 42.2 / 41.3 tok/s |
| ISTA IQ2_XXS + tail skip + DSpark drafter | | 0.3266 | **48.3 / 47.3 tok/s** |
| ISTA IQ1_M | 121 GB | 0.3845 | – |
| unsloth UD-IQ1_S (reference) | 233 GB | 0.3557 | 40.1 tok/s |

```sh
.venv/bin/python tools/iq_pack.py --gguf <ISTA>/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf \
  --out <ISTA>/pack-iq2xxs --compat-bf16 --engram <unsloth>/UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00005-of-00005.gguf
```

**The DSpark drafter** (v1.3.53, opt-in): Lucebox's DeepSeek-V4.1-Flash-1017-DSpark-draft-Q2K-Q4K.gguf
(`deepseek41-dflash-draft`, 4.9 GB) proposes up to five tokens; the model verifies them, so it does not change what is
sampled. `STRATA_DS4_DSPARK=<file> STRATA_DS4_DSPARK_CHAIN=0.6` in the config's `env`. swe-bench 8 tasks on UD-IQ1_S:
40.4 -> 46.7 tok/s (+16%, 4 runs, Lexar NM790); on the old ~3.4 GB/s SSD it measured no gain, since a verify window's
columns wait on the disk. With an iGPU, leave its lane on: the windows are CPU-lane bound and the lane takes part of that work (+25-33% above). Its 1.67 GB of VRAM (dense 0.46 GB, 48 / 40 / 24 expert slots a stage) come out of the expert pool.

**Uncensoring**: the run-time `--uncensor` (an `--abl-build` file or mrfakename's edit via `tools/abl_export.py`) and
the `--abl` overlay are described and measured in [UNCENSOR.md](../UNCENSOR.md).

The `deepseek41` GGUFs from kyuz0 / antirez named "v4.1" without `1017` are the September checkpoint
(`general.source.revision` `df42c109`); check the revision before trusting a repo name. Lucebox's ROCMFP files use
their own expert types (107 / 104), which Abyss does not read; their dense tensors and Engram tables do (a hybrid,
below).

```sh
.venv/bin/python tools/iq_pack.py --gguf <DeepSeek-V4.1-Flash-1017-Q2.gguf> --out <dir>/pack --compat-bf16
# unsloth: shard 1, the pack in the shards' folder
.venv/bin/python tools/iq_pack.py --gguf UD-IQ1_S/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf --out UD-IQ1_S/pack --compat-bf16
```

The pack keeps only an index, ~2.5 GB of converted small tensors and the Engram table offsets; the experts and the
Engram rows are read from the GGUF. Copy the 1017 chat template (`tokenizer/chat_template.jinja` from a GGUF that has
one, e.g. mrfakename's) into the pack if the GGUF carries none - the ds4-layout files do not. Keep the GGUF on a fast
local NVMe: the experts that do not fit and every Engram row come from it.

## Running it

`./setup-deepseek.sh` installs DeepSeek V4 only. For V4.1, build the engine (the manual's
[Build](../MANUAL.md#build)), make the pack as above and write the config next to the others (`abyss-<name>.json`):

```json
{
 "exe": "<Abyss>/build/strata",
 "args": ["--glm-pack", "<dir>/pack", "--max-context", "65536", "--prefill", "auto"],
 "cwd": "<Abyss>",
 "tokenizer": "<dir>/pack/tokenizer",
 "model_name": "deepseek-v4.1-flash-1017-q2",
 "gpu": [0],
 "sampling": {"temperature": 1.0, "top_p": 0.95},
 "port": 8082,
 "log": "<Abyss>/abyss-dsv41-q2.log",
 "env": {"STRATA_GLM_RAM_GB": "60", "STRATA_GLM_DISK_READERS": "8", "STRATA_GLM_USAGE": "0",
         "STRATA_GLM_TAIL_SKIP": "3", "STRATA_GLM_TAIL_SKIP_DISK": "1"}
}
```

Then start the server in a memory-capped scope. 153 GB of experts on a 94 GB box keep the page cache full, and without
a cap systemd-oomd may kill the session that started the server:

```sh
systemd-run --user --scope -p MemoryMax=88G .venv/bin/python serve/server.py --engine strata \
  --config abyss-dsv41-q2.json --port 8082
```

Drop the two `TAIL_SKIP` entries for the exact routing (17.7 tok/s instead of ~28, below). Do not add
`ManagedOOMPreference=omit` to the scope: oomd then kills other programs instead.

### A lighter dense pack (Q5_K)

Every token reads all 8.14 GB of the dense weights, and 4.65 GiB of that is three Q8_0 tensors a layer: `attn_q_b`,
`attn_output_a` and `attn_output_b`. As for DeepSeek V4 ([the manual](../MANUAL.md#a-lighter-dense-pack-the-attention-projections-at-q6_k-or-q5_k)),
`tools/requant_dense.py` re-quantizes just those to Q5_K (3.01 GiB) and copies every other byte; from v1.3.36 it keeps
the GGUF's 16 KB alignment (before, the V4.1 output came out shifted and `iq_pack.py` refused it). The new GGUF is
another 366 GB (~25 minutes); the pack takes its expert statistics and tokenizer from the original one:

```sh
M=<models dir>
GGUF_PY=~/llm/apps/llama.cpp/gguf-py LIBGGML=~/llm/apps/llama.cpp/build-cuda/bin/libggml-base.so \
  .venv/bin/python -I tools/requant_dense.py $M/nazeshinjite/DeepSeek-V4.1-Flash-1017-Q2.gguf \
  $M/nazeshinjite-attn-q5k q5_k 'blk\.\d+\.attn_q_b\.weight' 'blk\.\d+\.attn_output_[ab]\.weight'
.venv/bin/python tools/iq_pack.py --gguf $M/nazeshinjite-attn-q5k/DeepSeek-V4.1-Flash-1017-Q2.gguf \
  --out $M/nazeshinjite-attn-q5k/pack --compat-bf16
cp -r $M/nazeshinjite/pack/{expert_prior.txt,expert_counts.txt,tokenizer} $M/nazeshinjite-attn-q5k/pack/
```

Then point the config's `--glm-pack` and `tokenizer` at `nazeshinjite-attn-q5k/pack`. On the 8 held-out sequences
(the speed config above, v1.3.36): KL 0.2416 -> 0.2508, top-1 78.96 -> 78.48%, ppl 9.78 -> 9.88, and 10-15% faster
a sequence where the SSD had no slow spell (seq 0 / 1 / 2 / 4 / 5: 36.6 / 36.7 / 29.3 / 29.5 / 29.0 ->
31.4 / 31.2 / 26.6 / 26.2 / 26.1 ms a token); the dense weights drop to 6.50 GB and the VRAM tier gains 177 expert
slots (2,321 -> 2,498). It stays optional until a swe-bench run shows the answers hold up.

### Overlays: changed dense tensors without another copy of the model (v1.3.52)

`tools/dense_overlay.py` writes only the changed tensors into a small first shard and links the model's own shards
behind it; `iq_pack.py --first-wins` then takes the changed copies (the engine looks tensors up in shard order too).

**UD-IQ2_XXS with Q5_K dense** (2026-11-01, lossy, opt-in, recommended on a 32 GB card): the layers' Q8_0 / Q6_K
projections (attention, shared experts, `engram_kv`; 283 tensors, 6.26 -> 4.36 GB) re-quantized to Q5_K - a 4.4 GB
overlay; the VRAM tier gets the 1.9 GB. RTX 5090 + Radeon 8060S lane, the everyday settings + DSpark, swe-bench 3 tasks
alternated 4 times: 47.5 -> 50.5 tok/s (+6%; a token -6 / -9 / -18% on the three tasks), 8 held-out texts with exact
routing: KL 0.1420 -> 0.1456, ppl 8.79 -> 8.87, same top 83.8 -> 83.7%.

```sh
U=<models dir>/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/UD-IQ2_XXS
O=<models dir>/ud-iq2xxs-q5k
.venv/bin/python -I tools/dense_overlay.py $U/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf $O \
  --type=q5_k --from=q8_0,q6_k \
  'blk\.\d+\.(attn_q_a|attn_q_b|attn_kv|attn_output_a|attn_output_b|ffn_(gate|up|down)_shexp|engram_kv)\.weight'
STRATA_GGUF_PY=third_party/llama.cpp/gguf-py .venv/bin/python tools/iq_pack.py \
  --gguf $O/overlay-00001-of-00007.gguf --out $O/pack --compat-bf16 --first-wins
cp $U/pack/expert_*.txt $O/pack/
```

Then point the config's `--glm-pack` and `tokenizer` at `$O/pack` (`GGUF_PY` / `LIBGGML` as for `requant_dense.py`).

Two uses on unsloth's UD-IQ1_S (AI PRO R9700 + 8060S lane, 94 GB RAM, the 8 held-out sequences and the swe-bench
subset of 8 tasks):

- **Q5_K dense** (lossy, opt-in): the layers' Q6_K projections (attention, shared experts, `engram_kv`; 218 tensors,
  3.85 -> 3.22 GB) re-quantized to Q5_K - a 3.2 GB overlay. GPU time a token -1.4 ms (~6%), dense VRAM 6.08 -> 5.50 GB,
  swe-bench 40.1 -> 43.0 tok/s (two runs each, alternated); KL 0.3557 -> 0.3652, ppl 10.79 -> 10.93. Q5_K on
  `output.weight` as well is no faster and costs more (KL 0.3686).
- **A published abliteration edit** (`--abl`): see [UNCENSOR.md](../UNCENSOR.md#older-ways).

```sh
U=<models dir>/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/UD-IQ1_S
O=<models dir>/ud-iq1s-q5k
.venv/bin/python -I tools/dense_overlay.py $U/DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf $O \
  'blk\.\d+\.(attn_q_a|attn_q_b|attn_kv|attn_output_a|attn_output_b|ffn_(gate|up|down)_shexp|engram_kv)\.weight'
.venv/bin/python tools/iq_pack.py --gguf $O/overlay-00001-of-00006.gguf --out $O/pack --compat-bf16 --first-wins
cp $U/pack/expert_*.txt $O/pack/
```

Then point the config's `--glm-pack` and `tokenizer` at `$O/pack`. `GGUF_PY` / `LIBGGML` as for `requant_dense.py`.

## Quality and speed (measured in v1.3.31 on an RTX 5090 + Radeon 8060S lane + 94 GB RAM + NVMe at 3.4 GB/s; the R9700 numbers are in the README and docs/BENCH)

Quality is measured against the unquantized 1017's top-256 log-probs on Lucebox's 8 held-out sequences
(8 x 1,023 positions, `eval/` of [Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF)):

| | decode | KL | top-1 agreement | ppl |
| --- | --- | --- | --- | --- |
| nazeshinjite Q2 + disk tail skip (the config above, v1.3.31) | 35.9 ms/token (27.9 tok/s); 29-37 ms on 7 of the 8 | 0.241 | 78.9% | 9.75 |
| nazeshinjite Q2, exact routing (v1.3.29) | 56.6 ms/token (17.7 tok/s) | 0.229 | 79.2% | 9.69 |
| hybrid: Lucebox Q8_0 dense + 8-bit Engram, mrfakename Q2_K/Q3_K experts | 118 ms/token (8.5 tok/s) | 0.168 | 83.4% | 9.25 |
| (Lucebox ROCMFP2S-IM1017, their numbers) | - | 0.253 | 78.7% | 9.78 |

swe-bench with the config above (v1.3.31, a partial run: the first 35 of 60 tasks x 5; the full run is still owed): of those 35 tasks it resolved 34 (quality 84.9;
DeepSeek V4 Flash 35 and 85.0) at **51.6 tok/s** end to end, 38-59 a task (V4 Flash on the same 35 tasks: 107.2).
The engine averaged 19.1 ms a token over those 175 answers: 15.3 ms where it read under 0.5 experts a token from the
SSD, 28.2 ms past 1.5. The run is valid only up to task 35, where a 15K-token reasoning hit the 16K bug fixed in
v1.3.32; a full re-run is pending. On the 16 long tasks of the L7 set it resolved 14 at 28.9 tok/s.

On the same GGUF Abyss matches llama.cpp's V4.1 pull request (teacher KL 0.639 against 0.636). The prompt path
(batched MoE, the attention a token at a time) agrees with the token path at KL 0.013; a chat's next turn reuses the
previous state (1,132-token turn: 44 s -> 4.7 s of prompt, identical text). Prompts are slow: on the hybrid pack a
1K-token prompt took ~27 s with 8 SSD readers, most of it reading the experts the prompt touches.

Two things make single numbers noisy here: this NVMe has slow spells (one run in eight at 2x the read waits, and it
reads slower once warm), and before v1.3.31 the iGPU lane stalled for 15-24 s now and then. Compare runs by their
disk waits (`STRATA_GLM_TIMING=1`, the `tiers` line), not by one mean.

The hybrid shard set is shard 1 = the dense and Engram tensors fetched from Lucebox's GGUF by HTTP range, shards
2..11 = symlinks to mrfakename's shards, packed with `iq_pack.py --first-wins`.

## Options for this model

| variable | effect |
| --- | --- |
| `STRATA_GLM_TAIL_SKIP=3 STRATA_GLM_TAIL_SKIP_DISK=1` | leave out a route's 4th-6th experts when they are only on disk: 56.6 -> 34.2 ms/token, KL 0.229 -> 0.240, top-1 79.2 -> 78.9% (lossy, off by default; `=4` keeps ranks 1-4: 43.9 ms, KL 0.235) |
| `STRATA_GLM_RAM_GB=60` | the RAM tier; on a 94 GB box keep ~25 GB for the page cache (the SSD experts and the Engram rows go through it) |
| `STRATA_GLM_DISK_READERS=8` | eight SSD readers for a prompt's experts (hybrid pack, 1K-token prompt: 33 -> 27 s) |
| `STRATA_GLM_USAGE=0` | no usage file: a short session's counts froze the tiers (137 -> 169 ms/token) |
| `STRATA_GLM_RAM_THP` | on (v1.3.31): the RAM tier in transparent huge pages; with 4 KB pages the APU driver re-validated them for 15-24 s at a time |
| `STRATA_GLM_STREAM_FETCH` | on (v1.3.31): a disk-landed expert goes to VRAM chunk by chunk while the SSD still reads it (seq 0: 43.6 -> 39.5 ms/token, same logits) |
| `STRATA_DS41_ENGRAM_CACHE_MB` / `_THREADS` | the Engram raw-row cache (default 256 MB) and read threads (default 15 + caller) |
| `STRATA_GLM_DISK_LANE=1` | a disk read is computed by the CPU / iGPU lane from its RAM landing instead of being pulled into VRAM (-5% on the Q2_K/Q3_K experts; none on the Q2) |
| `STRATA_DS41_ENGRAM_ASYNC` | on (v1.3.36): a token's 48 Engram rows are read by a background thread while layer 0 runs; layers 1 and 14 wait for them on the GPU and copy them host-mapped (-0.5 ms a token, same logits) |
| `STRATA_DS41_ENGRAM_PREFETCH` | 8 (v1.3.36): threads reading a prompt chunk's table 0 then table 1 rows into the row cache right after its hash (1K-token prompt: Engram 1.7-1.9 -> 0.6 s, row-cache hits 8 -> 79%, the prompt -6-7%); `0` off |
| `STRATA_GLM_IGPU_EARLY` | on (v1.3.36): the iGPU lane starts before a route's SSD reads instead of after them (-2%, same logits) |
| `STRATA_GLM_ROUTE_AFFINITY_RAM=0.08` | (v1.3.37, lossy, off by default) a resident expert's routing score +0.08, so a near-tie avoids an SSD-only expert: with the tail skip, SSD reads -22% and -8-12% a token on the sequences without iGPU stalls, KL 0.2406 -> 0.2438, ppl 9.76 -> 9.81 (8 sequences); `0.16`: -12-20% at KL 0.2501, ppl 9.89 - about the Q5_K pack's trade |

Measured and left off: `STRATA_GLM_AHEAD` (the predicted reads were 22% useful and crowded the SSD; v1.3.36 lets a
demand read wait on an in-flight lookahead read inside the streamed fetch, `STRATA_GLM_AHEAD_MERGE`, which saves 2%,
but with a 3.4 GB/s SSD 37.4 ms a token still loses to 34.9 without lookahead; re-measured 2026-11-01 on the Lexar NM790 (~6 GB/s) with the iGPU lane and DSpark, swe-bench 3 tasks: off 47.9 / 49.3 tok/s, `AHEAD=1` 40.0 / 42.1, `AHEAD=1 AHEAD_TOP=4` 42.6 / 46.0, `AHEAD=2 AHEAD_TOP=4` 39.5 / 40.2 - fewer disk reads a token, but longer waits: still off),
`STRATA_GLM_SEED_PRIOR=1` and `STRATA_GLM_DEMOTE_TIES=1` (46.8 -> 48.5 / 46.1 ms with more RAM re-reads),
`STRATA_GLM_DISK_ADAPT=1`, `STRATA_GLM_DEFER=1` (-4% at KL 0.15 -> 0.28 on one sequence).

## Not yet for V4.1

MTP drafts from the model's own layers (the 1017 FP8 release has three MTP layers; no GGUF carries them - the DSpark
drafter above, v1.3.53, is the opt-in alternative), conversation slots on disk, a layer split, and the
one-command setup. Images work in a limited way (see the Images row in SPEC.md section 8): the `vision` entry of the config is written by hand, and large or several pictures and the GPU encoder are untried. The prompt's attention has been batched since v1.3.60 (it ran a token at a time, ~5 ms each,
before).
