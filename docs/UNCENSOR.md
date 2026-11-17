# Uncensoring (`--uncensor`)

[日本語](UNCENSOR.ja.md)

Everything about running a model without its refusals is on this page: the run-time edit (`--uncensor`, v1.3.71),
how to make its file for your model, how to check it, what it measured, and the older ways (an abliteration's own
edit, an overlay shard, abliterated GGUFs).

The edit removes refusals. It adds no knowledge, and what the model then writes is the user's responsibility.

## Contents

- [Which way to use](#which-way-to-use)
- [How it works](#how-it-works)
- [Turning it on](#turning-it-on)
- [Making the file for your model](#making-the-file-for-your-model)
- [Checking a file](#checking-a-file)
- [Adding your own prompts](#adding-your-own-prompts)
- [Measured](#measured)
- [Limits](#limits)
- [Older ways](#older-ways)
- [File format](#file-format)

## Which way to use

| way | size | models | switch per request | quality cost |
| --- | --- | --- | :---: | --- |
| **`--uncensor <file>`, a file made by `--abl-build`** (recommended) | 16-82 KB | every `--glm-pack` model: DeepSeek V4 / V4.1, GLM-5.3-Flash, qwen35moe (Ornith, Qwen3.5), Qwen3.8-Flash-Next (qwen4exp) | yes | text KL 0.013-0.053 |
| `--uncensor` with an abliteration's own edit (mrfakename's V4.1 `abliteration-edit.npz`, ABL1) | 43 KB | V4.1 | yes | KL +0.002 on UD-IQ1_S |
| an overlay shard (`tools/dense_overlay.py --abl`) | 1.4 GB | V4.1 | no | KL +0.0006 (the tensors are rounded again) |
| an abliterated GGUF (huihui-ai's V4 0731 Q2, orcarouter's V4.1 Mix-256G, orcarouter's Qwen3.8-Flash-Next-Uncensored) | the whole model | that model | no | V4.1 Mix-256G: text KL 0.083 |

The run-time edit needs no second copy of the model, keeps the ordinary model one request away, and on V4.1 costs a
third of the abliterated GGUF's KL. Qwen3.8-Flash-Next (qwen4exp, the fast path of
[docs/qwen38](qwen38/README.md)) takes `--abl-build` and `--uncensor` too (Unreleased; before it `--abl-build`
measured nothing on that path and the PLE rows were not edited).

## How it works

A model's refusals mostly follow one or a few directions in its residual stream (Arditi et al., "Refusal in Language
Models Is Mediated by a Single Direction"). Abliteration rewrites the weights so that nothing is written along those
directions. Abyss does the same on the fly: every write into the residual loses its component along the directions,

```
w  <-  w - s · Q (Qᵀ w)        Q: n_embd × rank, orthonormal;  s: the strength
```

at these sites:

- the token embedding,
- each layer's attention output,
- each layer's whole FFN output: routed experts + shared expert + the CPU lane's part,
- V4.1's Engram adds, Qwen3.8's per-layer embedding (PLE) rows.

The routed experts on the SSD are not touched, so the edit costs no disk space and no second rounding. It runs on
the token path, the prompt path and the speculative verify windows. A GPU flag switches it, so it works inside the
CUDA / HIP graphs and a request can turn it off.

## Turning it on

**Which file.** The model's run config: `abyss-<name>.json` in the Abyss folder (for example
`abyss-dsv41-iq2xxs.json`). Setup writes it; it is the file `run-abyss-<name>.sh` passes to the server as
`--config`, and the one `serve/server.py --config <file>` reads if you start the server yourself
([SETTINGS](SETTINGS.md) shows where it is and how to find the one in use). Each model has its own, so add the line to the one of
the model the `.abl` file was made for.

Add one top-level key, beside `"exe"` and `"args"` (not inside `"args"` or `"env"`):

```json
{
 "exe": "/home/you/llm/apps/Abyss/build/abyss",
 "args": ["--glm-pack", "...", "--max-context", "65536"],
 "uncensor": "/home/you/llm/apps/Abyss-data/abl/dsv41-iq2xxs.abl",
 ...
}
```

Then restart the server (`../Abyss-tools/abyss.sh restart`, or stop it and run `run-abyss-<name>.sh` again); the file is read when the engine starts. At
start the engine log (the config's `"log"`) says `glm fast: uncensored (<file>): ...`.

Two other ways do the same: `"--uncensor", "<file>"` added to `"args"`, or `"STRATA_GLM_ABL": "<file>"` in the
config's `"env"`. Use one of the three, not several.

Requests then run with the edit:

| switch | edit |
| --- | --- |
| a request with `"uncensor": false` | off for that request |
| a request with `"uncensor": true` or without the key | on |
| `POST /settings` with `"uncensor": false/true` | the default other apps get (a shared setting, like `temperature`) |
| the engine protocol: `GEN ... abl=0` / `abl=1` | off / on |
| `STRATA_GLM_ABL_ON=0` | loaded, but off by default |

A conversation computed with the other setting is not reused (the server keys its cache by the switch).

**Off is unchanged.** With the file loaded and switched off, the logits match a run without the file bit for bit
(V4.1, score path KL 0, the same CPU-lane count). **On costs no speed** beyond run-to-run noise (V4.1 on an AI PRO
R9700: 20.4 tok/s on, 21.3 off, 22.2 without the file).

## Making the file for your model

A file is made from the model it is used with, in under an hour on an AI PRO R9700. Make one per model (a V4 file does
not fit V4.1, and a file is refused at start when the model's width differs). The files above were built on the quant
they were measured with; for another quant of the same model, check the KL or build one for it.

**1. The prompts.** The directions are measured on public prompt sets: AdvBench, MaliciousInstruct,
ForbiddenQuestions, SimpleSafetyTests, JailbreakBench (harmful and benign behaviors) and XSTest (unsafe and safe
prompts). Get them from their own pages into `Abyss-data/datasets/safety` (its `MANIFEST.txt` lists the files).
`tools/abl_prompts.py` mixes them evenly and writes them as token ids with the pack's chat template (thinking off):

```sh
.venv/bin/python -I tools/abl_prompts.py --pack <pack> --out prompts.txt
```

| option | default | meaning |
| --- | --- | --- |
| `--train N` | 64 | harmful and harmless prompts each, for the measurement |
| `--check N` | 24 | the next ones, for trying the candidates |
| `--exclude FILE ...` | – | drop prompts an evaluation uses (lines of text, or JSON lines with a `"prompt"`) |
| `--harmful mix\|advbench` | mix | AdvBench alone leaves the softer refusals (piracy, medicine, fraud) in GLM |
| `--cyber N` | 0 | force N malware / hacking prompts into the harmful mix (ForbiddenQuestions' Malware category and cyber keywords in AdvBench / MaliciousInstruct, none from JailbreakBench). Use it when the malware refusals survive the edit: on Qwen3.8 they were a direction of their own |
| `--data DIR` | `Abyss-data/datasets/safety` | where the sets are |

The picks are the same on every run. The datasets are untrusted data: run the tool with `python -I`.

**2. The build.**

```sh
build/abyss --glm-pack <pack> --max-context 4096 --abl-build model.abl --abl-prompts prompts.txt \
  --abl-gen 32 --abl-rank 3 --abl-range 0.3,0.9 --abl-candidates 9
```

What it does:

1. Reads each training prompt to its last token and keeps the residual at every layer.
2. A layer's direction = the harmful prompts' mean minus the harmless ones', **made orthogonal to the harmless
   means**. Without that step the direction carries the model's huge fixed channels and breaks it (KL 9 on one
   model).
3. Tries each candidate layer's direction on the 24 + 24 check prompts: how many harmful answers still refuse, and the
   harmless prompts' first-token KL.
4. Combines the best layers within `--abl-max-kl`, up to `--abl-rank`, and writes the winner (ABL2) plus
   `model.abl.dirs` with every layer's direction.

| option | default | meaning |
| --- | --- | --- |
| `--abl-candidates N` | 8 | layers tried, spread over `--abl-range` |
| `--abl-range LO,HI` | 0.2,0.8 | the candidates' depth, as fractions of the layers |
| `--abl-layer L` | – | try only this layer |
| `--abl-gen N` | 0 | generate N greedy tokens of each harmful answer and look for refusal words in them (" can't", " cannot", "Sorry", 申し訳, ...). Use 32: GLM opens with a heading and refuses in the second line, so the first token alone misses it |
| `--abl-max-kl K` | 0.2 | the harmless first-token KL a candidate may cost (raise it - see below) |
| `--abl-rank N` | 1 | also try the best N directions together (1-4) |
| `--abl-scale S` | 1 | strength: 1 takes the component out, above 1 over-projects |
| `--abl-dirs F` | – | reuse an earlier build's `.dirs`: skips the measurement, ~10 minutes to try other settings |
| `--abl-dir-scales C1,..` | – | each written direction's own scale on top of the layer strengths (an ABL3 file, one scale a direction) |
| `--abl-search N` | 0 | experimental: N trials of a per-layer strength profile instead of one strength (needs `T` lines - `abl_prompts.py --text` - and `--abl-gen`). On V4 0731 and GLM-5.3 it did not beat the uniform pick on held-out prompts |
| `--abl-dump F` | – | write every measured prompt's captured residuals to F and stop (for per-category directions) |

The log, one line a candidate (V4 0731, abridged):

```
ABL unedited: a refusal word in the first 32 tokens of 20 of 24 harmful check answers
ABL layer 26: ... harmless KL 0.7945, refused in 32 tokens 7
ABL layer 29: ... harmless KL 0.7295, refused in 32 tokens 3
ABL layer 39: ... harmless KL 0.3053, refused in 32 tokens 6
ABL chose layer 29 (rank 1, scale 1.30: ...) -> model.abl
```

The first-token KL on chat prompts overstates the cost: on V4 it read 0.7-1.0 where the text KL was 0.013. Pick by
refusals, then check the text KL (next section); `--abl-max-kl 1.5` let V4's build choose its best layer.

## Checking a file

**Quality** - the KL on plain text with the score path, once without the file and once with it:

```sh
STRATA_GLM_SCORE=400 STRATA_GLM_SCORE_DUMP=ref.bin        # without the file: the reference
STRATA_GLM_SCORE_REF=ref.bin STRATA_GLM_ABL=model.abl     # with it: "vs ref: KL ..."
```

(`STRATA_GLM_ABL_ON=0` with the file loaded must give KL 0.)

**Refusals** - ask held-out prompts (not the ones in `prompts.txt`) with `"uncensor": false` and `true`, greedy, and
read the answers' openings. Count the harmless prompts too: a good edit also lowers over-refusal.

## Adding your own prompts

**When.** After `--abl-build`, one kind of request is still refused while the rest are answered. Each kind of refusal
can sit on a direction of its own, and the build only finds the directions its prompts show it. On Qwen3.8-Flash-Next
(ISTA IQ3_S) the first file left Malware / Hacking refused (4 of 5 JailbreakBench items); building again with more
cyber prompts and one more direction fixed it:

| build | JBB refused | Malware / Hacking | XSTest over-refusal | math | text KL (en / ja / code / md) |
| --- | :---: | :---: | :---: | :---: | --- |
| default mix, `--abl-rank 3` | 11.1% | 4 of 5 | 0.8% | 44 / 50 | 0.028 / 0.015 / 0.046 / 0.029 |
| `--cyber 30`, `--abl-rank 4` | 1.2% | 0 of 5 | 0.0% (50 prompts) | 43 / 50 | 0.027 / 0.015 / 0.057 / 0.033 |

(81 JailbreakBench test prompts, 50 math problems at 512 tokens, greedy, refusals read on the whole answer. With
`--cyber 30` alone at rank 3 the malware refusals stayed: the fourth direction, layer 18, is the one that cleared them.)

**`--cyber N`** forces N malware / hacking prompts into the harmful side: ForbiddenQuestions' Malware category and
cyber keywords in AdvBench / MaliciousInstruct (139 prompts; none from JailbreakBench).

**`--extra-harmful FILE`** forces your own prompts into the harmful side the same way. Abyss ships none: you write
them for the kind of request your model still refuses. The file:

```
# one prompt a line, UTF-8; blank lines and lines starting with # are skipped
example prompt 1
example prompt 2
{"prompt": "example prompt 3"}
```

They go first and stay through the trim to `--train` + `--check` (the default mix fills the rest); `--extra-n N`
takes at most N of them, `--exclude` still applies, and the tool prints how many it took. Then build as usual, with
one more direction:

```sh
.venv/bin/python -I tools/abl_prompts.py --pack <pack> --out prompts.txt --cyber 30 --extra-harmful my.txt
build/abyss --glm-pack <pack> --max-context 4096 --abl-build model.abl --abl-prompts prompts.txt \
  --abl-gen 32 --abl-rank 4 --abl-range 0.3,0.9 --abl-candidates 9 --abl-max-kl 1.0
```

**Check** as in [Checking a file](#checking-a-file): the refusals on held-out prompts, and that long answers (math)
and the text KL did not get worse. Keep your file apart from whatever you test with - a prompt the build saw says
nothing about the edit.

## Measured

AI PRO R9700. 101 held-out prompts (66 harmful, 35 harmless), none of them in the build's sets; greedy; "refused" read
from the answer's opening.

| Model | Harmful refused (off -> on) | Harmless refused (off -> on) | Text KL | The edit | File |
| --- | :---: | :---: | :---: | --- | --- |
| DeepSeek V4.1 Flash 1017, unsloth UD-IQ2_XXS | 86% -> 3% | 17% -> 3% | 0.023 | layers 27+30+33+24, strength 1.3 | 82 KB |
| DeepSeek V4 Flash 0731, UD-Q2_K_XL attn-q5k | 85% -> 6% | 11% -> 3% | 0.013 | layer 29, strength 1.3 | 16 KB |
| GLM-5.3-Flash, Maya-S-v2-IQ2_XXS | 88% -> 8% | 11% -> 3% | 0.053 | layers 31+28+34, strength 1.0 | 50 KB |

Qwen3.8-Flash-Next, ISTA-DASLab GSQ-RCO IQ3_S (RTX 5090): the build chose layers 39+43+36, rank 3, strength 1.0
(31 KB, `--abl-gen 32 --abl-range 0.3,0.9 --abl-candidates 9 --abl-max-kl 1.0`, 5 minutes); a refusal word in the first
32 tokens of 23 of 24 harmful check answers without it, 3 with it.

On V4.1, harmful answers: 3.0% refused, 22.7% a short caveat followed by the answer, 74.2% answered. On V4: 6.1% /
28.8% / 65.2%.

For comparison, orcarouter's abliterated V4.1 Mix-256G (new weights) has a text KL of 0.083.

## Limits

- **Some refusals stay.** The ones left are mostly a short caveat followed by the answer, or medical and financial
  advice.
- **Two speed features stay off while a file is loaded**: the hc3 fusion of the CPU lane's answer and
  `STRATA_GLM_DEFER` (the FFN output must be complete when it is edited). With the file switched off they stay off too.
- **One file per model.** Another quant of the same model has not been measured with these files; check the KL.
- **`--glm-pack` models only.** Not on the older engine path (the Maya-era packs, Qwen3.8-Flash-Next's older pack).

## Older ways

**An abliteration's own edit** (v1.3.53). mrfakename/DeepSeek-V4.1-Flash-1017-abliterated ships
`abliteration-edit.npz` (43 KB): a rank-2 refusal subspace and per-layer strengths for the attention and the shared
expert's outputs. Convert it and load it like an `--abl-build` file:

```sh
.venv/bin/python -I tools/abl_export.py abliteration-edit.npz v41-mrfakename.abl
```

UD-IQ1_S: KL 0.3557 -> 0.3577, 16 refusal-bait prompts (12 English, 4 Japanese) refused 12 -> 0. It edits only those
two outputs; it was not run on the 101-prompt set.

**An overlay shard** (v1.3.52). The same edit written into the weights, `W - s Q Qᵀ W` on `attn_output_b` /
`ffn_down_shexp` (62 tensors, a 1.4 GB overlay GGUF):

```sh
.venv/bin/python -I tools/dense_overlay.py <shard 1 of the model> <out dir> --abl=<dir>/abliteration-edit.npz
.venv/bin/python tools/iq_pack.py --gguf <out dir>/overlay-00001-of-0000N.gguf --out <out dir>/pack --compat-bf16 --first-wins
cp <model dir>/pack/expert_*.txt <out dir>/pack/
```

UD-IQ1_S: KL 0.3557 -> 0.3563, refusals 14 of 16 -> 0; with the Q5_K overlay too KL 0.3738 (rounded twice). It cannot
be switched per request; the run-time edit replaces it.

**Abliterated GGUFs** are packed and run like any other GGUF of their model ([MODELS](MODELS.md)): huihui-ai's
DeepSeek V4 0731 Q2 (`--compat-bf16`), orcarouter's V4.1 Mix-256G (text KL 0.083, swe-bench 22.4 tok/s, disk-bound)
and orcarouter's Qwen3.8-Flash-Next-Uncensored.

## File format

Little-endian:

```
char[4]  "ABL1" | "ABL2" | "ABL3"
int32    n_embd, rank, layers
float32  Q[n_embd][rank]          row-major, orthonormal columns
float32  scale[rank]              ABL3 only: each direction's own factor
float32  strength[layers][2]      per layer: ABL1 attention, shared expert / ABL2 attention, whole FFN
float32  embedding strength       ABL2 / ABL3
```

ABL1 edits the attention and the shared expert's outputs (an abliteration's own edit); ABL2 edits every residual
write; ABL3 is ABL2 with a scale a direction - a write w becomes w - s · Σ c_j q_j (q_jᵀ w), so a second direction
(for example the medical / legal / financial advice refusals') can be taken out less than the first.  ABL1 and ABL2
files load as before. A file whose `n_embd` differs from the model's is refused at start.
